import os, argparse, yaml, timm, torch, joblib, numpy as np, pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.transforms import get_valid_transforms

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p): return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found: {stem}")

class TestDS(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = transform
        idc = [c for c in df.columns if c.lower() in ['id','image_id','filename','image']]
        self.id_col = idc[0] if idc else df.columns[0]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_image_path(self.img_dir, r[self.id_col])
        img = np.array(Image.open(p).convert('RGB'))
        x = self.tf(image=img)['image']
        return x, r[self.id_col]

@torch.no_grad()
def infer_fold(model, loader, device, tta=0):
    model.eval()
    all_logits, all_ids = [], []
    for xb, ids in loader:
        xb = xb.to(device)
        logits = model(xb)
        if tta and tta > 1:
            # simple TTA: 90 deg rotations (matching earlier logic) up to tta count
            cands = [logits]
            rot_x = xb
            for k in range(1, tta):
                rot_x = torch.rot90(xb, k, dims=(2,3))
                cands.append(model(rot_x))
            logits = torch.stack(cands, 0).mean(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_ids.extend(ids)
    return np.concatenate(all_logits,0), all_ids

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-9, None)

def entropy(p, eps=1e-9):
    return -(p*np.log(p+eps)).sum(1)

def top2_margin(p):
    part = np.partition(-p, 1, axis=1)
    top1 = -part[:,0]; top2 = -part[:,1]
    return top1 - top2

def build_meta_features(p):
    return np.concatenate([p, entropy(p)[:,None], top2_margin(p)[:,None]], axis=1)

def apply_pair_refiners(base_probs, meta_probs, pair_models):
    # base_probs: per-sample class probabilities before meta
    # meta_probs: stacking layer output (will be adjusted for pairs)
    adjusted = meta_probs.copy()
    for (a,b), clf in pair_models.items():
        p_a = base_probs[:, a]
        p_b = base_probs[:, b]
        # region: both among top4 OR close margin
        margin = np.abs(p_a - p_b)
        region_mask = (margin < 0.25) | ((base_probs.argsort(axis=1)[:,-1] == a) & (base_probs.argsort(axis=1)[:,-2] == b)) | ((base_probs.argsort(axis=1)[:,-1] == b) & (base_probs.argsort(axis=1)[:,-2] == a))
        if not np.any(region_mask):
            continue
        feats = np.concatenate([
            p_a[region_mask][:,None],
            p_b[region_mask][:,None],
            (p_a[region_mask]-p_b[region_mask])[:,None],
            entropy(base_probs[region_mask])[:,None],
            top2_margin(base_probs[region_mask])[:,None],
        ], axis=1).astype(np.float32)
        probs_bin = clf.predict_proba(feats)
        # probs_bin[:,1] corresponds to target class 'a' if we trained with (y==a)->1
        p_ref_a = probs_bin[:,1]
        p_ref_b = probs_bin[:,0]
        # blend with meta probabilities for those samples
        adjusted[region_mask, a] = 0.55*adjusted[region_mask, a] + 0.45*p_ref_a
        adjusted[region_mask, b] = 0.55*adjusted[region_mask, b] + 0.45*p_ref_b
        # renormalize row-wise
        row_sum = adjusted[region_mask].sum(1, keepdims=True)
        adjusted[region_mask] /= np.clip(row_sum, 1e-9, None)
    return adjusted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/temp_full_mult4_tta4.yaml')
    ap.add_argument('--exp_out_dir', default='outputs/full_mult4_tta4', help='Dir with fold checkpoints + temp.npy')
    ap.add_argument('--test_csv', default='/root/cv_project/data/sample_submission.csv')
    ap.add_argument('--test_dir', default='/root/cv_project/data/test')
    ap.add_argument('--save_csv', default='submission_meta.csv')
    ap.add_argument('--tta', type=int, default=4)
    ap.add_argument('--meta_model', default='extern/meta_full.joblib')
    ap.add_argument('--pair_dir', default='extern', help='Directory containing pair_{a}_{b}.joblib files')
    args = ap.parse_args()

    with open(args.config,'r') as f:
        cfg = yaml.safe_load(f)
    model_name = cfg['model']['name']
    img_size = cfg['data']['img_size']
    batch_size = int(cfg['train']['batch_size'])

    df_test = pd.read_csv(args.test_csv)
    tf = get_valid_transforms(img_size)
    ds = TestDS(df_test, args.test_dir, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load per-fold models + temperatures
    fold_dirs = sorted([d for d in glob.glob(os.path.join(args.exp_out_dir,'fold*')) if os.path.isdir(d)])
    fold_logits = []
    image_ids = None
    for fd in fold_dirs:
        ckpt = os.path.join(fd, 'best.pt')
        if not os.path.exists(ckpt):
            print(f"[WARN] missing {ckpt}, skipping fold")
            continue
        temp_path = os.path.join(fd, 'temp.npy')
        T = 1.0
        if os.path.exists(temp_path):
            T = float(np.load(temp_path).reshape(-1)[0])
        model = timm.create_model(model_name, pretrained=False, num_classes=cfg['data'].get('num_classes', 17), drop_rate=float(cfg['model'].get('dropout',0.0)))
        sd = torch.load(ckpt, map_location='cpu')['model']
        model.load_state_dict(sd); model.to(device)
        logits, ids = infer_fold(model, dl, device, tta=args.tta)
        if image_ids is None:
            image_ids = ids
        probs = softmax(logits / max(T,1e-4))
        fold_logits.append(probs)
        print(f"[FOLD] {fd} collected probs shape={probs.shape} T={T:.3f}")

    if not fold_logits:
        raise RuntimeError("No folds processed. Check exp_out_dir.")

    base_probs = np.mean(np.stack(fold_logits, 0), axis=0)

    # Meta stacking
    meta_bundle = joblib.load(args.meta_model)
    meta_model = meta_bundle['model'] if isinstance(meta_bundle, dict) else meta_bundle
    meta_feats = build_meta_features(base_probs)
    meta_probs = meta_model.predict_proba(meta_feats)

    # Load pair refiners
    pair_models = {}
    for pf in glob.glob(os.path.join(args.pair_dir, 'pair_*.joblib')):
        bundle = joblib.load(pf)
        if isinstance(bundle, dict) and 'pair' in bundle:
            a,b = bundle['pair']
            pair_models[(a,b)] = bundle['model']
            print(f"[PAIR-LOAD] {pf} -> pair ({a},{b})")
    if pair_models:
        final_probs = apply_pair_refiners(base_probs, meta_probs, pair_models)
    else:
        final_probs = meta_probs

    preds = final_probs.argmax(1)
    out_df = pd.DataFrame({'image_id': image_ids, 'label': preds})
    out_df.to_csv(args.save_csv, index=False)
    print(f"[SAVE] {args.save_csv} with {len(out_df)} rows")

if __name__ == '__main__':
    main()
