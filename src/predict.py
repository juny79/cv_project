import os, yaml, timm, torch
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.transforms import get_valid_transforms

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def load_cfg(p):
    try:
        with open(p, 'r') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a mapping/dictionary")
        return cfg
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load config {p}: {e}")

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p): return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found for {stem} under {img_dir}")

class TestDS(Dataset):
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        idc = [c for c in self.df.columns if c.lower() in ['id','image_id','filename']]
        self.id_col = idc[0] if idc else self.df.columns[0]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        stem = str(self.df.iloc[i][self.id_col])
        p = resolve_image_path(self.img_dir, stem)
        img = np.array(Image.open(p).convert('RGB'))
        x = self.transform(image=img)['image']
        return x, stem

@torch.no_grad()
def predict_ensemble(cfg_path, tta=4):
    cfg = load_cfg(cfg_path)
    paths, data = cfg['paths'], cfg['data']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = get_valid_transforms(int(data['img_size']))

    test = TestDS(paths['sample_csv'], paths['test_dir'], tfm)
    num_workers = min(int(cfg.get('num_workers', 0)), os.cpu_count() or 1)
    pin_memory = True if device.startswith('cuda') else False
    loader = DataLoader(test, batch_size=int(cfg['train']['batch_size']), shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    ckpts, folds = [], data.get('folds', [0,1,2,3,4])
    for k in folds:
        p = os.path.join(paths['out_dir'], f'fold{k}', 'best.pt')
        if os.path.exists(p): ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {paths['out_dir']}/fold*/best.pt")

    state0 = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    # num_classes 추정
    head_weight = None
    for k,v in state0['model'].items():
        if 'head.fc.weight' in k and hasattr(v,'shape'):
            head_weight = v
            break
    if head_weight is None:
        raise RuntimeError("cannot find head.fc.weight in checkpoint")
    num_classes = head_weight.shape[0]

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()

    def _four_rot_logits(xb):
        outs = []
        outs.append(model(xb))                                        # 0
        outs.append(model(torch.rot90(xb, 1, dims=[2,3])))            # 90
        outs.append(model(torch.rot90(xb, 2, dims=[2,3])))            # 180
        outs.append(model(torch.rot90(xb, 3, dims=[2,3])))            # 270
        return outs

    logits_folds = None
    for ck in ckpts:
        w = torch.load(ck, map_location=device, weights_only=False)
        try:
            model.load_state_dict(w['model'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Failed to load checkpoint {ck} with strict=True: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(w['model'], strict=False)
        model.eval()

        selected_logits = []
        img_ids = []  # Store image IDs for later
        for xb, _ids in loader:
            xb = xb.to(device)
            img_ids.extend(_ids)
            if int(tta) == 0 or int(tta) == 1:
                # no rotation TTA: single forward
                out = model(xb)
                selected_logits.append(out.detach().cpu())
                continue

            # default/4: rotation-based TTA
            cand = _four_rot_logits(xb)  # list of 4 tensors [B,C]
            probs = [torch.softmax(z, dim=1) for z in cand]
            maxp = torch.stack([p.max(dim=1).values for p in probs], dim=0)  # [4,B]
            best_idx = torch.argmax(maxp, dim=0)                              # [B]
            # pick best rotation per sample
            pick = torch.stack([cand[r][i] for i, r in enumerate(best_idx.tolist())], dim=0)  # [B,C]
            selected_logits.append(pick.detach().cpu())

        fold_logits = torch.cat(selected_logits)

        if logits_folds is None:
            logits_folds = [fold_logits]
        else:
            logits_folds.append(fold_logits)

    mean_logits = torch.stack(logits_folds).mean(0)
    preds = torch.argmax(mean_logits, 1).numpy()

    sub = pd.read_csv(paths['sample_csv'])
    idc = [c for c in sub.columns if c.lower() in ['id','image_id','filename']]
    ycol = [c for c in sub.columns if c.lower() in ['label','target','class']]
    id_col = idc[0] if idc else sub.columns[0]
    y_col = ycol[0] if ycol else sub.columns[-1]

    if len(sub) != len(preds):
        raise ValueError(f"sample rows {len(sub)} != preds {len(preds)}")

    sub[y_col] = preds
    sub = sub[[id_col, y_col]]   # 포맷 강제
    # Save predictions
    out_csv = os.path.join(paths['out_dir'], 'submission.csv')
    os.makedirs(paths['out_dir'], exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    
    # Save logits and image IDs for analysis
    logits_path = os.path.join(paths['out_dir'], 'predict_logits.pt')
    torch.save({
        'logits': mean_logits.numpy(),
        'img_ids': img_ids,
        'predictions': preds,
    }, logits_path)
    print(f"Saved logits → {logits_path}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=4, help='TTA mode: 0 or 1 = no TTA, 4 = rotation TTA')
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
