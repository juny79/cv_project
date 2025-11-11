import os, yaml, torch, timm, argparse, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src.transforms import get_valid_transforms
from PIL import Image

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found for {stem}")

class DocDS(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = transform
        idc = [c for c in df.columns if c.lower() in ['id','image_id','filename','image']]
        self.id_col = idc[0] if idc else df.columns[0]
        yc = [c for c in df.columns if c.lower() in ['label','target','class']]
        self.y_col = yc[0] if yc else df.columns[-1]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_image_path(self.img_dir, r[self.id_col])
        img = np.array(Image.open(p).convert('RGB'))
        x = self.tf(image=img)['image']
        y = int(r[self.y_col])
        return x, y

@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_tgts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        lg = model(xb).detach().cpu().numpy()
        all_logits.append(lg)
        all_tgts.append(yb.numpy())
    return np.concatenate(all_logits,0), np.concatenate(all_tgts,0)

def fit_temperature(logits, labels, init_T=1.0, lr=0.01, iters=200):
    import torch, torch.nn.functional as F
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    T = torch.tensor([init_T], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=iters)
    def closure():
        opt.zero_grad(); z = x / T.clamp_min(1e-4); loss = F.cross_entropy(z, y); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp(0.3,5.0).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/temp_full_mult4_tta4.yaml')
    ap.add_argument('--exp_out_dir', default='outputs/full_mult4_tta4', help='Directory with fold*/best.pt (target for saving oof/temp)')
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    with open(args.config,'r') as f:
        cfg = yaml.safe_load(f)

    paths = cfg['paths']; data = cfg['data']; model_cfg = cfg['model']; trn = cfg['train']
    df = pd.read_csv(paths['train_csv'])
    ycol = [c for c in df.columns if c.lower() in ['label','target','class']]
    ycol = ycol[0] if ycol else df.columns[-1]
    y = df[ycol].astype(int).values
    n_splits = data.get('n_splits',5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get('seed',42))

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(df, y)):
        va_df = df.iloc[va_idx].reset_index(drop=True)
        va_tf = get_valid_transforms(data['img_size'])
        ds_va = DocDS(va_df, paths['train_dir'], va_tf)
        dl_va = DataLoader(ds_va, batch_size=int(trn['batch_size']), shuffle=False, num_workers=args.num_workers, pin_memory=True)
        fold_dir = os.path.join(args.exp_out_dir, f'fold{fold_id}')
        ckpt = os.path.join(fold_dir, 'best.pt')
        if not os.path.exists(ckpt):
            print(f"[Fold {fold_id}] missing checkpoint {ckpt}, skipping")
            continue
        model = timm.create_model(model_cfg['name'], pretrained=False, num_classes=len(np.unique(y)), drop_rate=float(model_cfg.get('dropout',0.0)))
        sd = torch.load(ckpt, map_location='cpu')['model']
        model.load_state_dict(sd)
        model.to(device)
        logits, tgts = collect_logits(model, dl_va, device)
        T = fit_temperature(logits, tgts)
        np.save(os.path.join(fold_dir, 'temp.npy'), np.array([T], dtype=np.float32))
        np.savez_compressed(os.path.join(fold_dir, f'oof_fold{fold_id}.npz'), logits=logits, labels=tgts)
        print(f"[Fold {fold_id}] saved temp.npy (T={T:.3f}) and oof_fold{fold_id}.npz")

if __name__ == '__main__':
    main()
