# src/predict.py
import os, yaml, timm, torch
import numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.transforms import get_valid_transforms

def load_cfg(p):
    with open(p,'r') as f: return yaml.safe_load(f)

class TestDS(Dataset):
    EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        cols = self.df.columns.str.lower()
        self.id_col = self.df.columns[np.where(cols=='id')[0][0]]
        self.dir = img_dir; self.tfm = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        stem = str(self.df.iloc[i][self.id_col])
        img = None
        for e in self.EXTS:
            p = os.path.join(self.dir, stem+e)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB')); break
        if img is None: raise FileNotFoundError(stem)
        x = self.tfm(image=img)['image']
        return x, stem

def _safe_load_ckpt(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as e1:
        try:
            from torch.serialization import safe_globals, add_safe_globals
            import numpy as _np
            try: add_safe_globals([_np._core.multiarray.scalar])  # type: ignore[attr-defined]
            except Exception: pass
            with safe_globals():
                try: return torch.load(path, map_location=map_location, weights_only=False)
                except TypeError: return torch.load(path, map_location=map_location)
        except Exception as e2:
            raise RuntimeError(f"ckpt load failed: {e1}\n{e2}")

@torch.no_grad()
def predict_ensemble(cfg_path, tta=1):
    cfg = load_cfg(cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths, data_cfg, train_cfg = cfg['paths'], cfg['data'], cfg.get('train', {})
    out_dir = paths.get('out_dir', './outputs'); os.makedirs(out_dir, exist_ok=True)

    img_size = int(data_cfg.get('img_size', 640))
    tfm = get_valid_transforms(img_size)

    test = TestDS(paths['sample_csv'], paths['test_dir'], tfm)
    loader = DataLoader(test, batch_size=int(train_cfg.get('batch_size',32)),
                        shuffle=False, num_workers=int(cfg.get('num_workers',2)),
                        pin_memory=True)

    folds = data_cfg.get('folds', [0,1,2,3,4])
    ckpts = [os.path.join(out_dir, f'fold{k}', 'best.pt') for k in folds if os.path.exists(os.path.join(out_dir, f'fold{k}', 'best.pt'))]
    if not ckpts: raise FileNotFoundError("no fold best.pt found")

    state0 = _safe_load_ckpt(ckpts[0], map_location='cpu')
    num_classes = None
    for k,v in state0['model'].items():
        if k.endswith('weight') and hasattr(v,'shape') and len(v.shape)==2:
            num_classes = int(v.shape[0]); break
    if num_classes is None: raise RuntimeError("infer num_classes failed")

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)

    all_logits = []
    for ck in ckpts:
        w = _safe_load_ckpt(ck, map_location=device)
        try:
            model.load_state_dict(w['model'], strict=False)
        except RuntimeError:
            msd = model.state_dict(); matched = {k:v for k,v in w['model'].items() if k in msd and msd[k].shape==v.shape}
            msd.update(matched); model.load_state_dict(msd, strict=False)

        model.eval()
        fold_logits = []
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            if int(tta) > 0:
                xflip = torch.flip(xb, dims=[3])
                logits = (logits + model(xflip))/2
            fold_logits.append(logits.detach().cpu())
        all_logits.append(torch.cat(fold_logits))

    mean_logits = torch.stack(all_logits).mean(0)
    preds = torch.argmax(mean_logits, dim=1).numpy()

    sub = pd.read_csv(paths['sample_csv'])
    id_col = sub.columns[np.where(sub.columns.str.lower()=='id')[0][0]]
    # 제출 컬럼 자동 탐색 → 없으면 target 생성
    ycols = [c for c in sub.columns if c.lower() in ['target','label','class']]
    ycol = ycols[0] if ycols else 'target'
    out = pd.DataFrame({id_col: sub[id_col].values, ycol: preds})
    out.to_csv(os.path.join(out_dir, 'submission.csv'), index=False)
    print(f"[predict] Saved → {os.path.join(out_dir,'submission.csv')}")
    return True

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=1)
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
