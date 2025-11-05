# src/predict.py
import os, yaml, timm, torch
import numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.preprocessing import get_valid_transforms

def load_cfg(p): 
    with open(p,'r') as f: return yaml.safe_load(f)

class TestDS(Dataset):
    exts = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')
    def __init__(self, csv, img_dir, tf):
        self.df = pd.read_csv(csv); self.dir = img_dir; self.tf = tf
        idc = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.id_col = idc[0] if idc else self.df.columns[0]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        stem = str(self.df.iloc[i][self.id_col])
        img = None
        for e in self.exts:
            p = os.path.join(self.dir, stem+e)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB')); break
        if img is None: raise FileNotFoundError(stem)
        return self.tf(image=img)['image'], stem

def _tta_logits(model, xb):
    outs = []
    outs.append(model(xb))                             # 0°
    outs.append(model(torch.flip(xb, dims=[3])))       # hflip
    for k in [1,2,3]:                                  # 90/180/270
        outs.append(model(torch.rot90(xb, k=k, dims=(2,3))))
    return torch.stack(outs).mean(0)

@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths, data = cfg['paths'], cfg['data']
    img_size = int(data['img_size'])

    tf = get_valid_transforms(img_size)
    ds = TestDS(paths['sample_csv'], paths['test_dir'], tf)
    loader = DataLoader(ds, batch_size=int(cfg['train']['batch_size']),
                        shuffle=False, num_workers=2, pin_memory=True)

    # fold checkpoints
    folds = data.get('folds', [0,1,2,3,4])
    ckpts = []
    for k in folds:
        p = os.path.join(paths['out_dir'], f'fold{k}', 'best.pt')
        if os.path.exists(p): ckpts.append(p)
    assert ckpts, "no checkpoints found"

    # infer num_classes
    state0 = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    num_classes = None
    for k,v in state0['model'].items():
        if v.ndim==2 and k.endswith('weight'): num_classes = v.shape[0]; break
    assert num_classes is not None

    model = timm.create_model(cfg['model']['name'], pretrained=False,
                              num_classes=num_classes).to(device)
    model.eval()

    all_logits = []
    for ck in ckpts:
        w = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(w['model'], strict=False)
        fold_logits = []
        for xb, _ in loader:
            xb = xb.to(device)
            logits = _tta_logits(model, xb)           # TTA 적용
            fold_logits.append(logits.detach().cpu())
        all_logits.append(torch.cat(fold_logits))
    mean_logits = torch.stack(all_logits).mean(0)
    preds = torch.argmax(mean_logits, 1).numpy()

    sub = pd.read_csv(paths['sample_csv']).copy()
    ycol = [c for c in sub.columns if c.lower() in ['target','label','class']]
    ycol = ycol[0] if ycol else sub.columns[-1]
    sub[ycol] = preds
    out_csv = os.path.join(paths['out_dir'], 'submission.csv')
    os.makedirs(paths['out_dir'], exist_ok=True)
    sub[['ID', ycol]].to_csv(out_csv, index=False)     # ID,target만 저장
    print(f"Saved -> {out_csv}")

if __name__ == '__main__':
    main()
