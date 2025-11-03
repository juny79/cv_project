# src/predict.py
import os, yaml, timm, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.transforms import get_valid_transforms  # v3 재수출 사용

def load_cfg(p):
    with open(p,'r') as f: return yaml.safe_load(f)

class TestDS(Dataset):
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        idc = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.id_col = idc[0] if idc else self.df.columns[0]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        stem = str(self.df.iloc[i, self.df.columns.get_loc(self.id_col)])
        # 확장자 유연 처리
        for ext in ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG',''):
            p = os.path.join(self.img_dir, stem+ext)
            if os.path.exists(p): 
                img = np.array(Image.open(p).convert('RGB'))
                break
        else:
            raise FileNotFoundError(stem)
        x = self.transform(image=img)['image']
        return x, stem

@torch.no_grad()
def predict_ensemble(cfg_path, tta=0):
    cfg = load_cfg(cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = get_valid_transforms(cfg['data']['img_size'])
    test = TestDS(cfg['paths']['sample_csv'], cfg['paths']['test_dir'], tfm)
    loader = DataLoader(test, batch_size=cfg['train'].get('batch_size', 32), shuffle=False, num_workers=2)

    # fold ckpts 수집
    ckpts = []
    out_dir = cfg['paths']['out_dir']
    for k in cfg['data'].get('folds',[0,1,2,3,4]):
        p = os.path.join(out_dir, f'fold{k}', 'best.pt')
        if os.path.exists(p): ckpts.append(p)
    assert ckpts, 'no checkpoints found'

    # 모델 구성 (num_classes 추정)
    state = torch.load(ckpts[0], map_location='cpu')
    num_classes = None
    for k,v in state['model'].items():
        if k.endswith('weight') and len(v.shape)==2:
            num_classes = v.shape[0]; break
    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)

    all_logits = []
    for ck in ckpts:
        w = torch.load(ck, map_location=device)
        model.load_state_dict(w['model'], strict=False)
        model.eval()

        fold_logits = []
        for xb, _ids in loader:
            xb = xb.to(device)
            logits = model(xb)

            if tta > 0:
                # 간단 TTA: H-Flip + 작은 scale shift
                # (전처리는 transforms_v3가 처리하므로 픽셀 변환만)
                xflip = torch.flip(xb, dims=[3])
                logits_flip = model(xflip)
                logits = (logits + logits_flip) / 2

            fold_logits.append(logits.detach().cpu())
        fold_logits = torch.cat(fold_logits)
        all_logits.append(fold_logits)

    mean_logits = torch.stack(all_logits).mean(0)
    preds = mean_logits.argmax(1).numpy()

    sub = pd.read_csv(cfg['paths']['sample_csv'])
    # 정답 컬럼명 찾기
    ycol = [c for c in sub.columns if c.lower() in ['label','target','class']]
    ycol = ycol[0] if ycol else sub.columns[-1]
    sub[ycol] = preds
    out_csv = os.path.join(cfg['paths']['out_dir'], 'submission.csv')
    sub.to_csv(out_csv, index=False)
    print(f'Saved → {out_csv}')
    return out_csv

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=1)
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
