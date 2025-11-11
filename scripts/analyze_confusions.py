import os, yaml, timm, torch
import numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from src.transforms import get_valid_transforms

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def load_cfg(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found for {stem} under {img_dir}")

class SimpleDS(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = transform
        idc = [c for c in self.df.columns if c.lower() in ['id','image_id','filename','image']]
        self.id_col = idc[0] if idc else self.df.columns[0]
        ycs = [c for c in self.df.columns if c.lower() in ['label','target','class']]
        self.y_col = ycs[0] if ycs else self.df.columns[-1]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_image_path(self.img_dir, r[self.id_col])
        img = np.array(Image.open(p).convert('RGB'))
        x = self.tf(image=img)['image']
        y = int(r[self.y_col])
        return x, y

@torch.no_grad()
def run_validation_inference(cfg_path, tta=0):
    cfg = load_cfg(cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = cfg['paths']; data = cfg['data']
    out_dir = paths.get('out_dir','./outputs'); os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(paths['train_csv'])
    ycol = [c for c in df.columns if c.lower() in ['label','target','class']]
    ycol = ycol[0] if ycol else df.columns[-1]
    y = df[ycol].astype(int).values
    num_classes = len(np.unique(y))

    tfm = get_valid_transforms(int(data['img_size']))
    skf = StratifiedKFold(n_splits=int(data.get('n_splits',5)), shuffle=True, random_state=cfg.get('seed',42))

    # Prepare model
    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    use_amp = True if device.startswith('cuda') else False

    # Storage
    y_true_all, y_pred_all = [], []

    for fold_id, (_, va_idx) in enumerate(skf.split(df, y)):
        ck = os.path.join(paths['out_dir'], f'fold{fold_id}', 'best.pt')
        if not os.path.exists(ck):
            print(f"[Warn] Missing checkpoint for fold{fold_id}, skipping")
            continue
        w = torch.load(ck, map_location=device, weights_only=False)
        try:
            model.load_state_dict(w['model'], strict=True)
        except RuntimeError:
            model.load_state_dict(w['model'], strict=False)
        model.eval()

        df_va = df.iloc[va_idx].reset_index(drop=True)
        ds_va = SimpleDS(df_va, paths['train_dir'], tfm)
        dl_va = DataLoader(ds_va, batch_size=int(cfg['train']['batch_size']), shuffle=False,
                           num_workers=0, pin_memory=device.startswith('cuda'))

        for xb, yb in dl_va:
            xb = xb.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(xb)
            preds = torch.argmax(logits, 1).cpu().numpy().tolist()
            y_true_all.extend(yb.numpy().tolist())
            y_pred_all.extend(preds)

    # Confusion matrix and reports
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=[f'true_{i}' for i in range(num_classes)], columns=[f'pred_{i}' for i in range(num_classes)])
    cm_csv = os.path.join(out_dir, 'val_confusion.csv')
    cm_df.to_csv(cm_csv)
    print(f"[OUT] Saved confusion matrix → {cm_csv}")

    # Per-class report
    report = classification_report(y_true_all, y_pred_all, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).transpose()
    rep_csv = os.path.join(out_dir, 'val_per_class_report.csv')
    rep_df.to_csv(rep_csv)
    print(f"[OUT] Saved per-class report → {rep_csv}")

    # Pair analysis for (3,7) and (4,14)
    pairs = [(3,7),(4,14)]
    rows = []
    for a,b in pairs:
        a_to_b = int(((np.array(y_true_all)==a) & (np.array(y_pred_all)==b)).sum())
        b_to_a = int(((np.array(y_true_all)==b) & (np.array(y_pred_all)==a)).sum())
        a_total = int((np.array(y_true_all)==a).sum())
        b_total = int((np.array(y_true_all)==b).sum())
        rows.append({
            'pair': f'{a}-{b}',
            'a': a, 'b': b,
            'a_to_b': a_to_b,
            'b_to_a': b_to_a,
            'a_total': a_total,
            'b_total': b_total,
            'a_miss_rate': (a_to_b / a_total) if a_total>0 else 0.0,
            'b_miss_rate': (b_to_a / b_total) if b_total>0 else 0.0,
        })
    pair_df = pd.DataFrame(rows)
    pair_csv = os.path.join(out_dir, 'val_pair_confusions.csv')
    pair_df.to_csv(pair_csv, index=False)
    print(f"[OUT] Saved pair confusions → {pair_csv}")

    return {
        'cm_csv': cm_csv,
        'rep_csv': rep_csv,
        'pair_csv': pair_csv,
    }

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=0)
    args = ap.parse_args()
    run_validation_inference(args.config, tta=args.tta)
