# src/train.py
import os, yaml, timm, math, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter
from tqdm import tqdm

from src.transforms import get_train_transforms, get_valid_transforms

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_cfg(path):
    with open(path,'r') as f: return yaml.safe_load(f)

def auto_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name

# ---------- dataset / cache loader ----------
class TrainCsvDS(Dataset):
    exts = ['.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','']
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.tfm = transform
        cols = self.df.columns.str.lower()
        self.id_col = self.df.columns[np.where(cols=='id')[0][0]]
        self.y_col = self.df.columns[np.where(np.isin(cols, ['target','label','class']))[0][0]]
    def __len__(self): return len(self.df)
    def _resolve(self, stem):
        for e in self.exts:
            p = os.path.join(self.img_dir, str(stem)+e)
            if os.path.exists(p): return p
        raise FileNotFoundError(stem)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = self._resolve(r[self.id_col])
        import numpy as np
        from PIL import Image
        img = np.array(Image.open(path).convert('RGB'))
        x = self.tfm(image=img)['image']
        y = int(r[self.y_col])
        return x, y

class CachedTensorDS(Dataset):
    def __init__(self, pt_path):
        obj = torch.load(pt_path, map_location='cpu')
        self.x = obj['images']; self.y = obj.get('labels', None)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i):
        if self.y is None: return self.x[i], -1
        return self.x[i], int(self.y[i])

# ---------- utils ----------
def build_model(name, num_classes, pretrained=True, dropout=0.0):
    # 안전 alias
    alias = {
        'efficientnetv2_s': 'tf_efficientnetv2_s',
        'tf_efficientnetv2_s': 'tf_efficientnetv2_s',
        'convnext_tiny': 'convnext_tiny',
        'resnet50': 'resnet50',
        'resnet34': 'resnet34',
    }
    mname = alias.get(name, name)
    model = timm.create_model(mname, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    return model

class EarlyStopper:
    def __init__(self, patience=3, mode='max'):
        self.patience = patience; self.best = None; self.count = 0; self.mode = mode
    def __call__(self, score):
        if self.best is None or (score > self.best if self.mode=='max' else score < self.best):
            self.best = score; self.count = 0; return True
        self.count += 1; return False
    def should_stop(self): return self.count >= self.patience

def compute_class_weight(y, n_classes):
    cnt = Counter(y); total = sum(cnt.values())
    w = torch.ones(n_classes, dtype=torch.float32)
    for k in range(n_classes):
        w[k] = total / (n_classes * max(1, cnt.get(k, 0)))
    return w

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None):
    model.train()
    losses, preds_all, tgts_all = [], [], []
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(xb)
            loss = criterion(logits, yb)
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(loss.item())
        preds_all.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        tgts_all.extend(yb.detach().cpu().tolist())
    f1 = f1_score(tgts_all, preds_all, average='macro')
    return float(np.mean(losses)), f1

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds_all, tgts_all = [], [], []
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        preds_all.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        tgts_all.extend(yb.detach().cpu().tolist())
    f1 = f1_score(tgts_all, preds_all, average='macro')
    return float(np.mean(losses)), f1

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--use-cache', action='store_true')
    args = ap.parse_args()

    cfg = load_cfg(args.config); set_seed(cfg.get('seed', 42))
    device = auto_device(cfg.get('device', 'auto'))
    paths, data_cfg, train_cfg = cfg['paths'], cfg['data'], cfg['train']

    img_size = int(data_cfg['img_size'])
    folds = data_cfg.get('folds', [0,1,2,3,4])
    n_splits = int(data_cfg.get('n_splits', 5))

    # 데이터/라벨 로딩(캐시 우선)
    if args.use-cache and os.path.exists(paths['processed_train']):
        full_ds = CachedTensorDS(paths['processed_train'])
        # 라벨이 텐서에 포함되어 있으므로 그대로 사용
        y_all = [int(full_ds[i][1]) for i in range(len(full_ds))]
        X = list(range(len(full_ds)))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get('seed',42))
        splits = list(splitter.split(X, y_all))
        def subset(ds, idxs):
            class IdxDS(Dataset):
                def __init__(self, base, idl): self.b=base; self.id=idl
                def __len__(self): return len(self.id)
                def __getitem__(self, i): return self.b[self.id[i]]
            return IdxDS(ds, idxs)
        using_cache = True
    else:
        # csv 기반
        df = pd.read_csv(paths['train_csv'])
        cols = df.columns.str.lower()
        id_col = df.columns[np.where(cols=='id')[0][0]]
        y_col  = df.columns[np.where(np.isin(cols, ['target','label','class']))[0][0]]
        y_all = df[y_col].tolist()
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get('seed',42))
        splits = list(splitter.split(df, y_all))
        tr_tfm = get_train_transforms(img_size); va_tfm = get_valid_transforms(img_size)
        using_cache = False

    num_classes = len(set(y_all))
    out_dir = paths.get('out_dir', './outputs'); os.makedirs(out_dir, exist_ok=True)

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        if fold_id not in folds: continue
        print(f"\n========== Fold {fold_id} ({'cache' if using_cache else 'csv'}) ==========")

        if using_cache:
            tr_ds = subset(full_ds, tr_idx); va_ds = subset(full_ds, va_idx)
        else:
            tr_ds = TrainCsvDS(paths['train_csv'], paths['train_dir'], tr_tfm)
            va_ds = TrainCsvDS(paths['train_csv'], paths['train_dir'], va_tfm)
            # 인덱스 서브셋
            def take(ds, idxs):
                class IdxDS(Dataset):
                    def __init__(self, base, idl): self.b=base; self.id=idl
                    def __len__(self): return len(self.id)
                    def __getitem__(self, i): return self.b[self.id[i]]
                return IdxDS(ds, idxs)
            tr_ds = take(tr_ds, tr_idx); va_ds = take(va_ds, va_idx)

        tr_loader = DataLoader(tr_ds, batch_size=int(train_cfg['batch_size']), shuffle=True,
                               num_workers=int(cfg.get('num_workers',2)), pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=int(train_cfg['batch_size']), shuffle=False,
                               num_workers=int(cfg.get('num_workers',2)), pin_memory=True)

        model = build_model(cfg['model']['name'], num_classes=num_classes,
                            pretrained=bool(cfg['model'].get('pretrained', True)),
                            dropout=float(cfg['model'].get('dropout', 0.0))).to(device)

        # class weight
        weight = None
        if train_cfg.get('class_weight','').lower()=='balanced':
            weight = compute_class_weight([y_all[i] for i in tr_idx], num_classes).to(device)

        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(train_cfg['lr']),
                                      weight_decay=float(train_cfg['weight_decay']))

        # scheduler
        if train_cfg.get('scheduler','cosine') == 'cosine':
            tmax = int(train_cfg['epochs'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        else:
            scheduler = None

        scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get('mix_precision', True)))
        es = EarlyStopper(patience=int(train_cfg.get('early_stop_patience',3)), mode='max')
        grad_clip = float(train_cfg.get('grad_clip_norm', 0.0)) or None

        best_f1 = -1.0
        fold_dir = os.path.join(out_dir, f'fold{fold_id}'); os.makedirs(fold_dir, exist_ok=True)

        for ep in range(int(train_cfg['epochs'])):
            tr_loss, tr_f1 = train_one_epoch(model, tr_loader, criterion, optimizer, device, scaler, grad_clip)
            va_loss, va_f1 = valid_one_epoch(model, va_loader, criterion, device)
            if scheduler: scheduler.step()

            print(f"[Fold {fold_id}] Ep {ep+1}/{train_cfg['epochs']} | "
                  f"Train loss {tr_loss:.4f} f1 {tr_f1:.4f} | "
                  f"Valid loss {va_loss:.4f} f1 {va_f1:.4f}")

            if es(va_f1):
                best_f1 = va_f1
                torch.save({
                    'model': model.state_dict(),
                    'classes': list(sorted(set(y_all))),
                    'cfg': cfg
                }, os.path.join(fold_dir, 'best.pt'))
            if es.should_stop():
                print(f"Early stop at epoch {ep+1}")
                break

        print(f"[Fold {fold_id}] Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
