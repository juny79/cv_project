import os, yaml, timm, torch, random
import numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from PIL import Image
from tqdm import tqdm

from src.transforms import get_train_transforms, get_valid_transforms

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def auto_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name

def load_cfg(path):
    try:
        with open(path,'r') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Configuration must be a dictionary")
        return cfg
    except (yaml.YAMLError, OSError) as e:
        raise RuntimeError(f"Failed to load config from {path}: {str(e)}")

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found for {stem} under {img_dir}")

class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        idc = [c for c in self.df.columns if c.lower() in ['id','image_id','filename','image']]
        self.id_col = idc[0] if idc else self.df.columns[0]
        ycs = [c for c in self.df.columns if c.lower() in ['label','target','class']]
        self.y_col = ycs[0] if ycs else self.df.columns[-1]

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_image_path(self.img_dir, r[self.id_col])
        img = np.array(Image.open(p).convert('RGB'))
        x = self.transform(image=img)['image']
        y = int(r[self.y_col])
        return x, y


class RepeatAugDataset(Dataset):
    """Wrap a base dataset and repeat indices so that each epoch yields
    `multiplier * len(base)` samples. Each repeated access will call the
    underlying dataset which applies stochastic transforms, producing
    augmented variants of the same images.

    Use-case: expand 1,570 -> 6,280 by setting multiplier=4.
    """
    def __init__(self, base_ds: Dataset, multiplier: int = 1):
        self.base = base_ds
        self.multiplier = max(1, int(multiplier))
    def __len__(self):
        return len(self.base) * self.multiplier
    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, mixup_fn=None, batch_print_interval=20):
    """Train for one epoch.
    Added lightweight batch progress logging (every `batch_print_interval` batches) to
    help debug perceived stalls when stdout is redirected (tqdm may not flush).
    """
    model.train()
    losses, preds_all, tgts_all = [], [], []
    for bi, (xb, yb) in enumerate(tqdm(loader, leave=False)):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if mixup_fn is not None:
            xb, yb_mix = mixup_fn(xb, yb)
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                logits = model(xb)
                loss = criterion(logits, yb_mix)
        else:
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                logits = model(xb)
                loss = criterion(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        losses.append(loss.item())
        preds_all.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        tgts_all.extend(yb.detach().cpu().tolist())

        if batch_print_interval and bi % batch_print_interval == 0:
            print(f"[Train] Batch {bi}/{len(loader)} loss={losses[-1]:.4f}")
    f1 = f1_score(tgts_all, preds_all, average='macro')
    return float(np.mean(losses)), f1

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds_all, tgts_all = [], [], []
    for xb, yb in tqdm(loader, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        preds_all.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        tgts_all.extend(yb.detach().cpu().tolist())
    f1 = f1_score(tgts_all, preds_all, average='macro')
    return float(np.mean(losses)), f1

class MixupWrapper:
    def __init__(self, alpha=0.2, num_classes=17):
        self.alpha = alpha
        self.num_classes = num_classes
    def __call__(self, x, y):
        if self.alpha <= 0: return x, y
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        y1 = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        y2 = y1[idx]
        return mixed_x, lam * y1 + (1 - lam) * y2

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get('seed',42))
    device = auto_device(cfg.get('device','auto'))

    paths = cfg['paths']; data = cfg['data']; trn = cfg['train']; model_cfg = cfg['model']
    out_dir = paths.get('out_dir','./outputs'); os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(paths['train_csv'])
    ycol = [c for c in df.columns if c.lower() in ['label','target','class']]
    ycol = ycol[0] if ycol else df.columns[-1]
    y = df[ycol].astype(int).values
    num_classes = len(np.unique(y))

    weights = None
    if str(trn.get('class_weight','')).lower() == 'balanced':
        cls_w = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        # Optional: apply per-class weight boosts from config, e.g., {'3':1.2,'7':1.15}
        boosts = cfg.get('class_weight_boost', {}) or {}
        # accept int keys as well
        for k, mul in boosts.items():
            try:
                idx = int(k)
                if 0 <= idx < len(cls_w):
                    cls_w[idx] *= float(mul)
            except Exception:
                continue
        weights = torch.tensor(cls_w, dtype=torch.float32).to(device)

    skf = StratifiedKFold(n_splits=data.get('n_splits',5), shuffle=True, random_state=cfg.get('seed',42))

    # 실험 로그 저장용 리스트
    log_rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(df, y)):
        print(f"\n========== Fold {fold_id} ==========")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        tr_tf = get_train_transforms(data['img_size'])
        va_tf = get_valid_transforms(data['img_size'])

        ds_tr = DocumentDataset(df_tr, paths['train_dir'], tr_tf)
        ds_va = DocumentDataset(df_va, paths['train_dir'], va_tf)

        aug_mult = int(data.get('aug_multiplier', 1))
        if aug_mult > 1:
            print(f"Repeat-augmentation enabled: multiplier={aug_mult} -> "
                  f"train samples {len(ds_tr)} -> {len(ds_tr) * aug_mult}")
            ds_tr = RepeatAugDataset(ds_tr, multiplier=aug_mult)

        num_workers = min(int(cfg.get('num_workers', 0)), os.cpu_count() or 1)
        persistent_workers = num_workers > 0
        dl_tr = DataLoader(ds_tr, batch_size=int(trn['batch_size']), shuffle=True,
                          num_workers=num_workers, pin_memory=True, 
                          persistent_workers=persistent_workers, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=int(trn['batch_size']), shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=persistent_workers)

        model = timm.create_model(model_cfg['name'],
                                  pretrained=bool(model_cfg.get('pretrained', True)),
                                  num_classes=num_classes, drop_rate=float(model_cfg.get('dropout',0.0))).to(device)

        mixup_fn = MixupWrapper(alpha=float(trn.get('mixup_alpha',0)), num_classes=num_classes) if trn.get('mixup_alpha',0) else None
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(model_cfg.get('label_smoothing',0.0)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(trn['lr']), weight_decay=float(trn['weight_decay']))

        sched_name = str(trn.get('scheduler','cosine')).lower()
        if sched_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(trn['epochs']))
        elif sched_name == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(trn['lr']),
                                                            epochs=int(trn['epochs']), steps_per_epoch=len(dl_tr))
        else:
            scheduler = None

        use_amp = bool(trn.get('amp', True)) and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
        best_f1, bad = -1.0, 0
        patience = int(trn.get('early_stop_patience',3))

        for ep in range(1, int(trn['epochs'])+1):
            tr_loss, tr_f1 = train_one_epoch(model, dl_tr, criterion, optimizer, device, scaler, mixup_fn,
                                             batch_print_interval=int(cfg.get('batch_print_interval', 20)))
            va_loss, va_f1 = valid_one_epoch(model, dl_va, criterion, device)
            if scheduler is not None and sched_name != 'onecycle':
                scheduler.step()

            print(f"[Fold {fold_id}] Ep {ep}/{trn['epochs']} | Train loss {tr_loss:.4f} f1 {tr_f1:.4f} | "
                  f"Valid loss {va_loss:.4f} f1 {va_f1:.4f}")

            # 로그 저장
            log_rows.append({
                'fold': fold_id,
                'epoch': ep,
                'train_loss': tr_loss,
                'train_f1': tr_f1,
                'valid_loss': va_loss,
                'valid_f1': va_f1,
                'best_f1': max(best_f1, va_f1),
            })

            if va_f1 > best_f1:
                best_f1, bad = va_f1, 0
                fold_dir = os.path.join(out_dir, f'fold{fold_id}'); os.makedirs(fold_dir, exist_ok=True)
                torch.save({'model': model.state_dict()}, os.path.join(fold_dir, 'best.pt'))
                print(f"[Fold {fold_id}] ✓ Best updated (F1={best_f1:.4f})")
            else:
                bad += 1
                if bad >= patience:
                    print(f"[Fold {fold_id}] Early stop. Best F1={best_f1:.4f}")
                    break

        print(f"[Fold {fold_id}] Best F1: {best_f1:.4f}")

        # ====================== Temperature Scaling & OOF Collection ======================
        # Helper: collect raw logits over validation loader
        @torch.no_grad()
        def collect_logits(eval_model, loader, device):
            eval_model.eval()
            all_logits, all_tgts = [], []
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                lg = eval_model(xb).detach().cpu().numpy()
                all_logits.append(lg)
                all_tgts.append(yb.numpy())
            return np.concatenate(all_logits, 0), np.concatenate(all_tgts, 0)

        # Simple temperature fitting minimizing NLL (cross-entropy)
        def fit_temperature(logits, labels, init_T=1.0, lr=0.01, iters=200):
            import torch, torch.nn.functional as F
            x = torch.tensor(logits, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            T = torch.tensor([init_T], dtype=torch.float32, requires_grad=True)
            opt = torch.optim.LBFGS([T], lr=lr, max_iter=iters)

            def closure():
                opt.zero_grad()
                z = x / T.clamp_min(1e-4)
                loss = F.cross_entropy(z, y)
                loss.backward()
                return loss
            opt.step(closure)
            return float(T.detach().clamp(0.3, 5.0).item())

        eval_model = model  # If EMA used in future: eval_model = ema.ema.module
        va_logits, va_tgts = collect_logits(eval_model, dl_va, device)
        T = fit_temperature(va_logits, va_tgts)
        fold_dir = os.path.join(out_dir, f'fold{fold_id}')
        os.makedirs(fold_dir, exist_ok=True)
        np.save(os.path.join(fold_dir, 'temp.npy'), np.array([T], dtype=np.float32))
        np.savez_compressed(os.path.join(fold_dir, f'oof_fold{fold_id}.npz'), logits=va_logits, labels=va_tgts)
        print(f"[Fold {fold_id}] saved temperature T={T:.3f} and OOF logits.")
        # =============================================================================

    # 모든 fold/epoch 로그를 CSV로 저장
    log_df = pd.DataFrame(log_rows)
    log_csv = os.path.join(out_dir, 'train_log.csv')
    log_df.to_csv(log_csv, index=False)
    print(f"[LOG] Saved training log to {log_csv}")

if __name__ == '__main__':
    main()
