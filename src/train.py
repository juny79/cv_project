# src/train.py  (cache-enabled + safe pretrained fallback + 타입 가드)
import os, yaml, timm, torch, random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from src.engine import train_one_epoch, valid_one_epoch, EarlyStopper
from src.transforms import get_train_transforms, get_valid_transforms

# ------------------ 공통 유틸 ------------------
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_device(name):
    if name == 'cuda' and torch.cuda.is_available(): return 'cuda'
    if name == 'auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'

def load_cfg(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def ensure_folds_from_df(df, label_col, n_splits=5, seed=42):
    out = df.copy()
    if 'fold' in out.columns and not out['fold'].isna().any():
        return out
    y = out[label_col].astype(str).values
    y_enc, _ = pd.factorize(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(out), -1, dtype=int)
    for k, (_, va_idx) in enumerate(skf.split(np.zeros(len(y_enc)), y_enc)):
        folds[va_idx] = k
    out['fold'] = folds
    return out

def get_scheduler(cfg, optimizer, steps_per_epoch):
    sch = str(cfg['train'].get('scheduler', 'cosine')).lower()
    epochs = int(cfg['train'].get('epochs', 10))
    if sch == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(optimizer, max_lr=float(cfg['train'].get('lr', 3e-4)),
                          steps_per_epoch=steps_per_epoch, epochs=epochs)
    else:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=epochs)

def save_checkpoint(path, model, optimizer, scaler, epoch, metrics, classes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'metrics': metrics,
        'classes': list(classes)
    }, path)

def parse_train_hparams(cfg):
    tr = cfg['train']
    # 모든 하이퍼파라미터를 안전 캐스팅
    return {
        'epochs': int(tr.get('epochs', 10)),
        'batch_size': int(tr.get('batch_size', 32)),
        'lr': float(tr.get('lr', 3e-4)),
        'weight_decay': float(tr.get('weight_decay', 1e-4)),
        'scheduler': str(tr.get('scheduler', 'cosine')).lower(),
        'mix_precision': bool(tr.get('mix_precision', True)),
        'early_stop_patience': int(tr.get('early_stop_patience', 3)),
        'class_weight': str(tr.get('class_weight', 'none')).lower(),
    }

# ---------- 안전 모델 생성기: pretrained 없으면 자동 폴백 ----------
_ALIAS = {
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "efficientnetv2_m": "tf_efficientnetv2_m",
    "efficientnetv2_l": "tf_efficientnetv2_l",
}

def create_timm_model_safe(name: str, pretrained: bool, num_classes: int, device: str):
    try:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        return model.to(device)
    except RuntimeError as e:
        msg = str(e)
        if "No pretrained weights exist" in msg and pretrained:
            if name in _ALIAS:
                alias = _ALIAS[name]
                print(f"[WARN] '{name}' pretrained 없음 → '{alias}'로 재시도")
                try:
                    model = timm.create_model(alias, pretrained=True, num_classes=num_classes)
                    return model.to(device)
                except Exception as e2:
                    print(f"[WARN] '{alias}'도 실패: {e2}. → pretrained=False로 폴백")
            else:
                print(f"[WARN] '{name}' pretrained 없음 → pretrained=False로 폴백")
            model = timm.create_model(name, pretrained=False, num_classes=num_classes)
            return model.to(device)
        raise

# =================== A) 캐시(.pt) 직접 학습 ===================
def run_training_with_cache(cfg, device):
    paths = cfg.get('paths', {})
    processed_train = paths.get('processed_train', '/root/cv_project/data/processed_v2/train_processed.pt')
    out_dir = paths.get('out_dir', './outputs'); os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(processed_train):
        raise FileNotFoundError(f"Processed train file not found: {processed_train}. Run preprocessing pipeline to generate it or set paths.processed_train in config.")

    state = torch.load(processed_train, map_location='cpu')
    images = state['images']                # [N,3,H,W]
    labels = state['labels']                # [N]

    if labels.numel() == 0:
        raise ValueError("캐시 텐서에 labels 가 없습니다. 학습용 캐시를 생성했는지 확인하세요.")

    y_np = labels.detach().cpu().numpy()
    if not np.issubdtype(y_np.dtype, np.integer):
        y_enc, classes = pd.factorize(y_np.astype(str))
        labels = torch.from_numpy(y_enc).long()
        classes = np.asarray(classes)
    else:
        classes = np.unique(y_np)

    n_splits = int(cfg['data'].get('n_splits', 5))
    folds = cfg['data'].get('folds', list(range(n_splits)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get('seed', 42))
    y_for_split = labels.detach().cpu().numpy()
    split_map = {k: pair for k, pair in enumerate(skf.split(np.zeros(len(y_for_split)), y_for_split))}

    hp = parse_train_hparams(cfg)
    all_fold_metrics = {}
    num_classes = int(len(classes))
    bs = hp['batch_size']

    for fold in folds:
        print(f"\n========== Fold {fold} (cache) ==========")
        tr_idx, va_idx = split_map[fold]

        tr_ds = TensorDataset(images[tr_idx], labels[tr_idx])
        va_ds = TensorDataset(images[va_idx], labels[va_idx])

        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                               num_workers=int(cfg.get('num_workers', 2)), pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                               num_workers=int(cfg.get('num_workers', 2)), pin_memory=True)

        model = create_timm_model_safe(cfg['model']['name'],
                                       bool(cfg['model'].get('pretrained', True)),
                                       num_classes, device)

        if hp['class_weight'] == 'balanced':
            w = compute_class_weight(class_weight='balanced',
                                     classes=np.arange(num_classes),
                                     y=y_for_split[tr_idx])
            weight = torch.tensor(w, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=hp['lr'],
                                      weight_decay=hp['weight_decay'])
        scaler = torch.cuda.amp.GradScaler(enabled=hp['mix_precision'] and device=='cuda')
        scheduler = get_scheduler(cfg, optimizer, steps_per_epoch=len(tr_loader))
        early = EarlyStopper(patience=hp['early_stop_patience'], mode='max')

        best_path = os.path.join(out_dir, f'fold{fold}', 'best.pt')
        best_f1 = -1.0
        epochs = hp['epochs']

        for epoch in range(1, epochs+1):
            print(f"[Fold {fold}] Epoch {epoch}/{epochs} (cache)")
            tr_loss, tr_f1 = train_one_epoch(model, tr_loader, criterion, optimizer, device, scaler)
            va_loss, va_f1 = valid_one_epoch(model, va_loader, criterion, device)
            scheduler.step()
            print(f"Train | loss={tr_loss:.4f} f1={tr_f1:.4f}  ||  Valid | loss={va_loss:.4f} f1={va_f1:.4f}")

            if early.step(va_f1):
                best_f1 = va_f1
                save_checkpoint(best_path, model, optimizer, scaler, epoch,
                                metrics={'train_loss':tr_loss, 'train_f1':tr_f1, 'valid_loss':va_loss, 'valid_f1':va_f1},
                                classes=classes)
                print(f"✓ Best updated (F1={va_f1:.4f}) → {best_path}")

            if early.should_stop():
                print("Early stopping triggered.")
                break

        all_fold_metrics[fold] = best_f1

    print("\n==== Summary (Best F1 per fold, cache) ====")
    for k,v in all_fold_metrics.items():
        print(f"Fold {k}: F1={v:.4f}")
    print("Saved checkpoints under:", out_dir)

# =================== B) 실시간 전처리 학습 ===================
def run_training_realtime(cfg, device):
    from src.datasets import DocumentDataset

    paths = cfg.get('paths', {})
    train_csv = paths.get('train_csv')
    train_dir = paths.get('train_dir')
    out_dir   = paths.get('out_dir', './outputs')
    if train_csv is None or not os.path.exists(train_csv):
        raise FileNotFoundError(f"train_csv not found or not set in config.paths: {train_csv}")
    if train_dir is None or not os.path.exists(train_dir):
        raise FileNotFoundError(f"train_dir not found or not set in config.paths: {train_dir}")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    cand_lb = [c for c in df.columns if c.lower() in ['label','target','class','y']]
    label_col = cand_lb[0] if cand_lb else df.columns[-1]
    df = ensure_folds_from_df(df, label_col, n_splits=int(cfg['data'].get('n_splits',5)), seed=cfg.get('seed',42))

    classes_global = np.unique(df[label_col].astype(str).values)
    num_classes = len(classes_global)

    img_size = int(cfg['data'].get('img_size', 640))
    t_train = get_train_transforms(img_size)
    t_valid = get_valid_transforms(img_size)

    folds = cfg['data'].get('folds', list(range(int(cfg['data'].get('n_splits', 5)))))
    hp = parse_train_hparams(cfg)
    all_fold_metrics = {}

    for fold in folds:
        print(f"\n========== Fold {fold} (realtime) ==========")
        tr_idx = df.index[df['fold'] != fold].tolist()
        va_idx = df.index[df['fold'] == fold].tolist()

        id_cols = [c for c in df.columns if 'id' in c.lower() or 'file' in c.lower()]
        id_col = id_cols[0] if id_cols else df.columns[0]

        tmp_dir = os.path.join(out_dir, '_tmp_csv'); os.makedirs(tmp_dir, exist_ok=True)
        tr_csv = os.path.join(tmp_dir, f'train_fold{fold}.csv')
        va_csv = os.path.join(tmp_dir, f'valid_fold{fold}.csv')

        tr_df = df.loc[tr_idx, [id_col, label_col]].rename(columns={label_col:'label'})
        va_df = df.loc[va_idx, [id_col, label_col]].rename(columns={label_col:'label'})

        tr_df['label'] = pd.Categorical(tr_df['label'].astype(str), categories=classes_global).codes
        va_df['label'] = pd.Categorical(va_df['label'].astype(str), categories=classes_global).codes

        tr_df.to_csv(tr_csv, index=False)
        va_df.to_csv(va_csv, index=False)

        tr_ds = DocumentDataset(tr_csv, train_dir, transform=t_train, has_label=True, id_col=id_col, label_col='label')
        va_ds = DocumentDataset(va_csv, train_dir, transform=t_valid, has_label=True, id_col=id_col, label_col='label')

        tr_loader = DataLoader(tr_ds, batch_size=hp['batch_size'], shuffle=True,
                               num_workers=int(cfg.get('num_workers', 2)), pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=hp['batch_size'], shuffle=False,
                               num_workers=int(cfg.get('num_workers', 2)), pin_memory=True)

        model = create_timm_model_safe(cfg['model']['name'],
                                       bool(cfg['model'].get('pretrained', True)),
                                       num_classes, device)

        if hp['class_weight'] == 'balanced':
            y_tr = tr_df['label'].values
            w = compute_class_weight(class_weight='balanced',
                                     classes=np.arange(num_classes),
                                     y=y_tr)
            weight = torch.tensor(w, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=hp['lr'],
                                      weight_decay=hp['weight_decay'])
        scaler = torch.cuda.amp.GradScaler(enabled=hp['mix_precision'] and device=='cuda')
        scheduler = get_scheduler(cfg, optimizer, steps_per_epoch=len(tr_loader))
        early = EarlyStopper(patience=hp['early_stop_patience'], mode='max')

        best_path = os.path.join(out_dir, f'fold{fold}', 'best.pt')
        best_f1 = -1.0
        epochs = hp['epochs']

        for epoch in range(1, epochs+1):
            print(f"[Fold {fold}] Epoch {epoch}/{epochs} (realtime)")
            tr_loss, tr_f1 = train_one_epoch(model, tr_loader, criterion, optimizer, device, scaler)
            va_loss, va_f1 = valid_one_epoch(model, va_loader, criterion, device)
            scheduler.step()
            print(f"Train | loss={tr_loss:.4f} f1={tr_f1:.4f}  ||  Valid | loss={va_loss:.4f} f1={va_f1:.4f}")

            if early.step(va_f1):
                best_f1 = va_f1
                save_checkpoint(best_path, model, optimizer, scaler, epoch,
                                metrics={'train_loss':tr_loss, 'train_f1':tr_f1, 'valid_loss':va_loss, 'valid_f1':va_f1},
                                classes=classes_global)
                print(f"✓ Best updated (F1={va_f1:.4f}) → {best_path}")

            if early.should_stop():
                print("Early stopping triggered.")
                break

        all_fold_metrics[fold] = best_f1

    print("\n==== Summary (Best F1 per fold, realtime) ====")
    for k,v in all_fold_metrics.items():
        print(f"Fold {k}: F1={v:.4f}")
    print("Saved checkpoints under:", out_dir)

# ------------------ entry ------------------
def main():
    ap = ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--use-cache', action='store_true', help='전처리 캐시(.pt)에서 바로 학습')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get('seed', 42))
    device = auto_device(cfg.get('device', 'auto'))

    if args.use_cache:
        run_training_with_cache(cfg, device)
    else:
        run_training_realtime(cfg, device)

if __name__ == '__main__':
    main()
