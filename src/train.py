import os, yaml, timm, torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from src.transforms import get_train_transforms, get_valid_transforms
from src.datasets  import DocumentDataset
from src.engine    import train_one_epoch, valid_one_epoch, EarlyStopper

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_device(name):
    if name == 'cuda' and torch.cuda.is_available(): return 'cuda'
    if name == 'auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def build_model(cfg, num_classes):
    model = timm.create_model(cfg['model']['name'],
                              pretrained=cfg['model']['pretrained'],
                              num_classes=num_classes,
                              drop_rate=cfg['model'].get('dropout', 0.0))
    return model

def main(config_path="configs/base.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg['seed'])
    device = auto_device(cfg['device'])
    os.makedirs(cfg['paths']['out_dir'], exist_ok=True)

    # 데이터 로드
    train_df  = pd.read_csv(cfg['paths']['train_csv'])
    folds_df  = pd.read_csv(cfg['paths']['folds_csv'])
    # folds_df 안에 label/target 컬럼명이 다를 수 있으므로 통일
    if 'label' not in folds_df.columns:
        if 'target' in folds_df.columns: folds_df = folds_df.rename(columns={'target':'label'})
        elif 'class' in folds_df.columns: folds_df = folds_df.rename(columns={'class':'label'})
    df = folds_df  # id, label, fold 구조라고 가정 (EDA 단계에서 정리한 것 사용)

    n_classes = int(df['label'].nunique())
    print(f"Classes: {n_classes}, device: {device}")

    # 변환
    img_size = cfg['data']['img_size']
    t_train = get_train_transforms(img_size)
    t_valid = get_valid_transforms(img_size)

    # 폴드 루프
    for fold in cfg['data']['folds']:
        print(f"\n========== Fold {fold} ==========")
        trn_idx = df['fold'] != fold
        val_idx = df['fold'] == fold
        trn_df, val_df = df[trn_idx].copy(), df[val_idx].copy()

        train_ds = DocumentDataset(trn_df, cfg['paths']['train_dir'], transform=t_train)
        valid_ds = DocumentDataset(val_df, cfg['paths']['train_dir'], transform=t_valid)

        train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                                  shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=cfg['train']['batch_size'],
                                  shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

        model = build_model(cfg, n_classes).to(device)

        # class weight
        if cfg['train']['class_weight'] == 'balanced':
            weights = compute_class_weight('balanced', classes=np.arange(n_classes), y=trn_df['label'].values)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'],
                                      weight_decay=cfg['train']['weight_decay'])

        total_steps = len(train_loader) * cfg['train']['epochs']
        if cfg['train']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
        elif cfg['train']['scheduler'] == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['train']['lr'],
                                                            steps_per_epoch=len(train_loader),
                                                            epochs=cfg['train']['epochs'])
        else:
            scheduler = None

        scaler = torch.cuda.amp.GradScaler(enabled=(cfg['train']['mix_precision'] and device=='cuda'))
        stopper = EarlyStopper(patience=cfg['train']['early_stop_patience'], mode='max')

        best_metric = -1.0
        fold_dir = os.path.join(cfg['paths']['out_dir'], f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        best_path = os.path.join(fold_dir, "best.pt")

        for epoch in range(1, cfg['train']['epochs'] + 1):
            print(f"\n[Fold {fold}] Epoch {epoch}/{cfg['train']['epochs']}")
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            va_loss, va_f1 = valid_one_epoch(model, valid_loader, criterion, device)

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                else:
                    scheduler.step()

            print(f"Train  | loss={tr_loss:.4f} f1={tr_f1:.4f}")
            print(f"Valid  | loss={va_loss:.4f} f1={va_f1:.4f}")

            # F1 기준으로 저장/조기종료
            improved, stop = stopper.step(va_f1)
            if improved:
                best_metric = va_f1
                torch.save({'model': model.state_dict(),
                            'cfg': cfg,
                            'fold': fold}, best_path)
                print(f"✓ Best updated (F1={va_f1:.4f}) → {best_path}")
            if stop:
                print("Early stopping triggered.")
                break

        print(f"[Fold {fold}] Best F1: {best_metric:.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    main(args.config)
