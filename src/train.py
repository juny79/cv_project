# ============================================
# src/train.py
# - v3 전처리와 호환
# - folds_csv 있으면 사용, 없으면 생성(StratifiedKFold)
# - 문자열 라벨 자동 인코딩(일관성 보장)
# - class_weight=balanced 지원
# - Cosine/OneCycle 스케줄러 지원
# - fold별 best.pt 저장
# ============================================
import os, yaml, timm, torch, random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset
from torch import nn
from src.datasets import DocumentDataset
from src.engine import train_one_epoch, valid_one_epoch, EarlyStopper
from src.transforms import get_train_transforms, get_valid_transforms  # v3를 재수출한다고 가정

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_device(name):
    if name == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_folds(cfg, df, label_col):
    """
    - paths.folds_csv 가 있으면 그걸 사용
    - 없으면 StratifiedKFold 로 생성하고 저장
    반환: folds_df (원본 df에 'fold' 열 추가)
    """
    folds_csv = cfg['paths'].get('folds_csv', None)
    if folds_csv and os.path.exists(folds_csv):
        folds_df = pd.read_csv(folds_csv)
        if 'fold' in folds_df.columns:
            # id 컬럼 기준으로 merge 시도 (없으면 인덱스 기준)
            id_cols = [c for c in df.columns if 'id' in c.lower() or 'file' in c.lower()]
            if id_cols and id_cols[0] in folds_df.columns:
                folds_df = folds_df[[id_cols[0], 'fold']]
                merged = df.merge(folds_df, on=id_cols[0], how='left')
                if 'fold' not in merged.columns or merged['fold'].isna().any():
                    # 폴드 매칭 실패한 경우 재생성
                    pass
                else:
                    return merged
            # id 매칭 실패 → 재생성
        # 형식이 맞지 않으면 생성
    # StratifiedKFold 생성
    folds = np.full(len(df), -1, dtype=int)
    y = df[label_col].values
    # 라벨이 문자열일 수 있으므로 factorize
    y_enc, uniq = pd.factorize(y)
    skf = StratifiedKFold(n_splits=cfg['data'].get('n_splits', 5), shuffle=True, random_state=cfg.get('seed', 42))
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y_enc)), y_enc)):
        folds[val_idx] = k
    out = df.copy()
    out['fold'] = folds
    if folds_csv:
        out[[*[c for c in out.columns if 'id' in c.lower()][:1], 'fold']].to_csv(folds_csv, index=False)
    return out

def build_loaders(cfg, df, label_col, fold, train_tf, valid_tf):
    base_dir = cfg['paths']['base_dir']
    train_dir = cfg['paths']['train_dir']

    tr_idx = df.index[df['fold'] != fold].tolist()
    va_idx = df.index[df['fold'] == fold].tolist()

    # 라벨 인코딩(문자 라벨 대비)
    y_all = df[label_col].astype(str)
    classes, y_all_enc = np.unique(y_all, return_inverse=True)
    df = df.copy()
    df['__yenc__'] = y_all_enc

    # Datasets
    tr_df = df.loc[tr_idx].reset_index(drop=True)
    va_df = df.loc[va_idx].reset_index(drop=True)

    # 임시 CSV 저장 없이 in-memory Dataset을 만들기 위해, Dataset 내부 자동탐색 로직 사용
    # 여기서는 간단히 csv를 그대로 저장해서 써도 되지만, 원본 파일을 덮지 않기 위해 복사본을 생성하지 않고
    # DocumentDataset에서 직접 DataFrame을 쓰도록 바꾸는 건 과도하므로, 여기서는 편의상 임시 CSV를 만듭니다.
    # (대회 환경에서도 안전)
    tmp_dir = os.path.join(cfg['paths']['out_dir'], '_tmp_csv')
    os.makedirs(tmp_dir, exist_ok=True)
    tr_csv = os.path.join(tmp_dir, f'train_fold{fold}.csv')
    va_csv = os.path.join(tmp_dir, f'valid_fold{fold}.csv')

    # id 컬럼 추정
    id_cols = [c for c in df.columns if 'id' in c.lower() or 'file' in c.lower()]
    id_col = id_cols[0] if id_cols else df.columns[0]

    tr_df_out = tr_df[[id_col]].copy()
    tr_df_out['label'] = tr_df['__yenc__'].values
    va_df_out = va_df[[id_col]].copy()
    va_df_out['label'] = va_df['__yenc__'].values
    tr_df_out.to_csv(tr_csv, index=False)
    va_df_out.to_csv(va_csv, index=False)

    from src.datasets import DocumentDataset  # 재-import 안전
    tr_ds = DocumentDataset(tr_csv, train_dir, transform=train_tf, has_label=True, id_col=id_col, label_col='label')
    va_ds = DocumentDataset(va_csv, train_dir, transform=valid_tf, has_label=True, id_col=id_col, label_col='label')

    tr_loader = DataLoader(tr_ds,
                           batch_size=cfg['train']['batch_size'],
                           shuffle=True,
                           num_workers=cfg.get('num_workers', 2),
                           pin_memory=True)
    va_loader = DataLoader(va_ds,
                           batch_size=cfg['train']['batch_size'],
                           shuffle=False,
                           num_workers=cfg.get('num_workers', 2),
                           pin_memory=True)

    return tr_loader, va_loader, classes  # classes: 인코딩 역변환용

def get_scheduler(cfg, optimizer, steps_per_epoch):
    sc = cfg['train'].get('scheduler', 'cosine')
    epochs = cfg['train']['epochs']
    if sc == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        max_lr = cfg['train']['lr']
        return OneCycleLR(optimizer, max_lr=max_lr,
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

def main():
    ap = ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.get('seed', 42))
    device = auto_device(cfg.get('device', 'auto'))

    # 경로/파일 로드
    train_csv = cfg['paths']['train_csv']
    out_dir   = cfg['paths']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(train_csv)
    # 라벨 컬럼 탐색
    cand_lb = [c for c in df.columns if c.lower() in ['label', 'target', 'class', 'y']]
    label_col = cand_lb[0] if cand_lb else df.columns[-1]
    # 폴드 확보
    df_folds = ensure_folds(cfg, df, label_col)

    # transforms (v3 재수출)
    img_size = cfg['data'].get('img_size', 640)
    t_train = get_train_transforms(img_size)
    t_valid = get_valid_transforms(img_size)

    folds = cfg['data'].get('folds', list(range(cfg['data'].get('n_splits', 5))))
    all_fold_metrics = {}

    # 라벨 인코더(문자 라벨 → 정수)의 일관성을 folds 전체에서 맞추기 위해, 전체 클래스 목록 확보
    all_labels = df[label_col].astype(str)
    classes_global = np.unique(all_labels)

    for fold in folds:
        print(f"\n========== Fold {fold} ==========")
        tr_loader, va_loader, classes = build_loaders(cfg, df_folds, label_col, fold, t_train, t_valid)

        # 모델
        num_classes = len(classes_global)  # 전 fold 공통 클래스 수로 고정
        model = timm.create_model(cfg['model']['name'],
                                  pretrained=cfg['model'].get('pretrained', True),
                                  num_classes=num_classes)
        model = model.to(device)

        # 손실함수 (class_weight 옵션)
        if cfg['train'].get('class_weight', 'none') == 'balanced':
            # 유효 라벨 수집 (train loader 첫 epoch에서 구하기보단 df 기반으로)
            y_for_weight = df_folds.loc[df_folds['fold'] != fold, label_col].astype(str).values
            # 전역 클래스 순서에 맞춰 인코딩
            y_enc = pd.Categorical(y_for_weight, categories=classes_global).codes
            w = compute_class_weight(class_weight='balanced',
                                     classes=np.arange(len(classes_global)),
                                     y=y_enc)
            weight = torch.tensor(w, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg['train']['lr'],
                                      weight_decay=cfg['train']['weight_decay'])
        scaler = torch.cuda.amp.GradScaler(enabled=cfg['train'].get('mix_precision', True) and device=='cuda')
        scheduler = get_scheduler(cfg, optimizer, steps_per_epoch=len(tr_loader))

        early = EarlyStopper(patience=cfg['train'].get('early_stop_patience', 3), mode='max')
        best_path = os.path.join(out_dir, f'fold{fold}', 'best.pt')

        best_f1 = -1.0
        epochs = cfg['train']['epochs']
        for epoch in range(1, epochs+1):
            print(f"[Fold {fold}] Epoch {epoch}/{epochs}")
            tr_loss, tr_f1 = train_one_epoch(model, tr_loader, criterion, optimizer, device, scaler)
            va_loss, va_f1 = valid_one_epoch(model, va_loader, criterion, device)

            # 스케줄러 스텝
            sc = cfg['train'].get('scheduler', 'cosine')
            if sc == 'onecycle':
                scheduler.step()
            else:
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

    print("\n==== Summary (Best F1 per fold) ====")
    for k,v in all_fold_metrics.items():
        print(f"Fold {k}: F1={v:.4f}")
    print("Saved checkpoints under:", out_dir)

if __name__ == '__main__':
    main()
