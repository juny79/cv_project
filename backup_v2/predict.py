# src/predict.py  (PyTorch 2.6 대응 + 안전 로더만 사용 + 클래스 역매핑 + 견고한 입출력)
import os, yaml, timm, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ----- robust import of valid transforms -----
try:
    from src.transforms import get_valid_transforms
except Exception:
    import src.transforms as _transforms
    get_valid_transforms = getattr(_transforms, 'get_valid_transforms', None)
    if get_valid_transforms is None:
        raise ImportError('src.transforms does not provide get_valid_transforms')

# ----- config loader -----
def load_cfg(p):
    with open(p, 'r') as f:
        return yaml.safe_load(f)

# ----- dataset -----
class TestDS(Dataset):
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        idc = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.id_col = idc[0] if idc else self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        stem = str(self.df.iloc[i, self.df.columns.get_loc(self.id_col)])
        # 확장자 유연 처리
        exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG', '')
        img = None
        for ext in exts:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB'))
                break
        if img is None:
            raise FileNotFoundError(f"[predict] image not found for id={stem} under {self.img_dir}")
        x = self.transform(image=img)['image']
        return x, stem

# ----- PyTorch 2.6 안전 로더 (반드시 이 함수만 사용) -----
def _safe_load_ckpt(path, map_location):
    """
    PyTorch 2.6: default weights_only=True → UnpicklingError 방지.
    1) weights_only=False로 재시도 (신뢰된 로컬 ckpt 가정)
    2) 실패시 safe_globals allowlist 추가 후 재시도
    3) 구 torch 의 TypeError(인자 미지원) 대응
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 구버전 torch: weights_only 인자 없음
        return torch.load(path, map_location=map_location)
    except Exception as e1:
        try:
            from torch.serialization import safe_globals, add_safe_globals
            try:
                import numpy as _np
                # numpy._core.multiarray.scalar 허용
                add_safe_globals([_np._core.multiarray.scalar])  # type: ignore[attr-defined]
            except Exception:
                pass
            with safe_globals():
                try:
                    return torch.load(path, map_location=map_location, weights_only=False)
                except TypeError:
                    return torch.load(path, map_location=map_location)
        except Exception as e2:
            raise RuntimeError(
                f"[predict] Failed to load checkpoint '{path}'.\n"
                f"1st error: {e1}\n2nd error (safe_globals): {e2}"
            )

@torch.no_grad()
def predict_ensemble(cfg_path, tta=1):
    # ---------- config ----------
    cfg = load_cfg(cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_cfg = cfg.get('data', {})
    paths = cfg.get('paths', {})

    # 기본값 보강
    base_dir = paths.get('base_dir', '/root/cv_project/data')
    paths.setdefault('sample_csv', os.path.join(base_dir, 'sample_submission.csv'))
    paths.setdefault('test_dir',  os.path.join(base_dir, 'test'))
    paths.setdefault('out_dir',   './outputs')
    if 'img_size' not in data_cfg:
        data_cfg['img_size'] = 640  # 안전 기본값

    img_size = int(data_cfg['img_size'])
    tfm = get_valid_transforms(img_size)

    # ---------- dataset & loader ----------
    test = TestDS(paths['sample_csv'], paths['test_dir'], tfm)
    num_workers = int(cfg.get('num_workers', 2))
    batch_size = int(cfg.get('train', {}).get('batch_size', 32))
    loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # ---------- gather fold checkpoints ----------
    out_dir = paths['out_dir']
    folds = cfg.get('data', {}).get('folds', [0, 1, 2, 3, 4])
    ckpts = []
    for k in folds:
        p = os.path.join(out_dir, f'fold{k}', 'best.pt')
        if os.path.exists(p):
            ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"[predict] No checkpoints found under {out_dir}/fold*/best.pt")

    # ---------- build model with inferred num_classes ----------
    state0 = _safe_load_ckpt(ckpts[0], map_location='cpu')
    num_classes = None
    for k, v in state0['model'].items():
        if k.endswith('weight') and hasattr(v, 'shape') and len(v.shape) == 2:
            num_classes = int(v.shape[0])
            break
    if num_classes is None:
        raise RuntimeError("[predict] Failed to infer num_classes from checkpoint state_dict.")

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)

    # ---------- load each fold & collect logits ----------
    all_logits = []
    for ck in ckpts:
        w = _safe_load_ckpt(ck, map_location=device)
        # state_dict 로드(엄격 불일치 시 shape 일치 파라미터만 매칭)
        try:
            model.load_state_dict(w['model'], strict=False)
        except RuntimeError as e:
            print(f"[WARN] strict load failed for {ck}: {e}\n → partial shape-matched load")
            model_sd = model.state_dict()
            ck_sd = w.get('model', w)
            matched = {k: v for k, v in ck_sd.items()
                       if k in model_sd and model_sd[k].shape == v.shape}
            if not matched:
                raise RuntimeError(f"[predict] No matching parameter shapes between model and {ck}.")
            model_sd.update(matched)
            model.load_state_dict(model_sd, strict=False)

        model.eval()
        fold_logits = []
        for xb, _ids in loader:
            xb = xb.to(device)
            logits = model(xb)
            if tta and int(tta) > 0:
                # 간단 TTA: H-Flip
                xflip = torch.flip(xb, dims=[3])
                logits = (logits + model(xflip)) / 2
            fold_logits.append(logits.detach().cpu())
        fold_logits = torch.cat(fold_logits)
        all_logits.append(fold_logits)

    # ---------- ensemble + class mapping ----------
    mean_logits = torch.stack(all_logits).mean(0)
    probs = torch.softmax(mean_logits, dim=1)
    pred_idx = torch.argmax(mean_logits, dim=1).numpy()
    max_probs = probs.max(1).values.numpy()

    classes = state0.get('classes', None)  # train.py에서 저장한 classes
    if classes is not None:
        classes = np.array(classes)
        preds = classes[pred_idx]
    else:
        preds = pred_idx  # 대회가 정수 라벨이면 이 경로 사용

    # ---------- build submission ----------
    sub = pd.read_csv(paths['sample_csv'])
    ycol_candidates = [c for c in sub.columns if c.lower() in ['label', 'target', 'class']]
    ycol = ycol_candidates[0] if ycol_candidates else sub.columns[-1]
    if len(sub) != len(preds):
        raise ValueError(f"[predict] sample_csv length ({len(sub)}) != preds length ({len(preds)})")
    sub[ycol] = preds
    sub['confidence'] = max_probs

    # (optional) fold agreement
    try:
        fold_agree = np.mean([(l.argmax(1).numpy() == pred_idx).mean() for l in all_logits])
        sub['fold_agreement'] = fold_agree
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'submission.csv')
    sub.to_csv(out_csv, index=False)
    print(f"[predict] Saved → {out_csv}")
    return out_csv

# ----- entry -----
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=1)
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
