# src/predict.py  (PyTorch 2.6 대응 + 클래스 역매핑 + 견고한 로딩)
import os, yaml, timm, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# robust import of valid transforms
try:
    from src.transforms import get_valid_transforms
except Exception:
    import src.transforms as _transforms
    get_valid_transforms = getattr(_transforms, 'get_valid_transforms', None)
    if get_valid_transforms is None:
        raise ImportError('src.transforms does not provide get_valid_transforms')


def load_cfg(p):
    with open(p, 'r') as f:
        return yaml.safe_load(f)


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
        # try several common extensions (and bare stem)
        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG', ''):
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB'))
                break
        else:
            raise FileNotFoundError(f"image not found for id={stem} under {self.img_dir}")
        x = self.transform(image=img)['image']
        return x, stem


def _safe_load_ckpt(path, map_location):
    """
    PyTorch 2.6: default weights_only=True.
    1) try weights_only=False (trusted local ckpt)
    2) fallback: allowlist numpy scalar via safe_globals
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e1:
        try:
            from torch.serialization import safe_globals, add_safe_globals

            # allowlist numpy scalar if needed
            try:
                import numpy as _np

                add_safe_globals([_np._core.multiarray.scalar])  # type: ignore[attr-defined]
            except Exception:
                pass

            with safe_globals():
                return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}'. "
                f"1st error: {e1}\n2nd error (safe_globals): {e2}"
            )


@torch.no_grad()
def predict_ensemble(cfg_path, tta=0):
    cfg = load_cfg(cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_cfg = cfg.get('data', {})
    paths = cfg.get('paths', {})

    # config validation
    if 'img_size' not in data_cfg:
        raise ValueError('Missing config: data.img_size')
    if 'sample_csv' not in paths:
        raise ValueError('Missing config: paths.sample_csv (path to sample_submission.csv)')
    if 'test_dir' not in paths:
        raise ValueError('Missing config: paths.test_dir (directory with test images)')
    if 'out_dir' not in paths:
        paths['out_dir'] = './outputs'

    # dataset / loader
    tfm = get_valid_transforms(int(data_cfg['img_size']))
    test = TestDS(paths['sample_csv'], paths['test_dir'], tfm)
    num_workers = int(cfg.get('num_workers', 2))
    batch_size = int(cfg.get('train', {}).get('batch_size', 32))
    loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # gather ckpts
    out_dir = paths['out_dir']
    folds = cfg['data'].get('folds', [0, 1, 2, 3, 4])
    ckpts = []
    for k in folds:
        p = os.path.join(out_dir, f'fold{k}', 'best.pt')
        if os.path.exists(p):
            ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {out_dir}/fold*/best.pt")

    # build model with inferred num_classes
    state0 = _safe_load_ckpt(ckpts[0], map_location='cpu')
    state0_model = state0.get('model', state0)
    if not isinstance(state0_model, dict):
        raise RuntimeError(f"Checkpoint '{ckpts[0]}' does not contain a valid model state_dict")
    num_classes = None
    for k, v in state0_model.items():
        if k.endswith('weight') and hasattr(v, 'shape') and len(v.shape) == 2:
            num_classes = v.shape[0]
            break
    if num_classes is None:
        raise RuntimeError("Failed to infer num_classes from checkpoint state_dict.")

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)

    # load per-fold and collect logits
    all_logits = []
    for ck in ckpts:
        w = _safe_load_ckpt(ck, map_location=device)
        w_model = w.get('model', w)
        try:
            model.load_state_dict(w_model, strict=False)
        except RuntimeError as e:
            print(f"[WARN] strict load failed for {ck}: {e}\n → partial shape-matched load")
            model_sd = model.state_dict()
            ck_sd = w_model
            matched = {k: v for k, v in ck_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            if not matched:
                raise RuntimeError(f'No matching parameter shapes found between checkpoint {ck} and model.')
            model_sd.update(matched)
            model.load_state_dict(model_sd, strict=False)

        model.eval()
        fold_logits = []
        for xb, _ids in loader:
            xb = xb.to(device)
            logits = model(xb)
            if tta and tta > 0:
                xflip = torch.flip(xb, dims=[3])
                logits = (logits + model(xflip)) / 2
            fold_logits.append(logits.detach().cpu())
        fold_logits = torch.cat(fold_logits)
        all_logits.append(fold_logits)

    # ensemble
    mean_logits = torch.stack(all_logits).mean(0)
    probs = torch.softmax(mean_logits, dim=1)
    pred_idx = torch.argmax(mean_logits, dim=1).numpy()
    max_probs = probs.max(1).values.numpy()

    # class mapping (restore original labels if saved)
    classes = state0.get('classes', None)
    if classes is not None:
        # ensure numpy array for indexing
        classes = np.array(classes)
        preds = classes[pred_idx]
    else:
        preds = pred_idx

    # build submission
    sub = pd.read_csv(paths['sample_csv'])
    # detect label column
    ycol = [c for c in sub.columns if c.lower() in ['label', 'target', 'class']]
    ycol = ycol[0] if ycol else sub.columns[-1]
    sub[ycol] = preds
    sub['confidence'] = max_probs

    # (optional) agreement meter among folds
    try:
        fold_agree = np.mean([(l.argmax(1).numpy() == pred_idx).mean() for l in all_logits])
        sub['fold_agreement'] = fold_agree
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'submission.csv')
    sub.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    return out_csv

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=1)
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
