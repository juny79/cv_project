import os, yaml, timm, torch
import numpy as np, pandas as pd
import torch.nn.functional as F
from PIL import Image
from src.transforms import get_valid_transforms

SUSPECTS = [
    ("0583254a73b48ece.jpg", None),   # label seems correct (11)
    ("1ec14a14bbe633db.jpg", 7),      # 14 -> 7?
    ("38d1796b6ad99ddd.jpg", None),   # 11 correct?
    ("45f0d2dfc7e47c03.jpg", 7),      # 3 -> 7?
    ("7100c5c67aecadc5.jpg", 7),      # 3 -> 7?
    ("8646f2c3280a4f49.jpg", 3),      # 7 -> 3?
    ("aec62dced7af97cd.jpg", 14),     # 3 -> 14?
    ("c5182ab809478f12.jpg", 14),     # 4 -> 14?
]

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def load_cfg(p):
    with open(p,'r') as f:
        return yaml.safe_load(f)

def resolve_image_path(img_dir, stem):
    for ext in COMMON_EXTS:
        cand = os.path.join(img_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(stem)

@torch.no_grad()
def infer_single_batch(model, xb, use_amp=True):
    # 4-way rotation TTA picking most confident view (consistent with predict.py)
    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        out0 = model(xb)
        out90 = model(torch.rot90(xb,1,dims=[2,3]))
        out180 = model(torch.rot90(xb,2,dims=[2,3]))
        out270 = model(torch.rot90(xb,3,dims=[2,3]))
    cand = [out0, out90, out180, out270]
    probs = [torch.softmax(z, dim=1) for z in cand]
    maxp = torch.stack([p.max(dim=1).values for p in probs], 0)  # (4,B)
    best_idx = torch.argmax(maxp, 0)
    picked = torch.stack([cand[r][i] for i,r in enumerate(best_idx.tolist())], 0)
    return picked

@torch.no_grad()
def main(cfg_path="configs/base.yaml"):
    cfg = load_cfg(cfg_path)
    paths = cfg['paths']; data_cfg = cfg['data']; model_cfg = cfg['model']
    train_csv = paths['train_csv']; train_dir = paths['train_dir']
    df = pd.read_csv(train_csv)
    label_map = dict(zip(df['ID'], df['target']))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = get_valid_transforms(int(data_cfg['img_size']))

    # Collect images
    imgs = []
    stems = []
    for fname, _ in SUSPECTS:
        p = resolve_image_path(train_dir, fname)
        arr = np.array(Image.open(p).convert('RGB'))
        x = tfm(image=arr)['image']
        imgs.append(x)
        stems.append(fname)
    xb = torch.stack(imgs,0).to(device)

    # Load checkpoints
    folds = data_cfg.get('folds', [0,1,2,3,4])
    ckpts = []
    for k in folds:
        ck = os.path.join(paths['out_dir'], f'fold{k}', 'best.pt')
        if os.path.exists(ck):
            ckpts.append(ck)
    if not ckpts:
        raise FileNotFoundError('No fold checkpoints found.')

    # num_classes
    state0 = torch.load(ckpts[0], map_location='cpu')
    head_w = None
    for k,v in state0['model'].items():
        if ('head.fc.weight' in k or 'classifier.weight' in k or 'fc.weight' in k):
            head_w = v; break
    num_classes = head_w.shape[0] if head_w is not None else int(data_cfg.get('num_classes',17))

    model = timm.create_model(model_cfg['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    use_amp = bool(cfg.get('inference', {}).get('amp', True)) and device.startswith('cuda')

    all_probs = []
    for fold_i, ck in enumerate(ckpts):
        w = torch.load(ck, map_location=device)
        try:
            model.load_state_dict(w['model'], strict=True)
        except Exception:
            model.load_state_dict(w['model'], strict=False)
        model.eval()
        # temperature scaling
        temp_file = os.path.join(os.path.dirname(ck), 'temp.npy')
        T = float(np.load(temp_file)[0]) if os.path.exists(temp_file) else 1.0
        logits = infer_single_batch(model, xb, use_amp=use_amp)
        logits = logits / T
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N,C)
        all_probs.append(probs)
    probs_mean = np.mean(np.stack(all_probs,0),0)
    preds = probs_mean.argmax(1)

    # Build output table
    rows = []
    for i,(fname, alt) in enumerate(SUSPECTS):
        true_lbl = label_map.get(fname, None)
        p_row = probs_mean[i]
        top3_idx = p_row.argsort()[-3:][::-1]
        alt_prob = p_row[alt] if alt is not None else None
        rows.append({
            'image': fname,
            'true_label': true_lbl,
            'pred_label': int(preds[i]),
            'pred_conf': float(p_row[preds[i]]),
            'top3': {int(k): float(p_row[k]) for k in top3_idx},
            'alt_label': alt,
            'alt_prob': float(alt_prob) if alt_prob is not None else None,
            'delta_pred_alt': (float(p_row[preds[i]]) - float(alt_prob)) if alt_prob is not None else None
        })

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    # Save for later analysis
    out_path = os.path.join(paths['out_dir'], 'suspect_label_check.csv')
    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/base.yaml')
    args = ap.parse_args()
    main(args.config)
