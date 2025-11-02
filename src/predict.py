import os, yaml, timm, torch
import pandas as pd
from torch.utils.data import DataLoader
from src.transforms import get_valid_transforms
from src.datasets  import DocumentDataset

def load_cfg(path): 
    import yaml
    with open(path,'r') as f: return yaml.safe_load(f)

@torch.no_grad()
def predict_fold(ckpt_path, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(cfg['model']['name'], pretrained=False,
                              num_classes=state['model']['fc.weight'].shape[0] if 'fc.weight' in state['model'] else None)
    # 일반화: num_classes를 cfg에서 가져오기 (학습 시 저장된 cfg 사용)
    num_classes = cfg.get('num_classes', None)
    if num_classes is not None:
        model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes)
    model.load_state_dict(state['model'], strict=False)
    model.to(device).eval()

    sample = pd.read_csv(cfg['paths']['sample_csv'])
    test_ds = DocumentDataset(sample, cfg['paths']['test_dir'], transform=get_valid_transforms(cfg['data']['img_size']))
    test_loader = DataLoader(test_ds, batch_size=cfg['train']['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'])
    preds = []
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0)

def main():
    import argparse, glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="/root/cv_project/outputs/submission.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    fold_ckpts = sorted(glob.glob(os.path.join(cfg['paths']['out_dir'], "fold*/best.pt")))
    assert len(fold_ckpts) > 0, "학습된 fold 체크포인트를 찾지 못했습니다."

    probs = []
    for ckpt in fold_ckpts:
        print(f"Infer with {ckpt}")
        state = torch.load(ckpt, map_location='cpu')
        fold_cfg = state.get('cfg', cfg)
        p = predict_fold(ckpt, fold_cfg)
        probs.append(p)

    prob = torch.stack(probs).mean(0)      # 간단한 앙상블 평균
    pred = prob.argmax(1).numpy()

    sample = pd.read_csv(cfg['paths']['sample_csv'])
    # sample의 라벨 컬럼명 자동 탐색/생성
    out = sample.copy()
    out_col = next((c for c in out.columns if c.lower() in ['label','target','class']), 'target')
    out[out_col] = pred
    out.to_csv(args.out, index=False)
    print(f"Saved submission → {args.out}")

if __name__ == "__main__":
    main()
