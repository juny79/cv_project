# src/train.py
import os, yaml, random, timm, torch
import numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from PIL import Image
from src.preprocessing import get_train_transforms, get_valid_transforms

# ---------------- utils ----------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_cfg(p):
    with open(p,'r') as f: return yaml.safe_load(f)

class ImgDS(Dataset):
    exts = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')
    def __init__(self, csv, img_dir, transform, has_label=True):
        self.df = pd.read_csv(csv)
        self.dir = img_dir
        self.tf = transform
        self.has_label = has_label
        idc = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.id_col = idc[0] if idc else self.df.columns[0]
        if has_label:
            lcs = [c for c in self.df.columns if c.lower() in ['target','label','class']]
            self.y_col = lcs[0] if lcs else self.df.columns[-1]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        stem = str(row[self.id_col])
        img = None
        for e in self.exts:
            p = os.path.join(self.dir, stem+e)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB')); break
        if img is None:
            raise FileNotFoundError(stem)
        x = self.tf(image=img)['image']
        y = int(row[self.y_col]) if self.has_label else -1
        return x, y

# ----------- EMA (optional) -----------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

# -------- Label smoothing CE ----------
class SmoothCE(nn.Module):
    def __init__(self, eps=0.05):
        super().__init__(); self.eps = eps; self.ce = nn.CrossEntropyLoss()
    def forward(self, logits, target):
        if self.eps<=0: return self.ce(logits, target)
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.eps / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        logp = torch.log_softmax(logits, dim=1)
        return -(true_dist * logp).sum(dim=1).mean()

# -------- MixUp / CutMix (light) ------
def do_mix(inputs, targets, alpha):
    if alpha<=0: return inputs, targets, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    x = lam*inputs + (1-lam)*inputs[idx]
    return x, (targets, targets[idx], lam), 'mixup'

def do_cutmix(inputs, targets, alpha):
    if alpha<=0: return inputs, targets, None
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = inputs.size()
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W*np.sqrt(1-lam)); h = int(H*np.sqrt(1-lam))
    x1, y1 = np.clip(cx-w//2, 0, W), np.clip(cy-h//2, 0, H)
    x2, y2 = np.clip(cx+w//2, 0, W), np.clip(cy+h//2, 0, H)
    idx = torch.randperm(B, device=inputs.device)
    inputs[:, :, y1:y2, x1:x2] = inputs[idx, :, y1:y2, x1:x2]
    lam2 = 1 - ((x2-x1)*(y2-y1)/(W*H))
    return inputs, (targets, targets[idx], lam2), 'cutmix'

def mix_loss(criterion, logits, mix_tgt):
    y1, y2, lam = mix_tgt
    return lam*criterion(logits, y1) + (1-lam)*criterion(logits, y2)

# -------------- train/eval --------------
def train_one_epoch(model, loader, opt, criterion, device, scaler=None,
                    mixup=0.1, cutmix=0.1, ema: EMA|None=None):
    model.train()
    losses, preds, tgts = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        mix_tgt = None; kind = None
        r = np.random.rand()
        if r < cutmix: xb, mix_tgt, kind = do_cutmix(xb, yb, cutmix)
        elif r < cutmix + mixup: xb, mix_tgt, kind = do_mix(xb, yb, mixup)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(xb)
            if mix_tgt is None:
                loss = criterion(logits, yb)
            else:
                loss = mix_loss(criterion, logits, mix_tgt)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        if ema: ema.update(model)

        losses.append(loss.item())
        preds.extend(torch.argmax(logits,1).detach().cpu().tolist())
        tgts.extend(yb.detach().cpu().tolist())
    f1 = f1_score(tgts, preds, average='macro')
    return float(np.mean(losses)), f1

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, tgts = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits,1).cpu().tolist())
        tgts.extend(yb.cpu().tolist())
    f1 = f1_score(tgts, preds, average='macro')
    return float(np.mean(losses)), f1

# -------------- main --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config); set_seed(cfg.get('seed',42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    paths, data, tr = cfg['paths'], cfg['data'], cfg['train']
    img_size = int(data['img_size'])

    df = pd.read_csv(paths['train_csv'])
    ycol = [c for c in df.columns if c.lower() in ['target','label','class']]
    ycol = ycol[0] if ycol else df.columns[-1]
    X = df.index.values; y = df[ycol].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.get('seed',42))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== Fold {fold} ==========")
        tr_tf = get_train_transforms(img_size)
        va_tf = get_valid_transforms(img_size)
        tr_ds = ImgDS(paths['train_csv'], paths['train_dir'], tr_tf, True)
        va_ds = ImgDS(paths['train_csv'], paths['train_dir'], va_tf, True)
        tr_ds.df = tr_ds.df.iloc[tr_idx].reset_index(drop=True)
        va_ds.df = va_ds.df.iloc[va_idx].reset_index(drop=True)

        tr_loader = DataLoader(tr_ds, batch_size=tr['batch_size'], shuffle=True,
                               num_workers=tr.get('num_workers',2), pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=tr['batch_size'], shuffle=False,
                               num_workers=tr.get('num_workers',2), pin_memory=True)

        model = timm.create_model(cfg['model']['name'],
                                  pretrained=cfg['model'].get('pretrained', True),
                                  num_classes=int(data['num_classes'])).to(device)
        if cfg['model'].get('dropout'):
            if hasattr(model, 'drop_rate'): model.drop_rate = cfg['model']['dropout']

        criterion = SmoothCE(eps=tr.get('label_smoothing',0.0)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(tr['lr']),
                                weight_decay=float(tr['weight_decay']))
        if tr['scheduler'] == 'cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr['epochs'])
        else:
            sch = None
        scaler = torch.cuda.amp.GradScaler(enabled=tr.get('amp', True))
        ema = EMA(model) if tr.get('ema', True) else None

        best_f1, es_cnt = -1, 0
        out_fold = os.path.join(paths['out_dir'], f'fold{fold}')
        os.makedirs(out_fold, exist_ok=True)

        for ep in range(1, tr['epochs']+1):
            tl, tf1 = train_one_epoch(model, tr_loader, opt, criterion, device, scaler,
                                      mixup=tr.get('mixup_alpha',0.0),
                                      cutmix=tr.get('cutmix_alpha',0.0),
                                      ema=ema)
            if ema: ema.apply_to(model)
            vl, vf1 = evaluate(model, va_loader, criterion, device)
            if sch: sch.step()

            print(f"[Fold {fold}] Ep {ep}/{tr['epochs']} | Train loss {tl:.4f} f1 {tf1:.4f} | Valid loss {vl:.4f} f1 {vf1:.4f}")
            if vf1 > best_f1:
                best_f1 = vf1; es_cnt = 0
                torch.save({'model': model.state_dict()}, os.path.join(out_fold, 'best.pt'))
            else:
                es_cnt += 1
                if es_cnt >= tr['early_stop_patience']:
                    print(f"[Fold {fold}] Early stop. Best F1={best_f1:.4f}")
                    break
        print(f"[Fold {fold}] Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
