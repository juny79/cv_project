# ============================================
# src/engine.py
# - 학습/검증 루프, F1 계산, AMP 지원, EarlyStopping
# ============================================
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score

class EarlyStopper:
    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return True  # best 갱신
        improved = (value > self.best) if self.mode == 'max' else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    losses = []
    all_preds, all_tgts = [], []

    for imgs, tgts in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, tgts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, tgts)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_tgts.extend(tgts.detach().cpu().tolist())

    f1 = f1_score(all_tgts, all_preds, average='macro')
    return float(sum(losses)/max(len(losses),1)), float(f1)

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_tgts = [], []

    for imgs, tgts in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, tgts)

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_tgts.extend(tgts.detach().cpu().tolist())

    f1 = f1_score(all_tgts, all_preds, average='macro')
    return float(sum(losses)/max(len(losses),1)), float(f1)
