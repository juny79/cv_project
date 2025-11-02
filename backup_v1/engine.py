import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    losses = []
    all_preds, all_tgts = [], []
    for imgs, tgts in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss = criterion(logits, tgts)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        all_preds.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        all_tgts.extend(tgts.detach().cpu().tolist())

    f1 = f1_score(all_tgts, all_preds, average='macro')
    return sum(losses)/len(losses), f1

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
        all_preds.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        all_tgts.extend(tgts.detach().cpu().tolist())
    f1 = f1_score(all_tgts, all_preds, average='macro')
    return sum(losses)/len(losses), f1

class EarlyStopper:
    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, value):
        improved = (value > self.best) if (self.best is not None) else True
        if improved:
            self.best = value
            self.count = 0
        else:
            self.count += 1
        return improved, self.count >= self.patience
