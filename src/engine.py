
import torch, numpy as np
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model, self.opt, self.loss_fn, self.device = model, optimizer, loss_fn, device
        self.scaler = GradScaler()

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        losses, preds_all, t_all = [], [], []
        for imgs, targets in loader:
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            with autocast():
                logits = self.model(imgs)
                loss = self.loss_fn(logits, targets)
            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            losses.append(loss.item())
            preds_all.extend(logits.argmax(1).detach().cpu().numpy())
            t_all.extend(targets.detach().cpu().numpy())
        f1 = f1_score(t_all, preds_all, average='macro')
        return float(np.mean(losses)), float(f1)
