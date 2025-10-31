
import os, yaml, torch, timm
from torch.utils.data import DataLoader
from torch import nn
from src.datasets import DocDataset
from src.transforms import get_train_tf, get_valid_tf
from src.engine import Trainer

def load_cfg(path, overrides=None):
    with open(path, 'r') as f: cfg = yaml.safe_load(f)
    overrides = overrides or {}
    # naive dot-override
    for i in range(0, len(overrides), 2):
        k, v = overrides[i], overrides[i+1]
        sect, key = k.split('.')
        cfg[sect][key] = v if not v.isdigit() else int(v)
    return cfg

if __name__ == "__main__":
    import sys
    cfg = load_cfg("configs/base.yaml", sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = cfg["MODEL"]["IMG_SIZE"]
    data_path = cfg["DATA"]["PATH"]
    train_csv = os.path.join(data_path, "train.csv")
    train_dir = os.path.join(data_path, "train")

    ds = DocDataset(train_csv, train_dir, transform=get_train_tf(img_size))
    dl = DataLoader(ds, batch_size=cfg["TRAIN"]["BATCH_SIZE"], shuffle=True, num_workers=cfg["TRAIN"]["NUM_WORKERS"], pin_memory=True)

    model = timm.create_model(cfg["MODEL"]["NAME"], pretrained=True, num_classes=cfg["DATA"]["NUM_CLASSES"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["TRAIN"]["LR"], weight_decay=cfg["TRAIN"]["WD"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["LOSS"]["LABEL_SMOOTHING"])

    trainer = Trainer(model, opt, loss_fn, device)
    for ep in range(cfg["TRAIN"]["EPOCHS"]):
        tl, tf1 = trainer.run_epoch(dl, train=True)
        print(f"Epoch {ep+1}/{cfg['TRAIN']['EPOCHS']} | train_loss={tl:.4f} | train_f1={tf1:.4f}")
