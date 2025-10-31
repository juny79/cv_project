
import os, yaml, torch, timm, pandas as pd
from torch.utils.data import DataLoader
from src.datasets import DocDataset
from src.transforms import get_valid_tf

if __name__ == "__main__":
    import sys
    cfg = yaml.safe_load(open("configs/base.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = cfg["DATA"]["PATH"]
    test_csv = os.path.join(data, "sample_submission.csv")
    test_dir = os.path.join(data, "test")

    ds = DocDataset(test_csv, test_dir, transform=get_valid_tf(cfg["MODEL"]["IMG_SIZE"]))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # weights path via arg: OUT.PRED next arg is path
    weights = sys.argv[sys.argv.index("--weights")+1]
    out_csv = sys.argv[sys.argv.index("OUT.PRED")+1]

    model = timm.create_model(cfg["MODEL"]["NAME"], pretrained=False, num_classes=cfg["DATA"]["NUM_CLASSES"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            y = model(x).argmax(1).detach().cpu().tolist()
            preds += y

    sub = pd.read_csv(test_csv)
    sub['target'] = preds
    sub.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
