# src/preprocessing_pipeline_v3.py
import os, torch, random, cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from src.transforms_v3 import get_train_transforms_v3, get_valid_transforms_v3

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

BASE_DIR = '/root/cv_project/data'
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')
TEST_CSV  = os.path.join(BASE_DIR, 'sample_submission.csv')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')
SAVE_DIR  = os.path.join(BASE_DIR, 'processed_v2')  # 기존 디렉터리 재사용
os.makedirs(SAVE_DIR, exist_ok=True)

EXTS = ['.jpg','.jpeg','.png','.JPG','.PNG','.JPEG']
def resolve_path(d, stem):
    base = os.path.basename(str(stem))
    root, ext = os.path.splitext(base)
    cands = [os.path.join(d, base)]
    if ext:
        cands += [os.path.join(d, root+e) for e in EXTS]
    else:
        cands += [os.path.join(d, base+e) for e in EXTS]
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError(f'Image not found for {stem} under {d}')

class DocDS(Dataset):
    def __init__(self, csv, img_dir, transform=None, has_label=True):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label
        idc = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.id_col = idc[0] if idc else self.df.columns[0]
        if has_label:
            lcs = [c for c in self.df.columns if c.lower() in ['label','target','class']]
            self.label_col = lcs[0] if lcs else self.df.columns[-1]
        else:
            self.label_col = None
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_path = resolve_path(self.img_dir, str(r[self.id_col]))
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform: img = self.transform(image=img)['image']
        label = int(r[self.label_col]) if self.label_col else -1
        return img, label

def save_tensors(ds, name, bs=16, workers=0):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    xs, ys = [], []
    for x, y in tqdm(loader, desc=f'preprocess:{name}'):
        xs.append(x); ys.append(y)
    xs = torch.cat(xs); ys = torch.cat(ys) if ys[0].numel() > 0 else torch.tensor([])
    torch.save({'images': xs, 'labels': ys}, os.path.join(SAVE_DIR, f'{name}_processed.pt'))
    print(f'✓ saved {name}_processed.pt → {xs.shape}')

if __name__ == '__main__':
    set_seed(42)
    tr = get_train_transforms_v3(img_size=640)
    va = get_valid_transforms_v3(img_size=640)
    train_ds = DocDS(TRAIN_CSV, TRAIN_DIR, transform=tr, has_label=True)
    test_ds  = DocDS(TEST_CSV,  TEST_DIR,  transform=va, has_label=False)
    save_tensors(train_ds, 'train', bs=16, workers=0)
    save_tensors(test_ds,  'test',  bs=16, workers=0)
