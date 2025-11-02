import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

COMMON_EXTS = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']

def resolve_image_path(img_dir: str, name: str) -> str:
    # 절대경로면 바로 검증
    if os.path.isabs(name) and os.path.exists(name):
        return name
    base = os.path.basename(str(name))
    root, ext = os.path.splitext(base)
    # 1) 그대로 시도
    cand = os.path.join(img_dir, base)
    candidates = [cand]
    # 2) 확장자 교정
    if ext:
        candidates += [os.path.join(img_dir, root + e) for e in COMMON_EXTS]
    else:
        candidates += [os.path.join(img_dir, base + e) for e in COMMON_EXTS]
    seen = set()
    for p in candidates:
        if p in seen: 
            continue
        seen.add(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"이미지 경로를 찾지 못했습니다. name='{name}', tried={candidates[:5]} ...")

class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # 자동 컬럼 탐색
        self.id_col = next((c for c in self.df.columns if any(k in c.lower() for k in ['id','file','path','image'])), self.df.columns[0])
        self.label_col = next((c for c in self.df.columns if c.lower() in ['label','target','class']), None)
        # 레이블은 int로 캐스팅(분류)
        if self.label_col is not None:
            self.df[self.label_col] = self.df[self.label_col].astype(int)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row[self.id_col])
        img_path = resolve_image_path(self.img_dir, name)
        img = np.array(Image.open(img_path).convert("RGB"))
        y = int(row[self.label_col]) if self.label_col is not None else -1
        if self.transform:
            img = self.transform(image=img)['image']
        return img, y