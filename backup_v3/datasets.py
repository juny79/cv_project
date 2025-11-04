# ============================================
# src/datasets.py
# - 확장자 자동 해석, id/label 컬럼 자동 탐색
# - Albumentations(transform) 입력 그대로 호환
# ============================================
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

COMMON_EXTS = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']

def resolve_image_path(img_dir: str, name: str) -> str:
    """
    - name 이 'abc123' 이면 확장자 후보를 전부 시도
    - name 이 'abc123.jpg' 같이 확장자가 포함되어도 안전하게 처리
    - 절대경로면 존재 검사 후 바로 사용
    """
    if os.path.isabs(name) and os.path.exists(name):
        return name
    base = os.path.basename(str(name))
    root, ext = os.path.splitext(base)

    candidates = []
    # 1) 그대로 시도
    candidates.append(os.path.join(img_dir, base))
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

    raise FileNotFoundError(f"이미지 경로를 찾지 못했습니다: name={name} dir={img_dir}")

class DocumentDataset(Dataset):
    """
    - csv: train.csv 또는 sample_submission.csv
    - 라벨 컬럼이 없으면 has_label=False 로 사용
    - transform: Albumentations Compose (get_train_transforms / get_valid_transforms)
    """
    def __init__(self, csv_path, img_dir, transform=None, has_label=True, id_col=None, label_col=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

        # id 컬럼 자동 탐색
        if id_col is None:
            cand_id = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower() or 'image' in c.lower()]
            self.id_col = cand_id[0] if cand_id else self.df.columns[0]
        else:
            self.id_col = id_col

        # label 컬럼 자동 탐색
        if has_label:
            if label_col is None:
                cand_lb = [c for c in self.df.columns if c.lower() in ['label', 'target', 'class', 'y']]
                self.label_col = cand_lb[0] if cand_lb else self.df.columns[-1]
            else:
                self.label_col = label_col
        else:
            self.label_col = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row[self.id_col])
        img_path = resolve_image_path(self.img_dir, name)
        # RGB 로 통일
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.has_label:
            label = row[self.label_col]
            # 라벨이 문자열일 수 있으므로, 숫자 변환은 train.py에서 encoder로 일괄 처리 권장
            try:
                label = int(label)
            except Exception:
                # 숫자가 아니면 train.py의 라벨 인코딩 규칙에 맞춰 미가공 값으로 반환
                pass
            return img, label
        else:
            return img, name
