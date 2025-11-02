# ============================================================
# 📘 Document Image Preprocessing Pipeline v2 (F1-Optimized)
# ============================================================
import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import functional as F

# ============================================================
# 1️⃣ 재현성 보장
# ============================================================
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================
# 2️⃣ 경로 설정
# ============================================================
BASE_DIR = '/root/cv_project/data'
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')
TEST_CSV = os.path.join(BASE_DIR, 'sample_submission.csv')
TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train')
TEST_IMG_DIR = os.path.join(BASE_DIR, 'test')
SAVE_DIR = os.path.join(BASE_DIR, 'processed_v2')
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 3️⃣ 유틸 함수: Skew 교정 + CLAHE + Padding 유지형 Resize
# ============================================================
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def resize_with_padding(img, target_size=512, pad_value=255):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_h = (target_size - nh) // 2
    pad_w = (target_size - nw) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, target_size-nh-pad_h, pad_w, target_size-nw-pad_w,
                                cv2.BORDER_CONSTANT, value=pad_value)
    return padded

# ============================================================
# 4️⃣ Albumentations Transform (Class-aware)
# ============================================================
train_transform = A.Compose([
    A.Lambda(image=deskew, p=0.5),
    A.Lambda(image=apply_clahe, p=0.5),
    A.RandomShadow(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.GaussNoise(p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    A.Lambda(image=lambda x, **k: resize_with_padding(x, 512), p=1.0),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Lambda(image=deskew, p=1.0),
    A.Lambda(image=apply_clahe, p=1.0),
    A.Lambda(image=lambda x, **k: resize_with_padding(x, 512), p=1.0),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ============================================================
# 5️⃣ Dataset 정의
# ============================================================
class DocumentDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.id_col = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.label_col = [c for c in self.df.columns if c.lower() in ['label', 'target', 'class']]
        self.id_col = self.id_col[0] if self.id_col else self.df.columns[0]
        self.label_col = self.label_col[0] if self.label_col else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.id_col])
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        img = np.array(Image.open(img_path).convert("RGB"))
        label = int(row[self.label_col]) if self.label_col else -1
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

# ============================================================
# 6️⃣ 텐서 저장 함수 (GPU-friendly 캐시)
# ============================================================
def save_preprocessed_tensors(dataset, name="train", batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_images, all_labels = [], []

    print(f"🔹 {name} 데이터 전처리 중...")
    for imgs, labels in tqdm(loader):
        all_images.append(imgs)
        all_labels.append(labels)

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    torch.save({
        'images': all_images,
        'labels': all_labels
    }, os.path.join(SAVE_DIR, f"{name}_processed.pt"))
    print(f"✅ {name}_processed.pt 저장 완료 ({all_images.shape[0]}개 이미지)")

# ============================================================
# 7️⃣ 실행
# ============================================================
if __name__ == "__main__":
    print("📘 고도화된 문서 이미지 전처리 파이프라인 시작")

    train_ds = DocumentDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=train_transform)
    test_ds = DocumentDataset(TEST_CSV, TEST_IMG_DIR, transform=test_transform)

    save_preprocessed_tensors(train_ds, name="train")
    save_preprocessed_tensors(test_ds, name="test")

    print("✨ 모든 전처리 완료 및 저장 완료 (F1-Optimized v2)")
