# ============================================================
# 📘 Document Image Preprocessing Pipeline
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

# ============================================================
# 1️⃣ 시드 고정 (재현성 보장)
# ============================================================
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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

SAVE_DIR = os.path.join(BASE_DIR, 'processed')
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 3️⃣ Albumentations 전처리 정의
# ============================================================
train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
    A.Perspective(p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# ============================================================
# 4️⃣ Custom Dataset
# ============================================================
class DocumentImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # 컬럼 자동 탐색
        self.id_col = [c for c in self.df.columns if 'id' in c.lower() or 'file' in c.lower()]
        self.label_col = [c for c in self.df.columns if c.lower() in ['label', 'target', 'class']]

        self.id_col = self.id_col[0] if self.id_col else self.df.columns[0]
        self.label_col = self.label_col[0] if self.label_col else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row[self.id_col])
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {img_path}")

        img = np.array(Image.open(img_path).convert("RGB"))
        label = int(row[self.label_col]) if self.label_col else -1

        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

# ============================================================
# 5️⃣ 데이터셋 로드 및 저장
# ============================================================
def save_preprocessed_tensors(dataset, name="train"):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    all_images, all_labels = [], []

    print(f"🔹 {name} 데이터 전처리 및 텐서 저장 중...")
    for imgs, labels in tqdm(loader):
        all_images.append(imgs)
        all_labels.append(labels)

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    torch.save({
        'images': all_images,
        'labels': all_labels
    }, os.path.join(SAVE_DIR, f'{name}_processed.pt'))

    print(f"✅ {name}_processed.pt 저장 완료 ({all_images.shape[0]}개 이미지)")

# ============================================================
# 6️⃣ 실행 (Train/Test 자동 처리)
# ============================================================
if __name__ == "__main__":
    print("📘 문서 이미지 전처리 파이프라인 시작")

    train_dataset = DocumentImageDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=train_transform)
    test_dataset = DocumentImageDataset(TEST_CSV, TEST_IMG_DIR, transform=test_transform)

    save_preprocessed_tensors(train_dataset, name="train")
    save_preprocessed_tensors(test_dataset, name="test")

    print("✨ 모든 전처리 완료 및 저장 완료")
