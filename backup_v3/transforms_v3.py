# src/transforms_v3.py
import cv2, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------- util: letterbox (비율 유지 + 패딩)
def letterbox(img, size=640, pad=255):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad, np.uint8)
    s = size / max(h, w)
    nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

# ---------- util: deskew (간단/안전)
def deskew_cv(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 바이너리화(약하게)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cnts) == 0:
        return image
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

# ---------- util: CLAHE (문서 대비 향상)
def apply_clahe_cv(image, **kwargs):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def _norm():
    # ImageNet 통계 (timm 기본과 호환)
    return A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225))

def get_train_transforms_v3(img_size=640):
    return A.Compose([
        A.Lambda(image=deskew_cv, p=0.5),
        A.Lambda(image=apply_clahe_cv, p=0.6),
        # test 노이즈를 흉내: 그림자/밝기대비/JPEG/노이즈/살짝 블러
        A.RandomShadow(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.35),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.15),
        # 약한 기하학 왜곡 (문서 모서리 보정 학습)
        A.Perspective(scale=(0.02, 0.05), p=0.25),
        A.Rotate(limit=12, border_mode=cv2.BORDER_REPLICATE, p=0.4),
        A.Lambda(image=lambda x, **k: letterbox(x, size=img_size), p=1.0),
        _norm(),
        ToTensorV2(),
    ])

def get_valid_transforms_v3(img_size=640):
    return A.Compose([
        A.Lambda(image=deskew_cv, p=1.0),
        A.Lambda(image=apply_clahe_cv, p=1.0),
        A.Lambda(image=lambda x, **k: letterbox(x, size=img_size), p=1.0),
        _norm(),
        ToTensorV2(),
    ])

# 호환성: 기존 import 경로 사용 코드와 연결
def get_train_transforms(img_size=640):
    return get_train_transforms_v3(img_size)

def get_valid_transforms(img_size=640):
    return get_valid_transforms_v3(img_size)
