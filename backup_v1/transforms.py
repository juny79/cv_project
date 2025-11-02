import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def deskew_cv(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def apply_clahe_cv(image, **kwargs):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def resize_with_padding(image, size=512, **kwargs):
    h, w = image.shape[:2]
    if max(h, w) == 0:  # 안전장치
        return cv2.resize(image, (size, size))
    scale = size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(image, (nw, nh))
    pad_h = (size - nh) // 2
    pad_w = (size - nw) // 2
    return cv2.copyMakeBorder(resized, pad_h, size-nh-pad_h, pad_w, size-nw-pad_w,
                              cv2.BORDER_CONSTANT, value=255)

def get_train_transforms(img_size=512):
    return A.Compose([
        A.Lambda(image=deskew_cv, p=0.5),
        A.Lambda(image=apply_clahe_cv, p=0.5),
        A.RandomShadow(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.GaussNoise(p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.Lambda(image=lambda x, **k: resize_with_padding(x, img_size), p=1.0),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])

def get_valid_transforms(img_size=512):
    return A.Compose([
        A.Lambda(image=deskew_cv, p=1.0),
        A.Lambda(image=apply_clahe_cv, p=1.0),
        A.Lambda(image=lambda x, **k: resize_with_padding(x, img_size), p=1.0),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])
