# src/preprocessing.py
import cv2, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------- helpers ----------
def _letterbox(img, size, pad=255):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad, np.uint8)
    s = size / max(h, w)
    nh, nw = int(h * s), int(w * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bot = (size - nh) // 2, size - nh - (size - nh) // 2
    left, right = (size - nw) // 2, size - nw - (size - nw) // 2
    return cv2.copyMakeBorder(img, top, bot, left, right,
                              cv2.BORDER_CONSTANT, value=(pad, pad, pad))

def _clahe_bgr(x, clip=2.0, grid=8):
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def _hough_deskew(x):
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    g = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLines(g, 1, np.pi/180, 120)
    if lines is None: 
        return x
    angles = [(theta - np.pi/2) for rho, theta in lines[:,0]]
    ang = np.median(angles) * 180/np.pi
    h, w = x.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
    return cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def _reorient_and_deskew_inference(x):
    # 0/90/180/270 중 텍스트 대비(에지 합)가 가장 큰 방향 선택
    cand = [x]
    for k in [1,2,3]:
        cand.append(np.rot90(x, k).copy())
    scores = []
    for c in cand:
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        scores.append(cv2.Sobel(g, cv2.CV_32F, 1, 0).var() + cv2.Sobel(g, 0, 1, 0).var())
    x = cand[int(np.argmax(scores))]
    return _hough_deskew(x)

# ---------- pipelines ----------
def get_train_transforms(img_size: int):
    return A.Compose([
        A.Lambda(image=lambda x, **k: _hough_deskew(x), p=0.25),
        A.Lambda(image=lambda x, **k: _clahe_bgr(x, clip=1.8, grid=8), p=0.4),

        A.ImageCompression(quality_range=(35, 95), p=0.35),
        A.MotionBlur(blur_limit=5, p=0.15),
        A.Blur(blur_limit=3, p=0.15),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.25),
        A.Affine(scale=(0.95, 1.05), shear=(-5, 5),
                 translate_percent=(0.02, 0.04), p=0.25),
        A.Perspective(scale=(0.02, 0.06), p=0.25),

        A.RandomBrightnessContrast(p=0.25),
        A.Lambda(image=lambda x, **k: _letterbox(x, img_size), p=1.0),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])

def get_valid_transforms(img_size: int):
    return A.Compose([
        A.Lambda(image=lambda x, **k: _reorient_and_deskew_inference(x), p=1.0),
        A.Lambda(image=lambda x, **k: _clahe_bgr(x, clip=2.0, grid=8), p=1.0),
        A.Lambda(image=lambda x, **k: _letterbox(x, img_size), p=1.0),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])
