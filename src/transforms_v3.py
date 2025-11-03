# src/transforms_v3.py
# 문서 이미지 전처리 v3: dewarp(원근)/deskew/화이트밸런스/CLAHE-샤프닝/tri-channel + 안전한 Albumentations 변환
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ───────────────────────────────────────────────────────────
# 비율 유지 + 패딩 리사이즈
def _letterbox(img, size=640, pad_value=255):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad_value, np.uint8)
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), pad_value, np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

# 최대 사각형 추출 + 원근 보정 (dewarp)
def _largest_rect_dewarp(bgr, min_area_ratio=0.2):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mor = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, ker, iterations=2)
    cnts, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_ratio * (bgr.shape[0] * bgr.shape[1]):
        return bgr

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(c)
        return bgr[y:y+h, x:x+w].copy()

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    rect[1] = pts[np.argmin(d)]   # tr
    rect[3] = pts[np.argmax(d)]   # bl

    (tl, tr, br, bl) = rect
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    W = int(max(w1, w2)); H = int(max(h1, h2))
    W = max(W, 10); H = max(H, 10)

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Hough 기반 스큐 보정
def _deskew_hough(bgr, max_deg=10):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    if lines is None:
        return bgr
    angs = []
    for rho, theta in lines[:, 0]:
        deg = (theta * 180 / np.pi) - 90
        if -max_deg <= deg <= max_deg:
            angs.append(deg)
    if not angs:
        return bgr
    angle = np.median(angs)
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Gray-world 화이트밸런스 + CLAHE + 샤프닝
def _illum_enhance(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    m = (b.mean() + g.mean() + r.mean()) / 3.0 + 1e-6
    b *= m / (b.mean() + 1e-6); g *= m / (g.mean() + 1e-6); r *= m / (r.mean() + 1e-6)
    bgr = cv2.merge([b,g,r]).clip(0,255).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    bgr = cv2.cvtColor(cv2.merge([l, a, bb]), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(bgr, (0,0), 1.0)
    return cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)

# tri-channel: gray / adaptive thr / sobel edge
def _tri_channel(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    sx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5,
                                              cv2.convertScaleAbs(sy), 0.5, 0))
    return cv2.merge([g, thr, mag])

# 통합 전처리
def _preprocess_doc(bgr, size=640):
    img = _largest_rect_dewarp(bgr)
    img = _deskew_hough(img)
    img = _illum_enhance(img)
    tri = _tri_channel(img)
    tri = _letterbox(tri, size=size, pad_value=255)
    return tri

# Albumentations 안전 Transform
class PreprocessDocV3(A.ImageOnlyTransform):
    def __init__(self, size=640, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.size = size
    def apply(self, img, **params):
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = _preprocess_doc(bgr, size=self.size)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def get_train_transforms_v3(img_size=640):
    return A.Compose([
        PreprocessDocV3(size=img_size),
        A.ShiftScaleRotate(0.02, 0.05, 5, border_mode=cv2.BORDER_REPLICATE, p=0.4),
        A.RandomBrightnessContrast(0.1, 0.15, p=0.3),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        A.CoarseDropout(max_holes=2, max_height=int(0.08*img_size),
                        max_width=int(0.08*img_size), fill_value=255, p=0.2),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])

def get_valid_transforms_v3(img_size=640):
    return A.Compose([
        PreprocessDocV3(size=img_size),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])
# src/transforms.py (파일 맨 아래 추가)
get_train_transforms = get_train_transforms_v3
get_valid_transforms = get_valid_transforms_v3
