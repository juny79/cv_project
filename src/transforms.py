import cv2, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _letterbox(img, size=640, pad=255):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad, np.uint8)
    
    # Scale to the target size while preserving aspect ratio
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Calculate padding
    pad_h = (size - nh) // 2
    pad_w = (size - nw) // 2
    
    # Create output array with padding
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    out[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
    return out

def _deskew_minarea(img, **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(bin_ > 0))
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _clahe_bgr(img, **kwargs):
    clip = kwargs.get('clip', 1.8)
    grid = kwargs.get('grid', 8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    cl = clahe.apply(l)
    lab = cv2.merge((cl,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def _unsharp(img, **kwargs):
    amount = kwargs.get('amount', 1.0)
    blur = cv2.GaussianBlur(img, (0,0), 1.0)
    sharp = cv2.addWeighted(img, 1+amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

class Letterbox(A.ImageOnlyTransform):
    def __init__(self, size=640, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.size = size
    def apply(self, img, **params):
        return _letterbox(img, self.size)

def get_train_transforms(img_size):
    def _clahe_wrapper(img, **params):
        return _clahe_bgr(img, clip=1.6, grid=8)
    
    def _unsharp_wrapper(img, **params):
        return _unsharp(img, amount=0.6)
    
    return A.Compose([
        # First, ensure consistent size
        Letterbox(img_size, always_apply=True, p=1.0),
        
        # Then apply augmentations
        A.Lambda(name="_deskew", image=_deskew_minarea, p=0.35),
        A.Lambda(name="_clahe", image=_clahe_wrapper, p=0.50),
        A.Lambda(name="_unsharp", image=_unsharp_wrapper, p=0.30),

        A.OneOf([
            A.Affine(scale=0.94, translate_percent=0.02, rotate=(-10,10),
                    border_mode=cv2.BORDER_REPLICATE, p=0.7),
            A.Perspective(scale=(0.02,0.05), p=0.3),
        ], p=0.6),

        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.25),

        A.OneOf([
            A.Blur(blur_limit=(3,7), p=1.0),
            # Albumentations GaussNoise now expects normalized std_range (fraction of max value)
            # previous code used var_limit=(10,50) (variance on 0-255 scale). Map to std_range ~ sqrt(var/255).
            A.GaussNoise(std_range=(0.20, 0.44), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.35),

        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size):
    def _clahe_wrapper(img, **params):
        return _clahe_bgr(img, clip=1.9, grid=8)
    
    def _unsharp_wrapper(img, **params):
        return _unsharp(img, amount=0.5)
    
    return A.Compose([
        # First, ensure consistent size
        Letterbox(img_size, always_apply=True, p=1.0),
        
        # Then apply preprocessing
        A.Lambda(name="_deskew", image=_deskew_minarea, p=1.0),
        A.Lambda(name="_clahe", image=_clahe_wrapper, p=1.0),
        A.Lambda(name="_unsharp", image=_unsharp_wrapper, p=1.0),
        
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def _gamma(img, gamma=1.0):
    if gamma is None or abs(gamma-1.0) < 1e-3:
        return img
    # Apply simple gamma correction assuming input uint8
    inv = 1.0 / max(gamma, 1e-6)
    lut = (np.linspace(0,1,256) ** inv) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

def build_valid_transform(img_size, clahe_clip=1.9, unsharp_amount=0.5, deskew=True, gamma=1.0):
    """Build a single deterministic test-time preprocessing pipeline based on parameters.

    Parameters
    ----------
    img_size : int
        Target square size for letterboxing.
    clahe_clip : float
        CLAHE clip limit.
    unsharp_amount : float
        Strength of unsharp mask (0 disables sharpening).
    deskew : bool
        Whether to apply deskew minAreaRect correction.
    gamma : float
        Gamma correction factor (>1 brighter mid-tones, <1 darker).
    """
    def _clahe_wrapper(img, **params):
        return _clahe_bgr(img, clip=clahe_clip, grid=8)
    def _unsharp_wrapper(img, **params):
        return _unsharp(img, amount=unsharp_amount) if unsharp_amount > 0 else img
    def _deskew_wrapper(img, **params):
        return _deskew_minarea(img) if deskew else img
    def _gamma_wrapper(img, **params):
        return _gamma(img, gamma=gamma)

    return A.Compose([
        Letterbox(img_size, always_apply=True, p=1.0),
        A.Lambda(name="_deskew", image=_deskew_wrapper, p=1.0),
        A.Lambda(name="_clahe", image=_clahe_wrapper, p=1.0),
        A.Lambda(name="_unsharp", image=_unsharp_wrapper, p=1.0),
        A.Lambda(name="_gamma", image=_gamma_wrapper, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_valid_transform_variants(img_size, variant_specs):
    """Return a list of (name, transform) tuples for variant test-time preprocessing.

    variant_specs: list of dicts, each may contain keys:
        - clahe_clip
        - unsharp_amount
        - deskew (bool)
        - gamma
        - name (optional label for logging)
    """
    variants = []
    for i, spec in enumerate(variant_specs):
        name = spec.get('name', f'v{i}')
        t = build_valid_transform(
            img_size,
            clahe_clip=spec.get('clahe_clip', 1.9),
            unsharp_amount=spec.get('unsharp_amount', 0.5),
            deskew=spec.get('deskew', True),
            gamma=spec.get('gamma', 1.0)
        )
        variants.append((name, t))
    return variants
