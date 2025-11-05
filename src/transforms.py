# src/transforms.py  (람다/경고 제거, partial 사용)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial
from .preprocessing import _clahe_bgr, _deskew_hough, _letterbox, _reorient_and_deskew_inference

def get_train_transforms(img_size=640):
    return A.Compose([
        A.Lambda(image=_deskew_hough, p=0.3),
        A.Lambda(image=partial(_clahe_bgr, clip=1.5, grid=8), p=0.4),
        A.ImageCompression(quality_range=(60,95), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        A.Affine(scale=(0.95,1.05), rotate=(-8,8), shear=(-4,4),
                 cval=255, mode=0, fit_output=False, p=0.35),
        A.Perspective(scale=(0.02,0.05), p=0.2),
        A.CoarseDropout(num_holes=2, max_height=int(0.08*img_size),
                        max_width=int(0.08*img_size), fill_value=255, p=0.2),
        A.Lambda(image=partial(_letterbox, size=img_size), p=1.0),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])

def get_valid_transforms(img_size=640):
    return A.Compose([
        A.Lambda(image=_reorient_and_deskew_inference, p=1.0),
        A.Lambda(image=partial(_clahe_bgr, clip=2.0, grid=8), p=1.0),
        A.Lambda(image=partial(_letterbox, size=img_size), p=1.0),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])
