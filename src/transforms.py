
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_tf(img_size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=(255,255,255)),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.OneOf([A.MotionBlur(), A.GaussianBlur(), A.MedianBlur()], p=0.3),
        A.OneOf([A.ISONoise(), A.GaussNoise(), A.MultiplicativeNoise()], p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_valid_tf(img_size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=(255,255,255)),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
