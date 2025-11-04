# src/transforms.py
# 학습/추론은 항상 여기서만 transform을 가져가도록 통일(SSOT).
try:
    from .transforms_v3 import (
        get_train_transforms_v3,
        get_valid_transforms_v3,
        get_train_transforms,
        get_valid_transforms,
    )

# Explicitly expose stable names mapped to v3 implementations
    def get_train_transforms(img_size=640):
        return get_train_transforms_v3(img_size)

    def get_valid_transforms(img_size=640):
        return get_valid_transforms_v3(img_size)

    # ensure names exist at module level even if transforms_v3 behaves oddly
    globals()['get_train_transforms'] = get_train_transforms
    globals()['get_valid_transforms'] = get_valid_transforms

except ImportError:
    # v3 파일이 없더라도 실패하지 않도록 최소 폴백 제공
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2

    def get_train_transforms(img_size=640):
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=255),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

    def get_valid_transforms(img_size=640):
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=255),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
