"""OCR-specific preprocessing pipeline for keyword routing.

This module provides enhanced image preprocessing specifically optimized for OCR,
separate from classification inference preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, List, Tuple, Optional, Union

def preprocess_for_ocr(
    img: np.ndarray,
    upscale_factor: float = 2.0,
    clahe_clip: float = 2.0,
    unsharp_amount: float = 0.5,
    adaptive_method: str = 'sauvola',
    deskew: bool = True,
    denoise: bool = True,
    gamma: float = 1.0
) -> np.ndarray:
    """Apply OCR-optimized preprocessing pipeline.
    
    Args:
        img: Input image (H, W, 3) RGB uint8
        upscale_factor: Resolution upscale multiplier (default 2.0)
        clahe_clip: CLAHE clip limit (default 2.0)
        unsharp_amount: Unsharp mask strength (default 0.5)
        adaptive_method: 'sauvola', 'bradley', or 'gaussian' (default 'sauvola')
        deskew: Apply deskewing (default True)
        denoise: Apply denoising (default True)
        gamma: Gamma correction (default 1.0, no change)
    
    Returns:
        Preprocessed grayscale image ready for OCR
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # 1. Upscale for small text
    if upscale_factor != 1.0:
        h, w = gray.shape
        new_h, new_w = int(h * upscale_factor), int(w * upscale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 2. CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 3. Unsharp mask for sharpening
    if unsharp_amount > 0:
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.0 + unsharp_amount, gaussian, -unsharp_amount, 0)
    
    # 4. Denoise (bilateral filter preserves edges)
    if denoise:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 5. Gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
    
    # 6. Deskew
    if deskew:
        gray = _deskew_image(gray)
    
    # 7. Adaptive thresholding
    if adaptive_method == 'sauvola':
        binary = _sauvola_threshold(gray)
    elif adaptive_method == 'bradley':
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10
        )
    elif adaptive_method == 'gaussian':
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
    else:
        # Otsu's method fallback
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 8. Morphology cleanup (very light)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary


def _deskew_image(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Deskew image using minAreaRect."""
    # Detect edges
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    # Find largest contour and get rotation angle
    largest_contour = max(contours, key=cv2.contourArea)
    min_area_rect = cv2.minAreaRect(largest_contour)
    angle = min_area_rect[-1]
    
    # Correct angle range
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Limit rotation to avoid over-correction
    if abs(angle) > max_angle:
        return img
    
    # Rotate image
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def _sauvola_threshold(img: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola's adaptive binarization (better for documents with varying illumination)."""
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    # Compute local mean and std
    mean = cv2.blur(img, (window_size, window_size))
    mean_sq = cv2.blur(img ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    
    # Sauvola threshold
    R = 128  # Dynamic range of std (for uint8)
    threshold = mean * (1 + k * ((std / R) - 1))
    
    binary = np.where(img > threshold, 255, 0).astype(np.uint8)
    return binary


def extract_title_region(img: np.ndarray, top_percent: float = 0.20) -> np.ndarray:
    """Extract top region of image (typically contains document title)."""
    h = img.shape[0]
    top_h = int(h * top_percent)
    return img[:top_h, :]


def ocr_with_tta(
    img: np.ndarray,
    lang: str = 'kor+eng',
    psm_modes: List[int] = [6, 4, 11],
    scale_factors: List[float] = [0.9, 1.0, 1.1],
    rotation_angles: List[int] = [0, 90, 180, 270],
    select_by: str = 'longest'
) -> Dict[str, any]:
    """Run OCR with test-time augmentation and select best result.
    
    Args:
        img: Preprocessed binary image
        lang: Tesseract language (default 'kor+eng')
        psm_modes: Page segmentation modes to try
        scale_factors: Scale augmentations
        rotation_angles: Rotation augmentations (degrees)
        select_by: 'longest', 'confidence', or 'keyword_hits'
    
    Returns:
        Dict with 'text', 'confidence', 'method' keys
    """
    results = []
    
    for angle in rotation_angles:
        # Rotate image
        if angle == 0:
            rotated = img
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            continue
        
        for scale in scale_factors:
            if scale != 1.0:
                h, w = rotated.shape[:2]
                scaled = cv2.resize(rotated, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                scaled = rotated
            
            for psm in psm_modes:
                try:
                    config = f'--oem 1 --psm {psm} -l {lang}'
                    
                    # Get text and confidence
                    data = pytesseract.image_to_data(scaled, config=config, output_type=pytesseract.Output.DICT)
                    text = ' '.join([word for word in data['text'] if word.strip()])
                    
                    # Compute average confidence (exclude -1 values)
                    confs = [c for c in data['conf'] if c != -1]
                    avg_conf = np.mean(confs) if confs else 0.0
                    
                    results.append({
                        'text': text,
                        'confidence': avg_conf,
                        'method': f'rot{angle}_scale{scale:.1f}_psm{psm}',
                        'length': len(text)
                    })
                except Exception as e:
                    # Skip failed attempts
                    continue
    
    if not results:
        return {'text': '', 'confidence': 0.0, 'method': 'none'}
    
    # Select best result
    if select_by == 'longest':
        best = max(results, key=lambda x: x['length'])
    elif select_by == 'confidence':
        best = max(results, key=lambda x: x['confidence'])
    else:
        # Default to longest
        best = max(results, key=lambda x: x['length'])
    
    return best


def run_multi_engine_ocr(
    img: np.ndarray,
    use_tesseract: bool = True,
    use_easyocr: bool = True,
    use_paddleocr: bool = False,
    select_by: str = 'longest'
) -> Dict[str, any]:
    """Run multiple OCR engines and select best result.
    
    Args:
        img: Preprocessed image (grayscale or binary)
        use_tesseract: Enable Tesseract
        use_easyocr: Enable EasyOCR
        use_paddleocr: Enable PaddleOCR (requires installation)
        select_by: 'longest', 'keyword_hits', or 'confidence'
    
    Returns:
        Dict with 'text', 'engine', 'confidence' keys
    """
    results = []
    
    # Tesseract
    if use_tesseract:
        try:
            tess_result = ocr_with_tta(img, select_by='longest')
            results.append({
                'text': tess_result['text'],
                'engine': 'tesseract',
                'confidence': tess_result['confidence'],
                'length': len(tess_result['text'])
            })
        except Exception:
            pass
    
    # EasyOCR
    if use_easyocr:
        try:
            import easyocr
            reader = easyocr.Reader(['ko', 'en'], gpu=False)
            easy_results = reader.readtext(img, detail=1)
            text = ' '.join([res[1] for res in easy_results])
            conf = np.mean([res[2] for res in easy_results]) if easy_results else 0.0
            results.append({
                'text': text,
                'engine': 'easyocr',
                'confidence': conf * 100,  # Normalize to 0-100
                'length': len(text)
            })
        except Exception:
            pass
    
    # PaddleOCR
    if use_paddleocr:
        try:
            from paddleocr import PaddleOCR
            paddle = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False, show_log=False)
            paddle_results = paddle.ocr(img, cls=True)
            if paddle_results and paddle_results[0]:
                text = ' '.join([line[1][0] for line in paddle_results[0]])
                conf = np.mean([line[1][1] for line in paddle_results[0]]) * 100
                results.append({
                    'text': text,
                    'engine': 'paddleocr',
                    'confidence': conf,
                    'length': len(text)
                })
        except Exception:
            pass
    
    if not results:
        return {'text': '', 'engine': 'none', 'confidence': 0.0}
    
    # Select best
    if select_by == 'longest':
        best = max(results, key=lambda x: x['length'])
    elif select_by == 'confidence':
        best = max(results, key=lambda x: x['confidence'])
    else:
        best = max(results, key=lambda x: x['length'])
    
    return best


def enhanced_normalize_text(text: str, keep_spaces: bool = False) -> str:
    """Enhanced text normalization for keyword matching.
    
    - NFKC normalization
    - Hyphen/dash unification
    - 3+ repeated character collapse
    - Parentheses/special character removal
    - Optional space preservation
    """
    import unicodedata
    import re
    
    if not text or not isinstance(text, str):
        return ''
    
    # NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Unify various dash/hyphen variants
    text = re.sub(r'[\-‐‑‒–—﹣−]', '-', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove parentheses and content
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove special characters (keep Korean, English, numbers, hyphens)
    if keep_spaces:
        text = re.sub(r'[^0-9a-zA-Z가-힣\-\s]+', '', text)
    else:
        text = re.sub(r'[^0-9a-zA-Z가-힣\-]+', '', text)
        text = re.sub(r'\s+', '', text)
    
    # Collapse 3+ repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    return text.strip()


def count_keyword_hits(
    text: str,
    keywords: List[str],
    title_keywords: Optional[List[str]] = None,
    title_weight: float = 2.0
) -> Dict[str, any]:
    """Count keyword hits with optional title weighting.
    
    Args:
        text: Normalized text to search
        keywords: List of keyword patterns (regex)
        title_keywords: Optional separate list for title region
        title_weight: Multiplier for title keyword hits
    
    Returns:
        Dict with 'total_hits', 'body_hits', 'title_hits' keys
    """
    import re
    
    body_hits = 0
    title_hits = 0
    
    # Count body keywords
    for kw in keywords:
        try:
            pattern = re.compile(kw, re.IGNORECASE)
            body_hits += len(pattern.findall(text))
        except Exception:
            continue
    
    # Count title keywords if provided
    if title_keywords:
        for kw in title_keywords:
            try:
                pattern = re.compile(kw, re.IGNORECASE)
                title_hits += len(pattern.findall(text))
            except Exception:
                continue
    
    total_hits = body_hits + (title_hits * title_weight)
    
    return {
        'total_hits': total_hits,
        'body_hits': body_hits,
        'title_hits': title_hits
    }


# Class-specific keyword dictionaries (Korean medical documents)
MEDICAL_KEYWORDS = {
    3: {  # 입퇴원 확인서
        'core': ['입퇴원', '입원', '퇴원', '입원확인서', '퇴원확인서', '입퇴원확인서'],
        'title': ['입퇴원사실확인서', '입원퇴원확인서', '입원확인서', '퇴원확인서']
    },
    7: {  # 외래/통원 확인서
        'core': ['외래', '통원', '진료', '치료', '외래진료', '통원진료'],
        'title': ['통원진료확인서', '외래진료확인서', '통원치료확인서', '진료확인서']
    },
    4: {  # 진단서
        'core': ['진단서', '진단명', '진단내용', '진단소견'],
        'title': ['진단서', '의사진단서', '진단확인서']
    },
    14: {  # 소견서
        'core': ['소견서', '소견', '의견서', '소견내용'],
        'title': ['의학적소견서', '소견서', '의견서']
    }
}
