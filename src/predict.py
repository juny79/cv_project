# predict.py
import os, re, unicodedata, yaml, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.transforms import get_valid_transforms, get_valid_transform_variants
try:
    from src.ocr_pipeline import (
        preprocess_for_ocr, ocr_with_tta, run_multi_engine_ocr,
        extract_title_region, enhanced_normalize_text, count_keyword_hits,
        MEDICAL_KEYWORDS
    )
    OCR_PIPELINE_AVAILABLE = True
except ImportError:
    OCR_PIPELINE_AVAILABLE = False
    print("[Warning] ocr_pipeline module not found, runtime OCR disabled")

# ---------------------------
# 기본 상수 & 유틸
# ---------------------------
COMMON_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG', '')

def load_cfg(p):
    """YAML 로드 (configs/ 접두 자동 처리)"""
    try:
        path = p
        if not os.path.exists(path):
            alt = os.path.join('configs', p)
            if os.path.exists(alt):
                path = alt
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a mapping/dictionary")
        return cfg
    except (OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load config {p}: {e}")

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"image not found for {stem} under {img_dir}")

# ---------------------------
# 데이터셋
# ---------------------------
class TestDS(Dataset):
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        idc = [c for c in self.df.columns if c.lower() in ['id','image_id','filename']]
        self.id_col = idc[0] if idc else self.df.columns[0]
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, i):
        stem = str(self.df.iloc[i][self.id_col])
        p = resolve_image_path(self.img_dir, stem)
        img = np.array(Image.open(p).convert('RGB'))
        x = self.transform(image=img)['image']
        return x, stem

# ---------------------------
# 텍스트 정규화 & OCR 키워드
# ---------------------------
def _normalize_text(s, opts=None):
    """키워드 매칭을 위한 정규화."""
    if not s or not isinstance(s, str):
        return ''
    if opts is None:
        opts = {}
    to_lower         = bool(opts.get('to_lower', True))
    remove_spaces    = bool(opts.get('remove_spaces', True))
    remove_punct     = bool(opts.get('remove_punct', True))
    remove_digits    = bool(opts.get('remove_digits', True))
    use_nfkc         = bool(opts.get('nfkc', True))
    keep_hyphens     = bool(opts.get('keep_hyphens', True))
    collapse_repeats = bool(opts.get('collapse_repeats', True))

    if use_nfkc:
        s = unicodedata.normalize('NFKC', s)
    # 하이픈 계열 통합
    s = re.sub(r'[\-‐-‒–—﹣−]', '-', s)
    if to_lower:
        s = s.lower()
    if remove_spaces:
        s = re.sub(r'\s+', '', s)
    if remove_punct:
        if keep_hyphens:
            s = re.sub(r'[^0-9a-zA-Z가-힣\-]+', '', s)
        else:
            s = re.sub(r'[^0-9a-zA-Z가-힣]+', '', s)
    if remove_digits:
        s = re.sub(r'\d+', '', s)
    if collapse_repeats:
        s = re.sub(r'(.)\1{2,}', r'\1', s)
    return s

# 핵심 + 확장 패턴 (core/wide)
KEYWORDS = {
    3: {  # 입퇴원 확인서 계열
        'core': [r'입퇴원', r'입\s*퇴원', r'입[ \-–—]*퇴원', r'입원', r'퇴원', r'입원퇴원', r'입원\s*퇴원'],
        'wide': [
            r'입퇴원사실(확인서|증명서|증명원)?', r'입퇴원(확인서|증명서|증명원)',
            r'입원사실(확인서|증명서|증명원)?', r'입원(확인서|증명서|증명원)', r'퇴원(확인서|증명서|증명원)',
            r'입원진료확인서', r'입원요약지', r'입원퇴원증명서', r'입원퇴원확인서'
        ]
    },
    7: {  # 외래/통원 확인서 계열
        'core': [r'통원', r'외래', r'진료', r'치료', r'통원진료', r'외래진료', r'통원치료', r'통원\s*진료', r'외래\s*진료'],
        'wide': [
            r'통원(진료확인서|치료사실확인서|치료사실증명서|사실확인서|사실증명서|사실증명원|확인서)?',
            r'외래진료사실확인서', r'진료사실(확인서|증명서|증명원)', r'통원치료사실확인서',
            r'진료(입원|통원)?확인서', r'치료확인서', r'확인서'
        ]
    },
    4: {  # 진단서
        'core': [r'진단서', r'진단명', r'의사진단서', r'진단내용'],
        'wide': [r'진단(확인서|증명서|증명원)', r'진단소견']
    },
    14: { # 소견서
        'core': [r'소견서', r'소견', r'의견서', r'소견내용'],
        'wide': [r'의학적소견', r'소견(확인서|증명서|증명원)']
    }
}

def _build_compiled_keywords(tr_cfg):
    """KEYWORDS + config 추가/제거/CSV 로딩 → 컴파일."""
    merged = {}
    for cid, groups in KEYWORDS.items():
        merged[cid] = {
            'core': list(groups.get('core', [])),
            'wide': list(groups.get('wide', []))
        }
    # 추가 패턴
    extra = tr_cfg.get('extra_keywords', {}) or {}
    for k, pats in extra.items():
        try:
            cid = int(k)
        except Exception:
            continue
        merged.setdefault(cid, {'core': [], 'wide': []})
        for p in (pats or []):
            if p not in merged[cid]['core']:
                merged[cid]['core'].append(p)
    # 제거 패턴
    remove = tr_cfg.get('remove_keywords', {}) or {}
    for k, pats in remove.items():
        try:
            cid = int(k)
        except Exception:
            continue
        if cid not in merged:
            continue
        merged[cid]['core'] = [p for p in merged[cid]['core'] if p not in set(pats or [])]
    # CSV에서 패턴 읽기
    kw_file = tr_cfg.get('keyword_file', '')
    if kw_file and os.path.exists(kw_file):
        try:
            df = pd.read_csv(kw_file)
            cc = [c for c in df.columns if str(c).lower() in ['class','cls','label','y']]
            pc = [c for c in df.columns if str(c).lower() in ['pattern','regex','keyword']]
            if cc and pc:
                ccol, pcol = cc[0], pc[0]
                for _, r in df.iterrows():
                    try:
                        cid = int(r[ccol])
                    except Exception:
                        continue
                    pat = str(r[pcol]) if pd.notna(r[pcol]) else ''
                    if not pat:
                        continue
                    merged.setdefault(cid, {'core': [], 'wide': []})
                    if pat not in merged[cid]['core']:
                        merged[cid]['core'].append(pat)
        except Exception:
            pass
    compiled = {}
    for cid, groups in merged.items():
        compiled[cid] = {
            'core': [re.compile(p) for p in groups.get('core', [])],
            'wide': [re.compile(p) for p in groups.get('wide', [])]
        }
    return compiled

def _count_hits_groups(text, cls_id, compiled_kw=None, norm_opts=None):
    norm = _normalize_text(text, norm_opts)
    if compiled_kw is None:
        groups = KEYWORDS.get(cls_id, {})
        core = [re.compile(p) for p in groups.get('core', [])]
        wide = [re.compile(p) for p in groups.get('wide', [])]
    else:
        entry = compiled_kw.get(cls_id, {'core': [], 'wide': []})
        core = entry.get('core', [])
        wide = entry.get('wide', [])
    hc = sum(1 for rgx in core if rgx.search(norm))
    hw = sum(1 for rgx in wide if rgx.search(norm))
    return hc, hw

def _load_ocr_map(csv_path, id_col='id'):
    """Load OCR text map from CSV."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Find id and text columns
    idc = [c for c in df.columns if c.lower() in ['id','image_id','filename']]
    txtc = [c for c in df.columns if c.lower() in ['text','ocr','ocr_text']]
    if not idc or not txtc:
        return None
    id_col_actual = idc[0]; txt_col = txtc[0]
    ocr_map = {}
    for _, row in df.iterrows():
        img_id = str(row[id_col_actual])
        text = row[txt_col] if pd.notna(row[txt_col]) else ''
        ocr_map[img_id] = text
    return ocr_map

def _perform_runtime_ocr(img_path: str, ocr_cfg: dict, cls_a: int, cls_b: int) -> dict:
    """Perform enhanced runtime OCR on a candidate image.
    
    Returns dict with 'text', 'title_text', 'engine', 'confidence' keys.
    """
    import cv2
    
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return {'text': '', 'title_text': '', 'engine': 'failed', 'confidence': 0.0}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocessing parameters
        upscale = float(ocr_cfg.get('upscale_factor', 2.0))
        clahe_clip = float(ocr_cfg.get('clahe_clip', 2.0))
        unsharp = float(ocr_cfg.get('unsharp_amount', 0.5))
        adaptive_method = str(ocr_cfg.get('adaptive_method', 'sauvola'))
        deskew = bool(ocr_cfg.get('deskew', True))
        denoise = bool(ocr_cfg.get('denoise', True))
        gamma = float(ocr_cfg.get('gamma', 1.0))
        
        # Preprocess for OCR
        processed = preprocess_for_ocr(
            img_rgb, 
            upscale_factor=upscale,
            clahe_clip=clahe_clip,
            unsharp_amount=unsharp,
            adaptive_method=adaptive_method,
            deskew=deskew,
            denoise=denoise,
            gamma=gamma
        )
        
        # Extract title region if enabled
        use_title = bool(ocr_cfg.get('use_title_region', True))
        title_text = ''
        if use_title:
            title_region = extract_title_region(processed, top_percent=0.20)
            title_result = ocr_with_tta(
                title_region,
                psm_modes=[6],  # Single block
                rotation_angles=[0],  # No rotation for title
                scale_factors=[1.0],
                select_by='longest'
            )
            title_text = title_result.get('text', '')
        
        # Full page OCR
        use_multi_engine = bool(ocr_cfg.get('use_multi_engine', False))
        
        if use_multi_engine:
            # Multi-engine approach
            result = run_multi_engine_ocr(
                processed,
                use_tesseract=True,
                use_easyocr=bool(ocr_cfg.get('use_easyocr', False)),
                use_paddleocr=bool(ocr_cfg.get('use_paddleocr', False)),
                select_by='longest'
            )
            full_text = result.get('text', '')
            engine = result.get('engine', 'unknown')
            confidence = result.get('confidence', 0.0)
        else:
            # Tesseract with TTA
            psm_modes = ocr_cfg.get('psm_modes', [6, 4, 11])
            result = ocr_with_tta(
                processed,
                psm_modes=psm_modes,
                rotation_angles=[0, 90, 180, 270],
                scale_factors=[0.9, 1.0, 1.1],
                select_by='longest'
            )
            full_text = result.get('text', '')
            engine = 'tesseract_tta'
            confidence = result.get('confidence', 0.0)
        
        return {
            'text': full_text,
            'title_text': title_text,
            'engine': engine,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"[RuntimeOCR] Error processing {img_path}: {e}")
        return {'text': '', 'title_text': '', 'engine': 'error', 'confidence': 0.0}

# ---------------------------
# OCR 키워드 라우팅 (2-스테이지)
# ---------------------------
def apply_keyword_routing(cfg, test_ds, probs_img, preds):
    tr_cfg = cfg.get('text_routing', {}) or {}
    if not tr_cfg.get('enable', False):
        return preds
    source = tr_cfg.get('source', '')
    pairs  = tr_cfg.get('pairs', [[3,7], [4,14]])

    # Strict/Wide 임계
    delta_thr_strict = float(tr_cfg.get('delta_thr_strict', tr_cfg.get('delta_thr', 0.04)))
    conf_thr_strict  = float(tr_cfg.get('conf_thr_strict',  tr_cfg.get('conf_thr', 0.55)))
    min_hits_strict  = int(tr_cfg.get('min_hits_strict',    tr_cfg.get('min_hits', 2)))
    delta_thr_wide   = float(tr_cfg.get('delta_thr_wide',   max(delta_thr_strict, 0.08)))
    conf_thr_wide    = float(tr_cfg.get('conf_thr_wide',    max(conf_thr_strict,  0.55)))
    min_hits_wide    = int(tr_cfg.get('min_hits_wide',      max(min_hits_strict+1, 3)))
    entropy_thr_wide = float(tr_cfg.get('entropy_thr_wide', 1e9))
    margin_thr_wide  = float(tr_cfg.get('margin_thr_wide',  1e9))
    prefer_top       = bool(tr_cfg.get('prefer_top', True))
    easy_exclude     = bool(tr_cfg.get('easy_exclude', True))
    
    # Runtime OCR 설정
    ocr_runtime_cfg = tr_cfg.get('ocr_runtime', {}) or {}
    use_runtime_ocr = bool(ocr_runtime_cfg.get('enable', False)) and OCR_PIPELINE_AVAILABLE
    if use_runtime_ocr:
        print(f"[KeywordRouting] Runtime OCR enabled for candidates")

    ocr_map = _load_ocr_map(source)
    if ocr_map is None:
        print(f"[KeywordRouting] OCR source not found or invalid: {source}, skipping")
        return preds

    # Easy-Lock 설정(라우팅에서도 제외)
    ensemble_cfg  = cfg.get('ensemble', {})
    easy_classes  = ensemble_cfg.get('easy_classes', [])
    easy_conf     = float(ensemble_cfg.get('easy_conf', 0.92))
    easy_conf_map = ensemble_cfg.get('easy_conf_map', {})

    compiled_kw = _build_compiled_keywords(tr_cfg)
    norm_opts   = tr_cfg.get('normalization', {}) or {}

    dbg_cfg   = tr_cfg.get('debug', {}) or {}
    dbg_en    = bool(dbg_cfg.get('enable', False))
    dbg_rows  = []

    routed = 0; hit_route = 0; tie_route = 0
    new_preds = preds.copy()

    # 전역 게이트용 통계
    eps = 1e-9
    ent = -(probs_img * np.log(probs_img + eps)).sum(1)
    part = np.partition(-probs_img, 1, axis=1)
    top1 = -part[:,0]; top2 = -part[:,1]
    margin = top1 - top2

    total_near_strict = 0
    total_near_wide   = 0
    total_locked      = 0
    total_eligible    = 0
    stage_counts = {'strict': 0, 'wide': 0}

    for (a, b) in pairs:
        pa = probs_img[:, a]; pb = probs_img[:, b]
        delta = np.abs(pa - pb)
        maxc  = np.maximum(pa, pb)

        # 1) Strict
        m_strict = (delta < delta_thr_strict) & (maxc > conf_thr_strict)
        idx_s = np.where(m_strict)[0]
        total_near_strict += len(idx_s)

        for i in idx_s:
            pred_cls = preds[i]
            conf_i   = probs_img[i, pred_cls]
            # easy lock 제외
            if easy_exclude and (pred_cls in easy_classes):
                thr = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf_i >= thr:
                    total_locked += 1
                    continue
            total_eligible += 1
            img_id = str(test_ds.df.iloc[i][test_ds.id_col])
            ocr_tx = ocr_map.get(img_id, '')
            
            # Runtime OCR for candidates with low-quality or empty OCR (strict stage)
            ocr_engine_used = 'preloaded'
            ocr_length_before = len(ocr_tx)
            if use_runtime_ocr and (not ocr_tx or len(ocr_tx) < ocr_runtime_cfg.get('min_text_length', 10)):
                try:
                    img_path = resolve_image_path(test_ds.img_dir, img_id)
                    ocr_result = _perform_runtime_ocr(img_path, ocr_runtime_cfg, a, b)
                    ocr_tx = ocr_result['text']
                    ocr_engine_used = ocr_result['engine']
                    # Optionally combine title text with higher weight
                    if ocr_result.get('title_text'):
                        ocr_tx = ocr_result['title_text'] + ' ' + ocr_tx
                    print(f"[RuntimeOCR] {img_id}: {ocr_length_before}→{len(ocr_tx)} chars, engine={ocr_engine_used}")
                except Exception as e:
                    print(f"[RuntimeOCR] Error on {img_id}: {e}")

            core_a, wide_a = _count_hits_groups(ocr_tx, a, compiled_kw=compiled_kw, norm_opts=norm_opts)
            core_b, wide_b = _count_hits_groups(ocr_tx, b, compiled_kw=compiled_kw, norm_opts=norm_opts)

            eff_a = core_a if core_a >= min_hits_strict else 0
            eff_b = core_b if core_b >= min_hits_strict else 0

            if max(eff_a, eff_b) > 0:
                reason = ''
                if eff_a > eff_b:
                    new_preds[i] = a; routed += 1; hit_route += 1; stage_counts['strict'] += 1; reason='strict_hits_a_core'
                elif eff_b > eff_a:
                    new_preds[i] = b; routed += 1; hit_route += 1; stage_counts['strict'] += 1; reason='strict_hits_b_core'
                elif prefer_top:
                    new_preds[i] = a if pa[i] >= pb[i] else b
                    routed += 1; tie_route += 1; stage_counts['strict'] += 1; reason='strict_tie_prefer_top'
                if dbg_en and reason:
                    dbg_rows.append({
                        'id': img_id, 'pair': f'{a}-{b}', 'stage':'strict',
                        'prev_pred': int(pred_cls), 'new_pred': int(new_preds[i]),
                        'pa': float(pa[i]), 'pb': float(pb[i]),
                        'delta': float(abs(pa[i]-pb[i])), 'entropy': float(ent[i]), 'margin': float(margin[i]),
                        'core_a': int(core_a), 'wide_a': int(wide_a),
                        'core_b': int(core_b), 'wide_b': int(wide_b),
                        'min_hits_strict': int(min_hits_strict), 'reason': reason
                    })

        # 2) Wide (entropy/margin 추가)
        m_wide = (delta < delta_thr_wide) & (delta >= delta_thr_strict) & (maxc > conf_thr_wide)
        m_wide = m_wide & (ent > entropy_thr_wide) & (margin < margin_thr_wide)
        idx_w = np.where(m_wide)[0]
        total_near_wide += len(idx_w)

        for i in idx_w:
            pred_cls = preds[i]
            conf_i   = probs_img[i, pred_cls]
            if easy_exclude and (pred_cls in easy_classes):
                thr = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf_i >= thr:
                    total_locked += 1
                    continue
            total_eligible += 1
            img_id = str(test_ds.df.iloc[i][test_ds.id_col])
            ocr_tx = ocr_map.get(img_id, '')
            
            # Runtime OCR for candidates with low-quality or empty OCR (wide stage)
            ocr_engine_used = 'preloaded'
            ocr_length_before = len(ocr_tx)
            if use_runtime_ocr and (not ocr_tx or len(ocr_tx) < ocr_runtime_cfg.get('min_text_length', 10)):
                try:
                    img_path = resolve_image_path(test_ds.img_dir, img_id)
                    ocr_result = _perform_runtime_ocr(img_path, ocr_runtime_cfg, a, b)
                    ocr_tx = ocr_result['text']
                    ocr_engine_used = ocr_result['engine']
                    if ocr_result.get('title_text'):
                        ocr_tx = ocr_result['title_text'] + ' ' + ocr_tx
                    print(f"[RuntimeOCR] {img_id}: {ocr_length_before}→{len(ocr_tx)} chars, engine={ocr_engine_used}")
                except Exception as e:
                    print(f"[RuntimeOCR] Error on {img_id}: {e}")

            core_a, wide_a = _count_hits_groups(ocr_tx, a, compiled_kw=compiled_kw, norm_opts=norm_opts)
            core_b, wide_b = _count_hits_groups(ocr_tx, b, compiled_kw=compiled_kw, norm_opts=norm_opts)

            total_a = core_a + wide_a
            total_b = core_b + wide_b
            eff_a = core_a if core_a >= min_hits_strict else (total_a if total_a >= min_hits_wide else 0)
            eff_b = core_b if core_b >= min_hits_strict else (total_b if total_b >= min_hits_wide else 0)

            if max(eff_a, eff_b) > 0:
                reason = ''
                if eff_a > eff_b:
                    new_preds[i] = a; routed += 1; hit_route += 1; stage_counts['wide'] += 1; reason='wide_hits_a'
                elif eff_b > eff_a:
                    new_preds[i] = b; routed += 1; hit_route += 1; stage_counts['wide'] += 1; reason='wide_hits_b'
                elif prefer_top:
                    new_preds[i] = a if pa[i] >= pb[i] else b
                    routed += 1; tie_route += 1; stage_counts['wide'] += 1; reason='wide_tie_prefer_top'
                if dbg_en and reason:
                    dbg_rows.append({
                        'id': img_id, 'pair': f'{a}-{b}', 'stage':'wide',
                        'prev_pred': int(pred_cls), 'new_pred': int(new_preds[i]),
                        'pa': float(pa[i]), 'pb': float(pb[i]),
                        'delta': float(abs(pa[i]-pb[i])), 'entropy': float(ent[i]), 'margin': float(margin[i]),
                        'core_a': int(core_a), 'wide_a': int(wide_a),
                        'core_b': int(core_b), 'wide_b': int(wide_b),
                        'min_hits_strict': int(min_hits_strict), 'min_hits_wide': int(min_hits_wide),
                        'reason': reason
                    })

    print(f"[KeywordRouting] Candidates strict={total_near_strict}, wide={total_near_wide}, locked: {total_locked}, eligible: {total_eligible}")
    print(f"[KeywordRouting] Routed: {routed} (by hits: {hit_route}, by tie→top: {tie_route}) | per-stage: {stage_counts} | pairs={pairs}")

    if dbg_en and dbg_rows:
        try:
            out_csv = dbg_cfg.get('csv_path')
            if not out_csv:
                out_dir = cfg.get('paths', {}).get('out_dir', './outputs')
                os.makedirs(out_dir, exist_ok=True)
                out_csv = os.path.join(out_dir, 'keyword_routing_debug.csv')
            import csv
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(dbg_rows[0].keys()))
                w.writeheader(); w.writerows(dbg_rows)
            print(f"[KeywordRouting] Debug CSV saved → {out_csv}")
        except Exception as e:
            print(f"[KeywordRouting] Failed to save debug CSV: {e}")
    return new_preds

# ---------------------------
# 보조 피처 생성 / 페어 리파이너
# ---------------------------
def feats_from_probs_np(p):
    eps = 1e-9
    top2 = np.sort(p, axis=1)[:, -2:]
    margin = top2[:,1] - top2[:,0]
    ent = -(p*np.log(p+eps)).sum(1, keepdims=True)
    return np.concatenate([p, ent, margin[:,None]], 1)

def apply_pair_refiner_with_lock(P, preds, a, b, clf_pair, delta_thr, conf_thr, easy_cfg, pair_gate_cfg=None, global_ent=None, global_margin=None):
    if clf_pair is None:
        return preds
    new_preds = preds.copy()
    pa, pb = P[:, a], P[:, b]
    base_mask = (np.abs(pa - pb) < delta_thr) & (np.maximum(pa, pb) > conf_thr)
    mask = base_mask
    if pair_gate_cfg is not None:
        ent_thr = float(pair_gate_cfg.get('entropy_thr', 1e9))
        mar_thr = float(pair_gate_cfg.get('margin_thr', 1e9))
        if global_ent is not None:
            mask = mask & (global_ent > ent_thr)
        if global_margin is not None:
            mask = mask & (global_margin < mar_thr)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return new_preds
    # Easy-Lock 제외
    if easy_cfg:
        easy_classes = easy_cfg.get('easy_classes', [])
        easy_conf    = float(easy_cfg.get('easy_conf', 0.92))
        easy_conf_map= easy_cfg.get('easy_conf_map', {})
        unlocked = []
        for i in idx:
            pred_cls = preds[i]
            conf     = P[i, pred_cls]
            if pred_cls in easy_classes:
                thr = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf >= thr:
                    continue
            unlocked.append(i)
        idx = np.array(unlocked)
    if len(idx) == 0:
        return new_preds
    ent = -(P[idx]*np.log(P[idx]+1e-9)).sum(1, keepdims=True)
    X = np.concatenate([pa[idx][:,None], pb[idx][:,None], ent, (pb[idx]-pa[idx])[:,None]], 1)
    z = clf_pair.predict(X)
    new_preds[idx] = np.where(z==1, b, a)
    return new_preds

# ---------------------------
# 메인 추론 파이프라인
# ---------------------------
@torch.no_grad()
def predict_ensemble(config_path, tta=4):
    cfg = load_cfg(config_path)
    paths, data = cfg['paths'], cfg['data']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = int(data['img_size'])
    base_tfm = get_valid_transforms(img_size)

    # 전처리 variant
    inf_cfg = cfg.get('inference', {}) or {}
    pp_cfg  = (inf_cfg.get('preproc_variants', {}) or {})
    use_variants = bool(pp_cfg.get('enable', False)) and len(pp_cfg.get('variants', [])) > 0
    combine_mode = str(pp_cfg.get('combine', 'mean')).lower()
    if use_variants:
        variants = get_valid_transform_variants(img_size, pp_cfg.get('variants', []))
        print(f"[Preproc] Using {len(variants)} variants, combine={combine_mode}")
    else:
        variants = [("base", base_tfm)]

    num_workers = 0
    pin_memory  = device.startswith('cuda')
    infer_bs    = int(inf_cfg.get('batch_size', int(cfg['train']['batch_size'])))

    # 외부 메타/페어 분류기 로드(있을 때만)
    project_root = os.path.abspath(os.path.join(paths['base_dir'], '..'))
    meta_path   = os.path.join(project_root, 'extern', 'meta_full.joblib')
    pair37_path = os.path.join(project_root, 'extern', 'pair_3_7.joblib')
    pair414_path= os.path.join(project_root, 'extern', 'pair_4_14.joblib')
    META_CLF = joblib.load(meta_path) if os.path.exists(meta_path) else None
    PAIR_37  = joblib.load(pair37_path) if os.path.exists(pair37_path) else None
    PAIR_414 = joblib.load(pair414_path) if os.path.exists(pair414_path) else None

    # 체크포인트
    ckpts, folds = [], data.get('folds', [0,1,2,3,4])
    for k in folds:
        p = os.path.join(paths['out_dir'], f'fold{k}', 'best.pt')
        if os.path.exists(p):
            ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {paths['out_dir']}/fold*/best.pt")

    # 클래스 수 추정
    state0 = torch.load(ckpts[0], map_location='cpu')
    head_weight = None
    for k, v in state0['model'].items():
        if ('head.fc.weight' in k or 'classifier.weight' in k or 'fc.weight' in k) and hasattr(v, 'shape'):
            head_weight = v; break
    num_classes = head_weight.shape[0] if head_weight is not None else int(cfg.get('data', {}).get('num_classes', 17))

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    use_amp = bool(cfg.get('inference', {}).get('amp', True)) and device.startswith('cuda')

    # TTA 함수
    def _four_rot_logits(xb):
        outs = []
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))                                     # 0
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))         # 90
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))         # 180
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))         # 270
        return outs

    def _eight_rot_logits(xb):
        outs = []
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))
        # 45·135·225·315° (affine)
        B, C, H, W = xb.shape
        for angle in [45, 135, 225, 315]:
            theta = torch.tensor([
                [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]
            ], dtype=xb.dtype, device=xb.device).unsqueeze(0).repeat(B,1,1)
            grid = F.affine_grid(theta, size=xb.size(), align_corners=False)
            rot  = F.grid_sample(xb, grid, mode='bilinear', padding_mode='border', align_corners=False)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outs.append(model(rot))
        return outs

    # (A) Variant × Fold 추론 + votes 수집
    variant_logits = []
    image_ids = None
    votes_all  = []  # (vname, fold_idx, votes_np[N])

    for v_idx, (vname, vtfm) in enumerate(variants):
        test = TestDS(paths['sample_csv'], paths['test_dir'], vtfm)
        loader = DataLoader(test, batch_size=infer_bs, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

        logits_folds = []
        batch_ids = None

        for fold_i, ck in enumerate(ckpts):
            w = torch.load(ck, map_location=device)
            try:
                model.load_state_dict(w['model'], strict=True)
            except Exception:
                model.load_state_dict(w['model'], strict=False)
            model.eval()

            # temperature scaling
            temp_file = os.path.join(os.path.dirname(ck), 'temp.npy')
            T = float(np.load(temp_file)[0]) if os.path.exists(temp_file) else 1.0

            selected = []
            ids_all  = []
            fold_votes = []

            for bi, (xb, ids) in enumerate(loader):
                xb = xb.to(device)
                if int(tta) == 8:
                    cand = [z.detach().cpu() for z in _eight_rot_logits(xb)]
                elif int(tta) == 4:
                    cand = [z.detach().cpu() for z in _four_rot_logits(xb)]
                else:
                    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                        cand = [model(xb).detach().cpu()]

                # max-conf 변환 선택
                probs = [torch.softmax(z, dim=1) for z in cand]
                maxp  = torch.stack([p.max(dim=1).values for p in probs], 0)
                best_idx = torch.argmax(maxp, 0)
                pick = torch.stack([cand[r][i] for i, r in enumerate(best_idx.tolist())], 0)

                # temp 적용
                pick = (pick / T)

                selected.append(pick)
                ids_all.extend(ids)
                fold_votes.append(torch.argmax(pick, 1).numpy())

                if bi % 10 == 0:
                    print(f"[Infer] var={vname} fold={fold_i} batch={bi}/{len(loader)} done", flush=True)

            fold_logits = torch.cat(selected, 0)
            logits_folds.append(fold_logits)

            fold_votes = np.concatenate(fold_votes, 0)
            votes_all.append((vname, fold_i, fold_votes))

            if batch_ids is None:
                batch_ids = ids_all

        var_mean = torch.stack(logits_folds, 0).mean(0)
        variant_logits.append((vname, var_mean))
        if image_ids is None:
            image_ids = batch_ids

    # (B) Variant 결합
    if len(variant_logits) == 1 or combine_mode == 'mean':
        mean_logits = torch.stack([v for _, v in variant_logits], 0).mean(0)
    elif combine_mode in ('max_conf', 'max_confidence'):
        probs_all = [torch.softmax(v, dim=1) for _, v in variant_logits]
        maxp = torch.stack([p.max(1).values for p in probs_all], 0)  # (V,N)
        best_v = torch.argmax(maxp, 0)
        picked = []
        for i in range(best_v.shape[0]):
            vi = int(best_v[i])
            picked.append(variant_logits[vi][1][i].unsqueeze(0))
        mean_logits = torch.cat(picked, 0)
        print(f"[Preproc] Combined by max_conf across {len(variant_logits)} variants")
    else:
        mean_logits = torch.stack([v for _, v in variant_logits], 0).mean(0)

    probs_img = torch.softmax(mean_logits, dim=1).numpy()
    preds     = probs_img.argmax(1)

    # 전역 통계
    eps = 1e-9
    ent = -(probs_img * np.log(probs_img + eps)).sum(1)
    part = np.partition(-probs_img, 1, axis=1)
    top1 = -part[:,0]; top2 = -part[:,1]
    margin = top1 - top2

    # (C) Easy-Lock
    ensemble_cfg  = cfg.get('ensemble', {}) or {}
    easy_classes  = ensemble_cfg.get('easy_classes', [])
    easy_conf     = float(ensemble_cfg.get('easy_conf', 0.92))
    easy_conf_map = ensemble_cfg.get('easy_conf_map', {})
    locked_easy_mask = np.zeros_like(preds, dtype=bool)

    if ensemble_cfg:
        locked_count = 0
        for i in range(len(preds)):
            pred_cls = int(preds[i])
            conf     = float(probs_img[i, pred_cls])
            if pred_cls in easy_classes:
                thr = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf >= thr:
                    locked_easy_mask[i] = True
                    locked_count += 1
        if locked_count > 0:
            print(f"[Easy-Lock] Locked {locked_count}/{len(preds)} easy samples (conf >= thresholds)")

    # (D) Consensus-Lock (fold×variant 표 기반)
    cons_cfg = (ensemble_cfg.get('consensus_lock', {}) if ensemble_cfg else {}) or {}
    locked_cons_mask = np.zeros_like(preds, dtype=bool)
    if cons_cfg.get('enable', False) and votes_all:
        thr       = float(cons_cfg.get('thr', 0.90))
        min_votes = int(cons_cfg.get('min_votes', 6))
        votes_stack = np.stack([v[2] for v in votes_all], 0)  # (S, N)
        S, N = votes_stack.shape

        cons_pred  = np.zeros(N, dtype=int)
        cons_ratio = np.zeros(N, dtype=float)
        for i in range(N):
            col = votes_stack[:, i]
            vals, cnts = np.unique(col, return_counts=True)
            j = cnts.argmax()
            top_cls, top_cnt = int(vals[j]), int(cnts[j])
            cons_pred[i]  = top_cls
            cons_ratio[i] = top_cnt / max(S, 1)

        cons_mask = (~locked_easy_mask) & (cons_ratio >= thr) & (S >= min_votes)
        if np.any(cons_mask):
            preds[cons_mask] = cons_pred[cons_mask]
            locked_cons_mask[cons_mask] = True
        print(f"[Consensus-Lock] Locked {locked_cons_mask.sum()} by consensus (thr≥{thr}, votes={S})")

    # 이후 게이트에서 제외할 통합 락
    locked_mask = locked_easy_mask | locked_cons_mask

    # (E) 후보 한정 5-crop (선택)
    five_cfg = (ensemble_cfg.get('candidate_5crop', {}) if ensemble_cfg else {}) or {}
    if five_cfg.get('enable', False):
        K   = int(five_cfg.get('topk', 200))
        mth = float(five_cfg.get('margin_thr', 0.12))
        gain= float(five_cfg.get('conf_gain', 0.05))

        # 불확실 상위 K (margin 작은 순)
        cand_idx = np.where((~locked_mask) & (margin < mth))[0]
        if len(cand_idx) > K:
            cand_idx = cand_idx[np.argsort(margin[cand_idx])[:K]]

        if len(cand_idx) > 0:
            print(f"[5-crop] Try on {len(cand_idx)} candidates (topK={K}, margin<{mth})")

            # 단일 샘플 단위로 5-crop 수행
            # base_tfm으로 만든 텐서를 crop → 원크기로 리사이즈 후 모델 통과
            test_for_crop = TestDS(paths['sample_csv'], paths['test_dir'], base_tfm)
            improved = 0
            for i in cand_idx:
                # 현재 확률/클래스
                cur_cls = int(preds[i])
                cur_conf = float(probs_img[i, cur_cls])

                # 한 샘플 텐서 로드
                x, _id = test_for_crop[i]  # x: (C,H,W), torch.Tensor
                xb = x.unsqueeze(0).to(device)

                # 5-crop 구현 (center + 4 corners, 비율 0.9)
                B, C, H, W = xb.shape
                side = int(min(H, W) * 0.90)
                offs = [(0,0), (0, W-side), (H-side,0), (H-side, W-side), ((H-side)//2, (W-side)//2)]
                outs = []
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    for oy, ox in offs:
                        crop = xb[:, :, oy:oy+side, ox:ox+side]
                        crop = torch.nn.functional.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
                        outs.append(model(crop))
                # 평균 로짓 → 확률
                logits5 = torch.stack(outs, 0).mean(0)
                probs5  = torch.softmax(logits5, dim=1).squeeze(0).detach().cpu().numpy()
                new_cls = int(np.argmax(probs5))
                new_conf= float(probs5[new_cls])
                # 확신도 이득 검사
                if (new_cls == cur_cls and new_conf >= cur_conf + gain) or (new_cls != cur_cls and new_conf >= cur_conf + gain):
                    preds[i] = new_cls
                    probs_img[i] = probs5
                    improved += 1
            if improved:
                print(f"[5-crop] Replaced {improved} predictions with higher-confidence 5-crop")

            # 전역 통계 재계산 (이후 게이트 위해)
            eps = 1e-9
            ent = -(probs_img * np.log(probs_img + eps)).sum(1)
            part = np.partition(-probs_img, 1, axis=1)
            top1 = -part[:,0]; top2 = -part[:,1]
            margin = top1 - top2

    # (F) Meta/Text/Pair 게이트 — 모두 락 제외 샘플만
    # Meta-Gate
    if META_CLF is not None:
        mg_cfg = cfg.get('meta_gate', {}) or {}
        ENT_THR = float(mg_cfg.get('entropy_thr', 1.4))
        MAR_THR = float(mg_cfg.get('margin_thr', 0.10))

        Xmeta = feats_from_probs_np(probs_img)
        meta_model = META_CLF['model'] if isinstance(META_CLF, dict) and 'model' in META_CLF else META_CLF
        meta_preds = meta_model.predict(Xmeta)

        applied = 0
        for i in range(len(preds)):
            if locked_mask[i]:
                continue
            if (ent[i] > ENT_THR) and (margin[i] < MAR_THR):
                preds[i] = meta_preds[i]
                applied += 1
        if applied:
            print(f"[Meta-Gate] Applied meta to {applied}/{len(preds)} uncertain samples (entropy>{ENT_THR}, margin<{MAR_THR})")

    # Keyword Routing (락 제외; 내부에서도 Easy-Exclude 수행)
    unlocked_test = TestDS(paths['sample_csv'], paths['test_dir'], base_tfm)
    if np.any(~locked_mask):
        preds = apply_keyword_routing(cfg, unlocked_test, probs_img, preds)

    # Text-Gate (확률 휴리스틱; 락 제외)
    text_cfg = cfg.get('text_gate', {}) or {}
    if text_cfg.get('enable', False):
        pairs = text_cfg.get('pairs', [[3,7], [4,14]])
        delta_thr = float(text_cfg.get('delta_thr', 0.08))
        conf_thr  = float(text_cfg.get('conf_thr', 0.35))
        prefer_higher = text_cfg.get('prefer_higher', True)

        # Easy 설정
        easy_classes  = ensemble_cfg.get('easy_classes', [])
        easy_conf     = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})

        text_adjusted = 0
        for (a,b) in pairs:
            pa, pb = probs_img[:, a], probs_img[:, b]
            delta  = np.abs(pa - pb)
            maxc   = np.maximum(pa, pb)
            near   = (delta < delta_thr) & (maxc > conf_thr) & (~locked_mask)
            idx    = np.where(near)[0]
            for i in idx:
                pred_cls = int(preds[i])
                conf     = float(probs_img[i, pred_cls])
                # easy 재확인
                if pred_cls in easy_classes:
                    thr = float(easy_conf_map.get(str(pred_cls), easy_conf))
                    if conf >= thr:
                        continue
                preds[i] = (a if pa[i] >= pb[i] else b) if prefer_higher else a
                text_adjusted += 1
        if text_adjusted:
            print(f"[Text-Gate] Adjusted {text_adjusted} near-boundary samples")

    # Pair-Gate (락 제외 + 추가 게이트)
    pg_cfg = cfg.get('pair_gate', {}) or {}
    if bool(pg_cfg.get('enable', True)):
        dthr = float(pg_cfg.get('delta_thr', 0.05))
        cthr = float(pg_cfg.get('conf_thr', 0.55))
        preds = apply_pair_refiner_with_lock(
            probs_img, preds, 3, 7, PAIR_37, dthr, cthr, ensemble_cfg,
            pair_gate_cfg=pg_cfg, global_ent=ent, global_margin=margin
        )
        preds = apply_pair_refiner_with_lock(
            probs_img, preds, 4, 14, PAIR_414, dthr, cthr, ensemble_cfg,
            pair_gate_cfg=pg_cfg, global_ent=ent, global_margin=margin
        )

    # (선택) 후처리 3↔7 근접치
    pp_cfg = cfg.get('postprocess', {}) or {}
    if pp_cfg.get('enable', False):
        cls_a = int(pp_cfg.get('class_a', 3))
        cls_b = int(pp_cfg.get('class_b', 7))
        delta_thr = float(pp_cfg.get('delta_threshold', 0.04))
        conf_thr  = float(pp_cfg.get('confidence_threshold', 0.55))
        prefer    = str(pp_cfg.get('prefer', 'higher'))
        adjust = 0
        for i in range(len(preds)):
            pa, pb = probs_img[i, cls_a], probs_img[i, cls_b]
            if abs(pa - pb) < delta_thr and max(pa, pb) > conf_thr:
                if prefer == 'higher':
                    preds[i] = cls_a if pa >= pb else cls_b
                else:
                    try: preds[i] = int(prefer)
                    except: preds[i] = cls_a if pa >= pb else cls_b
                adjust += 1
        if adjust:
            print(f"[Postprocess] Adjusted {adjust} near 3↔7")

    # 제출 저장
    sub = pd.read_csv(paths['sample_csv'])
    idc = [c for c in sub.columns if c.lower() in ['id','image_id','filename']]
    ycol= [c for c in sub.columns if c.lower() in ['label','target','class']]
    id_col = idc[0] if idc else sub.columns[0]
    y_col  = ycol[0] if ycol else sub.columns[-1]

    if len(sub) != len(preds):
        raise ValueError(f"sample rows {len(sub)} != preds {len(preds)}")

    sub[y_col] = preds
    sub = sub[[id_col, y_col]]
    out_csv = os.path.join(paths['out_dir'], 'submission.csv')
    os.makedirs(paths['out_dir'], exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # 로짓 저장
    logits_path = os.path.join(paths['out_dir'], 'predict_logits.pt')
    torch.save({'logits': mean_logits.numpy(), 'img_ids': image_ids, 'predictions': preds}, logits_path)
    print(f"Saved logits → {logits_path}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=4, help='TTA mode: 0/1=no TTA, 4=90° rotations, 8=45° rotations')
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
