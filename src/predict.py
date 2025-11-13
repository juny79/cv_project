# predict.py
import os, re, yaml, timm, torch, unicodedata
import numpy as np
import pandas as pd
import joblib
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.transforms import (
    get_valid_transforms,
    get_valid_transform_variants,  # 변형 전처리(variants) 지원
)

# ───────────────────────────────────────────────────────────
# Config / IO 유틸
# ───────────────────────────────────────────────────────────
COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def load_cfg(p):
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
        if os.path.exists(p): return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found for {stem} under {img_dir}")

class TestDS(Dataset):
    def __init__(self, csv, img_dir, transform):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        idc = [c for c in self.df.columns if c.lower() in ['id','image_id','filename']]
        self.id_col = idc[0] if idc else self.df.columns[0]
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        stem = str(self.df.iloc[i][self.id_col])
        p = resolve_image_path(self.img_dir, stem)
        img = np.array(Image.open(p).convert('RGB'))
        x = self.transform(image=img)['image']
        return x, stem

# ───────────────────────────────────────────────────────────
# 텍스트 정규화 & 키워드 라우팅
# ───────────────────────────────────────────────────────────
def _normalize_text(s, opts=None):
    if not s or not isinstance(s, str):
        return ''
    if opts is None: opts = {}
    to_lower = bool(opts.get('to_lower', True))
    remove_spaces = bool(opts.get('remove_spaces', True))
    remove_punct = bool(opts.get('remove_punct', True))
    remove_digits = bool(opts.get('remove_digits', True))
    use_nfkc = bool(opts.get('nfkc', True))
    keep_hyphens = bool(opts.get('keep_hyphens', True))
    collapse_repeats = bool(opts.get('collapse_repeats', True))

    if use_nfkc:
        s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\-‐-‒–—﹣−]', '-', s)
    if to_lower: s = s.lower()
    if remove_spaces: s = re.sub(r'\s+', '', s)
    if remove_punct:
        if keep_hyphens:
            s = re.sub(r'[^0-9a-zA-Z가-힣\-]+', '', s)
        else:
            s = re.sub(r'[^0-9a-zA-Z가-힣]+', '', s)
    if remove_digits: s = re.sub(r'\d+', '', s)
    if collapse_repeats:
        s = re.sub(r'(.)\1{2,}', r'\1', s)
    return s

# 기본 키워드(핵심/확장)
KEYWORDS = {
    3: {
        'core': [r'입퇴원', r'입\s*퇴원', r'입[ \-–—]*퇴원', r'입원', r'퇴원', r'입원퇴원', r'입원\s*퇴원'],
        'wide': [r'입퇴원사실(확인서|증명서|증명원)?', r'입퇴원(확인서|증명서|증명원)',
                 r'입원사실(확인서|증명서|증명원)?', r'입원(확인서|증명서|증명원)', r'퇴원(확인서|증명서|증명원)',
                 r'입원진료확인서', r'입원요약지', r'입원퇴원증명서', r'입원퇴원확인서']
    },
    7: {
        'core': [r'통원', r'외래', r'진료', r'치료', r'통원진료', r'외래진료', r'통원치료', r'통원\s*진료', r'외래\s*진료'],
        'wide': [r'통원(진료확인서|치료사실확인서|치료사실증명서|사실확인서|사실증명서|사실증명원|확인서)?',
                 r'외래진료사실확인서', r'진료사실(확인서|증명서|증명원)', r'통원치료사실확인서',
                 r'진료(입원|통원)?확인서', r'치료확인서', r'확인서']
    },
    4: {
        'core': [r'진단서', r'진단명', r'의사진단서', r'진단내용'],
        'wide': [r'진단(확인서|증명서|증명원)', r'진단소견']
    },
    14: {
        'core': [r'소견서', r'소견', r'의견서', r'소견내용'],
        'wide': [r'의학적소견', r'소견(확인서|증명서|증명원)']
    }
}

def _build_compiled_keywords(tr_cfg):
    merged = {}
    for cid, groups in KEYWORDS.items():
        merged[cid] = {'core': list(groups.get('core', [])),
                       'wide': list(groups.get('wide', []))}
    extra = tr_cfg.get('extra_keywords', {}) or {}
    for k, pats in extra.items():
        try: cid = int(k)
        except: continue
        merged.setdefault(cid, {'core': [], 'wide': []})
        for p in (pats or []):
            if p not in merged[cid]['core']:
                merged[cid]['core'].append(p)
    remove = tr_cfg.get('remove_keywords', {}) or {}
    for k, pats in remove.items():
        try: cid = int(k)
        except: continue
        if cid not in merged: continue
        merged[cid]['core'] = [p for p in merged[cid]['core'] if p not in set(pats or [])]
    kw_file = tr_cfg.get('keyword_file', '')
    if kw_file and os.path.exists(kw_file):
        try:
            df = pd.read_csv(kw_file)
            cc = [c for c in df.columns if str(c).lower() in ['class','cls','label','y']]
            pc = [c for c in df.columns if str(c).lower() in ['pattern','regex','keyword']]
            if cc and pc:
                ccol, pcol = cc[0], pc[0]
                for _, r in df.iterrows():
                    try: cid = int(r[ccol])
                    except: continue
                    pat = str(r[pcol]) if pd.notna(r[pcol]) else ''
                    if not pat: continue
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

def _load_ocr_map(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    idc = [c for c in df.columns if c.lower() in ['id','image_id','filename']]
    txtc = [c for c in df.columns if c.lower() in ['text','ocr','ocr_text']]
    if not idc or not txtc:
        return None
    id_col = idc[0]; txt_col = txtc[0]
    ocr_map = {}
    for _, r in df.iterrows():
        k = str(r[id_col]); v = r[txt_col] if pd.notna(r[txt_col]) else ''
        ocr_map[k] = v
    return ocr_map

def apply_keyword_routing(cfg, test_ds, probs_img, preds):
    tr_cfg = cfg.get('text_routing', {}) or {}
    if not tr_cfg.get('enable', False):
        return preds

    source = tr_cfg.get('source', '')
    pairs = tr_cfg.get('pairs', [[3,7],[4,14]])

    # 임계치(2단계 구조)
    delta_thr_strict = float(tr_cfg.get('delta_thr_strict', tr_cfg.get('delta_thr', 0.06)))
    conf_thr_strict  = float(tr_cfg.get('conf_thr_strict',  0.55))
    min_hits_strict  = int(tr_cfg.get('min_hits_strict',    2))

    delta_thr_wide   = float(tr_cfg.get('delta_thr_wide',   0.15))
    conf_thr_wide    = float(tr_cfg.get('conf_thr_wide',    0.60))
    min_hits_wide    = int(tr_cfg.get('min_hits_wide',      3))
    entropy_thr_wide = float(tr_cfg.get('entropy_thr_wide', 1.55))
    margin_thr_wide  = float(tr_cfg.get('margin_thr_wide',  0.12))

    prefer_top = bool(tr_cfg.get('prefer_top', True))
    easy_exclude = bool(tr_cfg.get('easy_exclude', True))

    # OCR 로드
    ocr_map = _load_ocr_map(source)
    if ocr_map is None:
        print(f"[KeywordRouting] OCR source not found or invalid: {source}, skipping")
        return preds

    # Easy-Lock 설정
    ensemble_cfg = cfg.get('ensemble', {})
    easy_classes = ensemble_cfg.get('easy_classes', [])
    easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
    easy_conf_map = ensemble_cfg.get('easy_conf_map', {})

    compiled_kw = _build_compiled_keywords(tr_cfg)
    norm_opts = tr_cfg.get('normalization', {}) or {}

    dbg_cfg = tr_cfg.get('debug', {}) or {}
    dbg_enable = bool(dbg_cfg.get('enable', False))
    dbg_rows = []

    new_preds = preds.copy()
    eps = 1e-9
    ent = -(probs_img * np.log(probs_img + eps)).sum(1)
    part = np.partition(-probs_img, 1, axis=1)
    top1 = -part[:,0]; top2 = -part[:,1]
    margin = top1 - top2

    total_near_strict = 0
    total_near_wide = 0
    total_locked = 0
    total_eligible = 0
    hit_route = 0
    tie_route = 0
    stage_counts = {'strict': 0, 'wide': 0}

    for (a, b) in pairs:
        pa, pb = probs_img[:, a], probs_img[:, b]
        delta = np.abs(pa - pb)
        max_conf = np.maximum(pa, pb)

        # 1단계: strict
        mask_strict = (delta < delta_thr_strict) & (max_conf > conf_thr_strict)
        idx_strict = np.where(mask_strict)[0]
        total_near_strict += len(idx_strict)

        for i in idx_strict:
            pred_cls = preds[i]
            conf_i = probs_img[i, pred_cls]
            # Easy-Lock 제외
            if easy_exclude and (pred_cls in easy_classes):
                threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf_i >= threshold:
                    total_locked += 1
                    continue
            total_eligible += 1

            img_id = str(test_ds.df.iloc[i][test_ds.id_col])
            ocr_text = ocr_map.get(img_id, '')

            core_a, wide_a = _count_hits_groups(ocr_text, a, compiled_kw=compiled_kw, norm_opts=norm_opts)
            core_b, wide_b = _count_hits_groups(ocr_text, b, compiled_kw=compiled_kw, norm_opts=norm_opts)

            eff_a = core_a if core_a >= min_hits_strict else 0
            eff_b = core_b if core_b >= min_hits_strict else 0

            if max(eff_a, eff_b) > 0:
                reason = ''
                if eff_a > eff_b:
                    new_preds[i] = a; hit_route += 1; stage_counts['strict'] += 1; reason = 'strict_hits_a_core'
                elif eff_b > eff_a:
                    new_preds[i] = b; hit_route += 1; stage_counts['strict'] += 1; reason = 'strict_hits_b_core'
                elif prefer_top:
                    new_preds[i] = a if pa[i] >= pb[i] else b
                    tie_route += 1; stage_counts['strict'] += 1; reason = 'strict_tie_prefer_top'
                if dbg_enable and reason:
                    dbg_rows.append({
                        'id': img_id, 'pair': f'{a}-{b}', 'stage': 'strict',
                        'prev_pred': int(pred_cls), 'new_pred': int(new_preds[i]),
                        'pa': float(pa[i]), 'pb': float(pb[i]),
                        'delta': float(abs(pa[i]-pb[i])),
                        'entropy': float(ent[i]), 'margin': float(margin[i]),
                        'core_a': int(core_a), 'wide_a': int(wide_a),
                        'core_b': int(core_b), 'wide_b': int(wide_b),
                        'min_hits_strict': int(min_hits_strict), 'reason': reason
                    })

        # 2단계: wide (추가 게이팅)
        mask_wide = (delta < delta_thr_wide) & (delta >= delta_thr_strict) & (max_conf > conf_thr_wide)
        mask_wide = mask_wide & (ent > entropy_thr_wide) & (margin < margin_thr_wide)
        idx_wide = np.where(mask_wide)[0]
        total_near_wide += len(idx_wide)

        for i in idx_wide:
            pred_cls = preds[i]
            conf_i = probs_img[i, pred_cls]
            if easy_exclude and (pred_cls in easy_classes):
                threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf_i >= threshold:
                    total_locked += 1
                    continue
            total_eligible += 1

            img_id = str(test_ds.df.iloc[i][test_ds.id_col])
            ocr_text = ocr_map.get(img_id, '')

            core_a, wide_a = _count_hits_groups(ocr_text, a, compiled_kw=compiled_kw, norm_opts=norm_opts)
            core_b, wide_b = _count_hits_groups(ocr_text, b, compiled_kw=compiled_kw, norm_opts=norm_opts)

            total_a = core_a + wide_a
            total_b = core_b + wide_b
            eff_a = core_a if core_a >= min_hits_strict else (total_a if total_a >= min_hits_wide else 0)
            eff_b = core_b if core_b >= min_hits_strict else (total_b if total_b >= min_hits_wide else 0)

            if max(eff_a, eff_b) > 0:
                reason = ''
                if eff_a > eff_b:
                    new_preds[i] = a; hit_route += 1; stage_counts['wide'] += 1; reason = 'wide_hits_a'
                elif eff_b > eff_a:
                    new_preds[i] = b; hit_route += 1; stage_counts['wide'] += 1; reason = 'wide_hits_b'
                elif prefer_top:
                    new_preds[i] = a if pa[i] >= pb[i] else b
                    tie_route += 1; stage_counts['wide'] += 1; reason = 'wide_tie_prefer_top'
                if dbg_enable and reason:
                    dbg_rows.append({
                        'id': img_id, 'pair': f'{a}-{b}', 'stage': 'wide',
                        'prev_pred': int(pred_cls), 'new_pred': int(new_preds[i]),
                        'pa': float(pa[i]), 'pb': float(pb[i]),
                        'delta': float(abs(pa[i]-pb[i])),
                        'entropy': float(ent[i]), 'margin': float(margin[i]),
                        'core_a': int(core_a), 'wide_a': int(wide_a),
                        'core_b': int(core_b), 'wide_b': int(wide_b),
                        'min_hits_strict': int(min_hits_strict), 'min_hits_wide': int(min_hits_wide),
                        'reason': reason
                    })

    print(f"[KeywordRouting] Candidates strict={total_near_strict}, wide={total_near_wide}, locked: {total_locked}, eligible: {total_eligible}")
    print(f"[KeywordRouting] Routed (hit/tie): {hit_route}/{tie_route} | per-stage: {stage_counts} | pairs={pairs}")
    print(f"[KeywordRouting] thr: strict(delta<{delta_thr_strict}, conf>{conf_thr_strict}, min_core>={min_hits_strict}) | wide(delta<{delta_thr_wide}, conf>{conf_thr_wide}, ent>{entropy_thr_wide}, mar<{margin_thr_wide})")

    if dbg_enable and dbg_rows:
        try:
            out_csv = dbg_cfg.get('csv_path') or os.path.join(cfg.get('paths', {}).get('out_dir', './outputs'),
                                                              'keyword_routing_debug.csv')
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            import csv
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(dbg_rows[0].keys()))
                w.writeheader(); w.writerows(dbg_rows)
            print(f"[KeywordRouting] Debug CSV saved → {out_csv}")
        except Exception as e:
            print(f"[KeywordRouting] Failed to save debug CSV: {e}")
    return new_preds

# ───────────────────────────────────────────────────────────
# 메타/페어 게이트 보조 유틸
# ───────────────────────────────────────────────────────────
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

    # Easy-Lock: 확신 높은 쉬운 클래스는 보정 제외
    if easy_cfg:
        easy_classes = easy_cfg.get('easy_classes', [])
        easy_conf = float(easy_cfg.get('easy_conf', 0.92))
        easy_conf_map = easy_cfg.get('easy_conf_map', {})
        unlocked_idx = []
        for i in idx:
            pred_cls = preds[i]
            conf = P[i, pred_cls]
            if pred_cls in easy_classes:
                threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf >= threshold:
                    continue
            unlocked_idx.append(i)
        idx = np.array(unlocked_idx)
    if len(idx) == 0:
        return new_preds

    ent = -(P[idx]*np.log(P[idx]+1e-9)).sum(1, keepdims=True)
    X = np.concatenate([pa[idx][:,None], pb[idx][:,None], ent, (pb[idx]-pa[idx])[:,None]], 1)
    z = clf_pair.predict(X)
    new_preds[idx] = np.where(z==1, b, a)
    return new_preds

# ───────────────────────────────────────────────────────────
# 메인 추론 파이프라인
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def predict_ensemble(config_path, tta=4):
    cfg = load_cfg(config_path)
    paths, data = cfg['paths'], cfg['data']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = int(data['img_size'])
    base_tfm = get_valid_transforms(img_size)

    # 전처리 variant 구성
    inf_cfg = cfg.get('inference', {}) or {}
    pp_cfg = (inf_cfg.get('preproc_variants', {}) or {})
    use_variants = bool(pp_cfg.get('enable', False)) and len(pp_cfg.get('variants', [])) > 0
    # 결합 모드 결정 (config → env override → 표준화)
    combine_mode_raw = pp_cfg.get('combine') or pp_cfg.get('mode') or pp_cfg.get('strategy')
    if isinstance(combine_mode_raw, str):
        combine_mode = combine_mode_raw.strip().lower()
    else:
        combine_mode = 'mean'
    # 환경 변수 강제 오버라이드 (디버깅/신속 실험용)
    env_override = os.environ.get('PREPROC_COMBINE', '').strip().lower()
    if env_override:
        combine_mode = env_override
    # 동의어 및 검증
    valid_modes = {'mean','avg','average','max_conf','max_confidence'}
    if combine_mode in {'avg','average'}:
        combine_mode = 'mean'
    if combine_mode not in valid_modes:
        print(f"[Preproc] Warning: unknown combine='{combine_mode_raw}', fallback 'mean'")
        combine_mode = 'mean'
    if use_variants:
        variants = get_valid_transform_variants(img_size, pp_cfg.get('variants', []))
        print(f"[Preproc] Using {len(variants)} variants, combine={combine_mode}")
    else:
        variants = [("base", base_tfm)]

    num_workers = 0
    pin_memory = device.startswith('cuda')
    infer_bs = int(inf_cfg.get('batch_size', int(cfg['train']['batch_size'])))

    # 메타/페어 모델 로드(선택적)
    project_root = os.path.abspath(os.path.join(paths['base_dir'], '..'))
    meta_path = os.path.join(project_root, 'extern', 'meta_full.joblib')
    pair37_path = os.path.join(project_root, 'extern', 'pair_3_7.joblib')
    pair414_path = os.path.join(project_root, 'extern', 'pair_4_14.joblib')
    META_CLF = joblib.load(meta_path) if os.path.exists(meta_path) else None
    PAIR_37  = joblib.load(pair37_path) if os.path.exists(pair37_path) else None
    PAIR_414 = joblib.load(pair414_path) if os.path.exists(pair414_path) else None

    # 체크포인트 모으기
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
    for k,v in state0['model'].items():
        if ('head.fc.weight' in k or 'classifier.weight' in k or 'fc.weight' in k) and hasattr(v, 'shape'):
            head_weight = v; break
    num_classes = head_weight.shape[0] if head_weight is not None else int(cfg.get('data', {}).get('num_classes', 17))

    # 기본 모델
    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    use_amp = bool(cfg.get('inference', {}).get('amp', True)) and device.startswith('cuda')

    def _four_rot_logits(xb):
        outs = []
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))
        return outs

    def _eight_rot_logits(xb):
        outs = []
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))
        B, C, H, W = xb.shape
        for angle in [45, 135, 225, 315]:
            theta_rad = torch.tensor(angle * 3.14159265 / 180.0, device=xb.device)
            cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
            theta = torch.tensor([[cos_t, -sin_t, 0],[sin_t, cos_t, 0]], dtype=xb.dtype, device=xb.device)
            theta = theta.unsqueeze(0).repeat(B,1,1)
            grid = F.affine_grid(theta, size=xb.size(), align_corners=False)
            rotated = F.grid_sample(xb, grid, mode='bilinear', padding_mode='border', align_corners=False)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outs.append(model(rotated))
        return outs

    # Variant × Fold 추론
    variant_logits = []
    image_ids = None
    for vname, vtfm in variants:
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

            # Temperature scaling (fold별)
            temp_file = os.path.join(os.path.dirname(ck), 'temp.npy')
            T = float(np.load(temp_file)[0]) if os.path.exists(temp_file) else 1.0

            selected = []
            ids_all = []
            for bi, (xb, ids) in enumerate(loader):
                xb = xb.to(device)
                if int(tta) == 8:
                    cand = [z.detach().cpu() for z in _eight_rot_logits(xb)]
                elif int(tta) == 4:
                    cand = [z.detach().cpu() for z in _four_rot_logits(xb)]
                else:
                    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                        cand = [model(xb).detach().cpu()]

                probs = [torch.softmax(z, dim=1) for z in cand]
                maxp = torch.stack([p.max(dim=1).values for p in probs], 0)
                best_idx = torch.argmax(maxp, 0)
                pick = torch.stack([cand[r][i] for i, r in enumerate(best_idx.tolist())], 0)

                pick = (pick / T)
                selected.append(pick)
                ids_all.extend(ids)

                if bi % 10 == 0:
                    print(f"[Infer] var={vname} fold={fold_i} batch={bi}/{len(loader)} done", flush=True)

            fold_logits = torch.cat(selected, 0)
            logits_folds.append(fold_logits)
            if batch_ids is None:
                batch_ids = ids_all

        var_mean = torch.stack(logits_folds, 0).mean(0)
        variant_logits.append((vname, var_mean))
        if image_ids is None:
            image_ids = batch_ids

    # Variant 결합
    if len(variant_logits) == 1:
        mean_logits = variant_logits[0][1]
    else:
        combine = combine_mode
        if combine in ('max_conf', 'max_confidence'):
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
    preds = probs_img.argmax(1)

    # Easy-Lock: 쉬운 클래스 고확신 Lock (다른 게이트에서 변경 못하게)
    ensemble_cfg = cfg.get('ensemble', {})
    if ensemble_cfg:
        easy_classes = ensemble_cfg.get('easy_classes', [])
        easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})
        locked_count = 0
        for i in range(len(preds)):
            pc = preds[i]; conf = probs_img[i, pc]
            if pc in easy_classes:
                thr = float(easy_conf_map.get(str(pc), easy_conf))
                if conf >= thr:
                    locked_count += 1
        if locked_count > 0:
            print(f"[Easy-Lock] Locked {locked_count}/{len(preds)} easy samples (conf >= thresholds)")

    # 메타 게이트(불확실 샘플만)
    if META_CLF is not None:
        meta_model = META_CLF['model'] if isinstance(META_CLF, dict) and 'model' in META_CLF else META_CLF
        Xmeta = feats_from_probs_np(probs_img)
        meta_preds = meta_model.predict(Xmeta)

        eps = 1e-9
        ent = -(probs_img * np.log(probs_img + eps)).sum(1)
        part = np.partition(-probs_img, 1, axis=1)
        top1 = -part[:,0]; top2 = -part[:,1]
        margin = top1 - top2

        mg_cfg = cfg.get('meta_gate', {}) or {}
        ENT_THR = float(mg_cfg.get('entropy_thr', 1.4))
        MAR_THR = float(mg_cfg.get('margin_thr', 0.10))

        easy_classes = ensemble_cfg.get('easy_classes', [])
        easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})

        applied = 0
        for i in range(len(preds)):
            pc = preds[i]; conf = probs_img[i, pc]
            locked = False
            if pc in easy_classes:
                thr = float(easy_conf_map.get(str(pc), easy_conf))
                if conf >= thr:
                    locked = True
            if (not locked) and (ent[i] > ENT_THR) and (margin[i] < MAR_THR):
                preds[i] = meta_preds[i]
                applied += 1
        if applied:
            print(f"[Meta-Gate] Applied meta to {applied}/{len(preds)} uncertain samples (entropy>{ENT_THR}, margin<{MAR_THR})")

    # 키워드 라우팅(OCR 기반) — 3↔7, 4↔14
    preds = apply_keyword_routing(cfg, test, probs_img, preds)

    # Text-Gate(초근접 단순 보조) — OCR 매칭 실패 시 근소 우세로 밀어줌
    text_cfg = cfg.get('text_gate', {}) or {}
    if text_cfg.get('enable', False):
        pairs = text_cfg.get('pairs', [[3,7],[4,14]])
        delta_thr = float(text_cfg.get('delta_thr', 0.08))
        conf_thr = float(text_cfg.get('conf_thr', 0.35))
        prefer_higher = bool(text_cfg.get('prefer_higher', True))

        easy_classes = ensemble_cfg.get('easy_classes', [])
        easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})

        text_adjusted = 0
        for (a, b) in pairs:
            pa, pb = probs_img[:, a], probs_img[:, b]
            delta = np.abs(pa - pb)
            max_conf = np.maximum(pa, pb)
            near_idx = np.where((delta < delta_thr) & (max_conf > conf_thr))[0]
            for i in near_idx:
                pc = preds[i]; conf = probs_img[i, pc]
                locked = False
                if pc in easy_classes:
                    thr = float(easy_conf_map.get(str(pc), easy_conf))
                    if conf >= thr: locked = True
                if locked: continue
                preds[i] = a if (prefer_higher and pa[i] >= pb[i]) else (b if prefer_higher else a)
                text_adjusted += 1
        if text_adjusted > 0:
            print(f"[Text-Gate] Adjusted {text_adjusted} near-boundary samples (delta<{delta_thr}) for pairs {pairs}")

    # 페어 리파이너(학습된 이진 보정기, Easy-Lock 존중)
    pg_cfg = cfg.get('pair_gate', {}) or {}
    use_pg = bool(pg_cfg.get('enable', True))
    eps = 1e-9
    part_pg = np.partition(-probs_img, 1, axis=1)
    top1_pg = -part_pg[:,0]; top2_pg = -part_pg[:,1]
    margin_pg = top1_pg - top2_pg
    ent_pg = -(probs_img * np.log(probs_img + eps)).sum(1)

    dthr = float(pg_cfg.get('delta_thr', 0.05))
    cthr = float(pg_cfg.get('conf_thr', 0.55))
    preds = apply_pair_refiner_with_lock(
        probs_img, preds, 3, 7, PAIR_37, dthr, cthr, ensemble_cfg,
        pair_gate_cfg=(pg_cfg if use_pg else None), global_ent=ent_pg, global_margin=margin_pg
    )
    preds = apply_pair_refiner_with_lock(
        probs_img, preds, 4, 14, PAIR_414, dthr, cthr, ensemble_cfg,
        pair_gate_cfg=(pg_cfg if use_pg else None), global_ent=ent_pg, global_margin=margin_pg
    )

    # (옵션) 간단 사후보정
    pp_cfg = cfg.get('postprocess', {}) or {}
    if pp_cfg.get('enable', False):
        probs = torch.softmax(mean_logits, dim=1).cpu().numpy()
        cls_a = int(pp_cfg.get('class_a', 3))
        cls_b = int(pp_cfg.get('class_b', 7))
        delta_thr = float(pp_cfg.get('delta_threshold', 0.04))
        conf_thr = float(pp_cfg.get('confidence_threshold', 0.55))
        prefer = str(pp_cfg.get('prefer', 'higher'))
        adjust_count = 0
        for i in range(len(preds)):
            pa, pb = probs[i, cls_a], probs[i, cls_b]
            if abs(pa - pb) < delta_thr and max(pa, pb) > conf_thr:
                if prefer == 'higher':
                    preds[i] = cls_a if pa >= pb else cls_b
                else:
                    try: preds[i] = int(prefer)
                    except: preds[i] = cls_a if pa >= pb else cls_b
                adjust_count += 1
        print(f"[Postprocess] Adjusted {adjust_count} predictions under delta<{delta_thr} and conf>{conf_thr}")

        # Advanced flip logic (risk-focused second-best flip for target classes)
        adv = pp_cfg.get('advanced_flip', {}) or {}
        if adv.get('enable', False):
            tgt = set(int(c) for c in adv.get('target_classes', [cls_a, cls_b]))
            m_thr = float(adv.get('margin_thr', 0.05))
            ent_thr = float(adv.get('entropy_thr', 0.35))
            top1_max = float(adv.get('top1_conf_max', 0.94))
            second_gain_min = float(adv.get('second_gain_min', 0.015))

            eps = 1e-9
            ent = -(probs * np.log(probs + eps)).sum(1)
            # top1 / second best probability
            order = np.argsort(-probs, axis=1)
            top1_idx = order[:,0]
            top2_idx = order[:,1]
            top1_conf = probs[np.arange(len(probs)), top1_idx]
            top2_conf = probs[np.arange(len(probs)), top2_idx]
            margin = top1_conf - top2_conf

            flip_count = 0
            for i in range(len(preds)):
                if preds[i] not in tgt:
                    continue
                # only attempt flip if current pred is in target set
                if top1_idx[i] != preds[i]:
                    continue  # consistency check
                if margin[i] > m_thr:  # margin already wide
                    continue
                if ent[i] < ent_thr:  # not uncertain enough
                    continue
                if top1_conf[i] > top1_max:  # too confident
                    continue
                # ensure second best is also within target scope OR part of pair
                if top2_idx[i] not in tgt:
                    continue
                # require closeness threshold
                if (top1_conf[i] - top2_conf[i]) > second_gain_min:
                    continue
                # flip to second best
                prev = preds[i]
                preds[i] = int(top2_idx[i])
                flip_count += 1
            if flip_count:
                print(f"[Postprocess-Advanced] Flipped {flip_count} low-margin uncertain samples (m<{m_thr}, ent>{ent_thr})")

    # 제출 저장
    sub = pd.read_csv(paths['sample_csv'])
    idc = [c for c in sub.columns if c.lower() in ['id','image_id','filename']]
    ycol = [c for c in sub.columns if c.lower() in ['label','target','class']]
    id_col = idc[0] if idc else sub.columns[0]
    y_col = ycol[0] if ycol else sub.columns[-1]

    if len(sub) != len(preds):
        raise ValueError(f"sample rows {len(sub)} != preds {len(preds)}")

    sub[y_col] = preds
    sub = sub[[id_col, y_col]]
    out_csv = os.path.join(paths['out_dir'], 'submission.csv')
    os.makedirs(paths['out_dir'], exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # 로짓/ID 저장(분석용)
    logits_path = os.path.join(paths['out_dir'], 'predict_logits.pt')
    torch.save({'logits': mean_logits.numpy(), 'img_ids': image_ids, 'predictions': preds}, logits_path)
    print(f"Saved logits → {logits_path}")

# ───────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=4, help='TTA mode: 0/1=no TTA, 4=90° rotations, 8=45° rotations')
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
