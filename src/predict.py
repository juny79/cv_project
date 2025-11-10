import os, yaml, timm, torch, re, unicodedata
import numpy as np, pandas as pd
import joblib
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.transforms import get_valid_transforms, get_valid_transform_variants

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def load_cfg(p):
    try:
        # allow bare config name by auto-prepending configs/
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

def _normalize_text(s, opts=None):
    """Normalize text for keyword matching.
    opts: {
      to_lower: bool, remove_spaces: bool, remove_punct: bool,
      remove_digits: bool, nfkc: bool
    }
    """
    if not s or not isinstance(s, str):
        return ''
    if opts is None:
        opts = {}
    # Defaults tuned for noisy OCR
    to_lower = bool(opts.get('to_lower', True))
    remove_spaces = bool(opts.get('remove_spaces', True))
    remove_punct = bool(opts.get('remove_punct', True))
    remove_digits = bool(opts.get('remove_digits', True))
    use_nfkc = bool(opts.get('nfkc', True))

    if use_nfkc:
        s = unicodedata.normalize('NFKC', s)
    if to_lower:
        s = s.lower()
    if remove_spaces:
        s = re.sub(r'\s+', '', s)
    if remove_punct:
        # Remove common punctuation and symbols while preserving Korean/English letters
        s = re.sub(r'[^0-9a-zA-Z가-힣]+', '', s)
    if remove_digits:
        s = re.sub(r'\d+', '', s)
    return s

# Keyword patterns for OCR-based routing
KEYWORDS = {
    3: [r'입퇴원', r'입원', r'퇴원', r'입퇴원확인서', r'입원확인서', r'퇴원확인서'],
    7: [r'외래', r'통원', r'외래진료', r'통원치료', r'외래확인서'],
    4: [r'진단서', r'진단명', r'의사진단서', r'진단내용'],
    14: [r'소견서', r'소견', r'의견서', r'소견내용']
}

def _build_compiled_keywords(tr_cfg):
    """Merge base KEYWORDS with config extras/removals and compile regex."""
    # Start with a copy of base keywords
    merged = {k: list(v) for k, v in KEYWORDS.items()}
    # Extra keywords from config
    extra = tr_cfg.get('extra_keywords', {}) or {}
    for k, pats in extra.items():
        try:
            cid = int(k)
        except Exception:
            cid = int(k) if isinstance(k, int) else None
        if cid is None:
            continue
        merged.setdefault(cid, [])
        for p in (pats or []):
            if p not in merged[cid]:
                merged[cid].append(p)
    # Remove keywords
    remove = tr_cfg.get('remove_keywords', {}) or {}
    for k, pats in remove.items():
        try:
            cid = int(k)
        except Exception:
            cid = int(k) if isinstance(k, int) else None
        if cid is None or cid not in merged:
            continue
        merged[cid] = [p for p in merged[cid] if p not in set(pats or [])]
    # Optional: load from CSV (class,pattern)
    kw_file = tr_cfg.get('keyword_file', '')
    if kw_file and os.path.exists(kw_file):
        try:
            df = pd.read_csv(kw_file)
            # Expect columns: class, pattern
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
                    merged.setdefault(cid, [])
                    if pat not in merged[cid]:
                        merged[cid].append(pat)
        except Exception:
            pass
    # Compile
    compiled = {cid: [re.compile(p) for p in pats] for cid, pats in merged.items()}
    return compiled

def _count_hits(text, cls_id, compiled_kw=None, norm_opts=None):
    """Count keyword matches for a class using compiled regex if provided."""
    if compiled_kw is None:
        pats = KEYWORDS.get(cls_id, [])
        compiled = [re.compile(p) for p in pats]
    else:
        compiled = compiled_kw.get(cls_id, [])
    norm = _normalize_text(text, norm_opts)
    count = 0
    for rgx in compiled:
        if rgx.search(norm):
            count += 1
    return count

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
    id_col_actual = idc[0]
    txt_col = txtc[0]
    ocr_map = {}
    for _, row in df.iterrows():
        img_id = str(row[id_col_actual])
        text = row[txt_col] if pd.notna(row[txt_col]) else ''
        ocr_map[img_id] = text
    return ocr_map

def apply_keyword_routing(cfg, test_ds, probs_img, preds):
    """Apply OCR keyword routing for near-boundary pairs (3,7) and (4,14)."""
    tr_cfg = cfg.get('text_routing', {}) or {}
    if not tr_cfg.get('enable', False):
        return preds
    
    source = tr_cfg.get('source', '')
    pairs = tr_cfg.get('pairs', [[3,7], [4,14]])
    delta_thr = float(tr_cfg.get('delta_thr', 0.04))
    conf_thr = float(tr_cfg.get('conf_thr', 0.55))
    min_hits = int(tr_cfg.get('min_hits', 2))
    prefer_top = bool(tr_cfg.get('prefer_top', True))
    
    # Load OCR map
    ocr_map = _load_ocr_map(source)
    if ocr_map is None:
        print(f"[KeywordRouting] OCR source not found or invalid: {source}, skipping")
        return preds
    
    # Easy-Lock config
    ensemble_cfg = cfg.get('ensemble', {})
    easy_classes = ensemble_cfg.get('easy_classes', [])
    easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
    easy_conf_map = ensemble_cfg.get('easy_conf_map', {})
    
    # Build compiled keyword patterns and normalization options
    compiled_kw = _build_compiled_keywords(tr_cfg)
    norm_opts = tr_cfg.get('normalization', {}) or {}

    # Debug logging setup
    dbg_cfg = tr_cfg.get('debug', {}) or {}
    dbg_enable = bool(dbg_cfg.get('enable', False))
    dbg_rows = []

    routed = 0
    new_preds = preds.copy()
    
    total_near = 0
    total_locked = 0
    total_eligible = 0
    hit_route = 0
    tie_route = 0

    for (a, b) in pairs:
        pa = probs_img[:, a]
        pb = probs_img[:, b]
        delta = np.abs(pa - pb)
        max_conf = np.maximum(pa, pb)
        
        # Near-boundary mask
        near_mask = (delta < delta_thr) & (max_conf > conf_thr)
        near_idx = np.where(near_mask)[0]
        total_near += len(near_idx)
        
        for i in near_idx:
            pred_cls = preds[i]
            conf = probs_img[i, pred_cls]
            
            # Easy-Lock check
            locked = False
            if pred_cls in easy_classes:
                threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf >= threshold:
                    locked = True
            
            if locked:
                total_locked += 1
                continue
            
            total_eligible += 1
            # Get OCR text for this image
            img_id = str(test_ds.df.iloc[i][test_ds.id_col])
            ocr_text = ocr_map.get(img_id, '')
            
            # Count keyword hits for both classes
            hits_a = _count_hits(ocr_text, a, compiled_kw=compiled_kw, norm_opts=norm_opts)
            hits_b = _count_hits(ocr_text, b, compiled_kw=compiled_kw, norm_opts=norm_opts)
            
            # Route if sufficient hits
            if max(hits_a, hits_b) >= min_hits:
                reason = ''
                if hits_a > hits_b:
                    new_preds[i] = a
                    routed += 1
                    hit_route += 1
                    reason = 'hits_a'
                elif hits_b > hits_a:
                    new_preds[i] = b
                    routed += 1
                    hit_route += 1
                    reason = 'hits_b'
                elif prefer_top:
                    # Tie: prefer higher probability
                    new_preds[i] = a if pa[i] >= pb[i] else b
                    routed += 1
                    tie_route += 1
                    reason = 'tie_prefer_top'
                if dbg_enable and reason:
                    dbg_rows.append({
                        'id': img_id,
                        'pair': f'{a}-{b}',
                        'prev_pred': int(pred_cls),
                        'new_pred': int(new_preds[i]),
                        'pa': float(pa[i]), 'pb': float(pb[i]),
                        'delta': float(abs(pa[i]-pb[i])),
                        'hits_a': int(hits_a), 'hits_b': int(hits_b),
                        'reason': reason
                    })
    
    # One-line summaries (always print)
    print(f"[KeywordRouting] Near-boundary candidates: {total_near}, locked: {total_locked}, eligible: {total_eligible}")
    print(f"[KeywordRouting] Routed: {routed} (by hits: {hit_route}, by tie→top: {tie_route}), pairs={pairs}, thr(delta<{delta_thr}, conf>{conf_thr}, min_hits>={min_hits})")
    # Save debug CSV if enabled
    if dbg_enable and dbg_rows:
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

@torch.no_grad()
def predict_ensemble(config_path, tta=4):
    cfg = load_cfg(config_path)
    paths, data = cfg['paths'], cfg['data']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = int(data['img_size'])
    base_tfm = get_valid_transforms(img_size)

    # Optional: multiple deterministic preprocessing variants
    inf_cfg = cfg.get('inference', {}) or {}
    pp_cfg = (inf_cfg.get('preproc_variants', {}) or {})
    use_variants = bool(pp_cfg.get('enable', False)) and len(pp_cfg.get('variants', [])) > 0
    combine_mode = str(pp_cfg.get('combine', 'mean')).lower()
    if use_variants:
        variants = get_valid_transform_variants(img_size, pp_cfg.get('variants', []))
        print(f"[Preproc] Using {len(variants)} variants, combine={combine_mode}")
    else:
        variants = [("base", base_tfm)]

    num_workers = 0  # dataloader 안정화
    pin_memory = True if device.startswith('cuda') else False
    infer_bs = int(inf_cfg.get('batch_size', int(cfg['train']['batch_size'])))

    # 메타/페어 로더 (선택적)
    project_root = os.path.abspath(os.path.join(paths['base_dir'], '..'))
    meta_path = os.path.join(project_root, 'extern', 'meta_full.joblib')
    pair37_path = os.path.join(project_root, 'extern', 'pair_3_7.joblib')
    pair414_path = os.path.join(project_root, 'extern', 'pair_4_14.joblib')
    META_CLF = joblib.load(meta_path) if os.path.exists(meta_path) else None
    PAIR_37  = joblib.load(pair37_path) if os.path.exists(pair37_path) else None
    PAIR_414 = joblib.load(pair414_path) if os.path.exists(pair414_path) else None

    def feats_from_probs_np(p):
        eps = 1e-9
        top2 = np.sort(p, axis=1)[:, -2:]
        margin = top2[:,1] - top2[:,0]
        ent = -(p*np.log(p+eps)).sum(1, keepdims=True)
        return np.concatenate([p, ent, margin[:,None]], 1)

    def apply_pair_refiner(P, preds, a, b, clf_pair, delta_thr=0.06, conf_thr=0.55):
        if clf_pair is None:
            return preds
        new_preds = preds.copy()
        pa, pb = P[:, a], P[:, b]
        mask = (np.abs(pa - pb) < delta_thr) & (np.maximum(pa, pb) > conf_thr)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return new_preds
        ent = -(P[idx]*np.log(P[idx]+1e-9)).sum(1, keepdims=True)
        X = np.concatenate([pa[idx][:,None], pb[idx][:,None], ent, (pb[idx]-pa[idx])[:,None]], 1)
        z = clf_pair.predict(X)
        new_preds[idx] = np.where(z==1, b, a)
        return new_preds

    # 체크포인트 발견
    ckpts, folds = [], data.get('folds', [0,1,2,3,4])
    for k in folds:
        p = os.path.join(paths['out_dir'], f'fold{k}', 'best.pt')
        if os.path.exists(p):
            ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {paths['out_dir']}/fold*/best.pt")

    # num_classes 추정
    state0 = torch.load(ckpts[0], map_location='cpu')
    head_weight = None
    for k,v in state0['model'].items():
        if ('head.fc.weight' in k or 'classifier.weight' in k or 'fc.weight' in k) and hasattr(v, 'shape'):
            head_weight = v; break
    if head_weight is None:
        # fallback: use data hint if provided
        num_classes = int(cfg.get('data', {}).get('num_classes', 17))
    else:
        num_classes = head_weight.shape[0]

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    use_amp = bool(cfg.get('inference', {}).get('amp', True)) and device.startswith('cuda')

    def _four_rot_logits(xb):
        outs = []
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))                                     # 0
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))         # 90
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))         # 180
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))         # 270
        return outs

    def _eight_rot_logits(xb):
        """8-way TTA: 45도 단위 회전 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)"""
        outs = []
        
        # 0°
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))
        
        # 90° 단위 회전 (rot90 사용)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))   # 90°
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))   # 180°
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))   # 270°
        
        # 45° 단위 회전 (affine 사용)
        B, C, H, W = xb.shape
        for angle in [45, 135, 225, 315]:
            # 회전 행렬 생성
            theta_rad = torch.tensor(angle * 3.14159265 / 180.0, device=xb.device)
            cos_t = torch.cos(theta_rad)
            sin_t = torch.sin(theta_rad)
            
            # Affine 변환 행렬 (배치용)
            theta = torch.tensor([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0]
            ], dtype=xb.dtype, device=xb.device).unsqueeze(0).repeat(B, 1, 1)
            
            grid = F.affine_grid(theta, size=xb.size(), align_corners=False)
            rotated = F.grid_sample(xb, grid, mode='bilinear', 
                                   padding_mode='border', align_corners=False)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outs.append(model(rotated))
        
        return outs

    # Variant × fold inference
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

            # temperature
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

                # 원본 방식: 가장 confident한 변환 선택
                probs = [torch.softmax(z, dim=1) for z in cand]
                maxp = torch.stack([p.max(dim=1).values for p in probs], 0)
                best_idx = torch.argmax(maxp, 0)
                pick = torch.stack([cand[r][i] for i, r in enumerate(best_idx.tolist())], 0)

                # temp 적용 후 누적
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

    # Combine variants
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
    preds = probs_img.argmax(1)

    # Easy-Lock 게이트: 높은 확신도의 쉬운 클래스는 고정
    ensemble_cfg = cfg.get('ensemble', {})
    if ensemble_cfg:
        easy_classes = ensemble_cfg.get('easy_classes', [])
        easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})
        
        locked_count = 0
        for i in range(len(preds)):
            pred_cls = preds[i]
            conf = probs_img[i, pred_cls]
            
            # easy_classes에 속하고 임계치 초과 시 lock
            if pred_cls in easy_classes:
                threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf >= threshold:
                    # 이미 pred_cls로 예측되어 있으므로 lock (변경 방지)
                    locked_count += 1
        
        if locked_count > 0:
            print(f"[Easy-Lock] Locked {locked_count}/{len(preds)} easy samples (conf >= thresholds)")

    # 메타 스태킹 (불확실 구간에만)
    if META_CLF is not None:
        meta_model = META_CLF['model'] if isinstance(META_CLF, dict) and 'model' in META_CLF else META_CLF
        Xmeta = feats_from_probs_np(probs_img)
        meta_preds = meta_model.predict(Xmeta)

        # 불확실 게이트 기준 계산
        eps = 1e-9
        ent = -(probs_img * np.log(probs_img + eps)).sum(1)
        part = np.partition(-probs_img, 1, axis=1)
        top1 = -part[:,0]; top2 = -part[:,1]
        margin = top1 - top2

        mg_cfg = cfg.get('meta_gate', {}) or {}
        ENT_THR = float(mg_cfg.get('entropy_thr', 1.4))
        MAR_THR = float(mg_cfg.get('margin_thr', 0.10))

        # Easy-Lock 설정
        ensemble_cfg = cfg.get('ensemble', {})
        easy_classes = ensemble_cfg.get('easy_classes', [])
        easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})

        applied = 0
        for i in range(len(preds)):
            # Easy-Lock된 샘플은 제외
            pred_cls = preds[i]
            conf = probs_img[i, pred_cls]
            locked = False
            if pred_cls in easy_classes:
                threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                if conf >= threshold:
                    locked = True

            # 메타 게이트: 불확실(엔트로피 높고, 마진 낮음) AND not locked
            if (not locked) and (ent[i] > ENT_THR) and (margin[i] < MAR_THR):
                preds[i] = meta_preds[i]
                applied += 1
        if applied:
            print(f"[Meta-Gate] Applied meta to {applied}/{len(preds)} uncertain samples (entropy>{ENT_THR}, margin<{MAR_THR})")
    
    # Keyword routing for near-boundary pairs using OCR
    preds = apply_keyword_routing(cfg, test, probs_img, preds)
    
    # Text 보조 게이트 (3↔7, 4↔14 초근접 샘플)
    text_cfg = cfg.get('text_gate', {}) or {}
    if text_cfg.get('enable', False):
        pairs = text_cfg.get('pairs', [[3,7], [4,14]])
        delta_thr = float(text_cfg.get('delta_thr', 0.08))
        conf_thr = float(text_cfg.get('conf_thr', 0.35))
        prefer_higher = text_cfg.get('prefer_higher', True)
        
        # Easy-Lock 설정
        ensemble_cfg = cfg.get('ensemble', {})
        easy_classes = ensemble_cfg.get('easy_classes', [])
        easy_conf = float(ensemble_cfg.get('easy_conf', 0.92))
        easy_conf_map = ensemble_cfg.get('easy_conf_map', {})
        
        text_adjusted = 0
        for (a, b) in pairs:
            pa = probs_img[:, a]
            pb = probs_img[:, b]
            delta = np.abs(pa - pb)
            max_conf = np.maximum(pa, pb)
            
            # 초근접 마스크: delta < threshold AND 최소 확신도 충족
            near_mask = (delta < delta_thr) & (max_conf > conf_thr)
            near_idx = np.where(near_mask)[0]
            
            for i in near_idx:
                pred_cls = preds[i]
                conf = probs_img[i, pred_cls]
                
                # Easy-Lock 체크
                locked = False
                if pred_cls in easy_classes:
                    threshold = float(easy_conf_map.get(str(pred_cls), easy_conf))
                    if conf >= threshold:
                        locked = True
                
                if locked:
                    continue
                
                # 텍스트 휴리스틱 적용
                if prefer_higher:
                    # 높은 확률 선택
                    preds[i] = a if pa[i] >= pb[i] else b
                else:
                    # 특정 규칙 (예: 클래스 a 선호)
                    preds[i] = a
                
                text_adjusted += 1
        
        if text_adjusted > 0:
            print(f"[Text-Gate] Adjusted {text_adjusted} near-boundary samples (delta<{delta_thr}) for pairs {pairs}")
    
    # 페어 리파이너 (Easy-Lock 적용)
    def apply_pair_refiner_with_lock(P, preds, a, b, clf_pair, delta_thr, conf_thr, easy_cfg, pair_gate_cfg=None, global_ent=None, global_margin=None):
        if clf_pair is None:
            return preds
        
        new_preds = preds.copy()
        pa, pb = P[:, a], P[:, b]
        base_mask = (np.abs(pa - pb) < delta_thr) & (np.maximum(pa, pb) > conf_thr)
        mask = base_mask
        # Optional: additional gating by entropy/margin if provided
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
        
        # Easy-Lock 필터링
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
                        continue  # Locked, skip
                unlocked_idx.append(i)
            idx = np.array(unlocked_idx)
        
        if len(idx) == 0:
            return new_preds
        
        ent = -(P[idx]*np.log(P[idx]+1e-9)).sum(1, keepdims=True)
        X = np.concatenate([pa[idx][:,None], pb[idx][:,None], ent, (pb[idx]-pa[idx])[:,None]], 1)
        z = clf_pair.predict(X)
        new_preds[idx] = np.where(z==1, b, a)
        return new_preds

    # Pair-gate config
    pg_cfg = cfg.get('pair_gate', {}) or {}
    use_pg = bool(pg_cfg.get('enable', True))
    # Precompute global entropy/margin once
    eps = 1e-9
    part_pg = np.partition(-probs_img, 1, axis=1)
    top1_pg = -part_pg[:,0]; top2_pg = -part_pg[:,1]
    margin_pg = top1_pg - top2_pg
    ent_pg = -(probs_img * np.log(probs_img + eps)).sum(1)

    # thresholds
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

    # Optional postprocess for confusing classes (3 and 7)
    pp_cfg = cfg.get('postprocess', {}) or {}
    if pp_cfg.get('enable', False):
        probs = torch.softmax(mean_logits, dim=1).cpu().numpy()
        cls_a = int(pp_cfg.get('class_a', 3))
        cls_b = int(pp_cfg.get('class_b', 7))
        delta_thr = float(pp_cfg.get('delta_threshold', 0.04))
        conf_thr = float(pp_cfg.get('confidence_threshold', 0.55))
        prefer = str(pp_cfg.get('prefer', 'higher'))  # 'higher' or explicit class id
        adjust_count = 0
        for i in range(len(preds)):
            pa, pb = probs[i, cls_a], probs[i, cls_b]
            if abs(pa - pb) < delta_thr and max(pa, pb) > conf_thr:
                if prefer == 'higher':
                    preds[i] = cls_a if pa >= pb else cls_b
                else:
                    try:
                        preds[i] = int(prefer)
                    except Exception:
                        preds[i] = cls_a if pa >= pb else cls_b
                adjust_count += 1
        print(f"[Postprocess] Adjusted {adjust_count} predictions under delta<{delta_thr} and conf>{conf_thr}")

    sub = pd.read_csv(paths['sample_csv'])
    idc = [c for c in sub.columns if c.lower() in ['id','image_id','filename']]
    ycol = [c for c in sub.columns if c.lower() in ['label','target','class']]
    id_col = idc[0] if idc else sub.columns[0]
    y_col = ycol[0] if ycol else sub.columns[-1]

    if len(sub) != len(preds):
        raise ValueError(f"sample rows {len(sub)} != preds {len(preds)}")

    sub[y_col] = preds
    sub = sub[[id_col, y_col]]   # 포맷 강제
    # Save predictions
    out_csv = os.path.join(paths['out_dir'], 'submission.csv')
    os.makedirs(paths['out_dir'], exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    
    # Save logits and image IDs for analysis
    logits_path = os.path.join(paths['out_dir'], 'predict_logits.pt')
    torch.save({
        'logits': mean_logits.numpy(),
        'img_ids': image_ids,
        'predictions': preds,
    }, logits_path)
    print(f"Saved logits → {logits_path}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=4, help='TTA mode: 0/1=no TTA, 4=90° rotations, 8=45° rotations')
    args = ap.parse_args()
    predict_ensemble(args.config, tta=args.tta)
