#!/usr/bin/env python
import os, sys, argparse, json
import numpy as np, pandas as pd
import torch
import re

def _read_preds(p):
    df = pd.read_csv(p)
    idc = [c for c in df.columns if c.lower() in ['id','image_id','filename']]
    yc  = [c for c in df.columns if c.lower() in ['label','target','class']]
    if not idc or not yc:
        raise ValueError(f"CSV must contain id and label columns: {p}")
    return df[[idc[0], yc[0]]].rename(columns={idc[0]:'id', yc[0]:'label'})


def _load_logits_pt(p):
    if not p or not os.path.exists(p):
        return None
    obj = torch.load(p, map_location='cpu')
    logits = obj.get('logits', None)
    img_ids = obj.get('img_ids', None)
    preds = obj.get('predictions', None)
    if logits is None or img_ids is None:
        return None
    P = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return {'probs': P, 'img_ids': [str(x) for x in img_ids], 'preds': preds}


def _join_by_id(base_df, new_df):
    j = base_df.merge(new_df, on='id', how='inner', suffixes=('_base','_new'))
    return j


def summarize_flips(df, pairs=[(3,7),(4,14)]):
    out = {}
    total_flips = (df['label_base'] != df['label_new']).sum()
    out['total_flips'] = int(total_flips)
    for a,b in pairs:
        m = ((df['label_base']==a) & (df['label_new']==b)) | ((df['label_base']==b) & (df['label_new']==a))
        sub = df[m]
        out[f'pair_{a}_{b}_flips'] = int(len(sub))
        out[f'pair_{a}_{b}_{a}_to_{b}'] = int(((sub['label_base']==a) & (sub['label_new']==b)).sum())
        out[f'pair_{a}_{b}_{b}_to_{a}'] = int(((sub['label_base']==b) & (sub['label_new']==a)).sum())
    return out


def near_boundary_counts(logits_info, pairs=[(3,7),(4,14)], delta_thr=0.04, conf_thr=0.55):
    if logits_info is None:
        return {}
    P = logits_info['probs']
    res = {'delta_thr':delta_thr, 'conf_thr':conf_thr}
    total_near = 0
    for a,b in pairs:
        pa, pb = P[:,a], P[:,b]
        delta = np.abs(pa - pb)
        maxc = np.maximum(pa, pb)
        near = ((delta < delta_thr) & (maxc > conf_thr)).sum()
        res[f'pair_{a}_{b}_near'] = int(near)
        total_near += int(near)
    res['near_total'] = int(total_near)
    return res


def _normalize_text(s):
    if not isinstance(s,str):
        return ''
    s = s.lower()
    s = re.sub(r'\s+','',s)
    s = re.sub(r'[^0-9a-zA-Z가-힣]+','',s)
    s = re.sub(r'\d+','',s)
    return s

KEYWORDS = {
    3: [r'입퇴원', r'입원', r'퇴원', r'입퇴원확인서', r'입원확인서', r'퇴원확인서'],
    7: [r'외래', r'통원', r'외래진료', r'통원치료', r'외래확인서'],
    4: [r'진단서', r'진단명', r'의사진단서', r'진단내용'],
    14:[r'소견서', r'소견', r'의견서', r'소견내용']
}

def keyword_hit_stats(ocr_csv, ids, pairs=[(3,7),(4,14)], min_hits=2):
    if not ocr_csv or not os.path.exists(ocr_csv):
        return {'ocr':'missing'}
    df = pd.read_csv(ocr_csv)
    idc = [c for c in df.columns if c.lower() in ['id','image_id','filename']]
    txtc = [c for c in df.columns if c.lower() in ['text','ocr','ocr_text']]
    if not idc or not txtc:
        return {'ocr':'invalid'}
    df = df[[idc[0], txtc[0]]].rename(columns={idc[0]:'id', txtc[0]:'text'})
    m = df['id'].astype(str).isin(set([str(x) for x in ids]))
    df = df[m]
    df['norm'] = df['text'].map(_normalize_text)
    comp = {k:[re.compile(p) for p in v] for k,v in KEYWORDS.items()}

    def _hits(s, cid):
        c = 0
        for rgx in comp.get(cid, []):
            if rgx.search(s): c += 1
        return c

    stats = {'ocr':'ok','min_hits':min_hits}
    total = len(df)
    stats['count'] = int(total)
    eligible = 0
    hit_based = 0
    for a,b in pairs:
        ha = df['norm'].map(lambda x: _hits(x,a))
        hb = df['norm'].map(lambda x: _hits(x,b))
        elig = ((ha>=min_hits) | (hb>=min_hits)).sum()
        eligible += int(elig)
        hit_based += int(((ha>hb)&(ha>=min_hits)).sum() + ((hb>ha)&(hb>=min_hits)).sum())
        stats[f'pair_{a}_{b}_eligible'] = int(elig)
        stats[f'pair_{a}_{b}_ha_ge_min'] = int((ha>=min_hits).sum())
        stats[f'pair_{a}_{b}_hb_ge_min'] = int((hb>=min_hits).sum())
    stats['eligible_total'] = int(eligible)
    stats['hit_based_total'] = int(hit_based)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_csv', required=True, help='Baseline predictions CSV (id,label)')
    ap.add_argument('--new_csv', required=True, help='New predictions CSV (id,label)')
    ap.add_argument('--logits_pt', default='', help='predict_logits.pt from new run (to compute near-boundary)')
    ap.add_argument('--ocr_csv', default='', help='OCR text CSV for keyword hit stats')
    ap.add_argument('--delta_thr', type=float, default=0.04)
    ap.add_argument('--conf_thr', type=float, default=0.55)
    args = ap.parse_args()

    base = _read_preds(args.base_csv)
    new  = _read_preds(args.new_csv)
    j = _join_by_id(base, new)

    flips = summarize_flips(j)
    logits_info = _load_logits_pt(args.logits_pt)
    near = near_boundary_counts(logits_info, delta_thr=args.delta_thr, conf_thr=args.conf_thr)
    ocr = keyword_hit_stats(args.ocr_csv, j['id'].tolist())

    summary = {'flips':flips, 'near':near, 'ocr_hits':ocr}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Save flipped cases CSV for pairs
    if logits_info is not None:
        P = logits_info['probs']
        id_to_idx = {str(i):k for k,i in enumerate(logits_info['img_ids'])}
        pairs = [(3,7),(4,14)]
        recs = []
        for _, r in j.iterrows():
            if r['label_base'] == r['label_new']:
                continue
            sid = str(r['id'])
            if sid not in id_to_idx:
                continue
            k = id_to_idx[sid]
            for a,b in pairs:
                pa, pb = float(P[k,a]), float(P[k,b])
                delta = abs(pa - pb); maxc = max(pa, pb)
                recs.append({'id':sid,'base':int(r['label_base']),'new':int(r['label_new']),
                             'a':a,'b':b,'pa':pa,'pb':pb,'delta':delta,'maxc':maxc})
        if recs:
            out_csv = os.path.join(os.path.dirname(args.new_csv), 'routing_flips_detail.csv')
            pd.DataFrame(recs).to_csv(out_csv, index=False)
            print(f"Saved flips detail → {out_csv}")

if __name__ == '__main__':
    main()
