import argparse
import os
import pandas as pd
import numpy as np
import torch

ID_CANDIDATES = ['image_id','ID','id','filename']
LBL_CANDIDATES = ['label','target','class']

def find_cols(df):
    id_col = next((c for c in ID_CANDIDATES if c in df.columns), df.columns[0])
    y_col = next((c for c in LBL_CANDIDATES if c in df.columns), df.columns[-1])
    return id_col, y_col

def load_submission(path):
    df = pd.read_csv(path)
    id_col, y_col = find_cols(df)
    return df[[id_col, y_col]].rename(columns={id_col:'id', y_col:'label'})

def load_confidence(logits_path):
    if not logits_path or not os.path.exists(logits_path):
        return None
    # Explicitly allow loading non-weight pickle content saved via torch.save
    bundle = torch.load(logits_path, map_location='cpu', weights_only=False)
    logits = bundle.get('logits')
    if logits is None:
        return None
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    conf = probs.max(axis=1)
    return conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tta4_csv', required=True)
    ap.add_argument('--tta8_csv', required=True)
    ap.add_argument('--tta4_logits', default=None)
    ap.add_argument('--tta8_logits', default=None)
    ap.add_argument('--out_dir', default='outputs/tta_comparison')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df4 = load_submission(args.tta4_csv)
    df8 = load_submission(args.tta8_csv)
    df4 = df4.sort_values('id').reset_index(drop=True)
    df8 = df8.sort_values('id').reset_index(drop=True)

    if len(df4) != len(df8) or not np.all(df4['id'].values == df8['id'].values):
        raise ValueError('TTA submissions have different rows or order.')

    pred4 = df4['label'].values
    pred8 = df8['label'].values
    n = len(pred4)
    changed = pred4 != pred8
    n_changed = int(changed.sum())
    pct_changed = 100.0 * n_changed / n

    print('=== TTA=4 vs TTA=8 비교 ===')
    print(f'- 샘플 수: {n}')
    print(f'- 변경 수: {n_changed} ({pct_changed:.2f}%)')
    print(f'- 동일 수: {n - n_changed} ({100 - pct_changed:.2f}%)')

    # Changes detail
    detail = pd.DataFrame({
        'id': df4['id'],
        'tta4': pred4,
        'tta8': pred8,
    })
    detail = detail[changed]

    # Confidence (optional)
    conf4 = load_confidence(args.tta4_logits)
    conf8 = load_confidence(args.tta8_logits)
    if conf4 is not None and conf8 is not None:
        detail = detail.join(pd.DataFrame({
            'tta4_conf': conf4,
            'tta8_conf': conf8,
        }), how='left')
        if 'tta4_conf' in detail and 'tta8_conf' in detail:
            detail['conf_delta'] = detail['tta8_conf'] - detail['tta4_conf']

    # Transition counts
    if len(detail) > 0:
        trans = detail.groupby(['tta4','tta8']).size().reset_index(name='count').sort_values('count', ascending=False)
        trans.to_csv(os.path.join(args.out_dir, 'transitions.csv'), index=False)
    detail.to_csv(os.path.join(args.out_dir, 'differences.csv'), index=False)

    # Class distribution
    dist4 = pd.Series(pred4).value_counts().sort_index()
    dist8 = pd.Series(pred8).value_counts().sort_index()
    dist = pd.DataFrame({'class': dist4.index, 'tta4_count': dist4.values}).merge(
        pd.DataFrame({'class': dist8.index, 'tta8_count': dist8.values}), on='class', how='outer').fillna(0)
    dist['delta'] = dist['tta8_count'] - dist['tta4_count']
    dist['pct_change'] = np.where(dist['tta4_count']>0, 100.0*dist['delta']/dist['tta4_count'], np.nan)
    dist.to_csv(os.path.join(args.out_dir, 'distribution.csv'), index=False)

    # Quick text summary
    with open(os.path.join(args.out_dir, 'SUMMARY.txt'), 'w') as f:
        f.write('TTA=4 vs TTA=8 비교 요약\n')
        f.write(f'- 샘플 수: {n}\n')
        f.write(f'- 변경 수: {n_changed} ({pct_changed:.2f}%)\n')
        f.write(f'- 동일 수: {n - n_changed} ({100 - pct_changed:.2f}%)\n')
        if len(detail) > 0:
            top = trans.head(10) if 'trans' in locals() else None
            if top is not None:
                f.write('\n상위 전환 패턴:\n')
                for _, r in top.iterrows():
                    f.write(f"  {int(r['tta4'])} -> {int(r['tta8'])}: {int(r['count'])}\n")

    print(f'✓ 저장: {args.out_dir}/differences.csv, transitions.csv, distribution.csv, SUMMARY.txt')

if __name__ == '__main__':
    main()
