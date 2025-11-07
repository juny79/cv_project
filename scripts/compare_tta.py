import os
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax


def load_logits(path):
    d = torch.load(os.path.join(path, 'predict_logits.pt'), weights_only=False)
    # ensure numpy arrays
    logits = np.array(d['logits'])
    preds = np.array(d['predictions'])
    img_ids = np.array(d.get('img_ids', np.arange(len(preds))))
    return {'logits': logits, 'predictions': preds, 'img_ids': img_ids}


def analyze_pair(no_tta_prefix, tta_prefix, out_dir='outputs/plots', tag='model'):
    os.makedirs(out_dir, exist_ok=True)
    a = load_logits(no_tta_prefix)
    b = load_logits(tta_prefix)

    probs_a = softmax(a['logits'], axis=1)
    probs_b = softmax(b['logits'], axis=1)

    pred_a = a['predictions']
    pred_b = b['predictions']
    img_ids = a['img_ids']

    if len(pred_a) != len(pred_b):
        raise ValueError('length mismatch between no-TTA and TTA predictions')

    # disagreement
    changes = pred_a != pred_b
    n_changes = int(changes.sum())
    pct = n_changes / len(pred_a) * 100

    conf_a = probs_a.max(axis=1)
    conf_b = probs_b.max(axis=1)
    conf_delta = conf_b - conf_a

    summary = {
        'n_samples': int(len(pred_a)),
        'n_changed': n_changes,
        'pct_changed': float(pct),
        'mean_conf_no_tta': float(conf_a.mean()),
        'mean_conf_tta': float(conf_b.mean()),
        'mean_conf_delta': float(conf_delta.mean()),
        'median_conf_delta': float(np.median(conf_delta)),
    }

    # per-class breakdown
    n_classes = probs_a.shape[1]
    rows = []
    for c in range(n_classes):
        ca = (pred_a == c)
        cb = (pred_b == c)
        rows.append({
            'class': c,
            'count_no_tta': int(ca.sum()),
            'count_tta': int(cb.sum()),
            'mean_conf_no_tta': float(conf_a[ca].mean()) if ca.any() else 0.0,
            'mean_conf_tta': float(conf_b[cb].mean()) if cb.any() else 0.0,
            'changed_from_no_tta': int(((pred_a == c) & changes).sum()),
            'changed_to_tta': int(((pred_b == c) & changes).sum()),
            'net_change': int(cb.sum() - ca.sum())
        })
    per_class_df = pd.DataFrame(rows).sort_values('net_change', ascending=False)

    # top changed samples
    changed_idx = np.where(changes)[0]
    changed_rows = []
    for i in changed_idx:
        changed_rows.append({
            'img_id': img_ids[i],
            'pred_no_tta': int(pred_a[i]),
            'pred_tta': int(pred_b[i]),
            'conf_no_tta': float(conf_a[i]),
            'conf_tta': float(conf_b[i]),
            'conf_delta': float(conf_delta[i])
        })
    changed_df = pd.DataFrame(changed_rows).sort_values('conf_delta', ascending=False)

    # save outputs
    base = f'{tag}_no_tta_vs_tta'
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f'{base}_summary.csv'), index=False)
    per_class_df.to_csv(os.path.join(out_dir, f'{base}_per_class.csv'), index=False)
    changed_df.to_csv(os.path.join(out_dir, f'{base}_changed_samples.csv'), index=False)

    # print concise summary
    print(f"--- {tag} NO-TTA vs TTA ---")
    print(f"Samples: {summary['n_samples']}, Changed: {summary['n_changed']} ({summary['pct_changed']:.2f}%)")
    print(f"Mean conf no-TTA: {summary['mean_conf_no_tta']:.4f}, with TTA: {summary['mean_conf_tta']:.4f}, delta: {summary['mean_conf_delta']:.4f}")
    print('\nTop per-class net changes:')
    print(per_class_df[['class','count_no_tta','count_tta','net_change']].head(10).to_string(index=False))
    print(f"\nSaved summary CSVs to {out_dir} (prefix {base}_*)")

    return {'summary': summary, 'per_class': per_class_df, 'changed': changed_df}


if __name__ == '__main__':
    # Compare mult1
    r1 = analyze_pair('outputs/full_mult1', 'outputs/full_mult1_tta4', out_dir='outputs/plots', tag='mult1')
    # Compare mult4
    r2 = analyze_pair('outputs/full_mult4', 'outputs/full_mult4_tta4', out_dir='outputs/plots', tag='mult4')

    print('\nDone')
