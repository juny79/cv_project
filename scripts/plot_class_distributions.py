import os
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional imports
try:
    import seaborn as sns
    sns.set(style='whitegrid')
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# Metrics
try:
    from scipy.stats import ks_2samp, wasserstein_distance
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def load_logits(path_prefix):
    path = os.path.join(path_prefix, 'predict_logits.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logits file not found: {path}")
    # load with weights_only=False to allow non-standard objects stored
    logits = torch.load(path, weights_only=False)
    if 'logits' not in logits:
        raise KeyError('Loaded object does not contain "logits" key')
    return logits


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def summarize_and_plot(prefix1='outputs/full_mult1', prefix4='outputs/full_mult4', out_dir='outputs/plots'):
    ensure_dir(out_dir)
    l1 = load_logits(prefix1)
    l4 = load_logits(prefix4)

    probs1 = softmax(l1['logits'], axis=1)
    probs4 = softmax(l4['logits'], axis=1)

    n_classes = probs1.shape[1]
    rows = []

    for c in range(n_classes):
        arr1 = probs1[:, c]
        arr4 = probs4[:, c]

        mean1 = float(np.mean(arr1))
        mean4 = float(np.mean(arr4))
        std1 = float(np.std(arr1))
        std4 = float(np.std(arr4))
        mean_diff = mean4 - mean1

        ks_stat = None
        ks_p = None
        w_dist = None
        if HAS_SCIPY:
            ks_res = ks_2samp(arr1, arr4)
            ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)
            w_dist = float(wasserstein_distance(arr1, arr4))

        rows.append({
            'class': c,
            'mean_mult1': mean1,
            'mean_mult4': mean4,
            'std_mult1': std1,
            'std_mult4': std4,
            'mean_diff': mean_diff,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'wasserstein': w_dist,
        })

        # Plot
        plt.figure(figsize=(6,4))
        if HAS_SEABORN:
            sns.kdeplot(arr1, label='mult1', fill=False)
            sns.kdeplot(arr4, label='mult4', fill=False)
        else:
            plt.hist(arr1, bins=50, density=True, alpha=0.5, label='mult1')
            plt.hist(arr4, bins=50, density=True, alpha=0.5, label='mult4')

        plt.title(f'Class {c} probability distribution')
        plt.xlabel('Predicted probability for class')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'class_{c:02d}_prob_dist.png'))
        plt.close()

    df = pd.DataFrame(rows)
    df = df.sort_values(by='mean_diff', ascending=False)
    csv_path = os.path.join(out_dir, 'class_distribution_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved per-class plots to {out_dir} and summary CSV to {csv_path}")

    # Print top shifts
    print('\nTop classes by mean difference (mult4 - mult1):')
    print(df[['class','mean_mult1','mean_mult4','mean_diff','ks_stat','ks_p','wasserstein']].head(10).to_string(index=False))

    print('\nTop classes by Wasserstein (if available):')
    if HAS_SCIPY:
        print(df.sort_values('wasserstein', ascending=False)[['class','mean_diff','wasserstein']].head(10).to_string(index=False))
    else:
        print('SciPy not available; install scipy to compute distribution metrics')


if __name__ == '__main__':
    summarize_and_plot()
