"""Train meta stacking multinomial LR and pairwise binary refiners.

This script consumes per-fold OOF logits produced by `train.py` (files:
`oof_fold{K}.npz` containing `logits` and `labels`, plus `temp.npy` for
temperature scaling). It builds calibrated probabilities, extracts
meta features, trains:
  1. Multinomial Logistic Regression (stacking layer)
  2. Binary Logistic Regression refiners for specified class pairs

Outputs written under `--save_dir` (default: extern):
  - meta_full.joblib
  - pair_{a}_{b}.joblib for each requested pair

Usage:
  python scripts/train_meta.py \
      --exp_out_dir outputs/exp_mult4_tta4_sharpen \
      --save_dir extern \
      --pairs 3,7;4,14
"""
import os, glob, joblib, numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, log_loss

def softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)

def entropy(p, axis=1, eps=1e-9):
    return -(p * np.log(p + eps)).sum(axis=axis)

def top2_margin(p):
    # top1 - top2
    part = np.partition(-p, 1, axis=1)
    top1 = -part[:,0]; top2 = -part[:,1]
    return top1 - top2

def build_meta_features(p):
    # Concatenate probs + entropy + top2 margin
    ent = entropy(p)[:, None]
    mar = top2_margin(p)[:, None]
    return np.concatenate([p, ent, mar], axis=1)

def discover_folds(exp_out_dir):
    dirs = []
    for d in sorted(glob.glob(os.path.join(exp_out_dir, 'fold*'))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, 'oof_fold*.npz')):
            dirs.append(d)
    return dirs

def load_fold_oof(fold_dir):
    npz_files = sorted(glob.glob(os.path.join(fold_dir, 'oof_fold*.npz')))
    if not npz_files:
        raise FileNotFoundError(f"No oof_fold*.npz in {fold_dir}")
    arr = np.load(npz_files[0])
    logits = arr['logits']; labels = arr['labels']
    temp_path = os.path.join(fold_dir, 'temp.npy')
    T = 1.0
    if os.path.exists(temp_path):
        T = float(np.load(temp_path).reshape(-1)[0])
    probs = softmax(logits / max(T,1e-4), axis=1)
    return probs.astype(np.float32), labels.astype(np.int64)

def train_meta_classifier(probs, labels):
    X = build_meta_features(probs)
    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=2000,
        C=1.5, class_weight='balanced'
    )
    clf.fit(X, labels)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    f1 = f1_score(labels, pred, average='macro')
    nll = log_loss(labels, proba, labels=list(range(probs.shape[1])))
    print(f"[META] OOF Macro F1={f1:.4f} NLL={nll:.4f}")
    print(classification_report(labels, pred, digits=4))
    return clf

def select_pair(probs, labels, a, b):
    mask = np.isin(labels, [a,b])
    p = probs[mask]; y = labels[mask]
    if len(y) == 0:
        return None, None
    # binary target: a -> 1, b -> 0
    t = (y == a).astype(np.int64)
    feats = np.concatenate([
        p[:, a][:,None],
        p[:, b][:,None],
        (p[:, a] - p[:, b])[:,None],
        entropy(p)[:,None],
        top2_margin(p)[:,None],
    ], axis=1).astype(np.float32)
    return feats, t

def train_pairwise(feats, tgt, a, b):
    clf = LogisticRegression(solver='lbfgs', max_iter=1500, class_weight='balanced')
    clf.fit(feats, tgt)
    pred = clf.predict(feats)
    f1_bin = f1_score(tgt, pred, average='binary')
    print(f"[PAIR {a}-{b}] OOF Bin-F1={f1_bin:.4f} N={len(tgt)}")
    return clf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_out_dir', default='outputs/full_mult4_tta4', help='Experiment output dir e.g. outputs/full_mult4_tta4')
    ap.add_argument('--save_dir', default='extern', help='Directory to save joblib models')
    ap.add_argument('--pairs', default='3,7;4,14', help='Semicolon separated class pairs for refiners')
    args = ap.parse_args()

    folds = discover_folds(args.exp_out_dir)
    if not folds:
        print(f"[WARN] No folds discovered in {args.exp_out_dir}")
        return
    all_probs, all_labels = [], []
    for fd in folds:
        p, lab = load_fold_oof(fd)
        print(f"[LOAD] {fd} -> probs {p.shape} labels {lab.shape}")
        all_probs.append(p); all_labels.append(lab)
    probs = np.concatenate(all_probs, 0)
    labels = np.concatenate(all_labels, 0)
    print(f"[DATA] Aggregated OOF: N={len(labels)} C={probs.shape[1]} folds={len(folds)}")

    meta_clf = train_meta_classifier(probs, labels)
    os.makedirs(args.save_dir, exist_ok=True)
    meta_path = os.path.join(args.save_dir, 'meta_full.joblib')
    joblib.dump({'model': meta_clf}, meta_path)
    print(f"[SAVE] Meta model -> {meta_path}")

    for pair in args.pairs.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        a_str, b_str = pair.split(',')
        a, b = int(a_str), int(b_str)
        feats, tgt = select_pair(probs, labels, a, b)
        if feats is None or len(tgt) < 8:
            print(f"[PAIR {a}-{b}] insufficient samples ({0 if feats is None else len(tgt)}). Skipping.")
            continue
        clf_pair = train_pairwise(feats, tgt, a, b)
        path = os.path.join(args.save_dir, f'pair_{a}_{b}.joblib')
        joblib.dump({'model': clf_pair, 'pair': (a, b)}, path)
        print(f"[SAVE] Pair model -> {path}")

if __name__ == '__main__':
    main()
