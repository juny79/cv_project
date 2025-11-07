import pandas as pd
import numpy as np
import torch
from collections import defaultdict

def load_logits(model_dir, fold_range=range(5)):
    """Load and combine logits from all folds"""
    all_logits = []
    for k in fold_range:
        p = f'{model_dir}/fold{k}/best.pt'
        try:
            ckpt = torch.load(p, map_location='cpu')
            if 'logits' in ckpt:
                all_logits.append(ckpt['logits'])
        except:
            print(f"Warning: Could not load {p}")
    return np.mean(all_logits, axis=0) if all_logits else None

def analyze_confidence(logits):
    """Analyze prediction confidence from logits"""
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    return probs, conf, preds

def confidence_metrics(conf, preds, changes=None):
    """Calculate confidence metrics, optionally focusing on changed predictions"""
    if changes is not None:
        conf = conf[changes]
        preds = preds[changes]
    
    return {
        'mean': np.mean(conf),
        'std': np.std(conf),
        'min': np.min(conf),
        'max': np.max(conf),
        'median': np.median(conf),
        'q25': np.percentile(conf, 25),
        'q75': np.percentile(conf, 75)
    }

def class_confidence_analysis(probs, preds):
    """Analyze confidence by class"""
    class_conf = defaultdict(list)
    for i, (p, pred) in enumerate(zip(probs, preds)):
        class_conf[pred].append(p[pred])
    
    metrics = {}
    for cls in sorted(class_conf.keys()):
        conf = np.array(class_conf[cls])
        metrics[cls] = {
            'count': len(conf),
            'mean_conf': np.mean(conf),
            'std_conf': np.std(conf),
            'min_conf': np.min(conf),
            'max_conf': np.max(conf)
        }
    return metrics

def analyze_changes(df1, df4, logits1=None, logits4=None):
    """Comprehensive analysis of prediction changes and confidence"""
    # Basic change analysis
    changes = (df1['target'] != df4['target'])
    changed_idx = np.where(changes)[0]
    
    print(f"\n=== Change Analysis ===")
    print(f"Total predictions: {len(df1)}")
    print(f"Changed predictions: {sum(changes)} ({sum(changes)/len(df1)*100:.2f}%)")
    
    # Confidence analysis if logits available
    if logits1 is not None and logits4 is not None:
        probs1, conf1, preds1 = analyze_confidence(logits1)
        probs4, conf4, preds4 = analyze_confidence(logits4)
        
        print("\n=== Confidence Analysis ===")
        print("\nModel 1 (aug_multiplier=1):")
        m1 = confidence_metrics(conf1, preds1)
        print(f"Overall confidence: mean={m1['mean']:.3f} ± {m1['std']:.3f} (median={m1['median']:.3f})")
        m1_changed = confidence_metrics(conf1, preds1, changes)
        print(f"Changed predictions confidence: mean={m1_changed['mean']:.3f} ± {m1_changed['std']:.3f}")
        
        print("\nModel 4 (aug_multiplier=4):")
        m4 = confidence_metrics(conf4, preds4)
        print(f"Overall confidence: mean={m4['mean']:.3f} ± {m4['std']:.3f} (median={m4['median']:.3f})")
        m4_changed = confidence_metrics(conf4, preds4, changes)
        print(f"Changed predictions confidence: mean={m4_changed['mean']:.3f} ± {m4_changed['std']:.3f}")
        
        # Per-class analysis
        print("\n=== Per-class Analysis ===")
        class_conf1 = class_confidence_analysis(probs1, preds1)
        class_conf4 = class_confidence_analysis(probs4, preds4)
        
        # Compare classes with major changes
        major_changes = []
        for cls in range(17):
            n1 = (preds1 == cls).sum()
            n4 = (preds4 == cls).sum()
            if abs(n1 - n4) > 10:  # threshold for major changes
                delta = n4 - n1
                conf_delta = class_conf4[cls]['mean_conf'] - class_conf1[cls]['mean_conf']
                major_changes.append({
                    'class': cls,
                    'count_delta': delta,
                    'conf_delta': conf_delta,
                    'mult1_conf': class_conf1[cls]['mean_conf'],
                    'mult4_conf': class_conf4[cls]['mean_conf'],
                    'mult1_count': n1,
                    'mult4_count': n4
                })
        
        print("\nClasses with major changes:")
        changes_df = pd.DataFrame(major_changes)
        if len(changes_df) > 0:
            print(changes_df.sort_values('count_delta', ascending=False))
        else:
            print("No classes with major changes found")

if __name__ == '__main__':
    # Load predictions
    df1 = pd.read_csv('outputs/full_mult1/submission_17_14.csv')
    df4 = pd.read_csv('outputs/full_mult4/submission_17_44.csv')
    
    # Load logits if available
    logits1 = load_logits('outputs/full_mult1')
    logits4 = load_logits('outputs/full_mult4')
    
    analyze_changes(df1, df4, logits1, logits4)