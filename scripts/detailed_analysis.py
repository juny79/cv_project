import torch
import pandas as pd
import numpy as np
from scipy.special import softmax

def load_predictions(path_prefix):
    """Load predictions and logits"""
    logits = torch.load(f'{path_prefix}/predict_logits.pt', weights_only=False)
    df = pd.read_csv(f'{path_prefix}/submission.csv')
    return df, logits

def analyze_predictions(df1, logits1, df4, logits4):
    """Comprehensive prediction analysis"""
    probs1 = softmax(logits1['logits'], axis=1)
    probs4 = softmax(logits4['logits'], axis=1)
    
    # 1. Confidence Analysis
    conf1 = np.max(probs1, axis=1)
    conf4 = np.max(probs4, axis=1)
    
    print("\n=== Overall Confidence ===")
    print(f"Model 1 (aug_mult=1): {conf1.mean():.4f} ± {conf1.std():.4f}")
    print(f"Model 4 (aug_mult=4): {conf4.mean():.4f} ± {conf4.std():.4f}")
    
    # 2. Changed Predictions Analysis
    changes = (logits1['predictions'] != logits4['predictions'])
    changed_idx = np.where(changes)[0]
    
    print(f"\n=== Changed Predictions ({len(changed_idx)} samples) ===")
    print(f"Confidence in changed predictions:")
    print(f"Model 1: {conf1[changed_idx].mean():.4f} ± {conf1[changed_idx].std():.4f}")
    print(f"Model 4: {conf4[changed_idx].mean():.4f} ± {conf4[changed_idx].std():.4f}")
    
    # 3. Per-class Analysis
    print("\n=== Per-class Analysis ===")
    class_metrics = []
    for cls in range(17):
        # Original predictions
        pred1_cls = (logits1['predictions'] == cls)
        pred4_cls = (logits4['predictions'] == cls)
        
        # Changed predictions
        changed_from = np.sum(pred1_cls & changes)
        changed_to = np.sum(pred4_cls & changes)
        
        # Confidence
        conf1_cls = conf1[pred1_cls].mean() if np.any(pred1_cls) else 0
        conf4_cls = conf4[pred4_cls].mean() if np.any(pred4_cls) else 0
        
        class_metrics.append({
            'class': cls,
            'mult1_count': np.sum(pred1_cls),
            'mult4_count': np.sum(pred4_cls),
            'mult1_conf': conf1_cls,
            'mult4_conf': conf4_cls,
            'changed_from': changed_from,
            'changed_to': changed_to,
            'net_change': np.sum(pred4_cls) - np.sum(pred1_cls)
        })
    
    cls_df = pd.DataFrame(class_metrics)
    print("\nClasses with significant changes (|net_change| > 5):")
    significant = cls_df[abs(cls_df['net_change']) > 5].sort_values('net_change', ascending=False)
    print(significant[['class', 'mult1_count', 'mult4_count', 'net_change', 'mult1_conf', 'mult4_conf']])
    
    # 4. Top Confusion Patterns
    print("\n=== Top Confusion Patterns ===")
    confusion = []
    for i in changed_idx:
        old_cls = logits1['predictions'][i]
        new_cls = logits4['predictions'][i]
        old_conf = probs1[i, old_cls]
        new_conf = probs4[i, new_cls]
        confusion.append({
            'from_class': old_cls,
            'to_class': new_cls,
            'old_conf': old_conf,
            'new_conf': new_conf,
            'conf_delta': new_conf - old_conf
        })
    
    conf_df = pd.DataFrame(confusion)
    patterns = conf_df.groupby(['from_class', 'to_class']).agg({
        'old_conf': ['count', 'mean'],
        'new_conf': 'mean',
        'conf_delta': 'mean'
    }).round(4)
    
    print("\nTop changes sorted by frequency:")
    print(patterns.sort_values(('old_conf', 'count'), ascending=False).head(10))

if __name__ == '__main__':
    # Load data
    df1, logits1 = load_predictions('outputs/full_mult1')
    df4, logits4 = load_predictions('outputs/full_mult4')
    
    # Run analysis
    analyze_predictions(df1, logits1, df4, logits4)