import pandas as pd
import numpy as np
import torch
import os

print("=== TTA=4 vs TTA=8 예측 비교 ===\n")

# Load predictions
tta4_csv = '/root/cv_project/outputs/full_mult4_tta4/submission.csv'
tta8_csv = '/root/cv_project/outputs/full_mult6_tta8/submission.csv'

df_tta4 = pd.read_csv(tta4_csv)
df_tta8 = pd.read_csv(tta8_csv)

# Ensure same ordering
df_tta4 = df_tta4.sort_values('ID').reset_index(drop=True)
df_tta8 = df_tta8.sort_values('ID').reset_index(drop=True)

# Compare predictions
pred_tta4 = df_tta4['target'].values
pred_tta8 = df_tta8['target'].values

n_total = len(pred_tta4)
n_changed = (pred_tta4 != pred_tta8).sum()
pct_changed = 100.0 * n_changed / n_total

print(f"총 샘플 수: {n_total}")
print(f"예측 변경된 샘플: {n_changed} ({pct_changed:.2f}%)")
print(f"예측 유지된 샘플: {n_total - n_changed} ({100 - pct_changed:.2f}%)\n")

# Load logits for confidence analysis
logits_tta4 = torch.load('/root/cv_project/outputs/full_mult4_tta4/predict_logits.pt', weights_only=False)
logits_tta8 = torch.load('/root/cv_project/outputs/full_mult6_tta8/predict_logits.pt', weights_only=False)

probs_tta4 = torch.softmax(torch.tensor(logits_tta4['logits']), dim=1).numpy()
probs_tta8 = torch.softmax(torch.tensor(logits_tta8['logits']), dim=1).numpy()

conf_tta4 = probs_tta4.max(axis=1)
conf_tta8 = probs_tta8.max(axis=1)

print("=== 신뢰도 비교 ===")
print(f"TTA=4 평균 신뢰도: {conf_tta4.mean():.4f} (std: {conf_tta4.std():.4f})")
print(f"TTA=8 평균 신뢰도: {conf_tta8.mean():.4f} (std: {conf_tta8.std():.4f})")
print(f"신뢰도 변화: {(conf_tta8.mean() - conf_tta4.mean()):.4f}\n")

# Analyze changes by class
if n_changed > 0:
    print("=== 클래스별 예측 변화 ===")
    changes = []
    for i in range(n_total):
        if pred_tta4[i] != pred_tta8[i]:
            changes.append({
                'img_id': df_tta4.iloc[i]['ID'],
                'tta4_pred': pred_tta4[i],
                'tta8_pred': pred_tta8[i],
                'tta4_conf': conf_tta4[i],
                'tta8_conf': conf_tta8[i],
                'conf_delta': conf_tta8[i] - conf_tta4[i]
            })
    
    changes_df = pd.DataFrame(changes)
    
    # Count changes by class transition
    transition_counts = changes_df.groupby(['tta4_pred', 'tta8_pred']).size().reset_index(name='count')
    transition_counts = transition_counts.sort_values('count', ascending=False)
    
    print("\n주요 클래스 전환 패턴 (상위 10개):")
    print(transition_counts.head(10).to_string(index=False))
    
    # Class distribution changes
    print("\n=== 클래스별 분포 변화 ===")
    dist_tta4 = pd.Series(pred_tta4).value_counts().sort_index()
    dist_tta8 = pd.Series(pred_tta8).value_counts().sort_index()
    
    dist_df = pd.DataFrame({
        'class': dist_tta4.index,
        'tta4_count': dist_tta4.values,
        'tta8_count': dist_tta8.values
    })
    dist_df['delta'] = dist_df['tta8_count'] - dist_df['tta4_count']
    dist_df['pct_change'] = 100.0 * dist_df['delta'] / dist_df['tta4_count']
    
    print(dist_df.to_string(index=False))
    
    # Focus on class 3 and 7
    print("\n=== 클래스 3, 7 집중 분석 ===")
    class_3_7_changes = changes_df[
        ((changes_df['tta4_pred'] == 3) | (changes_df['tta4_pred'] == 7) | 
         (changes_df['tta8_pred'] == 3) | (changes_df['tta8_pred'] == 7))
    ]
    
    print(f"클래스 3/7 관련 변화: {len(class_3_7_changes)}건")
    
    if len(class_3_7_changes) > 0:
        print("\n클래스 3↔7 직접 전환:")
        c3_to_7 = len(class_3_7_changes[(class_3_7_changes['tta4_pred'] == 3) & 
                                         (class_3_7_changes['tta8_pred'] == 7)])
        c7_to_3 = len(class_3_7_changes[(class_3_7_changes['tta4_pred'] == 7) & 
                                         (class_3_7_changes['tta8_pred'] == 3)])
        print(f"  3 → 7: {c3_to_7}건")
        print(f"  7 → 3: {c7_to_3}건")
    
    # Save detailed changes
    os.makedirs('/root/cv_project/outputs/tta_comparison', exist_ok=True)
    changes_df.to_csv('/root/cv_project/outputs/tta_comparison/tta4_vs_tta8_changes.csv', index=False)
    dist_df.to_csv('/root/cv_project/outputs/tta_comparison/tta4_vs_tta8_distribution.csv', index=False)
    print("\n✓ 변화 분석 저장: outputs/tta_comparison/")
else:
    print("예측 변화가 없습니다. TTA=4와 TTA=8의 결과가 동일합니다.")

print("\n=== 완료 ===")
