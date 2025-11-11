import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 80)
print("데이터 증강 비교: aug_multiplier=4 vs aug_multiplier=6")
print("=" * 80)

# Load training logs
mult4_log = pd.read_csv('/root/cv_project/outputs/full_mult4_tta8/train_log.csv')
mult6_log = pd.read_csv('/root/cv_project/outputs/full_mult4_tta4/train_log.csv')

print("\n### 1. 학습 설정 ###")
print(f"aug_multiplier=4: 1,256 → {1256 * 4:,} samples per fold")
print(f"aug_multiplier=6: 1,256 → {1256 * 6:,} samples per fold")

# Best F1 per fold
mult4_best = mult4_log.groupby('fold')['valid_f1'].max()
mult6_best = mult6_log.groupby('fold')['valid_f1'].max()

print("\n### 2. Fold별 Best Validation F1 ###")
print("\n" + "-" * 60)
print(f"{'Fold':<10}{'4배 증강':<20}{'6배 증강':<20}{'차이':<15}")
print("-" * 60)
for fold in range(5):
    f4 = mult4_best[fold]
    f6 = mult6_best[fold]
    diff = f6 - f4
    symbol = "✓" if diff > 0 else "✗" if diff < 0 else "="
    print(f"{fold:<10}{f4:.4f} {symbol:<15}{f6:.4f} {symbol:<15}{diff:+.4f}")

print("-" * 60)
print(f"{'평균':<10}{mult4_best.mean():.4f}{'':15}{mult6_best.mean():.4f}{'':15}{mult6_best.mean() - mult4_best.mean():+.4f}")
print(f"{'표준편차':<10}{mult4_best.std():.4f}{'':15}{mult6_best.std():.4f}{'':15}{mult6_best.std() - mult4_best.std():+.4f}")
print("-" * 60)

# Statistical summary
print("\n### 3. 통계적 요약 ###")
improvement = mult6_best.mean() - mult4_best.mean()
improvement_pct = (improvement / mult4_best.mean()) * 100
print(f"평균 F1 향상: {improvement:+.4f} ({improvement_pct:+.2f}%)")
print(f"최고 성능 (4배): Fold {mult4_best.idxmax()} = {mult4_best.max():.4f}")
print(f"최고 성능 (6배): Fold {mult6_best.idxmax()} = {mult6_best.max():.4f}")
print(f"최저 성능 (4배): Fold {mult4_best.idxmin()} = {mult4_best.min():.4f}")
print(f"최저 성능 (6배): Fold {mult6_best.idxmin()} = {mult6_best.min():.4f}")

# Epoch statistics
print("\n### 4. 학습 효율성 분석 ###")
mult4_epochs = mult4_log.groupby('fold')['epoch'].max()
mult6_epochs = mult6_log.groupby('fold')['epoch'].max()
print(f"평균 학습 에포크 (4배): {mult4_epochs.mean():.1f}")
print(f"평균 학습 에포크 (6배): {mult6_epochs.mean():.1f}")
print(f"총 학습 에포크 (4배): {mult4_epochs.sum()}")
print(f"총 학습 에포크 (6배): {mult6_epochs.sum()}")

# Final epoch comparison
mult4_final = mult4_log.groupby('fold').last()
mult6_final = mult6_log.groupby('fold').last()
print(f"\n최종 에포크 평균 Valid F1:")
print(f"  4배 증강: {mult4_final['valid_f1'].mean():.4f}")
print(f"  6배 증강: {mult6_final['valid_f1'].mean():.4f}")

# Training stability (std of valid_f1 across epochs for each fold)
print("\n### 5. 학습 안정성 (각 fold내 valid F1 표준편차) ###")
mult4_stability = mult4_log.groupby('fold')['valid_f1'].std().mean()
mult6_stability = mult6_log.groupby('fold')['valid_f1'].std().mean()
print(f"4배 증강: {mult4_stability:.4f}")
print(f"6배 증강: {mult6_stability:.4f}")
print(f"차이: {mult6_stability - mult4_stability:+.4f} ({'더 안정적' if mult6_stability < mult4_stability else '덜 안정적'})")

# Loss comparison
mult4_best_loss = mult4_log.groupby('fold')['valid_loss'].min()
mult6_best_loss = mult6_log.groupby('fold')['valid_loss'].min()
print("\n### 6. 최저 Validation Loss ###")
print(f"평균 최저 Loss (4배): {mult4_best_loss.mean():.4f}")
print(f"평균 최저 Loss (6배): {mult6_best_loss.mean():.4f}")
print(f"차이: {mult6_best_loss.mean() - mult4_best_loss.mean():.4f}")

# Create visualization
output_dir = '/root/cv_project/outputs/aug_comparison'
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Best F1 comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 Bar plot of best F1
ax1 = axes[0, 0]
x = np.arange(5)
width = 0.35
bars1 = ax1.bar(x - width/2, mult4_best.values, width, label='4배 증강', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x + width/2, mult6_best.values, width, label='6배 증강', alpha=0.8, color='coral')
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Best Validation F1', fontsize=12)
ax1.set_title('Fold별 최고 F1 Score 비교', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Fold {i}' for i in range(5)])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)

# 1.2 Learning curves for fold 1 (best performing)
ax2 = axes[0, 1]
fold1_mult4 = mult4_log[mult4_log['fold'] == 1]
fold1_mult6 = mult6_log[mult6_log['fold'] == 1]
ax2.plot(fold1_mult4['epoch'], fold1_mult4['valid_f1'], 'o-', label='4배 증강', alpha=0.8, linewidth=2)
ax2.plot(fold1_mult6['epoch'], fold1_mult6['valid_f1'], 's-', label='6배 증강', alpha=0.8, linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation F1', fontsize=12)
ax2.set_title('Fold 1 학습 곡선 비교', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 1.3 Training loss comparison
ax3 = axes[1, 0]
mult4_train_loss = mult4_log.groupby('epoch')['train_loss'].mean()
mult6_train_loss = mult6_log.groupby('epoch')['train_loss'].mean()
ax3.plot(mult4_train_loss.index, mult4_train_loss.values, 'o-', label='4배 증강', alpha=0.8, linewidth=2)
ax3.plot(mult6_train_loss.index, mult6_train_loss.values, 's-', label='6배 증강', alpha=0.8, linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Training Loss (평균)', fontsize=12)
ax3.set_title('에포크별 평균 Training Loss', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 1.4 Valid loss comparison
ax4 = axes[1, 1]
mult4_valid_loss = mult4_log.groupby('epoch')['valid_loss'].mean()
mult6_valid_loss = mult6_log.groupby('epoch')['valid_loss'].mean()
ax4.plot(mult4_valid_loss.index, mult4_valid_loss.values, 'o-', label='4배 증강', alpha=0.8, linewidth=2)
ax4.plot(mult6_valid_loss.index, mult6_valid_loss.values, 's-', label='6배 증강', alpha=0.8, linewidth=2)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Validation Loss (평균)', fontsize=12)
ax4.set_title('에포크별 평균 Validation Loss', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'aug_mult_comparison.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ 시각화 저장: {os.path.join(output_dir, 'aug_mult_comparison.png')}")
plt.close()

# Save comparison summary
summary_df = pd.DataFrame({
    'fold': range(5),
    'mult4_best_f1': mult4_best.values,
    'mult6_best_f1': mult6_best.values,
    'improvement': (mult6_best - mult4_best).values,
    'mult4_epochs': mult4_epochs.values,
    'mult6_epochs': mult6_epochs.values,
})
summary_df.to_csv(os.path.join(output_dir, 'aug_comparison_summary.csv'), index=False)
print(f"✓ 요약 저장: {os.path.join(output_dir, 'aug_comparison_summary.csv')}")

print("\n" + "=" * 80)
print("### 결론 ###")
print("=" * 80)

if mult6_best.mean() > mult4_best.mean():
    print(f"✓ 6배 증강이 4배 증강보다 평균 {improvement:.4f} ({improvement_pct:.2f}%) 더 높은 F1 달성")
    print(f"✓ 6배 증강의 최고 F1: {mult6_best.max():.4f} (Fold {mult6_best.idxmax()})")
    print(f"✓ 6배 증강 권장")
else:
    print(f"✗ 4배 증강이 6배 증강보다 평균 {-improvement:.4f} ({-improvement_pct:.2f}%) 더 높은 F1 달성")
    print(f"✓ 4배 증강의 최고 F1: {mult4_best.max():.4f} (Fold {mult4_best.idxmax()})")
    print(f"✓ 4배 증강이 더 효율적")

print("\n학습 시간 효율성:")
print(f"  4배 증강: {mult4_epochs.sum()} epochs 소요")
print(f"  6배 증강: {mult6_epochs.sum()} epochs 소요")
print(f"  차이: {mult6_epochs.sum() - mult4_epochs.sum()} epochs")

print("\n" + "=" * 80)
