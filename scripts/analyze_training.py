import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the training log
log_path = '/root/cv_project/outputs/full_mult4_tta4/train_log.csv'
df = pd.read_csv(log_path)

# Create output directory for plots
plots_dir = '/root/cv_project/outputs/full_mult4_tta4/training_plots'
os.makedirs(plots_dir, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Create a comprehensive training visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Training Loss by Fold
ax1 = axes[0, 0]
for fold in df['fold'].unique():
    fold_data = df[df['fold'] == fold]
    ax1.plot(fold_data['epoch'], fold_data['train_loss'], 
             marker='o', label=f'Fold {fold}', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss by Fold (aug_mult=6)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Validation Loss by Fold
ax2 = axes[0, 1]
for fold in df['fold'].unique():
    fold_data = df[df['fold'] == fold]
    ax2.plot(fold_data['epoch'], fold_data['valid_loss'], 
             marker='s', label=f'Fold {fold}', alpha=0.7)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Loss', fontsize=12)
ax2.set_title('Validation Loss by Fold (aug_mult=6)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Training F1 Score by Fold
ax3 = axes[1, 0]
for fold in df['fold'].unique():
    fold_data = df[df['fold'] == fold]
    ax3.plot(fold_data['epoch'], fold_data['train_f1'], 
             marker='o', label=f'Fold {fold}', alpha=0.7)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Training F1 Score', fontsize=12)
ax3.set_title('Training F1 Score by Fold (aug_mult=6)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Validation F1 Score by Fold
ax4 = axes[1, 1]
for fold in df['fold'].unique():
    fold_data = df[df['fold'] == fold]
    ax4.plot(fold_data['epoch'], fold_data['valid_f1'], 
             marker='s', label=f'Fold {fold}', alpha=0.7)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Validation F1 Score', fontsize=12)
ax4.set_title('Validation F1 Score by Fold (aug_mult=6)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'training_overview.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(plots_dir, 'training_overview.png')}")
plt.close()

# Summary statistics
print("\n=== Training Summary (aug_multiplier=6) ===")
print(f"Total epochs trained: {len(df)}")
print(f"Number of folds: {df['fold'].nunique()}")

# Best F1 per fold
best_f1_per_fold = df.groupby('fold')['valid_f1'].max()
print("\n--- Best Validation F1 per Fold ---")
for fold, f1 in best_f1_per_fold.items():
    print(f"Fold {fold}: {f1:.4f}")
print(f"\nMean Best F1: {best_f1_per_fold.mean():.4f}")
print(f"Std Best F1: {best_f1_per_fold.std():.4f}")

# Final epoch stats
final_epoch = df.groupby('fold').last()
print("\n--- Final Epoch Stats ---")
print(f"Mean Valid F1 (final epoch): {final_epoch['valid_f1'].mean():.4f}")
print(f"Mean Valid Loss (final epoch): {final_epoch['valid_loss'].mean():.4f}")

# Save summary to CSV
summary = pd.DataFrame({
    'fold': best_f1_per_fold.index,
    'best_valid_f1': best_f1_per_fold.values,
})
summary_path = os.path.join(plots_dir, 'fold_summary.csv')
summary.to_csv(summary_path, index=False)
print(f"\nSaved summary: {summary_path}")

# Create comparison plot if mult4 data exists
mult4_log_path = '/root/cv_project/outputs/full_mult4/train_log.csv'
if os.path.exists(mult4_log_path):
    df_mult4 = pd.read_csv(mult4_log_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Get best F1 per fold for both
    best_mult4 = df_mult4.groupby('fold')['valid_f1'].max()
    best_mult6 = df.groupby('fold')['valid_f1'].max()
    
    x = range(len(best_mult4))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], best_mult4.values, width, 
           label='aug_mult=4', alpha=0.8, color='skyblue')
    ax.bar([i + width/2 for i in x], best_mult6.values, width, 
           label='aug_mult=6', alpha=0.8, color='coral')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Best Validation F1', fontsize=12)
    ax.set_title('Augmentation Multiplier Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(len(best_mult4))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (v4, v6) in enumerate(zip(best_mult4.values, best_mult6.values)):
        ax.text(i - width/2, v4 + 0.005, f'{v4:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v6 + 0.005, f'{v6:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'aug_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(plots_dir, 'aug_comparison.png')}")
    plt.close()
    
    print("\n=== Comparison: aug_mult=4 vs aug_mult=6 ===")
    print(f"Mean F1 (mult=4): {best_mult4.mean():.4f}")
    print(f"Mean F1 (mult=6): {best_mult6.mean():.4f}")
    print(f"Improvement: {(best_mult6.mean() - best_mult4.mean()):.4f} ({((best_mult6.mean() / best_mult4.mean() - 1) * 100):.2f}%)")

print("\n✓ Training analysis complete!")
