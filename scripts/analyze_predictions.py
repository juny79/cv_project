import pandas as pd
import numpy as np

# Load predictions
df1 = pd.read_csv('outputs/full_mult1/submission.csv')  # aug_multiplier=1 with TTA
df4 = pd.read_csv('outputs/full_mult4/submission.csv')  # aug_multiplier=4 with TTA

# Ensure same order by ID
df1 = df1.set_index('ID')
df4 = df4.set_index('ID')

# Find disagreements
disagree = df1[df1['target'] != df4['target']].copy()
disagree['mult1_pred'] = df1.loc[disagree.index, 'target']
disagree['mult4_pred'] = df4.loc[disagree.index, 'target']

# Analysis
print(f"\n=== Prediction Disagreement Analysis ===")
print(f"Total samples: {len(df1)}")
print(f"Number of disagreements: {len(disagree)} ({len(disagree)/len(df1)*100:.2f}%)")

print("\n=== Disagreement patterns ===")
patterns = pd.DataFrame({
    'mult1': disagree['mult1_pred'],
    'mult4': disagree['mult4_pred']
}).value_counts().reset_index()
patterns.columns = ['mult1_pred', 'mult4_pred', 'count']
print(patterns.head(10))

print("\n=== Per-class disagreement rates ===")
class_changes = []
for cls in range(17):
    pred1 = (df1['target'] == cls).sum()
    pred4 = (df4['target'] == cls).sum()
    changed = ((df1['target'] == cls) & (df4['target'] != cls)).sum()
    class_changes.append({
        'class': cls,
        'mult1_count': pred1,
        'mult4_count': pred4,
        'disagreements': changed,
        'change_rate': changed/pred1*100 if pred1 > 0 else 0
    })

changes_df = pd.DataFrame(class_changes)
print(changes_df.sort_values('change_rate', ascending=False))