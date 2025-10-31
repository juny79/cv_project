
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DF = pd.read_csv('data/datasets_fin/train.csv')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
DF['fold'] = -1
for i, (_, val_idx) in enumerate(skf.split(DF['ID'], DF['target'])):
    DF.loc[val_idx, 'fold'] = i
DF.to_csv('data/datasets_fin/train_folds.csv', index=False)
print(DF['fold'].value_counts())
