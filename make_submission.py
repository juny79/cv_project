
import pandas as pd, sys
pred = pd.read_csv(sys.argv[sys.argv.index('--pred')+1])
sample = pd.read_csv(sys.argv[sys.argv.index('--sample')+1])
assert (pred['ID'] == sample['ID']).all(), "ID order mismatch"
out = sys.argv[sys.argv.index('--out')+1]
pred.to_csv(out, index=False)
print(f'Saved submission: {out}')
