
import os, pandas as pd, cv2
from torch.utils.data import Dataset

class DocDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.has_target = 'target' in self.df.columns

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['ID'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(image=img)['image']
        if self.has_target: return img, int(row['target'])
        return img, -1
