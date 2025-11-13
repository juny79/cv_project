# src/train_layout_pair.py
import os, json, argparse, numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification, get_linear_schedule_with_warmup

# --- OCR (Detector→Recognizer) : 기본 PaddleOCR 사용 ---
def run_ocr_paddle(img_path):
    # 설치가 되어있다는 전제: pip install paddleocr>=2.7.0
    from paddleocr import PaddleOCR
    # lazy singleton
    if not hasattr(run_ocr_paddle, "_ocr"):
        run_ocr_paddle._ocr = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)
    res = run_ocr_paddle._ocr.ocr(img_path, cls=True)
    words, boxes = [], []
    if res and res[0]:
        for item in res[0]:
            # PaddleOCR returns: (box, (text, conf))
            if len(item) == 2:
                box, (txt, conf) = item
            else:
                continue
            t = txt.strip() if isinstance(txt, str) else str(txt).strip()
            if not t:
                continue
            # box: 4 points (x,y). Flatten to min/max box
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            words.append(t)
            boxes.append([x0,y0,x1,y1])
    return words, boxes

def crop_title(img, ratio=0.20):
    h, w = img.size[1], img.size[0]  # PIL: (W,H)
    return img.crop((0, 0, w, int(h*ratio)))

class PairDataset(Dataset):
    def __init__(self, df, img_dir, cls_a, cls_b, processor, title_ratio=0.20, max_words=512, ocr_min_word_len=2):
        self.df = df[(df['target'].isin([cls_a, cls_b]))].reset_index(drop=True)
        self.img_dir = img_dir
        self.a, self.b = cls_a, cls_b
        self.processor = processor
        self.title_ratio = title_ratio
        self.max_words = max_words
        self.ocr_min_word_len = ocr_min_word_len

        # 캐시
        self.cache = {}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_id = str(r['id']) if 'id' in r else str(r[0])
        y = 0 if int(r['target']) == self.a else 1
        p = self._resolve_path(img_id)

        if p in self.cache:
            words, boxes, w, h = self.cache[p]
        else:
            words, boxes = run_ocr_paddle(p)
            # 최소 글자수 필터 및 상한
            wb = [(w, b) for (w, b) in zip(words, boxes) if len(w) >= self.ocr_min_word_len]
            if len(wb) > self.max_words:
                wb = wb[:self.max_words]
            words, boxes = [w for (w, _) in wb], [b for (_, b) in wb]
            with Image.open(p) as im:
                w, h = im.size
            self.cache[p] = (words, boxes, w, h)

        with Image.open(p).convert('RGB') as im:
            title_im = crop_title(im, self.title_ratio)

        # words/boxes를 image size 기준의 상대 bbox(0~1000 범위)로 정규화
        norm_boxes = []
        for x0,y0,x1,y1 in boxes:
            x0 = int(1000 * x0 / max(1,w)); x1 = int(1000 * x1 / max(1,w))
            y0 = int(1000 * y0 / max(1,h)); y1 = int(1000 * y1 / max(1,h))
            norm_boxes.append([x0,y0,x1,y1])

        # 전체 이미지만 사용 (타이틀 ROI는 추후 개선시 추가)
        enc = self.processor(images=im, text=words, boxes=norm_boxes, word_labels=None, return_tensors="pt")
        
        item = {k: v.squeeze(0) for k, v in enc.items()}  # batch dim 제거
        item['labels'] = torch.tensor(y, dtype=torch.long)
        return item

    def _resolve_path(self, stem):
        for ext in ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG',''):
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return p
        return os.path.join(self.img_dir, stem)

def train_pair(df, img_dir, a, b, out_dir, epochs=3, bs=4, lr=5e-5, title_ratio=0.20, max_words=512, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=2).to(device)

    ds = PairDataset(df, img_dir, a, b, processor, title_ratio=title_ratio, max_words=max_words)
    
    def collate_fn(batch):
        # Manual padding for batch
        batch_dict = {}
        for key in batch[0].keys():
            if key == 'labels':
                batch_dict[key] = torch.stack([item[key] for item in batch])
            else:
                # Pad sequences to max length in batch
                max_len = max(item[key].shape[0] if item[key].ndim > 0 else 1 for item in batch)
                padded = []
                for item in batch:
                    val = item[key]
                    if val.ndim == 0:
                        padded.append(val.unsqueeze(0))
                    elif val.shape[0] < max_len:
                        padding = torch.zeros(max_len - val.shape[0], *val.shape[1:], dtype=val.dtype)
                        padded.append(torch.cat([val, padding], dim=0))
                    else:
                        padded.append(val)
                batch_dict[key] = torch.stack(padded)
        return batch_dict
    
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)

    opt = AdamW(model.parameters(), lr=lr)
    total = epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    model.train()
    for ep in range(1, epochs+1):
        losses = []
        for batch in tqdm(dl, desc=f"pair {a}-{b} ep{ep}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            losses.append(loss.item())
        print(f"[pair {a}-{b}] ep{ep} loss={np.mean(losses):.4f}")
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', default='/root/cv_project/data/train.csv')
    ap.add_argument('--train_dir', default='/root/cv_project/data/train')
    ap.add_argument('--out_3_7',  default='/root/cv_project/extern/layout_pair_3_7')
    ap.add_argument('--out_4_14', default='/root/cv_project/extern/layout_pair_4_14')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-5)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    # 열 이름 표준화
    if 'target' not in df.columns:
        ycol = [c for c in df.columns if c.lower() in ['label','class']]
        df = df.rename(columns={ycol[0]: 'target'})
    if 'id' not in df.columns:
        idc = [c for c in df.columns if c.lower() in ['id','image_id','filename','image']]
        df = df.rename(columns={idc[0]: 'id'})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # (3,7)
    train_pair(df, args.train_dir, 3, 7, args.out_3_7, epochs=args.epochs, bs=args.bs, lr=args.lr, device=device)
    # (4,14)
    train_pair(df, args.train_dir, 4, 14, args.out_4_14, epochs=args.epochs, bs=args.bs, lr=args.lr, device=device)

if __name__ == "__main__":
    main()
