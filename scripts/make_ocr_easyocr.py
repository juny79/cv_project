#!/usr/bin/env python
import os
from glob import glob
import argparse
import pandas as pd

# Avoid torch cuda init if not needed
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', default='/root/cv_project/data/test')
    ap.add_argument('--out_csv', default='/root/cv_project/extern/ocr_text.csv')
    ap.add_argument('--langs', nargs='+', default=['ko','en'])
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    try:
        import easyocr
    except Exception as e:
        raise SystemExit('easyocr not installed. Please pip install easyocr opencv-python-headless')

    rdr = easyocr.Reader(args.langs, gpu=False)

    paths = sorted(glob(os.path.join(args.img_dir, '*')))
    ids, texts = [], []
    for p in paths:
        try:
            res = rdr.readtext(p, detail=0, paragraph=True)
            txt = ' '.join([t.strip() for t in res if isinstance(t,str)])
        except Exception:
            txt = ''
        ids.append(os.path.basename(p).rsplit('.',1)[0])
        texts.append(txt)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame({'id':ids,'text':texts}).to_csv(args.out_csv, index=False)
    print('saved:', args.out_csv)

if __name__ == '__main__':
    main()
