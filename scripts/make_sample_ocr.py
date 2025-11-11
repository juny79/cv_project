#!/usr/bin/env python3
"""
Extract near-boundary samples for pairs (3,7) and (4,14), then run EasyOCR on ~100 samples.
Appends results to extern/ocr_text.csv with format: id,text
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import easyocr

def main():
    # Load logits
    logits_path = 'outputs/full_mult4/predict_logits.pt'
    if not os.path.exists(logits_path):
        print(f"Error: {logits_path} not found")
        sys.exit(1)
    
    print("Loading logits...")
    data = torch.load(logits_path, map_location='cpu', weights_only=False)
    logits = torch.tensor(data['logits'])
    probs = torch.softmax(logits, dim=1).numpy()
    img_ids = data['img_ids']
    
    # Find near-boundary samples for pairs (3,7) and (4,14)
    pairs = [(3, 7), (4, 14)]
    delta_thr = 0.15  # relaxed to get ~100 samples
    conf_thr = 0.45
    
    candidates = []
    for a, b in pairs:
        pa = probs[:, a]
        pb = probs[:, b]
        delta = np.abs(pa - pb)
        max_conf = np.maximum(pa, pb)
        
        mask = (delta < delta_thr) & (max_conf > conf_thr)
        idx = np.where(mask)[0]
        
        for i in idx:
            candidates.append({
                'idx': int(i),
                'id': str(img_ids[i]),
                'pair': f'{a}-{b}',
                'delta': float(delta[i]),
                'conf': float(max_conf[i])
            })
        print(f"Pair ({a},{b}): {len(idx)} candidates")
    
    # Sort by delta (most ambiguous first) and take top 100
    candidates = sorted(candidates, key=lambda x: x['delta'])[:100]
    print(f"\nSelected {len(candidates)} samples for OCR")
    
    # Initialize EasyOCR
    print("Initializing EasyOCR (Korean + English)...")
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    
    # Load or create OCR CSV
    ocr_path = 'extern/ocr_text.csv'
    if os.path.exists(ocr_path):
        ocr_df = pd.read_csv(ocr_path)
        existing = set(ocr_df['id'].astype(str))
        print(f"Loaded existing OCR CSV with {len(ocr_df)} entries")
    else:
        ocr_df = pd.DataFrame(columns=['id', 'text'])
        existing = set()
        print("Creating new OCR CSV")
    
    # Process images
    test_dir = 'data/test'
    ocr_results = []
    
    for i, cand in enumerate(candidates):
        img_id = cand['id']
        if img_id in existing:
            print(f"[{i+1}/{len(candidates)}] {img_id}: already in CSV, skipping")
            continue
        
        # Find image file
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG', '']:
            p = os.path.join(test_dir, img_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            print(f"[{i+1}/{len(candidates)}] {img_id}: image not found, skipping")
            continue
        
        # Run OCR
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            result = reader.readtext(img, detail=0)  # detail=0 returns list of strings
            text = ' '.join(result)
            ocr_results.append({'id': img_id, 'text': text})
            print(f"[{i+1}/{len(candidates)}] {img_id} ({cand['pair']}, delta={cand['delta']:.3f}): {len(text)} chars")
        except Exception as e:
            print(f"[{i+1}/{len(candidates)}] {img_id}: OCR error - {e}")
            ocr_results.append({'id': img_id, 'text': ''})
    
    # Append to existing CSV
    if ocr_results:
        new_df = pd.DataFrame(ocr_results)
        ocr_df = pd.concat([ocr_df, new_df], ignore_index=True)
        os.makedirs(os.path.dirname(ocr_path) or '.', exist_ok=True)
        ocr_df.to_csv(ocr_path, index=False)
        print(f"\nSaved {len(ocr_results)} new OCR results to {ocr_path}")
        print(f"Total entries: {len(ocr_df)}")
    else:
        print("\nNo new OCR results to save")
    
    # Summary stats
    if ocr_results:
        total_chars = sum(len(r['text']) for r in ocr_results)
        non_empty = sum(1 for r in ocr_results if r['text'])
        print(f"\nSummary:")
        print(f"  Non-empty: {non_empty}/{len(ocr_results)}")
        print(f"  Avg chars: {total_chars/len(ocr_results):.1f}")

if __name__ == '__main__':
    main()
