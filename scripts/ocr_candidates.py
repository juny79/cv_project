#!/usr/bin/env python3
"""
Run Tesseract OCR on candidates from outputs/ocr_candidates.csv
"""
import os
import pandas as pd
from PIL import Image
import pytesseract

def main():
    # Load candidates
    cand_path = 'outputs/ocr_candidates.csv'
    if not os.path.exists(cand_path):
        print(f"Error: {cand_path} not found")
        return
    
    candidates = pd.read_csv(cand_path)
    print(f"Processing {len(candidates)} candidates...")
    
    # Load existing OCR CSV
    ocr_path = 'extern/ocr_text.csv'
    if os.path.exists(ocr_path):
        ocr_df = pd.read_csv(ocr_path)
        existing = dict(zip(ocr_df['id'].astype(str), ocr_df['text']))
        print(f"Loaded existing OCR CSV with {len(ocr_df)} entries")
    else:
        ocr_df = pd.DataFrame(columns=['id', 'text'])
        existing = {}
        print("Creating new OCR CSV")
    
    # Process images
    test_dir = 'data/test'
    updated = 0
    added = 0
    
    for i, row in candidates.iterrows():
        img_id = str(row['id'])
        
        # Find image file
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG', '']:
            p = os.path.join(test_dir, img_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            print(f"[{i+1}/{len(candidates)}] {img_id}: image not found")
            continue
        
        # Run OCR
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang='kor+eng', config='--psm 6')
            text = text.strip().replace('\n', ' ')
            
            # Update or add
            if img_id in existing:
                # Update in place
                ocr_df.loc[ocr_df['id'] == img_id, 'text'] = text
                updated += 1
                status = "updated"
            else:
                # Append new
                new_row = pd.DataFrame([{'id': img_id, 'text': text}])
                ocr_df = pd.concat([ocr_df, new_row], ignore_index=True)
                added += 1
                status = "added"
            
            print(f"[{i+1}/{len(candidates)}] {img_id} ({row['pair']}, delta={row['delta']:.3f}): {len(text)} chars [{status}]")
            if text:
                print(f"    Text: {text[:100]}")
        except Exception as e:
            print(f"[{i+1}/{len(candidates)}] {img_id}: OCR error - {e}")
    
    # Save
    os.makedirs(os.path.dirname(ocr_path) or '.', exist_ok=True)
    ocr_df.to_csv(ocr_path, index=False)
    print(f"\nSaved to {ocr_path}")
    print(f"Total entries: {len(ocr_df)}")
    print(f"Updated: {updated}, Added: {added}")
    
    # Summary
    non_empty = (ocr_df['text'].str.len() > 0).sum()
    total_chars = ocr_df['text'].str.len().sum()
    print(f"\nSummary:")
    print(f"  Non-empty: {non_empty}/{len(ocr_df)}")
    print(f"  Avg chars: {total_chars/len(ocr_df):.1f}")

if __name__ == '__main__':
    main()
