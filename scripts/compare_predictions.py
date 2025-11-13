#!/usr/bin/env python3
"""
mean vs max_conf 예측 비교 분석 스크립트
백업된 이전 결과(mean)와 새 결과(max_conf) 간 차이 분석
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def load_logits(path):
    """로짓 파일 로드"""
    data = torch.load(path, map_location='cpu', weights_only=False)
    logits = torch.tensor(data['logits']) if 'logits' in data else torch.tensor(data.get('logits', []))
    preds = data.get('predictions', torch.argmax(logits, dim=1).numpy())
    ids = data.get('img_ids', data.get('ids', []))
    return logits, preds, ids

def compare_predictions(old_path, new_path, output_csv=None):
    """두 예측 결과 비교"""
    print(f"Loading old (mean): {old_path}")
    logits_old, preds_old, ids_old = load_logits(old_path)
    
    print(f"Loading new (max_conf): {new_path}")
    logits_new, preds_new, ids_new = load_logits(new_path)
    
    # 확률 계산
    probs_old = torch.softmax(logits_old, dim=1).numpy()
    probs_new = torch.softmax(logits_new, dim=1).numpy()
    
    # 차이 분석
    changed = (preds_old != preds_new)
    n_changed = changed.sum()
    n_total = len(preds_old)
    
    print(f"\n=== 예측 변화 요약 ===")
    print(f"전체 샘플: {n_total}")
    print(f"예측 변경: {n_changed} ({100*n_changed/n_total:.2f}%)")
    print(f"예측 유지: {n_total - n_changed} ({100*(n_total-n_changed)/n_total:.2f}%)")
    
    # 변경된 샘플 상세 분석
    if n_changed > 0:
        changed_idx = np.where(changed)[0]
        
        # 확신도 변화
        conf_old = np.array([probs_old[i, preds_old[i]] for i in changed_idx])
        conf_new = np.array([probs_new[i, preds_new[i]] for i in changed_idx])
        conf_delta = conf_new - conf_old
        
        print(f"\n=== 변경된 샘플 확신도 분석 ===")
        print(f"확신도 증가: {(conf_delta > 0).sum()} 샘플")
        print(f"확신도 감소: {(conf_delta < 0).sum()} 샘플")
        print(f"평균 확신도 변화: {conf_delta.mean():.4f}")
        print(f"확신도 변화 범위: [{conf_delta.min():.4f}, {conf_delta.max():.4f}]")
        
        # 클래스별 변화
        print(f"\n=== 클래스별 변화 (from → to) ===")
        class_changes = {}
        for i in changed_idx:
            pair = (int(preds_old[i]), int(preds_new[i]))
            class_changes[pair] = class_changes.get(pair, 0) + 1
        
        for (old_cls, new_cls), count in sorted(class_changes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {old_cls} → {new_cls}: {count}회")
        
        # CSV 저장
        if output_csv:
            changed_df = pd.DataFrame({
                'id': [ids_old[i] if ids_old else f'sample_{i}' for i in changed_idx],
                'pred_old': preds_old[changed_idx],
                'pred_new': preds_new[changed_idx],
                'conf_old': conf_old,
                'conf_new': conf_new,
                'conf_delta': conf_delta
            })
            changed_df.to_csv(output_csv, index=False)
            print(f"\n변경 샘플 상세 저장 → {output_csv}")
    
    # 전체 확신도 분포 비교
    conf_all_old = probs_old.max(axis=1)
    conf_all_new = probs_new.max(axis=1)
    
    print(f"\n=== 전체 확신도 분포 비교 ===")
    print(f"Mean (old): mean={conf_all_old.mean():.4f}, std={conf_all_old.std():.4f}")
    print(f"Max_conf (new): mean={conf_all_new.mean():.4f}, std={conf_all_new.std():.4f}")
    print(f"확신도 변화량 평균: {(conf_all_new - conf_all_old).mean():.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', required=True, help='이전 결과 logits.pt 경로 (mean)')
    parser.add_argument('--new', required=True, help='새 결과 logits.pt 경로 (max_conf)')
    parser.add_argument('--output', default='outputs/prediction_comparison.csv', help='변경 샘플 저장 경로')
    args = parser.parse_args()
    
    compare_predictions(args.old, args.new, args.output)
