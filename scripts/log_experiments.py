#!/usr/bin/env python3
"""
Experiment logging utility for CV project.
Collects experiment configs, fold results, and runtime metrics into CSV.
"""
import os
import re
import csv
import yaml
import json
from datetime import datetime
from collections import defaultdict

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def extract_exp_config(cfg):
    """Extract key experiment parameters from config."""
    return {
        'aug_multiplier': cfg.get('data', {}).get('aug_multiplier', 1),
        'img_size': cfg.get('data', {}).get('img_size', 640),
        'model': cfg.get('model', {}).get('name', ''),
        'batch_size': cfg.get('train', {}).get('batch_size', 32),
        'epochs': cfg.get('train', {}).get('epochs', 12),
        'lr': cfg.get('train', {}).get('lr', 3e-4),
        'mixup_alpha': cfg.get('train', {}).get('mixup_alpha', 0),
        'label_smoothing': cfg.get('model', {}).get('label_smoothing', 0),
    }

def parse_train_log(log_text):
    """Parse training log to extract metrics per fold."""
    metrics = defaultdict(list)
    
    # Extract Best F1 scores
    f1_pattern = re.compile(r"\[Fold (\d+)\] Best F1: ([0-9]*\.?[0-9]+)")
    for fold, f1 in f1_pattern.findall(log_text):
        metrics['fold'].append(int(fold))
        metrics['best_f1'].append(float(f1))
    
    # Extract early stop info
    early_pattern = re.compile(r"\[Fold (\d+)\] Early stop\.")
    for fold in early_pattern.findall(log_text):
        metrics['early_stopped'].append(int(fold))
    
    return metrics

def log_experiment(cfg_path, log_text, out_csv='outputs/experiments.csv'):
    """Log experiment config and results to CSV."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    # Load and extract config
    cfg = load_yaml(cfg_path)
    exp_cfg = extract_exp_config(cfg)
    
    # Parse metrics
    metrics = parse_train_log(log_text)
    
    # Prepare row data
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_file': cfg_path,
        'config': json.dumps(exp_cfg),
        'num_folds': len(metrics['fold']),
        'mean_f1': sum(metrics['best_f1']) / len(metrics['best_f1']),
        'min_f1': min(metrics['best_f1']),
        'max_f1': max(metrics['best_f1']),
        'early_stops': len(metrics.get('early_stopped', [])),
        'fold_f1s': json.dumps(list(zip(metrics['fold'], metrics['best_f1']))),
    }
    
    # Write/append to CSV
    file_exists = os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"\nLogged experiment results to {out_csv}")
    print(f"Config: {exp_cfg}")
    print(f"Mean F1: {row['mean_f1']:.4f} (min={row['min_f1']:.4f}, max={row['max_f1']:.4f})")
    print(f"Fold F1s: {metrics['best_f1']}")
    if metrics.get('early_stopped'):
        print(f"Early stopped folds: {metrics['early_stopped']}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python log_experiments.py <config.yaml> <train_log.txt>")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    with open(sys.argv[2], 'r') as f:
        log_text = f.read()
    
    log_experiment(cfg_path, log_text)