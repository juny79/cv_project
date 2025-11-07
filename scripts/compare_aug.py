#!/usr/bin/env python3
"""Run quick compare experiments for augmentation multiplier.

This script creates two temporary configs (multiplier=1 and multiplier=4),
runs `python -m src.train --config <cfg>` for each with reduced epochs and
folds (epochs=3, n_splits=3) to compare validation Best F1 per fold.

It parses stdout from train runs to extract per-fold Best F1 and prints a
summary table.
"""
import os
import yaml
import subprocess
import tempfile
import re
from datetime import datetime

BASE_CFG = 'configs/base.yaml'
RUN_DIR = 'outputs/aug_exp'

def load_cfg(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def write_cfg(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f:
        yaml.safe_dump(cfg, f)

def run_train(cfg_path):
    cmd = ['python','-m','src.train','--config', cfg_path]
    print('\nRunning:', ' '.join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + '\n' + p.stderr
    if p.returncode != 0:
        print('Train process failed with returncode', p.returncode)
        print(out)
        raise RuntimeError('Train run failed')
    return out

def parse_best_f1s(output):
    # matches lines like: "[Fold 0] Best F1: 0.9407"
    pattern = re.compile(r"Best F1:\s*([0-9]*\.?[0-9]+)")
    vals = [float(m.group(1)) for m in pattern.finditer(output)]
    return vals

def main():
    base = load_cfg(BASE_CFG)
    experiments = [1,4]
    summary = {}

    for m in experiments:
        cfg = dict(base)
        cfg['data'] = dict(base.get('data', {}))
        cfg['train'] = dict(base.get('train', {}))

        cfg['data']['aug_multiplier'] = m
        cfg['data']['n_splits'] = 3
        cfg['train']['epochs'] = 3

        out_dir = os.path.join(RUN_DIR, f'mult{m}')
        cfg['paths'] = dict(base.get('paths', {}))
        cfg['paths']['out_dir'] = out_dir

        cfg_path = os.path.join('configs', f'temp_aug_mult{m}.yaml')
        write_cfg(cfg, cfg_path)

        t0 = datetime.now()
        out = run_train(cfg_path)
        dt = datetime.now() - t0

        f1s = parse_best_f1s(out)
        mean_f1 = sum(f1s)/len(f1s) if f1s else None
        summary[m] = {'fold_f1s': f1s, 'mean_f1': mean_f1, 'time': str(dt)}

    # Print summary
    print('\n=== Augmentation comparison summary ===')
    for m, info in summary.items():
        print(f'\nMultiplier = {m}')
        print('Fold Best F1s:', info['fold_f1s'])
        print('Mean F1:', info['mean_f1'])
        print('Run time:', info['time'])

if __name__ == '__main__':
    main()
