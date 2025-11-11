"""
Comprehensive confusion analysis for document classification
- TTA=4 validation inference
- Probability distribution sampling for class pairs (3,7)
- Threshold sweep curves for postprocessing parameters
- Visualizations: heatmaps, histograms, ROC-style curves
"""
import os, yaml, timm, torch
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from src.transforms import get_valid_transforms

COMMON_EXTS = ('.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','')

def load_cfg(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def resolve_image_path(img_dir, stem):
    s = str(stem)
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, s + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(img_dir, s)
    if os.path.exists(p): return p
    raise FileNotFoundError(f"image not found for {stem} under {img_dir}")

class SimpleDS(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = transform
        idc = [c for c in self.df.columns if c.lower() in ['id','image_id','filename','image']]
        self.id_col = idc[0] if idc else self.df.columns[0]
        ycs = [c for c in self.df.columns if c.lower() in ['label','target','class']]
        self.y_col = ycs[0] if ycs else self.df.columns[-1]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_image_path(self.img_dir, r[self.id_col])
        img = np.array(Image.open(p).convert('RGB'))
        x = self.tf(image=img)['image']
        y = int(r[self.y_col])
        img_id = r[self.id_col]
        return x, y, img_id

@torch.no_grad()
def run_tta_validation(cfg_path, tta=4):
    """Run TTA inference on validation folds and collect logits + predictions"""
    cfg = load_cfg(cfg_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = cfg['paths']; data = cfg['data']

    df = pd.read_csv(paths['train_csv'])
    ycol = [c for c in df.columns if c.lower() in ['label','target','class']]
    ycol = ycol[0] if ycol else df.columns[-1]
    y = df[ycol].astype(int).values
    num_classes = len(np.unique(y))

    tfm = get_valid_transforms(int(data['img_size']))
    skf = StratifiedKFold(n_splits=int(data.get('n_splits',5)), shuffle=True, random_state=cfg.get('seed',42))

    model = timm.create_model(cfg['model']['name'], pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    use_amp = True if device.startswith('cuda') else False

    def _four_rot_logits(xb):
        outs = []
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outs.append(model(xb))
            outs.append(model(torch.rot90(xb, 1, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 2, dims=[2,3])))
            outs.append(model(torch.rot90(xb, 3, dims=[2,3])))
        return outs

    all_data = []  # store: img_id, true_label, probs (num_classes,)

    for fold_id, (_, va_idx) in enumerate(skf.split(df, y)):
        ck = os.path.join(paths['out_dir'], f'fold{fold_id}', 'best.pt')
        if not os.path.exists(ck):
            print(f"[Warn] Missing checkpoint for fold{fold_id}, skipping")
            continue
        w = torch.load(ck, map_location=device, weights_only=False)
        try:
            model.load_state_dict(w['model'], strict=True)
        except RuntimeError:
            model.load_state_dict(w['model'], strict=False)
        model.eval()

        df_va = df.iloc[va_idx].reset_index(drop=True)
        ds_va = SimpleDS(df_va, paths['train_dir'], tfm)
        dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0, pin_memory=device.startswith('cuda'))

        print(f"[TTA Val] fold={fold_id} batches={len(dl_va)}")
        for bi, (xb, yb, ids) in enumerate(dl_va):
            xb = xb.to(device)
            if int(tta) == 4:
                cand = [z.detach().cpu() for z in _four_rot_logits(xb)]
                probs = [torch.softmax(z, dim=1) for z in cand]
                maxp = torch.stack([p.max(dim=1).values for p in probs], dim=0)
                best_idx = torch.argmax(maxp, dim=0)
                pick = torch.stack([cand[r][i] for i, r in enumerate(best_idx.tolist())], dim=0)
                logits_batch = pick
            else:
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    logits_batch = model(xb).detach().cpu()
            
            probs_batch = torch.softmax(logits_batch, dim=1).numpy()
            for i in range(len(yb)):
                all_data.append({
                    'img_id': ids[i],
                    'true': int(yb[i]),
                    'probs': probs_batch[i],
                })
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(dl_va)}")

    return all_data, num_classes

def apply_postprocess(probs, cls_a, cls_b, delta_thr, conf_thr, prefer='higher'):
    """Apply postprocessing rule for class pair"""
    pred = np.argmax(probs)
    pa, pb = probs[cls_a], probs[cls_b]
    if abs(pa - pb) < delta_thr and max(pa, pb) > conf_thr:
        if prefer == 'higher':
            pred = cls_a if pa >= pb else cls_b
        else:
            try:
                pred = int(prefer)
            except:
                pred = cls_a if pa >= pb else cls_b
    return pred

def threshold_sweep(all_data, num_classes, cls_a=3, cls_b=7):
    """Sweep delta_threshold and confidence_threshold"""
    delta_vals = np.linspace(0.01, 0.15, 15)
    conf_vals = np.linspace(0.4, 0.7, 13)
    
    results = []
    y_true = np.array([d['true'] for d in all_data])
    probs_arr = np.array([d['probs'] for d in all_data])
    
    # Baseline (no postprocess)
    y_pred_base = np.argmax(probs_arr, axis=1)
    f1_base = f1_score(y_true, y_pred_base, average='macro')
    cm_base = confusion_matrix(y_true, y_pred_base, labels=list(range(num_classes)))
    a_to_b_base = int(cm_base[cls_a, cls_b])
    b_to_a_base = int(cm_base[cls_b, cls_a])
    
    print(f"[Sweep] Baseline F1={f1_base:.4f}, {cls_a}→{cls_b}={a_to_b_base}, {cls_b}→{cls_a}={b_to_a_base}")
    
    for delta in delta_vals:
        for conf in conf_vals:
            y_pred = []
            for d in all_data:
                pred = apply_postprocess(d['probs'], cls_a, cls_b, delta, conf, 'higher')
                y_pred.append(pred)
            y_pred = np.array(y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            a_to_b = int(cm[cls_a, cls_b])
            b_to_a = int(cm[cls_b, cls_a])
            
            results.append({
                'delta_threshold': delta,
                'confidence_threshold': conf,
                'f1_macro': f1,
                f'{cls_a}_to_{cls_b}': a_to_b,
                f'{cls_b}_to_{cls_a}': b_to_a,
                'total_confusion': a_to_b + b_to_a,
            })
    
    return pd.DataFrame(results), (f1_base, a_to_b_base, b_to_a_base)

def sample_probability_cases(all_data, cls_a=3, cls_b=7, top_k=10):
    """Sample top/bottom cases for class pair based on probability delta"""
    subset = [d for d in all_data if d['true'] in [cls_a, cls_b]]
    
    deltas = []
    for d in subset:
        pa, pb = d['probs'][cls_a], d['probs'][cls_b]
        delta = abs(pa - pb)
        pred = np.argmax(d['probs'])
        correct = (pred == d['true'])
        deltas.append({
            'img_id': d['img_id'],
            'true': d['true'],
            'pred': pred,
            'prob_a': pa,
            'prob_b': pb,
            'delta': delta,
            'correct': correct,
        })
    
    df_deltas = pd.DataFrame(deltas)
    
    # Ensure numeric types for sorting
    df_deltas['delta'] = df_deltas['delta'].astype(float)
    df_deltas['prob_a'] = df_deltas['prob_a'].astype(float)
    df_deltas['prob_b'] = df_deltas['prob_b'].astype(float)
    
    # Top: largest delta (confident)
    top = df_deltas.nlargest(top_k, 'delta')
    # Bottom: smallest delta (confused)
    bottom = df_deltas.nsmallest(top_k, 'delta')
    
    return top, bottom, df_deltas

def visualize_results(sweep_df, baseline, samples_df, cls_a, cls_b, out_dir):
    """Generate comprehensive visualizations"""
    os.makedirs(out_dir, exist_ok=True)
    
    f1_base, a_to_b_base, b_to_a_base = baseline
    
    # 1. Heatmap: F1 vs delta/conf
    pivot = sweep_df.pivot_table(values='f1_macro', index='confidence_threshold', columns='delta_threshold')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', vmin=pivot.min().min(), vmax=pivot.max().max())
    plt.title(f'Macro F1 Score Heatmap (Baseline={f1_base:.4f})')
    plt.xlabel('Delta Threshold')
    plt.ylabel('Confidence Threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'f1_heatmap.png'), dpi=150)
    plt.close()
    print(f"[OUT] Saved f1_heatmap.png")
    
    # 2. Confusion reduction heatmap
    pivot_conf = sweep_df.pivot_table(values='total_confusion', index='confidence_threshold', columns='delta_threshold')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_conf, annot=True, fmt='.0f', cmap='RdYlGn_r', vmin=pivot_conf.min().min(), vmax=pivot_conf.max().max())
    plt.title(f'Total {cls_a}↔{cls_b} Confusions (Baseline={a_to_b_base + b_to_a_base})')
    plt.xlabel('Delta Threshold')
    plt.ylabel('Confidence Threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_heatmap.png'), dpi=150)
    plt.close()
    print(f"[OUT] Saved confusion_heatmap.png")
    
    # 3. Line plot: F1 vs delta at fixed conf
    fixed_confs = [0.45, 0.50, 0.55, 0.60, 0.65]
    plt.figure(figsize=(10, 6))
    for conf in fixed_confs:
        subset = sweep_df[sweep_df['confidence_threshold'] == conf]
        if len(subset) > 0:
            plt.plot(subset['delta_threshold'], subset['f1_macro'], marker='o', label=f'conf={conf:.2f}')
    plt.axhline(f1_base, color='red', linestyle='--', label='Baseline (no postproc)')
    plt.xlabel('Delta Threshold')
    plt.ylabel('Macro F1')
    plt.title(f'F1 vs Delta Threshold at Fixed Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'f1_vs_delta.png'), dpi=150)
    plt.close()
    print(f"[OUT] Saved f1_vs_delta.png")
    
    # 4. Probability delta distribution
    plt.figure(figsize=(10, 6))
    for true_cls in [cls_a, cls_b]:
        subset = samples_df[samples_df['true'] == true_cls]
        plt.hist(subset['delta'], bins=30, alpha=0.5, label=f'True class {true_cls}', edgecolor='black')
    plt.axvline(0.04, color='red', linestyle='--', label='Current delta_thr=0.04')
    plt.xlabel(f'|P(class {cls_a}) - P(class {cls_b})|')
    plt.ylabel('Count')
    plt.title(f'Probability Delta Distribution for Classes {cls_a} and {cls_b}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prob_delta_distribution.png'), dpi=150)
    plt.close()
    print(f"[OUT] Saved prob_delta_distribution.png")
    
    # 5. Confusion by correctness
    correct_df = samples_df[samples_df['correct'] == True]
    incorrect_df = samples_df[samples_df['correct'] == False]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(correct_df['delta'], bins=25, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_title(f'Correct Predictions (n={len(correct_df)})')
    axes[0].set_xlabel('Probability Delta')
    axes[0].set_ylabel('Count')
    axes[0].axvline(0.04, color='red', linestyle='--', label='delta_thr=0.04')
    axes[0].legend()
    
    axes[1].hist(incorrect_df['delta'], bins=25, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_title(f'Incorrect Predictions (n={len(incorrect_df)})')
    axes[1].set_xlabel('Probability Delta')
    axes[1].set_ylabel('Count')
    axes[1].axvline(0.04, color='red', linestyle='--', label='delta_thr=0.04')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correctness_delta_distribution.png'), dpi=150)
    plt.close()
    print(f"[OUT] Saved correctness_delta_distribution.png")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--tta', type=int, default=4)
    args = ap.parse_args()
    
    cfg = load_cfg(args.config)
    out_dir = os.path.join(cfg['paths']['out_dir'], 'confusion_analysis')
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE CONFUSION ANALYSIS")
    print("="*60)
    
    # 1. Run TTA validation
    print("\n[1/5] Running TTA=4 validation inference...")
    all_data, num_classes = run_tta_validation(args.config, tta=args.tta)
    print(f"  Collected {len(all_data)} validation samples")
    
    # 2. Threshold sweep
    print("\n[2/5] Sweeping postprocess thresholds...")
    sweep_df, baseline = threshold_sweep(all_data, num_classes, cls_a=3, cls_b=7)
    sweep_csv = os.path.join(out_dir, 'threshold_sweep.csv')
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"  Saved sweep results → {sweep_csv}")
    
    # Find best config
    best_row = sweep_df.loc[sweep_df['f1_macro'].idxmax()]
    print(f"\n  Best config: delta={best_row['delta_threshold']:.3f}, conf={best_row['confidence_threshold']:.2f}")
    print(f"    F1={best_row['f1_macro']:.4f} (baseline={baseline[0]:.4f})")
    print(f"    3→7={best_row['3_to_7']:.0f}, 7→3={best_row['7_to_3']:.0f} (baseline: {baseline[1]}, {baseline[2]})")
    
    # 3. Sample probability cases
    print("\n[3/5] Sampling probability cases for classes 3 and 7...")
    top, bottom, samples_df = sample_probability_cases(all_data, cls_a=3, cls_b=7, top_k=10)
    
    top_csv = os.path.join(out_dir, 'top_confident_cases.csv')
    bottom_csv = os.path.join(out_dir, 'bottom_confused_cases.csv')
    top.to_csv(top_csv, index=False)
    bottom.to_csv(bottom_csv, index=False)
    print(f"  Saved top confident cases → {top_csv}")
    print(f"  Saved bottom confused cases → {bottom_csv}")
    
    # 4. Generate visualizations
    print("\n[4/5] Generating visualizations...")
    visualize_results(sweep_df, baseline, samples_df, cls_a=3, cls_b=7, out_dir=out_dir)
    
    # 5. Summary report
    print("\n[5/5] Writing summary report...")
    report_path = os.path.join(out_dir, 'ANALYSIS_SUMMARY.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CONFUSION ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total validation samples: {len(all_data)}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"TTA mode: {args.tta}\n\n")
        
        f.write("BASELINE (no postprocessing):\n")
        f.write(f"  Macro F1: {baseline[0]:.4f}\n")
        f.write(f"  Class 3 → 7: {baseline[1]} misclassifications\n")
        f.write(f"  Class 7 → 3: {baseline[2]} misclassifications\n")
        f.write(f"  Total 3↔7 confusion: {baseline[1] + baseline[2]}\n\n")
        
        f.write("BEST POSTPROCESS CONFIG:\n")
        f.write(f"  Delta threshold: {best_row['delta_threshold']:.3f}\n")
        f.write(f"  Confidence threshold: {best_row['confidence_threshold']:.2f}\n")
        f.write(f"  Macro F1: {best_row['f1_macro']:.4f} (Δ={best_row['f1_macro']-baseline[0]:+.4f})\n")
        f.write(f"  Class 3 → 7: {int(best_row['3_to_7'])} (Δ={int(best_row['3_to_7'])-baseline[1]:+d})\n")
        f.write(f"  Class 7 → 3: {int(best_row['7_to_3'])} (Δ={int(best_row['7_to_3'])-baseline[2]:+d})\n")
        f.write(f"  Total 3↔7 confusion: {int(best_row['total_confusion'])} (Δ={int(best_row['total_confusion'])-(baseline[1]+baseline[2]):+d})\n\n")
        
        f.write("PROBABILITY ANALYSIS:\n")
        f.write(f"  Mean delta (correct predictions): {samples_df[samples_df['correct']==True]['delta'].mean():.4f}\n")
        f.write(f"  Mean delta (incorrect predictions): {samples_df[samples_df['correct']==False]['delta'].mean():.4f}\n")
        f.write(f"  Median delta (all): {samples_df['delta'].median():.4f}\n\n")
        
        f.write("TOP 5 CONFUSED CASES (smallest delta):\n")
        for i, row in bottom.head(5).iterrows():
            f.write(f"  {row['img_id']}: true={row['true']}, pred={row['pred']}, "
                   f"P(3)={row['prob_a']:.4f}, P(7)={row['prob_b']:.4f}, delta={row['delta']:.4f}\n")
        
        f.write("\nOUTPUT FILES:\n")
        f.write(f"  - {sweep_csv}\n")
        f.write(f"  - {top_csv}\n")
        f.write(f"  - {bottom_csv}\n")
        f.write(f"  - {out_dir}/f1_heatmap.png\n")
        f.write(f"  - {out_dir}/confusion_heatmap.png\n")
        f.write(f"  - {out_dir}/f1_vs_delta.png\n")
        f.write(f"  - {out_dir}/prob_delta_distribution.png\n")
        f.write(f"  - {out_dir}/correctness_delta_distribution.png\n")
    
    print(f"  Saved summary → {report_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {out_dir}")

if __name__ == '__main__':
    main()
