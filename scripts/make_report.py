import os
import glob
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
try:
    import seaborn as sns
    sns.set(style='white')
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
import yaml
from datetime import datetime
import html

# 클래스 이름 매핑 (자연어)
CLASS_NAMES = {
    0: "정상",
    1: "경도 질환",
    2: "중등도 질환",
    3: "중증 질환 A",
    4: "중증 질환 B",
    5: "합병증 A",
    6: "합병증 B",
    7: "급성 상태",
    8: "만성 상태 A",
    9: "만성 상태 B",
    10: "이상 소견 A",
    11: "이상 소견 B",
    12: "특이 소견 A",
    13: "특이 소견 B",
    14: "위험 징후",
    15: "응급 상황",
    16: "기타"
}

def format_float(x, precision=4):
    """Format float with consistent precision"""
    try:
        return f"{float(x):.{precision}f}"
    except (ValueError, TypeError):
        return str(x)

PLOTS_DIR = 'outputs/plots'
PDF_PATH = os.path.join(PLOTS_DIR, 'report.pdf')
HTML_PATH = os.path.join(PLOTS_DIR, 'report.html')


def make_cover(summary_text, size=(1200, 800), fontsize=24):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', fontsize)
    except Exception:
        font = ImageFont.load_default()

    # Title
    draw.text((40, 40), 'TTA & Class Distribution Report', fill='black', font=font)

    # Summary block
    y = 120
    for line in summary_text.split('\n'):
        draw.text((40, y), line, fill='black', font=font)
        y += fontsize + 6

    return img


def collect_plots():
    # Collect class plots first (class_00 ... class_16), then other PNGs
    class_imgs = sorted(glob.glob(os.path.join(PLOTS_DIR, 'class_*.png')))
    other_imgs = sorted(glob.glob(os.path.join(PLOTS_DIR, '*.png')))
    # remove class imgs from other
    other_imgs = [p for p in other_imgs if p not in class_imgs]
    return class_imgs + other_imgs


def generate_class_description(class_id, stats):
    """Generate natural language description for a class based on stats"""
    desc = []
    name = CLASS_NAMES.get(class_id, f"클래스 {class_id}")
    
    # Count changes
    changed_from = stats.get('changed_from_no_tta', 0)
    changed_to = stats.get('changed_to_tta', 0)
    count_no_tta = stats.get('count_no_tta', 0)
    count_tta = stats.get('count_tta', 0)
    
    desc.append(f"{name}:")
    
    # Overall trend
    if count_tta > count_no_tta:
        delta = count_tta - count_no_tta
        desc.append(f"TTA 적용 후 {delta}개 증가 ({count_no_tta}→{count_tta})")
    elif count_tta < count_no_tta:
        delta = count_no_tta - count_tta
        desc.append(f"TTA 적용 후 {delta}개 감소 ({count_no_tta}→{count_tta})")
    
    # Confidence changes
    conf_no_tta = stats.get('mean_conf_no_tta', 0)
    conf_tta = stats.get('mean_conf_tta', 0)
    if conf_tta > conf_no_tta:
        desc.append(f"평균 신뢰도 {format_float(conf_tta-conf_no_tta)} 상승")
    elif conf_tta < conf_no_tta:
        desc.append(f"평균 신뢰도 {format_float(conf_no_tta-conf_tta)} 하락")
        
    # Changes summary
    if changed_from > 0 or changed_to > 0:
        desc.append(f"다른 클래스에서 유입: {changed_to}개, 다른 클래스로 이동: {changed_from}개")
    
    return ' '.join(desc)

def read_summaries():
    # Read any summary CSVs present
    csvs = glob.glob(os.path.join(PLOTS_DIR, '*_summary.csv')) + glob.glob(os.path.join(PLOTS_DIR, '*_per_class.csv'))
    summaries = {}
    for c in csvs:
        name = os.path.basename(c)
        try:
            df = pd.read_csv(c)
            # Format numeric columns
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].apply(format_float)
            summaries[name] = df
        except Exception:
            summaries[name] = None
    return summaries


def build_pdf():
    ensure = os.path.exists(PLOTS_DIR)
    if not ensure:
        raise FileNotFoundError(f"Plots dir not found: {PLOTS_DIR}")

    # Before collecting plots, generate confusion matrices and per-class table images
    def safe_load_preds(path):
        p = os.path.join(path, 'predict_logits.pt')
        if not os.path.exists(p):
            return None
        try:
            d = torch.load(p, weights_only=False)
        except Exception:
            d = torch.load(p, map_location='cpu')
        preds = np.array(d.get('predictions'))
        img_ids = np.array(d.get('img_ids', np.arange(len(preds))))
        return {'preds': preds, 'img_ids': img_ids}

    def make_confusion_image(pred_true, pred_pred, out_png, n_classes=None, title='Confusion'):
        if len(pred_true) != len(pred_pred):
            return False
        if n_classes is None:
            n_classes = int(max(pred_true.max(), pred_pred.max()) + 1)

        # Compute F1 scores
        f1_per_class = f1_score(pred_true, pred_pred, average=None, zero_division=0)
        mean_f1 = f1_score(pred_true, pred_pred, average='macro', zero_division=0)
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(pred_true, pred_pred):
            cm[int(t), int(p)] += 1
        # normalize rows
        with np.errstate(all='ignore'):
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16,6), gridspec_kw={'width_ratios': [2, 1]})
        if HAS_SEABORN:
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar_kws={'label':'Proportion'})
        else:
            im = ax.imshow(cm_norm, cmap='Blues')
            fig.colorbar(im, ax=ax)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha='center', va='center', color='k')
        
        # Add class names to ticks
        class_labels = [CLASS_NAMES.get(i, str(i)) for i in range(n_classes)]
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Reference')
        ax.set_title(f"{title}\nMean F1: {format_float(mean_f1)}")
        
        # Plot per-class F1 scores
        ax2.bar(range(n_classes), f1_per_class)
        ax2.set_xticks(range(n_classes))
        ax2.set_xticklabels(class_labels, rotation=45, ha='right')
        ax2.set_title('Per-class F1 scores')
        ax2.set_ylim(0, 1)
        for i, f1 in enumerate(f1_per_class):
            ax2.text(i, f1+0.02, format_float(f1), ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return {'mean_f1': mean_f1, 'f1_per_class': f1_per_class.tolist()}

    def df_to_image(df, out_png, title=None):
        # Format numeric columns
        df_formatted = df.copy()
        for col in df.select_dtypes(include=['float64']).columns:
            df_formatted[col] = df_formatted[col].apply(format_float)
        
        rows, cols = df_formatted.shape
        max_rows = 40  # limit rows for readability
        disp_df = df_formatted.head(max_rows)
        
        fig, ax = plt.subplots(figsize=(12, min(0.5 + 0.4*len(disp_df), 20)))
        ax.axis('off')
        if title:
            ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        
        # Style the table
        table = ax.table(cellText=disp_df.values.tolist(), 
                        colLabels=disp_df.columns, 
                        loc='center', 
                        cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Add alternating row colors and style header
        for i in range(len(disp_df)+1):
            for j in range(len(disp_df.columns)):
                cell = table._cells[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#e0e0e0')
                    cell.set_text_props(weight='bold')
                elif i % 2:  # Alternating rows
                    cell.set_facecolor('#f5f5f5')
        
        table.scale(1.2, 1.5)
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return True

    # Try to generate extra artifacts for mult1 and mult4 TTA comparisons
    preds_mult1 = safe_load_preds(os.path.join('outputs', 'full_mult1'))
    preds_mult1_tta = safe_load_preds(os.path.join('outputs', 'full_mult1_tta4'))
    preds_mult4 = safe_load_preds(os.path.join('outputs', 'full_mult4'))
    preds_mult4_tta = safe_load_preds(os.path.join('outputs', 'full_mult4_tta4'))

    # Generate TTA F1 score comparison table
    f1_rows = []
    if preds_mult1 and preds_mult1_tta:
        f1_no_tta = f1_score(preds_mult1['preds'], preds_mult1_tta['preds'], average='macro')
        f1_rows.append({
            'model': 'aug_mult=1',
            'mean_f1_no_tta_vs_tta': format_float(f1_no_tta)
        })
    if preds_mult4 and preds_mult4_tta:
        f1_no_tta = f1_score(preds_mult4['preds'], preds_mult4_tta['preds'], average='macro')
        f1_rows.append({
            'model': 'aug_mult=4',
            'mean_f1_no_tta_vs_tta': format_float(f1_no_tta)
        })
    
    if f1_rows:
        f1_df = pd.DataFrame(f1_rows)
        f1_img = os.path.join(PLOTS_DIR, 'tta_f1_comparison.png')
        df_to_image(f1_df, f1_img, title='TTA F1 Score Comparison')

    # Confusion between no-TTA and TTA per model
    if preds_mult1 is not None and preds_mult1_tta is not None:
        out_png = os.path.join(PLOTS_DIR, 'confusion_mult1_no_vs_tta.png')
        make_confusion_image(preds_mult1['preds'], preds_mult1_tta['preds'], 
                           out_png, title='aug_mult=1: no-TTA vs TTA')
    if preds_mult4 is not None and preds_mult4_tta is not None:
        out_png = os.path.join(PLOTS_DIR, 'confusion_mult4_no_vs_tta.png')
        make_confusion_image(preds_mult4['preds'], preds_mult4_tta['preds'], 
                           out_png, title='aug_mult=4: no-TTA vs TTA')

    # Inter-model confusion (mult1 no-TTA vs mult4 no-TTA)
    if preds_mult1 is not None and preds_mult4 is not None:
        out_png = os.path.join(PLOTS_DIR, 'confusion_mult1_vs_mult4.png')
        make_confusion_image(preds_mult1['preds'], preds_mult4['preds'], 
                           out_png, title='aug_mult=1 vs aug_mult=4 (no-TTA)')

    # Per-class CSVs to image
    for csv_name in ['class_distribution_stats.csv', 'mult1_no_tta_vs_tta_per_class.csv', 'mult4_no_tta_vs_tta_per_class.csv']:
        csv_path = os.path.join(PLOTS_DIR, csv_name)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                img_name = os.path.join(PLOTS_DIR, csv_name.replace('.csv', '.png'))
                df_to_image(df, img_name, title=csv_name)
            except Exception as e:
                print(f"Warning: failed to render {csv_path} to image: {e}")

    imgs = collect_plots()
    if not imgs:
        raise FileNotFoundError(f"No plot images found in {PLOTS_DIR}")

    # Build summary text including class descriptions
    summaries = read_summaries()
    txt_lines = []
    
    # Add summary statistics
    if 'mult1_no_tta_vs_tta_summary.csv' in summaries:
        s = summaries['mult1_no_tta_vs_tta_summary.csv']
        if isinstance(s, pd.DataFrame) and not s.empty:
            row = s.iloc[0]
            txt_lines.append(f"aug_mult=1: samples={int(row['n_samples'])}, changed={int(row['n_changed'])} ({row['pct_changed']:.2f}%)")
    if 'mult4_no_tta_vs_tta_summary.csv' in summaries:
        s = summaries['mult4_no_tta_vs_tta_summary.csv']
        if isinstance(s, pd.DataFrame) and not s.empty:
            row = s.iloc[0]
            txt_lines.append(f"aug_mult=4: samples={int(row['n_samples'])}, changed={int(row['n_changed'])} ({row['pct_changed']:.2f}%)")

    # Generate natural language descriptions for each class
    class_desc = []
    for name, df in summaries.items():
        if 'per_class.csv' in name and isinstance(df, pd.DataFrame):
            model = 'aug_mult=1' if 'mult1' in name else 'aug_mult=4'
            class_desc.append(f"\n=== {model} 클래스별 변화 ===")
            for _, row in df.iterrows():
                if 'class' in row:
                    desc = generate_class_description(int(row['class']), row)
                    class_desc.append(desc)

    txt_lines.extend(class_desc)
    cover_text = "\n".join(txt_lines) if txt_lines else 'Generated report of plots and summaries.'
    cover = make_cover(cover_text)

    # Open images and convert to RGB
    pil_imgs = [cover]
    for p in imgs:
        try:
            im = Image.open(p).convert('RGB')
            # resize wide images moderately if too large for PDF
            max_w = 1200
            if im.width > max_w:
                ratio = max_w / im.width
                im = im.resize((int(im.width*ratio), int(im.height*ratio)), Image.LANCZOS)
            pil_imgs.append(im)
        except Exception as e:
            print(f"Warning: failed to open {p}: {e}")

    # Save to single PDF
    first, rest = pil_imgs[0], pil_imgs[1:]
    first.save(PDF_PATH, save_all=True, append_images=rest)
    print(f"Saved PDF report to {PDF_PATH}")

    # Also create enhanced HTML embedding images, CSVs and metadata
    # Metadata
    meta = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'cwd': os.getcwd()
    }

    # Try to include small config snippets
    configs = {}
    for cfg in ['configs/temp_full_mult1.yaml', 'configs/temp_full_mult4.yaml', 'configs/temp_full_mult1_tta4.yaml', 'configs/temp_full_mult4_tta4.yaml']:
        if os.path.exists(cfg):
            try:
                with open(cfg, 'r') as fh:
                    configs[os.path.basename(cfg)] = yaml.safe_load(fh)
            except Exception:
                configs[os.path.basename(cfg)] = None

    css = """
    body{font-family: Arial, sans-serif; margin:24px; line-height:1.6; color:#333; max-width:1200px; margin:0 auto;}
    img{max-width:100%;height:auto;border:1px solid #ddd;padding:8px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
    .meta{font-size:0.9rem;color:#666;margin-bottom:12px;padding:8px;background:#f8f8f8;border-radius:4px}
    table.paged{border-collapse:collapse;width:100%;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1)}
    table.paged th{background:#f4f4f4;font-weight:bold;border:1px solid #ddd;padding:12px}
    table.paged td{border:1px solid #ddd;padding:12px;text-align:left}
    table.paged tr:nth-child(even){background:#fafafa}
    .pager{margin:12px 0}
    .pager button{margin:0 4px;padding:4px 12px;border:1px solid #ddd;background:#fff;cursor:pointer;border-radius:4px}
    .pager button:hover{background:#f0f0f0}
    .pager button.active{background:#007bff;color:white;border-color:#0056b3}
    section{margin:32px 0;padding:24px;background:#fff;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
    section:target{animation:highlight 2s}
    @keyframes highlight{from{background:#fff3b8}to{background:#fff}}
    h1,h2,h3{color:#2c3e50}
    h1{border-bottom:3px solid #eee;padding-bottom:12px;margin-top:32px}
    h2{border-bottom:2px solid #eee;padding-bottom:8px;margin-top:24px}
    pre{background:#f8f8f8;padding:16px;border-radius:4px;overflow-x:auto;font-size:14px}
    .class-description{background:#f8f8f8;padding:16px;margin:8px 0;border-radius:4px;line-height:1.6}
    """

    # JS for basic table pagination
    js = """
    function paginateTable(tableId, pageSize){
      const tbl = document.getElementById(tableId);
      if(!tbl) return;
      const tbody = tbl.tBodies[0];
      const rows = Array.from(tbody.rows);
      let current=0;
      function showPage(page){
        rows.forEach((r,i)=> r.style.display = (i>=page*pageSize && i<(page+1)*pageSize)?'table-row':'none');
        current=page;
      }
      const pages = Math.ceil(rows.length/pageSize);
      const pager = document.createElement('div'); pager.className='pager';
      for(let i=0;i<pages;i++){
        const b=document.createElement('button'); b.textContent=(i+1); b.onclick=(()=>showPage(i)); pager.appendChild(b);
      }
      tbl.parentNode.insertBefore(pager,tbl);
      if(pages>0) showPage(0);
    }
    window.onload = function(){
      // auto-paginate tables with class 'paged'
      document.querySelectorAll('table.paged').forEach((t,idx)=> paginateTable(t,30));
    }
    """

    html_header = f"<meta charset='utf-8'><style>{css}</style><script>{js}</script>"
    html_lines = ["<html>", f"<head>{html_header}</head>", "<body>"]
    html_lines.append('<h1>TTA & Class Distribution Report</h1>')
    html_lines.append(f"<div class='meta'>Generated: {meta['generated_at']} | CWD: {html.escape(meta['cwd'])}</div>")

    if configs:
        html_lines.append('<h2>Configs</h2>')
        for name, cfg in configs.items():
            html_lines.append(f"<h3>{html.escape(name)}</h3>")
            html_lines.append(f"<pre>{html.escape(str(cfg))}</pre>")

    html_lines.append('<h2>Summary</h2>')
    if txt_lines:
        html_lines.append('<ul>')
        for l in txt_lines:
            html_lines.append(f'<li>{l}</li>')
        html_lines.append('</ul>')

    html_lines.append('<h2>Plots</h2>')
    for p in imgs:
        rel = os.path.relpath(p, PLOTS_DIR)
        html_lines.append(f"<div style='margin-bottom:18px;'><h3>{os.path.basename(p)}</h3><img src='{rel}' width='800'></div>")

    # embed small CSV tables with pagination class
    html_lines.append('<h2>CSV Summaries</h2>')
    for idx, (name, df) in enumerate(summaries.items()):
        html_lines.append(f"<h3>{name}</h3>")
        if isinstance(df, pd.DataFrame):
            tbl_id = f'tbl_{idx}'
            html_lines.append(df.to_html(index=False, table_id=tbl_id, classes='paged', escape=False))
        else:
            html_lines.append('<p>Failed to load</p>')

    html_lines.append('</body></html>')

    # Write HTML with images referenced relatively (so it can be opened locally)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))
    print(f"Saved HTML report to {HTML_PATH}")

    # Create changed-samples visualization (per model)
    def find_image_path(img_id):
        # search data/test and data/train
        base_dirs = ['data/test', 'data/train']
        for d in base_dirs:
            for ext in ['.jpg','.jpeg','.png','.JPG','.PNG','.JPEG','']:
                p = os.path.join(d, str(img_id) + ext)
                if os.path.exists(p):
                    return p
        return None

    def make_sample_card(img_path, info, out_png):
        try:
            im = Image.open(img_path).convert('RGB') if img_path and os.path.exists(img_path) else None
        except Exception:
            im = None
        if im is None:
            im = Image.new('RGB', (320,240), color=(200,200,200))
        # create canvas
        w = 800; h = max(320, im.height)
        canvas = Image.new('RGB', (w, h), 'white')
        canvas.paste(im.resize((320, int(im.height * 320 / im.width))), (10,10))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 14)
        except Exception:
            font = ImageFont.load_default()
        x = 350; y = 20
        lines = [f"ID: {info.get('img_id')}", f"No-TTA: pred={info.get('pred_no_tta')} conf={info.get('conf_no_tta'):.3f}", f"TTA: pred={info.get('pred_tta')} conf={info.get('conf_tta'):.3f}", f"conf_delta={info.get('conf_delta'):.3f}"]
        for ln in lines:
            draw.text((x,y), ln, fill='black', font=font)
            y += 22
        canvas.save(out_png)
        return True

    changed_out_dir = os.path.join(PLOTS_DIR, 'changed_samples')
    os.makedirs(changed_out_dir, exist_ok=True)
    for tag in ['mult1','mult4']:
        csvp = os.path.join(PLOTS_DIR, f'{tag}_no_tta_vs_tta_changed_samples.csv')
        if not os.path.exists(csvp):
            continue
        df = pd.read_csv(csvp)
        # select top N by abs(conf_delta)
        df['abs_delta'] = df['conf_delta'].abs()
        df2 = df.sort_values('abs_delta', ascending=False).head(50)
        cards = []
        for idx, row in df2.reset_index(drop=True).iterrows():
            img_id = row['img_id']
            img_path = find_image_path(img_id)
            out_png = os.path.join(changed_out_dir, f'{tag}_sample_{idx}.png')
            info = {'img_id': img_id, 'pred_no_tta': row['pred_no_tta'], 'pred_tta': row['pred_tta'], 'conf_no_tta': row['conf_no_tta'], 'conf_tta': row['conf_tta'], 'conf_delta': row['conf_delta']}
            make_sample_card(img_path, info, out_png)
            cards.append(out_png)
        # make PDF of cards
        pil_pages = [Image.open(p).convert('RGB') for p in cards]
        if pil_pages:
            pdfp = os.path.join(changed_out_dir, f'{tag}_changed_samples.pdf')
            pil_pages[0].save(pdfp, save_all=True, append_images=pil_pages[1:])
            # also html page
            htm = os.path.join(changed_out_dir, f'{tag}_changed_samples.html')
            with open(htm, 'w', encoding='utf-8') as hf:
                hf.write('<html><body>')
                hf.write(f"<h1>Top changed samples - {tag}</h1>")
                for p in cards:
                    hf.write(f"<div style='margin-bottom:12px;'><img src='{os.path.relpath(p, changed_out_dir)}' width='800'></div>")
                hf.write('</body></html>')
            print(f"Saved changed samples PDF/HTML for {tag} to {changed_out_dir}")


if __name__ == '__main__':
    build_pdf()
