#!/bin/bash
set -e

echo "=========================================="
echo "Keyword Routing Impact Comparison Pipeline"
echo "=========================================="
echo ""

# Configuration
BASE_CONFIG="configs/exp_no_routing.yaml"
ROUTING_CONFIG="configs/exp_with_routing.yaml"
OCR_CSV="extern/ocr_text.csv"

# Resolve out_dir from configs to avoid mismatch between script variables and config paths
BASE_OUT=$(python - <<'PY'
import yaml
with open('configs/exp_no_routing.yaml','r') as f:
    cfg=yaml.safe_load(f)
print(cfg['paths']['out_dir'])
PY
)
ROUTING_OUT=$(python - <<'PY'
import yaml
with open('configs/exp_with_routing.yaml','r') as f:
    cfg=yaml.safe_load(f)
print(cfg['paths']['out_dir'])
PY
)

# Step 1: Ensure output directories exist
echo "[Step 1/4] Creating output directories..."
mkdir -p "$BASE_OUT"
mkdir -p "$ROUTING_OUT"
echo "✓ Directories created"
echo ""

# Step 2: Run baseline inference (routing disabled)
echo "[Step 2/4] Running baseline inference (no routing)..."
echo "Config: $BASE_CONFIG"
echo "Output: $BASE_OUT"
python src/predict.py --config "$BASE_CONFIG" --tta 4 2>&1 | tee "$BASE_OUT/predict_baseline.log"
cp outputs/full_mult4/submission.csv "$BASE_OUT/submission.csv" 2>/dev/null || true
cp outputs/full_mult4/predict_logits.pt "$BASE_OUT/predict_logits.pt" 2>/dev/null || true
if [ ! -f "$BASE_OUT/submission.csv" ]; then
    echo "ERROR: Baseline submission not generated! (Check logs under outputs/full_mult4)"
    exit 1
fi
echo "✓ Baseline inference complete"
echo ""

# Step 3: Run routing inference (routing enabled with debug)
echo "[Step 3/4] Running inference with keyword routing..."
echo "Config: $ROUTING_CONFIG"
echo "Output: $ROUTING_OUT"
python src/predict.py --config "$ROUTING_CONFIG" --tta 4 2>&1 | tee "$ROUTING_OUT/predict_routing.log"
cp outputs/full_mult4/submission.csv "$ROUTING_OUT/submission.csv" 2>/dev/null || true
cp outputs/full_mult4/predict_logits.pt "$ROUTING_OUT/predict_logits.pt" 2>/dev/null || true
cp outputs/full_mult4/keyword_routing_debug.csv "$ROUTING_OUT/keyword_routing_debug.csv" 2>/dev/null || true
if [ ! -f "$ROUTING_OUT/submission.csv" ]; then
    echo "ERROR: Routing submission not generated! (Check logs under outputs/full_mult4)"
    exit 1
fi
echo "✓ Routing inference complete"
echo ""

# Step 4: Analyze impact
echo "[Step 4/4] Analyzing routing impact..."
python scripts/analyze_routing_impact.py \
    --base_csv "$BASE_OUT/submission.csv" \
    --new_csv "$ROUTING_OUT/submission.csv" \
    --logits_pt "$ROUTING_OUT/predict_logits.pt" \
    --ocr_csv "$OCR_CSV" \
    2>&1 | tee "$ROUTING_OUT/impact_analysis.json"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Baseline:       $BASE_OUT/submission.csv"
echo "  With Routing:   $ROUTING_OUT/submission.csv"
echo "  Impact Summary: $ROUTING_OUT/impact_analysis.json"
echo "  Flip Details:   $ROUTING_OUT/routing_flips_detail.csv"
if [ -f "$ROUTING_OUT/keyword_routing_debug.csv" ]; then
    echo "  Debug CSV:      $ROUTING_OUT/keyword_routing_debug.csv"
fi
echo ""
