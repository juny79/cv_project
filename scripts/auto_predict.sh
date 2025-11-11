#!/bin/bash
# Auto-predict script: waits for fold4/best.pt then runs prediction

CONFIG="configs/exp_mult4_tta4_sharpen.yaml"
CHECKPOINT_PATH="outputs/exp_mult4_tta4_sharpen/fold4/best.pt"
TTA=4

echo "[Auto-Predict] Waiting for $CHECKPOINT_PATH..."

# Wait for fold4 checkpoint to appear
while [ ! -f "$CHECKPOINT_PATH" ]; do
    sleep 30
done

echo "[Auto-Predict] ✓ Found $CHECKPOINT_PATH"
echo "[Auto-Predict] Starting prediction with TTA=$TTA..."

# Run prediction
python src/predict.py --config "$CONFIG" --tta "$TTA"

echo "[Auto-Predict] ✓ Prediction complete!"
echo "[Auto-Predict] Results saved to outputs/exp_mult4_tta4_sharpen/submission.csv"
