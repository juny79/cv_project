
# 📄 Document Type Classification (Bootcamp CV Competition)

문서 이미지 17종 분류 대회 — **EDA → 전처리 → 학습/검증 → 추론/앙상블 → 제출**
- Train: 1,570 images / Test: 3,140 images / Classes: 17
- Metric: **Macro F1**
- 규정: 외부 데이터 금지, ImageNet 사전학습 가중치 허용, Test는 분석만(학습 금지)

## Quickstart
```bash
# 1) Create & activate venv
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Put dataset under: data/datasets_fin/
#    - train.csv, meta.csv, sample_submission.csv, train/, test/

# 4) Create folds
python fold_split.py

# 5) Train baseline
python train.py --config configs/model_resnet34.yaml OUT.DIR outputs/resnet34_224 TRAIN.EPOCHS 5

# 6) Predict & make submission
python predict.py --weights outputs/resnet34_224/fold0_best.pth OUT.PRED out/resnet34_f0.csv
python make_submission.py --pred out/resnet34_f0.csv --sample data/datasets_fin/sample_submission.csv --out pred.csv
```
