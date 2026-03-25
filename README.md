# Bank Customer Segmentation

Segmenting 72,000 bank customers using KMeans clustering + Random Forest — deployed as a two-level FastAPI app on Hugging Face Spaces.

---

## What It Does

Takes a customer's salary, balance, credit score, transaction history and 16 other behavioral features — predicts which segment they belong to.

**Two-level system:**
- Staff view `/` — Segment result only
- Manager view `/admin` — Segment result + business recommendation

---

## Pipeline

```
EDA → LabelEncoder → ColumnTransformer (RobustScaler + StandardScaler) →
Elbow + Silhouette → KMeans (K=5) → Cluster Analysis + PCA →
KNN / LR(L1,L2) / DT / RF → RandomizedSearchCV → Pickle → FastAPI → Docker → HF Spaces
```

---

## Clusters

| # | Segment | Salary | Balance | Credit Score | Late Payments |
|---|---------|--------|---------|--------------|---------------|
| 2 | Premium | 3,403 AZN | 27,211 AZN | 779 | 0.3x |
| 3 | Middle | 1,464 AZN | 2,584 AZN | 620 | 0.8x |
| 0 | Young/Digital | 1,169 AZN | 2,928 AZN | 650 | 1.2x |
| 4 | Risky | 698 AZN | 763 AZN | 489 | 3.8x |
| 1 | Very Risky | 646 AZN | 346 AZN | 449 | 4.6x |

---

## Results

| Model | Accuracy | ROC AUC | Log Loss | Brier |
|-------|----------|---------|----------|-------|
| KNN | 97.95% | 99.69% | 0.178 | 0.0072 |
| Logistic Regression (L2) | 99.79% | 100.00% | 0.012 | 0.0011 |
| Logistic Regression (L1) | 98.70% | 99.90% | 0.093 | 0.0089 |
| Decision Tree | 97.67% | 99.07% | 0.279 | 0.0085 |
| Random Forest | 98.20% | 99.95% | 0.057 | 0.0057 |
| **Random Forest (Tuned)** | **98.31%** | **99.96%** | **0.052** | **0.0052** |

Best model: Random Forest (Tuned)
```
n_estimators=200, max_depth=20, criterion=entropy,
min_samples_split=2, min_samples_leaf=1, class_weight=balanced
```

Overfitting check: Train=99.35% / Test=98.31% — gap=1.04%

Top features: `kredit_bali`, `balans_azn`, `maas_azn`, `gecikmis_odenis_sayi`, `churn_riski`

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

API will be available at `http://localhost:8000`

---

## Tech Stack

Python 3.11 | pandas | numpy | scikit-learn | matplotlib | seaborn | FastAPI | Uvicorn | Jinja2 | Docker | Hugging Face Spaces

---

## Live Demo

| Link | Description |
|------|-------------|
| [Staff App](https://eltonvaliyev11-bank-segmentation-app.hf.space) | Bank staff interface |
| [Manager App](https://eltonvaliyev11-bank-segmentation-app.hf.space/admin) | Manager interface with recommendations |
| [Health Check](https://eltonvaliyev11-bank-segmentation-app.hf.space/health) | API status |
| [API Docs](https://eltonvaliyev11-bank-segmentation-app.hf.space/docs) | Swagger UI |

---

Elton Valiyev — [LinkedIn](https://linkedin.com/in/eltonvaliyev)
