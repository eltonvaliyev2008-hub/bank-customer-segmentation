# Bank Customer Segmentation

Segmenting 72,000 bank customers using KMeans clustering + Random Forest — deployed as a two-level FastAPI app on Hugging Face Spaces.

---

## What It Does

Takes a customer's salary, balance, credit score, transaction history and 16 other behavioral features — predicts which segment they belong to.

**Two-level system:**
- Staff view `/` — Segment result only
- Manager view `/admin` — Segment result + business recommendation

---


```
1.  Data Loading        — pd.read_csv, shape, dtypes check
2.  EDA                 — head, tail, info, describe, isna, duplicated, value_counts, nunique
3.  Type Conversion     — maas_azn → int
4.  Drop Columns        — cluster, cluster_adi removed before modeling
5.  Label Encoding      — seher, pese, tehsil, aile_veziyyeti, bank_mehsullari
6.  Outlier Detection   — boxplot for all numeric columns
7.  Scaling             — ColumnTransformer: RobustScaler (outlier cols) + StandardScaler (normal cols)
8.  Optimal K           — Elbow Method + Silhouette Score (sample_size=10,000, random_state=42)
9.  KMeans              — K=5, k-means++, n_init=10, max_iter=350, random_state=42
10. Cluster Analysis    — groupby means, cluster heatmap, boxplots (salary/balance/credit)
11. Correlation Heatmap — all features correlation matrix
12. PCA Visualization   — 2D scatter plot with cluster colors
13. Export              — bank_clustered.csv saved
14. Train/Test Split    — 80/20, stratify=y, random_state=42
15. Re-scaling          — CT2 fit_transform(X_train), transform(X_test)
16. Model Training      — KNN, LR(L1), LR(L2), Decision Tree, Random Forest
17. Overfit Check       — train vs test accuracy comparison
18. Feature Importance  — top 10 features from Random Forest
19. Hyperparameter Tuning — RandomizedSearchCV (60 iter, StratifiedKFold n=5, f1_weighted)
20. Evaluation          — Accuracy, Precision, Recall, F1, ROC AUC (OVO), Log Loss, Brier Score
21. Prediction Test     — single customer prediction example
22. Deployment          — pickle → FastAPI → Docker → Hugging Face Spaces
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
> Note: model.pkl and scaler.pkl are hosted on Hugging Face Spaces due to file size.

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
