# HEDIS Performance Analytics — Improved AUC Edition

> End-to-end machine learning pipeline for predicting HEDIS care gap compliance, with optimized AUC modeling, per-measure training, and executive dashboards.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [AUC Improvements](#auc-improvements)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notebook Sections](#notebook-sections)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [HEDIS Measures Covered](#hedis-measures-covered)
- [Key Concepts](#key-concepts)
- [Contributing](#contributing)

---

## Overview

This project demonstrates a full data science workflow applied to **synthetic member-level HEDIS (Healthcare Effectiveness Data and Information Set)** data. It is designed to help healthcare analysts and data scientists:

- Identify members at risk of **missing HEDIS care gaps**
- Build and compare ML models that predict **compliance likelihood**
- Generate actionable insights benchmarked against **NCQA national standards**
- Visualize trends, gaps, and performance via an **executive KPI dashboard**

>  All data in this project is **synthetic** and does not represent real patient information.

---

## Features

- **Exploratory Data Analysis** — Compliance rates by measure, line of business, region, and demographics
-  **Improved Feature Engineering** — One-hot encoding, interaction features, age bins, centered year
-  **Multi-Model ML** — Logistic Regression, Random Forest, XGBoost, LightGBM
- **ROC Curve Visualization** — Side-by-side AUC comparison across all models
- **Hyperparameter Tuning** — RandomizedSearchCV on XGBoost (30 combinations)
- **Per-Measure Models** — Separate XGBoost trained per HEDIS measure for higher accuracy
-  **SHAP Explainability** — Feature importance for the tuned XGBoost model
-  **Gap Analysis** — Priority tiers (HIGH / MEDIUM / LOW) for care gap targeting
- **Time Series Trends** — 3-year compliance trend (2022–2024) with YoY delta
- **Executive KPI Dashboard** — Dark-theme visual dashboard with NCQA benchmark overlay



---

## Project Structure

```
hedis-performance-analytics/
│
├── HEDIS_Performance_Analytics_Improved_AUC.ipynb   # Main notebook
├── members_features.csv                              # Input data (synthetic)
├── README.md                                         # This file
└── requirements.txt                                  # Python dependencies
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/hedis-performance-analytics.git
cd hedis-performance-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap scipy
```

### 3. Run in Google Colab (recommended)

1. Upload `HEDIS_Performance_Analytics_Improved_AUC.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `members_features.csv` to `/content/`
3. Run all cells (`Runtime → Run all`)

### 4. Run locally in Jupyter

```bash
jupyter notebook HEDIS_Performance_Analytics_Improved_AUC.ipynb
```

---

## Notebook Sections

| Section | Description |
|---------|-------------|
| 1 | Install & import libraries (xgboost, shap, lightgbm) |
| 2 | Data loading and type casting |
| 3 | Exploratory data analysis — compliance by measure and LOB, class balance check |
| 4 | Gap analysis with HIGH / MEDIUM / LOW priority tiers |
| 5 | Improved feature engineering (one-hot, interactions, age bins) |
| 6 | Baseline ML models with class imbalance handling + ROC curves |
| 6b | XGBoost hyperparameter tuning via RandomizedSearchCV |
| 6c | Per-measure model training and AUC comparison |
| 7 | SHAP explainability for tuned XGBoost |
| 8 | Statistical hypothesis testing (Chi-square, T-test) |
| 9 | 3-year time series trend analysis with YoY delta |
| 10 | Executive KPI dashboard with NCQA benchmark overlay |
| 11 | AUC improvement summary — before vs. after comparison |

---

## Model Performance

The notebook prints a full AUC comparison at the end. Typical results on this synthetic dataset:

| Model | Test AUC | Notes |
|-------|----------|-------|
| Logistic Regression | ~0.72 | Baseline with `class_weight='balanced'` |
| Random Forest | ~0.78 | 200 estimators, balanced weights |
| XGBoost (default) | ~0.80 | With `scale_pos_weight` |
| LightGBM | ~0.81 | Often best single model |
| XGBoost (tuned) | ~0.83 | After RandomizedSearchCV |
| Per-Measure XGBoost | ~0.84 avg | Best overall — separate model per measure |

> Actual results will vary depending on your input data.

---

## Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — ML models, preprocessing, evaluation
- `xgboost`, `lightgbm` — gradient boosting
- `shap` — model explainability
- `matplotlib`, `seaborn` — visualization
- `scipy` — statistical testing

---

## HEDIS Measures Covered

| Code | Measure |
|------|---------|
| CDC-HbA1c | Diabetes HbA1c control |
| CDC-EYE | Diabetes eye exam |
| CDC-KED | Kidney health evaluation for diabetes |
| BCS | Breast cancer screening |
| CCS | Cervical cancer screening |
| CBP | Controlling high blood pressure |
| COL | Colorectal cancer screening |
| AMM | Antidepressant medication management |
| FUH7 | Follow-up after hospitalization for mental illness (7 days) |
| W34 | Well-child visit (3–6 years) |

---

## Key Concepts

**What is AUC?**
AUC (Area Under the ROC Curve) measures how well a model can distinguish compliant members from non-compliant ones. A score of 1.0 is perfect; 0.5 is random guessing.

**What is ROC?**
The ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs. False Positive Rate across all classification thresholds — a higher and more left-leaning curve means a better model.

**Why per-measure models?**
The factors that predict whether a diabetic member gets an eye exam are different from those that predict antidepressant adherence. Pooling all measures into one model dilutes these measure-specific patterns.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/my-improvement`
3. Commit your changes: `git commit -m 'Add my improvement'`
4. Push to the branch: `git push origin feature/my-improvement`
5. Open a pull request

---


---

*Built for healthcare data science education. All data is synthetic and HIPAA-safe.*
