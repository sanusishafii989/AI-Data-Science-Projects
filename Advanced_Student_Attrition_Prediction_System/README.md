# Student Attrition Prediction Using Data Mining

> An advanced machine learning system that predicts student dropout risk using ensemble methods, feature engineering, and interactive dashboards.

---

## Project Owner

| Detail | Information |
|---|---|
| **Developer** | Sanusi Shafii |
| **Institution** | Federal University Dutsin-Ma (FUDMA) |
| **Email** | sanusi33@gmail.com |
| **Date** | February 2026 |

---

## Overview

This project builds a comprehensive student attrition prediction pipeline. By combining advanced feature engineering, multiple class-balancing strategies, and an ensemble of optimised classifiers, the system identifies students at risk of dropping out — enabling institutions to intervene early and improve retention rates.

**Target Accuracy:** Minimum 80%

---

## Key Features

- **100+ engineered features** derived from raw student data
- **7 optimised machine learning models** trained and compared
- **3 advanced ensemble methods** for improved robustness
- **SMOTE-Tomek balancing** to handle class imbalance
- **Comprehensive feature importance analysis** (Random Forest, XGBoost, LightGBM)
- **K-Means clustering** with UMAP visualisation for student segmentation
- **Interactive HTML prediction dashboard** with a 15-feature input form
- **All outputs saved** — models, visualisations, and reports persisted to Google Drive

---

## How It Works

| Step | Description |
|---|---|
| **1. Data Loading** | Reads `dataset.csv` and displays shape, columns, and a preview |
| **2. Comprehensive Analysis** | Statistical summary, missing value audit, data type inspection, and target correlation ranking |
| **3. Preprocessing** | Median/mode imputation, polynomial and interaction feature creation, robust scaling |
| **4. Data Mining** | Feature importance aggregation across three models; K-Means clustering with UMAP 2D projection |
| **5. Model Training** | Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression, and more — all with tuned hyperparameters |
| **6. Evaluation** | Confusion matrix (Plotly heatmap), classification report, cross-validated scores |
| **7. Dashboard** | HTML prediction interface served inline in Colab; top 15 features exposed as form inputs |

---

## Models Trained

- Random Forest (500 estimators, depth 25)
- XGBoost (500 estimators, learning rate 0.03)
- LightGBM (300 estimators, depth 8)
- CatBoost
- Logistic Regression
- Advanced ensemble (stacking / voting)

---

## Dependencies

```
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
pandas
numpy
matplotlib
seaborn
plotly
umap-learn
shap
mlxtend
joblib
tqdm
optuna
```

Install all at once:

```bash
pip install xgboost lightgbm catboost scikit-learn imbalanced-learn \
            plotly seaborn umap-learn shap mlxtend joblib tqdm optuna
```

---

## Dataset

Place your dataset file at `/content/dataset.csv` before running. The file must contain:

- **Numerical and/or categorical student features** (academic records, demographics, course data, etc.)
- A **`Target`** column indicating attrition status (e.g., `Dropout` / `Graduate` / `Enrolled`)

---

## Output Files

All results are saved to your Google Drive under `MyDrive/Student_Attrition_Project/`:

```
Student_Attrition_Project/
├── models/                   # Serialised trained models (.pkl / .joblib)
├── visualisations/           # Plotly HTML charts (confusion matrix, UMAP, etc.)
└── reports/                  # Classification reports and feature importance tables
```

---

## File Structure

```
model.ipynb          # Main Jupyter notebook (21 cells)
dataset.csv          # Your student dataset (required, not included)
```

---

## Usage

1. Open `model.ipynb` in **Google Colab**.
2. Mount Google Drive when prompted (Cell 2).
3. Upload `dataset.csv` to `/content/`.
4. Run all cells in order (`Runtime → Run all`).
5. Review model performance metrics and the interactive prediction dashboard generated at the end.

---

## Project Summary

> A fully automated pipeline from raw data to deployed prediction dashboard — covering EDA, feature engineering, class balancing, multi-model training, ensemble optimisation, and interactive inference.
