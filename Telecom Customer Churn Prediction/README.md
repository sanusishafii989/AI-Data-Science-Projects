# Customer Churn Monitoring and Prediction System

> A data science project for analyzing and predicting customer churn using the Telco Customer Churn dataset.

---

## Project Owner

| Detail | Information |
|---|---|
| **Author** | Sanusi Shafii |
| **Dataset** | Telco Customer Churn |
| **Date** | 07 Jan 2026 |

---

## Overview

This project builds a complete end-to-end machine learning pipeline to **monitor and predict customer churn** for a telecommunications company. By identifying customers who are likely to leave, businesses can take proactive retention measures and reduce revenue loss.

The notebook covers the full data science workflow — from exploratory data analysis (EDA) and preprocessing through to model training, evaluation, and interpretation.

---

## Dataset

**Source:** Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

The dataset contains customer account information, service subscriptions, and whether the customer churned. Key attributes include:

| Feature | Description |
|---|---|
| `tenure` | Number of months the customer has been with the company |
| `MonthlyCharges` | The amount charged monthly |
| `TotalCharges` | Total charges over the customer's lifetime |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `Churn` | Target variable — Yes or No |

---

## How It Works

1. **Data Loading** — Reads the Telco Churn CSV dataset.
2. **Exploratory Data Analysis (EDA)** — Visualizes class balance, feature distributions, and correlations using Matplotlib, Seaborn, and Plotly.
3. **Preprocessing** — Handles missing values (SimpleImputer / KNNImputer), encodes categorical variables (LabelEncoder, OneHotEncoder), and scales features (StandardScaler / MinMaxScaler).
4. **Model Training** — Multiple classifiers are trained and compared.
5. **Evaluation** — Models are evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
6. **Churn Prediction** — The trained model can predict whether a new customer is at risk of churning.

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
plotly
```

Install all dependencies with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly
```

---

## File Structure

```
MISBAHU_ISHAQ_PROJECT_MODEL.ipynb   # Main Jupyter notebook
WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (required, not included)
```

---

## Usage

1. Open `MISBAHU_ISHAQ_PROJECT_MODEL.ipynb` in Google Colab or Jupyter Notebook.
2. Upload the dataset to `/content/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Run all cells in order.
4. Review the EDA visualizations, model performance metrics, and churn predictions.

---

## Key Sections in the Notebook

| Section | Description |
|---|---|
| Libraries | All imports and dependencies |
| Read Data | Dataset loading and initial inspection |
| EDA | Visual exploration of data balance, distributions, and patterns |
| Preprocessing | Imputation, encoding, and scaling |
| Modeling | Classifier training and hyperparameter configuration |
| Evaluation | Performance metrics and confusion matrix |
| Prediction | Inferring churn risk on new customer data |
