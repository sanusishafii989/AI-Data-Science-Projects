# SMS Spam Detection System

> Machine Learning-based SMS spam filtering optimized for the Nigerian telecommunications context.

---

## Project Owner

| Detail | Information |
|---|---|
| **Name** | Sanusi Shafii |
| **Matriculation Number** | CSA/2023/27683 |
| **Email** | s.shafii27683@fudutsinma.edu.ng.com |
| **Phone** | +2349125006811 |
| **Department** | Computer Science and Information Technology |
| **Faculty** | Computing |
| **University** | Federal University Dutsin-Ma, Katsina State, Nigeria |

**Project Duration:** November 2023 – November 2024

---

## Overview

This project implements a Multinomial Naive Bayes classifier to automatically detect and filter spam SMS messages. The system preprocesses raw text messages and classifies them as either **spam** or **ham** (legitimate) with high accuracy.

---

## How It Works

1. **Data Loading** — Reads the SMS spam dataset (`spam.csv`) encoded in ISO-8859-1.
2. **Text Preprocessing** — Lowercasing, tokenization, stopword removal, and Porter Stemming via NLTK.
3. **Feature Extraction** — Bag-of-Words representation using `CountVectorizer`.
4. **Model Training** — Multinomial Naive Bayes classifier trained on a 67/33 train-test split.
5. **Evaluation** — Accuracy score, classification report, and confusion matrix visualization.

---

## Dependencies

```
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
ipywidgets
```

Install all dependencies with:

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn ipywidgets
```

Also download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## Dataset

The project uses the **SMS Spam Collection Dataset** (`spam.csv`), which contains 5,572 labeled SMS messages. Place the file at `/content/spam.csv` when running in Google Colab.

| Column | Description |
|---|---|
| `v1` | Label — `ham` or `spam` |
| `v2` | Raw SMS message text |

---

## Results

The model achieves strong classification performance. Key metrics include:

- **Accuracy:** ~98%
- Detailed precision, recall, and F1-score are available via the classification report
- A confusion matrix heatmap (HAM vs. SPAM) is generated for visual evaluation

---

## File Structure

```
MUSTAPHA_SPAM_EMAIL.ipynb   # Main Jupyter notebook
spam.csv                    # Dataset (required, not included)
```

---

## Usage

1. Open `MUSTAPHA_SPAM_EMAIL.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `spam.csv` to `/content/`.
3. Run all cells in order.
4. The interactive widget at the end allows you to input custom SMS messages and receive real-time spam predictions.
