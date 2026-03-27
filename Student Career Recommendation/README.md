# Career Recommendation System

> A machine learning system that recommends personalized career paths based on a student's interests and skills.

---

## Project Owner

| Detail | Information |
|---|---|
| **Name** | Sanusi Shafii |
| **Matriculation Number** | CSA/2023/27683 |
| **Email** | s.shafii27683@fudutsinma.edu.ng.com |
| **Department** | Computer Science and Information Technology |
| **Faculty** | Computing |
| **University** | Federal University Dutsin-Ma, Katsina State, Nigeria |

---

## Overview

The **Career Recommendation System** helps students discover potential career paths aligned with their strengths and interests. By training a **Random Forest Classifier** on student skill/interest data, the system predicts suitable academic or professional courses and provides personalized career insights through an interactive widget interface.

---

## How It Works

1. **Dataset Loading** — Reads a student skills/interests dataset (`stud.csv`).
2. **Feature Identification** — Automatically separates input features (skills/interests) from the target column (`Courses`).
3. **Data Preparation** — Encodes categorical variables and handles preprocessing.
4. **Model Training** — Random Forest Classifier trained and validated with accuracy metrics.
5. **Evaluation** — Accuracy score, classification report, and confusion matrix.
6. **Interactive UI** — `ipywidgets` interface lets users input their skills and receive instant career recommendations.

---

## Dependencies

```
numpy
pandas
scikit-learn
ipywidgets
```

Install all dependencies with:

```bash
pip install numpy pandas scikit-learn ipywidgets
```

Enable the custom widget manager in Google Colab:

```python
from google.colab import output
output.enable_custom_widget_manager()
```

---

## Dataset

The project uses a custom student dataset (`stud.csv`). Place the file at `/content/stud.csv` when running in Google Colab.

| Column Type | Description |
|---|---|
| Feature columns | Student skills, interests, and competencies |
| `Courses` (target) | Recommended career/academic course |

---

## Results

The Random Forest model is evaluated on a held-out test set. Key outputs include:

- **Accuracy score** printed after training
- **Classification report** with per-class precision, recall, and F1-score
- **Confusion matrix** for visual performance evaluation
- **Interactive recommendation widget** for real-time predictions

---

## File Structure

```
model.ipynb     # Main Jupyter notebook
stud.csv        # Student dataset (required, not included)
```

---

## Usage

1. Open `model.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `stud.csv` to `/content/`.
3. Run all cells in order.
4. Use the interactive widget at the end to enter your skills and interests and receive career recommendations.
