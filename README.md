# 🤖 AI & Data Science Portfolio

> A collection of machine learning, deep learning, and NLP projects developed during undergraduate studies at Federal University Dutsin-Ma (FUDMA).

**Author:** Sanusi Shafii
**Institution:** Federal University Dutsin-Ma (FUDMA), Katsina State, Nigeria
**Department:** Computer Science and Information Technology
**Program:** B.Sc. Computer Science

---

## 📋 Table of Contents

- [About](#about)
- [Projects](#projects)
  - [1. Automated Bell Pepper Disease Detection](#1-automated-bell-pepper-disease-detection)
  - [2. AI-Based Voice Authentication](#2-ai-based-voice-authentication)
  - [3. Handwritten Writer Verification](#3-handwritten-writer-verification)
  - [4. StudentBuddy Chatbot](#4-studentbuddy-chatbot)
  - [5. Signature Authentication](#5-signature-authentication)
  - [6. Spam Email Detection](#6-spam-email-detection)
  - [7. Student Career Recommendation](#7-student-career-recommendation)
  - [8. Telecom Customer Churn Prediction](#8-telecom-customer-churn-prediction)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Contact](#contact)

---

## About

This repository showcases hands-on AI and data science projects spanning computer vision, natural language processing, biometric authentication, and predictive analytics. Each project addresses a real-world problem using modern machine learning techniques — from convolutional neural networks for plant disease detection to NLP-powered chatbots for student support.

The projects were built using Python and popular ML/DL frameworks, and are designed to be reproducible and well-documented.

---

## Projects

### 1. Automated Bell Pepper Disease Detection

| Detail | Info |
|---|---|
| **Domain** | Computer Vision / Agriculture |
| **Approach** | Convolutional Neural Network (CNN) |
| **Framework** | TensorFlow / Keras, OpenCV |

**Description:**
A deep learning model that automatically detects diseases in bell pepper plants from leaf images. The system classifies leaves as healthy or infected (e.g., Bacterial Spot, Phytophthora Blight) by learning visual patterns through a trained CNN. This tool can help farmers identify crop diseases early and reduce yield loss.

**Key Features:**
- Image preprocessing and augmentation pipeline
- Multi-class CNN classification
- High-accuracy disease identification from plant leaf images
- Potential for deployment as a mobile or web-based diagnostic tool

---

### 2. AI-Based Voice Authentication

| Detail | Info |
|---|---|
| **Domain** | Biometrics / Security |
| **Approach** | MFCC Feature Extraction + Deep Learning |
| **Framework** | TensorFlow / Keras, Librosa |

**Description:**
A speaker verification system that authenticates users based on their unique voice characteristics. The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio samples and uses a deep neural network to determine whether a voice matches a registered speaker's voiceprint.

**Key Features:**
- Audio preprocessing and noise reduction
- MFCC feature extraction for voice representation
- Binary verification — accepts or rejects a claimed identity
- Applicable to secure access systems and voice-controlled devices

---

### 3. Handwritten Writer Verification

| Detail | Info |
|---|---|
| **Domain** | Computer Vision / Forensics |
| **Approach** | Siamese Neural Network |
| **Framework** | TensorFlow / Keras, OpenCV |

**Description:**
A writer verification system that determines whether two handwriting samples belong to the same individual. Built on a Siamese network architecture, the model learns a similarity metric by comparing pairs of handwriting images — useful in forensic document analysis and authentication.

**Key Features:**
- Siamese network with contrastive loss for similarity learning
- Pairwise handwriting comparison (same writer vs. different writer)
- Robust feature extraction from raw handwriting images
- Applications in document forgery detection and forensic analysis

---

### 4. StudentBuddy Chatbot

| Detail | Info |
|---|---|
| **Domain** | Natural Language Processing / Education |
| **Approach** | TF-IDF + Cosine Similarity + Rule-based NLP |
| **Framework** | Scikit-learn, NLTK, Gradio |

**Description:**
An intelligent conversational chatbot designed specifically for students of Federal University Dutsin-Ma (FUDMA). The bot answers frequently asked questions about course registration, academic calendars, campus facilities, and university policies — reducing the burden on administrative staff and giving students 24/7 access to information.

**Key Features:**
- NLP preprocessing: tokenization, stopword removal, lemmatization
- Intent matching via TF-IDF vectorization and cosine similarity
- Gradio-based web interface with public shareable link
- Extensible knowledge base (JSON/pickle format)

---

### 5. Signature Authentication

| Detail | Info |
|---|---|
| **Domain** | Computer Vision / Banking & Finance |
| **Approach** | Deep Learning (CNN / Siamese Network) |
| **Framework** | TensorFlow / Keras, OpenCV |

**Description:**
A signature verification system designed for banking and financial institutions to distinguish genuine signatures from forgeries. The model learns the visual characteristics of a person's signature and detects anomalies that indicate fraud, helping reduce financial losses from signature-based fraud.

**Key Features:**
- Signature image preprocessing and normalization
- Deep learning-based genuine vs. forged classification
- Pairwise similarity learning for robust verification
- Applicable to cheque processing and document authorization workflows

---

### 6. Spam Email Detection

| Detail | Info |
|---|---|
| **Domain** | Natural Language Processing |
| **Approach** | Multinomial Naive Bayes / Bag-of-Words |
| **Framework** | Scikit-learn, NLTK, Pandas |

**Description:**
A text classification system that filters spam emails from legitimate (ham) messages. The pipeline preprocesses raw email text — tokenization, stopword removal, stemming — and feeds it into a Multinomial Naive Bayes classifier trained on the SMS/Email Spam Collection Dataset. Achieves ~98% classification accuracy.

**Key Features:**
- Full NLP preprocessing pipeline (NLTK)
- Bag-of-Words feature extraction via CountVectorizer
- Lightweight, fast Naive Bayes classifier
- Confusion matrix and classification report evaluation
- Interactive widget for real-time spam testing

---

### 7. Student Career Recommendation

| Detail | Info |
|---|---|
| **Domain** | Machine Learning / Education |
| **Approach** | Random Forest Classifier |
| **Framework** | Scikit-learn, Pandas, ipywidgets |

**Description:**
A career guidance system that recommends suitable academic courses and career paths based on a student's skills, interests, and academic strengths. The Random Forest model is trained on labeled student data and provides personalized recommendations through an interactive widget interface.

**Key Features:**
- Automated feature and target column detection
- Random Forest with accuracy metrics and classification report
- Confusion matrix evaluation
- Interactive ipywidgets dashboard for live career predictions
- Easily extensible to new career categories and datasets

---

### 8. Telecom Customer Churn Prediction

| Detail | Info |
|---|---|
| **Domain** | Machine Learning / Business Analytics |
| **Approach** | Ensemble Classification (Multiple Models) |
| **Dataset** | Telco Customer Churn (IBM/Kaggle) |
| **Framework** | Scikit-learn, Pandas, Plotly, Seaborn |

**Description:**
A customer retention system that predicts whether a telecom subscriber is likely to churn (cancel their service). The project covers the full data science workflow — EDA, preprocessing, feature engineering, and multi-model training — giving telecom companies actionable insights to reduce customer loss and optimize retention strategies.

**Key Features:**
- Exploratory Data Analysis with Plotly and Seaborn visualizations
- Missing value handling, encoding, and feature scaling
- Multiple classifiers trained and compared
- Class imbalance handling
- Business-oriented insights and churn risk scoring

---

## Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Languages** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **NLP** | NLTK, TF-IDF, CountVectorizer |
| **Computer Vision** | OpenCV, PIL/Pillow |
| **Audio Processing** | Librosa |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **UI / Deployment** | Gradio, ipywidgets |
| **Environment** | Google Colab, Jupyter Notebook |

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Then install the core dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn plotly \
            opencv-python librosa nltk gradio ipywidgets xgboost lightgbm
```

For NLTK resources, run once inside Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Running a Project

Each project is contained in its own Jupyter notebook (`.ipynb`). To run:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Open the desired notebook in Jupyter or Google Colab.

3. Follow the setup instructions in the notebook's first cells (dataset path, drive mounting, etc.).

4. Run all cells in order (`Runtime → Run all` in Colab, or `Cell → Run All` in Jupyter).

> **Note:** Datasets are not included in this repository. Dataset sources and download instructions are provided within each notebook.

---

## Contact

**Sanusi Shafii**
Federal University Dutsin-Ma (FUDMA), Katsina State, Nigeria
📧 sanusi33@gmail.com

---

*Built with curiosity and Python.*
