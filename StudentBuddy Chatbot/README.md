# FUDMA Student Buddy Bot

> An AI-powered chatbot designed to assist students of Federal University Dutsin-Ma (FUDMA) with university-related inquiries.

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

The **FUDMA Student Buddy Bot** is a conversational AI assistant built to help FUDMA students get quick, accurate answers to common university-related questions. It uses NLP techniques — TF-IDF vectorization and cosine similarity — to match user queries to the most relevant pre-defined responses.

---

## How It Works

1. **Intent Matching** — User input is preprocessed and compared against a knowledge base using TF-IDF and cosine similarity.
2. **NLP Preprocessing** — Tokenization, stopword removal, and lemmatization via NLTK.
3. **Response Generation** — The bot retrieves the best-matching answer from a structured response dataset (JSON/pickle).
4. **Interactive Interface** — Deployed with a Gradio interface for easy browser-based interaction (`share=True` for public access via Colab).

---

## Dependencies

```
numpy
scikit-learn
nltk
gradio
pickle
```

Install all dependencies with:

```bash
pip install numpy scikit-learn nltk gradio
```

Also download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Features

- Answers common FUDMA student questions (registration, courses, campus info, etc.)
- Natural language input — no need for exact keyword matching
- Browser-accessible Gradio UI with optional public sharing link
- Lightweight and fast — no deep learning model required

---

## File Structure

```
StudentBuddy.ipynb          # Main Jupyter notebook
intents.json / data.pkl     # Knowledge base (required, not included)
```

---

## Usage

1. Open `StudentBuddy.ipynb` in Google Colab or Jupyter Notebook.
2. Ensure the knowledge base file (JSON or pickle) is available.
3. Run all cells in order.
4. The final cell launches the Gradio interface:

```python
gradio_interface.launch(share=True)
```

5. Open the generated public URL in your browser and start chatting with the bot.

---

## Example Interactions

| User Input | Bot Response |
|---|---|
| "How do I register for courses?" | Step-by-step registration guidance |
| "What is the school portal link?" | FUDMA portal URL and login instructions |
| "When is the next semester?" | Academic calendar information |
