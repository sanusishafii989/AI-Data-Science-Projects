# 📚 RAG Study Assistant
## A Retrieval-Augmented Generation (RAG) and LLM System for Context-Aware, Multimodal Study Assistance

**Federal University Dutsin-Ma (FUDMA) — Final Year Project**

---

## 📌 What is This Project? (Start Here if You Know Nothing)

This project builds a **smart study assistant** that you can load with your own lecture notes, textbooks, or past exam papers (in PDF format), and then ask it any question — and it will answer you **based only on what is in those documents**.

Think of it like this: instead of searching through a 200-page textbook yourself, you upload it to this system, type your question, and the system finds the relevant paragraphs and gives you a clear, intelligent answer.

### A Real Example of What it Did

The notebook's output shows a student uploaded a document called **"SPECIAL TOPICS IN ICT.docx"** and asked:

> *"WHAT IS ICT? MENTION SPECIAL TOPICS UNDER ICT. BRIEFLY EXPLAIN THOSE SPECIAL TOPICS."*

The system returned a detailed answer listing all 13 special topics (Bioinformatics, Digital Forensics, Nanotechnology, Mobile and Wireless Networks, etc.) with explanations — all extracted from the uploaded document and answered by an AI language model.

---

## 🤔 Why is This Project Important?

Traditional study methods require students to manually search through long documents for answers. This project automates that process using two powerful AI techniques:

1. **Retrieval** — Automatically finds the most relevant sections of a document for any question
2. **Generation** — Uses a large language model (LLM) to write a fluent, understandable answer based on those sections

Together, this is called **Retrieval-Augmented Generation (RAG)** — one of the most important AI techniques in use today by companies like Google, Microsoft, and OpenAI.

The value for students and lecturers is enormous:
- Students can interact with their study materials like talking to a tutor
- Lecturers can test whether their notes contain sufficient coverage of a topic
- The system supports both typed (digital) and scanned PDFs via OCR

---

## 🧠 What is RAG? (Explained Simply)

Imagine you are in an exam and you are allowed to bring a textbook. You do not memorize the whole book — instead, when a question comes, you quickly flip to the relevant page and read the answer.

RAG works exactly the same way:

```
Student asks a question
       ↓
System searches through the uploaded documents
and finds the 4 most relevant paragraphs
       ↓
Those 4 paragraphs are given to the AI as context
       ↓
The AI reads those paragraphs and writes a clear answer
       ↓
Student receives the answer with context from their own materials
```

This is fundamentally different from a standard chatbot, which answers from its own training data (which might be outdated or irrelevant to your specific course). RAG answers from **your documents** — making it far more accurate and trustworthy for academic use.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE PHASE (Setup)                        │
│                                                                   │
│  PDF File(s) uploaded by user                                    │
│       ↓                                                          │
│  PyMuPDF extracts text from each page                            │
│  (Tesseract OCR is used if a page is scanned/image-based)        │
│       ↓                                                          │
│  Text is split into overlapping chunks (500 chars, 50 overlap)   │
│       ↓                                                          │
│  SentenceTransformer (all-MiniLM-L6-v2) converts each chunk     │
│  into a 384-dimensional embedding vector                         │
│       ↓                                                          │
│  All vectors stored in FAISS Index (fast nearest-neighbor search)│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     ONLINE PHASE (Query)                         │
│                                                                   │
│  User types a question in the widget                             │
│       ↓                                                          │
│  Question is embedded with the same SentenceTransformer          │
│       ↓                                                          │
│  FAISS finds the 4 most similar chunks (cosine/L2 distance)      │
│       ↓                                                          │
│  Those 4 chunks are concatenated as "context"                    │
│       ↓                                                          │
│  A prompt is constructed:                                        │
│      "You are a study assistant. Use ONLY this context:          │
│       [chunk 1] [chunk 2] [chunk 3] [chunk 4]                   │
│       Answer this question: [user's question]"                   │
│       ↓                                                          │
│  Groq LLM (LLaMA 3.1 8B) generates the final answer             │
│       ↓                                                          │
│  Answer displayed in a styled HTML card                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Libraries and Tools Used

| Library / Tool | Version | Role |
|---|---|---|
| `PyMuPDF` (`fitz`) | Latest | Read and extract text from PDF files, page by page |
| `pytesseract` | Latest | OCR (Optical Character Recognition) for scanned/image pages |
| `Tesseract-OCR` | 4.1.1 | The underlying OCR engine (installed via apt-get) |
| `sentence-transformers` | Latest | Load the `all-MiniLM-L6-v2` model to convert text to vectors |
| `faiss-cpu` | Latest | Build and query the vector similarity index |
| `groq` | Latest | API client to call the Groq LLM (LLaMA 3.1 8B Instant) |
| `google-generativeai` | Latest | Initially imported (Gemini), but Groq is actually used |
| `Pillow` (`PIL`) | Latest | Handle image conversion from PDF pages for OCR |
| `ipywidgets` | Latest | Build the interactive question/answer interface |
| `numpy` | Latest | Numerical array operations for embeddings |
| `google.colab.files` | Built-in | File upload utility in Google Colab |

---

## ⚙️ Configuration Settings

| Parameter | Value | What it Controls |
|---|---|---|
| `GROQ_API_KEY` | User's key | Authentication to access the Groq LLM API |
| `CHUNK_SIZE` | 500 | Maximum number of characters in each text chunk |
| `CHUNK_OVERLAP` | 50 | Characters repeated between consecutive chunks to avoid losing boundary sentences |
| `TOP_K` | 4 | Number of most relevant chunks retrieved per question |
| `embedding_model` | `all-MiniLM-L6-v2` | The sentence embedding model from HuggingFace |
| `LLM model` | `llama-3.1-8b-instant` | The Groq-hosted language model that generates answers |

---

## 📋 Notebook Structure — Cell by Cell

| Cell | ID | Type | Purpose |
|---|---|---|---|
| Title | `cell-md-title` | Markdown | Project overview and instructions |
| 1 | `cell-install` | Code | Install all required packages (`pymupdf`, `pytesseract`, `sentence-transformers`, `faiss-cpu`, `groq`, `pillow`, `ipywidgets`) and Tesseract OCR |
| 2 | `cell-imports` | Code | Import all libraries in one place |
| 3 | `cell-config` | Code | Set API key, configuration constants, initialize the Groq client, and download/load the embedding model |
| 4 | `cell-processing` | Code | Define `extract_text_from_pdf()` and `chunk_text()` functions |
| 5 | `cell-index` | Code | Define `build_faiss_index()` and `retrieve_relevant_chunks()` functions |
| 6 | `cell-upload` | Code | Interactive file upload → extract → chunk → embed → build FAISS index |
| 7 | `cell-query` | Code | Define `answer_question()` — the core RAG function |
| 8 | `cell-interactive` | Code | Launch the interactive study assistant widget (Ask/Clear buttons) |

---

## 🔧 Detailed Pipeline Explanation

### Step 1 — PDF Text Extraction (`extract_text_from_pdf`)

This function handles any PDF file, whether it is a typed digital document or a scanned paper.

**For digital PDFs:** PyMuPDF (`fitz`) reads each page and extracts the text directly. This is fast and accurate.

**For scanned PDFs (image-based):** When a page has no selectable text (common with scanned lecture slides or photocopied notes), the system:
1. Renders the page as a high-resolution image (200 DPI)
2. Converts it to a PIL Image object
3. Passes it to Tesseract OCR, which reads the text from the image

This means the system can handle **any PDF file** a student might have, including old scanned textbooks.

```
PDF page
   ↓
page.get_text()  ← PyMuPDF tries to extract text directly
   ↓
If text is empty:
   → Render page as image (200 DPI)
   → Run Tesseract OCR on the image
   → Extract text from OCR result
```

### Step 2 — Text Chunking (`chunk_text`)

A 200-page textbook might have 100,000+ words. You cannot send all of that to an AI at once — there are limits on how much text an AI can process in one request (called the "context window").

The solution is to break the text into small, manageable **chunks** of 500 characters each. The 50-character **overlap** between chunks ensures that sentences near the boundary between two chunks are not cut off and lost.

```
Full document text (e.g., 50,000 characters)
   ↓
Chunk 1: characters 0 → 500
Chunk 2: characters 450 → 950      ← overlaps by 50 chars
Chunk 3: characters 900 → 1400     ← overlaps by 50 chars
...
```

For the sample document ("SPECIAL TOPICS IN ICT"), this produced **214 chunks**.

### Step 3 — Embedding (`build_faiss_index`)

Each text chunk needs to be converted into a **numerical vector** — a list of numbers that captures the meaning of the text. This is done by the `all-MiniLM-L6-v2` model from HuggingFace.

This model is a **Sentence Transformer** — a neural network specifically trained to produce embeddings where texts with similar meanings produce vectors that are numerically close to each other.

```
Chunk: "Bioinformatics is the application of computational tools..."
   ↓
SentenceTransformer.encode(chunk)
   ↓
[0.23, -0.14, 0.87, 0.02, ..., 0.45]   ← 384 numbers (the embedding)
```

All 214 chunk embeddings are stored in a **FAISS index** — a specialized data structure created by Facebook AI Research (FAISS = Facebook AI Similarity Search) that can find the most similar vectors extremely fast, even with millions of entries.

### Step 4 — Retrieval (`retrieve_relevant_chunks`)

When a student asks a question, the question itself is converted to an embedding vector using the same model. FAISS then compares this question vector to all 214 chunk vectors and returns the **4 most similar chunks** (TOP_K = 4) based on L2 (Euclidean) distance.

```
Question: "What is bioinformatics?"
   ↓
Embed the question → question_vector
   ↓
FAISS searches all 214 chunk vectors
   ↓
Returns the 4 chunks with the smallest L2 distance to the question vector
   ↓
These 4 chunks likely contain information about bioinformatics
```

### Step 5 — Generation (`answer_question`)

The retrieved chunks are combined into a single block of text called the **context**. A carefully engineered **prompt** is then built:

```
"You are a helpful and accurate study assistant.
Use only the context provided below to answer the question.
If the context does not contain enough information, say so clearly.

Context:
[Chunk 1 text]

[Chunk 2 text]

[Chunk 3 text]

[Chunk 4 text]

Question: [user's question]

Answer:"
```

This prompt is sent to the **Groq API**, which runs the **LLaMA 3.1 8B Instant** model — a powerful open-source language model created by Meta AI and hosted by Groq on specialized AI hardware for very fast responses.

The model reads the prompt and generates a complete, fluent answer. The key constraint is that the AI is instructed to use **only the provided context** — this prevents it from making up information or answering from general knowledge that might not match the course content.

---

## 🖥️ The Interactive Interface (Cell 8)

The final cell creates a simple, clean user interface inside the Jupyter notebook using `ipywidgets`:

```
┌──────────────────────────────────────────────────────┐
│  RAG Study Assistant                                  │
│  Ask any question based on the documents you uploaded.│
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  Type your study question here and click Ask..│  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  [    Ask    ]   [  Clear  ]                          │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  Question                                      │  │
│  │  [user's question displayed here]              │  │
│  │  ─────────────────────────────────────────     │  │
│  │  Answer                                        │  │
│  │  [AI-generated answer displayed here]          │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Ask button:** Reads the question, calls `answer_question()`, and displays the result in a styled HTML card with a "Question" section and an "Answer" section.

**Clear button:** Empties the question box and clears the output area.

The answer is rendered with HTML styling — bordered card, bold headers, comfortable font size and line spacing — making it easy and pleasant to read.

---

## 🚀 How to Run the Project

### Requirements
- A Google account (for Google Colab — free)
- Internet connection
- A free Groq API key
- One or more PDF documents to use as your study material

### Getting a Free Groq API Key
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Go to API Keys and create a new key
4. Copy the key — it starts with `gsk_`

### Step-by-Step Instructions

1. Open the notebook in Google Colab (or upload it to Colab)
2. **Cell 1:** Run it to install all libraries (this takes 1–2 minutes on first run)
3. **Cell 2:** Run it to import libraries
4. **Cell 3:** Paste your Groq API key in the line `GROQ_API_KEY = "..."` then run
5. **Cell 4:** Run it to define the text extraction functions
6. **Cell 5:** Run it to define the embedding and indexing functions
7. **Cell 6:** Run it — a file upload dialog will appear. Select your PDF file(s) and wait for the index to build
8. **Cell 7:** Run it to define the query function
9. **Cell 8:** Run it — the interactive interface will appear. Type your question and click **Ask**

> **Important:** Always run cells in order from top to bottom. Do not skip any cell.

---

## 📁 Project File Structure

```
project/
├── RAG_Study_Assistant.ipynb     ← The main notebook
├── README.md                     ← This file
└── [your PDF files]              ← Uploaded temporarily in Colab session
```

> Note: Files uploaded in Colab are stored in the Colab session's temporary storage and are lost when the session ends. The notebook and knowledge index must be rebuilt each session.

---

## 🔑 API Keys and Security

The notebook currently has a hardcoded Groq API key in Cell 3:
```python
GROQ_API_KEY = "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"
```

> ⚠️ **Important Security Note:** Hardcoding API keys in notebooks is a security risk if the notebook is shared publicly. For production use or when sharing the notebook with others, store the key using Colab Secrets (the key icon in the left panel) and access it with: `from google.colab import userdata; GROQ_API_KEY = userdata.get('GROQ_API_KEY')`

---

## 📐 Key Concepts Glossary

| Term | Plain English Explanation |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | A technique that combines document search (retrieval) with AI text generation to produce answers grounded in specific documents |
| **LLM (Large Language Model)** | A very large AI model trained on billions of text examples that can understand and generate human language. Examples: GPT-4, LLaMA, Gemini |
| **Embedding** | A way of converting text into a list of numbers (a vector) such that similar texts produce numerically similar vectors |
| **Vector** | A list of numbers. In this context, each text chunk becomes a vector of 384 numbers |
| **FAISS** | Facebook AI Similarity Search — a library that can find the most similar vectors in a large collection very fast |
| **FAISS Index** | The data structure that stores all the chunk embeddings and allows fast nearest-neighbor search |
| **Cosine Similarity / L2 Distance** | Mathematical ways to measure how similar two vectors are. Smaller distance = more similar content |
| **SentenceTransformer** | A type of neural network fine-tuned to produce high-quality sentence/paragraph embeddings |
| **all-MiniLM-L6-v2** | The specific embedding model used. 384-dimensional embeddings. Fast, small, and high quality |
| **Groq** | A company that hosts open-source LLMs on specialized AI hardware (LPUs) for very fast inference |
| **LLaMA 3.1 8B Instant** | Meta AI's open-source language model with 8 billion parameters, hosted on Groq |
| **PyMuPDF (fitz)** | A Python library for reading, writing, and processing PDF files |
| **OCR (Optical Character Recognition)** | Technology that converts images of text (like scanned documents) into actual readable text |
| **Tesseract** | An open-source OCR engine originally developed by HP and now maintained by Google |
| **Chunk** | A small piece of a larger text document, used so that the AI can process documents piece by piece |
| **Chunk Size** | The maximum number of characters in one chunk (set to 500 in this project) |
| **Chunk Overlap** | The number of characters repeated between adjacent chunks (set to 50) to avoid cutting off sentences |
| **TOP_K** | The number of most relevant chunks returned for each question (set to 4) |
| **Prompt** | The full text sent to an LLM, including instructions, context, and the user's question |
| **Context Window** | The maximum amount of text an LLM can process at once. By retrieving only 4 chunks, we keep the context small enough |
| **ipywidgets** | A Python library for creating interactive UI elements (buttons, text boxes, sliders) in Jupyter notebooks |
| **HuggingFace** | A platform that hosts thousands of pre-trained AI models for free download and use |
| **DPI (Dots Per Inch)** | Resolution setting for rendering PDF pages as images. The notebook uses 200 DPI for OCR accuracy |
| **Grounded Answer** | An answer that is based on specific evidence (the retrieved chunks) rather than general knowledge |

---

## ⚡ Performance Notes

- **Embedding speed:** The `all-MiniLM-L6-v2` model processed 214 chunks in approximately 11 seconds (7 batches at ~1.42s each) in the sample run
- **Query speed:** Each question takes 1–3 seconds on Groq's hardware (near-instant compared to other providers)
- **OCR speed:** Scanned pages at 200 DPI take approximately 1–3 seconds per page with Tesseract
- **Model size:** `all-MiniLM-L6-v2` is approximately 90.9 MB — relatively small and fast to download and run

---

## 🔬 Sample Output

The notebook recorded this real interaction:

**Question asked:**
```
WHAT IS ICT
MENTION SPECIAL TOPICS UNDER ICT
BREIFLY EXPLAIN THOSE SPECIAL TOPICS MENTIONED
```

**Answer produced (from the uploaded document):**
The system correctly identified and listed all 13 special topics under ICT from the document:
Bioinformatics, Digital Forensics, Computer Centre Management, Information Technology Law, Modern Theory of Computation, Nanotechnology, Design and Laying of Optics Fibres, Design and Construction of Telecommunication Masts, Technology of ATM, The GSM Call Cards, Design and Construction of a Satellite, Distributed Computing, and Mobile and Wireless Network — then gave a brief explanation of each.

---

## 🛠️ Possible Improvements

- **Persistent storage:** Save the FAISS index to Google Drive so it does not need to be rebuilt each session
- **Multi-modal support:** Add image understanding for PDFs that contain diagrams and charts
- **Conversation history:** Allow the assistant to remember previous questions in the same session
- **Source citation:** Show which page and document each answer came from
- **Better chunking:** Use sentence-boundary detection instead of character count for smarter splitting
- **Re-ranking:** After FAISS retrieval, use a cross-encoder model to re-rank the chunks for better accuracy
- **Secure API key handling:** Use Colab Secrets instead of hardcoding the key

---

## 📚 References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS — Original RAG paper
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP — SentenceTransformers paper
- Johnson, J. et al. (2019). *Billion-scale similarity search with GPUs.* IEEE — FAISS paper
- Meta AI (2024). *LLaMA 3.1 Technical Report* — LLaMA model
- Groq — [https://console.groq.com](https://console.groq.com)
- HuggingFace — [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- PyMuPDF — [https://pymupdf.readthedocs.io](https://pymupdf.readthedocs.io)
- Tesseract OCR — [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

---

*Federal University Dutsin-Ma (FUDMA) — Department of Computer Science and Information Technology*
