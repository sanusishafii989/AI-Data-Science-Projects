#  AI-Based Voice Authentication System

A robust, production-grade voice authentication system built with deep learning. The system enrolls speakers by learning unique vocal characteristics and authenticates users in real time using multi-feature analysis and anti-spoofing measures.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Overview

This system addresses the challenge of secure, password-free access control by using a person's voice as a biometric identifier. It leverages a **multi-input deep learning model** that processes both raw mel spectrograms (via a CNN) and handcrafted acoustic features (via dense layers) to generate 128-dimensional speaker embeddings. Authentication decisions are made by comparing cosine similarity between a live voice sample and stored user profiles.

---

##  Features

- **Multi-feature voice analysis** — MFCC, spectral features, prosodic features, and mel spectrograms
- **Speaker enrollment & verification** — Register new users and authenticate them in real time
- **Anti-spoofing** — Noise reduction and confidence thresholding prevent unauthorized access
- **Data augmentation** — 13+ variations per sample (pitch shift, time stretch, noise injection) for robust training
- **Interactive dashboard** — Upload an audio file and receive instant authentication results via a styled HTML interface
- **Google Drive integration** — Seamlessly loads voice data from Google Drive on Colab

---

##  Architecture

The model uses a **dual-input deep learning architecture**:

| Branch | Input | Layers |
|---|---|---|
| CNN Branch | Mel Spectrogram | Conv2D (32→64→128) + BatchNorm + MaxPooling + Dropout |
| Statistical Branch | 60+ acoustic features | Dense layers with BatchNorm |
| Fusion | Combined features | Dense(256) → Dense(128, L2-normalized embeddings) |

- **Output:** 128-dimensional L2-normalized speaker embedding
- **Loss:** Categorical cross-entropy (training) → Cosine similarity (inference)
- **Verification threshold:** 0.75 cosine similarity (configurable)

---

##  Dataset Structure

Organize your voice samples in the following structure on Google Drive:

```
voice_samples/
├── alice/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── bob/
│   ├── sample1.wav
│   └── ...
└── charlie/
    └── ...
```

- Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`
- Recommended: **5–10 samples per speaker**, 3–5 seconds each
- Audio is resampled to **22,050 Hz** and padded/trimmed to **4 seconds**

---

##  Installation

Run in **Google Colab** (GPU runtime recommended):

```python
!pip install librosa soundfile numpy scikit-learn scipy tensorflow matplotlib ipywidgets noisereduce pydub -q
```

**Mount Google Drive** and update the path:

```python
VOICE_DATA_FOLDER = '/content/drive/MyDrive/Colab Notebooks/data/voice_samples'
```

---

##  Usage

1. **Open** the notebook in Google Colab
2. **Set runtime** to GPU (`Runtime → Change runtime type → GPU`)
3. **Run all cells** in order (Cells 1–13)
4. **Enroll users** by placing voice samples in the folder structure above
5. **Authenticate** using the interactive dashboard (Cell 13) — upload any `.wav`/`.mp3` file

---

##  How It Works

### 1. Feature Extraction
Each audio file is processed into two feature sets:
- **Statistical features (~60 values):** MFCC (with Δ and ΔΔ), spectral centroid, rolloff, bandwidth, contrast, pitch, energy, zero-crossing rate
- **Mel spectrogram:** Time-frequency image used as CNN input

### 2. Data Augmentation
Each audio sample generates 13+ variants:
- Pitch shifting: −1.5, −0.5, +0.5, +1.5 semitones
- Time stretching: 0.90×, 0.95×, 1.05×, 1.10×
- Additive noise: 3 noise levels (0.002, 0.005, 0.008)

### 3. Model Training
- Multi-input model merges both branches into a shared embedding space
- EarlyStopping + ReduceLROnPlateau prevent overfitting
- Best model checkpoint saved automatically

### 4. Voice Verification
- Uploaded audio → feature extraction → 128-d embedding
- Cosine similarity computed against all registered profiles
- Weighted score (mean + max + top-5 average) determines identity
- Authentication granted only if similarity ≥ `VERIFICATION_THRESHOLD` (0.75) and confidence ≥ `MIN_CONFIDENCE` (0.70)

### 5. Interactive Dashboard
- Styled HTML/CSS panel for real-time audio upload and authentication
- Displays matched user, confidence score, and color-coded similarity bar

---

##  Results

| Metric | Value |
|---|---|
| Validation Accuracy | Displayed after training |
| Embedding Dimension | 128 |
| Similarity Metric | Cosine Similarity |
| Verification Threshold | 0.75 |
| Minimum Confidence | 0.70 |

*Performance varies based on number of speakers, sample quality, and recording conditions.*

---

##  Project Structure

```
AI_Based_Voice_Authentication_System.ipynb
│
├── Cell 1  — Install dependencies
├── Cell 2  — Import libraries
├── Cell 3  — Mount Google Drive
├── Cell 4  — System configuration & thresholds
├── Cell 5  — VoiceFeatureExtractor class
├── Cell 6  — Data augmentation pipeline
├── Cell 7  — Load & process training data
├── Cell 8  — Normalize features (StandardScaler)
├── Cell 9  — Build dual-input deep learning model
├── Cell 10 — Train model (EarlyStopping, ReduceLR)
├── Cell 11 — Evaluate model & plot accuracy/loss curves
├── Cell 12 — Create user voice profiles (embeddings)
├── Cell 13 — VoiceAuthenticator class + Interactive Dashboard
└── Cell 14 — System documentation
```

---

##  Author

**Sanusi Shafii**  
Matric No: CSA/2023/27683  
Email: s.shafii27683@fudutsinma.edu.ng

---

##  License

This project is intended for educational and research purposes.
