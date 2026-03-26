# Handwritten Writer Identification System
### Forensic-Grade Writer Verification using Siamese CNN with Triplet Loss

> Given two handwriting samples, determine with confidence whether they were written by the same person — powered by deep metric learning on the IAM Handwriting Database.

---

## Project Details

| Detail | Information |
|--------|-------------|
| **Developer** | Sanusi Shafii |
| **Email** | s.shafii27683@fudutsinma.edu.ng |
| **Institution** | Federal University Dutsin-Ma, Katsina State, Nigeria |
| **Year** | 2025 |
| **TensorFlow Version** | 2.19.0 |
| **Training Environment** | Google Colab (T4 GPU) |

---

## Overview

This system addresses a **writer verification** task — a one-shot, open-set problem where the model must decide if two handwriting samples originate from the same writer, even for writers never seen during training.

Rather than learning fixed classifications, the model learns a **metric embedding space** where handwriting samples from the same writer cluster together and samples from different writers are pushed apart. Inference requires no retraining — any new writer can be verified immediately.

The system is designed for applications in **forensic document analysis**, **archival research**, **academic integrity checks**, and **biometric identity verification**.

---

## Key Features

- Writer verification from just two handwriting image samples
- Triplet loss metric learning — stronger class separation than contrastive loss
- Simplified 2-block CNN — optimised to avoid overfitting on limited per-writer data
- Calibrated confidence levels: **LOW / MODERATE / HIGH**
- Interactive HTML verification interface with side-by-side image comparison
- Automatic threshold optimisation on validation set
- Data augmentation that effectively triples training data
- One-time training, unlimited inference

---

## Dataset

### IAM Handwriting Database (Modified)

| Statistic | Value |
|-----------|-------|
| Total Writers | 505 |
| Total Images | 3,903 |
| Samples per Writer | 3 – 8 |
| Average Samples/Writer | 7.7 |
| Image Type | Grayscale handwriting on lined/plain paper |

The dataset is sourced from Kaggle and downloaded automatically via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("ashish2001/iam-dataset-modified")
```

📦 [IAM Dataset Modified — Kaggle](https://www.kaggle.com/datasets/ashish2001/iam-dataset-modified)

No manual download or Google Drive setup is required — the notebook handles everything.

---

## Model Architecture

The system uses a **Siamese Triplet Network**: three inputs (anchor, positive, negative) share a single CNN backbone with tied weights, and the network learns to minimise anchor-positive distance while maximising anchor-negative distance.

### CNN Backbone (Embedding Network)

```
Input: 384 × 384 × 1 (grayscale)

Block 1:  Conv2D(32, 5×5, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Block 2:  Conv2D(64, 5×5, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)

          GlobalAveragePooling2D
          Dense(256, relu) → Dropout(0.50)
          Dense(128)         ← embedding layer
          L2 Normalisation

Output: 128-dimensional unit-norm embedding vector
```

> **Design rationale:** A deliberately shallow 2-block architecture prevents overfitting on the limited 3–8 samples per writer. GlobalAveragePooling replaces Flatten to further reduce parameter count.

### Triplet Network

```
Anchor   ──┐
Positive ──┼──► Shared Embedding Net ──► Distance(A, P)  ┐
Negative ──┘                             Distance(A, N)  ┴──► Triplet Loss
```

### Loss Function

**Triplet Loss** with margin = 0.5:

```
L = max(0,  d(anchor, positive) − d(anchor, negative) + margin)
```

This directly optimises the embedding space geometry, encouraging intra-writer compactness and inter-writer separation.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 8 × 10⁻⁵ |
| Batch Size | 16 |
| Max Epochs | 50 |
| Early Stopping Patience | 12 epochs |
| LR Reduction Factor | 0.5 (on plateau) |
| Triplet Margin | 0.5 |
| Image Size | 384 × 384 |
| Embedding Dimension | 128 |
| Expected Training Time | 4 – 6 hours (Colab T4 GPU) |

### Data Augmentation

Applied to the training set to effectively multiply dataset size:
- Random rotation (±10°)
- Random zoom (±10%)

---

## Inference & Verification

The `compute_similarity` function compares any two handwriting images:

```python
score, percentage, verdict, confidence, distance = compute_similarity(path_A, path_B)
```

**Similarity Scoring:**

```
distance   = Euclidean distance between L2-normalised 128-dim embeddings
similarity = exp(−distance × 2)      # maps to [0, 1]
```

**Decision Rules:**

| Similarity | Verdict | Confidence |
|-----------|---------|------------|
| ≥ 0.90 | ✓ SAME WRITER | HIGH |
| 0.70 – 0.89 | ✓ SAME WRITER | MODERATE |
| < 0.70 | ✗ DIFFERENT WRITERS | LOW |

The decision threshold is automatically optimised on the validation set during training (`best_threshold`).

---

## Performance

| Metric | Value |
|--------|-------|
| **Target Accuracy** | 80%+ |
| **Typical Range** | 75% – 85% |

**Tuning guide:**

| Scenario | Recommended Action |
|----------|--------------------|
| Accuracy < 75% | Increase `max_samples_per_writer` to 10, `num_triplets` to 3000, patience to 15 |
| Accuracy 75–85% | Model is performing well — consider threshold fine-tuning |
| Accuracy > 85% | Production-ready — no further changes needed |

---

## Requirements

```
tensorflow >= 2.x
numpy
opencv-python (cv2)
matplotlib
scikit-learn
Pillow
kagglehub
google-colab
```

Install with:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn Pillow kagglehub -q
```

---

## How to Run

### 1. Open in Google Colab

Upload `model.ipynb` to [Google Colab](https://colab.research.google.com/) and enable GPU acceleration:  
`Runtime → Change runtime type → T4 GPU`

### 2. Run All Cells in Order

| Step | Description |
|------|-------------|
| 1 | Install required libraries |
| 2 | Import libraries, set seeds, verify GPU |
| 3 | Download IAM dataset via `kagglehub` |
| 4 | Load and explore dataset (505 writers, 3,903 images) |
| 5 | Preprocess images (resize to 384×384, normalise, CLAHE) |
| 6 | Generate triplet training pairs (anchor, positive, negative) |
| 7 | Build Siamese triplet model with 2-block CNN |
| 8 | Train with conservative settings (LR=8e-5, patience=12) |
| 9 | Plot training curves (loss over epochs) |
| 10 | Evaluate — find optimal threshold, compute accuracy/precision/recall/F1 |
| 11 | Run interactive writer verification interface |

### 3. Verify Two Handwriting Samples

After training, the interactive interface prompts you to upload two images:

```
Upload Sample A → select handwriting image
Upload Sample B → select handwriting image
→ System returns: Similarity %, Distance, Confidence, and Verdict
```

The result is rendered as a styled HTML card showing both images side by side with the verdict prominently displayed.

---

## Saved Files

| File | Description |
|------|-------------|
| `writer_identification_model.h5` | Full trained triplet model |
| `embedding_network.h5` | Embedding sub-network used for inference |
| `model_config.pkl` | Threshold and configuration saved for deployment |

---

## Project Structure

```
handwritten-writer-identification/
│
├── model.ipynb                    ← Main notebook (all steps)
├── embedding_network.h5           ← Saved embedding model
├── writer_identification_model.h5 ← Saved full model
├── model_config.pkl               ← Threshold + config
└── README.md
```

---

## Technical Optimisations

| Optimisation | Benefit |
|-------------|---------|
| 2-block CNN instead of 4-block | Prevents overfitting with limited per-writer data |
| GlobalAveragePooling instead of Flatten | Reduces parameters, improves generalisation |
| Triplet loss instead of contrastive loss | Better metric space geometry and class separation |
| Conservative LR (8e-5) + high patience (12) | Stable convergence, avoids premature stopping |
| 3–8 samples per writer enforced | Balances intra-writer variation with consistency |
| CLAHE preprocessing | Enhances contrast for variable scan/photo quality |
| Data augmentation (rotation, zoom) | Effectively triples training set size |

---

## Usage Notes

- **Training** is a one-time process (4–6 hours). The model saves automatically.
- **Inference** (Step 11) can be re-run repeatedly with different image pairs after training.
- **Image quality:** The system handles phone photos, scanned documents, ruled lined paper, and varying lighting conditions.
- **Deployment:** Save the `embedding_network.h5` and `model_config.pkl` for production use — only the embedding network is needed at inference time.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.19 / Keras |
| Computer Vision | OpenCV, Pillow |
| Data & Math | NumPy, scikit-learn |
| Visualisation | Matplotlib |
| Dataset Access | kagglehub |
| UI / Display | IPython HTML, base64 image rendering |
| Environment | Google Colab, T4 GPU |

---

## Disclaimer

This system is developed for academic and research purposes. Predictions carry an inherent uncertainty margin and should not be used as the sole basis for legal or forensic conclusions without expert review.
