# Banking Signature Authentication System
### Secure Login via Handwritten Signature Verification

A deep learning-based biometric authentication system that verifies user identity through handwritten signature comparison, modelled after banking-grade security workflows.

---

## Project Details

| Detail | Information |
|--------|-------------|
| **Author** | Sanusi Shafii |
| **Matriculation Number** | CSA\2023\27683 |
| **Email** | s.shafii27683@fudutsinma.edu.ng.com |
| **Phone** | +2349125006811 |
| **Department** | Computer Science and Information Technology |
| **Faculty** | Computing |
| **University** | Federal University Dutsin-Ma, Katsina State, Nigeria |
| **Project Duration** | November 2023 – November 2024 |
| **Project Type** | 4th Year Undergraduate Project |

---

## Overview

This system uses a **Siamese Convolutional Neural Network (CNN)** to authenticate users by comparing a submitted handwritten signature against enrolled reference signatures. Rather than classifying signatures into fixed categories, the model learns a similarity metric — determining whether two signatures belong to the same person.

---

## Key Features

- Real-time signature authentication via image upload
- Multi-user recognition and management
- Confidence score-based verification with a configurable similarity threshold
- Banking-style interactive UI built with `ipywidgets` and HTML
- Automated identity verification with clear VERIFIED / ACCESS DENIED feedback
- Model persistence — trained models and user databases are saved for reuse

---

## Model Architecture

The system uses a **Siamese Neural Network** with a shared CNN backbone:

**Feature Extractor (CNN Backbone)**
- 4 convolutional blocks: Conv2D → BatchNormalization → MaxPooling2D → Dropout
- Filter progression: 32 → 64 → 128 → 256
- Global Average Pooling followed by Dense layers (256 → 128 units)

**Similarity Computation**
- Both input signatures pass through the same shared network
- Cosine similarity is computed between the two resulting feature vectors
- A contrastive loss function is used during training

**Verification Decision**
- If the similarity score ≥ `SIMILARITY_THRESHOLD` (default: 0.65), the user is authenticated
- Otherwise, access is denied

---

## System Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 128 × 128 (grayscale) |
| Batch Size | 16 |
| Max Epochs | 30 |
| Learning Rate | 0.0001 |
| Similarity Threshold | 0.65 |
| Optimizer | Adam |
| Loss Function | Contrastive Loss |

---

## Dataset Structure

Signatures must be organised by user in the following folder structure. The folder name becomes the registered user's name.

```
signature_dataset/
    John_Doe/
        sig1.png
        sig2.png
        sig3.png
    Jane_Smith/
        sig1.png
        sig2.png
    Michael_Brown/
        sig1.png
        sig2.png
        sig3.png
```

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
opencv-python (cv2)
matplotlib
seaborn
scikit-learn
ipywidgets
google-colab (for Drive mounting and file upload)
```

---

## How to Run

1. **Open the notebook** in Google Colab.
2. **Mount Google Drive** and set `DATASET_PATH` to your signature dataset folder.
3. **Run all cells in order** (Sections 1–15).
4. The system will:
   - Load and preprocess all signatures
   - Generate training pairs (positive: same user, negative: different users)
   - Train the Siamese network with early stopping and learning rate scheduling
   - Save the trained model and user database
   - Launch the interactive authentication interface

### Authentication Interface

Once deployed, the interface allows you to:
1. Click **Upload Signature** to select a signature image
2. Click **Authenticate** to verify identity
3. View the result with confidence score, matched user name, and VERIFIED or ACCESS DENIED status

---

## Notebook Sections

| Section | Description |
|---------|-------------|
| 1 | Import libraries and verify GPU |
| 2 | Mount Google Drive and set dataset path |
| 3 | System configuration (image size, thresholds, seeds) |
| 4 | Load user signatures from dataset |
| 5 | Visualise sample signatures and user distribution |
| 6 | Create training pairs for Siamese network |
| 7 | Prepare train/validation splits |
| 8 | Build Siamese Neural Network |
| 9 | Configure training callbacks |
| 10 | Train the model |
| 11 | Plot training accuracy and loss curves |
| 12 | Save model and user database to disk |
| 13 | Evaluate on validation set (confusion matrix, classification report) |
| 14 | Launch interactive authentication interface |
| 15 | Display system performance summary |

---

## Saved Files

After training, the following files are saved:

| File | Description |
|------|-------------|
| `banking_signature_siamese.h5` | Full trained Siamese model |
| `banking_feature_extractor.h5` | CNN feature extractor sub-model |
| `user_database.pkl` | Registered users and their signature embeddings |
| `best_banking_signature_model.h5` | Best checkpoint saved during training |

---

## Performance Metrics

The system is evaluated on the validation set using:
- Accuracy
- Precision
- Recall
- F1 Score

Results are displayed in a formatted summary dashboard at the end of the notebook.

---

## Technologies Used

- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy, Pandas
- **Visualisation:** Matplotlib, Seaborn
- **Interactive UI:** ipywidgets, IPython HTML display
- **Environment:** Google Colab with GPU support
