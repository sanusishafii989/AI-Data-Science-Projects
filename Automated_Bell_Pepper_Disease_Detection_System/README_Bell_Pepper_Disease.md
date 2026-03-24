#  Bell Pepper Disease Detection using MobileNetV2

A deep learning project for classifying bell pepper leaf diseases from images using transfer learning with MobileNetV2. The system identifies whether a plant is **healthy** or **diseased**, enabling early detection to support precision agriculture.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Overview

Early detection of plant disease is critical for reducing crop loss. This notebook trains a **MobileNetV2-based transfer learning model** on bell pepper leaf images to classify disease states with high accuracy. The pretrained ImageNet weights give the model a strong visual feature foundation, while fine-tuning adapts it to the specific characteristics of bell pepper leaves.

An **interactive HTML dashboard** is included so users can upload their own leaf images and get instant predictions.

---

##  Features

- **Transfer learning** with MobileNetV2 pretrained on ImageNet
- **Automatic data augmentation** — rotation, zoom, flip, shear, brightness adjustments
- **Training callbacks** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Comprehensive evaluation** — classification report, confusion matrix, accuracy/loss curves
- **Interactive prediction dashboard** — upload any leaf image and get a real-time diagnosis
- **Google Drive integration** — loads dataset directly from Drive on Colab

---

##  Model Architecture

| Layer | Details |
|---|---|
| Base Model | MobileNetV2 (ImageNet weights, frozen) |
| Input Shape | 224 × 224 × 3 |
| Pooling | Global Average Pooling (built-in) |
| BatchNormalization | After base model |
| Dropout | 0.5 |
| Dense | 128 units, ReLU |
| BatchNormalization | — |
| Dropout | 0.3 |
| Output | Softmax (N classes) |

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Metrics:** Accuracy

---

##  Dataset Structure

Place your dataset on Google Drive in the following structure:

```
dataset/
├── Healthy/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── Diseased/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Images are resized to **224 × 224** automatically
- The notebook auto-detects all subdirectories as class labels

>  The dataset used is the **Bell Pepper Leaf Disease Dataset** available on Kaggle (PlantVillage subset).

---

##  Installation

Run in **Google Colab** (GPU runtime recommended):

```python
!pip install -q tensorflow opencv-python matplotlib seaborn pandas numpy scikit-learn pillow
```

**Update the dataset path** in the configuration cell:

```python
DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/ml2-main/dataset'
```

---

##  Usage

1. **Open** the notebook in Google Colab
2. **Set runtime** to GPU (`Runtime → Change runtime type → GPU`)
3. **Mount Google Drive** and verify your dataset path
4. **Run all cells** in order
5. **Use the interactive dashboard** (Cell 13) to upload a leaf image for prediction

---

##  How It Works

### 1. Data Exploration
The notebook automatically scans the dataset folder, counts images per class, and displays sample images from each category.

### 2. Data Augmentation
Training images are augmented on-the-fly using `ImageDataGenerator`:
- Rotation (±20°)
- Width/height shift (±20%)
- Zoom (±20%)
- Horizontal flip
- Brightness adjustment

Validation images are only rescaled (no augmentation).

### 3. Transfer Learning
- **MobileNetV2** base is loaded with ImageNet weights and frozen
- Custom classification head is added on top
- Only the new layers are trained in the initial phase

### 4. Training
- Batch size: 32 | Image size: 224×224 | Default epochs: 5
- `EarlyStopping` halts training when `val_loss` stops improving (patience = 5)
- `ReduceLROnPlateau` reduces learning rate by 50% when stuck (patience = 3)
- Best weights restored automatically

### 5. Evaluation
- Generates confusion matrix and per-class classification report
- Plots training/validation accuracy and loss curves

### 6. Interactive Prediction
- Upload any `.jpg`/`.png` leaf image via the dashboard
- The `BellPepperDiseaseDetector` class preprocesses the image and returns:
  - Predicted class (e.g., Healthy / Diseased)
  - Confidence percentage
  - Visual result card with color coding

---

##  Results

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Base Model | MobileNetV2 (ImageNet) |
| Training Epochs | 5 (with EarlyStopping) |
| Validation Split | 20% |
| Optimizer | Adam |

*Accuracy results depend on dataset size and quality.*

---

##  Project Structure

```
Bell_Pepper_Project.ipynb
│
├── Cell 1  — Imports & library setup
├── Cell 2  — Mount Google Drive & set dataset path
├── Cell 3  — Dataset exploration & class statistics
├── Cell 4  — Display sample images per class
├── Cell 5  — Data preprocessing & augmentation setup
├── Cell 6  — Visualize augmented images
├── Cell 7  — Build MobileNetV2 model
├── Cell 8  — Visualize model architecture
├── Cell 9  — Define training callbacks
├── Cell 10 — Train the model
├── Cell 11 — Plot training history (accuracy & loss)
├── Cell 12 — Evaluate model (confusion matrix, report)
├── Cell 13 — BellPepperDiseaseDetector class
├── Cell 14 — Interactive HTML prediction dashboard
└── Cell 15 — Test with dataset sample images
```
---
##  Author

**Sanusi Shafii**  
Matric No: CSA/2023/27683  
Email: s.shafii27683@fudutsinma.edu.ng


---

##  License

This project is intended for educational and research purposes.
