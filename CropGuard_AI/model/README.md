---
title: CropGuard AI
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: true
license: mit
---

# 🌿 CropGuard AI — Offline Crop Disease Detection

**NextGen Innovation Challenge 2026 | Nigeria**

CropGuard AI is an offline-first crop disease identification system that gives
Nigerian farmers instant AI-powered diagnosis and treatment advice directly
from a smartphone browser.

## Features

- 📸 Upload a leaf photo or use your phone camera
- 🤖 AI diagnosis in under 2 seconds
- 💊 Actionable treatment steps in plain language
- 📍 Optional GPS logging to track disease spread
- 🗺️ Map view of all logged detections
- 📡 Works offline after initial model load

## Supported Crops & Diseases

| Crop | Conditions Detected |
|------|-------------------|
| 🍅 Tomato | Early Blight, Late Blight, Leaf Mold, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🌽 Maize | Common Rust, Northern Leaf Blight, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍇 Vine Crops | Black Rot, Black Measles, Healthy |

## Technology

- **Model**: MobileNetV2 (ImageNet pretrained) fine-tuned on PlantVillage
- **Framework**: TensorFlow / Keras
- **App**: Streamlit
- **Edge model**: TFLite (quantized) for Android/iOS deployment

## How to Use

1. Upload a clear photo of a single crop leaf
2. Wait for the AI diagnosis (< 2 seconds)
3. Read the diagnosis and confidence score
4. Follow the treatment advice
5. Optionally log the case with GPS coordinates

---

*Built for Nigerian farmers. Powered by AI.*
