# ================================================================
# CropGuard AI -- Multi-Crop Disease Detection Training Script
# NextGen Innovation Challenge 2026
# Run this entirely in Google Colab (GPU runtime recommended)
# Runtime > Change runtime type > T4 GPU
# ================================================================

# ================================================================
# CELL 1 -- Install dependencies
# ================================================================
# !pip install tensorflow tensorflow-datasets scikit-learn --quiet
# print("Dependencies ready.")

# ================================================================
# CELL 2 -- Imports and global configuration
# ================================================================

import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------------------------------
# Configuration -- all tunable values in one place
# ----------------------------------------------------------------
IMAGE_SIZE      = 224      # Standard MobileNetV2 input size
NUM_CHANNELS    = 3
INPUT_SHAPE     = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
BATCH_SIZE      = 32

PHASE1_EPOCHS   = 15       # Head-only training
PHASE1_LR       = 1e-3

PHASE2_EPOCHS   = 20       # Fine-tuning top layers
PHASE2_LR       = 1e-5     # Must stay very small
FINETUNE_LAYERS = 50       # Top MobileNetV2 layers to unfreeze

CHECKPOINT_PATH  = '/content/best_cropguard_model.keras'
FINAL_H5_PATH    = '/content/crop_disease_model.h5'
TFLITE_PATH      = '/content/model.tflite'
CLASS_JSON_PATH  = '/content/class_names.json'

print("Configuration set.")
print(f"  Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Phase 1      : {PHASE1_EPOCHS} epochs  lr={PHASE1_LR}")
print(f"  Phase 2      : {PHASE2_EPOCHS} epochs  lr={PHASE2_LR}")

# ================================================================
# CELL 3 -- Define the 15 target classes across 5 crop groups
# ================================================================
# PlantVillage raw label names exactly as they appear in tfds
RAW_CLASS_NAMES = [
    # Tomato (4 classes)
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___healthy',
    # Bell Pepper (2 classes)
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    # Maize / Corn (3 classes)
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    # Potato (3 classes)
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    # Grape / Vine (3 classes)
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
]

# Human-readable display names used in the app
DISPLAY_NAMES = [
    'Tomato - Early Blight',
    'Tomato - Late Blight',
    'Tomato - Leaf Mold',
    'Tomato - Healthy',
    'Bell Pepper - Bacterial Spot',
    'Bell Pepper - Healthy',
    'Maize - Common Rust',
    'Maize - Northern Leaf Blight',
    'Maize - Healthy',
    'Potato - Early Blight',
    'Potato - Late Blight',
    'Potato - Healthy',
    'Vine Crop - Black Rot',
    'Vine Crop - Black Measles',
    'Vine Crop - Healthy',
]

NUM_CLASSES = len(RAW_CLASS_NAMES)
assert len(DISPLAY_NAMES) == NUM_CLASSES

print(f"\nTotal classes : {NUM_CLASSES}")
for i, (raw, disp) in enumerate(zip(RAW_CLASS_NAMES, DISPLAY_NAMES)):
    print(f"  [{i:>2}]  {disp}")

# ================================================================
# CELL 4 -- Load PlantVillage and extract selected classes
# ================================================================
# We scan the full dataset once and extract only our 15 classes
# into fixed numpy arrays. This avoids the lazy-pipeline split
# leakage bug that caused previous models to collapse.

print("\nLoading PlantVillage dataset from TensorFlow Datasets...")
raw_dataset, ds_info = tfds.load(
    'plant_village',
    split='train',
    with_info=True,
    as_supervised=True
)

all_label_names = ds_info.features['label'].names
TOTAL_IMAGES    = ds_info.splits['train'].num_examples

# Build a map: tfds global index -> our local 0..14 index
raw_to_local = {
    all_label_names.index(raw): local_idx
    for local_idx, raw in enumerate(RAW_CLASS_NAMES)
    if raw in all_label_names
}

missing = [r for r in RAW_CLASS_NAMES if r not in all_label_names]
if missing:
    print(f"WARNING: These class names were not found in the dataset:")
    for m in missing:
        print(f"  {m}")

print(f"\nScanning {TOTAL_IMAGES:,} images. Collecting our 15 classes...")

all_images = []
all_labels = []

for step, (image, label) in enumerate(raw_dataset):
    global_idx = int(label.numpy())
    if global_idx in raw_to_local:
        img = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        all_images.append(img.numpy())
        all_labels.append(raw_to_local[global_idx])
    if (step + 1) % 10_000 == 0:
        print(f"  {step+1:,} / {TOTAL_IMAGES:,} scanned -- "
              f"{len(all_images)} matching images found", end='\r')

X = np.array(all_images, dtype=np.float32)
y = np.array(all_labels, dtype=np.int32)

# Shuffle before splitting
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# Print class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\nExtraction complete. Total: {len(X):,} images")
print("-" * 50)
print(f"  {'Class':<32}  {'Count':>6}")
print(f"  {'':->42}")
for idx, cnt in zip(unique, counts):
    print(f"  {DISPLAY_NAMES[idx]:<32}  {cnt:>6}")
print("-" * 50)

# ================================================================
# CELL 5 -- Stratified 70 / 15 / 15 split
# ================================================================
# Stratify=y guarantees all 15 classes appear in every split
# in the correct proportion.

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

AUTOTUNE        = tf.data.AUTOTUNE
STEPS_PER_EPOCH = (len(X_train) // BATCH_SIZE) * 3   # 3 augmented views


def augment_image(image, label):
    """
    Applies random spatial and colour augmentation to a training image.
    Never applied to validation or test images.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k     = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


ds_train = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train), seed=SEED)
    .map(augment_image, num_parallel_calls=AUTOTUNE)
    .repeat()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

ds_val = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

ds_test = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Data split complete.")
print(f"  Training   : {len(X_train):,}  |  Validation : {len(X_val):,}  "
      f"|  Test : {len(X_test):,}")
print(f"  Steps/epoch: {STEPS_PER_EPOCH}")

# ================================================================
# CELL 6 -- Visualise sample images per class
# ================================================================

SAMPLES = 2
collected = {i: [] for i in range(NUM_CLASSES)}
for idx, lbl in enumerate(y_train):
    cid = int(lbl)
    if len(collected[cid]) < SAMPLES:
        collected[cid].append(X_train[idx])
    if all(len(v) == SAMPLES for v in collected.values()):
        break

fig, axes = plt.subplots(NUM_CLASSES, SAMPLES,
                         figsize=(SAMPLES * 3.5, NUM_CLASSES * 3.0))
fig.suptitle('Sample Training Images per Class',
             fontsize=14, fontweight='bold', y=1.01)

for row in range(NUM_CLASSES):
    for col in range(SAMPLES):
        if col < len(collected[row]):
            axes[row][col].imshow(collected[row][col], interpolation='bilinear')
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
        if col == 0:
            axes[row][col].set_ylabel(DISPLAY_NAMES[row], fontsize=8,
                                      fontweight='bold', labelpad=5)

plt.tight_layout()
plt.savefig('/content/sample_images.png', dpi=100, bbox_inches='tight')
plt.show()
print("Figure saved: /content/sample_images.png")

# ================================================================
# CELL 7 -- Build the MobileNetV2 transfer learning model
# ================================================================

def build_model(input_shape, num_classes):
    """
    Constructs a MobileNetV2-based multi-class disease classifier.

    The base model is fully frozen during Phase 1 (head-only training).
    Phase 2 (Cell 9) unfreezes the top FINETUNE_LAYERS layers.

    Args:
        input_shape : Tuple (H, W, C). Use (224, 224, 3).
        num_classes : Integer. Number of disease classes (15).

    Returns:
        Tuple of (compiled keras.Model, MobileNetV2 base model).
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False   # Freeze entire base for Phase 1

    inputs = keras.Input(shape=input_shape, name='leaf_image')

    # MobileNetV2 preprocessing: scales [0,1] images to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    # Frozen MobileNetV2 feature extraction
    x = base(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.45, name='drop1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.35, name='drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='CropGuard_Classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    return model, base


model, base_model = build_model(INPUT_SHAPE, NUM_CLASSES)

total_params     = model.count_params()
trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)

print(f"Model built: {model.name}")
print(f"  MobileNetV2 layers    : {len(base_model.layers)}")
print(f"  Total parameters      : {total_params:,}")
print(f"  Trainable (Phase 1)   : {trainable_params:,}  (head only)")
print(f"  Frozen (base)         : {total_params - trainable_params:,}")

# ================================================================
# CELL 8 -- Phase 1: Train classification head only
# ================================================================
# Goal: get the head to a stable starting point using fixed
# pretrained MobileNetV2 features as input.

early_stop_p1 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=5, restore_best_weights=True, verbose=1
)
checkpoint_p1 = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor='val_accuracy', mode='max',
    save_best_only=True, verbose=1
)

print("Phase 1 -- Head-only training")
print(f"  Base frozen : ALL {len(base_model.layers)} layers")
print(f"  Learning rate : {PHASE1_LR}")
print("=" * 60)

history_p1 = model.fit(
    ds_train,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=ds_val,
    epochs=PHASE1_EPOCHS,
    callbacks=[early_stop_p1, checkpoint_p1],
    verbose=1
)

print("=" * 60)
best_p1_acc = max(history_p1.history['val_accuracy'])
print(f"Phase 1 complete. Best val_accuracy : {best_p1_acc * 100:.2f}%")

# ================================================================
# CELL 9 -- Phase 2: Fine-tune top MobileNetV2 layers
# ================================================================
# Gently adapt the top layers of the pretrained base to the
# leaf disease domain. Learning rate MUST be very small (1e-5)
# to avoid destroying the pretrained ImageNet weights.

base_model.trainable = True
for layer in base_model.layers[:-FINETUNE_LAYERS]:
    layer.trainable = False

trainable_now = sum(tf.size(w).numpy() for w in model.trainable_weights)
unfrozen      = sum(1 for l in base_model.layers if l.trainable)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

print("Phase 2 -- Fine-tuning")
print(f"  Unfrozen base layers  : {unfrozen}")
print(f"  Trainable parameters  : {trainable_now:,}")
print(f"  Learning rate         : {PHASE2_LR}")
print("=" * 60)

early_stop_p2 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=8, restore_best_weights=True, verbose=1
)
reduce_lr_p2 = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1
)
checkpoint_p2 = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor='val_accuracy', mode='max',
    save_best_only=True, verbose=1
)

history_p2 = model.fit(
    ds_train,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=ds_val,
    epochs=PHASE2_EPOCHS,
    callbacks=[early_stop_p2, reduce_lr_p2, checkpoint_p2],
    verbose=1
)

print("=" * 60)
best_p2_acc = max(history_p2.history['val_accuracy'])
print(f"Phase 2 complete. Best val_accuracy : {best_p2_acc * 100:.2f}%")

# ================================================================
# CELL 10 -- Plot learning curves
# ================================================================

def join(h1, h2):
    return {k: h1[k] + h2.get(k, []) for k in h1}

hist   = join(history_p1.history, history_p2.history)
p1_end = len(history_p1.history['accuracy'])
epochs = range(1, len(hist['accuracy']) + 1)

fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History  |  CropGuard AI',
             fontsize=14, fontweight='bold')

ax_acc.plot(epochs, hist['accuracy'],     color='#1e7a1e', lw=2, label='Training')
ax_acc.plot(epochs, hist['val_accuracy'], color='#6dbf6d', lw=2, ls='--', label='Validation')
ax_acc.axvline(p1_end + 0.5, color='navy', lw=1.5, ls=':', label='Phase 1 | 2')
ax_acc.set_title('Accuracy Over Epochs', fontsize=12, fontweight='bold')
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_ylim(0, 1.05)
ax_acc.legend()
ax_acc.grid(axis='y', alpha=0.3)

ax_loss.plot(epochs, hist['loss'],     color='#b03030', lw=2, label='Training')
ax_loss.plot(epochs, hist['val_loss'], color='#e08080', lw=2, ls='--', label='Validation')
ax_loss.axvline(p1_end + 0.5, color='navy', lw=1.5, ls=':')
ax_loss.set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()
ax_loss.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/content/training_history.png', dpi=120, bbox_inches='tight')
plt.show()

# ================================================================
# CELL 11 -- Evaluate on test set
# ================================================================

best_model = keras.models.load_model(CHECKPOINT_PATH)
test_loss, test_acc, test_top3 = best_model.evaluate(ds_test, verbose=0)

print("Test Set Results")
print("-" * 38)
print(f"  Accuracy       : {test_acc  * 100:.2f}%")
print(f"  Top-3 Accuracy : {test_top3 * 100:.2f}%")
print(f"  Loss           : {test_loss:.4f}")

# Collect predictions for confusion matrix
probs_all = best_model.predict(X_test, verbose=0)
y_pred    = np.argmax(probs_all, axis=1)

print("\nClassification Report")
print("-" * 60)
print(classification_report(
    y_test, y_pred,
    target_names=[n.replace(' - ', '\n  ') for n in DISPLAY_NAMES]
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
short_names = [n.split(' - ')[1] if ' - ' in n else n for n in DISPLAY_NAMES]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names,
            linewidths=0.4, annot_kws={'size': 9}, ax=ax)
ax.set_title('Confusion Matrix  |  Test Set', fontsize=13,
             fontweight='bold', pad=14)
ax.set_ylabel('True Label', fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()

# ================================================================
# CELL 12 -- Save model as .h5 and export to TFLite
# ================================================================

# -- Save full Keras model (.h5 for Streamlit app) ---------------
best_model.save(FINAL_H5_PATH)
print(f"Keras model saved : {FINAL_H5_PATH}")

# -- Export to TFLite (for edge / offline mobile deployment) -----
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

# Apply dynamic range quantization to reduce file size ~4x
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

h5_size     = os.path.getsize(FINAL_H5_PATH)    / 1024 / 1024
tflite_size = os.path.getsize(TFLITE_PATH)       / 1024 / 1024

print(f"TFLite model saved: {TFLITE_PATH}")
print(f"  .h5     size : {h5_size:.1f} MB")
print(f"  .tflite size : {tflite_size:.1f} MB  (quantized)")

# -- Save class names as JSON ------------------------------------
class_data = {
    'raw_names':     RAW_CLASS_NAMES,
    'display_names': DISPLAY_NAMES,
    'num_classes':   NUM_CLASSES,
    'image_size':    IMAGE_SIZE
}
with open(CLASS_JSON_PATH, 'w') as f:
    json.dump(class_data, f, indent=2)

print(f"Class names saved : {CLASS_JSON_PATH}")

# ================================================================
# CELL 13 -- Download all outputs
# ================================================================

from google.colab import files

for path in [FINAL_H5_PATH, TFLITE_PATH, CLASS_JSON_PATH,
             '/content/training_history.png',
             '/content/confusion_matrix.png']:
    if os.path.exists(path):
        files.download(path)
        print(f"Downloading: {path}")

print("\nAll files downloaded. Upload them to your Hugging Face Space.")
