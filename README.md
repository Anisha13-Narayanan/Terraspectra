
# TerraSpectra 🌱🔬

## Hyperspectral Tomato Disease Classification Using Deep Learning

TerraSpectra is a deep learning project for classifying tomato leaf diseases using hyperspectral imaging data.

The project uses hyperspectral cubes containing **550 spectral bands** and classifies samples into five categories using deep learning, primarily a **3D Convolutional Neural Network (3D-CNN)**.

---

## 🎯 Project Objective

The objective of TerraSpectra is to develop a deep learning system capable of classifying hyperspectral tomato samples into the following five classes:

1. Alternaria alternata
2. Alternaria solani
3. Botrytis cinerea
4. Fusarium oxysporum
5. Healthy

The project focuses on using consistent hyperspectral preprocessing and deep learning to learn both spatial and spectral features.

---

# 📂 Dataset

The dataset contains **5 classes**, with **8 hyperspectral `.mat` files per class**.

### Dataset Classes

```text
1. Alternaria alternata
2. Alternaria solani
3. Botrytis cinerea
4. Fusarium oxysporum
5. Healthy
````

### Total Dataset

```text
5 classes × 8 files = 40 hyperspectral files
```

A typical hyperspectral cube has the following shape:

```text
(140, 280, 550)
```

Where:

* `140` = Image height
* `280` = Image width
* `550` = Spectral bands

---

# 📊 Dataset Split

The dataset is split at the **original hyperspectral file level**.

```text
Files 1–6 → Training
File 7    → Validation
File 8    → Testing
```

### Final Split

| Dataset    | Number of Files | Number of Patches |
| ---------- | --------------: | ----------------: |
| Training   |              30 |               960 |
| Validation |               5 |               160 |
| Testing    |               5 |               160 |

This keeps test files separate from training files.

---

# ⚙️ Initial Preprocessing Pipeline

The initial preprocessing pipeline performed:

```text
Raw .mat File
      ↓
Load Hyperspectral Cube
      ↓
Check NaN / Infinite Values
      ↓
Normalization
      ↓
PCA: 550 Bands → 30 Components
      ↓
Save Processed .npy File
```

Example transformation:

```text
Original:
(140, 280, 550)

After PCA:
(140, 280, 30)
```

The initial PCA preprocessing retained approximately **95%–99% of variance**, depending on the file.

---

# ⚠️ Initial Pipeline Issue

The original preprocessing pipeline fitted PCA independently for every `.mat` file.

Conceptually:

```text
File 1 → Fit PCA → 30 Components
File 2 → Fit Different PCA → 30 Components
File 3 → Fit Different PCA → 30 Components
...
```

This could result in inconsistent PCA feature representations between files.

The previous experimental model achieved:

```text
Best Validation Accuracy: 52.50%
```

However, evaluation on the held-out test set produced:

```text
Final Test Accuracy: 23.13%
```

This showed poor generalization and motivated an improved preprocessing pipeline.

---

# 🔄 Improved Shared PCA Pipeline

A new preprocessing pipeline was implemented using preprocessing models fitted only on the training data.

## New Pipeline

```text
Raw Hyperspectral Files
        ↓
Split into Train / Validation / Test
        ↓
Fit StandardScaler on TRAIN Files Only
        ↓
Fit ONE Shared PCA on TRAIN Files Only
        ↓
Save Scaler and PCA Models
        ↓
Transform Training Data
Transform Validation Data
Transform Test Data
        ↓
Create Patches
        ↓
Train 3D-CNN
```

### Important

The validation and test datasets are **not used to fit the scaler or PCA**.

They use the same preprocessing models fitted using the training data.

---

# 🧠 Shared Preprocessing Models

The following preprocessing models were created:

```text
models/
└── preprocessing/
    ├── shared_scaler.joblib
    └── shared_pca30.joblib
```

These preprocessing models were fitted using only the **30 training hyperspectral files**.

---

# 📁 Shared PCA Processed Dataset

The improved processed dataset is stored separately from the old preprocessing output:

```text
data/
├── processed/
│
└── processed_shared_pca/
    ├── train/
    │   ├── alternaria_alternata/
    │   ├── alternaria_solani/
    │   ├── botrytis_cinerea/
    │   ├── fusarium_oxysporum/
    │   └── healthy/
    │
    ├── val/
    │   ├── alternaria_alternata/
    │   ├── alternaria_solani/
    │   ├── botrytis_cinerea/
    │   ├── fusarium_oxysporum/
    │   └── healthy/
    │
    └── test/
        ├── alternaria_alternata/
        ├── alternaria_solani/
        ├── botrytis_cinerea/
        ├── fusarium_oxysporum/
        └── healthy/
```

Shared preprocessing completed successfully:

```text
Train files processed: 30
Validation files processed: 5
Test files processed: 5
```

---

# 🧩 Patch Creation

Each processed hyperspectral cube is divided into spatial patches.

### Patch Configuration

```text
Patch Size: 32 × 32
Stride: 32
Spectral/PCA Components: 30
```

Each patch has the shape:

```text
(32, 32, 30)
```

### Final Shared PCA Dataset

```text
TRAIN -> X: (960, 32, 32, 30), y: (960,)
VAL   -> X: (160, 32, 32, 30), y: (160,)
TEST  -> X: (160, 32, 32, 30), y: (160,)
```

---

# ✅ Dataset Validation

The shared PCA dataset was successfully validated.

Validation checks included:

* Correct patch dimensions
* Matching feature and label sample counts
* No NaN values
* No infinite values
* All 5 classes present
* Labels encoded correctly from `0` to `4`
* Balanced class distribution

Final validation:

```text
TRAIN | Samples: 960 | Shape: (960, 32, 32, 30)
VAL   | Samples: 160 | Shape: (160, 32, 32, 30)
TEST  | Samples: 160 | Shape: (160, 32, 32, 30)

✓ ALL SHARED PCA DATASET VALIDATION CHECKS PASSED
✓ READY FOR FRESH MODEL TRAINING
```

---

# 🤖 3D-CNN Model

The project uses a 3D Convolutional Neural Network to learn spatial and spectral features jointly.

The patch data is converted from:

```text
(samples, 32, 32, 30)
```

to:

```text
(samples, 32, 32, 30, 1)
```

for use with `Conv3D`.

### Model Architecture

```text
Input
  ↓
Conv3D (16 Filters)
  ↓
Batch Normalization
  ↓
MaxPooling3D
  ↓
Dropout
  ↓
Conv3D (32 Filters)
  ↓
Batch Normalization
  ↓
MaxPooling3D
  ↓
Dropout
  ↓
Conv3D (64 Filters)
  ↓
Batch Normalization
  ↓
GlobalAveragePooling3D
  ↓
Dense (64)
  ↓
Dropout
  ↓
Dense (5, Softmax)
```

---

# 📈 Experimental Results

## Initial Pipeline

```text
Best Validation Accuracy: 52.50%
Final Test Accuracy:      23.13%
```

The held-out test results showed poor generalization.

### Test Classification Results

```text
Test Accuracy: 23.13%
```

Some classes had zero recall, while the model predicted `Botrytis cinerea` much more frequently than other classes.

This motivated the implementation of a shared train-fitted preprocessing pipeline.

---

## Shared PCA Pipeline

A fresh 3D-CNN was trained using the corrected shared preprocessing pipeline.

Current best result:

```text
Best Training Accuracy:   75.73%
Best Validation Accuracy: 50.00%
Best Epoch:               8
```

Although this validation result is slightly below the previous 52.50% validation result, the shared PCA pipeline provides a more consistent preprocessing methodology because the scaler and PCA are fitted using training data only.

---

# 📁 Project Structure

```text
Terraspectra/
│
├── data/
│   ├── raw/
│   │   └── tomato_hsi/
│   │
│   ├── processed/
│   │
│   ├── processed_shared_pca/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── patches_shared_pca/
│       ├── train/
│       │   ├── X_train.npy
│       │   └── y_train.npy
│       ├── val/
│       │   ├── X_val.npy
│       │   └── y_val.npy
│       └── test/
│           ├── X_test.npy
│           └── y_test.npy
│
├── models/
│   ├── preprocessing/
│   │   ├── shared_scaler.joblib
│   │   └── shared_pca30.joblib
│   │
│   ├── best_shared_pca_3dcnn.keras
│   └── final_shared_pca_3dcnn.keras
│
├── results/
│   ├── training_history_shared_pca_3dcnn.csv
│   ├── classification_report_3dcnn.csv
│   ├── confusion_matrix_3dcnn.csv
│   ├── confusion_matrix_3dcnn.png
│   └── final_test_results_3dcnn.txt
│
├── src/
│   ├── inspect_data.py
│   ├── preprocess_data.py
│   ├── split_dataset.py
│   ├── create_patches.py
│   ├── validate_dataset.py
│   │
│   ├── preprocess_shared_pca.py
│   ├── create_patches_shared_pca.py
│   ├── validate_shared_pca_dataset.py
│   └── train_shared_pca_3dcnn.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Current Project Status

## Completed

* [x] Dataset selection
* [x] Dataset organization
* [x] Downloaded 40 hyperspectral `.mat` files
* [x] Inspected hyperspectral data
* [x] Automated multi-class preprocessing
* [x] PCA dimensionality reduction from 550 bands to 30 components
* [x] Source-file-level dataset splitting
* [x] Patch creation
* [x] Dataset validation
* [x] Initial 3D-CNN training
* [x] Hybrid model experiments
* [x] Initial held-out test evaluation
* [x] Identified preprocessing consistency issue
* [x] Implemented shared train-fitted StandardScaler
* [x] Implemented shared train-fitted PCA
* [x] Reprocessed all 40 files
* [x] Recreated train, validation, and test patches
* [x] Validated the shared PCA dataset
* [x] Trained a fresh Shared-PCA 3D-CNN
* [x] Achieved 50.00% best validation accuracy with the shared preprocessing pipeline

## Current Stage

```text
Shared PCA Preprocessing Completed
        ↓
Shared PCA Patches Created
        ↓
Dataset Validated
        ↓
Fresh 3D-CNN Trained
        ↓
Best Validation Accuracy: 50.00%
        ↓
NEXT: Improve Shared-PCA 3D-CNN
        ↓
Select Final Model Using Validation Data
        ↓
Final Evaluation on Untouched Test Data
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* SciPy
* Scikit-learn
* TensorFlow
* Keras
* Pandas
* Matplotlib
* Joblib

---

# ▶️ How to Run

## 1. Shared PCA Preprocessing

```powershell
python src\preprocess_shared_pca.py
```

## 2. Create Shared PCA Patches

```powershell
python src\create_patches_shared_pca.py
```

## 3. Validate Dataset

```powershell
python src\validate_shared_pca_dataset.py
```

## 4. Train Shared PCA 3D-CNN

```powershell
python src\train_shared_pca_3dcnn.py
```

---

# 🔜 Next Steps

1. Improve the Shared-PCA 3D-CNN using training and validation data only.
2. Compare experimental models based on validation performance.
3. Select the best model.
4. Evaluate the selected model once on the untouched test set.
5. Generate final evaluation metrics and visualizations.
6. Prepare the final project report and presentation.

---

## 📌 Current Best Shared-PCA Result

```text
Best Training Accuracy:   75.73%
Best Validation Accuracy: 50.00%
Best Epoch:               8
```

**Project Status: In Progress 🚀**







