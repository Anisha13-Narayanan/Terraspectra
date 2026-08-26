TerraSpectra 🌱🔬
Hyperspectral Tomato Disease Classification Using Deep Learning

TerraSpectra is a deep learning project for classifying tomato leaf conditions using hyperspectral imaging data.

The project uses hyperspectral cubes with 550 spectral bands and classifies samples into five categories using a 3D Convolutional Neural Network (3D-CNN).

🎯 Project Objective

The objective is to develop a deep learning system that can classify hyperspectral tomato samples into the following classes:

Alternaria alternata
Alternaria solani
Botrytis cinerea
Fusarium oxysporum
Healthy

The project investigates how consistent hyperspectral preprocessing and 3D-CNN models can be used for plant disease classification.

📂 Dataset

The dataset contains 5 classes with 8 hyperspectral .mat files per class.

Dataset structure
data/raw/tomato_hsi/
│
├── alternaria_alternata/
│   ├── Alternaria_alternata_1.mat
│   ├── ...
│   └── Alternaria_alternata_8.mat
│
├── alternaria_solani/
│   ├── Alternaria_solani_1.mat
│   └── ...
│
├── botrytis_cinerea/
│   ├── Botrytis_cinerea_1.mat
│   └── ...
│
├── fusarium_oxysporum/
│   ├── Fusarium_oxysporum_1.mat
│   └── ...
│
└── healthy/
    ├── Healthy_1.mat
    └── Healthy_8.mat
Total dataset
5 classes × 8 files = 40 hyperspectral files

A typical hyperspectral cube has the shape:

(140, 280, 550)

Where:

140 = Height
280 = Width
550 = Spectral bands
🧪 Dataset Split

The dataset was split at the original hyperspectral file level.

Files 1–6 → Training
File 7    → Validation
File 8    → Testing
Final split
Split	Files	Patches
Training	30	960
Validation	5	160
Testing	5	160

This ensures that the model is evaluated on hyperspectral files that were not used during training.

⚙️ Initial Preprocessing Pipeline

The initial preprocessing pipeline performed:

Raw .mat file
      ↓
Load hyperspectral cube
      ↓
Check NaN / Infinite values
      ↓
Min-Max normalization
      ↓
PCA: 550 spectral bands → 30 components
      ↓
Save processed .npy file

For example:

Original shape:
(140, 280, 550)

After PCA:
(140, 280, 30)

Initial PCA explained approximately 95%–99% variance, depending on the file.

⚠️ Initial Pipeline Issue

The first preprocessing pipeline fitted PCA independently for every .mat file.

Conceptually:

File 1 → Fit PCA → 30 components
File 2 → Fit different PCA → 30 components
File 3 → Fit different PCA → 30 components
...

This could make the PCA component channels inconsistent between different hyperspectral files.

The initial model achieved:

Best validation accuracy: 52.50%

However, final evaluation on the untouched test set produced:

Test accuracy: 23.13%

This indicated poor generalization and motivated a redesign of the preprocessing pipeline.

🔄 Improved Shared PCA Pipeline

A new preprocessing pipeline was implemented to ensure consistent feature representation.

New pipeline
Raw hyperspectral files
        ↓
Split files into Train / Validation / Test
        ↓
Fit StandardScaler using TRAIN files only
        ↓
Fit ONE shared PCA using TRAIN files only
        ↓
Save preprocessing models
        ↓
Transform Training data
Transform Validation data
Transform Test data
        ↓
Create patches
        ↓
Train 3D-CNN
Important

Validation and test data are transformed using the same scaler and PCA fitted on the training data.

They are not used to fit preprocessing models.

🧠 Shared Preprocessing Models

The following preprocessing models were created:

models/preprocessing/
├── shared_scaler.joblib
└── shared_pca30.joblib

These models were fitted using only the 30 training hyperspectral files.

🗂️ Shared PCA Processed Data

The improved processed data is stored separately:

data/processed_shared_pca/
│
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

Shared PCA preprocessing completed successfully:

Train files processed: 30
Validation files processed: 5
Test files processed: 5
🧩 Patch Creation

Each processed hyperspectral cube is divided into spatial patches.

Patch configuration
Patch size: 32 × 32
PCA components: 30
Stride: 32

Each patch has the shape:

(32, 32, 30)
Final dataset
TRAIN -> X: (960, 32, 32, 30)
VAL   -> X: (160, 32, 32, 30)
TEST  -> X: (160, 32, 32, 30)

The dataset is balanced:

5 classes
Equal number of patches per class within each split
✅ Dataset Validation

The shared PCA dataset was validated successfully.

Checks performed:

Correct patch dimensions
Matching X and y sample counts
No NaN values
No infinite values
All 5 classes present
Labels correctly encoded from 0 to 4
Balanced class distribution

Validation result:

TRAIN | Samples: 960 | Shape: (960, 32, 32, 30)
VAL   | Samples: 160 | Shape: (160, 32, 32, 30)
TEST  | Samples: 160 | Shape: (160, 32, 32, 30)

✓ ALL SHARED PCA DATASET VALIDATION CHECKS PASSED
✓ READY FOR FRESH MODEL TRAINING
🤖 3D-CNN Model

The project uses a 3D-CNN to learn spatial and spectral features jointly.

Input data is converted from:

(samples, 32, 32, 30)

to:

(samples, 32, 32, 30, 1)

for Conv3D.

Architecture
Input
  ↓
Conv3D (16 filters)
  ↓
Batch Normalization
  ↓
MaxPooling3D
  ↓
Dropout
  ↓
Conv3D (32 filters)
  ↓
Batch Normalization
  ↓
MaxPooling3D
  ↓
Dropout
  ↓
Conv3D (64 filters)
  ↓
Batch Normalization
  ↓
Global Average Pooling 3D
  ↓
Dense (64)
  ↓
Dropout
  ↓
Dense (5, Softmax)
📊 Current Experimental Results
Previous preprocessing pipeline
Best validation accuracy: 52.50%
Final test accuracy:      23.13%

The test results showed poor generalization across classes.

Shared train-fitted PCA pipeline

A fresh 3D-CNN was trained using the corrected preprocessing pipeline.

Current result:

Best training accuracy:   75.73%
Best validation accuracy: 50.00%
Best epoch:               8

Although the validation accuracy is slightly below the previous pipeline's 52.50%, the shared PCA pipeline uses a more consistent preprocessing methodology.

📁 Project Structure
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
│       ├── val/
│       └── test/
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
🚀 Current Project Status
Completed
 Dataset selection and organization
 Downloaded 40 hyperspectral .mat files
 Inspected hyperspectral data
 Automated multi-class preprocessing
 PCA dimensionality reduction
 Dataset split into train, validation and test
 Patch creation
 Dataset validation
 Initial 3D-CNN training
 Hybrid model experiments
 Initial test evaluation
 Identified preprocessing consistency issue
 Implemented train-fitted shared StandardScaler
 Implemented shared PCA with 30 components
 Reprocessed all train, validation and test files
 Recreated shared-PCA patches
 Validated the new dataset
 Trained a fresh Shared-PCA 3D-CNN
 Achieved 50.00% validation accuracy with the corrected pipeline
Current Stage
Shared PCA preprocessing completed
        ↓
Fresh 3D-CNN trained
        ↓
Best validation accuracy: 50.00%
        ↓
NEXT: Improve the Shared-PCA 3D-CNN
        ↓
Select final model using validation data
        ↓
Perform final evaluation on untouched test data
🛠️ Technologies Used
Python
NumPy
SciPy
Scikit-learn
TensorFlow / Keras
Pandas
Matplotlib
Joblib
▶️ Main Commands
Shared PCA preprocessing
python src\preprocess_shared_pca.py
Create patches
python src\create_patches_shared_pca.py
Validate dataset
python src\validate_shared_pca_dataset.py
Train Shared-PCA 3D-CNN
python src\train_shared_pca_3dcnn.py