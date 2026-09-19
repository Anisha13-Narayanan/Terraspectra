# TerraSpectra Presentation Outline

## Slide 1 — Problem

RGB monitoring often observes visible symptoms late. Hyperspectral cubes carry
hundreds of spectral bands that can capture plant-condition differences.

## Slide 2 — System Architecture

```text
MAT / HDF5 / GeoTIFF cube
        ↓
FastAPI ingestion + validation
        ↓
Training-fitted StandardScaler + shared PCA (550 → 30)
        ↓
32 × 32 spatial tiling → calibrated Shared-PCA 3D-CNN
        ↓
Patch predictions + WGS84 polygons + acreage summary
        ↓
React / Deck.gl disease-risk map and report export
```

## Slide 3 — Data Integrity

- 40 original cubes across five classes.
- Source-file split: 30 training, 5 validation, 5 untouched test cubes.
- Scaler and PCA fitted only on training data.

## Slide 4 — Model Choice

The production API serves the Shared-PCA 3D-CNN. It tied the strongest
source-level validation score and had the strongest existing held-out patch
evaluation. The PyTorch 3D-CNN + Transformer is retained as an experiment.

## Slide 5 — Evaluation

- Held-out patch accuracy: 46.88%; macro F1: 0.3725.
- Source-level validation: 60.00% accuracy; macro F1: 0.50.
- Source-level test: 40.00% accuracy.

Explain that source-level results are more conservative because patches from
the same cube are correlated.

## Slide 6 — Live Product Demo

Show file upload, the disease overlay, hover tooltip, risk/acreage analytics,
timeline, and downloaded JSON report.

## Slide 7 — Limitations and Next Research Step

It is not yet a validated early-warning forecast. The next research phase is
collecting independent, longitudinal, pre-symptom field observations and
running source-level cross-validation.
