# TerraSpectra Dataset Plan

## Stage A: Model Development
Use a manageable hyperspectral benchmark dataset to develop and test:
- Data loading
- Normalization
- PCA
- Train/validation/test split
- Hyperspectral patch creation
- 3D-CNN
- Vision Transformer integration

## Stage B: Geospatial Pipeline
Use a large hyperspectral raster/cube dataset to implement:
- Rasterio ingestion
- Large-file handling
- Tiling
- Chunked inference
- Prediction heatmap generation

## Target
Hyperspectral crop health analysis and early disease/stress prediction.