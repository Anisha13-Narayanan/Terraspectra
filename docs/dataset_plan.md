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

## Stage B: Geospatial Satellite-Farm Pipeline
Use multi-date satellite or explicitly labelled mock hyperspectral rasters to implement:
- Rasterio ingestion
- Large-file handling
- Tiling
- Chunked inference
- Prediction heatmap generation

## Stage C: Forecast Validation

Use independent, time-labelled field/satellite observations to validate whether
spectral risk precedes visually confirmed disease onset. Do not make an
early-forecasting claim from the benchmark or synthetic data alone.

## Target
Hyperspectral crop health analysis, spatial disease-risk mapping, and an
eventual evidence-backed early disease/stress forecasting workflow.
