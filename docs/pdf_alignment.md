# TerraSpectra Alignment with the Project Brief

TerraSpectra now implements the project brief as a **mock satellite-farm
prototype** alongside its original tomato hyperspectral benchmark.

## What is implemented

- Rasterio GeoTIFF/HDF5 ingestion, shared PCA, windowed tiling, and batched API inference.
- React + Deck.gl GIS heatmap with WGS84 patch polygons and acreage-at-risk.
- A relative PCA-space spectral-anomaly signal per patch. It is not labelled as
  a specific chemical measurement without wavelength calibration.
- A dated in-session timeline and a temporal-analysis endpoint.
- PyTorch 3D-CNN + Transformer research baseline plus a deployed 3D-CNN chosen
  from current validation evidence.
- GPU/tensor resource audit script: `src/audit_training_resources.py`.
- Device-aware PyTorch 3D-CNN + Transformer training with best-validation
  checkpointing and epoch/resource history: `src/pytorch_hybrid.py`.
- Synthetic multi-date 550-band farm GeoTIFF generator:
  `src/create_mock_satellite_farm.py`.

## Demo data workflow

```powershell
.venv\Scripts\python.exe src\create_mock_satellite_farm.py
.venv\Scripts\python.exe src\audit_training_resources.py
.venv\Scripts\python.exe src\pytorch_hybrid.py --epochs 20 --device auto
```

Upload the generated scenes from `data/mock_satellite_farm/` one at a time,
using their date in the dashboard's observation-date field. This demonstrates
ingestion, GIS mapping, anomalies, acreage, and temporal comparison.

## Scientific boundary

The mock scenes are synthetic and are tagged as such. They validate engineering
workflow only. The system cannot claim a real three-week disease forecast until
it is trained and evaluated on independently collected, time-labelled satellite
or field hyperspectral observations with disease-onset ground truth.
