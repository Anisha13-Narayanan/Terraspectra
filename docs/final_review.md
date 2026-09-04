# TerraSpectra Final Review

## Delivered system

TerraSpectra accepts hyperspectral MAT, HDF5, and GeoTIFF data; applies the
training-fitted shared scaler and PCA; tiles cubes into 32×32 patches; and
returns calibrated disease classifications through FastAPI. The React/Deck.gl
dashboard renders patch polygons, probabilities, risk counts, geodesic acreage
for georeferenced TIFFs, and a dated in-session analysis timeline.

## Evaluation evidence

| Evaluation | Shared-PCA 3D-CNN result |
| --- | ---: |
| Held-out patch accuracy | 46.88% |
| Held-out patch macro F1 | 0.3725 |
| Source-level validation accuracy | 60.00% (3 of 5 cubes) |
| Source-level validation macro F1 | 0.50 |
| Source-level test accuracy | 40.00% (2 of 5 cubes) |

The source-file split has 30 train, 5 validation, and 5 test cubes with no
filename overlap. Source-level metrics are essential because a cube's patches
are correlated.

## Deployment decision

The served model is the TensorFlow Shared-PCA 3D-CNN. It tied the strongest
source-level validation result and has the strongest existing held-out
patch-level result. The PyTorch hybrid remains an experimental baseline. See
`models/deployment_manifest.json` and `GET /model-info`.

## Operational safeguards

- GeoTIFFs are read as Rasterio windows; PCA and inference are batched.
- Uploads have a configurable `TERRASPECTRA_MAX_UPLOAD_BYTES` limit (5 GiB by
  default).
- Set `TERRASPECTRA_API_KEY` to require an `X-API-Key` header on inference
  endpoints.
- The API logs completed inference filename and patch-count events.

## Limitations

This project is a disease-classification prototype, not validated evidence of
pre-symptom disease forecasting. The small number of source cubes prevents
strong generalization claims. A production rollout requires more independent,
time-labelled field data, source-level cross-validation, monitoring, and an
external authentication/deployment layer.
