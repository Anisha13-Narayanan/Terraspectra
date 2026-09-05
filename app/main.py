from pathlib import Path
from typing import Any
from io import BytesIO
import json
import os
import tempfile
import logging


import numpy as np
import pandas as pd
import joblib
import scipy.io
import tensorflow as tf
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.geospatial import load_cube


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_shared_pca_3dcnn.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing" / "shared_scaler.joblib"
PCA_PATH = PROJECT_ROOT / "models" / "preprocessing" / "shared_pca30.joblib"
CALIBRATION_PATH = PROJECT_ROOT / "models" / "preprocessing" / "temperature_calibration.json"
MODEL_MANIFEST_PATH = PROJECT_ROOT / "models" / "deployment_manifest.json"
TEMPORAL_ANALYSIS_PATH = (
    PROJECT_ROOT / "data" / "temporal_prediction_analysis.csv"
)



CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum",
    "Healthy",
]
PCA_PIXEL_BATCH_SIZE = 65_536
INFERENCE_BATCH_SIZE = 32
SQUARE_METERS_PER_ACRE = 4_046.8564224
MAX_UPLOAD_BYTES = int(os.getenv("TERRASPECTRA_MAX_UPLOAD_BYTES", 5 * 1024 ** 3))
API_KEY = os.getenv("TERRASPECTRA_API_KEY")
logger = logging.getLogger("terraspectra.api")

app = FastAPI(title="TerraSpectra Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None
scaler = None
pca = None
temperature = 1.0
FRONTEND_PATH = PROJECT_ROOT / "app" / "static" / "index.html"


class PatchRequest(BaseModel):
    patch: list[list[list[float]]] = Field(
        ...,
        description="One PCA-reduced hyperspectral patch with shape (32, 32, 30).",
    )


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_PATH)


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model unavailable: {MODEL_PATH}",
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def get_preprocessors():
    global scaler, pca
    if scaler is None or pca is None:
        if not SCALER_PATH.exists() or not PCA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Shared scaler/PCA preprocessing artifacts are unavailable",
            )
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
    return scaler, pca


def get_temperature():
    global temperature
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, encoding="utf-8") as file:
            temperature = float(json.load(file).get("temperature", 1.0))
    return temperature


def get_deployment_metadata() -> dict[str, Any]:
    if not MODEL_MANIFEST_PATH.exists():
        return {
            "deployment_model": MODEL_PATH.name,
            "framework": "TensorFlow/Keras",
            "architecture": "Shared-PCA 3D-CNN",
            "manifest_available": False,
        }
    try:
        metadata = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Deployment model manifest is unreadable") from error
    metadata["manifest_available"] = True
    return metadata


def calibrate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-7, 1.0))
    logits = logits / get_temperature()
    logits -= logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def find_cube(mat_data):
    cubes = [
        value
        for key, value in mat_data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 3
    ]
    if not cubes:
        raise HTTPException(status_code=422, detail="No 3D hyperspectral cube found in MAT file")
    return max(cubes, key=lambda value: value.size)


def cube_to_patches(cube: np.ndarray):
    height, width, channels = cube.shape
    patches = []
    locations = []
    for row in range(0, height - 31, 32):
        for column in range(0, width - 31, 32):
            patch = cube[row:row + 32, column:column + 32, :]
            if patch.shape == (32, 32, channels):
                patches.append(patch)
                locations.append({"row": row, "column": column})
    if not patches:
        raise HTTPException(status_code=422, detail="Cube is smaller than one 32 x 32 patch")
    return np.asarray(patches, dtype=np.float32), locations


def reduce_cube_in_batches(cube: np.ndarray) -> np.ndarray:
    """Apply the training preprocessing without materializing full-size intermediates."""
    fitted_scaler, fitted_pca = get_preprocessors()
    height, width, bands = cube.shape
    expected_bands = getattr(fitted_scaler, "n_features_in_", bands)
    if bands != expected_bands:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_bands} spectral bands, got {bands}",
        )

    pixels = cube.reshape(-1, bands)
    component_count = int(getattr(fitted_pca, "n_components_", 30))
    reduced = np.empty((len(pixels), component_count), dtype=np.float32)
    for start in range(0, len(pixels), PCA_PIXEL_BATCH_SIZE):
        end = min(start + PCA_PIXEL_BATCH_SIZE, len(pixels))
        reduced[start:end] = fitted_pca.transform(
            fitted_scaler.transform(pixels[start:end])
        )
    return reduced.reshape(height, width, component_count)


def predict_patches_in_batches(patches: np.ndarray) -> np.ndarray:
    """Keep model inference bounded when a field produces many patches."""
    probabilities = np.empty((len(patches), len(CLASS_NAMES)), dtype=np.float32)
    loaded_model = get_model()
    for start in range(0, len(patches), INFERENCE_BATCH_SIZE):
        end = min(start + INFERENCE_BATCH_SIZE, len(patches))
        raw_probabilities = loaded_model(
            patches[start:end, ..., None],
            training=False,
        ).numpy()
        probabilities[start:end] = calibrate_probabilities(raw_probabilities)
    return probabilities


def add_geospatial_coordinates(locations: list[dict[str, Any]], metadata: dict[str, Any]):
    if not metadata.get("crs") or not metadata.get("transform"):
        return

    from rasterio.transform import Affine, xy
    from rasterio.warp import transform
    from pyproj import Geod

    affine = Affine(*metadata["transform"][:6])
    geod = Geod(ellps="WGS84")
    rows = [location["row"] + 16 for location in locations]
    columns = [location["column"] + 16 for location in locations]
    eastings, northings = zip(
        *(xy(affine, row, column, offset="center") for row, column in zip(rows, columns))
    )
    longitudes, latitudes = transform(metadata["crs"], "EPSG:4326", eastings, northings)
    for location, longitude, latitude in zip(locations, longitudes, latitudes):
        location["longitude"] = float(longitude)
        location["latitude"] = float(latitude)
        row = location["row"]
        column = location["column"]
        corner_rows = [row, row, row + 32, row + 32]
        corner_columns = [column, column + 32, column + 32, column]
        corner_x, corner_y = zip(
            *(xy(affine, corner_row, corner_column, offset="ul")
              for corner_row, corner_column in zip(corner_rows, corner_columns))
        )
        polygon_x, polygon_y = transform(metadata["crs"], "EPSG:4326", corner_x, corner_y)
        location["polygon"] = [[float(x), float(y)] for x, y in zip(polygon_x, polygon_y)]
        polygon_area, _ = geod.polygon_area_perimeter(polygon_x, polygon_y)
        location["area_acres"] = abs(float(polygon_area)) / SQUARE_METERS_PER_ACRE


def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Require a key only when TERRASPECTRA_API_KEY is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def read_upload_limited(file: UploadFile) -> bytes:
    chunks = []
    total_bytes = 0
    while chunk := await file.read(1024 * 1024):
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit",
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    return b"".join(chunks)


def risk_summary(patch_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    affected = [
        patch for patch in patch_predictions
        if patch["predicted_label"] != len(CLASS_NAMES) - 1
    ]
    summary = {
        "total_patches": len(patch_predictions),
        "patches_at_risk": len(affected),
        "patch_risk_percent": (
            len(affected) / len(patch_predictions) * 100 if patch_predictions else 0.0
        ),
    }
    if patch_predictions and all("area_acres" in patch for patch in patch_predictions):
        summary["total_acres"] = sum(patch["area_acres"] for patch in patch_predictions)
        summary["at_risk_acres"] = sum(patch["area_acres"] for patch in affected)
    return summary


def preprocess_geospatial(contents: bytes, filename: str):
    try:
        cube, metadata = load_cube(contents, filename)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not read geospatial file: {error}") from error

    cube = np.asarray(cube, dtype=np.float32)
    if not np.isfinite(cube).all():
        raise HTTPException(status_code=422, detail="Geospatial cube contains NaN or infinite values")
    reduced_cube = reduce_cube_in_batches(cube)
    patches, locations = cube_to_patches(reduced_cube)
    add_geospatial_coordinates(locations, metadata)
    return patches, locations, metadata


def preprocess_mat(contents: bytes):
    patches, locations, _ = preprocess_geospatial(contents, "upload.mat")
    return patches, locations


async def save_upload_to_temporary_file(file: UploadFile) -> Path:
    """Persist an upload incrementally so Rasterio can read large GeoTIFFs by window."""
    suffix = Path(file.filename or "upload.tif").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        total_bytes = 0
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                temporary_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit",
                )
            temporary_file.write(chunk)
    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    return temporary_path


def geotiff_patch_batches(path: Path):
    """Yield reduced patches from a GeoTIFF without reading its entire raster at once."""
    import rasterio
    from rasterio.windows import Window

    try:
        with rasterio.open(path) as dataset:
            metadata = {
                "crs": str(dataset.crs) if dataset.crs else None,
                "transform": list(dataset.transform),
                "width": dataset.width,
                "height": dataset.height,
                "bounds": list(dataset.bounds),
            }
            locations = [
                {"row": row, "column": column}
                for row in range(0, dataset.height - 31, 32)
                for column in range(0, dataset.width - 31, 32)
            ]
            if not locations:
                raise HTTPException(status_code=422, detail="Cube is smaller than one 32 x 32 patch")
            add_geospatial_coordinates(locations, metadata)

            for start in range(0, len(locations), INFERENCE_BATCH_SIZE):
                batch_locations = locations[start:start + INFERENCE_BATCH_SIZE]
                raw_patches = np.stack([
                    dataset.read(
                        window=Window(location["column"], location["row"], 32, 32)
                    ).transpose(1, 2, 0)
                    for location in batch_locations
                ])
                if not np.isfinite(raw_patches).all():
                    raise HTTPException(status_code=422, detail="GeoTIFF cube contains NaN or infinite values")
                reduced_patches = reduce_cube_in_batches(
                    raw_patches.reshape(-1, 32, dataset.count)
                ).reshape(len(batch_locations), 32, 32, -1)
                yield reduced_patches, batch_locations, metadata
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not read GeoTIFF: {error}") from error


def aggregate_patch_predictions(patch_batches, filename: str):
    probability_sum = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    patch_count = 0
    patch_predictions = []
    metadata = {}
    for patches, locations, metadata in patch_batches:
        probabilities = predict_patches_in_batches(patches)
        predicted_labels = np.argmax(probabilities, axis=1)
        confidences = probabilities.max(axis=1)
        probability_sum += probabilities.sum(axis=0)
        patch_count += len(patches)
        patch_predictions.extend(
            {
                **location,
                "predicted_label": int(label),
                "predicted_class": CLASS_NAMES[int(label)],
                "confidence": float(confidence),
            }
            for location, label, confidence in zip(locations, predicted_labels, confidences)
        )
    response = prediction_response(probability_sum / patch_count)
    response.update({
        "filename": filename,
        "patch_count": patch_count,
        "geospatial_metadata": metadata,
        "patch_predictions": patch_predictions,
        "risk_summary": risk_summary(patch_predictions),
    })
    return response


def prepare_patch(request: PatchRequest) -> np.ndarray:
    patch = np.asarray(request.patch, dtype=np.float32)
    if patch.shape != (32, 32, 30):
        raise HTTPException(
            status_code=422,
            detail=f"Expected patch shape [32, 32, 30], got {list(patch.shape)}",
        )
    if not np.isfinite(patch).all():
        raise HTTPException(status_code=422, detail="Patch contains NaN or infinite values")
    return patch[None, ..., None]


def prediction_response(probabilities: np.ndarray) -> dict[str, Any]:
    predicted_label = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_label])
    return {
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": confidence,
        "confidence_percent": confidence * 100,
        "confidence_warning": confidence >= 0.95,
        "calibration_temperature": get_temperature(),
        "probabilities": {
            name: float(probabilities[label])
            for label, name in enumerate(CLASS_NAMES)
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_missing",
        "model": MODEL_PATH.name,
        "model_available": MODEL_PATH.exists(),
        "framework": get_deployment_metadata()["framework"],
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "api_key_required": bool(API_KEY),
    }


@app.get("/model-info")
def model_info():
    """Expose the selected deployment model and its evaluation context."""
    return get_deployment_metadata()


@app.get("/temporal-analysis")
def temporal_analysis():
    """Return temporal prediction analysis for dashboard visualization."""

    if not TEMPORAL_ANALYSIS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Temporal analysis file not found: {TEMPORAL_ANALYSIS_PATH}",
        )

    try:
        dataframe = pd.read_csv(TEMPORAL_ANALYSIS_PATH)
        dataframe = dataframe.replace({np.nan: None})

        return {
            "status": "ok",
            "source": str(TEMPORAL_ANALYSIS_PATH),
            "records": dataframe.to_dict(orient="records"),
        }

    except Exception as error:
        logger.exception("temporal_analysis_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read temporal analysis: {error}",
        ) from error
    

@app.post("/predict")
def predict(request: PatchRequest, _: None = Depends(verify_api_key)):
    patch = prepare_patch(request)
    probabilities = calibrate_probabilities(
        get_model()(patch, training=False).numpy()
    )[0]
    return prediction_response(probabilities)


@app.post("/explain")
def explain(request: PatchRequest, _: None = Depends(verify_api_key)):
    patch = prepare_patch(request)
    loaded_model = get_model()
    input_tensor = tf.Variable(patch)
    with tf.GradientTape() as tape:
        raw_probabilities = loaded_model(input_tensor, training=False)
        predicted_label = tf.argmax(raw_probabilities[0])
        class_score = raw_probabilities[0, predicted_label]

    gradients = tape.gradient(class_score, input_tensor)
    attribution = np.abs(gradients.numpy()[0, ..., 0])
    maximum = float(attribution.max())
    minimum = float(attribution.min())
    if maximum > minimum:
        attribution = (attribution - minimum) / (maximum - minimum)
    else:
        attribution = np.zeros_like(attribution)

    probabilities = calibrate_probabilities(raw_probabilities.numpy())[0]
    response = prediction_response(probabilities)
    response["spatial_importance"] = attribution.mean(axis=2).tolist()
    response["spectral_importance"] = attribution.mean(axis=(0, 1)).tolist()
    return response


@app.post("/predict-mat")
async def predict_mat(file: UploadFile = File(...), _: None = Depends(verify_api_key)):
    if not file.filename or not file.filename.lower().endswith(".mat"):
        raise HTTPException(status_code=422, detail="Upload a MATLAB .mat file")

    contents = await read_upload_limited(file)

    patches, locations, metadata = preprocess_geospatial(contents, file.filename)
    probabilities = predict_patches_in_batches(patches)
    predicted_labels = np.argmax(probabilities, axis=1)
    patch_confidences = probabilities.max(axis=1)
    image_probabilities = probabilities.mean(axis=0)
    response = prediction_response(image_probabilities)
    response["filename"] = file.filename
    response["geospatial_metadata"] = metadata
    response["patch_count"] = len(patches)
    response["patch_predictions"] = [
        {
            **location,
            "predicted_label": int(label),
            "predicted_class": CLASS_NAMES[int(label)],
            "confidence": float(confidence),
        }
        for location, label, confidence in zip(
            locations,
            predicted_labels,
            patch_confidences,
        )
    ]
    response["risk_summary"] = risk_summary(response["patch_predictions"])
    logger.info("prediction_complete filename=%s patches=%s", file.filename, len(patches))
    return response


@app.post("/predict-geospatial")
async def predict_geospatial(file: UploadFile = File(...), _: None = Depends(verify_api_key)):
    if not file.filename or Path(file.filename).suffix.lower() not in {
        ".mat", ".h5", ".hdf5", ".tif", ".tiff"
    }:
        raise HTTPException(
            status_code=422,
            detail="Upload a .mat, .h5, .hdf5, .tif, or .tiff file",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        temporary_path = await save_upload_to_temporary_file(file)
        try:
            response = aggregate_patch_predictions(
                geotiff_patch_batches(temporary_path),
                file.filename,
            )
            logger.info("geotiff_prediction_complete filename=%s patches=%s", file.filename, response["patch_count"])
            return response
        finally:
            os.unlink(temporary_path)

    contents = await read_upload_limited(file)

    patches, locations, metadata = preprocess_geospatial(contents, file.filename)
    probabilities = predict_patches_in_batches(patches)
    predicted_labels = np.argmax(probabilities, axis=1)
    image_probabilities = probabilities.mean(axis=0)
    response = prediction_response(image_probabilities)
    response["filename"] = file.filename
    response["patch_count"] = len(patches)
    response["geospatial_metadata"] = metadata
    response["patch_predictions"] = [
        {
            **location,
            "predicted_label": int(label),
            "predicted_class": CLASS_NAMES[int(label)],
            "confidence": float(confidence),
        }
        for location, label, confidence in zip(
            locations,
            predicted_labels,
            probabilities.max(axis=1),
        )
    ]
    response["risk_summary"] = risk_summary(response["patch_predictions"])
    logger.info("geospatial_prediction_complete filename=%s patches=%s", file.filename, len(patches))
    return response
