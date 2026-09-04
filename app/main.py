from pathlib import Path
from typing import Any
from io import BytesIO
import json

import numpy as np
import joblib
import scipy.io
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.geospatial import load_cube


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_shared_pca_3dcnn.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing" / "shared_scaler.joblib"
PCA_PATH = PROJECT_ROOT / "models" / "preprocessing" / "shared_pca30.joblib"
CALIBRATION_PATH = PROJECT_ROOT / "models" / "preprocessing" / "temperature_calibration.json"

CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum",
    "Healthy",
]

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


def preprocess_geospatial(contents: bytes, filename: str):
    try:
        cube, metadata = load_cube(contents, filename)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not read geospatial file: {error}") from error

    cube = np.asarray(cube, dtype=np.float32)
    if not np.isfinite(cube).all():
        raise HTTPException(status_code=422, detail="Geospatial cube contains NaN or infinite values")
    if cube.shape[-1] < 30:
        raise HTTPException(status_code=422, detail=f"Expected at least 30 spectral bands, got {cube.shape[-1]}")

    fitted_scaler, fitted_pca = get_preprocessors()
    height, width, bands = cube.shape
    pixels = cube.reshape(-1, bands)
    reduced = fitted_pca.transform(fitted_scaler.transform(pixels))
    reduced_cube = reduced.reshape(height, width, 30).astype(np.float32)
    patches, locations = cube_to_patches(reduced_cube)
    if metadata.get("crs") and metadata.get("transform"):
        import rasterio
        from rasterio.transform import Affine, xy
        from rasterio.warp import transform

        affine = Affine(*metadata["transform"][:6])
        rows = [location["row"] + 16 for location in locations]
        columns = [location["column"] + 16 for location in locations]
        eastings, northings = zip(
            *(xy(affine, row, column, offset="center") for row, column in zip(rows, columns))
        )
        longitudes, latitudes = transform(
            metadata["crs"],
            "EPSG:4326",
            eastings,
            northings,
        )
        for location, longitude, latitude in zip(locations, longitudes, latitudes):
            location["longitude"] = float(longitude)
            location["latitude"] = float(latitude)
        for location in locations:
            row = location["row"]
            column = location["column"]
            corner_rows = [row, row, row + 32, row + 32]
            corner_columns = [column, column + 32, column + 32, column]
            corner_x, corner_y = zip(
                *(xy(affine, corner_row, corner_column, offset="ul")
                  for corner_row, corner_column in zip(corner_rows, corner_columns))
            )
            polygon_x, polygon_y = transform(
                metadata["crs"],
                "EPSG:4326",
                corner_x,
                corner_y,
            )
            location["polygon"] = [
                [float(longitude), float(latitude)]
                for longitude, latitude in zip(polygon_x, polygon_y)
            ]
    return patches, locations, metadata


def preprocess_mat(contents: bytes):
    patches, locations, _ = preprocess_geospatial(contents, "upload.mat")
    return patches, locations


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
    }


@app.post("/predict")
def predict(request: PatchRequest):
    patch = prepare_patch(request)
    probabilities = calibrate_probabilities(
        get_model()(patch, training=False).numpy()
    )[0]
    return prediction_response(probabilities)


@app.post("/explain")
def explain(request: PatchRequest):
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
async def predict_mat(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".mat"):
        raise HTTPException(status_code=422, detail="Upload a MATLAB .mat file")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    patches, locations, metadata = preprocess_geospatial(contents, file.filename)
    probabilities = get_model()(
        patches[..., None],
        training=False,
    ).numpy()
    probabilities = calibrate_probabilities(probabilities)
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
    return response


@app.post("/predict-geospatial")
async def predict_geospatial(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() not in {
        ".mat", ".h5", ".hdf5", ".tif", ".tiff"
    }:
        raise HTTPException(
            status_code=422,
            detail="Upload a .mat, .h5, .hdf5, .tif, or .tiff file",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    patches, locations, metadata = preprocess_geospatial(contents, file.filename)
    probabilities = calibrate_probabilities(
        get_model()(patches[..., None], training=False).numpy()
    )
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
    return response