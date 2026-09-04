from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(r"E:\Terraspectra")
MODEL_PATH = PROJECT_ROOT / "models" / "best_shared_pca_3dcnn.keras"
VAL_DIR = PROJECT_ROOT / "data" / "patches_shared_pca" / "val"
OUTPUT_PATH = PROJECT_ROOT / "models" / "preprocessing" / "temperature_calibration.json"

NUM_CLASSES = 5


def load_validation_data():
    X = np.load(VAL_DIR / "X_val.npy").astype(np.float32)
    y = np.load(VAL_DIR / "y_val.npy").astype(np.int64)
    return X[..., None], y


def calibrated_probabilities(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-7, 1.0))
    scaled_logits = logits / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled_logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities, labels):
    correct_probabilities = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(correct_probabilities, 1e-7, 1.0)).mean())


def expected_calibration_error(probabilities, labels, bins=10):
    predictions = np.argmax(probabilities, axis=1)
    confidences = probabilities.max(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(0.0, 1.0, bins + 1)[1:]):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            ece += selected.mean() * abs(
                (predictions[selected] == labels[selected]).mean()
                - confidences[selected].mean()
            )
    return float(ece)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    X_val, y_val = load_validation_data()
    model = tf.keras.models.load_model(MODEL_PATH)
    raw_probabilities = model.predict(X_val, verbose=0)

    result = minimize_scalar(
        lambda temperature: negative_log_likelihood(
            calibrated_probabilities(raw_probabilities, temperature), y_val
        ),
        bounds=(0.05, 20.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = float(result.x)
    calibrated = calibrated_probabilities(raw_probabilities, temperature)

    summary = {
        "model": str(MODEL_PATH),
        "validation_samples": int(len(y_val)),
        "temperature": temperature,
        "raw_nll": negative_log_likelihood(raw_probabilities, y_val),
        "calibrated_nll": negative_log_likelihood(calibrated, y_val),
        "raw_ece": expected_calibration_error(raw_probabilities, y_val),
        "calibrated_ece": expected_calibration_error(calibrated, y_val),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved calibration: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
