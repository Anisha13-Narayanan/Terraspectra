"""Record the hardware and tensor configuration used for a training run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "patches_shared_pca" / "train" / "X_train.npy"
OUTPUT_PATH = PROJECT_ROOT / "results" / "training_resource_audit.json"


def main():
    features = np.load(DATA_PATH, mmap_mode="r")
    audit = {
        "dataset_path": str(DATA_PATH),
        "tensor_shape": list(features.shape),
        "tensor_dtype": str(features.dtype),
        "single_patch_bytes": int(np.prod(features.shape[1:]) * features.dtype.itemsize),
        "tensorflow_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "tensorflow_version": tf.__version__,
        "note": "Tensor allocation audit; run the selected training script to capture epoch-time profiling.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
