"""Evaluate a saved TerraSpectra model at original-cube, not patch, level.

The training pipeline creates many patches from each hyperspectral source file.
This evaluator keeps those patches together, averages their probabilities, and
reports one prediction for every source cube.  It also rejects overlapping file
names across train/validation/test folders to guard against split leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_shared_pca"
MODEL_PATH = PROJECT_ROOT / "models" / "best_shared_pca_3dcnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"
PATCH_SIZE = 32
CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum",
    "Healthy",
]
CLASS_TO_LABEL = {
    class_name.lower().replace(" ", "_"): label
    for label, class_name in enumerate(CLASS_NAMES)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TerraSpectra by original hyperspectral source file."
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def source_file_sets(processed_dir: Path) -> dict[str, set[str]]:
    return {
        split: {path.name for path in (processed_dir / split).glob("*/*.npy")}
        for split in ("train", "val", "test")
    }


def assert_source_splits_are_disjoint(processed_dir: Path) -> dict[str, int]:
    file_sets = source_file_sets(processed_dir)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = file_sets[left] & file_sets[right]
        if overlap:
            raise ValueError(
                f"Source-file leakage between {left} and {right}: {sorted(overlap)}"
            )
    return {split: len(files) for split, files in file_sets.items()}


def cube_to_patches(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3 or cube.shape[-1] != 30:
        raise ValueError(f"Expected an H x W x 30 PCA cube, got {cube.shape}")
    height, width, channels = cube.shape
    patches = [
        cube[row:row + PATCH_SIZE, column:column + PATCH_SIZE, :]
        for row in range(0, height - PATCH_SIZE + 1, PATCH_SIZE)
        for column in range(0, width - PATCH_SIZE + 1, PATCH_SIZE)
    ]
    if not patches:
        raise ValueError("Cube is smaller than one 32 x 32 patch")
    return np.asarray(patches, dtype=np.float32)


def prepare_for_model(model: tf.keras.Model, patches: np.ndarray) -> np.ndarray:
    expected_rank = len(model.input_shape)
    if expected_rank == patches.ndim + 1:
        return patches[..., None]
    if expected_rank == patches.ndim:
        return patches
    raise ValueError(
        f"Model expects input rank {expected_rank}, but patches have rank {patches.ndim}."
    )


def evaluate_sources(model: tf.keras.Model, split_dir: Path, batch_size: int) -> pd.DataFrame:
    rows = []
    for class_directory in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = CLASS_TO_LABEL.get(class_directory.name)
        if label is None:
            continue
        for cube_path in sorted(class_directory.glob("*.npy")):
            cube = np.load(cube_path).astype(np.float32)
            if not np.isfinite(cube).all():
                raise ValueError(f"Non-finite values in {cube_path}")
            patches = prepare_for_model(model, cube_to_patches(cube))
            probabilities = model.predict(patches, batch_size=batch_size, verbose=0)
            mean_probabilities = probabilities.mean(axis=0)
            predicted_label = int(np.argmax(mean_probabilities))
            row = {
                "source_file": cube_path.name,
                "true_label": label,
                "true_class": CLASS_NAMES[label],
                "predicted_label": predicted_label,
                "predicted_class": CLASS_NAMES[predicted_label],
                "confidence": float(mean_probabilities[predicted_label]),
                "patch_count": len(patches),
            }
            row.update({
                f"probability_{class_name}": float(mean_probabilities[index])
                for index, class_name in enumerate(CLASS_NAMES)
            })
            rows.append(row)
    if not rows:
        raise ValueError(f"No source cubes found in {split_dir}")
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    split_counts = assert_source_splits_are_disjoint(args.processed_dir)
    split_dir = args.processed_dir / args.split
    if not args.model.exists():
        raise FileNotFoundError(f"Model unavailable: {args.model}")

    model = tf.keras.models.load_model(args.model)
    source_predictions = evaluate_sources(model, split_dir, args.batch_size)
    y_true = source_predictions["true_label"].to_numpy()
    y_pred = source_predictions["predicted_label"].to_numpy()
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))

    output_dir = args.results_dir / "source_level" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    source_predictions.to_csv(output_dir / "source_predictions.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    summary = {
        "model": str(args.model),
        "split": args.split,
        "source_file_counts": split_counts,
        "evaluated_sources": int(len(source_predictions)),
        "accuracy": float(accuracy),
        "note": "One prediction per original source cube; source filenames are disjoint across splits.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
