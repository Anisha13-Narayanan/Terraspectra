import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.io import loadmat

from app.main import (
    CLASS_NAMES,
    predict_patches_in_batches,
    reduce_cube_in_batches,
)


INDEX_PATH = PROJECT_ROOT / "data" / "temporal_index.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "temporal_predictions.csv"


def load_mat_cube(relative_path, variable_name):
    mat_path = PROJECT_ROOT / relative_path

    mat_data = loadmat(mat_path)

    if variable_name not in mat_data:
        raise KeyError(
            f"Variable {variable_name} not found in {mat_path.name}"
        )

    cube = np.asarray(
        mat_data[variable_name],
        dtype=np.float32
    )

    if cube.ndim != 3:
        raise ValueError(
            f"Expected 3D cube, got {cube.shape}"
        )

    if not np.isfinite(cube).all():
        raise ValueError(
            f"Cube contains invalid values: {mat_path.name}"
        )

    return cube


def cube_to_patches_local(cube, patch_size=32):
    height, width, bands = cube.shape

    patches = []

    for row in range(0, height - patch_size + 1, patch_size):
        for column in range(
            0,
            width - patch_size + 1,
            patch_size
        ):
            patch = cube[
                row:row + patch_size,
                column:column + patch_size,
                :
            ]

            if patch.shape == (
                patch_size,
                patch_size,
                bands
            ):
                patches.append(patch)

    if not patches:
        raise ValueError(
            "No 32x32 patches could be created."
        )

    return np.asarray(
        patches,
        dtype=np.float32
    )


def main():

    print("=" * 70)
    print("TERRASPECTRA TEMPORAL MODEL PREDICTIONS")
    print("=" * 70)

    df = pd.read_csv(INDEX_PATH)

    results = []

    print(f"Files to process: {len(df)}")
    print()

    for i, row in df.iterrows():

        print(
            f"[{i + 1}/{len(df)}] "
            f"Day {int(row['day'])} | "
            f"{row['class']} | "
            f"Replicate {int(row['replicate'])}"
        )

        cube = load_mat_cube(
            row["relative_path"],
            row["mat_variable"]
        )

        reduced_cube = reduce_cube_in_batches(cube)

        patches = cube_to_patches_local(
            reduced_cube
        )

        probabilities = predict_patches_in_batches(
            patches
        )

        mean_probabilities = probabilities.mean(
            axis=0
        )

        predicted_label = int(
            np.argmax(mean_probabilities)
        )

        confidence = float(
            mean_probabilities[predicted_label]
        )

        results.append(
            {
                "class": row["class"],
                "day": int(row["day"]),
                "replicate": int(row["replicate"]),
                "filename": row["filename"],
                "predicted_label": predicted_label,
                "predicted_class": CLASS_NAMES[
                    predicted_label
                ],
                "confidence": confidence,
                "patch_count": len(patches),
            }
        )

    output = pd.DataFrame(results)

    output = output.sort_values(
        ["class", "day", "replicate"]
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(f"Predictions: {len(output)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("Predictions by day:")

    print(
        output.groupby(
            ["day", "predicted_class"]
        )
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()