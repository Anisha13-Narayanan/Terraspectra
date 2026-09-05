from pathlib import Path
import re

import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tomato_hsi"
OUTPUT_PATH = PROJECT_ROOT / "data" / "temporal_index.csv"


DAY_PATTERN = re.compile(
    r"day[_\s-]*(one|three|five|seven)",
    re.IGNORECASE,
)

DAY_MAP = {
    "one": 1,
    "three": 3,
    "five": 5,
    "seven": 7,
}


def extract_day(variable_name: str) -> int:
    match = DAY_PATTERN.search(variable_name)

    if not match:
        raise ValueError(
            f"Could not determine day from MAT variable: {variable_name}"
        )

    return DAY_MAP[match.group(1).lower()]


def extract_class(mat_path: Path) -> str:
    return mat_path.parent.name.lower()


def extract_replicate(variable_name: str) -> int:
    match = re.search(r"_(\d+)$", variable_name)

    if not match:
        return 1

    return int(match.group(1))


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DIR}"
        )

    rows = []

    mat_files = sorted(RAW_DIR.glob("*/*.mat"))

    if not mat_files:
        raise FileNotFoundError(
            f"No MAT files found under: {RAW_DIR}"
        )

    for mat_path in mat_files:
        mat_data = loadmat(mat_path)

        variable_names = [
            key for key in mat_data.keys()
            if not key.startswith("__")
        ]

        if len(variable_names) != 1:
            raise ValueError(
                f"Expected exactly one data variable in "
                f"{mat_path.name}, found: {variable_names}"
            )

        mat_variable = variable_names[0]

        day = extract_day(mat_variable)
        class_name = extract_class(mat_path)
        replicate = extract_replicate(mat_variable)

        cube = mat_data[mat_variable]

        if cube.ndim != 3:
            raise ValueError(
                f"Expected a 3D hyperspectral cube in "
                f"{mat_path.name}, got shape {cube.shape}"
            )

        rows.append(
            {
                "class": class_name,
                "day": day,
                "replicate": replicate,
                "filename": mat_path.name,
                "relative_path": str(
                    mat_path.relative_to(PROJECT_ROOT)
                ),
                "mat_variable": mat_variable,
                "height": int(cube.shape[0]),
                "width": int(cube.shape[1]),
                "spectral_bands": int(cube.shape[2]),
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["class", "day", "replicate"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("TERRASPECTRA TEMPORAL DATASET INDEX")
    print("=" * 70)
    print(f"Total MAT files: {len(df)}")
    print(f"Classes: {df['class'].nunique()}")
    print(f"Days: {sorted(df['day'].unique().tolist())}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    print("Files by day:")
    print(
        df.groupby("day")
        .size()
        .rename("files")
        .to_string()
    )

    print()
    print("Files by class:")
    print(
        df.groupby("class")
        .size()
        .rename("files")
        .to_string()
    )

    print()
    print("Temporal distribution:")
    print(
        df.groupby(["class", "day"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print()
    print("First records:")
    print(df.head(12).to_string(index=False))

    print()
    print("Temporal index created successfully.")


if __name__ == "__main__":
    main()
