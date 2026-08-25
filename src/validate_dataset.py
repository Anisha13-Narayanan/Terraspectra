from pathlib import Path
import numpy as np

# ==========================================================
# TERRASPECTRA - DATASET VALIDATION
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")
PATCHES_DIR = PROJECT_ROOT / "data" / "patches"

CLASS_NAMES = {
    0: "alternaria_alternata",
    1: "alternaria_solani",
    2: "botrytis_cinerea",
    3: "fusarium_oxysporum",
    4: "healthy"
}

SPLITS = ["train", "val", "test"]


def validate_split(split_name):

    print("\n" + "=" * 60)
    print(f"VALIDATING: {split_name.upper()}")
    print("=" * 60)

    split_dir = PATCHES_DIR / split_name

    X_file = split_dir / f"X_{split_name}.npy"
    y_file = split_dir / f"y_{split_name}.npy"

    # Check files exist
    if not X_file.exists():
        raise FileNotFoundError(f"Missing file: {X_file}")

    if not y_file.exists():
        raise FileNotFoundError(f"Missing file: {y_file}")

    # Load data
    X = np.load(X_file)
    y = np.load(y_file)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")

    # Check sample counts
    print("\nDATA CHECKS")

    if len(X) == len(y):
        print("✓ X and y sample counts match")
    else:
        raise ValueError(
            f"Mismatch: X has {len(X)} samples, "
            f"but y has {len(y)} labels"
        )

    # Expected patch shape
    expected_shape = (32, 32, 30)

    if X.shape[1:] == expected_shape:
        print(f"✓ Patch shape is correct: {expected_shape}")
    else:
        raise ValueError(
            f"Unexpected patch shape: {X.shape[1:]}"
        )

    # NaN / Inf checks
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")

    if nan_count == 0 and inf_count == 0:
        print("✓ No NaN or Infinite values")
    else:
        raise ValueError("Invalid values found in dataset")

    # Label checks
    unique_labels = np.unique(y)

    print("\nLABEL CHECK")

    print(f"Unique labels: {unique_labels}")

    expected_labels = np.array([0, 1, 2, 3, 4])

    if np.array_equal(unique_labels, expected_labels):
        print("✓ All 5 classes are present")
    else:
        raise ValueError(
            f"Unexpected labels found: {unique_labels}"
        )

    # Class distribution
    print("\nCLASS DISTRIBUTION")

    class_counts = {}

    for label, class_name in CLASS_NAMES.items():
        count = np.sum(y == label)
        class_counts[label] = count
        print(f"{label} - {class_name}: {count} patches")

    # Check balance
    counts = list(class_counts.values())

    if len(set(counts)) == 1:
        print("✓ Dataset is balanced")
    else:
        print("⚠ Dataset is not perfectly balanced")

    return {
        "samples": len(X),
        "shape": X.shape,
        "class_counts": class_counts
    }


def main():

    print("=" * 60)
    print("TERRASPECTRA - FINAL DATASET VALIDATION")
    print("=" * 60)

    results = {}

    for split_name in SPLITS:
        results[split_name] = validate_split(split_name)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)

    for split_name, result in results.items():
        print(
            f"{split_name.upper():5} | "
            f"Samples: {result['samples']} | "
            f"Shape: {result['shape']}"
        )

    print("\n✓ ALL DATASET VALIDATION CHECKS COMPLETED SUCCESSFULLY")
    print("✓ DATA IS READY FOR MODEL TRAINING")


if __name__ == "__main__":
    main()