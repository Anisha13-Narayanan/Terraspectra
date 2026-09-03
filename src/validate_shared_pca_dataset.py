from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(r"E:\Terraspectra")
DATA_DIR = PROJECT_ROOT / "data" / "patches_shared_pca"

EXPECTED_CLASSES = 5


def check_split(split_name):

    print("\n" + "=" * 60)
    print(f"VALIDATING {split_name.upper()} SPLIT")
    print("=" * 60)

    split_dir = DATA_DIR / split_name

    X = np.load(split_dir / f"X_{split_name}.npy")
    y = np.load(split_dir / f"y_{split_name}.npy")

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Data type: {X.dtype}")

    # Shape checks
    assert X.ndim == 4, "X must have shape (N, H, W, C)"
    assert X.shape[1:] == (32, 32, 30), (
        f"Unexpected patch shape: {X.shape[1:]}"
    )
    assert len(X) == len(y), "X and y sample counts differ"

    # Invalid values
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")

    assert nan_count == 0, "NaN values found"
    assert inf_count == 0, "Infinite values found"

    # Labels
    unique_labels, counts = np.unique(y, return_counts=True)

    print("\nClass distribution:")

    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} patches")

    assert len(unique_labels) == EXPECTED_CLASSES, (
        f"Expected {EXPECTED_CLASSES} classes, "
        f"found {len(unique_labels)}"
    )

    assert np.array_equal(
        unique_labels,
        np.arange(EXPECTED_CLASSES)
    ), "Labels must be 0, 1, 2, 3, 4"

    print("\n✓ Split validation passed")

    return X, y


def main():

    print("=" * 60)
    print("TERRASPECTRA - SHARED PCA DATASET VALIDATION")
    print("=" * 60)

    train_X, train_y = check_split("train")
    val_X, val_y = check_split("val")
    test_X, test_y = check_split("test")

    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"TRAIN | Samples: {len(train_X)} | "
        f"Shape: {train_X.shape}"
    )
    print(
        f"VAL   | Samples: {len(val_X)} | "
        f"Shape: {val_X.shape}"
    )
    print(
        f"TEST  | Samples: {len(test_X)} | "
        f"Shape: {test_X.shape}"
    )

    print("\n✓ ALL SHARED PCA DATASET VALIDATION CHECKS PASSED")
    print("✓ READY FOR FRESH MODEL TRAINING")


if __name__ == "__main__":
    main()