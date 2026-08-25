from pathlib import Path
import numpy as np

# ==========================================================
# TERRASPECTRA - PATCH CREATION FOR TRAIN / VAL / TEST
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

INPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "patches"

PATCH_SIZE = 32
STRIDE = 32

CLASS_TO_LABEL = {
    "alternaria_alternata": 0,
    "alternaria_solani": 1,
    "botrytis_cinerea": 2,
    "fusarium_oxysporum": 3,
    "healthy": 4
}

SPLITS = ["train", "val", "test"]


def create_patches(cube):
    """
    Input:
        cube shape = (height, width, bands)

    Output:
        patches shape = (num_patches, 32, 32, bands)
    """
    height, width, bands = cube.shape
    patches = []

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = cube[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
                :
            ]

            patches.append(patch)

    return np.asarray(patches, dtype=np.float32)


def process_split(split_name):
    """
    Create patches for one split:
    train, val, or test.
    """

    split_dir = INPUT_DIR / split_name

    print("\n" + "=" * 60)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 60)

    all_patches = []
    all_labels = []
    total_files = 0

    # Process each class
    for class_name, label in CLASS_TO_LABEL.items():

        class_dir = split_dir / class_name
        npy_files = sorted(class_dir.glob("*.npy"))

        print(f"\nCLASS: {class_name.upper()}")
        print(f"Label: {label}")
        print(f"Files found: {len(npy_files)}")

        for npy_file in npy_files:

            total_files += 1

            cube = np.load(npy_file)
            patches = create_patches(cube)

            # One label for every patch
            labels = np.full(
                len(patches),
                label,
                dtype=np.int64
            )

            all_patches.append(patches)
            all_labels.append(labels)

            print(
                f"  {npy_file.name} "
                f"-> {len(patches)} patches"
            )

    if not all_patches:
        raise ValueError(
            f"No patches created for {split_name}."
        )

    # Combine all classes
    X = np.concatenate(all_patches, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Create output folder
    split_output_dir = OUTPUT_DIR / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Save
    X_file = split_output_dir / f"X_{split_name}.npy"
    y_file = split_output_dir / f"y_{split_name}.npy"

    np.save(X_file, X)
    np.save(y_file, y)

    # Summary
    print("\n" + "-" * 60)
    print(f"{split_name.upper()} SUMMARY")
    print("-" * 60)

    print(f"Total source files: {total_files}")
    print(f"Total patches: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nClass distribution:")
    for class_name, label in CLASS_TO_LABEL.items():
        count = np.sum(y == label)
        print(f"{label} - {class_name}: {count}")

    print(f"\nSaved: {X_file}")
    print(f"Saved: {y_file}")

    return X.shape, y.shape


def main():

    print("=" * 60)
    print("TERRASPECTRA - TRAIN / VAL / TEST PATCH CREATION")
    print("=" * 60)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Split dataset folder not found: {INPUT_DIR}"
        )

    results = {}

    # Process train, validation, and test separately
    for split_name in SPLITS:
        results[split_name] = process_split(split_name)

    print("\n" + "=" * 60)
    print("FINAL PATCH CREATION SUMMARY")
    print("=" * 60)

    for split_name, (x_shape, y_shape) in results.items():
        print(
            f"{split_name.upper():5} -> "
            f"X: {x_shape}, y: {y_shape}"
        )

    print("\nPatch creation completed successfully.")


if __name__ == "__main__":
    main()