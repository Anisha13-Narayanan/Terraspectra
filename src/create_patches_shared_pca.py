from pathlib import Path
import numpy as np


# ==========================================================
# TERRASPECTRA - PATCH CREATION FROM SHARED PCA DATA
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

INPUT_DIR = PROJECT_ROOT / "data" / "processed_shared_pca"
OUTPUT_DIR = PROJECT_ROOT / "data" / "patches_shared_pca"

PATCH_SIZE = 32
STRIDE = 32

CLASS_NAMES = [
    "alternaria_alternata",
    "alternaria_solani",
    "botrytis_cinerea",
    "fusarium_oxysporum",
    "healthy"
]

CLASS_TO_LABEL = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


def create_patches(cube):
    """
    Convert one H x W x C hyperspectral cube into
    spatial patches of shape PATCH_SIZE x PATCH_SIZE x C.
    """

    height, width, channels = cube.shape

    patches = []

    for row in range(0, height - PATCH_SIZE + 1, STRIDE):
        for col in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = cube[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            if patch.shape == (
                PATCH_SIZE,
                PATCH_SIZE,
                channels
            ):
                patches.append(patch)

    return np.array(patches, dtype=np.float32)


def process_split(split_name):

    print("\n" + "=" * 60)
    print(f"PROCESSING {split_name.upper()} DATA")
    print("=" * 60)

    split_dir = INPUT_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}"
        )

    all_patches = []
    all_labels = []

    for class_name in CLASS_NAMES:

        class_dir = split_dir / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing folder: {class_dir}")
            continue

        label = CLASS_TO_LABEL[class_name]

        npy_files = sorted(class_dir.glob("*.npy"))

        print(f"\nCLASS: {class_name.upper()}")
        print(f"LABEL: {label}")
        print(f"Files found: {len(npy_files)}")

        for npy_file in npy_files:

            cube = np.load(npy_file)

            patches = create_patches(cube)

            labels = np.full(
                len(patches),
                label,
                dtype=np.int64
            )

            all_patches.append(patches)
            all_labels.append(labels)

            print(
                f"{npy_file.name} -> "
                f"{len(patches)} patches"
            )

    if not all_patches:
        raise ValueError(
            f"No patches created for {split_name}"
        )

    X = np.concatenate(all_patches, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Shuffle patches within this split
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    output_split_dir = OUTPUT_DIR / split_name
    output_split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_split_dir / f"X_{split_name}.npy",
        X
    )

    np.save(
        output_split_dir / f"y_{split_name}.npy",
        y
    )

    print("\nSAVED")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nCLASS DISTRIBUTION")

    for class_name, label in CLASS_TO_LABEL.items():

        count = np.sum(y == label)

        print(
            f"{label} - {class_name}: "
            f"{count} patches"
        )

    return X.shape, y.shape


def main():

    print("=" * 60)
    print("TERRASPECTRA - SHARED PCA PATCH CREATION")
    print("=" * 60)

    results = {}

    for split_name in ["train", "val", "test"]:

        X_shape, y_shape = process_split(split_name)

        results[split_name] = (
            X_shape,
            y_shape
        )

    print("\n" + "=" * 60)
    print("FINAL PATCH CREATION SUMMARY")
    print("=" * 60)

    for split_name, (X_shape, y_shape) in results.items():

        print(
            f"{split_name.upper():5} -> "
            f"X: {X_shape}, "
            f"y: {y_shape}"
        )

    print("\nPATCH CREATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()