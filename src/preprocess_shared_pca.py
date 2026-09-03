from pathlib import Path
import numpy as np
import scipy.io
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# ==========================================================
# TERRASPECTRA - SHARED TRAIN-FITTED PCA PREPROCESSING
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tomato_hsi"

# New folder - old processed data remains untouched
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_shared_pca"

MODELS_DIR = PROJECT_ROOT / "models" / "preprocessing"

N_COMPONENTS = 30
CHUNK_SIZE = 5000


# ==========================================================
# FIND HYPERSPECTRAL CUBE
# ==========================================================

def find_hyperspectral_cube(mat_data):

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append((key, value))

    if not candidates:
        raise ValueError("No 3D hyperspectral array found.")

    variable_name, cube = max(
        candidates,
        key=lambda item: item[1].size
    )

    return variable_name, cube


# ==========================================================
# LOAD ONE FILE
# ==========================================================

def load_cube(mat_file):

    mat_data = scipy.io.loadmat(mat_file)

    variable_name, cube = find_hyperspectral_cube(mat_data)

    cube = cube.astype(np.float32)

    if np.isnan(cube).any():
        raise ValueError(f"NaN values found in {mat_file.name}")

    if np.isinf(cube).any():
        raise ValueError(f"Infinite values found in {mat_file.name}")

    return variable_name, cube


# ==========================================================
# GET SPLIT FROM FILE NUMBER
# ==========================================================

def get_split(filename):

    """
    Files 1-6 -> train
    File 7    -> validation
    File 8    -> test
    """

    file_number = int(filename.stem.split("_")[-1])

    if file_number <= 6:
        return "train"

    elif file_number == 7:
        return "val"

    elif file_number == 8:
        return "test"

    else:
        raise ValueError(
            f"Unexpected file number in: {filename.name}"
        )


# ==========================================================
# COLLECT FILES
# ==========================================================

def collect_files():

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    class_dirs = sorted(
        folder
        for folder in RAW_DIR.iterdir()
        if folder.is_dir()
    )

    for class_dir in class_dirs:

        mat_files = sorted(class_dir.glob("*.mat"))

        for mat_file in mat_files:

            split = get_split(mat_file)

            splits[split].append(
                (class_dir.name, mat_file)
            )

    return splits


# ==========================================================
# FIT SCALER ON TRAIN DATA ONLY
# ==========================================================

def fit_scaler(train_files):

    print("\n" + "=" * 60)
    print("FITTING SHARED SCALER ON TRAIN DATA ONLY")
    print("=" * 60)

    scaler = StandardScaler()

    for index, (class_name, mat_file) in enumerate(
        train_files, start=1
    ):

        print(
            f"[{index}/{len(train_files)}] "
            f"Scaler fit: {class_name}/{mat_file.name}"
        )

        _, cube = load_cube(mat_file)

        pixels = cube.reshape(-1, cube.shape[-1])

        # Process in chunks to reduce memory usage
        for start in range(0, len(pixels), CHUNK_SIZE):

            chunk = pixels[
                start:start + CHUNK_SIZE
            ]

            scaler.partial_fit(chunk)

    return scaler


# ==========================================================
# FIT PCA ON TRAIN DATA ONLY
# ==========================================================

def fit_pca(train_files, scaler):

    print("\n" + "=" * 60)
    print("FITTING SHARED PCA ON TRAIN DATA ONLY")
    print("=" * 60)

    pca = IncrementalPCA(
        n_components=N_COMPONENTS,
        batch_size=CHUNK_SIZE
    )

    for index, (class_name, mat_file) in enumerate(
        train_files, start=1
    ):

        print(
            f"[{index}/{len(train_files)}] "
            f"PCA fit: {class_name}/{mat_file.name}"
        )

        _, cube = load_cube(mat_file)

        pixels = cube.reshape(-1, cube.shape[-1])

        for start in range(0, len(pixels), CHUNK_SIZE):

            chunk = pixels[
                start:start + CHUNK_SIZE
            ]

            chunk_scaled = scaler.transform(chunk)

            pca.partial_fit(chunk_scaled)

    print(
        f"\nTotal PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    return pca


# ==========================================================
# TRANSFORM AND SAVE
# ==========================================================

def transform_and_save(files, split_name, scaler, pca):

    print("\n" + "=" * 60)
    print(
        f"PROCESSING {split_name.upper()} FILES "
        "WITH SHARED SCALER + PCA"
    )
    print("=" * 60)

    for index, (class_name, mat_file) in enumerate(
        files, start=1
    ):

        print(
            f"\n[{index}/{len(files)}] "
            f"{class_name}/{mat_file.name}"
        )

        variable_name, cube = load_cube(mat_file)

        height, width, bands = cube.shape

        pixels = cube.reshape(-1, bands)

        # Same scaler fitted on training data
        pixels_scaled = scaler.transform(pixels)

        # Same PCA fitted on training data
        pixels_pca = pca.transform(pixels_scaled)

        cube_pca = pixels_pca.reshape(
            height,
            width,
            N_COMPONENTS
        ).astype(np.float32)

        output_class_dir = (
            OUTPUT_DIR /
            split_name /
            class_name
        )

        output_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_class_dir /
            f"{mat_file.stem}_pca{N_COMPONENTS}.npy"
        )

        np.save(output_file, cube_pca)

        print(f"Variable: {variable_name}")
        print(f"Original shape: {cube.shape}")
        print(f"Processed shape: {cube_pca.shape}")
        print(f"Saved: {output_file}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("TERRASPECTRA - SHARED PCA PREPROCESSING")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found:\n{RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Collect files
    # ------------------------------------------------------

    splits = collect_files()

    print("\nDATASET SPLIT")

    for split_name, files in splits.items():

        print(
            f"{split_name.upper()}: "
            f"{len(files)} files"
        )

    # Expected:
    # Train = 30
    # Val   = 5
    # Test  = 5

    # ------------------------------------------------------
    # Fit preprocessing using TRAIN ONLY
    # ------------------------------------------------------

    scaler = fit_scaler(splits["train"])

    scaler_path = MODELS_DIR / "shared_scaler.joblib"

    joblib.dump(
        scaler,
        scaler_path
    )

    print(f"\nScaler saved: {scaler_path}")

    pca = fit_pca(
        splits["train"],
        scaler
    )

    pca_path = MODELS_DIR / "shared_pca30.joblib"

    joblib.dump(
        pca,
        pca_path
    )

    print(f"PCA saved: {pca_path}")

    # ------------------------------------------------------
    # Transform all splits
    # ------------------------------------------------------

    transform_and_save(
        splits["train"],
        "train",
        scaler,
        pca
    )

    transform_and_save(
        splits["val"],
        "val",
        scaler,
        pca
    )

    transform_and_save(
        splits["test"],
        "test",
        scaler,
        pca
    )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("SHARED PCA PREPROCESSING COMPLETED")
    print("=" * 60)

    print("Train files processed:", len(splits["train"]))
    print("Validation files processed:", len(splits["val"]))
    print("Test files processed:", len(splits["test"]))

    print("\nSaved preprocessing models:")
    print(scaler_path)
    print(pca_path)

    print("\nOutput folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()