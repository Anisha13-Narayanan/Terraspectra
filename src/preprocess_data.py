from pathlib import Path

import numpy as np
import scipy.io
from sklearn.decomposition import PCA


# ==========================================================
# TERRASPECTRA - AUTOMATIC MULTI-CLASS PREPROCESSING PIPELINE
# ==========================================================

PROJECT_ROOT = Path(r"E:\Terraspectra")

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tomato_hsi"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

N_COMPONENTS = 30


def find_hyperspectral_cube(mat_data):
    """
    Find the main 3D numerical array inside a MATLAB file.
    Expected shape: Height x Width x Spectral Bands
    """

    candidates = []

    for key, value in mat_data.items():

        # Ignore MATLAB metadata
        if key.startswith("__"):
            continue

        # Look for 3D NumPy arrays
        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append((key, value))

    if not candidates:
        raise ValueError("No 3D hyperspectral array found.")

    # Select the largest 3D array
    variable_name, cube = max(
        candidates,
        key=lambda item: item[1].size
    )

    return variable_name, cube


def preprocess_cube(cube):
    """
    Clean, normalize, and apply PCA.
    """

    # Convert to float32 early to reduce memory usage
    cube = cube.astype(np.float32)

    # Check invalid values
    nan_count = np.isnan(cube).sum()
    inf_count = np.isinf(cube).sum()

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Invalid data found: NaN={nan_count}, Inf={inf_count}"
        )

    # -------------------------------
    # Min-Max Normalization
    # -------------------------------

    cube_min = cube.min()
    cube_max = cube.max()

    if cube_max == cube_min:
        raise ValueError("Cannot normalize: all values are identical.")

    cube_normalized = (
        (cube - cube_min) / (cube_max - cube_min)
    )

    # -------------------------------
    # PCA
    # -------------------------------

    height, width, bands = cube_normalized.shape

    # Convert H x W x Bands -> Pixels x Bands
    pixels = cube_normalized.reshape(-1, bands)

    n_components = min(
        N_COMPONENTS,
        pixels.shape[0],
        pixels.shape[1]
    )

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    pixels_pca = pca.fit_transform(pixels)

    # Convert back to H x W x Components
    cube_pca = pixels_pca.reshape(
        height,
        width,
        n_components
    ).astype(np.float32)

    explained_variance = (
        pca.explained_variance_ratio_.sum() * 100
    )

    return cube_pca, explained_variance


def process_file(mat_file, class_name):
    """
    Process one .mat file and save the result.
    """

    print("\n" + "-" * 60)
    print(f"Processing: {class_name}/{mat_file.name}")

    # Load MATLAB file
    mat_data = scipy.io.loadmat(mat_file)

    # Automatically find hyperspectral cube
    variable_name, cube = find_hyperspectral_cube(mat_data)

    print(f"Variable: {variable_name}")
    print(f"Original shape: {cube.shape}")

    # Preprocess
    cube_pca, explained_variance = preprocess_cube(cube)

    print(f"Processed shape: {cube_pca.shape}")
    print(
        f"Explained variance: {explained_variance:.2f}%"
    )

    # Create matching output class folder
    class_output_dir = PROCESSED_DIR / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Save using same filename
    output_file = (
        class_output_dir /
        f"{mat_file.stem}_pca{N_COMPONENTS}.npy"
    )

    np.save(output_file, cube_pca)

    print(f"Saved: {output_file}")


def main():

    print("=" * 60)
    print("TERRASPECTRA - AUTOMATIC DATA PREPROCESSING")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found:\n{RAW_DIR}"
        )

    # Find all class folders
    class_dirs = sorted(
        [
            folder
            for folder in RAW_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    if not class_dirs:
        raise ValueError(
            "No class folders found in the raw dataset directory."
        )

    total_files = 0
    successful_files = 0
    failed_files = []

    # Process every class
    for class_dir in class_dirs:

        class_name = class_dir.name

        print("\n" + "=" * 60)
        print(f"CLASS: {class_name.upper()}")
        print("=" * 60)

        # Find all .mat files in this class folder
        mat_files = sorted(class_dir.glob("*.mat"))

        print(f"Files found: {len(mat_files)}")

        for mat_file in mat_files:

            total_files += 1

            try:
                process_file(mat_file, class_name)
                successful_files += 1

            except Exception as error:

                print(
                    f"FAILED: {mat_file.name}"
                )
                print(f"Reason: {error}")

                failed_files.append(
                    (class_name, mat_file.name, str(error))
                )

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:

        print("\nFailed files:")

        for class_name, filename, error in failed_files:
            print(
                f"- {class_name}/{filename}: {error}"
            )

    print("\nAutomatic preprocessing completed.")


if __name__ == "__main__":
    main()