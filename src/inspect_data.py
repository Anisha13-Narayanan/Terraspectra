from pathlib import Path
import scipy.io
import h5py

# Path to the downloaded hyperspectral file
file_path = Path(
    r"E:\Terraspectra\data\raw\tomato_hsi\Alternaria_alternata_1.mat"
)

print("=" * 60)
print("TERRASPECTRA - HYPERSPECTRAL DATA INSPECTION")
print("=" * 60)
print(f"\nFile: {file_path.name}")
print(f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

# First try: Standard MATLAB .mat format
try:
    print("\nTrying scipy.io.loadmat()...")
    data = scipy.io.loadmat(file_path)

    print("\nVariables found:")
    for key, value in data.items():
        if not key.startswith("__"):
            print(f"{key}: shape={getattr(value, 'shape', 'N/A')}, "
                  f"dtype={getattr(value, 'dtype', type(value))}")

except NotImplementedError:
    # Second try: MATLAB v7.3 / HDF5 format
    print("\nThis appears to be an HDF5-based MATLAB file.")
    print("Trying h5py...")

    with h5py.File(file_path, "r") as file:
        print("\nVariables found:")
        for key in file.keys():
            item = file[key]
            print(f"{key}: shape={getattr(item, 'shape', 'N/A')}, "
                  f"dtype={getattr(item, 'dtype', type(item))}")

except Exception as error:
    print(f"\nError: {error}")