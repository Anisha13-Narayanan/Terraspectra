from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.io


SUPPORTED_SUFFIXES = {".mat", ".h5", ".hdf5", ".tif", ".tiff"}


def _largest_3d_array(values):
    candidates = [
        value
        for value in values
        if isinstance(value, np.ndarray) and value.ndim == 3
    ]
    if not candidates:
        raise ValueError("No 3D hyperspectral cube found")
    return max(candidates, key=lambda value: value.size)


def _load_mat(contents):
    data = scipy.io.loadmat(BytesIO(contents))
    return _largest_3d_array(data.values()), {}


def _load_hdf5(contents):
    import h5py

    arrays = []
    with h5py.File(BytesIO(contents), "r") as handle:
        def collect(_, value):
            if isinstance(value, h5py.Dataset) and value.ndim == 3:
                arrays.append(np.asarray(value))
        handle.visititems(collect)
    return _largest_3d_array(arrays), {}


def _load_geotiff(contents):
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(contents) as memory_file:
        with memory_file.open() as dataset:
            cube = dataset.read().transpose(1, 2, 0)
            metadata = {
                "crs": str(dataset.crs) if dataset.crs else None,
                "transform": list(dataset.transform),
                "width": dataset.width,
                "height": dataset.height,
                "bounds": list(dataset.bounds),
            }
    return cube, metadata


def load_cube(contents: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported formats are .mat, .h5, .hdf5, .tif, and .tiff")
    if suffix == ".mat":
        return _load_mat(contents)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5(contents)
    return _load_geotiff(contents)
