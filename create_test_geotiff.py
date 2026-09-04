import numpy as np
import rasterio
from rasterio.transform import from_origin

OUTPUT = r"E:\Terraspectra\data\test.tif"

height = 64
width = 64
bands = 550

# Create a test hyperspectral cube with 550 spectral bands
data = np.random.rand(bands, height, width).astype("float32")

transform = from_origin(
    76.0,       # west
    11.0,       # north
    0.0001,     # pixel width
    0.0001      # pixel height
)

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": bands,
    "dtype": "float32",
    "crs": "EPSG:4326",
    "transform": transform,
}

with rasterio.open(OUTPUT, "w", **profile) as dst:
    dst.write(data)

print("Created:", OUTPUT)
print("Shape:", data.shape)
print("Bands:", bands)