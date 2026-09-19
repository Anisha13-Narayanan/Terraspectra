"""Create a clearly labelled mock multi-date satellite farm dataset for demos.

The generated 550-band GeoTIFFs exercise the real Rasterio, PCA, tiling and
Deck.gl paths. They are not satellite observations and must not be used for
model accuracy or early-forecasting claims.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "mock_satellite_farm"
DATES = ("2026-05-01", "2026-05-15", "2026-05-29")
HEIGHT = WIDTH = 256
BANDS = 550
TRANSFORM = from_origin(76.0, 11.0, 0.0001, 0.0001)


def create_scene(scene_index: int) -> np.ndarray:
    rng = np.random.default_rng(42 + scene_index)
    cube = rng.normal(0.35, 0.025, size=(BANDS, HEIGHT, WIDTH)).astype("float32")
    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    center_row, center_column = 128, 128
    radius = 32 + scene_index * 18
    risk_zone = (rows - center_row) ** 2 + (columns - center_column) ** 2 <= radius ** 2
    spectral_signature = np.linspace(-0.06, 0.10, BANDS, dtype="float32")[:, None]
    cube[:, risk_zone] += spectral_signature
    return cube


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_index, observation_date in enumerate(DATES):
        scene_path = OUTPUT_DIR / f"mock_farm_{observation_date}.tif"
        with rasterio.open(
            scene_path, "w", driver="GTiff", height=HEIGHT, width=WIDTH,
            count=BANDS, dtype="float32", crs="EPSG:4326", transform=TRANSFORM,
            compress="deflate",
        ) as dataset:
            dataset.write(create_scene(scene_index))
            dataset.update_tags(
                mock_data="true",
                description="Synthetic multi-date hyperspectral farm scene for TerraSpectra demo only",
                observation_date=observation_date,
            )
        records.append({
            "observation_date": observation_date,
            "scene": scene_path.name,
            "simulated_risk_zone_radius_pixels": 32 + scene_index * 18,
            "data_status": "synthetic demonstration data; not ground truth",
        })

    with (OUTPUT_DIR / "temporal_scene_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Created {len(records)} mock satellite GeoTIFFs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
