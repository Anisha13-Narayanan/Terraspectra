import io
import unittest

import h5py
import numpy as np
import scipy.io
import rasterio
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.main import app


class TerraSpectraApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.patch = np.zeros((32, 32, 30), dtype=np.float32).tolist()

    def test_dashboard_and_health(self):
        dashboard = self.client.get("/")
        health = self.client.get("/health")
        model_info = self.client.get("/model-info")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("TerraSpectra", dashboard.text)
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_available"])
        self.assertEqual(health.json()["framework"], "TensorFlow/Keras")
        self.assertEqual(model_info.status_code, 200)
        self.assertEqual(model_info.json()["deployment_model"], "best_shared_pca_3dcnn.keras")
        self.assertEqual(model_info.json()["pytorch_hybrid_status"], "Experimental one-epoch CPU baseline only; it is retained for architecture research and is not served by the production API.")

    def test_predict_returns_five_probabilities(self):
        response = self.client.post("/predict", json={"patch": self.patch})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn(body["predicted_class"], body["probabilities"])
        self.assertEqual(len(body["probabilities"]), 5)
        self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0, places=5)

    def test_explain_returns_expected_shapes(self):
        response = self.client.post("/explain", json={"patch": self.patch})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(np.asarray(body["spatial_importance"]).shape, (32, 32))
        self.assertEqual(np.asarray(body["spectral_importance"]).shape, (30,))

    def test_invalid_patch_is_rejected(self):
        invalid_patch = np.zeros((31, 32, 30), dtype=np.float32).tolist()
        response = self.client.post("/predict", json={"patch": invalid_patch})

        self.assertEqual(response.status_code, 422)

    def test_mat_upload_returns_patch_coordinates(self):
        buffer = io.BytesIO()
        scipy.io.savemat(
            buffer,
            {"cube": np.zeros((64, 64, 550), dtype=np.float32)},
        )
        response = self.client.post(
            "/predict-mat",
            files={"file": ("synthetic.mat", buffer.getvalue(), "application/octet-stream")},
        )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["patch_count"], 4)
        self.assertEqual(len(body["patch_predictions"]), 4)
        self.assertEqual(body["patch_predictions"][0]["row"], 0)
        self.assertEqual(body["patch_predictions"][0]["column"], 0)

    def test_non_mat_upload_is_rejected(self):
        response = self.client.post(
            "/predict-mat",
            files={"file": ("invalid.txt", b"not a MAT file", "text/plain")},
        )

        self.assertEqual(response.status_code, 422)

    def test_hdf5_upload_returns_predictions(self):
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as handle:
            handle.create_dataset(
                "cube",
                data=np.zeros((32, 32, 550), dtype=np.float32),
            )

        response = self.client.post(
            "/predict-geospatial",
            files={"file": ("synthetic.h5", buffer.getvalue(), "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["patch_count"], 1)

    def test_geotiff_upload_preserves_metadata(self):
        buffer = io.BytesIO()
        transform = from_origin(500000, 4600000, 10, 10)
        with MemoryFile() as memory_file:
            with memory_file.open(
                driver="GTiff",
                height=32,
                width=32,
                count=550,
                dtype="float32",
                crs="EPSG:32643",
                transform=transform,
            ) as dataset:
                dataset.write(np.zeros((550, 32, 32), dtype=np.float32))
            buffer.write(memory_file.read())

        response = self.client.post(
            "/predict-geospatial",
            files={"file": ("synthetic.tif", buffer.getvalue(), "image/tiff")},
        )

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["patch_count"], 1)
        self.assertEqual(body["geospatial_metadata"]["crs"], "EPSG:32643")
        self.assertAlmostEqual(body["patch_predictions"][0]["longitude"], 75.0019, places=3)
        self.assertAlmostEqual(body["patch_predictions"][0]["latitude"], 41.5502, places=3)
        self.assertEqual(len(body["patch_predictions"][0]["polygon"]), 4)
        self.assertGreater(body["patch_predictions"][0]["area_acres"], 0)
        self.assertGreater(body["risk_summary"]["total_acres"], 0)
        self.assertLessEqual(
            body["risk_summary"]["at_risk_acres"],
            body["risk_summary"]["total_acres"],
        )


if __name__ == "__main__":
    unittest.main()
