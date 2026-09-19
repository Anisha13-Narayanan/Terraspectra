# TerraSpectra Demonstration Walkthrough

## Before the demo

1. Start the API from the project root:

   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```

2. Start the React dashboard in another terminal:

   ```powershell
   cd frontend
   npm install
   $env:VITE_API_URL = "http://127.0.0.1:8000"
   npm run dev
   ```

3. Open `http://localhost:5173`. A Mapbox token is optional; without it, the
   pixel-coordinate map remains available.

## Five-minute demonstration

1. Open the dashboard and point out the model-online health indicator.
2. Upload a compatible MAT, HDF5, or GeoTIFF hyperspectral cube.
3. Run field analysis and explain that the system applies the training-fitted
   scaler and shared PCA, then processes 32×32 patches.
4. Inspect the disease overlay. Hover a polygon to show class and confidence.
5. For a georeferenced GeoTIFF, point out the CRS, total field acres, and
   estimated at-risk acres.
6. Download the JSON analysis report. It includes the image-level result,
   class probabilities, spatial metadata, risk summary, and every patch output.
7. Upload another observation dated differently and use the history slider to
   compare the saved in-session analyses.

## Presentation message

Describe TerraSpectra as a hyperspectral disease-classification and field-risk
mapping prototype. Do not claim validated pre-symptom forecasting: that needs
longitudinal field data with disease-onset labels.
