# 🌾 Crop Yield Prediction — Unit-Aware ML Dashboard

An end-to-end crop-yield estimation project built with a trained Random Forest model, Flask API, and a responsive HTML/CSS/JavaScript frontend.

## Dataset and evidence

The repository contains a cleaned 17,321-row CSV (from an original 17,400-row training split), covering 31 states/regions, 62 crops, 6 seasons, and years 2000–2026.

The project uses the following input conventions, cross-checked against the Kaggle dataset lineage:
- Area: hectares (ha)
- Annual rainfall: millimetres per year (mm)
- Fertilizer: total kilograms (kg) for the crop record
- Pesticide: total kilograms (kg) for the crop record
- Production: metric tonnes in the source description
- Yield: target column documented as production per unit area; conventionally metric tonnes per hectare for mass crops

Reference dataset documentation:
- Kaggle dataset lineage: https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset
- A public mirror describing the columns and units: https://github.com/Arihant-Bhandari/CROP_YIELD
- The 2000–2026 dataset description states that 2021–2026 records were trend-extended and gap-filled, rather than being all direct official observations: https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction

These are dataset conventions, not a claim that every record is error-free.

## Input units and conversion

The model's feature order remains:
`Year`, `State`, `Crop`, `Season`, `Area`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`.

The frontend accepts:
- Area in hectares or acres. Acres are converted to hectares using 1 acre = 0.40468564224 hectares before inference.
- Rainfall in mm/year.
- Fertilizer in total kg across the entered crop area (not kg/ha).
- Pesticide in total kg across the entered crop area (not kg/ha).

The application does not accept bigha directly because bigha size varies by state/region and sometimes by local land-record convention. Convert the local bigha quantity to hectares before entering it. Liquid pesticide volume in litres must not be entered as kilograms unless the product density is known and a mass conversion has been made.

The backend retains fertilizer and pesticide values in kg as represented by the project CSV; do not divide or multiply them by 1,000 unless the underlying training data is deliberately re-preprocessed and the model is retrained to the new unit convention.

## Yield result: important data-quality limitation

The model predicts the dataset's `Yield` target. The result is a model estimate, not a guaranteed harvest or an agronomic recommendation.

A repository audit compared `Yield` against `Production / Area` for all 17,321 valid rows. Only **66.12%** reconcile within a 5% relative tolerance; **33.88%** do not. Consequently:
- The app reports predicted yield and its declared target unit, with a visible data-quality warning.
- Coconut is labelled as **dataset units (physical unit unverified)** because its values are extremely large and the physical unit cannot safely be established from the repository's current data documentation.
- The app deliberately does **not** calculate total production by multiplying predicted yield by area. Doing so would imply consistency the source data does not currently demonstrate.
- Crop comparisons between Coconut and the other crops are blocked because their target scales are not confirmed to be comparable.

The audit percentage is shown in the dashboard and included in the API response. The audit tolerance is documented in the API response as a 5% relative difference.

## Historical / extended data caveat

The linked 2000–2026 dataset card says:
- 2000–2020 is based on historical agricultural statistics.
- 2021–2026 was extended using state/crop trend extrapolation and gap filling.

For years 2021–2026, the app adds a warning. The year field is a crop-year feature; entering 2026 does not fetch or forecast actual future weather. Rainfall remains a user-provided annual value.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information and unit contract |
| GET | `/health` | Checks required files are present |
| GET | `/options` | Dropdown values, year range, input units, data-quality audit |
| GET | `/summary` | Dataset summary and data-quality audit |
| POST | `/predict` | Predict one crop |
| POST | `/compare` | Compare 2–8 crops with the same target-unit group |

Example request (values are illustrative):

```json
{
  "Year": 2024,
  "State": "Uttar Pradesh",
  "Crop": "Rice",
  "Season": "Kharif",
  "Area": 1000,
  "Area_Unit": "acre",
  "Annual_Rainfall": 900,
  "Fertilizer": 50000,
  "Pesticide": 100
}
```

Example response fields:
- `predicted_yield`
- `yield_unit`
- `area_hectares`
- `fertilizer_kg`
- `pesticide_kg`
- `annual_rainfall_mm`
- `warnings`
- `total_production_estimate: null` (intentionally not calculated until source records reconcile)

## Local setup

Python 3.10 or 3.11 is recommended until the exact scikit-learn version used to create the committed joblib artifacts is confirmed.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `frontend/index.html` with VS Code Live Server or another static server. The default API URL in `frontend/script.js` points to the Render deployment. For local API use, set `window.CROP_YIELD_API_URL = "http://127.0.0.1:5000"` before loading `script.js`.

## Deployment

### Backend (Render)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Set `FRONTEND_ORIGINS` to the comma-separated list of exact Netlify/custom domains.

### Frontend (Netlify)
- Base directory: `frontend`
- Publish directory: `.` when base directory is `frontend`
- No build command is required for the static frontend.

## Model performance

Metrics previously documented in earlier project reports are historical notebook results. They have not been re-established by this unit update. Re-run training with a held-out, preferably time-based evaluation before presenting them as a current accuracy guarantee.

## Engineering safeguards

- File paths resolve relative to `app.py`.
- Model artifacts load lazily.
- Dataset/options are cached per worker.
- Inputs are validated server-side.
- Acre conversion is applied before model inference.
- Warnings are generated for out-of-range inputs and model outputs.
- Total-production arithmetic is suppressed while the source data reconciliation audit fails.
- API regression tests, real-model smoke test, and frontend syntax check run in GitHub Actions.
