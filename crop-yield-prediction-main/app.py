"""Crop Yield Prediction API with explicit, auditable input units."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_yield_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "crop_yield.csv"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("crop-yield-api")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://crop-yield-prediction-ai.netlify.app,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/*": {"origins": allowed_origins}})

model = None
preprocessor = None
options_cache = None
dataset_cache = None
artifact_lock = Lock()
dataset_lock = Lock()

FEATURES = [
    "Year", "State", "Crop", "Season", "Area",
    "Annual_Rainfall", "Fertilizer", "Pesticide"
]
NUMERIC_FIELDS = ["Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
CATEGORICAL_FIELDS = ["State", "Crop", "Season"]

# Kaggle dataset documentation describes Area in hectares, Fertilizer and
# Pesticide in kilograms, Production in metric tonnes, and Yield as production
# per unit area. Inputs are total quantities over the selected crop area.
ACRE_TO_HECTARE = 0.40468564224
AREA_UNIT_ALIASES = {
    "ha": "hectare", "hectare": "hectare", "hectares": "hectare",
    "acre": "acre", "acres": "acre",
}


def crop_yield_units(crop: str) -> dict:
    """Return a conservative label for the dataset target scale.

    The source data contains unusually large Coconut Yield values (and the
    source documentation does not clearly establish this crop's output unit).
    Do not present Coconut as verified tonnes/ha or nuts/ha.
    """
    if str(crop).strip().casefold() == "coconut":
        return {
            "yield_unit": "dataset units (Coconut; physical unit unverified)",
            "unit_group": "coconut_unverified",
            "unit_note": (
                "Coconut target values are unusually large and the source "
                "does not establish a consistent physical unit. Treat this "
                "as a dataset-scale estimate only."
            ),
        }
    return {
        "yield_unit": "metric tonnes/hectare (t/ha)",
        "unit_group": "mass",
        "unit_note": (
            "The dataset defines Yield as metric-tonne production per "
            "hectare. Source rows still contain production/yield consistency "
            "issues; see the data-quality warning."
        ),
    }


def load_artifacts():
    """Load trusted, version-matched joblib files once per worker."""
    global model, preprocessor
    if model is not None and preprocessor is not None:
        return
    with artifact_lock:
        if model is not None and preprocessor is not None:
            return
        if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
            raise FileNotFoundError("Model artifacts are missing from the models/ directory.")
        loaded_model = joblib.load(MODEL_PATH)
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        model, preprocessor = loaded_model, loaded_preprocessor


def load_dataset():
    """Read the source dataset once per worker."""
    global dataset_cache
    if dataset_cache is not None:
        return dataset_cache
    with dataset_lock:
        if dataset_cache is None:
            if not DATASET_PATH.is_file():
                raise FileNotFoundError("The crop dataset is missing from the dataset/ directory.")
            dataset_cache = pd.read_csv(DATASET_PATH)
    return dataset_cache


def dataset_quality_audit(frame: pd.DataFrame) -> dict:
    """Measure whether the source Yield target reconciles to Production / Area."""
    area = pd.to_numeric(frame["Area"], errors="coerce")
    production = pd.to_numeric(frame["Production"], errors="coerce")
    target = pd.to_numeric(frame["Yield"], errors="coerce")
    valid = (area > 0) & area.notna() & production.notna() & target.notna()
    if not valid.any():
        return {
            "checked_records": 0,
            "match_within_5pct_records": 0,
            "match_within_5pct_pct": 0.0,
            "mismatch_records": 0,
            "tolerance": "5% relative difference between Yield and Production / Area",
        }
    derived = production[valid] / area[valid]
    observed = target[valid]
    matches = np.isclose(
        derived.to_numpy(dtype=float),
        observed.to_numpy(dtype=float),
        rtol=0.05,
        atol=0.0001,
    )
    checked = int(valid.sum())
    matched = int(matches.sum())
    return {
        "checked_records": checked,
        "match_within_5pct_records": matched,
        "match_within_5pct_pct": round(100.0 * matched / checked, 2),
        "mismatch_records": checked - matched,
        "tolerance": "5% relative difference between Yield and Production / Area",
    }


def load_options():
    global options_cache
    if options_cache is None:
        frame = load_dataset()
        years = pd.to_numeric(frame["Year"], errors="coerce").dropna()
        options_cache = {
            "states": sorted(frame["State"].dropna().astype(str).unique().tolist()),
            "crops": sorted(frame["Crop"].dropna().astype(str).unique().tolist()),
            "seasons": sorted(frame["Season"].dropna().astype(str).unique().tolist()),
            "year_min": int(years.min()) if not years.empty else 2000,
            "year_max": int(years.max()) if not years.empty else 2026,
            "dataset_rows": int(len(frame)),
            "data_quality": dataset_quality_audit(frame),
            "input_units": {
                "Area": "hectares internally; acres accepted and converted",
                "Annual_Rainfall": "millimetres per year (mm)",
                "Fertilizer": "total kilograms (kg) over the selected crop area",
                "Pesticide": "total kilograms (kg) over the selected crop area",
            },
        }
    return options_cache


def parse_prediction_input(payload, crop_override=None):
    """Validate values and convert the selected area to hectares for the model."""
    if not isinstance(payload, dict):
        return None, "Send prediction inputs as a JSON object."

    missing = [field for field in FEATURES if field not in payload]
    if missing:
        return None, "Missing required fields: " + ", ".join(missing)

    values = {}
    for field in NUMERIC_FIELDS:
        try:
            number = float(payload[field])
        except (TypeError, ValueError, OverflowError):
            return None, f"{field} must be a valid number."
        if not np.isfinite(number):
            return None, f"{field} must be a finite number."
        values[field] = number

    if not values["Year"].is_integer():
        return None, "Year must be a whole number."
    if values["Area"] <= 0:
        return None, "Area must be greater than zero."
    for field in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
        if values[field] < 0:
            return None, f"{field} cannot be negative."

    area_unit_raw = payload.get("Area_Unit", "hectare")
    if not isinstance(area_unit_raw, str):
        return None, "Area_Unit must be hectare or acre."
    area_unit = AREA_UNIT_ALIASES.get(area_unit_raw.strip().casefold())
    if area_unit is None:
        return None, "Area unit must be hectare (ha) or acre. Bigha varies by region; convert it to hectares first."

    input_area = values["Area"]
    values["Area_Input"] = input_area
    values["Area_Unit"] = area_unit
    if area_unit == "acre":
        values["Area"] = input_area * ACRE_TO_HECTARE

    for field in CATEGORICAL_FIELDS:
        raw = crop_override if field == "Crop" and crop_override is not None else payload.get(field)
        if not isinstance(raw, str) or not raw.strip():
            return None, f"Please select a valid {field.lower()}."
        values[field] = raw.strip()

    return values, None


def validate_categories(values, available):
    allowed = {
        "State": available["states"],
        "Crop": available["crops"],
        "Season": available["seasons"],
    }
    for field, choices in allowed.items():
        if values[field] not in choices:
            return f"Selected {field.lower()} is not in the dataset options."
    return None


def predict_rows(rows):
    load_artifacts()
    input_frame = pd.DataFrame(rows, columns=FEATURES)
    transformed = preprocessor.transform(input_frame)
    predictions = np.asarray(model.predict(transformed), dtype=float).reshape(-1)
    if len(predictions) != len(rows) or not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid prediction values.")
    return predictions


def prediction_warnings(values, prediction):
    """Flag extrapolation and source-data limitations; these are not confidence intervals."""
    warnings = []
    frame = load_dataset()
    crop_rows = frame.loc[
        frame["Crop"].astype(str).str.casefold() == values["Crop"].casefold()
    ]

    if len(crop_rows) < 50:
        warnings.append(
            f"Limited training support: only {len(crop_rows)} records are available for {values['Crop']}."
        )

    for field, label in [
        ("Area", "Area"),
        ("Annual_Rainfall", "Rainfall"),
        ("Fertilizer", "Fertilizer"),
        ("Pesticide", "Pesticide"),
    ]:
        series = pd.to_numeric(crop_rows[field], errors="coerce").dropna()
        if len(series) < 10:
            continue
        lower, upper = series.quantile([0.01, 0.99]).tolist()
        if values[field] < lower or values[field] > upper:
            warnings.append(
                f"{label} is outside the central 98% of historical {values['Crop']} examples; prediction reliability may be lower."
            )

    yield_series = pd.to_numeric(crop_rows["Yield"], errors="coerce").dropna()
    if len(yield_series) >= 10:
        lower, upper = yield_series.quantile([0.01, 0.99]).tolist()
        if prediction < lower or prediction > upper:
            warnings.append(
                f"Predicted yield is outside the central 98% of observed {values['Crop']} target values; treat it as an extrapolation."
            )

    if values["Year"] >= 2021:
        warnings.append(
            "The linked 2000–2026 dataset description says 2021–2026 records are trend-extended/gap-filled, not all direct official observations. This is not a real-time weather forecast."
        )

    return warnings


def build_prediction_result(values, prediction):
    units = crop_yield_units(values["Crop"])
    quality = load_options()["data_quality"]
    warnings = prediction_warnings(values, prediction)
    warnings.append(
        f"Data-quality audit: only {quality['match_within_5pct_pct']}% of records reconcile Yield with Production / Area within 5%. Therefore total production is deliberately not calculated."
    )
    warnings.append(
        "This is a state-level statistical estimate, not a farm-specific guarantee or fertilizer/pesticide recommendation."
    )
    return {
        "predicted_yield": float(prediction),
        "yield_unit": units["yield_unit"],
        "yield_unit_note": units["unit_note"],
        "total_production_estimate": None,
        "total_production_note": (
            "Not calculated: source Yield and Production / Area disagree for a material share of records."
        ),
        "area_hectares": float(values["Area"]),
        "input_area": float(values["Area_Input"]),
        "input_area_unit": values["Area_Unit"],
        "fertilizer_kg": float(values["Fertilizer"]),
        "pesticide_kg": float(values["Pesticide"]),
        "annual_rainfall_mm": float(values["Annual_Rainfall"]),
        "warnings": warnings,
    }


@app.get("/")
def home():
    return jsonify({
        "service": "Crop Yield Prediction API",
        "status": "ok",
        "endpoints": ["/health", "/options", "/summary", "/predict", "/compare"],
        "input_units": {
            "Area": "hectare or acre (acre converted to hectare)",
            "Annual_Rainfall": "mm/year",
            "Fertilizer": "kg total over the selected crop area",
            "Pesticide": "kg total over the selected crop area",
        },
    })


@app.get("/health")
def health():
    artifacts_available = MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file()
    dataset_available = DATASET_PATH.is_file()
    if not artifacts_available or not dataset_available:
        return jsonify({
            "status": "unavailable",
            "model_loaded": model is not None,
            "artifacts_available": artifacts_available,
            "dataset_available": dataset_available,
        }), 503
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "artifacts_available": True,
        "dataset_available": True,
    })


@app.get("/options")
def options():
    try:
        return jsonify(load_options())
    except Exception:
        logger.exception("Could not load dataset options")
        return jsonify({"error": "Crop options are temporarily unavailable."}), 503


@app.get("/summary")
def summary():
    try:
        available = load_options()
        return jsonify({
            "dataset_records": available["dataset_rows"],
            "states": len(available["states"]),
            "crops": len(available["crops"]),
            "seasons": len(available["seasons"]),
            "year_min": available["year_min"],
            "year_max": available["year_max"],
            "data_quality": available["data_quality"],
        })
    except Exception:
        logger.exception("Could not build dataset summary")
        return jsonify({"error": "Dataset summary is temporarily unavailable."}), 503


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    values, error = parse_prediction_input(payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        available = load_options()
        if not available["year_min"] <= values["Year"] <= available["year_max"]:
            return jsonify({
                "error": f"Year must be between {available['year_min']} and {available['year_max']}."
            }), 400
        error = validate_categories(values, available)
        if error:
            return jsonify({"error": error}), 400

        prediction = float(predict_rows([values])[0])
        if prediction < 0:
            return jsonify({"error": "The model returned a negative yield. Please review the model and input data."}), 500
        return jsonify(build_prediction_result(values, prediction))
    except FileNotFoundError as exc:
        logger.exception("Required project artifact missing")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed. Please try again."}), 500


@app.post("/compare")
def compare():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send comparison inputs as a JSON object."}), 400
    crops = payload.get("crops")
    if not isinstance(crops, list) or not 2 <= len(crops) <= 8:
        return jsonify({"error": "Select between 2 and 8 crops to compare."}), 400
    if any(not isinstance(crop, str) for crop in crops) or len(set(crops)) != len(crops):
        return jsonify({"error": "Choose 2–8 different crop names."}), 400

    unit_groups = {crop_yield_units(crop)["unit_group"] for crop in crops}
    if len(unit_groups) != 1:
        return jsonify({
            "error": "These crops use different or unverified yield units. Coconut cannot be ranked against the other crops."
        }), 400

    base_payload = dict(payload)
    base_payload["Crop"] = crops[0]
    base_values, error = parse_prediction_input(base_payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        available = load_options()
        if not available["year_min"] <= base_values["Year"] <= available["year_max"]:
            return jsonify({
                "error": f"Year must be between {available['year_min']} and {available['year_max']}."
            }), 400
        for crop in crops:
            if crop not in available["crops"]:
                return jsonify({"error": f"Crop '{crop}' is not in the dataset options."}), 400
        error = validate_categories(base_values, available)
        if error:
            return jsonify({"error": error}), 400

        rows = [{**base_values, "Crop": crop} for crop in crops]
        predictions = predict_rows(rows)
        results = [
            {
                "crop": crop,
                "predicted_yield": float(prediction),
                "yield_unit": crop_yield_units(crop)["yield_unit"],
            }
            for crop, prediction in zip(crops, predictions)
        ]
        results.sort(key=lambda item: item["predicted_yield"], reverse=True)
        return jsonify({
            "predictions": results,
            "yield_unit": results[0]["yield_unit"],
            "area_hectares": float(base_values["Area"]),
            "note": "Compare only crops with the same target unit; ranks are model estimates, not agronomic recommendations.",
        })
    except FileNotFoundError as exc:
        logger.exception("Required project artifact missing")
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Crop comparison failed")
        return jsonify({"error": "Crop comparison could not be completed. Please try again."}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
