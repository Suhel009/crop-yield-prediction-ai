import unittest

import numpy as np
import pandas as pd

import app as api


class IdentityPreprocessor:
    def transform(self, frame):
        return frame


class ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 2.5, dtype=float)


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "model": api.model,
            "preprocessor": api.preprocessor,
            "options_cache": api.options_cache,
            "dataset_cache": api.dataset_cache,
            "testing": api.app.config.get("TESTING", False),
        }
        api.model = ConstantModel()
        api.preprocessor = IdentityPreprocessor()
        api.options_cache = {
            "states": ["Uttar Pradesh", "Bihar"],
            "crops": ["Rice", "Wheat", "Maize", "Coconut"],
            "seasons": ["Kharif", "Rabi"],
            "year_min": 2000,
            "year_max": 2026,
            "dataset_rows": 17321,
            "input_units": {
                "Area": "hectares internally; acres accepted and converted",
                "Annual_Rainfall": "millimetres per year (mm)",
                "Fertilizer": "total kilograms (kg) over the selected crop area",
                "Pesticide": "total kilograms (kg) over the selected crop area",
            },
            "data_quality": {
                "checked_records": 100,
                "match_within_5pct_records": 66,
                "match_within_5pct_pct": 66.0,
                "mismatch_records": 34,
                "tolerance": "5%",
            },
        }
        api.dataset_cache = pd.DataFrame([
            {"Crop": "Rice", "Area": 1000, "Production": 2000, "Annual_Rainfall": 900,
             "Fertilizer": 50000, "Pesticide": 100, "Yield": 2.0},
            {"Crop": "Rice", "Area": 900, "Production": 1800, "Annual_Rainfall": 950,
             "Fertilizer": 45000, "Pesticide": 90, "Yield": 2.0},
            {"Crop": "Wheat", "Area": 800, "Production": 1600, "Annual_Rainfall": 700,
             "Fertilizer": 40000, "Pesticide": 80, "Yield": 2.0},
            {"Crop": "Maize", "Area": 600, "Production": 900, "Annual_Rainfall": 800,
             "Fertilizer": 30000, "Pesticide": 60, "Yield": 1.5},
            {"Crop": "Coconut", "Area": 100, "Production": 100000, "Annual_Rainfall": 1200,
             "Fertilizer": 10000, "Pesticide": 20, "Yield": 1000},
        ])
        api.app.config["TESTING"] = True
        self.client = api.app.test_client()

    def tearDown(self):
        api.model = self.previous["model"]
        api.preprocessor = self.previous["preprocessor"]
        api.options_cache = self.previous["options_cache"]
        api.dataset_cache = self.previous["dataset_cache"]
        api.app.config["TESTING"] = self.previous["testing"]

    def valid_payload(self):
        return {
            "Year": 2024,
            "State": "Uttar Pradesh",
            "Crop": "Rice",
            "Season": "Kharif",
            "Area": 1000,
            "Area_Unit": "hectare",
            "Annual_Rainfall": 900,
            "Fertilizer": 50000,
            "Pesticide": 100,
        }

    def test_options_include_verified_dataset_coverage_and_units(self):
        response = self.client.get("/options")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["dataset_rows"], 17321)
        self.assertEqual(body["year_min"], 2000)
        self.assertIn("kg", body["input_units"]["Pesticide"])

    def test_prediction_returns_explicit_yield_and_input_units(self):
        response = self.client.post("/predict", json=self.valid_payload())
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["predicted_yield"], 2.5)
        self.assertEqual(body["yield_unit"], "metric tonnes/hectare (t/ha)")
        self.assertEqual(body["area_hectares"], 1000)
        self.assertEqual(body["pesticide_kg"], 100)
        self.assertIsNone(body["total_production_estimate"])
        self.assertTrue(any("total production is deliberately not calculated" in w.lower() for w in body["warnings"]))

    def test_acres_are_converted_to_hectares_before_prediction(self):
        payload = self.valid_payload()
        payload["Area"] = 100
        payload["Area_Unit"] = "acre"
        values, error = api.parse_prediction_input(payload)
        self.assertIsNone(error)
        self.assertAlmostEqual(values["Area"], 40.468564224)
        self.assertEqual(values["Area_Input"], 100)
        self.assertEqual(values["Area_Unit"], "acre")

    def test_bigha_is_not_silently_assumed(self):
        payload = self.valid_payload()
        payload["Area_Unit"] = "bigha"
        values, error = api.parse_prediction_input(payload)
        self.assertIsNone(values)
        self.assertIn("Bigha varies by region", error)

    def test_prediction_rejects_missing_fields(self):
        response = self.client.post("/predict", json={"Year": 2024})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required fields", response.get_json()["error"])

    def test_prediction_rejects_negative_rainfall(self):
        payload = self.valid_payload()
        payload["Annual_Rainfall"] = -1
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot be negative", response.get_json()["error"])

    def test_prediction_rejects_unknown_crop(self):
        payload = self.valid_payload()
        payload["Crop"] = "Unknown Crop"
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_coconut_unit_is_flagged_as_unverified(self):
        units = api.crop_yield_units("Coconut")
        self.assertIn("unverified", units["yield_unit"])
        self.assertEqual(units["unit_group"], "coconut_unverified")

    def test_comparison_ranks_predictions_highest_first(self):
        payload = self.valid_payload()
        payload["crops"] = ["Rice", "Wheat", "Maize"]
        response = self.client.post("/compare", json=payload)
        self.assertEqual(response.status_code, 200)
        predictions = response.get_json()["predictions"]
        self.assertEqual([item["crop"] for item in predictions], ["Rice", "Wheat", "Maize"])
        self.assertTrue(all(item["yield_unit"] == "metric tonnes/hectare (t/ha)" for item in predictions))

    def test_comparison_blocks_mixed_or_unverified_yield_units(self):
        payload = self.valid_payload()
        payload["crops"] = ["Rice", "Coconut"]
        response = self.client.post("/compare", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("different or unverified yield units", response.get_json()["error"])

    def test_comparison_rejects_one_crop(self):
        payload = self.valid_payload()
        payload["crops"] = ["Rice"]
        response = self.client.post("/compare", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_summary_returns_data_quality_audit(self):
        response = self.client.get("/summary")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["crops"], 4)
        self.assertEqual(body["dataset_records"], 17321)
        self.assertEqual(body["data_quality"]["match_within_5pct_pct"], 66.0)

    def test_yield_production_audit_counts_only_consistent_rows(self):
        frame = pd.DataFrame([
            {"Area": 10, "Production": 20, "Yield": 2},
            {"Area": 10, "Production": 40, "Yield": 2},
        ])
        result = api.dataset_quality_audit(frame)
        self.assertEqual(result["checked_records"], 2)
        self.assertEqual(result["match_within_5pct_records"], 1)
        self.assertEqual(result["mismatch_records"], 1)


if __name__ == "__main__":
    unittest.main()
