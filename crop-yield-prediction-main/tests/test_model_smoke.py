"""Loads the committed model artifacts and performs one real inference."""
import math
import unittest

import app as api


class TrainedModelSmokeTest(unittest.TestCase):
    def test_committed_model_and_preprocessor_can_predict(self):
        api.load_artifacts()
        dataset = api.load_dataset()
        row = dataset.iloc[0]
        payload = {
            field: row[field]
            for field in api.FEATURES
        }
        prediction = float(api.predict_rows([payload])[0])
        self.assertTrue(math.isfinite(prediction))
        self.assertIsNotNone(api.model)
        self.assertIsNotNone(api.preprocessor)


if __name__ == "__main__":
    unittest.main()
