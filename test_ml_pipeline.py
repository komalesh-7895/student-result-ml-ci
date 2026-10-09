import json
import unittest
from pathlib import Path
import joblib
import pandas as pd

class TestMLPipeline(unittest.TestCase):
    def test_model_exists_and_loads(self):
        self.assertTrue(Path("student_result_model.pkl").exists())
        model = joblib.load("student_result_model.pkl")
        self.assertTrue(hasattr(model, "predict"))

    def test_metrics_are_valid(self):
        self.assertTrue(Path("metrics.json").exists())
        with open("metrics.json", encoding="utf-8") as file:
            metrics = json.load(file)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreater(metrics["training_records"], 0)
        self.assertGreater(metrics["testing_records"], 0)

    def test_demo_dataset_exists(self):
        self.assertTrue(Path("student_results.csv").exists())
        data = pd.read_csv("student_results.csv")
        self.assertGreater(len(data), 0)
        self.assertIn("result", data.columns)

if __name__ == "__main__":
    unittest.main()
