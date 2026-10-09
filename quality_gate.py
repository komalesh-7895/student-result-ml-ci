import json
from pathlib import Path

metrics_path = Path("metrics.json")
model_path = Path("student_result_model.pkl")
dataset_path = Path("student_results.csv")

for path in (metrics_path, model_path, dataset_path):
    if not path.exists():
        raise SystemExit(f"QUALITY GATE FAILED: required output missing: {path}")

with metrics_path.open(encoding="utf-8") as file:
    metrics = json.load(file)

threshold = 0.70
accuracy = float(metrics.get("accuracy", 0))
print(f"Model accuracy: {accuracy:.4f}; required minimum: {threshold:.2f}")
if accuracy < threshold:
    raise SystemExit("QUALITY GATE FAILED: accuracy below threshold")

print("QUALITY GATE PASSED")
