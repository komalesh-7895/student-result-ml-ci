# Student Result Prediction ML CI

Practical 4: Preserve validated ML model artifacts using GitHub Actions.

The workflow trains a model, checks the ML quality gate, runs automated tests, and uploads:
- `student_result_model.pkl`
- `metrics.json`
- `student_results.csv`

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
```

The workflow artifact is named `student-result-ml-artifacts` and is retained for 7 days.
