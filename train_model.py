import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Deterministic demonstration data for student result prediction.
# Replace this generated data with your course dataset if your instructor provided one.
data = pd.DataFrame({
    "study_hours": [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,9,
                    1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,9,1,3,5,7,9,2,4,6,8,10,
                    1,2,4,5,7,8,3,6,9,10],
    "attendance": [50,55,60,65,70,75,80,90,45,52,62,68,72,78,85,95,50,58,66,74,82,88,92,96,
                   40,54,61,67,73,79,86,94,48,59,64,71,77,83,91,97,43,63,76,87,98,51,69,81,93,99,
                   46,57,70,75,84,90,60,80,95,100],
    "previous_score": [35,40,45,50,55,60,65,75,30,38,47,52,58,63,70,82,34,44,51,59,66,72,78,88,
                       25,39,46,53,57,64,71,80,32,43,49,56,62,68,76,85,28,48,60,73,90,36,54,67,79,92,
                       31,42,58,61,74,83,45,69,87,95]
})
data["result"] = ((data["study_hours"] * 4 + data["attendance"] * 0.25 + data["previous_score"] * 0.45) >= 42).astype(int)
data.to_csv("student_results.csv", index=False)

X = data[["study_hours", "attendance", "previous_score"]]
y = data["result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}
joblib.dump(model, "student_result_model.pkl")
with open("metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)
print("Training complete.")
print(json.dumps(metrics, indent=4))
