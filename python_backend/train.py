"""
Scikit-learn Random Forest + KNeighbors comparison training script for ISL landmarks.
Serializes the trained pipeline, scaler, label encoder, and evaluation report with Joblib/JSON.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocess import load_landmark_csv, session_wise_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "isl_landmarks.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
JOBLIB_PATH = os.path.join(MODELS_DIR, "isl_rf_pipeline.joblib")
REPORT_PATH = os.path.join(MODELS_DIR, "python_evaluation_report.json")


def train_and_evaluate(csv_path: str = CSV_PATH) -> dict:
    records = load_landmark_csv(csv_path)
    if len(records) < 10:
        raise ValueError("At least 10 valid landmark samples are required for training.")

    train_rec, val_rec, test_rec, split_strategy = session_wise_split(records)

    le = LabelEncoder()
    all_labels = sorted({r["label"] for r in records})
    le.fit(all_labels)

    X_train = np.array([r["features"] for r in train_rec], dtype=np.float32)
    y_train = le.transform([r["label"] for r in train_rec])
    X_test = np.array([r["features"] for r in test_rec], dtype=np.float32)
    y_test = le.transform([r["label"] for r in test_rec])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    t0 = time.perf_counter()
    y_pred_rf = rf.predict(X_test_scaled)
    rf_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X_test_scaled))

    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(X_train_scaled, y_train)

    t1 = time.perf_counter()
    y_pred_knn = knn.predict(X_test_scaled)
    knn_ms = (time.perf_counter() - t1) * 1000.0 / max(1, len(X_test_scaled))

    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_rf, average="macro", zero_division=0
    )
    knn_acc = float(accuracy_score(y_test, y_pred_knn))
    knn_p, knn_r, knn_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_knn, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred_rf, labels=list(range(len(all_labels)))).tolist()
    cls_report = classification_report(
        y_test, y_pred_rf, target_names=all_labels, output_dict=True, zero_division=0
    )

    # Fit production model on all verified records
    X_all = np.array([r["features"] for r in records], dtype=np.float32)
    y_all = le.transform([r["label"] for r in records])
    prod_scaler = StandardScaler()
    X_all_scaled = prod_scaler.fit_transform(X_all)
    prod_rf = RandomForestClassifier(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1)
    prod_rf.fit(X_all_scaled, y_all)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {"model": prod_rf, "scaler": prod_scaler, "label_encoder": le},
        JOBLIB_PATH,
    )

    report = {
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "totalSamples": len(records),
        "trainSamples": len(train_rec),
        "validationSamples": len(val_rec),
        "testSamples": len(test_rec),
        "classesEvaluated": all_labels,
        "splitStrategy": split_strategy,
        "accuracy": round(rf_acc, 4),
        "macroPrecision": round(float(rf_p), 4),
        "macroRecall": round(float(rf_r), 4),
        "macroF1": round(float(rf_f1), 4),
        "confusionMatrix": cm,
        "perClassReport": cls_report,
        "comparison": [
            {
                "modelName": "Scikit-learn RandomForestClassifier (150 trees)",
                "accuracy": round(rf_acc, 4),
                "macroPrecision": round(float(rf_p), 4),
                "macroRecall": round(float(rf_r), 4),
                "macroF1": round(float(rf_f1), 4),
                "inferenceLatencyMs": round(rf_ms, 3),
            },
            {
                "modelName": "Scikit-learn KNeighborsClassifier (k=3)",
                "accuracy": round(knn_acc, 4),
                "macroPrecision": round(float(knn_p), 4),
                "macroRecall": round(float(knn_r), 4),
                "macroF1": round(float(knn_f1), 4),
                "inferenceLatencyMs": round(knn_ms, 3),
            },
        ],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(json.dumps(summary, indent=2))
