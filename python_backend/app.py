"""
Flask REST API backend for Indian Sign Language (ISL) Real-Time Hand Gesture Recognition.
Deployable to Google Cloud Run, Render, or Railway for persistent Python ML execution.
"""
from __future__ import annotations
import csv
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from preprocess import DUAL_HAND_FEATURE_COUNT, to_canonical_features
from train import CSV_PATH, JOBLIB_PATH, REPORT_PATH, train_and_evaluate

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.42
PIPELINE = None


def load_pipeline():
    global PIPELINE
    if os.path.exists(JOBLIB_PATH):
        PIPELINE = joblib.load(JOBLIB_PATH)
    elif os.path.exists(CSV_PATH):
        train_and_evaluate(CSV_PATH)
        PIPELINE = joblib.load(JOBLIB_PATH)


@app.get("/api/health")
def health():
    load_pipeline()
    return jsonify({
        "status": "ok",
        "engine": "Python Flask + Scikit-Learn RandomForest",
        "modelLoaded": PIPELINE is not None,
    })


@app.post("/api/predict")
def predict():
    started = time.perf_counter()
    load_pipeline()
    if PIPELINE is None:
        return jsonify({"status": "error", "message": "Model not trained."}), 503

    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, list):
        return jsonify({"status": "error", "message": "Missing features list."}), 400

    try:
        canonical = np.array([to_canonical_features(features)], dtype=np.float32)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    scaled = PIPELINE["scaler"].transform(canonical)
    probs = PIPELINE["model"].predict_proba(scaled)[0]
    classes = PIPELINE["label_encoder"].classes_

    ranked = sorted(zip(classes, probs), key=lambda pair: pair[1], reverse=True)
    best_label, best_conf = ranked[0][0], float(ranked[0][1])
    top_candidates = [
        {"label": str(lbl), "confidence": round(float(prob), 4)}
        for lbl, prob in ranked[:3]
    ]
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    if best_conf < CONFIDENCE_THRESHOLD:
        return jsonify({
            "prediction": None,
            "confidence": round(best_conf, 4),
            "status": "uncertain",
            "handCount": int(payload.get("handCount", 1)),
            "isDynamic": False,
            "topCandidates": top_candidates,
            "latencyMs": latency_ms,
            "message": "Low confidence gesture.",
            "modelSourceNote": "Python Scikit-Learn Random Forest",
        })

    return jsonify({
        "prediction": str(best_label),
        "confidence": round(best_conf, 4),
        "status": "detected",
        "handCount": int(payload.get("handCount", 1)),
        "isDynamic": False,
        "topCandidates": top_candidates,
        "latencyMs": latency_ms,
        "message": f"Predicted ISL alphabet '{best_label}'.",
        "modelSourceNote": "Python Scikit-Learn Random Forest",
    })


@app.post("/api/collect")
def collect():
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        return jsonify({"error": "Label must be A-Z."}), 400

    try:
        canonical = to_canonical_features(payload.get("features", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)
    sample_id = f"py-{int(time.time() * 1000)}"
    created_at = datetime.now(timezone.utc).isoformat()
    row = [
        sample_id,
        label,
        str(payload.get("sessionId", "flask-session")),
        int(payload.get("handCount", 1)),
        "user_webcam",
        created_at,
        *[round(v, 6) for v in canonical],
    ]
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            header = ["id", "label", "session_id", "hand_count", "source", "created_at"] + [
                f"f_{i}" for i in range(DUAL_HAND_FEATURE_COUNT)
            ]
            writer.writerow(header)
        writer.writerow(row)

    return jsonify({"status": "saved", "id": sample_id}), 201


@app.post("/api/train")
def train_endpoint():
    global PIPELINE
    try:
        report = train_and_evaluate(CSV_PATH)
        PIPELINE = joblib.load(JOBLIB_PATH)
        return jsonify({"report": report})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/evaluate")
def evaluate_endpoint():
    if not os.path.exists(REPORT_PATH):
        return jsonify({"error": "No evaluation report found."}), 404
    with open(REPORT_PATH, "r", encoding="utf-8") as fh:
        return jsonify(json.load(fh))


@app.get("/api/dataset/export")
def export_csv():
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Dataset CSV not found."}), 404
    return send_file(CSV_PATH, as_attachment=True, download_name="isl_landmarks.csv")


if __name__ == "__main__":
    load_pipeline()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
