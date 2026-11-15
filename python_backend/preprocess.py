"""
Preprocessing and normalization utilities for ISL 21-point hand landmarks.
Matches the frontend TypeScript normalization contract exactly (63 single-hand / 129 dual-hand features).
"""
from __future__ import annotations
import csv
import math
from typing import Dict, List, Sequence, Tuple

SINGLE_HAND_FEATURE_COUNT = 63
DUAL_HAND_FEATURE_COUNT = 129


def distance_3d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def validate_landmarks(landmarks: Sequence[Sequence[float]]) -> bool:
    if len(landmarks) != 21:
        return False
    xs, ys = [], []
    for pt in landmarks:
        if len(pt) != 3 or any(not math.isfinite(float(v)) for v in pt):
            return False
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return span >= 1e-4


def normalize_single_hand(landmarks: Sequence[Sequence[float]]) -> List[float]:
    if not validate_landmarks(landmarks):
        raise ValueError("Invalid 21-point hand landmarks.")
    wrist = landmarks[0]
    scale = max(
        distance_3d(wrist, landmarks[9]),
        distance_3d(landmarks[5], landmarks[17]),
        1e-5,
    )
    features: List[float] = []
    for lm in landmarks:
        features.extend([
            round((float(lm[0]) - float(wrist[0])) / scale, 6),
            round((float(lm[1]) - float(wrist[1])) / scale, 6),
            round((float(lm[2]) - float(wrist[2])) / scale, 6),
        ])
    return features


def to_canonical_features(features: Sequence[float]) -> List[float]:
    vals = [float(v) for v in features]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("Features contain NaN or infinite values.")
    if len(vals) == DUAL_HAND_FEATURE_COUNT:
        return vals
    if len(vals) == SINGLE_HAND_FEATURE_COUNT:
        return vals + [0.0] * (DUAL_HAND_FEATURE_COUNT - SINGLE_HAND_FEATURE_COUNT)
    raise ValueError(f"Expected 63 or 129 features, got {len(vals)}.")


def load_landmark_csv(csv_path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = (row.get("label") or "").strip().upper()
            if len(label) != 1 or not ("A" <= label <= "Z"):
                continue
            try:
                feats = [float(row[f"f_{i}"]) for i in range(DUAL_HAND_FEATURE_COUNT)]
            except (KeyError, ValueError):
                continue
            if any(not math.isfinite(v) for v in feats):
                continue
            records.append({
                "id": row.get("id", ""),
                "label": label,
                "session_id": row.get("session_id", "default"),
                "hand_count": int(row.get("hand_count", 1)),
                "source": row.get("source", "user_webcam"),
                "created_at": row.get("created_at", ""),
                "features": feats,
            })
    return records


def session_wise_split(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], str]:
    sessions = sorted({r["session_id"] for r in records})
    if len(sessions) >= 3:
        train_s, val_s, test_s = set(), set(), set()
        for idx, s in enumerate(sessions):
            if idx % 5 == 3:
                val_s.add(s)
            elif idx % 5 == 4:
                test_s.add(s)
            else:
                train_s.add(s)
        if not val_s:
            val_s.add(sessions[1])
        if not test_s:
            test_s.add(sessions[2])
        train = [r for r in records if r["session_id"] in train_s]
        val = [r for r in records if r["session_id"] in val_s]
        test = [r for r in records if r["session_id"] in test_s]
        if len({r["label"] for r in train}) == len({r["label"] for r in records}) and val and test:
            return train, val, test, "session-wise"

    train, val, test = [], [], []
    by_label: Dict[str, List[Dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    for group in by_label.values():
        for idx, r in enumerate(group):
            if len(group) < 3:
                train.append(r)
                if idx == len(group) - 1:
                    test.append(r)
            elif idx % 6 == 4:
                val.append(r)
            elif idx % 6 == 5:
                test.append(r)
            else:
                train.append(r)
    return train, val, test, "stratified-fallback"
