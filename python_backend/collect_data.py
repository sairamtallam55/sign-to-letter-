"""
Standalone OpenCV + MediaPipe Hands webcam data collection CLI for ISL (A-Z).
Run locally on a workstation with a webcam to append labeled samples directly to data/isl_landmarks.csv.
"""
from __future__ import annotations
import csv
import os
import time
from datetime import datetime, timezone

import cv2
import mediapipe as mp

from preprocess import DUAL_HAND_FEATURE_COUNT, distance_3d, normalize_single_hand

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "isl_landmarks.csv")


def main():
    target_label = input("Enter target ISL letter (A-Z): ").strip().upper()
    if len(target_label) != 1 or not ("A" <= target_label <= "Z"):
        raise SystemExit("Invalid letter. Must be A-Z.")
    session_id = input("Enter session ID (e.g. user1-daylight): ").strip() or "local-cam"

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)

    print("Press SPACE to capture a landmark sample, Q to quit.")
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            cv2.putText(
                frame,
                f"Target: {target_label} | SPACE=Capture | Q=Quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 120, 255),
                2,
            )
            cv2.imshow("ISL Landmark Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and result.multi_hand_landmarks:
                detected = [
                    [(lm.x, lm.y, lm.z) for lm in h.landmark]
                    for h in result.multi_hand_landmarks[:2]
                ]
                f1 = normalize_single_hand(detected[0])
                if len(detected) == 2:
                    f2 = normalize_single_hand(detected[1])
                    scale1 = max(
                        distance_3d(detected[0][0], detected[0][9]),
                        distance_3d(detected[0][5], detected[0][17]),
                        1e-5,
                    )
                    rel = [
                        round((detected[1][0][i] - detected[0][0][i]) / scale1, 6)
                        for i in range(3)
                    ]
                    feats = f1 + f2 + rel
                else:
                    feats = f1 + [0.0] * 66

                with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    if not file_exists:
                        header = ["id", "label", "session_id", "hand_count", "source", "created_at"] + [
                            f"f_{i}" for i in range(DUAL_HAND_FEATURE_COUNT)
                        ]
                        writer.writerow(header)
                        file_exists = True
                    writer.writerow([
                        f"cv-{int(time.time() * 1000)}",
                        target_label,
                        session_id,
                        len(detected),
                        "user_webcam",
                        datetime.now(timezone.utc).isoformat(),
                        *feats,
                    ])
                print(f"Saved sample for {target_label} ({len(detected)} hand(s)).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
