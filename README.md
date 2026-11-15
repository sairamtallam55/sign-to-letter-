# ISL Studio — Real-Time Indian Sign Language Hand Gesture Recognition

An interactive, full-stack web application that recognizes **Indian Sign Language (ISL)** alphabet gestures in real time using **MediaPipe 21-point Hand Landmarks**, **wrist/scale coordinate normalization**, **Random Forest classification**, and **dynamic stroke trajectory recognition (J & Z)**.

---

## 1. Key Distinction: ISL vs. ASL

Unlike American Sign Language (ASL), which uses a one-handed alphabet, **Indian Sign Language (ISLRTC standard) uses a predominantly two-handed (bimanual) manual alphabet**.
- MediaPipe HandLandmarker is configured with `numHands: 2` to track both the dominant (active) hand and non-dominant (base) hand simultaneously.
- Each hand produces 21 3D landmarks `(x, y, z)` = **63 features/hand**, combined into a **129-feature canonical dual-hand vector** (`63 Hand 1 + 63 Hand 2 + 3 inter-hand relative wrist displacement`).
- Letters **J** and **Z** require physical stroke movement and are handled by a dedicated **temporal trajectory recognizer** rather than single-frame static classification.

---

## 2. Project Folder Structure

```text
├── data/
│   └── isl_landmarks.csv              # Normalized 129-feature landmark dataset + provenance metadata
├── models/
│   ├── isl_rf_model.json              # Serialized Random Forest + StandardScaler + Centroids (Node ML)
│   └── evaluation_report.json         # Latest session-wise evaluation metrics & confusion matrix
├── public/mediapipe/wasm/             # Local MediaPipe Tasks Vision WASM runtime assets
├── python_backend/                    # Standalone Python Flask + Scikit-Learn deployment service
│   ├── app.py                         # Flask REST API (/api/predict, /api/collect, /api/train, /api/evaluate)
│   ├── preprocess.py                  # 21-landmark normalization & session-wise splitter
│   ├── train.py                       # Scikit-learn RandomForestClassifier + KNeighbors comparison + Joblib
│   ├── collect_data.py                # OpenCV + MediaPipe local webcam CLI collector
│   ├── test_preprocess.py             # Python unittest suite
│   ├── requirements.txt
│   ├── Dockerfile                     # Google Cloud Run / container deployment
│   └── render.yaml                    # Render Python Web Service deployment
├── src/
│   ├── components/
│   │   ├── Header.tsx                 # 3-zone top navigation bar
│   │   ├── HomeView.tsx               # Landing page & pipeline overview
│   │   ├── RecognitionView.tsx        # Live webcam + MediaPipe + smoother + simulated tester + history
│   │   ├── LearnView.tsx              # Interactive A–Z ISLRTC guide + Practice Mode
│   │   ├── DataModelView.tsx          # Custom dataset collector, CSV export, training & confusion matrix
│   │   ├── AboutView.tsx              # Research citations, SVG architecture & deployment guide
│   │   └── HandPoseDiagram.tsx        # Parametric SVG two-handed/one-handed posture diagrams
│   ├── ml/
│   │   ├── types.ts                   # Shared TypeScript interfaces
│   │   ├── islAlphabet.ts             # Authentic ISL A–Z reference metadata
│   │   ├── landmarks.ts               # Validation, wrist-scale normalization, smoother, dynamic J/Z
│   │   ├── randomForest.ts            # Random Forest + Nearest Centroid baseline + confusion matrix
│   │   ├── calibrationSeed.ts         # Deterministic calibration prototypes & simulation helpers
│   │   ├── landmarks.test.ts          # Unit tests (node:test)
│   │   └── randomForest.test.ts       # Unit tests (node:test)
│   ├── utils/
│   │   ├── handCanvas.ts              # 21-landmark skeleton & motion trail renderer
│   │   └── mediapipeClient.ts         # @mediapipe/tasks-vision HandLandmarker wrapper
│   └── App.tsx
├── server.ts                          # Express REST API + Vite middleware (runs live on port 3000)
├── server.test.ts                     # End-to-end REST API integration test
└── package.json
```

---

## 3. REST API Documentation

Both `server.ts` (Express) and `python_backend/app.py` (Flask) implement the same REST contract:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Runtime health check, model readiness, and sample counts |
| `POST` | `/api/predict` | Classify a 63- or 129-feature normalized hand landmark vector |
| `POST` | `/api/predict-dynamic` | Classify a temporal `{ x, y, t }[]` trajectory as `J` or `Z` |
| `GET` | `/api/dataset` | Dataset summary, per-class counts (`user_webcam` vs `synthetic_calibration`) |
| `GET` | `/api/dataset/export` | Download raw `isl_landmarks.csv` |
| `POST` | `/api/collect` | Validate & append a labeled landmark sample (with duplicate rejection) |
| `DELETE` | `/api/dataset/sample/:id` | Remove a specific collected sample |
| `POST` | `/api/train` | Run session-wise split, train Random Forest + baseline, return metrics |
| `GET` | `/api/evaluate` | Retrieve latest evaluation report and confusion matrix |

---

## 4. Running Locally & Deploying Python Backend

### Integrated Web + ML Server (Node.js / Express)
```bash
npm install
npm run dev       # Starts server.ts + Vite on http://localhost:3000
npm run build     # Builds production frontend bundle
npm start         # Serves production bundle + REST API
```

### Standalone Python Flask Backend (Google Cloud Run / Render)
1. Deploy `/python_backend/` using `python_backend/Dockerfile` to **Google Cloud Run**:
   ```bash
   gcloud run deploy isl-studio-flask-api --source . --dockerfile python_backend/Dockerfile --allow-unauthenticated
   ```
   Or connect the repo to **Render** using `python_backend/render.yaml`.
2. Open **Data & Model → REST API Runtime Endpoint** in the web UI and paste your HTTPS Flask URL.
