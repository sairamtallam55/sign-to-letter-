import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer } from 'vite';
import { buildCalibrationDataset } from './src/ml/calibrationSeed';
import {
  DUAL_HAND_FEATURE_COUNT,
  formatPredictionCandidates,
  recognizeDynamicTrajectory,
  toCanonicalFeatures,
  validateFeatureVector,
  type TrajectoryPoint,
} from './src/ml/landmarks';
import { ISL_ALPHABET, STATIC_ISL_LETTERS } from './src/ml/islAlphabet';
import {
  RandomForestModel,
  trainAndEvaluatePipeline,
  type SerializedRandomForestModel,
} from './src/ml/randomForest';
import type {
  DatasetSample,
  DatasetSummary,
  EvaluationReport,
  SampleSource,
} from './src/ml/types';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.join(__dirname, 'data');
const MODELS_DIR = path.join(__dirname, 'models');
const CSV_PATH = path.join(DATA_DIR, 'isl_landmarks.csv');
const MODEL_PATH = path.join(MODELS_DIR, 'isl_rf_model.json');
const REPORT_PATH = path.join(MODELS_DIR, 'evaluation_report.json');

const CONFIDENCE_THRESHOLD = 0.42;

function getCsvHeader(): string {
  const cols = ['id', 'label', 'session_id', 'hand_count', 'source', 'created_at'];
  for (let i = 0; i < DUAL_HAND_FEATURE_COUNT; i++) {
    cols.push(`f_${i}`);
  }
  return cols.join(',');
}

function sampleToCsvRow(s: DatasetSample): string {
  const safeId = s.id.replace(/,/g, '_');
  const safeSession = s.sessionId.replace(/,/g, '_');
  return [
    safeId,
    s.label,
    safeSession,
    String(s.handCount),
    s.source,
    s.createdAt,
    ...s.features.map((v) => v.toFixed(6)),
  ].join(',');
}

function saveDatasetToCsv(samples: DatasetSample[]): void {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const lines = [getCsvHeader(), ...samples.map(sampleToCsvRow)];
  fs.writeFileSync(CSV_PATH, lines.join('\n') + '\n', 'utf-8');
}

function loadDatasetFromCsv(): DatasetSample[] {
  if (!fs.existsSync(CSV_PATH)) {
    const seed = buildCalibrationDataset(25);
    saveDatasetToCsv(seed);
    return seed;
  }

  const raw = fs.readFileSync(CSV_PATH, 'utf-8').trim();
  if (!raw) return [];
  const lines = raw.split('\n');
  if (lines.length <= 1) return [];

  const samples: DatasetSample[] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length !== 6 + DUAL_HAND_FEATURE_COUNT) continue;
    const [id, label, sessionId, handCountRaw, sourceRaw, createdAt, ...featStrs] = parts;
    const features = featStrs.map(Number);
    if (
      !label ||
      !validateFeatureVector(features) ||
      (handCountRaw !== '1' && handCountRaw !== '2')
    ) {
      continue;
    }
    const source: SampleSource =
      sourceRaw === 'user_webcam' ? 'user_webcam' : 'synthetic_calibration';
    samples.push({
      id,
      label,
      sessionId,
      handCount: handCountRaw === '2' ? 2 : 1,
      source,
      createdAt,
      features,
    });
  }
  return samples;
}

class MLRuntimeStore {
  dataset: DatasetSample[] = [];
  model: RandomForestModel | null = null;
  report: EvaluationReport | null = null;

  init(): void {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.mkdirSync(MODELS_DIR, { recursive: true });
    this.dataset = loadDatasetFromCsv();

    if (fs.existsSync(MODEL_PATH) && fs.existsSync(REPORT_PATH)) {
      try {
        const modelJson = JSON.parse(
          fs.readFileSync(MODEL_PATH, 'utf-8')
        ) as SerializedRandomForestModel;
        const reportJson = JSON.parse(
          fs.readFileSync(REPORT_PATH, 'utf-8')
        ) as EvaluationReport;
        if (modelJson.version === 2) {
          this.model = new RandomForestModel(modelJson);
          this.report = reportJson;
          return;
        }
      } catch {
        // Recompute cleanly if artifact file was corrupted
      }
    }

    const hasUserSamples = this.dataset.some((s) => s.source === 'user_webcam');
    if (!hasUserSamples && this.dataset.length < 600) {
      this.dataset = buildCalibrationDataset(25);
      saveDatasetToCsv(this.dataset);
    }

    this.retrain();
  }

  retrain(): EvaluationReport {
    const { productionModel, report } = trainAndEvaluatePipeline(this.dataset);
    this.model = productionModel;
    this.report = report;
    fs.writeFileSync(MODEL_PATH, JSON.stringify(productionModel.model), 'utf-8');
    fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2), 'utf-8');
    return report;
  }

  getSummary(): DatasetSummary {
    const perClassCounts: DatasetSummary['perClassCounts'] = {};
    for (const item of ISL_ALPHABET) {
      perClassCounts[item.letter] = { total: 0, userWebcam: 0, syntheticCalibration: 0 };
    }

    const sessions = new Set<string>();
    let userWebcamSamples = 0;
    let syntheticCalibrationSamples = 0;

    for (const s of this.dataset) {
      sessions.add(s.sessionId);
      if (!perClassCounts[s.label]) {
        perClassCounts[s.label] = { total: 0, userWebcam: 0, syntheticCalibration: 0 };
      }
      perClassCounts[s.label].total += 1;
      if (s.source === 'user_webcam') {
        userWebcamSamples += 1;
        perClassCounts[s.label].userWebcam += 1;
      } else {
        syntheticCalibrationSamples += 1;
        perClassCounts[s.label].syntheticCalibration += 1;
      }
    }

    const classesCovered = Object.values(perClassCounts).filter((c) => c.total > 0).length;
    const recentSamples = [...this.dataset]
      .reverse()
      .slice(0, 20)
      .map(({ id, label, sessionId, handCount, source, createdAt }) => ({
        id,
        label,
        sessionId,
        handCount,
        source,
        createdAt,
      }));

    return {
      totalSamples: this.dataset.length,
      userWebcamSamples,
      syntheticCalibrationSamples,
      uniqueSessions: sessions.size,
      classesCovered,
      perClassCounts,
      recentSamples,
    };
  }
}

export function createApp() {
  const store = new MLRuntimeStore();
  store.init();

  const app = express();
  app.use(express.json({ limit: '2mb' }));

  app.get('/api/health', (_req, res) => {
    const summary = store.getSummary();
    res.json({
      status: 'ok',
      engine: 'Express TypeScript ML Engine (Flask API Contract Compatible)',
      modelLoaded: Boolean(store.model),
      classesSupportedStatic: store.model?.model.classes.length ?? 0,
      classesSupportedDynamic: ['J', 'Z'],
      totalSamples: summary.totalSamples,
      userWebcamSamples: summary.userWebcamSamples,
      syntheticCalibrationSamples: summary.syntheticCalibrationSamples,
      pythonMirrorDirectory: '/python_backend',
    });
  });

  app.post('/api/predict', (req, res) => {
    const started = performance.now();
    const { features, handCount } = req.body ?? {};

    if (!store.model) {
      res.status(503).json({
        prediction: null,
        confidence: 0,
        status: 'error',
        handCount: 0,
        isDynamic: false,
        topCandidates: [],
        latencyMs: 0,
        message: 'Model is not trained or loaded yet.',
        modelSourceNote: 'Unavailable',
      });
      return;
    }

    if (!validateFeatureVector(features)) {
      res.status(400).json({
        prediction: null,
        confidence: 0,
        status: 'error',
        handCount: 0,
        isDynamic: false,
        topCandidates: [],
        latencyMs: Number((performance.now() - started).toFixed(2)),
        message:
          'Invalid feature vector. Expected 63 (single-hand) or 129 (dual-hand canonical) finite numbers.',
        modelSourceNote: '',
      });
      return;
    }

    const canonical = toCanonicalFeatures(features);
    const { label, confidence, probabilities } = store.model.predict(canonical);
    const topCandidates = formatPredictionCandidates(probabilities, 3);
    const latencyMs = Number((performance.now() - started).toFixed(2));
    const userSamples = store.dataset.filter((s) => s.source === 'user_webcam').length;
    const modelSourceNote =
      userSamples > 0
        ? `Hybrid model (${userSamples} webcam + calibration samples)`
        : 'Calibration prototype model (record your hands in Data Studio for personalized accuracy)';

    if (confidence < CONFIDENCE_THRESHOLD) {
      res.json({
        prediction: null,
        confidence: Number(confidence.toFixed(4)),
        status: 'uncertain',
        handCount: handCount === 2 ? 2 : 1,
        isDynamic: false,
        topCandidates,
        latencyMs,
        message: `Low confidence (${Math.round(confidence * 100)}%). Gesture appears uncertain or unsupported.`,
        modelSourceNote,
      });
      return;
    }

    res.json({
      prediction: label,
      confidence: Number(confidence.toFixed(4)),
      status: 'detected',
      handCount: handCount === 2 ? 2 : 1,
      isDynamic: false,
      topCandidates,
      latencyMs,
      message: `Predicted ISL static alphabet '${label}'.`,
      modelSourceNote,
    });
  });

  app.post('/api/predict-dynamic', (req, res) => {
    const started = performance.now();
    const { frames } = req.body ?? {};
    if (!Array.isArray(frames)) {
      res.status(400).json({
        prediction: null,
        confidence: 0,
        status: 'insufficient_motion',
        pathLength: 0,
        latencyMs: Number((performance.now() - started).toFixed(2)),
        message: 'Expected frames array of { x, y, t } trajectory points.',
      });
      return;
    }

    const validFrames: TrajectoryPoint[] = frames.filter(
      (f) =>
        f &&
        typeof f.x === 'number' &&
        typeof f.y === 'number' &&
        typeof f.t === 'number' &&
        Number.isFinite(f.x) &&
        Number.isFinite(f.y) &&
        Number.isFinite(f.t)
    );

    const result = recognizeDynamicTrajectory(validFrames);
    res.json({
      ...result,
      latencyMs: Number((performance.now() - started).toFixed(2)),
    });
  });

  app.get('/api/dataset', (_req, res) => {
    res.json(store.getSummary());
  });

  app.get('/api/dataset/export', (_req, res) => {
    if (!fs.existsSync(CSV_PATH)) {
      saveDatasetToCsv(store.dataset);
    }
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="isl_landmarks.csv"');
    res.send(fs.readFileSync(CSV_PATH, 'utf-8'));
  });

  app.post('/api/collect', (req, res) => {
    const { label, sessionId, handCount, features, source } = req.body ?? {};
    const upperLabel = typeof label === 'string' ? label.trim().toUpperCase() : '';

    if (!/^[A-Z]$/.test(upperLabel)) {
      res.status(400).json({ error: 'Label must be a single English alphabet letter (A-Z).' });
      return;
    }

    if (!validateFeatureVector(features)) {
      res.status(400).json({
        error: 'Features must be an array of 63 or 129 finite numbers.',
      });
      return;
    }

    const canonical = toCanonicalFeatures(features);
    // Duplicate prevention: compare rounded signature against last 10 samples of same label
    const sig = canonical.slice(0, 12).map((n) => n.toFixed(3)).join('|');
    const recentSame = store.dataset
      .filter((s) => s.label === upperLabel)
      .slice(-10)
      .some((s) => s.features.slice(0, 12).map((n) => n.toFixed(3)).join('|') === sig);

    if (recentSame) {
      res.status(409).json({
        error: 'Duplicate hand posture detected. Slightly vary angle or hand position before capturing again.',
      });
      return;
    }

    const newSample: DatasetSample = {
      id: `smp-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      label: upperLabel,
      sessionId:
        typeof sessionId === 'string' && sessionId.trim().length > 0
          ? sessionId.trim().slice(0, 40)
          : `webcam-${new Date().toISOString().slice(0, 10)}`,
      handCount: handCount === 2 ? 2 : 1,
      source: source === 'user_webcam' ? 'user_webcam' : 'synthetic_calibration',
      features: canonical,
      createdAt: new Date().toISOString(),
    };

    store.dataset.push(newSample);
    saveDatasetToCsv(store.dataset);

    res.status(201).json({
      sample: {
        id: newSample.id,
        label: newSample.label,
        sessionId: newSample.sessionId,
        handCount: newSample.handCount,
        source: newSample.source,
        createdAt: newSample.createdAt,
      },
      summary: store.getSummary(),
    });
  });

  app.delete('/api/dataset/sample/:id', (req, res) => {
    const targetId = req.params.id;
    const before = store.dataset.length;
    store.dataset = store.dataset.filter((s) => s.id !== targetId);
    if (store.dataset.length === before) {
      res.status(404).json({ error: 'Sample not found.' });
      return;
    }
    saveDatasetToCsv(store.dataset);
    res.json({ summary: store.getSummary() });
  });

  app.post('/api/dataset/reset', (_req, res) => {
    store.dataset = buildCalibrationDataset(25);
    saveDatasetToCsv(store.dataset);
    const report = store.retrain();
    res.json({
      summary: store.getSummary(),
      report,
    });
  });

  app.post('/api/train', (_req, res) => {
    try {
      // Validate static classes present
      const validSamples = store.dataset.filter((s) => STATIC_ISL_LETTERS.includes(s.label) || /^[A-Z]$/.test(s.label));
      if (validSamples.length < 10) {
        res.status(400).json({ error: 'Need at least 10 valid samples to train the classifier.' });
        return;
      }
      const report = store.retrain();
      res.json({ report, summary: store.getSummary() });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Training failed.',
      });
    }
  });

  app.get('/api/evaluate', (_req, res) => {
    if (!store.report) {
      res.status(404).json({ error: 'No evaluation report available.' });
      return;
    }
    res.json(store.report);
  });

  return app;
}

async function startServer() {
  const app = createApp();
  const port = Number(process.env.PORT || 3000);

  if (process.env.NODE_ENV === 'production') {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`ISL Studio server listening on http://localhost:${port}`);
  });
}

// Allow importing createApp in tests without binding port 3000
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename)) {
  startServer();
}
