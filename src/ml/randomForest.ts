import type {
  ClassMetrics,
  DatasetSample,
  EvaluationReport,
  ModelComparisonMetrics,
} from './types';

interface TreeNode {
  leaf: boolean;
  classCounts?: number[];
  leafTotal?: number;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface SerializedScaler {
  means: number[];
  stds: number[];
}

export interface SerializedRandomForestModel {
  version: number;
  trainedAt: string;
  classes: string[];
  nFeatures: number;
  nTrees: number;
  maxDepth: number;
  scaler: SerializedScaler;
  trees: TreeNode[];
  centroids: number[][];
}

function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function fitScaler(X: number[][]): SerializedScaler {
  const nSamples = X.length;
  const nFeatures = X[0]?.length ?? 0;
  const means = new Array<number>(nFeatures).fill(0);
  const stds = new Array<number>(nFeatures).fill(1);

  if (nSamples === 0) return { means, stds };

  for (const row of X) {
    for (let j = 0; j < nFeatures; j++) {
      means[j] += row[j];
    }
  }
  for (let j = 0; j < nFeatures; j++) {
    means[j] /= nSamples;
  }

  const variances = new Array<number>(nFeatures).fill(0);
  for (const row of X) {
    for (let j = 0; j < nFeatures; j++) {
      const d = row[j] - means[j];
      variances[j] += d * d;
    }
  }
  for (let j = 0; j < nFeatures; j++) {
    const std = Math.sqrt(variances[j] / nSamples);
    stds[j] = std > 1e-6 ? std : 1;
  }
  return { means, stds };
}

export function transformWithScaler(X: number[][], scaler: SerializedScaler): number[][] {
  return X.map((row) =>
    row.map((val, j) => (val - scaler.means[j]) / (scaler.stds[j] || 1))
  );
}

function scaleVectorInto(raw: number[], scaler: SerializedScaler, out: Float64Array): void {
  const means = scaler.means;
  const stds = scaler.stds;
  for (let j = 0; j < out.length; j++) {
    out[j] = (raw[j] - means[j]) / (stds[j] || 1);
  }
}

function sumOfSquares(counts: Int32Array | number[]): number {
  let sumSq = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i];
    sumSq += c * c;
  }
  return sumSq;
}

function buildTree(
  X: number[][],
  y: number[],
  indices: number[],
  nClasses: number,
  nFeatures: number,
  maxFeatures: number,
  maxDepth: number,
  minSamplesSplit: number,
  depth: number,
  rng: () => number,
  featurePool: number[]
): TreeNode {
  const counts = new Array<number>(nClasses).fill(0);
  for (let i = 0; i < indices.length; i++) {
    counts[y[indices[i]]] += 1;
  }

  const total = indices.length;
  let nonZeroClasses = 0;
  for (let c = 0; c < nClasses; c++) {
    if (counts[c] > 0) nonZeroClasses++;
  }

  if (depth >= maxDepth || total < minSamplesSplit || nonZeroClasses <= 1) {
    return { leaf: true, classCounts: counts, leafTotal: total || 1 };
  }

  // Partial Fisher-Yates shuffle to sample `maxFeatures` random candidate features in O(maxFeatures)
  for (let i = 0; i < maxFeatures; i++) {
    const j = i + Math.floor(rng() * (nFeatures - i));
    const tmp = featurePool[i];
    featurePool[i] = featurePool[j];
    featurePool[j] = tmp;
  }

  const parentScore = sumOfSquares(counts) / total;
  let bestScore = parentScore;
  let bestFeature = -1;
  let bestThreshold = 0;

  const pairs = new Array<{ idx: number; val: number }>(total);
  const leftCounts = new Int32Array(nClasses);
  const rightCounts = new Int32Array(nClasses);

  for (let fi = 0; fi < maxFeatures; fi++) {
    const f = featurePool[fi];
    for (let i = 0; i < total; i++) {
      const idx = indices[i];
      pairs[i] = { idx, val: X[idx][f] };
    }
    pairs.sort((a, b) => a.val - b.val);

    if (pairs[total - 1].val - pairs[0].val < 1e-8) continue;

    // Linear single-pass O(N) prefix-sum Gini sweep across sorted boundaries
    leftCounts.fill(0);
    for (let c = 0; c < nClasses; c++) rightCounts[c] = counts[c];

    let leftSumSq = 0;
    let rightSumSq = sumOfSquares(counts);

    for (let i = 0; i < total - 1; i++) {
      const cls = y[pairs[i].idx];
      const lc = leftCounts[cls];
      const rc = rightCounts[cls];

      leftSumSq += 2 * lc + 1;
      rightSumSq += -2 * rc + 1;
      leftCounts[cls] = lc + 1;
      rightCounts[cls] = rc - 1;

      const v0 = pairs[i].val;
      const v1 = pairs[i + 1].val;
      if (v1 - v0 <= 1e-8) continue;

      const leftTotal = i + 1;
      const rightTotal = total - leftTotal;
      const score = leftSumSq / leftTotal + rightSumSq / rightTotal;

      if (score > bestScore + 1e-7) {
        bestScore = score;
        bestFeature = f;
        bestThreshold = 0.5 * (v0 + v1);
      }
    }
  }

  if (bestFeature === -1 || (bestScore - parentScore) / total <= 1e-7) {
    return { leaf: true, classCounts: counts, leafTotal: total || 1 };
  }

  const bestLeftIndices: number[] = [];
  const bestRightIndices: number[] = [];
  for (let i = 0; i < total; i++) {
    const idx = indices[i];
    if (X[idx][bestFeature] <= bestThreshold) {
      bestLeftIndices.push(idx);
    } else {
      bestRightIndices.push(idx);
    }
  }

  if (bestLeftIndices.length === 0 || bestRightIndices.length === 0) {
    return { leaf: true, classCounts: counts, leafTotal: total || 1 };
  }

  return {
    leaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildTree(
      X,
      y,
      bestLeftIndices,
      nClasses,
      nFeatures,
      maxFeatures,
      maxDepth,
      minSamplesSplit,
      depth + 1,
      rng,
      featurePool
    ),
    right: buildTree(
      X,
      y,
      bestRightIndices,
      nClasses,
      nFeatures,
      maxFeatures,
      maxDepth,
      minSamplesSplit,
      depth + 1,
      rng,
      featurePool
    ),
  };
}

export class RandomForestModel {
  readonly model: SerializedRandomForestModel;
  private readonly scaledBuf: Float64Array;
  private readonly probsBuf: Float64Array;

  constructor(model: SerializedRandomForestModel) {
    this.model = model;
    this.scaledBuf = new Float64Array(model.nFeatures);
    this.probsBuf = new Float64Array(model.classes.length);
  }

  static train(
    samples: DatasetSample[],
    options: { nTrees?: number; maxDepth?: number; minSamplesSplit?: number; seed?: number } = {}
  ): RandomForestModel {
    if (samples.length < 2) {
      throw new Error('At least 2 samples are required to train the Random Forest model.');
    }

    const nTrees = options.nTrees ?? 40;
    const maxDepth = options.maxDepth ?? 14;
    const minSamplesSplit = options.minSamplesSplit ?? 2;
    const rng = createRng(options.seed ?? 42);

    const classes = Array.from(new Set(samples.map((s) => s.label))).sort();
    const classToIdx = new Map(classes.map((c, i) => [c, i]));
    const nClasses = classes.length;

    const rawX = samples.map((s) => s.features);
    const y = samples.map((s) => classToIdx.get(s.label)!);
    const nFeatures = rawX[0].length;
    const scaler = fitScaler(rawX);
    const X = transformWithScaler(rawX, scaler);

    const maxFeatures = Math.max(4, Math.floor(Math.sqrt(nFeatures)) + 2);
    const featurePool = Array.from({ length: nFeatures }, (_, i) => i);
    const trees: TreeNode[] = [];

    for (let t = 0; t < nTrees; t++) {
      const bootIndices = new Array<number>(X.length);
      for (let i = 0; i < X.length; i++) {
        bootIndices[i] = Math.floor(rng() * X.length);
      }
      trees.push(
        buildTree(
          X,
          y,
          bootIndices,
          nClasses,
          nFeatures,
          Math.min(maxFeatures, nFeatures),
          maxDepth,
          minSamplesSplit,
          0,
          rng,
          featurePool
        )
      );
    }

    // Compute scaled class centroids for lightweight baseline comparison
    const centroids = classes.map((_, cIdx) => {
      const sum = new Array<number>(nFeatures).fill(0);
      let count = 0;
      for (let i = 0; i < X.length; i++) {
        if (y[i] === cIdx) {
          for (let j = 0; j < nFeatures; j++) sum[j] += X[i][j];
          count += 1;
        }
      }
      return sum.map((v) => v / Math.max(1, count));
    });

    return new RandomForestModel({
      version: 2,
      trainedAt: new Date().toISOString(),
      classes,
      nFeatures,
      nTrees,
      maxDepth,
      scaler,
      trees,
      centroids,
    });
  }

  predictProba(rawFeatures: number[]): Record<string, number> {
    scaleVectorInto(rawFeatures, this.model.scaler, this.scaledBuf);
    const nClasses = this.model.classes.length;
    const avgProbs = this.probsBuf;
    avgProbs.fill(0);

    const trees = this.model.trees;
    const nTrees = trees.length;
    for (let t = 0; t < nTrees; t++) {
      let node = trees[t];
      while (!node.leaf && node.featureIndex !== undefined && node.threshold !== undefined) {
        node = this.scaledBuf[node.featureIndex] <= node.threshold ? node.left! : node.right!;
      }
      const counts = node.classCounts;
      if (counts) {
        let total = node.leafTotal;
        if (!total) {
          total = 0;
          for (let c = 0; c < nClasses; c++) total += counts[c];
          if (total === 0) total = 1;
        }
        const inv = 1 / total;
        for (let c = 0; c < nClasses; c++) {
          avgProbs[c] += counts[c] * inv;
        }
      }
    }

    const invTrees = 1 / Math.max(1, nTrees);
    const result: Record<string, number> = {};
    for (let c = 0; c < nClasses; c++) {
      result[this.model.classes[c]] = avgProbs[c] * invTrees;
    }
    return result;
  }

  predict(rawFeatures: number[]): { label: string; confidence: number; probabilities: Record<string, number> } {
    const probabilities = this.predictProba(rawFeatures);
    let bestLabel = this.model.classes[0];
    let bestConf = -1;
    for (const [label, prob] of Object.entries(probabilities)) {
      if (prob > bestConf) {
        bestConf = prob;
        bestLabel = label;
      }
    }
    return { label: bestLabel, confidence: bestConf, probabilities };
  }

  predictCentroidBaseline(rawFeatures: number[]): { label: string; confidence: number } {
    scaleVectorInto(rawFeatures, this.model.scaler, this.scaledBuf);
    let bestIdx = 0;
    let bestDist = Infinity;
    const nClasses = this.model.centroids.length;
    const distances = new Float64Array(nClasses);

    for (let c = 0; c < nClasses; c++) {
      const centroid = this.model.centroids[c];
      let d2 = 0;
      for (let j = 0; j < this.scaledBuf.length; j++) {
        const diff = this.scaledBuf[j] - centroid[j];
        d2 += diff * diff;
      }
      const d = Math.sqrt(d2);
      distances[c] = d;
      if (d < bestDist) {
        bestDist = d;
        bestIdx = c;
      }
    }

    let sumExp = 0;
    let bestExp = 0;
    for (let c = 0; c < nClasses; c++) {
      const e = Math.exp(-distances[c] * 0.25);
      sumExp += e;
      if (c === bestIdx) bestExp = e;
    }

    return {
      label: this.model.classes[bestIdx],
      confidence: bestExp / (sumExp || 1),
    };
  }
}

/**
 * Splits samples into train / validation / test sets.
 * Prefers session-wise splitting so samples from the same recording session never leak across splits.
 */
export function splitDataset(samples: DatasetSample[]): {
  train: DatasetSample[];
  validation: DatasetSample[];
  test: DatasetSample[];
  splitStrategy: 'session-wise' | 'stratified-fallback';
} {
  const uniqueSessions = Array.from(new Set(samples.map((s) => s.sessionId))).sort();

  if (uniqueSessions.length >= 3) {
    const trainSessions = new Set<string>();
    const valSessions = new Set<string>();
    const testSessions = new Set<string>();

    uniqueSessions.forEach((sess, idx) => {
      if (idx % 5 === 3) valSessions.add(sess);
      else if (idx % 5 === 4) testSessions.add(sess);
      else trainSessions.add(sess);
    });

    if (valSessions.size === 0) valSessions.add(uniqueSessions[1]);
    if (testSessions.size === 0) testSessions.add(uniqueSessions[2]);

    const train = samples.filter((s) => trainSessions.has(s.sessionId));
    const validation = samples.filter((s) => valSessions.has(s.sessionId));
    const test = samples.filter((s) => testSessions.has(s.sessionId));

    const trainClasses = new Set(train.map((s) => s.label));
    const allClasses = new Set(samples.map((s) => s.label));
    if (trainClasses.size === allClasses.size && test.length > 0 && validation.length > 0) {
      return { train, validation, test, splitStrategy: 'session-wise' };
    }
  }

  const byClass = new Map<string, DatasetSample[]>();
  for (const s of samples) {
    const list = byClass.get(s.label) ?? [];
    list.push(s);
    byClass.set(s.label, list);
  }

  const train: DatasetSample[] = [];
  const validation: DatasetSample[] = [];
  const test: DatasetSample[] = [];

  for (const list of byClass.values()) {
    list.forEach((s, idx) => {
      if (list.length < 3) {
        train.push(s);
        if (idx === list.length - 1) test.push(s);
      } else if (idx % 6 === 4) {
        validation.push(s);
      } else if (idx % 6 === 5) {
        test.push(s);
      } else {
        train.push(s);
      }
    });
  }

  return { train, validation, test, splitStrategy: 'stratified-fallback' };
}

function computeClassificationMetrics(
  yTrue: string[],
  yPred: string[],
  classes: string[]
): {
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  perClassMetrics: ClassMetrics[];
  confusionMatrix: number[][];
} {
  const n = classes.length;
  const idxMap = new Map(classes.map((c, i) => [c, i]));
  const confusionMatrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));

  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const ti = idxMap.get(yTrue[i]);
    const pi = idxMap.get(yPred[i]);
    if (ti !== undefined && pi !== undefined) {
      confusionMatrix[ti][pi] += 1;
      if (ti === pi) correct += 1;
    }
  }

  const perClassMetrics: ClassMetrics[] = classes.map((label, c) => {
    const tp = confusionMatrix[c][c];
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (let r = 0; r < n; r++) {
      support += confusionMatrix[c][r];
      if (r !== c) {
        fp += confusionMatrix[r][c];
        fn += confusionMatrix[c][r];
      }
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return {
      label,
      support,
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1Score: Number(f1Score.toFixed(4)),
    };
  });

  const activeClasses = perClassMetrics.filter((m) => m.support > 0);
  const denom = Math.max(1, activeClasses.length);
  const macroPrecision = activeClasses.reduce((s, m) => s + m.precision, 0) / denom;
  const macroRecall = activeClasses.reduce((s, m) => s + m.recall, 0) / denom;
  const macroF1 = activeClasses.reduce((s, m) => s + m.f1Score, 0) / denom;
  const accuracy = yTrue.length > 0 ? correct / yTrue.length : 0;

  return {
    accuracy: Number(accuracy.toFixed(4)),
    macroPrecision: Number(macroPrecision.toFixed(4)),
    macroRecall: Number(macroRecall.toFixed(4)),
    macroF1: Number(macroF1.toFixed(4)),
    perClassMetrics,
    confusionMatrix,
  };
}

export function trainAndEvaluatePipeline(samples: DatasetSample[]): {
  productionModel: RandomForestModel;
  report: EvaluationReport;
} {
  const { train, validation, test, splitStrategy } = splitDataset(samples);

  const evalModel = RandomForestModel.train(train, {
    nTrees: 40,
    maxDepth: 14,
    minSamplesSplit: 2,
    seed: 101,
  });

  const classes = evalModel.model.classes;
  const yTest = test.map((s) => s.label);

  const rfStart = performance.now();
  const yPredRf = test.map((s) => evalModel.predict(s.features).label);
  const rfElapsed = performance.now() - rfStart;

  const centroidStart = performance.now();
  const yPredCentroid = test.map((s) => evalModel.predictCentroidBaseline(s.features).label);
  const centroidElapsed = performance.now() - centroidStart;

  const rfMetrics = computeClassificationMetrics(yTest, yPredRf, classes);
  const centroidMetrics = computeClassificationMetrics(yTest, yPredCentroid, classes);

  // Train final production model on all verified samples after held-out evaluation
  const productionModel = RandomForestModel.train(samples, {
    nTrees: 40,
    maxDepth: 14,
    minSamplesSplit: 2,
    seed: 202,
  });

  const userWebcamSamples = samples.filter((s) => s.source === 'user_webcam').length;
  const syntheticCalibrationSamples = samples.filter(
    (s) => s.source === 'synthetic_calibration'
  ).length;

  const comparison: ModelComparisonMetrics[] = [
    {
      modelName: 'Random Forest Classifier (40 trees, Optimized)',
      accuracy: rfMetrics.accuracy,
      macroPrecision: rfMetrics.macroPrecision,
      macroRecall: rfMetrics.macroRecall,
      macroF1: rfMetrics.macroF1,
      inferenceLatencyMs: Number((rfElapsed / Math.max(1, test.length)).toFixed(3)),
    },
    {
      modelName: 'Nearest Centroid Classifier (Baseline)',
      accuracy: centroidMetrics.accuracy,
      macroPrecision: centroidMetrics.macroPrecision,
      macroRecall: centroidMetrics.macroRecall,
      macroF1: centroidMetrics.macroF1,
      inferenceLatencyMs: Number((centroidElapsed / Math.max(1, test.length)).toFixed(3)),
    },
  ];

  const honestyNote =
    userWebcamSamples === 0
      ? 'Evaluated exclusively on synthetic landmark calibration prototypes (0 real webcam samples recorded yet). Collect live webcam samples in Data & Model Studio to measure real-world human signing accuracy.'
      : `Evaluated on ${samples.length} total samples (${userWebcamSamples} real webcam + ${syntheticCalibrationSamples} calibration prototypes) using ${splitStrategy} split.`;

  return {
    productionModel,
    report: {
      trainedAt: new Date().toISOString(),
      totalSamples: samples.length,
      trainSamples: train.length,
      validationSamples: validation.length,
      testSamples: test.length,
      classesEvaluated: classes,
      userWebcamSamples,
      syntheticCalibrationSamples,
      splitStrategy,
      accuracy: rfMetrics.accuracy,
      macroPrecision: rfMetrics.macroPrecision,
      macroRecall: rfMetrics.macroRecall,
      macroF1: rfMetrics.macroF1,
      perClassMetrics: rfMetrics.perClassMetrics,
      confusionMatrix: rfMetrics.confusionMatrix,
      comparison,
      honestyNote,
    },
  };
}
