import type { DetectedHandInput, Landmark3D, PredictionCandidate } from './types';

export const SINGLE_HAND_FEATURE_COUNT = 63; // 21 landmarks * (x, y, z)
export const DUAL_HAND_FEATURE_COUNT = 129; // 63 (hand1) + 63 (hand2) + 3 (inter-hand relative wrist delta)

export function distance3D(a: Landmark3D, b: Landmark3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function validateHandLandmarks(landmarks: Landmark3D[]): {
  valid: boolean;
  reason?: string;
} {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) {
    return { valid: false, reason: 'Expected exactly 21 hand landmarks.' };
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (
      !lm ||
      typeof lm.x !== 'number' ||
      typeof lm.y !== 'number' ||
      typeof lm.z !== 'number' ||
      !Number.isFinite(lm.x) ||
      !Number.isFinite(lm.y) ||
      !Number.isFinite(lm.z)
    ) {
      return { valid: false, reason: `Invalid non-finite coordinates at landmark ${i}.` };
    }
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }

  const span = Math.max(maxX - minX, maxY - minY);
  if (span < 1e-4) {
    return { valid: false, reason: 'Degenerate hand landmarks with zero spatial span.' };
  }

  return { valid: true };
}

/**
 * Translates landmark 0 (wrist) to origin and normalizes by hand reference scale
 * (max of wrist-to-middle-MCP and index-MCP-to-pinky-MCP) -> 63 translation- & scale-invariant features.
 */
export function normalizeSingleHand(landmarks: Landmark3D[]): number[] {
  const check = validateHandLandmarks(landmarks);
  if (!check.valid) {
    throw new Error(check.reason || 'Invalid hand landmarks');
  }

  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const indexMcp = landmarks[5];
  const pinkyMcp = landmarks[17];

  const scale = Math.max(distance3D(wrist, middleMcp), distance3D(indexMcp, pinkyMcp), 1e-5);

  const features: number[] = [];
  for (let i = 0; i < 21; i++) {
    features.push(
      Number(((landmarks[i].x - wrist.x) / scale).toFixed(6)),
      Number(((landmarks[i].y - wrist.y) / scale).toFixed(6)),
      Number(((landmarks[i].z - wrist.z) / scale).toFixed(6))
    );
  }
  return features;
}

/**
 * Canonical 129-feature extractor supporting both 1-handed and 2-handed ISL signs:
 * - Hand 1 (dominant / Right if available, else leftmost): 63 normalized features
 * - Hand 2 (base / Left if available): 63 normalized features (zeros if 1 hand)
 * - Inter-hand relative wrist vector (wrist2 - wrist1) / scale1: 3 features (zeros if 1 hand)
 */
export function extractFeatureVector(hands: DetectedHandInput[]): {
  features: number[];
  handCount: 1 | 2;
} {
  if (!Array.isArray(hands) || hands.length === 0) {
    throw new Error('At least 1 valid hand is required to extract features.');
  }

  const validHands = hands.slice(0, 2);
  for (const h of validHands) {
    const check = validateHandLandmarks(h.landmarks);
    if (!check.valid) {
      throw new Error(check.reason || 'Invalid hand landmarks');
    }
  }

  // Order consistently: Right/Dominant first, else sort by wrist.x descending (right side of image first)
  const ordered = [...validHands].sort((a, b) => {
    if (a.handedness === 'Right' && b.handedness !== 'Right') return -1;
    if (b.handedness === 'Right' && a.handedness !== 'Right') return 1;
    return b.landmarks[0].x - a.landmarks[0].x;
  });

  const hand1 = ordered[0].landmarks;
  const f1 = normalizeSingleHand(hand1);

  if (ordered.length === 1) {
    return {
      features: [...f1, ...new Array(63).fill(0), 0, 0, 0],
      handCount: 1,
    };
  }

  const hand2 = ordered[1].landmarks;
  const f2 = normalizeSingleHand(hand2);

  const scale1 = Math.max(
    distance3D(hand1[0], hand1[9]),
    distance3D(hand1[5], hand1[17]),
    1e-5
  );
  const relX = Number(((hand2[0].x - hand1[0].x) / scale1).toFixed(6));
  const relY = Number(((hand2[0].y - hand1[0].y) / scale1).toFixed(6));
  const relZ = Number(((hand2[0].z - hand1[0].z) / scale1).toFixed(6));

  return {
    features: [...f1, ...f2, relX, relY, relZ],
    handCount: 2,
  };
}

export function validateFeatureVector(features: unknown): features is number[] {
  if (!Array.isArray(features)) return false;
  if (features.length !== SINGLE_HAND_FEATURE_COUNT && features.length !== DUAL_HAND_FEATURE_COUNT) {
    return false;
  }
  for (let i = 0; i < features.length; i++) {
    if (typeof features[i] !== 'number' || !Number.isFinite(features[i])) {
      return false;
    }
  }
  return true;
}

/**
 * Adapts a 63-feature single-hand vector into the canonical 129-feature space
 * so API clients sending either 63 or 129 features are seamlessly supported.
 */
export function toCanonicalFeatures(features: number[]): number[] {
  if (features.length === DUAL_HAND_FEATURE_COUNT) {
    return features;
  }
  if (features.length === SINGLE_HAND_FEATURE_COUNT) {
    return [...features, ...new Array(66).fill(0)];
  }
  throw new Error(
    `Expected ${SINGLE_HAND_FEATURE_COUNT} or ${DUAL_HAND_FEATURE_COUNT} features, got ${features.length}.`
  );
}

/**
 * Temporal prediction smoother preventing erratic letter flicker across consecutive webcam frames.
 */
export class PredictionSmoother {
  private buffer: Array<{ label: string | null; confidence: number }> = [];
  private readonly windowSize: number;
  private readonly minConsensus: number;
  private readonly confidenceThreshold: number;

  constructor(windowSize = 6, minConsensus = 3, confidenceThreshold = 0.42) {
    this.windowSize = windowSize;
    this.minConsensus = minConsensus;
    this.confidenceThreshold = confidenceThreshold;
  }

  push(label: string | null, confidence: number): {
    stableLabel: string | null;
    smoothedConfidence: number;
    isStable: boolean;
  } {
    this.buffer.push({ label, confidence });
    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }

    const counts = new Map<string, { count: number; confSum: number }>();
    for (const item of this.buffer) {
      if (!item.label || item.confidence < this.confidenceThreshold) continue;
      const entry = counts.get(item.label) ?? { count: 0, confSum: 0 };
      entry.count += 1;
      entry.confSum += item.confidence;
      counts.set(item.label, entry);
    }

    let bestLabel: string | null = null;
    let bestCount = 0;
    let bestConfAvg = 0;

    for (const [key, val] of counts.entries()) {
      const avg = val.confSum / val.count;
      if (val.count > bestCount || (val.count === bestCount && avg > bestConfAvg)) {
        bestLabel = key;
        bestCount = val.count;
        bestConfAvg = avg;
      }
    }

    if (bestLabel && bestCount >= this.minConsensus) {
      return {
        stableLabel: bestLabel,
        smoothedConfidence: Number(bestConfAvg.toFixed(4)),
        isStable: true,
      };
    }

    return {
      stableLabel: null,
      smoothedConfidence: Number(confidence.toFixed(4)),
      isStable: false,
    };
  }

  reset(): void {
    this.buffer = [];
  }
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  t: number;
}

/**
 * Resamples a 2D motion stroke into N evenly spaced normalized points in [0, 1] x [0, 1].
 */
export function resampleTrajectory(points: TrajectoryPoint[], n = 16): Array<{ x: number; y: number }> {
  if (points.length < 2) return [];
  const segLengths: number[] = [];
  let totalLength = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    segLengths.push(d);
    totalLength += d;
  }
  if (totalLength < 1e-4) return [];

  const resampled: Array<{ x: number; y: number }> = [{ x: points[0].x, y: points[0].y }];
  const step = totalLength / (n - 1);
  let accum = 0;
  let segIdx = 0;

  for (let k = 1; k < n - 1; k++) {
    const target = k * step;
    while (segIdx < segLengths.length - 1 && accum + segLengths[segIdx] < target) {
      accum += segLengths[segIdx];
      segIdx++;
    }
    const len = segLengths[segIdx] || 1e-6;
    const ratio = Math.max(0, Math.min(1, (target - accum) / len));
    const p0 = points[segIdx];
    const p1 = points[segIdx + 1] ?? points[segIdx];
    resampled.push({
      x: p0.x + ratio * (p1.x - p0.x),
      y: p0.y + ratio * (p1.y - p0.y),
    });
  }
  resampled.push({ x: points[points.length - 1].x, y: points[points.length - 1].y });

  // Normalize into [0, 1] x [0, 1]
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of resampled) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const spanX = Math.max(maxX - minX, 0.05);
  const spanY = Math.max(maxY - minY, 0.05);

  return resampled.map((p) => ({
    x: (p.x - minX) / spanX,
    y: (p.y - minY) / spanY,
  }));
}

// Canonical normalized 16-point trajectory templates (screen coords: y increases downward)
const TEMPLATE_J: Array<{ x: number; y: number }> = resampleTrajectory(
  [
    { x: 0.75, y: 0.0, t: 0 },
    { x: 0.75, y: 0.4, t: 1 },
    { x: 0.75, y: 0.75, t: 2 },
    { x: 0.65, y: 0.95, t: 3 },
    { x: 0.4, y: 1.0, t: 4 },
    { x: 0.15, y: 0.9, t: 5 },
    { x: 0.05, y: 0.7, t: 6 },
  ],
  16
);

const TEMPLATE_Z: Array<{ x: number; y: number }> = resampleTrajectory(
  [
    { x: 0.0, y: 0.0, t: 0 },
    { x: 0.5, y: 0.0, t: 1 },
    { x: 1.0, y: 0.0, t: 2 },
    { x: 0.5, y: 0.5, t: 3 },
    { x: 0.0, y: 1.0, t: 4 },
    { x: 0.5, y: 1.0, t: 5 },
    { x: 1.0, y: 1.0, t: 6 },
  ],
  16
);

export function recognizeDynamicTrajectory(points: TrajectoryPoint[]): {
  prediction: 'J' | 'Z' | null;
  confidence: number;
  status: 'detected' | 'uncertain' | 'insufficient_motion';
  pathLength: number;
  message: string;
} {
  if (!Array.isArray(points) || points.length < 10) {
    return {
      prediction: null,
      confidence: 0,
      status: 'insufficient_motion',
      pathLength: 0,
      message: 'Need at least 10 sequential frames to classify a dynamic sign (J or Z).',
    };
  }

  let pathLength = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < points.length; i++) {
    if (points[i].x < minX) minX = points[i].x;
    if (points[i].x > maxX) maxX = points[i].x;
    if (points[i].y < minY) minY = points[i].y;
    if (points[i].y > maxY) maxY = points[i].y;
    if (i > 0) {
      pathLength += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    }
  }

  const bboxDiag = Math.hypot(maxX - minX, maxY - minY);
  if (pathLength < 0.18 || bboxDiag < 0.09) {
    return {
      prediction: null,
      confidence: 0,
      status: 'insufficient_motion',
      pathLength: Number(pathLength.toFixed(4)),
      message: 'Hand appears stationary; static classifier handles held signs.',
    };
  }

  const resampled = resampleTrajectory(points, 16);
  if (resampled.length !== 16) {
    return {
      prediction: null,
      confidence: 0,
      status: 'insufficient_motion',
      pathLength: Number(pathLength.toFixed(4)),
      message: 'Unable to resample motion trajectory.',
    };
  }

  const distTo = (tpl: Array<{ x: number; y: number }>) => {
    let sum = 0;
    for (let i = 0; i < 16; i++) {
      sum += Math.hypot(resampled[i].x - tpl[i].x, resampled[i].y - tpl[i].y);
    }
    return sum / 16;
  };

  const dJ = distTo(TEMPLATE_J);
  const dZ = distTo(TEMPLATE_Z);
  const bestDist = Math.min(dJ, dZ);
  const bestLabel: 'J' | 'Z' = dJ <= dZ ? 'J' : 'Z';

  // Convert normalized point-wise mean distance into calibrated confidence [0..1]
  const confidence = Math.max(0, Math.min(0.99, 1 - bestDist / 0.55));
  if (confidence < 0.52) {
    return {
      prediction: null,
      confidence: Number(confidence.toFixed(4)),
      status: 'uncertain',
      pathLength: Number(pathLength.toFixed(4)),
      message: 'Motion detected, but stroke trajectory does not clearly match J or Z.',
    };
  }

  return {
    prediction: bestLabel,
    confidence: Number(confidence.toFixed(4)),
    status: 'detected',
    pathLength: Number(pathLength.toFixed(4)),
    message: `Dynamic trajectory matched ISL letter ${bestLabel}.`,
  };
}

export function formatPredictionCandidates(
  probabilities: Record<string, number>,
  limit = 3
): PredictionCandidate[] {
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, confidence]) => ({
      label,
      confidence: Number(confidence.toFixed(4)),
    }));
}
