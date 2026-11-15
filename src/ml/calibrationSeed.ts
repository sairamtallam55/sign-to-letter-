import { extractFeatureVector, type TrajectoryPoint } from './landmarks';
import type { DatasetSample, DetectedHandInput, Landmark3D } from './types';
import { STATIC_ISL_LETTERS } from './islAlphabet';

function seededRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

interface FingerCurlProfile {
  thumbCurl: number; // 0 = extended, 1 = tightly curled
  indexCurl: number;
  middleCurl: number;
  ringCurl: number;
  pinkyCurl: number;
  spread: number;
  tiltRad: number;
}

const STATIC_HAND_PROFILES: Record<
  string,
  { hands: 1 | 2; dom: FingerCurlProfile; base?: FingerCurlProfile; offset?: { x: number; y: number; z: number } }
> = {
  A: {
    hands: 2,
    dom: { thumbCurl: 0.05, indexCurl: 0.85, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.1, tiltRad: 0.0 },
    base: { thumbCurl: 0.05, indexCurl: 0.15, middleCurl: 0.15, ringCurl: 0.2, pinkyCurl: 0.2, spread: 0.15, tiltRad: -0.3 },
    offset: { x: -0.14, y: 0.04, z: 0.02 },
  },
  B: {
    hands: 2,
    dom: { thumbCurl: 0.35, indexCurl: 0.1, middleCurl: 0.1, ringCurl: 0.15, pinkyCurl: 0.15, spread: 0.08, tiltRad: 0.2 },
    base: { thumbCurl: 0.2, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.05, tiltRad: 0.0 },
    offset: { x: -0.1, y: 0.0, z: 0.01 },
  },
  C: {
    hands: 1,
    dom: { thumbCurl: 0.3, indexCurl: 0.45, middleCurl: 0.45, ringCurl: 0.45, pinkyCurl: 0.45, spread: 0.12, tiltRad: 0.35 },
  },
  D: {
    hands: 2,
    dom: { thumbCurl: 0.4, indexCurl: 0.5, middleCurl: 0.85, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.1, tiltRad: 0.25 },
    base: { thumbCurl: 0.8, indexCurl: 0.02, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: 0.0 },
    offset: { x: -0.11, y: -0.04, z: 0.0 },
  },
  E: {
    hands: 2,
    dom: { thumbCurl: 0.6, indexCurl: 0.35, middleCurl: 0.65, ringCurl: 0.75, pinkyCurl: 0.8, spread: 0.1, tiltRad: -0.2 },
    base: { thumbCurl: 0.7, indexCurl: 0.05, middleCurl: 0.85, ringCurl: 0.85, pinkyCurl: 0.85, spread: 0.08, tiltRad: 0.1 },
    offset: { x: -0.16, y: 0.02, z: 0.02 },
  },
  F: {
    hands: 2,
    dom: { thumbCurl: 0.75, indexCurl: 0.08, middleCurl: 0.08, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.08, tiltRad: -1.45 },
    base: { thumbCurl: 0.75, indexCurl: 0.08, middleCurl: 0.08, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.08, tiltRad: 0.05 },
    offset: { x: -0.09, y: -0.03, z: -0.02 },
  },
  G: {
    hands: 2,
    dom: { thumbCurl: 0.8, indexCurl: 0.92, middleCurl: 0.92, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.05, tiltRad: 0.05 },
    base: { thumbCurl: 0.8, indexCurl: 0.92, middleCurl: 0.92, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.05, tiltRad: -0.05 },
    offset: { x: -0.02, y: 0.14, z: 0.0 },
  },
  H: {
    hands: 2,
    dom: { thumbCurl: 0.25, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.06, tiltRad: -1.5 },
    base: { thumbCurl: 0.2, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.06, tiltRad: 0.0 },
    offset: { x: -0.05, y: 0.06, z: -0.03 },
  },
  I: {
    hands: 1,
    dom: { thumbCurl: 0.85, indexCurl: 0.02, middleCurl: 0.92, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.04, tiltRad: 0.0 },
  },
  K: {
    hands: 2,
    dom: { thumbCurl: 0.6, indexCurl: 0.42, middleCurl: 0.88, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.1, tiltRad: -0.55 },
    base: { thumbCurl: 0.8, indexCurl: 0.03, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: 0.0 },
    offset: { x: -0.14, y: -0.02, z: 0.01 },
  },
  L: {
    hands: 1,
    dom: { thumbCurl: 0.02, indexCurl: 0.02, middleCurl: 0.92, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.45, tiltRad: 0.0 },
  },
  M: {
    hands: 2,
    dom: { thumbCurl: 0.8, indexCurl: 0.12, middleCurl: 0.12, ringCurl: 0.12, pinkyCurl: 0.92, spread: 0.14, tiltRad: 1.4 },
    base: { thumbCurl: 0.15, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.1, tiltRad: -0.1 },
    offset: { x: -0.06, y: 0.11, z: -0.02 },
  },
  N: {
    hands: 2,
    dom: { thumbCurl: 0.8, indexCurl: 0.12, middleCurl: 0.12, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.12, tiltRad: 1.4 },
    base: { thumbCurl: 0.15, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.1, tiltRad: -0.1 },
    offset: { x: -0.06, y: 0.11, z: -0.02 },
  },
  O: {
    hands: 1,
    dom: { thumbCurl: 0.55, indexCurl: 0.62, middleCurl: 0.64, ringCurl: 0.66, pinkyCurl: 0.68, spread: 0.04, tiltRad: 0.15 },
  },
  P: {
    hands: 2,
    dom: { thumbCurl: 0.45, indexCurl: 0.55, middleCurl: 0.88, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.08, tiltRad: -0.35 },
    base: { thumbCurl: 0.82, indexCurl: 0.03, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: 0.0 },
    offset: { x: -0.1, y: -0.12, z: 0.0 },
  },
  Q: {
    hands: 2,
    dom: { thumbCurl: 0.5, indexCurl: 0.58, middleCurl: 0.9, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.05, tiltRad: 0.85 },
    base: { thumbCurl: 0.45, indexCurl: 0.55, middleCurl: 0.6, ringCurl: 0.62, pinkyCurl: 0.65, spread: 0.06, tiltRad: 0.0 },
    offset: { x: -0.12, y: 0.08, z: 0.01 },
  },
  R: {
    hands: 2,
    dom: { thumbCurl: 0.82, indexCurl: 0.48, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: 1.1 },
    base: { thumbCurl: 0.2, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.08, tiltRad: 0.0 },
    offset: { x: -0.07, y: 0.09, z: -0.01 },
  },
  S: {
    hands: 2,
    dom: { thumbCurl: 0.78, indexCurl: 0.85, middleCurl: 0.85, ringCurl: 0.85, pinkyCurl: 0.35, spread: 0.12, tiltRad: 0.4 },
    base: { thumbCurl: 0.78, indexCurl: 0.85, middleCurl: 0.85, ringCurl: 0.85, pinkyCurl: 0.35, spread: 0.12, tiltRad: -0.4 },
    offset: { x: -0.15, y: 0.01, z: 0.01 },
  },
  T: {
    hands: 2,
    dom: { thumbCurl: 0.75, indexCurl: 0.05, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: -1.55 },
    base: { thumbCurl: 0.2, indexCurl: 0.05, middleCurl: 0.05, ringCurl: 0.05, pinkyCurl: 0.05, spread: 0.06, tiltRad: 0.0 },
    offset: { x: -0.18, y: -0.05, z: 0.0 },
  },
  U: {
    hands: 1,
    dom: { thumbCurl: 0.85, indexCurl: 0.03, middleCurl: 0.03, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.02, tiltRad: 0.0 },
  },
  V: {
    hands: 1,
    dom: { thumbCurl: 0.85, indexCurl: 0.03, middleCurl: 0.03, ringCurl: 0.92, pinkyCurl: 0.92, spread: 0.42, tiltRad: 0.0 },
  },
  W: {
    hands: 2,
    dom: { thumbCurl: 0.1, indexCurl: 0.06, middleCurl: 0.06, ringCurl: 0.06, pinkyCurl: 0.06, spread: 0.38, tiltRad: 0.35 },
    base: { thumbCurl: 0.1, indexCurl: 0.06, middleCurl: 0.06, ringCurl: 0.06, pinkyCurl: 0.06, spread: 0.38, tiltRad: -0.35 },
    offset: { x: -0.09, y: 0.02, z: -0.02 },
  },
  X: {
    hands: 2,
    dom: { thumbCurl: 0.82, indexCurl: 0.04, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: -0.75 },
    base: { thumbCurl: 0.82, indexCurl: 0.04, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: 0.75 },
    offset: { x: -0.14, y: 0.0, z: -0.02 },
  },
  Y: {
    hands: 2,
    dom: { thumbCurl: 0.8, indexCurl: 0.05, middleCurl: 0.9, ringCurl: 0.9, pinkyCurl: 0.9, spread: 0.05, tiltRad: 1.2 },
    base: { thumbCurl: 0.05, indexCurl: 0.05, middleCurl: 0.88, ringCurl: 0.88, pinkyCurl: 0.88, spread: 0.48, tiltRad: -0.2 },
    offset: { x: -0.13, y: -0.04, z: 0.01 },
  },
};

function synthesizeHandLandmarks(
  profile: FingerCurlProfile,
  wristOrigin: { x: number; y: number; z: number },
  scale: number,
  mirrorX: boolean,
  rng: () => number,
  noiseStd: number
): Landmark3D[] {
  const landmarks: Landmark3D[] = new Array(21);
  const signX = mirrorX ? -1 : 1;

  const jitter = () => (rng() - 0.5) * 2 * noiseStd;
  const rot = (dx: number, dy: number): { x: number; y: number } => {
    const c = Math.cos(profile.tiltRad);
    const s = Math.sin(profile.tiltRad);
    const rx = dx * c - dy * s;
    const ry = dx * s + dy * c;
    return { x: rx * signX, y: ry };
  };

  landmarks[0] = { x: wristOrigin.x + jitter(), y: wristOrigin.y + jitter(), z: wristOrigin.z + jitter() };

  // Thumb (1..4)
  const thumbAngles = [-0.55 - profile.spread * 0.4, -0.35 + profile.thumbCurl * 0.7, profile.thumbCurl * 0.65, profile.thumbCurl * 0.5];
  let tx = 0;
  let ty = 0;
  let tz = 0;
  const thumbLens = [0.22, 0.2, 0.16, 0.14];
  for (let i = 0; i < 4; i++) {
    const a = thumbAngles[i];
    tx += Math.sin(a) * thumbLens[i] * scale;
    ty -= Math.cos(a) * thumbLens[i] * scale;
    tz -= profile.thumbCurl * 0.05 * scale;
    const r = rot(tx, ty);
    landmarks[1 + i] = {
      x: wristOrigin.x + r.x + jitter(),
      y: wristOrigin.y + r.y + jitter(),
      z: wristOrigin.z + tz + jitter(),
    };
  }

  // Four fingers: Index(5..8), Middle(9..12), Ring(13..16), Pinky(17..20)
  const curls = [profile.indexCurl, profile.middleCurl, profile.ringCurl, profile.pinkyCurl];
  const baseOffsetsX = [-0.22, -0.07, 0.08, 0.22];
  const mcpY = [-0.55, -0.58, -0.55, -0.5];
  const boneLens = [0.24, 0.16, 0.12];

  for (let f = 0; f < 4; f++) {
    const curl = curls[f];
    const spreadOffset = (f - 1.5) * profile.spread * 0.28;
    const mcpLocalX = (baseOffsetsX[f] + spreadOffset * 0.35) * scale;
    const mcpLocalY = mcpY[f] * scale;
    const rMcp = rot(mcpLocalX, mcpLocalY);
    const mcpIdx = 5 + f * 4;
    landmarks[mcpIdx] = {
      x: wristOrigin.x + rMcp.x + jitter(),
      y: wristOrigin.y + rMcp.y + jitter(),
      z: wristOrigin.z + jitter(),
    };

    let fx = mcpLocalX;
    let fy = mcpLocalY;
    let fz = 0;
    for (let seg = 0; seg < 3; seg++) {
      // Curling pushes segments back toward the palm (positive y, negative/positive z)
      const len = boneLens[seg] * scale;
      const forwardDrop = Math.cos(curl * Math.PI * 0.85);
      fx += spreadOffset * 0.14 * scale;
      fy -= len * forwardDrop;
      fz -= Math.sin(curl * Math.PI * 0.7) * len;
      const r = rot(fx, fy);
      landmarks[mcpIdx + 1 + seg] = {
        x: wristOrigin.x + r.x + jitter(),
        y: wristOrigin.y + r.y + jitter(),
        z: wristOrigin.z + fz + jitter(),
      };
    }
  }

  return landmarks;
}

export function generateSimulatedHandsForLetter(
  letter: string,
  seed = 1234,
  noiseStd = 0.004
): DetectedHandInput[] {
  const upper = letter.toUpperCase();
  const spec = STATIC_HAND_PROFILES[upper] ?? STATIC_HAND_PROFILES.A;
  const rng = seededRng(seed);

  const hand1Origin = {
    x: 0.58 + (rng() - 0.5) * 0.06,
    y: 0.68 + (rng() - 0.5) * 0.06,
    z: 0,
  };
  const scale = 0.24 + (rng() - 0.5) * 0.04;
  const domLandmarks = synthesizeHandLandmarks(spec.dom, hand1Origin, scale, false, rng, noiseStd);

  const hands: DetectedHandInput[] = [
    { landmarks: domLandmarks, handedness: 'Right', score: 0.98 },
  ];

  if (spec.hands === 2 && spec.base && spec.offset) {
    const hand2Origin = {
      x: hand1Origin.x + spec.offset.x + (rng() - 0.5) * 0.015,
      y: hand1Origin.y + spec.offset.y + (rng() - 0.5) * 0.015,
      z: hand1Origin.z + spec.offset.z,
    };
    const baseLandmarks = synthesizeHandLandmarks(
      spec.base,
      hand2Origin,
      scale * (0.97 + rng() * 0.06),
      true,
      rng,
      noiseStd
    );
    hands.push({ landmarks: baseLandmarks, handedness: 'Left', score: 0.96 });
  }

  return hands;
}

export function generateSimulatedDynamicTrajectory(letter: 'J' | 'Z'): TrajectoryPoint[] {
  const baseTime = Date.now();
  if (letter === 'J') {
    const pts = [
      { x: 0.62, y: 0.25 },
      { x: 0.62, y: 0.35 },
      { x: 0.62, y: 0.45 },
      { x: 0.62, y: 0.55 },
      { x: 0.61, y: 0.65 },
      { x: 0.58, y: 0.72 },
      { x: 0.52, y: 0.76 },
      { x: 0.45, y: 0.75 },
      { x: 0.39, y: 0.71 },
      { x: 0.36, y: 0.64 },
      { x: 0.35, y: 0.58 },
      { x: 0.35, y: 0.54 },
    ];
    return pts.map((p, idx) => ({ ...p, t: baseTime + idx * 45 }));
  }

  const pts = [
    { x: 0.32, y: 0.28 },
    { x: 0.42, y: 0.28 },
    { x: 0.54, y: 0.28 },
    { x: 0.65, y: 0.28 },
    { x: 0.57, y: 0.38 },
    { x: 0.48, y: 0.48 },
    { x: 0.40, y: 0.58 },
    { x: 0.33, y: 0.68 },
    { x: 0.43, y: 0.68 },
    { x: 0.55, y: 0.68 },
    { x: 0.66, y: 0.68 },
    { x: 0.69, y: 0.68 },
  ];
  return pts.map((p, idx) => ({ ...p, t: baseTime + idx * 45 }));
}

export function generateAmbiguousHands(seed = 999): DetectedHandInput[] {
  const rng = seededRng(seed);
  const profile: FingerCurlProfile = {
    thumbCurl: 0.52,
    indexCurl: 0.52,
    middleCurl: 0.52,
    ringCurl: 0.52,
    pinkyCurl: 0.52,
    spread: 0.22,
    tiltRad: 0.65,
  };
  return [
    {
      landmarks: synthesizeHandLandmarks(profile, { x: 0.5, y: 0.6, z: 0 }, 0.22, false, rng, 0.025),
      handedness: 'Right',
      score: 0.75,
    },
  ];
}

/**
 * Generates deterministic calibration samples across all 24 static ISL letters
 * across 5 distinct simulated sessions so session-wise splitting has zero leakage.
 */
export function buildCalibrationDataset(samplesPerClass = 25): DatasetSample[] {
  const dataset: DatasetSample[] = [];
  const sessions = ['cal-sess-1', 'cal-sess-2', 'cal-sess-3', 'cal-sess-4', 'cal-sess-5'];

  for (let cIdx = 0; cIdx < STATIC_ISL_LETTERS.length; cIdx++) {
    const letter = STATIC_ISL_LETTERS[cIdx];
    for (let s = 0; s < samplesPerClass; s++) {
      const seed = (cIdx + 1) * 1000 + s * 37;
      const noiseStd = 0.004 + (s % 4) * 0.0018;
      const hands = generateSimulatedHandsForLetter(letter, seed, noiseStd);
      const { features, handCount } = extractFeatureVector(hands);
      dataset.push({
        id: `cal-${letter}-${s + 1}`,
        label: letter,
        sessionId: sessions[s % sessions.length],
        handCount,
        source: 'synthetic_calibration',
        features,
        createdAt: '2026-11-15T00:00:00.000Z',
      });
    }
  }

  return dataset;
}
