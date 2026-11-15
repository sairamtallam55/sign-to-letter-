import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DUAL_HAND_FEATURE_COUNT,
  PredictionSmoother,
  SINGLE_HAND_FEATURE_COUNT,
  extractFeatureVector,
  normalizeSingleHand,
  recognizeDynamicTrajectory,
  toCanonicalFeatures,
  validateFeatureVector,
  validateHandLandmarks,
} from './landmarks';
import {
  generateSimulatedDynamicTrajectory,
  generateSimulatedHandsForLetter,
} from './calibrationSeed';

test('validateHandLandmarks accepts 21 valid landmarks and rejects malformed/degenerate inputs', () => {
  const validHands = generateSimulatedHandsForLetter('V', 42);
  assert.equal(validateHandLandmarks(validHands[0].landmarks).valid, true);

  assert.equal(validateHandLandmarks([]).valid, false);
  assert.equal(validateHandLandmarks(validHands[0].landmarks.slice(0, 20)).valid, false);

  const degenerate = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  assert.equal(validateHandLandmarks(degenerate).valid, false);

  const withNaN = validHands[0].landmarks.map((p, idx) =>
    idx === 5 ? { ...p, x: Number.NaN } : p
  );
  assert.equal(validateHandLandmarks(withNaN).valid, false);
});

test('normalizeSingleHand produces 63 features invariant to translation and uniform scaling', () => {
  const hand = generateSimulatedHandsForLetter('I', 101, 0)[0].landmarks;
  const fOriginal = normalizeSingleHand(hand);
  assert.equal(fOriginal.length, SINGLE_HAND_FEATURE_COUNT);
  assert.equal(fOriginal[0], 0);
  assert.equal(fOriginal[1], 0);
  assert.equal(fOriginal[2], 0);

  const translatedAndScaled = hand.map((p) => ({
    x: p.x * 1.75 + 0.3,
    y: p.y * 1.75 - 0.15,
    z: p.z * 1.75 + 0.08,
  }));
  const fTransformed = normalizeSingleHand(translatedAndScaled);

  for (let i = 0; i < SINGLE_HAND_FEATURE_COUNT; i++) {
    assert.ok(
      Math.abs(fOriginal[i] - fTransformed[i]) < 1e-4,
      `Feature ${i} drifted: ${fOriginal[i]} vs ${fTransformed[i]}`
    );
  }
});

test('extractFeatureVector produces 129 canonical features for both 1-hand and 2-hand ISL signs', () => {
  const single = generateSimulatedHandsForLetter('C', 10, 0);
  const res1 = extractFeatureVector(single);
  assert.equal(res1.handCount, 1);
  assert.equal(res1.features.length, DUAL_HAND_FEATURE_COUNT);
  assert.equal(res1.features[126], 0);

  const dual = generateSimulatedHandsForLetter('M', 10, 0);
  const res2 = extractFeatureVector(dual);
  assert.equal(res2.handCount, 2);
  assert.equal(res2.features.length, DUAL_HAND_FEATURE_COUNT);
  assert.ok(Math.abs(res2.features[126]) > 0 || Math.abs(res2.features[127]) > 0);

  assert.equal(validateFeatureVector(res1.features), true);
  assert.equal(validateFeatureVector(new Array(63).fill(0.1)), true);
  assert.equal(validateFeatureVector([1, 2, 3]), false);
  assert.equal(toCanonicalFeatures(new Array(63).fill(0.2)).length, DUAL_HAND_FEATURE_COUNT);
});

test('PredictionSmoother stabilizes consecutive predictions and suppresses low-confidence noise', () => {
  const smoother = new PredictionSmoother(5, 3, 0.45);
  assert.equal(smoother.push('A', 0.85).isStable, false);
  assert.equal(smoother.push('B', 0.80).isStable, false);
  assert.equal(smoother.push('A', 0.88).isStable, false);
  const fourth = smoother.push('A', 0.90);
  assert.equal(fourth.isStable, true);
  assert.equal(fourth.stableLabel, 'A');

  smoother.reset();
  assert.equal(smoother.push('A', 0.25).isStable, false);
  assert.equal(smoother.push('A', 0.20).isStable, false);
  assert.equal(smoother.push('A', 0.30).isStable, false);
});

test('recognizeDynamicTrajectory distinguishes dynamic J and Z strokes from stationary frames', () => {
  const jPoints = generateSimulatedDynamicTrajectory('J');
  const resJ = recognizeDynamicTrajectory(jPoints);
  assert.equal(resJ.status, 'detected');
  assert.equal(resJ.prediction, 'J');
  assert.ok(resJ.confidence >= 0.7);

  const zPoints = generateSimulatedDynamicTrajectory('Z');
  const resZ = recognizeDynamicTrajectory(zPoints);
  assert.equal(resZ.status, 'detected');
  assert.equal(resZ.prediction, 'Z');
  assert.ok(resZ.confidence >= 0.7);

  const stationary = Array.from({ length: 15 }, (_, i) => ({
    x: 0.5,
    y: 0.5,
    t: 1000 + i * 33,
  }));
  assert.equal(recognizeDynamicTrajectory(stationary).status, 'insufficient_motion');
});
