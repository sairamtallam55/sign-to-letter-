import test from 'node:test';
import assert from 'node:assert/strict';
import { buildCalibrationDataset, generateSimulatedHandsForLetter } from './calibrationSeed';
import { extractFeatureVector } from './landmarks';
import { RandomForestModel, splitDataset, trainAndEvaluatePipeline } from './randomForest';

test('splitDataset performs zero-leakage session-wise splitting when multiple sessions exist', () => {
  const dataset = buildCalibrationDataset(10);
  const { train, validation, test: testSet, splitStrategy } = splitDataset(dataset);
  assert.equal(splitStrategy, 'session-wise');
  assert.ok(train.length > 0 && validation.length > 0 && testSet.length > 0);

  const trainSessions = new Set(train.map((s) => s.sessionId));
  const testSessions = new Set(testSet.map((s) => s.sessionId));
  for (const s of testSessions) {
    assert.equal(trainSessions.has(s), false, `Session ${s} leaked across train and test!`);
  }
});

test('trainAndEvaluatePipeline trains Random Forest + Nearest Centroid baseline and evaluates all 24 static classes', () => {
  const dataset = buildCalibrationDataset(15);
  const { productionModel, report } = trainAndEvaluatePipeline(dataset);

  assert.equal(report.classesEvaluated.length, 24);
  assert.equal(report.confusionMatrix.length, 24);
  assert.equal(report.comparison.length, 2);
  assert.ok(report.accuracy > 0.85, `Expected high calibration accuracy, got ${report.accuracy}`);
  assert.ok(report.macroF1 > 0.85);

  // Verify live inference on unseen noisy samples for representative 1-hand and 2-hand letters
  for (const letter of ['A', 'B', 'C', 'M', 'N', 'V', 'W', 'Y']) {
    const { features } = extractFeatureVector(generateSimulatedHandsForLetter(letter, 99999, 0.004));
    const pred = productionModel.predict(features);
    assert.equal(pred.label, letter, `Expected ${letter} but predicted ${pred.label}`);
    assert.ok(pred.confidence > 0.4);
  }

  // Verify JSON serialization roundtrip
  const serialized = JSON.parse(JSON.stringify(productionModel.model));
  const restored = new RandomForestModel(serialized);
  const sampleFeatures = extractFeatureVector(generateSimulatedHandsForLetter('K', 888, 0.002)).features;
  assert.deepEqual(restored.predict(sampleFeatures).label, productionModel.predict(sampleFeatures).label);
});
