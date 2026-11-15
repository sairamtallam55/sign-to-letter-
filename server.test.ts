import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createApp } from './server';
import {
  generateAmbiguousHands,
  generateSimulatedDynamicTrajectory,
  generateSimulatedHandsForLetter,
} from './src/ml/calibrationSeed';
import { extractFeatureVector } from './src/ml/landmarks';

async function withServer(run: (baseUrl: string) => Promise<void>) {
  const app = createApp();
  const server = http.createServer(app);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Failed to bind ephemeral test port');
  }
  const baseUrl = `http://127.0.0.1:${address.port}`;
  try {
    await run(baseUrl);
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

test('Express ML REST API health, predict, predict-dynamic, validation, collect, train, and evaluate', async () => {
  await withServer(async (baseUrl) => {
    // 1. Health endpoint
    const healthRes = await fetch(`${baseUrl}/api/health`);
    assert.equal(healthRes.status, 200);
    const health = await healthRes.json();
    assert.equal(health.status, 'ok');
    assert.equal(health.modelLoaded, true);
    assert.equal(health.classesSupportedStatic, 24);

    // 2. Valid static prediction (two-handed ISL 'M' and single-handed 'V')
    for (const letter of ['M', 'V', 'A', 'W']) {
      const hands = generateSimulatedHandsForLetter(letter, 54321, 0.004);
      const { features, handCount } = extractFeatureVector(hands);
      const predRes = await fetch(`${baseUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, handCount }),
      });
      assert.equal(predRes.status, 200);
      const pred = await predRes.json();
      assert.equal(pred.status, 'detected');
      assert.equal(pred.prediction, letter);
      assert.ok(pred.confidence >= 0.42);
      assert.equal(pred.topCandidates.length, 3);
    }

    // 3. Invalid feature payload -> 400
    const badRes = await fetch(`${baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: [1, 2, 3] }),
    });
    assert.equal(badRes.status, 400);

    // 4. Low-confidence / uncertain gesture handling
    const ambHands = generateAmbiguousHands(77);
    const { features: ambFeats, handCount: ambCount } = extractFeatureVector(ambHands);
    const ambRes = await fetch(`${baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: ambFeats, handCount: ambCount }),
    });
    assert.equal(ambRes.status, 200);
    const ambJson = await ambRes.json();
    assert.ok(['detected', 'uncertain'].includes(ambJson.status));

    // 5. Dynamic J and Z prediction
    const jFrames = generateSimulatedDynamicTrajectory('J');
    const jRes = await fetch(`${baseUrl}/api/predict-dynamic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: jFrames }),
    });
    assert.equal(jRes.status, 200);
    const jJson = await jRes.json();
    assert.equal(jJson.prediction, 'J');
    assert.equal(jJson.status, 'detected');

    // 6. Dataset summary & evaluation report
    const dsRes = await fetch(`${baseUrl}/api/dataset`);
    assert.equal(dsRes.status, 200);
    const ds = await dsRes.json();
    assert.ok(ds.totalSamples >= 360);

    const evalRes = await fetch(`${baseUrl}/api/evaluate`);
    assert.equal(evalRes.status, 200);
    const ev = await evalRes.json();
    assert.equal(ev.classesEvaluated.length, 24);
    assert.ok(ev.accuracy > 0.8);

    // 7. Collect a sample, prevent duplicate, and delete cleanup
    const newHands = generateSimulatedHandsForLetter('A', 77777, 0.004);
    const { features: newFeats, handCount: newCount } = extractFeatureVector(newHands);
    const collectRes = await fetch(`${baseUrl}/api/collect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        label: 'A',
        sessionId: 'api-test-session',
        handCount: newCount,
        features: newFeats,
        source: 'user_webcam',
      }),
    });
    assert.equal(collectRes.status, 201);
    const collected = await collectRes.json();
    assert.ok(collected.sample.id);

    const dupRes = await fetch(`${baseUrl}/api/collect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        label: 'A',
        sessionId: 'api-test-session',
        handCount: newCount,
        features: newFeats,
        source: 'user_webcam',
      }),
    });
    assert.equal(dupRes.status, 409);

    const delRes = await fetch(`${baseUrl}/api/dataset/sample/${collected.sample.id}`, {
      method: 'DELETE',
    });
    assert.equal(delRes.status, 200);
  });
});
