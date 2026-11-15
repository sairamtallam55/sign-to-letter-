import test from 'node:test';
import assert from 'node:assert/strict';
import {
  assessPerformanceBottleneck,
  averageMs,
  computeRollingFps,
} from './diagnostics';

test('computeRollingFps calculates accurate FPS from sliding window timestamps', () => {
  // 31 frames spaced 33.33ms apart over 1000ms => 30 fps
  const timestamps: number[] = [];
  for (let i = 0; i <= 30; i++) {
    timestamps.push(1000 + i * (1000 / 30));
  }
  assert.equal(computeRollingFps(timestamps, 2000, 1000), 30);

  // Older timestamps outside window should be ignored
  const mixed = [0, 100, 200, 1500, 1600, 1700, 1800, 1900, 2000];
  // Within [1000, 2000]: 6 timestamps, 5 intervals over 500ms => 10 fps
  assert.equal(computeRollingFps(mixed, 2000, 1000), 10);

  // Insufficient samples return 0
  assert.equal(computeRollingFps([], 1000), 0);
  assert.equal(computeRollingFps([950], 1000), 0);
});

test('averageMs computes rounded arithmetic mean and handles empty arrays', () => {
  assert.equal(averageMs([]), null);
  assert.equal(averageMs([10, 12.5, 13.5]), 12);
  assert.equal(averageMs([4.12, 4.18]), 4.2);
});

test('assessPerformanceBottleneck identifies network, ML inference, and client FPS bottlenecks', () => {
  const networkBottleneck = assessPerformanceBottleneck({
    cameraActive: false,
    simulatedActive: true,
    fps: null,
    frameDetectMs: null,
    roundTripLatencyMs: 215.4,
    modelLatencyMs: 0.45,
  });
  assert.equal(networkBottleneck.severity, 'critical');
  assert.equal(networkBottleneck.label, 'Network / API');

  const inferenceBottleneck = assessPerformanceBottleneck({
    cameraActive: true,
    simulatedActive: false,
    fps: 28,
    frameDetectMs: 14,
    roundTripLatencyMs: 48,
    modelLatencyMs: 38.2,
  });
  assert.equal(inferenceBottleneck.severity, 'critical');
  assert.equal(inferenceBottleneck.label, 'ML Inference');

  const fpsBottleneck = assessPerformanceBottleneck({
    cameraActive: true,
    simulatedActive: false,
    fps: 11,
    frameDetectMs: 52.3,
    roundTripLatencyMs: 18,
    modelLatencyMs: 0.5,
  });
  assert.equal(fpsBottleneck.severity, 'critical');
  assert.equal(fpsBottleneck.label, 'Client FPS Low');

  const optimalLive = assessPerformanceBottleneck({
    cameraActive: true,
    simulatedActive: false,
    fps: 30,
    frameDetectMs: 11.2,
    roundTripLatencyMs: 8.4,
    modelLatencyMs: 0.4,
  });
  assert.equal(optimalLive.severity, 'optimal');
  assert.equal(optimalLive.label, 'Optimal');

  const standby = assessPerformanceBottleneck({
    cameraActive: false,
    simulatedActive: false,
    fps: null,
    frameDetectMs: null,
    roundTripLatencyMs: null,
    modelLatencyMs: null,
  });
  assert.equal(standby.severity, 'idle');
  assert.equal(standby.label, 'Standby');
});
