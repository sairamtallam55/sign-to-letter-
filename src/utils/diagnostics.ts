export type BottleneckSeverity = 'idle' | 'optimal' | 'warning' | 'critical';

export interface DiagnosticMetricsInput {
  cameraActive: boolean;
  simulatedActive: boolean;
  fps: number | null;
  frameDetectMs: number | null;
  roundTripLatencyMs: number | null;
  modelLatencyMs: number | null;
}

export interface BottleneckAssessment {
  severity: BottleneckSeverity;
  label: string;
  summary: string;
}

/**
 * Computes frames per second over a sliding time window (default 1000ms).
 * Expects monotonic timestamps in milliseconds (e.g. from performance.now()).
 */
export function computeRollingFps(
  timestamps: number[],
  nowMs: number,
  windowMs = 1000
): number {
  if (timestamps.length < 2 || windowMs <= 0) {
    return 0;
  }

  const cutoff = nowMs - windowMs;
  const recent = timestamps.filter((t) => t >= cutoff && t <= nowMs);
  if (recent.length < 2) {
    return 0;
  }

  const elapsedMs = recent[recent.length - 1] - recent[0];
  if (elapsedMs <= 0) {
    return 0;
  }

  // Use interval count (N - 1 gaps across elapsed span) for accurate instantaneous FPS
  const rawFps = ((recent.length - 1) * 1000) / elapsedMs;
  return Math.round(rawFps);
}

/**
 * Computes arithmetic mean rounded to 1 decimal place.
 */
export function averageMs(samples: number[]): number | null {
  if (samples.length === 0) return null;
  const total = samples.reduce((sum, val) => sum + val, 0);
  return Math.round((total / samples.length) * 10) / 10;
}

/**
 * Evaluates frame rate and prediction latency to surface performance bottlenecks.
 */
export function assessPerformanceBottleneck(
  input: DiagnosticMetricsInput
): BottleneckAssessment {
  const {
    cameraActive,
    simulatedActive,
    fps,
    frameDetectMs,
    roundTripLatencyMs,
    modelLatencyMs,
  } = input;

  const effectiveLatency = roundTripLatencyMs ?? modelLatencyMs;

  // 1. Severe prediction latency checks (works in both live and simulated modes)
  if (modelLatencyMs !== null && modelLatencyMs > 30) {
    return {
      severity: 'critical',
      label: 'ML Inference',
      summary: `Model scoring took ${modelLatencyMs.toFixed(1)} ms; check server CPU load.`,
    };
  }

  if (
    roundTripLatencyMs !== null &&
    roundTripLatencyMs > 120 &&
    (modelLatencyMs === null || roundTripLatencyMs - modelLatencyMs > 80)
  ) {
    return {
      severity: 'critical',
      label: 'Network / API',
      summary: `Round-trip ${roundTripLatencyMs.toFixed(1)} ms vs ${
        modelLatencyMs !== null ? `${modelLatencyMs.toFixed(2)} ms` : 'fast'
      } model.`,
    };
  }

  // 2. Live camera frame-rate & MediaPipe vision checks
  if (cameraActive) {
    if (fps !== null && fps > 0 && fps < 15) {
      return {
        severity: 'critical',
        label: 'Client FPS Low',
        summary:
          frameDetectMs !== null && frameDetectMs > 40
            ? `MediaPipe detect averaging ${frameDetectMs.toFixed(1)} ms/frame.`
            : 'Low camera frame rate; check lighting or close heavy tabs.',
      };
    }

    if (
      (fps !== null && fps >= 15 && fps < 22) ||
      (frameDetectMs !== null && frameDetectMs > 30) ||
      (effectiveLatency !== null && effectiveLatency > 80)
    ) {
      return {
        severity: 'warning',
        label: 'Elevated Load',
        summary: 'Pipeline running with moderate frame or network overhead.',
      };
    }

    if (fps !== null && fps >= 22) {
      return {
        severity: 'optimal',
        label: 'Optimal',
        summary: 'Real-time landmark tracking and inference within budget.',
      };
    }

    return {
      severity: 'idle',
      label: 'Measuring',
      summary: 'Sampling initial webcam frames...',
    };
  }

  // 3. Simulated mode (no webcam FPS loop)
  if (simulatedActive || effectiveLatency !== null) {
    if (effectiveLatency !== null && effectiveLatency > 80) {
      return {
        severity: 'warning',
        label: 'API Latency',
        summary: 'Prediction response exceeded 80 ms threshold.',
      };
    }
    return {
      severity: 'optimal',
      label: 'API Ready',
      summary: 'Simulated input active · start camera to monitor live FPS.',
    };
  }

  return {
    severity: 'idle',
    label: 'Standby',
    summary: 'Start camera for live FPS or run a simulated sign for latency.',
  };
}
