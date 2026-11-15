import { Activity } from 'lucide-react';
import { assessPerformanceBottleneck } from '../utils/diagnostics';

interface DiagnosticOverlayProps {
  cameraActive: boolean;
  simulatedActive: boolean;
  fps: number | null;
  frameDetectMs: number | null;
  roundTripLatencyMs: number | null;
  modelLatencyMs: number | null;
}

export function DiagnosticOverlay({
  cameraActive,
  simulatedActive,
  fps,
  frameDetectMs,
  roundTripLatencyMs,
  modelLatencyMs,
}: DiagnosticOverlayProps) {
  const assessment = assessPerformanceBottleneck({
    cameraActive,
    simulatedActive,
    fps,
    frameDetectMs,
    roundTripLatencyMs,
    modelLatencyMs,
  });

  const statusToneClass =
    assessment.severity === 'optimal'
      ? 'text-emerald-400'
      : assessment.severity === 'warning'
      ? 'text-amber-400'
      : assessment.severity === 'critical'
      ? 'text-rose-400'
      : 'text-slate-400';

  const effectiveLatency = roundTripLatencyMs ?? modelLatencyMs;
  const latencyToneClass =
    effectiveLatency === null
      ? 'text-slate-400'
      : effectiveLatency > 120
      ? 'text-rose-400'
      : effectiveLatency > 80
      ? 'text-amber-400'
      : 'text-slate-100';

  const fpsToneClass =
    !cameraActive || fps === null
      ? 'text-slate-400'
      : fps < 15
      ? 'text-rose-400'
      : fps < 22
      ? 'text-amber-400'
      : 'text-emerald-400';

  return (
    <aside
      data-testid="diagnostic-overlay"
      aria-label="Performance diagnostics"
      className="absolute top-2.5 left-2.5 right-2.5 sm:left-auto sm:top-3 sm:right-3 sm:w-56 z-20 rounded-lg bg-slate-950/85 backdrop-blur-sm border border-slate-700/80 px-3 py-2 sm:py-2.5 text-slate-200 shadow-lg pointer-events-auto"
    >
      <div className="flex items-center justify-between gap-2 pb-1.5 border-b border-slate-800">
        <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-slate-300">
          <Activity className="w-3.5 h-3.5 text-blue-400" aria-hidden="true" />
          <span>Diagnostics</span>
        </span>
        <span
          data-testid="diagnostic-status"
          className={`text-[11px] font-mono font-semibold whitespace-nowrap ${statusToneClass}`}
        >
          {assessment.label}
        </span>
      </div>

      <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1.5 font-mono tabular-nums">
        <div>
          <dt className="text-[10px] font-sans text-slate-400">Frame Rate</dt>
          <dd
            data-testid="diagnostic-fps"
            className={`text-sm font-bold leading-tight ${fpsToneClass}`}
          >
            {cameraActive && fps !== null ? `${fps} fps` : '-- fps'}
          </dd>
        </div>

        <div>
          <dt className="text-[10px] font-sans text-slate-400">Pred. Latency</dt>
          <dd
            data-testid="diagnostic-latency"
            className={`text-sm font-bold leading-tight ${latencyToneClass}`}
          >
            {effectiveLatency !== null ? `${effectiveLatency.toFixed(1)} ms` : '-- ms'}
          </dd>
        </div>
      </dl>

      {(modelLatencyMs !== null || (cameraActive && frameDetectMs !== null)) && (
        <div className="mt-1.5 pt-1.5 border-t border-slate-800/80 flex items-center justify-between gap-2 text-[10px] font-mono tabular-nums text-slate-400">
          {modelLatencyMs !== null && (
            <span data-testid="diagnostic-model-latency">
              Model: {modelLatencyMs.toFixed(2)} ms
            </span>
          )}
          {cameraActive && frameDetectMs !== null && (
            <span data-testid="diagnostic-detect-time">
              Detect: {frameDetectMs.toFixed(1)} ms
            </span>
          )}
        </div>
      )}

      <p
        data-testid="diagnostic-summary"
        className="mt-1.5 text-[10px] leading-snug text-slate-400"
      >
        {assessment.summary}
      </p>
    </aside>
  );
}
