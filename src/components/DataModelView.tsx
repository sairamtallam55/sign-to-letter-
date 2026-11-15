import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  Camera,
  CameraOff,
  CheckCircle2,
  Download,
  Play,
  RefreshCw,
  RotateCcw,
  Trash2,
} from 'lucide-react';
import { generateSimulatedHandsForLetter } from '../ml/calibrationSeed';
import { ISL_ALPHABET, STATIC_ISL_LETTERS } from '../ml/islAlphabet';
import { extractFeatureVector } from '../ml/landmarks';
import type {
  DatasetSummary,
  DetectedHandInput,
  EvaluationReport,
} from '../ml/types';
import { drawHandLandmarks } from '../utils/handCanvas';
import { getHandLandmarker, mapMediaPipeResult } from '../utils/mediapipeClient';
import { LoadingIndicator } from './LoadingIndicator';

interface DataModelViewProps {
  apiBaseUrl: string;
  onApiBaseUrlChange: (url: string) => void;
  summary: DatasetSummary | null;
  isSummaryLoading?: boolean;
  onRefreshSummary: () => Promise<void>;
}

export function DataModelView({
  apiBaseUrl,
  onApiBaseUrlChange,
  summary,
  isSummaryLoading = false,
  onRefreshSummary,
}: DataModelViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const latestHandsRef = useRef<DetectedHandInput[]>([]);

  const [selectedLetter, setSelectedLetter] = useState<string>('A');
  const [sessionId, setSessionId] = useState<string>('webcam-session-1');
  const [camActive, setCamActive] = useState<boolean>(false);
  const [detectedCount, setDetectedCount] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<{
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [isReportLoading, setIsReportLoading] = useState<boolean>(true);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isCamStarting, setIsCamStarting] = useState<boolean>(false);
  const [isCollecting, setIsCollecting] = useState<boolean>(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [customUrlInput, setCustomUrlInput] = useState<string>(apiBaseUrl);

  const fetchEvaluation = useCallback(async () => {
    setIsReportLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/api/evaluate`);
      if (res.ok) {
        const data = (await res.json()) as EvaluationReport;
        setReport(data);
      }
    } catch {
      // Report will show empty/error state cleanly
    } finally {
      setIsReportLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void fetchEvaluation();
  }, [fetchEvaluation]);

  const stopCollectorCamera = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    latestHandsRef.current = [];
    setDetectedCount(0);
    setCamActive(false);
  }, []);

  useEffect(() => {
    return () => stopCollectorCamera();
  }, [stopCollectorCamera]);

  const startCollectorCamera = async () => {
    setStatusMessage(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatusMessage({
        type: 'error',
        text: 'Webcam API unavailable in this browser. Use "Capture Test Sample" below or run over HTTPS.',
      });
      return;
    }

    setIsCamStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 420 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const landmarker = await getHandLandmarker();
      setCamActive(true);

      const loop = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || !streamRef.current) return;
        if (video.readyState >= 2) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 420;
          }
          const result = landmarker.detectForVideo(video, performance.now());
          const hands = mapMediaPipeResult(result);
          latestHandsRef.current = hands;
          setDetectedCount(hands.length);
          const ctx = canvas.getContext('2d');
          if (ctx) {
            drawHandLandmarks(ctx, canvas.width, canvas.height, hands, [], true);
          }
        }
        rafRef.current = requestAnimationFrame(loop);
      };
      rafRef.current = requestAnimationFrame(loop);
    } catch (err) {
      stopCollectorCamera();
      setStatusMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Unable to start webcam.',
      });
    } finally {
      setIsCamStarting(false);
    }
  };

  const submitSample = async (hands: DetectedHandInput[], source: 'user_webcam' | 'synthetic_calibration') => {
    setIsCollecting(true);
    try {
      const { features, handCount } = extractFeatureVector(hands);
      const res = await fetch(`${apiBaseUrl}/api/collect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label: selectedLetter,
          sessionId: sessionId.trim() || 'webcam-session-1',
          handCount,
          features,
          source,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setStatusMessage({ type: 'error', text: data.error || 'Failed to save sample.' });
        return;
      }
      setStatusMessage({
        type: 'success',
        text: `Saved ${handCount}-hand sample for letter '${selectedLetter}' (${source === 'user_webcam' ? 'Webcam' : 'Simulated'}).`,
      });
      await onRefreshSummary();
    } catch (err) {
      setStatusMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Network error saving sample.',
      });
    } finally {
      setIsCollecting(false);
    }
  };

  const handleCaptureLive = async () => {
    if (latestHandsRef.current.length === 0) {
      setStatusMessage({
        type: 'error',
        text: 'No hand detected in the webcam frame. Position your hand(s) clearly and try again.',
      });
      return;
    }
    await submitSample(latestHandsRef.current, 'user_webcam');
  };

  const handleCaptureSimulated = async () => {
    const seed = Math.floor(Date.now() % 100000);
    const hands = generateSimulatedHandsForLetter(selectedLetter, seed, 0.005);
    if (canvasRef.current && !camActive) {
      canvasRef.current.width = 640;
      canvasRef.current.height = 420;
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) drawHandLandmarks(ctx, 640, 420, hands, [], false);
    }
    await submitSample(hands, 'user_webcam');
  };

  const handleDeleteSample = async (id: string) => {
    setDeletingId(id);
    try {
      const res = await fetch(`${apiBaseUrl}/api/dataset/sample/${id}`, { method: 'DELETE' });
      if (res.ok) {
        await onRefreshSummary();
        setStatusMessage({ type: 'info', text: 'Sample removed from CSV dataset.' });
      }
    } finally {
      setDeletingId(null);
    }
  };

  const handleResetCalibration = async () => {
    setIsTraining(true);
    try {
      const res = await fetch(`${apiBaseUrl}/api/dataset/reset`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setReport(data.report);
        await onRefreshSummary();
        setStatusMessage({
          type: 'info',
          text: 'Dataset reset to baseline 24-class calibration prototypes and model retrained.',
        });
      }
    } finally {
      setIsTraining(false);
    }
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    setStatusMessage(null);
    try {
      const res = await fetch(`${apiBaseUrl}/api/train`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) {
        setStatusMessage({ type: 'error', text: data.error || 'Training failed.' });
        return;
      }
      setReport(data.report);
      await onRefreshSummary();
      setStatusMessage({
        type: 'success',
        text: `Random Forest retrained on ${data.report.totalSamples} samples (${Math.round(
          data.report.accuracy * 100
        )}% held-out test accuracy).`,
      });
    } catch (err) {
      setStatusMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Training request failed.',
      });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-6">
        <div className="max-w-xl">
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">
            Dataset Collection &amp; ML Evaluation Studio
          </h1>
          <p className="text-sm sm:text-base text-slate-600 mt-1">
            Collect labeled 21-point landmarks, export CSV datasets, and train/benchmark the Random
            Forest classifier with session-wise splits.
          </p>
        </div>

        <div className="flex flex-wrap lg:flex-nowrap items-center gap-2.5 shrink-0">
          <a
            href={`${apiBaseUrl}/api/dataset/export`}
            download="isl_landmarks.csv"
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-800 bg-white border border-slate-300 hover:bg-slate-50 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </a>
          <button
            type="button"
            onClick={handleResetCalibration}
            disabled={isTraining}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 disabled:opacity-50 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset Baseline</span>
          </button>
          <button
            type="button"
            onClick={handleTrainModel}
            disabled={isTraining}
            className="inline-flex items-center gap-2 px-5 py-2 text-xs font-semibold text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            <span>{isTraining ? 'Training...' : 'Train & Evaluate Model'}</span>
          </button>
        </div>
      </div>

      {isTraining && (
        <LoadingIndicator
          variant="banner"
          label="Training Random Forest classifier & evaluating session-wise split..."
          sublabel="Updating CSV model weights, precision/recall metrics, and confusion matrix."
        />
      )}

      {statusMessage && !isTraining && (
        <div
          role="status"
          className={`p-4 rounded-xl border flex items-center gap-3 text-sm ${
            statusMessage.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
              : statusMessage.type === 'error'
              ? 'bg-rose-50 border-rose-200 text-rose-900'
              : 'bg-slate-100 border-slate-200 text-slate-800'
          }`}
        >
          {statusMessage.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-700" />
          ) : (
            <AlertCircle className="w-4 h-4 shrink-0" />
          )}
          <span>{statusMessage.text}</span>
        </div>
      )}

      {/* Section 1: Custom Webcam Collector + Coverage Matrix */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-lg font-semibold text-slate-900">
              Custom Landmark Collector
            </h2>
            <span className="font-mono text-xs text-slate-500">
              {detectedCount} hand{detectedCount === 1 ? '' : 's'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="collect-letter" className="block text-xs font-medium text-slate-700 mb-1">
                Target Alphabet
              </label>
              <select
                id="collect-letter"
                value={selectedLetter}
                onChange={(e) => setSelectedLetter(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              >
                {STATIC_ISL_LETTERS.map((l) => (
                  <option key={l} value={l}>
                    Letter {l}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="session-id" className="block text-xs font-medium text-slate-700 mb-1">
                Session ID (Split Key)
              </label>
              <input
                id="session-id"
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="e.g. user1-daylight"
                className="w-full px-3 py-2 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              />
            </div>
          </div>

          <div className="relative aspect-[4/3] bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover -scale-x-100 ${camActive ? 'block' : 'hidden'}`}
            />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            {isCamStarting && (
              <LoadingIndicator
                variant="overlay"
                tone="dark"
                label="Starting webcam & MediaPipe..."
                sublabel="Preparing 21-landmark hand tracker."
              />
            )}
            {!camActive && !isCamStarting && (
              <p className="text-xs text-slate-400 text-center px-4">
                Start collector webcam to record live hand landmarks, or capture a simulated sample.
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {!camActive ? (
              <button
                type="button"
                onClick={startCollectorCamera}
                disabled={isCamStarting}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 disabled:opacity-60 rounded-lg cursor-pointer"
              >
                <Camera className="w-3.5 h-3.5" />
                <span>{isCamStarting ? 'Starting...' : 'Open Webcam'}</span>
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={stopCollectorCamera}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 rounded-lg cursor-pointer"
                >
                  <CameraOff className="w-3.5 h-3.5" />
                  <span>Stop</span>
                </button>
                <button
                  type="button"
                  onClick={handleCaptureLive}
                  disabled={isCollecting}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-white bg-blue-700 hover:bg-blue-800 disabled:opacity-60 rounded-lg cursor-pointer"
                >
                  {isCollecting ? (
                    <LoadingIndicator variant="inline" tone="dark" size="sm" label="Saving..." />
                  ) : (
                    <span>Capture Live Frame</span>
                  )}
                </button>
              </>
            )}

            <button
              type="button"
              onClick={handleCaptureSimulated}
              disabled={isCollecting}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 disabled:opacity-60 rounded-lg cursor-pointer ml-auto"
            >
              {isCollecting ? (
                <LoadingIndicator variant="inline" size="sm" label="Saving..." />
              ) : (
                <span>Capture Test Sample ({selectedLetter})</span>
              )}
            </button>
          </div>
        </div>

        {/* Coverage Matrix + Recent Captures */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 space-y-5">
          {isSummaryLoading && !summary ? (
            <LoadingIndicator
              variant="card"
              label="Loading dataset records & class coverage..."
              sublabel="Fetching CSV sample counts across all 26 ISL alphabet classes."
              className="border-0"
            />
          ) : (
            <>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div>
                  <h2 className="font-display text-lg font-semibold text-slate-900">
                    Class Sample Coverage (A–Z)
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Total: <strong className="font-mono tabular-nums">{summary?.totalSamples ?? 0}</strong>{' '}
                    samples across{' '}
                    <strong className="font-mono tabular-nums">{summary?.uniqueSessions ?? 0}</strong>{' '}
                    sessions ({summary?.userWebcamSamples ?? 0} user-recorded ·{' '}
                    {summary?.syntheticCalibrationSamples ?? 0} calibration)
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-6 sm:grid-cols-9 md:grid-cols-13 gap-2">
                {ISL_ALPHABET.map((item) => {
                  const counts = summary?.perClassCounts[item.letter] ?? {
                    total: 0,
                    userWebcam: 0,
                    syntheticCalibration: 0,
                  };
                  const isDynamic = item.modality === 'dynamic-motion';
                  const isSelected = selectedLetter === item.letter;
                  return (
                    <button
                      key={item.letter}
                      type="button"
                      disabled={isDynamic}
                      onClick={() => !isDynamic && setSelectedLetter(item.letter)}
                      title={
                        isDynamic
                          ? `${item.letter} uses temporal dynamic motion tracking`
                          : `${item.letter}: ${counts.total} samples (${counts.userWebcam} webcam)`
                      }
                      className={`p-2 rounded-lg border text-center transition-colors ${
                        isDynamic
                          ? 'bg-amber-50 border-amber-200 text-amber-800 cursor-default'
                          : isSelected
                          ? 'bg-blue-700 border-blue-700 text-white cursor-pointer'
                          : counts.total > 0
                          ? 'bg-slate-50 border-slate-200 text-slate-900 hover:border-blue-400 cursor-pointer'
                          : 'bg-white border-dashed border-slate-300 text-slate-400 cursor-pointer'
                      }`}
                    >
                      <div className="font-display text-sm font-bold">{item.letter}</div>
                      <div className="font-mono text-[11px] tabular-nums opacity-85">
                        {isDynamic ? 'DYN' : counts.total}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Recent Collected Samples */}
              <div className="pt-3 border-t border-slate-100">
                <h3 className="text-xs font-semibold text-slate-600 mb-2">
                  Recent Dataset Records
                </h3>
                <div className="max-h-48 overflow-y-auto border border-slate-200 rounded-lg divide-y divide-slate-100">
                  {(summary?.recentSamples ?? []).slice(0, 8).map((s) => (
                    <div
                      key={s.id}
                      className="px-3 py-2 flex items-center justify-between gap-3 text-xs"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <span className="font-display font-bold text-slate-900 w-5">{s.label}</span>
                        <span className="font-mono text-slate-600 truncate">{s.sessionId}</span>
                        <span className="text-slate-500 whitespace-nowrap">
                          {s.handCount}H · {s.source === 'user_webcam' ? 'User' : 'Calibration'}
                        </span>
                      </div>
                      <button
                        type="button"
                        disabled={deletingId === s.id}
                        onClick={() => handleDeleteSample(s.id)}
                        aria-label={`Delete sample ${s.id}`}
                        className="p-1 text-slate-400 hover:text-rose-700 disabled:opacity-60 cursor-pointer"
                      >
                        {deletingId === s.id ? (
                          <LoadingIndicator variant="inline" size="sm" label="" />
                        ) : (
                          <Trash2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      {/* Section 2: Model Evaluation & Comparison */}
      {isReportLoading && !report ? (
        <LoadingIndicator
          variant="card"
          label="Loading model evaluation report & confusion matrix..."
          sublabel="Reading Random Forest and baseline metrics from /api/evaluate."
        />
      ) : report ? (
        <section className="space-y-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div>
                <h2 className="font-display text-xl font-bold text-slate-900">
                  Model Evaluation Report
                </h2>
                <p className="text-xs text-slate-600 mt-1">{report.honestyNote}</p>
              </div>
              <div className="text-xs font-mono text-slate-500 tabular-nums shrink-0">
                Split: {report.splitStrategy} ({report.trainSamples} train / {report.validationSamples}{' '}
                val / {report.testSamples} test)
              </div>
            </div>

            {/* Key Metrics Row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                <p className="text-xs text-slate-500">Test Accuracy</p>
                <p className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
                  {(report.accuracy * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                <p className="text-xs text-slate-500">Macro Precision</p>
                <p className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
                  {(report.macroPrecision * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                <p className="text-xs text-slate-500">Macro Recall</p>
                <p className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
                  {(report.macroRecall * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                <p className="text-xs text-slate-500">Macro F1-Score</p>
                <p className="font-mono text-2xl font-bold text-slate-900 mt-1 tabular-nums">
                  {(report.macroF1 * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Model Comparison Table */}
            <div>
              <h3 className="text-sm font-semibold text-slate-900 mb-3">
                Model Architecture Comparison (Held-Out Test Split)
              </h3>
              <div className="overflow-x-auto border border-slate-200 rounded-lg">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600">
                    <tr>
                      <th className="py-2.5 px-4 font-semibold">Model</th>
                      <th className="py-2.5 px-4 font-semibold text-right">Accuracy</th>
                      <th className="py-2.5 px-4 font-semibold text-right">Precision</th>
                      <th className="py-2.5 px-4 font-semibold text-right">Recall</th>
                      <th className="py-2.5 px-4 font-semibold text-right">F1-Score</th>
                      <th className="py-2.5 px-4 font-semibold text-right">Avg Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {report.comparison.map((row) => (
                      <tr key={row.modelName} className="hover:bg-slate-50">
                        <td className="py-2.5 px-4 font-medium text-slate-900">{row.modelName}</td>
                        <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                          {(row.accuracy * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                          {(row.macroPrecision * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                          {(row.macroRecall * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                          {(row.macroF1 * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                          {row.inferenceLatencyMs} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrix Heatmap */}
            <div>
              <h3 className="text-sm font-semibold text-slate-900 mb-2">
                Confusion Matrix ({report.classesEvaluated.length} Evaluated Static Classes)
              </h3>
              <p className="text-xs text-slate-500 mb-3">
                Rows represent true alphabet labels; columns represent predicted labels on the test split.
              </p>
              <div className="overflow-x-auto border border-slate-200 rounded-lg p-3 bg-slate-50">
                <table className="text-[10px] font-mono tabular-nums border-collapse">
                  <thead>
                    <tr>
                      <th className="p-1 text-slate-400">T\P</th>
                      {report.classesEvaluated.map((c) => (
                        <th key={c} className="p-1 w-6 text-center text-slate-700 font-semibold">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {report.confusionMatrix.map((row, rIdx) => (
                      <tr key={report.classesEvaluated[rIdx]}>
                        <td className="p-1 font-semibold text-slate-700">
                          {report.classesEvaluated[rIdx]}
                        </td>
                        {row.map((val, cIdx) => {
                          const isDiag = rIdx === cIdx;
                          const cellStyle =
                            val === 0
                              ? 'bg-white text-slate-300'
                              : isDiag
                              ? 'bg-blue-700 text-white font-bold'
                              : 'bg-rose-100 text-rose-800 font-semibold';
                          return (
                            <td
                              key={cIdx}
                              className={`p-1 w-6 h-6 text-center border border-slate-200 ${cellStyle}`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {/* Backend Endpoint Configuration */}
      <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            REST API Runtime Endpoint
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Default uses the integrated Express ML Engine on this server. Point to an external Python
            Flask deployment URL (Cloud Run / Render) if running Python separately.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            value={customUrlInput}
            onChange={(e) => setCustomUrlInput(e.target.value)}
            aria-label="REST API Base URL"
            placeholder="Same-origin (leave empty)"
            className="px-3 py-2 text-xs font-mono bg-white border border-slate-300 rounded-lg w-64"
          />
          <button
            type="button"
            onClick={() => onApiBaseUrlChange(customUrlInput.trim())}
            className="px-3.5 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg cursor-pointer whitespace-nowrap"
          >
            Apply URL
          </button>
          {apiBaseUrl !== '' && (
            <button
              type="button"
              onClick={() => {
                setCustomUrlInput('');
                onApiBaseUrlChange('');
              }}
              className="px-3 py-2 text-xs font-medium text-slate-600 hover:text-slate-900 cursor-pointer"
            >
              Reset
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
