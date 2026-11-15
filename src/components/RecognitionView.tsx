import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Camera,
  CameraOff,
  CheckCircle2,
  Hand,
  HelpCircle,
  RotateCcw,
  Space,
  Trash2,
  Delete,
  Sparkles,
  Activity,
} from 'lucide-react';
import {
  generateAmbiguousHands,
  generateSimulatedDynamicTrajectory,
  generateSimulatedHandsForLetter,
} from '../ml/calibrationSeed';
import { ISL_ALPHABET, ISL_ALPHABET_MAP } from '../ml/islAlphabet';
import {
  PredictionSmoother,
  extractFeatureVector,
  type TrajectoryPoint,
} from '../ml/landmarks';
import type {
  DetectedHandInput,
  DynamicPredictionResponse,
  HistoryEntry,
  PredictionResponse,
} from '../ml/types';
import { averageMs, computeRollingFps } from '../utils/diagnostics';
import { drawHandLandmarks } from '../utils/handCanvas';
import { ISL_SIGN_IMAGES } from '../utils/islSignImages';
import { getHandLandmarker, mapMediaPipeResult } from '../utils/mediapipeClient';
import { DiagnosticOverlay } from './DiagnosticOverlay';
import { HandPoseDiagram } from './HandPoseDiagram';
import { LoadingIndicator } from './LoadingIndicator';

interface RecognitionViewProps {
  apiBaseUrl: string;
  practiceTarget: string | null;
  onClearPracticeTarget: () => void;
}

type RecognitionMode = 'static' | 'dynamic';

export function RecognitionView({
  apiBaseUrl,
  practiceTarget,
  onClearPracticeTarget,
}: RecognitionViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const smootherRef = useRef(new PredictionSmoother(6, 3, 0.42));
  const lastApiCallRef = useRef<number>(0);
  const inFlightRef = useRef<boolean>(false);
  const trajectoryRef = useRef<TrajectoryPoint[]>([]);
  const lastCommittedRef = useRef<{ letter: string; at: number } | null>(null);
  const frameTimestampsRef = useRef<number[]>([]);
  const detectSamplesRef = useRef<number[]>([]);
  const lastDiagUpdateRef = useRef<number>(0);

  const [cameraState, setCameraState] = useState<'idle' | 'starting' | 'active' | 'error'>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [mediapipeReady, setMediapipeReady] = useState<boolean>(false);
  const [mode, setMode] = useState<RecognitionMode>(
    practiceTarget === 'J' || practiceTarget === 'Z' ? 'dynamic' : 'static'
  );
  const [handCount, setHandCount] = useState<number>(0);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [dynamicResult, setDynamicResult] = useState<DynamicPredictionResponse | null>(null);
  const [stableLetter, setStableLetter] = useState<string | null>(null);
  const [composedText, setComposedText] = useState<string>('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [autoAppend, setAutoAppend] = useState<boolean>(true);
  const [simulatedLabel, setSimulatedLabel] = useState<string | null>(null);
  const [previewLetter, setPreviewLetter] = useState<string>(practiceTarget || 'A');
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(true);
  const [fps, setFps] = useState<number | null>(null);
  const [frameDetectMs, setFrameDetectMs] = useState<number | null>(null);
  const [roundTripLatencyMs, setRoundTripLatencyMs] = useState<number | null>(null);

  useEffect(() => {
    if (practiceTarget === 'J' || practiceTarget === 'Z') {
      setMode('dynamic');
      setPreviewLetter(practiceTarget);
    } else if (practiceTarget) {
      setMode('static');
      setPreviewLetter(practiceTarget);
    }
  }, [practiceTarget]);

  const appendHistory = useCallback(
    (letter: string, confidence: number, hands: number, isDynamic: boolean) => {
      const now = Date.now();
      // Avoid flooding history with identical consecutive letter within 1.5s
      if (
        lastCommittedRef.current &&
        lastCommittedRef.current.letter === letter &&
        now - lastCommittedRef.current.at < 1500
      ) {
        return;
      }
      lastCommittedRef.current = { letter, at: now };

      const entry: HistoryEntry = {
        id: `hist-${now}-${Math.random().toString(36).slice(2, 6)}`,
        letter,
        confidence,
        modality: isDynamic
          ? 'dynamic-motion'
          : hands === 2
          ? 'two-handed-static'
          : 'single-handed-static',
        handCount: hands,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
      };

      setHistory((prev) => [entry, ...prev].slice(0, 30));
      if (autoAppend) {
        setComposedText((prev) => prev + letter);
      }
    },
    [autoAppend]
  );

  const sendStaticFeatures = useCallback(
    async (hands: DetectedHandInput[], forceCommit = false) => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      if (forceCommit) setIsPredicting(true);
      try {
        const { features, handCount: count } = extractFeatureVector(hands);
        const reqStartedAt = performance.now();
        const res = await fetch(`${apiBaseUrl}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features, handCount: count }),
        });
        const elapsedMs = Math.round((performance.now() - reqStartedAt) * 10) / 10;
        setRoundTripLatencyMs(elapsedMs);

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          setPrediction({
            prediction: null,
            confidence: 0,
            status: 'error',
            handCount: count,
            isDynamic: false,
            topCandidates: [],
            latencyMs: 0,
            message: errData.message || `API returned HTTP ${res.status}`,
            modelSourceNote: '',
          });
          return;
        }

        const data = (await res.json()) as PredictionResponse;
        setPrediction(data);

        if (forceCommit) {
          smootherRef.current.reset();
          setStableLetter(data.prediction);
          if (data.status === 'detected' && data.prediction) {
            appendHistory(data.prediction, data.confidence, data.handCount, false);
          }
        } else {
          const smoothed = smootherRef.current.push(data.prediction, data.confidence);
          if (smoothed.isStable && smoothed.stableLabel) {
            setStableLetter(smoothed.stableLabel);
            appendHistory(smoothed.stableLabel, smoothed.smoothedConfidence, data.handCount, false);
          } else if (data.status === 'uncertain') {
            setStableLetter(null);
          }
        }
      } catch (err) {
        setPrediction({
          prediction: null,
          confidence: 0,
          status: 'error',
          handCount: 0,
          isDynamic: false,
          topCandidates: [],
          latencyMs: 0,
          message:
            err instanceof Error ? `Backend unreachable: ${err.message}` : 'Backend connection error.',
          modelSourceNote: '',
        });
      } finally {
        inFlightRef.current = false;
        if (forceCommit) setIsPredicting(false);
      }
    },
    [apiBaseUrl, appendHistory]
  );

  const sendDynamicTrajectory = useCallback(
    async (points: TrajectoryPoint[]) => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      setIsPredicting(true);
      try {
        const reqStartedAt = performance.now();
        const res = await fetch(`${apiBaseUrl}/api/predict-dynamic`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frames: points }),
        });
        const elapsedMs = Math.round((performance.now() - reqStartedAt) * 10) / 10;
        setRoundTripLatencyMs(elapsedMs);
        if (!res.ok) return;
        const data = (await res.json()) as DynamicPredictionResponse;
        setDynamicResult(data);
        if (data.status === 'detected' && data.prediction) {
          setStableLetter(data.prediction);
          appendHistory(data.prediction, data.confidence, 1, true);
        }
      } catch {
        // Handled cleanly in UI state
      } finally {
        inFlightRef.current = false;
        setIsPredicting(false);
      }
    },
    [apiBaseUrl, appendHistory]
  );

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    smootherRef.current.reset();
    trajectoryRef.current = [];
    frameTimestampsRef.current = [];
    detectSamplesRef.current = [];
    lastDiagUpdateRef.current = 0;
    setFps(null);
    setFrameDetectMs(null);
    setHandCount(0);
    setCameraState('idle');
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setSimulatedLabel(null);
    setCameraState('starting');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraState('error');
      setCameraError(
        'Webcam API is unavailable in this browser context. Use HTTPS or the simulated gesture tester below.'
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 } },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const landmarker = await getHandLandmarker();
      setMediapipeReady(true);
      setCameraState('active');

      const processFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || !streamRef.current) return;

        if (video.readyState >= 2) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
          }

          const detectStart = performance.now();
          const result = landmarker.detectForVideo(video, detectStart);
          const detectEnd = performance.now();

          frameTimestampsRef.current.push(detectEnd);
          if (frameTimestampsRef.current.length > 90) {
            frameTimestampsRef.current.shift();
          }
          detectSamplesRef.current.push(detectEnd - detectStart);
          if (detectSamplesRef.current.length > 24) {
            detectSamplesRef.current.shift();
          }
          if (detectEnd - lastDiagUpdateRef.current >= 400) {
            lastDiagUpdateRef.current = detectEnd;
            setFps(computeRollingFps(frameTimestampsRef.current, detectEnd, 1000));
            setFrameDetectMs(averageMs(detectSamplesRef.current));
          }

          const hands = mapMediaPipeResult(result);
          setHandCount(hands.length);

          const ctx = canvas.getContext('2d');
          if (hands.length > 0) {
            if (mode === 'dynamic') {
              const tip = hands[0].landmarks[8];
              const now = Date.now();
              trajectoryRef.current.push({ x: tip.x, y: tip.y, t: now });
              if (trajectoryRef.current.length > 36) {
                trajectoryRef.current.shift();
              }
            }
            if (ctx) {
              drawHandLandmarks(
                ctx,
                canvas.width,
                canvas.height,
                hands,
                mode === 'dynamic' ? trajectoryRef.current : [],
                true
              );
            }

            const now = performance.now();
            if (now - lastApiCallRef.current > 150) {
              lastApiCallRef.current = now;
              if (mode === 'static') {
                void sendStaticFeatures(hands, false);
              } else if (trajectoryRef.current.length >= 12) {
                void sendDynamicTrajectory(trajectoryRef.current);
              }
            }
          } else {
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            smootherRef.current.reset();
            setStableLetter(null);
            if (mode === 'dynamic' && trajectoryRef.current.length > 0) {
              trajectoryRef.current = [];
            }
          }
        }

        rafRef.current = requestAnimationFrame(processFrame);
      };

      rafRef.current = requestAnimationFrame(processFrame);
    } catch (err) {
      stopCamera();
      setCameraState('error');
      const msg =
        err instanceof Error
          ? err.message
          : 'Camera permission was denied or no webcam device was found.';
      setCameraError(
        `${msg} — You can still test the full ML pipeline using the Simulated Hand Tester below.`
      );
    }
  }, [mode, sendDynamicTrajectory, sendStaticFeatures, stopCamera]);

  const restartCamera = useCallback(async () => {
    stopCamera();
    await startCamera();
  }, [startCamera, stopCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Simulate real MediaPipe 21-landmark hand detection on canvas & dispatch through API
  const runSimulatedGesture = async (target: string) => {
    if (cameraState === 'active') stopCamera();
    setCameraError(null);
    setSimulatedLabel(target);
    if (target !== 'UNCERTAIN') {
      setPreviewLetter(target);
    }

    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = 640;
      canvas.height = 420;
    }
    const ctx = canvas?.getContext('2d');

    if (target === 'J' || target === 'Z') {
      setMode('dynamic');
      setPrediction(null);
      const traj = generateSimulatedDynamicTrajectory(target);
      const hands = generateSimulatedHandsForLetter('I', 42, 0.002);
      setHandCount(1);
      if (ctx && canvas) {
        drawHandLandmarks(ctx, canvas.width, canvas.height, hands, traj, false);
      }
      await sendDynamicTrajectory(traj);
      return;
    }

    setMode('static');
    setDynamicResult(null);
    if (target === 'UNCERTAIN') {
      const amb = generateAmbiguousHands(Math.floor(Date.now() % 1000));
      setHandCount(amb.length);
      if (ctx && canvas) {
        drawHandLandmarks(ctx, canvas.width, canvas.height, amb, [], false);
      }
      await sendStaticFeatures(amb, true);
      return;
    }

    const hands = generateSimulatedHandsForLetter(target, 700 + target.charCodeAt(0), 0.003);
    setHandCount(hands.length);
    if (ctx && canvas) {
      drawHandLandmarks(ctx, canvas.width, canvas.height, hands, [], false);
    }
    await sendStaticFeatures(hands, true);
  };

  const activeLetter = mode === 'dynamic' ? dynamicResult?.prediction : stableLetter ?? prediction?.prediction;
  const activeConfidence =
    mode === 'dynamic' ? dynamicResult?.confidence ?? 0 : prediction?.confidence ?? 0;
  const activeStatus =
    handCount === 0
      ? 'no_hand'
      : mode === 'dynamic'
      ? dynamicResult?.status ?? 'uncertain'
      : prediction?.status ?? 'uncertain';

  const practiceGuide = practiceTarget ? ISL_ALPHABET_MAP[practiceTarget] : null;
  const practiceMatched = Boolean(practiceTarget && activeLetter === practiceTarget);
  const displayLetter =
    activeStatus === 'detected' && activeLetter
      ? activeLetter
      : practiceGuide?.letter || previewLetter;
  const displayGuide = ISL_ALPHABET_MAP[displayLetter] ?? ISL_ALPHABET[0];
  const displayLatency =
    mode === 'dynamic' ? dynamicResult?.latencyMs : prediction?.latencyMs;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Page Header & Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">
            Live ISL Recognition Studio
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Hold an authentic ISL gesture steadily in frame. Switch to Dynamic Mode to trace J or Z.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg self-start sm:self-auto">
          <button
            type="button"
            onClick={() => {
              setMode('static');
              setDynamicResult(null);
              trajectoryRef.current = [];
            }}
            className={`px-3.5 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              mode === 'static'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Static Alphabet (24)
          </button>
          <button
            type="button"
            onClick={() => {
              setMode('dynamic');
              setPrediction(null);
              smootherRef.current.reset();
            }}
            className={`px-3.5 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              mode === 'dynamic'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Dynamic Motion (J, Z)
          </button>
        </div>
      </div>

      {/* Practice Mode Target Banner (when triggered from Learn ISL) */}
      {practiceGuide && (
        <div className="bg-white border border-slate-200 rounded-xl p-4 sm:p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-28 shrink-0">
              <HandPoseDiagram guide={practiceGuide} size="sm" />
            </div>
            <div>
              <p className="text-xs font-medium text-blue-700">
                Practice Mode Active · Target Letter {practiceGuide.letter}
              </p>
              <p className="text-sm font-semibold text-slate-900 mt-0.5">
                {practiceGuide.formationSummary}
              </p>
              <p className="text-xs text-slate-600 mt-1">{practiceGuide.dominantHand}</p>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {practiceMatched ? (
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-700">
                <CheckCircle2 className="w-4 h-4" />
                Matched {practiceGuide.letter}!
              </span>
            ) : (
              <span className="text-xs font-medium text-slate-500">
                Waiting for {practiceGuide.letter}
              </span>
            )}
            <button
              type="button"
              onClick={onClearPracticeTarget}
              className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg cursor-pointer"
            >
              Exit Practice
            </button>
          </div>
        </div>
      )}

      {/* Main Two-Zone Recognition Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Zone (65%): Dark Camera & 21-Landmark Canvas Stage */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
            {/* Viewport Header Bar */}
            <div className="px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                {handCount > 0 ? (
                  <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400">
                    <Hand className="w-3.5 h-3.5" />
                    <span>
                      HAND DETECTED ({handCount} {handCount === 1 ? 'hand' : 'hands'} ·{' '}
                      {handCount * 21} landmarks)
                    </span>
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400">
                    <Hand className="w-3.5 h-3.5" />
                    <span>
                      {cameraState === 'active'
                        ? 'NO HAND IN FRAME'
                        : simulatedLabel
                        ? 'SIMULATED FRAME'
                        : 'CAMERA STANDBY'}
                    </span>
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowDiagnostics((prev) => !prev)}
                  aria-pressed={showDiagnostics}
                  title="Toggle FPS & prediction latency diagnostic overlay"
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    showDiagnostics
                      ? 'bg-slate-800 text-slate-100 border-slate-700 hover:bg-slate-700'
                      : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <Activity className="w-3.5 h-3.5 text-blue-400" aria-hidden="true" />
                  <span>Diagnostics</span>
                </button>

                {cameraState !== 'active' ? (
                  <button
                    type="button"
                    onClick={startCamera}
                    disabled={cameraState === 'starting'}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Camera className="w-3.5 h-3.5" />
                    <span>{cameraState === 'starting' ? 'Starting...' : 'Start Camera'}</span>
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={restartCamera}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Restart</span>
                    </button>
                    <button
                      type="button"
                      onClick={stopCamera}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <CameraOff className="w-3.5 h-3.5" />
                      <span>Stop Camera</span>
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Video + Overlay Canvas */}
            <div className="relative min-h-[340px] sm:min-h-0 sm:aspect-[16/10] w-full bg-slate-900 flex items-center justify-center">
              <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full h-full object-cover -scale-x-100 ${
                  cameraState === 'active' ? 'block' : 'hidden'
                }`}
              />
              <canvas
                ref={canvasRef}
                data-testid="landmark-canvas"
                className="absolute inset-0 w-full h-full pointer-events-none"
              />

              {showDiagnostics && cameraState !== 'starting' && (
                <DiagnosticOverlay
                  cameraActive={cameraState === 'active'}
                  simulatedActive={Boolean(simulatedLabel)}
                  fps={fps}
                  frameDetectMs={frameDetectMs}
                  roundTripLatencyMs={roundTripLatencyMs}
                  modelLatencyMs={displayLatency ?? null}
                />
              )}

              {cameraState === 'starting' && (
                <LoadingIndicator
                  variant="overlay"
                  tone="dark"
                  label="Loading MediaPipe Hand Landmarker & Camera..."
                  sublabel="Initializing browser WASM runtime and 21-point dual-hand detector."
                />
              )}

              {cameraState !== 'active' && cameraState !== 'starting' && !simulatedLabel && (
                <div
                  className={`text-center px-6 max-w-md space-y-3 ${
                    showDiagnostics ? 'pt-36 pb-8 sm:py-8' : 'py-8'
                  }`}
                >
                  <Camera className="w-10 h-10 text-slate-500 mx-auto" />
                  <p className="text-sm font-semibold text-slate-200">
                    {cameraState === 'error'
                      ? 'Webcam Unavailable'
                      : 'Ready for Live Hand Detection'}
                  </p>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {cameraError ||
                      'Click Start Camera to grant webcam permission and track 21 landmarks per hand via MediaPipe, or use the Simulated Hand Tester below.'}
                  </p>
                </div>
              )}

              {simulatedLabel && cameraState !== 'active' && (
                <div className="absolute bottom-2.5 left-2.5 sm:top-3 sm:bottom-auto sm:left-3 px-2.5 py-1 rounded bg-slate-900/85 border border-slate-700 text-[11px] font-mono text-slate-200">
                  Simulated Landmark Input: {simulatedLabel}
                </div>
              )}
            </div>

            {/* Viewport Instructions Footer */}
            <div className="px-4 py-3 bg-slate-900/70 border-t border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
              <span>
                {mediapipeReady
                  ? 'MediaPipe HandLandmarker loaded (max 2 hands)'
                  : 'Keep both hands centered at chest height in even lighting'}
              </span>
              {(mode === 'dynamic' ? dynamicResult : prediction) && (
                <span className="font-mono tabular-nums text-slate-300">
                  API Latency: {(mode === 'dynamic' ? dynamicResult?.latencyMs : prediction?.latencyMs) ?? 0} ms
                </span>
              )}
            </div>
          </div>

          {/* Offline / No-Camera Pipeline Tester */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">
                  Test Pipeline Without Physical Camera
                </h2>
                <p className="text-xs text-slate-500">
                  Injects calibrated 21-point landmarks into the canvas, normalizes features, and
                  calls the live REST API.
                </p>
              </div>
              {simulatedLabel && (
                <button
                  type="button"
                  onClick={() => {
                    setSimulatedLabel(null);
                    setHandCount(0);
                    setPrediction(null);
                    setDynamicResult(null);
                    setStableLetter(null);
                    setRoundTripLatencyMs(null);
                    const ctx = canvasRef.current?.getContext('2d');
                    if (ctx && canvasRef.current) {
                      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                    }
                  }}
                  className="text-xs font-medium text-slate-600 hover:text-slate-900 underline whitespace-nowrap shrink-0 cursor-pointer"
                >
                  Clear Simulation
                </button>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              {['A', 'B', 'C', 'M', 'N', 'V', 'W', 'Y'].map((letter) => (
                <button
                  key={letter}
                  type="button"
                  onClick={() => runSimulatedGesture(letter)}
                  className={`px-3 py-1.5 text-xs font-mono font-semibold rounded-lg border transition-colors cursor-pointer ${
                    simulatedLabel === letter
                      ? 'bg-blue-700 text-white border-blue-700'
                      : 'bg-slate-50 text-slate-800 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  Sign {letter}
                </button>
              ))}
              {(['J', 'Z'] as const).map((dyn) => (
                <button
                  key={dyn}
                  type="button"
                  onClick={() => runSimulatedGesture(dyn)}
                  className={`px-3 py-1.5 text-xs font-mono font-semibold rounded-lg border transition-colors cursor-pointer ${
                    simulatedLabel === dyn
                      ? 'bg-amber-600 text-white border-amber-600'
                      : 'bg-amber-50 text-amber-900 border-amber-200 hover:bg-amber-100'
                  }`}
                >
                  Dynamic {dyn}
                </button>
              ))}
              <button
                type="button"
                onClick={() => runSimulatedGesture('UNCERTAIN')}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer ${
                  simulatedLabel === 'UNCERTAIN'
                    ? 'bg-slate-800 text-white border-slate-800'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
              >
                Test Uncertain
              </button>
            </div>
          </div>
        </div>

        {/* Right Zone (35%): Prediction, Word Composer & History */}
        <div className="lg:col-span-5 space-y-5">
          {/* Primary Prediction Readout Card */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            {/* Dark Status Header Bar */}
            <div className="bg-slate-950 text-white px-5 py-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-300 truncate">
                  {mode === 'dynamic'
                    ? 'DYNAMIC STROKE · J / Z'
                    : 'STATIC RF CLASSIFIER · 40 TREES'}
                </span>
                {displayLatency !== undefined && (
                  <span className="text-[11px] font-mono text-slate-400 tabular-nums whitespace-nowrap shrink-0">
                    · {displayLatency} ms
                  </span>
                )}
              </div>

              {isPredicting ? (
                <LoadingIndicator variant="inline" tone="dark" size="sm" label="Classifying..." />
              ) : activeStatus === 'detected' ? (
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-400 whitespace-nowrap shrink-0">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>RECOGNIZED</span>
                </span>
              ) : activeStatus === 'uncertain' ? (
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-400 whitespace-nowrap shrink-0">
                  <HelpCircle className="w-3.5 h-3.5" />
                  <span>LOW CONFIDENCE</span>
                </span>
              ) : activeStatus === 'error' ? (
                <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-400 whitespace-nowrap shrink-0">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>API ERROR</span>
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-300 whitespace-nowrap shrink-0">
                  <Hand className="w-3.5 h-3.5" />
                  <span>{practiceGuide ? `TARGET ${practiceGuide.letter}` : 'READY'}</span>
                </span>
              )}
            </div>

            <div className="p-5 space-y-5">
              {/* Alphabet + Real ISL Reference Photo Stage */}
              <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
                <div className="sm:col-span-6 space-y-3">
                  <div className="flex items-center gap-3.5">
                    <div
                      data-testid="predicted-letter"
                      className={`w-16 h-16 rounded-xl flex items-center justify-center font-display text-4xl font-bold leading-none shrink-0 transition-colors ${
                        activeStatus === 'detected' && activeLetter
                          ? 'bg-blue-700 text-white shadow-sm'
                          : activeStatus === 'uncertain'
                          ? 'bg-amber-100 text-amber-900 border border-amber-300'
                          : 'bg-slate-900 text-white'
                      }`}
                    >
                      {activeStatus === 'detected' && activeLetter
                        ? activeLetter
                        : activeStatus === 'uncertain'
                        ? '?'
                        : displayLetter}
                    </div>

                    <div className="min-w-0">
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-blue-700">
                        {activeStatus === 'detected' && activeLetter
                          ? 'Detected Alphabet'
                          : activeStatus === 'uncertain'
                          ? 'Uncertain Gesture'
                          : practiceGuide
                          ? 'Target Reference'
                          : 'Alphabet Preview'}
                      </p>
                      <h2 className="font-display text-xl font-bold text-slate-900 leading-tight">
                        Letter {displayGuide.letter}
                      </h2>
                      <p className="text-xs text-slate-600 mt-0.5 line-clamp-1">
                        {displayGuide.svgPose.contactHint}
                      </p>
                    </div>
                  </div>

                  {/* Confidence Meter */}
                  <div className="space-y-1.5 pt-1">
                    <div className="flex items-baseline justify-between text-xs">
                      <span className="text-slate-500 font-medium">Model Confidence</span>
                      <span className="font-mono text-base font-bold text-slate-900 tabular-nums">
                        {Math.round(activeConfidence * 100)}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-150 ${
                          activeConfidence >= 0.65
                            ? 'bg-emerald-600'
                            : activeConfidence >= 0.42
                            ? 'bg-amber-500'
                            : 'bg-slate-300'
                        }`}
                        style={{ width: `${Math.round(activeConfidence * 100)}%` }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      {activeStatus === 'detected' && activeLetter
                        ? mode === 'dynamic'
                          ? `Dynamic trajectory (${dynamicResult?.pathLength ?? 0} normalized path)`
                          : `${displayGuide.handsRequired}-hand ISL · threshold 42%`
                        : prediction?.message ||
                          dynamicResult?.message ||
                          'Show gesture in camera or tap any letter A–Z below.'}
                    </p>
                  </div>
                </div>

                {/* Live ISL Reference Photograph */}
                <div className="sm:col-span-6 space-y-1.5">
                  <HandPoseDiagram guide={displayGuide} size="sm" />
                  <p className="text-[11px] text-slate-500 text-center line-clamp-1">
                    {displayGuide.formationSummary}
                  </p>
                </div>
              </div>

              {/* Top-3 Candidates with ISL Thumbnail Photos */}
              {mode === 'static' && prediction && prediction.topCandidates.length > 0 && (
                <div className="pt-4 border-t border-slate-100 space-y-2">
                  <p className="text-xs font-semibold text-slate-600">
                    Top Candidate Probabilities
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {prediction.topCandidates.map((cand) => {
                      const thumbSrc = ISL_SIGN_IMAGES[cand.label];
                      return (
                        <button
                          key={cand.label}
                          type="button"
                          onClick={() => runSimulatedGesture(cand.label)}
                          className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 flex items-center gap-2 text-left transition-colors cursor-pointer"
                        >
                          {thumbSrc && (
                            <img
                              src={thumbSrc}
                              alt={`ISL sign ${cand.label}`}
                              referrerPolicy="no-referrer"
                              className="w-9 h-7 rounded object-cover shrink-0 border border-slate-200"
                            />
                          )}
                          <div className="min-w-0">
                            <div className="font-display text-sm font-bold text-slate-900 leading-none">
                              {cand.label}
                            </div>
                            <div className="font-mono text-[11px] text-slate-600 tabular-nums mt-0.5">
                              {Math.round(cand.confidence * 100)}%
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Interactive A-Z Alphabet Selector to Show & Classify Any Letter */}
              <div className="pt-4 border-t border-slate-100 space-y-2.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-semibold text-slate-700">
                    ISL Alphabet Selector (A–Z)
                  </p>
                  <span className="text-[11px] text-slate-500">Tap to predict &amp; show photo</span>
                </div>
                <div className="grid grid-cols-7 sm:grid-cols-9 gap-1.5">
                  {ISL_ALPHABET.map((item) => {
                    const isCurrent = displayLetter === item.letter;
                    const isDynamic = item.modality === 'dynamic-motion';
                    return (
                      <button
                        key={item.letter}
                        type="button"
                        onClick={() => runSimulatedGesture(item.letter)}
                        title={`Show and test ISL letter ${item.letter} (${
                          isDynamic ? 'Dynamic motion' : `${item.handsRequired}-hand static`
                        })`}
                        className={`py-1.5 rounded-md font-display text-xs font-bold border transition-colors cursor-pointer ${
                          isCurrent
                            ? 'bg-blue-700 text-white border-blue-700 shadow-xs'
                            : isDynamic
                            ? 'bg-amber-50 text-amber-900 border-amber-200 hover:bg-amber-100'
                            : 'bg-slate-50 text-slate-800 border-slate-200 hover:bg-slate-100 hover:border-slate-300'
                        }`}
                      >
                        {item.letter}
                      </button>
                    );
                  })}
                </div>
              </div>

              {prediction?.modelSourceNote && (
                <p className="text-[11px] text-slate-500 pt-2 border-t border-slate-100">
                  {prediction.modelSourceNote}
                </p>
              )}
            </div>
          </div>

          {/* Fingerspelling Word Builder */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-900">Word &amp; Phrase Builder</h2>
              <label className="inline-flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoAppend}
                  onChange={(e) => setAutoAppend(e.target.checked)}
                  className="rounded border-slate-300 text-blue-700 focus:ring-blue-600"
                />
                <span>Auto-append stable letters</span>
              </label>
            </div>

            <div className="min-h-[48px] px-3.5 py-2.5 rounded-lg bg-slate-50 border border-slate-200 font-mono text-base font-medium text-slate-900 tracking-wider break-all">
              {composedText || <span className="text-slate-400 font-sans text-xs">Recognized letters assemble here...</span>}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                disabled={!activeLetter || activeStatus !== 'detected'}
                onClick={() => activeLetter && setComposedText((p) => p + activeLetter)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 disabled:opacity-45 rounded-lg cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Add Letter</span>
              </button>
              <button
                type="button"
                onClick={() => setComposedText((p) => p + ' ')}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-lg cursor-pointer"
              >
                <Space className="w-3.5 h-3.5" />
                <span>Space</span>
              </button>
              <button
                type="button"
                onClick={() => setComposedText((p) => p.slice(0, -1))}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 hover:bg-slate-50 rounded-lg cursor-pointer"
              >
                <Delete className="w-3.5 h-3.5" />
                <span>Backspace</span>
              </button>
              <button
                type="button"
                onClick={() => setComposedText('')}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-rose-700 ml-auto cursor-pointer"
              >
                <span>Clear Text</span>
              </button>
            </div>
          </div>

          {/* Recognition History */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-900">
                Recognition History ({history.length})
              </h2>
              <button
                type="button"
                onClick={() => setHistory([])}
                disabled={history.length === 0}
                className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-rose-700 disabled:opacity-40 cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear History</span>
              </button>
            </div>

            {history.length === 0 ? (
              <p className="text-xs text-slate-500 py-4 text-center border border-dashed border-slate-200 rounded-lg">
                No letters recognized yet. Start your camera or run a simulated gesture test.
              </p>
            ) : (
              <ul
                data-testid="recognition-history"
                className="divide-y divide-slate-100 max-h-60 overflow-y-auto"
              >
                {history.map((item) => (
                  <li
                    key={item.id}
                    className="py-2 flex items-center justify-between gap-3 text-xs"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-display text-base font-bold text-slate-900 w-6">
                        {item.letter}
                      </span>
                      <span className="text-slate-600">
                        {item.modality === 'dynamic-motion'
                          ? 'Dynamic Motion'
                          : `${item.handCount}-Hand Static`}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 font-mono tabular-nums text-slate-500">
                      <span>{Math.round(item.confidence * 100)}%</span>
                      <span>{item.timestamp}</span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
