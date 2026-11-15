export interface Landmark3D {
  x: number;
  y: number;
  z: number;
}

export type HandednessLabel = 'Left' | 'Right' | 'Unknown';

export interface DetectedHandInput {
  landmarks: Landmark3D[];
  handedness?: HandednessLabel;
  score?: number;
}

export type SignModality = 'two-handed-static' | 'single-handed-static' | 'dynamic-motion';

export type SampleSource = 'user_webcam' | 'synthetic_calibration';

export interface DatasetSample {
  id: string;
  label: string;
  sessionId: string;
  handCount: 1 | 2;
  source: SampleSource;
  features: number[]; // 129 features: 63 (hand1) + 63 (hand2) + 3 (inter-hand wrist offset)
  createdAt: string;
}

export interface PredictionCandidate {
  label: string;
  confidence: number;
}

export interface PredictionResponse {
  prediction: string | null;
  confidence: number;
  status: 'detected' | 'uncertain' | 'no_hand' | 'error';
  handCount: number;
  isDynamic: boolean;
  topCandidates: PredictionCandidate[];
  latencyMs: number;
  message: string;
  modelSourceNote: string;
}

export interface DynamicPredictionResponse {
  prediction: 'J' | 'Z' | null;
  confidence: number;
  status: 'detected' | 'uncertain' | 'insufficient_motion';
  pathLength: number;
  latencyMs: number;
  message: string;
}

export interface ClassMetrics {
  label: string;
  support: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface ModelComparisonMetrics {
  modelName: string;
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  inferenceLatencyMs: number;
}

export interface EvaluationReport {
  trainedAt: string;
  totalSamples: number;
  trainSamples: number;
  validationSamples: number;
  testSamples: number;
  classesEvaluated: string[];
  userWebcamSamples: number;
  syntheticCalibrationSamples: number;
  splitStrategy: 'session-wise' | 'stratified-fallback';
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  perClassMetrics: ClassMetrics[];
  confusionMatrix: number[][];
  comparison: ModelComparisonMetrics[];
  honestyNote: string;
}

export interface DatasetSummary {
  totalSamples: number;
  userWebcamSamples: number;
  syntheticCalibrationSamples: number;
  uniqueSessions: number;
  classesCovered: number;
  perClassCounts: Record<
    string,
    { total: number; userWebcam: number; syntheticCalibration: number }
  >;
  recentSamples: Array<{
    id: string;
    label: string;
    sessionId: string;
    handCount: 1 | 2;
    source: SampleSource;
    createdAt: string;
  }>;
}

export interface HistoryEntry {
  id: string;
  letter: string;
  confidence: number;
  modality: SignModality;
  handCount: number;
  timestamp: string;
}

export interface ISLLetterGuide {
  letter: string;
  modality: SignModality;
  handsRequired: 1 | 2;
  dominantHand: string;
  baseHand: string;
  formationSummary: string;
  keyLandmarks: string;
  islVsAslNote: string;
  svgPose: {
    baseHandShape:
      | 'none'
      | 'flat-palm'
      | 'open-spread'
      | 'index-up'
      | 'circle-o'
      | 'curve-c'
      | 'fist';
    activeHandShape:
      | 'thumb-touch'
      | 'arch-loop'
      | 'curve-c'
      | 'index-up'
      | 'tip-touch'
      | 'cross-bar'
      | 'hook-g'
      | 'chop-flat'
      | 'index-point'
      | 'trace-j'
      | 'v-touch'
      | 'l-shape'
      | 'three-fingers'
      | 'two-fingers'
      | 'circle-o'
      | 'loop-p'
      | 'pinch-q'
      | 'hook-r'
      | 'snake-s'
      | 't-cross'
      | 'u-two'
      | 'v-sign'
      | 'interlock-w'
      | 'hook-x'
      | 'yoke-y'
      | 'trace-z';
    contactHint: string;
  };
}
