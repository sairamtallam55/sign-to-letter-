import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import type { DetectedHandInput, HandednessLabel } from '../ml/types';

let cachedLandmarker: HandLandmarker | null = null;
let loadingPromise: Promise<HandLandmarker> | null = null;

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export async function getHandLandmarker(): Promise<HandLandmarker> {
  if (cachedLandmarker) return cachedLandmarker;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    let vision;
    try {
      vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    } catch {
      vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm'
      );
    }

    try {
      cachedLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });
    } catch {
      // Fallback to CPU delegate when WebGL/GPU is unavailable
      cachedLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.5,
      });
    }

    return cachedLandmarker;
  })();

  return loadingPromise.catch((err) => {
    loadingPromise = null;
    throw err;
  });
}

export function mapMediaPipeResult(result: HandLandmarkerResult): DetectedHandInput[] {
  if (!result.landmarks || result.landmarks.length === 0) return [];
  return result.landmarks.slice(0, 2).map((lms, idx) => {
    const cat = result.handednesses?.[idx]?.[0];
    const rawName = cat?.categoryName;
    const handedness: HandednessLabel =
      rawName === 'Left' || rawName === 'Right' ? rawName : 'Unknown';
    return {
      landmarks: lms.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      handedness,
      score: cat?.score ?? 1,
    };
  });
}
