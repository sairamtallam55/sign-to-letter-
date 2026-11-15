import type { DetectedHandInput, Landmark3D } from '../ml/types';
import type { TrajectoryPoint } from '../ml/landmarks';

// Standard MediaPipe 21-point hand bone connections
export const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // Pinky & Palm
];

export function drawHandLandmarks(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  hands: DetectedHandInput[],
  trajectory: TrajectoryPoint[] = [],
  mirror = false
): void {
  ctx.clearRect(0, 0, width, height);

  const mapX = (x: number) => (mirror ? (1 - x) * width : x * width);
  const mapY = (y: number) => y * height;

  // Draw dynamic motion trail if present
  if (trajectory.length > 1) {
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#f59e0b';
    trajectory.forEach((pt, idx) => {
      const px = mapX(pt.x);
      const py = mapY(pt.y);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    const last = trajectory[trajectory.length - 1];
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(mapX(last.x), mapY(last.y), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  const palettes = [
    { bone: '#60a5fa', joint: '#2563eb', ring: '#eff6ff', label: 'Dominant Hand' },
    { bone: '#2dd4bf', joint: '#0d9488', ring: '#f0fdfa', label: 'Base Hand' },
  ];

  hands.slice(0, 2).forEach((hand, hIdx) => {
    const palette = palettes[hIdx] ?? palettes[0];
    const lms: Landmark3D[] = hand.landmarks;
    if (!lms || lms.length !== 21) return;

    // Draw bones
    ctx.save();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = palette.bone;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = lms[a];
      const pb = lms[b];
      ctx.beginPath();
      ctx.moveTo(mapX(pa.x), mapY(pa.y));
      ctx.lineTo(mapX(pb.x), mapY(pb.y));
      ctx.stroke();
    }

    // Draw 21 landmark nodes
    lms.forEach((pt, idx) => {
      const cx = mapX(pt.x);
      const cy = mapY(pt.y);
      const isTip = idx === 4 || idx === 8 || idx === 12 || idx === 16 || idx === 20;
      const radius = idx === 0 ? 5.5 : isTip ? 5 : 3.5;

      ctx.beginPath();
      ctx.arc(cx, cy, radius + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = palette.ring;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = palette.joint;
      ctx.fill();
    });

    // Label near wrist, offset left/right per hand to avoid overlap during two-handed ISL signs
    const wrist = lms[0];
    const labelText = `${hand.handedness && hand.handedness !== 'Unknown' ? hand.handedness : palette.label}`;
    ctx.font = '600 11px "Plus Jakarta Sans", sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    const xOffset = hIdx === 0 ? 10 : -(textWidth + 18);
    const tx = Math.max(8, Math.min(width - textWidth - 16, mapX(wrist.x) + xOffset));
    const ty = Math.min(height - 12, mapY(wrist.y) + 22);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
    ctx.fillRect(tx - 6, ty - 12, textWidth + 12, 18);
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(labelText, tx, ty);
    ctx.restore();
  });
}
