import { ExternalLink } from 'lucide-react';
import type { DatasetSummary } from '../ml/types';
import { LoadingIndicator } from './LoadingIndicator';

interface AboutViewProps {
  summary: DatasetSummary | null;
  isLoading?: boolean;
}

export function AboutView({ summary, isLoading = false }: AboutViewProps) {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Header */}
      <header className="border-b border-slate-200 pb-6 space-y-3">
        <p className="text-xs font-medium text-blue-700">
          Phase 1–4 Research, Architecture &amp; Deployment Documentation
        </p>
        <h1 className="font-display text-3xl sm:text-4xl font-bold text-slate-900">
          Research, System Architecture &amp; Feasibility
        </h1>
        <p className="text-base text-slate-600 max-w-3xl leading-relaxed">
          A transparent engineering breakdown of how Indian Sign Language differs from ASL, which
          verified datasets exist, how our 21-landmark dual-hand architecture works, and how to
          deploy the Python Flask service.
        </p>
        <div className="pt-1 text-xs text-slate-600 flex flex-wrap items-center gap-3">
          {isLoading || !summary ? (
            <LoadingIndicator variant="inline" size="sm" label="Loading live runtime dataset status..." />
          ) : (
            <span className="font-mono text-slate-700">
              Active Runtime Dataset: <strong>{summary.totalSamples}</strong> samples ·{' '}
              <strong>{summary.classesCovered}/26</strong> classes covered ·{' '}
              <strong>{summary.userWebcamSamples}</strong> user webcam recordings
            </span>
          )}
        </div>
      </header>

      {/* 1. ISL vs ASL Research Findings */}
      <section className="space-y-4">
        <h2 className="font-display text-2xl font-bold text-slate-900">
          01. Authentic ISL vs. ASL Alphabet Findings
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2">
            <h3 className="font-display text-lg font-semibold text-slate-900">
              Indian Sign Language (ISLRTC Standard)
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Standardized by the Indian Sign Language Research and Training Centre (ISLRTC,
              Ministry of Social Justice and Empowerment), ISL predominantly uses a{' '}
              <strong className="text-slate-900">two-handed (bimanual) manual alphabet</strong>,
              historically related to the BANZSL family. Most letters require two hands cooperating
              (e.g., M places three fingers onto the opposite palm; T crosses index onto the base
              hand), while C, I, L, O, U, V can be formed single-handedly and J and Z require
              temporal motion.
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2">
            <h3 className="font-display text-lg font-semibold text-slate-900">
              Why ASL Models Cannot Be Reused
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              American Sign Language (ASL) fingerspelling uses a one-handed alphabet for all 26
              letters. Training or evaluating an ISL system on ASL datasets produces visually false
              results for letters such as A, B, D, E, F, G, H, K, M, N, P, Q, R, S, T, W, X, and Y.
              Accordingly, this system configures MediaPipe with{' '}
              <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">numHands = 2</code>{' '}
              and normalizes 129 canonical dual-hand features.
            </p>
          </div>
        </div>
      </section>

      {/* 2. Verified Public Datasets & Official References */}
      <section className="space-y-4">
        <h2 className="font-display text-2xl font-bold text-slate-900">
          02. Verified Reference Sources &amp; Public Datasets
        </h2>
        <div className="overflow-x-auto bg-white border border-slate-200 rounded-xl">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-700">
              <tr>
                <th className="py-3 px-4 font-semibold">Source / Dataset</th>
                <th className="py-3 px-4 font-semibold">License</th>
                <th className="py-3 px-4 font-semibold">Classes &amp; Volume</th>
                <th className="py-3 px-4 font-semibold">Format &amp; Suitability</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-600">
              <tr>
                <td className="py-3.5 px-4 font-medium text-slate-900">
                  <a
                    href="https://islrtc.nic.in/"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-blue-700 hover:underline"
                  >
                    <span>ISLRTC Official Manual Alphabet</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </td>
                <td className="py-3.5 px-4">Govt. of India Educational</td>
                <td className="py-3.5 px-4">26 English letters (A–Z)</td>
                <td className="py-3.5 px-4">
                  Authoritative visual posters &amp; self-learning video reference for authentic
                  two-handed ISL formations.
                </td>
              </tr>

              <tr>
                <td className="py-3.5 px-4 font-medium text-slate-900">
                  <a
                    href="https://data.mendeley.com/datasets/2tj6c7pzfk/1"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-blue-700 hover:underline"
                  >
                    <span>Multimodal ISL Dataset (Mendeley)</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </td>
                <td className="py-3.5 px-4 font-mono text-xs">CC BY 4.0</td>
                <td className="py-3.5 px-4">
                  79 classes (26 English A–Z, 43 Marathi, 10 digits); 1,425 unique captures (4,740
                  multimodal images, 20/class)
                </td>
                <td className="py-3.5 px-4">
                  RGB images, MediaPipe 21-point skeletal images, &amp; skin masks. Static only;
                  requires local download &amp; landmark CSV extraction.
                </td>
              </tr>

              <tr>
                <td className="py-3.5 px-4 font-medium text-slate-900">
                  Built-In Custom Landmark Collector
                </td>
                <td className="py-3.5 px-4 font-mono text-xs">Apache-2.0 (Local CSV)</td>
                <td className="py-3.5 px-4">24 static classes + J/Z dynamic module</td>
                <td className="py-3.5 px-4">
                  Ships with synthetic calibration prototypes for immediate testing; records real
                  user webcam landmarks directly to CSV for genuine accuracy.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 3. System Architecture SVG Diagram */}
      <section className="space-y-4">
        <h2 className="font-display text-2xl font-bold text-slate-900">
          03. System &amp; Computer Vision Pipeline Architecture
        </h2>
        <div className="bg-white border border-slate-200 rounded-xl p-6 overflow-x-auto">
          <svg
            viewBox="0 0 880 220"
            className="w-full min-w-[680px] h-auto"
            role="img"
            aria-label="System Architecture diagram from Live Webcam through MediaPipe, Normalization, REST API, and Random Forest Classifier"
          >
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
              </marker>
            </defs>

            {/* Stage Boxes */}
            {[
              { x: 16, title: '1. Live Webcam', sub: 'Browser getUserMedia', accent: '#0f172a' },
              { x: 190, title: '2. MediaPipe Hands', sub: '21 × 3 Landmarks (≤2)', accent: '#1d4ed8' },
              { x: 364, title: '3. Normalizer', sub: 'Wrist Origin + Scale', accent: '#1d4ed8' },
              { x: 538, title: '4. REST API', sub: 'Express / Flask JSON', accent: '#0f766e' },
              { x: 712, title: '5. ML Classifiers', sub: 'RF Static + DTW (J,Z)', accent: '#047857' },
            ].map((box, idx) => (
              <g key={box.title}>
                <rect x={box.x} y="30" width="152" height="78" rx="10" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.5" />
                <rect x={box.x} y="30" width="152" height="6" rx="3" fill={box.accent} />
                <text x={box.x + 14} y="62" fontSize="13" fontWeight="700" fill="#0f172a" fontFamily="Plus Jakarta Sans, sans-serif">
                  {box.title}
                </text>
                <text x={box.x + 14} y="84" fontSize="11" fill="#475569" fontFamily="JetBrains Mono, monospace">
                  {box.sub}
                </text>
                {idx < 4 && (
                  <line
                    x1={box.x + 154}
                    y1="69"
                    x2={box.x + 172}
                    y2="69"
                    stroke="#475569"
                    strokeWidth="1.8"
                    markerEnd="url(#arrow)"
                  />
                )}
              </g>
            ))}

            {/* Feedback & Training Loop */}
            <rect x="190" y="142" width="500" height="54" rx="10" fill="#eff6ff" stroke="#bfdbfe" strokeWidth="1.5" />
            <text x="210" y="166" fontSize="12" fontWeight="700" fill="#1e3a8a" fontFamily="Plus Jakarta Sans, sans-serif">
              Feedback, Smoothing &amp; Custom Training Loop
            </text>
            <text x="210" y="184" fontSize="11" fill="#1e40af" fontFamily="JetBrains Mono, monospace">
              6-frame Majority Smoother → History → /api/collect (CSV) → /api/train (Session-wise Split)
            </text>
          </svg>
        </div>
      </section>

      {/* 4. Cloud Runtime Analysis & Python Flask Deployment */}
      <section className="space-y-4">
        <h2 className="font-display text-2xl font-bold text-slate-900">
          04. Cloud Runtime Architecture &amp; Python Flask Deployment
        </h2>
        <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4 text-sm text-slate-600 leading-relaxed">
          <p>
            <strong className="text-slate-900">AI Studio Runtime Inspection:</strong> Google AI
            Studio runs a persistent Node.js/Express container on port 3000 with HTTPS preview, while
            Python packages (<code className="font-mono text-xs">flask</code>,{' '}
            <code className="font-mono text-xs">scikit-learn</code>) are not preinstalled via pip.
            To provide a zero-setup live application right inside AI Studio, the integrated{' '}
            <code className="font-mono text-xs">server.ts</code> Express server implements the
            complete Random Forest training, CSV dataset management, and static/dynamic prediction
            REST API natively in TypeScript.
          </p>
          <p>
            <strong className="text-slate-900">Dedicated Python Flask Service:</strong> The complete
            Python Flask + Scikit-Learn + Pandas + Joblib + OpenCV/MediaPipe backend is included in{' '}
            <code className="font-mono text-xs">/python_backend/</code> along with{' '}
            <code className="font-mono text-xs">Dockerfile</code> and{' '}
            <code className="font-mono text-xs">render.yaml</code> for one-click deployment to{' '}
            <strong className="text-slate-900">Google Cloud Run</strong> or{' '}
            <strong className="text-slate-900">Render</strong> (both provide HTTPS endpoints
            required for browser webcam access). Once deployed, paste the Python service URL into
            the Data &amp; Model Studio endpoint switcher.
          </p>
        </div>
      </section>

      {/* 5. Known Limitations & Future Scope */}
      <section className="space-y-4">
        <h2 className="font-display text-2xl font-bold text-slate-900">
          05. Honest Limitations &amp; Future Improvements
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2">
            <h3 className="font-display text-base font-semibold text-slate-900">
              Current Limitations
            </h3>
            <ul className="list-disc pl-5 space-y-1.5 text-sm text-slate-600">
              <li>
                Default preloaded weights are trained on synthetic biomechanical calibration
                prototypes—not real human signing recordings—until you record webcam samples.
              </li>
              <li>
                Two-handed occlusion (when one hand hides behind the other from a single 2D webcam)
                can reduce MediaPipe landmark stability.
              </li>
              <li>
                Dynamic recognition covers fingerspelled J and Z trajectories rather than full-body
                or facial non-manual ISL sentence grammar.
              </li>
            </ul>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-2">
            <h3 className="font-display text-base font-semibold text-slate-900">
              Future Improvements
            </h3>
            <ul className="list-disc pl-5 space-y-1.5 text-sm text-slate-600">
              <li>
                Import extracted MediaPipe CSV landmarks from Mendeley DOI{' '}
                <code className="font-mono text-xs">10.17632/2tj6c7pzfk.1</code> across diverse
                signers and lighting conditions.
              </li>
              <li>
                Upgrade the dynamic J/Z trajectory classifier to a sequence LSTM/GRU trained on
                multi-user 30-frame landmark tensors.
              </li>
              <li>
                Add Text-to-Speech (Web Speech API) output for assembled ISL words and full ISL
                word-level vocabulary beyond A–Z.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
