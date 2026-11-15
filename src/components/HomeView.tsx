import { ArrowRight, BookOpen, Camera, Database, Layers } from 'lucide-react';
import type { AppView } from './Header';
import { LoadingIndicator } from './LoadingIndicator';
import heroImage from '../assets/images/isl_hero_studio_1794752569629.jpg';

interface HomeViewProps {
  onNavigate: (view: AppView) => void;
  totalSamples: number | null;
  classesCovered: number | null;
  isLoading?: boolean;
}

export function HomeView({ onNavigate, totalSamples, classesCovered, isLoading = false }: HomeViewProps) {
  return (
    <div className="space-y-16 pb-16">
      {/* Editorial 2-Column Hero */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 lg:pt-14">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-stretch">
          <div className="lg:col-span-7 flex flex-col justify-between py-2">
            <div className="space-y-6">
              <p className="text-sm font-medium text-blue-700">
                Indian Sign Language · Computer Vision &amp; Machine Learning
              </p>

              <h1 className="font-display text-4xl sm:text-5xl font-bold tracking-tight text-slate-900 leading-[1.1] text-balance">
                Learn and practice Indian Sign Language in real time.
              </h1>

              <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-2xl">
                Unlike American Sign Language (ASL), which relies mostly on one hand,{' '}
                <strong className="font-semibold text-slate-900">
                  Indian Sign Language (ISL) uses a predominantly two-handed alphabet
                </strong>
                . ISL Studio tracks 21 skeletal landmarks per hand via MediaPipe, normalizes 3D
                geometry, and classifies alphabet gestures with a Random Forest and dynamic stroke
                pipeline.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-2">
                <button
                  type="button"
                  onClick={() => onNavigate('recognize')}
                  className="inline-flex items-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
                >
                  <Camera className="w-4 h-4" />
                  <span>Start Live Recognition</span>
                  <ArrowRight className="w-4 h-4" />
                </button>

                <button
                  type="button"
                  onClick={() => onNavigate('learn')}
                  className="inline-flex items-center gap-2 px-5 py-3 text-sm font-semibold text-slate-800 bg-white border border-slate-300 hover:bg-slate-50 rounded-lg transition-colors cursor-pointer"
                >
                  <BookOpen className="w-4 h-4 text-slate-600" />
                  <span>Explore ISL A–Z Guide</span>
                </button>
              </div>
            </div>

            <div className="pt-8 mt-8 border-t border-slate-200 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-500">
              <span>ISLRTC Two-Handed Standard</span>
              <span aria-hidden="true">·</span>
              <span>Static A–Z &amp; Dynamic J, Z</span>
              <span aria-hidden="true">·</span>
              <span>On-device camera privacy (no video stored)</span>
            </div>
          </div>

          <div className="lg:col-span-5">
            <div className="relative h-72 sm:h-96 lg:h-full min-h-[340px] rounded-xl overflow-hidden border border-slate-200 bg-slate-200">
              <img
                src={heroImage}
                alt="Educator demonstrating a two-handed Indian Sign Language alphabet gesture in front of a laptop camera"
                referrerPolicy="no-referrer"
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/80 via-slate-950/35 to-transparent p-5">
                <p className="text-sm font-medium text-white">
                  Two-handed bimanual tracking with 21 landmarks per hand
                </p>
                <p className="text-xs text-slate-200 mt-0.5">
                  Captures both dominant and base hand geometry in real time
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Full-Width 4-Column Capability Strip */}
        <div className="mt-10 grid grid-cols-2 lg:grid-cols-4 gap-4 border-y border-slate-200 py-6">
          <div>
            <p className="text-xs font-medium text-slate-500">Landmark Geometry</p>
            <p className="font-mono text-2xl font-semibold text-slate-900 mt-1 tabular-nums">
              21 × 3
            </p>
            <p className="text-xs text-slate-600 mt-0.5">
              63 coords/hand · 129 canonical dual-hand features
            </p>
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500">Alphabet Coverage</p>
            <div className="font-mono text-2xl font-semibold text-slate-900 mt-1 tabular-nums min-h-[32px] flex items-center">
              {isLoading || classesCovered === null ? (
                <LoadingIndicator variant="inline" size="sm" label="Syncing..." />
              ) : (
                `${classesCovered} / 26`
              )}
            </div>
            <p className="text-xs text-slate-600 mt-0.5">
              24 static classes + 2 dynamic motion (J, Z)
            </p>
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500">Active Dataset</p>
            <div className="font-mono text-2xl font-semibold text-slate-900 mt-1 tabular-nums min-h-[32px] flex items-center">
              {isLoading || totalSamples === null ? (
                <LoadingIndicator variant="inline" size="sm" label="Loading..." />
              ) : (
                totalSamples
              )}
            </div>
            <p className="text-xs text-slate-600 mt-0.5">
              Normalized CSV samples ready for retraining
            </p>
          </div>
          <div>
            <p className="text-xs font-medium text-slate-500">ML Classifier</p>
            <p className="font-mono text-2xl font-semibold text-slate-900 mt-1">
              Random Forest
            </p>
            <p className="text-xs text-slate-600 mt-0.5">
              With frame smoothing &amp; uncertainty rejection
            </p>
          </div>
        </div>
      </section>

      {/* How the System Works */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-2xl mb-8">
          <h2 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">
            How the recognition pipeline works
          </h2>
          <p className="text-sm sm:text-base text-slate-600 mt-2">
            Instead of sending raw webcam video to a server, only lightweight numerical coordinates
            travel over the API—protecting user privacy and keeping latency low.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <p className="font-mono text-xs font-semibold text-blue-700">01. Capture</p>
            <h3 className="font-display text-lg font-semibold text-slate-900">
              Live Webcam &amp; MediaPipe
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Your browser detects up to 2 hands per frame and extracts 21 3D skeletal keypoints
              (wrist, knuckles, and fingertips) locally.
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <p className="font-mono text-xs font-semibold text-blue-700">02. Normalize</p>
            <h3 className="font-display text-lg font-semibold text-slate-900">
              Scale &amp; Position Invariance
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Coordinates are centered at the wrist (landmark 0) and scaled by palm size so signs
              work regardless of distance from the camera.
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <p className="font-mono text-xs font-semibold text-blue-700">03. Classify</p>
            <h3 className="font-display text-lg font-semibold text-slate-900">
              Static Forest &amp; Dynamic J/Z
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              The REST API scores static handshapes with a Random Forest ensemble, while dynamic
              letters J and Z use temporal trajectory matching.
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <p className="font-mono text-xs font-semibold text-blue-700">04. Practice &amp; Train</p>
            <h3 className="font-display text-lg font-semibold text-slate-900">
              Custom Dataset Studio
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Record authentic samples of your own hands, export the CSV dataset, and retrain the
              classifier with session-wise evaluation metrics.
            </p>
          </div>
        </div>
      </section>

      {/* Direct Workspace Entry Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <button
            type="button"
            onClick={() => onNavigate('learn')}
            className="text-left bg-white border border-slate-200 hover:border-blue-300 rounded-xl p-6 transition-colors cursor-pointer group"
          >
            <BookOpen className="w-5 h-5 text-blue-700 mb-3" />
            <h3 className="font-display text-lg font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
              Authentic ISL Alphabet Guide
            </h3>
            <p className="text-sm text-slate-600 mt-1.5">
              Study all 26 ISL formations with two-handed diagrams, ASL contrast notes, and direct
              practice mode.
            </p>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('studio')}
            className="text-left bg-white border border-slate-200 hover:border-blue-300 rounded-xl p-6 transition-colors cursor-pointer group"
          >
            <Database className="w-5 h-5 text-blue-700 mb-3" />
            <h3 className="font-display text-lg font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
              Webcam Data &amp; ML Studio
            </h3>
            <p className="text-sm text-slate-600 mt-1.5">
              Collect labeled 21-landmark samples, inspect confusion matrices, and benchmark Random
              Forest vs baseline models.
            </p>
          </button>

          <button
            type="button"
            onClick={() => onNavigate('about')}
            className="text-left bg-white border border-slate-200 hover:border-blue-300 rounded-xl p-6 transition-colors cursor-pointer group"
          >
            <Layers className="w-5 h-5 text-blue-700 mb-3" />
            <h3 className="font-display text-lg font-semibold text-slate-900 group-hover:text-blue-700 transition-colors">
              Research, Architecture &amp; Cloud Docs
            </h3>
            <p className="text-sm text-slate-600 mt-1.5">
              Review verified ISL datasets, system diagrams, Python Flask deployment steps, and
              honest feasibility findings.
            </p>
          </button>
        </div>
      </section>
    </div>
  );
}
