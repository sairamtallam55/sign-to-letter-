import { useMemo, useState } from 'react';
import { ArrowRight, ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { ISL_ALPHABET } from '../ml/islAlphabet';
import type { ISLLetterGuide, SignModality } from '../ml/types';
import { ISL_SIGN_IMAGES } from '../utils/islSignImages';
import { HandPoseDiagram } from './HandPoseDiagram';

interface LearnViewProps {
  onPracticeLetter: (letter: string) => void;
}

type FilterType = 'all' | SignModality;

interface ISLVisualGuideEntry extends ISLLetterGuide {
  imageSrc: string;
}

const ISL_VISUAL_REFERENCE: ISLVisualGuideEntry[] = ISL_ALPHABET.map((item) => ({
  ...item,
  imageSrc: ISL_SIGN_IMAGES[item.letter] ?? '',
}));

export function LearnView({ onPracticeLetter }: LearnViewProps) {
  const [filter, setFilter] = useState<FilterType>('all');
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState<ISLVisualGuideEntry>(ISL_VISUAL_REFERENCE[0]);

  const filteredLetters = useMemo(() => {
    const q = query.trim().toUpperCase();
    return ISL_VISUAL_REFERENCE.filter((item) => {
      const matchesFilter = filter === 'all' || item.modality === filter;
      const matchesQuery =
        !q ||
        item.letter.includes(q) ||
        item.formationSummary.toUpperCase().includes(q) ||
        item.svgPose.contactHint.toUpperCase().includes(q) ||
        item.keyLandmarks.toUpperCase().includes(q) ||
        item.islVsAslNote.toUpperCase().includes(q);
      return matchesFilter && matchesQuery;
    });
  }, [filter, query]);

  const selectedIndex = ISL_VISUAL_REFERENCE.findIndex((item) => item.letter === selected.letter);

  const handleStep = (delta: -1 | 1) => {
    const nextIdx =
      (selectedIndex + delta + ISL_VISUAL_REFERENCE.length) % ISL_VISUAL_REFERENCE.length;
    setSelected(ISL_VISUAL_REFERENCE[nextIdx]);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-6">
        <div className="max-w-2xl">
          <h1 className="font-display text-2xl sm:text-3xl font-bold text-slate-900">
            Interactive Indian Sign Language (ISL) A–Z Guide
          </h1>
          <p className="text-sm sm:text-base text-slate-600 mt-1.5">
            Comprehensive visual reference mapped to the ISLRTC two-handed manual alphabet
            standard. Select any letter to inspect hand posture photographs, landmark cues, and
            launch live practice.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Filter ISL alphabet letters"
              placeholder="Search letter or cue..."
              className="pl-9 pr-3 py-2 text-sm bg-white border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 w-full sm:w-52"
            />
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg overflow-x-auto">
            {(
              [
                { id: 'all', label: 'All 26' },
                { id: 'two-handed-static', label: 'Two-Handed' },
                { id: 'single-handed-static', label: 'Single-Handed' },
                { id: 'dynamic-motion', label: 'Dynamic (J, Z)' },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setFilter(tab.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  filter === tab.id
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Visual Alphabet Reference Grid */}
        <div className="lg:col-span-7 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>
              Showing <strong className="font-mono text-slate-800">{filteredLetters.length}</strong>{' '}
              of 26 ISL visual gestures
            </span>
            <span>Click any card to inspect full formation</span>
          </div>

          {filteredLetters.length === 0 ? (
            <div className="bg-white border border-slate-200 rounded-xl p-10 text-center space-y-2">
              <p className="text-sm font-semibold text-slate-800">No matching ISL letters</p>
              <p className="text-xs text-slate-500">
                Clear your search filter to view all 26 visual reference cards.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3.5">
              {filteredLetters.map((item) => {
                const isSelected = selected.letter === item.letter;
                return (
                  <button
                    key={item.letter}
                    type="button"
                    onClick={() => setSelected(item)}
                    className={`text-left rounded-xl border p-3 transition-all cursor-pointer bg-white ${
                      isSelected
                        ? 'border-blue-700 ring-2 ring-blue-100 shadow-xs'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-baseline justify-between mb-2">
                      <span className="font-display text-xl font-bold text-slate-900">
                        {item.letter}
                      </span>
                      <span className="text-[11px] font-medium text-slate-500">
                        {item.modality === 'dynamic-motion'
                          ? 'Dynamic'
                          : `${item.handsRequired}-Hand`}
                      </span>
                    </div>
                    <HandPoseDiagram guide={item} imageSrc={item.imageSrc} size="sm" />
                    <p className="text-xs text-slate-600 mt-2 line-clamp-2">
                      {item.svgPose.contactHint}
                    </p>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Selected Letter Detail & Practice Inspector */}
        <aside className="lg:col-span-5 lg:sticky lg:top-24 bg-white border border-slate-200 rounded-xl p-6 space-y-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs font-medium text-blue-700">
                ISL Reference ·{' '}
                {selected.modality === 'dynamic-motion'
                  ? 'Dynamic Motion Sequence'
                  : selected.modality === 'two-handed-static'
                  ? 'Two-Handed Static Posture'
                  : 'Single-Handed Static Posture'}
              </p>
              <h2 className="font-display text-3xl font-bold text-slate-900 mt-1">
                Letter {selected.letter}
              </h2>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <div className="inline-flex items-center border border-slate-200 rounded-lg bg-slate-50 p-0.5">
                <button
                  type="button"
                  onClick={() => handleStep(-1)}
                  aria-label="Previous ISL letter"
                  className="p-1.5 text-slate-600 hover:text-slate-900 hover:bg-white rounded-md transition-colors cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="px-2 font-mono text-[11px] text-slate-600 tabular-nums">
                  {selectedIndex + 1}/26
                </span>
                <button
                  type="button"
                  onClick={() => handleStep(1)}
                  aria-label="Next ISL letter"
                  className="p-1.5 text-slate-600 hover:text-slate-900 hover:bg-white rounded-md transition-colors cursor-pointer"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              <button
                type="button"
                onClick={() => onPracticeLetter(selected.letter)}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <span>Practice {selected.letter}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="space-y-1.5">
            <HandPoseDiagram guide={selected} imageSrc={selected.imageSrc} size="lg" />
            <p className="text-[11px] text-slate-500 flex items-center justify-between">
              <span>Visual Posture Reference · {selected.svgPose.contactHint}</span>
              <span className="font-mono">{selected.handsRequired === 2 ? 'Two Hands' : 'Single Hand'}</span>
            </p>
          </div>

          <div className="space-y-4 text-sm">
            <div>
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Formation Summary
              </h3>
              <p className="text-slate-800 mt-1 leading-relaxed">{selected.formationSummary}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-slate-100">
              <div>
                <h4 className="text-xs font-semibold text-blue-800">Dominant (Active) Hand</h4>
                <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                  {selected.dominantHand}
                </p>
              </div>
              <div>
                <h4 className="text-xs font-semibold text-teal-800">Non-Dominant (Base) Hand</h4>
                <p className="text-xs text-slate-600 mt-1 leading-relaxed">{selected.baseHand}</p>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100">
              <h4 className="text-xs font-semibold text-slate-700">MediaPipe Key Landmarks</h4>
              <p className="font-mono text-xs text-slate-600 mt-1">{selected.keyLandmarks}</p>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
              <h4 className="text-xs font-semibold text-slate-800">ISL vs. ASL Distinction</h4>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                {selected.islVsAslNote}
              </p>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
