import { useCallback, useEffect, useState } from 'react';
import type { DatasetSummary } from './ml/types';
import { AboutView } from './components/AboutView';
import { DataModelView } from './components/DataModelView';
import { Header, type AppView } from './components/Header';
import { HomeView } from './components/HomeView';
import { LearnView } from './components/LearnView';
import { RecognitionView } from './components/RecognitionView';

export default function App() {
  const [activeView, setActiveView] = useState<AppView>(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      if (params.get('autostart') === '1') return 'recognize';
    }
    return 'home';
  });
  const [apiBaseUrl, setApiBaseUrl] = useState<string>('');
  const [practiceTarget, setPracticeTarget] = useState<string | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [isSummaryLoading, setIsSummaryLoading] = useState<boolean>(true);

  const refreshSummary = useCallback(async () => {
    setIsSummaryLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/api/dataset`);
      if (res.ok) {
        const data = (await res.json()) as DatasetSummary;
        setSummary(data);
      }
    } catch {
      // Handled gracefully by child views
    } finally {
      setIsSummaryLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void refreshSummary();
  }, [refreshSummary]);

  const handlePracticeLetter = (letter: string) => {
    setPracticeTarget(letter);
    setActiveView('recognize');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <Header activeView={activeView} onNavigate={setActiveView} />

      <main className="flex-1">
        {activeView === 'home' && (
          <HomeView
            onNavigate={setActiveView}
            totalSamples={summary?.totalSamples ?? null}
            classesCovered={summary?.classesCovered ?? null}
            isLoading={isSummaryLoading && summary === null}
          />
        )}
        {activeView === 'recognize' && (
          <RecognitionView
            apiBaseUrl={apiBaseUrl}
            practiceTarget={practiceTarget}
            onClearPracticeTarget={() => setPracticeTarget(null)}
          />
        )}
        {activeView === 'learn' && <LearnView onPracticeLetter={handlePracticeLetter} />}
        {activeView === 'studio' && (
          <DataModelView
            apiBaseUrl={apiBaseUrl}
            onApiBaseUrlChange={setApiBaseUrl}
            summary={summary}
            isSummaryLoading={isSummaryLoading && summary === null}
            onRefreshSummary={refreshSummary}
          />
        )}
        {activeView === 'about' && (
          <AboutView summary={summary} isLoading={isSummaryLoading && summary === null} />
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <span>ISL Studio · Indian Sign Language Real-Time Hand Gesture Recognition</span>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setActiveView('learn')}
              className="hover:text-slate-900 cursor-pointer"
            >
              ISLRTC Alphabet Guide
            </button>
            <button
              type="button"
              onClick={() => setActiveView('studio')}
              className="hover:text-slate-900 cursor-pointer"
            >
              Data &amp; Evaluation
            </button>
            <button
              type="button"
              onClick={() => setActiveView('about')}
              className="hover:text-slate-900 cursor-pointer"
            >
              Architecture &amp; Docs
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
