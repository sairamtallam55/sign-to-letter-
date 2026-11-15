import { Loader2 } from 'lucide-react';

export interface LoadingIndicatorProps {
  label?: string;
  sublabel?: string;
  variant?: 'inline' | 'banner' | 'card' | 'overlay';
  tone?: 'light' | 'dark';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingIndicator({
  label = 'Loading...',
  sublabel,
  variant = 'inline',
  tone = 'light',
  size = 'md',
  className = '',
}: LoadingIndicatorProps) {
  const iconSize =
    size === 'sm' ? 'w-3.5 h-3.5' : size === 'lg' ? 'w-7 h-7' : 'w-4 h-4';

  const textSize =
    size === 'sm' ? 'text-xs' : size === 'lg' ? 'text-sm font-semibold' : 'text-xs sm:text-sm font-medium';

  if (variant === 'overlay') {
    const surface =
      tone === 'dark'
        ? 'bg-slate-950/85 text-slate-100'
        : 'bg-white/90 text-slate-900';
    const subColor = tone === 'dark' ? 'text-slate-300' : 'text-slate-600';
    const iconColor = tone === 'dark' ? 'text-blue-400' : 'text-blue-700';

    return (
      <div
        role="status"
        aria-live="polite"
        className={`absolute inset-0 z-20 flex flex-col items-center justify-center text-center px-6 py-6 backdrop-blur-[2px] ${surface} ${className}`}
      >
        <Loader2 className={`w-8 h-8 animate-spin ${iconColor} mb-3`} aria-hidden="true" />
        <p className="text-sm font-semibold">{label}</p>
        {sublabel && <p className={`text-xs mt-1 max-w-sm leading-relaxed ${subColor}`}>{sublabel}</p>}
      </div>
    );
  }

  if (variant === 'card') {
    return (
      <div
        role="status"
        aria-live="polite"
        className={`bg-white border border-slate-200 rounded-xl p-8 flex flex-col items-center justify-center text-center space-y-2.5 ${className}`}
      >
        <Loader2 className="w-6 h-6 animate-spin text-blue-700" aria-hidden="true" />
        <p className="text-sm font-semibold text-slate-900">{label}</p>
        {sublabel && <p className="text-xs text-slate-500 max-w-md">{sublabel}</p>}
      </div>
    );
  }

  if (variant === 'banner') {
    return (
      <div
        role="status"
        aria-live="polite"
        className={`p-4 rounded-xl border border-blue-200 bg-blue-50 text-blue-950 flex items-center gap-3 ${className}`}
      >
        <Loader2 className="w-4 h-4 animate-spin text-blue-700 shrink-0" aria-hidden="true" />
        <div className="min-w-0">
          <p className="text-sm font-semibold">{label}</p>
          {sublabel && <p className="text-xs text-blue-800 mt-0.5">{sublabel}</p>}
        </div>
      </div>
    );
  }

  const inlineColor = tone === 'dark' ? 'text-slate-200' : 'text-slate-600';
  const inlineSpinColor = tone === 'dark' ? 'text-blue-400' : 'text-blue-700';

  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 ${inlineColor} ${className}`}
    >
      <Loader2 className={`${iconSize} animate-spin ${inlineSpinColor} shrink-0`} aria-hidden="true" />
      <span className={textSize}>{label}</span>
    </span>
  );
}
