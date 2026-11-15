import { useEffect, useRef, useState } from 'react';
import type { ISLLetterGuide } from '../ml/types';
import { ISL_SIGN_IMAGES } from '../utils/islSignImages';
import { LoadingIndicator } from './LoadingIndicator';

interface HandPoseDiagramProps {
  guide: ISLLetterGuide;
  imageSrc?: string;
  size?: 'sm' | 'lg';
}

export function HandPoseDiagram({ guide, imageSrc, size = 'sm' }: HandPoseDiagramProps) {
  const [failedLetter, setFailedLetter] = useState<string | null>(null);
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const resolvedSrc = imageSrc ?? ISL_SIGN_IMAGES[guide.letter];
  const hasError = failedLetter === guide.letter || !resolvedSrc;
  const isLoaded = Boolean(resolvedSrc && loadedSrc === resolvedSrc);

  useEffect(() => {
    setFailedLetter(null);
    if (imgRef.current && imgRef.current.complete && imgRef.current.naturalWidth > 0 && resolvedSrc) {
      setLoadedSrc(resolvedSrc);
    }
  }, [guide.letter, resolvedSrc]);

  return (
    <div
      className={`relative w-full aspect-[4/3] rounded-lg bg-slate-100 border border-slate-200 flex items-center justify-center overflow-hidden ${
        size === 'lg' ? 'max-h-[340px]' : ''
      }`}
    >
      {hasError ? (
        <div className="flex flex-col items-center justify-center text-center px-3 py-2 text-slate-600">
          <span className="font-display text-2xl font-bold text-slate-800">{guide.letter}</span>
          <span className="text-[11px] mt-1 line-clamp-2">{guide.svgPose.contactHint}</span>
        </div>
      ) : (
        <>
          {!isLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-100">
              <LoadingIndicator variant="inline" size="sm" label="Loading..." />
            </div>
          )}
          <img
            ref={imgRef}
            src={resolvedSrc}
            alt={`Indian Sign Language hand gesture for letter ${guide.letter} — ${guide.svgPose.contactHint}`}
            referrerPolicy="no-referrer"
            onLoad={() => resolvedSrc && setLoadedSrc(resolvedSrc)}
            onError={() => setFailedLetter(guide.letter)}
            className={`w-full h-full object-cover object-center transition-opacity duration-150 ${
              isLoaded ? 'opacity-100' : 'opacity-0'
            }`}
          />
        </>
      )}
      <span className="absolute bottom-1.5 right-1.5 px-1.5 py-0.5 rounded bg-white/90 border border-slate-200/80 text-[10px] font-mono text-slate-700 leading-none">
        {guide.handsRequired}H · {guide.modality === 'dynamic-motion' ? 'Motion' : 'Static'}
      </span>
    </div>
  );
}
