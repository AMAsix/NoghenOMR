import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

type Detection = {
  staves: Array<{ lines: number[]; bbox: [number, number, number, number] }>;
  systems: Array<{ bbox: [number, number, number, number] }>;
  measures: Array<{ x: number; y0: number; y1: number }>;
  image?: string;
};

export default function OMRCanvas({ file }: { file: File }) {
  const backendBase = useMemo(() => 
    (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'))
      ? `${window.location.protocol}//${window.location.hostname}:4000`
      : ''
  , []);

  const apiFetch = useCallback(async (path: string, opts?: RequestInit) => {
    if (backendBase) {
      try {
        const r = await fetch(backendBase + path, opts);
        if (r.status !== 404) return r;
      } catch {
        // fallthrough to relative
      }
    }
    return fetch(path, opts);
  }, [backendBase]);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [pageWidth, setPageWidth] = useState(0);
  const [pageHeight, setPageHeight] = useState(0);
  const [scaleX, setScaleX] = useState(1);
  const [scaleY, setScaleY] = useState(1);
  const [page, setPage] = useState(0);
  const [pageCount, setPageCount] = useState<number | null>(null);
  const [detections, setDetections] = useState<Detection | null>(null);
  const [cache, setCache] = useState<Record<number, Detection>>({});
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [loadingPages, setLoadingPages] = useState<Record<number, boolean>>({});
  
  // Refs to avoid stale closures
  const cacheRef = useRef<Record<number, Detection>>({});
  cacheRef.current = cache;
  const pageRef = useRef(page);
  pageRef.current = page;
  const jobIdRef = useRef<string | null>(null);
  jobIdRef.current = jobId;

  const startJob = useCallback(async () => {
    setErrorMsg(null);
    setCache({});
    setDetections(null);
    setImgSrc(null);
    setJobId(null);
    setLoadingPages({});
    setPage(0);

    try {
      const arrayBuffer = await file.arrayBuffer();
      const form = new FormData();
      form.append('file', new Blob([arrayBuffer]), file.name);
      const r = await apiFetch('/api/omr/start_job', { method: 'POST', body: form });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(`${r.status}: ${body}`);
      }
      const j = await r.json();
      setJobId(j.job_id || null);
      if (j.page_count !== undefined) setPageCount(j.page_count);
      return j;
    } catch (e: unknown) {
      console.error('start_job failed', e);
      setErrorMsg(String(e instanceof Error ? e.message : e));
      return null;
    }
  }, [file, apiFetch]);

  // Start job when file changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      await startJob();
    })();
    return () => { cancelled = true; };
  }, [file, startJob]);

  // Update image dimensions on load
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const onLoad = () => {
      const rect = img.getBoundingClientRect();
      setPageWidth(rect.width);
      setPageHeight(rect.height);
      const natW = img.naturalWidth || rect.width;
      const natH = img.naturalHeight || rect.height;
      setScaleX(rect.width / natW);
      setScaleY(rect.height / natH);
    };
    img.addEventListener('load', onLoad);
    if (img.complete) onLoad();
    return () => img.removeEventListener('load', onLoad);
  }, [imgSrc]);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let restarted = false;

    const poll = async () => {
      if (cancelled) return;
      try {
        const currentJobId = jobIdRef.current;
        if (!currentJobId) return;
        const r = await apiFetch(`/api/omr/job/${currentJobId}/status`);
        if (r.status === 404) {
          if (!restarted) {
            restarted = true;
            setErrorMsg('Job not found, restarting...');
            await startJob();
            return;
          }
          setErrorMsg('Job not found. Please try again.');
          setJobId(null);
          return;
        }
        if (!r.ok) return;
        const s = await r.json();
        const done: number[] = s.done || [];
        const currentCache = cacheRef.current || {};
        const currentPage = pageRef.current;

        for (const idx of done) {
          if (currentCache[idx]) continue;
          try {
            const rp = await apiFetch(`/api/omr/job/${currentJobId}/page/${idx}`);
            if (rp.status === 202) continue;
            if (!rp.ok) continue;
            const pj = await rp.json();
            setCache((c) => ({ ...c, [idx]: pj }));
            setLoadingPages((l) => { const copy = { ...l }; delete copy[idx]; return copy; });
            if (idx === currentPage) {
              setDetections(pj);
              if (pj.image) setImgSrc(`data:image/png;base64,${pj.image}`);
            }
          } catch {
            // ignore
          }
        }
      } catch {
        // ignore polling errors
      }
    };

    poll();
    const iv = setInterval(poll, 1500);
    return () => { cancelled = true; clearInterval(iv); };
  }, [jobId, apiFetch, startJob]);

  const prioritizePage = useCallback(async (p: number) => {
    const currentJobId = jobIdRef.current;
    if (!currentJobId) return;
    try {
      const form = new FormData();
      form.append('page', String(p));
      await apiFetch(`/api/omr/job/${currentJobId}/prioritize`, { method: 'POST', body: form });
    } catch {
      // ignore
    }
  }, [apiFetch]);

  // Handle page navigation
  useEffect(() => {
    if (!jobId) return;
    const cachedPage = cache?.[page];
    if (cachedPage) {
      setDetections(cachedPage);
      if (cachedPage.image) setImgSrc(`data:image/png;base64,${cachedPage.image}`);
      return;
    }
    setDetections(null);
    setImgSrc(null);
    setLoadingPages((l) => ({ ...l, [page]: true }));
    prioritizePage(page);
  }, [page, jobId, prioritizePage, cache]);

  // Update display when cache changes for current page
  useEffect(() => {
    const cachedPage = cache?.[page];
    if (cachedPage) {
      setDetections(cachedPage);
      if (cachedPage.image) setImgSrc(`data:image/png;base64,${cachedPage.image}`);
      setLoadingPages((l) => { const copy = { ...l }; delete copy[page]; return copy; });
    }
  }, [cache, page]);

  return (
    <div style={{ padding: 16 }}>
      {errorMsg && (
        <div style={{ 
          padding: 12, 
          marginBottom: 16, 
          background: '#fee', 
          border: '1px solid #fcc',
          borderRadius: 4,
          color: '#c00'
        }}>
          {errorMsg}
        </div>
      )}

      {imgSrc ? (
        <>
          <div style={{ 
            padding: 12, 
            marginBottom: 16, 
            display: 'flex', 
            gap: 12, 
            alignItems: 'center',
            background: '#f9f9f9',
            borderRadius: 4
          }}>
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page <= 0}
              style={{ padding: '8px 16px', cursor: page <= 0 ? 'not-allowed' : 'pointer' }}
            >
              ◀ Prev
            </button>
            <div>
              Page{' '}
              <input
                type="number"
                value={page + 1}
                min={1}
                max={pageCount ?? 1}
                onChange={(e) => {
                  const v = Number(e.target.value || 1) - 1;
                  setPage(isNaN(v) ? 0 : Math.max(0, Math.min(v, (pageCount ?? 1) - 1)));
                }}
                style={{ width: 64, padding: 4 }}
              />
              {pageCount ? <> / {pageCount}</> : null}
            </div>
            <button
              onClick={() => setPage((p) => (pageCount ? Math.min(pageCount - 1, p + 1) : p + 1))}
              disabled={pageCount !== null && page >= pageCount - 1}
              style={{ 
                padding: '8px 16px', 
                cursor: (pageCount !== null && page >= pageCount - 1) ? 'not-allowed' : 'pointer' 
              }}
            >
              Next ▶
            </button>
          </div>

          <div style={{ position: 'relative', display: 'inline-block' }}>
            {cache[page] ? (
              <img 
                ref={imgRef} 
                src={`data:image/png;base64,${cache[page].image}`} 
                alt={`Page ${page + 1}`}
                style={{ display: 'block', maxWidth: '100%', height: 'auto' }} 
              />
            ) : (
              <div style={{ 
                width: 800, 
                height: 600, 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center', 
                color: '#666',
                background: '#f0f0f0'
              }}>
                {loadingPages[page] ? `Loading page ${page + 1}...` : 'Processing...'}
              </div>
            )}

            <svg
              width={pageWidth}
              height={pageHeight}
              style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }}
            >
              {detections && (
                <g>
                  {detections.systems.map((s, i) => (
                    <rect
                      key={`sys-${i}`}
                      x={s.bbox[0] * scaleX}
                      y={s.bbox[1] * scaleY}
                      width={s.bbox[2] * scaleX}
                      height={s.bbox[3] * scaleY}
                      fill="rgba(0,128,255,0.12)"
                      stroke="rgba(0,128,255,0.6)"
                      strokeWidth={1}
                    />
                  ))}

                  {detections.staves.map((s, i) => (
                    <g key={`staff-${i}`}>
                      {s.lines.map((y, j) => (
                        <line
                          key={`staffline-${j}`}
                          x1={0}
                          x2={pageWidth}
                          y1={y * scaleY}
                          y2={y * scaleY}
                          stroke={j % 2 === 0 ? 'rgba(220,20,60,0.9)' : 'rgba(220,20,60,0.6)'}
                          strokeWidth={1}
                        />
                      ))}
                    </g>
                  ))}

                  {detections.measures.map((m, i) => (
                    <line
                      key={`measure-${i}`}
                      x1={m.x * scaleX}
                      x2={m.x * scaleX}
                      y1={m.y0 * scaleY}
                      y2={m.y1 * scaleY}
                      stroke="rgba(34,139,34,0.95)"
                      strokeWidth={3}
                    />
                  ))}
                </g>
              )}
            </svg>
          </div>
        </>
      ) : (
        <div style={{ 
          padding: 48, 
          textAlign: 'center', 
          color: '#666',
          background: '#f9f9f9',
          borderRadius: 8
        }}>
          {jobId ? 'Processing PDF...' : 'Starting...'}
        </div>
      )}
    </div>
  );
}
