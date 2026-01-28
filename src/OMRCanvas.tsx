import { useEffect, useRef, useState } from 'react';

const API = 'http://localhost:4000';

type PageData = {
  measures: Array<{ index: number; x: number; y: number }>;
  image: string;
  image_width: number;
  image_height: number;
};

export default function OMRCanvas({ file }: { file: File }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [page, setPage] = useState(0);
  const [pages, setPages] = useState<Record<number, PageData>>({});
  const [donePages, setDonePages] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const imgRef = useRef<HTMLImageElement>(null);

  // Start job when file changes
  useEffect(() => {
    setJobId(null);
    setPages({});
    setDonePages([]);
    setPage(0);
    setError(null);

    const form = new FormData();
    form.append('file', file);
    
    fetch(`${API}/api/omr/start_job`, { method: 'POST', body: form })
      .then(r => r.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        setJobId(data.job_id);
        setPageCount(data.page_count);
      })
      .catch(e => setError(e.message));
  }, [file]);

  // Poll for completed pages
  useEffect(() => {
    if (!jobId) return;
    
    const poll = async () => {
      try {
        const r = await fetch(`${API}/api/omr/job/${jobId}`);
        if (!r.ok) return;
        const status = await r.json();
        const newDone: number[] = status.done || [];
        
        // Fetch any newly completed pages we don't have
        for (const idx of newDone) {
          if (!pages[idx]) {
            const pr = await fetch(`${API}/api/omr/job/${jobId}/page/${idx}`);
            if (pr.ok) {
              const pageData = await pr.json();
              if (!pageData.error) {
                setPages(p => ({ ...p, [idx]: pageData }));
              }
            }
          }
        }
        setDonePages(newDone);
        
        // Stop polling when all pages done
        if (status.finished) return true;
      } catch { /* ignore */ }
      return false;
    };

    let stopped = false;
    const loop = async () => {
      while (!stopped) {
        const finished = await poll();
        if (finished) break;
        await new Promise(r => setTimeout(r, 2000));
      }
    };
    loop();
    return () => { stopped = true; };
  }, [jobId, pages]);

  const currentPage = pages[page];
  const isLoading = !currentPage && donePages.length < pageCount;

  if (error) {
    return <div style={{ padding: 16, color: '#c00', background: '#fee' }}>{error}</div>;
  }

  return (
    <div style={{ padding: 16 }}>
      {/* Navigation */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16, padding: 12, background: '#f5f5f5', borderRadius: 4 }}>
        <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page <= 0}>◀ Prev</button>
        <span>Page {page + 1} / {pageCount || '?'}</span>
        <button onClick={() => setPage(p => Math.min(pageCount - 1, p + 1))} disabled={page >= pageCount - 1}>Next ▶</button>
        <span style={{ marginLeft: 'auto', color: '#666', fontSize: 14 }}>
          {donePages.length}/{pageCount} processed
        </span>
      </div>

      {/* Page display */}
      <div style={{ position: 'relative', display: 'inline-block' }}>
        {currentPage ? (
          <>
            <img
              ref={imgRef}
              src={`data:image/png;base64,${currentPage.image}`}
              alt={`Page ${page + 1}`}
              style={{ display: 'block', maxWidth: '100%' }}
            />
            <svg
              style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
              viewBox={`0 0 ${currentPage.image_width} ${currentPage.image_height}`}
              preserveAspectRatio="xMinYMin meet"
            >
              {currentPage.measures.map((m, i) => (
                <g key={i}>
                  {/* Marker at barline position */}
                  <circle cx={m.x} cy={m.y} r={8} fill="rgba(34,139,34,0.8)" />
                  <text x={m.x + 12} y={m.y + 6} fill="#960096" fontSize={18} fontWeight="bold">{m.index}</text>
                </g>
              ))}
            </svg>
          </>
        ) : (
          <div style={{ width: 800, height: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f0f0f0', color: '#666' }}>
            {isLoading ? `Processing page ${page + 1}...` : 'Page not available'}
          </div>
        )}
      </div>
    </div>
  );
}
