import { useState } from 'react';
import OMRCanvas from './OMRCanvas';

export default function App() {
  const [file, setFile] = useState<File | null>(null);

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 28, marginBottom: 8 }}>🎼 NoghenOMR</h1>
        <p style={{ color: '#666' }}>Upload a scanned sheet-music PDF to detect barlines.</p>
      </header>

      <div style={{ marginBottom: 16, padding: 16, background: '#fff', borderRadius: 8, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>Select PDF:</label>
        <input type="file" accept="application/pdf" onChange={e => setFile(e.target.files?.[0] ?? null)} />
      </div>

      <div style={{ background: '#fff', borderRadius: 8, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        {file ? (
          <OMRCanvas file={file} />
        ) : (
          <div style={{ padding: 48, textAlign: 'center', color: '#999', border: '2px dashed #ddd', borderRadius: 8 }}>
            Upload a PDF to begin
          </div>
        )}
      </div>
    </div>
  );
}
