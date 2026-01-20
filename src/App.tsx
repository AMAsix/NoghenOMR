import { useState } from 'react';
import OMRCanvas from './OMRCanvas';

export default function App() {
  const [file, setFile] = useState<File | null>(null);

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 28, marginBottom: 8 }}>🎼 NoghenOMR</h1>
        <p style={{ color: '#666' }}>
          Optical Music Recognition - Upload a scanned sheet-music PDF to detect staves and measure lines.
        </p>
      </header>

      <div style={{ 
        marginBottom: 16, 
        padding: 16, 
        background: '#fff', 
        borderRadius: 8,
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
      }}>
        <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
          Select PDF file:
        </label>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          style={{ fontSize: 16 }}
        />
      </div>

      <div style={{ 
        background: '#fff', 
        borderRadius: 8, 
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        overflow: 'hidden'
      }}>
        {file ? (
          <OMRCanvas file={file} />
        ) : (
          <div style={{ 
            padding: 48, 
            textAlign: 'center', 
            color: '#999',
            borderRadius: 8,
            border: '2px dashed #ddd'
          }}>
            <p style={{ fontSize: 18 }}>No file selected</p>
            <p style={{ marginTop: 8 }}>Upload a PDF to begin OMR processing</p>
          </div>
        )}
      </div>

      <footer style={{ marginTop: 24, padding: 16, textAlign: 'center', color: '#999', fontSize: 14 }}>
        <p>
          <strong>Legend:</strong>{' '}
          <span style={{ color: 'rgba(0,128,255,0.8)' }}>■ Systems</span>{' · '}
          <span style={{ color: 'rgba(220,20,60,0.8)' }}>— Staff Lines</span>{' · '}
          <span style={{ color: 'rgba(34,139,34,0.9)' }}>│ Measure Lines</span>
        </p>
      </footer>
    </div>
  );
}
