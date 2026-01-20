# NoghenOMR

Optical Music Recognition (OMR) web application for detecting staves and measure lines in scanned sheet music PDFs.

## Features

- Upload PDF files of scanned sheet music
- Automatic detection of music systems (staff groups)
- Staff line detection
- Measure/barline detection
- Multi-page support with navigation
- Visual overlay showing detected elements

## Prerequisites

- Node.js 18+
- Python 3.10+
- Optional: homr library for advanced OMR (requires working scipy)

## Quick Start

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Create Python virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run development servers

```bash
npm run dev
```

This starts:
- Frontend (Vite): http://localhost:5173
- Backend (Flask): http://localhost:4000

## Usage

1. Open http://localhost:5173 in your browser
2. Upload a PDF file containing scanned sheet music
3. The system will process the pages and overlay detected elements:
   - **Blue boxes**: Detected music systems
   - **Red lines**: Detected staff lines
   - **Green lines**: Detected measure/bar lines

## Architecture

### Frontend (React + TypeScript)
- `src/App.tsx` - Main application component
- `src/OMRCanvas.tsx` - Canvas component for displaying PDF pages with detection overlays

### Backend (Python + Flask)
- `server/main.py` - Flask application entry point
- `server/omr.py` - OMR processing logic (PDF rendering, staff/barline detection)

## Detection Algorithm

The application uses a projection-based algorithm for detecting musical elements:

1. **PDF Rendering**: PyMuPDF renders PDF pages to high-resolution PNG images
2. **System Detection**: Horizontal projection analysis finds gaps between music systems
3. **Staff Line Detection**: Identifies the characteristic 5-line patterns of musical staves
4. **Barline Detection**: Vertical projection analysis finds measure dividers

When the `homr` library is available, it provides enhanced detection using machine learning models.

## License

MIT
