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

### 4. Apply homr patches (required for numpy 2.x)

homr has compatibility issues with numpy 2.x. Run the patch script after installing:

```bash
python patches/apply_homr_numpy2_patches.py
```

> **Note**: Re-run this script whenever you reinstall homr.

### 5. Run development servers

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

The application uses the `homr` library for detecting musical elements:

1. **PDF Rendering**: PyMuPDF renders PDF pages to high-resolution PNG images
2. **OMR Processing**: homr uses machine learning models to detect:
   - Staff lines and systems
   - Noteheads and stems
   - Barlines and measure boundaries
   - Clefs and key signatures

## License

MIT
