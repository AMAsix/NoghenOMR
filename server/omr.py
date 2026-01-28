"""
NoghenOMR - Optical Music Recognition Processing
Minimal Flask API for PDF barline detection
"""

from flask import Blueprint, request, jsonify
import base64
import io
import logging
import threading
import uuid

import numpy as np
from PIL import Image
import fitz

from .barline_detection import (
    binarize,
    detect_vertical_candidates,
    keep_longest_vertical_lines,
    merge_close_vertical_lines,
    enumerate_vertical_lines,
)

logger = logging.getLogger(__name__)
omr_bp = Blueprint('omr', __name__, url_prefix='/api/omr')

# In-memory job store
_jobs = {}


def _process_page(pdf_bytes: bytes, page_idx: int) -> dict:
    """Render a PDF page and detect barlines."""
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes('png')
    
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    rgb = np.array(im)
    h, w = rgb.shape[:2]

    # Barline detection pipeline - all logic in barline_detection.py
    bw = binarize(rgb)
    vraw = detect_vertical_candidates(bw)
    filtered = keep_longest_vertical_lines(vraw)
    barlines = merge_close_vertical_lines(filtered)
    coords = enumerate_vertical_lines(rgb, barlines, return_coords=True)

    # coords is list of (index, x, y)
    measures = [{'index': idx, 'x': x, 'y': y} for idx, x, y in coords]

    return {
        'page_index': page_idx,
        'measures': measures,
        'image_width': w,
        'image_height': h,
        'image': base64.b64encode(img_bytes).decode('utf-8'),
    }


@omr_bp.route('/start_job', methods=['POST'])
def start_job():
    """Start async job to process PDF. Returns job_id and page_count."""
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    
    pdf_bytes = request.files['file'].read()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        page_count = doc.page_count
    except Exception as e:
        return jsonify({'error': 'Invalid PDF', 'detail': str(e)}), 400

    job_id = str(uuid.uuid4())
    job = {
        'pdf_bytes': pdf_bytes,
        'total': page_count,
        'results': {},
        'current': 0,
        'done': False,
        'lock': threading.Lock(),
    }
    _jobs[job_id] = job

    def worker():
        for i in range(page_count):
            with job['lock']:
                job['current'] = i
            try:
                result = _process_page(pdf_bytes, i)
            except Exception as e:
                logger.exception('Page %d failed', i)
                result = {'page_index': i, 'error': str(e)}
            with job['lock']:
                job['results'][i] = result
        with job['lock']:
            job['done'] = True

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'job_id': job_id, 'page_count': page_count})


@omr_bp.route('/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get job status and list of completed pages."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    with job['lock']:
        return jsonify({
            'page_count': job['total'],
            'done': sorted(job['results'].keys()),
            'current': job['current'],
            'finished': job['done'],
        })


@omr_bp.route('/job/<job_id>/page/<int:page_idx>', methods=['GET'])
def job_page(job_id, page_idx):
    """Get result for a specific page."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job'}), 404
    with job['lock']:
        result = job['results'].get(page_idx)
    if result is None:
        return jsonify({'status': 'pending'}), 202
    return jsonify(result)
