"""
NoghenOMR - Optical Music Recognition Processing
Handles PDF rendering and staff/barline detection
"""

from flask import Blueprint, request, jsonify
import io
import logging
import threading
import uuid
import time
from barline_detection import (
    binarize,
    detect_vertical_candidates,
    keep_longest_vertical_lines,
    merge_close_vertical_lines,
    enumerate_vertical_lines,
)

# Core imaging / numeric imports at module scope (fail fast / log if missing)
try:
    from PIL import Image
except Exception as e:
    Image = None
    logging.getLogger(__name__).warning('Pillow (PIL) unavailable: %s', e)

try:
    import numpy as np
except Exception as e:
    np = None
    logging.getLogger(__name__).warning('numpy unavailable: %s', e)

try:
    import fitz  # PyMuPDF
except Exception as e:
    fitz = None
    logging.getLogger(__name__).warning('PyMuPDF (fitz) unavailable: %s', e)

# homr is required for OMR detection
try:
    import homr
    from homr import main as hm
except Exception as e:
    hm = None
    logging.getLogger(__name__).error('homr is required but not available: %s', e)

logger = logging.getLogger(__name__)

omr_bp = Blueprint('omr', __name__, url_prefix='/api/omr')

# Simple in-memory job store for async batch processing
_jobs = {}


def _detect_barlines_in_region(arr_gray, bbox, min_vertical_coverage=0.6, col_gap=4):
    """Detect vertical barline x-positions inside bbox on a grayscale numpy array.
    Returns list of absolute x positions (int) and y0,y1 covering the bbox vertical span.
    """
    try:
        import numpy as np
        x0, y0, w, h = bbox
        x0 = int(max(0, x0))
        y0 = int(max(0, y0))
        w = int(max(0, w))
        h = int(max(0, h))
        if w <= 0 or h <= 0:
            return []
        band = arr_gray[y0:y0 + h, x0:x0 + w]
        if band.size == 0:
            return []
        inv = 255 - band
        # column darkness metric
        col_dark = inv.mean(axis=0)
        cmean = float(col_dark.mean())
        cstd = float(col_dark.std())
        # threshold tuned to prefer tall verticals
        thresh = cmean + max(3.0, 0.6 * cstd)

        peaks = []
        for i in range(1, len(col_dark) - 1):
            if col_dark[i] > thresh and col_dark[i] > col_dark[i - 1] and col_dark[i] >= col_dark[i + 1]:
                peaks.append(i)

        # cluster nearby peaks into single columns
        clustered = []
        cluster = []
        for p in peaks:
            if not cluster or p - cluster[-1] <= col_gap:
                cluster.append(p)
            else:
                clustered.append(int(sum(cluster) / len(cluster)))
                cluster = [p]
        if cluster:
            clustered.append(int(sum(cluster) / len(cluster)))

        results = []
        for cx in clustered:
            col = inv[:, cx]
            dark_mask = col > 30
            if dark_mask.sum() == 0:
                continue
            # largest contiguous dark segment length
            max_seg = 0
            cur = 0
            for v in dark_mask:
                if v:
                    cur += 1
                    if cur > max_seg:
                        max_seg = cur
                else:
                    cur = 0
            if max_seg / float(h) >= float(min_vertical_coverage):
                # convert local x to absolute image x
                abs_x = x0 + int(cx)
                results.append({'x': abs_x, 'y0': y0, 'y1': y0 + h})
        return results
    except Exception:
        return []


def _process_page_bytes(img_bytes_local, page_idx):
    try:
        im = Image.open(io.BytesIO(img_bytes_local)).convert('RGB')
        rgb = np.array(im)
        h, w, _ = rgb.shape

        bw = binarize(rgb)
        vraw = detect_vertical_candidates(bw)
        filtered = keep_longest_vertical_lines(vraw)
        barlines = merge_close_vertical_lines(filtered)

        coords = enumerate_vertical_lines(rgb, barlines, return_coords=True)

        return {
            'page_index': int(page_idx),
            'measures': [
                {'index': i, 'x': x, 'y': y}
                for (i, x, y) in coords
            ],
            'image_width': int(w),
            'image_height': int(h),
        }

    except Exception as e:
        logger.exception('Noghen barline processing failed for page %s', page_idx)
        return {
            'error': 'processing_failed',
            'detail': str(e),
            'page_index': int(page_idx)
        }


@omr_bp.route('/process', methods=['POST'])
def process_pdf():
    """Accept a PDF file, render the requested page and run OMR detection."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Missing file field'}), 400

        f = request.files['file']
        pdf_bytes = f.read()

        try:
            import fitz
        except Exception as e:
            logger.exception('Missing PyMuPDF')
            return jsonify({'error': 'Server missing PyMuPDF', 'detail': str(e)}), 500

        # page index
        page_index = 0
        try:
            page_str = request.form.get('page') or request.args.get('page')
            if page_str is not None:
                page_index = max(0, int(page_str))
        except Exception:
            page_index = 0

        # batch flag
        batch_flag = False
        try:
            b = request.form.get('batch') or request.args.get('batch')
            if b is not None and str(b).lower() in ('1', 'true', 'yes'):
                batch_flag = True
        except Exception:
            batch_flag = False

        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        page_count = doc.page_count
        if page_index >= page_count:
            page_index = max(0, page_count - 1)

        if batch_flag:
            results = []
            for idx in range(page_count):
                page = doc.load_page(idx)
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes('png')
                res = _process_page_bytes(img_bytes, idx)
                results.append(res)
            return jsonify({'pages': results, 'page_count': int(page_count)})

        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes('png')
        single_res = _process_page_bytes(img_bytes, page_index)
        single_res['page_count'] = int(page_count)
        return jsonify(single_res)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception('OMR processing failed')
        return jsonify({'error': str(e), 'traceback': tb}), 500


@omr_bp.route('/start_job', methods=['POST'])
def start_job():
    """Start an async job to process all pages. Returns a job_id and page_count."""
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file field'}), 400
    f = request.files['file']
    pdf_bytes = f.read()
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        page_count = doc.page_count
    except Exception as e:
        logger.exception('Failed opening PDF')
        return jsonify({'error': 'invalid pdf', 'detail': str(e)}), 400

    job_id = str(uuid.uuid4())
    job = {
        'id': job_id,
        'total': int(page_count),
        'results': {},
        'queue': list(range(int(page_count))),
        'processing': False,
        'current': None,
        'lock': threading.Lock(),
        'pdf_bytes': pdf_bytes,
        'created': time.time()
    }
    _jobs[job_id] = job

    def worker(j):
        j['processing'] = True
        try:
            import fitz
        except Exception:
            pass
        while True:
            with j['lock']:
                if not j['queue']:
                    j['current'] = None
                    j['processing'] = False
                    break
                next_idx = j['queue'].pop(0)
                j['current'] = next_idx
            try:
                try:
                    doc = fitz.open(stream=j['pdf_bytes'], filetype='pdf')
                    page = doc.load_page(next_idx)
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes('png')
                except Exception as re:
                    res = {'error': 'render_failed', 'detail': str(re), 'page_index': next_idx}
                    with j['lock']:
                        j['results'][next_idx] = res
                    continue

                res = _process_page_bytes(img_bytes, next_idx)
                with j['lock']:
                    j['results'][next_idx] = res
            except Exception as we:
                with j['lock']:
                    j['results'][next_idx] = {'error': 'worker_failed', 'detail': str(we), 'page_index': next_idx}
                continue

    t = threading.Thread(target=worker, args=(job,), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'page_count': int(page_count)})


@omr_bp.route('/job/<job_id>/status', methods=['GET'])
def job_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'unknown job'}), 404
    with job['lock']:
        done = sorted(list(job['results'].keys()))
        return jsonify({
            'job_id': job_id,
            'page_count': int(job['total']),
            'done': done,
            'processing': bool(job['processing']),
            'current': job['current']
        })


@omr_bp.route('/job/<job_id>/page/<int:page_idx>', methods=['GET'])
def job_get_page(job_id, page_idx):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'unknown job'}), 404
    with job['lock']:
        res = job['results'].get(page_idx)
        if res is None:
            return jsonify({'status': 'pending'}), 202
        return jsonify(res)


@omr_bp.route('/job/<job_id>/prioritize', methods=['POST'])
def job_prioritize(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'unknown job'}), 404
    try:
        page = int(request.form.get('page') or request.args.get('page'))
    except Exception:
        return jsonify({'error': 'missing page param'}), 400
    with job['lock']:
        if page in job['results']:
            return jsonify({'status': 'already_done'}), 200
        if job['current'] == page:
            return jsonify({'status': 'currently_processing'}), 200
        if page in job['queue']:
            job['queue'].remove(page)
            job['queue'].insert(0, page)
            return jsonify({'status': 'prioritized'}), 200
        else:
            return jsonify({'status': 'not_found'}), 404


if __name__ == '__main__':
    with open("../examples/mozart_opera.pdf", "rb") as f:
        pdf = f.read()

    doc = fitz.open(stream=pdf, filetype="pdf")
    pix = doc.load_page(0).get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")

    res = _process_page_bytes(img_bytes, 0)
    print(res)
