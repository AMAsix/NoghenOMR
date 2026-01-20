"""
NoghenOMR - Optical Music Recognition Processing
Handles PDF rendering and staff/barline detection
"""

from flask import Blueprint, request, jsonify
import io
import logging
import base64
import threading
import uuid
import time
import tempfile
import os

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

# homr is optional; try import once at module load and fall back gracefully
try:
    import homr
    from homr import main as hm
except Exception as e:
    hm = None
    logging.getLogger(__name__).info('homr not available: %s', e)

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


def _fallback_barline_detector(img_bytes_local, page_idx):
    """A simple fallback that finds systems (vertical whitespace separators)
    and then detects vertical barlines inside each system using projections.
    """
    try:
        if Image is None or np is None:
            return {'error': 'missing_imaging_libs', 'detail': 'PIL or numpy not available', 'page_index': page_idx}
        im_fb = Image.open(io.BytesIO(img_bytes_local)).convert('L')
        arr_fb = np.array(im_fb)
        h_fb, w_fb = arr_fb.shape
        inv_fb = 255 - arr_fb

        # row projections to find separators between systems
        row_sums_fb = inv_fb.mean(axis=1)
        try:
            window = max(3, int(max(1, h_fb * 0.01)))
            kernel = np.ones(window, dtype=float) / float(window)
            row_smooth_fb = np.convolve(row_sums_fb, kernel, mode='same')
        except Exception:
            row_smooth_fb = row_sums_fb

        try:
            whitespace_thresh = float(np.percentile(row_smooth_fb, 25))
        except Exception:
            whitespace_thresh = float(row_smooth_fb.mean())
        sep_mask = row_smooth_fb < whitespace_thresh
        separators_fb = []
        in_sep_fb = False
        sep_start_fb = 0
        for y in range(h_fb):
            if sep_mask[y] and not in_sep_fb:
                in_sep_fb = True
                sep_start_fb = y
            elif not sep_mask[y] and in_sep_fb:
                in_sep_fb = False
                separators_fb.append((sep_start_fb, y))
        if in_sep_fb:
            separators_fb.append((sep_start_fb, h_fb - 1))

        systems_fb = []
        last_fb = 0
        for (s0, s1) in separators_fb:
            if s0 - last_fb > 20:
                systems_fb.append({'bbox': [0, int(last_fb), int(w_fb), int(s0 - last_fb)]})
            last_fb = s1 + 1
        if h_fb - last_fb > 20:
            systems_fb.append({'bbox': [0, int(last_fb), int(w_fb), int(h_fb - last_fb)]})
        if not systems_fb:
            systems_fb = [{'bbox': [0, 0, int(w_fb), int(h_fb)]}]

        # detect column peaks globally then test per-system
        col_sums_fb = inv_fb.mean(axis=0)
        cmean = float(col_sums_fb.mean())
        cstd = float(col_sums_fb.std())
        col_thresh_fb = cmean + cstd * 0.9

        col_peaks_fb = []
        for x in range(1, len(col_sums_fb) - 1):
            if col_sums_fb[x] > col_thresh_fb and col_sums_fb[x] > col_sums_fb[x - 1] and col_sums_fb[x] >= col_sums_fb[x + 1]:
                col_peaks_fb.append(int(x))

        # cluster peaks
        clustered_cols_fb = []
        cluster_fb = []
        for x in sorted(col_peaks_fb):
            if not cluster_fb or x - cluster_fb[-1] <= 4:
                cluster_fb.append(x)
            else:
                clustered_cols_fb.append(int(sum(cluster_fb) / len(cluster_fb)))
                cluster_fb = [x]
        if cluster_fb:
            clustered_cols_fb.append(int(sum(cluster_fb) / len(cluster_fb)))

        measures_fb = []
        for sys_fb in systems_fb:
            sy = int(sys_fb['bbox'][1])
            sh = int(sys_fb['bbox'][3])
            sys_top = sy
            sys_bottom = sy + sh
            sys_measures = []
            for x in clustered_cols_fb:
                col_slice = inv_fb[sys_top:sys_bottom, x]
                if col_slice.size == 0:
                    continue
                dark_frac = float((col_slice > 30).sum()) / float(col_slice.size)
                if dark_frac > 0.18:
                    sys_measures.append(int(x))
            sys_measures = sorted(sys_measures)
            clustered_sys_fb = []
            cl_fb = []
            for x in sys_measures:
                if not cl_fb or x - cl_fb[-1] <= 4:
                    cl_fb.append(x)
                else:
                    clustered_sys_fb.append(int(sum(cl_fb) / len(cl_fb)))
                    cl_fb = [x]
            if cl_fb:
                clustered_sys_fb.append(int(sum(cl_fb) / len(cl_fb)))
            for x in clustered_sys_fb:
                measures_fb.append({'x': int(x), 'y0': int(sys_top), 'y1': int(sys_bottom)})

        image_b64_fb = base64.b64encode(img_bytes_local).decode('ascii')
        return {
            'staves': [],
            'systems': systems_fb,
            'measures': measures_fb,
            'image': image_b64_fb,
            'image_width': int(w_fb),
            'image_height': int(h_fb),
            'page_index': int(page_idx)
        }
    except Exception as e:
        logger.exception('fallback barline detection failed for page %s', page_idx)
        return {'error': 'fallback detection failed', 'detail': str(e), 'page_index': page_idx}


def _process_page_bytes(img_bytes_local, page_idx):
    """Process a single page image bytes and return normalized measures/systems."""
    try:
        from PIL import Image
        import numpy as np
        try:
            from homr import main as hm
        except Exception:
            hm = None

        try:
            im_tmp = Image.open(io.BytesIO(img_bytes_local)).convert('L')
            arr_tmp = np.array(im_tmp)
            h_local, w_local = arr_tmp.shape
        except Exception:
            h_local, w_local = 0, 0

        # if homr available try it first
        if hm is not None:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            try:
                tmpf.write(img_bytes_local)
                tmpf.flush()
                tmpf.close()
                # ProcessingConfig(enable_debug, enable_cache, write_staff_positions, 
                #                  read_staff_positions, selected_staff, use_gpu_inference)
                cfg = hm.ProcessingConfig(False, False, False, False, -1, False)
                try:
                    multi_staffs, pre, debug, title_future = hm.detect_staffs_in_image(tmpf.name, cfg)
                except Exception as e:
                    logger.info('homr detect_staffs_in_image failed: %s', e)
                    try:
                        if hasattr(hm, 'download_weights'):
                            hm.download_weights(use_gpu_inference=False)
                            multi_staffs, pre, debug, title_future = hm.detect_staffs_in_image(tmpf.name, cfg)
                        else:
                            raise
                    except Exception as e2:
                        logger.warning('homr failed: %s', e2)
                        multi_staffs = None
                        pre = None

                if multi_staffs:
                    try:
                        pre_h, pre_w = int(pre.shape[0]), int(pre.shape[1])
                    except Exception:
                        pre_h, pre_w = (h_local or 1), (w_local or 1)
                    scale_x_map = float(w_local) / float(pre_w) if pre_w else 1.0
                    scale_y_map = float(h_local) / float(pre_h) if pre_h else 1.0

                    staves = []
                    systems = []
                    measures = []

                    for multi in multi_staffs:
                        staffs = None
                        for attr in dir(multi):
                            if attr.startswith('_'):
                                continue
                            try:
                                val = getattr(multi, attr)
                            except Exception:
                                continue
                            if isinstance(val, (list, tuple)) and len(val) > 0:
                                if hasattr(val[0], 'min_x') and hasattr(val[0], 'min_y'):
                                    staffs = val
                                    break
                        if not staffs and isinstance(multi, (list, tuple)) and len(multi) > 0 and hasattr(multi[0], 'min_x'):
                            staffs = list(multi)
                        if not staffs:
                            continue

                        for staff in staffs:
                            min_x = int(getattr(staff, 'min_x', 0))
                            min_y = int(getattr(staff, 'min_y', 0))
                            max_x = int(getattr(staff, 'max_x', min_x))
                            max_y = int(getattr(staff, 'max_y', min_y))
                            systems.append({
                                'bbox': [
                                    int(min_x * scale_x_map),
                                    int(min_y * scale_y_map),
                                    int((max_x - min_x) * scale_x_map),
                                    int((max_y - min_y) * scale_y_map)
                                ]
                            })

                            grid = getattr(staff, 'grid', []) or []
                            lines = []
                            if grid:
                                try:
                                    import numpy as _np
                                    for i in range(5):
                                        ys = [getattr(p, 'y', [])[i] for p in grid if hasattr(p, 'y') and len(getattr(p, 'y', [])) > i]
                                        if ys:
                                            lines.append(int(float(_np.median(ys))))
                                except Exception:
                                    lines = []

                            lines_mapped = [int(y * scale_y_map) for y in lines]
                            staves.append({
                                'lines': lines_mapped,
                                'bbox': [
                                    int(min_x * scale_x_map),
                                    int(min_y * scale_y_map),
                                    int((max_x - min_x) * scale_x_map),
                                    int((max_y - min_y) * scale_y_map)
                                ]
                            })

                            # Detect barlines inside the staff bbox
                            try:
                                staff_bbox = [
                                    int(min_x * scale_x_map),
                                    int(min_y * scale_y_map),
                                    int((max_x - min_x) * scale_x_map),
                                    int((max_y - min_y) * scale_y_map)
                                ]
                                local_measures = _detect_barlines_in_region(arr_tmp, staff_bbox)
                                for mm in local_measures:
                                    measures.append({'x': int(mm['x']), 'y0': int(mm['y0']), 'y1': int(mm['y1'])})
                            except Exception:
                                pass

                    def _uniq(lst, key):
                        seen = set()
                        out = []
                        for item in lst:
                            k = tuple(item.get(key) if isinstance(item.get(key), (list, tuple)) else (item.get(key),))
                            if k in seen:
                                continue
                            seen.add(k)
                            out.append(item)
                        return out

                    systems = _uniq(systems, 'bbox')
                    staves = _uniq(staves, 'bbox')
                    measures = _uniq(measures, 'x')

                    image_b64 = base64.b64encode(img_bytes_local).decode('ascii')
                    return {
                        'staves': staves,
                        'systems': systems,
                        'measures': measures,
                        'image': image_b64,
                        'image_width': int(w_local),
                        'image_height': int(h_local),
                        'page_index': int(page_idx)
                    }
            finally:
                try:
                    os.unlink(tmpf.name)
                except Exception:
                    pass

        # fallback to simple detector
        return _fallback_barline_detector(img_bytes_local, page_idx)
    except Exception as e:
        return {'error': 'processing failed', 'detail': str(e), 'page_index': page_idx}


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
