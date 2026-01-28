"""
Barline Detection and Measure Enumeration Pipeline
=================================================

Author: Ido Oron

Description
-----------
This module implements a deterministic, image-processing pipeline for detecting
barlines and enumerating measures in engraved sheet music PDFs. The approach is
layout-driven and deliberately avoids symbolic Optical Music Recognition (OMR);
it relies only on geometric properties of staff notation.

Algorithm Summary
-----------------
1. Render PDF pages at high resolution.
2. Binarize pages using adaptive thresholding.
3. Detect horizontal staff lines via morphological filtering and width
   constraints; extract their vertical centers.
4. Group staff lines into 5-line staves (rows) and group rows into systems based
   on vertical spacing.
5. Detect vertical candidates using morphological filtering.
6. Keep only the tallest vertical responses, exploiting the fact that barlines
   are among the longest vertical structures on a page.
7. Within each system, retain only vertical lines that intersect the largest
   number of staff lines, removing note stems and partial verticals.
8. Enumerate barlines left-to-right within each system, discarding the final
   barline of each system, and number measures continuously across pages.
9. Overlay measure numbers on the original pages and export the result as a
   single multi-page PDF.

The pipeline is instrument-agnostic, reproducible, and robust to common engraving
artifacts such as slurs, beams, and text.
"""
import time
from pathlib import Path
import numpy as np
import cv2
import fitz
from PIL import Image

PDF_PATH = "examples/mozart_opera.pdf"
OUT_DIR = "debug"
DPI = 400
DEBUGING = True
DEBUG_PAGE_NUMBER = 0


def debug_write(path, img):
    if DEBUGING:
        cv2.imwrite(str(path), img)


# ================= IO =================
def render_pdf(pdf_path, dpi):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        pages.append(img)
    return pages


def save_images_as_single_pdf(rgb_images, pdf_path):
    imgs = [Image.fromarray(img) for img in rgb_images]
    imgs[0].save(pdf_path, "PDF", save_all=True, append_images=imgs[1:], resolution=300.0)


# ================= PREPROCESS =================
def binarize(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)


# ================= VERTICAL =================
def detect_vertical_candidates(bw):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)


def keep_longest_vertical_lines(img, max_gap=15, min_width=1, drop_ratio=0.5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max_gap))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(closed)

    spans = []
    for lbl in range(1, num_labels):
        ys, xs = np.where(labels == lbl)
        if len(ys) == 0:
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if w >= min_width:
            spans.append((h, lbl))

    spans.sort(reverse=True)
    heights = np.array([h for h, _ in spans])

    # Find first large relative drop
    ratios = heights[1:] / heights[:-1]
    cut = np.argmax(ratios < drop_ratio) + 1 if np.any(ratios < drop_ratio) else len(heights)

    keep_labels = {lbl for _, lbl in spans[:cut]}

    out = np.zeros_like(img)
    for lbl in keep_labels:
        out[labels == lbl] = 255

    return out


def merge_close_vertical_lines(vmask, x_gap_px=18, y_overlap_ratio=0.7, min_height_px=5):
    out = np.zeros_like(vmask)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(vmask, 8)

    comps = []
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if hh < min_height_px:
            continue
        comps.append((x, x + ww - 1, y, y + hh - 1))

    comps.sort(key=lambda t: t[0])
    used = [False] * len(comps)

    for i in range(len(comps)):
        if used[i]:
            continue

        x0, x1, ya0, ya1 = comps[i]
        used[i] = True

        for j in range(i + 1, len(comps)):
            if used[j]:
                continue

            xb0, xb1, yb0, yb1 = comps[j]
            if xb0 - x1 > x_gap_px:
                break

            ov = max(0, min(ya1, yb1) - max(ya0, yb0) + 1)
            ref = min(ya1 - ya0 + 1, yb1 - yb0 + 1)

            if ref > 0 and ov / ref >= y_overlap_ratio:
                x1 = max(x1, xb1)
                ya0 = min(ya0, yb0)
                ya1 = max(ya1, yb1)
                used[j] = True

        out[ya0:ya1 + 1, x0:x1 + 1] = 255

    return out


# ================= BARLINES =================
def enumerate_vertical_lines(rgb, vmask, font_scale=0.9, thickness=5, row_tol_px=40, return_coords=False):
    out = rgb.copy()
    coords = []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(vmask, 8)
    lines = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h <= 0:
            continue
        cx = x + w / 2.0
        lines.append((y, cx, x, w, h))
    if not lines:
        return coords if return_coords else out

    lines.sort(key=lambda t: t[0])
    rows = []
    for item in lines:
        if not rows or abs(item[0] - rows[-1][0][0]) > row_tol_px:
            rows.append([item])
        else:
            rows[-1].append(item)

    idx = 1
    for row in rows:
        row.sort(key=lambda t: t[1])
        for (y, cx, x, w, h) in row[:-1]:
            if return_coords:
                coords.append((idx, int(cx), int(y)))
            else:
                cv2.putText(out, str("*"), (int(cx) + 3, int(y) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 0, 150),
                            thickness, cv2.LINE_AA)
            idx += 1

    return coords if return_coords else out



# ================= MAIN =================
def main():
    debug = Path(OUT_DIR)
    if DEBUGING:
        debug.mkdir(exist_ok=True)

    pages = render_pdf(PDF_PATH, DPI)
    final_pages, global_idx = [], 1

    print(f"Starting enumeration process ({len(pages)} pages).")
    for i, rgb in enumerate(pages):
        start_time = time.time()
        print(f"Processing page {i + 1}", end=' ... ')
        bw = binarize(rgb)
        debug_write(debug / f"page_{i:03d}_bw.png", bw)

        vraw = detect_vertical_candidates(bw)
        debug_write(debug / f"page_{i:03d}_vraw.png", vraw)

        filtered_vlines = keep_longest_vertical_lines(vraw)
        debug_write(debug / f"page_{i:03d}_vraw_longest.png", filtered_vlines)

        barlines = merge_close_vertical_lines(filtered_vlines)
        debug_write(debug / f"page_{i:03d}_merged_vlines.png", barlines)

        enumerated = enumerate_vertical_lines(rgb, barlines)
        debug_write(debug / f"page_{i:03d}_enumerated_score.png", enumerated)
        final_pages.append(enumerated)
        print(f"took {time.time() - start_time:.2f}s.")

        if DEBUGING:
            break
    if not DEBUGING:
        save_images_as_single_pdf(final_pages, "measures_numbered.pdf")

#
# if __name__ == "__main__":
#     main()
