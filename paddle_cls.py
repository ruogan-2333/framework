"""paddle_cls.py

PaddleOCR wrapper (kept because OCR is required for icon-heavy UIs).

This module is intentionally self-contained (no utils.py dependency).

Performance notes:
- Uses a global OCR singleton to avoid repeated model initialization.
- The UI pipeline should cache OCR outputs by screenshot hash and only call OCR
  when needed.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import contextmanager
from typing import List, Tuple

import cv2 as cv
import numpy as np

logger = logging.getLogger(__name__)


from utils import time_consumed


def cv2PutText(img, text: str, org, font, font_scale, color, thickness=1):
    """Readable text overlay helper (outline + fill)."""
    try:
        x, y = int(org[0]), int(org[1])
        cv.putText(img, text, (x, y), font, font_scale, (0, 0, 0),
                   thickness + 3, cv.LINE_AA)
        cv.putText(img, text, (x, y), font, font_scale, color,
                   thickness, cv.LINE_AA)
    except Exception:
        pass


@contextmanager
def keep_root_logging():
    """Prevent PaddleOCR from reconfiguring global logging."""
    root = logging.getLogger()
    saved = (root.level, root.handlers[:], root.filters[:], root.disabled)
    try:
        yield
    finally:
        level, handlers, filters, disabled = saved
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in handlers:
            root.addHandler(h)
        for f in root.filters[:]:
            root.removeFilter(f)
        for f in filters:
            root.addFilter(f)
        root.setLevel(level)
        root.disabled = disabled


_OCR = None


def _get_ocr():
    """Create (or reuse) PaddleOCR predictor. Heavy init => cache singleton."""
    global _OCR
    if _OCR is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("DISABLE_AUTO_LOGGING_CONFIG", "1")

        original_level = logging.getLogger().getEffectiveLevel()
        with keep_root_logging():
            from paddleocr import PaddleOCR

            _OCR = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=2000,
                text_det_thresh=0.3,
                text_det_box_thresh=0.3,
                text_det_unclip_ratio=2.0,
                text_rec_score_thresh=0.5,
            )
        logging.getLogger().setLevel(original_level)
    return _OCR


class PaddleOCRClient:
    """Return OCR detections: (x, y, w, h, text, conf)."""

    @staticmethod
    @time_consumed
    def ocr(base64_screenshot: str, debug_output: bool = False) -> List[Tuple[int, int, int, int, str, float]]:
        logging.getLogger("ppocr").setLevel(logging.DEBUG if debug_output else logging.ERROR)
        logging.getLogger("paddlex").setLevel(logging.DEBUG if debug_output else logging.ERROR)

        ocr = _get_ocr()

        img = cv.imdecode(
            np.frombuffer(base64.b64decode(base64_screenshot), np.uint8),
            cv.IMREAD_UNCHANGED,
        )
        if img is None:
            raise ValueError("Failed to decode screenshot image")

        # Force to BGR
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        vis = img.copy() if debug_output else None
        result = ocr.ocr(img, cls=False)

        elements: List[Tuple[int, int, int, int, str, float]] = []
        lines = result[0] if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list) else result
        if not isinstance(lines, list):
            lines = []

        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            poly_raw, rec = line[0], line[1]
            if not poly_raw:
                continue

            poly = np.asarray(poly_raw, dtype=np.float32).reshape(-1, 2)
            if poly.size < 2:
                continue
            x1, y1 = poly.min(axis=0)
            x2, y2 = poly.max(axis=0)

            text = ""
            conf = 0.0
            if isinstance(rec, (list, tuple)):
                if len(rec) >= 1:
                    text = str(rec[0])
                if len(rec) >= 2:
                    try:
                        conf = float(rec[1])
                    except Exception:
                        conf = 0.0
            else:
                text = str(rec)

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            elements.append((x1, y1, w, h, text, conf))

            if debug_output and vis is not None:
                cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2PutText(
                    vis,
                    text[:40],
                    (x1, max(0, y1 - 10)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    1,
                )

        if debug_output and vis is not None:
            cv.imwrite("screenshot-ocr.png", vis)

        return elements


__all__ = ["PaddleOCRClient"]
