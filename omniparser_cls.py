"""In-process OmniParser client for framework visual UI parsing.

Inputs:
- A screenshot image path or a screenshot base64 string.
- An output directory where parser artifacts should be written.

Outputs:
- ``overlay.png`` with numbered OmniParser boxes.
- ``parsed_elements.json`` with OmniParser element records.
- ``label_coordinates.json`` with overlay label coordinates.
- ``summary.json`` with timing and artifact metadata.

Function:
- Loads the external OmniParser models once inside the framework Python process,
  then reuses those model objects for later screenshots. This file is intended
  to be tested independently before replacing the current subprocess backend in
  ``ui_cls.py``.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image


OMNIPARSER_ROOT = Path(r"F:\workplace\external_tools\OmniParser")
"""Path to the local external OmniParser checkout used by this framework."""

_YOLO_MODEL: Any = None
"""Cached OmniParser icon detection model, loaded on the first parse call."""

_CAPTION_MODEL_PROCESSOR: Any = None
"""Cached OmniParser Florence caption model/processor pair, loaded on the first parse call."""

_OMNIPARSER_UTILS: dict[str, Any] | None = None
"""Cached OmniParser utility functions imported from the external checkout."""

_MODEL_LOCK = threading.Lock()
"""Lock that prevents concurrent first-time model initialization in one process."""


def _ensure_omniparser_on_path() -> None:
    """Ensure Python can import OmniParser utility modules.

    Input:
    - No arguments; uses the module-level ``OMNIPARSER_ROOT`` path.

    Output:
    - None.

    Function:
    - Validates the external OmniParser checkout and prepends it to ``sys.path``
      so imports like ``from util.utils import ...`` resolve correctly.
    """

    if not OMNIPARSER_ROOT.exists():
        raise FileNotFoundError(f"OmniParser root not found: {OMNIPARSER_ROOT}")
    root_str = str(OMNIPARSER_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _get_omniparser_utils() -> dict[str, Any]:
    """Import and cache OmniParser utility functions.

    Input:
    - No arguments.

    Output:
    - Dictionary containing ``check_ocr_box``, ``get_caption_model_processor``,
      ``get_som_labeled_img``, and ``get_yolo_model``.

    Function:
    - Keeps external imports in one place so the rest of the framework can call
      this module without knowing OmniParser's internal package layout.
    """

    global _OMNIPARSER_UTILS
    if _OMNIPARSER_UTILS is not None:
        return _OMNIPARSER_UTILS

    _ensure_omniparser_on_path()
    from util.utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model

    _OMNIPARSER_UTILS = {
        "check_ocr_box": check_ocr_box,
        "get_caption_model_processor": get_caption_model_processor,
        "get_som_labeled_img": get_som_labeled_img,
        "get_yolo_model": get_yolo_model,
    }
    return _OMNIPARSER_UTILS


def _ensure_models() -> tuple[Any, Any]:
    """Load OmniParser models once and return cached instances.

    Input:
    - No arguments.

    Output:
    - Tuple ``(yolo_model, caption_model_processor)``.

    Function:
    - Initializes the YOLO icon detector and Florence caption model on the first
      call, then returns the same objects for subsequent screenshots.
    """

    global _YOLO_MODEL, _CAPTION_MODEL_PROCESSOR
    if _YOLO_MODEL is not None and _CAPTION_MODEL_PROCESSOR is not None:
        return _YOLO_MODEL, _CAPTION_MODEL_PROCESSOR

    with _MODEL_LOCK:
        if _YOLO_MODEL is not None and _CAPTION_MODEL_PROCESSOR is not None:
            return _YOLO_MODEL, _CAPTION_MODEL_PROCESSOR

        utils = _get_omniparser_utils()
        _YOLO_MODEL = utils["get_yolo_model"](
            model_path=str(OMNIPARSER_ROOT / "weights" / "icon_detect" / "model.pt")
        )
        _CAPTION_MODEL_PROCESSOR = utils["get_caption_model_processor"](
            model_name="florence2",
            model_name_or_path=str(OMNIPARSER_ROOT / "weights" / "icon_caption_florence"),
        )
        return _YOLO_MODEL, _CAPTION_MODEL_PROCESSOR


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload with UTF-8 encoding.

    Input:
    - ``path``: target JSON file path.
    - ``payload``: JSON-serializable object.

    Output:
    - None.

    Function:
    - Centralizes JSON writing settings so all OmniParser artifacts are readable
      and stable for local debugging.
    """

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _decode_base64_image(screenshot_b64: str, image_path: Path) -> None:
    """Decode a screenshot base64 string to a PNG file.

    Input:
    - ``screenshot_b64``: raw base64 string, with or without data URI prefix.
    - ``image_path``: target PNG path.

    Output:
    - None.

    Function:
    - Converts the framework's in-memory screenshot representation into the file
      path form expected by OmniParser utilities.
    """

    b64 = str(screenshot_b64 or "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    padding = "=" * (-len(b64) % 4)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(base64.b64decode(b64 + padding, validate=False))


class OmniParserClient:
    """Public in-process OmniParser client used by tests and future workflow integration.

    Input:
    - Screenshot image paths or screenshot base64 strings.

    Output:
    - Dictionaries containing parsed elements, label coordinates, timing summary,
      and output artifact paths.

    Function:
    - Provides a stable framework-side API that hides OmniParser model loading
      and artifact writing details.
    """

    @staticmethod
    def warmup() -> dict[str, Any]:
        """Load OmniParser models without parsing a screenshot.

        Input:
        - No arguments.

        Output:
        - Timing dictionary with ``model_seconds`` and ``already_loaded``.

        Function:
        - Lets the workflow pay the heavy OmniParser model initialization cost at
          run startup instead of on the first XML-unreliable UI.
        """

        already_loaded = _YOLO_MODEL is not None and _CAPTION_MODEL_PROCESSOR is not None
        start = time.perf_counter()
        _ensure_models()
        return {
            "already_loaded": bool(already_loaded),
            "model_seconds": round(time.perf_counter() - start, 3),
        }

    @staticmethod
    def parse_image(
        image_path: str,
        out_dir: str,
        *,
        use_ocr: bool = True,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        imgsz: int = 640,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Parse one screenshot image with in-process OmniParser.

        Input:
        - ``image_path``: PNG/JPG screenshot path.
        - ``out_dir``: directory where overlay and JSON artifacts are written.
        - Parser threshold and batch-size settings matching the subprocess smoke
          test defaults.

        Output:
        - Dictionary with ``parsed_elements``, ``label_coordinates``, ``summary``,
          and artifact paths.

        Function:
        - Loads cached OmniParser models, optionally runs OCR, runs icon
          detection/captioning, and writes inspectable artifacts to disk.
        """

        image = Path(image_path).resolve()
        if not image.exists():
            raise FileNotFoundError(f"Screenshot image not found: {image}")

        output = Path(out_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)

        total_start = time.perf_counter()
        model_start = time.perf_counter()
        yolo_model, caption_model_processor = _ensure_models()
        model_seconds = time.perf_counter() - model_start

        utils = _get_omniparser_utils()
        pil_image = Image.open(image).convert("RGB")
        box_overlay_ratio = max(pil_image.size) / 3200
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }

        ocr_text: list[str] = []
        ocr_bbox: list[Any] = []
        ocr_seconds = 0.0
        if use_ocr:
            ocr_start = time.perf_counter()
            (ocr_text, ocr_bbox), _ = utils["check_ocr_box"](
                pil_image,
                display_img=False,
                output_bb_format="xyxy",
                easyocr_args={"paragraph": False, "text_threshold": 0.8},
                use_paddleocr=False,
            )
            ocr_seconds = time.perf_counter() - ocr_start

        parse_start = time.perf_counter()
        overlay_b64, label_coordinates, parsed_content_list = utils["get_som_labeled_img"](
            pil_image,
            yolo_model,
            BOX_TRESHOLD=float(box_threshold),
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=ocr_text,
            iou_threshold=float(iou_threshold),
            imgsz=int(imgsz),
            batch_size=int(batch_size),
        )
        parse_seconds = time.perf_counter() - parse_start

        overlay = Image.open(io.BytesIO(base64.b64decode(overlay_b64)))
        overlay_path = output / "overlay.png"
        parsed_path = output / "parsed_elements.json"
        coords_path = output / "label_coordinates.json"
        summary_path = output / "summary.json"

        overlay.save(overlay_path)
        _write_json(parsed_path, parsed_content_list)
        _write_json(coords_path, label_coordinates)

        summary = {
            "image": str(image),
            "omniparser_root": str(OMNIPARSER_ROOT),
            "out_dir": str(output),
            "use_ocr": bool(use_ocr),
            "ocr_count": len(ocr_text),
            "parsed_count": len(parsed_content_list),
            "model_seconds": round(model_seconds, 3),
            "ocr_seconds": round(ocr_seconds, 3),
            "parse_seconds": round(parse_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_start, 3),
        }
        _write_json(summary_path, summary)

        return {
            "parsed_elements": parsed_content_list,
            "label_coordinates": label_coordinates,
            "summary": summary,
            "out_dir": str(output),
            "overlay_path": str(overlay_path),
            "parsed_path": str(parsed_path),
            "coords_path": str(coords_path),
            "summary_path": str(summary_path),
        }

    @staticmethod
    def parse_base64(screenshot_b64: str, out_dir: str, **kwargs: Any) -> dict[str, Any]:
        """Parse one base64 screenshot with in-process OmniParser.

        Input:
        - ``screenshot_b64``: framework screenshot string.
        - ``out_dir``: directory where input image and parser artifacts are written.
        - ``kwargs``: parser options forwarded to ``parse_image``.

        Output:
        - Same result dictionary as ``parse_image``.

        Function:
        - Writes ``omniparser_input.png`` under the output directory and delegates
          all parsing to ``parse_image``.
        """

        output = Path(out_dir).resolve()
        input_path = output / "omniparser_input.png"
        _decode_base64_image(screenshot_b64, input_path)
        return OmniParserClient.parse_image(str(input_path), str(output), **kwargs)
