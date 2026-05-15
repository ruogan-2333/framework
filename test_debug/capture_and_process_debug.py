#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline image-only tester for WorkflowRunner._capture_and_process_debug.

Input:
- Images from test_debug/test_pic, or one image name configured below.

Output:
- For each image, writes only the files needed to inspect post-processing:
  screenshot.png, uist.json, vid_map.json, vid_map_overlay.png.
- Also writes snap.json to test_debug/intermediate for later LLM tests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from appium_android import AndroidAppiumClient
from trace_callbacks import NoOpCallbacks
from workflow import BudgetConfig, WorkflowRunner


# ====================== Config ======================
INPUT_DIR = PROJECT_ROOT / "test_debug" / "test_pic"
OUTPUT_DIR = PROJECT_ROOT / "test_debug" / "output"
INTERMEDIATE_DIR = PROJECT_ROOT / "test_debug" / "intermediate"

# Empty string means processing all supported images under INPUT_DIR.
IMAGE_NAME = ""

TARGET_PACKAGE = "debug.image.only"
TARGET_ACTIVITY = None
PIXEL_RATIO = 1.0
CAPTURE_TIMEOUT = 10.0
DEBUG_LOG = True
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
# ====================================================


logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    """
    Input: debug flag.
    Output: configures process-wide logging for this debug script.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)


def safe_token(value: Any, default: str = "image") -> str:
    """
    Input: arbitrary value used in a path segment.
    Output: Windows-safe filename token.
    """
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or default


def md5_bytes(data: bytes) -> str:
    """
    Input: raw bytes.
    Output: lowercase MD5 hex digest.
    """
    return hashlib.md5(data).hexdigest()


def image_to_png_bytes(image_path: Path) -> bytes:
    """
    Input: path to a png/jpg/jpeg/webp image.
    Output: normalized RGB PNG bytes for the workflow raw payload.
    """
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def build_raw_payload(png_bytes: bytes) -> Dict[str, Any]:
    """
    Input: normalized PNG bytes.
    Output: raw payload shaped like AndroidAppiumClient.capture_snapshot().
    """
    screenshot_b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {
        "xml": "",
        "xml_hash": hashlib.md5(b"").hexdigest(),
        "screenshot": screenshot_b64,
        "screenshot_hash": md5_bytes(png_bytes),
        "device_info": {"pixelRatio": float(PIXEL_RATIO)},
    }


def iter_input_images(input_dir: Path, image_name: str) -> List[Path]:
    """
    Input: input directory and optional exact image name.
    Output: sorted list of image paths to process.
    """
    if image_name.strip():
        image_path = input_dir / image_name.strip()
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        return [image_path]

    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def make_minimal_questionnaires() -> SimpleNamespace:
    """
    Input: none.
    Output: minimal questionnaire object sufficient for capture-only runner use.
    """
    return SimpleNamespace(routers=[], blocks=[], block_status={})


def build_runner(run_id: str) -> WorkflowRunner:
    """
    Input: run_id used by WorkflowRunner internal context.
    Output: WorkflowRunner configured for offline image-only debug mode.
    """
    appium = AndroidAppiumClient(server_url="http://127.0.0.1:4723", device_name=None)
    budget = BudgetConfig(time_budget_s=30.0, max_actions=1, per_page_probe_cap=1, max_workers=1)
    return WorkflowRunner(
        appium=appium,
        gpt=None,  # _capture_and_process_debug does not call GPT.
        questionnaires=make_minimal_questionnaires(),
        budget=budget,
        target_package=TARGET_PACKAGE,
        target_activity=TARGET_ACTIVITY,
        pause=False,
        callbacks=NoOpCallbacks(),
        run_id=run_id,
    )


def decode_screenshot_b64(screenshot_b64: str) -> Image.Image:
    """
    Input: base64 encoded screenshot from snap["screenshot"].
    Output: RGB PIL image.
    """
    raw = base64.b64decode(str(screenshot_b64 or "") + "==")
    return Image.open(BytesIO(raw)).convert("RGB")


def draw_vid_map_overlay(screenshot_b64: str, vid_map: Dict[Any, Any], out_path: Path) -> None:
    """
    Input: processed screenshot base64, vid_map, and output path.
    Output: writes an overlay image whose boxes use the same coordinate space as vid_map.
    """
    image = decode_screenshot_b64(screenshot_b64)
    draw = ImageDraw.Draw(image)

    for element_id, node in (vid_map or {}).items():
        frame = node.get("absolute_frame") or node.get("frame") or {}
        x = int(frame.get("x", 0) or 0)
        y = int(frame.get("y", 0) or 0)
        w = int(frame.get("width", 0) or 0)
        h = int(frame.get("height", 0) or 0)
        if w <= 1 or h <= 1:
            continue

        color = (220, 50, 47) if bool(node.get("clickable")) else (46, 160, 67)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)

        label = str(element_id)
        box = draw.textbbox((x, y), label)
        text_w = box[2] - box[0]
        text_h = box[3] - box[1]
        label_x = max(0, x)
        label_y = max(0, y - text_h - 8)
        if label_y == 0 and y + h + text_h + 8 < image.height:
            label_y = y + h + 2
        draw.rectangle((label_x, label_y, label_x + text_w + 10, label_y + text_h + 8), fill=color)
        draw.text((label_x + 5, label_y + 4), label, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def write_json(path: Path, payload: Any) -> None:
    """
    Input: path and JSON-serializable payload.
    Output: writes pretty UTF-8 JSON to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_success_outputs(step_dir: Path, snap: Dict[str, Any]) -> None:
    """
    Input: per-image output directory and snap returned by _capture_and_process_debug.
    Output: writes screenshot.png, uist.json, vid_map.json, and vid_map_overlay.png.
    """
    step_dir.mkdir(parents=True, exist_ok=True)

    screenshot_b64 = str(snap.get("screenshot") or "")
    if screenshot_b64:
        (step_dir / "screenshot.png").write_bytes(base64.b64decode(screenshot_b64 + "=="))

    uist = snap.get("uist") or {}
    vid_map = snap.get("vid_map") or {}
    write_json(step_dir / "uist.json", uist)
    write_json(step_dir / "vid_map.json", vid_map)

    if screenshot_b64 and vid_map:
        draw_vid_map_overlay(screenshot_b64, vid_map, step_dir / "vid_map_overlay.png")


def save_intermediate_snap(intermediate_dir: Path, snap: Dict[str, Any]) -> None:
    """
    Input: per-image intermediate directory and complete snap dictionary.
    Output: writes snap.json for later LLM navigation/router tests.
    """
    write_json(intermediate_dir / "snap.json", snap)


def save_failure_outputs(step_dir: Path, image_path: Path, elapsed_ms: float, error: str) -> None:
    """
    Input: failure context.
    Output: writes failure.json only when a test image fails.
    """
    payload = {
        "ok": False,
        "image_path": str(image_path),
        "elapsed_ms": round(elapsed_ms, 3),
        "error": error,
        "saved_at_ms": int(time.time() * 1000),
    }
    write_json(step_dir / "failure.json", payload)


def print_success_summary(image_path: Path, step_dir: Path, intermediate_dir: Path, snap: Dict[str, Any], elapsed_ms: float) -> None:
    """
    Input: successful snap, output paths, and timing info.
    Output: prints the concise result summary to stdout.
    """
    meta = snap.get("meta") or {}
    print(f"[OK] {image_path.name}")
    print(f"     output={step_dir}")
    print(f"     intermediate={intermediate_dir}")
    print(f"     elapsed_ms={elapsed_ms:.3f}")
    print(f"     state_sig={snap.get('state_sig')}")
    print(f"     xml_reliable={snap.get('xml_reliable')} mode={'xml_only' if snap.get('xml_reliable') else 'three_tools'}")
    print(f"     uist_roots={len((snap.get('uist') or {}).get('elements') or [])} vid_map={len(snap.get('vid_map') or {})}")
    print(f"     crop_px={meta.get('status_bar_crop_px')} coord_scale={meta.get('coord_scale')}")


def process_one_image(runner: WorkflowRunner, image_path: Path, output_root: Path, intermediate_root: Path) -> bool:
    """
    Input: runner, one image path, output root, and intermediate root.
    Output: returns True on success; writes visual artifacts and snap.json.
    """
    image_token = safe_token(image_path.stem)
    step_dir = output_root / image_token
    intermediate_dir = intermediate_root / image_token
    t0 = time.perf_counter()

    try:
        png_bytes = image_to_png_bytes(image_path)
        raw_payload = build_raw_payload(png_bytes)
        snap = runner._capture_and_process_debug(
            timeout=float(CAPTURE_TIMEOUT),
            debug=True,
            raw=raw_payload,
            xml_raw="",
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("capture_and_process_debug crashed for %s", image_path)
        save_failure_outputs(step_dir, image_path, elapsed_ms, error)
        print(f"[FAILED] {image_path.name} error={error} output={step_dir}")
        return False

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not snap:
        save_failure_outputs(step_dir, image_path, elapsed_ms, "_capture_and_process_debug returned None")
        print(f"[FAILED] {image_path.name} returned None output={step_dir}")
        return False

    save_success_outputs(step_dir, snap)
    save_intermediate_snap(intermediate_dir, snap)
    print_success_summary(image_path, step_dir, intermediate_dir, snap, elapsed_ms)
    return True


def main() -> int:
    """
    Input: module-level configuration constants.
    Output: process exit code, 0 when all selected images succeed.
    """
    setup_logging(DEBUG_LOG)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"input dir not found: {INPUT_DIR}")

    images = iter_input_images(INPUT_DIR, IMAGE_NAME)
    if not images:
        print(f"No supported images found under {INPUT_DIR}")
        return 1

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = OUTPUT_DIR / run_id
    intermediate_root = INTERMEDIATE_DIR / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    intermediate_root.mkdir(parents=True, exist_ok=True)

    runner = build_runner(run_id=run_id)
    print("=== _capture_and_process_debug image-only test ===")
    print(f"input_dir={INPUT_DIR}")
    print(f"output_root={output_root}")
    print(f"intermediate_root={intermediate_root}")
    print(f"image_count={len(images)}")

    results = [process_one_image(runner, image_path, output_root, intermediate_root) for image_path in images]
    ok_count = sum(1 for ok in results if ok)
    print(f"=== done: ok={ok_count} failed={len(results) - ok_count} ===")
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
