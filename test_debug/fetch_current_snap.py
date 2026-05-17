#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch snaps from current emulator pages through the real Appium capture path.

Input:
- The emulator/app is already open on the target page.
- Appium server is running and can create a UiAutomator2 session.
- Press Enter once for each page that should be captured; type q/quit/exit to stop.

Output:
- test_debug/fetch_snap/<session_id>/<capture_id>/xml_raw.xml
- test_debug/fetch_snap/<session_id>/<capture_id>/screenshot_raw.png
- test_debug/fetch_snap/<session_id>/<capture_id>/snap.json
- test_debug/fetch_snap/<session_id>/<capture_id>/vid_map_overlay.png
- test_debug/fetch_snap/<session_id>/<capture_id>/failure.json when capture fails

Function:
- Calls WorkflowRunner._capture_and_process(...) repeatedly, without launching or restarting the app.
- Saves only the raw page source, raw screenshot, complete snap, and vid_map overlay for debugging.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from appium_android import AndroidAppiumClient
from trace_callbacks import NoOpCallbacks
from workflow import BudgetConfig, WorkflowRunner


# ====================== Config ======================
OUTPUT_ROOT = PROJECT_ROOT / "test_debug" / "fetch_snap"

APPIUM_SERVER_URL = "http://127.0.0.1:4723"
DEVICE_NAME = None

# Keep empty to avoid app launch/foreground enforcement. This script only captures the current page.
TARGET_PACKAGE = ""
TARGET_ACTIVITY = None

CAPTURE_TIMEOUT = 10.0
PAUSE_BEFORE_CAPTURE = True
DEBUG_LOG = True
CAPTURE_ID_WIDTH = 4
# ====================================================


logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    """
    Input: debug flag.
    Output: configures process-wide logging for this fetch script.
    Function: keeps Appium/Selenium logs readable while preserving workflow debug logs.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def write_json(path: Path, payload: Any) -> None:
    """
    Input: output path and JSON-serializable payload.
    Output: writes pretty UTF-8 JSON to disk.
    Function: centralizes JSON writing and creates parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def decode_screenshot_b64(screenshot_b64: str) -> Image.Image:
    """
    Input: base64-encoded PNG screenshot.
    Output: RGB PIL image.
    Function: converts snap screenshots into an image canvas for saving or overlay rendering.
    """
    raw = base64.b64decode(str(screenshot_b64 or "") + "==")
    return Image.open(BytesIO(raw)).convert("RGB")


def write_screenshot_png(path: Path, screenshot_b64: str) -> None:
    """
    Input: output path and base64-encoded screenshot.
    Output: writes PNG image bytes to disk.
    Function: saves raw screenshot bytes with a PIL fallback for malformed padding.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(base64.b64decode(str(screenshot_b64 or "") + "=="))
    except Exception:
        decode_screenshot_b64(screenshot_b64).save(path)


def draw_vid_map_overlay(screenshot_b64: str, vid_map: Dict[Any, Any], out_path: Path) -> None:
    """
    Input: processed screenshot base64, vid_map, and output path.
    Output: writes a PNG with element id boxes.
    Function: draws red boxes for clickable nodes and green boxes for non-clickable nodes.
    """
    image = decode_screenshot_b64(screenshot_b64)
    draw = ImageDraw.Draw(image)

    for element_id, node in (vid_map or {}).items():
        if not isinstance(node, dict):
            continue
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


def make_minimal_questionnaires() -> SimpleNamespace:
    """
    Input: none.
    Output: minimal questionnaire namespace.
    Function: satisfies WorkflowRunner construction for capture-only use.
    """
    return SimpleNamespace(routers=[], blocks=[], block_status={})


def build_runner(run_id: str) -> WorkflowRunner:
    """
    Input: run id used for trace context.
    Output: WorkflowRunner with a live Appium client.
    Function: initializes Appium but does not launch, activate, stop, or restart any app.
    """
    appium = AndroidAppiumClient(server_url=APPIUM_SERVER_URL, device_name=DEVICE_NAME)
    appium.init_connection()
    budget = BudgetConfig(time_budget_s=30.0, max_actions=1, per_page_probe_cap=1, max_workers=1)
    return WorkflowRunner(
        appium=appium,
        gpt=None,
        questionnaires=make_minimal_questionnaires(),
        budget=budget,
        target_package=TARGET_PACKAGE,
        target_activity=TARGET_ACTIVITY,
        pause=False,
        callbacks=NoOpCallbacks(),
        run_id=run_id,
    )


def save_success_outputs(out_dir: Path, snap: Dict[str, Any]) -> None:
    """
    Input: output directory and complete snap.
    Output: writes xml_raw.xml, screenshot_raw.png, snap.json, and vid_map_overlay.png.
    Function: keeps current-page snap capture outputs minimal and directly inspectable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_raw = str(snap.get("xml_raw") or snap.get("xml") or "")
    (out_dir / "xml_raw.xml").write_text(xml_raw, encoding="utf-8")

    screenshot_raw_b64 = str(snap.get("screenshot_raw") or snap.get("screenshot") or "")
    if screenshot_raw_b64:
        write_screenshot_png(out_dir / "screenshot_raw.png", screenshot_raw_b64)

    write_json(out_dir / "snap.json", snap)

    screenshot_b64 = str(snap.get("screenshot") or screenshot_raw_b64)
    vid_map = snap.get("vid_map") or {}
    if screenshot_b64 and vid_map:
        draw_vid_map_overlay(screenshot_b64, vid_map, out_dir / "vid_map_overlay.png")


def save_failure_outputs(out_dir: Path, elapsed_ms: float, error: str) -> None:
    """
    Input: output directory, elapsed milliseconds, and error message.
    Output: writes failure.json.
    Function: captures enough context to diagnose Appium/capture failures.
    """
    write_json(
        out_dir / "failure.json",
        {
            "ok": False,
            "elapsed_ms": round(elapsed_ms, 3),
            "error": error,
            "saved_at_ms": int(time.time() * 1000),
        },
    )


def print_success_summary(out_dir: Path, snap: Dict[str, Any], elapsed_ms: float) -> None:
    """
    Input: output directory, snap, and elapsed milliseconds.
    Output: prints a concise capture summary.
    Function: tells the user where outputs are and what state was captured.
    """
    meta = snap.get("meta") or {}
    print("[OK] current page captured")
    print(f"     output={out_dir}")
    print(f"     elapsed_ms={elapsed_ms:.3f}")
    print(f"     state_sig={snap.get('state_sig')}")
    print(f"     xml_reliable={snap.get('xml_reliable')} mode={'xml_only' if snap.get('xml_reliable') else 'three_tools'}")
    print(f"     uist_roots={len((snap.get('uist') or {}).get('elements') or [])} vid_map={len(snap.get('vid_map') or {})}")
    print(f"     foreground_package={meta.get('foreground_package')}")
    print(f"     foreground_activity={meta.get('foreground_activity')}")


def capture_once(runner: WorkflowRunner, out_dir: Path) -> bool:
    """
    Input: live WorkflowRunner and one capture output directory.
    Output: returns True when a snap is captured and saved successfully.
    Function: executes one current-page capture and writes success or failure artifacts.
    """
    t0 = time.perf_counter()
    try:
        snap = runner._capture_and_process(timeout=float(CAPTURE_TIMEOUT))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not snap:
            save_failure_outputs(out_dir, elapsed_ms, "_capture_and_process returned None")
            print(f"[FAILED] _capture_and_process returned None output={out_dir}")
            return False
        save_success_outputs(out_dir, snap)
        print_success_summary(out_dir, snap, elapsed_ms)
        return True
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("fetch current page snap failed")
        save_failure_outputs(out_dir, elapsed_ms, error)
        print(f"[FAILED] {error} output={out_dir}")
        return False


def main() -> int:
    """
    Input: module-level configuration constants.
    Output: process exit code.
    Function: opens one Appium session and captures current emulator pages in a manual loop.
    """
    setup_logging(DEBUG_LOG)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=== fetch current emulator page snaps ===")
    print(f"session_output={session_dir}")
    print("This script will not launch or restart the app.")

    runner = None
    try:
        runner = build_runner(run_id=session_id)
        capture_index = 1
        ok_count = 0
        failed_count = 0

        while True:
            prompt = (
                "Open the target page in the emulator, then press Enter to capture "
                "(q/quit/exit to stop)... "
                if PAUSE_BEFORE_CAPTURE
                else "Press Enter to capture (q/quit/exit to stop)... "
            )
            command = input(prompt).strip().lower()
            if command in {"q", "quit", "exit"}:
                break

            capture_id = f"{capture_index:0{CAPTURE_ID_WIDTH}d}"
            out_dir = session_dir / capture_id
            print(f"--- capture {capture_id} ---")
            if capture_once(runner, out_dir):
                ok_count += 1
            else:
                failed_count += 1
            capture_index += 1

        print(f"=== done: session={session_dir} ok={ok_count} failed={failed_count} ===")
        return 0 if failed_count == 0 else 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("fetch current page snap loop failed")
        write_json(
            session_dir / "session_failure.json",
            {"ok": False, "error": error, "saved_at_ms": int(time.time() * 1000)},
        )
        print(f"[FAILED] {error} session_output={session_dir}")
        return 1
    finally:
        try:
            if runner is not None:
                runner.appium.quit()
        except Exception:
            logger.debug("Appium quit failed", exc_info=True)


if __name__ == "__main__":
    raise SystemExit(main())
