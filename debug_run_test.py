from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from appium_android import AndroidAppiumClient
from env_config import load_project_env
from gpt_cls import GPTClient
from questionnaire_state import QuestionnaireState
from trace_callbacks import NoOpCallbacks
from workflow import BudgetConfig, RecoveryReason, WorkflowRunner


# Load the project environment file as the source of truth for API and proxy settings.
load_project_env(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger(__name__)
OUTPUT_ROOT = PROJECT_ROOT / "mytest2" / "outputs"


def setup_logging(debug: bool = True, level: str = "INFO") -> None:
    root_level = logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(root_level)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Editable debug copy of WorkflowRunner.run().")
    p.add_argument("--appium-url", type=str, default="http://127.0.0.1:4723")
    p.add_argument("--device-name", type=str, default=None)
    p.add_argument("--package", type=str, required=True)
    p.add_argument("--activity", type=str, default=None)
    p.add_argument("--questionnaire-dir", type=str, required=True)
    p.add_argument("--task", type=str, default="Explore the app to fill the questionnaire.")
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--time-budget", type=float, default=300.0)
    p.add_argument("--max-actions", type=int, default=300)
    p.add_argument("--probe-cap", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--pause", action="store_true")
    p.add_argument("--relaunch", action="store_true")
    return p.parse_args()


# ============================
package = "bim.app"

sys.argv = [
    sys.argv[0],
    # "--appium-url", "http://localhost:4723",
    "--appium-url", "http://127.0.0.1:4723",
    "--device-name", "emulator-5554",
    "--package", package,
    # "--questionnaire-dir", "./questionnaire-v2/others",
    "--questionnaire-dir", "./questionnaire-v2/games",
    "--relaunch",
    "--debug",
]
# ============================


def _decode_screenshot_to_image(screenshot_b64: str) -> Image.Image:
    raw = base64.b64decode(screenshot_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _draw_vid_map_overlay(screenshot_b64: str, vid_map: dict, out_path: Path) -> None:
    image = _decode_screenshot_to_image(screenshot_b64)
    draw = ImageDraw.Draw(image)

    for element_id, node in (vid_map or {}).items():
        frame = node.get("absolute_frame") or node.get("frame") or {}
        x = int(frame.get("x", 0))
        y = int(frame.get("y", 0))
        w = int(frame.get("width", 0))
        h = int(frame.get("height", 0))
        if w <= 0 or h <= 0:
            continue

        is_clickable = bool(node.get("clickable"))
        color = (220, 50, 47) if is_clickable else (46, 160, 67)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)

        label_text = str(element_id)
        label_box = draw.textbbox((x, y), label_text)
        text_w = label_box[2] - label_box[0]
        text_h = label_box[3] - label_box[1]
        text_left = max(0, x)
        text_top = max(0, y - text_h - 8)
        if text_top == 0 and y + h + text_h + 8 < image.height:
            text_top = y + h + 2
        draw.rectangle(
            (text_left, text_top, text_left + text_w + 10, text_top + text_h + 8),
            fill=color,
        )
        draw.text((text_left + 5, text_top + 4), label_text, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _write_debug_outputs(snap: dict, package: str) -> Path:
    run_dir = OUTPUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{package.replace('.', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snap_path = run_dir / "snap.json"
    snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    if snap.get("xml"):
        (run_dir / "xml.xml").write_text(str(snap.get("xml") or ""), encoding="utf-8")
    if snap.get("xml_raw"):
        (run_dir / "xml_raw.xml").write_text(str(snap.get("xml_raw") or ""), encoding="utf-8")

    if snap.get("screenshot"):
        (run_dir / "screenshot.png").write_bytes(base64.b64decode(str(snap.get("screenshot") or "") + "=="))
    if snap.get("screenshot_raw"):
        (run_dir / "screenshot_raw.png").write_bytes(base64.b64decode(str(snap.get("screenshot_raw") or "") + "=="))

    if snap.get("uist") is not None:
        (run_dir / "uist.json").write_text(json.dumps(snap.get("uist") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    if snap.get("vid_map") is not None:
        (run_dir / "vid_map.json").write_text(json.dumps(snap.get("vid_map") or {}, ensure_ascii=False, indent=2), encoding="utf-8")

    if snap.get("screenshot") and snap.get("vid_map"):
        _draw_vid_map_overlay(str(snap.get("screenshot") or ""), snap.get("vid_map") or {}, run_dir / "vid_map_overlay.png")

    return run_dir


def debug_run_copy(runner: WorkflowRunner, task: str) -> None:
    """
    Copy of WorkflowRunner.run() for local debugging.
    You can edit this function freely without touching the framework's real run().
    """
    start = time.time()

    if runner.target_package:
        runner.appium.ensure_foreground(runner.target_package, runner.target_activity)

    runner._pool = ThreadPoolExecutor(max_workers=runner.budget.max_workers)

    snap = runner._capture_and_process()
    if not snap:
        logger.error("Initial capture failed; abort.")
        return

    cur_sig: str = snap["state_sig"]
    runner.entry_sig = cur_sig
    runner.restart_entry_sig = cur_sig
    runner.dfs_stack = [cur_sig]
    runner.dfs_via = [None]
    runner.parent_map = {}

    runner._graph_record_observation(cur_sig, meta={"entry": True})
    runner._schedule_state(cur_sig, snap, task)
    runner._mark_progress("entry_state", {"sig": cur_sig})

    # ------------------------------------------------------------
    # IMPORTANT:
    # Start your step-by-step debugging here.
    # For the first phase, we stop right after capture/snap generation.
    # ------------------------------------------------------------
    print("\n=== debug_run_copy: after _capture_and_process() ===")
    print(f"state_sig={snap.get('state_sig')}")
    print(f"foreground_package={(snap.get('meta') or {}).get('foreground_package')}")
    print(f"foreground_activity={(snap.get('meta') or {}).get('foreground_activity')}")
    print(f"uist_root_count={len((snap.get('uist') or {}).get('elements') or [])}")
    print(f"vid_count={len(snap.get('vid_map') or {})}")
    print(f"elapsed_s={time.time() - start:.2f}")
    out_dir = _write_debug_outputs(snap, runner.target_package or "unknown")
    print(f"saved_debug_dir={out_dir}")
    input("Paused after snap generation. Press Enter to exit this debug copy...")

    try:
        if runner._pool is not None:
            runner._pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    setup_logging(debug=bool(args.debug), level="INFO")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    questionnaires = QuestionnaireState.load_from_dir(args.questionnaire_dir)
    appium = AndroidAppiumClient(server_url=args.appium_url, device_name=args.device_name).init_connection()
    gpt = GPTClient(api_key=api_key, model=args.model, temperature=args.temperature, timeout_s=args.timeout)

    budget = BudgetConfig(
        time_budget_s=float(args.time_budget),
        max_actions=int(args.max_actions),
        per_page_probe_cap=int(args.probe_cap),
        max_workers=int(args.workers),
    )

    runner = WorkflowRunner(
        appium=appium,
        gpt=gpt,
        questionnaires=questionnaires,
        budget=budget,
        target_package=args.package,
        target_activity=args.activity,
        pause=bool(args.pause),
        callbacks=NoOpCallbacks(),
    )

    if args.relaunch and args.package:
        appium.force_stop(args.package)

    try:
        debug_run_copy(runner, args.task)
        return 0
    finally:
        try:
            if args.relaunch and args.package:
                appium.force_stop(args.package)
        except Exception:
            pass
        appium.quit()


if __name__ == "__main__":
    raise SystemExit(main())
