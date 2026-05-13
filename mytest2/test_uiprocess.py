from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from appium_android import AndroidAppiumClient
from gpt_cls import GPTClient
from questionnaire_state2 import QuestionnaireState as QuestionnaireState2
from trace_callbacks import NoOpCallbacks
from workflow import BudgetConfig, WorkflowRunner


load_dotenv()
logger = logging.getLogger(__name__)
OUTPUT_ROOT = PROJECT_ROOT / "mytest2" / "capture_and_process_debug"


def setup_logging(debug: bool = False, level: str = "INFO") -> None:
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
    p = argparse.ArgumentParser(
        description=(
            "Interactive debug runner for WorkflowRunner._capture_and_process(). "
            "Press Enter once to run one capture pass."
        )
    )
    p.add_argument("--appium-url", type=str, default="http://127.0.0.1:4723")
    p.add_argument("--device-name", type=str, default=None)
    p.add_argument("--package", type=str, required=True)
    p.add_argument("--activity", type=str, default=None)
    p.add_argument("--questionnaire-dir", type=str, required=True)

    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--api-key", type=str, default=None)

    p.add_argument("--time-budget", type=float, default=300.0)
    p.add_argument("--max-actions", type=int, default=300)
    p.add_argument("--probe-cap", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--capture-timeout", type=float, default=10.0, help="Timeout passed into _capture_and_process().")
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N captures. 0 means unlimited.")
    p.add_argument("--run-id", type=str, default=None, help="Optional run id for output folder naming.")
    p.add_argument("--relaunch", action="store_true", help="Force-stop target package before and after debug run.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_") or "unknown"


def _decode_screenshot_to_image(screenshot_b64: str) -> Image.Image:
    raw = base64.b64decode(screenshot_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _draw_vid_map_overlay(screenshot_b64: str, vid_map: Dict[str, Any], out_path: Path) -> None:
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


def _save_snapshot_artifacts(snap: Dict[str, Any], step_dir: Path) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)

    (step_dir / "snap.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    if snap.get("xml"):
        (step_dir / "xml.xml").write_text(str(snap.get("xml") or ""), encoding="utf-8")
    if snap.get("xml_raw"):
        (step_dir / "xml_raw.xml").write_text(str(snap.get("xml_raw") or ""), encoding="utf-8")

    if snap.get("screenshot"):
        (step_dir / "screenshot.png").write_bytes(base64.b64decode(str(snap.get("screenshot") or "") + "=="))
    if snap.get("screenshot_raw"):
        (step_dir / "screenshot_raw.png").write_bytes(base64.b64decode(str(snap.get("screenshot_raw") or "") + "=="))

    if snap.get("uist") is not None:
        (step_dir / "uist.json").write_text(
            json.dumps(snap.get("uist") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if snap.get("vid_map") is not None:
        (step_dir / "vid_map.json").write_text(
            json.dumps(snap.get("vid_map") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if snap.get("screenshot") and snap.get("vid_map"):
        _draw_vid_map_overlay(str(snap.get("screenshot") or ""), snap.get("vid_map") or {}, step_dir / "vid_map_overlay.png")


def _build_summary_record(
    *,
    step: int,
    elapsed_s: float,
    step_dir: Path,
    snap: Optional[Dict[str, Any]],
    error: str = "",
) -> Dict[str, Any]:
    if not snap:
        return {
            "step": step,
            "ok": False,
            "elapsed_s": elapsed_s,
            "step_dir": str(step_dir),
            "error": error or "capture_failed",
            "ts": int(time.time() * 1000),
        }

    meta = snap.get("meta") or {}
    return {
        "step": step,
        "ok": True,
        "elapsed_s": elapsed_s,
        "step_dir": str(step_dir),
        "state_sig": str(snap.get("state_sig") or ""),
        "xml_reliable": bool(snap.get("xml_reliable")),
        "vid_count": len(snap.get("vid_map") or {}),
        "uist_root_count": len((snap.get("uist") or {}).get("elements") or []),
        "cache_hit": bool(meta.get("cache_hit")),
        "coord_scale": meta.get("coord_scale"),
        "identity_source": meta.get("identity_source"),
        "foreground_package": meta.get("foreground_package"),
        "foreground_activity": meta.get("foreground_activity"),
        "xml_hash": meta.get("xml_hash"),
        "screenshot_hash": meta.get("screenshot_hash"),
        "xml_hash_raw": meta.get("xml_hash_raw"),
        "screenshot_hash_raw": meta.get("screenshot_hash_raw"),
        "ts": int(time.time() * 1000),
    }


def _build_runner(args: argparse.Namespace) -> Tuple[WorkflowRunner, AndroidAppiumClient]:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY is empty; this is fine for capture-only debug.")

    questionnaires = QuestionnaireState2.load_from_questionnaire_dir(args.questionnaire_dir)
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
        pause=False,
        callbacks=NoOpCallbacks(),
    )
    return runner, appium


def main() -> int:
    args = parse_args()
    setup_logging(debug=bool(args.debug), level="INFO")

    run_id = str(args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{_sanitize_token(args.package)}")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "capture_summary.jsonl"

    runner, appium = _build_runner(args)

    if args.relaunch and args.package:
        appium.force_stop(args.package)

    try:
        if runner.target_package:
            # Keep the same startup behavior as WorkflowRunner.run().
            runner.appium.ensure_foreground(runner.target_package, runner.target_activity)

        print("=== capture_and_process debug ===")
        print(f"run_dir={run_dir}")
        print("Press Enter to run one _capture_and_process() pass.")
        print("Type 'q' then Enter to quit.")

        step = 0
        while True:
            user_input = input("\n[Next] Enter=run, q=quit > ").strip().lower()
            if user_input in {"q", "quit", "exit"}:
                break

            step += 1
            step_dir = run_dir / f"step_{step:03d}"
            t0 = time.perf_counter()
            snap: Optional[Dict[str, Any]] = None
            error_text = ""

            try:
                snap = runner._capture_and_process(timeout=float(args.capture_timeout))
            except Exception as exc:  # defensive; method already handles most failures internally
                error_text = f"{type(exc).__name__}: {exc}"
                logger.exception("capture step failed with uncaught exception")

            elapsed_s = time.perf_counter() - t0
            if snap:
                _save_snapshot_artifacts(snap, step_dir)
                meta = snap.get("meta") or {}
                print(
                    f"[step {step}] ok elapsed={elapsed_s:.3f}s sig={snap.get('state_sig')} "
                    f"xml_reliable={bool(snap.get('xml_reliable'))} "
                    f"cache_hit={bool(meta.get('cache_hit'))} "
                    f"vid={len(snap.get('vid_map') or {})}"
                )
                print(
                    f"          fg_pkg={meta.get('foreground_package') or '-'} "
                    f"fg_act={meta.get('foreground_activity') or '-'}"
                )
                print(f"          saved={step_dir}")
            else:
                print(f"[step {step}] failed elapsed={elapsed_s:.3f}s error={error_text or 'capture_and_process returned None'}")

            record = _build_summary_record(
                step=step,
                elapsed_s=elapsed_s,
                step_dir=step_dir,
                snap=snap,
                error=error_text,
            )
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if int(args.max_steps or 0) > 0 and step >= int(args.max_steps):
                print(f"Reached max_steps={args.max_steps}, stopping.")
                break

        print(f"\nsummary_jsonl={summary_path}")
        print("done.")
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
