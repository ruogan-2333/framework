from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

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

load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(
        description="Interactive framework for real-time capture_and_process + LLM analysis only."
    )
    parser.add_argument("--appium-url", type=str, default="http://127.0.0.1:4723")
    parser.add_argument("--device-name", type=str, default=None)
    parser.add_argument("--package", type=str, default="", help="Target app package name for context")
    parser.add_argument("--activity", type=str, default=None)
    parser.add_argument("--questionnaire-dir", type=str, default=str(PROJECT_ROOT / "questionnaire-v2" / "games"))
    parser.add_argument("--task", type=str, default="Explore the app to fill the questionnaire.")
    parser.add_argument("--app-intro", type=str, default="")
    parser.add_argument("--focus-hints", type=str, default="")
    parser.add_argument("--llm-mode", type=str, choices=["nav", "router", "both"], default="both")

    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--capture-timeout", type=float, default=10.0)

    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "mytest2" / "llm_interactive_debug"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _safe_token(value: Any, default: str = "unknown") -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or default


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_uist_nodes(uist: Dict[str, Any]) -> int:
    stack = list((uist or {}).get("elements", []) or [])
    total = 0
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.get("subviews", []) or [])
    return total


def _draw_vid_map_overlay(screenshot_b64: str, vid_map: Dict[str, Any], out_path: Path) -> bool:
    if not screenshot_b64 or not vid_map:
        return False
    try:
        raw = base64.b64decode(str(screenshot_b64) + "==")
        image = Image.open(BytesIO(raw)).convert("RGB")
        draw = ImageDraw.Draw(image)

        for element_id, node in (vid_map or {}).items():
            frame = node.get("absolute_frame") or node.get("frame") or {}
            x = int(frame.get("x", 0))
            y = int(frame.get("y", 0))
            w = int(frame.get("width", 0))
            h = int(frame.get("height", 0))
            if w <= 1 or h <= 1:
                continue

            clickable = bool(node.get("clickable"))
            color = (220, 50, 47) if clickable else (46, 160, 67)
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3)

            label = str(element_id)
            box = draw.textbbox((x, y), label)
            tw = box[2] - box[0]
            th = box[3] - box[1]
            left = max(0, x)
            top = max(0, y - th - 8)
            if top == 0 and y + h + th + 8 < image.height:
                top = y + h + 2
            draw.rectangle((left, top, left + tw + 10, top + th + 8), fill=color)
            draw.text((left + 5, top + 4), label, fill=(255, 255, 255))

        image.save(out_path)
        return True
    except Exception:
        logger.debug("draw vid_map overlay failed", exc_info=True)
        return False


def _save_capture(step_dir: Path, snap: Dict[str, Any]) -> Dict[str, Any]:
    step_dir.mkdir(parents=True, exist_ok=True)

    _json_dump(step_dir / "snap.json", snap)
    (step_dir / "xml.xml").write_text(str(snap.get("xml") or ""), encoding="utf-8")
    (step_dir / "xml_raw.xml").write_text(str(snap.get("xml_raw") or ""), encoding="utf-8")

    screenshot = str(snap.get("screenshot") or "")
    screenshot_raw = str(snap.get("screenshot_raw") or "")
    if screenshot:
        (step_dir / "screenshot.png").write_bytes(base64.b64decode(screenshot + "=="))
    if screenshot_raw:
        (step_dir / "screenshot_raw.png").write_bytes(base64.b64decode(screenshot_raw + "=="))

    uist = snap.get("uist") or {}
    vid_map = snap.get("vid_map") or {}
    _json_dump(step_dir / "uist.json", uist)
    _json_dump(step_dir / "vid_map.json", vid_map)
    overlay_ok = _draw_vid_map_overlay(screenshot, vid_map, step_dir / "vid_map_overlay.png")

    meta = snap.get("meta") or {}
    summary = {
        "saved_at_ms": int(time.time() * 1000),
        "state_sig": str(snap.get("state_sig") or ""),
        "xml_reliable": bool(snap.get("xml_reliable")),
        "uist_root_count": int(len((uist or {}).get("elements") or [])),
        "uist_total_nodes": int(_count_uist_nodes(uist)),
        "vid_map_count": int(len(vid_map or {})),
        "overlay_written": bool(overlay_ok),
        "identity_source": str(meta.get("identity_source") or ""),
        "coord_scale": float(meta.get("coord_scale") or 0.0),
        "foreground_package": str(meta.get("foreground_package") or ""),
        "foreground_activity": str(meta.get("foreground_activity") or ""),
        "xml_hash": str(meta.get("xml_hash") or ""),
        "screenshot_hash": str(meta.get("screenshot_hash") or ""),
        "screenshot_phash": str(meta.get("screenshot_phash") or ""),
        "cache_hit": bool(meta.get("cache_hit")),
    }
    _json_dump(step_dir / "summary.json", summary)
    return summary


def _build_runner(args: argparse.Namespace, run_id: str) -> WorkflowRunner:
    questionnaires = QuestionnaireState2.load_from_questionnaire_dir(str(args.questionnaire_dir))
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set .env or pass --api-key.")

    gpt = GPTClient(
        api_key=api_key,
        model=str(args.model),
        temperature=float(args.temperature),
        timeout_s=int(args.timeout),
    )
    appium = AndroidAppiumClient(server_url=str(args.appium_url), device_name=args.device_name).init_connection()
    budget = BudgetConfig(time_budget_s=60.0, max_actions=1, per_page_probe_cap=1, max_workers=1)

    return WorkflowRunner(
        appium=appium,
        gpt=gpt,
        questionnaires=questionnaires,
        budget=budget,
        target_package=str(args.package or ""),
        target_activity=args.activity,
        pause=False,
        app_intro=(str(args.app_intro).strip() or None),
        focus_hints=(str(args.focus_hints).strip() or None),
        callbacks=NoOpCallbacks(),
        run_id=run_id,
    )


def _run_nav(runner: WorkflowRunner, snap: Dict[str, Any], step_dir: Path, task: str) -> Dict[str, Any]:
    nav_input = {
        "state_sig": str(snap.get("state_sig") or ""),
        "task": str(task or ""),
        "block_status": copy.deepcopy(getattr(runner.questionnaires, "block_status", {}) or {}),
        "app_intro": runner.app_intro,
        "focus_hints": runner.focus_hints,
        "history": list(getattr(runner, "history", []) or []),
        "meta": copy.deepcopy(snap.get("meta") or {}),
    }
    _json_dump(step_dir / "nav_input.json", nav_input)

    t0 = time.perf_counter()
    nav = runner.gpt.propose_navigation(
        screenshot_b64=str(snap.get("screenshot") or ""),
        ui_json=snap.get("uist") or {},
        block_status=nav_input["block_status"],
        task=task,
        app_intro=runner.app_intro,
        focus_hints=runner.focus_hints,
        history=nav_input["history"],
        state_sig=nav_input["state_sig"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    _json_dump(step_dir / "nav_output.json", nav.model_dump(mode="json"))

    candidates = runner._candidate_actions(nav=nav, snap=snap, allow_heuristics=False)
    candidate_list = []
    for i, cand in enumerate(candidates, start=1):
        actions = list(getattr(cand, "actions", None) or [])
        candidate_list.append(
            {
                "index": i,
                "candidate_key": runner._candidate_key(cand),
                "score": float(getattr(cand, "score", 0.0) or 0.0),
                "actions": [
                    {
                        "action": str(getattr(getattr(s, "action", None), "value", getattr(s, "action", ""))),
                        "element_id": getattr(s, "element_id", None),
                        "text": getattr(s, "text", None),
                        "reasoning": getattr(s, "reasoning", ""),
                    }
                    for s in actions
                ],
            }
        )
    _json_dump(step_dir / "candidate_list.json", candidate_list)

    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "overlay_kind": str(getattr(nav, "overlay_kind", "")),
        "candidate_count_raw": len(getattr(nav, "candidate_actions", []) or []),
        "candidate_count_filtered": len(candidate_list),
    }


def _run_router(runner: WorkflowRunner, snap: Dict[str, Any], step_dir: Path) -> Dict[str, Any]:
    router_input = {
        "state_sig": str(snap.get("state_sig") or ""),
        "app_intro": runner.app_intro,
        "focus_hints": runner.focus_hints,
        "router_question_count": len(getattr(runner.questionnaires, "routers", []) or []),
    }
    _json_dump(step_dir / "router_input.json", router_input)

    t0 = time.perf_counter()
    router_result = runner.gpt.propose_router_answers(
        screenshot_b64=str(snap.get("screenshot") or ""),
        router_questions=list(getattr(runner.questionnaires, "routers", []) or []),
        app_intro=runner.app_intro,
        focus_hints=runner.focus_hints,
        state_sig=router_input["state_sig"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    _json_dump(step_dir / "router_output.json", router_result.model_dump(mode="json"))

    router_updates = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else getattr(item, "__dict__", {})
        for item in (getattr(router_result, "router_updates", None) or [])
    ]
    matched_blocks = runner.questionnaires.match_blocks_from_router_answers(router_updates)
    runner.questionnaires.mark_blocks_hit(matched_blocks)

    matched_payload = [
        {
            "id": str(block.get("id") or ""),
            "module": str(block.get("module") or ""),
            "topic": str(block.get("topic") or ""),
            "question_count": int(len(block.get("questions") or {})),
            "question_ids": list((block.get("questions") or {}).keys())[:80],
        }
        for block in matched_blocks
    ]
    _json_dump(step_dir / "router_matched_blocks.json", matched_payload)

    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "router_update_count": len(router_updates),
        "matched_block_count": len(matched_payload),
    }


def _print_help() -> None:
    print("Commands:")
    print("  Enter : capture_and_process + LLM analysis")
    print("  c     : capture only")
    print("  a     : analyze last capture only")
    print("  h     : help")
    print("  q     : quit")


def run_interactive_framework(args: argparse.Namespace) -> int:
    setup_logging(debug=bool(args.debug), level="INFO")

    run_stamp = str(args.run_id or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    package_token = _safe_token(args.package or "current_foreground")
    run_id = f"{run_stamp}_{package_token}"
    run_dir = Path(str(args.output_root)).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(args, run_id=run_id)
    mode = str(args.llm_mode or "both").strip().lower()

    _json_dump(
        run_dir / "session_meta.json",
        {
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_mode": mode,
            "appium_url": str(args.appium_url),
            "device_name": args.device_name,
            "package": str(args.package or ""),
            "activity": args.activity,
            "questionnaire_dir": str(Path(str(args.questionnaire_dir)).resolve()),
            "model": str(args.model),
            "temperature": float(args.temperature),
            "timeout_s": int(args.timeout),
            "capture_timeout_s": float(args.capture_timeout),
            "output_dir": str(run_dir),
        },
    )

    print("=== Interactive LLM Framework (Capture + Analyze only) ===")
    print(f"Output dir: {run_dir}")
    print(f"LLM mode: {mode}")
    print("Note: this script does not execute UI actions.")
    _print_help()

    step_idx = 0
    last_snap: Optional[Dict[str, Any]] = None
    last_step_dir: Optional[Path] = None

    try:
        while True:
            cmd = input("\\n[loop] Enter/c/a/h/q > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "h":
                _print_help()
                continue

            do_capture = cmd in {"", "c"}
            do_analyze = cmd in {"", "a"}
            if cmd not in {"", "c", "a"}:
                print(f"Unknown command: {cmd}")
                continue

            if do_capture:
                step_idx += 1
                step_dir = run_dir / f"step_{step_idx:03d}"
                t0 = time.perf_counter()
                snap = runner._capture_and_process(timeout=float(args.capture_timeout))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if not snap:
                    print("capture failed: _capture_and_process returned None")
                    step_idx -= 1
                    continue
                summary = _save_capture(step_dir, snap)
                summary["capture_elapsed_ms"] = round(elapsed_ms, 3)
                _json_dump(step_dir / "summary.json", summary)
                last_snap = snap
                last_step_dir = step_dir
                print(
                    f"[step {step_idx:03d}] capture done: sig={summary.get('state_sig')} "
                    f"xml_reliable={summary.get('xml_reliable')} "
                    f"uist_nodes={summary.get('uist_total_nodes')} vid_map={summary.get('vid_map_count')}"
                )
                if cmd == "c":
                    continue

            if do_analyze:
                if last_snap is None or last_step_dir is None:
                    print("no captured snapshot; run capture first")
                    continue
                analysis_summary: Dict[str, Any] = {
                    "saved_at_ms": int(time.time() * 1000),
                    "state_sig": str(last_snap.get("state_sig") or ""),
                    "llm_mode": mode,
                }
                if mode in {"nav", "both"}:
                    nav_summary = _run_nav(runner, last_snap, last_step_dir, task=str(args.task))
                    analysis_summary.update(
                        {
                            "nav_elapsed_ms": nav_summary["elapsed_ms"],
                            "nav_overlay_kind": nav_summary["overlay_kind"],
                            "nav_candidate_count_raw": nav_summary["candidate_count_raw"],
                            "nav_candidate_count_filtered": nav_summary["candidate_count_filtered"],
                        }
                    )
                if mode in {"router", "both"}:
                    router_summary = _run_router(runner, last_snap, last_step_dir)
                    analysis_summary.update(
                        {
                            "router_elapsed_ms": router_summary["elapsed_ms"],
                            "router_update_count": router_summary["router_update_count"],
                            "matched_block_count": router_summary["matched_block_count"],
                        }
                    )
                _json_dump(last_step_dir / "analysis_summary.json", analysis_summary)
                print(
                    f"[step {step_idx:03d}] analysis done: "
                    f"nav_ms={analysis_summary.get('nav_elapsed_ms')} "
                    f"router_ms={analysis_summary.get('router_elapsed_ms')} "
                    f"router_updates={analysis_summary.get('router_update_count')} "
                    f"matched_blocks={analysis_summary.get('matched_block_count')}"
                )
                print(f"[step {step_idx:03d}] files saved: {last_step_dir}")
    finally:
        runner.appium.quit()

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_interactive_framework(parse_args()))
