#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synchronous tester for GPTClient.propose_navigation_and_router.

Input:
- snap.json files produced by test_debug/capture_and_process_debug.py.
- Real questionnaire routers loaded from QUESTIONNAIRE_DIR.

Output:
- Combined LLM result and locally matched full block payloads.
- ui_digest.json stores the exact compact UI digest value sent to the LLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt_cls import GPTClient, _compact_digest
from questionnaire_state2 import QuestionnaireState as QuestionnaireState2


# ====================== Config ======================
INTERMEDIATE_ROOT = PROJECT_ROOT / "test_debug" / "intermediate"
OUTPUT_ROOT = PROJECT_ROOT / "test_debug" / "llm_navigation_router_output"
QUESTIONNAIRE_DIR = PROJECT_ROOT / "questionnaire-UI" / "games"

# Empty string means using the latest run directory under INTERMEDIATE_ROOT.
SOURCE_RUN_ID = r"F:\workplace\framework\test_debug\intermediate\20260516_033838"

# Empty string means processing all snap.json files in the selected run.
SNAP_NAME = ""

TASK = "Explore the app UI and propose useful navigation actions for questionnaire evidence collection."
APP_INTRO = None
FOCUS_HINTS = None
HISTORY: List[str] = []

# MODEL = "gpt-4o"
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.2
TIMEOUT_S = 60
MAX_SNAPS = 0  # 0 means unlimited.
DEBUG_LOG = True
UI_DIGEST_LIMIT = 260
# ====================================================


logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    """
    Input: debug flag.
    Output: configures process-wide logging for this combined LLM test script.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def safe_token(value: Any, default: str = "snap") -> str:
    """
    Input: arbitrary value used in an output path.
    Output: Windows-safe filename token.
    """
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or default


def read_json(path: Path) -> Any:
    """
    Input: JSON file path.
    Output: parsed JSON object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """
    Input: path and JSON-serializable payload.
    Output: writes pretty UTF-8 JSON to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def decode_screenshot_b64(screenshot_b64: str) -> Image.Image:
    """
    Input: base64 encoded processed screenshot from snap["screenshot"].
    Output: RGB PIL image used as the coordinate base for action overlays.
    """
    raw = base64.b64decode(str(screenshot_b64 or "") + "==")
    return Image.open(BytesIO(raw)).convert("RGB")


def lookup_vid_node(vid_map: Dict[Any, Any], element_id: Any) -> Dict[str, Any] | None:
    """
    Input: vid_map from snap and an LLM element_id.
    Output: matching UI node dictionary, or None when the id is absent.
    """
    if element_id is None:
        return None
    if element_id in vid_map:
        node = vid_map.get(element_id)
        return node if isinstance(node, dict) else None
    text_id = str(element_id)
    if text_id in vid_map:
        node = vid_map.get(text_id)
        return node if isinstance(node, dict) else None
    try:
        int_id = int(element_id)
    except Exception:
        return None
    node = vid_map.get(int_id) or vid_map.get(str(int_id))
    return node if isinstance(node, dict) else None


def frame_from_step(step: Dict[str, Any], vid_map: Dict[Any, Any]) -> Dict[str, int] | None:
    """
    Input: one LLM action step and vid_map.
    Output: absolute frame dict {x,y,width,height} for drawing, or None.
    """
    node = lookup_vid_node(vid_map, step.get("element_id"))
    if node:
        frame = node.get("absolute_frame") or node.get("frame") or {}
        if frame:
            return {
                "x": int(frame.get("x", 0) or 0),
                "y": int(frame.get("y", 0) or 0),
                "width": int(frame.get("width", 0) or 0),
                "height": int(frame.get("height", 0) or 0),
            }

    return None


def draw_labeled_box(draw: ImageDraw.ImageDraw, frame: Dict[str, int], label: str, color: tuple[int, int, int]) -> None:
    """
    Input: drawing context, frame, label, and RGB color.
    Output: draws one labeled rectangle on the current image.
    """
    x = int(frame.get("x", 0) or 0)
    y = int(frame.get("y", 0) or 0)
    w = int(frame.get("width", 0) or 0)
    h = int(frame.get("height", 0) or 0)
    if w <= 0 or h <= 0:
        return

    draw.rectangle((x, y, x + w, y + h), outline=color, width=4)
    box = draw.textbbox((x, y), label)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    label_x = max(0, x)
    label_y = max(0, y - text_h - 8)
    if label_y == 0:
        label_y = y + 2
    draw.rectangle((label_x, label_y, label_x + text_w + 10, label_y + text_h + 8), fill=color)
    draw.text((label_x + 5, label_y + 4), label, fill=(255, 255, 255))


def iter_action_steps_for_overlay(navigation: Dict[str, Any]) -> List[tuple[str, Dict[str, Any], tuple[int, int, int]]]:
    """
    Input: NavigationProposal JSON payload from the combined result.
    Output: labeled overlay/candidate/page-return action steps with colors for rendering.
    """
    items: List[tuple[str, Dict[str, Any], tuple[int, int, int]]] = []
    for idx, step in enumerate(navigation.get("overlay_dismiss_actions") or [], start=1):
        if isinstance(step, dict):
            items.append((f"O{idx}", step, (220, 50, 47)))

    for cand_idx, candidate in enumerate(navigation.get("candidate_actions") or [], start=1):
        if not isinstance(candidate, dict):
            continue
        for step_idx, step in enumerate(candidate.get("actions") or [], start=1):
            if isinstance(step, dict):
                items.append((f"C{cand_idx}.{step_idx}", step, (30, 102, 245)))

    for idx, step in enumerate(navigation.get("page_return_actions") or [], start=1):
        if isinstance(step, dict):
            items.append((f"PR{idx}", step, (191, 97, 0)))
    return items


def draw_llm_actions_overlay(snap: Dict[str, Any], navigation: Dict[str, Any], out_path: Path) -> None:
    """
    Input: source snap, NavigationProposal JSON payload, and output path.
    Output: writes an image with LLM action targets overlaid.
    """
    screenshot_b64 = str(snap.get("screenshot") or "")
    if not screenshot_b64:
        return
    vid_map = snap.get("vid_map") or {}
    image = decode_screenshot_b64(screenshot_b64)
    draw = ImageDraw.Draw(image)

    for label, step, color in iter_action_steps_for_overlay(navigation):
        frame = frame_from_step(step, vid_map)
        if frame is not None:
            draw_labeled_box(draw, frame, label, color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def latest_run_dir(root: Path) -> Path:
    """
    Input: intermediate root containing timestamped run directories.
    Output: most recently modified run directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"intermediate root not found: {root}")
    runs = [path for path in root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no run directories found under: {root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def resolve_source_run_dir() -> Path:
    """
    Input: module-level INTERMEDIATE_ROOT and SOURCE_RUN_ID.
    Output: selected source run directory containing snap.json files.
    """
    if SOURCE_RUN_ID.strip():
        run_dir = INTERMEDIATE_ROOT / SOURCE_RUN_ID.strip()
        if not run_dir.exists():
            raise FileNotFoundError(f"source run dir not found: {run_dir}")
        return run_dir
    return latest_run_dir(INTERMEDIATE_ROOT)


def iter_snap_files(source_run_dir: Path) -> List[Path]:
    """
    Input: selected source run directory and optional SNAP_NAME.
    Output: sorted list of snap.json files to process.
    """
    if SNAP_NAME.strip():
        snap_path = source_run_dir / safe_token(SNAP_NAME.strip()) / "snap.json"
        if not snap_path.exists():
            snap_path = source_run_dir / SNAP_NAME.strip() / "snap.json"
        if not snap_path.exists():
            raise FileNotFoundError(f"snap not found for SNAP_NAME={SNAP_NAME!r} under {source_run_dir}")
        return [snap_path]

    snap_files = sorted(source_run_dir.glob("*/snap.json"), key=lambda path: path.parent.name.lower())
    if MAX_SNAPS and MAX_SNAPS > 0:
        snap_files = snap_files[: int(MAX_SNAPS)]
    return snap_files


def model_to_jsonable(value: Any) -> Any:
    """
    Input: pydantic model or plain Python value.
    Output: JSON-serializable representation suitable for local inspection.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def load_questionnaires() -> QuestionnaireState2:
    """
    Input: module-level QUESTIONNAIRE_DIR.
    Output: loaded QuestionnaireState2 with routers, blocks, and block_status.
    """
    if not QUESTIONNAIRE_DIR.exists():
        raise FileNotFoundError(f"questionnaire dir not found: {QUESTIONNAIRE_DIR}")
    return QuestionnaireState2.load_from_questionnaire_dir(str(QUESTIONNAIRE_DIR))


def build_input_record(snap_path: Path, snap: Dict[str, Any], router_count: int, block_status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: snap metadata plus questionnaire summary.
    Output: compact combined LLM input summary without screenshot base64.
    """
    ui_json = snap.get("uist") or {}
    meta = snap.get("meta") or {}
    xml_reliable = snap.get("xml_reliable", meta.get("xml_reliable"))
    return {
        "snap_path": str(snap_path),
        "questionnaire_dir": str(QUESTIONNAIRE_DIR),
        "state_sig": str(snap.get("state_sig") or ""),
        "task": TASK,
        "router_question_count": int(router_count),
        "block_status_count": len(block_status or {}),
        "app_intro": APP_INTRO,
        "focus_hints": FOCUS_HINTS,
        "history": HISTORY,
        "has_screenshot": bool(snap.get("screenshot")),
        "uist_root_count": len((ui_json or {}).get("elements") or []),
        "ui_digest_limit": UI_DIGEST_LIMIT,
        "xml_reliable": xml_reliable,
        "postprocess_mode": "xml_only" if xml_reliable is True else ("three_tools" if xml_reliable is False else "unknown"),
    }


def build_ui_digest(ui_json: Dict[str, Any]) -> Any:
    """
    Input: post-processed UI tree from snap["uist"].
    Output: compact UI digest using the same limit as GPTClient.propose_navigation_and_router.
    """
    return _compact_digest(ui_json, limit=UI_DIGEST_LIMIT)


def build_gpt_client() -> GPTClient:
    """
    Input: module-level model settings and OPENAI_API_KEY from environment.
    Output: initialized GPTClient for synchronous combined LLM calls.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY is empty; the LLM call will probably fail.")
    return GPTClient(api_key=api_key, model=MODEL, temperature=TEMPERATURE, timeout_s=TIMEOUT_S)


def router_updates_to_answers(result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Input: JSON payload from NavigationRouterResult.
    Output: router_updates list shaped for QuestionnaireState2.match_blocks_from_router_answers.
    """
    router = result_payload.get("router") or {}
    updates = router.get("router_updates") or []
    return [item for item in updates if isinstance(item, dict)]


def process_one_snap(
    gpt: GPTClient,
    questionnaires: QuestionnaireState2,
    snap_path: Path,
    output_run_dir: Path,
) -> bool:
    """
    Input: GPTClient, loaded questionnaire state, one snap.json path, and output run directory.
    Output: returns True on successful LLM result; writes input/result/matched-block/failure JSON.
    """
    snap = read_json(snap_path)
    router_questions = list(getattr(questionnaires, "routers", []) or [])
    block_status = dict(getattr(questionnaires, "block_status", {}) or {})
    image_token = safe_token(snap_path.parent.name)
    out_dir = output_run_dir / image_token
    input_record = build_input_record(snap_path, snap, len(router_questions), block_status)
    write_json(out_dir / "llm_navigation_router_input.json", input_record)

    screenshot_b64 = str(snap.get("screenshot") or "")
    ui_json = snap.get("uist") or {}
    state_sig = str(snap.get("state_sig") or "")
    meta = snap.get("meta") or {}
    xml_reliable = snap.get("xml_reliable", meta.get("xml_reliable"))
    if not screenshot_b64 or not ui_json:
        write_json(
            out_dir / "llm_navigation_router_failure.json",
            {"ok": False, "error": "snap missing screenshot or uist", "input": input_record},
        )
        print(f"[FAILED] {snap_path} missing screenshot/uist")
        return False

    ui_digest = build_ui_digest(ui_json)
    write_json(out_dir / "ui_digest.json", ui_digest)

    t0 = time.perf_counter()
    try:
        combined = gpt.propose_navigation_and_router(
            screenshot_b64=screenshot_b64,
            ui_json=ui_json,
            router_questions=router_questions,
            block_status=block_status,
            task=TASK,
            app_intro=APP_INTRO,
            focus_hints=FOCUS_HINTS,
            history=list(HISTORY),
            state_sig=state_sig,
            xml_reliable=xml_reliable if isinstance(xml_reliable, bool) else None,
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - t0
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("Navigation+Router LLM failed for %s", snap_path)
        write_json(
            out_dir / "llm_navigation_router_failure.json",
            {"ok": False, "elapsed_s": elapsed_s, "error": error, "input": input_record},
        )
        print(f"[FAILED] {snap_path.parent.name} error={error}")
        return False

    elapsed_s = time.perf_counter() - t0
    result_payload = model_to_jsonable(combined)
    router_answers = router_updates_to_answers(result_payload if isinstance(result_payload, dict) else {})
    matched_blocks = questionnaires.match_blocks_from_router_answers(router_answers)

    write_json(
        out_dir / "llm_navigation_router_result.json",
        {
            "ok": True,
            "elapsed_s": elapsed_s,
            "result": result_payload,
            "usage": {
                "total_prompt_tokens": getattr(gpt, "total_prompt_tokens", 0),
                "total_completion_tokens": getattr(gpt, "total_completion_tokens", 0),
                "total_llm_calls": getattr(gpt, "total_llm_calls", 0),
                "usage_by_op": getattr(gpt, "usage_by_op", {}),
            },
        },
    )
    write_json(out_dir / "matched_blocks.json", matched_blocks)

    navigation = (result_payload or {}).get("navigation") if isinstance(result_payload, dict) else {}
    router = (result_payload or {}).get("router") if isinstance(result_payload, dict) else {}
    draw_llm_actions_overlay(snap, navigation or {}, out_dir / "llm_navigation_router_actions_overlay.png")
    candidate_count = len((navigation or {}).get("candidate_actions") or [])
    router_update_count = len((router or {}).get("router_updates") or [])
    print(
        f"[OK] {snap_path.parent.name} elapsed_s={elapsed_s:.3f} "
        f"candidates={candidate_count} router_updates={router_update_count} matched_blocks={len(matched_blocks)}"
    )
    return True


def main() -> int:
    """
    Input: module-level configuration constants.
    Output: process exit code, 0 when all selected combined LLM calls succeed.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging(DEBUG_LOG)

    source_run_dir = resolve_source_run_dir()
    snap_files = iter_snap_files(source_run_dir)
    if not snap_files:
        print(f"No snap.json files found under {source_run_dir}")
        return 1

    questionnaires = load_questionnaires()
    output_run_dir = OUTPUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_from_{safe_token(source_run_dir.name)}"
    output_run_dir.mkdir(parents=True, exist_ok=True)

    print("=== LLM Navigation+Router synchronous test ===")
    print(f"source_run_dir={source_run_dir}")
    print(f"questionnaire_dir={QUESTIONNAIRE_DIR}")
    print(f"output_run_dir={output_run_dir}")
    print(f"snap_count={len(snap_files)}")
    print(f"router_count={len(getattr(questionnaires, 'routers', []) or [])}")
    print(f"block_count={len(getattr(questionnaires, 'blocks', []) or [])}")

    gpt = build_gpt_client()
    results = [process_one_snap(gpt, questionnaires, snap_path, output_run_dir) for snap_path in snap_files]
    ok_count = sum(1 for ok in results if ok)
    print(f"=== done: ok={ok_count} failed={len(results) - ok_count} ===")
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
