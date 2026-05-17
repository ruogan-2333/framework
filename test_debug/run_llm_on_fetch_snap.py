#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Navigation or Navigation+Router LLM analysis on snaps captured by fetch_current_snap.py.

Input:
- test_debug/fetch_snap/<session_id>/<capture_id>/snap.json.
- For LLM_MODE="navigation_router", questionnaire routers are loaded from QUESTIONNAIRE_DIR.

Output:
- test_debug/fetch_snap/<session_id>/<capture_id>/ui_digest.json.
- test_debug/fetch_snap/<session_id>/<capture_id>/llm_navigation_input.json.
- test_debug/fetch_snap/<session_id>/<capture_id>/llm_navigation_result.json.
- test_debug/fetch_snap/<session_id>/<capture_id>/llm_navigation_actions_overlay.png.
- test_debug/fetch_snap/<session_id>/<capture_id>/llm_navigation_router_input.json.
- test_debug/fetch_snap/<session_id>/<capture_id>/llm_navigation_router_result.json.
- test_debug/fetch_snap/<session_id>/<capture_id>/matched_blocks.json.
- test_debug/fetch_snap/<session_id>/<capture_id>/llm_navigation_router_actions_overlay.png.
- Failure JSON files are written in the same capture directory when one snap fails.

Function:
- Replays saved current-page snaps through GPTClient.propose_navigation or
  GPTClient.propose_navigation_and_router without thread pools or action execution.
- Draws LLM-selected element_id targets on the captured page image for manual inspection.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt_cls import GPTClient, _compact_digest
from questionnaire_state2 import QuestionnaireState as QuestionnaireState2


# ====================== Config ======================
FETCH_ROOT = PROJECT_ROOT / "test_debug" / "fetch_snap"
QUESTIONNAIRE_DIR = PROJECT_ROOT / "questionnaire-v2" / "games"

# Empty string means using the latest session directory under FETCH_ROOT.
# Absolute paths are also supported, for example:
# SOURCE_SESSION_ID = r"F:\workplace\framework\test_debug\fetch_snap\20260517_103000"
SOURCE_SESSION_ID = ""

# Empty string means processing all capture folders in the selected session.
# Set to "0001" or another capture folder name to process only one snap.
CAPTURE_NAME = ""

# Supported values: "navigation" or "navigation_router".
LLM_MODE = "navigation"

TASK = "Explore the app UI and propose useful navigation actions for questionnaire evidence collection."
APP_INTRO = None
FOCUS_HINTS = None
BLOCK_STATUS: Dict[str, Any] = {}
HISTORY: List[str] = []

# MODEL = "gpt-4o"
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.2
TIMEOUT_S = 60
MAX_CAPTURES = 0  # 0 means unlimited.
DEBUG_LOG = True
NAV_UI_DIGEST_LIMIT = 240
NAV_ROUTER_UI_DIGEST_LIMIT = 260
# ====================================================


logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    """
    Input: debug flag.
    Output: configures process-wide logging for this fetch-snap LLM replay script.
    Function: reduces noisy HTTP logs while keeping local failure diagnostics visible.
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
    Input: arbitrary value used as a path token.
    Output: Windows-safe filename token.
    Function: mirrors the naming behavior used by the other debug scripts.
    """
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or default


def read_json(path: Path) -> Any:
    """
    Input: JSON file path.
    Output: parsed JSON object.
    Function: centralizes UTF-8 JSON reads.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """
    Input: output path and JSON-serializable payload.
    Output: writes pretty UTF-8 JSON to disk.
    Function: centralizes result/failure JSON persistence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def decode_screenshot_b64(screenshot_b64: str) -> Image.Image:
    """
    Input: base64 encoded processed screenshot from snap["screenshot"].
    Output: RGB PIL image.
    Function: creates the coordinate base used for LLM action overlays.
    """
    raw = base64.b64decode(str(screenshot_b64 or "") + "==")
    return Image.open(BytesIO(raw)).convert("RGB")


def lookup_vid_node(vid_map: Dict[Any, Any], element_id: Any) -> Optional[Dict[str, Any]]:
    """
    Input: vid_map from snap and an LLM element_id.
    Output: matching UI node dictionary, or None when the element id is absent.
    Function: handles both string and integer element_id keys saved in JSON.
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


def frame_from_step(step: Dict[str, Any], vid_map: Dict[Any, Any]) -> Optional[Dict[str, int]]:
    """
    Input: one LLM action step and snap vid_map.
    Output: absolute frame dict {x,y,width,height}, or None when the id cannot be drawn.
    Function: resolves LLM element_id output back to the UI node frame.
    """
    node = lookup_vid_node(vid_map, step.get("element_id"))
    if not node:
        return None
    frame = node.get("absolute_frame") or node.get("frame") or {}
    if not frame:
        return None
    return {
        "x": int(frame.get("x", 0) or 0),
        "y": int(frame.get("y", 0) or 0),
        "width": int(frame.get("width", 0) or 0),
        "height": int(frame.get("height", 0) or 0),
    }


def draw_labeled_box(draw: ImageDraw.ImageDraw, frame: Dict[str, int], label: str, color: tuple[int, int, int]) -> None:
    """
    Input: drawing context, frame, label text, and RGB color.
    Output: draws one labeled rectangle on the current image.
    Function: renders O/C/PR labels consistently with the existing LLM debug scripts.
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
    Input: NavigationProposal JSON payload.
    Output: labeled overlay/candidate/page-return action steps with colors.
    Function: converts LLM action lists into drawable items.
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
    Output: writes one PNG with LLM action targets overlaid.
    Function: draws only element_id-resolved actions; bbox and coordinate fallbacks are intentionally unsupported.
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


def latest_session_dir(root: Path) -> Path:
    """
    Input: fetch_snap root containing timestamped session directories.
    Output: most recently modified session directory.
    Function: supports quick replay without manually editing SOURCE_SESSION_ID.
    """
    if not root.exists():
        raise FileNotFoundError(f"fetch_snap root not found: {root}")
    sessions = [path for path in root.iterdir() if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"no session directories found under: {root}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def resolve_source_session_dir() -> Path:
    """
    Input: module-level FETCH_ROOT and SOURCE_SESSION_ID.
    Output: selected fetch_snap session directory.
    Function: accepts empty latest-session selection, timestamp folder names, or absolute paths.
    """
    source = SOURCE_SESSION_ID.strip()
    if not source:
        return latest_session_dir(FETCH_ROOT)

    source_path = Path(source)
    session_dir = source_path if source_path.is_absolute() else FETCH_ROOT / source
    if not session_dir.exists():
        raise FileNotFoundError(f"source session dir not found: {session_dir}")
    return session_dir


def iter_snap_files(session_dir: Path) -> List[Path]:
    """
    Input: selected fetch_snap session directory and optional CAPTURE_NAME.
    Output: sorted list of snap.json files to process.
    Function: supports either whole-session replay or one-capture replay.
    """
    capture_name = CAPTURE_NAME.strip()
    if capture_name:
        snap_path = session_dir / safe_token(capture_name) / "snap.json"
        if not snap_path.exists():
            snap_path = session_dir / capture_name / "snap.json"
        if not snap_path.exists():
            raise FileNotFoundError(f"snap not found for CAPTURE_NAME={CAPTURE_NAME!r} under {session_dir}")
        return [snap_path]

    snap_files = sorted(session_dir.glob("*/snap.json"), key=lambda path: path.parent.name.lower())
    if MAX_CAPTURES and MAX_CAPTURES > 0:
        snap_files = snap_files[: int(MAX_CAPTURES)]
    return snap_files


def model_to_jsonable(value: Any) -> Any:
    """
    Input: pydantic model or plain Python value.
    Output: JSON-serializable representation.
    Function: makes GPTClient structured outputs safe to persist.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def build_gpt_client() -> GPTClient:
    """
    Input: module-level model settings and OPENAI_API_KEY from environment.
    Output: initialized GPTClient.
    Function: creates the same direct synchronous client used by the existing debug scripts.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY is empty; the LLM call will probably fail.")
    return GPTClient(api_key=api_key, model=MODEL, temperature=TEMPERATURE, timeout_s=TIMEOUT_S)


def load_questionnaires() -> QuestionnaireState2:
    """
    Input: module-level QUESTIONNAIRE_DIR.
    Output: loaded QuestionnaireState2 with routers, blocks, and block_status.
    Function: supplies real local router questions for the combined LLM mode.
    """
    if not QUESTIONNAIRE_DIR.exists():
        raise FileNotFoundError(f"questionnaire dir not found: {QUESTIONNAIRE_DIR}")
    return QuestionnaireState2.load_from_questionnaire_dir(str(QUESTIONNAIRE_DIR))


def build_ui_digest(ui_json: Dict[str, Any], limit: int) -> Any:
    """
    Input: post-processed UI tree from snap["uist"] and compact-digest limit.
    Output: compact UI digest matching the selected LLM function's internal limit.
    Function: writes exactly the UI structure intended for LLM prompt inspection.
    """
    return _compact_digest(ui_json, limit=limit)


def build_navigation_input_record(snap_path: Path, snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: one snap path and loaded snap dictionary.
    Output: compact Navigation input summary without screenshot base64.
    Function: documents the effective parameters used by propose_navigation.
    """
    ui_json = snap.get("uist") or {}
    meta = snap.get("meta") or {}
    xml_reliable = snap.get("xml_reliable", meta.get("xml_reliable"))
    return {
        "snap_path": str(snap_path),
        "mode": "navigation",
        "state_sig": str(snap.get("state_sig") or ""),
        "task": TASK,
        "block_status": BLOCK_STATUS,
        "app_intro": APP_INTRO,
        "focus_hints": FOCUS_HINTS,
        "history": HISTORY,
        "has_screenshot": bool(snap.get("screenshot")),
        "uist_root_count": len((ui_json or {}).get("elements") or []),
        "ui_digest_limit": NAV_UI_DIGEST_LIMIT,
        "xml_reliable": xml_reliable,
        "postprocess_mode": "xml_only" if xml_reliable is True else ("three_tools" if xml_reliable is False else "unknown"),
    }


def build_router_input_record(
    snap_path: Path,
    snap: Dict[str, Any],
    router_count: int,
    block_status: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Input: one snap, router count, and questionnaire block status.
    Output: compact combined Navigation+Router input summary without screenshot base64.
    Function: documents the effective parameters used by propose_navigation_and_router.
    """
    ui_json = snap.get("uist") or {}
    meta = snap.get("meta") or {}
    xml_reliable = snap.get("xml_reliable", meta.get("xml_reliable"))
    return {
        "snap_path": str(snap_path),
        "mode": "navigation_router",
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
        "ui_digest_limit": NAV_ROUTER_UI_DIGEST_LIMIT,
        "xml_reliable": xml_reliable,
        "postprocess_mode": "xml_only" if xml_reliable is True else ("three_tools" if xml_reliable is False else "unknown"),
    }


def router_updates_to_answers(result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Input: JSON payload from NavigationRouterResult.
    Output: router_updates list shaped for QuestionnaireState2.match_blocks_from_router_answers.
    Function: extracts the existing router answer format from the combined result.
    """
    router = result_payload.get("router") or {}
    updates = router.get("router_updates") or []
    return [item for item in updates if isinstance(item, dict)]


def process_one_navigation(gpt: GPTClient, snap_path: Path) -> bool:
    """
    Input: GPTClient and one fetch_snap capture snap.json path.
    Output: returns True on successful Navigation LLM replay.
    Function: writes Navigation input, ui_digest, result, failure, and action overlay into the capture folder.
    """
    out_dir = snap_path.parent
    snap = read_json(snap_path)
    input_record = build_navigation_input_record(snap_path, snap)
    write_json(out_dir / "llm_navigation_input.json", input_record)

    screenshot_b64 = str(snap.get("screenshot") or "")
    ui_json = snap.get("uist") or {}
    state_sig = str(snap.get("state_sig") or "")
    meta = snap.get("meta") or {}
    xml_reliable = snap.get("xml_reliable", meta.get("xml_reliable"))
    if not screenshot_b64 or not ui_json:
        write_json(
            out_dir / "llm_navigation_failure.json",
            {"ok": False, "error": "snap missing screenshot or uist", "input": input_record},
        )
        print(f"[FAILED] {snap_path.parent.name} missing screenshot/uist")
        return False

    write_json(out_dir / "ui_digest.json", build_ui_digest(ui_json, NAV_UI_DIGEST_LIMIT))

    t0 = time.perf_counter()
    try:
        nav = gpt.propose_navigation(
            screenshot_b64=screenshot_b64,
            ui_json=ui_json,
            block_status=dict(BLOCK_STATUS),
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
        logger.exception("Navigation LLM failed for %s", snap_path)
        write_json(out_dir / "llm_navigation_failure.json", {"ok": False, "elapsed_s": elapsed_s, "error": error})
        print(f"[FAILED] {snap_path.parent.name} error={error}")
        return False

    elapsed_s = time.perf_counter() - t0
    result = model_to_jsonable(nav)
    write_json(
        out_dir / "llm_navigation_result.json",
        {
            "ok": True,
            "elapsed_s": elapsed_s,
            "result": result,
            "usage": {
                "total_prompt_tokens": getattr(gpt, "total_prompt_tokens", 0),
                "total_completion_tokens": getattr(gpt, "total_completion_tokens", 0),
                "total_llm_calls": getattr(gpt, "total_llm_calls", 0),
                "usage_by_op": getattr(gpt, "usage_by_op", {}),
            },
        },
    )
    draw_llm_actions_overlay(snap, result if isinstance(result, dict) else {}, out_dir / "llm_navigation_actions_overlay.png")

    candidate_count = len((result or {}).get("candidate_actions") or []) if isinstance(result, dict) else 0
    overlay_kind = (result or {}).get("overlay_kind") if isinstance(result, dict) else ""
    return_count = len((result or {}).get("page_return_actions") or []) if isinstance(result, dict) else 0
    print(
        f"[OK] {snap_path.parent.name} mode=navigation elapsed_s={elapsed_s:.3f} "
        f"overlay={overlay_kind} candidates={candidate_count} page_return={return_count}"
    )
    return True


def process_one_navigation_router(
    gpt: GPTClient,
    questionnaires: QuestionnaireState2,
    snap_path: Path,
) -> bool:
    """
    Input: GPTClient, loaded questionnaire state, and one fetch_snap capture snap.json path.
    Output: returns True on successful combined LLM replay.
    Function: writes combined input, ui_digest, result, matched blocks, failure, and action overlay into the capture folder.
    """
    out_dir = snap_path.parent
    snap = read_json(snap_path)
    router_questions = list(getattr(questionnaires, "routers", []) or [])
    block_status = dict(getattr(questionnaires, "block_status", {}) or {})
    input_record = build_router_input_record(snap_path, snap, len(router_questions), block_status)
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
        print(f"[FAILED] {snap_path.parent.name} missing screenshot/uist")
        return False

    write_json(out_dir / "ui_digest.json", build_ui_digest(ui_json, NAV_ROUTER_UI_DIGEST_LIMIT))

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
        write_json(out_dir / "llm_navigation_router_failure.json", {"ok": False, "elapsed_s": elapsed_s, "error": error})
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
    return_count = len((navigation or {}).get("page_return_actions") or [])
    router_update_count = len((router or {}).get("router_updates") or [])
    print(
        f"[OK] {snap_path.parent.name} mode=navigation_router elapsed_s={elapsed_s:.3f} "
        f"candidates={candidate_count} page_return={return_count} "
        f"router_updates={router_update_count} matched_blocks={len(matched_blocks)}"
    )
    return True


def main() -> int:
    """
    Input: module-level configuration constants.
    Output: process exit code, 0 when all selected LLM calls succeed.
    Function: selects one fetch_snap session, replays selected snaps through the chosen LLM mode, and writes results in place.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging(DEBUG_LOG)

    mode = LLM_MODE.strip().lower()
    if mode not in {"navigation", "navigation_router"}:
        print(f"[FAILED] unsupported LLM_MODE={LLM_MODE!r}; expected navigation or navigation_router")
        return 1

    session_dir = resolve_source_session_dir()
    snap_files = iter_snap_files(session_dir)
    if not snap_files:
        print(f"No snap.json files found under {session_dir}")
        return 1

    print("=== LLM replay on fetch_snap ===")
    print(f"mode={mode}")
    print(f"source_session_dir={session_dir}")
    print(f"snap_count={len(snap_files)}")

    gpt = build_gpt_client()
    questionnaires = None
    if mode == "navigation_router":
        questionnaires = load_questionnaires()
        print(f"questionnaire_dir={QUESTIONNAIRE_DIR}")
        print(f"router_count={len(getattr(questionnaires, 'routers', []) or [])}")
        print(f"block_count={len(getattr(questionnaires, 'blocks', []) or [])}")

    results: List[bool] = []
    for snap_path in snap_files:
        if mode == "navigation":
            results.append(process_one_navigation(gpt, snap_path))
        else:
            assert questionnaires is not None
            results.append(process_one_navigation_router(gpt, questionnaires, snap_path))

    ok_count = sum(1 for ok in results if ok)
    print(f"=== done: ok={ok_count} failed={len(results) - ok_count} ===")
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
