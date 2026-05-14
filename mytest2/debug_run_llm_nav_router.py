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
from typing import Any, Dict, List, Optional

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
        description=(
            "Interactive debug script for capture + LLM navigation/block_router analysis. "
            "Press Enter to alternate capture and analysis."
        )
    )
    parser.add_argument("--appium-url", type=str, default="http://127.0.0.1:4723")
    parser.add_argument("--device-name", type=str, default=None)
    parser.add_argument("--package", type=str, default="", help="Target app package name (used for state context)")
    parser.add_argument("--activity", type=str, default=None)
    parser.add_argument("--questionnaire-dir", type=str, default=str(PROJECT_ROOT / "questionnaire-v2" / "games"))
    parser.add_argument("--task", type=str, default="Explore the app to fill the questionnaire.")
    parser.add_argument("--app-intro", type=str, default="")
    parser.add_argument("--focus-hints", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--capture-timeout", type=float, default=10.0)
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "mytest2" / "llm_nav_router_debug"))
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


def _save_capture(step_dir: Path, snap: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    def n(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    step_dir.mkdir(parents=True, exist_ok=True)

    _json_dump(step_dir / n("snap.json"), snap)

    xml = str(snap.get("xml") or "")
    xml_raw = str(snap.get("xml_raw") or "")
    (step_dir / n("xml.xml")).write_text(xml, encoding="utf-8")
    (step_dir / n("xml_raw.xml")).write_text(xml_raw, encoding="utf-8")

    screenshot = str(snap.get("screenshot") or "")
    screenshot_raw = str(snap.get("screenshot_raw") or "")
    if screenshot:
        (step_dir / n("screenshot.png")).write_bytes(base64.b64decode(screenshot + "=="))
    if screenshot_raw:
        (step_dir / n("screenshot_raw.png")).write_bytes(base64.b64decode(screenshot_raw + "=="))

    uist = snap.get("uist") or {}
    vid_map = snap.get("vid_map") or {}
    _json_dump(step_dir / n("uist.json"), uist)
    _json_dump(step_dir / n("vid_map.json"), vid_map)

    overlay_ok = _draw_vid_map_overlay(screenshot, vid_map, step_dir / n("vid_map_overlay.png"))

    uist_total_nodes = _count_uist_nodes(uist)
    vid_map_count = len(vid_map or {})
    meta = snap.get("meta") or {}
    summary = {
        "saved_at_ms": int(time.time() * 1000),
        "state_sig": str(snap.get("state_sig") or ""),
        "xml_reliable": bool(snap.get("xml_reliable")),
        "uist_root_count": int(len((uist or {}).get("elements") or [])),
        "uist_total_nodes": int(uist_total_nodes),
        "vid_map_count": int(vid_map_count),
        "vid_map_matches_all_uist_nodes": bool(vid_map_count == uist_total_nodes),
        "vid_map_missing_nodes": int(max(0, uist_total_nodes - vid_map_count)),
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
    _json_dump(step_dir / n("summary.json"), summary)
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


def _nav_input_payload(runner: WorkflowRunner, snap: Dict[str, Any], task: str) -> Dict[str, Any]:
    return {
        "state_sig": str(snap.get("state_sig") or ""),
        "task": str(task or ""),
        "block_status": copy.deepcopy(getattr(runner.questionnaires, "block_status", {}) or {}),
        "app_intro": runner.app_intro,
        "focus_hints": runner.focus_hints,
        "history": list(getattr(runner, "history", []) or []),
        "meta": copy.deepcopy(snap.get("meta") or {}),
    }


def _router_input_payload(runner: WorkflowRunner, snap: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "state_sig": str(snap.get("state_sig") or ""),
        "app_intro": runner.app_intro,
        "focus_hints": runner.focus_hints,
        "router_question_count": len(getattr(runner.questionnaires, "routers", []) or []),
    }


def _step_to_text(step: Any) -> str:
    action = getattr(step, "action", None)
    action_text = str(getattr(action, "value", action) or "unknown")
    element_id = getattr(step, "element_id", None)
    text = str(getattr(step, "text", "") or "")
    if len(text) > 28:
        text = text[:28] + "..."
    return f"{action_text}(id={element_id}, text={text})"


def _candidate_to_payload(cand: Any) -> Dict[str, Any]:
    if hasattr(cand, "model_dump"):
        try:
            return cand.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(cand, "__dict__"):
        try:
            return dict(cand.__dict__)
        except Exception:
            pass
    return {"raw": str(cand)}


def _build_candidate_briefs(runner: WorkflowRunner, candidates: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates, start=1):
        actions = list(getattr(cand, "actions", None) or [])
        score = float(getattr(cand, "score", 0.0) or 0.0)
        key = runner._candidate_key(cand)
        out.append(
            {
                "index": i,
                "candidate_key": key,
                "score": score,
                "action_count": len(actions),
                "actions_preview": [_step_to_text(s) for s in actions[:3]],
                "candidate": _candidate_to_payload(cand),
            }
        )
    return out


def _run_llm_analysis(runner: WorkflowRunner, snap: Dict[str, Any], step_dir: Path, task: str) -> Dict[str, Any]:
    state_sig = str(snap.get("state_sig") or "")
    screenshot_b64 = str(snap.get("screenshot") or "")
    uist = snap.get("uist") or {}

    nav_input = _nav_input_payload(runner, snap, task)
    _json_dump(step_dir / "nav_input.json", nav_input)

    t0 = time.perf_counter()
    nav = runner.gpt.propose_navigation(
        screenshot_b64=screenshot_b64,
        ui_json=uist,
        block_status=nav_input["block_status"],
        task=task,
        app_intro=runner.app_intro,
        focus_hints=runner.focus_hints,
        history=nav_input["history"],
        state_sig=state_sig,
    )
    nav_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    nav_payload = nav.model_dump(mode="json")
    _json_dump(step_dir / "nav_output.json", nav_payload)
    candidates = runner._candidate_actions(nav=nav, snap=snap, allow_heuristics=False)
    candidate_briefs = _build_candidate_briefs(runner, candidates)
    _json_dump(step_dir / "candidate_list.json", candidate_briefs)

    router_input = _router_input_payload(runner, snap)
    _json_dump(step_dir / "router_input.json", router_input)

    t1 = time.perf_counter()
    router_result = runner.gpt.propose_router_answers(
        screenshot_b64=screenshot_b64,
        router_questions=list(getattr(runner.questionnaires, "routers", []) or []),
        app_intro=runner.app_intro,
        focus_hints=runner.focus_hints,
        state_sig=state_sig,
    )
    router_elapsed_ms = (time.perf_counter() - t1) * 1000.0
    router_payload = router_result.model_dump(mode="json")
    _json_dump(step_dir / "router_output.json", router_payload)

    router_updates = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else getattr(item, "__dict__", {})
        for item in (getattr(router_result, "router_updates", None) or [])
    ]
    matched_blocks = runner.questionnaires.match_blocks_from_router_answers(router_updates)
    runner.questionnaires.mark_blocks_hit(matched_blocks)

    matched_payload: List[Dict[str, Any]] = []
    for block in matched_blocks:
        questions = block.get("questions") or {}
        matched_payload.append(
            {
                "id": str(block.get("id") or ""),
                "module": str(block.get("module") or ""),
                "topic": str(block.get("topic") or ""),
                "question_count": int(len(questions)),
                "question_ids": list(questions.keys())[:80],
            }
        )
    _json_dump(step_dir / "router_matched_blocks.json", matched_payload)

    analysis_summary = {
        "saved_at_ms": int(time.time() * 1000),
        "state_sig": state_sig,
        "nav_elapsed_ms": round(nav_elapsed_ms, 3),
        "router_elapsed_ms": round(router_elapsed_ms, 3),
        "nav_overlay_kind": str(getattr(nav, "overlay_kind", "")),
        "nav_candidate_count_raw": len(getattr(nav, "candidate_actions", []) or []),
        "nav_candidate_count_filtered": len(candidates),
        "router_update_count": len(router_updates),
        "matched_block_count": len(matched_payload),
    }
    _json_dump(step_dir / "analysis_summary.json", analysis_summary)
    return {
        "summary": analysis_summary,
        "nav": nav,
        "candidates": candidates,
        "candidate_briefs": candidate_briefs,
        "router_updates": router_updates,
        "matched_blocks": matched_payload,
    }


def _print_help() -> None:
    print("命令说明:")
    print("  回车: 按当前阶段执行（抓取 / 分析 / 执行）")
    print("  c    : 强制切回抓取阶段")
    print("  a    : 在分析阶段执行分析")
    print("  在执行阶段输入数字: 执行对应候选动作（例: 1）")
    print("  h    : 显示帮助")
    print("  q    : 退出")


def _print_candidate_menu(candidate_briefs: List[Dict[str, Any]]) -> None:
    if not candidate_briefs:
        print("当前没有可执行候选动作。")
        return
    print("\n候选动作列表:")
    for row in candidate_briefs:
        idx = int(row.get("index") or 0)
        score = float(row.get("score") or 0.0)
        preview = "; ".join(row.get("actions_preview") or [])
        print(f"  {idx}. score={score:.3f} {preview}")


def run_interactive_debug(args: argparse.Namespace) -> int:
    setup_logging(debug=bool(args.debug), level="INFO")

    run_stamp = str(args.run_id or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    package_token = _safe_token(args.package or "current_foreground")
    run_id = f"{run_stamp}_{package_token}"
    run_dir = Path(str(args.output_root)).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(args, run_id=run_id)

    session_meta = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    }
    _json_dump(run_dir / "session_meta.json", session_meta)

    print("=== LLM Navigation + Block Router Interactive Debug ===")
    print(f"输出目录: {run_dir}")
    print("说明: 不会启动/重启目标 APP，只会连接 Appium 后抓取当前前台页面。")
    _print_help()

    phase = "capture"
    step_idx = 0
    current_snap: Optional[Dict[str, Any]] = None
    current_step_dir: Optional[Path] = None
    last_analysis: Optional[Dict[str, Any]] = None

    try:
        while True:
            if phase == "capture":
                cmd = input("\n[capture] 回车抓取当前页面, q退出, h帮助 > ").strip().lower()
                if cmd == "q":
                    break
                if cmd == "h":
                    _print_help()
                    continue
                if cmd not in {"", "c"}:
                    print(f"未知命令: {cmd}")
                    continue

                step_idx += 1
                current_step_dir = run_dir / f"step_{step_idx:03d}"

                t0 = time.perf_counter()
                snap = runner._capture_and_process(timeout=float(args.capture_timeout))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if not snap:
                    print("抓取失败: _capture_and_process 返回 None")
                    step_idx -= 1
                    continue

                current_snap = snap
                summary = _save_capture(current_step_dir, current_snap, prefix="")
                summary["capture_elapsed_ms"] = round(elapsed_ms, 3)
                _json_dump(current_step_dir / "summary.json", summary)
                print(
                    f"[step {step_idx:03d}] 抓取完成: state_sig={summary.get('state_sig')} "
                    f"xml_reliable={summary.get('xml_reliable')} "
                    f"uist_nodes={summary.get('uist_total_nodes')} "
                    f"vid_map={summary.get('vid_map_count')}"
                )
                print(f"[step {step_idx:03d}] 文件已保存: {current_step_dir}")
                phase = "analyze"
                last_analysis = None
                continue

            cmd = input("\n[analyze] 回车分析当前快照, c重新抓取, q退出, h帮助 > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "h":
                _print_help()
                continue
            if cmd == "c":
                phase = "capture"
                continue
            if cmd not in {"", "a"}:
                print(f"未知命令: {cmd}")
                continue
            if current_snap is None or current_step_dir is None:
                print("当前没有可分析快照，请先抓取。")
                phase = "capture"
                continue

            try:
                result = _run_llm_analysis(runner, current_snap, current_step_dir, task=str(args.task))
            except Exception as exc:
                error_payload = {
                    "saved_at_ms": int(time.time() * 1000),
                    "state_sig": str(current_snap.get("state_sig") or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _json_dump(current_step_dir / "analysis_error.json", error_payload)
                print(f"分析失败，详情见: {current_step_dir / 'analysis_error.json'}")
                logger.exception("LLM analysis failed at step %s", step_idx)
                phase = "capture"
                continue

            last_analysis = result
            summary = result["summary"]
            print(
                f"[step {step_idx:03d}] 分析完成: "
                f"nav_ms={summary.get('nav_elapsed_ms')} "
                f"router_ms={summary.get('router_elapsed_ms')} "
                f"nav_candidates_raw={summary.get('nav_candidate_count_raw')} "
                f"nav_candidates_filtered={summary.get('nav_candidate_count_filtered')} "
                f"router_updates={summary.get('router_update_count')} "
                f"matched_blocks={summary.get('matched_block_count')}"
            )
            print(f"[step {step_idx:03d}] 结果文件: {current_step_dir}")
            _print_candidate_menu(result.get("candidate_briefs") or [])
            phase = "execute"
            while phase == "execute":
                cmd2 = input(
                    "\n[execute] 回车执行第1个候选, 输入序号执行, a重新分析, c重抓, s跳过执行, q退出 > "
                ).strip().lower()
                if cmd2 == "q":
                    return 0
                if cmd2 == "h":
                    _print_help()
                    _print_candidate_menu((last_analysis or {}).get("candidate_briefs") or [])
                    continue
                if cmd2 == "c":
                    phase = "capture"
                    break
                if cmd2 == "a":
                    phase = "analyze"
                    break
                if cmd2 == "s":
                    phase = "capture"
                    break

                candidate_briefs = (last_analysis or {}).get("candidate_briefs") or []
                candidates = (last_analysis or {}).get("candidates") or []
                if not candidate_briefs or not candidates:
                    print("没有可执行候选，请先重新抓取或分析。")
                    phase = "capture"
                    break

                if cmd2 == "":
                    choose_idx = 1
                else:
                    if not cmd2.isdigit():
                        print(f"无效输入: {cmd2}")
                        continue
                    choose_idx = int(cmd2)
                if choose_idx < 1 or choose_idx > len(candidates):
                    print(f"候选序号越界: {choose_idx}, 可选范围 1..{len(candidates)}")
                    continue

                selected = candidates[choose_idx - 1]
                selected_brief = candidate_briefs[choose_idx - 1]
                action_input = {
                    "saved_at_ms": int(time.time() * 1000),
                    "state_sig": str(current_snap.get("state_sig") or ""),
                    "selected_index": choose_idx,
                    "candidate_key": selected_brief.get("candidate_key"),
                    "candidate": selected_brief.get("candidate"),
                    "available_candidates_count": len(candidates),
                }
                _json_dump(current_step_dir / "action_input.json", action_input)

                source_sig = str(current_snap.get("state_sig") or "")
                action_count_before = int(runner.action_count)
                t_exec = time.perf_counter()
                ok = runner._execute_action_sequence(list(getattr(selected, "actions", []) or []), current_snap)
                exec_elapsed_ms = (time.perf_counter() - t_exec) * 1000.0

                action_result: Dict[str, Any] = {
                    "saved_at_ms": int(time.time() * 1000),
                    "source_state_sig": source_sig,
                    "selected_index": choose_idx,
                    "candidate_key": selected_brief.get("candidate_key"),
                    "execute_ok": bool(ok),
                    "execute_elapsed_ms": round(exec_elapsed_ms, 3),
                    "action_count_before": action_count_before,
                    "action_count_after": int(runner.action_count),
                }

                if not ok:
                    _json_dump(current_step_dir / "action_result.json", action_result)
                    print(f"[step {step_idx:03d}] 动作执行失败，详情见 action_result.json")
                    phase = "capture"
                    break

                t_post = time.perf_counter()
                snap_next = runner._capture_and_process(timeout=float(args.capture_timeout))
                post_elapsed_ms = (time.perf_counter() - t_post) * 1000.0
                action_result["post_action_capture_elapsed_ms"] = round(post_elapsed_ms, 3)

                if snap_next:
                    post_summary = _save_capture(current_step_dir, snap_next, prefix="post_action_")
                    post_summary["capture_elapsed_ms"] = round(post_elapsed_ms, 3)
                    _json_dump(current_step_dir / "post_action_summary.json", post_summary)

                    transition = runner._actions_signature(
                        list(getattr(selected, "actions", []) or []),
                        vid_map=current_snap.get("vid_map") or {},
                    )
                    try:
                        dst_was_new = runner._graph_record_transition(
                            source_sig,
                            str(snap_next.get("state_sig") or ""),
                            transition,
                            src_snap=current_snap,
                            dst_snap=snap_next,
                        )
                    except Exception:
                        dst_was_new = False

                    action_result["post_action_capture_ok"] = True
                    action_result["dst_state_sig"] = str(snap_next.get("state_sig") or "")
                    action_result["dst_was_new"] = bool(dst_was_new)
                    current_snap = snap_next
                else:
                    action_result["post_action_capture_ok"] = False

                _json_dump(current_step_dir / "action_result.json", action_result)
                print(
                    f"[step {step_idx:03d}] 动作执行完成: ok={action_result['execute_ok']} "
                    f"dst={action_result.get('dst_state_sig', '')} "
                    f"post_capture={action_result.get('post_action_capture_ok')}"
                )
                phase = "capture"
                break

    finally:
        runner.appium.quit()

    print("调试结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_interactive_debug(parse_args()))
