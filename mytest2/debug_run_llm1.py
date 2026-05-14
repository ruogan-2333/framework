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

from dotenv import load_dotenv
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_cls import ActionStep, ActionType
from appium_android import AndroidAppiumClient
from gpt_cls import GPTClient, OverlayKind
from questionnaire_state import QuestionnaireState
from trace_callbacks import NoOpCallbacks
from workflow import BudgetConfig, RecoveryReason, WorkflowRunner


load_dotenv()
logger = logging.getLogger(__name__)
OUTPUT_ROOT = PROJECT_ROOT / "mytest2" / "outputs"
LLM1_OUTPUT_ROOT = PROJECT_ROOT / "mytest2" / "llm1_debug"


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
# package="com.Polystation.BuildABabe"
# package="com.netease.uu"
sys.argv = [
    sys.argv[0],
    # "--appium-url", "http://localhost:4723",
    "--appium-url", "http://127.0.0.1:4723",
    "--device-name", "emulator-5554",
    "--package", package,
    # "--questionnaire-dir", "./questionnaire-v2/others",
    "--questionnaire-dir", "../questionnaire-v2/games",
    # "--relaunch",
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


def _write_debug_outputs(snap: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "snap.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    if snap.get("xml"):
        (out_dir / "xml.xml").write_text(str(snap.get("xml") or ""), encoding="utf-8")
    if snap.get("xml_raw"):
        (out_dir / "xml_raw.xml").write_text(str(snap.get("xml_raw") or ""), encoding="utf-8")

    if snap.get("screenshot"):
        (out_dir / "screenshot.png").write_bytes(base64.b64decode(str(snap.get("screenshot") or "") + "=="))
    if snap.get("screenshot_raw"):
        (out_dir / "screenshot_raw.png").write_bytes(base64.b64decode(str(snap.get("screenshot_raw") or "") + "=="))

    if snap.get("uist") is not None:
        (out_dir / "uist.json").write_text(json.dumps(snap.get("uist") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    if snap.get("vid_map") is not None:
        (out_dir / "vid_map.json").write_text(json.dumps(snap.get("vid_map") or {}, ensure_ascii=False, indent=2), encoding="utf-8")

    if snap.get("screenshot") and snap.get("vid_map"):
        _draw_vid_map_overlay(str(snap.get("screenshot") or ""), snap.get("vid_map") or {}, out_dir / "vid_map_overlay.png")

    return out_dir


def _safe_json_dump(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    return value


def _write_llm1_outputs(
    loop_dir: Path,
    *,
    cur_sig: str,
    task: str,
    snap: dict,
    nav: object,
    using_heuristics: bool,
    candidates: list,
    forward: object,
    forward_detail: dict | None,
) -> None:
    llm1_input = {
        "state_sig": cur_sig,
        "task": task,
        "xml_reliable": snap.get("xml_reliable"),
        "identity_source": (snap.get("meta") or {}).get("identity_source"),
        "foreground_package": (snap.get("meta") or {}).get("foreground_package"),
        "foreground_activity": (snap.get("meta") or {}).get("foreground_activity"),
        "vid_count": len(snap.get("vid_map") or {}),
    }
    (loop_dir / "llm1_input.json").write_text(json.dumps(llm1_input, ensure_ascii=False, indent=2), encoding="utf-8")

    llm1_output = {
        "using_heuristics": using_heuristics,
        "nav": _safe_json_dump(nav),
        "candidates": [_safe_json_dump(c) for c in (candidates or [])],
        "forward": _safe_json_dump(forward),
        "forward_detail": forward_detail or {},
    }
    (loop_dir / "llm1_output.json").write_text(json.dumps(llm1_output, ensure_ascii=False, indent=2), encoding="utf-8")


def debug_run_test(runner: WorkflowRunner, task: str) -> None:
    """
    Copy of WorkflowRunner.run() for local debugging.
    You can edit this function freely without touching the framework's real run().
    """
    start = time.time()

    if runner.target_package:
        runner.appium.ensure_foreground(runner.target_package, runner.target_activity)

    runner._pool = ThreadPoolExecutor(max_workers=runner.budget.max_workers)
    runner.budget.enable_probe_return = False
    run_root = LLM1_OUTPUT_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{(runner.target_package or 'app').replace('.', '_')}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"LLM1调试输出目录: {run_root}")

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
    _write_debug_outputs(snap, run_root / "loop_000_entry")

    loop_index = 0

    while True:
            loop_index += 1
            loop_dir = run_root / f"loop_{loop_index:03d}"
            _write_debug_outputs(snap, loop_dir)
            # # 每轮一开始先看是否应该结束，例如超时、动作数超限、长时间没新状态等。
            # if runner._stop_condition(start):
            #     # 记录停止日志，方便从 trace / log 中看 run 为什么结束。
            #     logger.info("Stop condition reached.")
            #     # 跳出主循环，进入资源清理阶段。
            #     break

            # # Each loop without progress increments; _mark_progress resets it.
            # # 默认先把“无进展轮数”加一；如果本轮真的有进展，后续会被 _mark_progress 重置。
            runner.no_progress_loops += 1

            # WHERE: main loop top; drain completed LLM futures so decisions see latest analysis.
            # 回收已经完成的异步 LLM 任务，把结果写入缓存 / 问卷状态，保证本轮决策看到的是最新信息。
            runner._drain_futures()

            # 读取当前仍未填完的问卷问题，作为主循环的目标集合。
            gaps = runner.questionnaires.open_gaps()
            # 计算当前问卷完成比例，用于日志和调试观察整体推进情况。
            answered_ratio = runner.questionnaires.progress_score()

            # # 打印当前循环的核心状态，包括页面 sig、栈深、动作数、剩余 gap、异步任务数量等。
            # logger.info(
            #     "Loop: sig=%s stack_depth=%d actions=%d gaps=%d answered=%.2f pending(nav=%d route=%d fill=%d) no_progress=%d no_new=%d",
            #     cur_sig[:8],
            #     len(runner.dfs_stack),
            #     runner.action_count,
            #     len(gaps),
            #     answered_ratio,
            #     len(runner._nav_futures),
            #     len(runner._topic_route_futures),
            #     len(runner._topic_fill_futures),
            #     runner.no_progress_loops,
            #     runner.no_new_state_count,
            # )

            # # 如果已经没有未完成问卷项，说明本次探索目标达成，可以结束。
            # if not gaps:
            #     # 记录“所有 gap 都填完”的结束原因。
            #     logger.info("All questionnaire gaps filled. Stopping.")
            #     # 跳出主循环。
            #     break





            # 页面前台检测部分:


            # CONDITION: FOREGROUND PACKAGE MISMATCH (ads/external browser/system overlays)
            # 先检查前台包名是否还在目标 app 内；如果跳去了广告、浏览器、系统页，需要先拉回来。
            gate_snap = runner._handle_foreground_gate(cur_sig, snap, task)
            # 如果前台修复逻辑真的造成了页面变化，这里会返回一个新的快照。
            if gate_snap:
                # 用修复后的快照替换当前快照。
                print ("前台APP错误,切换前台APP并重新分析.")
                snap = gate_snap
                # 同步更新当前状态签名。
                cur_sig = snap["state_sig"]
                # We can't safely attribute this relocation to a precise UI edge.
                # 这种“位置变更”通常不是由一个明确的业务动作引起，所以按 external move 方式修正栈结构。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=True)
                # 对修复后的新页面重新安排分析任务。
                runner._schedule_state(cur_sig, snap, task)
                # 本轮剩余逻辑作废，直接开始下一轮。
                continue



            # Drift检测部分:


            # CONDITION: DRIFT (UI changed without our explicit logged action)
            # WHEN/WHERE:
            # - checked once per loop BEFORE trusting NAV proposals/caches for cur_sig.
            # 在真正相信当前缓存和导航建议之前，先做一次轻量预检，看看页面是否自己变了。
            snap_live = runner._preflight_refresh_if_changed(snap, timeout_s=0.8)
            # 如果预检发现页面确实发生了变化，就进入 drift 分支。
            if snap_live:
                # 如果状态签名已经变了，说明页面漂移到了另一个状态，当前计划要作废重排。
                if snap_live["state_sig"] != cur_sig:
                    # 取出漂移后的新状态签名。
                    new_sig: str = snap_live["state_sig"]
                    # 输出 drift 日志，帮助定位“页面自己变了”的情况。
                    logger.warning("触发DRIFT: 原页面=%s 新页面=%s. 重新分析.", cur_sig[:8], new_sig[:8])
                    # logger.warning("DRIFT: old=%s new=%s. Replanning.", cur_sig[:8], new_sig[:8])
                    # 发出决策事件，标记这是一次 drift 导致的重规划。
                    runner._emit_decision(cur_sig, "drift_detected", {"from_sig": cur_sig, "to_sig": new_sig})

                    # 构造一个伪动作，用来在状态图里表示“等待时页面自己变了”这类漂移迁移。
                    drift_step = ActionStep(action=ActionType.WAIT, element_id=None, text="0.2", priority=1, reasoning="drift")
                    # 把 drift 也记成一条状态迁移边，并拿到目标状态在发现当时是否是新状态。
                    dst_was_new = runner._graph_record_transition(
                        cur_sig,
                        new_sig,
                        runner._actions_signature([drift_step]),
                        src_snap=snap,
                        dst_snap=snap_live,
                    )
                    # 更新 DFS 路径，让当前所在位置切换到漂移后的状态。
                    runner._enter_state(from_sig=cur_sig, to_sig=new_sig, via_action="drift")

                    # 当前状态与快照都切到漂移后的页面。
                    cur_sig, snap = new_sig, snap_live
                    # 对新状态重新安排异步分析。
                    runner._schedule_state(cur_sig, snap, task)

                    # no_new_state_count tracks novelty-at-discovery
                    # 如果漂移到了一个从未见过的新状态，就清零“无新状态”计数。
                    if dst_was_new:
                        runner.no_new_state_count = 0
                    # 否则说明只是跳到旧状态，“无新状态”计数继续累计。
                    else:
                        runner.no_new_state_count += 1
                    # 当前轮因为页面基准已经变化，直接重开下一轮。
                    continue

                # Same state_sig but raw content changed; refresh snap for accurate vid_map.
                # 如果 sig 没变但底层内容更新了，也要用新快照替换，保证 vid_map / 元素 id 不过期。
                snap = snap_live



            
            # NAV barrier:
            # - may wait; may detect drift while waiting; may timeout -> heuristics
            # 在根节点适当放宽等待时间，给首页导航分析更多时间。
            timeout_s = runner.budget.nav_timeout_s + (30.0 if len(runner.dfs_stack) == 1 else 0.0)
            runner._emit_decision(
                cur_sig,
                "next_step",
                {"plan": "wait_nav_or_fallback", "timeout_s": timeout_s, "stack_depth": len(runner.dfs_stack)},
            )
            # 等待 NAV 结果；如果超时或进入 cooldown，则允许走启发式候选。
            nav, using_heuristics = runner._wait_nav_or_fallback(cur_sig, snap, task, timeout_s=timeout_s)

            # NAV wait can set forced replan (drift while waiting)
            # NAV 等待期间也可能检测到 drift / 强制重规划信号，这里优先处理。
            if runner._consume_forced_replan():
                # 取出强制重规划指定的新状态与快照。
                cur_sig, snap = runner._force_replan_sig, runner._force_replan_snap  # type: ignore[assignment]
                # 标记这次重规划前是否已经记录过迁移边。
                has_edge = runner._force_replan_has_edge
                # 清空强制重规划标记，避免后续重复消费。
                runner._clear_forced_replan()
                # 把当前栈位置修正到新状态；如果已经有边，就不再补 observation。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=not has_edge)
                # 为新页面重新安排分析。
                runner._schedule_state(cur_sig, snap, task)
                # 直接进入下一轮。
                continue

            # Candidate memory for DFS completion
            # 如果拿到了 NAV 候选，把这组候选记到当前页面 family 上，供 DFS exhausted 判定使用。
            if nav is not None and getattr(nav, "candidate_actions", None) is not None:
                runner.state_candidates[runner._family_id(cur_sig)] = list(getattr(nav, "candidate_actions") or [])

            # Track overlay info into graph node WITHOUT inflating visit_count.
            # 把 overlay 类型写回状态图节点元信息，但不增加 visit_count，避免污染统计。
            if nav:
                runner._graph_annotate(cur_sig, overlay_kind=runner._overlay_kind_value(nav))

            # 从 NAV 结果里提取当前页面的 overlay 类型，后面按类型分流处理。
            overlay_kind = runner._overlay_kind_value(nav)

            # CONDITION: OVERLAY (blocking dismiss / loading)
            # 如果当前页面被可关闭弹窗阻塞，先优先处理弹窗，而不是继续正常探索。
            if overlay_kind == OverlayKind.DISMISS.value:
                # 记录“发现可关闭 overlay”的信息日志。
                logger.info("Overlay(DISMISS) at sig=%s: %s", cur_sig[:8], (getattr(nav, "overlay_reason", "") or "")[:140])
                # 把 overlay 检测结果写入 trace / log。
                runner._log_event("overlay_detected", sig=cur_sig, overlay_kind=overlay_kind, reason=getattr(nav, "overlay_reason", ""))

                # 先尝试按 LLM1 给出的 dismiss 动作去关闭弹窗。
                resolved = runner._dismiss_overlay_with_nav(cur_sig, snap, nav, task)
                # 如果 LLM1 没能关掉弹窗，则进入更通用的恢复流程。
                if not resolved:
                    # 记录告警，说明 overlay 处理失败，准备 recovery。
                    logger.warning("Overlay unresolved by LLM1 overlay_dismiss_actions; invoking recovery.")
                    # 调用 recovery 尝试把页面带回稳定非阻塞状态。
                    ok = runner._recover(cur_sig, snap, reason=RecoveryReason.OVERLAY_UNRESOLVED, task=task, target_sig=None)
                    # 如果 recovery 也失败，只能通过重启 app + best-effort replay 自救。
                    if not ok:
                        # 输出错误日志，说明进入最重的恢复分支。
                        logger.error("Recovery failed. Restart app + replay best-effort.")
                        # 重启应用，并尽可能回到可继续探索的位置。
                        runner._restart_and_replay(best_target=None, task=task, reason="overlay_unresolved_recovery_failed")

                # After overlay/recovery/restart: re-capture and replan at the true current state.
                # 不管 overlay 是怎么被解除的，最后都重新抓一次真实页面，重新建立当前基准。
                snap = runner._capture_and_process() or snap
                # 同步更新当前状态签名。
                cur_sig = snap["state_sig"]
                # overlay 关闭后的状态通常不适合直接沿用旧栈，需要做一次外部移动式修正。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                # 对新页面重新安排分析。
                runner._schedule_state(cur_sig, snap, task)
                # 本轮结束，开始下一轮。
                continue

            # 如果是 loading overlay，则先等它过去，而不是盲目点页面。
            if overlay_kind == OverlayKind.LOADING.value:
                # 记录 loading overlay 的出现。
                logger.info("Overlay(LOADING) at sig=%s: %s", cur_sig[:8], (getattr(nav, "overlay_reason", "") or "")[:140])
                # 把 loading 事件写入 trace。
                runner._log_event("overlay_loading", sig=cur_sig, reason=getattr(nav, "overlay_reason", ""))
                # 调用 loading 专用处理逻辑，例如等待页面稳定。
                runner._handle_loading_overlay(cur_sig, snap, task)
                # loading 结束后重新抓快照，拿到真正稳定的页面状态。
                snap = runner._capture_and_process() or snap
                # 更新当前 sig。
                cur_sig = snap["state_sig"]
                # 修正 DFS 栈，使其与真实当前位置一致。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                # 重新安排后续分析。
                runner._schedule_state(cur_sig, snap, task)
                # 当前轮结束。
                continue

            # WORKFLOW 类型 overlay 不是阻塞弹窗，而是流程提示类信息，这里先只记录下来供调试使用。
            if overlay_kind == OverlayKind.WORKFLOW.value:
                runner._log_event("overlay_workflow", sig=cur_sig, reason=getattr(nav, "overlay_reason", ""), hints=getattr(nav, "workflow_hints", []))

            # Candidates: NAV if available; heuristics ONLY if NAV timed out/cooldown.
            # 生成本轮可尝试的候选动作；优先使用 NAV 结果，只有 NAV 不可用时才退化到启发式。
            candidates = runner._candidate_actions(nav=nav, snap=snap, allow_heuristics=using_heuristics)
            runner._emit_decision(
                cur_sig,
                "next_step",
                {
                    "plan": "evaluate_candidates",
                    "candidate_count": len(candidates or []),
                    "using_heuristics": bool(using_heuristics),
                    "overlay_kind": overlay_kind,
                },
            )
            # 如果本轮拿到的是正常 NAV 结果，就顺手检查候选是否还有真正可用的动作。
            if nav is not None and (not using_heuristics):
                # 统计有没有至少一个候选动作还没被探索、尝试或拉黑。
                usable = 0
                # 逐个检查候选。
                for c in candidates:
                    # 没有动作内容的候选直接跳过。
                    if not getattr(c, "actions", None):
                        continue
                    # 构造候选动作的稳定 key，方便和尝试/黑名单集合做比对。
                    ckey = runner._candidate_key(c)
                    # 如果这个候选已经探索过、尝试过或被临时拉黑，就不算“可用”。
                    if runner._is_explored(cur_sig, c) or runner._already_attempted(cur_sig, c) or runner._is_action_blacklisted(cur_sig, ckey):
                        continue
                    # 找到一个可用候选就够了，不需要继续统计。
                    usable += 1
                    # 直接结束循环。
                    break
                # 如果 NAV 给了一堆候选，但最后一个都用不上，说明这次 NAV 质量偏低。
                if usable == 0:
                    # 记录低质量 NAV 的上下文信息，方便后续排查模型输出问题。
                    runner._log_event(
                        "nav_low_quality",
                        sig=cur_sig,
                        candidate_count=len(getattr(nav, "candidate_actions", []) or []),
                        overlay_kind=runner._overlay_kind_value(nav),
                        has_page_summary=bool(getattr(nav, "page_summary", "") or ""),
                        tag_count=len(getattr(nav, "page_tags", []) or []),
                    )
                    print("没找到可执行的动作。")
                    # 发出一个决策事件，标记这次 NAV 结果虽然完成了，但实际不可用。
                    runner._emit_decision(cur_sig, "nav_low_quality", {"candidate_count": len(getattr(nav, "candidate_actions", []) or [])})

            # Probe-return exploration (evidence gathering for forward scoring)
            # 先对候选做 probe 探测，而不是立刻前进；目的是先知道这些动作分别会通向哪里。
            if runner.budget.enable_probe_return:
                runner._emit_decision(
                    cur_sig,
                    "next_step",
                    {"plan": "probe_candidates", "candidate_count": len(candidates or []), "probe_cap": int(runner.budget.per_page_probe_cap)},
                )
                runner._probe_candidates(cur_sig, snap, nav, candidates, task)
            else:
                runner._log_event("probe_skipped", sig=cur_sig, reason="probe_return_disabled")
                runner._emit_decision(cur_sig, "probe_skipped", {"reason": "probe_return_disabled"})

            # Probe can request forced replan (back-like / return_method=none / source mismatch)
            # probe 期间如果发现“实际上已经跳走了”，这里会触发强制重规划。
            if runner.budget.enable_probe_return and runner._consume_forced_replan():
                # 切换到 probe 过程确定的新状态与快照。
                cur_sig, snap = runner._force_replan_sig, runner._force_replan_snap  # type: ignore[assignment]
                # 读取是否已有记录边。
                has_edge = runner._force_replan_has_edge
                # 清除该标记。
                runner._clear_forced_replan()
                # 根据是否已有边来修正当前位置与 DFS 栈。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=not has_edge)
                # 对当前位置重新安排分析。
                runner._schedule_state(cur_sig, snap, task)
                # 结束当前轮。
                continue

            # Optional post-probe wait (let pipelined Q updates land)
            # probe 之后可短暂等待一下，让刚刚 pipeline 出去的问卷更新任务有时间完成并落回状态。
            if runner.budget.enable_probe_return:
                runner._post_probe_wait()

            # Choose forward commit based on probe evidence + novelty + questionnaire yield.
            # 基于 probe 的落点、新颖度、问卷收益等信息，挑出本轮真正要 commit 的前进动作。
            runner._emit_decision(cur_sig, "next_step", {"plan": "choose_forward"})
            forward = runner._choose_forward(cur_sig, nav, candidates=candidates)
            _write_llm1_outputs(
                loop_dir,
                cur_sig=cur_sig,
                task=task,
                snap=snap,
                nav=nav,
                using_heuristics=using_heuristics,
                candidates=candidates,
                forward=forward,
                forward_detail=runner._last_forward_detail,
            )

            # Beam switching compares against a *verified* local option only.
            # If forward is speculative (no probe outcome), treat local_score as -inf.

            # 计算当前候选动作的beam_saerch分数
            # 默认把本地候选分数记成负无穷，意思是“还没有足够证据与全局切换比较”。
            local_score = float("-inf")
            # 只有当 forward 候选有明确评分且来自 probe 证据时，才把它当作有效本地方案。
            if forward is not None and runner._last_forward_detail and runner._last_forward_detail.get("score") is not None:
                try:
                    # 拿到 forward 候选的稳定 key。
                    fkey = runner._candidate_key(forward)
                    # 查出 probe 期间这个候选实际落到的目标状态。
                    dst_sig = (runner.probe_outcomes.get(runner._family_id(cur_sig), {}) or {}).get(fkey)
                    # 只有确实落到了不同页面，才算一个有效的“前进选项”。
                    dst_ok = bool(dst_sig and dst_sig != cur_sig)
                    # 目标状态不能已经被探索穷尽。
                    if dst_ok and (not runner._is_state_exhausted(str(dst_sig))):
                        # 看一下目标状态是否已经被判断为 overlay 阻塞页。
                        dst_nav = runner.nav_cache.get(str(dst_sig))
                        # 提取目标状态的 overlay 类型。
                        overlay = runner._overlay_kind_value(dst_nav)
                        # 只有目标状态不是 dismiss/loading 阻塞页，才采用它的本地评分。
                        if overlay not in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
                            local_score = float(runner._last_forward_detail.get("score") or 0.0)
                # 如果评分计算过程中出了异常，退回到“没有可靠本地分数”。
                except Exception:
                    local_score = float("-inf")

            # Beam-first: switch to a higher-value global opportunity if worth the travel cost.
            # 在真正前进之前，再比较一下全局状态图里是否存在更值得切换过去的目标页面。
            beam_snap = runner._maybe_beam_switch(cur_sig, local_score, snap, task)
            # 如果 beam 策略决定切换，并且已经把我们导航到了新页面，就以新页面作为当前基准继续。
            if beam_snap:
                # 更新当前快照。
                snap = beam_snap
                # 更新当前状态签名。
                cur_sig = snap["state_sig"]
                # 对“通过 beam 跳转后的位置”修正 DFS 栈。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                # 为新页面重新安排分析。
                runner._schedule_state(cur_sig, snap, task)
                # 当前轮结束。
                continue

            # 如果没有选出 forward 候选，说明当前页此刻没有明确的前进动作可 commit。
            if forward is None:
                # Only backtrace when truly exhausted (all branches explored).
                # 如果当前状态已经被完全探索完，就优先考虑 DFS 回退，而不是原地乱试。
                if runner._is_state_exhausted(cur_sig):
                    # 找最近一个还有剩余工作可做的祖先状态。
                    target = runner._nearest_ancestor_with_remaining_work(cur_sig)

                    # 如果能找到这样的祖先，就执行回溯。
                    if target:
                        # 记录回溯目标。
                        logger.info("Exhausted sig=%s. Backtrace to ancestor sig=%s.", cur_sig[:8], target[:8])
                        # 试着按 DFS 路径退回去。
                        runner._emit_decision(cur_sig, "next_step", {"plan": "backtrace_to_ancestor", "target_sig": target})
                        ok = runner._backtrace_to(target_sig=target, task=task, start_snap=snap)
                        # 如果回溯失败，则进入恢复流程。
                        if not ok:
                            # 记录回溯失败。
                            logger.warning("Backtrace failed. Recover/restart.")
                            # recovery 尝试把我们拉回目标祖先或稳定页。
                            ok2 = runner._recover(cur_sig, snap, reason=RecoveryReason.BACKTRACE_FAILED, task=task, target_sig=target)
                            # 如果 recovery 还失败，再升级到重启 + replay。
                            if not ok2:
                                runner._restart_and_replay(best_target=target, task=task, reason="backtrace_failed_recovery_failed")

                        # 回溯 / 恢复 / 重启之后，重新抓当前真实页面。
                        snap = runner._capture_and_process() or snap
                        # 更新当前 sig。
                        cur_sig = snap["state_sig"]
                        # 修正 DFS 栈与当前位置。
                        runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                        # 重新安排分析。
                        runner._schedule_state(cur_sig, snap, task)
                        # 当前轮结束。
                        continue

                    # 如果祖先里也没有剩余工作，就尝试从全局状态图里挑一个 frontier 状态。
                    frontier = runner._pick_global_frontier()
                    # 只有 frontier 合法且不是当前页时，才值得进行跨分支跳转。
                    if frontier and frontier != cur_sig:
                        # 记录准备前往全局 frontier。
                        logger.info("No ancestor work. Attempt to reach global frontier sig=%s", frontier[:8])
                        # 先尝试不重启，直接沿已知状态图导航过去。
                        runner._emit_decision(cur_sig, "next_step", {"plan": "navigate_to_global_frontier", "target_sig": frontier})
                        reached, snap2 = runner._navigate_via_graph(start_sig=cur_sig, start_snap=snap, target_sig=frontier, task=task)
                        # 如果状态图导航失败，则升级到重启 + replay。
                        if not reached:
                            runner._restart_and_replay(best_target=frontier, task=task, reason="global_frontier_unreachable")
                            # 重启后重新抓一次页面，得到最新快照。
                            snap2 = runner._capture_and_process(timeout=10.0) or snap2
                        # 采用导航后的快照；如果没有新快照则保留原快照。
                        snap = snap2 or snap
                        # 更新当前 sig。
                        cur_sig = snap["state_sig"]
                        # 修正当前位置在 DFS 栈中的表达。
                        runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                        # 对新页面重新安排分析。
                        runner._schedule_state(cur_sig, snap, task)
                        # 本轮结束。
                        continue

                # 如果当前页不是 exhausted，但整体表现出“卡住”征兆，就走恢复分支。
                if runner._should_recover_stuck():
                    # 打印 stuck 告警，包括卡了多少轮、多久没强进展。
                    logger.warning(
                        "STUCK: no progress loops=%d age=%.1fs at sig=%s -> recovery.",
                        runner.no_progress_loops,
                        (time.time() - runner.last_strong_progress_ts),
                        cur_sig[:8],
                    )
                    # 把 stuck 事件写入日志与 trace。
                    runner._log_event("stuck_detected", sig=cur_sig, loops=runner.no_progress_loops, age_s=(time.time() - runner.last_strong_progress_ts))
                    # 尝试 recovery，把流程拉回一个可继续探索的状态。
                    runner._emit_decision(
                        cur_sig,
                        "next_step",
                        {"plan": "recover_stuck", "loops": runner.no_progress_loops, "age_s": (time.time() - runner.last_strong_progress_ts)},
                    )
                    ok = runner._recover(cur_sig, snap, reason=RecoveryReason.STUCK_NO_PROGRESS, task=task, target_sig=None)
                    # recovery 失败则走重启恢复。
                    if not ok:
                        runner._restart_and_replay(best_target=None, task=task, reason="stuck_recovery_failed")

                    # 恢复后重新抓当前页面。
                    snap = runner._capture_and_process() or snap
                    # 更新当前 sig。
                    cur_sig = snap["state_sig"]
                    # 修正当前位置与 DFS 栈。
                    runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                    # 重新安排分析。
                    runner._schedule_state(cur_sig, snap, task)
                else:
                    # keep scheduling/waiting; do not heuristic-click unless NAV timed out
                    # 如果既不 exhausted 也不 stuck，就继续等待更多分析结果，不强行点击。
                    runner._schedule_state(cur_sig, snap, task)
                # forward 不存在的分支到这里结束，本轮不再执行下面的 commit 逻辑。
                continue

            # Execute forward move (commit)
            # 走到这里，说明已经为当前页选出了一个真正值得提交执行的前进候选。
            cand_key = runner._candidate_key(forward)
            # 取出第一步动作附带的 reasoning，方便日志里看到“为什么选它”。
            first_reason = (forward.actions[0].reasoning if forward.actions else "") or ""
            # 记录本轮前进提交的候选动作。
            runner._emit_decision(
                cur_sig,
                "next_step",
                {"plan": "forward_commit", "candidate_key": cand_key, "reason": first_reason, "actions": runner._actions_signature(forward.actions)},
            )
            logger.info("Forward commit from sig=%s: %s (%s)", cur_sig[:8], cand_key, first_reason[:90])
            # 如果有上一轮 forward 评分细节，也一起落到 trace 里。
            if runner._last_forward_detail:
                runner._log_event("forward_score", sig=cur_sig, **runner._last_forward_detail)
            # 发出决策事件，标记当前正式选择了哪个动作序列。
            runner._emit_decision(cur_sig, "forward_selected", {"actions": runner._actions_signature(forward.actions)})
            # 真正执行这个动作序列。
            print(f"[loop {loop_index:03d}] 已保存本轮 snap、标注图、LLM1 输入输出: {loop_dir}")
            input("按回车执行当前 forward 动作并进入下一轮...")
            ok = runner._execute_action_sequence(forward.actions, snap)

            # 如果动作执行层面就失败了，不能继续信任当前状态，需要恢复。
            if not ok:
                # 记录前进动作执行失败。
                logger.warning("Forward action failed at sig=%s -> recovery.", cur_sig[:8])
                # 进入 recovery。
                runner._recover(cur_sig, snap, reason=RecoveryReason.FORWARD_ACTION_FAILED, task=task, target_sig=None)
                # 恢复后重新抓页面。
                snap = runner._capture_and_process() or snap
                # 更新当前 sig。
                cur_sig = snap["state_sig"]
                # 修正 DFS 栈。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                # 为当前位置重新安排分析。
                runner._schedule_state(cur_sig, snap, task)
                # 本轮结束。
                continue

            # Capture post-forward authoritative snapshot
            # 动作执行成功后，马上抓一份新的权威快照，确认我们实际落到了哪里。
            snap_next = runner._capture_and_process()
            # 如果连前进后的快照都抓不到，同样需要恢复。
            if not snap_next:
                # 记录抓取失败。
                print("动作执行后的页面获取snap失败")
                logger.warning("Capture failed after forward at sig=%s -> recovery.", cur_sig[:8])
                # 进入 recovery。
                runner._recover(cur_sig, snap, reason=RecoveryReason.CAPTURE_FAILED_AFTER_FORWARD, task=task, target_sig=None)
                # 恢复后重新抓当前页。
                snap = runner._capture_and_process() or snap
                # 更新 sig。
                cur_sig = snap["state_sig"]
                # 修正 DFS 栈。
                runner._reconcile_stack_on_external_move(cur_sig, record_observation=False)
                # 重新安排分析。
                runner._schedule_state(cur_sig, snap, task)
                # 本轮结束。
                continue

            # 提取前进后真正到达的新状态签名。
            new_sig = snap_next["state_sig"]

            # Record transition and get novelty-at-discovery
            # 把这次“当前页 -> 前进后页面”的迁移正式写入状态图，并拿到目标页是否为新状态。
            dst_was_new = runner._graph_record_transition(
                cur_sig,
                new_sig,
                runner._actions_signature(forward.actions, vid_map=snap.get("vid_map") or {}),
                src_snap=snap,
                dst_snap=snap_next,
            )
            # 标记当前页上的这个 forward 候选已经至少尝试过一次。
            runner._mark_attempted(cur_sig, forward)

            # If forward returned to an ancestor or did nothing, mark explored to avoid re-committing endlessly.
            # 如果这个 forward 实际上没动，或者回到了祖先页，就把它标成 explored，避免以后重复提交。
            if new_sig == cur_sig or new_sig in runner.dfs_stack:
                runner._mark_explored(cur_sig, forward)

            # 用这次前进迁移更新 DFS 路径与 parent 关系。
            runner._enter_state(from_sig=cur_sig, to_sig=new_sig, via_action=cand_key)

            # Saturation counter uses novelty-at-discovery (not visit_count after touch)
            # 如果发现了新状态，就重置“无新状态”计数。
            if dst_was_new:
                runner.no_new_state_count = 0
            # 否则继续累计“连续没有发现新状态”的次数。
            else:
                runner.no_new_state_count += 1

            # 把当前基准切到前进后的新状态。
            cur_sig, snap = new_sig, snap_next
            # 为新状态安排下一轮异步分析。
            runner._schedule_state(cur_sig, snap, task)

        # 退出主循环后，尽力关闭线程池并取消剩余 future，避免后台任务继续占资源。
    try:
        # 不等待未完成任务自然结束，而是尽快取消它们，适合 run 结束时的清理语义。
        runner._pool.shutdown(wait=False, cancel_futures=True)
    # 清理失败不影响 run 的最终返回，所以这里吞掉异常。
    except Exception:
        pass

    # try:
    #     if runner._pool is not None:
    #         runner._pool.shutdown(wait=False, cancel_futures=True)
    # except Exception:
    #     pass


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
        debug_run_test(runner, args.task)
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
