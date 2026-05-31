"""Lightweight callback interfaces and a JSONL trace writer.

This module defines a small callback protocol that the workflow can invoke at
authoritative choke points (snapshots, LLM I/O, actions, transitions,
questionnaire updates, and policy decisions). A no-op implementation is the
default; JsonlTraceCallbacks writes compact events and large artifacts (XML/
screenshot/UI trees) to disk for offline replay and analysis.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from PIL import Image, ImageDraw


@dataclass
class StepCtx:
    """
    Trace context passed to callback events.

    Input:
    - Workflow state at the moment an event is emitted.

    Output:
    - A JSON-friendly context object embedded in trace.jsonl records.

    Function:
    - Carries run, DFS, questionnaire, and task-stack summaries without forcing
      callbacks to know about WorkflowRunner internals.
    """

    run_id: str
    step_id: int
    ts: float
    cur_sig: str
    stack: List[str]
    block_status: Dict[str, Any]
    open_gaps: List[str]
    answered_ratio: float
    task_context: Optional[Dict[str, Any]] = None


class Callbacks(Protocol):
    def on_snapshot(self, ctx: StepCtx, snap: Dict[str, Any]) -> None: ...

    def on_llm_enqueued(self, ctx: StepCtx, kind: str, payload: Dict[str, Any]) -> None: ...

    def on_llm_result(self, ctx: StepCtx, kind: str, result: Dict[str, Any]) -> None: ...

    def on_action(self, ctx: StepCtx, action: Dict[str, Any], phase: str, extra: Dict[str, Any]) -> None: ...

    def on_transition(self, ctx: StepCtx, tr: Dict[str, Any]) -> None: ...

    def on_questionnaire_update(self, ctx: StepCtx, upd: Dict[str, Any]) -> None: ...

    def on_decision(self, ctx: StepCtx, name: str, detail: Dict[str, Any]) -> None: ...


class NoOpCallbacks:
    def on_snapshot(self, ctx: StepCtx, snap: Dict[str, Any]) -> None:
        return None

    def on_llm_enqueued(self, ctx: StepCtx, kind: str, payload: Dict[str, Any]) -> None:
        return None

    def on_llm_result(self, ctx: StepCtx, kind: str, result: Dict[str, Any]) -> None:
        return None

    def on_action(self, ctx: StepCtx, action: Dict[str, Any], phase: str, extra: Dict[str, Any]) -> None:
        return None

    def on_transition(self, ctx: StepCtx, tr: Dict[str, Any]) -> None:
        return None

    def on_questionnaire_update(self, ctx: StepCtx, upd: Dict[str, Any]) -> None:
        return None

    def on_decision(self, ctx: StepCtx, name: str, detail: Dict[str, Any]) -> None:
        return None


class InteractiveDebugCallbacks(NoOpCallbacks):
    """Print concise Chinese step summaries and wait for a key between steps."""

    manages_action_pause = True

    def __init__(self, *, pause_on: Optional[List[str]] = None) -> None:
        self.pause_on = set(
            pause_on
            or [
                "snapshot",
                "llm_result",
                "action_before",
                "action_after",
                "transition",
                "questionnaire_update",
                "decision",
            ]
        )
        self._seq = 0
        self._lock = threading.Lock()

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _short_sig(self, sig: Any) -> str:
        text = str(sig or "")
        return text[:8] if text else "-"

    def _short_text(self, value: Any, limit: int = 80) -> str:
        text = str(value or "").strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _fmt_action(self, action: Dict[str, Any]) -> str:
        kind = str(action.get("action") or "?")
        element_id = action.get("element_id")
        text = self._short_text(action.get("text"), 30)
        parts = [kind]
        if element_id is not None:
            parts.append(f"id={element_id}")
        if text:
            parts.append(f"text={text}")
        return " ".join(parts)

    def _fmt_json(self, payload: Dict[str, Any], limit: int = 160) -> str:
        return self._short_text(json.dumps(payload, ensure_ascii=False), limit)

    def _print_block(self, title: str, lines: List[str], *, pause_key: Optional[str] = None) -> None:
        idx = self._next_seq()
        print(f"\n[{idx:03d}] {title}")
        for line in lines:
            print(f"  {line}")
        if pause_key and pause_key in self.pause_on:
            self._wait_for_key()

    def _wait_for_key(self) -> None:
        prompt = "  按任意键继续..."
        if os.name == "nt":
            try:
                import msvcrt

                print(prompt, end="", flush=True)
                msvcrt.getwch()
                print("")
                return
            except Exception:
                pass
        input(f"{prompt}（或直接回车）")

    def on_snapshot(self, ctx: StepCtx, snap: Dict[str, Any]) -> None:  # type: ignore[override]
        meta = snap.get("meta") or {}
        vid_count = len(snap.get("vid_map") or {})
        root_count = len((snap.get("uist") or {}).get("elements") or [])
        self._print_block(
            "快照",
            [
                f"step_id={ctx.step_id} sig={self._short_sig(snap.get('state_sig'))} stack_depth={len(ctx.stack)}",
                f"前台包={meta.get('foreground_package') or '-'} activity={meta.get('foreground_activity') or '-'}",
                f"元素数={vid_count} 根节点数={root_count} cache_hit={bool(meta.get('cache_hit'))}",
            ],
            pause_key="snapshot",
        )

    def on_llm_result(self, ctx: StepCtx, kind: str, result: Dict[str, Any]) -> None:  # type: ignore[override]
        keys = sorted(result.keys())
        self._print_block(
            f"LLM结果: {kind}",
            [
                f"sig={self._short_sig(ctx.cur_sig)} 字段={', '.join(keys[:8]) or '-'}",
                f"摘要={self._short_text(json.dumps(result, ensure_ascii=False), 160)}",
            ],
            pause_key="llm_result",
        )

    def on_action(self, ctx: StepCtx, action: Dict[str, Any], phase: str, extra: Dict[str, Any]) -> None:  # type: ignore[override]
        origin = str(extra.get("origin") or "-")
        reasoning = self._short_text(extra.get("reasoning"), 100) or "-"
        if phase == "before":
            self._print_block(
                "动作准备",
                [
                    f"sig={self._short_sig(ctx.cur_sig)} 动作={self._fmt_action(action)}",
                    f"来源函数={origin}",
                    f"执行原因={reasoning}",
                    f"附加信息={self._fmt_json(extra, 140)}",
                ],
                pause_key="action_before",
            )
            return

        self._print_block(
            "动作结果",
            [
                f"sig={self._short_sig(ctx.cur_sig)} 动作={self._fmt_action(action)}",
                f"来源函数={origin}",
                f"执行原因={reasoning}",
                f"结果={self._fmt_json(extra, 140)}",
            ],
            pause_key="action_after",
        )

    def on_transition(self, ctx: StepCtx, tr: Dict[str, Any]) -> None:  # type: ignore[override]
        self._print_block(
            "状态迁移",
            [
                f"kind={tr.get('kind') or '-'} src={self._short_sig(tr.get('src'))} dst={self._short_sig(tr.get('dst') or tr.get('sig'))}",
                f"详情={self._short_text(json.dumps(tr, ensure_ascii=False), 160)}",
            ],
            pause_key="transition",
        )

    def on_questionnaire_update(self, ctx: StepCtx, upd: Dict[str, Any]) -> None:  # type: ignore[override]
        self._print_block(
            "问卷更新",
            [
                f"sig={self._short_sig(ctx.cur_sig)}",
                f"详情={self._short_text(json.dumps(upd, ensure_ascii=False), 160)}",
            ],
            pause_key="questionnaire_update",
        )

    def on_decision(self, ctx: StepCtx, name: str, detail: Dict[str, Any]) -> None:  # type: ignore[override]
        if name == "next_step":
            plan = str(detail.get("plan") or "-")
            self._print_block(
                "下一步计划",
                [
                    f"当前sig={self._short_sig(ctx.cur_sig)}",
                    f"计划={plan}",
                    f"详情={self._short_text(json.dumps(detail, ensure_ascii=False), 220)}",
                ],
                pause_key="decision",
            )
            return
        self._print_block(
            f"流程决策: {name}",
            [
                f"sig={self._short_sig(ctx.cur_sig)} blocks={len(ctx.block_status)}",
                f"详情={self._short_text(json.dumps(detail, ensure_ascii=False), 220)}",
            ],
            pause_key="decision",
        )


class JsonlTraceCallbacks(NoOpCallbacks):
    """Persist events + large blobs to disk for offline replay/metrics."""

    def __init__(self, root_dir: str, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.root_dir = os.path.abspath(os.path.join(root_dir, self.run_id))
        self.trace_path = os.path.join(self.root_dir, "trace.jsonl")
        self.states_dir = os.path.join(self.root_dir, "states")

        os.makedirs(self.states_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._page_dirs_by_sig: Dict[str, str] = {}
        self._page_prefix_by_sig: Dict[str, str] = {}
        self._snap_cache_by_sig: Dict[str, Dict[str, Any]] = {}
        self._llm_result_cache_by_sig: Dict[str, Dict[str, Any]] = {}
        self._timeline: List[Dict[str, Any]] = []
        self._pending_actions: Dict[str, Dict[str, Any]] = {}
        self._timeline_logger = logging.getLogger("run_timeline")

    # ------------ helpers ------------
    @staticmethod
    def _block_status_summary(block_status: Dict[str, Any]) -> Dict[str, int]:
        """
        Input: full block_status mapping from workflow context.
        Output: small count summary safe to repeat inside trace.jsonl.
        Function: avoids writing a huge questionnaire payload into every event.
        """
        rows = list((block_status or {}).values())
        visited = 0
        hit = 0
        for row in rows:
            try:
                if int((row or {}).get("visit_count", 0) or 0) > 0:
                    visited += 1
                if int((row or {}).get("hit_count", 0) or 0) > 0:
                    hit += 1
            except Exception:
                continue
        return {"count": len(rows), "visited": visited, "hit": hit}

    def _write_event(self, event: str, ctx: StepCtx, payload: Dict[str, Any]) -> None:
        """
        Input: event name, workflow step context, and event payload.
        Output: appends one compact JSON line to trace.jsonl.
        Function: preserves machine-readable tracing without repeating large questionnaire state.
        """
        rec = {
            "event": event,
            "ts": time.time(),
            "ctx": {
                "run_id": ctx.run_id,
                "step_id": ctx.step_id,
                "ts": ctx.ts,
                "cur_sig": ctx.cur_sig,
                "stack": ctx.stack,
                "block_status_summary": self._block_status_summary(ctx.block_status),
                "open_gaps": ctx.open_gaps,
                "answered_ratio": ctx.answered_ratio,
                "task": ctx.task_context or {},
            },
            "data": payload,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _add_timeline(self, entry: Dict[str, Any], message: Optional[str] = None) -> None:
        """
        Input: normalized timeline entry and optional concise console message.
        Output: stores the entry in memory and logs the message to run_timeline.
        Function: builds the final human-readable run timeline while keeping console concise.
        """
        row = {"ts": time.time(), **entry}
        with self._lock:
            self._timeline.append(row)
        if message:
            self._timeline_logger.info(message)

    @staticmethod
    def _safe_filename_token(value: Any, default: str = "state") -> str:
        """
        Convert state ids into Windows-safe filename tokens.

        Why:
        - New state_sig values may contain ":" (for example "xml:abcd...").
        - On Windows, ":" creates an alternate data stream, causing normal
          files to appear as 0-byte placeholders.
        """
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
        return token or default

    def _page_prefix(self, step_id: int, sig: str) -> str:
        """
        Input: workflow step id and state signature.
        Output: standard state directory prefix such as UI000005_phash_abcd.
        Function: keeps per-state artifact folders stable and human-readable.
        """
        return f"UI{int(step_id):06d}_{self._safe_filename_token(sig)[:48]}"

    def _page_dir_for_sig(self, sig: str, step_id: int, *, create: bool = True) -> str:
        """
        Input: state signature and current step id.
        Output: existing or newly created states directory for that state.
        Function: lets snapshot, LLM, and action artifacts land in one human-readable page folder.
        """
        if sig in self._page_dirs_by_sig:
            return self._page_dirs_by_sig[sig]
        prefix = self._page_prefix(step_id, sig)
        page_dir = os.path.join(self.states_dir, prefix)
        if create:
            os.makedirs(page_dir, exist_ok=True)
        self._page_dirs_by_sig[sig] = page_dir
        self._page_prefix_by_sig[sig] = prefix
        return page_dir

    @staticmethod
    def _ensure_subdir(page_dir: str, name: str) -> str:
        """
        Input: page directory and subdirectory name.
        Output: absolute path to the ensured subdirectory.
        Function: keeps overlays and LLM artifacts separated inside each UI folder.
        """
        out_dir = os.path.join(page_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _llm_filename_for_kind(kind: str, suffix: str) -> str:
        """
        Input: LLM callback kind and suffix, either input or result.
        Output: canonical filename inside states/<UI>/llm.
        Function: avoids legacy split navigation/router filenames.
        """
        normalized = str(kind or "").strip()
        if normalized == "navigation_router":
            return f"navigation_router_{suffix}.json"
        if normalized == "blocks_fill":
            return f"blocks_fill_{suffix}.json"
        return f"{normalized}_{suffix}.json"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """
        Input: arbitrary callback payload.
        Output: JSON-serializable value.
        Function: protects debug artifact writes from Pydantic or custom object instances.
        """
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(k): JsonlTraceCallbacks._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [JsonlTraceCallbacks._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [JsonlTraceCallbacks._json_safe(v) for v in value]
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            return str(value)

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        """
        Input: target path and JSON-like payload.
        Output: writes UTF-8 pretty JSON.
        Function: centralizes debug artifact JSON serialization.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(JsonlTraceCallbacks._json_safe(payload), fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _vid_map_summary(vid_map: Dict[Any, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (vid_map or {}).items():
            try:
                f = v.get("absolute_frame") or v.get("frame") or {}
                out[str(k)] = {
                    "bounds": [
                        int(f.get("x", 0)),
                        int(f.get("y", 0)),
                        int(f.get("width", 0)),
                        int(f.get("height", 0)),
                    ],
                    "text": v.get("text"),
                    "content_desc": v.get("content_desc"),
                    "ocr_text": v.get("ocr_text"),
                    "class": v.get("class"),
                    "clickable": v.get("clickable"),
                    "enabled": v.get("enabled"),
                }
            except Exception:
                continue
        return out

    @staticmethod
    def _iter_uist_nodes(uist: Dict[str, Any]):
        stack = list(uist.get("elements", []) or [])
        while stack:
            node = stack.pop()
            yield node
            subs = node.get("subviews", []) or []
            if subs:
                stack.extend(reversed(subs))

    @staticmethod
    def _decode_screenshot(screenshot_b64: str) -> Optional[Image.Image]:
        if not screenshot_b64:
            return None
        try:
            raw = base64.b64decode(screenshot_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    @staticmethod
    def _label_for_node(node: Dict[str, Any]) -> str:
        for key in ("text", "content_desc", "ocr_text", "semantic_label", "icon_label"):
            value = node.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _draw_boxes(
        image: Image.Image,
        items: List[Dict[str, Any]],
        *,
        id_getter,
        color_getter,
    ) -> Image.Image:
        out = image.copy()
        draw = ImageDraw.Draw(out)

        for item in items:
            frame = item.get("absolute_frame") or item.get("frame") or {}
            x = int(frame.get("x", 0))
            y = int(frame.get("y", 0))
            w = int(frame.get("width", 0))
            h = int(frame.get("height", 0))
            if w <= 0 or h <= 0:
                continue

            color = color_getter(item)
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3)

            label_text = str(id_getter(item))
            if not label_text:
                continue
            label_box = draw.textbbox((x, y), label_text)
            text_w = label_box[2] - label_box[0]
            text_h = label_box[3] - label_box[1]
            text_left = max(0, x)
            text_top = max(0, y - text_h - 8)
            if text_top == 0 and y + h + text_h + 8 < out.height:
                text_top = y + h + 2
            draw.rectangle(
                (text_left, text_top, text_left + text_w + 10, text_top + text_h + 8),
                fill=color,
            )
            draw.text((text_left + 5, text_top + 4), label_text, fill=(255, 255, 255))

        return out

    @staticmethod
    def _frame_for_element(vid_map: Dict[Any, Any], element_id: Any) -> Optional[Dict[str, int]]:
        """
        Input: vid_map and an LLM element_id.
        Output: drawable frame dict, or None when the element cannot be resolved.
        Function: maps LLM action targets back to screenshot coordinates.
        """
        if element_id is None:
            return None
        node = (vid_map or {}).get(element_id)
        if node is None:
            node = (vid_map or {}).get(str(element_id))
        if not isinstance(node, dict):
            return None
        frame = node.get("absolute_frame") or node.get("frame") or {}
        try:
            x = int(frame.get("x", 0))
            y = int(frame.get("y", 0))
            w = int(frame.get("width", 0))
            h = int(frame.get("height", 0))
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return {"x": x, "y": y, "width": w, "height": h}

    @staticmethod
    def _iter_action_steps_for_overlay(navigation: Dict[str, Any]) -> List[tuple[str, Dict[str, Any], tuple[int, int, int]]]:
        """
        Input: navigation result dict.
        Output: labeled action steps and colors.
        Function: mirrors test_debug overlays for dismiss, candidate, and page-return actions.
        """
        items: List[tuple[str, Dict[str, Any], tuple[int, int, int]]] = []
        for idx, step in enumerate(navigation.get("overlay_dismiss_actions") or [], start=1):
            if isinstance(step, dict):
                items.append((f"D{idx}", step, (220, 50, 47)))
        for cand_idx, cand in enumerate(navigation.get("candidate_actions") or [], start=1):
            if not isinstance(cand, dict):
                continue
            for step_idx, step in enumerate(cand.get("actions") or [], start=1):
                if isinstance(step, dict):
                    items.append((f"C{cand_idx}.{step_idx}", step, (38, 139, 210)))
        for idx, step in enumerate(navigation.get("page_return_actions") or [], start=1):
            if isinstance(step, dict):
                items.append((f"R{idx}", step, (181, 137, 0)))
        return items

    @staticmethod
    def _draw_labeled_box(draw: ImageDraw.ImageDraw, frame: Dict[str, int], label: str, color: tuple[int, int, int]) -> None:
        """
        Input: drawing context, element frame, text label, and RGB color.
        Output: draws one labeled rectangle.
        Function: renders LLM candidate action targets on screenshots.
        """
        x = int(frame["x"])
        y = int(frame["y"])
        w = int(frame["width"])
        h = int(frame["height"])
        draw.rectangle((x, y, x + w, y + h), outline=color, width=4)
        box = draw.textbbox((x, y), label)
        text_w = box[2] - box[0]
        text_h = box[3] - box[1]
        label_x = max(0, x)
        label_y = max(0, y - text_h - 8)
        if label_y == 0:
            label_y = y + h + 2
        draw.rectangle((label_x, label_y, label_x + text_w + 10, label_y + text_h + 8), fill=color)
        draw.text((label_x + 5, label_y + 4), label, fill=(255, 255, 255))

    def _write_llm_actions_overlay(self, sig: str, result: Dict[str, Any], out_path: str) -> Optional[str]:
        """
        Input: state signature, navigation result, and output image path.
        Output: output path when the overlay was written.
        Function: draws LLM action candidates into the page debug directory.
        """
        snap = self._snap_cache_by_sig.get(sig) or {}
        image = self._decode_screenshot(str(snap.get("screenshot") or ""))
        if image is None:
            return None
        draw = ImageDraw.Draw(image)
        vid_map = snap.get("vid_map") or {}
        drawn = 0
        for label, step, color in self._iter_action_steps_for_overlay(result):
            frame = self._frame_for_element(vid_map, step.get("element_id"))
            if frame:
                self._draw_labeled_box(draw, frame, label, color)
                drawn += 1
        if drawn <= 0:
            return None
        image.save(out_path)
        return out_path

    def _write_snapshot_overlays(self, page_dir: str, snap: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Input: page debug directory and snapshot.
        Output: paths for vid_map and uist overlay images.
        Function: writes human-readable element overlays beside each snapshot.
        """
        screenshot_b64 = str(snap.get("screenshot") or "")
        base_img = self._decode_screenshot(screenshot_b64)
        if base_img is None:
            return {"vidmap_overlay_path": None, "uist_overlay_path": None}

        overlays_dir = self._ensure_subdir(page_dir, "overlays")
        vidmap_path = None
        uist_path = None

        try:
            vid_items = list((snap.get("vid_map") or {}).values())
            vid_img = self._draw_boxes(
                base_img,
                vid_items,
                id_getter=lambda n: n.get("id", ""),
                color_getter=lambda n: (220, 50, 47) if bool(n.get("clickable")) else (46, 160, 67),
            )
            vidmap_path = os.path.join(overlays_dir, "vid_map_overlay.png")
            vid_img.save(vidmap_path)
        except Exception:
            vidmap_path = None

        try:
            uist_items = list(self._iter_uist_nodes(snap.get("uist") or {}))
            # uist 视图没有稳定 id 时，优先显示节点 id；没有 id 时显示流水号会不稳定，
            # 所以这里退化成只显示已有 id，没有就不显示文字标签。
            uist_img = self._draw_boxes(
                base_img,
                uist_items,
                id_getter=lambda n: n.get("id", ""),
                color_getter=lambda n: (203, 75, 22) if bool(n.get("clickable")) else (133, 153, 0),
            )
            uist_path = os.path.join(overlays_dir, "uist_overlay.png")
            uist_img.save(uist_path)
        except Exception:
            uist_path = None

        return {"vidmap_overlay_path": vidmap_path, "uist_overlay_path": uist_path}

    def _write_debug_summary(self, sig: str, page_dir: str) -> Optional[str]:
        """
        Input: state signature and UI artifact directory.
        Output: path to llm/debug_summary.md when written.
        Function: provides a compact human-readable index of the LLM outputs for one UI.
        """
        llm_dir = self._ensure_subdir(page_dir, "llm")
        out_path = os.path.join(llm_dir, "debug_summary.md")
        bundle = self._llm_result_cache_by_sig.get(sig) or {}
        nav_router = bundle.get("navigation_router") or {}
        blocks_fill = bundle.get("blocks_fill") or {}
        nav_body = {}
        router_body = {}
        task_decision_body = {}
        proposed_tasks_body = []
        if isinstance(nav_router, dict):
            result = nav_router.get("result") if isinstance(nav_router.get("result"), dict) else nav_router
            nav_body = result.get("navigation") if isinstance(result.get("navigation"), dict) else {}
            router_body = result.get("router") if isinstance(result.get("router"), dict) else {}
            task_decision_body = result.get("task_decision") if isinstance(result.get("task_decision"), dict) else {}
            proposed_tasks_body = result.get("proposed_tasks") if isinstance(result.get("proposed_tasks"), list) else []
        block_body = blocks_fill.get("result") if isinstance(blocks_fill.get("result"), dict) else {}

        lines = [
            "# UI LLM Debug Summary",
            "",
            f"- state_sig: `{sig}`",
            f"- navigation_router: {'yes' if nav_router else 'no'}",
            f"- blocks_fill: {'yes' if blocks_fill else 'no'}",
            "",
            "## Navigation Router",
            "",
        ]
        if nav_body:
            lines.extend(
                [
                    f"- page_summary: {str(nav_body.get('page_summary') or '')}",
                    f"- overlay_kind: {str(nav_body.get('overlay_kind') or '')}",
                    f"- candidate_actions: {len(nav_body.get('candidate_actions') or [])}",
                    f"- page_return_actions: {len(nav_body.get('page_return_actions') or [])}",
                ]
            )
            candidates = list(nav_body.get("candidate_actions") or [])
            if candidates:
                lines.extend(["", "### Candidate Actions", ""])
                for idx, cand in enumerate(candidates, start=1):
                    if not isinstance(cand, dict):
                        continue
                    step = {}
                    steps = cand.get("actions") if isinstance(cand.get("actions"), list) else []
                    if steps and isinstance(steps[0], dict):
                        step = steps[0]
                    role = str(cand.get("action_role") or "")
                    starts = str(cand.get("starts_task_type") or "")
                    starts_depth = str(cand.get("starts_task_depth") or "")
                    label = str(step.get("anchor_label") or step.get("text") or step.get("reasoning") or "")
                    lines.append(
                        f"- C{idx}: role={role}, starts_task_type={starts}, starts_task_depth={starts_depth}, score={cand.get('score')}, "
                        f"action={step.get('action')}:{step.get('element_id')} {label}"
                    )
        if router_body:
            lines.extend(
                [
                    f"- router_updates: {len(router_body.get('router_updates') or [])}",
                    f"- matched_block_ids: {', '.join([str(x) for x in (nav_router.get('matched_block_ids') or [])])}",
                ]
            )
        lines.extend(["", "## Task", ""])
        if task_decision_body:
            lines.extend(
                [
                    f"- task_id: {str((nav_router.get('result') or {}).get('task_id') or '')}",
                    f"- current_task_done: {bool(task_decision_body.get('current_task_done'))}",
                    f"- current_task_failed: {bool(task_decision_body.get('current_task_failed'))}",
                    f"- should_return: {bool(task_decision_body.get('should_return'))}",
                    f"- reason: {str(task_decision_body.get('reason') or '')}",
                    f"- proposed_tasks: {len(proposed_tasks_body)}",
                ]
            )
            if proposed_tasks_body:
                lines.extend(["", "### Proposed Tasks", ""])
                for idx, task in enumerate(proposed_tasks_body, start=1):
                    if not isinstance(task, dict):
                        continue
                    entry = task.get("entry_action") if isinstance(task.get("entry_action"), dict) else {}
                    label = str(entry.get("anchor_label") or entry.get("text") or entry.get("reasoning") or "")
                    lines.append(
                        f"- T{idx}: priority={task.get('priority')}, depth={task.get('exploration_depth')}, type={task.get('task_type')}, "
                        f"entry={entry.get('action')}:{entry.get('element_id')} {label}, "
                        f"prompt={str(task.get('prompt') or '')}"
                    )
        else:
            lines.append("- task_decision: no")
        lines.extend(["", "## Blocks Fill", ""])
        if block_body:
            lines.append(f"- block_results: {len(block_body.get('block_results') or [])}")
        else:
            lines.append("- block_results: 0")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            return out_path
        except Exception:
            return None

    # ------------ event handlers ------------
    def on_snapshot(self, ctx: StepCtx, snap: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: authoritative workflow snapshot.
        Output: writes one states/<UI...>/ folder and a compact trace event.
        Function: keeps all human-facing snapshot artifacts together for manual debugging.
        """
        sig = str(snap.get("state_sig") or "")
        prefix = self._page_prefix(ctx.step_id, sig)
        page_dir = self._page_dir_for_sig(sig, ctx.step_id, create=True)
        self._snap_cache_by_sig[sig] = {
            "screenshot": snap.get("screenshot"),
            "vid_map": snap.get("vid_map") or {},
        }

        screenshot_path = None
        screenshot_hash = None
        if snap.get("screenshot"):
            try:
                raw = base64.b64decode(snap.get("screenshot") or b"")
                screenshot_hash = hashlib.md5(raw).hexdigest()
                screenshot_path = os.path.join(page_dir, "screenshot.png")
                with open(screenshot_path, "wb") as fh:
                    fh.write(raw)
            except Exception:
                screenshot_path = None

        screenshot_raw_path = None
        screenshot_raw_hash = None
        if snap.get("screenshot_raw"):
            try:
                raw2 = base64.b64decode(snap.get("screenshot_raw") or b"")
                screenshot_raw_hash = hashlib.md5(raw2).hexdigest()
                screenshot_raw_path = os.path.join(page_dir, "screenshot_raw.png")
                with open(screenshot_raw_path, "wb") as fh:
                    fh.write(raw2)
            except Exception:
                screenshot_raw_path = None

        xml_path = None
        if snap.get("xml"):
            try:
                xml_path = os.path.join(page_dir, "xml.xml")
                with open(xml_path, "w", encoding="utf-8") as fh:
                    fh.write(str(snap.get("xml") or ""))
            except Exception:
                xml_path = None

        xml_raw_path = None
        if snap.get("xml_raw"):
            try:
                xml_raw_path = os.path.join(page_dir, "xml_raw.xml")
                with open(xml_raw_path, "w", encoding="utf-8") as fh:
                    fh.write(str(snap.get("xml_raw") or ""))
            except Exception:
                xml_raw_path = None

        uist_path = None
        if snap.get("uist"):
            try:
                uist_path = os.path.join(page_dir, "uist.json")
                self._write_json(uist_path, snap.get("uist") or {})
            except Exception:
                uist_path = None

        vid_map_path = None
        try:
            vid_map_path = os.path.join(page_dir, "vid_map.json")
            self._write_json(vid_map_path, snap.get("vid_map") or {})
        except Exception:
            vid_map_path = None

        overlay_paths = self._write_snapshot_overlays(page_dir, snap)

        meta = snap.get("meta", {}) or {}
        snap_path = None
        try:
            snap_path = os.path.join(page_dir, "snap.json")
            self._write_json(snap_path, snap)
        except Exception:
            snap_path = None

        payload = {
            "state_sig": sig,
            "page_dir": page_dir,
            "page": prefix,
            "page_prefix": prefix,
            "known": bool(meta.get("cache_hit") or meta.get("matched_existing")),
            "xml_reliable": meta.get("xml_reliable"),
            "xml_path": xml_path,
            "xml_raw_path": xml_raw_path,
            "screenshot_path": screenshot_path,
            "screenshot_hash": screenshot_hash,
            "screenshot_raw_path": screenshot_raw_path,
            "screenshot_raw_hash": screenshot_raw_hash,
            "uist_path": uist_path,
            "vid_map_path": vid_map_path,
            "snap_path": snap_path,
            **overlay_paths,
            "vid_map_summary": self._vid_map_summary(snap.get("vid_map", {})),
            "device_info": snap.get("device_info"),
            "meta": meta,
        }

        self._write_event("snapshot", ctx, payload)
        known = bool(meta.get("cache_hit") or meta.get("matched_existing"))
        self._add_timeline(
            {
                "type": "snapshot",
                "step_id": ctx.step_id,
                "sig": sig,
                "page": prefix,
                "known": known,
                "xml_reliable": meta.get("xml_reliable"),
                "vid_count": len(snap.get("vid_map") or {}),
                "page_dir": page_dir,
            },
            f"[SNAP] {prefix} known={str(known).lower()} xml_reliable={meta.get('xml_reliable')} vid={len(snap.get('vid_map') or {})}",
        )

    def on_llm_enqueued(self, ctx: StepCtx, kind: str, payload: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: LLM kind and business input payload.
        Output: writes llm_<kind>_input.json into the page debug directory.
        Function: preserves the model-call input visible to workflow callbacks.
        """
        sig = str(payload.get("state_sig") or ctx.cur_sig or "")
        page_dir = self._page_dir_for_sig(sig, ctx.step_id, create=True)
        llm_dir = self._ensure_subdir(page_dir, "llm")
        input_path = os.path.join(llm_dir, self._llm_filename_for_kind(kind, "input"))
        try:
            self._write_json(input_path, payload)
        except Exception:
            input_path = ""
        self._write_event(
            "llm_enqueued",
            ctx,
            {
                "kind": kind,
                "state_sig": sig,
                "input_path": input_path,
                "payload_keys": sorted([str(k) for k in payload.keys()]),
                "block_status_summary": self._block_status_summary(payload.get("block_status") or {}),
                "router_question_count": payload.get("router_question_count"),
                "block_count": payload.get("block_count"),
            },
        )
        self._add_timeline(
            {"type": "llm_enqueued", "step_id": ctx.step_id, "sig": sig, "kind": kind, "input_path": input_path},
            f"[LLM] enqueue {kind} sig={self._safe_filename_token(sig)[:16]} input={os.path.basename(input_path) if input_path else '-'}",
        )

    def on_llm_result(self, ctx: StepCtx, kind: str, result: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: LLM kind and structured result payload.
        Output: writes llm_<kind>_result.json and optional navigation action overlay.
        Function: keeps model outputs and visual action suggestions beside the relevant page.
        """
        sig = str(result.get("state_sig") or ctx.cur_sig or "")
        page_dir = self._page_dir_for_sig(sig, ctx.step_id, create=True)
        llm_dir = self._ensure_subdir(page_dir, "llm")
        result_path = os.path.join(llm_dir, self._llm_filename_for_kind(kind, "result"))
        overlay_path = ""
        try:
            self._write_json(result_path, result)
        except Exception:
            result_path = ""
        try:
            result_body_for_overlay = result.get("result") if isinstance(result.get("result"), dict) else {}
            nav_result = result_body_for_overlay.get("navigation") if isinstance(result_body_for_overlay.get("navigation"), dict) else result_body_for_overlay
            if kind in {"navigation_router", "nav"} and nav_result:
                overlay_path = self._write_llm_actions_overlay(
                    sig,
                    nav_result,
                    os.path.join(self._ensure_subdir(page_dir, "overlays"), "llm_actions_overlay.png"),
                ) or ""
        except Exception:
            overlay_path = ""
        result_body = result.get("result") if isinstance(result.get("result"), dict) else {}
        if kind == "navigation_router" and isinstance(result_body.get("navigation"), dict):
            result_body = result_body.get("navigation") or {}
        if result_path:
            self._llm_result_cache_by_sig.setdefault(sig, {})[kind] = result
            self._write_debug_summary(sig, page_dir)
        cand_count = len((result_body or {}).get("candidate_actions") or [])
        return_count = len((result_body or {}).get("page_return_actions") or [])
        overlay_kind = (result_body or {}).get("overlay_kind") or "-"
        self._write_event(
            "llm_result",
            ctx,
            {
                "kind": kind,
                "state_sig": sig,
                "duration_s": result.get("duration_s"),
                "result_path": result_path,
                "actions_overlay_path": overlay_path,
                "error": result.get("error"),
                "candidate_count": cand_count,
                "return_count": return_count,
                "overlay_kind": overlay_kind,
                "matched_block_ids": result.get("matched_block_ids"),
            },
        )
        self._add_timeline(
            {
                "type": "llm_result",
                "step_id": ctx.step_id,
                "sig": sig,
                "kind": kind,
                "duration_s": result.get("duration_s"),
                "candidate_count": cand_count,
                "return_count": return_count,
                "overlay_kind": overlay_kind,
                "result_path": result_path,
                "actions_overlay_path": overlay_path,
            },
            f"[LLM] result {kind} sig={self._safe_filename_token(sig)[:16]} overlay={overlay_kind} candidates={cand_count} return={return_count} elapsed={result.get('duration_s')}",
        )

    def on_action(self, ctx: StepCtx, action: Dict[str, Any], phase: str, extra: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: action event before/after execution.
        Output: compact trace event and timeline row.
        Function: records what was executed without creating per-action artifact files.
        """
        self._write_event("action", ctx, {"phase": phase, "action": action, "extra": extra})
        action_key = str(extra.get("action_key") or f"{action.get('action')}:{action.get('element_id')}")
        if phase == "before":
            self._pending_actions[action_key] = {"action": action, "extra": extra, "step_id": ctx.step_id, "sig": ctx.cur_sig}
            label = action.get("text") or extra.get("label") or extra.get("reasoning") or ""
            self._add_timeline(
                {
                    "type": "action_before",
                    "step_id": ctx.step_id,
                    "sig": ctx.cur_sig,
                    "action_key": action_key,
                    "action": action,
                    "reasoning": extra.get("reasoning"),
                },
                f"[ACT] start sig={self._safe_filename_token(ctx.cur_sig)[:16]} {action.get('action')} element={action.get('element_id')} label={str(label)[:60]}",
            )
            return
        success = bool(extra.get("success"))
        reason = extra.get("reason") or ""
        self._add_timeline(
            {
                "type": "action_after",
                "step_id": ctx.step_id,
                "sig": ctx.cur_sig,
                "action_key": action_key,
                "action": action,
                "success": success,
                "reason": reason,
                "extra": extra,
            },
            f"[ACT] done sig={self._safe_filename_token(ctx.cur_sig)[:16]} {action.get('action')} element={action.get('element_id')} success={str(success).lower()} reason={reason or '-'}",
        )

    def on_transition(self, ctx: StepCtx, tr: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: graph or drift transition payload.
        Output: compact trace event and timeline row.
        Function: makes page changes visible without opening trace.jsonl manually.
        """
        self._write_event("transition", ctx, tr)
        kind = str(tr.get("kind") or tr.get("action", {}).get("outcome", {}).get("change_type") or "-")
        src = str(tr.get("src") or ctx.cur_sig or "")
        dst = str(tr.get("dst") or tr.get("sig") or "")
        dst_was_new = tr.get("dst_was_new")
        self._add_timeline(
            {
                "type": "transition",
                "step_id": ctx.step_id,
                "sig": ctx.cur_sig,
                "kind": kind,
                "src": src,
                "dst": dst,
                "dst_was_new": dst_was_new,
                "detail": tr,
            },
            f"[STATE] {self._safe_filename_token(src)[:16]} -> {self._safe_filename_token(dst)[:16]} kind={kind} new={dst_was_new}",
        )

    def on_questionnaire_update(self, ctx: StepCtx, upd: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: questionnaire update payload.
        Output: compact trace event.
        Function: records questionnaire progress without console noise.
        """
        self._write_event("questionnaire_update", ctx, upd)

    def on_decision(self, ctx: StepCtx, name: str, detail: Dict[str, Any]) -> None:  # type: ignore[override]
        """
        Input: workflow decision name and detail payload.
        Output: compact trace event and selected timeline messages.
        Function: surfaces only decisions that are important for debugging the main route.
        """
        self._write_event("decision", ctx, {"name": name, **detail})
        if name == "drift_detected":
            src = str(detail.get("from_sig") or ctx.cur_sig or "")
            dst = str(detail.get("to_sig") or "")
            self._add_timeline(
                {"type": "drift", "step_id": ctx.step_id, "from_sig": src, "to_sig": dst, "detail": detail},
                f"[DRIFT] {self._safe_filename_token(src)[:16]} -> {self._safe_filename_token(dst)[:16]}",
            )
            return
        if name == "next_step":
            plan = str(detail.get("plan") or "-")
            if plan in {"dfs_evaluate_candidates", "dfs_candidate_commit", "dfs_return_from_exhausted_state", "page_return_actions", "fallback_back_after_page_return"}:
                self._add_timeline(
                    {"type": "decision", "step_id": ctx.step_id, "sig": ctx.cur_sig, "name": name, "detail": detail},
                    f"[DFS] sig={self._safe_filename_token(ctx.cur_sig)[:16]} plan={plan} detail={json.dumps(self._json_safe(detail), ensure_ascii=False)[:180]}",
                )

    def export_timeline(self, *, stop_reason: str = "run_exit") -> Dict[str, str]:
        """
        Input: final stop reason.
        Output: path to timeline.md.
        Function: writes the complete human-readable run timeline after workflow completion.
        """
        timeline_md = os.path.join(self.root_dir, "timeline.md")
        with self._lock:
            rows = list(self._timeline)
        lines = [f"# Run Timeline", "", f"- run_id: `{self.run_id}`", f"- stop_reason: `{stop_reason}`", ""]
        for idx, row in enumerate(rows, start=1):
            typ = row.get("type") or "event"
            step_id = row.get("step_id", "-")
            if typ == "snapshot":
                lines.append(f"{idx}. SNAP step={step_id} page={row.get('page')} known={row.get('known')} xml_reliable={row.get('xml_reliable')} vid={row.get('vid_count')}")
            elif typ == "llm_result":
                lines.append(f"{idx}. LLM step={step_id} kind={row.get('kind')} candidates={row.get('candidate_count')} return={row.get('return_count')} overlay={row.get('overlay_kind')}")
            elif typ == "action_after":
                act = row.get("action") or {}
                lines.append(f"{idx}. ACT step={step_id} {act.get('action')} element={act.get('element_id')} success={row.get('success')} reason={row.get('reason') or '-'}")
            elif typ == "transition":
                lines.append(f"{idx}. STATE step={step_id} {self._safe_filename_token(row.get('src'))[:16]} -> {self._safe_filename_token(row.get('dst'))[:16]} kind={row.get('kind')} new={row.get('dst_was_new')}")
            elif typ == "drift":
                lines.append(f"{idx}. DRIFT step={step_id} {self._safe_filename_token(row.get('from_sig'))[:16]} -> {self._safe_filename_token(row.get('to_sig'))[:16]}")
            elif typ == "decision":
                detail = row.get("detail") or {}
                lines.append(f"{idx}. DFS step={step_id} plan={detail.get('plan') or '-'} sig={self._safe_filename_token(row.get('sig'))[:16]}")
            else:
                lines.append(f"{idx}. {str(typ).upper()} step={step_id}")
        with open(timeline_md, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self._timeline_logger.info("[TRACE] timeline saved md=%s", timeline_md)
        return {"timeline_md": timeline_md}


__all__ = [
    "Callbacks",
    "StepCtx",
    "NoOpCallbacks",
    "JsonlTraceCallbacks",
]
