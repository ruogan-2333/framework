"""Build an offline task execution report for one workflow trace run.

Inputs:
- ``--trace-dir``: a workflow trace directory containing ``trace.jsonl`` and
  optional ``tasks.json``, ``task_report.json``, and ``states/*`` artifacts.

Outputs:
- ``task_execution_report.json``: normalized machine-readable report data.
- ``task_execution_report.md``: compact human-readable task timeline.
- ``task_execution_report.html``: static HTML report with task chains, UI
  thumbnails, action summaries, and drift/state-family checks.

Function:
- This script does not modify workflow behavior. It reconstructs the actual
  run path from existing trace artifacts so task execution can be inspected
  separately from the UTG state graph.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STATE_PREFIX_RE = re.compile(r"^(UI\d{6})_(.+)$")


@dataclass
class UiRef:
    """A human-readable reference to one UI state.

    Inputs:
    - state_sig: internal state signature such as ``phash:abc``.
    - ui_prefix: trace folder name such as ``UI000001_phash_abc``.
    - folder: absolute state artifact directory.

    Output:
    - Used by Markdown/HTML renderers to link state signatures back to files.
    """

    state_sig: str
    ui_prefix: str
    folder: Path


@dataclass
class TaskStep:
    """One observed UI/action step associated with a task.

    Inputs:
    - state_sig/ui_prefix: UI where this step occurred.
    - page_summary/progress: page and task text from LLM artifacts when present.
    - selected_action/action_intent: action selected for this task step.
    - next_state_sig: state reached after the action when trace can infer it.

    Output:
    - A normalized report row inside a task chain.
    """

    step_id: int
    state_sig: str
    ui_prefix: str = ""
    page_summary: str = ""
    page_kind: str = ""
    progress: str = ""
    selected_action: str = ""
    action_intent: str = ""
    action_success: Optional[bool] = None
    action_reason: str = ""
    next_state_sig: str = ""
    next_ui_prefix: str = ""
    source: str = "trace"
    drift_checks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskNode:
    """One task and its reconstructed execution chain.

    Inputs:
    - task fields from ``tasks.json`` or ``task_report.json``.
    - steps reconstructed from task history and trace events.

    Output:
    - A serializable task object for JSON/Markdown/HTML reporting.
    """

    task_id: str
    task_type: str = ""
    status: str = ""
    initial_goal: str = ""
    current_goal: str = ""
    progress_summary: str = ""
    parent_task_id: str = ""
    origin_state_sig: str = ""
    resume_state_sig: str = ""
    created_from_sig: str = ""
    created_from_ui: str = ""
    created_by_action: str = ""
    created_source: str = "inferred"
    finish_reason: str = ""
    finish_state_sig: str = ""
    finish_ui: str = ""
    finish_source: str = "final_tasks"
    priority: Any = ""
    step_budget: Any = ""
    used_steps: Any = ""
    inferred: bool = True
    steps: List[TaskStep] = field(default_factory=list)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Inputs:
    - argv: optional argument list; process argv is used when omitted.

    Output:
    - argparse Namespace containing trace-dir and output options.
    """

    parser = argparse.ArgumentParser(description="Build an offline task execution report for one trace run.")
    parser.add_argument("--trace-dir", required=True, help="Trace run directory containing trace.jsonl.")
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional output Markdown path.")
    parser.add_argument("--output-html", default="", help="Optional output HTML path.")
    return parser.parse_args(argv)


def read_json(path: Path, default: Any) -> Any:
    """Read JSON with a fallback value.

    Inputs:
    - path: JSON file path.
    - default: value returned when the file is missing or malformed.

    Output:
    - Parsed JSON object or default.
    """

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read trace JSONL records.

    Inputs:
    - path: ``trace.jsonl`` path.

    Output:
    - List of parsed JSON records; malformed lines are skipped.
    """

    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def safe_text(value: Any, limit: int = 600) -> str:
    """Convert arbitrary values into short report text.

    Inputs:
    - value: any object from trace artifacts.
    - limit: maximum returned character length.

    Output:
    - Single-line trimmed text.
    """

    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit > 0 and len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def action_to_text(action: Any, extra: Optional[Dict[str, Any]] = None) -> str:
    """Format one action as compact text.

    Inputs:
    - action: action dict from trace/task artifacts.
    - extra: optional trace extra dict containing action_key or reasoning.

    Output:
    - Human-readable action label such as ``click:31 ICON_SHOPPING_CART``.
    """

    extra = extra or {}
    if not isinstance(action, dict):
        return safe_text(action, 300)
    kind = action.get("action") or action.get("type") or "action"
    element = action.get("element_id")
    label = (
        action.get("text")
        or action.get("anchor_label")
        or extra.get("label")
        or extra.get("reasoning")
        or action.get("reasoning")
        or ""
    )
    if element is None or element == "":
        return safe_text(f"{kind} {label}".strip(), 300)
    return safe_text(f"{kind}:{element} {label}".strip(), 300)


def state_sig_from_prefix(prefix_tail: str) -> str:
    """Infer state signature from a trace UI folder suffix.

    Inputs:
    - prefix_tail: folder suffix after ``UI000001_``.

    Output:
    - Best-effort state signature.
    """

    if prefix_tail.startswith("phash_"):
        return "phash:" + prefix_tail[len("phash_") :]
    if prefix_tail.startswith("xml_"):
        return "xml:" + prefix_tail[len("xml_") :]
    return prefix_tail.replace("_", ":", 1)


def build_ui_refs(trace_dir: Path, events: Sequence[Dict[str, Any]]) -> Dict[str, UiRef]:
    """Build a state-signature to UI-folder mapping.

    Inputs:
    - trace_dir: trace run directory.
    - events: parsed trace records.

    Output:
    - Mapping from ``state_sig`` to ``UiRef``.
    """

    refs: Dict[str, UiRef] = {}
    states_dir = trace_dir / "states"
    if states_dir.exists():
        for folder in sorted(p for p in states_dir.iterdir() if p.is_dir()):
            match = STATE_PREFIX_RE.match(folder.name)
            if not match:
                continue
            sig = state_sig_from_prefix(match.group(2))
            refs.setdefault(sig, UiRef(state_sig=sig, ui_prefix=folder.name, folder=folder))

    for event in events:
        ctx = event.get("ctx") or {}
        data = event.get("data") or {}
        candidates = [
            ctx.get("cur_sig"),
            data.get("state_sig"),
            data.get("sig"),
            data.get("src"),
            data.get("dst"),
            data.get("from_sig"),
            data.get("to_sig"),
        ]
        for sig in candidates:
            if not sig or sig in refs:
                continue
            token = str(sig).replace(":", "_")
            refs[str(sig)] = UiRef(state_sig=str(sig), ui_prefix=token, folder=states_dir / token)
    return refs


def ui_prefix(refs: Dict[str, UiRef], state_sig: str) -> str:
    """Return a readable UI prefix for one state signature.

    Inputs:
    - refs: state-to-UI mapping.
    - state_sig: state signature.

    Output:
    - UI folder prefix or shortened signature.
    """

    if not state_sig:
        return ""
    ref = refs.get(state_sig)
    if ref:
        return ref.ui_prefix
    return state_sig.replace(":", "_")[:32]


def load_task_nodes(trace_dir: Path, refs: Dict[str, UiRef]) -> Dict[str, TaskNode]:
    """Load task nodes from task artifacts.

    Inputs:
    - trace_dir: trace run directory.
    - refs: state-to-UI mapping.

    Output:
    - Mapping from task id to TaskNode.
    """

    tasks_data = read_json(trace_dir / "tasks.json", {})
    report_data = read_json(trace_dir / "task_report.json", {})
    raw_tasks = []
    if isinstance(tasks_data, dict) and isinstance(tasks_data.get("tasks"), list):
        raw_tasks = tasks_data.get("tasks", [])
    elif isinstance(report_data, dict) and isinstance(report_data.get("tasks"), list):
        raw_tasks = report_data.get("tasks", [])

    nodes: Dict[str, TaskNode] = {}
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            continue
        task_id = safe_text(raw.get("task_id"), 80)
        if not task_id:
            continue
        entry_action = raw.get("entry_action") if isinstance(raw.get("entry_action"), dict) else {}
        origin_sig = safe_text(raw.get("origin_state_sig"), 120)
        parent = safe_text(raw.get("parent_task_id"), 80)
        node = TaskNode(
            task_id=task_id,
            task_type=safe_text(raw.get("task_type"), 120),
            status=safe_text(raw.get("status"), 80),
            initial_goal=safe_text(raw.get("initial_goal"), 1200),
            current_goal=safe_text(raw.get("current_goal"), 1200),
            progress_summary=safe_text(raw.get("progress_summary"), 1200),
            parent_task_id=parent,
            origin_state_sig=origin_sig,
            resume_state_sig=safe_text(raw.get("resume_state_sig"), 120),
            created_from_sig=origin_sig,
            created_from_ui=ui_prefix(refs, origin_sig),
            created_by_action=action_to_text(entry_action),
            created_source="final_tasks",
            finish_reason=safe_text(raw.get("finish_reason"), 500),
            finish_state_sig="",
            finish_ui="",
            finish_source="final_tasks",
            priority=raw.get("priority", ""),
            step_budget=raw.get("step_budget", ""),
            used_steps=raw.get("used_steps", ""),
            inferred=True,
        )
        nodes[task_id] = node

        for idx, hist in enumerate(raw.get("history") or [], start=1):
            if not isinstance(hist, dict):
                continue
            sig = safe_text(hist.get("state_sig"), 120)
            node.steps.append(
                TaskStep(
                    step_id=idx,
                    state_sig=sig,
                    ui_prefix=ui_prefix(refs, sig),
                    page_summary=safe_text(hist.get("page_summary"), 800),
                    progress=safe_text(hist.get("progress"), 1000),
                    selected_action=safe_text(hist.get("selected_action"), 500),
                    action_intent=safe_text(hist.get("action_intent"), 1000),
                    source="task_history",
                )
            )
    return nodes


def apply_task_lifecycle_events(events: Sequence[Dict[str, Any]], nodes: Dict[str, TaskNode], refs: Dict[str, UiRef]) -> int:
    """Apply explicit task creation/finish events to task nodes.

    Inputs:
    - events: parsed trace records.
    - nodes: task node mapping loaded from task artifacts.
    - refs: UI reference mapping.

    Output:
    - Number of lifecycle events applied.
    """

    applied = 0
    for event, detail in collect_task_trace_events(events):
        event_type = safe_text(detail.get("event_type"), 120)
        if event_type not in {"task_created", "task_finished"}:
            continue
        ctx = event.get("ctx") or {}
        task_id = safe_text(detail.get("task_id"), 80) or current_task_id(event)
        if not task_id:
            continue
        if task_id not in nodes:
            nodes[task_id] = TaskNode(task_id=task_id, status="unknown")
        node = nodes[task_id]
        if event_type == "task_created":
            origin_sig = safe_text(detail.get("origin_state_sig") or detail.get("state_sig") or ctx.get("cur_sig"), 120)
            entry_action = detail.get("entry_action") if isinstance(detail.get("entry_action"), dict) else {}
            node.task_type = safe_text(detail.get("task_type") or node.task_type, 120)
            node.initial_goal = safe_text(detail.get("initial_goal") or node.initial_goal, 1200)
            node.current_goal = safe_text(detail.get("current_goal") or node.current_goal, 1200)
            node.parent_task_id = safe_text(detail.get("parent_task_id") or node.parent_task_id, 80)
            node.origin_state_sig = origin_sig or node.origin_state_sig
            node.resume_state_sig = safe_text(detail.get("resume_state_sig") or node.resume_state_sig, 120)
            node.created_from_sig = origin_sig or node.created_from_sig
            node.created_from_ui = ui_prefix(refs, origin_sig) if origin_sig else node.created_from_ui
            node.created_by_action = action_to_text(entry_action) or node.created_by_action
            node.created_source = "task_trace"
            node.priority = detail.get("priority", node.priority)
            node.inferred = False
        elif event_type == "task_finished":
            finish_sig = safe_text(detail.get("state_sig") or ctx.get("cur_sig"), 120)
            node.status = safe_text(detail.get("status") or node.status, 80)
            node.finish_reason = safe_text(detail.get("finish_reason") or node.finish_reason, 800)
            node.finish_state_sig = finish_sig
            node.finish_ui = ui_prefix(refs, finish_sig)
            node.finish_source = "task_trace"
            if detail.get("current_goal"):
                node.current_goal = safe_text(detail.get("current_goal"), 1200)
            node.inferred = False
        applied += 1
    return applied


def load_page_summaries(trace_dir: Path, refs: Dict[str, UiRef]) -> Dict[str, Dict[str, Any]]:
    """Load compact per-page summaries from state LLM artifacts.

    Inputs:
    - trace_dir: trace run directory.
    - refs: UI reference mapping.

    Output:
    - Mapping from state signature to summary fields.
    """

    out: Dict[str, Dict[str, Any]] = {}
    for sig, ref in refs.items():
        result = read_json(ref.folder / "navigation_router_result.json", {})
        if not isinstance(result, dict):
            continue
        page_tags = result.get("page_tags")
        if not isinstance(page_tags, list):
            page_tags = []
        out[sig] = {
            "page_summary": safe_text(result.get("page_summary") or result.get("summary"), 900),
            "page_kind": safe_text(result.get("page_kind"), 80),
            "task_progress": safe_text(result.get("task_progress"), 1000),
            "page_tags": page_tags,
        }
    return out


def current_task_id(event: Dict[str, Any]) -> str:
    """Extract current task id from one trace record.

    Inputs:
    - event: parsed trace record.

    Output:
    - Current task id or empty string.
    """

    ctx = event.get("ctx") or {}
    task_ctx = ctx.get("task") or {}
    current = task_ctx.get("current_task") or {}
    return safe_text(current.get("task_id"), 80)


def task_trace_detail(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a task_trace decision payload from one trace event.

    Inputs:
    - event: parsed trace record.

    Output:
    - task_trace detail dictionary, or empty dict when this event is unrelated.
    """

    if event.get("event") != "decision":
        return {}
    data = event.get("data") or {}
    if data.get("name") != "task_trace":
        return {}
    detail = data.get("detail") or {}
    return detail if isinstance(detail, dict) else {}


def collect_task_trace_events(events: Sequence[Dict[str, Any]], event_type: str = "") -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Collect task_trace records, optionally filtered by event type.

    Inputs:
    - events: parsed trace records.
    - event_type: optional task event type such as ``task_created``.

    Output:
    - List of ``(event, detail)`` pairs preserving trace order.
    """

    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for event in events:
        detail = task_trace_detail(event)
        if not detail:
            continue
        if event_type and safe_text(detail.get("event_type"), 120) != event_type:
            continue
        rows.append((event, detail))
    return rows


def nearest_transition_after(events: Sequence[Dict[str, Any]], start_index: int, source_sig: str) -> Tuple[str, str]:
    """Find the next transition destination after an action.

    Inputs:
    - events: all trace records in time order.
    - start_index: index after which to search.
    - source_sig: action source signature.

    Output:
    - Tuple of ``(next_state_sig, transition_kind)``. Empty strings if unknown.
    """

    for event in events[start_index + 1 : min(len(events), start_index + 8)]:
        if event.get("event") != "transition":
            continue
        data = event.get("data") or {}
        src = safe_text(data.get("src") or (event.get("ctx") or {}).get("cur_sig"), 120)
        dst = safe_text(data.get("dst") or data.get("sig"), 120)
        kind = safe_text(data.get("kind"), 120)
        if not source_sig or src == source_sig or dst:
            return dst, kind
    return "", ""


def append_trace_action_steps(
    events: Sequence[Dict[str, Any]],
    nodes: Dict[str, TaskNode],
    refs: Dict[str, UiRef],
    page_summaries: Dict[str, Dict[str, Any]],
) -> None:
    """Add action-derived task steps to task nodes.

    Inputs:
    - events: parsed trace records.
    - nodes: task node mapping.
    - refs: UI reference mapping.
    - page_summaries: per-page summaries.

    Output:
    - Mutates nodes in place by appending trace action steps.
    """

    seen = {(node.task_id, step.state_sig, step.selected_action) for node in nodes.values() for step in node.steps}
    for idx, event in enumerate(events):
        if event.get("event") != "action":
            continue
        data = event.get("data") or {}
        if data.get("phase") != "after":
            continue
        task_id = current_task_id(event)
        if not task_id:
            task_id = "unassigned"
        if task_id not in nodes:
            nodes[task_id] = TaskNode(task_id=task_id, task_type="unassigned", status="unknown")
        ctx = event.get("ctx") or {}
        sig = safe_text(ctx.get("cur_sig"), 120)
        action = data.get("action") or {}
        extra = data.get("extra") or {}
        action_text = safe_text(extra.get("action_key") or action_to_text(action, extra), 500)
        key = (task_id, sig, action_text)
        if key in seen:
            continue
        seen.add(key)
        next_sig, trans_kind = nearest_transition_after(events, idx, sig)
        summary = page_summaries.get(sig, {})
        nodes[task_id].steps.append(
            TaskStep(
                step_id=int(ctx.get("step_id") or 0),
                state_sig=sig,
                ui_prefix=ui_prefix(refs, sig),
                page_summary=safe_text(summary.get("page_summary"), 900),
                page_kind=safe_text(summary.get("page_kind"), 80),
                progress=safe_text(summary.get("task_progress"), 1000),
                selected_action=action_text,
                action_intent=safe_text(extra.get("reasoning") or action.get("reasoning"), 1000),
                action_success=bool(extra.get("success")) if "success" in extra else None,
                action_reason=safe_text(extra.get("reason") or trans_kind, 300),
                next_state_sig=next_sig,
                next_ui_prefix=ui_prefix(refs, next_sig),
                source="trace_action",
            )
        )


def collect_drift_and_family_checks(events: Sequence[Dict[str, Any]], refs: Dict[str, UiRef]) -> List[Dict[str, Any]]:
    """Collect drift and state-family comparison records.

    Inputs:
    - events: parsed trace records.
    - refs: UI reference mapping.

    Output:
    - List of normalized check dictionaries.
    """

    checks: List[Dict[str, Any]] = []
    trace_check_found = False
    for event, detail in collect_task_trace_events(events, "state_family_check"):
        ctx = event.get("ctx") or {}
        result = detail.get("result") if isinstance(detail.get("result"), dict) else {}
        phase = safe_text(detail.get("phase"), 80)
        if phase not in {"after", "error"}:
            continue
        trace_check_found = True
        ref_sig = safe_text(
            detail.get("reference_state_sig")
            or detail.get("expected_state_sig")
            or result.get("reference_state_sig")
            or result.get("expected_state_sig")
            or ctx.get("cur_sig"),
            120,
        )
        obs_sig = safe_text(
            detail.get("observed_state_sig")
            or result.get("observed_state_sig")
            or result.get("actual_state_sig")
            or result.get("new_state_sig")
            or detail.get("state_sig"),
            120,
        )
        nearest_sig = safe_text(detail.get("nearest_state_sig") or result.get("nearest_state_sig"), 120)
        same_family = result.get("same_family", "")
        if same_family == "":
            same_family = result.get("same_page", "")
        checks.append(
            {
                "kind": safe_text(detail.get("check_kind") or detail.get("event_type"), 120),
                "phase": phase,
                "step_id": ctx.get("step_id"),
                "task_id": safe_text(detail.get("task_id") or current_task_id(event), 80),
                "reference_state_sig": ref_sig,
                "observed_state_sig": obs_sig,
                "nearest_state_sig": nearest_sig,
                "reference_ui": ui_prefix(refs, ref_sig),
                "observed_ui": ui_prefix(refs, obs_sig),
                "nearest_ui": ui_prefix(refs, nearest_sig),
                "same_state": result.get("same_state", ""),
                "same_family": same_family,
                "match_type": safe_text(result.get("match_type"), 120),
                "matched_state_sig": safe_text(result.get("matched_state_sig"), 120),
                "recommended_next_step": safe_text(result.get("recommended_next_step"), 500),
                "visible_change_summary": safe_text(result.get("visible_change_summary") or result.get("reason"), 800),
                "source": safe_text(detail.get("source"), 120),
                "duration_s": detail.get("duration_s", ""),
                "error": safe_text(detail.get("error"), 500),
                "result_path": "",
            }
        )
    for event in events:
        ctx = event.get("ctx") or {}
        data = event.get("data") or {}
        task_id = current_task_id(event)
        if event.get("event") == "decision" and data.get("name") == "drift_detected":
            from_sig = safe_text(data.get("from_sig") or ctx.get("cur_sig"), 120)
            to_sig = safe_text(data.get("to_sig"), 120)
            checks.append(
                {
                    "kind": "drift_detected",
                    "step_id": ctx.get("step_id"),
                    "task_id": task_id,
                    "reference_state_sig": from_sig,
                    "observed_state_sig": to_sig,
                    "reference_ui": ui_prefix(refs, from_sig),
                    "observed_ui": ui_prefix(refs, to_sig),
                    "summary": safe_text(data, 1000),
                }
            )
        if (not trace_check_found) and event.get("event") == "llm_result" and data.get("kind") in {"state_family_drift", "state_family_transition"}:
            result_path = Path(str(data.get("result_path") or ""))
            result_file = read_json(result_path, {}) if result_path else {}
            result = result_file.get("result") if isinstance(result_file, dict) else {}
            if not isinstance(result, dict):
                result = result_file if isinstance(result_file, dict) else {}
            ref_sig = safe_text(
                result.get("reference_state_sig")
                or result_file.get("reference_state_sig")
                or result.get("expected_state_sig")
                or result.get("old_state_sig")
                or ctx.get("cur_sig"),
                120,
            )
            obs_sig = safe_text(
                result.get("observed_state_sig")
                or result_file.get("observed_state_sig")
                or result.get("actual_state_sig")
                or result.get("new_state_sig")
                or result_file.get("state_sig")
                or data.get("state_sig"),
                120,
            )
            same_family = result.get("same_family", "")
            if same_family == "":
                same_family = result.get("same_page", "")
            checks.append(
                {
                    "kind": data.get("kind"),
                    "step_id": ctx.get("step_id"),
                    "task_id": task_id,
                    "reference_state_sig": ref_sig,
                    "observed_state_sig": obs_sig,
                    "reference_ui": ui_prefix(refs, ref_sig),
                    "observed_ui": ui_prefix(refs, obs_sig),
                    "same_state": result.get("same_state", ""),
                    "same_family": same_family,
                    "recommended_next_step": safe_text(result.get("recommended_next_step"), 500),
                    "visible_change_summary": safe_text(result.get("visible_change_summary"), 800),
                    "result_path": str(result_path) if result_path else "",
                }
            )
    return checks


def attach_checks_to_steps(nodes: Dict[str, TaskNode], checks: Sequence[Dict[str, Any]]) -> None:
    """Attach drift/state-family checks to nearby task steps.

    Inputs:
    - nodes: task node mapping.
    - checks: normalized check records.

    Output:
    - Mutates task steps by appending matching checks.
    """

    by_task = {task_id: node for task_id, node in nodes.items()}
    for check in checks:
        task_id = safe_text(check.get("task_id"), 80)
        node = by_task.get(task_id)
        if not node or not node.steps:
            continue
        step_id = int(check.get("step_id") or 0)
        best = min(node.steps, key=lambda step: abs(int(step.step_id or 0) - step_id))
        best.drift_checks.append(dict(check))


def infer_task_switches(events: Sequence[Dict[str, Any]], refs: Dict[str, UiRef]) -> List[Dict[str, Any]]:
    """Infer task switch events from current-task changes in trace context.

    Inputs:
    - events: parsed trace records.
    - refs: UI reference mapping.

    Output:
    - List of inferred task switch dictionaries.
    """

    switches: List[Dict[str, Any]] = []
    previous_task = ""
    previous_sig = ""
    previous_step = 0
    for event in events:
        task_id = current_task_id(event)
        ctx = event.get("ctx") or {}
        sig = safe_text(ctx.get("cur_sig"), 120)
        step_id = int(ctx.get("step_id") or 0)
        if task_id and previous_task and task_id != previous_task:
            switches.append(
                {
                    "from_task_id": previous_task,
                    "to_task_id": task_id,
                    "from_state_sig": previous_sig,
                    "to_state_sig": sig,
                    "from_ui": ui_prefix(refs, previous_sig),
                    "to_ui": ui_prefix(refs, sig),
                    "step_id": step_id,
                    "previous_step_id": previous_step,
                    "route_strategy": "inferred_from_current_task_change",
                    "route_steps": "",
                    "route_success": "",
                    "inferred": True,
                }
            )
        if task_id:
            previous_task = task_id
            previous_sig = sig
            previous_step = step_id
    return switches


def collect_task_switch_events(events: Sequence[Dict[str, Any]], refs: Dict[str, UiRef]) -> List[Dict[str, Any]]:
    """Collect explicit task switch events from task_trace records.

    Inputs:
    - events: parsed trace records.
    - refs: UI reference mapping.

    Output:
    - List of task switch dictionaries.
    """

    rows: List[Dict[str, Any]] = []
    for event, detail in collect_task_trace_events(events, "task_switch"):
        ctx = event.get("ctx") or {}
        from_sig = safe_text(detail.get("from_state_sig") or ctx.get("cur_sig"), 120)
        to_sig = safe_text(detail.get("to_state_sig") or ctx.get("cur_sig"), 120)
        rows.append(
            {
                "from_task_id": safe_text(detail.get("from_task_id"), 80),
                "to_task_id": safe_text(detail.get("to_task_id") or detail.get("task_id"), 80),
                "from_state_sig": from_sig,
                "to_state_sig": to_sig,
                "from_ui": ui_prefix(refs, from_sig),
                "to_ui": ui_prefix(refs, to_sig),
                "step_id": int(ctx.get("step_id") or 0),
                "previous_step_id": "",
                "route_strategy": safe_text(detail.get("reason") or "task_trace", 120),
                "route_steps": "",
                "route_success": "",
                "inferred": False,
            }
        )
    return rows


def collect_task_route_events(events: Sequence[Dict[str, Any]], refs: Dict[str, UiRef]) -> List[Dict[str, Any]]:
    """Collect task resume/route start and finish events.

    Inputs:
    - events: parsed trace records.
    - refs: UI reference mapping.

    Output:
    - List of route dictionaries combining start and finish events by route_id.
    """

    routes: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for event, detail in collect_task_trace_events(events):
        event_type = safe_text(detail.get("event_type"), 120)
        if event_type not in {"task_route_started", "task_route_finished"}:
            continue
        ctx = event.get("ctx") or {}
        route_id = safe_text(detail.get("route_id"), 200)
        if not route_id:
            route_id = f"{safe_text(detail.get('task_id'), 80)}:{ctx.get('step_id')}"
        if route_id not in routes:
            routes[route_id] = {"route_id": route_id}
            order.append(route_id)
        row = routes[route_id]
        from_sig = safe_text(detail.get("from_state_sig") or row.get("from_state_sig") or ctx.get("cur_sig"), 120)
        target_sig = safe_text(detail.get("target_state_sig") or row.get("target_state_sig"), 120)
        reached_sig = safe_text(detail.get("reached_state_sig") or row.get("reached_state_sig"), 120)
        row.update(
            {
                "task_id": safe_text(detail.get("task_id") or row.get("task_id"), 80),
                "route_context": safe_text(detail.get("route_context") or row.get("route_context"), 120),
                "from_state_sig": from_sig,
                "from_ui": ui_prefix(refs, from_sig),
                "target_state_sig": target_sig,
                "target_ui": ui_prefix(refs, target_sig),
                "reached_state_sig": reached_sig,
                "reached_ui": ui_prefix(refs, reached_sig),
                "start_step_id": row.get("start_step_id", ""),
                "finish_step_id": row.get("finish_step_id", ""),
                "strategy": safe_text(detail.get("strategy") or row.get("strategy"), 120),
                "success": row.get("success", ""),
            }
        )
        if event_type == "task_route_started":
            row["start_step_id"] = int(ctx.get("step_id") or 0)
        if event_type == "task_route_finished":
            row["finish_step_id"] = int(ctx.get("step_id") or 0)
            row["success"] = detail.get("success")
    return [routes[key] for key in order]


def screenshot_for_ui(ref: UiRef) -> str:
    """Choose the best screenshot path for a UI state.

    Inputs:
    - ref: UI reference.

    Output:
    - Absolute image path string, or empty string when unavailable.
    """

    if not ref.folder.exists():
        return ""
    overlay_dir = ref.folder / "overlays"
    if overlay_dir.exists():
        preferred = sorted(list(overlay_dir.glob("*action*.png")) + list(overlay_dir.glob("*actions*.png")))
        if preferred:
            return str(preferred[0])
    for name in ("screenshot.png", "screenshot_raw.png", "vid_map_overlay.png", "uist_overlay.png"):
        path = ref.folder / name
        if path.exists():
            return str(path)
    if overlay_dir.exists():
        any_png = sorted(overlay_dir.glob("*.png"))
        if any_png:
            return str(any_png[0])
    return ""


def task_to_dict(node: TaskNode) -> Dict[str, Any]:
    """Serialize one TaskNode.

    Inputs:
    - node: task node dataclass.

    Output:
    - JSON-serializable dictionary.
    """

    return {
        "task_id": node.task_id,
        "task_type": node.task_type,
        "status": node.status,
        "initial_goal": node.initial_goal,
        "current_goal": node.current_goal,
        "progress_summary": node.progress_summary,
        "parent_task_id": node.parent_task_id,
        "origin_state_sig": node.origin_state_sig,
        "resume_state_sig": node.resume_state_sig,
        "created_from_sig": node.created_from_sig,
        "created_from_ui": node.created_from_ui,
        "created_by_action": node.created_by_action,
        "created_source": node.created_source,
        "finish_reason": node.finish_reason,
        "finish_state_sig": node.finish_state_sig,
        "finish_ui": node.finish_ui,
        "finish_source": node.finish_source,
        "priority": node.priority,
        "step_budget": node.step_budget,
        "used_steps": node.used_steps,
        "inferred": node.inferred,
        "steps": [step.__dict__ for step in sorted(node.steps, key=lambda s: (int(s.step_id or 0), s.state_sig, s.selected_action))],
    }


def build_report(trace_dir: Path) -> Dict[str, Any]:
    """Build normalized report data for one trace directory.

    Inputs:
    - trace_dir: workflow trace run directory.

    Output:
    - JSON-serializable report dictionary.
    """

    trace_path = trace_dir / "trace.jsonl"
    events = read_jsonl(trace_path)
    run_data = read_json(trace_dir / "run.json", {})
    refs = build_ui_refs(trace_dir, events)
    page_summaries = load_page_summaries(trace_dir, refs)
    nodes = load_task_nodes(trace_dir, refs)
    lifecycle_event_count = apply_task_lifecycle_events(events, nodes, refs)
    append_trace_action_steps(events, nodes, refs, page_summaries)
    checks = collect_drift_and_family_checks(events, refs)
    attach_checks_to_steps(nodes, checks)
    explicit_switches = collect_task_switch_events(events, refs)
    switches = explicit_switches or infer_task_switches(events, refs)
    routes = collect_task_route_events(events, refs)
    task_trace_event_count = len(collect_task_trace_events(events))

    event_counts = Counter(str(row.get("event") or "") for row in events)
    status_counts = Counter(node.status or "unknown" for node in nodes.values())
    tasks = [task_to_dict(node) for node in sorted(nodes.values(), key=lambda n: n.task_id)]
    return {
        "trace_dir": str(trace_dir),
        "run_id": trace_dir.name,
        "run": run_data if isinstance(run_data, dict) else {},
        "summary": {
            "event_count": len(events),
            "event_counts": dict(event_counts),
            "task_count": len(tasks),
            "task_status_counts": dict(status_counts),
            "action_event_count": event_counts.get("action", 0),
            "drift_or_family_check_count": len(checks),
            "task_switch_count": len(switches),
            "task_route_count": len(routes),
            "task_trace_event_count": task_trace_event_count,
            "task_lifecycle_event_count": lifecycle_event_count,
            "task_switch_source": "task_trace" if explicit_switches else "inferred",
            "ui_count": len(refs),
            "trace_jsonl_found": trace_path.exists(),
        },
        "tasks": tasks,
        "task_switches": switches,
        "task_routes": routes,
        "state_family_checks": checks,
        "ui_refs": {
            sig: {"ui_prefix": ref.ui_prefix, "folder": str(ref.folder), "screenshot": screenshot_for_ui(ref)}
            for sig, ref in refs.items()
        },
    }


def render_md(report: Dict[str, Any]) -> str:
    """Render report data as Markdown.

    Inputs:
    - report: normalized report dictionary.

    Output:
    - Markdown string.
    """

    summary = report.get("summary") or {}
    lines = [
        "# Task Execution Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- trace_dir: `{report.get('trace_dir', '')}`",
        f"- tasks: {summary.get('task_count', 0)}",
        f"- ui_count: {summary.get('ui_count', 0)}",
        f"- action_events: {summary.get('action_event_count', 0)}",
        f"- drift_or_family_checks: {summary.get('drift_or_family_check_count', 0)}",
        f"- task_switches: {summary.get('task_switch_count', 0)} source=`{summary.get('task_switch_source', '')}`",
        f"- task_routes: {summary.get('task_route_count', 0)}",
        f"- task_trace_events: {summary.get('task_trace_event_count', 0)}",
        "",
        "## Task Tree",
        "",
    ]
    for task in report.get("tasks") or []:
        parent = task.get("parent_task_id") or "root"
        lines.append(
            f"- `{task.get('task_id')}` type=`{task.get('task_type')}` status=`{task.get('status')}` parent=`{parent}` "
            f"created_ui=`{task.get('created_from_ui')}` action={task.get('created_by_action') or '-'}"
        )
    lines.extend(["", "## Task Chains", ""])
    for task in report.get("tasks") or []:
        lines.append(f"### {task.get('task_id')} {task.get('task_type')}")
        lines.append("")
        lines.append(f"- status: `{task.get('status')}`")
        lines.append(f"- goal: {task.get('current_goal') or task.get('initial_goal') or '-'}")
        lines.append(f"- created: `{task.get('created_from_ui')}` via {task.get('created_by_action') or '-'}")
        if task.get("finish_ui"):
            lines.append(f"- finished: `{task.get('finish_ui')}` source=`{task.get('finish_source')}`")
        if task.get("finish_reason"):
            lines.append(f"- finish_reason: {task.get('finish_reason')}")
        lines.append("")
        steps = task.get("steps") or []
        if not steps:
            lines.append("- No reconstructed steps.")
            lines.append("")
            continue
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. `{step.get('ui_prefix')}` `{step.get('state_sig')}`")
            if step.get("page_summary"):
                lines.append(f"   - page: {step.get('page_summary')}")
            if step.get("progress"):
                lines.append(f"   - progress: {step.get('progress')}")
            if step.get("selected_action"):
                lines.append(f"   - action: {step.get('selected_action')}")
            if step.get("action_intent"):
                lines.append(f"   - intent: {step.get('action_intent')}")
            if step.get("next_ui_prefix"):
                lines.append(f"   - next: `{step.get('next_ui_prefix')}` `{step.get('next_state_sig')}`")
            checks = step.get("drift_checks") or []
            for check in checks:
                lines.append(
                    f"   - check: `{check.get('kind')}` ref=`{check.get('reference_ui')}` obs=`{check.get('observed_ui')}` "
                    f"same_family=`{display_value(check.get('same_family'))}` step={check.get('step_id')}"
                )
        lines.append("")
    lines.extend(["## Task Switches", ""])
    switches = report.get("task_switches") or []
    if not switches:
        lines.append("- No task switches inferred.")
    for row in switches:
        lines.append(
            f"- step {row.get('step_id')}: `{row.get('from_task_id')}` at `{row.get('from_ui')}` -> "
            f"`{row.get('to_task_id')}` at `{row.get('to_ui')}` strategy=`{row.get('route_strategy')}`"
        )
    lines.extend(["", "## Task Routes", ""])
    routes = report.get("task_routes") or []
    if not routes:
        lines.append("- No task route events found.")
    for row in routes:
        lines.append(
            f"- route `{row.get('route_id')}` task=`{row.get('task_id')}` "
            f"from=`{row.get('from_ui')}` target=`{row.get('target_ui')}` reached=`{row.get('reached_ui')}` "
            f"success=`{display_value(row.get('success'))}` strategy=`{row.get('strategy')}`"
        )
    lines.extend(["", "## Drift / State-Family Checks", ""])
    checks = report.get("state_family_checks") or []
    if not checks:
        lines.append("- No drift/state-family checks found.")
    for row in checks:
        lines.append(
            f"- step {row.get('step_id')}: `{row.get('kind')}` task=`{row.get('task_id')}` "
            f"ref=`{row.get('reference_ui')}` obs=`{row.get('observed_ui')}` "
            f"same_family=`{display_value(row.get('same_family'))}` {row.get('visible_change_summary') or row.get('summary') or ''}"
        )
    lines.append("")
    return "\n".join(lines)


def html_escape(value: Any) -> str:
    """Escape a value for HTML.

    Inputs:
    - value: arbitrary value.

    Output:
    - HTML-safe string.
    """

    return html.escape(safe_text(value, 2000))


def display_value(value: Any) -> str:
    """Format report scalar values without losing False/0.

    Inputs:
    - value: scalar value from normalized report data.

    Output:
    - String that preserves booleans and numbers for display.
    """

    if value is None:
        return ""
    return safe_text(value, 500)


def render_html(report: Dict[str, Any]) -> str:
    """Render report data as a standalone HTML page.

    Inputs:
    - report: normalized report dictionary.

    Output:
    - HTML document string.
    """

    ui_refs = report.get("ui_refs") or {}
    summary = report.get("summary") or {}

    def img_for(sig: str) -> str:
        """Return HTML image tag for one state signature."""
        item = ui_refs.get(sig) or {}
        path = item.get("screenshot") or ""
        if not path:
            return '<div class="no-img">No screenshot</div>'
        return f'<img class="thumb" src="{html_escape(path)}" alt="{html_escape(sig)}">'

    task_rows = []
    for task in report.get("tasks") or []:
        task_rows.append(
            "<tr>"
            f"<td><code>{html_escape(task.get('task_id'))}</code></td>"
            f"<td>{html_escape(task.get('task_type'))}</td>"
            f"<td>{html_escape(task.get('status'))}</td>"
            f"<td><code>{html_escape(task.get('parent_task_id') or 'root')}</code></td>"
            f"<td><code>{html_escape(task.get('created_from_ui'))}</code></td>"
            f"<td>{html_escape(task.get('created_by_action') or '-')}</td>"
            f"<td>{len(task.get('steps') or [])}</td>"
            "</tr>"
        )

    task_sections = []
    for task in report.get("tasks") or []:
        step_cards = []
        for step in task.get("steps") or []:
            checks = step.get("drift_checks") or []
            check_html = ""
            if checks:
                check_items = []
                for check in checks:
                    check_items.append(
                        f"<li><b>{html_escape(check.get('kind'))}</b> "
                        f"ref=<code>{html_escape(check.get('reference_ui'))}</code> "
                        f"obs=<code>{html_escape(check.get('observed_ui'))}</code> "
                        f"same_family=<code>{html_escape(display_value(check.get('same_family')))}</code><br>"
                        f"{html_escape(check.get('visible_change_summary') or check.get('summary'))}</li>"
                    )
                check_html = '<ul class="checks">' + "".join(check_items) + "</ul>"
            next_html = ""
            if step.get("next_ui_prefix"):
                next_html = (
                    f'<div class="meta">next: <code>{html_escape(step.get("next_ui_prefix"))}</code> '
                    f'<code>{html_escape(step.get("next_state_sig"))}</code></div>'
                )
            folder = (ui_refs.get(step.get("state_sig") or "") or {}).get("folder") or ""
            step_cards.append(
                '<div class="step-card">'
                f'<div class="shot">{img_for(step.get("state_sig") or "")}</div>'
                '<div class="step-body">'
                f'<div class="meta">step {html_escape(step.get("step_id"))} | '
                f'<code>{html_escape(step.get("ui_prefix"))}</code> | '
                f'<code>{html_escape(step.get("state_sig"))}</code></div>'
                f'<div class="folder">{html_escape(folder)}</div>'
                f'<div><b>page</b>: {html_escape(step.get("page_summary") or "-")}</div>'
                f'<div><b>progress</b>: {html_escape(step.get("progress") or "-")}</div>'
                f'<div><b>action</b>: {html_escape(step.get("selected_action") or "-")}</div>'
                f'<div><b>intent</b>: {html_escape(step.get("action_intent") or "-")}</div>'
                f"{next_html}{check_html}"
                "</div></div>"
            )
        if not step_cards:
            step_cards.append('<div class="empty">No reconstructed steps.</div>')
        task_sections.append(
            f'<details class="task" open><summary><code>{html_escape(task.get("task_id"))}</code> '
            f'{html_escape(task.get("task_type"))} '
            f'<span class="badge">{html_escape(task.get("status"))}</span></summary>'
            f'<div class="goal"><b>goal</b>: {html_escape(task.get("current_goal") or task.get("initial_goal") or "-")}</div>'
            f'<div class="goal"><b>created</b>: <code>{html_escape(task.get("created_from_ui"))}</code> '
            f'via {html_escape(task.get("created_by_action") or "-")} '
            f'<span class="meta">source={html_escape(task.get("created_source"))}</span></div>'
            f'<div class="goal"><b>finished</b>: <code>{html_escape(task.get("finish_ui") or "-")}</code> '
            f'{html_escape(task.get("finish_reason") or "")} '
            f'<span class="meta">source={html_escape(task.get("finish_source"))}</span></div>'
            + "".join(step_cards)
            + "</details>"
        )

    switch_rows = []
    for row in report.get("task_switches") or []:
        switch_rows.append(
            "<tr>"
            f"<td>{html_escape(row.get('step_id'))}</td>"
            f"<td><code>{html_escape(row.get('from_task_id'))}</code></td>"
            f"<td><code>{html_escape(row.get('from_ui'))}</code></td>"
            f"<td><code>{html_escape(row.get('to_task_id'))}</code></td>"
            f"<td><code>{html_escape(row.get('to_ui'))}</code></td>"
            f"<td>{html_escape(row.get('route_strategy'))}</td>"
            "</tr>"
        )

    route_rows = []
    for row in report.get("task_routes") or []:
        route_rows.append(
            "<tr>"
            f"<td>{html_escape(row.get('start_step_id'))}</td>"
            f"<td>{html_escape(row.get('finish_step_id'))}</td>"
            f"<td><code>{html_escape(row.get('task_id'))}</code></td>"
            f"<td>{html_escape(row.get('route_context'))}</td>"
            f"<td><code>{html_escape(row.get('from_ui'))}</code></td>"
            f"<td><code>{html_escape(row.get('target_ui'))}</code></td>"
            f"<td><code>{html_escape(row.get('reached_ui'))}</code></td>"
            f"<td><code>{html_escape(display_value(row.get('success')))}</code></td>"
            f"<td>{html_escape(row.get('strategy'))}</td>"
            "</tr>"
        )

    check_rows = []
    for row in report.get("state_family_checks") or []:
        check_rows.append(
            "<tr>"
            f"<td>{html_escape(row.get('step_id'))}</td>"
            f"<td>{html_escape(row.get('kind'))}</td>"
            f"<td><code>{html_escape(row.get('task_id'))}</code></td>"
            f"<td><code>{html_escape(row.get('reference_ui'))}</code></td>"
            f"<td><code>{html_escape(row.get('observed_ui'))}</code></td>"
            f"<td><code>{html_escape(display_value(row.get('same_family')))}</code></td>"
            f"<td>{html_escape(row.get('visible_change_summary') or row.get('summary'))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Task Execution Report - {html_escape(report.get('run_id'))}</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2937; background: #f7f8fa; }}
h1, h2 {{ margin: 18px 0 10px; }}
code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0 22px; }}
th, td {{ border: 1px solid #d9dee7; padding: 7px 8px; vertical-align: top; font-size: 13px; }}
th {{ background: #eef2f7; text-align: left; }}
.summary {{ display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 8px; margin-bottom: 18px; }}
.metric {{ background: white; border: 1px solid #d9dee7; padding: 10px; border-radius: 6px; }}
.metric b {{ display: block; font-size: 20px; }}
.task {{ background: white; border: 1px solid #d9dee7; border-radius: 6px; margin: 12px 0; padding: 10px; }}
.task summary {{ cursor: pointer; font-weight: 700; }}
.badge {{ background: #e8f2ff; border: 1px solid #bcd7ff; padding: 1px 6px; border-radius: 999px; font-size: 12px; margin-left: 8px; }}
.goal {{ margin: 8px 0; font-size: 13px; }}
.step-card {{ display: grid; grid-template-columns: 220px 1fr; gap: 12px; border-top: 1px solid #eceff5; padding: 12px 0; }}
.thumb {{ max-width: 220px; max-height: 150px; object-fit: contain; border: 1px solid #cfd6e0; background: #111; }}
.no-img {{ width: 220px; height: 120px; display: flex; align-items: center; justify-content: center; border: 1px dashed #9aa4b2; color: #6b7280; }}
.meta, .folder {{ color: #526070; font-size: 12px; margin-bottom: 5px; word-break: break-all; }}
.checks {{ background: #fff8e6; border: 1px solid #f2da8a; padding: 8px 8px 8px 24px; }}
.empty {{ color: #6b7280; padding: 8px; }}
</style>
</head>
<body>
<h1>Task Execution Report</h1>
<div class="meta"><code>{html_escape(report.get('run_id'))}</code></div>
<div class="meta">{html_escape(report.get('trace_dir'))}</div>
<section class="summary">
  <div class="metric">Tasks <b>{html_escape(summary.get('task_count'))}</b></div>
  <div class="metric">UIs <b>{html_escape(summary.get('ui_count'))}</b></div>
  <div class="metric">Actions <b>{html_escape(summary.get('action_event_count'))}</b></div>
  <div class="metric">Drift/Family <b>{html_escape(summary.get('drift_or_family_check_count'))}</b></div>
  <div class="metric">Task Routes <b>{html_escape(summary.get('task_route_count'))}</b></div>
</section>
<h2>Task Overview</h2>
<table><thead><tr><th>task</th><th>type</th><th>status</th><th>parent</th><th>created UI</th><th>created action</th><th>steps</th></tr></thead>
<tbody>{''.join(task_rows) or '<tr><td colspan="7">No tasks</td></tr>'}</tbody></table>
<h2>Task Chains</h2>
{''.join(task_sections)}
<h2>Task Switches</h2>
<table><thead><tr><th>step</th><th>from task</th><th>from UI</th><th>to task</th><th>to UI</th><th>strategy</th></tr></thead>
<tbody>{''.join(switch_rows) or '<tr><td colspan="6">No task switches inferred.</td></tr>'}</tbody></table>
<h2>Task Routes</h2>
<table><thead><tr><th>start step</th><th>finish step</th><th>task</th><th>context</th><th>from UI</th><th>target UI</th><th>reached UI</th><th>success</th><th>strategy</th></tr></thead>
<tbody>{''.join(route_rows) or '<tr><td colspan="9">No task route events found.</td></tr>'}</tbody></table>
<h2>Drift / State-Family Checks</h2>
<table><thead><tr><th>step</th><th>kind</th><th>task</th><th>reference UI</th><th>observed UI</th><th>same_family</th><th>summary</th></tr></thead>
<tbody>{''.join(check_rows) or '<tr><td colspan="7">No drift/state-family checks found.</td></tr>'}</tbody></table>
</body>
</html>
"""


def write_outputs(report: Dict[str, Any], args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """Write JSON, Markdown, and HTML report files.

    Inputs:
    - report: normalized report dictionary.
    - args: CLI arguments containing optional output paths.

    Output:
    - Tuple of written JSON, Markdown, and HTML paths.
    """

    trace_dir = Path(report["trace_dir"])
    json_path = Path(args.output_json) if args.output_json else trace_dir / "task_execution_report.json"
    md_path = Path(args.output_md) if args.output_md else trace_dir / "task_execution_report.md"
    html_path = Path(args.output_html) if args.output_html else trace_dir / "task_execution_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, md_path, html_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the task execution report builder.

    Inputs:
    - argv: optional CLI argument list.

    Output:
    - Process exit code. ``0`` means report files were written or a readable
      missing-trace report was produced; ``2`` means the trace directory is
      invalid.
    """

    args = parse_args(argv)
    trace_dir = Path(args.trace_dir).resolve()
    if not trace_dir.exists() or not trace_dir.is_dir():
        print(f"[TASK-REPORT][ERROR] trace_dir_not_found={trace_dir}")
        return 2
    report = build_report(trace_dir)
    json_path, md_path, html_path = write_outputs(report, args)
    summary = report.get("summary") or {}
    print(f"[TASK-REPORT] trace_dir={trace_dir}")
    print(f"[TASK-REPORT] events={summary.get('event_count', 0)} tasks={summary.get('task_count', 0)} uis={summary.get('ui_count', 0)}")
    print(
        f"[TASK-REPORT] drift_or_family_checks={summary.get('drift_or_family_check_count', 0)} "
        f"task_switches={summary.get('task_switch_count', 0)} task_routes={summary.get('task_route_count', 0)}"
    )
    print(f"[TASK-REPORT] output_json={json_path}")
    print(f"[TASK-REPORT] output_md={md_path}")
    print(f"[TASK-REPORT] output_html={html_path}")
    if not summary.get("trace_jsonl_found"):
        print("[TASK-REPORT][WARN] trace.jsonl not found; report only contains available static artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
