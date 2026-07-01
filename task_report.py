"""
Task-oriented report generation for one trace run.

Input:
- A run directory containing `tasks.json` and `trace.jsonl`.

Output:
- `task_report.json` and `task_report.md` under the same run directory.

Function:
- Groups UI observations and selected actions by task_id so a human can review
  each task's path through the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    """
    Input: JSON file path.
    Output: decoded JSON object, or an empty dict when the file is missing/invalid.
    Function: keeps report generation tolerant of partial run artifacts.
    """
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_trace_events(path: Path) -> List[Dict[str, Any]]:
    """
    Input: trace.jsonl path.
    Output: list of decoded event dictionaries.
    Function: reads valid JSON object lines and skips malformed lines.
    """
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _event_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: raw trace event.
    Output: payload dictionary containing business fields such as kind/task_id.
    Function: supports both flat test events and real trace events with nested data.
    """
    for key in ("data", "extra", "payload"):
        value = event.get(key)
        if isinstance(value, dict) and (value.get("kind") or value.get("name")):
            return value
    return event


def _task_id_value(value: Any) -> str:
    """
    Input: arbitrary task id-like value.
    Output: normalized string task id.
    Function: avoids repeated None/string cleanup in grouping code.
    """
    return str(value or "").strip()


def _seed_task_rows(tasks_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Input: decoded tasks.json payload.
    Output: task_id -> normalized task report row.
    Function: preserves task metadata even if no per-UI events were recorded.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for task in list(tasks_payload.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        task_id = _task_id_value(task.get("task_id"))
        if not task_id:
            continue
        initial_goal = str(task.get("initial_goal") or task.get("prompt") or "")
        rows[task_id] = {
            "task_id": task_id,
            "task_type": task.get("task_type", ""),
            "status": task.get("status", ""),
            "initial_goal": initial_goal,
            "current_goal": str(task.get("current_goal") or initial_goal),
            "progress_summary": str(task.get("progress_summary") or ""),
            "priority": task.get("priority"),
            "type_priority": task.get("type_priority"),
            "llm_priority": task.get("llm_priority"),
            "used_steps": task.get("used_steps"),
            "step_budget": task.get("step_budget"),
            "finish_reason": task.get("finish_reason", ""),
            "history": list(task.get("history") or []),
            "policy_captures": [],
            "steps": [],
        }
    return rows


def _observation_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: task_ui_observation event payload.
    Output: normalized report step without selected action.
    Function: converts one UI observation into a task timeline row.
    """
    return {
        "state_sig": str(payload.get("state_sig") or payload.get("sig") or ""),
        "page_summary": payload.get("page_summary", ""),
        "page_kind": payload.get("page_kind", ""),
        "page_tags": list(payload.get("page_tags") or []),
        "task_progress": payload.get("task_progress", ""),
        "task_decision": payload.get("task_decision") or {},
        "selected_action": None,
    }


def _selected_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: task_action_selected event payload.
    Output: normalized selected action dictionary.
    Function: extracts action intent and executable action summary for the report.
    """
    return {
        "candidate_key": payload.get("candidate_key", ""),
        "action_role": payload.get("action_role", ""),
        "starts_task_type": payload.get("starts_task_type", ""),
        "starts_task_depth": payload.get("starts_task_depth", ""),
        "score": payload.get("score"),
        "action_intent": payload.get("action_intent", ""),
        "source": payload.get("source", ""),
        "action": payload.get("action"),
        "element_id": payload.get("element_id"),
        "label": payload.get("label", ""),
        "reasoning": payload.get("reasoning", ""),
    }


def _iter_task_events(trace_events: Iterable[Dict[str, Any]]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """
    Input: raw trace event dictionaries.
    Output: iterable of (kind, task_id, payload) rows for task report events.
    Function: filters trace.jsonl down to task UI/action events and policy capture events.
    """
    for event in trace_events:
        payload = _event_payload(event)
        kind = str(payload.get("kind") or "")
        name = str(payload.get("name") or "")
        if kind not in {"task_ui_observation", "task_action_selected"} and name not in {"policy_capture_done", "policy_capture_failed"}:
            continue
        task_id = _task_id_value(payload.get("task_id") or payload.get("source_task_id"))
        if not task_id:
            continue
        if name in {"policy_capture_done", "policy_capture_failed"}:
            yield name, task_id, payload
            continue
        yield kind, task_id, payload


def build_task_report(run_dir: Path) -> Dict[str, Any]:
    """
    Input: run directory path.
    Output: task-oriented report dictionary.
    Function: combines tasks.json with task_ui_observation and task_action_selected events.
    """
    run_dir = Path(run_dir)
    tasks_payload = _read_json(run_dir / "tasks.json")
    task_rows = _seed_task_rows(tasks_payload)
    pending_by_task_state: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for kind, task_id, payload in _iter_task_events(_read_trace_events(run_dir / "trace.jsonl")):
        if task_id not in task_rows:
            task_rows[task_id] = {
                "task_id": task_id,
                "task_type": payload.get("task_type", ""),
                "status": "",
                "initial_goal": "",
                "current_goal": "",
                "progress_summary": "",
                "priority": None,
                "type_priority": None,
                "llm_priority": None,
                "used_steps": None,
                "step_budget": None,
                "finish_reason": "",
                "history": [],
                "policy_captures": [],
                "steps": [],
            }
        if kind in {"policy_capture_done", "policy_capture_failed"}:
            task_rows[task_id].setdefault("policy_captures", []).append(
                {
                    "status": payload.get("status", ""),
                    "failure_reason": payload.get("failure_reason", ""),
                    "document_title": payload.get("document_title", ""),
                    "capture_location": payload.get("capture_location", ""),
                    "url_raw": payload.get("url_raw", ""),
                    "text_char_count": payload.get("text_char_count"),
                    "output_dir": payload.get("output_dir", ""),
                    "document_text_path": payload.get("document_text_path", ""),
                    "screenshot_path": payload.get("screenshot_path", ""),
                    "metadata_path": payload.get("metadata_path", ""),
                    "state_sig": payload.get("state_sig", "") or payload.get("sig", ""),
                }
            )
            continue
        state_sig = str(payload.get("state_sig") or payload.get("sig") or "")
        key = (task_id, state_sig)
        if kind == "task_ui_observation":
            step = _observation_step(payload)
            task_rows[task_id]["steps"].append(step)
            pending_by_task_state[key] = step
            continue
        action = _selected_action(payload)
        step = pending_by_task_state.get(key)
        if step is None:
            step = _observation_step({"state_sig": state_sig})
            task_rows[task_id]["steps"].append(step)
            pending_by_task_state[key] = step
        step["selected_action"] = action

    return {
        "run_id": tasks_payload.get("run_id") or run_dir.name,
        "task_count": len(task_rows),
        "tasks": list(task_rows.values()),
    }


def render_task_report_markdown(report: Dict[str, Any]) -> str:
    """
    Input: task report dictionary.
    Output: markdown report text.
    Function: provides a compact human-readable review grouped by task.
    """
    lines = [
        "# Task Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- task_count: `{report.get('task_count', 0)}`",
        "",
    ]
    for task in report.get("tasks") or []:
        lines.extend(
            [
                f"## {task.get('task_id')} / {task.get('task_type')}",
                "",
                f"- status: `{task.get('status', '')}`",
                f"- priority: `{task.get('priority', '')}` / type `{task.get('type_priority', '')}` / llm `{task.get('llm_priority', '')}`",
                f"- steps: `{task.get('used_steps', '')}/{task.get('step_budget', '')}`",
                f"- initial_goal: {task.get('initial_goal', '')}",
                f"- current_goal: {task.get('current_goal', '')}",
                f"- progress_summary: {task.get('progress_summary', '')}",
            ]
        )
        if task.get("finish_reason"):
            lines.append(f"- finish_reason: {task.get('finish_reason')}")
        lines.append("")
        if task.get("policy_captures"):
            lines.extend(["### Policy Capture", ""])
            for item in task.get("policy_captures") or []:
                lines.extend(
                    [
                        f"- document_title: {item.get('document_title', '')}",
                        f"  - status: `{item.get('status', '')}`",
                        f"  - location: `{item.get('capture_location', '')}`",
                        f"  - url_raw: {item.get('url_raw', '')}",
                        f"  - text_chars: `{item.get('text_char_count', '')}`",
                        f"  - output: `{item.get('output_dir', '')}`",
                        f"  - text: `{item.get('document_text_path', '')}`",
                    ]
                )
                if item.get("failure_reason"):
                    lines.append(f"  - failure_reason: {item.get('failure_reason')}")
            lines.append("")
        if task.get("history"):
            lines.extend(["### History", ""])
            for idx, row in enumerate(task.get("history") or [], start=1):
                lines.extend(
                    [
                        f"{idx}. `{row.get('state_sig', '')}`",
                        f"   - page: {row.get('page_summary', '')}",
                        f"   - previous_goal: {row.get('previous_goal', '')}",
                        f"   - current_goal: {row.get('current_goal', '')}",
                        f"   - progress: {row.get('progress', '')}",
                        f"   - action: {row.get('selected_action', '')}",
                        f"   - intent: {row.get('action_intent', '')}",
                    ]
                )
            lines.append("")
        for idx, step in enumerate(task.get("steps") or [], start=1):
            action = step.get("selected_action") or {}
            lines.extend(
                [
                    f"{idx}. `{step.get('state_sig', '')}` {step.get('page_kind', '')}",
                    f"   - page: {step.get('page_summary', '')}",
                    f"   - progress: {step.get('task_progress', '')}",
                    f"   - action: {action.get('action', '')}:{action.get('element_id', '')} {action.get('label', '')}",
                    f"   - intent: {action.get('action_intent', '')}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_task_report(run_dir: Path) -> Dict[str, Any]:
    """
    Input: run directory path.
    Output: generated report dictionary.
    Function: writes task_report.json and task_report.md to the run directory.
    """
    run_dir = Path(run_dir)
    report = build_task_report(run_dir)
    (run_dir / "task_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "task_report.md").write_text(render_task_report_markdown(report), encoding="utf-8")
    return report
