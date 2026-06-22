"""
Offline task analysis report generation for one UI exploration run.

Input:
- A run directory containing `tasks.json`, `trace.jsonl`, and
  `states/<UI...>/llm/navigation_router_result.json` files.

Output:
- `task_analysis_report.json` and `task_analysis_report.md` under the same run
  directory.

Function:
- Reconstructs each task from its origin UI through the UI states it visited,
  using saved LLM results for page/progress/action intent and trace events for
  the actually executed path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    """
    Input: JSON file path.
    Output: decoded JSON object, or an empty dict on missing/invalid input.
    Function: provides tolerant reads for partially generated trace runs.
    """
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Input: JSONL file path.
    Output: list of decoded JSON object lines.
    Function: reads trace.jsonl while skipping malformed rows.
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


def _payload(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: raw trace event.
    Output: event business payload.
    Function: supports real trace rows where payload lives under `data`.
    """
    data = event.get("data")
    if isinstance(data, dict):
        return data
    return event


def _task_from_ctx(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: raw trace event.
    Output: current task object from the event context, or an empty dict.
    Function: recovers task_id for action/transition rows that do not store it
    directly inside data.
    """
    ctx = event.get("ctx")
    if not isinstance(ctx, dict):
        return {}
    task_ctx = ctx.get("task")
    if not isinstance(task_ctx, dict):
        return {}
    current = task_ctx.get("current_task")
    return current if isinstance(current, dict) else {}


def _task_id_from_event(event: Dict[str, Any]) -> str:
    """
    Input: raw trace event.
    Output: normalized current task id.
    Function: finds task_id from data first, then trace context.
    """
    data = _payload(event)
    task_id = str(data.get("task_id") or "").strip()
    if task_id:
        return task_id
    return str(_task_from_ctx(event).get("task_id") or "").strip()


def _state_sig_from_event(event: Dict[str, Any]) -> str:
    """
    Input: raw trace event.
    Output: normalized source/current state signature.
    Function: supports state lookup for action and transition rows.
    """
    data = _payload(event)
    if data.get("state_sig"):
        return str(data.get("state_sig") or "")
    if data.get("sig"):
        return str(data.get("sig") or "")
    ctx = event.get("ctx")
    if isinstance(ctx, dict):
        return str(ctx.get("cur_sig") or "")
    return ""


def _first_action(action_like: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: ActionCandidate-like or ActionStep-like dictionary.
    Output: first ActionStep-like dictionary.
    Function: lets matching code compare entry actions and candidate actions.
    """
    actions = action_like.get("actions")
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        return actions[0]
    return action_like if isinstance(action_like, dict) else {}


def _action_match_key(action_like: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Input: ActionStep-like or ActionCandidate-like dictionary.
    Output: loose comparable key `(action, element_id, text, label)`.
    Function: matches tasks.json entry_action to LLM candidate_actions.
    """
    step = _first_action(action_like)
    return (
        str(step.get("action") or "").split(".")[-1].lower(),
        str(step.get("element_id") if step.get("element_id") is not None else ""),
        str(step.get("text") or ""),
        str(step.get("anchor_label") or step.get("label") or ""),
    )


def _format_action(action_like: Dict[str, Any]) -> str:
    """
    Input: ActionStep-like or selected-action dictionary.
    Output: compact human-readable action string.
    Function: keeps markdown action formatting consistent.
    """
    step = _first_action(action_like)
    action = str(step.get("action") or "")
    element = step.get("element_id")
    label = str(step.get("anchor_label") or step.get("label") or step.get("text") or "")
    if element is None or element == "":
        return f"{action} {label}".strip()
    return f"{action}:{element} {label}".strip()


def _load_llm_results(run_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Input: run directory.
    Output: `(state_sig -> LLM result, state_sig -> UI alias)`.
    Function: loads saved per-UI navigation_router_result.json files.
    """
    by_sig: Dict[str, Dict[str, Any]] = {}
    alias: Dict[str, str] = {}
    states_dir = run_dir / "states"
    for state_dir in sorted(states_dir.glob("UI*")):
        path = state_dir / "llm" / "navigation_router_result.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        sig = str(payload.get("state_sig") or result.get("state_sig") or "").strip()
        if not sig:
            nav = result.get("navigation") if isinstance(result.get("navigation"), dict) else {}
            sig = str(nav.get("state_sig") or "").strip()
        if not sig:
            continue
        by_sig[sig] = {
            "state_sig": sig,
            "ui_alias": state_dir.name.split("_", 1)[0],
            "path": str(path),
            "result": result,
        }
        alias[sig] = state_dir.name.split("_", 1)[0]
    return by_sig, alias


def _candidate_key(candidate: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Input: LLM candidate action dictionary.
    Output: loose action match key.
    Function: wraps _action_match_key for readability in candidate lookups.
    """
    return _action_match_key(candidate)


def _find_candidate_by_action(llm_result: Dict[str, Any], action_like: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: one LLM result and an action dictionary.
    Output: matching candidate action dictionary, or empty dict.
    Function: recovers action_intent for an entry/executed action from saved LLM output.
    """
    target = _action_match_key(action_like)
    navigation = llm_result.get("navigation") if isinstance(llm_result.get("navigation"), dict) else {}
    for candidate in list(navigation.get("candidate_actions") or []):
        if isinstance(candidate, dict) and _candidate_key(candidate) == target:
            return candidate
    return {}


def _llm_page_summary(llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: saved LLM result for one UI.
    Output: page/progress/decision summary dictionary.
    Function: extracts the page analysis fields used by both origin and path steps.
    """
    navigation = llm_result.get("navigation") if isinstance(llm_result.get("navigation"), dict) else {}
    return {
        "analysis_task_id": str(llm_result.get("task_id") or ""),
        "page_summary": str(navigation.get("page_summary") or ""),
        "page_kind": str(navigation.get("page_kind") or ""),
        "page_tags": list(navigation.get("page_tags") or []),
        "task_progress": str(llm_result.get("task_progress") or ""),
        "task_decision": llm_result.get("task_decision") if isinstance(llm_result.get("task_decision"), dict) else {},
        "llm_result_path": "",
    }


def _index_executed_actions(trace_events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Input: raw trace events.
    Output: task_id -> selected action rows in execution order.
    Function: captures the actual task path from task_action_selected rows.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for event in trace_events:
        data = _payload(event)
        if data.get("kind") != "task_action_selected":
            continue
        task_id = _task_id_from_event(event)
        if not task_id:
            continue
        out.setdefault(task_id, []).append(
            {
                "task_id": task_id,
                "state_sig": _state_sig_from_event(event),
                "candidate_key": str(data.get("candidate_key") or ""),
                "action_role": str(data.get("action_role") or ""),
                "starts_task_type": str(data.get("starts_task_type") or ""),
                "starts_task_depth": str(data.get("starts_task_depth") or ""),
                "score": data.get("score"),
                "action_intent": str(data.get("action_intent") or ""),
                "action": str(data.get("action") or ""),
                "element_id": data.get("element_id"),
                "label": str(data.get("label") or ""),
                "reasoning": str(data.get("reasoning") or ""),
                "step_id": (event.get("ctx") or {}).get("step_id") if isinstance(event.get("ctx"), dict) else None,
            }
        )
    return out


def _index_action_results(trace_events: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Input: raw trace events.
    Output: `(state_sig, action_key-ish) -> result summary`.
    Function: best-effort mapping from selected actions to the next transition target.
    """
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pending: Optional[Dict[str, Any]] = None
    for event in trace_events:
        data = _payload(event)
        event_name = str(event.get("event") or "")
        if data.get("kind") == "task_action_selected":
            pending = {
                "task_id": _task_id_from_event(event),
                "state_sig": _state_sig_from_event(event),
                "action": str(data.get("action") or ""),
                "element_id": data.get("element_id"),
                "label": str(data.get("label") or ""),
                "candidate_key": str(data.get("candidate_key") or ""),
            }
            continue
        if pending and event_name == "transition":
            kind = str(data.get("kind") or "")
            src = str(data.get("src") or data.get("state_sig") or "")
            dst = str(data.get("dst") or data.get("sig") or "")
            if kind == "transition" and (not src or src == pending.get("state_sig")):
                key = (str(pending.get("state_sig") or ""), str(pending.get("candidate_key") or ""))
                results[key] = {"dst_sig": dst, "transition_kind": kind}
                pending = None
    return results


def _index_observations(trace_events: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Input: raw trace events.
    Output: observation indexes keyed by `(task_id, state_sig)` and by `state_sig`.
    Function: recovers task_progress when saved LLM JSON lacks root-level fields.
    """
    by_task_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_state: Dict[str, Dict[str, Any]] = {}
    for event in trace_events:
        data = _payload(event)
        if data.get("kind") != "task_ui_observation":
            continue
        task_id = _task_id_from_event(event)
        state_sig = _state_sig_from_event(event)
        if not state_sig:
            continue
        row = {
            "task_id": task_id,
            "state_sig": state_sig,
            "page_summary": str(data.get("page_summary") or ""),
            "page_kind": str(data.get("page_kind") or ""),
            "page_tags": list(data.get("page_tags") or []),
            "task_progress": str(data.get("task_progress") or ""),
            "task_decision": data.get("task_decision") if isinstance(data.get("task_decision"), dict) else {},
        }
        if task_id:
            by_task_state[(task_id, state_sig)] = row
        by_state[state_sig] = row
    return by_task_state, by_state


def _merge_observation_page(
    page: Dict[str, Any],
    *,
    state_sig: str,
    task_id: str,
    obs_by_task_state: Dict[Tuple[str, str], Dict[str, Any]],
    obs_by_state: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Input: page summary from LLM JSON plus observation indexes.
    Output: page summary with missing progress/decision filled from trace observations.
    Function: keeps the report complete when LLM result files omit root-level task_progress.
    """
    merged = dict(page)
    source = "llm_result"
    obs = obs_by_task_state.get((task_id, state_sig))
    if obs is None:
        obs = obs_by_state.get(state_sig)
        if obs is not None:
            source = "same_state_observation"
    else:
        source = "task_observation"
    if obs:
        if not merged.get("page_summary"):
            merged["page_summary"] = obs.get("page_summary", "")
        if not merged.get("page_kind"):
            merged["page_kind"] = obs.get("page_kind", "")
        if not merged.get("page_tags"):
            merged["page_tags"] = list(obs.get("page_tags") or [])
        if not merged.get("task_progress"):
            merged["task_progress"] = obs.get("task_progress", "")
        if not merged.get("task_decision"):
            merged["task_decision"] = obs.get("task_decision") or {}
        merged["observation_source"] = source
        merged["observation_task_id"] = obs.get("task_id", "")
    else:
        merged["observation_source"] = ""
        merged["observation_task_id"] = ""
    return merged


def _origin_from_task(
    task: Dict[str, Any],
    llm_by_sig: Dict[str, Dict[str, Any]],
    obs_by_task_state: Dict[Tuple[str, str], Dict[str, Any]],
    obs_by_state: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Input: one task row and loaded LLM results.
    Output: origin section for the task report.
    Function: explains where and why the task was created.
    """
    origin_sig = str(task.get("origin_state_sig") or "")
    llm_row = llm_by_sig.get(origin_sig) or {}
    llm_result = llm_row.get("result") if isinstance(llm_row.get("result"), dict) else {}
    candidate = _find_candidate_by_action(llm_result, task.get("entry_action") if isinstance(task.get("entry_action"), dict) else {})
    proposed_match = _find_proposed_task(llm_result, task)
    page = _llm_page_summary(llm_result)
    page["llm_result_path"] = str(llm_row.get("path") or "")
    page = _merge_observation_page(
        page,
        state_sig=origin_sig,
        task_id=str(task.get("parent_task_id") or task.get("task_id") or ""),
        obs_by_task_state=obs_by_task_state,
        obs_by_state=obs_by_state,
    )
    return {
        "state_sig": origin_sig,
        "ui_alias": str(llm_row.get("ui_alias") or ""),
        "page": page,
        "parent_task_id": str(task.get("parent_task_id") or ""),
        "entry_action": task.get("entry_action") or {},
        "entry_action_text": _format_action(task.get("entry_action") if isinstance(task.get("entry_action"), dict) else {}),
        "entry_intent": str(candidate.get("action_intent") or proposed_match.get("reason") or task.get("notes") or ""),
        "proposed_reason": str(proposed_match.get("reason") or task.get("notes") or ""),
    }


def _find_proposed_task(llm_result: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: origin LLM result and task row from tasks.json.
    Output: matching proposed task dictionary, or empty dict.
    Function: recovers proposed task reason/goal for the origin section.
    """
    target_type = str(task.get("task_type") or "")
    target_goal = str(task.get("initial_goal") or task.get("prompt") or "")
    target_action_key = _action_match_key(task.get("entry_action") if isinstance(task.get("entry_action"), dict) else {})
    for proposed in list(llm_result.get("proposed_tasks") or []):
        if not isinstance(proposed, dict):
            continue
        entry = proposed.get("entry_action") if isinstance(proposed.get("entry_action"), dict) else {}
        if _action_match_key(entry) == target_action_key:
            return proposed
        proposed_goal = str(proposed.get("initial_goal") or proposed.get("prompt") or "")
        if target_type and str(proposed.get("task_type") or "") == target_type and proposed_goal == target_goal:
            return proposed
    return {}


def _step_from_action(
    action_row: Dict[str, Any],
    llm_by_sig: Dict[str, Dict[str, Any]],
    result_by_action: Dict[Tuple[str, str], Dict[str, Any]],
    alias_by_sig: Dict[str, str],
    obs_by_task_state: Dict[Tuple[str, str], Dict[str, Any]],
    obs_by_state: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Input: executed action row plus LLM/result indexes.
    Output: one task path step.
    Function: combines saved LLM page analysis with the action actually selected.
    """
    state_sig = str(action_row.get("state_sig") or "")
    llm_row = llm_by_sig.get(state_sig) or {}
    llm_result = llm_row.get("result") if isinstance(llm_row.get("result"), dict) else {}
    page = _llm_page_summary(llm_result)
    page["llm_result_path"] = str(llm_row.get("path") or "")
    page = _merge_observation_page(
        page,
        state_sig=state_sig,
        task_id=str(action_row.get("task_id") or ""),
        obs_by_task_state=obs_by_task_state,
        obs_by_state=obs_by_state,
    )
    candidate = _find_candidate_by_action(
        llm_result,
        {
            "action": action_row.get("action"),
            "element_id": action_row.get("element_id"),
            "anchor_label": action_row.get("label"),
        },
    )
    action_key = str(action_row.get("candidate_key") or "")
    result = result_by_action.get((state_sig, action_key)) or {}
    dst_sig = str(result.get("dst_sig") or "")
    return {
        "state_sig": state_sig,
        "ui_alias": str(llm_row.get("ui_alias") or alias_by_sig.get(state_sig) or ""),
        "page": page,
        "page_source": "same_state_llm_result" if page.get("analysis_task_id") and page.get("analysis_task_id") != action_row.get("task_id") else "task_llm_result",
        "selected_action": {
            **action_row,
            "action_text": _format_action(
                {
                    "action": action_row.get("action"),
                    "element_id": action_row.get("element_id"),
                    "anchor_label": action_row.get("label"),
                }
            ),
            "action_intent": str(action_row.get("action_intent") or candidate.get("action_intent") or ""),
            "candidate_llm_score": candidate.get("score"),
            "candidate_reasoning": (_first_action(candidate).get("reasoning") if candidate else ""),
        },
        "result": {
            "dst_sig": dst_sig,
            "dst_ui_alias": alias_by_sig.get(dst_sig, ""),
            "transition_kind": result.get("transition_kind", ""),
        },
    }


def build_task_analysis_report(run_dir: Path) -> Dict[str, Any]:
    """
    Input: run directory path.
    Output: task analysis report dictionary.
    Function: builds an offline task-centered report from saved LLM JSON and trace execution.
    """
    run_dir = Path(run_dir)
    tasks_payload = _read_json(run_dir / "tasks.json")
    trace_events = _read_jsonl(run_dir / "trace.jsonl")
    llm_by_sig, alias_by_sig = _load_llm_results(run_dir)
    actions_by_task = _index_executed_actions(trace_events)
    result_by_action = _index_action_results(trace_events)
    obs_by_task_state, obs_by_state = _index_observations(trace_events)

    task_reports: List[Dict[str, Any]] = []
    for task in list(tasks_payload.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        actions = actions_by_task.get(task_id, [])
        initial_goal = str(task.get("initial_goal") or task.get("prompt") or "")
        task_reports.append(
            {
                "task_id": task_id,
                "task_type": str(task.get("task_type") or ""),
                "status": str(task.get("status") or ""),
                "initial_goal": initial_goal,
                "current_goal": str(task.get("current_goal") or initial_goal),
                "progress_summary": str(task.get("progress_summary") or ""),
                "priority": task.get("priority"),
                "type_priority": task.get("type_priority"),
                "llm_priority": task.get("llm_priority"),
                "exploration_depth": str(task.get("exploration_depth") or ""),
                "used_steps": task.get("used_steps"),
                "step_budget": task.get("step_budget"),
                "finish_reason": str(task.get("finish_reason") or ""),
                "origin": _origin_from_task(task, llm_by_sig, obs_by_task_state, obs_by_state),
                "path": [
                    _step_from_action(row, llm_by_sig, result_by_action, alias_by_sig, obs_by_task_state, obs_by_state)
                    for row in actions
                ],
            }
        )

    return {
        "run_id": str(tasks_payload.get("run_id") or run_dir.name),
        "run_dir": str(run_dir),
        "task_count": len(task_reports),
        "tasks": task_reports,
    }


def _md_value(value: Any) -> str:
    """
    Input: arbitrary value.
    Output: markdown-safe-ish one-line string.
    Function: avoids printing Python None values as meaningful content.
    """
    if value is None:
        return ""
    return str(value)


def render_task_analysis_markdown(report: Dict[str, Any]) -> str:
    """
    Input: task analysis report dictionary.
    Output: markdown text.
    Function: renders each task as origin, path, and end sections.
    """
    lines = [
        "# Task Analysis Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- task_count: `{report.get('task_count', 0)}`",
        "",
    ]
    for task in report.get("tasks") or []:
        origin = task.get("origin") or {}
        origin_page = origin.get("page") or {}
        lines.extend(
            [
                f"## {task.get('task_id')} / {task.get('task_type')}",
                "",
                f"- status: `{task.get('status', '')}`",
                f"- initial_goal: {task.get('initial_goal', '')}",
                f"- current_goal: {task.get('current_goal', '')}",
                f"- progress_summary: {task.get('progress_summary', '')}",
                f"- priority: `{task.get('priority', '')}` / type `{task.get('type_priority', '')}` / llm `{task.get('llm_priority', '')}`",
                f"- depth: `{task.get('exploration_depth', '')}`",
                f"- steps: `{task.get('used_steps', '')}/{task.get('step_budget', '')}`",
                "",
                "### Origin",
                "",
                f"- origin_ui: `{origin.get('ui_alias', '')}` / `{origin.get('state_sig', '')}`",
                f"- parent_task: `{origin.get('parent_task_id', '')}`",
                f"- page: {origin_page.get('page_summary', '')}",
                f"- analysis_task_id: `{origin_page.get('analysis_task_id', '')}`",
                f"- proposed_reason: {origin.get('proposed_reason', '')}",
                f"- entry_action: {_md_value(origin.get('entry_action_text'))}",
                f"- entry_intent: {origin.get('entry_intent', '')}",
                "",
                "### Path",
                "",
            ]
        )
        path = list(task.get("path") or [])
        if not path:
            lines.append("- no executed task actions recorded")
            lines.append("")
        for idx, step in enumerate(path, start=1):
            page = step.get("page") or {}
            action = step.get("selected_action") or {}
            result = step.get("result") or {}
            lines.extend(
                [
                    f"{idx}. `{step.get('ui_alias', '')}` / `{step.get('state_sig', '')}`",
                    f"   - page: {page.get('page_summary', '')}",
                    f"   - progress: {page.get('task_progress', '')}",
                    f"   - page_source: `{step.get('page_source', '')}` / analysis_task_id `{page.get('analysis_task_id', '')}`",
                    f"   - action: {_md_value(action.get('action_text'))}",
                    f"   - role: `{action.get('action_role', '')}` starts `{action.get('starts_task_type', '')}` depth `{action.get('starts_task_depth', '')}`",
                    f"   - intent: {action.get('action_intent', '')}",
                    f"   - result: `{result.get('dst_ui_alias', '')}` / `{result.get('dst_sig', '')}`",
                ]
            )
        lines.extend(
            [
                "",
                "### End",
                "",
                f"- status: `{task.get('status', '')}`",
                f"- reason: {task.get('finish_reason', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_task_analysis_report(run_dir: Path) -> Dict[str, Any]:
    """
    Input: run directory path.
    Output: generated report dictionary.
    Function: writes task_analysis_report.json and task_analysis_report.md.
    """
    run_dir = Path(run_dir)
    report = build_task_analysis_report(run_dir)
    (run_dir / "task_analysis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "task_analysis_report.md").write_text(render_task_analysis_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    """
    Input: command-line arguments.
    Output: report file paths printed to stdout.
    Function: CLI entry point for offline task analysis report generation.
    """
    parser = argparse.ArgumentParser(description="Generate an offline task analysis report for one trace run.")
    parser.add_argument("--run-dir", required=True, help="Trace run directory containing tasks.json and states/.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    report = write_task_analysis_report(run_dir)
    print(run_dir / "task_analysis_report.md")
    print(run_dir / "task_analysis_report.json")
    print(f"task_count={report.get('task_count', 0)}")


if __name__ == "__main__":
    main()
