"""
Generate brief task-to-block coverage reports for UI exploration runs.

Input:
- One run directory via `--run-dir`, or the latest N run directories via
  `--latest` and `--trace-root`.

Output:
- `task_router_brief.md` and `task_router_brief.json` inside each run directory.

Function:
- Lists each task's origin UI, path UIs, matched questionnaire blocks hit by
  each UI, and per-task block hit counts in a compact human-readable format.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    """
    Input: JSON file path.
    Output: decoded dictionary, or an empty dictionary on missing/invalid input.
    Function: makes the offline report tolerant of partial run directories.
    """
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _load_html_alias_by_sig(run_dir: Path) -> Dict[str, str]:
    """
    Input: one trace run directory containing `index.html`.
    Output: mapping from state_sig to HTML graph alias such as `UI3`.
    Function: reads the same alias map used by the interactive UTG HTML.
    """
    html_path = Path(run_dir) / "index.html"
    if not html_path.exists():
        return {}
    text = html_path.read_text(encoding="utf-8", errors="replace")
    marker = "const payload = "
    start = text.find(marker)
    if start < 0:
        return {}
    start += len(marker)
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    alias_map = payload.get("alias_map")
    if not isinstance(alias_map, dict):
        return {}
    return {str(sig): str(alias) for sig, alias in alias_map.items() if str(sig) and str(alias)}


def _format_ui_display(html_alias: str, state_alias: str) -> str:
    """
    Input: HTML graph alias and state-directory alias.
    Output: readable UI label, e.g. `UI3 (UI000042)`.
    Function: makes offline reports align with the interactive UTG labels.
    """
    html_alias = str(html_alias or "").strip()
    state_alias = str(state_alias or "").strip()
    if html_alias and state_alias and html_alias != state_alias:
        return f"{html_alias} ({state_alias})"
    return html_alias or state_alias


def _load_llm_observation_by_sig(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Input: one trace run directory.
    Output: mapping from state_sig to page/router/block information.
    Function: loads saved navigation_router_result.json files for UI-level block hits.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for state_dir in sorted((run_dir / "states").glob("UI*")):
        path = state_dir / "llm" / "navigation_router_result.json"
        payload = _read_json(path)
        if not payload:
            continue
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        nav = result.get("navigation") if isinstance(result.get("navigation"), dict) else {}
        router = result.get("router") if isinstance(result.get("router"), dict) else {}
        sig = str(payload.get("state_sig") or result.get("state_sig") or nav.get("state_sig") or "").strip()
        if not sig:
            continue
        updates = [u for u in list(router.get("router_updates") or []) if isinstance(u, dict)]
        matched_block_ids = [
            str(block_id)
            for block_id in list(payload.get("matched_block_ids") or result.get("matched_block_ids") or [])
            if str(block_id)
        ]
        out[sig] = {
            "state_sig": sig,
            "ui_alias": state_dir.name.split("_", 1)[0],
            "page_summary": str(nav.get("page_summary") or ""),
            "page_kind": str(nav.get("page_kind") or ""),
            "matched_block_ids": matched_block_ids,
            "router_updates": updates,
            "router_question_ids": [str(u.get("question_id") or "") for u in updates if str(u.get("question_id") or "")],
            "llm_result_path": str(path),
        }
    return out


def _unique_path_steps(path_steps: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: task path rows from task_analysis_report.json.
    Output: path rows with exact duplicate consecutive UI/action rows collapsed.
    Function: keeps the brief report readable without hiding repeated non-consecutive visits.
    """
    out: List[Dict[str, Any]] = []
    last_key: Tuple[str, str] = ("", "")
    for row in path_steps:
        if not isinstance(row, dict):
            continue
        action = row.get("selected_action") if isinstance(row.get("selected_action"), dict) else {}
        key = (str(row.get("state_sig") or ""), str(action.get("candidate_key") or action.get("action_text") or ""))
        if key == last_key:
            continue
        out.append(row)
        last_key = key
    return out


def _block_summary_for_steps(path_steps: List[Dict[str, Any]], observation_by_sig: Dict[str, Dict[str, Any]]) -> Counter:
    """
    Input: task path rows and UI observation mapping.
    Output: counter of matched block ids hit across the task path.
    Function: aggregates per-UI block hits into task-level hit counts.
    """
    counts: Counter = Counter()
    for row in path_steps:
        sig = str(row.get("state_sig") or "")
        observation = observation_by_sig.get(sig) or {}
        counts.update([block_id for block_id in list(observation.get("matched_block_ids") or []) if block_id])
    return counts


def _brief_task(
    task: Dict[str, Any],
    observation_by_sig: Dict[str, Dict[str, Any]],
    html_alias_by_sig: Dict[str, str],
) -> Dict[str, Any]:
    """
    Input: one task row from task_analysis_report.json and UI observation mapping.
    Output: compact task-block coverage dictionary.
    Function: extracts origin, path UI hits, and task-level matched-block totals.
    """
    origin = task.get("origin") if isinstance(task.get("origin"), dict) else {}
    origin_sig = str(origin.get("state_sig") or "")
    origin_state_alias = str(origin.get("ui_alias") or "")
    origin_html_alias = str(html_alias_by_sig.get(origin_sig) or "")
    path_steps = _unique_path_steps(task.get("path") if isinstance(task.get("path"), list) else [])
    ui_rows: List[Dict[str, Any]] = []
    for idx, step in enumerate(path_steps, start=1):
        sig = str(step.get("state_sig") or "")
        page = step.get("page") if isinstance(step.get("page"), dict) else {}
        action = step.get("selected_action") if isinstance(step.get("selected_action"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        observation = observation_by_sig.get(sig) or {}
        block_ids = list(observation.get("matched_block_ids") or [])
        state_alias = str(step.get("ui_alias") or observation.get("ui_alias") or "")
        html_alias = str(html_alias_by_sig.get(sig) or "")
        ui_rows.append(
            {
                "index": idx,
                "ui_alias": state_alias,
                "html_ui_alias": html_alias,
                "ui_display": _format_ui_display(html_alias, state_alias),
                "state_sig": sig,
                "page_summary": str(page.get("page_summary") or observation.get("page_summary") or ""),
                "selected_action": str(action.get("action_text") or ""),
                "dst_ui_alias": str(result.get("dst_ui_alias") or ""),
                "dst_sig": str(result.get("dst_sig") or ""),
                "matched_block_count": len(block_ids),
                "matched_block_ids": block_ids,
                "router_updates": list(observation.get("router_updates") or []),
            }
        )
    counts = _block_summary_for_steps(path_steps, observation_by_sig)
    return {
        "task_id": str(task.get("task_id") or ""),
        "task_type": str(task.get("task_type") or ""),
        "status": str(task.get("status") or ""),
        "prompt": str(task.get("prompt") or ""),
        "origin_ui_alias": origin_state_alias,
        "origin_html_ui_alias": origin_html_alias,
        "origin_ui_display": _format_ui_display(origin_html_alias, origin_state_alias),
        "origin_state_sig": origin_sig,
        "origin_page": str(((origin.get("page") or {}) if isinstance(origin.get("page"), dict) else {}).get("page_summary") or ""),
        "entry_action": str(origin.get("entry_action_text") or ""),
        "path_ui_count": len(ui_rows),
        "path_unique_ui_count": len({row["state_sig"] for row in ui_rows if row["state_sig"]}),
        "finish_reason": str(task.get("finish_reason") or ""),
        "used_steps": task.get("used_steps"),
        "step_budget": task.get("step_budget"),
        "matched_block_count": int(sum(counts.values())),
        "unique_matched_block_count": len(counts),
        "matched_block_counts": dict(sorted(counts.items())),
        "ui_hits": ui_rows,
    }


def build_brief_report(run_dir: Path) -> Dict[str, Any]:
    """
    Input: one trace run directory.
    Output: compact task-router report dictionary.
    Function: combines task_analysis_report.json with saved LLM block-match outputs.
    """
    run_dir = Path(run_dir)
    analysis_path = run_dir / "task_analysis_report.json"
    analysis = _read_json(analysis_path)
    if not analysis:
        raise FileNotFoundError(f"Missing or invalid task_analysis_report.json: {analysis_path}")
    observation_by_sig = _load_llm_observation_by_sig(run_dir)
    html_alias_by_sig = _load_html_alias_by_sig(run_dir)
    tasks = [
        _brief_task(t, observation_by_sig, html_alias_by_sig)
        for t in list(analysis.get("tasks") or [])
        if isinstance(t, dict)
    ]
    type_counts: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_type = str(task.get("task_type") or "")
        row = type_counts.setdefault(
            task_type,
            {
                "task_type": task_type,
                "task_count": 0,
                "matched_block_count": 0,
                "unique_matched_blocks": set(),
            },
        )
        row["task_count"] += 1
        row["matched_block_count"] += int(task.get("matched_block_count") or 0)
        row["unique_matched_blocks"].update(task.get("matched_block_counts", {}).keys())
    type_summary = []
    for row in type_counts.values():
        type_summary.append(
            {
                "task_type": row["task_type"],
                "task_count": row["task_count"],
                "matched_block_count": row["matched_block_count"],
                "unique_matched_block_count": len(row["unique_matched_blocks"]),
                "matched_blocks": sorted(row["unique_matched_blocks"]),
            }
        )
    type_summary.sort(key=lambda x: (str(x["task_type"])))
    return {
        "run_id": str(analysis.get("run_id") or run_dir.name),
        "run_dir": str(run_dir),
        "task_count": len(tasks),
        "type_summary": type_summary,
        "tasks": tasks,
    }


def _join_block_ids(ids: Sequence[str]) -> str:
    """
    Input: matched block id sequence.
    Output: comma-separated text, or `none`.
    Function: keeps markdown concise for UI hit rows.
    """
    clean = [str(x) for x in ids if str(x)]
    return ", ".join(clean) if clean else "none"


def render_brief_markdown(report: Dict[str, Any]) -> str:
    """
    Input: compact task-block report dictionary.
    Output: markdown text.
    Function: renders a simple task -> UI -> matched-block hit report.
    """
    lines = [
        "# Task Block Brief Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- task_count: `{report.get('task_count', 0)}`",
        "",
        "## Task Type Summary",
        "",
        "| task_type | tasks | matched_blocks | unique_blocks |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("type_summary") or []:
        lines.append(
            f"| `{row.get('task_type', '')}` | {row.get('task_count', 0)} | "
            f"{row.get('matched_block_count', 0)} | {row.get('unique_matched_block_count', 0)} |"
        )
    lines.extend(["", "## Tasks", ""])
    for task in report.get("tasks") or []:
        lines.extend(
            [
                f"### {task.get('task_id')} / {task.get('task_type')}",
                "",
                f"- status: `{task.get('status', '')}`",
                f"- origin: `{task.get('origin_ui_display') or task.get('origin_ui_alias', '')}` / {task.get('origin_page', '')}",
                f"- entry_action: {task.get('entry_action', '')}",
                f"- path_ui_count: `{task.get('path_ui_count', 0)}` unique `{task.get('path_unique_ui_count', 0)}`",
                f"- used_steps: `{task.get('used_steps', '')}/{task.get('step_budget', '')}`",
                f"- finish_reason: {task.get('finish_reason', '')}",
                "",
                "#### UI Block Hits",
                "",
            ]
        )
        ui_hits = list(task.get("ui_hits") or [])
        if not ui_hits:
            lines.extend(["- no path UI recorded", ""])
        for row in ui_hits:
            lines.extend(
                [
                    f"{row.get('index')}. `{row.get('ui_display') or row.get('ui_alias', '')}` / {row.get('page_summary', '')}",
                    f"   - action: {row.get('selected_action', '')}",
                    f"   - matched_blocks: {_join_block_ids(row.get('matched_block_ids') or [])}",
                ]
            )
        lines.extend(["", "#### Task Block Summary", ""])
        counts = task.get("matched_block_counts") if isinstance(task.get("matched_block_counts"), dict) else {}
        if not counts:
            lines.extend(["- none", ""])
            continue
        lines.extend(["| matched_block | hit_count |", "|---|---:|"])
        for block_id, count in counts.items():
            lines.append(f"| `{block_id}` | {count} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_brief_report(run_dir: Path) -> Dict[str, Any]:
    """
    Input: one trace run directory.
    Output: compact report dictionary.
    Function: writes task_router_brief.json and task_router_brief.md.
    """
    run_dir = Path(run_dir)
    report = build_brief_report(run_dir)
    (run_dir / "task_router_brief.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "task_router_brief.md").write_text(render_brief_markdown(report), encoding="utf-8")
    return report


def _latest_run_dirs(trace_root: Path, count: int) -> List[Path]:
    """
    Input: trace root and number of runs.
    Output: latest run directories by LastWriteTime.
    Function: supports quick batch reporting for recent runs.
    """
    dirs = [p for p in Path(trace_root).iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[: max(0, int(count))]


def main() -> None:
    """
    Input: command-line arguments.
    Output: generated report paths printed to stdout.
    Function: CLI entry point for single-run or latest-N brief report generation.
    """
    parser = argparse.ArgumentParser(description="Generate brief task-router coverage reports.")
    parser.add_argument("--run-dir", default="", help="Single trace run directory.")
    parser.add_argument("--trace-root", default="traces", help="Trace root used with --latest.")
    parser.add_argument("--latest", type=int, default=0, help="Generate reports for latest N run directories.")
    args = parser.parse_args()

    run_dirs: List[Path] = []
    if args.run_dir:
        run_dirs.append(Path(args.run_dir).resolve())
    if args.latest:
        run_dirs.extend(_latest_run_dirs(Path(args.trace_root).resolve(), int(args.latest)))

    seen = set()
    for run_dir in run_dirs:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        report = write_brief_report(run_dir)
        print(run_dir / "task_router_brief.md")
        print(run_dir / "task_router_brief.json")
        print(f"task_count={report.get('task_count', 0)}")


if __name__ == "__main__":
    main()
