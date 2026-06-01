#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline UTG return-hints evaluation for saved trace runs.

Input:
- A trace directory that already contains graph/state_graph_snapshot.json.
- A parent UI state signature.
- A child UI state signature reached by a real forward edge.
- Optional --call-llm to ask the configured OpenAI-compatible model.

Output:
- input.json records script parameters, resolved UI aliases, and the forward edge.
- home_paths.md/json list each UI alias and its path from home.
- utg.txt is the text-only UTG prompt context.
- prompt.md is the return-hints question sent to the LLM.
- result.json/md contain either the LLM result or a skipped-call marker.

Function:
- Lets us test whether a model can use a saved UTG, parent UI, and child UI
  to propose a small set of predicted return actions for the child state.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_config import load_project_env
from gpt_cls import GPTClient, _b64_image_url, _compact_digest


class ReturnActionHint(BaseModel):
    """One predicted return action from a child UI to home, parent, or ancestor."""

    target_type: str = Field("", description="Return target type: home, parent, ancestor, or other.")
    target_alias: str = Field("", description="Target UI alias such as UI2.")
    target_state_sig: str = Field("", description="Target UI state signature.")
    action_type: str = Field("", description="Action type such as click, back, none, or unknown.")
    element_id: Optional[int] = Field(None, description="UI element id on child UI when identifiable.")
    action_summary: str = Field("", description="Short action summary, e.g. click Home tab.")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in the recommendation.")
    evidence: List[str] = Field(default_factory=list, description="UTG and visual evidence supporting this hint.")


class UtgReturnHintEvalResult(BaseModel):
    """Structured LLM result for one forward-edge return-hints evaluation."""

    parent_ui: str = Field("", description="Parent UI alias or state signature.")
    child_ui: str = Field("", description="Child/current UI alias or state signature.")
    no_return_needed: bool = Field(False, description="True when this forward edge should not receive return hints.")
    reason: str = Field("", description="Short reason for no_return_needed or overall return-hint strategy.")
    return_actions: List[ReturnActionHint] = Field(default_factory=list, description="Predicted return actions for the child UI.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments and return the script configuration."""

    load_project_env(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Evaluate UTG-based return hint reasoning on a saved trace.")
    parser.add_argument("--trace-dir", required=True, help="Saved trace directory.")
    parser.add_argument("--parent-sig", required=True, help="Parent/source UI state signature.")
    parser.add_argument("--child-sig", required=True, help="Child/current UI state signature reached from parent.")
    parser.add_argument("--home-sig", default="", help="Home UI state signature. Defaults to --parent-sig.")
    parser.add_argument("--mode", default="text_only", choices=["text_only"], help="UTG input mode for this first version.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated evaluation artifacts.")
    parser.add_argument("--call-llm", action="store_true", help="Actually call the configured LLM.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gemini-2.5-flash"), help="Model name for --call-llm.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature for --call-llm.")
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout seconds for --call-llm.")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk and raise a clear error when it is missing."""

    if not path.exists():
        raise FileNotFoundError(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write JSON data with UTF-8 encoding and stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def short_sig(sig: str, width: int = 8) -> str:
    """Return a compact state signature for readable text output."""

    if ":" in sig:
        prefix, value = sig.split(":", 1)
        return f"{prefix}:{value[:width]}"
    return sig[:width]


def make_aliases(graph: Dict[str, Any]) -> Dict[str, str]:
    """Create UI aliases that match the interactive HTML ordering when graph node order is unchanged."""

    nodes = graph.get("nodes") or {}
    return {sig: f"UI{idx}" for idx, sig in enumerate(nodes.keys(), start=1)}


def find_state_dir(trace_dir: Path, sig: str) -> Optional[Path]:
    """Find the trace states/<UI...> directory corresponding to a state signature."""

    states_dir = trace_dir / "states"
    if not states_dir.exists():
        return None
    token = sig.split(":", 1)[-1]
    for child in states_dir.iterdir():
        if child.is_dir() and token in child.name:
            return child
    return None


def load_page_summary(trace_dir: Path, sig: str) -> str:
    """Load NavigationProposal.page_summary for a state when a saved LLM result exists."""

    state_dir = find_state_dir(trace_dir, sig)
    if not state_dir:
        return ""
    result_path = state_dir / "llm" / "navigation_router_result.json"
    if not result_path.exists():
        return ""
    data = read_json(result_path)
    body = data.get("result") if isinstance(data.get("result"), dict) else data
    nav = body.get("navigation") if isinstance(body.get("navigation"), dict) else body
    return str((nav or {}).get("page_summary") or "").strip()


def edge_action_summary(edge: Dict[str, Any]) -> str:
    """Summarize a graph edge action into a short human-readable path segment."""

    action_payload = edge.get("action") or {}
    steps = action_payload.get("actions") or []
    labels: List[str] = []
    for step in steps:
        action = str(step.get("action") or "").replace("ActionType.", "").lower()
        label = str(step.get("anchor_label") or step.get("text") or "").strip()
        if label:
            labels.append(f"{action} {label}".strip())
        elif action:
            labels.append(action)
    if labels:
        return " + ".join(labels)
    return "unknown action"


def find_forward_edge(graph: Dict[str, Any], parent_sig: str, child_sig: str) -> Optional[Dict[str, Any]]:
    """Find the first saved graph edge from parent UI to child UI."""

    for edge in graph.get("edges") or []:
        if str(edge.get("src") or "") == parent_sig and str(edge.get("dst") or "") == child_sig:
            return edge
    return None


def build_adjacency(graph: Dict[str, Any]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """Build a directed adjacency list from graph edges."""

    adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for edge in graph.get("edges") or []:
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if not src or not dst:
            continue
        adjacency.setdefault(src, []).append((dst, edge))
    return adjacency


def shortest_paths_from_home(graph: Dict[str, Any], home_sig: str) -> Dict[str, List[Dict[str, str]]]:
    """Compute one shortest edge path from home to every reachable UI."""

    adjacency = build_adjacency(graph)
    paths: Dict[str, List[Dict[str, str]]] = {home_sig: []}
    queue: deque[str] = deque([home_sig])
    while queue:
        src = queue.popleft()
        for dst, edge in adjacency.get(src, []):
            if dst in paths:
                continue
            paths[dst] = paths[src] + [{"from": src, "to": dst, "action": edge_action_summary(edge)}]
            queue.append(dst)
    return paths


def path_to_text(sig: str, aliases: Dict[str, str], paths: Dict[str, List[Dict[str, str]]], home_sig: str) -> str:
    """Render one UI's path from home as a compact string."""

    if sig == home_sig:
        return "home"
    if sig not in paths:
        return "not reachable from home"
    parts = ["home"]
    for step in paths[sig]:
        parts.append(step["action"])
    return " > ".join(parts)


def build_home_path_rows(graph: Dict[str, Any], aliases: Dict[str, str], home_sig: str) -> List[Dict[str, str]]:
    """Build home path rows for all graph nodes in alias order."""

    paths = shortest_paths_from_home(graph, home_sig)
    rows: List[Dict[str, str]] = []
    for sig in (graph.get("nodes") or {}).keys():
        rows.append(
            {
                "alias": aliases.get(sig, short_sig(sig)),
                "state_sig": sig,
                "short_sig": short_sig(sig),
                "home_path": path_to_text(sig, aliases, paths, home_sig),
            }
        )
    return rows


def render_home_paths_md(rows: Iterable[Dict[str, str]]) -> str:
    """Render all home path rows as a markdown table for manual review."""

    lines = ["# Home Paths", "", "| UI | state_sig | home_path |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['alias']} | `{row['state_sig']}` | {row['home_path']} |")
    lines.append("")
    return "\n".join(lines)


def render_utg_text(
    trace_dir: Path,
    graph: Dict[str, Any],
    aliases: Dict[str, str],
    home_sig: str,
    parent_sig: str,
    child_sig: str,
    forward_edge: Optional[Dict[str, Any]],
    home_rows: List[Dict[str, str]],
) -> str:
    """Render a text-only UTG with nodes, edges, home paths, and return-hints context."""

    lines: List[str] = [
        f"HOME: {aliases.get(home_sig, short_sig(home_sig))} ({home_sig})",
        f"PARENT: {aliases.get(parent_sig, short_sig(parent_sig))} ({parent_sig})",
        f"CHILD: {aliases.get(child_sig, short_sig(child_sig))} ({child_sig})",
        "",
        "Nodes:",
    ]
    for sig, node in (graph.get("nodes") or {}).items():
        alias = aliases.get(sig, short_sig(sig))
        summary = load_page_summary(trace_dir, sig)
        activity = str(((node.get("meta") or {}).get("foreground_activity")) or "")
        overlay = str(node.get("overlay_kind") or "")
        description = summary or activity or "no saved page summary"
        lines.append(f"- {alias}: {description}. overlay={overlay}; sig={sig}")

    lines.extend(["", "Edges:"])
    for edge in graph.get("edges") or []:
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if not src or not dst:
            continue
        lines.append(f"- {aliases.get(src, short_sig(src))} -> {aliases.get(dst, short_sig(dst))}: {edge_action_summary(edge)}")

    lines.extend(["", "Home paths:"])
    for row in home_rows:
        lines.append(f"- {row['alias']}: {row['home_path']}")

    lines.extend(
        [
            "",
            "Forward edge under evaluation:",
            f"- Parent UI: {aliases.get(parent_sig, short_sig(parent_sig))}",
            f"- Child UI: {aliases.get(child_sig, short_sig(child_sig))}",
            f"- Forward action: {edge_action_summary(forward_edge or {})}",
            "",
            "Return-hints task:",
            "- Decide whether the child UI needs predicted return actions.",
            "- Prefer return targets in this order: home, parent, nearest useful ancestor.",
            "- If the edge is one-way startup/loading or return is not useful, set no_return_needed=true.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_ui_digest(trace_dir: Path, sig: str) -> Dict[str, Any]:
    """Load and compact one UI tree for LLM-visible element ids."""

    state_dir = find_state_dir(trace_dir, sig)
    if not state_dir:
        raise FileNotFoundError(f"missing state directory for state_sig: {sig}")
    uist_path = state_dir / "uist.json"
    if not uist_path.exists():
        uist_path = state_dir / "snap.json"
    ui_json = read_json(uist_path)
    return _compact_digest(ui_json, limit=260)


def read_image_b64(path: Path) -> str:
    """Read an image file as base64 text for an OpenAI-compatible image_url message."""

    if not path.exists():
        raise FileNotFoundError(f"missing image: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def screenshot_path(trace_dir: Path, sig: str) -> Path:
    """Return the processed screenshot path for one UI state."""

    state_dir = find_state_dir(trace_dir, sig)
    if not state_dir:
        raise FileNotFoundError(f"missing state directory for state_sig: {sig}")
    path = state_dir / "screenshot.png"
    if not path.exists():
        raise FileNotFoundError(f"missing screenshot for state_sig: {path}")
    return path


def render_prompt(
    utg_text: str,
    aliases: Dict[str, str],
    parent_sig: str,
    child_sig: str,
    forward_edge: Optional[Dict[str, Any]],
    child_ui_digest: Dict[str, Any],
) -> str:
    """Build the text prompt for the UTG return-hints evaluation."""

    parent_alias = aliases.get(parent_sig, short_sig(parent_sig))
    child_alias = aliases.get(child_sig, short_sig(child_sig))
    payload = {
        "parent_ui": {"alias": parent_alias, "state_sig": parent_sig},
        "child_ui": {"alias": child_alias, "state_sig": child_sig},
        "forward_action": edge_action_summary(forward_edge or {}),
        "child_ui_digest": child_ui_digest,
    }
    return (
        "You are evaluating whether a UI exploration agent can use a UTG to generate return hints.\n\n"
        "Task:\n"
        f"- The parent UI is {parent_alias}.\n"
        f"- The child UI is {child_alias}.\n"
        f"- The saved forward action is: {edge_action_summary(forward_edge or {})}.\n"
        "- Generate a small set of predicted return actions for the child UI.\n"
        "- Prefer targets in this order: home, parent, nearest useful ancestor.\n"
        "- If the forward edge is a one-way startup/loading transition, or returning is not useful, set no_return_needed=true.\n"
        "- Do not blindly choose Android system back.\n"
        "- If the UTG suggests this transition came from a bottom tab switch, prefer switching back to the original tab.\n"
        "- Use the child screenshot and child_ui_digest to identify visible controls and element_id values when possible.\n"
        "- Return at most three return_actions.\n\n"
        "UTG:\n"
        f"{utg_text}\n"
        "Evaluation payload:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def call_llm(prompt: str, parent_screenshot_b64: str, child_screenshot_b64: str, args: argparse.Namespace) -> UtgReturnHintEvalResult:
    """Call the configured LLM with the UTG return-hints prompt."""

    load_project_env(PROJECT_ROOT / ".env")
    client = GPTClient(model=args.model, temperature=args.temperature, timeout_s=args.timeout)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise Android UI graph reasoning evaluator. "
                "Return strict JSON matching the requested schema. Do not include chain-of-thought."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Parent UI screenshot:"},
                {"type": "image_url", "image_url": {"url": _b64_image_url(parent_screenshot_b64)}} if parent_screenshot_b64 else {"type": "text", "text": "(no parent screenshot)"},
                {"type": "text", "text": "Child UI screenshot:"},
                {"type": "image_url", "image_url": {"url": _b64_image_url(child_screenshot_b64)}} if child_screenshot_b64 else {"type": "text", "text": "(no child screenshot)"},
            ],
        },
    ]
    return client._call_structured(messages, UtgReturnHintEvalResult, opname="utg_return_eval")


def render_result_md(result: Dict[str, Any], called_llm: bool) -> str:
    """Render the LLM or skipped-call result as markdown."""

    lines = ["# UTG Return Hints Eval Result", "", f"- call_llm: `{called_llm}`", ""]
    for key in ["parent_ui", "child_ui", "no_return_needed", "reason"]:
        lines.append(f"- {key}: `{result.get(key, '')}`")
    lines.append("")
    lines.append("## Return Actions")
    for idx, item in enumerate(result.get("return_actions") or [], start=1):
        lines.append(f"{idx}. target={item.get('target_alias') or ''} ({item.get('target_type') or ''})")
        lines.append(f"   - state_sig: `{item.get('target_state_sig') or ''}`")
        lines.append(f"   - action: `{item.get('action_type') or ''}` element=`{item.get('element_id')}` summary=`{item.get('action_summary') or ''}`")
        lines.append(f"   - confidence: `{item.get('confidence')}`")
        for ev in item.get("evidence") or []:
            lines.append(f"   - evidence: {ev}")
    if not result.get("return_actions"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run the offline UTG return-hints evaluation and write artifacts."""

    args = parse_args()
    trace_dir = Path(args.trace_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    graph_path = trace_dir / "graph" / "state_graph_snapshot.json"
    graph = read_json(graph_path)
    parent_sig = str(args.parent_sig)
    child_sig = str(args.child_sig)
    home_sig = str(args.home_sig or parent_sig)

    aliases = make_aliases(graph)
    forward_edge = find_forward_edge(graph, parent_sig, child_sig)
    home_rows = build_home_path_rows(graph, aliases, home_sig)
    child_ui_digest = load_ui_digest(trace_dir, child_sig)
    utg_text = render_utg_text(trace_dir, graph, aliases, home_sig, parent_sig, child_sig, forward_edge, home_rows)
    prompt = render_prompt(utg_text, aliases, parent_sig, child_sig, forward_edge, child_ui_digest)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "input.json",
        {
            "trace_dir": str(trace_dir),
            "parent_sig": parent_sig,
            "parent_alias": aliases.get(parent_sig, ""),
            "child_sig": child_sig,
            "child_alias": aliases.get(child_sig, ""),
            "home_sig": home_sig,
            "home_alias": aliases.get(home_sig, ""),
            "forward_edge_found": bool(forward_edge),
            "forward_action": edge_action_summary(forward_edge or {}),
            "mode": args.mode,
            "call_llm": bool(args.call_llm),
        },
    )
    write_json(out_dir / "home_paths.json", home_rows)
    (out_dir / "home_paths.md").write_text(render_home_paths_md(home_rows), encoding="utf-8")
    (out_dir / "utg.txt").write_text(utg_text, encoding="utf-8")
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    if args.call_llm:
        parent_screenshot_b64 = read_image_b64(screenshot_path(trace_dir, parent_sig))
        child_screenshot_b64 = read_image_b64(screenshot_path(trace_dir, child_sig))
        parsed = call_llm(prompt, parent_screenshot_b64, child_screenshot_b64, args)
        result = parsed.model_dump()
    else:
        result = {
            "parent_ui": aliases.get(parent_sig, parent_sig),
            "child_ui": aliases.get(child_sig, child_sig),
            "no_return_needed": False,
            "reason": "LLM call skipped. Re-run with --call-llm.",
            "return_actions": [],
        }
    write_json(out_dir / "result.json", result)
    (out_dir / "result.md").write_text(render_result_md(result, bool(args.call_llm)), encoding="utf-8")

    print(f"[OK] wrote UTG return eval artifacts: {out_dir}")
    print(f"[OK] parent={aliases.get(parent_sig, parent_sig)} child={aliases.get(child_sig, child_sig)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
