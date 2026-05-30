"""Render workflow analysis artifacts into visual reports.

Usage:
  python visualize_run_analysis.py --run-dir traces/<run_id>

Expected input files (auto-exported by WorkflowRunner):
  <run-dir>/graph/state_graph_snapshot.json
  <run-dir>/graph/dfs_snapshot.json
  <run-dir>/graph/state_action_snapshot.json
  <run-dir>/trace.jsonl

Outputs:
  <run-dir>/graph/visual/ui_transition_graph.png
  <run-dir>/graph/visual/dfs_stack.png
  <run-dir>/graph/visual/report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_short_sig(sig: str) -> str:
    text = str(sig or "")
    if len(text) <= 28:
        return text
    return text[:28] + "..."


def _extract_edge_label(edge: Dict[str, Any]) -> str:
    action = edge.get("action") or {}
    steps = list(action.get("actions") or [])
    if not steps:
        return ""
    step = steps[0] or {}
    a = str(step.get("action") or "").strip()
    eid = step.get("element_id")
    if eid is None:
        return a
    return f"{a}:{eid}"


def _candidate_key_from_steps(steps: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for step in steps:
        a = str(step.get("action") or "")
        eid = step.get("element_id")
        txt = str(step.get("text") or "")
        parts.append(f"{a}:{eid}:{txt}")
    return "||".join(parts) if parts else "None:None:"


def _load_sig_to_screenshot(trace_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not trace_path.exists():
        return mapping
    with trace_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if row.get("event") != "snapshot":
                continue
            data = row.get("data") or {}
            sig = str(data.get("state_sig") or "").strip()
            # Prefer annotated vid_map overlay for UTG nodes.
            # Fallback order: vid_map overlay -> processed screenshot -> raw screenshot.
            shot_vid = str(data.get("vidmap_overlay_path") or "").strip()
            shot = str(data.get("screenshot_path") or "").strip()
            shot_raw = str(data.get("screenshot_raw_path") or "").strip()
            chosen = ""
            for cand in (shot_vid, shot, shot_raw):
                if cand and Path(cand).exists():
                    chosen = cand
                    break
            if not chosen:
                chosen = shot_vid or shot or shot_raw
            if sig and chosen and sig not in mapping:
                mapping[sig] = chosen
    return mapping


def _rebuild_from_trace(trace_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fallback builder for historical runs that don't have analysis/*.json snapshots yet.
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    last_stack: List[str] = []
    last_sig = ""
    entry_sig = ""
    run_id = trace_path.parent.name

    sig_to_family: Dict[str, str] = {}
    nav_candidates_by_state: Dict[str, List[Dict[str, Any]]] = {}
    attempted_by_family: Dict[str, set[str]] = defaultdict(set)

    if not trace_path.exists():
        return (
            {"run_id": run_id, "entry_sig": "", "cur_sig": "", "node_count": 0, "edge_count": 0, "nodes": {}, "edges": []},
            {"run_id": run_id, "entry_sig": "", "cur_sig": "", "dfs_stack": [], "dfs_via": [], "parent_map": {}, "stack_depth": 0},
            {"per_state": {}, "unfinished_states": []},
            {"run_id": run_id, "stop_reason": "trace_missing"},
        )

    with trace_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
            except Exception:
                continue
            ctx = evt.get("ctx") or {}
            data = evt.get("data") or {}
            event = str(evt.get("event") or "")

            if not entry_sig and event == "snapshot":
                entry_sig = str(data.get("state_sig") or "")
            if ctx.get("stack"):
                last_stack = list(ctx.get("stack") or [])
            if ctx.get("cur_sig"):
                last_sig = str(ctx.get("cur_sig") or "")

            if event == "snapshot":
                sig = str(data.get("state_sig") or "")
                if not sig:
                    continue
                meta = data.get("meta") or {}
                fam = str(meta.get("struct_sig") or sig)
                sig_to_family[sig] = fam
                row = nodes.setdefault(
                    sig,
                    {
                        "sig": sig,
                        "visit_count": 0,
                        "overlay_kind": "none",
                        "meta": {},
                        "outgoing_count": 0,
                        "incoming_count": 0,
                    },
                )
                row["visit_count"] = int(row.get("visit_count", 0) or 0) + 1
                row["meta"] = dict(meta)

            if event == "transition":
                kind = str(data.get("kind") or "")
                if kind == "observation":
                    sig = str(data.get("sig") or "")
                    if sig:
                        row = nodes.setdefault(
                            sig,
                            {
                                "sig": sig,
                                "visit_count": 0,
                                "overlay_kind": "none",
                                "meta": {},
                                "outgoing_count": 0,
                                "incoming_count": 0,
                            },
                        )
                        row["visit_count"] = int(row.get("visit_count", 0) or 0) + 1
                        if isinstance(data.get("meta"), dict):
                            row["meta"] = dict(data.get("meta") or {})
                elif kind == "transition":
                    src = str(data.get("src") or "")
                    dst = str(data.get("dst") or "")
                    action = data.get("action") or {}
                    akey = json.dumps(action, ensure_ascii=False, sort_keys=True)
                    if src and dst:
                        edge_key = (src, dst, akey)
                        row = edge_map.setdefault(
                            edge_key,
                            {
                                "src": src,
                                "dst": dst,
                                "action": action,
                                "count": 0,
                                "no_effect": bool(src == dst),
                                "verified_ok": 0,
                                "verified_fail": 0,
                                "last_ts": float(evt.get("ts") or 0.0),
                                "last_verified_ts": 0.0,
                            },
                        )
                        row["count"] = int(row.get("count", 0) or 0) + 1
                        row["last_ts"] = float(evt.get("ts") or row["last_ts"] or 0.0)
                        nodes.setdefault(src, {"sig": src, "visit_count": 0, "overlay_kind": "none", "meta": {}, "outgoing_count": 0, "incoming_count": 0})
                        nodes.setdefault(dst, {"sig": dst, "visit_count": 0, "overlay_kind": "none", "meta": {}, "outgoing_count": 0, "incoming_count": 0})
                        fam = sig_to_family.get(src, src)
                        steps = list((action or {}).get("actions") or [])
                        key = _candidate_key_from_steps(steps)
                        attempted_by_family[fam].add(key)

            if event == "llm_result" and str(data.get("kind") or "") == "nav":
                sig = str(data.get("state_sig") or "")
                result = data.get("result") or {}
                cands = list(result.get("candidate_actions") or [])
                nav_candidates_by_state[sig] = cands

    # compute in/out degree
    outs: Dict[str, set[str]] = defaultdict(set)
    ins: Dict[str, set[str]] = defaultdict(set)
    edges: List[Dict[str, Any]] = []
    for row in edge_map.values():
        edges.append(row)
        outs[row["src"]].add(row["dst"])
        ins[row["dst"]].add(row["src"])
    for sig, row in nodes.items():
        row["outgoing_count"] = len(outs.get(sig, set()))
        row["incoming_count"] = len(ins.get(sig, set()))

    # Build action snapshot fallback
    per_state: Dict[str, Any] = {}
    for sig in sorted(nodes.keys()):
        fam = sig_to_family.get(sig, sig)
        raw_cands = nav_candidates_by_state.get(sig, [])
        candidate_rows: List[Dict[str, Any]] = []
        candidate_keys: List[str] = []
        for cand in raw_cands:
            steps = list(cand.get("actions") or [])
            key = _candidate_key_from_steps(steps)
            candidate_keys.append(key)
            candidate_rows.append(
                {
                    "candidate_key": key,
                    "score": float(cand.get("score", 0.0) or 0.0),
                    "tags": list(cand.get("tags") or []),
                    "actions": [
                        {
                            "action": str(st.get("action") or ""),
                            "element_id": st.get("element_id"),
                            "text": str(st.get("text") or ""),
                            "reasoning": str(st.get("reasoning") or ""),
                        }
                        for st in steps[:4]
                    ],
                }
            )
        attempted = set(attempted_by_family.get(fam, set()))
        remaining = [k for k in candidate_keys if k not in attempted]
        per_state[sig] = {
            "state_sig": sig,
            "family_id": fam,
            "candidate_count": len(candidate_keys),
            "explored_count": len([k for k in candidate_keys if k in attempted]),
            "attempted_count": len([k for k in candidate_keys if k in attempted]),
            "remaining_count": len(remaining),
            "is_exhausted": len(candidate_keys) > 0 and len(remaining) == 0,
            "candidate_keys": candidate_keys,
            "remaining_candidate_keys": remaining,
            "explored_keys": sorted(attempted),
            "attempted_keys": sorted(attempted),
            "candidates": candidate_rows,
        }
    actions = {
        "per_state": per_state,
        "unfinished_states": sorted([sig for sig, row in per_state.items() if int(row.get("remaining_count", 0) or 0) > 0]),
    }
    graph = {
        "run_id": run_id,
        "cur_sig": last_sig,
        "entry_sig": entry_sig,
        "restart_entry_sig": entry_sig,
        "stop_reason": "reconstructed_from_trace",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
    dfs = {
        "run_id": run_id,
        "cur_sig": last_sig,
        "stop_reason": "reconstructed_from_trace",
        "entry_sig": entry_sig,
        "restart_entry_sig": entry_sig,
        "dfs_stack": last_stack,
        "dfs_via": [None] * len(last_stack),
        "parent_map": {},
        "stack_depth": len(last_stack),
    }
    summary = {
        "run_id": run_id,
        "stop_reason": "reconstructed_from_trace",
        "graph_node_count": len(nodes),
        "graph_edge_count": len(edges),
        "dfs_stack_depth": len(last_stack),
        "unfinished_state_count": len(actions.get("unfinished_states") or []),
    }
    return graph, dfs, actions, summary


def _assign_layers(nodes: List[str], edges: List[Tuple[str, str]], entry_sig: str) -> Dict[str, int]:
    adj: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = defaultdict(int)
    for s, d in edges:
        adj[s].append(d)
        indeg[d] += 1
        indeg.setdefault(s, indeg.get(s, 0))

    layers: Dict[str, int] = {}
    q: deque[str] = deque()
    if entry_sig and entry_sig in set(nodes):
        layers[entry_sig] = 0
        q.append(entry_sig)
    else:
        roots = sorted([n for n in nodes if indeg.get(n, 0) == 0])[:1]
        for r in roots:
            layers[r] = 0
            q.append(r)

    while q:
        cur = q.popleft()
        base = layers.get(cur, 0)
        for nxt in adj.get(cur, []):
            cand = base + 1
            if nxt not in layers or cand < layers[nxt]:
                layers[nxt] = cand
                q.append(nxt)

    max_layer = max(layers.values()) if layers else 0
    for sig in sorted(nodes):
        if sig not in layers:
            max_layer += 1
            layers[sig] = max_layer
    return layers


def _draw_arrow(draw: ImageDraw.ImageDraw, src: Tuple[int, int], dst: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x1, y1 = src
    x2, y2 = dst
    draw.line((x1, y1, x2, y2), fill=color, width=3)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 10
    p1 = (x2, y2)
    p2 = (int(x2 - ux * size + px * (size * 0.6)), int(y2 - uy * size + py * (size * 0.6)))
    p3 = (int(x2 - ux * size - px * (size * 0.6)), int(y2 - uy * size - py * (size * 0.6)))
    draw.polygon([p1, p2, p3], fill=color)


def _fit_image(path: str, width: int, height: int) -> Image.Image:
    bg = Image.new("RGB", (width, height), (245, 245, 245))
    p = Path(path)
    if not p.exists():
        return bg
    try:
        img = Image.open(p).convert("RGB")
        img.thumbnail((width, height))
        x = (width - img.width) // 2
        y = (height - img.height) // 2
        bg.paste(img, (x, y))
        return bg
    except Exception:
        return bg


def render_ui_transition_graph(
    graph: Dict[str, Any],
    actions: Dict[str, Any],
    sig_to_shot: Dict[str, str],
    out_path: Path,
) -> None:
    nodes: Dict[str, Any] = dict(graph.get("nodes") or {})
    edges_raw: List[Dict[str, Any]] = list(graph.get("edges") or [])
    if not nodes:
        img = Image.new("RGB", (800, 320), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 20), "No graph nodes found.", fill=(0, 0, 0))
        img.save(out_path)
        return

    edge_pairs = [(str(e.get("src") or ""), str(e.get("dst") or "")) for e in edges_raw if e.get("src") and e.get("dst")]
    layers = _assign_layers(list(nodes.keys()), edge_pairs, str(graph.get("entry_sig") or ""))
    grouped: Dict[int, List[str]] = defaultdict(list)
    for sig in nodes.keys():
        grouped[int(layers.get(sig, 0))].append(sig)
    for _, arr in grouped.items():
        arr.sort()

    box_w, box_h = 220, 380
    thumb_w, thumb_h = 200, 300
    margin = 60
    col_gap = 300
    row_gap = 430
    max_layer = max(grouped.keys()) if grouped else 0
    max_rows = max((len(v) for v in grouped.values()), default=1)
    canvas_w = margin * 2 + (max_layer + 1) * col_gap + box_w
    canvas_h = margin * 2 + max_rows * row_gap + 100

    image = Image.new("RGB", (canvas_w, canvas_h), (252, 252, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    positions: Dict[str, Tuple[int, int]] = {}
    for layer in sorted(grouped.keys()):
        for idx, sig in enumerate(grouped[layer]):
            x = margin + layer * col_gap
            y = margin + idx * row_gap
            positions[sig] = (x, y)

    # Draw edges first
    for edge in edges_raw:
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if src not in positions or dst not in positions:
            continue
        sx, sy = positions[src]
        dx, dy = positions[dst]
        start = (sx + box_w, sy + box_h // 2)
        end = (dx, dy + box_h // 2)
        _draw_arrow(draw, start, end, (65, 105, 225))
        label = _extract_edge_label(edge)
        if label:
            mx = (start[0] + end[0]) // 2
            my = (start[1] + end[1]) // 2 - 10
            draw.rectangle((mx - 50, my - 8, mx + 50, my + 10), fill=(255, 255, 255))
            draw.text((mx - 46, my - 6), label[:14], fill=(35, 35, 35), font=font)

    per_state = (actions.get("per_state") or {}) if isinstance(actions, dict) else {}

    # Draw nodes with screenshots
    for sig, (x, y) in positions.items():
        node = nodes.get(sig) or {}
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=12, fill=(255, 255, 255), outline=(120, 120, 120), width=2)

        shot = sig_to_shot.get(sig, "")
        thumb = _fit_image(shot, thumb_w, thumb_h)
        image.paste(thumb, (x + 10, y + 10))

        visit = int(node.get("visit_count", 0) or 0)
        rem = int(((per_state.get(sig) or {}).get("remaining_count", 0)) or 0)
        exhausted = bool((per_state.get(sig) or {}).get("is_exhausted", False))
        status = "done" if exhausted else ("pending" if rem > 0 else "-")

        draw.text((x + 10, y + 318), _safe_short_sig(sig), fill=(20, 20, 20), font=font)
        draw.text((x + 10, y + 334), f"visit={visit} remaining={rem}", fill=(20, 20, 20), font=font)
        draw.text((x + 10, y + 350), f"status={status}", fill=(20, 20, 20), font=font)

    title = "UI Transition Graph (node screenshot + directed edges)"
    draw.text((margin, 20), title, fill=(10, 10, 10), font=font)
    image.save(out_path)


def render_dfs_stack(dfs: Dict[str, Any], actions: Dict[str, Any], out_path: Path) -> None:
    stack = list(dfs.get("dfs_stack") or [])
    vias = list(dfs.get("dfs_via") or [])
    per_state = (actions.get("per_state") or {}) if isinstance(actions, dict) else {}
    if not stack:
        img = Image.new("RGB", (900, 280), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 20), "DFS stack is empty.", fill=(0, 0, 0))
        img.save(out_path)
        return

    card_w, card_h = 860, 78
    margin = 30
    gap = 18
    canvas_h = margin * 2 + len(stack) * (card_h + gap) + 80
    img = Image.new("RGB", (950, canvas_h), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((margin, 12), "DFS Stack (top is the last frame)", fill=(10, 10, 10), font=font)
    for i, sig in enumerate(reversed(stack)):
        stack_idx = len(stack) - 1 - i
        y = margin + i * (card_h + gap) + 28
        x = margin
        is_top = (stack_idx == len(stack) - 1)
        fill = (232, 245, 255) if is_top else (255, 255, 255)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=10, fill=fill, outline=(110, 110, 110), width=2)
        via = vias[stack_idx] if stack_idx < len(vias) else None
        row = per_state.get(sig) or {}
        rem = int(row.get("remaining_count", 0) or 0)
        explored = int(row.get("explored_count", 0) or 0)
        total = int(row.get("candidate_count", 0) or 0)
        head = "TOP" if is_top else f"frame#{stack_idx}"
        draw.text((x + 12, y + 10), f"{head}  {_safe_short_sig(sig)}", fill=(20, 20, 20), font=font)
        draw.text((x + 12, y + 30), f"via={str(via or '-')[:80]}", fill=(45, 45, 45), font=font)
        draw.text((x + 12, y + 48), f"candidates={total} explored={explored} remaining={rem}", fill=(45, 45, 45), font=font)

        if i < len(stack) - 1:
            sx = x + card_w // 2
            sy = y + card_h
            ex = sx
            ey = y + card_h + gap - 4
            _draw_arrow(draw, (sx, sy), (ex, ey), (120, 120, 120))

    img.save(out_path)


def write_markdown_report(
    summary: Dict[str, Any],
    graph: Dict[str, Any],
    dfs: Dict[str, Any],
    actions: Dict[str, Any],
    out_path: Path,
) -> None:
    per_state = actions.get("per_state") or {}
    unfinished = list(actions.get("unfinished_states") or [])
    lines: List[str] = []
    lines.append("# Workflow Analysis Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- run_id: `{summary.get('run_id', '')}`")
    lines.append(f"- stop_reason: `{summary.get('stop_reason', '')}`")
    lines.append(f"- graph nodes: `{graph.get('node_count', 0)}`")
    lines.append(f"- graph edges: `{graph.get('edge_count', 0)}`")
    lines.append(f"- dfs_stack_depth: `{dfs.get('stack_depth', 0)}`")
    lines.append(f"- unfinished_states: `{len(unfinished)}`")
    lines.append("")
    lines.append("## Visuals")
    lines.append("- UI transition graph: `ui_transition_graph.png`")
    lines.append("- DFS stack view: `dfs_stack.png`")
    lines.append("")
    lines.append("## Top Unfinished States")
    if not unfinished:
        lines.append("- none")
    else:
        for sig in unfinished[:30]:
            row = per_state.get(sig) or {}
            lines.append(
                f"- `{sig}` remaining={row.get('remaining_count', 0)} explored={row.get('explored_count', 0)}/"
                f"{row.get('candidate_count', 0)}"
            )
    lines.append("")
    lines.append("## Current DFS Stack")
    stack = list(dfs.get("dfs_stack") or [])
    if not stack:
        lines.append("- empty")
    else:
        for idx, sig in enumerate(stack):
            lines.append(f"- [{idx}] `{sig}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render workflow analysis artifacts into PNG/Markdown reports.")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. traces/20260425_200637_com.beenverified.android")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    analysis_dir = run_dir / "graph"
    if not analysis_dir.exists() and (run_dir / "analysis").exists():
        analysis_dir = run_dir / "analysis"
    visual_dir = analysis_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    graph_path = analysis_dir / "state_graph_snapshot.json"
    dfs_path = analysis_dir / "dfs_snapshot.json"
    action_path = analysis_dir / "state_action_snapshot.json"
    summary_path = analysis_dir / "run_analysis_summary.json"
    trace_path = run_dir / "trace.jsonl"

    if graph_path.exists() and dfs_path.exists() and action_path.exists():
        graph = _read_json(graph_path)
        dfs = _read_json(dfs_path)
        actions = _read_json(action_path)
        summary = _read_json(summary_path) if summary_path.exists() else {"run_id": str(run_dir.name), "stop_reason": "unknown"}
    else:
        graph, dfs, actions, summary = _rebuild_from_trace(trace_path)
        # Persist reconstructed snapshots for convenience.
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        dfs_path.write_text(json.dumps(dfs, ensure_ascii=False, indent=2), encoding="utf-8")
        action_path.write_text(json.dumps(actions, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[INFO] analysis snapshots were missing; reconstructed from trace.jsonl")
    sig_to_shot = _load_sig_to_screenshot(trace_path)

    graph_img = visual_dir / "ui_transition_graph.png"
    dfs_img = visual_dir / "dfs_stack.png"
    report_md = visual_dir / "report.md"

    render_ui_transition_graph(graph, actions, sig_to_shot, graph_img)
    render_dfs_stack(dfs, actions, dfs_img)
    write_markdown_report(summary, graph, dfs, actions, report_md)

    print(f"[OK] graph image: {graph_img}")
    print(f"[OK] dfs image: {dfs_img}")
    print(f"[OK] report: {report_md}")


if __name__ == "__main__":
    main()
