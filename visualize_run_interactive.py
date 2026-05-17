"""Build an interactive HTML UI-transition graph for one workflow run.

Usage:
  python visualize_run_interactive.py --run-dir traces/<run_id>

Output:
  <run-dir>/analysis/interactive/ui_transition_interactive.html
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.request import Request, urlopen


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _canon_action_key(action_payload: Dict[str, Any]) -> str:
    return json.dumps(action_payload or {}, ensure_ascii=False, sort_keys=True)


def _guess_target_package(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes") or {}
    entry_sig = str(graph.get("entry_sig") or "")
    if entry_sig and entry_sig in nodes:
        meta = (nodes.get(entry_sig) or {}).get("meta") or {}
        pkg = str(meta.get("foreground_package") or "").strip()
        if pkg:
            return pkg
    counter: Counter[str] = Counter()
    for node in (nodes or {}).values():
        meta = (node or {}).get("meta") or {}
        pkg = str(meta.get("foreground_package") or "").strip()
        if pkg:
            counter[pkg] += 1
    return counter.most_common(1)[0][0] if counter else ""


def _load_trace_data(
    trace_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str, str], List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Collect state snapshots, edge occurrences, and a human-readable action timeline.

    Input:
      trace_path: one run's trace.jsonl file.

    Output:
      per_state: first-seen snapshot metadata keyed by state signature.
      per_edge_occurs: transition occurrences keyed by graph edge identity.
      flow_rows: compact chronological rows for the right-side HTML timeline and Markdown export.

    Function:
      Converts low-level trace events into a UI-centric flow: discovered state, action, result, and recovery events.
    """
    per_state: Dict[str, Dict[str, Any]] = {}
    per_edge_occurs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    flow_rows: List[Dict[str, Any]] = []
    if not trace_path.exists():
        return per_state, per_edge_occurs, flow_rows

    snapshot_seq = 0
    transition_seq = 0
    flow_seq = 0
    seen_snapshot_sigs: set[str] = set()

    def _first_step(action_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return the first action step from a transition payload."""
        steps = list((action_payload or {}).get("actions") or [])
        return steps[0] if steps else {}

    def _step_label(step: Dict[str, Any]) -> str:
        """Return a short human label for a trace action step."""
        for key in ("anchor_label", "text", "reasoning", "anchor_class"):
            val = str((step or {}).get(key) or "").strip()
            if val:
                return " ".join(val.split())[:80]
        return ""

    def _append_flow(row: Dict[str, Any]) -> None:
        """Append one compact timeline row with a stable sequence number."""
        nonlocal flow_seq
        flow_seq += 1
        row["flow_seq"] = flow_seq
        flow_rows.append(row)

    with trace_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            event = str(row.get("event") or "")
            data = row.get("data") or {}

            if event == "snapshot":
                sig = str(data.get("state_sig") or "").strip()
                if not sig:
                    continue
                snapshot_seq += 1
                if sig not in seen_snapshot_sigs:
                    seen_snapshot_sigs.add(sig)
                    _append_flow(
                        {
                            "kind": "state_seen",
                            "ts": float(row.get("ts") or 0.0),
                            "src": "",
                            "dst": sig,
                            "action": "observe",
                            "element_id": None,
                            "label": str(data.get("page") or ""),
                            "changed": None,
                            "change_type": "snapshot",
                            "confidence": None,
                            "result": "first_seen",
                            "page": str(data.get("page") or ""),
                            "known": bool(data.get("known")),
                            "xml_reliable": data.get("xml_reliable"),
                            "action_key": "",
                        }
                    )
                if sig not in per_state:
                    meta = data.get("meta") or {}
                    per_state[sig] = {
                        "first_snapshot_seq": snapshot_seq,
                        "first_snapshot_ts": float(row.get("ts") or 0.0),
                        "snapshot_paths": {
                            "screenshot_path": str(data.get("screenshot_path") or ""),
                            "screenshot_raw_path": str(data.get("screenshot_raw_path") or ""),
                            "vidmap_overlay_path": str(data.get("vidmap_overlay_path") or ""),
                            "uist_overlay_path": str(data.get("uist_overlay_path") or ""),
                            "xml_path": str(data.get("xml_path") or ""),
                            "xml_raw_path": str(data.get("xml_raw_path") or ""),
                            "uist_path": str(data.get("uist_path") or ""),
                        },
                        "vid_map_summary": data.get("vid_map_summary") or {},
                        "snapshot_meta": meta,
                    }
                else:
                    per_state[sig]["last_snapshot_ts"] = float(row.get("ts") or 0.0)

            if event == "transition" and str(data.get("kind") or "") == "transition":
                src = str(data.get("src") or "")
                dst = str(data.get("dst") or "")
                action_payload = data.get("action") or {}
                if not src or not dst:
                    continue
                transition_seq += 1
                key = (src, dst, _canon_action_key(action_payload))
                step0 = _first_step(action_payload)
                outcome = action_payload.get("outcome") or {}
                action_name = str(step0.get("action") or "")
                element_id = step0.get("element_id")
                label = _step_label(step0)
                action_key = f"{action_name}:{element_id}:{label}".strip(":")
                per_edge_occurs[key].append(
                    {
                        "seq": transition_seq,
                        "ts": float(row.get("ts") or 0.0),
                        "src": src,
                        "dst": dst,
                        "action": action_name,
                        "element_id": element_id,
                        "text": step0.get("text"),
                        "label": label,
                        "action_key": action_key,
                        "changed": outcome.get("changed"),
                        "change_type": str(outcome.get("change_type") or ""),
                        "confidence": float(outcome.get("confidence") or 0.0),
                    }
                )
                _append_flow(
                    {
                        "kind": "transition",
                        "ts": float(row.get("ts") or 0.0),
                        "src": src,
                        "dst": dst,
                        "action": action_name,
                        "element_id": element_id,
                        "label": label,
                        "changed": outcome.get("changed"),
                        "change_type": str(outcome.get("change_type") or ""),
                        "confidence": outcome.get("confidence"),
                        "result": "changed" if bool(outcome.get("changed")) else "same_state",
                        "page": "",
                        "known": None,
                        "xml_reliable": None,
                        "action_key": action_key,
                    }
                )

            if event in {
                "external_foreground_after_action",
                "app_restart_triggered",
                "state_return_exhausted",
                "return_exhausted_frontier_replay",
                "replay_path_missing",
                "utg_restore_failed",
                "utg_restore_success",
                "run_stop_condition",
                "loop_detected",
                "action_blacklisted",
            }:
                _append_flow(
                    {
                        "kind": event,
                        "ts": float(row.get("ts") or 0.0),
                        "src": str(data.get("from_sig") or data.get("src") or ctx.get("cur_sig") or ""),
                        "dst": str(data.get("target_sig") or data.get("dst") or data.get("actual_sig") or data.get("frontier_sig") or ""),
                        "action": str(data.get("action_key") or data.get("reason") or event),
                        "element_id": None,
                        "label": str(data.get("reason") or data.get("kind") or ""),
                        "changed": None,
                        "change_type": event,
                        "confidence": None,
                        "result": str(data.get("reason") or data.get("kind") or event),
                        "page": "",
                        "known": None,
                        "xml_reliable": None,
                        "action_key": str(data.get("action_key") or ""),
                    }
                )
    return per_state, per_edge_occurs, flow_rows


def _load_latest_observations(obs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load latest observation json for each state_sig from one observation directory."""
    out: Dict[str, Dict[str, Any]] = {}
    if not obs_dir.exists():
        return out

    def _ts_from_name(path: Path) -> int:
        stem = path.stem
        p = stem.split("_", 1)[0]
        try:
            return int(p)
        except Exception:
            return 0

    files = sorted(obs_dir.glob("*.json"), key=_ts_from_name)
    for fp in files:
        try:
            obj = _read_json(fp)
        except Exception:
            continue
        sig = str(obj.get("state_sig") or "").strip()
        if not sig:
            continue
        out[sig] = obj
    return out


def _relpath_or_raw(path_str: str, anchor_dir: Path) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    try:
        if p.exists():
            return str(p.resolve().relative_to(anchor_dir.resolve())).replace("\\", "/")
    except Exception:
        pass
    try:
        return str(Path(path_str).as_posix())
    except Exception:
        return str(path_str)


def _short_sig(sig: str, n: int = 14) -> str:
    if len(sig) <= n:
        return sig
    return sig[:n] + "..."


def _ensure_vis_network_vendor(interactive_dir: Path) -> str:
    """Ensure local vis-network standalone js exists.

    Input:
      interactive_dir: output dir of interactive html

    Output:
      relative path from interactive_dir to local js, or empty string if unavailable
    """
    vendor_dir = interactive_dir / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    out_js = vendor_dir / "vis-network.min.js"
    if out_js.exists() and out_js.stat().st_size > 100_000:
        return _relpath_or_raw(str(out_js), interactive_dir)

    urls = [
        "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.0/standalone/umd/vis-network.min.js",
    ]
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=12) as resp:
                data = resp.read()
            if b"vis-network" not in data[:20_000] and b"vis.Network" not in data:
                continue
            out_js.write_bytes(data)
            if out_js.stat().st_size > 100_000:
                return _relpath_or_raw(str(out_js), interactive_dir)
        except Exception:
            continue
    return _relpath_or_raw(str(out_js), interactive_dir) if out_js.exists() else ""


def _edge_label(edge: Dict[str, Any]) -> str:
    action = edge.get("action") or {}
    steps = list(action.get("actions") or [])
    step = steps[0] if steps else {}
    a = str(step.get("action") or "")
    eid = step.get("element_id")
    outcome = action.get("outcome") or {}
    ctype = str(outcome.get("change_type") or "")
    if eid is None:
        base = a
    else:
        base = f"{a}:{eid}"
    if ctype:
        return f"{base} [{ctype}]"
    return base


def _match_action_candidate(
    src_sig: str,
    action_payload: Dict[str, Any],
    per_state_action: Dict[str, Any],
    nav_obs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Try to map one executed edge action back to LLM candidate details.

    Input:
      src_sig: source state signature
      action_payload: edge action payload from state_graph_snapshot
      per_state_action: loaded state_action_snapshot["per_state"]
      nav_obs: latest nav observation map by state signature

    Output:
      dict containing:
        - state_action_match: matched candidate info from state_action_snapshot
        - nav_candidate_match: matched candidate info from observations/nav
      Missing branch returns {} for corresponding fields.
    """
    out: Dict[str, Any] = {"state_action_match": {}, "nav_candidate_match": {}}
    steps = list((action_payload or {}).get("actions") or [])
    step0 = steps[0] if steps else {}
    action_name = str(step0.get("action") or "")
    element_id = step0.get("element_id")
    text_val = step0.get("text")

    state_row = per_state_action.get(src_sig) or {}
    for cand in list(state_row.get("candidates") or []):
        csteps = list(cand.get("actions") or [])
        c0 = csteps[0] if csteps else {}
        if str(c0.get("action") or "") != action_name:
            continue
        if c0.get("element_id") != element_id:
            continue
        c_text = c0.get("text")
        if text_val not in (None, "") and c_text not in (None, "") and c_text != text_val:
            continue
        out["state_action_match"] = {
            "candidate_key": cand.get("candidate_key"),
            "score": cand.get("score"),
            "tags": cand.get("tags") or [],
            "reasoning": c0.get("reasoning"),
        }
        break

    nav = nav_obs.get(src_sig) or {}
    nav_result = nav.get("nav_result") if isinstance(nav.get("nav_result"), dict) else {}
    nav_result = nav_result or {}
    for cand in list(nav_result.get("candidate_actions") or []):
        csteps = list(cand.get("actions") or [])
        c0 = csteps[0] if csteps else {}
        if str(c0.get("action") or "") != action_name:
            continue
        if c0.get("element_id") != element_id:
            continue
        c_text = c0.get("text")
        if text_val not in (None, "") and c_text not in (None, "") and c_text != text_val:
            continue
        out["nav_candidate_match"] = {
            "score": cand.get("score"),
            "tags": cand.get("tags") or [],
            "reasoning": c0.get("reasoning"),
            "return_method": cand.get("return_method"),
        }
        break
    return out


def _enrich_flow_rows(flow_rows: List[Dict[str, Any]], alias: Dict[str, str]) -> List[Dict[str, Any]]:
    """Attach UI aliases and compact display text to timeline rows.

    Input:
      flow_rows: raw chronological rows from trace.jsonl.
      alias: state signature to UI alias map, for example phash:* -> UI2.

    Output:
      rows with src_alias, dst_alias, action_display, and result_display.

    Function:
      Keeps HTML and Markdown timelines readable without losing the underlying signatures.
    """
    out: List[Dict[str, Any]] = []
    for row in list(flow_rows or []):
        item = dict(row)
        src = str(item.get("src") or "")
        dst = str(item.get("dst") or "")
        action = str(item.get("action") or "")
        element_id = item.get("element_id")
        label = str(item.get("label") or "").strip()
        action_bits = [action]
        if element_id is not None:
            action_bits.append(f"element={element_id}")
        if label:
            action_bits.append(label)
        item["src_alias"] = alias.get(src, _short_sig(src) if src else "")
        item["dst_alias"] = alias.get(dst, _short_sig(dst) if dst else "")
        item["action_display"] = " ".join(x for x in action_bits if x).strip() or str(item.get("kind") or "")
        change_type = str(item.get("change_type") or "")
        result = str(item.get("result") or "")
        if item.get("kind") == "state_seen":
            item["result_display"] = f"发现 {item['dst_alias']} ({'known' if item.get('known') else 'new'})"
        elif src or dst:
            arrow = f"{item['src_alias'] or '-'} -> {item['dst_alias'] or '-'}"
            item["result_display"] = f"{arrow} / {change_type or result}"
        else:
            item["result_display"] = result or change_type
        out.append(item)
    return out


def _write_action_timeline_md(out_path: Path, payload: Dict[str, Any]) -> None:
    """Write the simplified UI/action timeline beside the interactive HTML.

    Input:
      out_path: Markdown destination path.
      payload: page payload containing run_id, stop_reason, and enriched timeline rows.

    Output:
      Creates or replaces action_timeline.md.

    Function:
      Provides a compact text version of the same right-side HTML timeline for quick debugging.
    """
    rows = list(payload.get("timeline") or [])
    lines = [
        "# Action Timeline",
        "",
        f"- run_id: `{payload.get('run_id') or ''}`",
        f"- stop_reason: `{payload.get('stop_reason') or ''}`",
        "",
        "| # | source | action | result |",
        "|---:|---|---|---|",
    ]
    for row in rows:
        seq = row.get("flow_seq")
        src = row.get("src_alias") or "-"
        action = str(row.get("action_display") or "").replace("|", "\\|")
        result = str(row.get("result_display") or "").replace("|", "\\|")
        lines.append(f"| {seq} | `{src}` | {action} | {result} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_interactive_html(run_dir: Path) -> Path:
    analysis_dir = run_dir / "analysis"
    graph_path = analysis_dir / "state_graph_snapshot.json"
    actions_path = analysis_dir / "state_action_snapshot.json"
    trace_path = run_dir / "trace.jsonl"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph snapshot: {graph_path}")
    graph = _read_json(graph_path)
    actions = _read_json(actions_path) if actions_path.exists() else {"per_state": {}, "unfinished_states": []}

    per_state_trace, edge_occurs, flow_rows = _load_trace_data(trace_path)
    nav_obs = _load_latest_observations(run_dir / "observations" / "nav")
    router_obs = _load_latest_observations(run_dir / "observations" / "router")
    blocks_obs = _load_latest_observations(run_dir / "observations" / "blocks_fill")

    interactive_dir = analysis_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    out_html = interactive_dir / "ui_transition_interactive.html"
    local_vis_js = _ensure_vis_network_vendor(interactive_dir)

    nodes_raw: Dict[str, Any] = graph.get("nodes") or {}
    edges_raw: List[Dict[str, Any]] = list(graph.get("edges") or [])
    per_state_action: Dict[str, Any] = actions.get("per_state") or {}
    unfinished = set(actions.get("unfinished_states") or [])
    target_pkg = _guess_target_package(graph)

    # Stable alias ordering: trace first-seen, then remaining.
    sigs = list(nodes_raw.keys())
    sigs.sort(key=lambda s: (int((per_state_trace.get(s) or {}).get("first_snapshot_seq", 10**9)), s))
    alias = {sig: f"UI{i+1}" for i, sig in enumerate(sigs)}

    vis_nodes: List[Dict[str, Any]] = []
    node_detail_map: Dict[str, Dict[str, Any]] = {}

    for sig in sigs:
        node = nodes_raw.get(sig) or {}
        meta = node.get("meta") or {}
        trace_state = per_state_trace.get(sig) or {}
        snap_paths = trace_state.get("snapshot_paths") or {}
        nav = nav_obs.get(sig) or {}
        nav_result = nav.get("nav_result") if isinstance(nav.get("nav_result"), dict) else {}
        nav_result = nav_result or {}

        row_action = per_state_action.get(sig) or {}
        cand_ct = int(row_action.get("candidate_count", 0) or 0)
        rem_ct = int(row_action.get("remaining_count", 0) or 0)
        visit = int(node.get("visit_count", 0) or 0)
        overlay = str(node.get("overlay_kind") or "none")
        fg_pkg = str(meta.get("foreground_package") or "")
        is_external = bool(target_pkg and fg_pkg and fg_pkg != target_pkg)

        color = "#ffe9e9" if is_external else ("#e8f5ff" if sig in unfinished else "#eef6ee")
        border = "#d23f3f" if is_external else ("#2f6fad" if sig in unfinished else "#5a8f5a")
        label = f"{alias[sig]}\nvisit={visit} rem={rem_ct}\n{overlay}"

        vis_nodes.append(
            {
                "id": sig,
                "label": label,
                "shape": "box",
                "color": {"background": color, "border": border},
                "font": {"face": "Consolas", "size": 14},
                "margin": 10,
            }
        )

        preview_path = (
            snap_paths.get("vidmap_overlay_path")
            or snap_paths.get("uist_overlay_path")
            or snap_paths.get("screenshot_path")
            or snap_paths.get("screenshot_raw_path")
            or ""
        )

        node_detail_map[sig] = {
            "alias": alias[sig],
            "state_sig": sig,
            "visit_count": visit,
            "unfinished": bool(sig in unfinished),
            "target_package": target_pkg,
            "foreground_package": fg_pkg,
            "foreground_activity": str(meta.get("foreground_activity") or ""),
            "overlay_kind": overlay,
            "xml_reliable": meta.get("xml_reliable"),
            "candidate_summary": {
                "candidate_count": cand_ct,
                "explored_count": int(row_action.get("explored_count", 0) or 0),
                "attempted_count": int(row_action.get("attempted_count", 0) or 0),
                "remaining_count": rem_ct,
            },
            "snapshot_paths": {
                "preview": _relpath_or_raw(str(preview_path), interactive_dir),
                "screenshot_path": _relpath_or_raw(str(snap_paths.get("screenshot_path") or ""), interactive_dir),
                "screenshot_raw_path": _relpath_or_raw(str(snap_paths.get("screenshot_raw_path") or ""), interactive_dir),
                "xml_path": _relpath_or_raw(str(snap_paths.get("xml_path") or ""), interactive_dir),
                "xml_raw_path": _relpath_or_raw(str(snap_paths.get("xml_raw_path") or ""), interactive_dir),
                "uist_path": _relpath_or_raw(str(snap_paths.get("uist_path") or ""), interactive_dir),
                "uist_overlay_path": _relpath_or_raw(str(snap_paths.get("uist_overlay_path") or ""), interactive_dir),
                "vidmap_overlay_path": _relpath_or_raw(str(snap_paths.get("vidmap_overlay_path") or ""), interactive_dir),
            },
            "vid_map_summary": trace_state.get("vid_map_summary") or {},
            "snapshot_meta": trace_state.get("snapshot_meta") or {},
            "nav_result": nav_result,
            "router_observation": router_obs.get(sig) or {},
            "blocks_observation": blocks_obs.get(sig) or {},
            "graph_meta": meta,
            "state_action": row_action,
        }

    vis_edges: List[Dict[str, Any]] = []
    edge_detail_map: Dict[str, Dict[str, Any]] = {}

    for i, edge in enumerate(edges_raw):
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        action_payload = edge.get("action") or {}
        if not src or not dst:
            continue
        edge_id = f"e{i+1}"
        label = _edge_label(edge)
        no_effect = bool(edge.get("no_effect"))
        vis_edges.append(
            {
                "id": edge_id,
                "from": src,
                "to": dst,
                "label": label[:42],
                "arrows": "to",
                "color": {"color": "#6f6f6f" if no_effect else "#2c5aa0"},
                "dashes": bool(no_effect),
                "font": {"align": "middle", "size": 12},
            }
        )

        key = (src, dst, _canon_action_key(action_payload))
        occurs = edge_occurs.get(key) or []
        edge_match = _match_action_candidate(src, action_payload, per_state_action, nav_obs)
        edge_detail_map[edge_id] = {
            "id": edge_id,
            "src": src,
            "src_alias": alias.get(src, src),
            "dst": dst,
            "dst_alias": alias.get(dst, dst),
            "label": label,
            "count": int(edge.get("count", 0) or 0),
            "no_effect": no_effect,
            "verified_ok": int(edge.get("verified_ok", 0) or 0),
            "verified_fail": int(edge.get("verified_fail", 0) or 0),
            "last_ts": edge.get("last_ts"),
            "action_payload": action_payload,
            "occurrences": occurs,
            "state_action_match": edge_match.get("state_action_match") or {},
            "nav_candidate_match": edge_match.get("nav_candidate_match") or {},
        }

    timeline = _enrich_flow_rows(flow_rows, alias)
    timeline.sort(key=lambda x: int(x.get("flow_seq") or 0))

    page_payload = {
        "run_id": str(graph.get("run_id") or run_dir.name),
        "entry_sig": str(graph.get("entry_sig") or ""),
        "cur_sig": str(graph.get("cur_sig") or ""),
        "stop_reason": str(graph.get("stop_reason") or ""),
        "target_package": target_pkg,
        "node_count": len(vis_nodes),
        "edge_count": len(vis_edges),
        "nodes_vis": vis_nodes,
        "edges_vis": vis_edges,
        "node_details": node_detail_map,
        "edge_details": edge_detail_map,
        "timeline": timeline,
        "alias_map": alias,
    }

    _write_action_timeline_md(interactive_dir / "action_timeline.md", page_payload)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UI Transition Interactive - {page_payload['run_id']}</title>
  <script src="{local_vis_js}"></script>
  <script>
    if (!window.vis) {{
      document.write('<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"><\\/script>');
    }}
    if (!window.vis) {{
      document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.0/standalone/umd/vis-network.min.js"><\\/script>');
    }}
  </script>
  <style>
    html, body {{
      margin: 0; padding: 0; width: 100%; height: 100%;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      background: #f5f7fb;
    }}
    #root {{ display: flex; width: 100%; height: 100%; }}
    #left {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
    #toolbar {{
      background: #fff; border-bottom: 1px solid #d9e1ef; padding: 10px 14px;
      display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    }}
    #network {{ flex: 1; background: #f8fbff; }}
    #network {{ min-height: 520px; }}
    #right {{
      width: 460px; max-width: 46vw; min-width: 340px;
      border-left: 1px solid #d9e1ef; background: #fff; overflow: auto;
      padding: 12px 14px;
    }}
    .meta {{ font-size: 12px; color: #5f6b7a; }}
    .k {{ font-weight: 600; color: #23303f; }}
    .card {{ border: 1px solid #e4e9f2; border-radius: 8px; padding: 10px; margin: 10px 0; }}
    .mono {{ font-family: Consolas, Menlo, monospace; font-size: 12px; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }}
    .ok {{ background: #e7f6ec; color: #17653a; }}
    .warn {{ background: #fff2e3; color: #8a5315; }}
    .bad {{ background: #fde8ea; color: #9a2230; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #e5eaf3; padding: 6px; vertical-align: top; text-align: left; }}
    th {{ background: #f6f9ff; }}
    img.preview {{ width: 100%; border: 1px solid #dde5f3; border-radius: 8px; }}
    details > summary {{ cursor: pointer; color: #2d5fa8; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background:#f7f9fc; padding:8px; border-radius:6px; }}
    a {{ color: #2a66b0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <noscript>
    <div style="padding:12px;background:#fde8ea;color:#9a2230;border-bottom:1px solid #e6bfc5;">
      JavaScript is disabled, so the interactive graph cannot be rendered. Open this file in a browser with scripts enabled.
    </div>
  </noscript>
  <div id="root">
    <div id="left">
      <div id="toolbar">
        <div><span class="k">run_id</span>: <span class="mono">{page_payload['run_id']}</span></div>
        <div><span class="k">stop_reason</span>: <span class="mono">{page_payload['stop_reason']}</span></div>
        <div><span class="k">nodes</span>: <span class="mono">{page_payload['node_count']}</span></div>
        <div><span class="k">edges</span>: <span class="mono">{page_payload['edge_count']}</span></div>
        <label><input id="hideNoop" type="checkbox" /> Hide noop edges</label>
      </div>
      <div id="network"></div>
    </div>
    <div id="right">
      <h3 style="margin:4px 0 6px 0;">Details</h3>
      <div class="meta">Click a node or edge to inspect details. Click blank graph space to return to the global timeline.</div>
      <div id="detail"></div>
    </div>
  </div>

  <script>
    const payload = {_safe_json_dumps(page_payload)};
    const nodeDetails = payload.node_details || {{}};
    const edgeDetails = payload.edge_details || {{}};
    const timeline = payload.timeline || [];

    function showRuntimeError(msg) {{
      const detail = document.getElementById('detail');
      const net = document.getElementById('network');
      if (detail) {{
        detail.innerHTML = `<div class="card" style="border-color:#e6bfc5;background:#fff5f6;">
          <div class="k" style="color:#9a2230;">渲染失败</div>
          <div class="mono">${{esc(msg)}}</div>
          <div class="meta" style="margin-top:8px;">建议：使用 Chrome/Edge 打开；或在目录下执行 python -m http.server 后通过 http://127.0.0.1:8765 访问。</div>
        </div>`;
      }}
      if (net) {{
        net.innerHTML = `<div style="padding:14px;color:#9a2230;background:#fff5f6;border:1px solid #e6bfc5;border-radius:8px;margin:12px;">
          交互图未渲染：${{esc(msg)}}
        </div>`;
      }}
    }}

    function esc(s) {{
      if (s === null || s === undefined) return '';
      return String(s)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function renderPathLink(label, p) {{
      if (!p) return `<div><span class="k">${{esc(label)}}:</span> -</div>`;
      return `<div><span class="k">${{esc(label)}}:</span> <a class="mono" href="${{esc(p)}}" target="_blank">${{esc(p)}}</a></div>`;
    }}

    function renderNode(sig) {{
      const d = nodeDetails[sig];
      if (!d) return '<div class="card">节点详情不存在。</div>';
      const nav = d.nav_result || {{}};
      const router = d.router_observation || {{}};
      const blocks = d.blocks_observation || {{}};
      const cand = d.state_action || {{}};
      const cands = cand.candidates || [];

      const tag = d.foreground_package && d.target_package && d.foreground_package !== d.target_package
        ? `<span class="pill bad">外部包: ${{esc(d.foreground_package)}}</span>`
        : `<span class="pill ok">目标包: ${{esc(d.foreground_package || '-')}}</span>`;

      let html = '';
      html += `<div class="card"><div class="k">节点</div><div class="mono">${{esc(d.alias)}} | ${{esc(d.state_sig)}}</div>`;
      html += `<div style="margin-top:6px">${{tag}}`;
      html += `<span class="pill warn">overlay=${{esc(d.overlay_kind)}}</span>`;
      html += `<span class="pill warn">visit=${{esc(d.visit_count)}}</span>`;
      html += `<span class="pill warn">remaining=${{esc((d.candidate_summary || {{}}).remaining_count)}}</span>`;
      html += `</div></div>`;

      const sp = d.snapshot_paths || {{}};
      html += `<div class="card"><div class="k">快照文件</div>`;
      if (sp.preview) {{
        html += `<div style="margin:8px 0"><img class="preview" src="${{esc(sp.preview)}}" /></div>`;
      }}
      html += renderPathLink('screenshot', sp.screenshot_path);
      html += renderPathLink('screenshot_raw', sp.screenshot_raw_path);
      html += renderPathLink('xml', sp.xml_path);
      html += renderPathLink('xml_raw', sp.xml_raw_path);
      html += renderPathLink('uist', sp.uist_path);
      html += renderPathLink('uist_overlay', sp.uist_overlay_path);
      html += renderPathLink('vidmap_overlay', sp.vidmap_overlay_path);
      html += `</div>`;

      html += `<div class="card"><div class="k">LLM NAV 结果</div>`;
      html += `<div>overlay_kind: <span class="mono">${{esc(nav.overlay_kind ?? '-')}}</span></div>`;
      html += `<div>overlay_reason: <span class="mono">${{esc(nav.overlay_reason ?? '-')}}</span></div>`;
      html += `<div>candidate_count: <span class="mono">${{esc((nav.candidate_actions || []).length)}}</span></div>`;
      html += `<div>exhausted: <span class="mono">${{esc(nav.exhausted)}}</span>, confidence: <span class="mono">${{esc(nav.exhausted_confidence)}}</span></div>`;
      html += `<div>why_these_actions: <span class="mono">${{esc(nav.why_these_actions ?? '')}}</span></div>`;
      if (cands.length) {{
        html += `<details style="margin-top:8px" open><summary>候选动作（state_action_snapshot）</summary><table><thead><tr><th>key</th><th>score</th><th>action</th><th>reason</th></tr></thead><tbody>`;
        for (const c of cands) {{
          const a = (c.actions || [])[0] || {{}};
          html += `<tr><td class="mono">${{esc(c.candidate_key)}}</td><td>${{esc(c.score)}}</td><td class="mono">${{esc((a.action || '') + ':' + (a.element_id ?? 'None'))}}</td><td>${{esc(a.reasoning || '')}}</td></tr>`;
        }}
        html += `</tbody></table></details>`;
      }}
      html += `</div>`;

      html += `<div class="card"><div class="k">VID Map 摘要</div>`;
      html += `<pre>${{esc(JSON.stringify(d.vid_map_summary || {{}}, null, 2))}}</pre>`;
      html += `</div>`;

      html += `<div class="card"><div class="k">Snapshot 元信息</div>`;
      html += `<pre>${{esc(JSON.stringify(d.snapshot_meta || {{}}, null, 2))}}</pre>`;
      html += `<div class="meta">说明：当前 run 的 snapshot 未落盘完整 snap 对象，因此这里展示的是已落盘字段。</div>`;
      html += `</div>`;

      html += `<div class="card"><div class="k">Router / Blocks 观测</div>`;
      html += `<div>router_answers: <span class="mono">${{esc((router.router_answers || []).length)}}</span></div>`;
      html += `<div>matched_block_ids: <span class="mono">${{esc((router.matched_block_ids || []).length)}}</span></div>`;
      html += `<div>block_fill_results: <span class="mono">${{esc((blocks.block_fill_results || []).length)}}</span></div>`;
      html += `</div>`;

      html += `<details class="card"><summary>Raw JSON</summary><pre>${{esc(JSON.stringify(d, null, 2))}}</pre></details>`;
      return html;
    }}

    function renderEdge(edgeId) {{
      const d = edgeDetails[edgeId];
      if (!d) return '<div class="card">边详情不存在。</div>';
      let html = '';
      html += `<div class="card"><div class="k">边</div>`;
      html += `<div class="mono">${{esc(d.id)}}: ${{esc(d.src_alias)}} -> ${{esc(d.dst_alias)}}</div>`;
      html += `<div>label: <span class="mono">${{esc(d.label)}}</span></div>`;
      html += `<div>count=${{esc(d.count)}}, no_effect=${{esc(d.no_effect)}}, verified_ok=${{esc(d.verified_ok)}}, verified_fail=${{esc(d.verified_fail)}}</div>`;
      html += `</div>`;

      const sm = d.state_action_match || {{}};
      const nm = d.nav_candidate_match || {{}};
      html += `<div class="card"><div class="k">动作来源与原因</div>`;
      if (Object.keys(sm).length) {{
        html += `<div><span class="k">state_action match</span></div>`;
        html += `<div>candidate_key: <span class="mono">${{esc(sm.candidate_key)}}</span></div>`;
        html += `<div>score: <span class="mono">${{esc(sm.score)}}</span></div>`;
        html += `<div>reasoning: <span class="mono">${{esc(sm.reasoning || '-')}}</span></div>`;
      }} else {{
        html += `<div class="meta">state_action_snapshot 中未匹配到对应候选。</div>`;
      }}
      if (Object.keys(nm).length) {{
        html += `<div style="margin-top:6px"><span class="k">nav candidate match</span></div>`;
        html += `<div>score: <span class="mono">${{esc(nm.score)}}</span>, return_method: <span class="mono">${{esc(nm.return_method || '-')}}</span></div>`;
        html += `<div>reasoning: <span class="mono">${{esc(nm.reasoning || '-')}}</span></div>`;
      }} else {{
        html += `<div class="meta">observations/nav 中未匹配到对应候选。</div>`;
      }}
      html += `</div>`;

      const occ = d.occurrences || [];
      if (occ.length) {{
        html += `<div class="card"><div class="k">执行序列（来自 trace）</div>`;
        html += `<table><thead><tr><th>seq</th><th>action</th><th>element_id</th><th>change_type</th><th>conf</th></tr></thead><tbody>`;
        for (const x of occ) {{
          html += `<tr><td>${{esc(x.seq)}}</td><td class="mono">${{esc(x.action)}}</td><td>${{esc(x.element_id)}}</td><td>${{esc(x.change_type)}}</td><td>${{esc(x.confidence)}}</td></tr>`;
        }}
        html += `</tbody></table></div>`;
      }}

      html += `<details class="card"><summary>Raw Action Payload</summary><pre>${{esc(JSON.stringify(d.action_payload || {{}}, null, 2))}}</pre></details>`;
      return html;
    }}

    // Input: global timeline rows from the trace payload. Output: HTML summary table for human debugging.
    function renderTimeline() {{
      let html = '<div class="card"><div class="k">Global Action Timeline</div>';
      if (!timeline.length) return html + '<div class="meta">No timeline data.</div></div>';
      html += '<table><thead><tr><th>#</th><th>Source</th><th>Action</th><th>Result</th><th>Confidence</th></tr></thead><tbody>';
      for (const x of timeline) {{
        const conf = (x.confidence === null || x.confidence === undefined) ? '' : x.confidence;
        html += `<tr>
          <td>${{esc(x.flow_seq)}}</td>
          <td class="mono">${{esc(x.src_alias || '-')}}</td>
          <td><span class="mono">${{esc(x.action_display || x.action || '')}}</span></td>
          <td>${{esc(x.result_display || x.result || x.change_type || '')}}</td>
          <td>${{esc(conf)}}</td>
        </tr>`;
      }}
      html += '</tbody></table></div>';
      return html;
    }}

    function setDetail(html) {{
      document.getElementById('detail').innerHTML = html;
    }}
    try {{
      if (!window.vis || !window.vis.DataSet || !window.vis.Network) {{
        throw new Error('vis-network 未加载（window.vis 不可用）');
      }}
      const allNodes = new vis.DataSet(payload.nodes_vis || []);
      const allEdges = new vis.DataSet(payload.edges_vis || []);
      let activeEdgeIds = (payload.edges_vis || []).map(e => e.id);

      const network = new vis.Network(
        document.getElementById('network'),
        {{ nodes: allNodes, edges: allEdges }},
        {{
          autoResize: true,
          interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
          physics: {{
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{ gravitationalConstant: -55, springLength: 180, springConstant: 0.06 }},
            stabilization: {{ iterations: 220, fit: true }}
          }},
          layout: {{ improvedLayout: true }},
          edges: {{ smooth: {{ type: 'dynamic' }} }},
        }}
      );

      network.once('stabilizationIterationsDone', function() {{
        try {{ network.fit({{ animation: false }}); }} catch (e) {{}}
      }});

      network.on('click', function (params) {{
        // Input: vis-network click event. Output: right-panel HTML; node/edge details are shown above the timeline.
        if (params.nodes && params.nodes.length > 0) {{
          setDetail(renderNode(params.nodes[0]) + renderTimeline());
          return;
        }}
        if (params.edges && params.edges.length > 0) {{
          setDetail(renderEdge(params.edges[0]) + renderTimeline());
          return;
        }}
        setDetail(renderTimeline());
      }});

      document.getElementById('hideNoop').addEventListener('change', function (e) {{
        const hide = !!e.target.checked;
        allEdges.clear();
        if (!hide) {{
          allEdges.add(payload.edges_vis || []);
          activeEdgeIds = (payload.edges_vis || []).map(e => e.id);
        }} else {{
          const filtered = (payload.edges_vis || []).filter(e => {{
            const d = edgeDetails[e.id] || {{}};
            return !d.no_effect;
          }});
          allEdges.add(filtered);
          activeEdgeIds = filtered.map(e => e.id);
        }}
      }});

      setDetail(renderTimeline());
    }} catch (err) {{
      showRuntimeError((err && err.message) ? err.message : String(err));
    }}
  </script>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Render interactive HTML UI transition graph for one run.")
    parser.add_argument("--run-dir", required=True, help="Run directory under traces, e.g. traces/20260513_211332_bim.app")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    out_html = build_interactive_html(run_dir)
    print(f"[OK] interactive html: {out_html}")


if __name__ == "__main__":
    main()
