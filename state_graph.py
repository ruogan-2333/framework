# state_graph.py (patched; compatible with your "updated" version)

"""
state_graph.py

ROLE:
- Store navigation states (state signatures) and transitions (edges) with replayable action payloads.
- Support best-effort replay after restart via shortest_action_path(src, dst) -> (state_path, action_path).

CRITICAL SEMANTICS (workflow depends on these; do NOT re-implement elsewhere):
1) add_edge() touches src/dst (visit_count++).
2) Therefore, "dst was new at discovery" MUST be computed BEFORE add_edge().
   => record_transition() returns dst_was_new_at_discovery (bool).
3) We must sometimes attach page_kind/meta WITHOUT incrementing visit_count.
   => annotate() updates node metadata/flags without counting a "visit".

WHEN USED IN WORKFLOW:
- record_observation(sig): called when we are at a state without a clean predecessor edge
  (entry, post-restart landing, post-recovery reconciliation).
- record_transition(src,dst,action): called for EVERY executed UI action that yields a new snapshot.
  Action payloads are stored as {"actions": [ ... ]} (single-step is length-1 list).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


@dataclass
class Edge:
    # Directed transition labeled with the executed action(s); count tracks stability for replay.
    # action must be a dict with "actions": [ ... ] (single-step stored as length-1 list).
    src: str
    dst: str
    action: Dict[str, Any] = field(default_factory=dict)
    count: int = 0
    last_ts: float = 0.0
    no_effect: bool = False
    # Replay verification stats (separate from discovery count).
    verified_ok: int = 0
    verified_fail: int = 0
    last_verified_ts: float = 0.0


@dataclass
class Node:
    # State signature node with visit metadata and page-shape metadata.
    sig: str
    visit_count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    page_kind: str = "stable"
    page_tags: Set[str] = field(default_factory=set)
    meta: Dict[str, Any] = field(default_factory=dict)

    outgoing: Set[str] = field(default_factory=set)
    incoming: Set[str] = field(default_factory=set)


@dataclass
class TextUTGContext:
    """
    Text export result for feeding the current StateGraph into an LLM prompt.

    Input:
    - StateGraph nodes and edges plus current/home/parent state signatures.

    Output:
    - text: compact human-readable UTG text.
    - aliases: state signature to UI label map, such as {"xml:abc": "UI1"}.
    - home_paths: state signature to semantic path from home.

    Function:
    - Keeps prompt-facing UTG context structured without creating a second graph model.
    """

    text: str
    aliases: Dict[str, str] = field(default_factory=dict)
    home_paths: Dict[str, str] = field(default_factory=dict)
    home_sig: Optional[str] = None
    current_sig: Optional[str] = None
    parent_sig: Optional[str] = None


class StateGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        # key: (src, dst, action_key)
        self.edges: Dict[Tuple[str, str, str], Edge] = {}
        # adjacency for path search
        self._adj: Dict[str, List[Edge]] = defaultdict(list)

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "StateGraph":
        """
        Rebuild a StateGraph from graph/state_graph_snapshot.json.

        Input:
        - snapshot: JSON object with "nodes" and "edges" fields from a saved run.

        Output:
        - StateGraph containing the saved nodes, edges, metadata, and adjacency.

        Function:
        - Lets offline debug scripts reuse StateGraph methods without duplicating UTG parsing.
        """
        graph = cls()
        nodes = snapshot.get("nodes") if isinstance(snapshot, Mapping) else {}
        if isinstance(nodes, Mapping):
            for sig, raw in nodes.items():
                if not sig:
                    continue
                row = raw if isinstance(raw, Mapping) else {}
                node = Node(
                    sig=str(sig),
                    visit_count=int(row.get("visit_count", 0) or 0),
                    first_ts=float(row.get("first_ts", 0.0) or 0.0),
                    last_ts=float(row.get("last_ts", 0.0) or 0.0),
                    page_kind=str(row.get("page_kind", "stable") or "stable"),
                    page_tags=cls._norm_page_tags(row.get("page_tags") or []),
                    meta=dict(row.get("meta") or {}) if isinstance(row.get("meta"), Mapping) else {},
                )
                graph.nodes[str(sig)] = node

        edges = snapshot.get("edges") if isinstance(snapshot, Mapping) else []
        if isinstance(edges, list):
            for raw_edge in edges:
                if not isinstance(raw_edge, Mapping):
                    continue
                src = str(raw_edge.get("src") or "")
                dst = str(raw_edge.get("dst") or "")
                if not src or not dst:
                    continue
                action = raw_edge.get("action") if isinstance(raw_edge.get("action"), Mapping) else {}
                edge = Edge(
                    src=src,
                    dst=dst,
                    action=dict(action),
                    count=int(raw_edge.get("count", 0) or 0),
                    last_ts=float(raw_edge.get("last_ts", 0.0) or 0.0),
                    no_effect=bool(raw_edge.get("no_effect", False)),
                    verified_ok=int(raw_edge.get("verified_ok", 0) or 0),
                    verified_fail=int(raw_edge.get("verified_fail", 0) or 0),
                    last_verified_ts=float(raw_edge.get("last_verified_ts", 0.0) or 0.0),
                )
                if src not in graph.nodes:
                    graph.nodes[src] = Node(sig=src)
                if dst not in graph.nodes:
                    graph.nodes[dst] = Node(sig=dst)
                graph.nodes[src].outgoing.add(dst)
                graph.nodes[dst].incoming.add(src)
                key = (src, dst, graph._action_key(edge.action))
                graph.edges[key] = edge
                graph._adj[src].append(edge)
        return graph

    # -------------------------
    # Introspection helpers
    # -------------------------

    @staticmethod
    def _norm_page_kind(page_kind: Optional[str]) -> str:
        """
        Normalize page_kind values stored on StateGraph nodes.

        Input:
        - page_kind: raw string or enum-like object from LLM schema.

        Output:
        - stable/popup/loading string, defaulting to stable when empty.

        Function:
        - Keeps StateGraph independent from the Pydantic enum class.
        """
        if not page_kind:
            return "stable"
        if hasattr(page_kind, "value"):
            return str(getattr(page_kind, "value"))
        return str(page_kind)

    @staticmethod
    def _norm_page_tags(page_tags: Optional[List[Any]]) -> Set[str]:
        """
        Normalize page_tags values stored on StateGraph nodes.

        Input:
        - page_tags: raw strings or enum-like objects from LLM schema/workflow.

        Output:
        - Deduplicated non-empty string set.

        Function:
        - Keeps StateGraph independent from the Pydantic PageTag enum class.
        """
        out: Set[str] = set()
        for item in page_tags or []:
            value = getattr(item, "value", item)
            text = str(value or "").strip()
            if text:
                out.add(text)
        return out

    def has_state(self, sig: str) -> bool:
        """
        IPO:
          in : sig
          out: True if node exists
        WHEN called (workflow):
          - before record_transition() to reason about novelty at discovery time
        """
        return sig in self.nodes

    def get_node(self, sig: str) -> Optional[Node]:
        return self.nodes.get(sig)

    # -------------------------
    # Node touch / annotate
    # -------------------------

    def touch(
        self,
        sig: str,
        page_kind: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        page_tags: Optional[List[Any]] = None,
    ) -> Node:
        """
        IPO:
          in : sig; optional page_kind/meta
          out: Node (created if needed) with visit_count incremented
        WHEN called:
          - record_observation()
          - add_edge() (for src and dst)
        """
        now = time.time()
        n = self.nodes.get(sig)
        if n is None:
            n = Node(sig=sig, visit_count=0, first_ts=now, last_ts=now, page_kind=self._norm_page_kind(page_kind), meta=meta or {})
            self.nodes[sig] = n
        n.visit_count += 1
        n.last_ts = now
        if page_kind is not None:
            n.page_kind = self._norm_page_kind(page_kind)
        tags = self._norm_page_tags(page_tags)
        if tags:
            n.page_tags.update(tags)
        if meta:
            n.meta.update(meta)
        return n

    def annotate(
        self,
        sig: str,
        page_kind: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        page_tags: Optional[List[Any]] = None,
    ) -> None:
        """
        IPO:
          in : sig; optional page_kind/meta
          out: updates node flags/meta WITHOUT incrementing visit_count
        WHEN called (workflow):
          - after receiving NAV results (LLM1) to mark page_kind
          - whenever we want to attach extra metadata to existing nodes
        WHY:
          - add_edge/touch already counts visits; we must not inflate visit_count just to store flags.
        """
        now = time.time()
        n = self.nodes.get(sig)
        if n is None:
            # If node doesn't exist, we DO want to create it, but still do NOT count as a visit.
            n = Node(
                sig=sig,
                visit_count=0,
                first_ts=now,
                last_ts=now,
                page_kind=self._norm_page_kind(page_kind),
                page_tags=self._norm_page_tags(page_tags),
                meta=meta or {},
            )
            self.nodes[sig] = n
            return
        n.last_ts = now
        if page_kind is not None:
            n.page_kind = self._norm_page_kind(page_kind)
        tags = self._norm_page_tags(page_tags)
        if tags:
            n.page_tags.update(tags)
        if meta:
            n.meta.update(meta)

    def annotate_meta(self, sig: str, **meta: Any) -> None:
        """
        Attach metadata to a state node without incrementing its visit count.

        Input:
        - sig: state signature to annotate.
        - meta: JSON-safe metadata fields such as page_summary.

        Output:
        - None; the node metadata is updated in-place.

        Function:
        - Provides a small explicit wrapper so workflow code does not manipulate Node.meta directly.
        """
        self.annotate(sig, meta=dict(meta or {}))

    @staticmethod
    def _clip_text(value: Any, limit: int) -> str:
        """
        Convert a value to a one-line clipped string.

        Input:
        - value: arbitrary text-like value.
        - limit: maximum output length.

        Output:
        - One-line string clipped to limit characters.

        Function:
        - Keeps generated UTG prompt context compact and stable.
        """
        text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
        text = " ".join(text.split())
        if limit > 0 and len(text) > limit:
            return text[: max(0, limit - 1)].rstrip() + "…"
        return text

    def _action_text_for_context(self, action: Mapping[str, Any], limit: int = 120) -> str:
        """
        Build a short human-readable action label for UTG text.

        Input:
        - action: stored edge action payload, usually {"actions": [step, ...]}.
        - limit: maximum length for each step summary.

        Output:
        - Text such as "click Personal" or "click Settings ; click OK".

        Function:
        - Converts replay action payloads into semantic edge labels for LLM context.
        """
        steps = action.get("actions") if isinstance(action, Mapping) else []
        if not isinstance(steps, list):
            steps = []
        parts: List[str] = []
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            verb = self._clip_text(step.get("action") or step.get("type") or "action", 24)
            label = (
                step.get("action_summary")
                or step.get("semantic_action")
                or step.get("anchor_label")
                or step.get("label")
                or step.get("text")
                or step.get("content_desc")
                or step.get("resource_id")
                or step.get("reasoning")
                or ""
            )
            label_text = self._clip_text(label, limit)
            parts.append(f"{verb} {label_text}".strip())
        if not parts:
            return "unknown action"
        return " ; ".join(parts)

    def _home_paths_for_context(self, home_sig: Optional[str]) -> Dict[str, str]:
        """
        Compute semantic paths from the home UI to every known node.

        Input:
        - home_sig: state signature currently treated as the app home page.

        Output:
        - Mapping from state signature to path text.

        Function:
        - Gives the LLM a compact branch-position cue without adding extra schema fields.
        """
        if not home_sig or home_sig not in self.nodes:
            return {sig: "unknown_home" for sig in self.nodes.keys()}

        paths: Dict[str, str] = {home_sig: "home"}
        q = deque([home_sig])
        while q:
            cur = q.popleft()
            edges = sorted(self._adj.get(cur, []), key=lambda e: e.count, reverse=True)
            for edge in edges:
                nxt = edge.dst
                if nxt in paths:
                    continue
                action_text = self._action_text_for_context(edge.action)
                paths[nxt] = f"{paths[cur]} > {action_text}"
                q.append(nxt)

        for sig in self.nodes.keys():
            paths.setdefault(sig, "unreachable_from_home")
        return paths

    def build_text_utg_context(
        self,
        current_sig: Optional[str],
        home_sig: Optional[str],
        parent_sig: Optional[str] = None,
        max_nodes: int = 40,
        max_edges: int = 80,
        summary_chars: int = 160,
    ) -> TextUTGContext:
        """
        Export a compact text UTG context for the navigation/router LLM.

        Input:
        - current_sig: state currently being analyzed.
        - home_sig: best-known app home state, or None when unknown.
        - parent_sig: direct predecessor state when known.
        - max_nodes/max_edges/summary_chars: prompt-size controls.

        Output:
        - TextUTGContext containing text plus alias and home-path maps.

        Function:
        - Lets LLM see known UI graph position, node summaries, edges, and home paths.
        """
        special = [sig for sig in (home_sig, parent_sig, current_sig) if sig]
        selected: List[str] = []
        for sig in self.nodes.keys():
            if len(selected) >= max_nodes and sig not in special:
                continue
            if sig not in selected:
                selected.append(sig)
        for sig in special:
            if sig and sig in self.nodes and sig not in selected:
                selected.append(sig)

        aliases = {sig: f"UI{idx + 1}" for idx, sig in enumerate(selected)}
        all_home_paths = self._home_paths_for_context(home_sig)
        home_paths = {sig: all_home_paths.get(sig, "unreachable_from_home") for sig in selected}

        def alias_or_unknown(sig: Optional[str]) -> str:
            if not sig:
                return "unknown"
            return aliases.get(sig, sig[:12])

        lines: List[str] = [
            f"HOME: {alias_or_unknown(home_sig)}",
            f"CURRENT: {alias_or_unknown(current_sig)}",
            f"PARENT: {alias_or_unknown(parent_sig)}",
            "",
            "Nodes:",
        ]
        for sig in selected:
            node = self.nodes.get(sig)
            summary = "no page summary"
            page_kind = "stable"
            if node is not None:
                summary = self._clip_text((node.meta or {}).get("page_summary") or "no page summary", summary_chars)
                page_kind = str(node.page_kind or "stable")
            page_tags: List[str] = []
            if node is not None:
                page_tags = sorted(getattr(node, "page_tags", set()) or [])
            tags_text = f", tags={','.join(page_tags)}" if page_tags else ""
            lines.append(f"- {aliases[sig]}: {summary} [sig={sig}, page_kind={page_kind}{tags_text}]")

        lines.extend(["", "Edges:"])
        edge_count = 0
        selected_set = set(selected)
        for edge in sorted(self.edges.values(), key=lambda e: (-int(e.count or 0), str(e.src), str(e.dst))):
            if edge.src not in selected_set or edge.dst not in selected_set:
                continue
            if edge_count >= max_edges:
                break
            edge_count += 1
            action_text = self._action_text_for_context(edge.action)
            lines.append(f"- {aliases[edge.src]} -> {aliases[edge.dst]}: {action_text} [count={int(edge.count or 0)}]")
        omitted_edges = max(0, len(self.edges) - edge_count)
        if omitted_edges:
            lines.append(f"- ... omitted_edges={omitted_edges}")

        lines.extend(["", "Home paths:"])
        for sig in selected:
            lines.append(f"- {aliases[sig]}: {home_paths.get(sig, 'unreachable_from_home')}")
        omitted_nodes = max(0, len(self.nodes) - len(selected))
        if omitted_nodes:
            lines.append(f"- ... omitted_nodes={omitted_nodes}")

        return TextUTGContext(
            text="\n".join(lines),
            aliases=aliases,
            home_paths=home_paths,
            home_sig=home_sig,
            current_sig=current_sig,
            parent_sig=parent_sig,
        )

    # -------------------------
    # Edges / transitions
    # -------------------------

    def _action_key(self, action: Dict[str, Any]) -> str:
        """
        Build a stable-ish key for an action payload.
        Expects action dicts with an `actions` list (single-step => list of len 1).
        """
        if not action:
            return "None:None:"

        parts: List[str] = []
        for a in action.get("actions") or []:
            parts.append(f"{a.get('action')}:{a.get('element_id')}:{a.get('text') or ''}")
        return "||".join(parts) or "None:None:"

    def action_key(self, action: Dict[str, Any]) -> str:
        """
        Public wrapper for building an action payload key.
        """
        return self._action_key(action)

    def get_edge(self, src: str, dst: str, action: Dict[str, Any]) -> Optional[Edge]:
        """
        Look up the exact stored edge instance for (src,dst,action_key).
        Returns None if not found.
        """
        try:
            akey = self._action_key(action)
            return self.edges.get((src, dst, akey))
        except Exception:
            return None

    def record_edge_verification(self, src: str, dst: str, action: Dict[str, Any], *, ok: bool) -> None:
        """
        Update replay verification stats for a specific stored edge.
        Intended to be called by workflow replay/navigation when an edge is used as part of a plan.
        """
        e = self.get_edge(src, dst, action)
        if e is None:
            return
        now = time.time()
        if ok:
            e.verified_ok += 1
            e.last_verified_ts = now
        else:
            e.verified_fail += 1

    def add_edge(self, src: str, dst: str, action: Dict[str, Any]) -> None:
        """
        IPO:
          in : src,dst,action_dict
          out: edge recorded; edge.count++ ; src/dst nodes touched (visit_count++)
        WHEN called:
          - record_transition() (workflow should call record_transition, not add_edge directly)
        """
        akey = self._action_key(action)
        k = (src, dst, akey)
        e = self.edges.get(k)
        now = time.time()
        if e is None:
            e = Edge(src=src, dst=dst, action=dict(action), count=0, last_ts=now, no_effect=(src == dst))
            self.edges[k] = e
            self._adj[src].append(e)
        e.count += 1
        e.last_ts = now
        e.no_effect = (src == dst)

        self.touch(src)
        self.touch(dst)
        self.nodes[src].outgoing.add(dst)
        self.nodes[dst].incoming.add(src)

    def add_edge_no_touch(self, src: str, dst: str, action: Dict[str, Any]) -> None:
        """
        Record an edge WITHOUT incrementing src/dst visit_count.

        WHY:
          - Probe/return edges can be very frequent and will otherwise drown visit_count-based
            frontier heuristics, while still being useful for replay graph connectivity.
        """
        akey = self._action_key(action)
        k = (src, dst, akey)
        e = self.edges.get(k)
        now = time.time()
        if e is None:
            e = Edge(src=src, dst=dst, action=dict(action), count=0, last_ts=now, no_effect=(src == dst))
            self.edges[k] = e
            self._adj[src].append(e)
        e.count += 1
        e.last_ts = now
        e.no_effect = (src == dst)

        # Ensure nodes exist but do not count as "visits".
        if src not in self.nodes:
            self.nodes[src] = Node(sig=src, visit_count=0, first_ts=now, last_ts=now)
        else:
            self.nodes[src].last_ts = now
        if dst not in self.nodes:
            self.nodes[dst] = Node(sig=dst, visit_count=0, first_ts=now, last_ts=now)
        else:
            self.nodes[dst].last_ts = now

        self.nodes[src].outgoing.add(dst)
        self.nodes[dst].incoming.add(src)

    def record_observation(
        self,
        sig: str,
        page_kind: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        page_tags: Optional[List[Any]] = None,
    ) -> Node:
        """
        IPO:
          in : sig observed without a clean predecessor action edge
          out: touch(sig) -> increments visit_count exactly once
        WHEN called (workflow):
          - entry state (no predecessor)
          - post-restart landing state
          - post-recovery reconciliation when we can't safely label the move as an action edge
        """
        return self.touch(sig, page_kind=page_kind, meta=meta, page_tags=page_tags)

    def record_transition(
        self,
        src: str,
        dst: str,
        action: Dict[str, Any],
        *,
        touch: bool = True,
        dst_page_kind: Optional[str] = None,
        dst_meta: Optional[Dict[str, Any]] = None,
        src_page_kind: Optional[str] = None,
        src_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        IPO:
          in : (src, dst, action_dict) and optional page_kind/meta annotations
               action_dict may represent a single step or include an `actions` list for multi-step probes.
          out: dst_was_new_at_discovery (bool)

        WHEN called (workflow):
          - EVERY time we execute something and then capture a new authoritative snapshot
            (probe, forward, recovery step, replay step, drift).

        CRITICAL:
          - dst_was_new_at_discovery is computed BEFORE add_edge(), because add_edge touches nodes.
        """
        dst_was_new = (dst not in self.nodes)
        if touch:
            self.add_edge(src, dst, action)
        else:
            self.add_edge_no_touch(src, dst, action)

        # annotate WITHOUT incrementing (avoid double-touch inflation)
        if src_page_kind is not None or src_meta:
            self.annotate(src, page_kind=src_page_kind, meta=src_meta)
        if dst_page_kind is not None or dst_meta:
            self.annotate(dst, page_kind=dst_page_kind, meta=dst_meta)

        return dst_was_new

    # -------------------------
    # Frontier / replay
    # -------------------------

    def is_new_state(self, sig: str) -> bool:
        # Kept for compatibility, but note:
        # - With add_edge touching nodes, this reflects "currently rare", not "novel at discovery time".
        return sig not in self.nodes or self.nodes[sig].visit_count <= 1

    def frontier_states(self, max_items: int = 10) -> List[str]:
        # Heuristic frontier: prefer rarely visited, low-degree, recently seen states.
        scored: List[Tuple[float, str]] = []
        now = time.time()
        for sig, n in self.nodes.items():
            out_deg = len(n.outgoing)
            age = now - n.last_ts
            score = 0.0
            score += 6.0 / max(1.0, float(n.visit_count))
            score += 5.0 / max(1.0, float(out_deg + 1))
            score += 1.5 / max(1.0, float(age / 60.0 + 1.0))
            scored.append((score, sig))
        scored.sort(reverse=True)
        return [s for _, s in scored[:max_items]]

    def frontier_hint(self) -> str:
        front = self.frontier_states(8)
        if not front:
            return "No frontier; likely stuck or saturated."
        return "Frontier: " + ", ".join([f[:8] for f in front])

    def shortest_action_path(self, src: str, dst: str, max_depth: int = 25) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        BFS over states, storing predecessor + edge action.

        Returns:
          (state_path, action_path)
        action_path length = len(state_path)-1

        Used for best-effort replay after app restart; edges with higher counts are preferred first.
        """
        if src == dst:
            return ([src], [])

        if src not in self.nodes or dst not in self.nodes:
            return ([], [])

        q = deque([src])
        prev: Dict[str, Optional[str]] = {src: None}
        prev_action: Dict[str, Optional[Dict[str, Any]]] = {src: None}
        depth: Dict[str, int] = {src: 0}

        while q:
            cur = q.popleft()
            if cur == dst:
                break
            if depth[cur] >= max_depth:
                continue

            edges = sorted(self._adj.get(cur, []), key=lambda e: e.count, reverse=True)
            for e in edges:
                nxt = e.dst
                if nxt not in prev:
                    prev[nxt] = cur
                    prev_action[nxt] = dict(e.action)
                    depth[nxt] = depth[cur] + 1
                    q.append(nxt)

        if dst not in prev:
            return ([], [])

        states: List[str] = []
        actions: List[Dict[str, Any]] = []
        cur = dst
        while cur is not None:
            states.append(cur)
            act = prev_action.get(cur)
            if act is not None:
                actions.append(act)
            cur = prev.get(cur)
        states.reverse()
        actions.reverse()
        return (states, actions)
