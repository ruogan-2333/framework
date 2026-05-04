# workflow.py (patched; full; uses StateGraph record_* helpers; custom return supported)

"""
workflow.py

GOAL (non-negotiable):
- Fully explore the app UI (navigation) to collect enough evidence to COMPLETE a fixed questionnaire.
- Exploration is purposeful: each navigation choice is driven by remaining "open gaps" in the questionnaire.

Core design constraints (robustness over cleverness):
- Decisions are only made on authoritative snapshots (xml+png captured together).
- LLM calls are slow; UI actions are fast; pipeline LLM calls with a thread pool.
- Never "guess click" unless NAV truly failed/timed-out (NAV barrier).
- Navigation is "complete": exhaust all candidates in a state, then backtrace (DFS-style).
- Overlays/popups are resolved first; if not resolvable -> recovery -> restart+replay -> replan.

Critical StateGraph alignment (must match patched state_graph.py):
- graph.add_edge() touches src/dst (increments visit_count).
- Therefore: "dst is new" must be computed BEFORE recording edge.
- Workflow does NOT duplicate that semantic:
  => We call graph.record_transition() which returns dst_was_new_at_discovery.
- Overlay/meta updates must NOT inflate visit_count:
  => We call graph.annotate() for flags.

CONDITIONS explicitly handled:
- DRIFT: UI changes without our logged action -> record drift edge, replan at new state.
- OVERLAY: blocking popup/dialog -> LLM1 overlay_dismiss_actions, else recovery, else restart+replay.
- BACK-LIKE: a click lands in an ancestor state -> do NOT probe-return; replan at that ancestor.
- NAV timeout: LLM1 not ready -> cancel best-effort; only then heuristics.
- EXHAUSTED: state has no remaining unprobed candidates -> backtrace to nearest ancestor with work.
- STUCK: progress watchdog (time/loops without progress) -> recovery then restart fallback.

IMPORTANT subtlety (tabs / per-candidate return):
- Different probes on the same page can require different unwind strategies (tab click vs close vs back).
- LLM1 now attaches return_method/return_actions per candidate; global defaults are only fallbacks.
- Workflow executes a candidate's return_actions FIRST; BACK is a LAST resort for tab-back/custom contexts.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import io
import json
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from collections import OrderedDict, deque

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat

from appium_android import AndroidAppiumClient
from gpt_cls import (
    ActionCandidate,
    ActionStep,
    ActionType,
    BlocksFillResult,
    GPTClient,
    NavigationProposal,
    OverlayKind,
    QuestionnaireUpdate,
    RecoveryProposal,
    RouterResult,
    TopicRouteResult,
    UIView,
)
from questionnaire_state2 import QuestionnaireState as QuestionnaireState2
from state_graph import StateGraph
from ui_cls import BaseUI
from trace_callbacks import Callbacks, StepCtx, NoOpCallbacks

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    # End conditions (checked at top of each main-loop iteration)
    time_budget_s: float = 300.0
    max_actions: int = 300
    saturation_limit: int = 30              # too many loops with no novel states => stop
    stuck_time_s: float = 60.0              # time since last progress -> recover
    strong_stall_stop_s: float = 180.0      # hard stop if no strong progress for this long
    stuck_loops_limit: int = 18             # consecutive no-progress loops -> recover (if NAV ready)

    # Per-page probing (bounded per loop; across loops we exhaust remaining candidates)
    enable_probe_return: bool = True
    per_page_probe_cap: int = 10
    post_action_settle_s: float = 0.6

    # NAV barrier (wait + drift detection + timeout fallback)
    nav_timeout_s: float = 120.0  # LLM1 wait timeout before heuristic fallback
    nav_poll_interval_s: float = 0.08
    nav_drift_check_interval_s: float = 0.9# 每隔多久进行一次drift检测
    nav_cooldown_s: float = 6.0
    loading_wait_s: float = 6.0
    topic_route_conf_threshold: float = 0.55
    topic_fill_cooldown_s: float = 60.0
    topic_pack_limit: int = 18
    screenshot_phash_similarity_threshold: float = 0.80# phash相似度判断阈值
    meaningful_xml_nodes_threshold: int = 2  #计算xml中有意义节点数阈值,用于判断xml是否可信
    visual_probe_max_taps: int = 6
    visual_probe_settle_s: float = 0.25
    visual_probe_roi_delta_threshold: float = 0.035

    # Optional short wait after probing to let pipelined analysis land
    post_probe_wait_s: float = 1.2

    # Forward scoring weights
    cand_score_weight: float = 3.0  # LLM candidate score in [-1,1] -> [-3,3] influence (minor vs novelty/q-yield)
    min_candidate_score: float = -1.0  # LLM1 candidates below this score are filtered before probe/forward.

    # Beam-first global switching
    beam_width: int = 10
    min_switch_gain: float = 4.0
    jump_cooldown_s: float = 10.0
    restart_penalty: float = 3.0
    travel_cost_weight: float = 1.0
    restart_budget_window_s: float = 120.0
    restart_budget_max: int = 2
    beam_commit_min_s: float = 25.0
    beam_commit_min_actions: int = 14
    beam_commit_override_gain: float = 8.0
    edge_unverified_penalty: float = 1.2
    edge_stale_after_s: float = 240.0
    edge_stale_penalty: float = 0.3

    # Recovery / replay / backtrace
    recovery_attempts: int = 2
    recovery_back_steps: int = 2
    replay_max_depth: int = 18
    backtrace_max_steps: int = 6

    # Concurrency
    max_workers: int = 4

    # Foreground package gate
    foreground_mismatch_limit: int = 3

    # Repeat/loop control
    blacklist_nochange_threshold: int = 2
    blacklist_ttl_s: float = 90.0
    loop_blacklist_ttl_s: float = 120.0
    repeat_penalty_alpha: float = 1.0


class RecoveryReason(str, Enum):
    OVERLAY_UNRESOLVED = "overlay_unresolved"
    FOREGROUND_MISMATCH = "foreground_mismatch"
    RETURN_FAILED = "return_failed"
    STUCK_NO_PROGRESS = "stuck_no_progress"
    CAPTURE_FAILED = "capture_failed"
    BACKTRACE_FAILED = "backtrace_failed"
    FORWARD_ACTION_FAILED = "forward_action_failed"
    PROBE_CAPTURE_FAILED = "probe_capture_failed"
    OVERLAY_DURING_BACKTRACE = "overlay_during_backtrace"
    CAPTURE_FAILED_AFTER_FORWARD = "capture_failed_after_forward"


def compute_state_signature(uist: Dict[str, Any], *, foreground_package: str = "", foreground_activity: str = "") -> str:
    """
    IPO:
      in : processed UI tree (uist) after BaseUI.post_process_ui
      out: stable-ish signature string used for caches/graph keys

    WHEN called:
      - Every snapshot capture in _capture_and_process()

    WHY:
      - state_sig is the core "staleness barrier": LLM outputs/caches are trusted only for matching sig.
      - Must distinguish tab/frame content changes even if outer structure is similar.
        => include a SMALL normalized label token (text/content_desc/icon_label/ocr_text) per node.

    Robustness notes:
      - We strip digits and overlong strings to avoid volatility from counters/timestamps.
    """
    items: List[Tuple] = []

    def q(v: int) -> int:
        return int(v // 12)

    def norm_label(n: Dict[str, Any]) -> str:
        raw = (
            n.get("text")
            or n.get("content_desc")
            or n.get("semantic_label")
            or n.get("icon_label")
            or n.get("ocr_text")
            or ""
        )
        s = str(raw).strip().lower()
        if not s:
            return ""
        # remove digits to reduce volatility (badge counts, prices, timestamps)
        s = re.sub(r"\d+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s[:18]

    def walk(n: Dict[str, Any]):
        f = BaseUI.get_frame(n)
        items.append(
            (
                str(n.get("class") or ""),
                str(n.get("resource_id") or "")[-40:],
                bool(n.get("clickable")),
                bool(n.get("selected", False)),
                norm_label(n),
                q(int(f["x"])),
                q(int(f["y"])),
                q(int(f["width"])),
                q(int(f["height"])),
            )
        )
        for ch in n.get("subviews", []) or []:
            walk(ch)

    for r in uist.get("elements", []) or []:
        walk(r)

    blob = json.dumps(
        {
            "foreground_package": str(foreground_package or ""),
            "foreground_activity": str(foreground_activity or ""),
            "items": items[:2600],
        },
        ensure_ascii=False,
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def compute_structural_signature(uist: Dict[str, Any], *, foreground_package: str = "", foreground_activity: str = "") -> str:
    """
    A more stable signature than state_sig for "return matching" when a page changes
    minor dynamic text while remaining structurally the same.
    """
    items: List[Tuple] = []

    def q(v: int) -> int:
        return int(v // 12)

    def walk(n: Dict[str, Any]):
        f = BaseUI.get_frame(n)
        items.append(
            (
                str(n.get("class") or ""),
                str(n.get("resource_id") or "")[-40:],
                bool(n.get("clickable")),
                bool(n.get("selected", False)),
                q(int(f["x"])),
                q(int(f["y"])),
                q(int(f["width"])),
                q(int(f["height"])),
            )
        )
        for ch in n.get("subviews", []) or []:
            walk(ch)

    for r in uist.get("elements", []) or []:
        walk(r)

    blob = json.dumps(
        {
            "foreground_package": str(foreground_package or ""),
            "foreground_activity": str(foreground_activity or ""),
            "items": items[:2600],
        },
        ensure_ascii=False,
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def compute_coarse_signature(uist: Dict[str, Any], *, foreground_package: str = "", foreground_activity: str = "") -> str:
    """
    Coarser state hash (more stable): ignores labels and uses larger geometry buckets.
    Useful for fast grouping / loop detection and as a fallback identity when text is volatile.
    """
    items: List[Tuple] = []

    def q(v: int) -> int:
        return int(v // 24)

    def walk(n: Dict[str, Any]):
        f = BaseUI.get_frame(n)
        items.append(
            (
                str(n.get("class") or ""),
                str(n.get("resource_id") or "")[-40:],
                bool(n.get("clickable")),
                bool(n.get("selected", False)),
                q(int(f["x"])),
                q(int(f["y"])),
                q(int(f["width"])),
                q(int(f["height"])),
            )
        )
        for ch in n.get("subviews", []) or []:
            walk(ch)

    for r in uist.get("elements", []) or []:
        walk(r)

    blob = json.dumps(
        {
            "foreground_package": str(foreground_package or ""),
            "foreground_activity": str(foreground_activity or ""),
            "items": items[:2600],
        },
        ensure_ascii=False,
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def compute_fine_signature(uist: Dict[str, Any], *, foreground_package: str = "", foreground_activity: str = "") -> str:
    """
    Finer state hash (more discriminative): uses smaller geometry buckets and longer label tokens.
    Useful for detecting signature collisions and for high-confidence "no-change" verification.
    """
    items: List[Tuple] = []

    def q(v: int) -> int:
        return int(v // 6)

    def norm_label(n: Dict[str, Any]) -> str:
        raw = (
            n.get("text")
            or n.get("content_desc")
            or n.get("semantic_label")
            or n.get("icon_label")
            or n.get("ocr_text")
            or ""
        )
        s = str(raw).strip().lower()
        if not s:
            return ""
        s = re.sub(r"\d+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s[:32]

    def walk(n: Dict[str, Any]):
        f = BaseUI.get_frame(n)
        items.append(
            (
                str(n.get("class") or ""),
                str(n.get("resource_id") or "")[-60:],
                bool(n.get("clickable")),
                bool(n.get("enabled", True)),
                bool(n.get("selected", False)),
                norm_label(n),
                q(int(f["x"])),
                q(int(f["y"])),
                q(int(f["width"])),
                q(int(f["height"])),
            )
        )
        for ch in n.get("subviews", []) or []:
            walk(ch)

    for r in uist.get("elements", []) or []:
        walk(r)

    blob = json.dumps(
        {
            "foreground_package": str(foreground_package or ""),
            "foreground_activity": str(foreground_activity or ""),
            "items": items[:3000],
        },
        ensure_ascii=False,
    )
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def compute_screenshot_phash(image_b64: str, size: int = 32, lowfreq: int = 8) -> str:
    """Return a perceptual hash hex string for a base64-encoded screenshot."""
    if not image_b64:
        return ""
    try:
        raw = base64.b64decode(str(image_b64) + "==", validate=False)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ""
        resized = cv2.resize(image, (int(size), int(size)), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        low = dct[: int(lowfreq), : int(lowfreq)]
        flat = low.flatten()
        threshold = np.median(flat[1:]) if flat.size > 1 else flat[0]
        bits = (low > threshold).astype(np.uint8).flatten()
        pad_len = (-len(bits)) % 8
        if pad_len:
            bits = np.pad(bits, (0, pad_len), constant_values=0)
        return "".join(f"{b:02x}" for b in np.packbits(bits).tolist())
    except Exception:
        return ""


def compare_phash_similarity(hash1: str, hash2: str) -> float:
    """Return similarity in [0, 1] from two perceptual-hash hex strings."""
    if not hash1 or not hash2:
        return 0.0
    try:
        bits1 = np.unpackbits(np.frombuffer(bytes.fromhex(hash1), dtype=np.uint8))
        bits2 = np.unpackbits(np.frombuffer(bytes.fromhex(hash2), dtype=np.uint8))
        if bits1.shape != bits2.shape:
            return 0.0
        total = int(bits1.size)
        if total <= 0:
            return 0.0
        dist = int(np.count_nonzero(bits1 != bits2))
        return 1.0 - (dist / total)
    except Exception:
        return 0.0


def count_meaningful_xml_nodes(xml_text: str) -> int:
    """
    Count meaningful nodes from Appium XML only.

    Meaningful node:
    - not fullscreen / near-fullscreen
    - has text / content-desc / meaningful resource-id / clickable=true
    """
    if not xml_text or "<hierarchy" not in xml_text:
        return 0
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return 0

    screen_w = int(root.attrib.get("width", "0") or 0)
    screen_h = int(root.attrib.get("height", "0") or 0)
    count = 0

    for node in root.iter():
        if node.tag == "hierarchy":
            continue

        bounds_text = str(node.attrib.get("bounds", "") or "").strip()
        bounds: Optional[Tuple[int, int, int, int]] = None
        if bounds_text.startswith("[") and "][" in bounds_text and bounds_text.endswith("]"):
            try:
                left, right = bounds_text[1:-1].split("][", 1)
                x1_s, y1_s = left.split(",", 1)
                x2_s, y2_s = right.split(",", 1)
                x1, y1, x2, y2 = int(x1_s), int(y1_s), int(x2_s), int(y2_s)
                if x2 > x1 and y2 > y1:
                    bounds = (x1, y1, x2, y2)
            except Exception:
                bounds = None

        if bounds and screen_w > 0 and screen_h > 0:
            x1, y1, x2, y2 = bounds
            node_w = x2 - x1
            node_h = y2 - y1
            area_ratio = (node_w * node_h) / float(screen_w * screen_h)
            width_ratio = node_w / float(screen_w)
            height_ratio = node_h / float(screen_h)
            if area_ratio >= 0.90 or (width_ratio >= 0.98 and height_ratio >= 0.98):
                continue

        rid = str(node.attrib.get("resource-id", "") or "").strip()
        has_rid = bool(rid) and rid not in {"android:id/content"}
        has_text = bool(str(node.attrib.get("text", "") or "").strip())
        has_desc = bool(str(node.attrib.get("content-desc", "") or "").strip())
        clickable = str(node.attrib.get("clickable", "false")).lower() == "true"

        if has_text or has_desc or has_rid or clickable:
            count += 1

    return count


@dataclass
class WorkflowRunner:
    appium: AndroidAppiumClient
    gpt: GPTClient
    questionnaires: QuestionnaireState2
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    target_package: str = ""
    target_activity: Optional[str] = None
    allowed_packages: Set[str] = field(default_factory=set)
    allowed_packages_recovery: Set[str] = field(default_factory=set)

    pause: bool = False

    # App-level metadata context (computed before run starts).
    app_intro: Optional[str] = None
    focus_hints: Optional[str] = None
    questionnaire_type: str = ""
    questionnaire_type_source: str = "manual"
    metadata_csv_path: str = ""
    metadata_entry_found: bool = False
    metadata_notes: str = ""

    callbacks: Callbacks = field(default_factory=NoOpCallbacks)
    run_id: str = ""

    graph: StateGraph = field(default_factory=StateGraph, init=False)

    # Thread pool for LLM calls
    _pool: Optional[ThreadPoolExecutor] = field(default=None, init=False)

    # Futures keyed by state_sig (stale-safe)
    _nav_futures: Dict[str, Future] = field(default_factory=dict, init=False)
    _topic_route_futures: Dict[str, Future] = field(default_factory=dict, init=False)
    _topic_fill_futures: Dict[Tuple[str, str, str], Future] = field(default_factory=dict, init=False)
    _block_router_futures: Dict[str, Future] = field(default_factory=dict, init=False)
    _blocks_fill_futures: Dict[str, Future] = field(default_factory=dict, init=False)

    # LLM caches keyed by state_sig
    nav_cache: Dict[str, NavigationProposal] = field(default_factory=dict, init=False)
    topic_route_cache: Dict[str, TopicRouteResult] = field(default_factory=dict, init=False)
    q_cache: Dict[str, QuestionnaireUpdate] = field(default_factory=dict, init=False)
    topic_fill_cache: Dict[Tuple[str, str, str], QuestionnaireUpdate] = field(default_factory=dict, init=False)
    block_router_cache: Dict[str, RouterResult] = field(default_factory=dict, init=False)
    block_match_cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)
    blocks_fill_cache: Dict[str, BlocksFillResult] = field(default_factory=dict, init=False)

    # LLM enqueue timestamps
    _nav_enqueue_ts: Dict[str, float] = field(default_factory=dict, init=False)
    _topic_route_enqueue_ts: Dict[str, float] = field(default_factory=dict, init=False)
    _topic_fill_enqueue_ts: Dict[Tuple[str, str, str], float] = field(default_factory=dict, init=False)
    _block_router_enqueue_ts: Dict[str, float] = field(default_factory=dict, init=False)
    _blocks_fill_enqueue_ts: Dict[str, float] = field(default_factory=dict, init=False)

    # Topic fill attempt tracking (cooldown per (sig,topic_id))
    topic_fill_attempt_ts: Dict[Tuple[str, str], float] = field(default_factory=dict, init=False)

    # Snapshot cache for LLM2-1/2 pipelines (limited; keeps screenshot+uist for async fills)
    llm_snap_cache: "OrderedDict[str, Dict[str, Any]]" = field(default_factory=OrderedDict, init=False)
    llm_snap_cache_max: int = 12

    # Per-state probing bookkeeping
    # attempted_actions: clicked at least once (probe/forward)
    # explored_actions: branch finished or proven no-effect
    attempted_actions: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    explored_actions: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    probe_outcomes: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False)

    # Novelty at discovery time per probed action (for forward scoring)
    probe_novelty: Dict[str, Dict[str, bool]] = field(default_factory=dict, init=False)

    # NAV reliability bookkeeping
    nav_failures: Dict[str, int] = field(default_factory=dict, init=False)
    nav_cooldown_until: Dict[str, float] = field(default_factory=dict, init=False)

    # Candidate memory per state (DFS completion)
    state_candidates: Dict[str, List[ActionCandidate]] = field(default_factory=dict, init=False)  # keyed by family_id
    nav_candidates_noted: Set[str] = field(default_factory=set, init=False)

    # Counters (global)
    action_count: int = 0
    no_new_state_count: int = 0
    no_progress_loops: int = 0
    last_strong_progress_ts: float = field(default=0.0, init=False)
    last_weak_progress_ts: float = field(default=0.0, init=False)
    nav_ready_once: bool = False

    # Last action failure detail (best-effort; cleared on success).
    # Used to treat certain failures (e.g., foreground mismatches) as state faults rather than mere candidate failures.
    last_action_failure: Optional[Dict[str, Any]] = field(default=None, init=False)

    # Action history for LLM context
    history: List[str] = field(default_factory=list, init=False)

    # DFS stack / parent tracking (represents the *current* navigation path)
    dfs_stack: List[str] = field(default_factory=list, init=False)
    # Parallel stack: dfs_via[i] is the action_key used to enter dfs_stack[i] from dfs_stack[i-1].
    # Root has dfs_via[0] == None.
    dfs_via: List[Optional[str]] = field(default_factory=list, init=False)
    parent_map: Dict[str, str] = field(default_factory=dict, init=False)

    # Debug-only: recent observed states (should NOT affect graph visit_count)
    recent_states: List[str] = field(default_factory=list, init=False)
    sig_to_family: Dict[str, str] = field(default_factory=dict, init=False)
    visual_state_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # Restart baselines for replay
    entry_sig: str = ""
    restart_entry_sig: Optional[str] = None
    restart_recent: Deque[float] = field(default_factory=deque, init=False)

    # Snapshot post-process cache
    snapshot_cache: "OrderedDict[str, Dict[str, Any]]" = field(default_factory=OrderedDict, init=False)
    snapshot_cache_max: int = 64
    snapshot_cache_hits: int = 0
    snapshot_cache_misses: int = 0

    # Trace step counter
    _step_seq: int = field(default=0, init=False)

    # Force-replan signals (set by NAV-wait drift detection OR probe back-like behavior)
    _force_replan_sig: Optional[str] = field(default=None, init=False)
    _force_replan_snap: Optional[Dict[str, Any]] = field(default=None, init=False)
    _force_replan_has_edge: bool = field(default=False, init=False)

    # Forward scoring detail (last selection)
    _last_forward_detail: Optional[Dict[str, Any]] = field(default=None, init=False)

    # Questionnaire yield tracking per state_sig (used by beam/global scoring)
    state_update_counts: Dict[str, int] = field(default_factory=dict, init=False)

    # Replay path cache for beam cost estimation: (src,dst) -> (steps, ts, unverified_edges, stale_edges)
    replay_distance_cache: Dict[Tuple[str, str], Tuple[int, float, int, int]] = field(default_factory=dict, init=False)

    # Beam switching cooldown
    last_jump_ts: float = field(default=0.0, init=False)
    _last_beam_detail: Optional[Dict[str, Any]] = field(default=None, init=False)
    committed_target_sig: Optional[str] = field(default=None, init=False)
    committed_target_score: float = field(default=-1e9, init=False)
    commit_until_ts: float = field(default=0.0, init=False)
    commit_until_action_count: int = field(default=0, init=False)

    # Foreground gate bookkeeping
    foreground_mismatch_count: int = field(default=0, init=False)
    foreground_mismatch_streak: int = field(default=0, init=False)
    foreground_recoveries: int = field(default=0, init=False)

    # Repeat/loop control (family_id + action_key -> counts/expiry)
    action_attempt_counts: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)
    action_nochange_counts: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)
    action_blacklist_until: Dict[Tuple[str, str], float] = field(default_factory=dict, init=False)
    # (src_sig, dst_sig, action_key, src_family, dst_family)
    recent_transitions: List[Tuple[str, str, str, str, str]] = field(default_factory=list, init=False)
    loop_detected_count: int = field(default=0, init=False)
    analysis_export_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        cb_run = getattr(self.callbacks, "run_id", None)
        if self.run_id and not cb_run:
            try:
                setattr(self.callbacks, "run_id", self.run_id)
            except Exception:
                pass
        if not self.run_id:
            self.run_id = cb_run or time.strftime("%Y%m%d_%H%M%S")
        elif cb_run:
            self.run_id = cb_run
        now = time.time()
        self.last_strong_progress_ts = now
        self.last_weak_progress_ts = now

        # Default allowed packages: target app + Android permission controller(s).
        # You can override by passing allowed_packages explicitly.
        if not self.allowed_packages:
            if self.target_package:
                self.allowed_packages.add(self.target_package)
        self.allowed_packages.update({"com.android.permissioncontroller", "com.google.android.permissioncontroller"})
        if not self.allowed_packages_recovery:
            self.allowed_packages_recovery = set(self.allowed_packages)
        else:
            self.allowed_packages_recovery.update(self.allowed_packages)
        # Recovery-safe foreground packages (BACK/WAIT only). Keep this list conservative.
        self.allowed_packages_recovery.update(
            {
                "com.android.systemui",
                "com.android.settings",
                "com.android.chrome",
                "com.google.android.gms",
                "com.google.android.googlequicksearchbox",
            }
        )
        try:
            logger.info(
                "QuestionnaireState2 active: routers=%d blocks=%d",
                len(getattr(self.questionnaires, "routers", []) or []),
                len(getattr(self.questionnaires, "blocks", []) or []),
            )
        except Exception:
            logger.debug("QuestionnaireState2 summary failed", exc_info=True)
        self._save_app_metadata_context()

    # ---------------------------
    # Trace helpers
    # ---------------------------

    def _mk_ctx(self, cur_sig: str) -> StepCtx:
        self._step_seq += 1
        try:
            block_status = copy.deepcopy(getattr(self.questionnaires, "block_status", {}) or {})
        except Exception:
            block_status = {}
        return StepCtx(
            run_id=self.run_id,
            step_id=self._step_seq,
            ts=time.time(),
            cur_sig=cur_sig,
            stack=list(self.dfs_stack),
            block_status=block_status,
            open_gaps=[],
            answered_ratio=0.0,
        )

    def _emit_snapshot(self, snap: Dict[str, Any]) -> None:
        sig = str(snap.get("state_sig") or "")
        try:
            self.callbacks.on_snapshot(self._mk_ctx(sig), snap)
        except Exception:
            logger.debug("on_snapshot failed", exc_info=True)

    def _emit_llm_enqueued(self, kind: str, sig: str, payload: Dict[str, Any]) -> None:
        try:
            self.callbacks.on_llm_enqueued(self._mk_ctx(sig), kind, payload)
        except Exception:
            logger.debug("on_llm_enqueued failed", exc_info=True)

    def _emit_llm_result(self, kind: str, sig: str, payload: Dict[str, Any]) -> None:
        try:
            self.callbacks.on_llm_result(self._mk_ctx(sig), kind, payload)
        except Exception:
            logger.debug("on_llm_result failed", exc_info=True)

    def _emit_action(self, sig: str, action: Dict[str, Any], phase: str, extra: Dict[str, Any]) -> None:
        try:
            self.callbacks.on_action(self._mk_ctx(sig), action, phase, extra)
        except Exception:
            logger.debug("on_action failed", exc_info=True)

    def _emit_transition(self, sig: str, payload: Dict[str, Any]) -> None:
        try:
            self.callbacks.on_transition(self._mk_ctx(sig), payload)
        except Exception:
            logger.debug("on_transition failed", exc_info=True)

    def _emit_questionnaire_update(self, sig: str, payload: Dict[str, Any]) -> None:
        try:
            self.callbacks.on_questionnaire_update(self._mk_ctx(sig), payload)
        except Exception:
            logger.debug("on_questionnaire_update failed", exc_info=True)

    def _emit_decision(self, sig: str, name: str, detail: Dict[str, Any]) -> None:
        try:
            self.callbacks.on_decision(self._mk_ctx(sig), name, detail)
        except Exception:
            logger.debug("on_decision failed", exc_info=True)

    def _questionnaire2_observation_dir(self) -> Path:
        """
        Return where the new router/block observations should be saved.

        Input:
        - No explicit input. Uses callback/run metadata already attached to
          the workflow runner.

        Processing:
        - If JsonlTraceCallbacks is active, write beside the trace under:
          `<trace_run_root>/observations`.
        - Otherwise write to a stable debug folder under `mytest2`.

        Output:
        - Path to an observation directory. The directory is created by
          `QuestionnaireState2.save_observation(...)`.
        """
        cb_root = str(getattr(self.callbacks, "root_dir", "") or "").strip()
        if cb_root:
            return Path(cb_root) / "observations"
        run_token = self.run_id or time.strftime("%Y%m%d_%H%M%S")
        return Path("mytest2") / "questionnaire_handler" / "chain_debug" / "workflow_observations" / run_token

    def _run_output_root(self) -> Path:
        """
        Return the run-level root directory for auxiliary artifacts.

        - With trace callbacks: <trace_run_root>
        - Without trace callbacks: local debug run folder
        """
        cb_root = str(getattr(self.callbacks, "root_dir", "") or "").strip()
        if cb_root:
            return Path(cb_root)
        run_token = self.run_id or time.strftime("%Y%m%d_%H%M%S")
        return Path("mytest2") / "questionnaire_handler" / "chain_debug" / "workflow_observations" / run_token

    def _save_app_metadata_context(self) -> Optional[Path]:
        """
        Persist app-level metadata context once per run for reproducibility.

        Output:
        - Path to `app_metadata_context.json`, or None on failure.
        """
        try:
            out_root = self._run_output_root()
            out_root.mkdir(parents=True, exist_ok=True)
            out_path = out_root / "app_metadata_context.json"
            payload = {
                "run_id": self.run_id,
                "target_package": str(self.target_package or ""),
                "questionnaire_type": str(self.questionnaire_type or ""),
                "questionnaire_type_source": str(self.questionnaire_type_source or ""),
                "app_intro": self.app_intro,
                "focus_hints": self.focus_hints,
                "metadata_csv_path": str(self.metadata_csv_path or ""),
                "metadata_entry_found": bool(self.metadata_entry_found),
                "metadata_notes": str(self.metadata_notes or ""),
                "saved_at": int(time.time() * 1000),
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log_event("app_metadata_context_saved", sig="", path=str(out_path))
            return out_path
        except Exception:
            logger.debug("Failed to save app_metadata_context", exc_info=True)
            return None

    def _save_questionnaire2_observation(
        self,
        sig: str,
        router_answers: List[Dict[str, Any]],
        matched_blocks: List[Dict[str, Any]],
        block_fill_results: Optional[List[Dict[str, Any]]] = None,
        *,
        stage: str = "blocks_fill",
        screenshot_path: str = "",
    ) -> Optional[Path]:
        """
        Persist one observation produced by the new router/block path.

        Input:
        - sig: current state signature.
        - router_answers: LLM2-1 router answers, already converted to dicts.
        - matched_blocks: full block payloads matched locally by block_show_if.
        - block_fill_results: optional LLM2-2 results grouped by block.
        - stage: subfolder name, e.g. "router" or "blocks_fill".
        - screenshot_path: optional path to the screenshot asset.

        Processing:
        - Delegate JSON writing to QuestionnaireState2.save_observation.
        - Save router answers, matched block ids, and any block fill results
          available for this state.

        Output:
        - Path if saved, otherwise None.
        """
        q2 = self.questionnaires
        if q2 is None or not hasattr(q2, "save_observation"):
            return None
        try:
            return q2.save_observation(
                out_dir=str(self._questionnaire2_observation_dir() / stage),
                state_sig=sig,
                router_answers=router_answers,
                matched_blocks=matched_blocks,
                block_fill_results=list(block_fill_results or []),
                screenshot_path=screenshot_path,
            )
        except Exception:
            logger.debug("QuestionnaireState2 observation save failed sig=%s", sig[:8], exc_info=True)
            return None

    def _save_nav_observation(self, sig: str, nav: NavigationProposal) -> Optional[Path]:
        """
        Persist one LLM1 (navigation) result per state.

        Input:
        - sig: current state signature.
        - nav: LLM1 NavigationProposal.

        Output:
        - Path to written JSON, or None on failure.
        """
        try:
            root = self._questionnaire2_observation_dir() / "nav"
            root.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            safe_sig = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sig or "unknown")).strip("_") or "unknown"
            out_path = root / f"{stamp}_{safe_sig}.json"
            payload = {
                "state_sig": sig,
                "nav_result": nav.model_dump(mode="json") if hasattr(nav, "model_dump") else getattr(nav, "__dict__", {}),
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return out_path
        except Exception:
            logger.debug("NAV observation save failed sig=%s", str(sig)[:8], exc_info=True)
            return None

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Best-effort conversion for dataclass/pydantic/custom objects into JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): WorkflowRunner._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [WorkflowRunner._jsonable(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return WorkflowRunner._jsonable(value.model_dump(mode="json"))
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return WorkflowRunner._jsonable(vars(value))
            except Exception:
                pass
        return str(value)

    def _candidate_snapshot(self, cand: Any) -> Dict[str, Any]:
        """Serialize ActionCandidate-like objects for analysis export."""
        try:
            score = float(getattr(cand, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        try:
            tags = [str(x) for x in (getattr(cand, "tags", None) or [])[:8]]
        except Exception:
            tags = []
        actions: List[Dict[str, Any]] = []
        try:
            for step in list(getattr(cand, "actions", None) or [])[:4]:
                try:
                    actions.append(
                        {
                            "action": str(getattr(step, "action", "") or ""),
                            "element_id": getattr(step, "element_id", None),
                            "text": str(getattr(step, "text", "") or ""),
                            "reasoning": str(getattr(step, "reasoning", "") or ""),
                        }
                    )
                except Exception:
                    actions.append({"raw": str(step)})
        except Exception:
            pass
        return {
            "candidate_key": self._candidate_key(cand),
            "score": score,
            "tags": tags,
            "actions": actions,
        }

    def _collect_state_action_snapshot(self) -> Dict[str, Any]:
        """
        Build a per-state summary: explored/attempted/remaining candidates.
        Remaining candidates are the primary signal for unfinished DFS work.
        """
        per_state: Dict[str, Any] = {}
        all_states: Set[str] = set(self.graph.nodes.keys()) | set(self.sig_to_family.keys())

        for sig in sorted(all_states):
            fam = self._family_id(sig)
            candidates = list(self.state_candidates.get(fam) or [])
            if not candidates:
                nav = self.nav_cache.get(sig)
                if nav and getattr(nav, "candidate_actions", None) is not None:
                    candidates = list(getattr(nav, "candidate_actions") or [])

            candidate_rows: List[Dict[str, Any]] = []
            candidate_keys: List[str] = []
            for cand in candidates:
                if not getattr(cand, "actions", None):
                    continue
                row = self._candidate_snapshot(cand)
                key = str(row.get("candidate_key") or "")
                if not key:
                    continue
                candidate_rows.append(row)
                candidate_keys.append(key)

            explored = set(self.explored_actions.get(fam, set()))
            attempted = set(self.attempted_actions.get(fam, set()))
            remaining = [k for k in candidate_keys if k not in explored]
            per_state[sig] = {
                "state_sig": sig,
                "family_id": fam,
                "candidate_count": len(candidate_keys),
                "explored_count": len([k for k in candidate_keys if k in explored]),
                "attempted_count": len([k for k in candidate_keys if k in attempted]),
                "remaining_count": len(remaining),
                "is_exhausted": len(candidate_keys) > 0 and len(remaining) == 0,
                "candidate_keys": candidate_keys,
                "remaining_candidate_keys": remaining,
                "explored_keys": sorted(explored),
                "attempted_keys": sorted(attempted),
                "candidates": candidate_rows,
            }

        return {
            "per_state": per_state,
            "unfinished_states": sorted([sig for sig, row in per_state.items() if int(row.get("remaining_count", 0) or 0) > 0]),
        }

    def _export_analysis_snapshot(self, *, cur_sig: str, stop_reason: str) -> Optional[Path]:
        """
        Export run-time analysis artifacts for post-stop visualization.
        Files are overwritten by the latest snapshot.
        """
        try:
            out_dir = self._run_output_root() / "analysis"
            out_dir.mkdir(parents=True, exist_ok=True)

            graph_nodes: Dict[str, Any] = {}
            for sig, node in (self.graph.nodes or {}).items():
                graph_nodes[sig] = {
                    "sig": sig,
                    "visit_count": int(getattr(node, "visit_count", 0) or 0),
                    "first_ts": float(getattr(node, "first_ts", 0.0) or 0.0),
                    "last_ts": float(getattr(node, "last_ts", 0.0) or 0.0),
                    "overlay_kind": str(getattr(node, "overlay_kind", "none") or "none"),
                    "meta": self._jsonable(getattr(node, "meta", {}) or {}),
                    "outgoing_count": len(getattr(node, "outgoing", set()) or set()),
                    "incoming_count": len(getattr(node, "incoming", set()) or set()),
                }
            graph_edges: List[Dict[str, Any]] = []
            for edge in (self.graph.edges or {}).values():
                graph_edges.append(
                    {
                        "src": str(getattr(edge, "src", "") or ""),
                        "dst": str(getattr(edge, "dst", "") or ""),
                        "action": self._jsonable(getattr(edge, "action", {}) or {}),
                        "count": int(getattr(edge, "count", 0) or 0),
                        "no_effect": bool(getattr(edge, "no_effect", False)),
                        "verified_ok": int(getattr(edge, "verified_ok", 0) or 0),
                        "verified_fail": int(getattr(edge, "verified_fail", 0) or 0),
                        "last_ts": float(getattr(edge, "last_ts", 0.0) or 0.0),
                        "last_verified_ts": float(getattr(edge, "last_verified_ts", 0.0) or 0.0),
                    }
                )
            graph_payload = {
                "run_id": self.run_id,
                "cur_sig": cur_sig,
                "entry_sig": self.entry_sig,
                "restart_entry_sig": self.restart_entry_sig,
                "stop_reason": stop_reason,
                "node_count": len(graph_nodes),
                "edge_count": len(graph_edges),
                "nodes": graph_nodes,
                "edges": graph_edges,
            }
            (out_dir / "state_graph_snapshot.json").write_text(
                json.dumps(graph_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            dfs_payload = {
                "run_id": self.run_id,
                "cur_sig": cur_sig,
                "stop_reason": stop_reason,
                "entry_sig": self.entry_sig,
                "restart_entry_sig": self.restart_entry_sig,
                "dfs_stack": list(self.dfs_stack),
                "dfs_via": list(self.dfs_via),
                "parent_map": dict(self.parent_map),
                "stack_depth": len(self.dfs_stack),
            }
            (out_dir / "dfs_snapshot.json").write_text(
                json.dumps(dfs_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            action_payload = self._collect_state_action_snapshot()
            (out_dir / "state_action_snapshot.json").write_text(
                json.dumps(action_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self.analysis_export_count += 1
            summary_payload = {
                "run_id": self.run_id,
                "export_seq": self.analysis_export_count,
                "exported_at": int(time.time() * 1000),
                "stop_reason": stop_reason,
                "cur_sig": cur_sig,
                "entry_sig": self.entry_sig,
                "action_count": int(self.action_count),
                "no_new_state_count": int(self.no_new_state_count),
                "no_progress_loops": int(self.no_progress_loops),
                "graph_node_count": int(len(graph_nodes)),
                "graph_edge_count": int(len(graph_edges)),
                "dfs_stack_depth": int(len(self.dfs_stack)),
                "unfinished_state_count": int(len(action_payload.get("unfinished_states", []) or [])),
            }
            (out_dir / "run_analysis_summary.json").write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._log_event("analysis_snapshot_exported", sig=cur_sig, out_dir=str(out_dir), stop_reason=stop_reason)
            return out_dir
        except Exception:
            logger.debug("Failed to export analysis snapshot", exc_info=True)
            return None

    # ---------------------------
    # Structured log helpers
    # ---------------------------

    def _log_event(self, name: str, **fields: Any) -> None:
        try:
            sig = fields.pop("sig", None) or (self.recent_states[-1] if self.recent_states else "")
            payload = {
                "event": name,
                "run_id": self.run_id,
                "step_id": self._step_seq,
                "cur_sig": sig[:8] if isinstance(sig, str) else sig,
                "stack_depth": len(self.dfs_stack),
                "action_count": self.action_count,
                "no_progress_loops": self.no_progress_loops,
                "snapshot_cache_hits": self.snapshot_cache_hits,
                "snapshot_cache_misses": self.snapshot_cache_misses,
                **fields,
            }
            logger.info(json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.debug("log_event failed", exc_info=True)

    def _infer_action_origin(self) -> str:
        """
        Best-effort caller attribution for debugging action execution.
        """
        interesting = {
            "_probe_candidates",
            "_return_to_expected",
            "_dismiss_overlay_with_nav",
            "_handle_loading_overlay",
            "_backtrace_to",
            "_recover",
            "_restart_and_replay",
            "_navigate_via_graph",
            "_execute_action_payload",
            "_replay_actions_to_target",
            "run",
        }
        try:
            for frame_info in inspect.stack()[1:12]:
                fn = str(frame_info.function or "")
                if fn in interesting:
                    return fn
        except Exception:
            return "unknown"
        return "unknown"

    def _mark_progress(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        self.last_weak_progress_ts = now

        strong_kinds = {
            "entry_state",
            "new_state_discovered",
            "q_update_applied",
            "overlay_resolved",
            "candidates_discovered",
        }
        is_strong = kind in strong_kinds
        if is_strong:
            self.last_strong_progress_ts = now
            self.no_progress_loops = 0

        self._log_event("progress_event", kind=kind, level=("strong" if is_strong else "weak"), **(detail or {}))

    def _overlay_kind_value(self, nav: Optional[NavigationProposal]) -> str:
        if nav is None:
            return OverlayKind.NONE.value
        ok = getattr(nav, "overlay_kind", None)
        if ok is None:
            return OverlayKind.NONE.value
        return ok.value if isinstance(ok, OverlayKind) else str(ok)

    def _overlay_kind_for_sig(self, sig: str, fallback: Optional[str] = None) -> str:
        nav = self.nav_cache.get(sig)
        if nav:
            return self._overlay_kind_value(nav)
        return fallback if fallback is not None else OverlayKind.NONE.value

    # ---------------------------
    # Foreground package gate
    # ---------------------------

    def _foreground_is_allowed(self, *, mode: str = "action") -> Tuple[bool, str]:
        """
        Returns (allowed, current_package).
        If we cannot read the package, treat as allowed (fail-open to avoid deadlocks).
        """
        allowed: Set[str] = set()
        if str(mode).lower().strip() == "recovery":
            allowed = set(self.allowed_packages_recovery or set(self.allowed_packages or set()))
        else:
            allowed = set(self.allowed_packages or set())
        if not allowed:
            return True, ""
        try:
            pkg = self.appium.foreground_package()
        except Exception:
            return True, ""
        if not pkg:
            return True, ""
        return (pkg in allowed), pkg

    def _handle_foreground_gate(self, cur_sig: str, snap: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
        """
        Gate that runs before any goal-directed in-app actions.
        If the foreground package is not allowed, we attempt bounded recovery and return a refreshed snapshot.
        Returns:
          - None if allowed (no action needed)
          - snap' if we attempted recovery / state changed and caller should replan
        """
        allowed, pkg = self._foreground_is_allowed()
        if allowed:
            if self.foreground_mismatch_streak:
                self._log_event("foreground_recovered", sig=cur_sig, prev_streak=self.foreground_mismatch_streak)
            self.foreground_mismatch_streak = 0
            return None

        recovery_allowed = bool(pkg and (pkg in set(self.allowed_packages_recovery or set())))
        self.foreground_mismatch_count += 1
        self.foreground_mismatch_streak += 1
        self._emit_decision(
            cur_sig,
            "foreground_mismatch",
            {
                "foreground_package": pkg,
                "allowed_packages": sorted(list(self.allowed_packages))[:8],
                "recovery_allowed": recovery_allowed,
                "streak": self.foreground_mismatch_streak,
            },
        )
        self._log_event(
            "foreground_mismatch",
            sig=cur_sig,
            foreground_package=pkg,
            allowed_count=len(self.allowed_packages),
            recovery_allowed=recovery_allowed,
            streak=self.foreground_mismatch_streak,
            total=self.foreground_mismatch_count,
        )

        # Tier-1 recovery: try to bring the target app back to foreground without restart.
        try:
            if self.target_package:
                self.appium.ensure_foreground(self.target_package, self.target_activity)
                self.foreground_recoveries += 1
        except Exception:
            logger.debug("ensure_foreground failed during foreground gate", exc_info=True)

        snap2 = self._capture_and_process(timeout=6.0)
        if not snap2:
            # If we can't capture, fall back to deterministic recovery.
            self._recover(cur_sig, snap, reason=RecoveryReason.FOREGROUND_MISMATCH, task=task, target_sig=None)
            return self._capture_and_process(timeout=8.0)

        # If mismatch persists repeatedly, escalate to tier-3 restart via recovery.
        if self.foreground_mismatch_streak >= int(self.budget.foreground_mismatch_limit):
            self._recover(snap2.get("state_sig") or cur_sig, snap2, reason=RecoveryReason.FOREGROUND_MISMATCH, task=task, target_sig=None)
            snap3 = self._capture_and_process(timeout=8.0)
            return snap3 or snap2

        return snap2

    # ---------------------------
    # Main loop
    # ---------------------------

    def run(self, task: str) -> None:
        """
        IPO:
          in : task string (questionnaire exploration goal)
          out: questionnaire filled as much as possible; graph expanded; logs

        WHEN called:
          - single entrypoint per exploration session

        WHERE graph is updated:
          - EVERY transition (probe/forward/overlay/recovery/replay/drift) -> _graph_record_transition()
          - Observations without clean predecessor edge -> _graph_record_observation()
          - Overlay flags from NAV -> _graph_annotate(sig, overlay_kind=...)

        WHICH state variables are authoritative:
          - cur_sig always matches snap["state_sig"] for the most recent authoritative snapshot.
          - nav_cache/q_cache are only trusted when key == cur_sig (staleness barrier).
        """
        # 记录本次 run 开始时间，后面的停止条件会基于它计算总耗时。
        start = time.time()

        # 如果配置了目标包名，先确保目标应用已经在前台，避免一开始就跑在错误页面上。
        if self.target_package:
            # 把目标 app 拉到前台；如果已经在前台，这一步基本是幂等的。
            self.appium.ensure_foreground(self.target_package, self.target_activity)

        # 创建线程池，后续 NAV / topic route / topic fill 等 LLM 任务会异步提交到这里。
        self._pool = ThreadPoolExecutor(max_workers=self.budget.max_workers)

        # Authoritative entry snapshot
        # 抓取首次“权威快照”：包含当前页面截图、XML、处理后的 UI 树、状态签名等。
        snap = self._capture_and_process()
        # 如果连初始页面都抓不到，整个探索流程无法继续，直接终止。
        if not snap:
            # 记录错误日志，说明这次 run 连入口状态都没有建立起来。
            logger.error("Initial capture failed; abort.")
            # 提前返回，不进入主循环。
            return

        # 当前状态签名，后面所有调度、缓存、图记录都围绕这个 sig 展开。
        cur_sig: str = snap["state_sig"]
        # 记录首次进入 app 时的入口状态，用于后续 replay / restart 的基准。
        self.entry_sig = cur_sig
        # restart 后的回放也默认从这个入口状态重新出发。
        self.restart_entry_sig = cur_sig
        # DFS 栈初始化为当前入口状态，表示当前探索路径从这个页面开始。
        self.dfs_stack = [cur_sig]
        # 根节点没有“通过哪个动作进入”，所以对应的 via 记为 None。
        self.dfs_via = [None]
        # 父节点映射在入口处清空，后面随着状态迁移逐步建立。
        self.parent_map = {}

        # Entry is an observation (no predecessor edge).
        # 把入口状态记为一次“观察到的页面”，因为它没有前驱动作，不适合记成边。
        self._graph_record_observation(cur_sig, meta={"entry": True}, snap=snap)
        # 为入口状态安排异步分析：包括导航建议和问卷相关的 LLM 任务。
        self._schedule_state(cur_sig, snap, task)
        # 把进入入口页记为一次进展，便于重置 no_progress 之类的计数器。
        self._mark_progress("entry_state", {"sig": cur_sig})

        # 主循环：不断围绕“当前页面”做分析、探测、前进、回退、恢复，直到触发停止条件。
        while True:
            # 每轮一开始先看是否应该结束，例如超时、动作数超限、长时间没新状态等。
            stop_reason = self._stop_condition_reason(start)
            if stop_reason:
                # 记录停止日志，方便从 trace / log 中看 run 为什么结束。
                logger.warning("Stop condition reached: %s", stop_reason)
                print(f"[STOP] {stop_reason}")
                self._log_event("run_stop_condition", sig=cur_sig, reason=stop_reason)
                self._export_analysis_snapshot(cur_sig=cur_sig, stop_reason=stop_reason)
                # 跳出主循环，进入资源清理阶段。
                break

            # Each loop without progress increments; _mark_progress resets it.
            # 默认先把“无进展轮数”加一；如果本轮真的有进展，后续会被 _mark_progress 重置。
            self.no_progress_loops += 1

            # WHERE: main loop top; drain completed LLM futures so decisions see latest analysis.
            # 回收已经完成的异步 LLM 任务，把结果写入缓存 / 问卷状态，保证本轮决策看到的是最新信息。
            self._drain_futures()

            # 读取当前仍未填完的问卷问题，作为主循环的目标集合。
            block_status = getattr(self.questionnaires, "block_status", {}) or {}
            # 计算当前问卷完成比例，用于日志和调试观察整体推进情况。
            # 打印当前循环的核心状态，包括页面 sig、栈深、动作数、剩余 gap、异步任务数量等。
            logger.info(
                "Loop: sig=%s stack_depth=%d actions=%d blocks=%d pending(nav=%d block_router=%d) no_progress=%d no_new=%d",
                cur_sig[:8],
                len(self.dfs_stack),
                self.action_count,
                len(block_status),
                len(self._nav_futures),
                len(self._block_router_futures),
                self.no_progress_loops,
                self.no_new_state_count,
            )

            # 如果已经没有未完成问卷项，说明本次探索目标达成，可以结束。
            if False and block_status:
                # 记录“所有 gap 都填完”的结束原因。
                logger.info("All questionnaire gaps filled. Stopping.")
                # 跳出主循环。
                break

            # CONDITION: FOREGROUND PACKAGE MISMATCH (ads/external browser/system overlays)
            # 先检查前台包名是否还在目标 app 内；如果跳去了广告、浏览器、系统页，需要先拉回来。
            gate_snap = self._handle_foreground_gate(cur_sig, snap, task)
            # 如果前台修复逻辑真的造成了页面变化，这里会返回一个新的快照。
            if gate_snap:
                # 用修复后的快照替换当前快照。
                snap = gate_snap
                # 同步更新当前状态签名。
                cur_sig = snap["state_sig"]
                # We can't safely attribute this relocation to a precise UI edge.
                # 这种“位置变更”通常不是由一个明确的业务动作引起，所以按 external move 方式修正栈结构。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=True)
                # 对修复后的新页面重新安排分析任务。
                self._schedule_state(cur_sig, snap, task)
                # 本轮剩余逻辑作废，直接开始下一轮。
                continue

            # CONDITION: DRIFT (UI changed without our explicit logged action)
            # WHEN/WHERE:
            # - checked once per loop BEFORE trusting NAV proposals/caches for cur_sig.
            # 在真正相信当前缓存和导航建议之前，先做一次轻量预检，看看页面是否自己变了。
            snap_live = self._preflight_refresh_if_changed(snap, timeout_s=0.8)
            # 如果预检发现页面确实发生了变化，就进入 drift 分支。
            if snap_live:
                # 如果状态签名已经变了，说明页面漂移到了另一个状态，当前计划要作废重排。
                if snap_live["state_sig"] != cur_sig:
                    # 取出漂移后的新状态签名。
                    new_sig: str = snap_live["state_sig"]
                    # 输出 drift 日志，帮助定位“页面自己变了”的情况。
                    logger.warning("DRIFT: old=%s new=%s. Replanning.", cur_sig[:8], new_sig[:8])
                    # 发出决策事件，标记这是一次 drift 导致的重规划。
                    self._emit_decision(cur_sig, "drift_detected", {"from_sig": cur_sig, "to_sig": new_sig})

                    # 构造一个伪动作，用来在状态图里表示“等待时页面自己变了”这类漂移迁移。
                    drift_step = ActionStep(action=ActionType.WAIT, element_id=None, text="0.2", priority=1, reasoning="drift")
                    # 把 drift 也记成一条状态迁移边，并拿到目标状态在发现当时是否是新状态。
                    dst_was_new = self._graph_record_transition(
                        cur_sig,
                        new_sig,
                        self._actions_signature([drift_step]),
                        src_snap=snap,
                        dst_snap=snap_live,
                    )
                    # 更新 DFS 路径，让当前所在位置切换到漂移后的状态。
                    self._enter_state(from_sig=cur_sig, to_sig=new_sig, via_action="drift")

                    # 当前状态与快照都切到漂移后的页面。
                    cur_sig, snap = new_sig, snap_live
                    # 对新状态重新安排异步分析。
                    self._schedule_state(cur_sig, snap, task)

                    # no_new_state_count tracks novelty-at-discovery
                    # 如果漂移到了一个从未见过的新状态，就清零“无新状态”计数。
                    if dst_was_new:
                        self.no_new_state_count = 0
                    # 否则说明只是跳到旧状态，“无新状态”计数继续累计。
                    else:
                        self.no_new_state_count += 1
                    # 当前轮因为页面基准已经变化，直接重开下一轮。
                    continue

                # Same state_sig but raw content changed; refresh snap for accurate vid_map.
                # 如果 sig 没变但底层内容更新了，也要用新快照替换，保证 vid_map / 元素 id 不过期。
                snap = snap_live

            # NAV barrier:
            # - may wait; may detect drift while waiting; may timeout -> heuristics
            # 在根节点适当放宽等待时间，给首页导航分析更多时间。
            timeout_s = self.budget.nav_timeout_s + (30.0 if len(self.dfs_stack) == 1 else 0.0)
            self._emit_decision(
                cur_sig,
                "next_step",
                {"plan": "wait_nav_or_fallback", "timeout_s": timeout_s, "stack_depth": len(self.dfs_stack)},
            )
            # 等待 NAV 结果；如果超时或进入 cooldown，则允许走启发式候选。
            nav, using_heuristics = self._wait_nav_or_fallback(cur_sig, snap, task, timeout_s=timeout_s)

            # NAV wait can set forced replan (drift while waiting)
            # NAV 等待期间也可能检测到 drift / 强制重规划信号，这里优先处理。
            if self._consume_forced_replan():
                # 取出强制重规划指定的新状态与快照。
                cur_sig, snap = self._force_replan_sig, self._force_replan_snap  # type: ignore[assignment]
                # 标记这次重规划前是否已经记录过迁移边。
                has_edge = self._force_replan_has_edge
                # 清空强制重规划标记，避免后续重复消费。
                self._clear_forced_replan()
                # 把当前栈位置修正到新状态；如果已经有边，就不再补 observation。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=not has_edge)
                # 为新页面重新安排分析。
                self._schedule_state(cur_sig, snap, task)
                # 直接进入下一轮。
                continue

            # Candidate memory for DFS completion
            # 如果拿到了 NAV 候选，把这组候选记到当前页面 family 上，供 DFS exhausted 判定使用。
            if nav is not None and getattr(nav, "candidate_actions", None) is not None:
                self.state_candidates[self._family_id(cur_sig)] = list(getattr(nav, "candidate_actions") or [])

            # Track overlay info into graph node WITHOUT inflating visit_count.
            # 把 overlay 类型写回状态图节点元信息，但不增加 visit_count，避免污染统计。
            if nav:
                self._graph_annotate(cur_sig, overlay_kind=self._overlay_kind_value(nav))

            # 从 NAV 结果里提取当前页面的 overlay 类型，后面按类型分流处理。
            overlay_kind = self._overlay_kind_value(nav)

            nav_exhausted = False
            nav_exhausted_conf = 0.0
            nav_exhausted_reason = ""
            if nav is not None:
                try:
                    nav_exhausted = bool(getattr(nav, "exhausted", False))
                    nav_exhausted_conf = float(getattr(nav, "exhausted_confidence", 0.0) or 0.0)
                    nav_exhausted_reason = str(getattr(nav, "exhausted_reason", "") or "")
                except Exception:
                    nav_exhausted = False
                    nav_exhausted_conf = 0.0
                    nav_exhausted_reason = ""

            # Honor exhausted only with strong confidence to avoid false positives.
            if nav_exhausted and nav_exhausted_conf >= 0.75:
                fam = self._family_id(cur_sig)
                self.state_candidates[fam] = []
                self._log_event(
                    "nav_exhausted",
                    sig=cur_sig,
                    exhausted=True,
                    exhausted_confidence=nav_exhausted_conf,
                    exhausted_reason=nav_exhausted_reason[:220],
                    ui_type=str(getattr(nav, "ui_type", "") or "") if nav else "",
                    overlay_kind=overlay_kind,
                    candidate_count=len(getattr(nav, "candidate_actions", []) or []) if nav else 0,
                )
                self._emit_decision(
                    cur_sig,
                    "nav_exhausted",
                    {
                        "exhausted": True,
                        "confidence": nav_exhausted_conf,
                        "reason": nav_exhausted_reason[:220],
                        "ui_type": str(getattr(nav, "ui_type", "") or "") if nav else "",
                        "overlay_kind": overlay_kind,
                    },
                )

            # CONDITION: OVERLAY (blocking dismiss / loading)
            # 如果当前页面被可关闭弹窗阻塞，先优先处理弹窗，而不是继续正常探索。
            if overlay_kind == OverlayKind.DISMISS.value:
                # 记录“发现可关闭 overlay”的信息日志。
                logger.info("Overlay(DISMISS) at sig=%s: %s", cur_sig[:8], (getattr(nav, "overlay_reason", "") or "")[:140])
                # 把 overlay 检测结果写入 trace / log。
                self._log_event("overlay_detected", sig=cur_sig, overlay_kind=overlay_kind, reason=getattr(nav, "overlay_reason", ""))

                # 先尝试按 LLM1 给出的 dismiss 动作去关闭弹窗。
                resolved = self._dismiss_overlay_with_nav(cur_sig, snap, nav, task)
                # 如果 LLM1 没能关掉弹窗，则进入更通用的恢复流程。
                if not resolved:
                    # 记录告警，说明 overlay 处理失败，准备 recovery。
                    logger.warning("Overlay unresolved by LLM1 overlay_dismiss_actions; invoking recovery.")
                    # 调用 recovery 尝试把页面带回稳定非阻塞状态。
                    ok = self._recover(cur_sig, snap, reason=RecoveryReason.OVERLAY_UNRESOLVED, task=task, target_sig=None)
                    # 如果 recovery 也失败，只能通过重启 app + best-effort replay 自救。
                    if not ok:
                        # 输出错误日志，说明进入最重的恢复分支。
                        logger.error("Recovery failed. Restart app + replay best-effort.")
                        # 重启应用，并尽可能回到可继续探索的位置。
                        self._restart_and_replay(best_target=None, task=task, reason="overlay_unresolved_recovery_failed")

                # After overlay/recovery/restart: re-capture and replan at the true current state.
                # 不管 overlay 是怎么被解除的，最后都重新抓一次真实页面，重新建立当前基准。
                snap = self._capture_and_process() or snap
                # 同步更新当前状态签名。
                cur_sig = snap["state_sig"]
                # overlay 关闭后的状态通常不适合直接沿用旧栈，需要做一次外部移动式修正。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                # 对新页面重新安排分析。
                self._schedule_state(cur_sig, snap, task)
                # 本轮结束，开始下一轮。
                continue

            # 如果是 loading overlay，则先等它过去，而不是盲目点页面。
            if overlay_kind == OverlayKind.LOADING.value:
                # 记录 loading overlay 的出现。
                logger.info("Overlay(LOADING) at sig=%s: %s", cur_sig[:8], (getattr(nav, "overlay_reason", "") or "")[:140])
                # 把 loading 事件写入 trace。
                self._log_event("overlay_loading", sig=cur_sig, reason=getattr(nav, "overlay_reason", ""))
                # 调用 loading 专用处理逻辑，例如等待页面稳定。
                self._handle_loading_overlay(cur_sig, snap, task)
                # loading 结束后重新抓快照，拿到真正稳定的页面状态。
                snap = self._capture_and_process() or snap
                # 更新当前 sig。
                cur_sig = snap["state_sig"]
                # 修正 DFS 栈，使其与真实当前位置一致。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                # 重新安排后续分析。
                self._schedule_state(cur_sig, snap, task)
                # 当前轮结束。
                continue

            # WORKFLOW 类型 overlay 不是阻塞弹窗，而是流程提示类信息，这里先只记录下来供调试使用。
            if overlay_kind == OverlayKind.WORKFLOW.value:
                self._log_event("overlay_workflow", sig=cur_sig, reason=getattr(nav, "overlay_reason", ""), hints=getattr(nav, "workflow_hints", []))

            # Candidates: NAV if available; heuristics ONLY if NAV timed out/cooldown.
            # 生成本轮可尝试的候选动作；优先使用 NAV 结果，只有 NAV 不可用时才退化到启发式。
            if nav_exhausted and nav_exhausted_conf >= 0.75:
                candidates = []
            else:
                candidates = self._candidate_actions(nav=nav, snap=snap, allow_heuristics=using_heuristics)
            self._emit_decision(
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
            if nav is not None and (not using_heuristics) and not (nav_exhausted and nav_exhausted_conf >= 0.75):
                # 统计有没有至少一个候选动作还没被探索、尝试或拉黑。
                usable = 0
                # 逐个检查候选。
                for c in candidates:
                    # 没有动作内容的候选直接跳过。
                    if not getattr(c, "actions", None):
                        continue
                    # 构造候选动作的稳定 key，方便和尝试/黑名单集合做比对。
                    ckey = self._candidate_key(c)
                    # 如果这个候选已经探索过、尝试过或被临时拉黑，就不算“可用”。
                    if self._is_explored(cur_sig, c) or self._already_attempted(cur_sig, c) or self._is_action_blacklisted(cur_sig, ckey):
                        continue
                    # 找到一个可用候选就够了，不需要继续统计。
                    usable += 1
                    # 直接结束循环。
                    break
                # 如果 NAV 给了一堆候选，但最后一个都用不上，说明这次 NAV 质量偏低。
                if usable == 0:
                    # 记录低质量 NAV 的上下文信息，方便后续排查模型输出问题。
                    self._log_event(
                        "nav_low_quality",
                        sig=cur_sig,
                        candidate_count=len(getattr(nav, "candidate_actions", []) or []),
                        overlay_kind=self._overlay_kind_value(nav),
                        has_page_summary=bool(getattr(nav, "page_summary", "") or ""),
                        tag_count=len(getattr(nav, "page_tags", []) or []),
                    )
                    # 发出一个决策事件，标记这次 NAV 结果虽然完成了，但实际不可用。
                    self._emit_decision(cur_sig, "nav_low_quality", {"candidate_count": len(getattr(nav, "candidate_actions", []) or [])})

            # Probe-return exploration (evidence gathering for forward scoring)
            # 先对候选做 probe 探测，而不是立刻前进；目的是先知道这些动作分别会通向哪里。
            if self.budget.enable_probe_return:
                self._emit_decision(
                    cur_sig,
                    "next_step",
                    {"plan": "probe_candidates", "candidate_count": len(candidates or []), "probe_cap": int(self.budget.per_page_probe_cap)},
                )
                self._probe_candidates(cur_sig, snap, nav, candidates, task)
            else:
                self._log_event("probe_skipped", sig=cur_sig, reason="probe_return_disabled")
                self._emit_decision(cur_sig, "probe_skipped", {"reason": "probe_return_disabled"})

            # Probe can request forced replan (back-like / return_method=none / source mismatch)
            # probe 期间如果发现“实际上已经跳走了”，这里会触发强制重规划。
            if self.budget.enable_probe_return and self._consume_forced_replan():
                # 切换到 probe 过程确定的新状态与快照。
                cur_sig, snap = self._force_replan_sig, self._force_replan_snap  # type: ignore[assignment]
                # 读取是否已有记录边。
                has_edge = self._force_replan_has_edge
                # 清除该标记。
                self._clear_forced_replan()
                # 根据是否已有边来修正当前位置与 DFS 栈。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=not has_edge)
                # 对当前位置重新安排分析。
                self._schedule_state(cur_sig, snap, task)
                # 结束当前轮。
                continue

            # Optional post-probe wait (let pipelined Q updates land)
            # probe 之后可短暂等待一下，让刚刚 pipeline 出去的问卷更新任务有时间完成并落回状态。
            if self.budget.enable_probe_return:
                self._post_probe_wait()

            # Choose forward commit based on probe evidence + novelty + questionnaire yield.
            # 基于 probe 的落点、新颖度、问卷收益等信息，挑出本轮真正要 commit 的前进动作。
            self._emit_decision(cur_sig, "next_step", {"plan": "choose_forward"})
            forward = self._choose_forward(cur_sig, nav, candidates=candidates)

            # Beam switching compares against a *verified* local option only.
            # If forward is speculative (no probe outcome), treat local_score as -inf.
            # 默认把本地候选分数记成负无穷，意思是“还没有足够证据与全局切换比较”。
            local_score = float("-inf")
            # 只有当 forward 候选有明确评分且来自 probe 证据时，才把它当作有效本地方案。
            if forward is not None and self._last_forward_detail and self._last_forward_detail.get("score") is not None:
                try:
                    # 拿到 forward 候选的稳定 key。
                    fkey = self._candidate_key(forward)
                    # 查出 probe 期间这个候选实际落到的目标状态。
                    dst_sig = (self.probe_outcomes.get(self._family_id(cur_sig), {}) or {}).get(fkey)
                    # 只有确实落到了不同页面，才算一个有效的“前进选项”。
                    dst_ok = bool(dst_sig and dst_sig != cur_sig)
                    # 目标状态不能已经被探索穷尽。
                    if dst_ok and (not self._is_state_exhausted(str(dst_sig))):
                        # 看一下目标状态是否已经被判断为 overlay 阻塞页。
                        dst_nav = self.nav_cache.get(str(dst_sig))
                        # 提取目标状态的 overlay 类型。
                        overlay = self._overlay_kind_value(dst_nav)
                        # 只有目标状态不是 dismiss/loading 阻塞页，才采用它的本地评分。
                        if overlay not in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
                            local_score = float(self._last_forward_detail.get("score") or 0.0)
                # 如果评分计算过程中出了异常，退回到“没有可靠本地分数”。
                except Exception:
                    local_score = float("-inf")

            # Beam-first: switch to a higher-value global opportunity if worth the travel cost.
            # 在真正前进之前，再比较一下全局状态图里是否存在更值得切换过去的目标页面。
            beam_snap = self._maybe_beam_switch(cur_sig, local_score, snap, task)
            # 如果 beam 策略决定切换，并且已经把我们导航到了新页面，就以新页面作为当前基准继续。
            if beam_snap:
                # 更新当前快照。
                snap = beam_snap
                # 更新当前状态签名。
                cur_sig = snap["state_sig"]
                # 对“通过 beam 跳转后的位置”修正 DFS 栈。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                # 为新页面重新安排分析。
                self._schedule_state(cur_sig, snap, task)
                # 当前轮结束。
                continue

            # 如果没有选出 forward 候选，说明当前页此刻没有明确的前进动作可 commit。
            if forward is None:
                # Only backtrace when truly exhausted (all branches explored).
                # 如果当前状态已经被完全探索完，就优先考虑 DFS 回退，而不是原地乱试。
                if self._is_state_exhausted(cur_sig):
                    # 找最近一个还有剩余工作可做的祖先状态。
                    target = self._nearest_ancestor_with_remaining_work(cur_sig)

                    # 如果能找到这样的祖先，就执行回溯。
                    if target:
                        # 记录回溯目标。
                        logger.info("Exhausted sig=%s. Backtrace to ancestor sig=%s.", cur_sig[:8], target[:8])
                        # 试着按 DFS 路径退回去。
                        self._emit_decision(cur_sig, "next_step", {"plan": "backtrace_to_ancestor", "target_sig": target})
                        ok = self._backtrace_to(target_sig=target, task=task, start_snap=snap)
                        # 如果回溯失败，则进入恢复流程。
                        if not ok:
                            # 记录回溯失败。
                            logger.warning("Backtrace failed. Recover/restart.")
                            # recovery 尝试把我们拉回目标祖先或稳定页。
                            ok2 = self._recover(cur_sig, snap, reason=RecoveryReason.BACKTRACE_FAILED, task=task, target_sig=target)
                            # 如果 recovery 还失败，再升级到重启 + replay。
                            if not ok2:
                                self._restart_and_replay(best_target=target, task=task, reason="backtrace_failed_recovery_failed")

                        # 回溯 / 恢复 / 重启之后，重新抓当前真实页面。
                        snap = self._capture_and_process() or snap
                        # 更新当前 sig。
                        cur_sig = snap["state_sig"]
                        # 修正 DFS 栈与当前位置。
                        self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                        # 重新安排分析。
                        self._schedule_state(cur_sig, snap, task)
                        # 当前轮结束。
                        continue

                    # 如果祖先里也没有剩余工作，就尝试从全局状态图里挑一个 frontier 状态。
                    frontier = self._pick_global_frontier()
                    # 只有 frontier 合法且不是当前页时，才值得进行跨分支跳转。
                    if frontier and frontier != cur_sig:
                        # 记录准备前往全局 frontier。
                        logger.info("No ancestor work. Attempt to reach global frontier sig=%s", frontier[:8])
                        # 先尝试不重启，直接沿已知状态图导航过去。
                        self._emit_decision(cur_sig, "next_step", {"plan": "navigate_to_global_frontier", "target_sig": frontier})
                        reached, snap2 = self._navigate_via_graph(start_sig=cur_sig, start_snap=snap, target_sig=frontier, task=task)
                        # 如果状态图导航失败，则升级到重启 + replay。
                        if not reached:
                            self._restart_and_replay(best_target=frontier, task=task, reason="global_frontier_unreachable")
                            # 重启后重新抓一次页面，得到最新快照。
                            snap2 = self._capture_and_process(timeout=10.0) or snap2
                        # 采用导航后的快照；如果没有新快照则保留原快照。
                        snap = snap2 or snap
                        # 更新当前 sig。
                        cur_sig = snap["state_sig"]
                        # 修正当前位置在 DFS 栈中的表达。
                        self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                        # 对新页面重新安排分析。
                        self._schedule_state(cur_sig, snap, task)
                        # 本轮结束。
                        continue

                # 如果当前页不是 exhausted，但整体表现出“卡住”征兆，就走恢复分支。
                if self._should_recover_stuck():
                    # 打印 stuck 告警，包括卡了多少轮、多久没强进展。
                    logger.warning(
                        "STUCK: no progress loops=%d age=%.1fs at sig=%s -> recovery.",
                        self.no_progress_loops,
                        (time.time() - self.last_strong_progress_ts),
                        cur_sig[:8],
                    )
                    # 把 stuck 事件写入日志与 trace。
                    self._log_event("stuck_detected", sig=cur_sig, loops=self.no_progress_loops, age_s=(time.time() - self.last_strong_progress_ts))
                    # 尝试 recovery，把流程拉回一个可继续探索的状态。
                    self._emit_decision(
                        cur_sig,
                        "next_step",
                        {"plan": "recover_stuck", "loops": self.no_progress_loops, "age_s": (time.time() - self.last_strong_progress_ts)},
                    )
                    ok = self._recover(cur_sig, snap, reason=RecoveryReason.STUCK_NO_PROGRESS, task=task, target_sig=None)
                    # recovery 失败则走重启恢复。
                    if not ok:
                        self._restart_and_replay(best_target=None, task=task, reason="stuck_recovery_failed")

                    # 恢复后重新抓当前页面。
                    snap = self._capture_and_process() or snap
                    # 更新当前 sig。
                    cur_sig = snap["state_sig"]
                    # 修正当前位置与 DFS 栈。
                    self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                    # 重新安排分析。
                    self._schedule_state(cur_sig, snap, task)
                else:
                    # keep scheduling/waiting; do not heuristic-click unless NAV timed out
                    # 如果既不 exhausted 也不 stuck，就继续等待更多分析结果，不强行点击。
                    self._schedule_state(cur_sig, snap, task)
                # forward 不存在的分支到这里结束，本轮不再执行下面的 commit 逻辑。
                continue

            # Execute forward move (commit)
            # 走到这里，说明已经为当前页选出了一个真正值得提交执行的前进候选。
            cand_key = self._candidate_key(forward)
            # 取出第一步动作附带的 reasoning，方便日志里看到“为什么选它”。
            first_reason = (forward.actions[0].reasoning if forward.actions else "") or ""
            # 记录本轮前进提交的候选动作。
            self._emit_decision(
                cur_sig,
                "next_step",
                {"plan": "forward_commit", "candidate_key": cand_key, "reason": first_reason, "actions": self._actions_signature(forward.actions)},
            )
            logger.info("Forward commit from sig=%s: %s (%s)", cur_sig[:8], cand_key, first_reason[:90])
            # 如果有上一轮 forward 评分细节，也一起落到 trace 里。
            if self._last_forward_detail:
                self._log_event("forward_score", sig=cur_sig, **self._last_forward_detail)
            # 发出决策事件，标记当前正式选择了哪个动作序列。
            self._emit_decision(cur_sig, "forward_selected", {"actions": self._actions_signature(forward.actions)})
            # 真正执行这个动作序列。
            ok = self._execute_action_sequence(forward.actions, snap)

            # 如果动作执行层面就失败了，不能继续信任当前状态，需要恢复。
            if not ok:
                # 记录前进动作执行失败。
                logger.warning("Forward action failed at sig=%s -> recovery.", cur_sig[:8])
                # 进入 recovery。
                self._recover(cur_sig, snap, reason=RecoveryReason.FORWARD_ACTION_FAILED, task=task, target_sig=None)
                # 恢复后重新抓页面。
                snap = self._capture_and_process() or snap
                # 更新当前 sig。
                cur_sig = snap["state_sig"]
                # 修正 DFS 栈。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                # 为当前位置重新安排分析。
                self._schedule_state(cur_sig, snap, task)
                # 本轮结束。
                continue

            # Capture post-forward authoritative snapshot
            # 动作执行成功后，马上抓一份新的权威快照，确认我们实际落到了哪里。
            snap_next = self._capture_and_process()
            # 如果连前进后的快照都抓不到，同样需要恢复。
            if not snap_next:
                # 记录抓取失败。
                logger.warning("Capture failed after forward at sig=%s -> recovery.", cur_sig[:8])
                # 进入 recovery。
                self._recover(cur_sig, snap, reason=RecoveryReason.CAPTURE_FAILED_AFTER_FORWARD, task=task, target_sig=None)
                # 恢复后重新抓当前页。
                snap = self._capture_and_process() or snap
                # 更新 sig。
                cur_sig = snap["state_sig"]
                # 修正 DFS 栈。
                self._reconcile_stack_on_external_move(cur_sig, snap=snap, record_observation=False)
                # 重新安排分析。
                self._schedule_state(cur_sig, snap, task)
                # 本轮结束。
                continue

            # 提取前进后真正到达的新状态签名。
            new_sig = snap_next["state_sig"]

            # Record transition and get novelty-at-discovery
            # 把这次“当前页 -> 前进后页面”的迁移正式写入状态图，并拿到目标页是否为新状态。
            dst_was_new = self._graph_record_transition(
                cur_sig,
                new_sig,
                self._actions_signature(forward.actions, vid_map=snap.get("vid_map") or {}),
                src_snap=snap,
                dst_snap=snap_next,
            )
            # 标记当前页上的这个 forward 候选已经至少尝试过一次。
            self._mark_attempted(cur_sig, forward)

            # If forward returned to an ancestor or did nothing, mark explored to avoid re-committing endlessly.
            # 如果这个 forward 实际上没动，或者回到了祖先页，就把它标成 explored，避免以后重复提交。
            if new_sig == cur_sig or new_sig in self.dfs_stack:
                self._mark_explored(cur_sig, forward)

            # 用这次前进迁移更新 DFS 路径与 parent 关系。
            self._enter_state(from_sig=cur_sig, to_sig=new_sig, via_action=cand_key)

            # Saturation counter uses novelty-at-discovery (not visit_count after touch)
            # 如果发现了新状态，就重置“无新状态”计数。
            if dst_was_new:
                self.no_new_state_count = 0
            # 否则继续累计“连续没有发现新状态”的次数。
            else:
                self.no_new_state_count += 1

            # 把当前基准切到前进后的新状态。
            cur_sig, snap = new_sig, snap_next
            # 为新状态安排下一轮异步分析。
            self._schedule_state(cur_sig, snap, task)

        # 退出主循环后，尽力关闭线程池并取消剩余 future，避免后台任务继续占资源。
        try:
            # 不等待未完成任务自然结束，而是尽快取消它们，适合 run 结束时的清理语义。
            self._pool.shutdown(wait=False, cancel_futures=True)
        # 清理失败不影响 run 的最终返回，所以这里吞掉异常。
        except Exception:
            pass
        try:
            self._export_analysis_snapshot(cur_sig=cur_sig, stop_reason="run_exit")
        except Exception:
            logger.debug("final analysis export failed", exc_info=True)

    # ---------------------------
    # Snapshot
    # ---------------------------

    @staticmethod
    def _png_dimensions(png: bytes) -> Tuple[int, int]:
        """
        Extract PNG (width,height) from bytes via IHDR header (no full decode).
        Returns (0,0) on failure.
        """
        try:
            if not png or len(png) < 24:
                return (0, 0)
            if png[:8] != b"\x89PNG\r\n\x1a\n":
                return (0, 0)
            # PNG signature (8) + length (4) + type (4) => IHDR data starts at offset 16.
            if png[12:16] != b"IHDR":
                return (0, 0)
            w = int.from_bytes(png[16:20], "big", signed=False)
            h = int.from_bytes(png[20:24], "big", signed=False)
            return (int(w), int(h))
        except Exception:
            return (0, 0)

    @staticmethod
    def _xml_bounds_p95(xml: str) -> Tuple[int, int, int]:
        """
        Robust estimate of screen extents in XML coordinate space from UiAutomator2 bounds.
        Uses 95th percentile of x2/y2 to avoid rare outliers.
        Returns (w,h,count).
        """
        if not xml:
            return (0, 0, 0)
        try:
            m = re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", xml)
            if not m:
                return (0, 0, 0)
            x2s: List[int] = []
            y2s: List[int] = []
            for _, _, x2, y2 in m:
                try:
                    x2s.append(int(x2))
                    y2s.append(int(y2))
                except Exception:
                    continue
            if not x2s or not y2s:
                return (0, 0, 0)
            x2s.sort()
            y2s.sort()
            idx = int(0.95 * float(len(x2s) - 1)) if len(x2s) > 1 else 0
            w = int(x2s[max(0, min(len(x2s) - 1, idx))])
            h = int(y2s[max(0, min(len(y2s) - 1, idx))])
            return (w, h, int(len(x2s)))
        except Exception:
            return (0, 0, 0)

    @staticmethod
    def _infer_coord_scale(
        *,
        xml: str,
        png: bytes,
        device_pixel_ratio: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Infer a safe coordinate scale between UiAutomator2 XML bounds and screenshot pixels.

        UiAutomator2 bounds are *usually already in pixels*. DeviceInfo.pixelRatio is often a density
        scale (dp->px) and should NOT be blindly applied. We default to 1.0 and only scale up when
        the screenshot is consistently larger than XML bounds.
        """
        meta: Dict[str, Any] = {"device_pixel_ratio": float(device_pixel_ratio or 1.0)}

        png_w, png_h = WorkflowRunner._png_dimensions(png)
        xml_w, xml_h, n_bounds = WorkflowRunner._xml_bounds_p95(xml)
        meta.update({"png_w": png_w, "png_h": png_h, "xml_w": xml_w, "xml_h": xml_h, "xml_bounds_n": n_bounds})

        if png_w <= 0 or png_h <= 0 or xml_w <= 0 or xml_h <= 0:
            meta["coord_scale_reason"] = "missing_dims"
            return (1.0, meta)

        rw = float(png_w) / float(max(1, xml_w))
        rh = float(png_h) / float(max(1, xml_h))
        meta.update({"ratio_w": rw, "ratio_h": rh})

        if max(rw, rh) > 0:
            rel = abs(rw - rh) / max(rw, rh)
            meta["ratio_rel_diff"] = rel
            if rel > 0.12:
                meta["coord_scale_reason"] = "ratio_inconsistent"
                return (1.0, meta)

        ratio = 0.5 * (rw + rh)
        meta["ratio_avg"] = ratio

        if 0.90 <= ratio <= 1.10:
            meta["coord_scale_reason"] = "ratio_near_1"
            return (1.0, meta)

        # Only scale up (XML likely in dp). Scaling down is risky; prefer leaving as-is.
        if ratio < 1.15 or ratio > 6.0:
            meta["coord_scale_reason"] = "default_no_scale"
            return (1.0, meta)

        pr = float(device_pixel_ratio or 0.0)
        if pr > 0 and abs(ratio - pr) <= 0.20:
            meta["coord_scale_reason"] = "snapped_to_device_pixel_ratio"
            return (float(pr), meta)

        meta["coord_scale_reason"] = "inferred"
        return (float(round(ratio, 4)), meta)

    @staticmethod
    def _parse_uia_bounds(bounds: str) -> Optional[Tuple[int, int, int, int]]:
        if not bounds:
            return None
        m = re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", str(bounds))
        if not m:
            return None
        x1, y1, x2, y2 = m[0]
        try:
            return (int(x1), int(y1), int(x2), int(y2))
        except Exception:
            return None

    @staticmethod
    def _crop_png_top(png: bytes, crop_top_px: int) -> bytes:
        """
        Mask (blank) the top `crop_top_px` pixels of a PNG image with white.
        Keeps the coordinate system intact for actuation (taps) while making screenshots stable for diffs.
        Returns original bytes on failure.
        """
        try:
            crop_top_px = int(crop_top_px or 0)
            if crop_top_px <= 0:
                return png
            img = Image.open(io.BytesIO(png))
            w, h = img.size
            if crop_top_px >= h:
                return png
            # Ensure writable image
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            blank = Image.new(img.mode, (w, crop_top_px), (255, 255, 255, 255) if img.mode == "RGBA" else (255, 255, 255))
            img.paste(blank, (0, 0))
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
        except Exception:
            return png

    @staticmethod
    def _compute_status_bar_crop(
        *,
        xml: str,
        png_h: int,
        coord_scale: float,
        target_package: str = "",
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Derive a single top cutoff so screenshot + hierarchy can be normalized consistently.

        Returns:
          (crop_height_px, meta)
        """
        meta: Dict[str, Any] = {
            "status_bar_crop_px": 0,
            "status_bar_crop_xml": 0,
            "status_bar_crop_method": "none",
        }
        try:
            png_h = int(png_h or 0)
        except Exception:
            png_h = 0
        try:
            coord_scale = float(coord_scale or 1.0)
        except Exception:
            coord_scale = 1.0
        if coord_scale <= 0:
            coord_scale = 1.0

        if not xml or png_h <= 0:
            return 0, meta

        # Estimate screen height in XML coordinate units (dp/pixels depending on hierarchy).
        screen_h_xml = int(round(float(png_h) / float(coord_scale)))
        if screen_h_xml <= 0:
            screen_h_xml = 0

        crop_xml: Optional[int] = None
        try:
            root = ET.fromstring(xml)
        except Exception:
            root = None

        # Method 0 (preferred when available): use systemui status bar bounds directly.
        if crop_xml is None and root is not None:
            y2s: List[int] = []
            for el in root.iter():
                try:
                    if str(el.attrib.get("package") or "") != "com.android.systemui":
                        continue
                    b = el.attrib.get("bounds")
                    bb = WorkflowRunner._parse_uia_bounds(str(b) if b else "")
                    if not bb:
                        continue
                    _x1, y1, _x2, y2 = bb
                    if y2 <= 0:
                        continue
                    if y1 > 0:
                        continue
                    if screen_h_xml and y2 >= int(0.35 * float(screen_h_xml)):
                        continue
                    y2s.append(int(y2))
                except Exception:
                    continue
            if y2s:
                crop_xml = int(max(y2s))
                meta["status_bar_crop_method"] = "systemui_max_y2"

        # Method A: content_top_y from the first non-0 app container (fallback when systemui is absent).
        if crop_xml is None and root is not None and target_package:
            ys: List[int] = []
            for el in root.iter():
                try:
                    if str(el.attrib.get("package") or "") != str(target_package):
                        continue
                    b = el.attrib.get("bounds")
                    bb = WorkflowRunner._parse_uia_bounds(str(b) if b else "")
                    if not bb:
                        continue
                    _, y1, _, y2 = bb
                    if y1 <= 0:
                        continue
                    if screen_h_xml and y1 >= int(0.35 * float(screen_h_xml)):
                        continue
                    if y2 <= y1:
                        continue
                    ys.append(int(y1))
                except Exception:
                    continue
            if ys:
                crop_xml = int(min(ys))
                meta["status_bar_crop_method"] = "content_top_y"

        # Method B: hierarchy height delta (works on many UiAutomator2 dumps).
        if crop_xml is None and root is not None and screen_h_xml:
            try:
                h_attr = root.attrib.get("height")
                h_xml = int(float(h_attr)) if h_attr is not None else 0
                cand = int(screen_h_xml - h_xml)
                if cand > 0:
                    crop_xml = cand
                    meta["status_bar_crop_method"] = "hierarchy_height_delta"
            except Exception:
                pass

        # Method C: min positive y across non-system nodes.
        if crop_xml is None and root is not None:
            ys: List[int] = []
            for el in root.iter():
                try:
                    pkg = str(el.attrib.get("package") or "")
                    if pkg == "com.android.systemui":
                        continue
                    b = el.attrib.get("bounds")
                    bb = WorkflowRunner._parse_uia_bounds(str(b) if b else "")
                    if not bb:
                        continue
                    _, y1, _, y2 = bb
                    if y1 <= 0:
                        continue
                    if screen_h_xml and y1 >= int(0.35 * float(screen_h_xml)):
                        continue
                    if y2 <= y1:
                        continue
                    ys.append(int(y1))
                except Exception:
                    continue
            if ys:
                crop_xml = int(min(ys))
                meta["status_bar_crop_method"] = "min_positive_y"

        if crop_xml is None:
            return 0, meta

        # Sanity bounds: avoid cropping too aggressively.
        try:
            crop_xml = int(crop_xml)
        except Exception:
            return 0, meta
        if crop_xml <= 0:
            return 0, meta
        if screen_h_xml and crop_xml > int(0.30 * float(screen_h_xml)):
            return 0, meta
        if crop_xml > 600:
            return 0, meta

        crop_px = int(round(float(crop_xml) * float(coord_scale)))
        crop_px = max(0, min(int(png_h), int(crop_px)))
        meta["status_bar_crop_px"] = crop_px
        meta["status_bar_crop_xml"] = crop_xml
        meta["status_bar_screen_h_xml_est"] = screen_h_xml
        return crop_px, meta

    @staticmethod
    def _preprocess_hierarchy_xml(xml: str, *, crop_top_xml: int) -> str:
        """
        Produce a processed hierarchy XML with the status bar region removed:
        - removes nodes fully within top crop region

        IMPORTANT:
        - We DO NOT shift bounds. This preserves the original coordinate system so coordinate taps remain correct.
          (Processed screenshot is masked rather than cropped.)
        """
        try:
            crop_top_xml = int(crop_top_xml or 0)
        except Exception:
            crop_top_xml = 0
        if crop_top_xml <= 0 or not xml:
            return xml

        try:
            root = ET.fromstring(xml)
        except Exception:
            return xml

        def keep(elem: ET.Element) -> bool:
            # Process children first so parents can be removed if empty.
            kept_children: List[ET.Element] = []
            for ch in list(elem):
                if keep(ch):
                    kept_children.append(ch)
            elem[:] = kept_children

            b = elem.attrib.get("bounds")
            bb = WorkflowRunner._parse_uia_bounds(str(b) if b else "")
            if not bb:
                # Keep structural nodes if they still have children.
                return bool(list(elem)) or (elem is root)

            _x1, _y1, _x2, y2 = bb
            # Remove nodes fully within the masked status bar region.
            if int(y2) <= int(crop_top_xml):
                return False
            return True

        # Root cannot be removed.
        for ch in list(root):
            if not keep(ch):
                root.remove(ch)

        try:
            body = ET.tostring(root, encoding="unicode")
        except Exception:
            return xml

        # Standardize header for deterministic diffs.
        return "<?xml version='1.0' encoding='UTF-8'?>\n" + body

    def _capture_and_process(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """

        (use uied)
        IPO:
          in : device UI state (implicit)
          out: {xml, screenshot(b64), uist, vid_map, state_sig} or None

        WHEN called:
          - At entry
          - Once per main-loop (drift check)
          - After every action sequence (probe/overlay/recovery/forward/backtrace/replay)

        WHY:
          - Defines the single source of truth for current state_sig.
          - Prevents planning on mixed frames (xml from one, screenshot from another).
        """
        try:
            raw = self.appium.capture_snapshot(timeout=timeout)
            xml_raw = raw.get("xml", "")
            xml_hash_raw = str(raw.get("xml_hash") or "")
            screenshot_b64_raw = raw.get("screenshot", "")
            screenshot_hash_raw = str(raw.get("screenshot_hash") or "")
            info = raw.get("device_info") or {}
            device_pixel_ratio = float(info.get("pixelRatio", 1.0) or 1.0)

            # Decode screenshot bytes once for hashing/scale inference (best-effort).
            png_bytes_raw: bytes = b""
            if screenshot_b64_raw:
                try:
                    png_bytes_raw = base64.b64decode(str(screenshot_b64_raw) + "==", validate=False)
                except Exception:
                    png_bytes_raw = b""

            # Back-compat: if older capture_snapshot doesn't provide hashes, compute them here.
            if not xml_hash_raw:
                xml_hash_raw = hashlib.md5((xml_raw or "").encode("utf-8")).hexdigest()
            if not screenshot_hash_raw:
                if png_bytes_raw:
                    screenshot_hash_raw = hashlib.md5(png_bytes_raw).hexdigest()
                else:
                    # Last resort: stable-ish fallback (won't match screenshot_png_hash()).
                    screenshot_hash_raw = hashlib.md5((screenshot_b64_raw or "").encode("utf-8")).hexdigest()

            coord_scale, scale_meta = self._infer_coord_scale(xml=xml_raw, png=png_bytes_raw, device_pixel_ratio=device_pixel_ratio)

            # Foreground context (package/activity) is part of the signature space:
            # Launcher and target app must never collide even if UI text/structure looks similar.
            try:
                foreground_package = self.appium.foreground_package()
            except Exception:
                foreground_package = ""
            try:
                foreground_activity = self.appium.foreground_activity()
            except Exception:
                foreground_activity = ""

            # --------- Normalization: mask status bar (screenshot + hierarchy) ---------
            png_h_raw = int(scale_meta.get("png_h") or 0) if isinstance(scale_meta, dict) else 0
            if png_h_raw <= 0 and png_bytes_raw:
                _, png_h_raw = self._png_dimensions(png_bytes_raw)

            crop_px, crop_meta = self._compute_status_bar_crop(
                xml=xml_raw,
                png_h=png_h_raw,
                coord_scale=coord_scale,
                target_package=str(self.target_package or ""),
            )
            crop_xml = int(crop_meta.get("status_bar_crop_xml") or 0)

            # Processed screenshot (masked) for stable diffs; keep raw for debugging.
            png_bytes = png_bytes_raw
            screenshot_b64 = screenshot_b64_raw
            if png_bytes_raw and crop_px > 0:
                png_bytes = self._crop_png_top(png_bytes_raw, crop_px)
                screenshot_b64 = base64.b64encode(png_bytes).decode("utf-8")
            screenshot_hash = hashlib.md5((png_bytes or b"")).hexdigest() if png_bytes else hashlib.md5((screenshot_b64 or "").encode("utf-8")).hexdigest()
            screenshot_phash = compute_screenshot_phash(screenshot_b64)

            # Processed XML: remove top status bar region; keep raw for debugging.
            xml = self._preprocess_hierarchy_xml(xml_raw, crop_top_xml=crop_xml) if crop_xml > 0 else (xml_raw or "")
            xml_hash = hashlib.md5((xml or "").encode("utf-8")).hexdigest()

            xml_reliable = count_meaningful_xml_nodes(xml) >= int(self.budget.meaningful_xml_nodes_threshold)
            postprocess_mode = "xml_only" if xml_reliable else "uied_first"
            logger.info("UIED mode: %s (xml_reliable=%s)", "skipped" if xml_reliable else "used", xml_reliable)

            raw_key = self._snapshot_cache_key(
                xml_hash,
                screenshot_hash,
                coord_scale,
                foreground_package=foreground_package,
                foreground_activity=foreground_activity,
            ) + f":{postprocess_mode}"
            cached = self.snapshot_cache.get(raw_key)
            if cached:
                self.snapshot_cache.move_to_end(raw_key)
                self.snapshot_cache_hits += 1
                uist2 = copy.deepcopy(cached["uist"])
                vid_map = copy.deepcopy(cached["vid_map"])
                sig = compute_state_signature(
                    uist2,
                    foreground_package=foreground_package,
                    foreground_activity=foreground_activity,
                )
                self._log_event("snapshot_cache_hit", sig=sig, raw_key=raw_key, cache_size=len(self.snapshot_cache))
            else:
                self.snapshot_cache_misses += 1
                uist = self.appium.parse_xml_to_uist(xml, pixel_ratio=coord_scale)
                if xml_reliable:
                    uist2, vid_map = BaseUI.post_process_ui(uist, screenshot_b64, device_info=info)
                else:
                    uist2, vid_map = BaseUI.post_process_ui_uied_first(uist, screenshot_b64, device_info=info)
                sig = compute_state_signature(
                    uist2,
                    foreground_package=foreground_package,
                    foreground_activity=foreground_activity,
                )
                # Store deep copies to avoid mutable sharing across cache hits.
                self.snapshot_cache[raw_key] = {"uist": copy.deepcopy(uist2), "vid_map": copy.deepcopy(vid_map), "state_sig": sig}
                if len(self.snapshot_cache) > self.snapshot_cache_max:
                    self.snapshot_cache.popitem(last=False)
                self._log_event("snapshot_cache_miss", sig=sig, raw_key=raw_key, cache_size=len(self.snapshot_cache))

            coarse_sig = compute_coarse_signature(uist2, foreground_package=foreground_package, foreground_activity=foreground_activity)
            struct_sig = compute_structural_signature(uist2, foreground_package=foreground_package, foreground_activity=foreground_activity)
            fine_sig = compute_fine_signature(uist2, foreground_package=foreground_package, foreground_activity=foreground_activity)
            identity = self._resolve_state_identity(
                xml_reliable=xml_reliable,
                xml_state_sig=sig,
                struct_sig=struct_sig,
                screenshot_phash=screenshot_phash,
                foreground_package=foreground_package,
                foreground_activity=foreground_activity,
            )
            sig = str(identity.get("state_sig") or sig or "")
            try:
                if sig and struct_sig:
                    self.sig_to_family[sig] = struct_sig
            except Exception:
                pass
            snap = {
                "xml": xml,  # 处理后的 XML：后续解析/UI 判断真正使用的层级树
                "xml_raw": xml_raw,  # 原始 XML：保留给调试对照
                "screenshot": screenshot_b64,  # 处理后的截图：后续 OCR/UIED/相似度使用
                "screenshot_raw": screenshot_b64_raw,  # 原始截图：保留给调试对照
                "uist": uist2,  # 后处理后的 UI 树：签名、LLM、动作定位都基于它
                "vid_map": vid_map,  # element_id -> 节点映射：动作执行时靠它找元素
                "state_sig": sig,  # 当前页面主签名：状态图/cache 的核心 key
                "xml_reliable": xml_reliable,  # 顶层冗余一份 XML 可信标记，方便调试和状态判断
                "device_info": info,  # 设备信息：分辨率、像素比等
                "meta": {
                    "raw_key": raw_key,  # 这次快照在 snapshot cache 里的 key
                    "cache_hit": bool(cached),  # 是否命中了 snapshot cache
                    "cache_size": len(self.snapshot_cache),  # 当前 snapshot cache 大小
                    "xml_hash": xml_hash,  # 处理后 XML 的 hash
                    "screenshot_hash": screenshot_hash,  # 处理后截图的 hash
                    "screenshot_phash": screenshot_phash,  # 处理后截图的感知 hash（视觉相似度）
                    "xml_hash_raw": xml_hash_raw,  # 原始 XML 的 hash
                    "screenshot_hash_raw": screenshot_hash_raw,  # 原始截图的 hash
                    "xml_reliable": xml_reliable,  # XML 是否足够可靠，可用于状态判断
                    "coord_scale": coord_scale,  # XML 坐标到截图坐标的缩放比例
                    "coarse_sig": coarse_sig,  # 粗粒度签名：宽松比较页面
                    "struct_sig": struct_sig,  # 结构签名：probe-return 回页时常用
                    "fine_sig": fine_sig,  # 细粒度签名：更严格地区分页面
                    "identity_source": str(identity.get("identity_source") or ""),  # 当前 state_sig 来自 xml 还是 phash
                    "identity_hash": str(identity.get("identity_hash") or ""),  # 本次状态身份判定依赖的核心 hash
                    "matched_existing": bool(identity.get("matched_existing")),  # phash 模式下是否复用了历史视觉状态
                    "matched_similarity": float(identity.get("matched_similarity") or 0.0),  # 命中历史视觉状态时的相似度
                    "foreground_package": foreground_package,  # 当前前台包名
                    "foreground_activity": foreground_activity,  # 当前前台 activity
                    **(crop_meta or {}),  # 裁切相关元信息：状态栏裁掉了多少等
                    **scale_meta,  # 尺寸/缩放相关元信息：png/xml 尺寸等
                },
            }
            # Keep hashes attached to graph nodes for debugging/disambiguation without inflating visits.
            self._graph_annotate(
                sig,
                meta={
                    "coarse_sig": coarse_sig,
                    "struct_sig": struct_sig,
                    "fine_sig": fine_sig,
                    "xml_reliable": xml_reliable,
                    "xml_hash": xml_hash,
                    "screenshot_hash": screenshot_hash,
                    "screenshot_phash": screenshot_phash,
                    "xml_hash_raw": xml_hash_raw,
                    "screenshot_hash_raw": screenshot_hash_raw,
                    "identity_source": str(identity.get("identity_source") or ""),
                    "identity_hash": str(identity.get("identity_hash") or ""),
                    "matched_existing": bool(identity.get("matched_existing")),
                    "matched_similarity": float(identity.get("matched_similarity") or 0.0),
                    "coord_scale": coord_scale,
                    "foreground_package": foreground_package,
                    "foreground_activity": foreground_activity,
                    **(crop_meta or {}),
                },
            )
            self._emit_snapshot(snap)
            return snap
        except Exception:
            logger.exception("capture_and_process failed")
            return None
        

    def _capture_and_process2(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        (no uied)
        IPO:
          in : device UI state (implicit)
          out: {xml, screenshot(b64), uist, vid_map, state_sig} or None

        WHEN called:
          - At entry
          - Once per main-loop (drift check)
          - After every action sequence (probe/overlay/recovery/forward/backtrace/replay)

        WHY:
          - Defines the single source of truth for current state_sig.
          - Prevents planning on mixed frames (xml from one, screenshot from another).
        """
        try:
            raw = self.appium.capture_snapshot(timeout=timeout)
            xml_raw = raw.get("xml", "")
            xml_hash_raw = str(raw.get("xml_hash") or "")
            screenshot_b64_raw = raw.get("screenshot", "")
            screenshot_hash_raw = str(raw.get("screenshot_hash") or "")
            info = raw.get("device_info") or {}
            device_pixel_ratio = float(info.get("pixelRatio", 1.0) or 1.0)

            # Decode screenshot bytes once for hashing/scale inference (best-effort).
            png_bytes_raw: bytes = b""
            if screenshot_b64_raw:
                try:
                    png_bytes_raw = base64.b64decode(str(screenshot_b64_raw) + "==", validate=False)
                except Exception:
                    png_bytes_raw = b""

            # Back-compat: if older capture_snapshot doesn't provide hashes, compute them here.
            if not xml_hash_raw:
                xml_hash_raw = hashlib.md5((xml_raw or "").encode("utf-8")).hexdigest()
            if not screenshot_hash_raw:
                if png_bytes_raw:
                    screenshot_hash_raw = hashlib.md5(png_bytes_raw).hexdigest()
                else:
                    # Last resort: stable-ish fallback (won't match screenshot_png_hash()).
                    screenshot_hash_raw = hashlib.md5((screenshot_b64_raw or "").encode("utf-8")).hexdigest()

            coord_scale, scale_meta = self._infer_coord_scale(xml=xml_raw, png=png_bytes_raw, device_pixel_ratio=device_pixel_ratio)

            # Foreground context (package/activity) is part of the signature space:
            # Launcher and target app must never collide even if UI text/structure looks similar.
            try:
                foreground_package = self.appium.foreground_package()
            except Exception:
                foreground_package = ""
            try:
                foreground_activity = self.appium.foreground_activity()
            except Exception:
                foreground_activity = ""

            # --------- Normalization: mask status bar (screenshot + hierarchy) ---------
            png_h_raw = int(scale_meta.get("png_h") or 0) if isinstance(scale_meta, dict) else 0
            if png_h_raw <= 0 and png_bytes_raw:
                _, png_h_raw = self._png_dimensions(png_bytes_raw)

            crop_px, crop_meta = self._compute_status_bar_crop(
                xml=xml_raw,
                png_h=png_h_raw,
                coord_scale=coord_scale,
                target_package=str(self.target_package or ""),
            )
            crop_xml = int(crop_meta.get("status_bar_crop_xml") or 0)

            # Processed screenshot (masked) for stable diffs; keep raw for debugging.
            png_bytes = png_bytes_raw
            screenshot_b64 = screenshot_b64_raw
            if png_bytes_raw and crop_px > 0:
                png_bytes = self._crop_png_top(png_bytes_raw, crop_px)
                screenshot_b64 = base64.b64encode(png_bytes).decode("utf-8")
            screenshot_hash = hashlib.md5((png_bytes or b"")).hexdigest() if png_bytes else hashlib.md5((screenshot_b64 or "").encode("utf-8")).hexdigest()

            # Processed XML: remove top status bar region; keep raw for debugging.
            xml = self._preprocess_hierarchy_xml(xml_raw, crop_top_xml=crop_xml) if crop_xml > 0 else (xml_raw or "")
            xml_hash = hashlib.md5((xml or "").encode("utf-8")).hexdigest()

            raw_key = self._snapshot_cache_key(
                xml_hash,
                screenshot_hash,
                coord_scale,
                foreground_package=foreground_package,
                foreground_activity=foreground_activity,
            )
            cached = self.snapshot_cache.get(raw_key)
            if cached:
                self.snapshot_cache.move_to_end(raw_key)
                self.snapshot_cache_hits += 1
                uist2 = copy.deepcopy(cached["uist"])
                vid_map = copy.deepcopy(cached["vid_map"])
                sig = compute_state_signature(
                    uist2,
                    foreground_package=foreground_package,
                    foreground_activity=foreground_activity,
                )
                self._log_event("snapshot_cache_hit", sig=sig, raw_key=raw_key, cache_size=len(self.snapshot_cache))
            else:
                self.snapshot_cache_misses += 1
                uist = self.appium.parse_xml_to_uist(xml, pixel_ratio=coord_scale)
                # 这里是“页面分析”的真正入口：
                # 1. 先把 Appium XML 解析成内部树 uist
                # 2. 再交给 BaseUI.post_process_ui* 做后处理
                #
                # 当前这条链走的是原始 post_process_ui()。
                # 如果后面要切到 UIED-first 调试链，可以在这里改成
                # BaseUI.post_process_ui_uied_first(...)
                uist2, vid_map = BaseUI.post_process_ui(uist, screenshot_b64, device_info=info)
                sig = compute_state_signature(
                    uist2,
                    foreground_package=foreground_package,
                    foreground_activity=foreground_activity,
                )
                # Store deep copies to avoid mutable sharing across cache hits.
                self.snapshot_cache[raw_key] = {"uist": copy.deepcopy(uist2), "vid_map": copy.deepcopy(vid_map), "state_sig": sig}
                if len(self.snapshot_cache) > self.snapshot_cache_max:
                    self.snapshot_cache.popitem(last=False)
                self._log_event("snapshot_cache_miss", sig=sig, raw_key=raw_key, cache_size=len(self.snapshot_cache))

            screenshot_phash = compute_screenshot_phash(screenshot_b64)
            xml_reliable = count_meaningful_xml_nodes(xml) >= int(self.budget.meaningful_xml_nodes_threshold)
            coarse_sig = compute_coarse_signature(uist2, foreground_package=foreground_package, foreground_activity=foreground_activity)
            struct_sig = compute_structural_signature(uist2, foreground_package=foreground_package, foreground_activity=foreground_activity)
            fine_sig = compute_fine_signature(uist2, foreground_package=foreground_package, foreground_activity=foreground_activity)
            identity = self._resolve_state_identity(
                xml_reliable=xml_reliable,
                xml_state_sig=sig,
                struct_sig=struct_sig,
                screenshot_phash=screenshot_phash,
                foreground_package=foreground_package,
                foreground_activity=foreground_activity,
            )
            sig = str(identity.get("state_sig") or sig or "")
            try:
                if sig and struct_sig:
                    self.sig_to_family[sig] = struct_sig
            except Exception:
                pass
            snap = {
                "xml": xml,  # 处理后的 XML：后续解析/UI 判断真正使用的层级树
                "xml_raw": xml_raw,  # 原始 XML：保留给调试对照
                "screenshot": screenshot_b64,  # 处理后的截图：后续 OCR/相似度使用
                "screenshot_raw": screenshot_b64_raw,  # 原始截图：保留给调试对照
                "uist": uist2,  # 后处理后的 UI 树：签名、LLM、动作定位都基于它
                "vid_map": vid_map,  # element_id -> 节点映射：动作执行时靠它找元素
                "state_sig": sig,  # 当前页面主签名：状态图/cache 的核心 key
                "xml_reliable": xml_reliable,  # 顶层冗余一份 XML 可信标记，方便调试和状态判断
                "device_info": info,  # 设备信息：分辨率、像素比等
                "meta": {
                    "raw_key": raw_key,  # 这次快照在 snapshot cache 里的 key
                    "cache_hit": bool(cached),  # 是否命中了 snapshot cache
                    "cache_size": len(self.snapshot_cache),  # 当前 snapshot cache 大小
                    "xml_hash": xml_hash,  # 处理后 XML 的 hash
                    "screenshot_hash": screenshot_hash,  # 处理后截图的 hash
                    "xml_hash_raw": xml_hash_raw,  # 原始 XML 的 hash
                    "screenshot_hash_raw": screenshot_hash_raw,  # 原始截图的 hash
                    "screenshot_phash": screenshot_phash,  # 处理后截图的感知 hash（视觉相似度）
                    "xml_reliable": xml_reliable,  # XML 是否足够可靠，可用于状态判断
                    "coord_scale": coord_scale,  # XML 坐标到截图坐标的缩放比例
                    "coarse_sig": coarse_sig,  # 粗粒度签名：宽松比较页面
                    "struct_sig": struct_sig,  # 结构签名：probe-return 回页时常用
                    "fine_sig": fine_sig,  # 细粒度签名：更严格地区分页面
                    "identity_source": str(identity.get("identity_source") or ""),  # 当前 state_sig 来自 xml 还是 phash
                    "identity_hash": str(identity.get("identity_hash") or ""),  # 本次状态身份判定依赖的核心 hash
                    "matched_existing": bool(identity.get("matched_existing")),  # phash 模式下是否复用了历史视觉状态
                    "matched_similarity": float(identity.get("matched_similarity") or 0.0),  # 命中历史视觉状态时的相似度
                    "foreground_package": foreground_package,  # 当前前台包名
                    "foreground_activity": foreground_activity,  # 当前前台 activity
                    **(crop_meta or {}),  # 裁切相关元信息：状态栏裁掉了多少等
                    **scale_meta,  # 尺寸/缩放相关元信息：png/xml 尺寸等
                },
            }
            # Keep hashes attached to graph nodes for debugging/disambiguation without inflating visits.
            self._graph_annotate(
                sig,
                meta={
                    "coarse_sig": coarse_sig,
                    "struct_sig": struct_sig,
                    "fine_sig": fine_sig,
                    "xml_reliable": xml_reliable,
                    "xml_hash": xml_hash,
                    "screenshot_hash": screenshot_hash,
                    "screenshot_phash": screenshot_phash,
                    "xml_hash_raw": xml_hash_raw,
                    "screenshot_hash_raw": screenshot_hash_raw,
                    "identity_source": str(identity.get("identity_source") or ""),
                    "identity_hash": str(identity.get("identity_hash") or ""),
                    "matched_existing": bool(identity.get("matched_existing")),
                    "matched_similarity": float(identity.get("matched_similarity") or 0.0),
                    "coord_scale": coord_scale,
                    "foreground_package": foreground_package,
                    "foreground_activity": foreground_activity,
                    **(crop_meta or {}),
                },
            )
            self._emit_snapshot(snap)
            return snap
        except Exception:
            logger.exception("capture_and_process failed")
            return None

    def _snapshot_cache_key(
        self,
        xml_hash: str,
        screenshot_hash: str,
        coord_scale: float,
        *,
        foreground_package: str = "",
        foreground_activity: str = "",
    ) -> str:
        bundle = BaseUI.postprocess_version_bundle()
        m3 = hashlib.md5(bundle.encode("utf-8")).hexdigest()
        return f"{xml_hash}:{screenshot_hash}:{float(coord_scale or 1.0):.4f}:{m3}:{str(foreground_package or '')}:{str(foreground_activity or '')}"

    @staticmethod
    def _state_sig_from_xml(xml_state_sig: str) -> str:
        xml_state_sig = str(xml_state_sig or "").strip()
        return f"xml:{xml_state_sig}" if xml_state_sig else ""

    @staticmethod
    def _state_sig_from_phash(screenshot_phash: str, *, foreground_package: str = "", foreground_activity: str = "") -> str:
        blob = json.dumps(
            {
                "foreground_package": str(foreground_package or ""),
                "foreground_activity": str(foreground_activity or ""),
                "screenshot_phash": str(screenshot_phash or ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"phash:{hashlib.md5(blob.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _family_sig_from_struct(struct_sig: str) -> str:
        struct_sig = str(struct_sig or "").strip()
        return f"struct:{struct_sig}" if struct_sig else ""

    def _resolve_state_identity(
        self,
        *,
        xml_reliable: bool,
        xml_state_sig: str,
        struct_sig: str,
        screenshot_phash: str,
        foreground_package: str = "",
        foreground_activity: str = "",
    ) -> Dict[str, Any]:
        """
        Select the canonical state key used everywhere else in the workflow.

        POLICY:
          - Reliable XML: keep using XML-derived state signature.
          - Unreliable XML: switch to a screenshot-phash-backed signature and
            reuse an existing visual state when similarity is high enough.
        """
        xml_state_sig = str(xml_state_sig or "")
        struct_sig = str(struct_sig or "")
        screenshot_phash = str(screenshot_phash or "")
        foreground_package = str(foreground_package or "")
        foreground_activity = str(foreground_activity or "")

        if xml_reliable and xml_state_sig:
            state_sig = self._state_sig_from_xml(xml_state_sig)
            family_sig = self._family_sig_from_struct(struct_sig) or state_sig
            return {
                "state_sig": state_sig,
                "family_sig": family_sig,
                "identity_source": "xml",
                "identity_hash": xml_state_sig,
                "matched_existing": False,
                "matched_similarity": 1.0,
            }

        if screenshot_phash:
            best_sig = ""
            best_similarity = -1.0
            for known_sig, meta in self.visual_state_registry.items():
                if str(meta.get("identity_source") or "") != "phash":
                    continue
                if str(meta.get("foreground_package") or "") != foreground_package:
                    continue
                if str(meta.get("foreground_activity") or "") != foreground_activity:
                    continue
                known_phash = str(meta.get("screenshot_phash") or "")
                if not known_phash:
                    continue
                sim = compare_phash_similarity(screenshot_phash, known_phash)
                if sim > best_similarity:
                    best_similarity = sim
                    best_sig = known_sig
            if best_sig and best_similarity >= float(self.budget.screenshot_phash_similarity_threshold):
                self.visual_state_registry[best_sig] = {
                    "identity_source": "phash",
                    "screenshot_phash": screenshot_phash,
                    "foreground_package": foreground_package,
                    "foreground_activity": foreground_activity,
                    "last_similarity": best_similarity,
                    "updated_ts": time.time(),
                }
                return {
                    "state_sig": best_sig,
                    "family_sig": best_sig,
                    "identity_source": "phash",
                    "identity_hash": screenshot_phash,
                    "matched_existing": True,
                    "matched_similarity": best_similarity,
                }

            state_sig = self._state_sig_from_phash(
                screenshot_phash,
                foreground_package=foreground_package,
                foreground_activity=foreground_activity,
            )
            self.visual_state_registry[state_sig] = {
                "identity_source": "phash",
                "screenshot_phash": screenshot_phash,
                "foreground_package": foreground_package,
                "foreground_activity": foreground_activity,
                "last_similarity": 1.0,
                "updated_ts": time.time(),
            }
            return {
                "state_sig": state_sig,
                "family_sig": state_sig,
                "identity_source": "phash",
                "identity_hash": screenshot_phash,
                "matched_existing": False,
                "matched_similarity": 1.0,
            }

        # Last-resort fallback: keep the XML signature so the workflow still has a state key.
        state_sig = self._state_sig_from_xml(xml_state_sig)
        family_sig = self._family_sig_from_struct(struct_sig) or state_sig
        return {
            "state_sig": state_sig,
            "family_sig": family_sig,
            "identity_source": "xml_fallback",
            "identity_hash": xml_state_sig,
            "matched_existing": False,
            "matched_similarity": 0.0,
        }

    def _preflight_refresh_if_changed(self, snap: Dict[str, Any], *, timeout_s: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Cheap change detector to avoid full capture+postprocess when UI is stable.

        POLICY:
          - Prefer XML hash when the hierarchy is reliable (most apps).
          - Fall back to screenshot hash when XML is empty/sparse (Unity/canvas UIs).
          - If the chosen signal changed, do ONE authoritative refresh via _capture_and_process().

        NOTE:
          - This function is NOT authoritative; it only decides whether to refresh.
          - The authoritative barrier remains _capture_and_process() (paired xml+screenshot).
        """
        meta = snap.get("meta") or {}
        # Preflight compares against the *processed* representation used by the workflow (status bar masked/removed).
        exp_xml_hash = str(meta.get("xml_hash") or "")
        exp_shot_hash = str(meta.get("screenshot_hash") or "")
        exp_shot_phash = str(meta.get("screenshot_phash") or "")
        xml_reliable = bool(meta.get("xml_reliable", True))
        crop_top_xml = int(meta.get("status_bar_crop_xml") or 0)
        crop_top_px = int(meta.get("status_bar_crop_px") or 0)

        # If NAV classified this as a screenshot-driven surface, prefer screenshot for change detection.
        try:
            sig = str(snap.get("state_sig") or "")
            nav = self.nav_cache.get(sig)
            ui_type = str(getattr(nav, "ui_type", "") or "").strip().lower() if nav else ""
            if ui_type in ("webview", "game_canvas", "video_player"):
                xml_reliable = False
        except Exception:
            pass

        try:
            if xml_reliable and exp_xml_hash:# xml 可信且有历史 hash，优先用 XML 判断变化（大多数 app 都是这个情况）
                xml_now = self.appium.page_source_once()
                if not xml_now:
                    return self._capture_and_process(timeout=timeout_s)
                if crop_top_xml > 0:
                    xml_now = self._preprocess_hierarchy_xml(xml_now, crop_top_xml=crop_top_xml)
                cur = hashlib.md5((xml_now or "").encode("utf-8")).hexdigest()
                if cur == exp_xml_hash:
                    return None
            elif exp_shot_hash:# XML 不可信但有历史截图 hash，先判断截图hash是否相同(完全相同界面),在判断Phash是否相似(处理动态界面).
                png_now = self.appium.screenshot_png_once()
                if not png_now:
                    return self._capture_and_process(timeout=timeout_s)
                if crop_top_px > 0:
                    png_now = self._crop_png_top(png_now, crop_top_px)
                cur = hashlib.md5(png_now).hexdigest()
                if cur == exp_shot_hash:
                    return None
                if exp_shot_phash:
                    shot_now_b64 = base64.b64encode(png_now).decode("utf-8")
                    cur_phash = compute_screenshot_phash(shot_now_b64)
                    sim = compare_phash_similarity(exp_shot_phash, cur_phash)
                    if sim >= float(self.budget.screenshot_phash_similarity_threshold):
                        return None
            else:
                # Missing expected hashes: force refresh once.
                return self._capture_and_process(timeout=timeout_s)
        except Exception:
            # If preflight fails, fall back to authoritative refresh (bounded).
            return self._capture_and_process(timeout=timeout_s)

        # Signal changed -> authoritative refresh.
        return self._capture_and_process(timeout=timeout_s)

    # ---------------------------
    # Graph recording (delegates semantics to StateGraph)
    # ---------------------------

    def _graph_state_meta(self, snap: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if snap:
            meta = snap.get("meta") or {}
            payload.update(
                {
                    "xml_reliable": bool(snap.get("xml_reliable", meta.get("xml_reliable", False))),
                    "screenshot_phash": str(meta.get("screenshot_phash") or ""),
                    "identity_source": str(meta.get("identity_source") or ""),
                    "identity_hash": str(meta.get("identity_hash") or ""),
                    "matched_existing": bool(meta.get("matched_existing")),
                    "matched_similarity": float(meta.get("matched_similarity") or 0.0),
                    "struct_sig": str(meta.get("struct_sig") or ""),
                    "coarse_sig": str(meta.get("coarse_sig") or ""),
                    "fine_sig": str(meta.get("fine_sig") or ""),
                    "foreground_package": str(meta.get("foreground_package") or ""),
                    "foreground_activity": str(meta.get("foreground_activity") or ""),
                }
            )
        if extra:
            payload.update(extra)
        return payload

    def _graph_record_observation(self, sig: str, meta: Optional[Dict[str, Any]] = None, snap: Optional[Dict[str, Any]] = None) -> None:
        """
        IPO:
          in : sig observed without a meaningful predecessor edge
          out: graph.record_observation(sig) -> visit_count++ once

        WHEN called:
          - Entry state
          - After restart/recovery when we reconcile current state without a reliable action edge

        WHICH state:
          - sig is the authoritative current snapshot sig
        """
        graph_meta = self._graph_state_meta(snap, meta)
        was_new = not self.graph.has_state(sig)
        try:
            self.graph.record_observation(sig, meta=graph_meta)
        except Exception:
            # fallback to touch
            self.graph.touch(sig, meta=graph_meta)
        self._record_state_trace(sig)
        self._emit_transition(sig, {"kind": "observation", "sig": sig, "meta": graph_meta})
        if was_new:
            self._mark_progress("new_state_discovered", {"sig": sig})

    def _transition_outcome(self, src_snap: Dict[str, Any], dst_snap: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a transition outcome for loop/blacklist logic and graph metadata.
        Conservative: only declare "noop" when both structural and fine signatures match.
        """
        try:
            src_meta = src_snap.get("meta") or {}
            dst_meta = dst_snap.get("meta") or {}

            src_sig = str(src_snap.get("state_sig") or "")
            dst_sig = str(dst_snap.get("state_sig") or "")

            src_struct = str(src_meta.get("struct_sig") or "")
            dst_struct = str(dst_meta.get("struct_sig") or "")
            src_fine = str(src_meta.get("fine_sig") or "")
            dst_fine = str(dst_meta.get("fine_sig") or "")

            src_xml = str(src_meta.get("xml_hash") or "")
            dst_xml = str(dst_meta.get("xml_hash") or "")
            src_shot = str(src_meta.get("screenshot_hash") or "")
            dst_shot = str(dst_meta.get("screenshot_hash") or "")

            verified_no_change = False
            if src_struct and dst_struct and src_fine and dst_fine:
                verified_no_change = (src_struct == dst_struct) and (src_fine == dst_fine)
            if not verified_no_change:
                verified_no_change = (src_sig == dst_sig)

            changed = not verified_no_change
            if not changed:
                return {"changed": False, "change_type": "noop", "confidence": 1.0}

            # If structure differs, treat as navigation (high confidence).
            if src_struct and dst_struct and src_struct != dst_struct:
                return {"changed": True, "change_type": "nav", "confidence": 0.9}

            # Same structure but different fine/screenshot: likely content/webview/animation.
            if src_xml and dst_xml and src_xml == dst_xml and src_shot and dst_shot and src_shot != dst_shot:
                return {"changed": True, "change_type": "visual", "confidence": 0.6}

            return {"changed": True, "change_type": "content", "confidence": 0.6}
        except Exception:
            return {"changed": (str(src_snap.get("state_sig") or "") != str(dst_snap.get("state_sig") or "")), "change_type": "unknown", "confidence": 0.3}

    def _action_payload_key(self, action: Dict[str, Any]) -> str:
        if not action:
            return "None:None:"
        parts: List[str] = []
        for a in (action.get("actions") or [])[:6]:
            parts.append(f"{a.get('action')}:{a.get('element_id')}:{a.get('text') or ''}")
        return "||".join(parts) or "None:None:"

    def _is_action_blacklisted(self, sig: str, action_key: str) -> bool:
        if not sig or not action_key:
            return False
        now = time.time()
        fam = self._family_id(sig)
        k = (fam, action_key)
        until = self.action_blacklist_until.get(k)
        if until is None:
            return False
        if now >= float(until or 0.0):
            self.action_blacklist_until.pop(k, None)
            return False
        return True

    def _blacklist_action(self, sig: str, action_key: str, *, ttl_s: float, reason: str) -> None:
        if not sig or not action_key:
            return
        fam = self._family_id(sig)
        self._blacklist_action_family(fam, action_key, ttl_s=float(ttl_s or 0.0), reason=reason, sig_for_log=sig)

    def _blacklist_action_family(self, fam: str, action_key: str, *, ttl_s: float, reason: str, sig_for_log: Optional[str] = None) -> None:
        """
        Blacklist an action for a structural family id (used by loop detection).
        Prefer _blacklist_action(sig, ...) when you have a concrete state_sig for accurate logging.
        """
        if not fam or not action_key:
            return
        until = time.time() + float(ttl_s or 0.0)
        self.action_blacklist_until[(fam, action_key)] = until
        payload = {
            "family": fam,
            "action_key": action_key,
            "ttl_s": ttl_s,
            "until_ts": until,
            "reason": reason,
        }
        if sig_for_log:
            payload["sig"] = sig_for_log
        self._log_event("action_blacklisted", **payload)

    def _record_action_outcome_stats(self, src_sig: str, action_key: str, outcome: Dict[str, Any]) -> None:
        if not src_sig or not action_key:
            return
        fam = self._family_id(src_sig)
        k = (fam, action_key)
        self.action_attempt_counts[k] = int(self.action_attempt_counts.get(k, 0) or 0) + 1

        ctype = str((outcome or {}).get("change_type") or "")
        if ctype == "noop":
            self.action_nochange_counts[k] = int(self.action_nochange_counts.get(k, 0) or 0) + 1
            if self.action_nochange_counts[k] >= int(self.budget.blacklist_nochange_threshold):
                self._blacklist_action(src_sig, action_key, ttl_s=float(self.budget.blacklist_ttl_s), reason="repeated_noop")

    def _maybe_detect_loop(self) -> None:
        """
        Detect short oscillation loops (length 2..5) and temporarily blacklist the involved actions.
        """
        if len(self.recent_transitions) < 4:
            return

        def scaled_ttl() -> float:
            base = float(self.budget.loop_blacklist_ttl_s)
            # Escalate TTL when loops repeat (bounded).
            mult = 1.0 + min(3.0, 0.5 * float(max(0, int(self.loop_detected_count) - 1)))
            return float(base * mult)

        # Normalize to family space so loops aren't bypassed by dynamic state_sig churn.
        norm = [(sf, df, ak) for (_ss, _ds, ak, sf, df) in self.recent_transitions]

        (a1, b1, k1), (a2, b2, k2), (a3, b3, k3), (a4, b4, k4) = norm[-4:]
        if not a1 or not b1:
            return
        if a1 == b1:
            return
        if (a1, b1) == (a3, b3) and (a2, b2) == (a4, b4) and (a2 == b1) and (b2 == a1):
            self.loop_detected_count += 1
            last_src_sig = self.recent_transitions[-1][0]
            self._log_event("loop_detected", sig=last_src_sig, kind="2-cycle", a=a1, b=b1, count=self.loop_detected_count)
            ttl = scaled_ttl()
            self._blacklist_action_family(a1, k1, ttl_s=ttl, reason="2-cycle_loop")
            self._blacklist_action_family(a2, k2, ttl_s=ttl, reason="2-cycle_loop")
            return

        # N-gram loop detection: ABCABC, ABCDABCD, ...
        for n in (5, 4, 3):
            if len(norm) < 2 * n:
                continue
            seq = norm[-2 * n :]
            if seq[:n] != seq[n:]:
                continue
            # Avoid declaring a loop if every transition is a self-loop.
            if all((src == dst) for (src, dst, _k) in seq):
                continue

            self.loop_detected_count += 1
            ttl = scaled_ttl()
            last_src_sig = self.recent_transitions[-1][0]
            self._log_event("loop_detected", sig=last_src_sig, kind=f"{n}-cycle", n=n, count=self.loop_detected_count)
            seen: Set[Tuple[str, str]] = set()
            for src, _dst, akey in seq[n:]:
                k = (src, str(akey))
                if k in seen:
                    continue
                seen.add(k)
                self._blacklist_action_family(src, str(akey), ttl_s=ttl, reason=f"{n}-cycle_loop")
            return

    def _graph_record_transition(
        self,
        src: str,
        dst: str,
        action: Dict[str, Any],
        *,
        touch: bool = True,
        src_snap: Optional[Dict[str, Any]] = None,
        dst_snap: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        IPO:
          in : src_sig, dst_sig, action payload dict
          out: dst_was_new_at_discovery (bool)

        WHEN called:
          - After EVERY executed action + subsequent snapshot capture.

        WHICH states:
          - src is the state before the action (the last authoritative sig)
          - dst is the state after the action (new authoritative sig)
        """
        payload = dict(action or {})
        if src_snap is not None and dst_snap is not None:
            payload["outcome"] = self._transition_outcome(src_snap, dst_snap)
        else:
            payload["outcome"] = {"changed": (src != dst), "change_type": "unknown", "confidence": 0.3}

        # Update action repetition stats + loop detection before recording.
        akey = self._action_payload_key(payload)
        self._record_action_outcome_stats(src, akey, payload.get("outcome") or {})
        self.recent_transitions.append((src, dst, akey, self._family_id(src), self._family_id(dst)))
        if len(self.recent_transitions) > 18:
            self.recent_transitions = self.recent_transitions[-18:]
        self._maybe_detect_loop()

        src_graph_meta = self._graph_state_meta(src_snap)
        dst_graph_meta = self._graph_state_meta(dst_snap)
        dst_was_new = self.graph.record_transition(
            src,
            dst,
            payload,
            touch=bool(touch),
            src_meta=src_graph_meta,
            dst_meta=dst_graph_meta,
        )
        self._record_state_trace(dst)
        self._emit_transition(
            dst,
            {"kind": "transition", "src": src, "dst": dst, "action": payload, "dst_was_new": bool(dst_was_new)},
        )
        if src != dst:
            if dst_was_new:
                self.no_new_state_count = 0   # reset also here
                self._mark_progress("new_state_discovered", {"src": src, "dst": dst})
            else:
                self._mark_progress("state_changed", {"src": src, "dst": dst})
        return bool(dst_was_new)

    def _graph_record_aux_transition(self, src: str, dst: str, action: Dict[str, Any], *, kind: str) -> None:
        """
        Record an edge without treating it as "progress" for the main loop.
        Used for probe-return/unwind steps (record everything, but avoid masking stuck heuristics).
        """
        dst_was_new = False
        try:
            dst_was_new = bool(self.graph.record_transition(src, dst, action, touch=False))
        except Exception:
            try:
                # fallback: still keep node touch for visibility in graph
                self.graph.touch(src)
                self.graph.touch(dst)
            except Exception:
                pass
        self._record_state_trace(dst)
        self._emit_transition(dst, {"kind": kind, "src": src, "dst": dst, "action": action, "dst_was_new": bool(dst_was_new)})

    def _graph_annotate(self, sig: str, overlay_kind: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        IPO:
          in : sig and flags/meta
          out: graph.annotate(sig) -> DOES NOT increment visit_count

        WHEN called:
          - After NAV results to store overlay state as a graph hint
        """
        try:
            self.graph.annotate(sig, overlay_kind=overlay_kind, meta=meta)
        except Exception:
            pass

    def _record_state_trace(self, sig: str) -> None:
        """
        Debug-only trace. DOES NOT touch graph.
        WHEN called:
          - after recording observation or transition
        """
        self.recent_states.append(sig)
        if len(self.recent_states) > 10:
            self.recent_states = self.recent_states[-10:]

    # ---------------------------
    # DFS stack helpers
    # ---------------------------

    def _pop_stack_to(self, sig: str, mark_explored: bool) -> None:
        """
        Pop DFS stack until sig is on top (or stack empty).
        If mark_explored=True, mark popped child branches explored in their parent.
        """
        # Keep dfs_stack and dfs_via aligned.
        while self.dfs_stack and self.dfs_stack[-1] != sig:
            popped_sig = self.dfs_stack.pop()
            popped_via = self.dfs_via.pop() if self.dfs_via else None
            parent_sig = self.dfs_stack[-1] if self.dfs_stack else None
            if mark_explored and parent_sig and popped_via:
                self.explored_actions.setdefault(self._family_id(parent_sig), set()).add(str(popped_via))
        if sig not in self.dfs_stack:
            if self.dfs_stack:
                self.parent_map.setdefault(sig, self.dfs_stack[-1])
            self.dfs_stack.append(sig)
            self.dfs_via.append(None)

    def _enter_state(self, from_sig: str, to_sig: str, via_action: str) -> None:
        """
        IPO:
          in : from_sig, to_sig, via_action
          out: updates dfs_stack and parent_map; does NOT touch graph

        WHEN called:
          - after recording a transition (probe/forward/overlay/recovery/replay/drift)

        Cases:
          1) to_sig already on stack => back-like jump to ancestor
             -> pop until ancestor is top
          2) to_sig visited but not on stack => convergence from another branch
             -> append to represent current path
        """
        if to_sig not in self.parent_map:
            self.parent_map[to_sig] = from_sig

        if to_sig in self.dfs_stack:
            # back-like jump; DO NOT mark explored (external/unexpected)
            self._pop_stack_to(to_sig, mark_explored=False)
        else:
            self.dfs_stack.append(to_sig)
            self.dfs_via.append(str(via_action or ""))

    def _reconcile_stack_on_external_move(self, cur_sig: str, snap: Optional[Dict[str, Any]] = None, *, record_observation: bool = True) -> None:
        """
        IPO:
          in : cur_sig after recovery/restart/overlay resolution
          out: stack updated to include cur_sig as current; optionally records an observation

        WHEN called:
          - After any procedure that can relocate us without a clean recorded edge:
            recovery end, restart end, overlay resolution end.

        WHY:
          - Keeps DFS completion/backtrace meaningful even after discontinuities.
        """
        if record_observation:
            self._graph_record_observation(cur_sig, snap=snap)
            self._mark_progress("state_changed", {"sig": cur_sig})
        else:
            # If a caller believes the move had a recorded edge but a recapture produced a new sig,
            # ensure the graph still "sees" this state (avoids current-state-without-touch skew).
            last = self.recent_states[-1] if self.recent_states else ""
            if last and last != cur_sig:
                self._log_event("implicit_observation", sig=cur_sig, prev_sig=last)
                self._graph_record_observation(cur_sig, meta={"implicit": True, "prev_sig": last}, snap=snap)
        if cur_sig in self.dfs_stack:
            # external move: pop without marking explored
            self._pop_stack_to(cur_sig, mark_explored=False)
        else:
            if self.dfs_stack:
                self.parent_map.setdefault(cur_sig, self.dfs_stack[-1])
            self.dfs_stack.append(cur_sig)
            self.dfs_via.append(None)

    # ---------------------------
    # Scheduling (LLM pipelining)
    # ---------------------------

    def _remember_llm_snap(self, sig: str, snap: Dict[str, Any]) -> None:
        """
        Keep a small LRU of (screenshot,uist) keyed by state_sig so that async
        LLM2-1/2 futures can schedule follow-on work without re-capturing.
        """
        try:
            self.llm_snap_cache[sig] = {"screenshot": snap.get("screenshot", ""), "uist": snap.get("uist", {})}
            self.llm_snap_cache.move_to_end(sig)
            if len(self.llm_snap_cache) > int(self.llm_snap_cache_max):
                self.llm_snap_cache.popitem(last=False)
        except Exception:
            pass

    def _page_signals(self, sig: str, snap: Dict[str, Any], nav: Optional[NavigationProposal] = None) -> Dict[str, Any]:
        """
        Small, stable page signals for topic routing/filling.
        Avoid including full UI trees or full questionnaire context.
        """
        lines = BaseUI.get_text_list(snap.get("uist") or {})
        uniq: List[str] = []
        seen = set()
        for t in lines:
            s = str(t).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s[:90])
            if len(uniq) >= 14:
                break

        blob = " ".join(uniq).lower()
        tags: Set[str] = set()
        if any(k in blob for k in ("privacy", "policy", "terms", "legal")):
            tags.add("privacy")
            tags.add("legal")
        if any(k in blob for k in ("permission", "allow", "deny")):
            tags.add("permissions")
        if any(k in blob for k in ("location", "gps")):
            tags.add("location")
        if any(k in blob for k in ("camera", "photo")):
            tags.add("camera")
        if any(k in blob for k in ("microphone", "mic")):
            tags.add("microphone")
        if any(k in blob for k in ("notification", "notifications")):
            tags.add("notifications")
        if any(k in blob for k in ("subscription", "subscribe", "billing", "purchase", "$", "trial", "restore")):
            tags.add("billing")
        if any(k in blob for k in ("login", "log in", "sign in", "sign up", "register")):
            tags.add("auth")
        if any(k in blob for k in ("account", "profile", "password", "security")):
            tags.add("account")
            tags.add("security")
        if any(k in blob for k in ("help", "support", "faq", "about")):
            tags.add("help")

        ui_type = "unknown"
        if "permission" in blob and any(k in blob for k in ("allow", "deny")):
            ui_type = "permissions_dialog"
        elif "settings" in blob:
            ui_type = "settings_list"
        elif "subscription" in blob or "billing" in blob:
            ui_type = "subscription_paywall"
        elif any(k in blob for k in ("login", "sign in", "sign up")):
            ui_type = "auth_flow"

        page_summary = ""
        nav_tags: List[Dict[str, Any]] = []
        if nav is not None:
            page_summary = (getattr(nav, "page_summary", "") or "")[:120]
            nav_ui_type = str(getattr(nav, "ui_type", "") or "").strip()
            if nav_ui_type:
                ui_type = nav_ui_type
            for tg in (getattr(nav, "page_tags", None) or [])[:12]:
                try:
                    nav_tags.append({"tag": str(getattr(tg, "tag", "") or ""), "weight": float(getattr(tg, "weight", 0.0) or 0.0)})
                except Exception:
                    continue

        return {
            "state_sig": sig,
            "ui_type": ui_type,
            "page_summary": page_summary,
            "page_tags": nav_tags,
            "local_tags": sorted(tags),
            "ocr_top_lines": uniq[:12],
        }

    def _schedule_state(self, sig: str, snap: Dict[str, Any], task: str) -> None:
        """
        IPO:
          in : sig + authoritative snapshot
          out: schedules LLM1 (NAV) and LLM2 (Q) futures if not already cached/in-flight

        WHEN called:
          - on entry
          - whenever cur_sig changes
          - after probe destinations are discovered (pipeline analysis)
        """
        if not self._pool:
            return

        now = time.time()
        self._remember_llm_snap(sig, snap)

        # NAV scheduling (LLM1)
        if now >= self.nav_cooldown_until.get(sig, 0.0):
            if sig not in self.nav_cache and sig not in self._nav_futures:
                block_status = copy.deepcopy(getattr(self.questionnaires, "block_status", {}) or {})
                logger.debug("Schedule NAV for sig=%s blocks=%d", sig[:8], len(block_status))
                self._nav_enqueue_ts[sig] = time.time()
                self._log_event("nav_scheduled", sig=sig, blocks=len(block_status), enqueue_ts=self._nav_enqueue_ts[sig])
                self._emit_llm_enqueued(
                    "nav",
                    sig,
                    {
                        "state_sig": sig,
                        "block_status": block_status,
                        "app_intro": self.app_intro,
                        "focus_hints": self.focus_hints,
                        "task": task,
                        "history": list(self.history),
                        "enqueue_ts": self._nav_enqueue_ts[sig],
                    },
                )
                self._nav_futures[sig] = self._pool.submit(
                    self.gpt.propose_navigation,
                    screenshot_b64=snap["screenshot"],
                    ui_json=snap["uist"],
                    block_status=block_status,
                    task=task,
                    app_intro=self.app_intro,
                    focus_hints=self.focus_hints,
                    history=list(self.history),
                    state_sig=sig,
                )
        else:
            logger.debug("NAV cooldown active for sig=%s", sig[:8])
            self._log_event("nav_cooldown", sig=sig, cooldown_until=self.nav_cooldown_until.get(sig))

        # Old topic_route/topic_fill are disabled in the block_status workflow.

        # New UI-level router/block path, observation mode only.
        # Input:
        # - current screenshot
        # - flat router list from QuestionnaireState2
        # Processing:
        # - LLM2-1 answers router questions
        # - local code maps router answers to matched blocks
        # Output:
        # - an observation JSON saved by `_drain_futures`.
        q2 = self.questionnaires
        if (
            sig not in self.block_router_cache
            and sig not in self.block_match_cache
            and sig not in self._block_router_futures
        ):
            router_questions = list(getattr(q2, "routers", []) or [])
            if router_questions:
                self._block_router_enqueue_ts[sig] = time.time()
                self._emit_llm_enqueued(
                    "block_router",
                    sig,
                    {
                        "state_sig": sig,
                        "app_intro": self.app_intro,
                        "focus_hints": self.focus_hints,
                        "router_question_count": len(router_questions),
                        "enqueue_ts": self._block_router_enqueue_ts[sig],
                    },
                )
                self._block_router_futures[sig] = self._pool.submit(
                    self.gpt.propose_router_answers,
                    screenshot_b64=snap.get("screenshot", ""),
                    router_questions=router_questions,
                    app_intro=self.app_intro,
                    focus_hints=self.focus_hints,
                    state_sig=sig,
                )
            else:
                try:
                    matched_blocks = q2.match_blocks_from_router_answers([])
                    q2.mark_blocks_hit(matched_blocks)
                    self.block_match_cache[sig] = matched_blocks
                    obs_path = self._save_questionnaire2_observation(sig, [], matched_blocks, stage="router")
                    self._log_event(
                        "block_router_no_routers",
                        sig=sig,
                        matched_block_count=len(matched_blocks),
                        observation_path=str(obs_path or ""),
                    )
                except Exception:
                    logger.debug("QuestionnaireState2 no-router matching failed sig=%s", sig[:8], exc_info=True)


    def _schedule_topic_fills(self, sig: str, snap: Dict[str, Any]) -> None:
        # Disabled: old topic_fill is not used by the block_status workflow.
        return
        if not self._pool:
            return
        route = self.topic_route_cache.get(sig)
        if not route:
            return

        now = time.time()
        conf_th = float(self.budget.topic_route_conf_threshold)
        cooldown_s = float(self.budget.topic_fill_cooldown_s)

        signals = self._page_signals(sig, snap, nav=self.nav_cache.get(sig))

        for t in (getattr(route, "relevant_topics", None) or [])[:8]:
            topic_id = str(getattr(t, "topic_id", "") or "").strip()
            conf = float(getattr(t, "confidence", 0.0) or 0.0)
            if not topic_id or conf < conf_th:
                continue

            if not self.questionnaires.open_gaps_in_topic(topic_id):
                continue

            digest = self.questionnaires.answers_digest_for_topic(topic_id)
            fut_key = (sig, topic_id, digest)
            if fut_key in self.topic_fill_cache or fut_key in self._topic_fill_futures:
                continue

            last_try = self.topic_fill_attempt_ts.get((sig, topic_id), 0.0)
            if last_try and (now - last_try) < cooldown_s:
                continue

            pack = self.questionnaires.build_topic_question_pack(topic_id, limit=int(self.budget.topic_pack_limit))
            if not pack:
                continue

            memory = self.questionnaires.topic_memory_summary(topic_id, max_items=18)
            current_answers = self.questionnaires.answers_for_ids([p["id"] for p in pack])

            self.topic_fill_attempt_ts[(sig, topic_id)] = now
            self._topic_fill_enqueue_ts[fut_key] = now
            self._emit_llm_enqueued(
                "topic_fill",
                sig,
                {
                    "state_sig": sig,
                    "topic_id": topic_id,
                    "pack_size": len(pack),
                    "answers_digest": digest[:10],
                    "enqueue_ts": now,
                },
            )
            self._topic_fill_futures[fut_key] = self._pool.submit(
                self.gpt.propose_topic_fill,
                snap.get("screenshot", ""),
                snap.get("uist", {}),
                topic_id,
                pack,
                memory,
                current_answers,
                signals,
                sig,
            )

    def _schedule_blocks_fill(self, sig: str, snap: Dict[str, Any], matched_blocks: List[Dict[str, Any]]) -> None:
        """
        Schedule one LLM2-2 call for all blocks matched on the current UI.

        Input:
        - sig: current state signature.
        - snap: authoritative snapshot containing screenshot.
        - matched_blocks: full block payloads from QuestionnaireState2.

        Processing:
        - Skip if there are no matched blocks or a fill is already cached/in-flight.
        - Mark matched block ids as visited because they are being sent to LLM2-2.
        - Submit one `propose_blocks_fill` future with all matched blocks.

        Output:
        - None. Future state is stored in `_blocks_fill_futures`.
        """
        if not self._pool or not matched_blocks:
            return
        if sig in self.blocks_fill_cache or sig in self._blocks_fill_futures:
            return

        block_ids = [str(block.get("id") or "") for block in matched_blocks if block.get("id")]
        try:
            self.questionnaires.mark_blocks_visited(block_ids)
        except Exception:
            logger.debug("mark_blocks_visited failed sig=%s", sig[:8], exc_info=True)

        now = time.time()
        self._blocks_fill_enqueue_ts[sig] = now
        self._emit_llm_enqueued(
            "blocks_fill",
            sig,
            {
                "state_sig": sig,
                "app_intro": self.app_intro,
                "focus_hints": self.focus_hints,
                "block_count": len(matched_blocks),
                "block_ids": block_ids,
                "enqueue_ts": now,
            },
        )
        self._blocks_fill_futures[sig] = self._pool.submit(
            self.gpt.propose_blocks_fill,
            screenshot_b64=snap.get("screenshot", ""),
            blocks_payload=matched_blocks,
            app_intro=self.app_intro,
            focus_hints=self.focus_hints,
            state_sig=sig,
        )

    def _drain_futures(self) -> None:
        """
        WHEN called:
          - top of each main-loop iteration
          - during NAV waiting and post-probe wait

        WHAT it does:
          - moves completed futures into caches
          - applies questionnaire updates immediately (sticky state)
        """
        for sig in list(self._nav_futures.keys()):
            fut = self._nav_futures[sig]
            if fut.done():
                try:
                    nav = fut.result()
                    self.nav_cache[sig] = nav
                    self.nav_ready_once = True
                    try:
                        cand_ct = len(getattr(nav, "candidate_actions", []) or [])
                        fam = self._family_id(sig)
                        if cand_ct and fam not in self.nav_candidates_noted:
                            self.nav_candidates_noted.add(fam)
                            self._mark_progress("candidates_discovered", {"sig": sig, "family": fam, "count": cand_ct})
                    except Exception:
                        pass
                    logger.debug(
                        "NAV ready sig=%s overlay=%s cand=%d return_method=%s return_actions=%d",
                        sig[:8],
                        self._overlay_kind_value(nav),
                        len(getattr(nav, "candidate_actions", []) or []),
                        getattr(nav, "return_method", "back"),
                        len(getattr(nav, "return_actions", []) or []),
                    )
                    start = self._nav_enqueue_ts.pop(sig, None)
                    duration_s = (time.time() - start) if start else None
                    self._log_event(
                        "nav_ready",
                        sig=sig,
                        overlay_kind=self._overlay_kind_value(nav),
                        candidate_count=len(getattr(nav, "candidate_actions", []) or []),
                        duration_s=duration_s,
                    )
                    nav_obs_path = self._save_nav_observation(sig, nav)
                    if nav_obs_path is not None:
                        self._log_event("nav_observation_saved", sig=sig, observation_path=str(nav_obs_path))
                    self._emit_llm_result(
                        "nav",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": duration_s,
                            "observation_path": str(nav_obs_path or ""),
                            "result": nav.model_dump(mode="json") if hasattr(nav, "model_dump") else getattr(nav, "__dict__", {}),
                        },
                    )
                except CancelledError:
                    # Most commonly cancelled by NAV timeout; do not double-count failures/backoff here.
                    start = self._nav_enqueue_ts.pop(sig, None)
                    self._log_event("nav_cancelled", sig=sig, duration_s=(time.time() - start) if start else None)
                    self._emit_llm_result(
                        "nav",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "error": "nav_cancelled",
                        },
                    )
                except Exception:
                    logger.debug("NAV future failed sig=%s", sig[:8], exc_info=True)
                    self.nav_failures[sig] = self.nav_failures.get(sig, 0) + 1
                    failures = int(self.nav_failures.get(sig, 0) or 0)
                    backoff_s = min(30.0, 2.0 * (2.0 ** float(min(max(0, failures - 1), 4))))
                    self.nav_cooldown_until[sig] = max(float(self.nav_cooldown_until.get(sig, 0.0) or 0.0), time.time() + backoff_s)
                    self._log_event("nav_failure", sig=sig, failures=failures, backoff_s=backoff_s, cooldown_until=self.nav_cooldown_until.get(sig))
                    start = self._nav_enqueue_ts.pop(sig, None)
                    self._emit_llm_result(
                        "nav",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "error": "nav_future_failed",
                        },
                    )
                finally:
                    self._nav_futures.pop(sig, None)

        for sig in list(self._block_router_futures.keys()):
            fut = self._block_router_futures[sig]
            if fut.done():
                try:
                    route: RouterResult = fut.result()
                    if getattr(route, "state_sig", sig) and route.state_sig != sig:
                        self._log_event("block_router_reject", sig=sig, reason="stale_state_sig", got=route.state_sig)
                    else:
                        q2 = self.questionnaires
                        router_answers = [
                            item.model_dump(mode="json") if hasattr(item, "model_dump") else getattr(item, "__dict__", {})
                            for item in (getattr(route, "router_updates", None) or [])
                        ]
                        matched_blocks: List[Dict[str, Any]] = []
                        matched_blocks = q2.match_blocks_from_router_answers(router_answers)
                        q2.mark_blocks_hit(matched_blocks)
                        self.block_router_cache[sig] = route
                        self.block_match_cache[sig] = matched_blocks

                        obs_path = self._save_questionnaire2_observation(sig, router_answers, matched_blocks, stage="router")
                        self._log_event(
                            "block_router_ready",
                            sig=sig,
                            router_update_count=len(router_answers),
                            matched_block_count=len(matched_blocks),
                            observation_path=str(obs_path or ""),
                        )
                        snap = self.llm_snap_cache.get(sig)
                        if snap and matched_blocks:
                            self._schedule_blocks_fill(sig, snap, matched_blocks)

                    start = self._block_router_enqueue_ts.pop(sig, None)
                    result_payload = route.model_dump(mode="json") if hasattr(route, "model_dump") else getattr(route, "__dict__", {})
                    self._emit_llm_result(
                        "block_router",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "matched_block_ids": [block.get("id") for block in self.block_match_cache.get(sig, [])],
                            "result": result_payload,
                        },
                    )
                except Exception:
                    logger.debug("BlockRouter future failed sig=%s", sig[:8], exc_info=True)
                    start = self._block_router_enqueue_ts.pop(sig, None)
                    self._emit_llm_result(
                        "block_router",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "error": "block_router_future_failed",
                        },
                    )
                finally:
                    self._block_router_futures.pop(sig, None)

        for sig in list(self._blocks_fill_futures.keys()):
            fut = self._blocks_fill_futures[sig]
            if fut.done():
                try:
                    result: BlocksFillResult = fut.result()
                    self.blocks_fill_cache[sig] = result

                    router_result = self.block_router_cache.get(sig)
                    router_answers = [
                        item.model_dump(mode="json") if hasattr(item, "model_dump") else getattr(item, "__dict__", {})
                        for item in (getattr(router_result, "router_updates", None) or [])
                    ] if router_result is not None else []
                    matched_blocks = self.block_match_cache.get(sig, [])
                    block_fill_results = [
                        item.model_dump(mode="json") if hasattr(item, "model_dump") else getattr(item, "__dict__", {})
                        for item in (getattr(result, "block_results", None) or [])
                    ]
                    obs_path = self._save_questionnaire2_observation(
                        sig,
                        router_answers,
                        matched_blocks,
                        block_fill_results=block_fill_results,
                        stage="blocks_fill",
                    )

                    proposed_ct = sum(len(row.get("proposed_updates") or []) for row in block_fill_results)
                    if proposed_ct:
                        self.state_update_counts[sig] = int(self.state_update_counts.get(sig, 0) or 0) + int(proposed_ct)
                        self._mark_progress("blocks_fill_observed", {"sig": sig, "proposed_count": proposed_ct})

                    start = self._blocks_fill_enqueue_ts.pop(sig, None)
                    self._log_event(
                        "blocks_fill_ready",
                        sig=sig,
                        block_result_count=len(block_fill_results),
                        proposed_count=proposed_ct,
                        observation_path=str(obs_path or ""),
                    )
                    self._emit_llm_result(
                        "blocks_fill",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "observation_path": str(obs_path or ""),
                            "result": result.model_dump(mode="json") if hasattr(result, "model_dump") else getattr(result, "__dict__", {}),
                        },
                    )
                except Exception:
                    logger.debug("BlocksFill future failed sig=%s", sig[:8], exc_info=True)
                    start = self._blocks_fill_enqueue_ts.pop(sig, None)
                    self._emit_llm_result(
                        "blocks_fill",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "error": "blocks_fill_future_failed",
                        },
                    )
                finally:
                    self._blocks_fill_futures.pop(sig, None)

        for sig in list(self._topic_route_futures.keys()):
            fut = self._topic_route_futures[sig]
            if fut.done():
                try:
                    route: TopicRouteResult = fut.result()
                    if getattr(route, "state_sig", sig) and route.state_sig != sig:
                        self._log_event("topic_route_reject", sig=sig, reason="stale_state_sig", got=route.state_sig)
                    else:
                        self.topic_route_cache[sig] = route
                        self._log_event("topic_route_ready", sig=sig, topic_count=len(getattr(route, "relevant_topics", []) or []))

                        snap = self.llm_snap_cache.get(sig)
                        if snap:
                            self._schedule_topic_fills(sig, snap)

                    start = self._topic_route_enqueue_ts.pop(sig, None)
                    self._emit_llm_result(
                        "topic_route",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "result": route.model_dump(mode="json") if hasattr(route, "model_dump") else getattr(route, "__dict__", {}),
                        },
                    )
                except Exception:
                    logger.debug("TopicRoute future failed sig=%s", sig[:8], exc_info=True)
                    start = self._topic_route_enqueue_ts.pop(sig, None)
                    self._emit_llm_result(
                        "topic_route",
                        sig,
                        {
                            "state_sig": sig,
                            "duration_s": (time.time() - start) if start else None,
                            "error": "topic_route_future_failed",
                        },
                    )
                finally:
                    self._topic_route_futures.pop(sig, None)

        for key in list(self._topic_fill_futures.keys()):
            fut = self._topic_fill_futures[key]
            if fut.done():
                sig, topic_id, digest = key
                try:
                    upd: QuestionnaireUpdate = fut.result()
                    self.topic_fill_cache[key] = upd
                    self.q_cache[sig] = upd

                    nav = self.nav_cache.get(sig)
                    page_summary = getattr(nav, "page_summary", "") if nav else ""
                    self.questionnaires.apply_updates(
                        upd,
                        context={"state_sig": sig, "page_summary": page_summary, "topic_id": topic_id, "answers_digest": digest},
                    )

                    proposed_ct = len(getattr(upd, "proposed_updates", []) or [])
                    if proposed_ct:
                        self.state_update_counts[sig] = int(self.state_update_counts.get(sig, 0) or 0) + int(proposed_ct)
                    if proposed_ct:
                        self._mark_progress("q_update_applied", {"sig": sig, "topic_id": topic_id, "proposed_count": proposed_ct})
                    self._log_event("topic_fill_update", sig=sig, topic_id=topic_id, proposed_count=proposed_ct)

                    start = self._topic_fill_enqueue_ts.pop(key, None)
                    payload = upd.model_dump(mode="json") if hasattr(upd, "model_dump") else getattr(upd, "__dict__", {})
                    self._emit_llm_result(
                        "topic_fill",
                        sig,
                        {
                            "state_sig": sig,
                            "topic_id": topic_id,
                            "answers_digest": digest[:10],
                            "duration_s": (time.time() - start) if start else None,
                            "result": payload,
                        },
                    )
                    self._emit_questionnaire_update(sig, {"state_sig": sig, "topic_id": topic_id, **payload})
                except Exception:
                    logger.debug("TopicFill future failed sig=%s topic=%s", sig[:8], topic_id, exc_info=True)
                    start = self._topic_fill_enqueue_ts.pop(key, None)
                    self._emit_llm_result(
                        "topic_fill",
                        sig,
                        {
                            "state_sig": sig,
                            "topic_id": topic_id,
                            "answers_digest": digest[:10],
                            "duration_s": (time.time() - start) if start else None,
                            "error": "topic_fill_future_failed",
                        },
                    )
                finally:
                    self._topic_fill_futures.pop(key, None)

    # ---------------------------
    # NAV barrier with timeout fallback
    # ---------------------------

    def _wait_nav_or_fallback(
        self, sig: str, snap: Dict[str, Any], task: str, *, timeout_s: Optional[float] = None
    ) -> Tuple[Optional[NavigationProposal], bool]:
        """
        IPO:
          in : current sig + snapshot + task
          out: (nav, using_heuristics)

        WHEN called:
          - once per main-loop iteration, AFTER drift check, BEFORE overlay/probe/forward

        GUARANTEES:
          - We do NOT heuristic-click unless NAV timed out or is under cooldown.
          - While waiting, we periodically recapture to detect drift/popups.

        WHICH state:
          - sig is the authoritative current snapshot state_sig.
        """
        nav = self.nav_cache.get(sig)
        if nav:
            self.nav_ready_once = True
            try:
                cand_ct = len(getattr(nav, "candidate_actions", []) or [])
                fam = self._family_id(sig)
                if cand_ct and fam not in self.nav_candidates_noted:
                    self.nav_candidates_noted.add(fam)
                    self._mark_progress("candidates_discovered", {"sig": sig, "family": fam, "count": cand_ct})
            except Exception:
                pass
            self._emit_decision(sig, "nav_ready_cached", {"using_cache": True})
            self._log_event("nav_cache_hit", sig=sig)
            return nav, False

        # Ensure NAV is scheduled
        self._schedule_state(sig, snap, task)

        now = time.time()
        if now < self.nav_cooldown_until.get(sig, 0.0):
            return None, True

        t0 = time.time()
        timeout_s = float(timeout_s if timeout_s is not None else self.budget.nav_timeout_s)
        last_drift_check = t0
        rescheduled_once = False

        while time.time() - t0 < timeout_s:
            self._drain_futures()
            nav = self.nav_cache.get(sig)
            if nav:
                self.nav_ready_once = True
                return nav, False
            # NAV may fail fast and set a cooldown in _drain_futures; don't keep waiting.
            if time.time() < self.nav_cooldown_until.get(sig, 0.0):
                return None, True

            # If NAV future failed early (removed) and nothing is in-flight, reschedule once.
            if (sig not in self._nav_futures) and (sig not in self.nav_cache) and (not rescheduled_once):
                logger.debug("NAV missing in-flight for sig=%s -> reschedule once.", sig[:8])
                self._schedule_state(sig, snap, task)
                rescheduled_once = True

            # Drift check while waiting
            if time.time() - last_drift_check >= self.budget.nav_drift_check_interval_s:
                last_drift_check = time.time()
                snap_live = self._preflight_refresh_if_changed(snap, timeout_s=0.8)
                if snap_live:
                    # Always refresh local snapshot to keep vid_map/hashes aligned with the real UI,
                    # even when state_sig stays stable.
                    if snap_live["state_sig"] != sig:
                        new_sig = snap_live["state_sig"]
                        logger.warning("DRIFT during NAV wait: old=%s new=%s", sig[:8], new_sig[:8])
                        drift_step = ActionStep(action=ActionType.WAIT, element_id=None, text="0.2", priority=1, reasoning="drift_wait_nav")
                        self._graph_record_transition(sig, new_sig, self._actions_signature([drift_step]), src_snap=snap, dst_snap=snap_live)
                        self._set_forced_replan(new_sig, snap_live, has_edge=True)
                        self._emit_decision(sig, "drift_during_nav_wait", {"from_sig": sig, "to_sig": new_sig})
                        return None, False
                    try:
                        snap.update(snap_live)
                    except Exception:
                        pass

            time.sleep(self.budget.nav_poll_interval_s)

        # Timeout: cancel best-effort and allow heuristics
        fut = self._nav_futures.get(sig)
        if fut is not None:
            try:
                cancelled = fut.cancel()
                logger.warning("NAV timeout sig=%s cancelled=%s -> heuristics allowed.", sig[:8], cancelled)
            except Exception:
                logger.warning("NAV timeout sig=%s -> heuristics allowed.", sig[:8])
        else:
            logger.warning("NAV timeout sig=%s (no in-flight future) -> heuristics allowed.", sig[:8])

        self.nav_cooldown_until[sig] = time.time() + self.budget.nav_cooldown_s
        self.nav_failures[sig] = self.nav_failures.get(sig, 0) + 1
        self._nav_enqueue_ts.pop(sig, None)
        self._emit_decision(sig, "nav_timeout", {"timeout_s": timeout_s})
        self._log_event("nav_timeout", sig=sig, timeout_s=timeout_s)
        return None, True

    # ---------------------------
    # Overlay handling
    # ---------------------------

    def _dismiss_overlay_with_nav(self, sig: str, snap: Dict[str, Any], nav: NavigationProposal, task: str) -> bool:
        """
        IPO:
          in : sig+snap (overlay state), nav (LLM1 overlay plan)
          out: True if overlay cleared

        WHEN called:
          - main loop when nav.overlay_kind==dismiss
          - backtrace when encountering overlay

        WHICH state:
          - sig must match the snapshot the overlay is on
        """
        actions = getattr(nav, "overlay_dismiss_actions", []) or []
        if not actions:
            logger.debug("LLM1 says overlay at sig=%s but gave no overlay_dismiss_actions.", sig[:8])
            self._log_event("overlay_dismiss_missing_actions", sig=sig)
            return False

        for step in actions[:5]:
            prev_sig = sig
            prev_snap = snap
            logger.info("Overlay action at sig=%s: %s (%s)", sig[:8], self._action_key(step), (step.reasoning or "")[:70])
            self._log_event("overlay_dismiss_attempt", sig=sig, action=self._action_signature(step))
            if not self._execute_action(step, snap["vid_map"], sig):
                break
            self.action_count += 1
            self.history.append(self._action_key(step))
            time.sleep(self.budget.post_action_settle_s)

            ns = self._capture_and_process()
            if not ns:
                continue

            new_sig = ns["state_sig"]
            self._graph_record_transition(
                prev_sig,
                new_sig,
                self._actions_signature([step], vid_map=prev_snap.get("vid_map") or {}),
                src_snap=prev_snap,
                dst_snap=ns,
            )
            self._enter_state(from_sig=sig, to_sig=new_sig, via_action=self._action_key(step))

            sig, snap = new_sig, ns

            self._schedule_state(sig, snap, task)
            self._drain_futures()

            new_nav = self.nav_cache.get(sig)
            if new_nav and self._overlay_kind_value(new_nav) != OverlayKind.DISMISS.value:
                logger.info("Overlay resolved; now at sig=%s", sig[:8])
                self._emit_decision(sig, "overlay_resolved", {"via_action": self._action_signature(step)})
                self._mark_progress("overlay_resolved", {"sig": sig})
                return True

        self._emit_decision(sig, "overlay_unresolved", {"actions_tried": len(actions)})
        return False

    def _handle_loading_overlay(self, sig: str, snap: Dict[str, Any], task: str) -> None:
        """
        Bounded wait for transient loading overlays; recapture and replan.
        """
        t0 = time.time()
        last_sig = sig
        while time.time() - t0 < self.budget.loading_wait_s:
            time.sleep(0.5)
            ns = self._capture_and_process()
            if not ns:
                continue
            new_sig = ns["state_sig"]
            if new_sig != last_sig:
                wait_step = ActionStep(action=ActionType.WAIT, element_id=None, text="0.5", priority=1, reasoning="loading_wait")
                self._graph_record_transition(last_sig, new_sig, self._actions_signature([wait_step]), src_snap=snap, dst_snap=ns)
                self._enter_state(from_sig=last_sig, to_sig=new_sig, via_action="wait:None:loading")
                last_sig = new_sig
                sig = new_sig
                snap = ns
                self._schedule_state(sig, snap, task)
                self._drain_futures()

            nav = self.nav_cache.get(sig)
            if nav and self._overlay_kind_value(nav) != OverlayKind.LOADING.value:
                return

    # ---------------------------
    # Candidate selection
    # ---------------------------

    def _filter_nav_candidates(self, sig: str, candidates: List[ActionCandidate]) -> List[ActionCandidate]:
        threshold = float(self.budget.min_candidate_score)
        if threshold <= -1.0:
            return list(candidates or [])

        kept: List[ActionCandidate] = []
        dropped: List[Dict[str, Any]] = []
        for cand in list(candidates or []):
            score = float(getattr(cand, "score", 0.0) or 0.0)
            if score < threshold:
                dropped.append(
                    {
                        "action_key": self._candidate_key(cand) if getattr(cand, "actions", None) else "",
                        "score": score,
                    }
                )
                continue
            kept.append(cand)

        if dropped:
            self._log_event(
                "candidate_score_filtered",
                sig=sig,
                threshold=threshold,
                kept=len(kept),
                dropped=len(dropped),
                dropped_preview=dropped[:5],
            )
            self._emit_decision(
                sig,
                "candidate_score_filtered",
                {
                    "threshold": threshold,
                    "kept": len(kept),
                    "dropped": len(dropped),
                    "dropped_preview": dropped[:5],
                },
            )
        return kept

    def _candidate_actions(self, nav: Optional[NavigationProposal], snap: Dict[str, Any], allow_heuristics: bool) -> List[ActionCandidate]:
        """
        WHEN called:
          - main loop after overlay gate to decide which actions to probe

        POLICY:
          - Prefer NAV candidates
          - Heuristics ONLY if allow_heuristics==True (NAV timed out)
        """
        if nav and getattr(nav, "candidate_actions", None):
            sig = str(snap.get("state_sig") or "")
            return self._filter_nav_candidates(sig, list(getattr(nav, "candidate_actions") or []))

        if not allow_heuristics:
            return []

        # Heuristic fallback: do not fully trust attributes (Flutter may mislabel).
        sig = str(snap.get("state_sig") or "")
        vid_map = snap["vid_map"]
        scored: List[Tuple[float, int]] = []

        def has_meaning(n: Dict[str, Any]) -> bool:
            return bool(
                (n.get("text") or "").strip()
                or (n.get("content_desc") or "").strip()
                or (n.get("ocr_text") or "").strip()
                or (n.get("icon_label") or "").strip()
            )

        for eid, node in vid_map.items():
            try:
                f = BaseUI.get_frame(node)
                w = float(f.get("width", 0))
                h = float(f.get("height", 0))
                area = w * h
                if area < 18 * 18:
                    continue
                if area > 1400 * 1400:
                    continue

                score = 0.0
                if node.get("clickable"):
                    score += 2.0
                if has_meaning(node):
                    score += 1.7
                cls = (node.get("class") or "").lower()
                if "button" in cls or "tab" in cls or "menu" in cls:
                    score += 1.0
                if node.get("enabled") is False:
                    score -= 0.3

                scored.append((score, eid))
            except Exception:
                continue

        scored.sort(reverse=True)
        out: List[ActionCandidate] = []
        for _, eid in scored:
            if len(out) >= 8:
                break
            step = ActionStep(action=ActionType.CLICK, element_id=eid, priority=len(out) + 1, reasoning="heuristic_timeout_nav")
            if sig and self._is_action_blacklisted(sig, self._actions_key([step])):
                continue
            out.append(ActionCandidate(actions=[step], return_method=None, return_actions=[]))
        if not out and scored:
            # Break-glass: if everything is blacklisted, take the top few anyway.
            self._log_event("blacklist_break_glass", sig=sig, kind="heuristic_candidates")
            for _, eid in scored[:4]:
                step = ActionStep(action=ActionType.CLICK, element_id=eid, priority=len(out) + 1, reasoning="heuristic_timeout_nav_break_glass")
                out.append(ActionCandidate(actions=[step], return_method=None, return_actions=[]))
        return out

    # ---------------------------
    # Probing with back-like detection + custom return support
    # ---------------------------

    def _probe_candidates(
        self,
        src_sig: str,
        snap: Dict[str, Any],
        nav: Optional[NavigationProposal],
        candidates: List[ActionCandidate],
        task: str,
    ) -> None:
        """
        IPO:
          in : src_sig (current state), snap (authoritative snapshot for src_sig), nav (LLM1), candidates (actions to probe)
          out: updates probe_outcomes/probe_novelty; schedules dst analysis; may set forced replan

        WHEN called:
          - once per main-loop iteration after overlay gate

        WHICH state:
          - src_sig must match snap["state_sig"] on entry
          - probe actions use snap["vid_map"] (element ids are only valid for that snapshot)
        """
        if not candidates:
            return

        global_return_method = getattr(nav, "return_method", "back") if nav else "back"
        global_return_actions = list(getattr(nav, "return_actions", []) or []) if nav else []
        expected_struct_sig = str((snap.get("meta") or {}).get("struct_sig") or "")
        if not expected_struct_sig:
            meta = snap.get("meta") or {}
            expected_struct_sig = compute_structural_signature(
                snap.get("uist") or {},
                foreground_package=str(meta.get("foreground_package") or ""),
                foreground_activity=str(meta.get("foreground_activity") or ""),
            )

        probed = 0
        for cand in candidates:
            if probed >= self.budget.per_page_probe_cap:
                break
            if not cand.actions:
                continue
            ckey = self._candidate_key(cand)
            if self._is_explored(src_sig, cand):
                continue
            if self._already_attempted(src_sig, cand):
                continue
            if self._is_action_blacklisted(src_sig, ckey):
                continue

            primary = cand.actions[0]
            primary_key = self._action_key(primary)
            logger.debug("Probe: src_sig=%s action=%s", src_sig[:8], ckey)
            self._log_event("probe_attempt", sig=src_sig, action_key=ckey, primary=primary_key)

            ok = self._execute_action_sequence(cand.actions, snap)
            if not ok:
                logger.debug("Probe action failed immediately at src_sig=%s", src_sig[:8])
                # Foreground mismatch is a STATE fault: stop probing and replan on a fresh snapshot.
                try:
                    fail_reason = (self.last_action_failure or {}).get("reason")
                    if fail_reason in ("foreground_mismatch", "post_back_foreground_mismatch"):
                        snap2 = self._capture_and_process(timeout=6.0)
                        if snap2:
                            self._set_forced_replan(str(snap2.get("state_sig") or ""), snap2, has_edge=False)
                        return
                except Exception:
                    return
                continue

            probed += 1

            dst = self._capture_and_process()
            if not dst:
                logger.warning("Probe capture failed after action=%s -> recovery.", ckey)
                self._recover(src_sig, snap, reason=RecoveryReason.PROBE_CAPTURE_FAILED, task=task, target_sig=None)
                snap2 = self._capture_and_process()
                if snap2:
                    self._set_forced_replan(snap2["state_sig"], snap2, has_edge=True)
                return

            dst_sig = dst["state_sig"]
            if dst_sig != src_sig:
                self._mark_progress("probe_dst_found", {"src": src_sig, "dst": dst_sig, "action": ckey})

            # Record transition and novelty-at-discovery
            dst_was_new = self._graph_record_transition(
                src_sig,
                dst_sig,
                self._actions_signature(cand.actions, vid_map=snap.get("vid_map") or {}),
                touch=False,
                src_snap=snap,
                dst_snap=dst,
            )
            src_fam = self._family_id(src_sig)
            self.probe_novelty.setdefault(src_fam, {})[ckey] = dst_was_new
            self.probe_outcomes.setdefault(src_fam, {})[ckey] = dst_sig
            self._mark_attempted(src_sig, cand)
            self._log_event("probe_result", sig=src_sig, action_key=ckey, dst_sig=dst_sig, dst_was_new=dst_was_new)
            if dst_was_new:
                self.no_new_state_count = 0

            # No-op probe (dst == src): mark this candidate explored/dead and continue probing others.
            if dst_sig == src_sig:
                self._mark_explored(src_sig, cand)
                snap.update(dst)
                continue

            # Pipeline analysis on dst
            self._schedule_state(dst_sig, dst, task)

            # CONDITION: BACK-LIKE (dst is an ancestor already on stack)
            # ACTION:
            # - stop probe-return; replan at ancestor (continuing probe list is invalid now)
            if dst_sig in self.dfs_stack and dst_sig != src_sig:
                logger.warning("Back-like probe: src=%s dst=%s (ancestor). Replan at dst.", src_sig[:8], dst_sig[:8])
                self._set_forced_replan(dst_sig, dst, has_edge=True)
                return

            # Pick candidate-specific return strategy (candidate overrides global; empty => inherit)
            cand_return_method = cand.return_method or global_return_method
            cand_return_actions = list(cand.return_actions or []) or list(global_return_actions)
            return_action_hints = self._build_return_action_hints(cand_return_actions, snap.get("vid_map") or {})

            # If return_method == none/custom "stay", we intentionally continue at dst.
            if cand_return_method == "none" and dst_sig != src_sig:
                logger.info("Probe return_method=none at dst_sig=%s -> replan at dst.", dst_sig[:8])
                self._set_forced_replan(dst_sig, dst, has_edge=True)
                return

            # Probe-return required: restore src_sig to probe the next candidate.
            ok_return = self._return_to_expected(
                expected_sig=src_sig,
                expected_struct_sig=expected_struct_sig,
                start_snap=dst,
                return_method=cand_return_method,
                return_actions=cand_return_actions,
                return_action_hints=return_action_hints,
                task=task,
            )
            self._log_event(
                "probe_return",
                sig=src_sig,
                action_key=ckey,
                return_method=cand_return_method,
                ok=bool(ok_return),
            )
            if not ok_return:
                logger.warning("Return failed after probe (src_sig=%s, dst_sig=%s) -> recovery.", src_sig[:8], dst_sig[:8])
                ok2 = self._recover(dst_sig, dst, reason=RecoveryReason.RETURN_FAILED, task=task, target_sig=src_sig)
                if not ok2:
                    self._restart_and_replay(best_target=src_sig, task=task, reason="probe_return_failed_recovery_failed")

                snap2 = self._capture_and_process()
                if snap2:
                    self._set_forced_replan(snap2["state_sig"], snap2, has_edge=True)
                return

            # Refresh source snapshot (avoid using stale vid_map for subsequent probes)
            src_snap2 = self._capture_and_process()
            if not src_snap2:
                return
            if src_snap2["state_sig"] != src_sig:
                new_sig = str(src_snap2["state_sig"])
                new_struct = str((src_snap2.get("meta") or {}).get("struct_sig") or "")
                if not new_struct:
                    meta2 = src_snap2.get("meta") or {}
                    new_struct = compute_structural_signature(
                        src_snap2.get("uist") or {},
                        foreground_package=str(meta2.get("foreground_package") or ""),
                        foreground_activity=str(meta2.get("foreground_activity") or ""),
                    )
                if expected_struct_sig and new_struct == expected_struct_sig:
                    # Benign dynamic change: treat as same source family but refresh planning/candidates on the new sig.
                    logger.info("Source state_sig changed but structural match old=%s new=%s -> refresh.", src_sig[:8], new_sig[:8])
                    self._set_forced_replan(new_sig, src_snap2, has_edge=True)
                else:
                    logger.warning("Source changed after return old=%s new=%s -> replan.", src_sig[:8], new_sig[:8])
                    self._set_forced_replan(new_sig, src_snap2, has_edge=True)
                return

            snap.update(src_snap2)

    def _return_to_expected(
        self,
        expected_sig: str,
        expected_struct_sig: Optional[str],
        start_snap: Optional[Dict[str, Any]],
        return_method: str,
        return_actions: List[ActionStep],
        task: str,
        return_action_hints: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> bool:
        """
        IPO:
          in : expected_sig (source state we must return to), return_method, return_actions
          out: True if we reach expected_sig again, else False

        WHEN called:
          - after a probe click (src -> dst) when we want to continue probing other src candidates

        WHY (tabs):
          - For tab UIs, Android BACK frequently exits the page instead of switching tabs.
          - return_actions allows LLM1 to specify "click the original tab id" to restore state.

        POLICY (order matters; careful):
          1) If return_actions provided: execute them FIRST (custom is authoritative).
          2) If return_method is tab-back (and no return_actions): try clicking likely tab elements (heuristic).
          3) If return_method is close (and no return_actions): try close-like heuristics.
          4) BACK is a fallback:
             - For "tab-back/custom": BACK is LAST resort.
             - For "back": BACK is primary.
          5) If still not reached: allow LLM3 recovery upstream (caller).
        """
        def matches(s: Dict[str, Any]) -> bool:
            try:
                if not s:
                    return False
                if s.get("state_sig") == expected_sig:
                    return True
                if expected_struct_sig:
                    cur_struct = str((s.get("meta") or {}).get("struct_sig") or "")
                    if not cur_struct:
                        meta = s.get("meta") or {}
                        cur_struct = compute_structural_signature(
                            s.get("uist") or {},
                            foreground_package=str(meta.get("foreground_package") or ""),
                            foreground_activity=str(meta.get("foreground_activity") or ""),
                        )
                    return cur_struct == expected_struct_sig
            except Exception:
                return False
            return False

        cur = start_snap or self._capture_and_process()
        if not cur:
            return False
        cur_sig = str(cur.get("state_sig") or "")

        self._emit_decision(
            cur_sig,
            "probe_return_start",
            {
                "expected_sig": expected_sig,
                "expected_struct_sig": expected_struct_sig or "",
                "return_method": return_method,
                "return_actions_count": len(return_actions or []),
            },
        )

        # Fast check: sometimes already returned.
        if matches(cur):
            self._emit_decision(cur_sig, "probe_return_already_matched", {"expected_sig": expected_sig})
            return True

        def apply_step(step: ActionStep, *, kind: str, why: str) -> Optional[Dict[str, Any]]:
            nonlocal cur, cur_sig
            pre_sig = cur_sig
            self._emit_decision(
                pre_sig,
                "probe_return_step",
                {
                    "kind": kind,
                    "why": why,
                    "return_method": return_method,
                    "action": self._action_signature(step),
                },
            )
            ok = self._execute_action(step, cur.get("vid_map") or {}, pre_sig)
            if not ok:
                self._emit_decision(
                    pre_sig,
                    "probe_return_step_failed",
                    {
                        "kind": kind,
                        "why": why,
                        "return_method": return_method,
                        "action": self._action_signature(step),
                        "last_action_failure": dict(self.last_action_failure or {}),
                    },
                )
                return None
            self.action_count += 1
            self.history.append(self._action_key(step))
            time.sleep(self.budget.post_action_settle_s)
            nxt = self._capture_and_process()
            if not nxt:
                self._emit_decision(
                    pre_sig,
                    "probe_return_capture_failed",
                    {
                        "kind": kind,
                        "why": why,
                        "return_method": return_method,
                        "action": self._action_signature(step),
                    },
                )
                return None
            nxt_sig = str(nxt.get("state_sig") or "")
            self._graph_record_aux_transition(pre_sig, nxt_sig, self._actions_signature([step]), kind=kind)
            self._emit_decision(
                pre_sig,
                "probe_return_step_result",
                {
                    "kind": kind,
                    "why": why,
                    "return_method": return_method,
                    "from_sig": pre_sig,
                    "to_sig": nxt_sig,
                    "matched_expected": bool(matches(nxt)),
                },
            )
            cur = nxt
            cur_sig = nxt_sig
            return nxt

        # 1) Custom return actions (LLM1-provided)
        if return_actions:
            for i, st in enumerate(return_actions[:4]):
                if not cur:
                    return False

                eff = st
                if st.action == ActionType.CLICK:
                    vid_map = cur.get("vid_map") or {}

                    hint: Optional[Dict[str, Any]] = None
                    if return_action_hints and i < len(return_action_hints):
                        hint = return_action_hints[i]

                    # If we can't fingerprint the SOURCE element, don't trust the id (it may point to a different node).
                    if not hint:
                        self._log_event(
                            "return_action_missing_hint",
                            sig=cur_sig,
                            return_method=return_method,
                            element_id=st.element_id,
                        )
                        # Safe-ish fallback: if the current snapshot still has this element_id and the node has a
                        # strong stable anchor, allow one attempt rather than forcing recovery immediately.
                        try:
                            cur_id = int(st.element_id) if st.element_id is not None else None
                        except Exception:
                            cur_id = None
                        node = vid_map.get(cur_id) if cur_id is not None else None
                        if node and bool(node.get("clickable")):
                            rid = self._normalize_str(node.get("resource_id"))
                            cd = self._normalize_str(node.get("content_desc"))
                            txt = self._normalize_str(node.get("text"))
                            il = self._normalize_str(node.get("icon_label"))
                            lbl = (txt or cd or il).strip()
                            strong_anchor = bool(rid) or (bool(cd) and len(cd) <= 48) or (bool(txt) and len(txt) <= 32 and (not txt.isdigit())) or (bool(il) and len(il) <= 32)
                            if strong_anchor:
                                self._log_event(
                                    "return_action_no_hint_fallback",
                                    sig=cur_sig,
                                    return_method=return_method,
                                    element_id=cur_id,
                                    has_rid=bool(rid),
                                    label=(lbl[:32] if lbl else ""),
                                )
                                # Proceed with eff==st (no remap); apply_step will still use stable selectors first.
                            else:
                                break
                        else:
                            break

                    if hint:
                        cur_id = int(st.element_id) if st.element_id is not None else None
                        if cur_id is not None and cur_id in vid_map:
                            s, _, rid_m, lbl_m = self._fingerprint_match(vid_map[cur_id], hint)
                            strong = (rid_m and s >= 6.0) or (lbl_m and s >= 4.5)
                            if not strong:
                                new_id, best_s = self._resolve_element_id_by_fingerprint(vid_map, hint)
                                if new_id is None:
                                    self._log_event(
                                        "return_action_remap_failed",
                                        sig=cur_sig,
                                        return_method=return_method,
                                        from_id=cur_id,
                                        best_score=best_s,
                                    )
                                    break
                                self._log_event(
                                    "return_action_remap",
                                    sig=cur_sig,
                                    return_method=return_method,
                                    from_id=cur_id,
                                    to_id=new_id,
                                    score=best_s,
                                )
                                eff = ActionStep(action=st.action, element_id=new_id, text=st.text, priority=st.priority, reasoning=st.reasoning)
                        else:
                            new_id, best_s = self._resolve_element_id_by_fingerprint(vid_map, hint)
                            if new_id is None:
                                self._log_event(
                                    "return_action_missing_element",
                                    sig=cur_sig,
                                    return_method=return_method,
                                    element_id=st.element_id,
                                    best_score=best_s,
                                )
                                break
                            self._log_event(
                                "return_action_remap",
                                sig=cur_sig,
                                return_method=return_method,
                                from_id=st.element_id,
                                to_id=new_id,
                                score=best_s,
                            )
                            eff = ActionStep(action=st.action, element_id=new_id, text=st.text, priority=st.priority, reasoning=st.reasoning)

                nxt = apply_step(eff, kind="return", why="llm_return_actions")
                if nxt and matches(nxt):
                    return True

        # 2) Tab-back heuristic (ONLY if method indicates tab context and no custom steps worked)
        if return_method in ("tab-back", "custom"):
            tab_ids = self._heuristic_tab_elements(cur.get("vid_map") or {})
            for eid in tab_ids[:4]:
                st = ActionStep(action=ActionType.CLICK, element_id=eid, priority=1, reasoning="heuristic_tab_back")
                nxt = apply_step(st, kind="return", why="heuristic_tab_back")
                if nxt and matches(nxt):
                    return True

        # 3) Close heuristic for close/custom contexts
        if return_method in ("close", "custom"):
            close_ids = self._heuristic_close_elements(cur.get("vid_map") or {})
            for eid in close_ids[:3]:
                st = ActionStep(action=ActionType.CLICK, element_id=eid, priority=1, reasoning="heuristic_close")
                nxt = apply_step(st, kind="return", why="heuristic_close")
                if nxt and matches(nxt):
                    return True

        # 4) BACK fallback (ordering depends on return_method)
        back_first = (return_method == "back")
        back_attempts = 2 if back_first else 1

        for _ in range(back_attempts):
            st = ActionStep(action=ActionType.BACK, element_id=None, priority=1, reasoning="return_back")
            nxt = apply_step(st, kind="return", why="back_fallback_primary")
            if (self.last_action_failure or {}).get("reason") == "back_blocked_settings_root":
                return False
            if (self.last_action_failure or {}).get("reason") == "post_back_foreground_mismatch":
                return False
            if nxt and matches(nxt):
                return True

        # If return_method wasn't "back" and we still didn't reach, try one more BACK as last resort
        if not back_first:
            st = ActionStep(action=ActionType.BACK, element_id=None, priority=1, reasoning="return_back_last")
            nxt = apply_step(st, kind="return", why="back_fallback_last_resort")
            if (self.last_action_failure or {}).get("reason") == "back_blocked_settings_root":
                return False
            if (self.last_action_failure or {}).get("reason") == "post_back_foreground_mismatch":
                return False
            if nxt and matches(nxt):
                return True

        return False

    def _heuristic_tab_elements(self, vid_map: Dict[int, Any]) -> List[int]:
        """
        Heuristic tab finder.
        WHEN used:
          - return_method tab-back/custom without return_actions

        Strategy:
          - prefer nodes whose class/content_desc/text suggests tab
          - prefer bottom area elements
          - return left-to-right ordering by x

        This is intentionally conservative; it is a fallback when LLM1 did not provide return_actions.
        """
        tabs: List[Tuple[float, int]] = []
        for eid, node in vid_map.items():
            try:
                cls = str(node.get("class") or "").lower()
                txt = str(node.get("text") or "").lower()
                cd = str(node.get("content_desc") or "").lower()
                il = str(node.get("icon_label") or "").lower()
                hint = "tab" in cls or "tab" in txt or "tab" in cd
                # many apps don't say "tab"; allow bottom navigation keywords
                hint = hint or any(k in txt or k in cd or k in il for k in ["home", "search", "profile", "account", "me", "settings"])
                if not hint:
                    continue
                f = BaseUI.get_frame(node)
                y = float(f.get("y", 0))
                h = float(f.get("height", 0))
                # prefer bottom-ish controls
                score = y + 0.25 * h
                tabs.append((score, eid))
            except Exception:
                continue

        if not tabs:
            return []

        # First: prefer bottom-ish candidates by score, then order left-to-right for determinism.
        tabs.sort(key=lambda t: float(t[0]), reverse=True)
        top = tabs[:8]

        eids: List[Tuple[float, int]] = []
        for _, eid in top:
            try:
                f = BaseUI.get_frame(vid_map[eid])
                x = float(f.get("x", 0))
                eids.append((x, eid))
            except Exception:
                continue
        eids.sort()
        return [eid for _, eid in eids]

    def _heuristic_close_elements(self, vid_map: Dict[int, Any]) -> List[int]:
        """
        Heuristic close finder for dialogs/webviews.
        WHEN used:
          - return_method close without return_actions

        Strategy:
          - match labels like close/x/done/cancel/back
          - prefer top area and small icon-like bounds
        """
        cands: List[Tuple[float, int]] = []
        pat = re.compile(r"\b(close|cancel|done|dismiss|back|exit|x)\b", re.IGNORECASE)
        for eid, node in vid_map.items():
            try:
                txt = str(node.get("text") or "")
                cd = str(node.get("content_desc") or "")
                il = str(node.get("icon_label") or "")
                s = " ".join([txt, cd, il]).strip()
                if not s or not pat.search(s):
                    continue
                f = BaseUI.get_frame(node)
                x = float(f.get("x", 0))
                y = float(f.get("y", 0))
                w = float(f.get("width", 0))
                h = float(f.get("height", 0))
                # prefer top and icon-sized
                score = 0.0
                score += max(0.0, 1200.0 - y)
                score += max(0.0, 400.0 - (w * h))
                score += max(0.0, 800.0 - x) * 0.01
                cands.append((score, eid))
            except Exception:
                continue
        cands.sort(reverse=True)
        return [eid for _, eid in cands]

    def _node_fingerprint(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a compact, snapshot-independent-ish fingerprint for a UI node.

        Used to remap snapshot-scoped element_ids (especially for return_actions).
        """
        try:
            f = BaseUI.get_frame(node)
            rid = str(node.get("resource_id") or "")
            raw_label = (
                node.get("text")
                or node.get("content_desc")
                or node.get("semantic_label")
                or node.get("icon_label")
                or node.get("ocr_text")
                or ""
            )
            s = str(raw_label).strip().lower()
            s = re.sub(r"\d+", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            label = s[:24]
            return {
                "class": str(node.get("class") or "")[:80],
                "resource_id_tail": rid[-40:],
                "label": label,
                "frame": [int(f["x"]), int(f["y"]), int(f["width"]), int(f["height"])],
            }
        except Exception:
            return {}

    def _fingerprint_match(self, node: Dict[str, Any], fp: Dict[str, Any]) -> Tuple[float, float, bool, bool]:
        """
        Returns:
          (score, iou, rid_match, label_match)
        """
        try:
            score = 0.0
            rid_match = False
            label_match = False

            f = BaseUI.get_frame(node)
            bx1, by1 = int(f["x"]), int(f["y"])
            bx2, by2 = int(f["x"] + f["width"]), int(f["y"] + f["height"])

            af = fp.get("frame") or [0, 0, 0, 0]
            ax1, ay1, aw, ah = [int(x or 0) for x in af]
            ax2, ay2 = ax1 + max(0, aw), ay1 + max(0, ah)

            # IoU
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = float(iw * ih)
            a_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
            b_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
            iou = inter / float(max(1.0, (a_area + b_area - inter)))

            score += 4.0 * float(iou)

            # resource-id tail
            fp_rid = str(fp.get("resource_id_tail") or "")
            rid = str(node.get("resource_id") or "")
            rid_tail = rid[-40:]
            if fp_rid and rid_tail:
                if rid_tail == fp_rid:
                    rid_match = True
                    score += 6.0
                elif fp_rid in rid_tail or rid_tail in fp_rid:
                    score += 3.0

            # label token
            fp_lbl = str(fp.get("label") or "").strip().lower()
            raw_label = (
                node.get("text")
                or node.get("content_desc")
                or node.get("semantic_label")
                or node.get("icon_label")
                or node.get("ocr_text")
                or ""
            )
            cand_lbl = str(raw_label).strip().lower()
            cand_lbl = re.sub(r"\d+", "", cand_lbl)
            cand_lbl = re.sub(r"\s+", " ", cand_lbl).strip()
            cand_lbl = cand_lbl[:24]
            if fp_lbl and cand_lbl:
                if cand_lbl == fp_lbl:
                    label_match = True
                    score += 3.0
                elif fp_lbl in cand_lbl or cand_lbl in fp_lbl:
                    score += 1.5

            # class
            fp_cls = str(fp.get("class") or "")
            cls = str(node.get("class") or "")[:80]
            if fp_cls and cls and cls == fp_cls:
                score += 1.0

            # clickable bias (return actions are usually clicks)
            if node.get("clickable"):
                score += 0.5
            else:
                score -= 0.5

            return float(score), float(iou), bool(rid_match), bool(label_match)
        except Exception:
            return 0.0, 0.0, False, False

    def _resolve_element_id_by_fingerprint(self, vid_map: Dict[int, Any], fp: Dict[str, Any]) -> Tuple[Optional[int], float]:
        if not fp or not vid_map:
            return None, 0.0

        best_eid: Optional[int] = None
        best_score = -1e9
        best_iou = 0.0
        best_rid = False
        best_lbl = False

        for eid, node in vid_map.items():
            s, iou, rid_m, lbl_m = self._fingerprint_match(node, fp)
            if s > best_score:
                best_score = s
                best_eid = int(eid)
                best_iou = float(iou)
                best_rid = bool(rid_m)
                best_lbl = bool(lbl_m)

        # Conservative acceptance: avoid misclicking unrelated elements.
        if best_eid is None:
            return None, float(best_score)
        if best_rid and best_score >= 6.0:
            return best_eid, float(best_score)
        if best_lbl and best_iou >= 0.15 and best_score >= 4.5:
            return best_eid, float(best_score)
        if best_iou >= 0.75 and best_score >= 4.0:
            return best_eid, float(best_score)

        return None, float(best_score)

    def _build_return_action_hints(self, steps: List[ActionStep], src_vid_map: Dict[int, Any]) -> List[Optional[Dict[str, Any]]]:
        """
        Build per-step fingerprints from the SOURCE snapshot so that return_actions can be
        remapped onto the CURRENT snapshot's vid_map at execution time.
        """
        hints: List[Optional[Dict[str, Any]]] = []
        for st in (steps or []):
            if st.action != ActionType.CLICK:
                hints.append(None)
                continue
            if st.element_id is None:
                hints.append(None)
                continue
            node = src_vid_map.get(int(st.element_id))
            if not node:
                hints.append(None)
                continue
            hints.append(self._node_fingerprint(node))
        return hints

    def _post_probe_wait(self) -> None:
        if self.budget.post_probe_wait_s <= 0:
            return
        t0 = time.time()
        while time.time() - t0 < self.budget.post_probe_wait_s:
            self._drain_futures()
            time.sleep(0.05)

    # ---------------------------
    # Exhaustion / backtrace / frontier
    # ---------------------------

    def _is_state_exhausted(self, sig: str) -> bool:
        """
        A state is exhausted when every candidate branch is explored (completed or proven no-effect).
        Attempted != explored.
        """
        fam = self._family_id(sig)
        if fam in self.state_candidates:
            candidates = self.state_candidates.get(fam) or []
        else:
            candidates = []
            nav = self.nav_cache.get(sig)
            if nav and getattr(nav, "candidate_actions", None) is not None:
                self.state_candidates[fam] = list(getattr(nav, "candidate_actions") or [])
                candidates = self.state_candidates.get(fam) or []
            else:
                return False

        # NAV returned but gave no candidates (or all filtered upstream): exhausted for this family.
        if not candidates:
            return True

        explored = self.explored_actions.get(fam, set())
        keys: List[str] = []
        for c in candidates:
            if not c.actions:
                continue
            keys.append(self._candidate_key(c))

        if not keys:
            return True

        return all(k in explored for k in keys)

    def _nearest_ancestor_with_remaining_work(self, cur_sig: str) -> Optional[str]:
        """
        DFS completion:
          - if current state exhausted, backtrace to nearest ancestor on stack that is not exhausted.
        """
        if not self.dfs_stack or self.dfs_stack[-1] != cur_sig:
            return None
        for s in reversed(self.dfs_stack[:-1]):
            if not self._is_state_exhausted(s):
                return s
        return None

    def _pick_global_frontier(self) -> Optional[str]:
        """
        Global frontier:
          - when no ancestor has remaining work, choose an unfinished state that is:
              * non-overlay
              * low visit_count
              * likely to yield questionnaire updates (if known)
        """
        best_sig = None
        best_score = -1e9

        for sig, node in (self.graph.nodes or {}).items():
            if getattr(node, "overlay_kind", "none") in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
                continue
            if self._is_state_exhausted(sig):
                continue

            visits = float(getattr(node, "visit_count", 1) or 1)
            score = 3.0 / max(1.0, visits)

            upd = self.q_cache.get(sig)
            if upd:
                score += min(6.0, 1.2 * float(len(getattr(upd, "proposed_updates", []) or [])))
            score += min(6.0, 1.0 * float(self.state_update_counts.get(sig, 0) or 0))

            if score > best_score:
                best_score = score
                best_sig = sig

        return best_sig

    # ---------------------------
    # Beam-first global opportunities (cost-aware switching)
    # ---------------------------

    def _estimate_replay_steps(self, src_sig: str, dst_sig: str) -> Optional[int]:
        if not src_sig or not dst_sig:
            return None
        if src_sig == dst_sig:
            return 0
        now = time.time()
        key = (src_sig, dst_sig)
        cached = self.replay_distance_cache.get(key)
        if cached and (now - float(cached[1] or 0.0)) <= 120.0:
            steps = int(cached[0])
            return None if steps >= 900 else steps

        state_path, actions = self.graph.shortest_action_path(src_sig, dst_sig, max_depth=self.budget.replay_max_depth)
        if not actions:
            self.replay_distance_cache[key] = (999, now, 0, 0)
            return None

        unverified = 0
        stale = 0
        try:
            stale_after_s = float(self.budget.edge_stale_after_s)
        except Exception:
            stale_after_s = 0.0
        for i, act in enumerate(actions):
            try:
                if i + 1 >= len(state_path):
                    break
                e = self.graph.get_edge(state_path[i], state_path[i + 1], act)
                if not e or float(getattr(e, "last_verified_ts", 0.0) or 0.0) <= 0.0:
                    unverified += 1
                else:
                    if stale_after_s > 0.0 and (now - float(getattr(e, "last_verified_ts", 0.0) or 0.0)) >= stale_after_s:
                        stale += 1
            except Exception:
                unverified += 1

        steps = int(len(actions))
        self.replay_distance_cache[key] = (steps, now, int(unverified), int(stale))
        return steps

    def _active_topic_context(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Legacy beam helper kept for compatibility.

        The block_status workflow no longer computes old active topics. Beam
        scoring can still run, but questionnaire-specific score is neutral
        until we design a block_status-aware frontier policy.
        """
        return [], ""

    def _topic_match_score(self, sig: str, topic_blob: str) -> float:
        score = 0.0

        nav = self.nav_cache.get(sig)
        if nav is not None:
            for tg in (getattr(nav, "page_tags", None) or [])[:10]:
                tag = str(getattr(tg, "tag", "") or "").strip().lower()
                if not tag:
                    continue
                if tag in topic_blob:
                    w = float(getattr(tg, "weight", 0.0) or 0.0)
                    score += 4.0 * max(0.2, min(1.0, w))

        return float(min(10.0, score))

    def _ui_type_bonus(self, sig: str) -> float:
        nav = self.nav_cache.get(sig)
        if nav is None:
            return 0.0
        ut = str(getattr(nav, "ui_type", "") or "").strip()
        if ut in ("settings_list", "detail_form", "permissions_dialog", "subscription_paywall", "auth_flow"):
            return 2.5
        return 0.0

    def _prune_restart_recent(self, now: float) -> None:
        try:
            window_s = float(self.budget.restart_budget_window_s)
        except Exception:
            window_s = 120.0
        if window_s <= 0:
            return
        try:
            while self.restart_recent and (now - float(self.restart_recent[0] or 0.0)) > window_s:
                self.restart_recent.popleft()
        except Exception:
            pass

    def _restart_budget_exceeded(self, now: float) -> bool:
        self._prune_restart_recent(now)
        try:
            limit = int(self.budget.restart_budget_max)
        except Exception:
            limit = 0
        if limit <= 0:
            return False
        return len(self.restart_recent) >= limit

    def _restart_penalty_dynamic(self, now: float) -> float:
        self._prune_restart_recent(now)
        base = float(self.budget.restart_penalty)
        try:
            limit = int(self.budget.restart_budget_max)
        except Exception:
            limit = 0
        if limit <= 0:
            return base
        # Increase cost as we approach/exceed the restart budget (so beam avoids restart-hopping).
        excess = max(0, len(self.restart_recent) - limit + 1)
        return float(base * (1.0 + float(excess)))

    def _beam_commit_active(self, now: float) -> bool:
        if not self.committed_target_sig:
            return False
        # Commit expires only after BOTH conditions are satisfied (min time + min actions).
        return not (now >= float(self.commit_until_ts or 0.0) and self.action_count >= int(self.commit_until_action_count or 0))

    def _clear_beam_commit(self, *, cur_sig: str, reason: str) -> None:
        if self.committed_target_sig:
            self._log_event("beam_commit_cleared", sig=cur_sig, committed_target=self.committed_target_sig, reason=reason)
        self.committed_target_sig = None
        self.committed_target_score = -1e9
        self.commit_until_ts = 0.0
        self.commit_until_action_count = 0

    def _set_beam_commit(self, *, cur_sig: str, target_sig: str, target_score: float, now: float) -> None:
        if self.committed_target_sig == target_sig and self._beam_commit_active(now):
            return
        self.committed_target_sig = target_sig
        self.committed_target_score = float(target_score)
        self.commit_until_ts = float(now) + float(self.budget.beam_commit_min_s)
        self.commit_until_action_count = int(self.action_count) + int(self.budget.beam_commit_min_actions)
        self._log_event(
            "beam_commit_set",
            sig=cur_sig,
            target_sig=target_sig,
            score=float(target_score),
            until_ts=self.commit_until_ts,
            until_action_count=self.commit_until_action_count,
        )

    def _beam_score_target(self, cur_sig: str, target_sig: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        """
        Compute the same score used in _beam_best_target(), but for a specific target_sig.
        Used for commit-to-target hysteresis.
        """
        if not target_sig or not self.graph.nodes:
            return None

        node = self.graph.get_node(target_sig)
        if node and getattr(node, "overlay_kind", "none") in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
            return None
        if self._is_state_exhausted(target_sig):
            return None

        questionnaire_context, topic_blob = self._active_topic_context()

        visits = float(getattr(node, "visit_count", 1) or 1) if node else 1.0
        novelty = 8.0 if visits <= 1 else 2.0
        visit_score = 3.0 / max(1.0, visits)
        yield_score = min(8.0, 1.2 * float(self.state_update_counts.get(target_sig, 0) or 0))

        topic_score = self._topic_match_score(target_sig, topic_blob)
        ui_type_bonus = self._ui_type_bonus(target_sig)
        value = float(novelty + visit_score + yield_score + topic_score + ui_type_bonus)

        replay_src = (self.restart_entry_sig or self.entry_sig or cur_sig) or cur_sig
        reach_mode = "in_graph"
        path_src = cur_sig
        steps = self._estimate_replay_steps(cur_sig, target_sig)
        if steps is None:
            reach_mode = "restart_replay"
            path_src = replay_src
            steps = self._estimate_replay_steps(replay_src, target_sig)
        if steps is None:
            return None

        if reach_mode == "restart_replay" and self._restart_budget_exceeded(time.time()):
            return None

        cost = float(self.budget.travel_cost_weight) * float(steps)
        if reach_mode == "restart_replay":
            cost += float(self._restart_penalty_dynamic(time.time()))

        unverified_edges = 0
        stale_edges = 0
        cached = self.replay_distance_cache.get((path_src, target_sig))
        if cached:
            try:
                unverified_edges = int(cached[2])
                stale_edges = int(cached[3])
            except Exception:
                unverified_edges = 0
                stale_edges = 0
        edge_penalty = float(self.budget.edge_unverified_penalty) * float(unverified_edges) + float(self.budget.edge_stale_penalty) * float(stale_edges)
        cost += edge_penalty

        score = float(value) - float(cost)
        detail = {
            "target_sig": target_sig,
            "visits": visits,
            "novelty": novelty,
            "visit_score": visit_score,
            "yield_score": yield_score,
            "topic_score": topic_score,
            "ui_type_bonus": ui_type_bonus,
            "value": value,
            "reach_mode": reach_mode,
            "replay_src": replay_src,
            "replay_steps": steps,
            "path_src": path_src,
            "unverified_edges": unverified_edges,
            "stale_edges": stale_edges,
            "edge_penalty": edge_penalty,
            "cost": cost,
            "score": score,
            "questionnaire_context": questionnaire_context,
        }
        return score, detail

    def _beam_best_target(self, cur_sig: str) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        if not self.graph.nodes:
            return None

        questionnaire_context, topic_blob = self._active_topic_context()

        # 1) Build value-only candidates (cheap) then cost them for a small beam.
        candidates: List[Tuple[float, str, Dict[str, Any]]] = []
        for sig, node in (self.graph.nodes or {}).items():
            if sig == cur_sig:
                continue
            if sig in self.dfs_stack:
                # Ancestors are handled by deterministic backtrace; avoid restart thrash.
                continue
            if getattr(node, "overlay_kind", "none") in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
                continue
            if self._is_state_exhausted(sig):
                continue

            visits = float(getattr(node, "visit_count", 1) or 1)
            novelty = 8.0 if visits <= 1 else 2.0
            visit_score = 3.0 / max(1.0, visits)
            yield_score = min(8.0, 1.2 * float(self.state_update_counts.get(sig, 0) or 0))

            topic_score = self._topic_match_score(sig, topic_blob)
            ui_type_bonus = self._ui_type_bonus(sig)

            value = novelty + visit_score + yield_score + topic_score + ui_type_bonus
            candidates.append(
                (
                    value,
                    sig,
                    {
                        "target_sig": sig,
                        "visits": visits,
                        "novelty": novelty,
                        "visit_score": visit_score,
                        "yield_score": yield_score,
                        "topic_score": topic_score,
                        "ui_type_bonus": ui_type_bonus,
                        "value": value,
                    },
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)
        beam = candidates[: max(1, int(self.budget.beam_width))]

        replay_src = (self.restart_entry_sig or self.entry_sig or cur_sig) or cur_sig
        best_sig = None
        best_score = -1e9
        best_detail: Optional[Dict[str, Any]] = None

        for value, sig, detail in beam:
            reach_mode = "in_graph"
            path_src = cur_sig
            steps = self._estimate_replay_steps(cur_sig, sig)
            if steps is None:
                # Fall back to restart+replay planning if no in-graph path is known.
                reach_mode = "restart_replay"
                path_src = replay_src
                steps = self._estimate_replay_steps(replay_src, sig)
            if steps is None:
                continue

            if reach_mode == "restart_replay" and self._restart_budget_exceeded(time.time()):
                continue

            cost = float(self.budget.travel_cost_weight) * float(steps)
            if reach_mode == "restart_replay":
                cost += float(self._restart_penalty_dynamic(time.time()))

            # Penalize paths that rely on unverified/stale edges (replay reliability).
            unverified_edges = 0
            stale_edges = 0
            cached = self.replay_distance_cache.get((path_src, sig))
            if cached:
                try:
                    unverified_edges = int(cached[2])
                    stale_edges = int(cached[3])
                except Exception:
                    unverified_edges = 0
                    stale_edges = 0
            edge_penalty = float(self.budget.edge_unverified_penalty) * float(unverified_edges) + float(self.budget.edge_stale_penalty) * float(stale_edges)
            cost += edge_penalty
            score = float(value) - float(cost)

            if score > best_score:
                best_score = score
                best_sig = sig
                best_detail = {
                    **detail,
                    "reach_mode": reach_mode,
                    "replay_src": replay_src,
                    "replay_steps": steps,
                    "path_src": path_src,
                    "unverified_edges": unverified_edges,
                    "stale_edges": stale_edges,
                    "edge_penalty": edge_penalty,
                    "cost": cost,
                    "score": score,
                    "questionnaire_context": questionnaire_context,
                }

        if not best_sig or best_detail is None:
            return None
        return best_sig, float(best_score), best_detail

    def _maybe_beam_switch(self, cur_sig: str, local_score: float, cur_snap: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        if self.last_jump_ts and (now - self.last_jump_ts) < float(self.budget.jump_cooldown_s):
            return None

        # Commit-to-target hysteresis: avoid bouncing between global targets.
        commit_active = self._beam_commit_active(now)
        if self.committed_target_sig and not commit_active:
            self._clear_beam_commit(cur_sig=cur_sig, reason="expired")
            commit_active = False

        if commit_active and self.committed_target_sig:
            ct = str(self.committed_target_sig)
            if self._is_state_exhausted(ct):
                self._clear_beam_commit(cur_sig=cur_sig, reason="target_exhausted")
                commit_active = False
            else:
                n = self.graph.get_node(ct)
                if n and getattr(n, "overlay_kind", "none") in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
                    self._clear_beam_commit(cur_sig=cur_sig, reason="target_overlay")
                    commit_active = False

        best = self._beam_best_target(cur_sig)
        if not best:
            return None
        target_sig, target_score, detail = best

        # If a commit is active, keep pursuing the committed target unless a new one wins by a big margin.
        if commit_active and self.committed_target_sig:
            ct = str(self.committed_target_sig)
            if ct != target_sig:
                if ct in self.dfs_stack:
                    # We are already within the committed branch; block global jumps unless override wins by a lot.
                    if float(target_score) < float(self.committed_target_score) + float(self.budget.beam_commit_override_gain):
                        self._log_event(
                            "beam_commit_hold",
                            sig=cur_sig,
                            committed_target=ct,
                            committed_score=self.committed_target_score,
                            candidate_target=target_sig,
                            candidate_score=target_score,
                            reason="committed_on_stack",
                        )
                        return None
                else:
                    scored = self._beam_score_target(cur_sig, ct)
                    if not scored:
                        self._clear_beam_commit(cur_sig=cur_sig, reason="committed_unreachable")
                        commit_active = False
                    else:
                        ct_score, ct_detail = scored
                        self.committed_target_score = float(ct_score)
                        if float(target_score) < float(ct_score) + float(self.budget.beam_commit_override_gain):
                            target_sig, target_score, detail = ct, float(ct_score), ct_detail
                        else:
                            self._clear_beam_commit(cur_sig=cur_sig, reason="override_by_better_target")
                            commit_active = False

        if target_sig == cur_sig:
            return None

        if float(target_score) < float(local_score) + float(self.budget.min_switch_gain):
            return None

        self.last_jump_ts = now
        self._last_beam_detail = {"from_sig": cur_sig, "to_sig": target_sig, "local_score": local_score, **detail}
        self._emit_decision(cur_sig, "beam_switch", dict(self._last_beam_detail))
        self._log_event("beam_switch", sig=cur_sig, **self._last_beam_detail)
        self._set_beam_commit(cur_sig=cur_sig, target_sig=target_sig, target_score=float(target_score), now=now)

        mode = str(detail.get("reach_mode") or "restart_replay")
        if mode == "in_graph":
            reached, snap2 = self._navigate_via_graph(start_sig=cur_sig, start_snap=cur_snap, target_sig=target_sig, task=task)
            if reached:
                return snap2
            # If in-graph navigation diverged, fall back to restart+replay to avoid thrash.
            self._restart_and_replay(best_target=target_sig, task=task, reason="beam_in_graph_navigation_diverged")
            return self._capture_and_process(timeout=10.0)

        self._restart_and_replay(best_target=target_sig, task=task, reason="beam_restart_replay_selected")
        return self._capture_and_process(timeout=10.0)

    def _backtrace_to(self, target_sig: str, task: str, *, start_snap: Optional[Dict[str, Any]] = None) -> bool:
        """
        IPO:
          in : target_sig (ancestor state we want to return to)
          out: True if reached

        WHEN called:
          - after a state is exhausted, to perform complete-navigation backtracking

        NOTE:
          - Backtrace itself is not recorded as edges (we don’t know the UI element).
          - We DO record observations on each step so visit_count reflects time spent/seen.
        """
        snap = start_snap or self._capture_and_process()
        if not snap:
            return False

        cur_sig = str(snap.get("state_sig") or "")
        for _ in range(self.budget.backtrace_max_steps):
            st = ActionStep(action=ActionType.BACK, element_id=None, priority=1, reasoning="backtrace_back")
            ok = self._execute_action(st, snap.get("vid_map") or {}, cur_sig)
            if not ok:
                return False
            self.action_count += 1
            self.history.append("back:None:")
            time.sleep(self.budget.post_action_settle_s)

            snap2 = self._capture_and_process()
            if not snap2:
                continue
            cur_sig = str(snap2.get("state_sig") or "")
            self._graph_record_observation(cur_sig, snap=snap2)
            self._mark_progress("state_changed", {"sig": cur_sig, "via": "backtrace"})
            # Intentional backtrace: pop stack and mark explored for popped branches
            self._pop_stack_to(cur_sig, mark_explored=True)

            if cur_sig == target_sig:
                return True

            # Overlay during backtrace: resolve before continuing
            nav, _ = self._wait_nav_or_fallback(cur_sig, snap2, task)
            if nav and self._overlay_kind_value(nav) == OverlayKind.DISMISS.value:
                logger.info("Overlay during backtrace at sig=%s -> resolve.", cur_sig[:8])
                resolved = self._dismiss_overlay_with_nav(cur_sig, snap2, nav, task)
                if not resolved:
                    ok2 = self._recover(cur_sig, snap2, reason=RecoveryReason.OVERLAY_DURING_BACKTRACE, task=task, target_sig=target_sig)
                    if not ok2:
                        return False

                snap3 = self._capture_and_process()
                if snap3 and str(snap3.get("state_sig") or "") == target_sig:
                    return True

            snap = snap2

        return False

    # ---------------------------
    # Forward selection (visited-aware)
    # ---------------------------

    def _choose_forward(
        self, src_sig: str, nav: Optional[NavigationProposal], *, candidates: Optional[List[ActionCandidate]] = None
    ) -> Optional[ActionCandidate]:
        """
        Evidence-driven forward selection:
          - Prefer destinations that were novel-at-probe time.
          - Still allow visited destinations if they improve questionnaire progress or remain unfinished.

        WHICH state:
          - src_sig is current state
          - probe_outcomes[src_sig] maps probed candidate_key -> dst_sig
        """
        outcomes = self.probe_outcomes.get(self._family_id(src_sig), {})
        explored = self.explored_actions.get(self._family_id(src_sig), set())
        pool = list(candidates or []) or list(getattr(nav, "candidate_actions", []) or [])
        if not outcomes:
            logger.warning("No probe outcomes for forward selection at src_sig=%s", src_sig[:8])
            # No probes available: pick first available candidate if any (avoid explored/attempted).
            best = None
            best_score = -1e9
            best_detail = None
            for c in pool:
                if not c.actions:
                    continue
                if self._is_explored(src_sig, c) or self._already_attempted(src_sig, c):
                    continue
                cand_score = float(getattr(c, "score", 0.0) or 0.0)
                score = self.budget.cand_score_weight * cand_score
                if score > best_score:
                    best_score = score
                    best = c
                    best_detail = {
                        "action_key": self._candidate_key(c),
                        "dst_sig": None,
                        "score": score,
                        "candidate_score": cand_score,
                        "novelty_score": 0.0,
                        "q_yield": 0.0,
                        "visit_score": 0.0,
                        "back_penalty": 0.0,
                    }
            self._last_forward_detail = best_detail
            if best:
                return best
            return None

        # 把候选列表整理成 {候选稳定key -> 候选对象} 的映射，后面按 key 查 candidate 会更方便。
        cand_map = {self._candidate_key(c): c for c in pool if getattr(c, "actions", None)}
        # 当前找到的最优 forward 候选，初始化为空。
        best = None
        # 当前最优分数，先给一个很小的初值，保证第一个合法候选能覆盖它。
        best_score = -1e9
        # 记录最优候选的打分细节，方便调试“为什么选了它”。
        best_detail: Optional[Dict[str, Any]] = None
        # 取当前活跃 topic 的上下文，用来给“更贴近当前问卷主题”的页面加分。
        _, topic_blob = self._active_topic_context()

        for ckey, dst_sig in outcomes.items():
            # 如果这个候选已经在当前页被标记为 explored，就不再重复提交。
            if ckey in explored:
                continue
            # 如果这个候选当前处于黑名单中，也先跳过。
            if self._is_action_blacklisted(src_sig, ckey):
                continue
            # 通过 key 找回对应的候选对象；如果候选池里已经没有它，就没法继续算分。
            cand = cand_map.get(ckey)
            if not cand:
                continue

            # probe 结果如果显示“目标页还是当前页”，说明这个候选基本没带来前进价值，跳过。
            if dst_sig == src_sig:
                continue

            # 如果目标页已经被识别成 dismiss / loading 这类 overlay，就不把它当作正式 forward 目标。
            dst_nav = self.nav_cache.get(dst_sig)
            if dst_nav and self._overlay_kind_value(dst_nav) in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value):
                continue

            # 如果目标子页面本身已经 exhausted（没有可继续探索的价值），就不要再 commit 过去。
            if self._is_state_exhausted(dst_sig):
                continue

            # 看 probe 阶段是否把这个候选对应的目标页判成“新状态”。
            is_new_at_probe = self.probe_novelty.get(self._family_id(src_sig), {}).get(ckey, False)

            # 新页面奖励更高；不是全新页面也给一个较小的基础分，避免所有旧页面一票否决。
            novelty_score = 10.0 if is_new_at_probe else 2.0
            # 总分从 novelty 分起步。
            score = novelty_score

            # 如果目标页已经产出了问卷更新，就把“问卷收益”计入分数。
            upd = self.q_cache.get(dst_sig)
            q_yield = 0.0
            if upd:
                # proposed_updates 越多，说明这个页面越可能对填问卷有帮助；这里封顶到 8 分。
                q_yield = min(8.0, 1.4 * float(len(getattr(upd, "proposed_updates", []) or [])))
                score += q_yield

            # 目标页和当前活跃 topic 越匹配，越加分。
            topic_score = self._topic_match_score(dst_sig, topic_blob)
            # 某些 UI 类型（例如更像主流程页面）会有额外 bonus。
            ui_type_bonus = self._ui_type_bonus(dst_sig)
            score += topic_score
            score += ui_type_bonus

            # 访问次数少的页面更值得去，避免总在老页面之间绕圈。
            node = self.graph.get_node(dst_sig)
            visit_score = 0.0
            if node:
                visits = float(getattr(node, "visit_count", 1) or 1)
                visit_score = 3.0 / max(1.0, visits)
                score += visit_score

            # 尽量找出当前页面在 DFS 路径里的父节点是谁，后面用来识别“这个候选是不是明显在往回走”。
            parent = None
            try:
                if src_sig in self.dfs_stack:
                    i = self.dfs_stack.index(src_sig)
                    parent = self.dfs_stack[i - 1] if i > 0 else None
            except Exception:
                parent = None
            if parent is None:
                parent = self.parent_map.get(src_sig)
            back_penalty = 0.0
            if parent and dst_sig == parent:
                # 如果目标页正好就是父节点，说明这个 forward 很像“往回退”，这里做一个惩罚。
                # 明确的回退由 DFS backtrace 去处理，不希望 forward 阶段浪费在明显后退上。
                back_penalty = -2.0
                score += back_penalty

            # LLM1 给候选自带的 score 也会参与总分，但只是综合因素中的一项。
            cand_score = float(getattr(cand, "score", 0.0) or 0.0)
            # 把候选原始分乘上权重，得到它对总分的贡献。
            cand_score_term = self.budget.cand_score_weight * cand_score
            score += cand_score_term

            # 这个候选在当前页/当前 family 下尝试次数越多，重复惩罚越大。
            attempts = int(self.action_attempt_counts.get((self._family_id(src_sig), ckey), 0) or 0)
            # 用 log1p 做惩罚，让第一次、第二次重复更敏感，后面增长逐渐变缓。
            repeat_pen = float(self.budget.repeat_penalty_alpha) * float(math.log1p(max(0, attempts)))
            score -= repeat_pen

            # 维护一个“当前分数最高的候选”。
            if score > best_score:
                best_score = score
                best = cand
                # 把每个组成部分都记下来，便于后面输出 explain / trace。
                best_detail = {
                    "action_key": ckey,
                    "dst_sig": dst_sig,
                    "score": score,
                    "novelty_score": novelty_score,
                    "q_yield": q_yield,
                    "topic_score": topic_score,
                    "ui_type_bonus": ui_type_bonus,
                    "visit_score": visit_score,
                    "back_penalty": back_penalty,
                    "candidate_score": cand_score,
                    "candidate_score_term": cand_score_term,
                    "repeat_penalty": repeat_pen,
                    "attempts": attempts,
                }

        # 保存这次 forward 选择的细节，供日志/调试查看。
        self._last_forward_detail = best_detail
        if best is not None:
            return best

        # Break-glass:
        # 如果前面的严格过滤把所有候选都筛掉了（例如全被 blacklist/exhausted 过滤掉），
        # 那就退回一个更保守的兜底策略：只按 candidate 自带分数再选一次。
        if outcomes:
            self._log_event("blacklist_break_glass", sig=src_sig, kind="forward_selection")
            best = None
            best_score = -1e9
            best_detail = None
            for ckey, dst_sig in outcomes.items():
                # explored 的仍然不考虑，避免明显重复。
                if ckey in explored:
                    continue
                cand = cand_map.get(ckey)
                # 没 candidate 或根本没离开当前页的，兜底也不选。
                if not cand or dst_sig == src_sig:
                    continue
                # 兜底模式下不再用复杂因素，只看 LLM 原始 candidate 分数。
                cand_score = float(getattr(cand, "score", 0.0) or 0.0)
                score = self.budget.cand_score_weight * cand_score
                if score > best_score:
                    best_score = score
                    best = cand
                    best_detail = {"action_key": ckey, "dst_sig": dst_sig, "score": score, "candidate_score": cand_score}
            self._last_forward_detail = best_detail
        return best

    # ---------------------------
    # Recovery / Restart / Replay
    # ---------------------------

    def _is_nonblocking_state(self, sig: str) -> bool:
        nav = self.nav_cache.get(sig)
        if not nav:
            return False
        return self._overlay_kind_value(nav) not in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value)

    def _validate_recovery_step(self, step: ActionStep, snap: Dict[str, Any]) -> Tuple[bool, str]:
        vid_map = snap.get("vid_map") or {}
        if step.action in (ActionType.NONE, ActionType.COMPLETE):
            return False, "noop_action"

        if step.action == ActionType.CLICK:
            node = vid_map.get(step.element_id or -1)
            if not node:
                if step.element_id is None and (getattr(step, "bbox", None) or (getattr(step, "x", None) is not None and getattr(step, "y", None) is not None)):
                    return True, ""
                return False, "missing_element"
            f = BaseUI.get_frame(node)
            if float(f.get("width", 0)) <= 1 or float(f.get("height", 0)) <= 1:
                return False, "degenerate_bounds"
            return True, ""

        if step.action == ActionType.INPUT:
            if step.element_id is not None:
                node = vid_map.get(step.element_id)
                if not node:
                    return False, "missing_element"
                f = BaseUI.get_frame(node)
                if float(f.get("width", 0)) <= 1 or float(f.get("height", 0)) <= 1:
                    return False, "degenerate_bounds"
                return True, ""
            # no element_id: ensure a focused input exists
            for node in vid_map.values():
                if node.get("focused"):
                    return True, ""
            return False, "no_focused_input"

        # BACK / WAIT / RESTART are always valid
        return True, ""

    def _apply_recovery_step(
        self, step: ActionStep, sig: str, snap: Dict[str, Any], task: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        ok = self._execute_action(step, snap["vid_map"], sig)
        if not ok:
            return False, sig, snap
        self.action_count += 1
        self.history.append(self._action_key(step))
        time.sleep(self.budget.post_action_settle_s)

        ns = self._capture_and_process()
        if not ns:
            return False, sig, snap

        new_sig = ns["state_sig"]
        self._graph_record_transition(
            sig,
            new_sig,
            self._actions_signature([step], vid_map=snap.get("vid_map") or {}),
            src_snap=snap,
            dst_snap=ns,
        )
        self._enter_state(from_sig=sig, to_sig=new_sig, via_action=self._action_key(step))
        sig, snap = new_sig, ns

        self._schedule_state(sig, snap, task)
        self._drain_futures()
        return True, sig, snap

    def _deterministic_recovery(
        self, sig: str, snap: Dict[str, Any], reason: RecoveryReason, task: str, target_sig: Optional[str]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if reason == RecoveryReason.FOREGROUND_MISMATCH:
            # Try BACK a couple times; then force foreground; then restart as last resort.
            for _ in range(2):
                try:
                    self.appium.back()
                    self.action_count += 1
                    self.history.append("back:None:")
                    time.sleep(self.budget.post_action_settle_s)
                except Exception:
                    pass
                ns = self._capture_and_process()
                if ns:
                    sig, snap = ns["state_sig"], ns
                    ok, _pkg = self._foreground_is_allowed()
                    if ok:
                        self.foreground_mismatch_streak = 0
                        return True, sig, snap

            try:
                if self.target_package:
                    self.appium.ensure_foreground(self.target_package, self.target_activity)
                    self.foreground_recoveries += 1
            except Exception:
                pass

            ns = self._capture_and_process()
            if ns:
                sig, snap = ns["state_sig"], ns
                ok, _pkg = self._foreground_is_allowed()
                if ok:
                    self.foreground_mismatch_streak = 0
                    return True, sig, snap

            self._restart_and_replay(best_target=target_sig, task=task, reason=f"recovery_{reason.value}_foreground_mismatch")
            return True, sig, snap

        if reason in (RecoveryReason.CAPTURE_FAILED, RecoveryReason.CAPTURE_FAILED_AFTER_FORWARD, RecoveryReason.PROBE_CAPTURE_FAILED):
            time.sleep(1.0)
            ns = self._capture_and_process()
            if ns:
                sig, snap = ns["state_sig"], ns
                self._schedule_state(sig, snap, task)
                self._drain_futures()
                if target_sig and sig == target_sig:
                    return True, sig, snap
                if self._is_nonblocking_state(sig):
                    return True, sig, snap
                return False, sig, snap
            self._restart_and_replay(best_target=target_sig, task=task, reason=f"recovery_{reason.value}_capture_retry_failed")
            return True, sig, snap

        if reason == RecoveryReason.RETURN_FAILED:
            for _ in range(3):
                st = ActionStep(action=ActionType.BACK, element_id=None, text=None, priority=1, reasoning="recovery_back")
                self._log_event("recovery_deterministic_step", sig=sig, action=self._action_signature(st))
                ok, sig, snap = self._apply_recovery_step(st, sig, snap, task)
                if not ok:
                    continue
                if target_sig and sig == target_sig:
                    return True, sig, snap
                if self._is_nonblocking_state(sig):
                    return True, sig, snap
            return False, sig, snap

        if reason in (RecoveryReason.OVERLAY_UNRESOLVED, RecoveryReason.OVERLAY_DURING_BACKTRACE):
            close_ids = self._heuristic_close_elements(snap["vid_map"])
            for eid in close_ids[:3]:
                st = ActionStep(action=ActionType.CLICK, element_id=eid, priority=1, reasoning="recovery_close")
                self._log_event("recovery_deterministic_step", sig=sig, action=self._action_signature(st))
                ok, sig, snap = self._apply_recovery_step(st, sig, snap, task)
                if not ok:
                    continue
                if self._is_nonblocking_state(sig):
                    return True, sig, snap
            for _ in range(2):
                st = ActionStep(action=ActionType.BACK, element_id=None, priority=1, reasoning="recovery_back")
                self._log_event("recovery_deterministic_step", sig=sig, action=self._action_signature(st))
                ok, sig, snap = self._apply_recovery_step(st, sig, snap, task)
                if not ok:
                    continue
                if self._is_nonblocking_state(sig):
                    return True, sig, snap
            return False, sig, snap

        if reason == RecoveryReason.STUCK_NO_PROGRESS:
            target = self._nearest_ancestor_with_remaining_work(sig)
            if target:
                ok = self._backtrace_to(target_sig=target, task=task, start_snap=snap)
                if ok:
                    ns = self._capture_and_process()
                    if ns:
                        sig, snap = ns["state_sig"], ns
                    return True, sig, snap
            frontier = self._pick_global_frontier()
            if frontier and frontier != sig:
                reached, snap2 = self._navigate_via_graph(start_sig=sig, start_snap=snap, target_sig=frontier, task=task)
                if reached:
                    return True, str(snap2.get("state_sig") or sig), snap2
            self._restart_and_replay(best_target=frontier, task=task, reason="recovery_return_failed_frontier_fallback")
            ns = self._capture_and_process()
            if ns:
                return True, ns["state_sig"], ns
            return True, sig, snap

        if reason == RecoveryReason.BACKTRACE_FAILED and target_sig:
            reached, snap2 = self._navigate_via_graph(start_sig=sig, start_snap=snap, target_sig=target_sig, task=task)
            if reached:
                return True, str(snap2.get("state_sig") or sig), snap2
            self._restart_and_replay(best_target=target_sig, task=task, reason="recovery_backtrace_failed_target_unreachable")
            ns = self._capture_and_process()
            if ns:
                return True, ns["state_sig"], ns
            return True, sig, snap

        return False, sig, snap

    def _recover(self, sig: str, snap: Dict[str, Any], reason: RecoveryReason, task: str, target_sig: Optional[str]) -> bool:
        """
        IPO:
          in : current sig+snap, reason, optional target_sig
          out: True if we reach a stable non-overlay state or target_sig
        """
        if not isinstance(reason, RecoveryReason):
            try:
                reason = RecoveryReason(str(reason))
            except Exception:
                reason = RecoveryReason.STUCK_NO_PROGRESS

        self._emit_decision(sig, "recovery_start", {"reason": reason.value, "target_sig": target_sig})
        self._log_event("recovery_start", sig=sig, reason=reason.value, target_sig=target_sig)

        # Deterministic ladder first
        det_ok, sig, snap = self._deterministic_recovery(sig, snap, reason, task, target_sig)
        if det_ok:
            self._emit_decision(sig, "recovery_success", {"via": "deterministic", "target_sig": target_sig})
            return True

        for attempt in range(self.budget.recovery_attempts):
            logger.info("Recovery attempt %d/%d at sig=%s reason=%s", attempt + 1, self.budget.recovery_attempts, sig[:8], reason.value)
            last_nav = self.nav_cache.get(sig)

            try:
                self._emit_llm_enqueued("recovery", sig, {"state_sig": sig, "reason": reason.value, "attempt": attempt + 1})
                t0 = time.time()
                rec: RecoveryProposal = self.gpt.recover_state(
                    snap["screenshot"],
                    snap["uist"],
                    frontier_hint=self.graph.frontier_hint(),
                    last_nav=last_nav,
                    note=f"{reason.value}; target={target_sig}",
                    state_sig=sig,
                )
                self._emit_llm_result(
                    "recovery",
                    sig,
                    {
                        "state_sig": sig,
                        "duration_s": time.time() - t0,
                        "result": rec.model_dump(mode="json") if hasattr(rec, "model_dump") else getattr(rec, "__dict__", {}),
                    },
                )
            except Exception:
                logger.debug("LLM3 recover_state failed", exc_info=True)
                rec = RecoveryProposal(state_sig=sig, ui_view=UIView(), page_summary="", candidate_actions=[])
                self._emit_llm_result(
                    "recovery",
                    sig,
                    {
                        "state_sig": sig,
                        "error": "recover_state_failed",
                        "duration_s": None,
                    },
                )

            if getattr(rec, "state_sig", "") and rec.state_sig != sig:
                self._log_event("recovery_reject", sig=sig, reason="stale_state_sig", rec_state=rec.state_sig)
                snap = self._capture_and_process() or snap
                sig = snap["state_sig"]
                continue

            steps = getattr(rec, "candidate_actions", []) or []
            logger.info("LLM3 recovery steps=%d why=%s", len(steps), (getattr(rec, "why", "") or "")[:140])

            no_effect_steps = 0
            prev_overlay = self._overlay_kind_for_sig(sig)
            for st in steps[:5]:
                valid, why = self._validate_recovery_step(st, snap)
                if not valid:
                    self._log_event("recovery_step_reject", sig=sig, action=self._action_signature(st), reason=why)
                    continue
                self._log_event("recovery_step_accept", sig=sig, action=self._action_signature(st))

                prev_sig = sig
                prev_overlay = self._overlay_kind_for_sig(sig, fallback=prev_overlay)
                ok, sig, snap = self._apply_recovery_step(st, sig, snap, task)
                if not ok:
                    continue

                new_overlay = self._overlay_kind_for_sig(sig, fallback=prev_overlay)
                if sig == prev_sig and new_overlay == prev_overlay:
                    no_effect_steps += 1
                else:
                    no_effect_steps = 0

                if no_effect_steps >= 2:
                    self._log_event("recovery_no_effect", sig=sig, steps=no_effect_steps)
                    self._restart_and_replay(best_target=target_sig, task=task, reason="recovery_no_effect")
                    return True

                if target_sig and sig == target_sig:
                    self._emit_decision(sig, "recovery_success", {"reached_target": True})
                    return True

                if self._is_nonblocking_state(sig):
                    self._emit_decision(sig, "recovery_success", {"reached_target": False})
                    return True

        return False

    def _safe_action_type(self, action_value: str) -> ActionType:
        try:
            return ActionType(action_value)
        except Exception:
            return ActionType.WAIT

    def _action_dict_to_steps(self, act: Dict[str, Any]) -> List[ActionStep]:
        """
        Convert a graph-stored action payload back into ActionStep list.
        Graph actions are stored as {"actions": [ ... ]}.
        """
        if not act:
            return []

        steps_data = act.get("actions")
        if isinstance(steps_data, list) and steps_data:
            out: List[ActionStep] = []
            for a in steps_data:
                out.append(
                    ActionStep(
                        action=self._safe_action_type(a.get("action", "click")),
                        element_id=a.get("element_id"),
                        text=a.get("text"),
                        priority=1,
                        reasoning="replay",
                    )
                )
            return out
        return []

    def _execute_action_payload(self, act: Dict[str, Any], snap: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a graph-stored action payload of the form {"actions":[...]}.
        Supports fingerprint-based remapping when element_id is snapshot-scoped/missing.

        Returns:
          (ok, used_payload) where used_payload keeps original element_id/fingerprint and adds
          per-step "used_element_id" when a remap was required.
        """
        if not act or not isinstance(act.get("actions"), list):
            return False, act or {}

        vid_map: Dict[int, Any] = snap.get("vid_map") or {}
        used_actions: List[Dict[str, Any]] = []

        for raw in (act.get("actions") or [])[:5]:
            if not isinstance(raw, dict):
                continue
            action_val = str(raw.get("action") or "wait")
            step_type = self._safe_action_type(action_val)
            text = raw.get("text")
            fp = raw.get("fingerprint") if isinstance(raw.get("fingerprint"), dict) else None
            elem = raw.get("element_id")

            used_id: Optional[int] = None
            if step_type in (ActionType.CLICK, ActionType.INPUT):
                try:
                    if elem is not None:
                        used_id = int(elem)
                except Exception:
                    used_id = None

                if used_id is not None and used_id in vid_map:
                    if fp:
                        s, _iou, rid_m, lbl_m = self._fingerprint_match(vid_map[used_id], fp)
                        strong = (rid_m and s >= 6.0) or (lbl_m and s >= 4.5)
                        if not strong:
                            remap_id, best_s = self._resolve_element_id_by_fingerprint(vid_map, fp)
                            if remap_id is None:
                                self._log_event(
                                    "replay_remap_failed",
                                    sig=str(snap.get("state_sig") or ""),
                                    from_id=used_id,
                                    best_score=best_s,
                                )
                                return False, act
                            used_id = int(remap_id)
                else:
                    if fp:
                        remap_id, best_s = self._resolve_element_id_by_fingerprint(vid_map, fp)
                        if remap_id is None:
                            self._log_event(
                                "replay_missing_element",
                                sig=str(snap.get("state_sig") or ""),
                                element_id=elem,
                                best_score=best_s,
                            )
                            return False, act
                        used_id = int(remap_id)
                    else:
                        return False, act

            step = ActionStep(action=step_type, element_id=used_id, text=text, priority=1, reasoning="graph_replay")
            if not self._execute_action(step, vid_map, str(snap.get("state_sig") or "")):
                return False, act
            self.action_count += 1
            self.history.append(self._action_key(step))
            time.sleep(self.budget.post_action_settle_s)

            # Preserve original step dict, but attach used id for audit/debug.
            used_step = dict(raw)
            if used_id is not None:
                used_step["used_element_id"] = int(used_id)
            used_actions.append(used_step)

        used_payload = dict(act)
        used_payload["actions"] = used_actions
        return True, used_payload

    def _navigate_via_graph(
        self,
        *,
        start_sig: str,
        start_snap: Dict[str, Any],
        target_sig: str,
        task: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Best-effort in-app navigation using only known graph edges (no restart).
        Returns (reached, final_snapshot).
        """
        if not start_sig or not target_sig or not start_snap:
            return False, start_snap
        if start_sig == target_sig:
            return True, start_snap

        state_path, actions = self.graph.shortest_action_path(start_sig, target_sig, max_depth=self.budget.replay_max_depth)
        if not actions or not state_path:
            return False, start_snap

        cur_sig = start_sig
        snap = start_snap

        for i, act in enumerate(actions):
            ok, used_act = self._execute_action_payload(act, snap)
            if not ok:
                self._log_event("graph_nav_failed", sig=cur_sig, step_index=i + 1, reason="action_exec_failed")
                return False, snap

            snap2 = self._capture_and_process()
            if not snap2:
                self._log_event("graph_nav_failed", sig=cur_sig, step_index=i + 1, reason="capture_failed")
                return False, snap

            new_sig = snap2["state_sig"]

            # Update replay verification stats for the planned edge (before any divergence abort).
            try:
                expected = state_path[i + 1] if (i + 1) < len(state_path) else None
                if expected:
                    ok_edge = bool(new_sig == expected) or (self._family_id(str(new_sig)) == self._family_id(str(expected)))
                    self.graph.record_edge_verification(state_path[i], expected, act, ok=ok_edge)
            except Exception:
                pass

            self._graph_record_transition(cur_sig, new_sig, used_act, src_snap=snap, dst_snap=snap2)
            self._enter_state(from_sig=cur_sig, to_sig=new_sig, via_action=self._action_payload_key(used_act))

            snap = snap2
            cur_sig = new_sig

            self._schedule_state(cur_sig, snap, task)
            self._drain_futures()

            # Divergence check: if we don't land on the expected next state, abort (graph edge stale).
            try:
                expected = state_path[i + 1]
                reached_target = (cur_sig == target_sig) or (self._family_id(str(cur_sig)) == self._family_id(str(target_sig)))
                expected_ok = bool(expected and (cur_sig == expected or self._family_id(str(cur_sig)) == self._family_id(str(expected))))
                if reached_target:
                    return True, snap
                if expected and (not expected_ok):
                    self._log_event(
                        "graph_nav_diverged",
                        sig=cur_sig,
                        step_index=i + 1,
                        expected_sig=str(expected),
                        got_sig=str(cur_sig),
                    )
                    return False, snap
            except Exception:
                pass

            if (cur_sig == target_sig) or (self._family_id(str(cur_sig)) == self._family_id(str(target_sig))):
                return True, snap

        return ((cur_sig == target_sig) or (self._family_id(str(cur_sig)) == self._family_id(str(target_sig)))), snap

    def _restart_and_replay(self, best_target: Optional[str], task: str, *, reason: str = "") -> None:
        """
        IPO:
          in : best_target (desired state_sig to return to), task
          out: restarts app; attempts best-effort replay; then resumes navigation wherever we land

        WHEN called:
          - after recovery fails
          - to reach a global frontier when DFS is exhausted locally

        WHICH states:
          - src for replay is the post-restart landing sig
          - dst is best_target (if provided and path exists)
        """
        if not self.target_package:
            logger.error("Restart requested but target_package not configured.")
            return

        now = time.time()
        if self._restart_budget_exceeded(now):
            self._log_event(
                "restart_budget_exceeded",
                sig=(best_target or ""),
                best_target=(best_target or ""),
                recent_restarts=len(self.restart_recent),
                window_s=float(self.budget.restart_budget_window_s),
                limit=int(self.budget.restart_budget_max),
            )
            # If we have a concrete target, try in-graph navigation before spending a restart.
            if best_target:
                snap0 = self._capture_and_process(timeout=8.0)
                if snap0:
                    start_sig = str(snap0.get("state_sig") or "")
                    reached, _snap1 = self._navigate_via_graph(start_sig=start_sig, start_snap=snap0, target_sig=best_target, task=task)
                    if reached:
                        self._log_event("restart_budget_avoided", sig=start_sig, target_sig=best_target)
                        return
            return

        self._prune_restart_recent(now)
        try:
            self.restart_recent.append(now)
        except Exception:
            pass

        restart_reason = (reason or "unspecified").strip()
        logger.warning(
            "APP restart triggered: reason=%s best_target=%s",
            restart_reason,
            (best_target[:8] if best_target else "None"),
        )
        self._log_event(
            "app_restart_triggered",
            sig=(best_target or ""),
            best_target=(best_target or ""),
            reason=restart_reason,
        )
        try:
            self.appium.force_stop(self.target_package)
            time.sleep(0.6)
            self.appium.ensure_foreground(self.target_package, self.target_activity)
            time.sleep(0.8)
        except Exception:
            logger.debug("Restart failed", exc_info=True)
            return

        snap = self._capture_and_process()
        if not snap:
            return

        cur = snap["state_sig"]
        self.restart_entry_sig = cur
        self._graph_record_observation(cur, meta={"post_restart": True}, snap=snap)

        if not best_target:
            self._schedule_state(cur, snap, task)
            return

        state_path, actions = self.graph.shortest_action_path(cur, best_target, max_depth=self.budget.replay_max_depth)
        if not actions:
            logger.warning("Replay path not found from post-restart sig=%s to target=%s. Resume here.", cur[:8], best_target[:8])
            self._log_event("replay_path_missing", sig=cur, target_sig=best_target)
            self._schedule_state(cur, snap, task)
            return
        self._log_event("replay_path", sig=cur, target_sig=best_target, steps=len(actions))

        reached = False
        for i, act in enumerate(actions):
            if not act or not act.get("actions"):
                logger.debug("Replay step %d had no parsable actions; abort replay.", i + 1)
                self._log_event("replay_diverged", sig=cur, step_index=i + 1, reason="no_parsable_actions")
                break

            logger.info("Replay %d/%d from sig=%s: %s", i + 1, len(actions), cur[:8], self._action_payload_key(act))
            self._log_event("replay_step", sig=cur, step_index=i + 1, action=act)

            ok, used_act = self._execute_action_payload(act, snap)

            snap2 = self._capture_and_process()
            if not snap2:
                self._log_event("replay_diverged", sig=cur, step_index=i + 1, reason="capture_failed")
                break
            new_sig = snap2["state_sig"]

            # Update replay verification stats for the planned edge (before any early abort).
            try:
                expected = state_path[i + 1] if (i + 1) < len(state_path) else None
                if expected:
                    ok_edge = bool(new_sig == expected) or (self._family_id(str(new_sig)) == self._family_id(str(expected)))
                    self.graph.record_edge_verification(state_path[i], expected, act, ok=ok_edge)
            except Exception:
                pass

            self._graph_record_transition(cur, new_sig, used_act, src_snap=snap, dst_snap=snap2)
            self._enter_state(from_sig=cur, to_sig=new_sig, via_action=self._action_payload_key(used_act))

            snap = snap2
            cur = new_sig

            self._schedule_state(cur, snap, task)
            self._drain_futures()

            if cur == best_target or (self._family_id(str(cur)) == self._family_id(str(best_target))):
                reached = True
                self._log_event("replay_reached", sig=cur, target_sig=best_target)
                break

            # Overlay during replay: attempt resolve; if fails, stop replay
            nav = self.nav_cache.get(cur)
            if nav and self._overlay_kind_value(nav) == OverlayKind.DISMISS.value:
                if not self._dismiss_overlay_with_nav(cur, snap, nav, task):
                    self._log_event("replay_diverged", sig=cur, step_index=i + 1, reason="overlay_unresolved")
                    break

            if not ok:
                self._log_event("replay_diverged", sig=cur, step_index=i + 1, reason="action_failed")
                break

        if not reached:
            logger.info("Replay did not reach target. Resume navigation at current sig=%s", cur[:8])
            self._log_event("replay_incomplete", sig=cur, target_sig=best_target)
            self._schedule_state(cur, snap, task)

    # ---------------------------
    # Action execution (Flutter-robust)
    # ---------------------------

    @staticmethod
    def _normalize_str(v: Any) -> str:
        try:
            return str(v or "").strip()
        except Exception:
            return ""

    def _find_navigate_up_element(self, vid_map: Dict[int, Any]) -> Optional[int]:
        """
        Find an explicit "Navigate up" affordance in the current vid_map.
        Prefer it over system BACK to avoid leaving the app (especially at Settings root).
        """
        best: Optional[Tuple[float, int]] = None
        for eid, node in (vid_map or {}).items():
            try:
                cd = self._normalize_str(node.get("content_desc")).lower()
                txt = self._normalize_str(node.get("text")).lower()
                if ("navigate up" not in cd) and ("navigate up" not in txt):
                    continue
                f = BaseUI.get_frame(node)
                x = float(f.get("x", 0))
                y = float(f.get("y", 0))
                w = float(f.get("width", 0))
                h = float(f.get("height", 0))
                if w <= 1 or h <= 1:
                    continue
                # Prefer top-left-ish, small-ish buttons for determinism.
                score = (y + 0.6 * x) + 0.02 * (w * h)
                cand = (score, int(eid))
                if best is None or cand[0] < best[0]:
                    best = cand
            except Exception:
                continue
        return best[1] if best else None

    def _is_settings_root_state(self, vid_map: Dict[int, Any]) -> bool:
        """
        Heuristic: detect Settings "root/top" where system BACK would exit to launcher.

        This is intentionally conservative and only applies when the target app is Settings.
        """
        if str(self.target_package or "") != "com.android.settings":
            return False
        try:
            pkg = self.appium.foreground_package()
        except Exception:
            pkg = ""
        if pkg != "com.android.settings":
            return False
        try:
            act = self._normalize_str(self.appium.foreground_activity())
        except Exception:
            act = ""
        if act.startswith("."):
            act = "com.android.settings" + act
        act_l = act.lower()
        if "settingshomepageactivity" in act_l:
            return True
        if act_l.endswith(".settings"):
            return True
        # Fallback: view-hierarchy anchor present on homepage containers (best-effort).
        try:
            for node in (vid_map or {}).values():
                rid = self._normalize_str(node.get("resource_id")).lower()
                if "settings_homepage_container" in rid:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _xpath_literal(s: str) -> str:
        """
        Build an XPath string literal that safely handles quotes.
        """
        if "'" not in s:
            return f"'{s}'"
        if '"' not in s:
            return f'"{s}"'
        parts = s.split("'")
        # concat('foo', "'", 'bar', "'", 'baz')
        inner = ", \"'\", ".join([f"'{p}'" for p in parts])
        return f"concat({inner})"

    def _try_click_stable(self, node: Dict[str, Any]) -> bool:
        """
        Prefer stable selectors (resource-id/content-desc/text) over coordinate taps.
        Returns True if an element was clicked via selector.
        """
        rid = self._normalize_str(node.get("resource_id"))
        if rid:
            try:
                self.appium.click("id", rid)
                return True
            except Exception:
                pass

        cdesc = self._normalize_str(node.get("content_desc"))
        if cdesc:
            try:
                self.appium.click("accessibility id", cdesc)
                return True
            except Exception:
                pass

        txt = self._normalize_str(node.get("text"))
        if txt:
            # XPath is slower but widely supported; keep as a last resort.
            try:
                self.appium.click("xpath", f"//*[@text={self._xpath_literal(txt)}]")
                return True
            except Exception:
                pass

        return False

    def _execute_action_sequence(self, steps: List[ActionStep], snap: Dict[str, Any]) -> bool:
        """
        Execute an ordered list of ActionSteps. Updates action_count/history per step.
        NOTE: We intentionally do NOT recapture between steps to keep element_ids aligned
        with the original snapshot (ids are snapshot-scoped).
        """
        if not steps:
            return False

        cur_snap = snap
        vid_map = cur_snap["vid_map"]

        for st in steps:
            ok = self._execute_action(st, vid_map, cur_snap.get("state_sig"))
            if not ok:
                return False
            self.action_count += 1
            self.history.append(self._action_key(st))
            time.sleep(self.budget.post_action_settle_s)

        return True

    def _visual_click_bbox(self, step: ActionStep, png_w: int, png_h: int) -> Optional[List[int]]:
        raw_bbox = getattr(step, "bbox", None) or None
        bbox: Optional[List[int]] = None
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox[:4]]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                bbox = [x1, y1, x2, y2]
            except Exception:
                bbox = None

        x = getattr(step, "x", None)
        y = getattr(step, "y", None)
        if bbox is None and x is not None and y is not None:
            try:
                cx = int(round(float(x)))
                cy = int(round(float(y)))
                r = 72
                bbox = [cx - r, cy - r, cx + r, cy + r]
            except Exception:
                bbox = None

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(png_w) - 1, x1))
        x2 = max(0, min(int(png_w), x2))
        y1 = max(0, min(int(png_h) - 1, y1))
        y2 = max(0, min(int(png_h), y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        # Keep this as a local, bounded fallback. A near-full-screen bbox is too close to random clicking.
        if (x2 - x1) * (y2 - y1) > 0.35 * max(1, int(png_w) * int(png_h)):
            return None
        return [x1, y1, x2, y2]

    def _visual_probe_points(self, step: ActionStep, bbox: List[int]) -> List[Tuple[int, int]]:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        points: List[Tuple[int, int]] = []

        def add(px: float, py: float) -> None:
            x = int(round(max(x1 + 1, min(x2 - 1, px))))
            y = int(round(max(y1 + 1, min(y2 - 1, py))))
            p = (x, y)
            if p not in points:
                points.append(p)

        x = getattr(step, "x", None)
        y = getattr(step, "y", None)
        if x is not None and y is not None:
            try:
                add(float(x), float(y))
            except Exception:
                pass

        reasoning = f"{getattr(step, 'reasoning', '') or ''} {getattr(step, 'text', '') or ''}".lower()
        top_right_hint = any(s in reasoning for s in ("top right", "upper right", "right upper", "close", "dismiss", "叉", "关闭", "右上")) or bool(re.search(r"\b[x×]\b", reasoning))
        if top_right_hint:
            add(x2 - min(40.0, w * 0.12), y1 + min(40.0, h * 0.12))
            add(x2 - min(18.0, w * 0.06), y1 + min(18.0, h * 0.06))
        else:
            add(x1 + w * 0.5, y1 + h * 0.5)

        grid = int(getattr(step, "probe_grid", 3) or 3)
        grid = max(1, min(5, grid))
        cells: List[Tuple[float, float]] = []
        for row in range(grid):
            for col in range(grid):
                cells.append((x1 + (col + 0.5) * w / grid, y1 + (row + 0.5) * h / grid))
        if top_right_hint:
            cells.sort(key=lambda p: (p[0] - x2) ** 2 + (p[1] - y1) ** 2)
        else:
            cx, cy = x1 + w * 0.5, y1 + h * 0.5
            cells.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        for px, py in cells:
            add(px, py)
        return points[: max(1, int(self.budget.visual_probe_max_taps))]

    def _map_screenshot_point_to_tap(self, x: int, y: int, screenshot_w: int, screenshot_h: int) -> Tuple[int, int]:
        try:
            driver = getattr(self.appium, "driver", None)
            ws = driver.get_window_size() if driver else {}
            win_w = int(ws.get("width") or 0)
            win_h = int(ws.get("height") or 0)
            if win_w > 0 and win_h > 0 and screenshot_w > 0 and screenshot_h > 0:
                return int(round(float(x) * win_w / screenshot_w)), int(round(float(y) * win_h / screenshot_h))
        except Exception:
            pass
        return int(x), int(y)

    def _roi_delta(self, before_png: bytes, after_png: bytes, bbox: List[int]) -> float:
        try:
            with Image.open(io.BytesIO(before_png)).convert("RGB") as before_img:
                with Image.open(io.BytesIO(after_png)).convert("RGB") as after_img:
                    x1, y1, x2, y2 = bbox
                    x2 = min(x2, before_img.width, after_img.width)
                    y2 = min(y2, before_img.height, after_img.height)
                    x1 = max(0, min(x1, x2 - 1))
                    y1 = max(0, min(y1, y2 - 1))
                    a = before_img.crop((x1, y1, x2, y2)).resize((48, 48))
                    b = after_img.crop((x1, y1, x2, y2)).resize((48, 48))
                    stat = ImageStat.Stat(ImageChops.difference(a, b))
                    return float(sum(stat.mean) / (len(stat.mean) * 255.0))
        except Exception:
            return 0.0

    def _execute_visual_probe_click(self, step: ActionStep, sig_for_trace: str, action_sig: Dict[str, Any]) -> bool:
        """
        Bounded screenshot-only click fallback for targets visible in the screenshot but absent from vid_map.
        It tries a few points inside a model-provided bbox and uses a cheap ROI image delta between taps.
        Full authoritative capture still happens once in the caller after this returns.
        """
        try:
            before_png = self.appium.screenshot_png_once()
            if not before_png:
                self.last_action_failure = {"reason": "visual_probe_no_screenshot", "action": action_sig}
                self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "visual_probe_no_screenshot"})
                return False
            with Image.open(io.BytesIO(before_png)) as img:
                png_w, png_h = int(img.width), int(img.height)

            bbox = self._visual_click_bbox(step, png_w, png_h)
            if not bbox:
                self.last_action_failure = {"reason": "visual_probe_invalid_bbox", "action": action_sig}
                self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "visual_probe_invalid_bbox"})
                return False

            points = self._visual_probe_points(step, bbox)
            if not points:
                self.last_action_failure = {"reason": "visual_probe_no_points", "action": action_sig}
                self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "visual_probe_no_points", "bbox": bbox})
                return False

            cur_png = before_png
            threshold = float(self.budget.visual_probe_roi_delta_threshold)
            for idx, (sx, sy) in enumerate(points, start=1):
                tx, ty = self._map_screenshot_point_to_tap(sx, sy, png_w, png_h)
                self._log_event("visual_probe_tap", sig=sig_for_trace, index=idx, screenshot_x=sx, screenshot_y=sy, tap_x=tx, tap_y=ty, bbox=bbox)
                self.appium.tap(tx, ty)
                time.sleep(float(self.budget.visual_probe_settle_s))
                after_png = self.appium.screenshot_png_once()
                delta = self._roi_delta(cur_png, after_png, bbox) if after_png else 0.0
                self._log_event("visual_probe_delta", sig=sig_for_trace, index=idx, delta=delta, threshold=threshold)
                if delta >= threshold:
                    self._emit_action(
                        sig_for_trace,
                        action_sig,
                        "after",
                        {"success": True, "via": "visual_probe", "taps": idx, "bbox": bbox, "delta": delta},
                    )
                    return True
                if after_png:
                    cur_png = after_png

            self.last_action_failure = {"reason": "visual_probe_no_change", "action": action_sig, "bbox": bbox, "points": points}
            self._emit_action(
                sig_for_trace,
                action_sig,
                "after",
                {"success": False, "reason": "visual_probe_no_change", "bbox": bbox, "points": points},
            )
            return False
        except Exception:
            logger.debug("visual probe click failed", exc_info=True)
            self.last_action_failure = {"reason": "visual_probe_exception", "action": action_sig}
            self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "visual_probe_exception"})
            return False

    def _execute_action(self, step: ActionStep, vid_map: Dict[int, Any], cur_sig: Optional[str] = None) -> bool:
        """
        IPO:
          in : step (ActionStep), vid_map for the CURRENT snapshot
          out: True if action executed (not necessarily succeeded in UI), False if cannot execute

        WHEN called:
          - for every action step (probe/forward/overlay/recovery/replay)

        IMPORTANT:
          - element_id validity is snapshot-scoped; always use vid_map from the snapshot you are acting on.
        """
        try:
            sig_for_trace = cur_sig or (self.recent_states[-1] if self.recent_states else "")
            action_sig = self._action_signature(step)
            origin = self._infer_action_origin()
            common_extra = {
                "reasoning": step.reasoning or "",
                "origin": origin,
                "action_key": self._action_key(step),
            }
            self._emit_action(sig_for_trace, action_sig, "before", {"has_element": step.element_id in vid_map, **common_extra})
            logger.debug("Executing action: %s on element %s", self._action_key(step), str(vid_map.get(step.element_id or -1, {}) | {"subviews": {}}))
            if self.pause and not getattr(self.callbacks, "manages_action_pause", False):
                input("Paused before action execution. Press Enter to continue...")

            self.last_action_failure = None

            # Foreground-package gate: never interact (click/input/back) when we're in an unexpected package.
            if step.action in (ActionType.CLICK, ActionType.INPUT, ActionType.BACK):
                ok_pkg, pkg = self._foreground_is_allowed()
                if not ok_pkg:
                    self.foreground_mismatch_count += 1
                    self.foreground_mismatch_streak += 1
                    self._log_event(
                        "foreground_mismatch_during_action",
                        sig=sig_for_trace,
                        action=self._action_signature(step),
                        foreground_package=pkg,
                        streak=self.foreground_mismatch_streak,
                        total=self.foreground_mismatch_count,
                    )
                    self.last_action_failure = {"reason": "foreground_mismatch", "foreground_package": pkg, "action": self._action_signature(step)}
                    try:
                        if self.target_package:
                            self.appium.ensure_foreground(self.target_package, self.target_activity)
                            self.foreground_recoveries += 1
                    except Exception:
                        pass
                    self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "foreground_mismatch", **common_extra})
                    return False

            if step.action == ActionType.CLICK:
                node = vid_map.get(step.element_id or -1)
                if not node:
                    if step.element_id is None and (getattr(step, "bbox", None) or (getattr(step, "x", None) is not None and getattr(step, "y", None) is not None)):
                        return self._execute_visual_probe_click(step, sig_for_trace, action_sig)
                    logger.debug("CLICK failed: element_id=%s not in vid_map", step.element_id)
                    self.last_action_failure = {"reason": "missing_element", "action": self._action_signature(step)}
                    self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "missing_element", **common_extra})
                    return False
                try:
                    if self._try_click_stable(node):
                        self._emit_action(sig_for_trace, action_sig, "after", {"success": True, "via": "stable_selector", **common_extra})
                        return True
                except Exception:
                    # Fall back to coordinate tap below.
                    pass
                f = BaseUI.get_frame(node)
                w = float(f.get("width", 0))
                h = float(f.get("height", 0))
                if w <= 1 or h <= 1:
                    logger.debug("CLICK failed: degenerate bounds id=%s w=%.1f h=%.1f", step.element_id, w, h)
                    self.last_action_failure = {"reason": "degenerate_bounds", "action": self._action_signature(step)}
                    self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "degenerate_bounds", **common_extra})
                    return False
                c = BaseUI.get_center(node)
                self.appium.tap(c["x"], c["y"])
                self._emit_action(sig_for_trace, action_sig, "after", {"success": True, **common_extra})
                return True

            if step.action == ActionType.BACK:
                # Prefer explicit "Navigate up" affordance when present (safer than system BACK).
                nav_up_id = self._find_navigate_up_element(vid_map)
                if nav_up_id is not None and nav_up_id in vid_map:
                    node = vid_map.get(nav_up_id)
                    if node:
                        c = BaseUI.get_center(node)
                        self.appium.tap(c["x"], c["y"])
                        self._emit_action(sig_for_trace, action_sig, "after", {"success": True, "via": "navigate_up", **common_extra})
                        return True

                # Settings root safety: never press system BACK from the top activity (would exit to launcher),
                # unless we're clearly dismissing an overlay.
                if self._is_settings_root_state(vid_map):
                    try:
                        nav = self.nav_cache.get(sig_for_trace)
                        ok_overlay = nav and (self._overlay_kind_value(nav) in (OverlayKind.DISMISS.value, OverlayKind.LOADING.value))
                    except Exception:
                        ok_overlay = False
                    if not ok_overlay:
                        self.last_action_failure = {"reason": "back_blocked_settings_root", "action": self._action_signature(step)}
                        self._log_event("back_blocked_settings_root", sig=sig_for_trace, action=self._action_signature(step))
                        self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "back_blocked_settings_root", **common_extra})
                        return False

                self.appium.back()

                # Post-BACK foreground guard: BACK may exit target app (launcher/external app).
                ok_pkg_after, pkg_after = self._foreground_is_allowed()
                if not ok_pkg_after:
                    self.foreground_mismatch_count += 1
                    self.foreground_mismatch_streak += 1
                    self._log_event(
                        "foreground_mismatch_during_action",
                        sig=sig_for_trace,
                        action=self._action_signature(step),
                        foreground_package=pkg_after,
                        streak=self.foreground_mismatch_streak,
                        total=self.foreground_mismatch_count,
                        phase="post_back",
                    )
                    self.last_action_failure = {
                        "reason": "post_back_foreground_mismatch",
                        "foreground_package": pkg_after,
                        "action": self._action_signature(step),
                    }
                    try:
                        if self.target_package:
                            self.appium.ensure_foreground(self.target_package, self.target_activity)
                            self.foreground_recoveries += 1
                    except Exception:
                        pass
                    self._emit_action(
                        sig_for_trace,
                        action_sig,
                        "after",
                        {"success": False, "reason": "post_back_foreground_mismatch", "foreground_package": pkg_after, **common_extra},
                    )
                    return False

                self._emit_action(sig_for_trace, action_sig, "after", {"success": True, **common_extra})
                return True

            if step.action == ActionType.WAIT:
                seconds = 0.7
                if step.text is not None:
                    try:
                        seconds = float(step.text)
                    except (TypeError, ValueError):
                        seconds = 0.7
                time.sleep(seconds)
                self._emit_action(sig_for_trace, action_sig, "after", {"success": True, **common_extra})
                return True

            if step.action == ActionType.INPUT:
                node = vid_map.get(step.element_id or -1)
                if node:
                    try:
                        if not self._try_click_stable(node):
                            c = BaseUI.get_center(node)
                            self.appium.tap(c["x"], c["y"])
                    except Exception:
                        c = BaseUI.get_center(node)
                        self.appium.tap(c["x"], c["y"])
                    time.sleep(0.2)
                self.appium.type_text(step.text or "")
                self._emit_action(sig_for_trace, action_sig, "after", {"success": True, **common_extra})
                return True

            if step.action == ActionType.RESTART:
                if self.target_package:
                    self.appium.force_stop(self.target_package)
                    time.sleep(0.6)
                    self.appium.ensure_foreground(self.target_package, self.target_activity)
                    time.sleep(0.8)
                    self._emit_action(sig_for_trace, action_sig, "after", {"success": True, **common_extra})
                    return True
                self._emit_action(sig_for_trace, action_sig, "after", {"success": False, "reason": "no_target_package", **common_extra})
                return False

            if step.action in (ActionType.NONE, ActionType.COMPLETE):
                self._emit_action(sig_for_trace, action_sig, "after", {"success": True, **common_extra})
                return True

            self._emit_action(sig_for_trace, action_sig, "after", {"success": False, **common_extra})
            return False
        except Exception:
            logger.debug("execute_action error", exc_info=True)
            sig_for_trace = cur_sig or (self.recent_states[-1] if self.recent_states else "")
            self._emit_action(
                sig_for_trace,
                self._action_signature(step),
                "after",
                {
                    "success": False,
                    "exception": True,
                    "reasoning": step.reasoning or "",
                    "origin": locals().get("origin", "unknown"),
                    "action_key": self._action_key(step),
                },
            )
            return False

    # ---------------------------
    # Force-replan signaling
    # ---------------------------

    def _set_forced_replan(self, sig: str, snap: Dict[str, Any], *, has_edge: bool = False) -> None:
        self._force_replan_sig = sig
        self._force_replan_snap = snap
        self._force_replan_has_edge = bool(has_edge)

    def _consume_forced_replan(self) -> bool:
        return self._force_replan_sig is not None and self._force_replan_snap is not None

    def _clear_forced_replan(self) -> None:
        self._force_replan_sig = None
        self._force_replan_snap = None
        self._force_replan_has_edge = False

    # ---------------------------
    # Probe bookkeeping
    # ---------------------------

    def _visual_action_suffix(self, step: ActionStep) -> str:
        if getattr(step, "element_id", None) is not None:
            return ""
        bbox = getattr(step, "bbox", None) or None
        x = getattr(step, "x", None)
        y = getattr(step, "y", None)
        if not bbox and (x is None or y is None):
            return ""
        return f":xy={x},{y}:bbox={bbox}:grid={getattr(step, 'probe_grid', '')}"

    def _action_key(self, step: ActionStep) -> str:
        return f"{step.action.value}:{step.element_id}:{step.text or ''}{self._visual_action_suffix(step)}"

    def _actions_key(self, steps: List[ActionStep]) -> str:
        if not steps:
            return "None:None:"
        parts: List[str] = []
        for s in steps:
            action_val = s.action.value if hasattr(s.action, "value") else str(s.action)
            parts.append(f"{action_val}:{s.element_id}:{s.text or ''}{self._visual_action_suffix(s)}")
        return "||".join(parts) or "None:None:"

    def _candidate_key(self, cand: Any) -> str:
        if isinstance(cand, str):
            return cand
        if isinstance(cand, ActionCandidate):
            return self._actions_key(cand.actions or [])
        if isinstance(cand, ActionStep):
            return self._actions_key([cand])
        if isinstance(cand, list):
            return self._actions_key(cand)
        return str(cand)

    def _action_signature(self, step: ActionStep, *, vid_map: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"action": step.action.value, "element_id": step.element_id, "text": step.text}
        try:
            if step.element_id is None:
                if getattr(step, "x", None) is not None:
                    out["x"] = getattr(step, "x", None)
                if getattr(step, "y", None) is not None:
                    out["y"] = getattr(step, "y", None)
                if getattr(step, "bbox", None):
                    out["bbox"] = getattr(step, "bbox", None)
                if getattr(step, "bbox", None):
                    out["probe_grid"] = getattr(step, "probe_grid", None)
        except Exception:
            pass
        try:
            if vid_map is not None and step.element_id is not None and step.element_id in vid_map:
                if step.action in (ActionType.CLICK, ActionType.INPUT):
                    out["fingerprint"] = self._node_fingerprint(vid_map[int(step.element_id)])
        except Exception:
            pass
        return out

    def _actions_signature(self, steps: List[ActionStep], *, vid_map: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
        """
        Signature for multi-step candidates.
        """
        if not steps:
            return {"actions": []}
        return {"actions": [self._action_signature(s, vid_map=vid_map) for s in steps]}

    def _family_id(self, sig: str) -> str:
        """
        Structural "family" id for a state signature. Used for per-page bookkeeping when
        state_sig changes due to dynamic text while structure stays stable.
        """
        if not sig:
            return ""
        return self.sig_to_family.get(sig, sig)

    def _mark_attempted(self, sig: str, cand: Any) -> None:
        fam = self._family_id(sig)
        self.attempted_actions.setdefault(fam, set()).add(self._candidate_key(cand))

    def _already_attempted(self, sig: str, cand: Any) -> bool:
        fam = self._family_id(sig)
        return self._candidate_key(cand) in self.attempted_actions.get(fam, set())

    def _mark_explored(self, sig: str, cand: Any) -> None:
        fam = self._family_id(sig)
        self.explored_actions.setdefault(fam, set()).add(self._candidate_key(cand))

    def _is_explored(self, sig: str, cand: Any) -> bool:
        fam = self._family_id(sig)
        return self._candidate_key(cand) in self.explored_actions.get(fam, set())

    # ---------------------------
    # Stop condition
    # ---------------------------

    def _should_recover_stuck(self) -> bool:
        age_strong = time.time() - float(self.last_strong_progress_ts or 0.0)
        if age_strong >= self.budget.stuck_time_s:
            return True
        if self.no_progress_loops >= self.budget.stuck_loops_limit and self.nav_ready_once:
            return True
        return False

    def _stop_condition_reason(self, start: float) -> str:
        if (time.time() - start) >= self.budget.time_budget_s:
            return "time_budget_reached"
        # Hard stop if we haven't made strong progress for too long (prevents infinite recover loops).
        if float(self.budget.strong_stall_stop_s or 0.0) > 0.0:
            if (time.time() - float(self.last_strong_progress_ts or 0.0)) >= float(self.budget.strong_stall_stop_s):
                return "strong_stall_timeout"
        if self.action_count >= self.budget.max_actions:
            return "max_actions_reached"
        if self.no_new_state_count >= self.budget.saturation_limit:
            return "state_saturation_reached"
        return ""
