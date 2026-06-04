# gpt_cls.py (patched; key change: compact navigation output for debugging)

"""
gpt_cls.py

KEY CHANGE (requested):
- Navigation output is compact for debug runs.
- Candidate actions are exploration-only.
- Page return controls are emitted separately as NavigationProposal.page_return_actions.

WHAT WE ADD:
- ActionCandidate wrapper with actions + score only.
- NavigationProposal.candidate_actions is List[ActionCandidate].
- NavigationProposal.page_return_actions stores visible page-level return/exit controls.

WORKFLOW CONTRACT:
- This compact schema is intended for LLM debugging scripts first.
- Main workflow integration requires separate adaptation if it still expects legacy return fields.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


from utils import time_consumed, token_record  # do NOT modify user's utils.py


R = TypeVar("R", bound=BaseModel)

# Default OpenAI-compatible API endpoint. Override with OPENAI_BASE_URL when
# moving the project between machines/providers.
DEFAULT_OPENAI_BASE_URL = "https://api.uniapi.io/v1"



class ActionType(str, Enum):
    CLICK = "click"
    INPUT = "input"
    WAIT = "wait"
    BACK = "back"
    RESTART = "restart"
    COMPLETE = "complete"
    NONE = "none"

class OverlayKind(str, Enum):
    NONE = "none"
    DISMISS = "dismiss"      # permission/rate/cookie/etc
    WORKFLOW = "workflow"    # login/search/filter/form/otp
    LOADING = "loading"      # spinner / transition


class PageReturnStatus(str, Enum):
    HAS_VISIBLE_RETURN = "has_visible_return"
    NO_RETURN_NEEDED = "no_return_needed"
    NEEDS_RETURN_BUT_NO_VISIBLE_CONTROL = "needs_return_but_no_visible_control"


class UIElementType(str, Enum):
    BUTTON = "Button"
    TEXT_BUTTON = "TextButton"
    ICON_BUTTON = "IconButton"
    INPUT_TEXT = "InputText"
    TOGGLE = "Toggle"
    CHECKBOX = "Checkbox"
    RADIO = "Radio"
    TAB = "Tab"
    LIST_ITEM = "ListItem"
    TEXT_ONLY = "TextOnly"
    ICON_ONLY = "IconOnly"
    DIALOG = "Dialog"
    OTHER_INTERACTABLE = "OtherInteractable"
    UNKNOWN = "Unknown"


class UIElement(BaseModel):
    id: int = Field(..., description="Stable id matching UI tree after post-processing")
    ui_type: UIElementType = Field(..., description="Element category")
    description: str = Field("", description="Inferred purpose or label (<=15 words)")
    text: str = Field("", description="Visible text/value if any")
    clickability: float = Field(0.0, description="0-1 likelihood the element is actionable")
    location: str = Field("", description="Relative position (e.g., top-left, header, nav bar)")


class TagSignal(BaseModel):
    tag: str = Field(..., description="Short taxonomy tag (snake_case preferred)")
    weight: float = Field(0.0, ge=0.0, le=1.0, description="0-1 strength of association")


class UIView(BaseModel):
    """
    A concise, model-generated interpretation of the current screen.
    Keep it SMALL (the prompt enforces item caps).
    """
    description: str = Field("", description="One-paragraph summary of layout and purpose")
    feedback_message: str = Field("", description="Visible status/toast/error text if any")
    is_alert_topmost: bool = Field(False, description="True if a blocking dialog/overlay is on top")
    hint_elements: List[UIElement] = Field(default_factory=list, description="Informative, mostly non-interactive elements")
    action_elements: List[UIElement] = Field(default_factory=list, description="Interactive elements")
    match_rate: float = Field(1.0, description="0-1 match confidence between UI tree + screenshot")

    @property
    def elements(self) -> List[UIElement]:
        return (self.hint_elements or []) + (self.action_elements or [])


class ActionStep(BaseModel):
    action: ActionType = Field(..., description="Concrete action type")
    element_id: Optional[int] = Field(None, description="UI element id for click/input/back-like visible controls")
    text: Optional[str] = Field(None, description="Text to input when action==input")
    anchor_frame: Optional[List[int]] = Field(None, description="Frozen [x,y,width,height] frame from the snapshot where this action was planned")
    anchor_center: Optional[List[float]] = Field(None, description="Frozen [x,y] center from the snapshot where this action was planned")
    anchor_label: str = Field("", description="Frozen short label/text/icon hint for the planned element")
    anchor_class: str = Field("", description="Frozen UI class/source hint for the planned element")
    priority: int = Field(0, description="Higher executes earlier when same group")
    reasoning: str = Field("", description="Short reason, preferably <=6 words. No chain-of-thought.")


class ActionCandidate(BaseModel):
    """
    LLM navigation candidate for exploration only.

    Input:
    - Ordered action steps that may reveal new UI/evidence.

    Output:
    - A scored candidate that should not include page return/close/back controls.
    """

    actions: List[ActionStep] = Field(
        ...,
        description="Ordered exploration steps for this candidate. Must not be empty. Do not include page-return controls.",
        min_items=1,
    )
    score: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="LLM priority score for this exploration candidate in [-1, 1].",
    )
    action_role: Literal["continue_current_task", "start_child_task"] = Field(
        "continue_current_task",
        description="Whether this action continues the active task or starts a proposed child task.",
    )
    starts_task_type: str = Field(
        "",
        description="When action_role is start_child_task, the coarse child task type this action enters.",
    )
    starts_task_depth: str = Field(
        "",
        description="When action_role is start_child_task, the proposed child task exploration depth.",
    )


class NavigationProposal(BaseModel):
    """
    LLM1 output: compact overlay handling, exploration candidates, and page-level return plan.
    """
    state_sig: str = Field(..., description="Echo input state_sig for staleness/debug")

    page_summary: str = Field(..., description="One-line summary of current page")

    overlay_kind: OverlayKind = Field(OverlayKind.NONE, description="Overlay type classification")
    overlay_reason: str = Field("", description="Short reason for overlay_kind")
    overlay_dismiss_actions: List[ActionStep] = Field(
        default_factory=list,
        description="Actions to dismiss blocking overlays; only for overlay_kind=dismiss.",
    )

    candidate_actions: List[ActionCandidate] = Field(
        default_factory=list,
        description=(
            "Exploration candidates only. Do not include back, close, up, or already-visited "
            "tab-switch controls unless they are likely to reveal new content."
        ),
    )

    page_return_actions: List[ActionStep] = Field(
        default_factory=list,
        description=(
            "Visible page-level return/exit actions to use after candidate_actions are completed. "
            "Examples: visible back arrow, close button, or tab switch to a previous/root page. "
            "Do not duplicate candidate_actions. If no visible return control exists, keep empty."
        ),
    )
    page_return_status: PageReturnStatus = Field(
        PageReturnStatus.NO_RETURN_NEEDED,
        description=(
            "Page-level return status. Use has_visible_return when page_return_actions contains usable visible controls; "
            "no_return_needed for home/loading/splash/startup-only pages; "
            "needs_return_but_no_visible_control when returning is useful but no reliable visible return control is found."
        ),
    )
    page_return_reason: str = Field("", description="Short reason for page_return_status.")


class RecoveryProposal(BaseModel):
    """
    LLM3 output: recover to stable exploration state with custom actions allowed.
    """
    state_sig: str = Field(..., description="Echo input state_sig for staleness/debug")
    ui_view: UIView = Field(..., description="UI interpretation for recovery context")

    page_summary: str = Field(..., description="What screen looks like and why stuck")
    target_hint: str = Field("", description="Frontier/goal hint")
    overlay_kind: OverlayKind = Field(OverlayKind.NONE, description="Overlay type classification")
    overlay_reason: str = Field("", description="Why overlay_kind was chosen")
    candidate_actions: List[ActionStep] = Field(default_factory=list, description="Up to 5 steps to escape")
    why: str = Field("", description="Short explanation of recovery strategy (no hidden CoT)")


class ProposedUpdate(BaseModel):
    question_id: str
    new_answer: Union[str, List[str], None] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(default_factory=list)
    note: str = ""


class QuestionnaireUpdate(BaseModel):
    detected_signals: List[str] = Field(
        default_factory=list,
        description="Evidence strings observed on this screen that may inform answers (conservative, short).",
    )
    proposed_updates: List[ProposedUpdate] = Field(
        default_factory=list,
        description="Minimal set of questionnaire changes justified by CURRENT SCREEN evidence.",
    )
    open_gaps_recommended: List[str] = Field(
        default_factory=list,
        description="IDs of gaps that should be prioritized next based on hints from this screen.",
    )
    conflicts: List[str] = Field(
        default_factory=list,
        description="Detected inconsistencies between current screen evidence and existing answers (string IDs).",
    )


class BlockFillResult(BaseModel):
    block_id: str = Field(..., description="Block id from questionnaire_blocks.json")
    detected_signals: List[str] = Field(default_factory=list, description="Short evidence strings observed for this block")
    proposed_updates: List[ProposedUpdate] = Field(default_factory=list, description="Question updates for this block only")
    conflicts: List[str] = Field(default_factory=list, description="Optional conflict notes for this block")


class BlocksFillResult(BaseModel):
    state_sig: str = Field("", description="Echoed UI state signature")
    block_results: List[BlockFillResult] = Field(default_factory=list, description="One result per answered block")


class RelevantTopic(BaseModel):
    topic_id: str = Field(..., description="Stable topic_id from topic_tree_shallow")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="0-1 confidence this topic is answerable from this UI")
    rationale: str = Field("", description="Why this topic matches this screen (<=1 sentence)")
    expected_evidence: List[str] = Field(default_factory=list, description="What evidence to look for (short bullets)")


class TopicRouteResult(BaseModel):
    """
    LLM2-1 output: choose which questionnaire topics are relevant on this screen.
    """
    state_sig: str = Field(..., description="Echo input state_sig for staleness/debug")
    relevant_topics: List[RelevantTopic] = Field(default_factory=list, description="Relevant topics with confidence")
    skip_reason: str = Field("", description="If no relevant topics, why (short)")
    followups: List[str] = Field(default_factory=list, description="Optional followups like 'open Settings > Privacy'")


class RouterQuestionAnswer(BaseModel):
    """
    One router-question update inferred from the current screen.

    Notes:
      - This is still a questionnaire question, so question_id must be an existing router question id.
      - new_answer follows the same single/multi conventions as QuestionnaireUpdate.
    """
    question_id: str = Field(..., description="Router question id from router_questions input")
    new_answer: Union[str, List[str], None] = Field(
        None,
        description="For single-select use ONE option id/value string; for multi-select use a list of option ids/values.",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="0-1 confidence for this router answer")
    rationale: str = Field("", description="Short justification from the current UI (<=1 sentence)")


class RouterResult(BaseModel):
    """
    New LLM2-1 output for the router-only execution view.
    """
    state_sig: str = Field(..., description="Echo input state_sig for staleness/debug")
    router_updates: List[RouterQuestionAnswer] = Field(
        default_factory=list,
        description="Router questions that can be answered from the current screen",
    )


class TaskDecision(BaseModel):
    """
    LLM decision about the active exploration task.

    Input:
    - Current UI, current task, and router findings.

    Output:
    - Whether the current task should end or return.

    Function:
    - Lets workflow keep task-stack state separate from navigation candidates.
    """

    current_task_done: bool = Field(False, description="True when the current task is complete")
    current_task_failed: bool = Field(False, description="True when the current task cannot be usefully advanced")
    should_return: bool = Field(False, description="True when workflow should try to return to the parent task context")
    reason: str = Field("", description="Short natural-language reason for this task decision")


class ProposedTask(BaseModel):
    """
    LLM-proposed child exploration task.

    Input:
    - Router/navigation analysis of the current UI.

    Output:
    - A child task plus the entry action that opens it.

    Function:
    - Represents valuable side branches such as policy, payment, settings, or account pages.
    """

    prompt: str = Field("", description="Natural-language child task prompt")
    task_type: str = Field("generic", description="Coarse task type such as explore_policy or explore_payment")
    priority: float = Field(0.5, ge=0.0, le=1.0, description="Task importance in [0, 1]; first version records it but does not sort by it")
    exploration_depth: Literal["shallow", "normal", "deep"] = Field(
        "normal",
        description="How deeply this task should be explored: shallow overview, normal key paths, or deep questionnaire-focused exploration.",
    )
    initial_steps: int = Field(4, ge=1, le=20, description="Initial action-step budget for this child task")
    entry_action: Optional[ActionStep] = Field(
        None,
        description="Action from the current UI that enters this task. Required for workflow execution.",
    )
    reason: str = Field("", description="Why this task is worth exploring")
    related_router_questions: List[str] = Field(default_factory=list, description="Router question ids related to this task")


class NavigationRouterResult(BaseModel):
    """
    Combined LLM output for navigation planning and router answering.
    """
    state_sig: str = Field(..., description="Echo input state_sig for staleness/debug")
    task_id: str = Field("", description="Echo current_task.task_id when provided")
    navigation: NavigationProposal = Field(..., description="LLM1-compatible navigation proposal")
    router: RouterResult = Field(..., description="LLM2-1-compatible router answer result")
    task_decision: TaskDecision = Field(default_factory=TaskDecision, description="Decision about the active task")
    proposed_tasks: List[ProposedTask] = Field(default_factory=list, description="Child tasks discovered on this UI")


class AppMetadataSummary(BaseModel):
    """
    One-shot summary generated from app-level metadata (store description/category/etc.).
    This output is intended to be reused as stable context in later LLM calls.
    """
    app_id: str = Field("", description="App package id (e.g., com.example.app)")
    app_intro: str = Field("", description="One-sentence app overview inferred from description fields")
    focus_hints: str = Field(
        "",
        description=(
            "Potential content focus hints for UI review, written as short phrases separated by semicolons. "
            "Leave empty when evidence is insufficient."
        ),
    )
    questionnaire_type: Literal["games", "social_apps", "others", ""] = Field(
        "",
        description="Questionnaire bucket inferred from genre fields. Empty when insufficient evidence.",
    )
    notes: str = Field(
        "",
        description="Brief reason when any output field is empty (e.g., missing source field, empty value, garbled text).",
    )


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model output")
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in: {text[:200]!r}")
    blob = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    return json.loads(blob)


def _b64_image_url(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def _compact_digest(ui_json: Dict[str, Any], limit: int = 220) -> Dict[str, Any]:
    """
    Compact UI for LLM. Keep fields stable and small.
    NOTE: ids must stay aligned with BaseUI post-processing so NavigationProposal can reference them.
    """
    out: List[Dict[str, Any]] = []

    def walk(n: Dict[str, Any], depth: int = 0, parent_id: Optional[int] = None):
        if len(out) >= limit:
            return
        f = n.get("absolute_frame") or n.get("frame") or {}
        label = n.get("text") or n.get("content_desc") or n.get("semantic_label") or n.get("ocr_text") or n.get("icon_label")
        out.append(
            {
                "id": n.get("id"),
                "parent_id": parent_id,
                "depth": depth,
                "label": (str(label)[:90] if label else None),
                "text": (str(n.get("text"))[:90] if n.get("text") else None),
                "content_desc": (str(n.get("content_desc"))[:90] if n.get("content_desc") else None),
                "resource_id": (str(n.get("resource_id"))[:90] if n.get("resource_id") else None),
                "semantic_label": (str(n.get("semantic_label"))[:90] if n.get("semantic_label") else None),
                "semantic_type": (str(n.get("semantic_type"))[:40] if n.get("semantic_type") else None),
                "ocr_text": (str(n.get("ocr_text"))[:90] if n.get("ocr_text") else None),
                "icon_label": (str(n.get("icon_label"))[:60] if n.get("icon_label") else None),
                "clickable": bool(n.get("clickable")),
                "enabled": bool(n.get("enabled", True)),
                "bounds": [
                    int(f.get("x", 0)),
                    int(f.get("y", 0)),
                    int(f.get("width", 0)),
                    int(f.get("height", 0)),
                ],
                "class": (str(n.get("class") or "")[:60] or None),
            }
        )
        for ch in (n.get("subviews") or []):
            walk(ch, depth + 1, n.get("id"))

    for r in (ui_json.get("elements") or []):
        walk(r, 0, None)

    return {"elements": out, "screenscale": ui_json.get("screenscale", 1.0)}


_NAV_SYSTEM = """You are LLM1 for an Android UI exploration agent.

GOAL:
- Produce a compact navigation plan for the current Android screen.
- Separate exploration actions from page-return actions.
- Prefer actions that can reveal new UI surfaces or questionnaire evidence.

INPUTS:
- state_sig: current UI signature for debug/staleness checks.
- task: current exploration objective.
- block_status: questionnaire block coverage context; use it only as a weak priority signal.
- app_intro/focus_hints: weak app-level priors; current-screen evidence has priority.
- history: recent action context; avoid obvious loops.
- ui_digest: compact UI tree. Action element_id values MUST come from ui_digest ids.
- xml_reliable: whether XML-derived structure is reliable. If false, trust screenshot evidence more than labels/tree semantics.
- screenshot: primary visual evidence.

OUTPUT (strict JSON matching NavigationProposal):
- state_sig
- page_summary
- overlay_kind
- overlay_reason
- overlay_dismiss_actions
- candidate_actions
- page_return_actions
- page_return_status
- page_return_reason

ACTION STEP RULES:
- action must be one of: click | input | wait | back | restart | complete | none.
- element_id must be an id from ui_digest when targeting a visible UI control.
- Do not output screenshot coordinates or bounding boxes.
- text is used only for input actions.
- reasoning should be short, preferably <=6 words.

OVERLAY RULES:
- overlay_kind must be one of: none | dismiss | workflow | loading.
- If overlay_kind=dismiss, you MUST provide overlay_dismiss_actions.
- If overlay_kind=dismiss, put ALL safe close/deny/not-now/OK/skip/continue-past-popup actions in overlay_dismiss_actions, ordered safest first.
- If overlay_kind=dismiss, candidate_actions MUST be empty because the screen is blocked until the overlay is gone.
- Do not choose fullscreen/root/container elements as dismiss targets unless that element is the only clearly tappable close/continue control.
- Prefer explicit X, Close, Cancel, No thanks, Not now, Later, Skip, Continue, Agree, OK, or Got it controls.
- If overlay_kind is not dismiss, keep overlay_dismiss_actions empty.

CANDIDATE ACTION RULES:
- candidate_actions are exploration actions only.
- If overlay_kind=dismiss, candidate_actions MUST be [].
- Do not include back, close, up, return, or already-visited tab-switch controls in candidate_actions unless they likely open genuinely new content.
- Do not include login/sign-in/social-login actions unless the task explicitly requires account login or no other useful non-auth exploration exists.
- Actions whose main effect is returning to a previous/visited page should be omitted or receive very low priority.
- Prefer visible primary actions, menus, tabs, settings, shop/purchase, profile/account, help/about, game start/level, rewards, or other evidence-bearing surfaces.
- Each candidate should have 1..3 actions and score in [-1, 1].

PAGE RETURN RULES:
- page_return_actions are page-level return/exit actions to use only after all candidate_actions on this page are completed.
- Typical page_return_actions: visible back arrow, close/X button, or tab switch back to a previous/root page.
- First set page_return_status:
  * has_visible_return: this page should be able to return and a visible return/close/up/tab-return control exists.
  * no_return_needed: this is home, loading, splash, startup-only, or otherwise should not return to a previous UI.
  * needs_return_but_no_visible_control: returning would be useful but no reliable visible return control is found.
- Always explain the status in page_return_reason.
- If page_return_status=has_visible_return, fill page_return_actions with the visible controls.
- If page_return_status=no_return_needed or needs_return_but_no_visible_control, keep page_return_actions=[].
- Do not duplicate any element_id/action already listed in candidate_actions or overlay_dismiss_actions.
- If no visible return/close/up/tab-return control exists, output page_return_actions=[]. Do not invent a return action.

CONSISTENCY RULES:
- Every referenced element_id must exist in ui_digest.
- Never put the same element_id in candidate_actions and page_return_actions.
- Keep uncertain actions omitted rather than guessed.
- Do not spam random clicks.
- Do not intentionally leave the app unless clearly necessary for questionnaire evidence.
"""
#prompt调整

_Q_SYSTEM = """You are LLM2 for questionnaire filling based on CURRENT SCREEN evidence.

PRIMARY GOAL:
- Update only the MOST relevant questionnaire parts using evidence on this screen.
- Use hierarchy to infer parent/child relations when confident.

ANTI FLIP-FLOP (CRITICAL):
- Never set "No" just because evidence isn't visible on this screen.
- Do NOT flip prior Yes->No or remove multi-selections unless explicit contradiction exists AND confidence >= 0.95.

OUTPUT (strict JSON matching QuestionnaireUpdate):
- proposed_updates: minimal, evidence-backed changes
- For single-select questions: new_answer should be ONE option id/value (string), not a list.
- For multi-select questions: new_answer should be a list of option ids/values.
- evidence_refs must include ui ids when possible, e.g. {"ui_id": 12, "label": "In-app purchases"}
"""


_TOPIC_ROUTE_SYSTEM = """You are LLM2-1 (Topic Router) for an Android UI exploration agent.

GOAL:
- Decide which questionnaire TOPICS are relevant/answerable from this screen, cheaply.
- Do NOT attempt to answer questions here; only route topics.

INPUTS:
- state_sig: UI signature for staleness/debug
- topic_tree_shallow: list of topic entries {topic_id,title,keywords?,example_questions?}
- page_signals: small page cues (ocr_top_lines, ui_type?, local_tags?, page_summary?)

OUTPUT (strict JSON matching TopicRouteResult):
- relevant_topics: 0..8 topics, each with:
  - topic_id (must exist in topic_tree_shallow)
  - confidence 0..1
  - rationale (<=1 sentence)
  - expected_evidence (0..4 short strings)
- If none, set skip_reason (short).
- followups: optional suggestions to navigate to evidence surfaces (0..4).

RULES:
- Prefer precision over recall. If unsure, omit.
- Do not invent topic_ids.
"""

# 新的LLM2-1
_ROUTER_SYSTEM = """You are LLM2-1 (Router Filler) for an Android UI exploration agent.

GOAL:
- Answer only the provided router questions when the current screen gives clear evidence.

INPUTS:
- state_sig: UI signature for staleness/debug
- app_intro: one-sentence app-level description inferred from metadata (may be empty/None)
- focus_hints: semicolon-separated app-level content focus hints inferred from metadata (may be empty/None)
- router_questions: router question entries:
  {
    id,
    full_id,
    module,
    category,
    question,
    type,
    options,
    show_if?
  }
- screenshot: the current UI screenshot; this is the primary evidence source

OUTPUT (strict JSON matching RouterResult):
- router_updates: 0..12 router answers, each with:
  - question_id (prefer full_id from router_questions; local id is acceptable only if full_id is absent)
  - new_answer
  - confidence
  - rationale(<=1 sentence)

RULES:
- Prefer precision over recall. If unsure, omit.
- Do not invent question_ids.
- If the current screen does not contain enough evidence for a router question, skip that question and do not include it in router_updates.
- Router questions are provided as a flat list. Treat `show_if` only as background context, not as an availability gate.
- For multiple-choice routers, `new_answer` should be a list of option ids.
- For single-choice routers, `new_answer` should be one option id string.
- app_intro/focus_hints are weak priors only; current-screen evidence has priority.
"""


_NAV_ROUTER_SYSTEM = """You are a combined Navigation + Router model for an Android UI exploration agent.

GOAL:
- navigation: produce a compact exploration plan plus page-level return actions.
- router: answer provided router questions supported by current-screen evidence.

INPUTS:
- state_sig: UI signature for staleness/debug.
- task: current exploration goal.
- current_task: structured active task context. Its task_id should be echoed as task_id.
- task_stack: compact depth-first task stack. The last item is the active task.
- utg_context: text UI transition graph context. It may be unavailable early in a run, but when present it describes known UI nodes, edges, aliases, home path, CURRENT, PARENT, and HOME.
- app_intro/focus_hints: weak app-level priors; current-screen evidence has priority.
- history: recent action/context strings.
- block_status: existing block runtime status for navigation context.
- router_questions: flat router question entries from the selected questionnaire.
- ui_digest: compact UI tree; element ids here are the only valid ids for navigation actions.
- xml_reliable: tells whether XML-derived UI structure is reliable for this snap.
- screenshot: current UI screenshot; primary visual evidence.

OUTPUT (strict JSON matching NavigationRouterResult):
- state_sig
- task_id: echo current_task.task_id when present.
- navigation: strict JSON matching NavigationProposal.
- router: strict JSON matching RouterResult.
- task_decision: whether the active task is done/failed/should return, with a short reason.
- proposed_tasks: child tasks discovered on this UI. Each executable task needs an entry_action and exploration_depth.

DECISION ORDER:
1. First perform router analysis: identify direct evidence for router questions and visible entries that may lead to relevant evidence.
2. Then decide whether any visible entry should become a proposed child task.
3. Then choose navigation actions for the current UI action pool. Each candidate must have action_role:
   - continue_current_task: the action advances the active task.
   - start_child_task: the action enters a newly proposed child task.
4. If the page is unrelated to the active task, prefer closing, returning, skipping, or waiting rather than exploring unrelated controls.
5. If the current page already satisfies the active task, set task_decision.current_task_done=true and explain why.
6. If current_task.task_type is enter_main_page and this is already a stable main/home/menu page, the enter_main_page task is complete; do not mark ordinary main-page feature buttons as continue_current_task.
7. Use current_task.step_budget and current_task.used_steps to judge whether to continue. If many steps have already been used, prefer finishing the task unless a questionnaire-relevant path clearly needs more evidence.

NAVIGATION RULES:
- Use utg_context to understand where CURRENT is in the known UI transition graph before selecting actions.
- If navigation.overlay_kind=dismiss, put all close/deny/not-now/OK/skip/continue-past-popup actions in navigation.overlay_dismiss_actions and set navigation.candidate_actions=[].
- If navigation.overlay_kind=dismiss, do not choose fullscreen/root/container elements as dismiss targets unless that element is the only clearly tappable close/continue control.
- navigation.candidate_actions are the current UI action pool. They may continue the active task or start child tasks.
- For actions that continue the active task, set action_role=continue_current_task and starts_task_type="".
- For actions that enter proposed_tasks, set action_role=start_child_task, starts_task_type to that proposed task's task_type, and starts_task_depth to that proposed task's exploration_depth.
- If task_decision.current_task_done=true, do not output continue_current_task candidate_actions unless a necessary final close/return action is required.
- Do not include back, close, up, return, or already-visited tab-switch controls in candidate_actions unless they likely open genuinely new content.
- Do not include login/sign-in/social-login actions unless the task explicitly requires account login or no other useful non-auth exploration exists.
- Actions whose main effect is returning to a previous/visited page should be omitted or receive very low priority.
- navigation.page_return_actions are page-level return/exit actions to use only after all candidate_actions on this page are completed.
- Typical page_return_actions: visible back arrow, close/X button, or tab switch back to a previous/root page.
- PAGE RETURN RULES WITH UTG:
  - Use utg_context to decide page_return_actions when utg_context.text is present.
  - First set navigation.page_return_status:
    * has_visible_return: CURRENT should be able to return and a visible return/close/up/tab-return control exists.
    * no_return_needed: CURRENT is home, loading, splash, startup-only, or otherwise should not return to a previous UI.
    * needs_return_but_no_visible_control: CURRENT should return to parent/home/ancestor but no reliable visible return control is available.
  - Always explain the status in navigation.page_return_reason.
  - If page_return_status=has_visible_return, fill page_return_actions with the visible controls.
  - If page_return_status=no_return_needed or needs_return_but_no_visible_control, keep page_return_actions=[].
  - page_return_actions should be visible actions on CURRENT that return to one of: home UI, parent UI, or nearest useful ancestor UI.
  - Prefer home if it is visible and meaningful.
  - Prefer parent when home is not directly reachable.
  - Use nearest useful ancestor when parent is loading, splash, one-way, or overlay.
  - Do not generate page_return_actions back to splash/loading/startup pages.
  - If CURRENT was reached by bottom-tab switching, prefer switching back to the previous/home tab instead of Android back.
  - If no visible return, close, up, or tab-return control exists, keep page_return_actions=[].
- Do not duplicate any element_id/action between candidate_actions, overlay_dismiss_actions, and page_return_actions.
- If no visible return/close/up/tab-return control exists, output page_return_actions=[]. Do not invent a return action.
- Do not output screenshot coordinates or bounding boxes.
- Navigation action element_id values MUST come from ui_digest ids.
- If xml_reliable is false, trust screenshot evidence more than labels/tree semantics.

ROUTER RULES:
- Router question_id values MUST come from router_questions; prefer full_id when present.
- Answer router questions only when the current screen provides clear evidence.
- If unsure, omit rather than hallucinate.

TASK RULES:
- For proposed_tasks, create child tasks only for visible entries that are relevant to questionnaire evidence or the active task.
- First-version executable proposed tasks require entry_action. If no entry action is visible, do not propose that task.
- If proposed_tasks includes an entry_action, include the same action in navigation.candidate_actions with action_role=start_child_task.
- Do not use internal task ids in candidate actions. Workflow creates ids like task_0002 after your response.
- Useful task_type values include enter_main_page, explore_policy, explore_payment, explore_settings, explore_account, explore_core_feature, dismiss_overlay, and generic.
- Use explore_core_feature for ordinary important app/game features such as play, online mode, statistics, feature pages, content browsing, or other primary app functions.
- proposed_tasks.priority is a 0-1 importance score. Use values near 1.0 for core app features and strong questionnaire-relevant entries; use lower values for minor or weakly relevant entries.
- proposed_tasks.exploration_depth controls detail level only. It does not change the numeric step budget.
- Use exploration_depth=deep for questionnaire/router-relevant tasks that need enough evidence to answer questions, such as privacy, permissions, payment/subscription, account, data, ads, or safety.
- Use exploration_depth=normal for important settings/account/store/core paths where one or two key subpages may be useful.
- Use exploration_depth=shallow for broad app-feature overview, share flows, theme/appearance, leaderboard, tutorials, content browsing, or weakly relevant branches. For shallow tasks, identify the feature and avoid drilling into every sub-option or external share channel.
- If current_task.exploration_depth=shallow and the page already reveals the feature's purpose, set current_task_done=true instead of continuing deeper.
- If current_task.exploration_depth=normal, inspect only key branches and stop when the page gives enough useful context.
- If current_task.exploration_depth=deep, continue while evidence is still needed for router/questionnaire questions, but avoid real purchases, irreversible submissions, or credential entry.
- Do not artificially limit proposed_tasks to only the top three. Include visible valuable child tasks and use priority to rank their importance.
"""

_APP_METADATA_SYSTEM = """You are an assistant that summarizes Android app metadata for downstream UI analysis.

GOAL:
- Convert selected app metadata fields into a compact reusable context for later app UI exploration and UI analysis.
- Keep the output factual and concise; do not invent details not supported by the metadata.

INPUTS:
- app_id:
  - Android package id (unique app identifier).
- app_metadata: contains only the following selected fields:
  - description
    - Main app-store description text; primary source for app functionality/content.
  - descriptionHTML
    - HTML-formatted description text; may overlap with description and include markup artifacts.
  - summary
    - Short app tagline/summary of core purpose.
  - contentRating
    - Store-provided content-rating label.
  - contentRatingDescription
    - Optional explanation text for the content-rating decision.
  - offersIAP
    - Whether the app provides in-app purchases.
  - inAppProductPrice
    - In-app purchase price range or price note.
  - genre
    - Human-readable app category.
  - genreId
    - Normalized category id from store taxonomy.
  - categories
    - Category list payload (often serialized list/dict string).

WORKFLOW:
1) Build `app_intro` (based on description / descriptionHTML / summary).
   - Output one concise sentence that briefly introduces the app and provides hints for subsequent app exploration and UI analysis.
   - Do not mention the app's specific name; refer to it as "the app".

2) Build `focus_hints` (primarily based on contentRating / contentRatingDescription / offersIAP / inAppProductPrice, and also referencing description / descriptionHTML / summary).
   - Output short natural-language review hints for downstream UI inspection, summarizing what types of content in the app may affect age-related content considerations.
   - Format as semicolon-separated phrases, where each phrase represents one aspect.
   - IMPORTANT: do NOT explicitly output the app's age rating; only describe the related content.

3) Infer `questionnaire_type` (based on genre / genreId / categories).
   - Based on the app's category/type information, determine which questionnaire type applies.
   - The output must be exactly one of: games | social_apps | others.

4) Fill `notes`.
   - If any of the above three fields is empty, briefly explain why:
     e.g., missing source field, empty source value, garbled/unusable text, or insufficient evidence.
   - If all fields are confidently filled, `notes` should be empty.

OUTPUT (strict JSON matching AppMetadataSummary):
- app_id
- app_intro
- focus_hints
- questionnaire_type
- notes

RULES:
- Use only provided information; no guessing.
- Follow output format strictly.
"""


_TOPIC_FILL_SYSTEM = """You are LLM2-2 (Topic Filler) for an Android UI exploration agent.

GOAL:
- Propose questionnaire updates ONLY for the provided question_pack (topic-scoped).
- Use ONLY evidence present on the current screen (UI digest + screenshot).

ANTI FLIP-FLOP:
- Never set "No" just because evidence is not visible.
- Do not downgrade previous Yes->No or remove multi-select choices unless explicit contradiction is present.

INPUTS:
- topic_id: current topic being filled
- question_pack: list of questions (id,type,options,parents,children,current_answer,evidence_summary)
- current_answers: existing answers for those ids
- memory: topic-scoped memory (high confidence items)
- ui_digest + screenshot: authoritative current UI

OUTPUT (strict JSON matching QuestionnaireUpdate):
- proposed_updates: only include items you can justify from this screen.
- For single-select questions: new_answer should be ONE option id/value (string).
- For multi-select questions: new_answer should be a list of option ids/values.
"""


_BLOCK_FILL_SYSTEM = """You are LLM2-2 (Block Filler) for an Android UI exploration agent.

GOAL:
- Propose questionnaire updates ONLY for the provided block payload.
- Use ONLY evidence present on the current screen (screenshot + block questions).
- Skip any question that does not have enough visible evidence on this screen.

INPUTS:
- state_sig: UI signature for staleness/debug
- app_intro: one-sentence app-level description inferred from metadata (may be empty/None)
- focus_hints: semicolon-separated app-level content focus hints inferred from metadata (may be empty/None)
- block_payload:
  {
    id,
    module,
    topic,
    block_show_if,
    questions: {
      question_id: {
        category,
        question,
        type,
        options,
        show_if
      }
    }
  }
- screenshot: current UI screenshot; this is the primary evidence source

OUTPUT (strict JSON matching QuestionnaireUpdate):
- proposed_updates: only include updates justified by this screen.
- For single-select questions: new_answer should be ONE option id/value string.
- For multi-select questions: new_answer should be a list of option ids/values.

RULES:
- Do not answer questions outside the given block.
- Prefer precision over recall. If unsure, omit.
- Never answer "No" only because evidence is absent.
- Use question ids from `block_payload.questions` as `question_id` in proposed_updates.
- app_intro/focus_hints are weak priors only; current-screen evidence has priority.
- Category hints for future merge:
  - mutual: options are different meanings; answer only the visibly supported option.
  - severity: options imply severity; answer the visibly supported severity.
  - frequency: answer only when the screenshot clearly supports a frequency/yes-no observation.
  - multiple_class: select all visibly supported classes.
  - multiple_freq: select all visibly supported frequency-tracked options.
"""


_BLOCKS_FILL_SYSTEM = """You are LLM2-2 (Blocks Filler) for an Android UI exploration agent.

GOAL:
- Fill the provided matched questionnaire blocks for the CURRENT screenshot.
- Use ONLY visible evidence in this screenshot.
- Skip questions that cannot be answered from the current screenshot.

INPUTS:
- state_sig: UI signature for debugging
- app_intro: one-sentence app-level description inferred from metadata (may be empty/None)
- focus_hints: semicolon-separated app-level content focus hints inferred from metadata (may be empty/None)
- blocks: list of block payloads:
  {
    id,
    module,
    topic,
    block_show_if,
    questions: {
      question_id: {
        category,
        question,
        type,
        options,
        show_if
      }
    }
  }
- screenshot: current UI screenshot

OUTPUT (strict JSON matching BlocksFillResult):
- block_results: one item per block that has answerable questions:
  - block_id: must be one of the provided block ids
  - detected_signals: short evidence strings
  - proposed_updates: only questions inside that block
  - conflicts: optional notes

RULES:
- Do not answer questions outside the provided blocks.
- Use question ids exactly as keys in each block's `questions`.
- For single-select questions: new_answer should be one option id string.
- For multi-select questions: new_answer should be a list of option ids.
- Prefer precision over recall. If unsure, omit.
- Never answer "No" only because evidence is absent.
- app_intro/focus_hints are weak priors only; current-screen evidence has priority.
- Respect internal question show_if: answer a child only when its parent answer is supported in the same block.
- Category hints:
  - mutual: choose the visible matching meaning.
  - severity: choose the visible supported severity.
  - frequency: answer only when the screenshot clearly supports the yes/frequency observation.
  - multiple_class: select all visible classes.
  - multiple_freq: select visible frequency-tracked options.
"""


_RECOVERY_SYSTEM = """You are LLM3 for recovery in Android UI automation.

PRIMARY GOAL:
- Resolve popups/dialogs/ads/permission prompts
- Escape stuck states (no UI change, wrong page, return failed)
- Restore a stable exploration state

OUTPUT (strict JSON matching RecoveryProposal):
- ui_view required (concise)
- candidate_actions <= 5
- Allowed actions: click, input, back, wait, restart, none, complete
- Choose overlay_kind from: none | dismiss | workflow | loading
- For visible recovery controls, use element_id from ui_digest. Do not output screenshot coordinates or bounding boxes.

STRATEGY:
1) If overlay likely, propose click actions on Close/X/Cancel/Deny/Not now/OK (safe first).
2) If return failed, propose actions to return to stable state (close dialog, back, or click top-left back icon).
3) If truly stuck, propose restart as last resort.
"""


class GPTClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        timeout_s: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = (os.getenv("OPENAI_BASE_URL", "").strip() or DEFAULT_OPENAI_BASE_URL).strip()
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_llm_calls: int = 0
        self.usage_by_op: Dict[str, Dict[str, int]] = {}

        self.client = None
        try:
            import openai  # type: ignore
            import httpx

            proxy_url = (
                os.getenv("HTTPS_PROXY")
                or os.getenv("https_proxy")
                or os.getenv("HTTP_PROXY")
                or os.getenv("http_proxy")
            )
            http_client = httpx.Client(proxy=proxy_url) if proxy_url else None

            if hasattr(openai, "OpenAI"):
                # Original official-endpoint initialization:
                # kwargs = {"api_key": self.api_key}
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                if http_client is not None:
                    kwargs["http_client"] = http_client
                self.client = openai.OpenAI(**kwargs)
            elif hasattr(openai, "Client"):
                # Original official-endpoint initialization:
                # self.client = openai.Client(api_key=self.api_key) if self.api_key else openai.Client()
                client_kwargs: Dict[str, Any] = {}
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = openai.Client(**client_kwargs)
            else:
                self.client = openai
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning("OpenAI client init failed: %s", e)
            self.client = None

    @time_consumed
    def propose_navigation(
        self,
        screenshot_b64: str,
        ui_json: Dict[str, Any],
        block_status: Dict[str, Any],
        task: str,
        app_intro: Optional[str] = None,
        focus_hints: Optional[str] = None,
        history: Optional[List[str]] = None,
        state_sig: str = "",
        xml_reliable: Optional[bool] = None,
        debug_payload_path: Optional[str] = None,
    ) -> NavigationProposal:
        """
        Ask LLM1 to interpret the current UI and propose navigation actions.

        Input:
        - screenshot_b64: current UI screenshot.
        - ui_json: compacted UI tree source.
        - block_status: questionnaire block runtime status; this is the
          navigation hint for questionnaire coverage.
        - task/history/state_sig: exploration goal and recent context.
        - xml_reliable: whether XML-derived UI structure should be trusted.
        - debug_payload_path: optional JSON file receiving the exact payload sent as user text.

        Processing:
        - Build a small UI digest.
        - Send block_status as questionnaire state context.
        - Persist the exact payload when debug_payload_path is provided.

        Output:
        - NavigationProposal with overlay handling and candidate actions.
        """
        ui_digest = _compact_digest(ui_json, limit=240)
        payload = {
            "state_sig": state_sig,
            "task": task,
            "block_status": block_status,
            "app_intro": app_intro,
            "focus_hints": focus_hints,
            "history": (history or [])[-12:],
            "ui_digest": ui_digest,
            "xml_reliable": xml_reliable

        }
        if debug_payload_path:
            try:
                path = Path(debug_payload_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                logger.debug("failed to write navigation debug payload: %s", debug_payload_path, exc_info=True)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _NAV_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, NavigationProposal, opname="propose_navigation")
        out.overlay_dismiss_actions = list(out.overlay_dismiss_actions or [])[:5]
        out.candidate_actions = list(out.candidate_actions or [])[:10]
        for c in out.candidate_actions:
            c.actions = list(c.actions or [])[:3]
        out.page_return_actions = list(out.page_return_actions or [])[:3]
        out.state_sig = state_sig or out.state_sig
        return out

    @time_consumed
    def propose_questionnaire_updates(
        self,
        screenshot_b64: str,
        ui_json: Dict[str, Any],
        open_gaps: List[str],
        hierarchy: Any,
        memory: Optional[List[Dict[str, Any]]] = None,
        current_answers: Optional[Dict[str, Any]] = None,
    ) -> QuestionnaireUpdate:
        ui_digest = _compact_digest(ui_json, limit=260)
        payload = {
            "open_gaps": open_gaps[:120],
            "hierarchy": hierarchy,
            "memory": memory or [],
            "current_answers": current_answers or {},
            "ui_digest": ui_digest,
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _Q_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, QuestionnaireUpdate, opname="propose_questionnaire_updates")
        out.proposed_updates = list(out.proposed_updates or [])[:24]
        return out

    @time_consumed
    def propose_topic_routes(
        self,
        topic_tree_shallow: List[Dict[str, Any]],
        page_signals: Dict[str, Any],
        state_sig: str = "",
    ) -> TopicRouteResult:
        payload = {
            "state_sig": state_sig,
            "topic_tree_shallow": topic_tree_shallow[:80],
            "page_signals": page_signals,
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _TOPIC_ROUTE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        out = self._call_structured(messages, TopicRouteResult, opname="propose_topic_routes")
        out.relevant_topics = list(out.relevant_topics or [])[:8]
        for t in out.relevant_topics:
            t.expected_evidence = list(t.expected_evidence or [])[:4]
        out.followups = list(out.followups or [])[:4]
        out.state_sig = state_sig or out.state_sig
        return out

    @time_consumed
    def propose_topic_fill(
        self,
        screenshot_b64: str,
        ui_json: Dict[str, Any],
        topic_id: str,
        question_pack: List[Dict[str, Any]],
        memory: Optional[List[Dict[str, Any]]] = None,
        current_answers: Optional[Dict[str, Any]] = None,
        page_signals: Optional[Dict[str, Any]] = None,
        state_sig: str = "",
    ) -> QuestionnaireUpdate:
        ui_digest = _compact_digest(ui_json, limit=280)
        payload = {
            "state_sig": state_sig,
            "topic_id": topic_id,
            "page_signals": page_signals or {},
            "question_pack": question_pack[:28],
            "current_answers": current_answers or {},
            "memory": memory or [],
            "ui_digest": ui_digest,
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _TOPIC_FILL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, QuestionnaireUpdate, opname="propose_topic_fill")
        out.proposed_updates = list(out.proposed_updates or [])[:24]
        return out

    @time_consumed
    def propose_router_answers(
        self,
        screenshot_b64: str,
        router_questions: List[Dict[str, Any]],
        app_intro: Optional[str] = None,
        focus_hints: Optional[str] = None,
        state_sig: str = "",
    ) -> RouterResult:
        """
        New LLM2-1 for the router-only execution view.

        Inputs:
        - screenshot_b64: current screen screenshot, primary evidence source
        - router_questions:
          Executable router questions extracted from `questionnaire_routers.json`.
          Each router keeps the UI-level question shape plus:
          `id`, `full_id`, and `module`.
        - state_sig: current page state signature

        Output:
        - RouterResult:
          - router_updates: only the router questions answerable on this screen
        """
        payload = {
            "state_sig": state_sig,
            "app_intro": app_intro,
            "focus_hints": focus_hints,
            "router_questions": router_questions[:40],
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, RouterResult, opname="propose_router_answers")
        out.router_updates = list(out.router_updates or [])[:12]
        out.state_sig = state_sig or out.state_sig
        return out

    @time_consumed
    def propose_navigation_and_router(
        self,
        screenshot_b64: str,
        ui_json: Dict[str, Any],
        router_questions: List[Dict[str, Any]],
        block_status: Dict[str, Any],
        task: str,
        app_intro: Optional[str] = None,
        focus_hints: Optional[str] = None,
        history: Optional[List[str]] = None,
        current_task: Optional[Dict[str, Any]] = None,
        task_stack: Optional[List[Dict[str, Any]]] = None,
        utg_context: Optional[Dict[str, Any]] = None,
        max_proposed_tasks: int = 10,
        state_sig: str = "",
        xml_reliable: Optional[bool] = None,
        debug_payload_path: str = "",
    ) -> NavigationRouterResult:
        """
        Ask one LLM call to produce both navigation planning and router answers.

        Input:
        - screenshot_b64: current screen screenshot.
        - ui_json: post-processed UI tree from snap["uist"].
        - router_questions: executable router questions loaded from the selected questionnaire.
        - block_status: current block runtime status, used only as navigation context.
        - task/history/state_sig: exploration goal, recent context, and current state id.
        - current_task/task_stack/utg_context: task-stack context that constrains navigation.
        - max_proposed_tasks: upper bound for retained proposed child tasks.

        Output:
        - NavigationRouterResult containing:
          navigation: NavigationProposal-compatible result.
          router: RouterResult-compatible result.
          task_decision/proposed_tasks: task-stack updates for workflow.
        """
        ui_digest = _compact_digest(ui_json, limit=260)
        payload = {
            "state_sig": state_sig,
            "task": task,
            "current_task": current_task or {},
            "task_stack": task_stack or [],
            "utg_context": utg_context or {},
            "max_proposed_tasks": int(max_proposed_tasks),
            "app_intro": app_intro,
            "focus_hints": focus_hints,
            "history": (history or [])[-12:],
            "block_status": block_status,
            "router_questions": router_questions[:40],
            "ui_digest": ui_digest,
            "xml_reliable": xml_reliable

        }
        if debug_payload_path:
            try:
                out_path = Path(debug_payload_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                logger.debug("Failed to write navigation/router debug payload", exc_info=True)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _NAV_ROUTER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, NavigationRouterResult, opname="propose_navigation_and_router")
        out.state_sig = state_sig or out.state_sig
        if not out.task_id and isinstance(current_task, dict):
            out.task_id = str(current_task.get("task_id") or "")
        out.navigation.state_sig = state_sig or out.navigation.state_sig
        out.router.state_sig = state_sig or out.router.state_sig

        out.navigation.overlay_dismiss_actions = list(out.navigation.overlay_dismiss_actions or [])[:5]
        out.navigation.candidate_actions = list(out.navigation.candidate_actions or [])[:10]
        for candidate in out.navigation.candidate_actions:
            candidate.actions = list(candidate.actions or [])[:3]
        out.navigation.page_return_actions = list(out.navigation.page_return_actions or [])[:3]
        out.router.router_updates = list(out.router.router_updates or [])[:12]
        proposed_limit = max(0, int(max_proposed_tasks))
        out.proposed_tasks = list(out.proposed_tasks or [])[:proposed_limit]
        depth_by_type: Dict[str, str] = {}
        for task_item in out.proposed_tasks:
            depth = str(getattr(task_item, "exploration_depth", "") or "normal")
            if depth not in {"shallow", "normal", "deep"}:
                task_item.exploration_depth = "normal"
                depth = "normal"
            task_type = str(getattr(task_item, "task_type", "") or "")
            if task_type and task_type not in depth_by_type:
                depth_by_type[task_type] = depth
        for candidate in out.navigation.candidate_actions:
            role = str(getattr(candidate, "action_role", "") or "")
            starts_type = str(getattr(candidate, "starts_task_type", "") or "")
            if role != "start_child_task":
                candidate.starts_task_type = ""
                candidate.starts_task_depth = ""
                continue
            depth = str(getattr(candidate, "starts_task_depth", "") or "")
            if depth not in {"shallow", "normal", "deep"}:
                candidate.starts_task_depth = depth_by_type.get(starts_type, "normal")
        return out

    @time_consumed
    def analyze_app_metadata(
        self,
        app_id: str,
        app_metadata: Dict[str, Any],
    ) -> AppMetadataSummary:
        """
        Analyze selected app metadata fields and return a compact reusable summary.

        Inputs:
        - app_id:
          App package id, usually from CSV column `appId`.
        - app_metadata:
          Selected metadata fields only (description/summary/rating/iap/genre related).

        Processing:
        - Keep payload compact (trim long string fields).
        - Ask structured LLM with AppMetadataSummary schema.

        Output:
        - AppMetadataSummary:
          app_intro + focus_hints + questionnaire_type (+ notes for empty fields).
        """
        compact_meta: Dict[str, Any] = {}
        for k, v in (app_metadata or {}).items():
            key = str(k or "").strip()
            if not key:
                continue
            if v is None:
                compact_meta[key] = ""
                continue
            s = str(v)
            compact_meta[key] = s[:4000] if len(s) > 4000 else s

        payload = {
            "app_id": app_id,
            "app_metadata": compact_meta,
        }
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _APP_METADATA_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        out = self._call_structured(messages, AppMetadataSummary, opname="analyze_app_metadata")
        out.app_id = app_id or out.app_id
        out.app_intro = str(out.app_intro or "")[:280]
        out.focus_hints = str(out.focus_hints or "")[:500]
        out.notes = str(out.notes or "")[:500]
        if out.questionnaire_type not in ("games", "social_apps", "others", ""):
            out.questionnaire_type = ""
        return out

    @time_consumed
    def propose_block_fill(
        self,
        screenshot_b64: str,
        block_payload: Dict[str, Any],
        app_intro: Optional[str] = None,
        focus_hints: Optional[str] = None,
        state_sig: str = "",
    ) -> QuestionnaireUpdate:
        """
        Fill one matched questionnaire block using the current screenshot.

        Inputs:
        - screenshot_b64:
          Base64-encoded screenshot for the current UI. This is the primary evidence source.
        - block_payload:
          The full payload of one block from `questionnaire_blocks.json`.
          Expected keys include:
          - id
          - module
          - topic
          - block_show_if
          - questions
        - state_sig:
          Current UI state signature for debugging/staleness.

        Processing:
        - Build a compact payload containing only this block and state_sig.
        - Send the payload and screenshot to the structured LLM call.
        - Trim the returned updates to a safe small size.

        Output:
        - QuestionnaireUpdate
          The `proposed_updates` field represents this block's answer candidates
          on the current screen.
        """
        payload = {
            "state_sig": state_sig,
            "app_intro": app_intro,
            "focus_hints": focus_hints,
            "block_payload": block_payload,
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _BLOCK_FILL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, QuestionnaireUpdate, opname="propose_block_fill")
        out.proposed_updates = list(out.proposed_updates or [])[:24]
        return out

    @time_consumed
    def propose_blocks_fill(
        self,
        screenshot_b64: str,
        blocks_payload: List[Dict[str, Any]],
        app_intro: Optional[str] = None,
        focus_hints: Optional[str] = None,
        state_sig: str = "",
        debug_payload_path: str = "",
    ) -> BlocksFillResult:
        """
        Fill all matched blocks for one UI state in a single LLM2-2 call.

        Input:
        - screenshot_b64: current UI screenshot.
        - blocks_payload: matched block payloads from QuestionnaireState2.
        - state_sig: current UI state signature.

        Processing:
        - Send the current screenshot and all matched blocks together.
        - The model returns results grouped by block_id.

        Output:
        - BlocksFillResult. Each block result contains proposed question updates
          for that block only.
        """
        payload = {
            "state_sig": state_sig,
            "app_intro": app_intro,
            "focus_hints": focus_hints,
            "blocks": blocks_payload,
        }
        if debug_payload_path:
            try:
                out_path = Path(debug_payload_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                logger.debug("Failed to write blocks_fill debug payload", exc_info=True)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _BLOCKS_FILL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, BlocksFillResult, opname="propose_blocks_fill")
        out.state_sig = state_sig or out.state_sig
        out.block_results = list(out.block_results or [])[:20]
        for block_result in out.block_results:
            block_result.proposed_updates = list(block_result.proposed_updates or [])[:24]
        return out

    @time_consumed
    def recover_state(
        self,
        screenshot_b64: str,
        ui_json: Dict[str, Any],
        frontier_hint: str = "",
        last_nav: Optional[NavigationProposal] = None,
        note: str = "",
        state_sig: str = "",
    ) -> RecoveryProposal:
        ui_digest = _compact_digest(ui_json, limit=240)
        payload = {
            "state_sig": state_sig,
            "frontier_hint": frontier_hint,
            "note": note,
            "last_nav": last_nav.model_dump() if last_nav else None,
            "ui_digest": ui_digest,
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _RECOVERY_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}} if screenshot_b64 else {"type": "text", "text": "(no screenshot)"},
                ],
            },
        ]

        out = self._call_structured(messages, RecoveryProposal, opname="recover_state")
        out.candidate_actions = list(out.candidate_actions or [])[:5]
        out.state_sig = state_sig or out.state_sig
        return out

    def _call_structured(self, messages: List[Dict[str, Any]], model_cls: Type[R], opname: str) -> R:
        """
        IPO:
          in : messages (system+user), model_cls (Pydantic), opname
          out: parsed Pydantic instance
        WHEN called:
          - internal helper for each LLM call
        FALLBACKS:
          - prefer beta.chat.completions.parse when available
          - otherwise use chat.completions.create and JSON-extract
        """
        if self.client is None:
            raise RuntimeError(f"OpenAI client not available for {opname}")

        start_ts = time.time()
        stage = self._llm_stage_from_opname(opname)
        state_sig = self._extract_state_sig(messages)

        # Preferred: structured parse
        while True:
            try:
                resp = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout_s,
                    response_format=model_cls,
                )
                try:
                    prompt_tokens = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(resp.usage, "completion_tokens", 0) or 0)
                    self._record_token_usage(opname, prompt_tokens, completion_tokens)
                    self._log_llm_usage(
                        stage=stage,
                        opname=opname,
                        state_sig=state_sig,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration_s=(time.time() - start_ts),
                        path="parse",
                    )
                except Exception:
                    pass
                return resp.choices[0].message.parsed
            except Exception as exc:
                if self._is_rate_limit_error(exc):
                    logger.warning(
                        "Rate limited for %s (parse). Sleep 10s then retry. err=%s",
                        opname,
                        str(exc)[:300],
                    )
                    time.sleep(10.0)
                    continue
                logger.error("Structured parse failed for %s; falling back to JSON extraction.", opname, exc_info=True)
                break

        # Fallback: normal completion (best-effort across OpenAI client versions)
        try:
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                while True:
                    try:
                        resp2 = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            timeout=self.timeout_s,
                        )
                        break
                    except Exception as exc:
                        if self._is_rate_limit_error(exc):
                            logger.warning(
                                "Rate limited for %s (create). Sleep 10s then retry. err=%s",
                                opname,
                                str(exc)[:300],
                            )
                            time.sleep(10.0)
                            continue
                        raise
                try:
                    prompt_tokens = int(getattr(resp2.usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(resp2.usage, "completion_tokens", 0) or 0)
                    self._record_token_usage(opname, prompt_tokens, completion_tokens)
                    self._log_llm_usage(
                        stage=stage,
                        opname=opname,
                        state_sig=state_sig,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration_s=(time.time() - start_ts),
                        path="create",
                    )
                except Exception:
                    pass
                text = resp2.choices[0].message.content or ""
            elif hasattr(self.client, "ChatCompletion"):
                # legacy module-style client
                while True:
                    try:
                        resp2 = self.client.ChatCompletion.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            timeout=self.timeout_s,
                        )
                        break
                    except Exception as exc:
                        if self._is_rate_limit_error(exc):
                            logger.warning(
                                "Rate limited for %s (legacy_create). Sleep 10s then retry. err=%s",
                                opname,
                                str(exc)[:300],
                            )
                            time.sleep(10.0)
                            continue
                        raise
                try:
                    usage = resp2.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    self._record_token_usage(opname, prompt_tokens, completion_tokens)
                    self._log_llm_usage(
                        stage=stage,
                        opname=opname,
                        state_sig=state_sig,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration_s=(time.time() - start_ts),
                        path="legacy_create",
                    )
                except Exception:
                    pass
                text = ((resp2.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            else:
                raise RuntimeError("No supported chat completion method on client")

            data = _safe_json_from_text(text)
            if hasattr(model_cls, "model_validate"):
                return model_cls.model_validate(data)  # type: ignore[return-value]
            return model_cls.parse_obj(data)  # type: ignore[return-value]
        except Exception as exc:
            raise RuntimeError(f"LLM call failed for {opname}: {exc}") from exc

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """
        Detect OpenAI token/request rate-limit exceptions in a version-tolerant way.
        """
        if exc is None:
            return False
        name = str(type(exc).__name__ or "").lower()
        text = str(exc or "").lower()
        if "ratelimit" in name or "rate_limit" in name:
            return True
        if "rate limit" in text or "rate_limit" in text:
            return True
        if "error code: 429" in text or "status code: 429" in text:
            return True
        return False

    def _record_token_usage(self, opname: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Record token usage in both legacy utils tracker and local aggregate counters.
        """
        token_record(opname, int(prompt_tokens), int(completion_tokens))
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_completion_tokens += int(completion_tokens)
        self.total_llm_calls += 1
        row = self.usage_by_op.setdefault(
            str(opname or ""),
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        row["calls"] += 1
        row["prompt_tokens"] += int(prompt_tokens)
        row["completion_tokens"] += int(completion_tokens)
        row["total_tokens"] += int(prompt_tokens) + int(completion_tokens)

    def usage_summary(self) -> Dict[str, Any]:
        """
        Return aggregate LLM token usage collected during this process.
        """
        return {
            "calls": int(self.total_llm_calls),
            "prompt_tokens": int(self.total_prompt_tokens),
            "completion_tokens": int(self.total_completion_tokens),
            "total_tokens": int(self.total_prompt_tokens + self.total_completion_tokens),
            "by_op": copy.deepcopy(self.usage_by_op),
        }

    @staticmethod
    def _llm_stage_from_opname(opname: str) -> str:
        if opname == "propose_navigation":
            return "LLM1"
        if opname == "propose_navigation_and_router":
            return "LLM1+2"
        if opname in {
            "propose_questionnaire_updates",
            "propose_topic_routes",
            "propose_topic_fill",
            "propose_router_answers",
            "propose_block_fill",
            "propose_blocks_fill",
        }:
            return "LLM2"
        if opname == "recover_state":
            return "LLM3"
        return "LLM"

    @staticmethod
    def _extract_state_sig(messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages or []):
            content = msg.get("content")
            try:
                if isinstance(content, str):
                    payload = json.loads(content)
                    sig = str((payload or {}).get("state_sig") or "").strip()
                    if sig:
                        return sig
                elif isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("type") or "") != "text":
                            continue
                        text = str(item.get("text") or "").strip()
                        if not text:
                            continue
                        payload = json.loads(text)
                        sig = str((payload or {}).get("state_sig") or "").strip()
                        if sig:
                            return sig
            except Exception:
                continue
        return ""

    @staticmethod
    def _log_llm_usage(
        *,
        stage: str,
        opname: str,
        state_sig: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_s: float,
        path: str,
    ) -> None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        sig8 = (state_sig or "")[:8]
        logger.info(
            "LLM_USAGE stage=%s op=%s sig=%s prompt=%d completion=%d total=%d duration_s=%.3f path=%s",
            stage,
            opname,
            sig8,
            int(prompt_tokens),
            int(completion_tokens),
            total_tokens,
            float(duration_s),
            path,
        )



__all__ = [
    "GPTClient",
    "UIView",
    "UIElement",
    "UIElementType",
    "ActionStep",
    "ActionCandidate",
    "ActionType",
    "OverlayKind",
    "NavigationProposal",
    "NavigationRouterResult",
    "TaskDecision",
    "ProposedTask",
    "RecoveryProposal",
    "QuestionnaireUpdate",
    "TopicRouteResult",
    "RelevantTopic",
    "RouterResult",
    "RouterQuestionAnswer",
    "ProposedUpdate",
]
