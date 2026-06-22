"""
Task stack state for task-oriented Android UI exploration.

Input:
- Workflow state signatures and LLM-proposed task dictionaries.

Output:
- JSON-safe task snapshots for LLM context and trace artifacts.

Function:
- Maintains a simple depth-first task stack. The first version intentionally
  avoids queue scheduling, task ranking, and UTG path planning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


EXPLORATION_DEPTHS = {"shallow", "normal", "deep"}


class TaskType(str, Enum):
    """
    Predefined task categories for task-oriented UI exploration.

    Input:
    - LLM proposed task_type strings.

    Output:
    - Stable enum values accepted by workflow and prompt schema.

    Function:
    - Keeps task creation constrained to navigation goals instead of arbitrary questionnaire topics.
    """

    ENTER_MAIN_PAGE = "enter_main_page"
    EXPLORE_MAIN_FUNCTION = "explore_main_function"
    EXPLORE_PAYMENT = "explore_payment"
    EXPLORE_POLICY = "explore_policy"
    EXPLORE_SETTINGS = "explore_settings"
    GENERIC = "generic"


@dataclass(frozen=True)
class TaskTypeSpec:
    """
    Configuration for one predefined task type.

    Input:
    - Task type enum plus human-authored description, completion goal, and defaults.

    Output:
    - Prompt-facing task metadata and workflow scheduling defaults.

    Function:
    - Separates task taxonomy, creation quota, and step budget from task-stack runtime state.
    """

    task_type: TaskType
    description: str
    completion_goal: str
    default_priority: float
    default_depth: str
    max_created: int
    step_budget: int


TASK_TYPE_SPECS: Dict[TaskType, TaskTypeSpec] = {
    TaskType.ENTER_MAIN_PAGE: TaskTypeSpec(
        task_type=TaskType.ENTER_MAIN_PAGE,
        description="进入 APP 的稳定主界面，跳过登录、注册、广告、引导页和无关弹窗。",
        completion_goal="到达可以正常使用 APP 主要功能的稳定页面；如果已经到达主界面，则结束该任务，并基于主界面提出后续探索任务。",
        default_priority=1.0,
        default_depth="normal",
        max_created=1,
        step_budget=8,
    ),
    TaskType.EXPLORE_MAIN_FUNCTION: TaskTypeSpec(
        task_type=TaskType.EXPLORE_MAIN_FUNCTION,
        description="探索 APP 的主要内容、主要功能或游戏核心玩法，并在过程中观察问卷相关证据。",
        completion_goal="覆盖 APP 的代表性主功能页面，理解 APP 主要用途；在探索过程中记录问卷可见证据，例如内容风险、用户互动、位置分享、广告、年龄验证、防沉迷、AI 功能、儿童接触风险等。",
        default_priority=0.75,
        default_depth="normal",
        max_created=8,
        step_budget=5,
    ),
    TaskType.EXPLORE_PAYMENT: TaskTypeSpec(
        task_type=TaskType.EXPLORE_PAYMENT,
        description="探索 APP 的支付、订阅、商店、premium、虚拟货币、随机奖励、loot box、现金兑换、NFT 或可转移数字资产等相关功能。",
        completion_goal="找到能够回答支付相关问卷问题的页面证据，例如是否存在内购、订阅、随机奖励、虚拟货币、现金兑换或 NFT/可转移数字资产；不要执行真实购买或不可逆操作。",
        default_priority=0.90,
        default_depth="normal",
        max_created=3,
        step_budget=3,
    ),
    TaskType.EXPLORE_POLICY: TaskTypeSpec(
        task_type=TaskType.EXPLORE_POLICY,
        description="找到并打开隐私政策、服务条款、用户协议、数据政策、儿童隐私或类似政策页面。",
        completion_goal="记录政策页面及其入口；任务完成时应将当前页面标记为 policy 类页面，便于后续对政策页面做额外处理；当前任务不需要深入阅读全文。",
        default_priority=0.70,
        default_depth="shallow",
        max_created=3,
        step_budget=3,
    ),
    TaskType.EXPLORE_SETTINGS: TaskTypeSpec(
        task_type=TaskType.EXPLORE_SETTINGS,
        description="探索和问卷相关的设置、控制或安全入口，重点包括隐私/安全设置、用户屏蔽、举报、聊天审核、好友/互动限制、位置分享控制、家长控制、防沉迷、儿童安全、广告/隐私控制等。",
        completion_goal="找到和问卷关注点相关的设置项或控制项，或确认设置页没有明显相关入口；不需要深入语言、主题、声音、震动等无关设置。",
        default_priority=0.80,
        default_depth="normal",
        max_created=3,
        step_budget=3,
    ),
    TaskType.GENERIC: TaskTypeSpec(
        task_type=TaskType.GENERIC,
        description="无法归入以上类型但可能有少量问卷价值的入口。",
        completion_goal="只做轻度确认；如果与问卷关注点无关，应快速结束或跳过。",
        default_priority=0.20,
        default_depth="shallow",
        max_created=3,
        step_budget=4,
    ),
}


def normalize_task_type(value: Any) -> TaskType:
    """
    Input: raw task type value from LLM output or internal code.
    Output: valid TaskType enum.
    Function: falls back to generic when the value is empty, unknown, or malformed.
    """
    if isinstance(value, TaskType):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        value = enum_value
    try:
        return TaskType(str(value or "").strip())
    except ValueError:
        return TaskType.GENERIC


def normalize_depth(value: Any, default: str = "normal") -> str:
    """
    Input: raw exploration depth and fallback value.
    Output: one of shallow, normal, or deep.
    Function: keeps task depth stable when LLM omits or misspells it.
    """
    fallback = str(default or "normal").strip().lower()
    if fallback not in EXPLORATION_DEPTHS:
        fallback = "normal"
    text = str(value or fallback).strip().lower()
    return text if text in EXPLORATION_DEPTHS else fallback


def clamp_priority(value: Any) -> float:
    """
    Input: raw priority value from config or LLM output.
    Output: priority clamped to [0.0, 1.0].
    Function: protects scheduling math from malformed priority values.
    """
    try:
        score = float(value)
    except Exception:
        score = 0.5
    return max(0.0, min(1.0, score))


def compose_task_priority(type_priority: Any, llm_priority: Any) -> float:
    """
    Input: task-type default priority and LLM local priority.
    Output: final runtime task priority in [0.0, 1.0].
    Function: first version uses equal weights for global type importance and local LLM judgment.
    """
    return round(0.5 * clamp_priority(type_priority) + 0.5 * clamp_priority(llm_priority), 6)


def get_task_type_spec(value: Any) -> TaskTypeSpec:
    """
    Input: raw task type value.
    Output: TaskTypeSpec for that type, falling back to generic.
    Function: centralizes lookup for workflow and prompt construction.
    """
    return TASK_TYPE_SPECS[normalize_task_type(value)]


def task_type_prompt_rows() -> List[Dict[str, Any]]:
    """
    Input: global task type specs.
    Output: JSON-safe rows for LLM prompt input.
    Function: exposes only the fields the LLM needs to choose task types.
    """
    rows: List[Dict[str, Any]] = []
    for spec in TASK_TYPE_SPECS.values():
        rows.append(
            {
                "task_type": spec.task_type.value,
                "description": spec.description,
                "completion_goal": spec.completion_goal,
                "default_priority": clamp_priority(spec.default_priority),
                "default_depth": normalize_depth(spec.default_depth),
                "max_created": max(0, int(spec.max_created)),
                "step_budget": max(1, int(spec.step_budget)),
            }
        )
    return rows


INITIAL_TASK_PROMPT = (
    "进入 APP 主界面。跳过登录、注册、广告、引导页和无关弹窗。"
    "尽量到达一个稳定的、可以正常使用 APP 主要功能的页面。"
    "如果已经到达稳定主界面，则根据当前页面结构提出后续探索任务，"
    "例如设置、隐私政策、支付/订阅、账号等入口。"
)


@dataclass
class Task:
    """
    One exploration task tracked by the workflow.

    Input:
    - Initial/current natural-language goals, task type, parent task id, and optional entry action.

    Output:
    - JSON-safe dictionaries for LLM payloads and trace snapshots.

    Function:
    - Represents the current purpose that should constrain navigation choices.
    """

    task_id: str
    initial_goal: str
    current_goal: str
    task_type: str
    status: str
    priority: float
    type_priority: float
    llm_priority: float
    exploration_depth: str
    parent_task_id: str
    origin_state_sig: str
    entry_action: Dict[str, Any]
    step_budget: int
    used_steps: int
    created_by: str
    created_at_ms: int
    progress_summary: str = ""
    finish_reason: str = ""
    notes: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Input: this Task instance.
        Output: JSON-safe task dictionary.
        Function: provides a stable representation for LLM payloads and tasks.json.
        """
        return {
            "task_id": self.task_id,
            "initial_goal": self.initial_goal,
            "current_goal": self.current_goal,
            "task_type": self.task_type,
            "status": self.status,
            "priority": self.priority,
            "type_priority": self.type_priority,
            "llm_priority": self.llm_priority,
            "exploration_depth": self.exploration_depth,
            "parent_task_id": self.parent_task_id,
            "origin_state_sig": self.origin_state_sig,
            "entry_action": self.entry_action,
            "step_budget": self.step_budget,
            "used_steps": max(0, int(self.used_steps)),
            "created_by": self.created_by,
            "created_at_ms": self.created_at_ms,
            "progress_summary": self.progress_summary,
            "finish_reason": self.finish_reason,
            "notes": self.notes,
            "history": list(self.history),
        }


class TaskManager:
    """
    Depth-first task stack manager.

    Input:
    - Initial state signature, proposed child task payloads, and task decisions.

    Output:
    - Current task, stack summaries, and full snapshots.

    Function:
    - Keeps task state isolated from the main workflow so it can be unit-tested
      without Appium or LLM calls.
    """

    def __init__(self, *, default_steps: int = 8) -> None:
        """
        Input: optional default step budget for newly created tasks.
        Output: initialized empty task manager.
        Function: prepares counters and in-memory task indexes.
        """
        self.default_steps = int(default_steps)
        self.tasks_by_id: Dict[str, Task] = {}
        self.stack: List[str] = []
        self._next_id = 1
        self.ignored_proposed_tasks: List[Dict[str, Any]] = []

    def ensure_initial_task(self, origin_state_sig: str = "") -> Task:
        """
        Input: entry UI state signature.
        Output: existing or newly created initial task.
        Function: lazily creates the single first-version main task.
        """
        if self.stack:
            cur = self.current_task()
            if cur and not cur.origin_state_sig and origin_state_sig:
                cur.origin_state_sig = origin_state_sig
            return cur  # type: ignore[return-value]

        task = self._new_task(
            initial_goal=INITIAL_TASK_PROMPT,
            task_type="enter_main_page",
            parent_task_id="",
            origin_state_sig=origin_state_sig,
            entry_action={},
            step_budget=max(self.default_steps, 1),
            created_by="system_init",
            priority=1.0,
            type_priority=1.0,
            llm_priority=1.0,
            exploration_depth="normal",
            notes="initial main task",
        )
        task.status = "running"
        self.tasks_by_id[task.task_id] = task
        self.stack.append(task.task_id)
        return task

    def current_task(self) -> Optional[Task]:
        """
        Input: current task stack.
        Output: stack-top task, or None when the stack is empty.
        Function: gives workflow the task that should constrain the next LLM call.
        """
        while self.stack:
            task_id = self.stack[-1]
            task = self.tasks_by_id.get(task_id)
            if task:
                if task.status == "pending":
                    task.status = "running"
                return task
            self.stack.pop()
        return None

    def created_count_by_type(self, task_type: str) -> int:
        """
        Input: task type string.
        Output: number of already-created tasks for this normalized task type.
        Function: supports first-version per-task-type creation quotas.
        """
        normalized = normalize_task_type(task_type).value
        return sum(1 for task in self.tasks_by_id.values() if task.task_type == normalized)

    def push_child_task(
        self,
        *,
        initial_goal: str,
        task_type: str = "generic",
        priority: float = 0.5,
        type_priority: float = 0.5,
        llm_priority: float = 0.5,
        exploration_depth: str = "normal",
        initial_steps: Optional[int] = None,
        entry_action: Optional[Dict[str, Any]] = None,
        reason: str = "",
        related_router_questions: Optional[List[str]] = None,
        origin_state_sig: str = "",
        parent_task_id: Optional[str] = None,
    ) -> Optional[Task]:
        """
        Input: LLM-proposed child task fields.
        Output: created Task, or None when the proposal is invalid.
        Function: validates the first-version requirement that child tasks must
        have both an initial goal and an entry action, then pushes the task onto stack.
        """
        clean_goal = str(initial_goal or "").strip()
        clean_action = dict(entry_action or {})
        if not clean_goal or not clean_action:
            ignored = {
                "initial_goal": clean_goal,
                "task_type": task_type or "generic",
                "exploration_depth": self._normalize_depth(exploration_depth),
                "entry_action": clean_action,
                "reason": reason,
                "origin_state_sig": origin_state_sig,
                "ignored_reason": "missing_initial_goal_or_entry_action",
                "ts_ms": self._now_ms(),
            }
            self.ignored_proposed_tasks.append(ignored)
            return None

        normalized_task_type = normalize_task_type(task_type)
        spec = get_task_type_spec(normalized_task_type)
        configured_step_budget = max(1, int(spec.step_budget))
        created_count = self.created_count_by_type(normalized_task_type.value)
        max_created = max(0, int(spec.max_created))
        if max_created > 0 and created_count >= max_created:
            ignored = {
                "initial_goal": clean_goal,
                "task_type": normalized_task_type.value,
                "exploration_depth": self._normalize_depth(exploration_depth),
                "entry_action": clean_action,
                "reason": reason,
                "origin_state_sig": origin_state_sig,
                "ignored_reason": "max_created_per_type_exceeded",
                "created_count": created_count,
                "max_created": max_created,
                "ts_ms": self._now_ms(),
            }
            self.ignored_proposed_tasks.append(ignored)
            return None

        parent_id = parent_task_id
        if parent_id is None:
            parent = self.current_task()
            parent_id = parent.task_id if parent else ""

        task = self._new_task(
            initial_goal=clean_goal,
            task_type=normalized_task_type.value,
            parent_task_id=str(parent_id or ""),
            origin_state_sig=origin_state_sig,
            entry_action=clean_action,
            step_budget=configured_step_budget,
            created_by="llm_proposed",
            priority=float(priority if priority is not None else 0.5),
            type_priority=clamp_priority(type_priority),
            llm_priority=clamp_priority(llm_priority),
            exploration_depth=self._normalize_depth(exploration_depth),
            notes=reason,
        )
        task.status = "running"
        self.tasks_by_id[task.task_id] = task
        self.stack.append(task.task_id)
        return task

    def record_task_progress(
        self,
        *,
        task_id: str = "",
        state_sig: str,
        page_summary: str,
        previous_goal: str,
        current_goal: str,
        progress: str,
        selected_action: str = "",
        action_intent: str = "",
    ) -> Optional[Task]:
        """
        Input: one UI-level task progress record plus optional selected action.
        Output: updated task, or None when the target task is unavailable.
        Function: stores the human-readable task history used by reports and debugging.
        """
        task = self.tasks_by_id.get(task_id) if task_id else self.current_task()
        if not task:
            return None
        old_goal = str(previous_goal or task.current_goal or task.initial_goal or "")
        new_goal = str(current_goal or old_goal).strip()
        if new_goal:
            task.current_goal = new_goal
        progress_text = str(progress or "").strip()
        if progress_text:
            task.progress_summary = progress_text
        row = {
            "state_sig": str(state_sig or ""),
            "page_summary": str(page_summary or ""),
            "previous_goal": old_goal,
            "current_goal": task.current_goal,
            "progress": progress_text,
            "selected_action": str(selected_action or ""),
            "action_intent": str(action_intent or ""),
        }
        action_only_update = bool(row["selected_action"] or row["action_intent"]) and not bool(row["page_summary"] or row["progress"])
        if task.history and str(task.history[-1].get("state_sig") or "") == row["state_sig"]:
            for key, value in row.items():
                if action_only_update and key not in {"selected_action", "action_intent"}:
                    continue
                if value or key in {"previous_goal", "current_goal"}:
                    task.history[-1][key] = value
        else:
            task.history.append(row)
        return task

    def finish_current_task(self, status: str, reason: str = "", state_sig: str = "") -> Optional[Task]:
        """
        Input: terminal status and natural-language reason.
        Output: the finished task, or None when no task is active.
        Function: marks the stack-top task as ended and returns control to its parent.
        """
        task = self.current_task()
        if not task:
            return None
        task.status = status if status in {"done", "failed", "blocked", "expired"} else "done"
        task.finish_reason = str(reason or "")
        if self.stack and self.stack[-1] == task.task_id:
            self.stack.pop()
        parent = self.current_task()
        if parent and parent.status == "pending":
            parent.status = "running"
        return task

    def consume_step(self, state_sig: str = "", action_key: str = "", source: str = "") -> Optional[Task]:
        """
        Input: state/action context for one executed action.
        Output: the task after consuming the step, or the expired task.
        Function: decrements the current task budget and expires it at zero.
        """
        task = self.current_task()
        if not task:
            return None
        task.used_steps = max(0, int(task.used_steps)) + 1
        if task.used_steps >= task.step_budget:
            self.finish_current_task("expired", "step_budget exhausted", state_sig=state_sig)
        return task

    def stack_summary(self) -> List[Dict[str, Any]]:
        """
        Input: current stack ids.
        Output: compact list of task dictionaries for LLM context.
        Function: avoids sending full task histories in every LLM call.
        """
        rows: List[Dict[str, Any]] = []
        for task_id in self.stack:
            task = self.tasks_by_id.get(task_id)
            if not task:
                continue
            rows.append(
                {
                    "task_id": task.task_id,
                    "initial_goal": task.initial_goal,
                    "current_goal": task.current_goal,
                    "progress_summary": task.progress_summary,
                    "task_type": task.task_type,
                    "status": task.status,
                    "priority": task.priority,
                    "type_priority": task.type_priority,
                    "llm_priority": task.llm_priority,
                    "exploration_depth": task.exploration_depth,
                    "step_budget": task.step_budget,
                    "used_steps": max(0, int(task.used_steps)),
                    "parent_task_id": task.parent_task_id,
                    "origin_state_sig": task.origin_state_sig,
                }
            )
        return rows

    def snapshot(self) -> Dict[str, Any]:
        """
        Input: all task manager state.
        Output: JSON-safe full task snapshot.
        Function: writes the run-level tasks.json artifact.
        """
        current = self.current_task()
        return {
            "current_task_id": current.task_id if current else "",
            "stack": list(self.stack),
            "tasks": [task.to_dict() for task in self.tasks_by_id.values()],
            "ignored_proposed_tasks": list(self.ignored_proposed_tasks[-50:]),
        }

    def _new_task(
        self,
        *,
        initial_goal: str,
        task_type: str,
        parent_task_id: str,
        origin_state_sig: str,
        entry_action: Dict[str, Any],
        step_budget: int,
        created_by: str,
        priority: float,
        type_priority: float,
        llm_priority: float,
        exploration_depth: str,
        notes: str,
    ) -> Task:
        """
        Input: normalized task fields.
        Output: Task with a generated id.
        Function: centralizes task id generation and default normalization.
        """
        task_id = f"task_{self._next_id:04d}"
        self._next_id += 1
        clean_goal = str(initial_goal or "").strip()
        return Task(
            task_id=task_id,
            initial_goal=clean_goal,
            current_goal=clean_goal,
            task_type=task_type or "generic",
            status="pending",
            priority=float(priority),
            type_priority=clamp_priority(type_priority),
            llm_priority=clamp_priority(llm_priority),
            exploration_depth=self._normalize_depth(exploration_depth),
            parent_task_id=parent_task_id,
            origin_state_sig=origin_state_sig,
            entry_action=dict(entry_action or {}),
            step_budget=max(int(step_budget), 1),
            used_steps=0,
            created_by=created_by,
            created_at_ms=self._now_ms(),
            notes=notes or "",
        )

    @staticmethod
    def _normalize_depth(value: str) -> str:
        """
        Input: raw exploration depth string from code or LLM output.
        Output: one of shallow, normal, or deep.
        Function: keeps task JSON stable when the model omits or misspells the field.
        """
        return normalize_depth(value)

    @staticmethod
    def _now_ms() -> int:
        """
        Input: current system clock.
        Output: milliseconds since epoch.
        Function: provides consistent integer timestamps for task trace records.
        """
        return int(time.time() * 1000)
