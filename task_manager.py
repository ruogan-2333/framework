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
from typing import Any, Dict, List, Optional


EXPLORATION_DEPTHS = {"shallow", "normal", "deep"}


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
    - Natural-language prompt, task type, parent task id, and optional entry action.

    Output:
    - JSON-safe dictionaries for LLM payloads and trace snapshots.

    Function:
    - Represents the current purpose that should constrain navigation choices.
    """

    task_id: str
    prompt: str
    task_type: str
    status: str
    priority: float
    exploration_depth: str
    parent_task_id: str
    origin_state_sig: str
    entry_action: Dict[str, Any]
    step_budget: int
    used_steps: int
    created_by: str
    created_at_ms: int
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
            "prompt": self.prompt,
            "task_type": self.task_type,
            "status": self.status,
            "priority": self.priority,
            "exploration_depth": self.exploration_depth,
            "parent_task_id": self.parent_task_id,
            "origin_state_sig": self.origin_state_sig,
            "entry_action": self.entry_action,
            "step_budget": self.step_budget,
            "used_steps": max(0, int(self.used_steps)),
            "created_by": self.created_by,
            "created_at_ms": self.created_at_ms,
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
            prompt=INITIAL_TASK_PROMPT,
            task_type="enter_main_page",
            parent_task_id="",
            origin_state_sig=origin_state_sig,
            entry_action={},
            step_budget=max(self.default_steps, 1),
            created_by="system_init",
            priority=1.0,
            exploration_depth="normal",
            notes="initial main task",
        )
        task.status = "running"
        self.tasks_by_id[task.task_id] = task
        self.stack.append(task.task_id)
        task.history.append({"event": "created", "state_sig": origin_state_sig, "ts_ms": self._now_ms()})
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

    def push_child_task(
        self,
        *,
        prompt: str,
        task_type: str = "generic",
        priority: float = 0.5,
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
        have both a prompt and an entry action, then pushes the task onto stack.
        """
        clean_prompt = str(prompt or "").strip()
        clean_action = dict(entry_action or {})
        if not clean_prompt or not clean_action:
            ignored = {
                "prompt": clean_prompt,
                "task_type": task_type or "generic",
                "exploration_depth": self._normalize_depth(exploration_depth),
                "entry_action": clean_action,
                "reason": reason,
                "origin_state_sig": origin_state_sig,
                "ignored_reason": "missing_prompt_or_entry_action",
                "ts_ms": self._now_ms(),
            }
            self.ignored_proposed_tasks.append(ignored)
            return None

        parent_id = parent_task_id
        if parent_id is None:
            parent = self.current_task()
            parent_id = parent.task_id if parent else ""

        task = self._new_task(
            prompt=clean_prompt,
            task_type=str(task_type or "generic"),
            parent_task_id=str(parent_id or ""),
            origin_state_sig=origin_state_sig,
            entry_action=clean_action,
            step_budget=int(initial_steps if initial_steps is not None else self.default_steps),
            created_by="llm_proposed",
            priority=float(priority if priority is not None else 0.5),
            exploration_depth=self._normalize_depth(exploration_depth),
            notes=reason,
        )
        task.status = "running"
        task.history.append(
            {
                "event": "created",
                "state_sig": origin_state_sig,
                "reason": reason,
                "related_router_questions": list(related_router_questions or []),
                "ts_ms": self._now_ms(),
            }
        )
        self.tasks_by_id[task.task_id] = task
        self.stack.append(task.task_id)
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
        task.history.append({"event": "finished", "status": task.status, "reason": task.finish_reason, "state_sig": state_sig, "ts_ms": self._now_ms()})
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
        task.history.append(
            {
                "event": "step_consumed",
                "state_sig": state_sig,
                "action_key": action_key,
                "source": source,
                "used_steps": task.used_steps,
                "step_budget": task.step_budget,
                "ts_ms": self._now_ms(),
            }
        )
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
                    "prompt": task.prompt,
                    "task_type": task.task_type,
                    "status": task.status,
                    "priority": task.priority,
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
        prompt: str,
        task_type: str,
        parent_task_id: str,
        origin_state_sig: str,
        entry_action: Dict[str, Any],
        step_budget: int,
        created_by: str,
        priority: float,
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
        return Task(
            task_id=task_id,
            prompt=prompt,
            task_type=task_type or "generic",
            status="pending",
            priority=float(priority),
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
        text = str(value or "normal").strip().lower()
        return text if text in EXPLORATION_DEPTHS else "normal"

    @staticmethod
    def _now_ms() -> int:
        """
        Input: current system clock.
        Output: milliseconds since epoch.
        Function: provides consistent integer timestamps for task trace records.
        """
        return int(time.time() * 1000)
