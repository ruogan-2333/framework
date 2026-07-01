"""
Tests for task-owned action candidate binding.

Input:
- Small in-memory ActionCandidate objects and a workflow instance created without Appium.

Output:
- Assertions about candidate-to-task binding and task-aware filtering.

Function:
- Verifies the workflow can bind candidates to real task ids after LLM output is applied.
"""

from gpt_cls import ActionCandidate, ActionStep, ActionType
from workflow import WorkflowRunner


def _candidate(text: str, role: str = "continue_current_task") -> ActionCandidate:
    """
    Input: visible action text and candidate role.
    Output: one ActionCandidate with a single click step.
    Function: keeps unit tests short while preserving the production schema object.
    """
    return ActionCandidate(
        actions=[ActionStep(action=ActionType.CLICK, element_id=1, text=text)],
        score=0.8,
        action_role=role,
        starts_task_type="explore_policy" if role == "start_child_task" else "",
        starts_task_depth="shallow" if role == "start_child_task" else "",
        action_intent=f"Open {text}.",
    )


def test_bind_candidate_to_task_id_roundtrip():
    """
    Input: one candidate and one workflow-created task id.
    Output: binding lookup returns the same task id.
    Function: confirms candidate_task_bindings records ownership without mutating ActionCandidate.
    """
    cand = _candidate("Privacy Policy", role="start_child_task")
    explorer = object.__new__(WorkflowRunner)
    explorer.candidate_task_bindings = {}
    explorer.sig_to_family = {}

    WorkflowRunner._bind_candidate_to_task(explorer, "state_policy_links", cand, "task_0012")

    assert WorkflowRunner._task_id_for_candidate(explorer, "state_policy_links", cand) == "task_0012"


def test_filter_candidates_for_current_task_keeps_only_bound_task():
    """
    Input: two candidates bound to different task ids and a fake current task.
    Output: only the candidate owned by the current task remains.
    Function: prevents a child task from executing its parent's or sibling's action.
    """
    keep = _candidate("Privacy Policy", role="start_child_task")
    drop = _candidate("ICON_X", role="continue_current_task")

    explorer = object.__new__(WorkflowRunner)
    explorer.candidate_task_bindings = {}
    explorer.sig_to_family = {}
    explorer.task_manager = type(
        "FakeTaskManager",
        (),
        {"current_task": lambda self: type("Task", (), {"task_id": "task_0012"})()},
    )()
    explorer._log_event = lambda *args, **kwargs: None

    WorkflowRunner._bind_candidate_to_task(explorer, "state_premium", keep, "task_0012")
    WorkflowRunner._bind_candidate_to_task(explorer, "state_premium", drop, "task_0009")

    filtered = WorkflowRunner._filter_candidates_for_current_task(explorer, "state_premium", [drop, keep])

    assert filtered == [keep]


def test_filter_candidates_without_bindings_keeps_legacy_candidates():
    """
    Input: candidates on a state with no binding table entries.
    Output: original candidates are preserved.
    Function: keeps legacy cached candidates and heuristic fallback usable.
    """
    cand = _candidate("Continue")
    explorer = object.__new__(WorkflowRunner)
    explorer.candidate_task_bindings = {}
    explorer.sig_to_family = {}
    explorer.task_manager = type(
        "FakeTaskManager",
        (),
        {"current_task": lambda self: type("Task", (), {"task_id": "task_0001"})()},
    )()

    filtered = WorkflowRunner._filter_candidates_for_current_task(explorer, "state_home", [cand])

    assert filtered == [cand]


class _FakeTask:
    """
    Input: task id, task type, and resume state signature.
    Output: lightweight task object.
    Function: gives task-first exhaustion tests the fields used by WorkflowRunner.
    """

    def __init__(self, task_id: str, task_type: str, resume_state_sig: str) -> None:
        self.task_id = task_id
        self.task_type = task_type
        self.resume_state_sig = resume_state_sig
        self.status = "running"
        self.finish_reason = ""


class _FakeTaskManager:
    """
    Input: ordered fake task list where the last item is stack top.
    Output: minimal TaskManager-compatible object for workflow unit tests.
    Function: verifies task switching without Appium, screenshots, or real task persistence.
    """

    def __init__(self, tasks: list[_FakeTask]) -> None:
        self.tasks = list(tasks)
        self.finished: list[tuple[str, str, str]] = []

    def current_task(self):
        """
        Input: current fake task list.
        Output: top running fake task, or None.
        Function: mirrors TaskManager.current_task enough for workflow helper tests.
        """
        while self.tasks and self.tasks[-1].status != "running":
            self.tasks.pop()
        return self.tasks[-1] if self.tasks else None

    def finish_current_task(self, status: str, reason: str = "", state_sig: str = ""):
        """
        Input: terminal status, reason, and source state signature.
        Output: finished fake task, or None.
        Function: records task completion and pops the stack top for assertions.
        """
        task = self.current_task()
        if not task:
            return None
        task.status = status
        task.finish_reason = reason
        self.finished.append((task.task_id, status, reason))
        self.tasks.pop()
        return task


def test_task_local_exhaustion_finishes_current_task_before_dfs_return():
    """
    Input: current UI with no candidate left for stack-top task but with sibling task still waiting.
    Output: stack-top task is failed and caller should replan on the same UI.
    Function: protects task scheduling from falling through to DFS return/back after a no-op task action.
    """
    current = _FakeTask("task_0005", "explore_payment", "state_home")
    sibling = _FakeTask("task_0004", "explore_settings", "state_home")

    explorer = object.__new__(WorkflowRunner)
    explorer.task_manager = _FakeTaskManager([sibling, current])
    explorer.candidate_task_bindings = {"state_home": {"settings_key": "task_0004"}}
    explorer._family_id = lambda sig: sig
    explorer._log_event = lambda *args, **kwargs: None
    explorer._save_tasks_snapshot = lambda *args, **kwargs: None

    handled = WorkflowRunner._finish_current_task_if_task_exhausted(
        explorer,
        "state_home",
        reason="task_candidate_noop_exhausted",
    )

    assert handled is True
    assert explorer.task_manager.finished == [
        ("task_0005", "failed", "task_candidate_noop_exhausted")
    ]
    assert explorer.task_manager.current_task().task_id == "task_0004"


def test_task_local_exhaustion_does_not_finish_when_no_sibling_task_exists():
    """
    Input: current UI with only one task on the task stack.
    Output: helper returns False so legacy DFS/root exhaustion can handle termination.
    Function: avoids hiding true run completion behind task failure.
    """
    current = _FakeTask("task_0005", "explore_payment", "state_home")

    explorer = object.__new__(WorkflowRunner)
    explorer.task_manager = _FakeTaskManager([current])
    explorer.candidate_task_bindings = {}
    explorer._family_id = lambda sig: sig
    explorer._log_event = lambda *args, **kwargs: None
    explorer._save_tasks_snapshot = lambda *args, **kwargs: None

    handled = WorkflowRunner._finish_current_task_if_task_exhausted(
        explorer,
        "state_home",
        reason="task_candidate_noop_exhausted",
    )

    assert handled is False
    assert explorer.task_manager.finished == []
    assert explorer.task_manager.current_task().task_id == "task_0005"
