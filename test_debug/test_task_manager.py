"""
Unit tests for the task stack manager.

Input:
- Pure Python task operations with fake state signatures and fake entry actions.

Output:
- Pytest assertions over task stack state.

Function:
- Verifies the task stack without Appium, LLM calls, screenshots, or snap files.
"""

from __future__ import annotations

from task_manager import (
    TASK_TYPE_SPECS,
    TaskManager,
    TaskType,
    compose_task_priority,
    normalize_task_type,
    task_type_prompt_rows,
)


def test_initial_task_exists_and_is_running() -> None:
    """
    Input: empty TaskManager and an entry state signature.
    Output: one running initial task.
    Function: confirms the main task is created lazily at run start.
    """
    manager = TaskManager(default_steps=3)
    task = manager.ensure_initial_task("xml:root")

    assert task.task_id == "task_0001"
    assert task.status == "running"
    assert task.task_type == "enter_main_page"
    assert manager.current_task() == task
    assert task.type_priority == 1.0
    assert task.llm_priority == 1.0


def test_task_type_specs_cover_allowed_types() -> None:
    """
    Input: predefined task type configuration.
    Output: all first-version task types are present.
    Function: protects the LLM-facing task taxonomy from accidental drift.
    """
    expected = {
        TaskType.ENTER_MAIN_PAGE,
        TaskType.EXPLORE_MAIN_FUNCTION,
        TaskType.EXPLORE_PAYMENT,
        TaskType.EXPLORE_POLICY,
        TaskType.EXPLORE_SETTINGS,
        TaskType.GENERIC,
    }

    assert set(TASK_TYPE_SPECS) == expected


def test_normalize_task_type_falls_back_to_generic() -> None:
    """
    Input: known, unknown, and empty task type strings.
    Output: known values survive and unknown values become generic.
    Function: prevents malformed LLM task_type strings from breaking task creation.
    """
    assert normalize_task_type("explore_payment") == TaskType.EXPLORE_PAYMENT
    assert normalize_task_type(TaskType.EXPLORE_PAYMENT) == TaskType.EXPLORE_PAYMENT
    assert normalize_task_type("explore_account") == TaskType.GENERIC
    assert normalize_task_type("") == TaskType.GENERIC


def test_compose_task_priority_uses_equal_weights_and_clamps() -> None:
    """
    Input: type-level priority and LLM local priority.
    Output: final priority in [0, 1] using equal weights.
    Function: verifies the first-version local task scheduling score.
    """
    assert compose_task_priority(0.8, 0.2) == 0.5
    assert compose_task_priority(2.0, -1.0) == 0.5


def test_task_type_prompt_rows_are_minimal_and_json_safe() -> None:
    """
    Input: predefined task type specs.
    Output: JSON-safe task type rows for LLM prompt payload.
    Function: keeps the LLM-facing task type contract stable.
    """
    rows = task_type_prompt_rows()

    assert rows
    assert {row["task_type"] for row in rows} == {item.value for item in TASK_TYPE_SPECS}
    for row in rows:
        assert set(row) == {
            "task_type",
            "description",
            "completion_goal",
            "default_priority",
            "default_depth",
        }


def test_child_task_pushes_on_top_and_finish_returns_to_parent() -> None:
    """
    Input: initial task plus one valid child task proposal.
    Output: child becomes current, then parent becomes current after finish.
    Function: verifies depth-first task stack behavior.
    """
    manager = TaskManager(default_steps=4)
    parent = manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        prompt="探索 Privacy Policy 页面",
        task_type="explore_policy",
        entry_action={"action": "click", "element_id": 7},
        origin_state_sig="xml:root",
        reason="policy link is visible",
    )

    assert child is not None
    assert manager.current_task() == child
    assert child.parent_task_id == parent.task_id
    assert child.task_type == "explore_policy"

    finished = manager.finish_current_task("done", "policy page captured", state_sig="xml:policy")

    assert finished == child
    assert manager.current_task() == parent
    assert child.finish_reason == "policy page captured"


def test_invalid_child_without_entry_action_is_ignored() -> None:
    """
    Input: child task proposal without an entry action.
    Output: no new task is pushed and the ignored proposal is recorded.
    Function: enforces the first-version entry_action requirement.
    """
    manager = TaskManager(default_steps=4)
    parent = manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        prompt="探索不明确页面",
        task_type="generic",
        entry_action={},
        origin_state_sig="xml:root",
        reason="missing entry action",
    )

    assert child is None
    assert manager.current_task() == parent
    assert len(manager.ignored_proposed_tasks) == 1


def test_child_task_records_priority_sources() -> None:
    """
    Input: child task proposal with final, type, and LLM priorities.
    Output: created task and stack summary both preserve the three priority fields.
    Function: lets trace, HTML, and future scheduling distinguish priority sources.
    """
    manager = TaskManager(default_steps=4)
    manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        prompt="探索支付页面",
        task_type="explore_payment",
        priority=0.8,
        type_priority=0.9,
        llm_priority=0.7,
        entry_action={"action": "click", "element_id": 9},
        origin_state_sig="xml:root",
        reason="store entry is visible",
    )

    assert child is not None
    assert child.priority == 0.8
    assert child.type_priority == 0.9
    assert child.llm_priority == 0.7

    summary = manager.stack_summary()[-1]
    assert summary["priority"] == 0.8
    assert summary["type_priority"] == 0.9
    assert summary["llm_priority"] == 0.7


def test_step_budget_expires_current_task() -> None:
    """
    Input: child task with a one-step budget.
    Output: consuming one step expires the child and returns to parent.
    Function: verifies action-based budget exhaustion.
    """
    manager = TaskManager(default_steps=4)
    parent = manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        prompt="探索 Store 页面",
        task_type="explore_payment",
        initial_steps=1,
        entry_action={"action": "click", "element_id": 9},
        origin_state_sig="xml:root",
        reason="store entry is visible",
    )

    assert child is not None
    assert child.step_budget == 1
    assert child.used_steps == 0
    manager.consume_step(state_sig="xml:root", action_key="click:9:Store", source="unit")

    assert child.used_steps == 1
    assert child.status == "expired"
    assert manager.current_task() == parent


def test_snapshot_is_json_safe() -> None:
    """
    Input: task stack with one ignored proposal.
    Output: dictionary containing current task id, stack, tasks, and ignored proposals.
    Function: verifies the shape expected by tasks.json.
    """
    manager = TaskManager(default_steps=2)
    manager.ensure_initial_task("xml:root")
    manager.push_child_task(prompt="", task_type="generic", entry_action={}, origin_state_sig="xml:root")

    snapshot = manager.snapshot()

    assert snapshot["current_task_id"] == "task_0001"
    assert snapshot["stack"] == ["task_0001"]
    assert snapshot["tasks"][0]["task_id"] == "task_0001"
    assert snapshot["ignored_proposed_tasks"][0]["ignored_reason"] == "missing_prompt_or_entry_action"
