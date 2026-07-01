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


def test_task_type_specs_include_quota_and_budget() -> None:
    """
    Input: predefined task type specs.
    Output: expected max_created and step_budget for each task type.
    Function: protects the currently configured task scheduling quota table.
    """
    expected = {
        TaskType.ENTER_MAIN_PAGE: (1, 10),
        TaskType.EXPLORE_MAIN_FUNCTION: (50, 8),
        TaskType.EXPLORE_PAYMENT: (50, 8),
        TaskType.EXPLORE_POLICY: (50, 8),
        TaskType.EXPLORE_SETTINGS: (50, 8),
        TaskType.GENERIC: (20, 6),
    }

    for task_type, (max_created, step_budget) in expected.items():
        spec = TASK_TYPE_SPECS[task_type]
        assert spec.max_created == max_created
        assert spec.step_budget == step_budget


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
            "max_created",
            "step_budget",
        }
        assert isinstance(row["max_created"], int)
        assert isinstance(row["step_budget"], int)


def test_child_task_pushes_on_top_and_finish_returns_to_parent() -> None:
    """
    Input: initial task plus one valid child task proposal.
    Output: child becomes current, then parent becomes current after finish.
    Function: verifies depth-first task stack behavior.
    """
    manager = TaskManager(default_steps=4)
    parent = manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        initial_goal="探索 Privacy Policy 页面",
        task_type="explore_policy",
        entry_action={"action": "click", "element_id": 7},
        origin_state_sig="xml:root",
        reason="policy link is visible",
    )

    assert child is not None
    assert manager.current_task() == child
    assert child.parent_task_id == parent.task_id
    assert child.task_type == "explore_policy"
    assert child.resume_state_sig == "xml:root"

    finished = manager.finish_current_task("done", "policy page captured", state_sig="xml:policy")

    assert finished == child
    assert manager.current_task() == parent
    assert child.finish_reason == "policy page captured"


def test_update_resume_state_tracks_task_resume_location() -> None:
    """
    Input: one child task and a later state signature.
    Output: the child task resume_state_sig is updated.
    Function: verifies task switching can restore to the latest UI for a task.
    """
    manager = TaskManager(default_steps=4)
    manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        initial_goal="探索 Privacy Policy 页面",
        task_type="explore_policy",
        entry_action={"action": "click", "element_id": 7},
        origin_state_sig="xml:root",
        reason="policy link is visible",
    )

    assert child is not None
    updated = manager.update_resume_state(child.task_id, "xml:policy")

    assert updated == child
    assert child.resume_state_sig == "xml:policy"


def test_invalid_child_without_entry_action_is_ignored() -> None:
    """
    Input: child task proposal without an entry action.
    Output: no new task is pushed and the ignored proposal is recorded.
    Function: enforces the first-version entry_action requirement.
    """
    manager = TaskManager(default_steps=4)
    parent = manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        initial_goal="探索不明确页面",
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
        initial_goal="探索支付页面",
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


def test_child_task_step_budget_uses_task_type_spec() -> None:
    """
    Input: child task proposal with an exaggerated LLM initial step count.
    Output: created task uses the task-type configured step budget.
    Function: keeps runtime budgets controlled by local scheduling config instead of raw LLM output.
    """
    manager = TaskManager(default_steps=8)
    manager.ensure_initial_task("xml:root")

    child = manager.push_child_task(
        initial_goal="explore payment entry",
        task_type="explore_payment",
        initial_steps=20,
        entry_action={"action": "click", "element_id": 9},
        origin_state_sig="xml:root",
        reason="store entry is visible",
    )

    assert child is not None
    assert child.step_budget == TASK_TYPE_SPECS[TaskType.EXPLORE_PAYMENT].step_budget


def test_child_task_creation_respects_max_created_per_type() -> None:
    """
    Input: one more proposed explore_payment child task than the configured quota.
    Output: tasks up to the quota are created and the extra proposal is recorded as ignored.
    Function: verifies task-type creation quota even while the quota is temporarily relaxed.
    """
    manager = TaskManager(default_steps=8)
    manager.ensure_initial_task("xml:root")
    max_created = TASK_TYPE_SPECS[TaskType.EXPLORE_PAYMENT].max_created

    created = []
    for idx in range(max_created + 1):
        created.append(
            manager.push_child_task(
                initial_goal=f"explore payment entry {idx}",
                task_type="explore_payment",
                entry_action={"action": "click", "element_id": idx + 1},
                origin_state_sig="xml:root",
                reason="payment entry is visible",
            )
        )

    assert [task is not None for task in created] == [True] * max_created + [False]
    assert len([task for task in manager.tasks_by_id.values() if task.task_type == "explore_payment"]) == max_created
    assert manager.ignored_proposed_tasks[-1]["ignored_reason"] == "max_created_per_type_exceeded"
    assert manager.ignored_proposed_tasks[-1]["task_type"] == "explore_payment"
    assert manager.ignored_proposed_tasks[-1]["max_created"] == max_created


def test_step_budget_expires_current_task() -> None:
    """
    Input: child task whose local task type has a configured step budget.
    Output: consuming that many steps expires the child and returns to parent.
    Function: verifies action-based budget exhaustion.
    """
    manager = TaskManager(default_steps=4)
    parent = manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        initial_goal="探索 Store 页面",
        task_type="explore_payment",
        initial_steps=1,
        entry_action={"action": "click", "element_id": 9},
        origin_state_sig="xml:root",
        reason="store entry is visible",
    )

    assert child is not None
    assert child.step_budget == TASK_TYPE_SPECS[TaskType.EXPLORE_PAYMENT].step_budget
    assert child.used_steps == 0
    for idx in range(child.step_budget):
        manager.consume_step(state_sig=f"xml:store{idx}", action_key=f"click:{idx}:More", source="unit")

    assert child.used_steps == child.step_budget
    assert child.status == "expired"
    assert manager.current_task() == parent


def test_record_task_progress_updates_current_goal_and_history() -> None:
    """
    Input: one active child task, one UI progress update, and one later action update.
    Output: current goal and human-readable history row are updated.
    Function: verifies the dynamic task-goal history format used by reports.
    """
    manager = TaskManager(default_steps=4)
    manager.ensure_initial_task("xml:root")
    child = manager.push_child_task(
        initial_goal="Open settings and inspect privacy controls.",
        task_type="explore_settings",
        entry_action={"action": "click", "element_id": 3},
        origin_state_sig="xml:root",
        reason="settings entry is visible",
    )

    assert child is not None
    manager.record_task_progress(
        task_id=child.task_id,
        state_sig="xml:settings",
        page_summary="Settings page with privacy entries.",
        previous_goal=child.current_goal,
        current_goal="Inspect privacy and blocking controls.",
        progress="Settings page reached and privacy-related options are visible.",
    )
    manager.record_task_progress(
        task_id=child.task_id,
        state_sig="xml:settings",
        page_summary="",
        previous_goal=child.current_goal,
        current_goal=child.current_goal,
        progress="",
        selected_action="click:12 Privacy",
        action_intent="Open Privacy settings.",
    )

    assert child.current_goal == "Inspect privacy and blocking controls."
    assert child.progress_summary == "Settings page reached and privacy-related options are visible."
    assert child.history == [
        {
            "state_sig": "xml:settings",
            "page_summary": "Settings page with privacy entries.",
            "previous_goal": "Open settings and inspect privacy controls.",
            "current_goal": "Inspect privacy and blocking controls.",
            "progress": "Settings page reached and privacy-related options are visible.",
            "selected_action": "click:12 Privacy",
            "action_intent": "Open Privacy settings.",
        }
    ]


def test_snapshot_is_json_safe() -> None:
    """
    Input: task stack with one ignored proposal.
    Output: dictionary containing current task id, stack, tasks, and ignored proposals.
    Function: verifies the shape expected by tasks.json.
    """
    manager = TaskManager(default_steps=2)
    manager.ensure_initial_task("xml:root")
    manager.push_child_task(initial_goal="", task_type="generic", entry_action={}, origin_state_sig="xml:root")

    snapshot = manager.snapshot()

    assert snapshot["current_task_id"] == "task_0001"
    assert snapshot["stack"] == ["task_0001"]
    assert snapshot["tasks"][0]["task_id"] == "task_0001"
    assert snapshot["ignored_proposed_tasks"][0]["ignored_reason"] == "missing_initial_goal_or_entry_action"


def test_snapshot_records_quota_ignored_tasks() -> None:
    """
    Input: task manager where one proposed task exceeds the configured per-type quota.
    Output: snapshot contains the quota rejection entry.
    Function: ensures run-level tasks.json can explain why LLM proposals were dropped.
    """
    manager = TaskManager(default_steps=8)
    manager.ensure_initial_task("xml:root")
    max_created = TASK_TYPE_SPECS[TaskType.EXPLORE_POLICY].max_created

    for idx in range(max_created + 1):
        manager.push_child_task(
            initial_goal=f"explore policy entry {idx}",
            task_type="explore_policy",
            entry_action={"action": "click", "element_id": idx + 1},
            origin_state_sig="xml:root",
            reason="policy link is visible",
        )

    snapshot = manager.snapshot()

    assert len([task for task in snapshot["tasks"] if task["task_type"] == "explore_policy"]) == max_created
    assert snapshot["ignored_proposed_tasks"][-1]["ignored_reason"] == "max_created_per_type_exceeded"
    assert snapshot["ignored_proposed_tasks"][-1]["task_type"] == "explore_policy"
