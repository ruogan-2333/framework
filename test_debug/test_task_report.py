"""Tests for task-oriented run report generation.

Input: synthetic tasks.json and trace.jsonl files.
Output: assertions about generated task report dictionaries.
Function: verifies that task observation/action events are grouped by task.
"""

from pathlib import Path

from task_report import build_task_report, render_task_report_markdown, write_task_report


def test_build_task_report_groups_observation_and_action_by_task(tmp_path: Path):
    """Report generation joins one UI observation with its selected action."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tasks.json").write_text(
        """
{
  "run_id": "run1",
  "tasks": [
    {
      "task_id": "task_0001",
      "task_type": "enter_main_page",
      "status": "running",
      "prompt": "Enter the main page.",
      "priority": 1.0,
      "type_priority": 1.0,
      "llm_priority": 1.0,
      "used_steps": 1,
      "step_budget": 8,
      "finish_reason": ""
    }
  ]
}
""",
        encoding="utf-8",
    )
    (run_dir / "trace.jsonl").write_text(
        "\n".join(
            [
                '{"event":"transition","data":{"kind":"task_ui_observation","task_id":"task_0001","state_sig":"xml:abc","page_summary":"Splash","page_kind":"loading","page_tags":[],"task_progress":"The page is loading, so the task should wait.","task_decision":{"current_task_done":false}}}',
                '{"event":"transition","data":{"kind":"task_action_selected","task_id":"task_0001","state_sig":"xml:abc","candidate_key":"wait:None:","action_role":"continue_current_task","action_intent":"Wait for the main page to finish loading.","action":"wait","element_id":null,"label":"","reasoning":"Wait loading"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_task_report(run_dir)

    assert report["task_count"] == 1
    task = report["tasks"][0]
    assert task["task_id"] == "task_0001"
    assert task["steps"][0]["page_summary"] == "Splash"
    assert task["steps"][0]["selected_action"]["action_intent"] == "Wait for the main page to finish loading."


def test_write_task_report_writes_json_and_markdown(tmp_path: Path):
    """write_task_report creates both machine-readable and human-readable files."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tasks.json").write_text('{"run_id":"run2","tasks":[]}', encoding="utf-8")
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")

    report = write_task_report(run_dir)
    md = render_task_report_markdown(report)

    assert (run_dir / "task_report.json").exists()
    assert (run_dir / "task_report.md").exists()
    assert "Task Report" in md
