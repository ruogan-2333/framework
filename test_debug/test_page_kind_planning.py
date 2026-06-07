"""Tests for page_kind and unified action planning schema.

Input: synthetic NavigationProposal objects.
Output: assertions about popup candidates and page return actions.
Function: locks the expected schema behavior before workflow changes.
"""

from gpt_cls import ActionCandidate, ActionStep, ActionType, NavigationProposal, PageKind, PageTag, ProposedTask


def test_popup_can_have_close_and_payment_candidate_actions():
    """Popup pages may expose both close and task-opening actions."""

    close_step = ActionStep(
        action=ActionType.CLICK,
        element_id=1,
        anchor_label="Close",
        reasoning="close popup and continue",
    )
    payment_step = ActionStep(
        action=ActionType.CLICK,
        element_id=2,
        anchor_label="Subscription details",
        reasoning="explore payment details",
    )

    nav = NavigationProposal(
        state_sig="xml:popup",
        page_summary="Subscription popup over the main page.",
        page_kind=PageKind.POPUP,
        page_kind_reason="A modal subscription panel overlays the underlying page.",
        candidate_actions=[
            ActionCandidate(
                actions=[close_step],
                score=0.5,
                action_role="continue_current_task",
                starts_task_type="",
                starts_task_depth="",
            ),
            ActionCandidate(
                actions=[payment_step],
                score=0.9,
                action_role="start_child_task",
                starts_task_type="explore_payment",
                starts_task_depth="deep",
            ),
        ],
        page_return_actions=[close_step],
    )

    assert nav.page_kind == PageKind.POPUP
    assert len(nav.candidate_actions) == 2
    assert nav.candidate_actions[0].action_role == "continue_current_task"
    assert nav.candidate_actions[1].starts_task_type == "explore_payment"
    assert nav.page_return_actions[0].anchor_label == "Close"


def test_navigation_proposal_accepts_page_tags():
    """NavigationProposal can mark special semantic page tags for UTG nodes."""

    nav = NavigationProposal(
        state_sig="xml:policy",
        page_summary="Privacy policy page",
        page_tags=[PageTag.POLICY],
    )

    assert nav.page_tags == [PageTag.POLICY]


def test_proposed_task_accepts_predefined_task_type():
    """ProposedTask.task_type is constrained to predefined task types."""

    task = ProposedTask(prompt="Open store", task_type="explore_payment", priority=0.9)

    assert task.task_type == "explore_payment"
