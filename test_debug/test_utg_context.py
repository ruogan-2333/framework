"""
Unit tests for text UTG context generation.

Input:
- Small in-memory StateGraph instances with fake UI state signatures and actions.

Output:
- Pytest assertions over generated aliases, home paths, and text UTG content.

Function:
- Verifies StateGraph can export compact LLM-readable UTG context without Appium,
  screenshots, or live LLM calls.
"""

from __future__ import annotations

from state_graph import StateGraph


def test_home_paths_follow_shortest_known_edges() -> None:
    """
    Input: a linear home -> personal -> settings graph.
    Output: UI3 has the expected semantic home path.
    Function: confirms text UTG paths follow known graph edges.
    """
    graph = StateGraph()
    graph.record_observation("xml:home", page_kind="stable", meta={"page_summary": "Main home screen."})
    graph.record_transition(
        "xml:home",
        "xml:personal",
        {"actions": [{"action": "click", "text": "Personal"}]},
        dst_page_kind="popup",
        dst_meta={"page_summary": "Personal tab."},
    )
    graph.record_transition(
        "xml:personal",
        "xml:settings",
        {"actions": [{"action": "click", "text": "Settings"}]},
        dst_page_kind="stable",
        dst_meta={"page_summary": "Settings page."},
    )

    result = graph.build_text_utg_context(
        current_sig="xml:settings",
        home_sig="xml:home",
        parent_sig="xml:personal",
    )

    assert "HOME: UI1" in result.text
    assert "CURRENT: UI3" in result.text
    assert "PARENT: UI2" in result.text
    assert "page_kind=popup" in result.text
    assert "page_kind=stable" in result.text
    assert "UI3: home > click Personal > click Settings" in result.text
    assert result.home_paths["xml:settings"] == "home > click Personal > click Settings"


def test_unreachable_nodes_are_marked_without_mutating_graph() -> None:
    """
    Input: a home node plus a disconnected settings node.
    Output: disconnected node is marked unreachable and visit counts remain unchanged.
    Function: ensures context generation is read-only and handles broken paths.
    """
    graph = StateGraph()
    graph.record_observation("xml:home", meta={"page_summary": "Main home screen."})
    graph.record_observation("xml:orphan")
    before_visits = {sig: node.visit_count for sig, node in graph.nodes.items()}

    result = graph.build_text_utg_context(
        current_sig="xml:orphan",
        home_sig="xml:home",
    )

    after_visits = {sig: node.visit_count for sig, node in graph.nodes.items()}
    assert before_visits == after_visits
    assert result.home_paths["xml:orphan"] == "unreachable_from_home"
    assert "UI2: no page summary" in result.text
    assert "UI2: unreachable_from_home" in result.text


def test_unknown_home_degrades_to_unknown_header() -> None:
    """
    Input: a graph without a known home signature.
    Output: text UTG keeps current UI and marks home unknown.
    Function: prevents missing home detection from blocking LLM scheduling.
    """
    graph = StateGraph()
    graph.record_observation("xml:first", meta={"page_summary": "First observed page."})

    result = graph.build_text_utg_context(
        current_sig="xml:first",
        home_sig=None,
    )

    assert "HOME: unknown" in result.text
    assert "CURRENT: UI1" in result.text
    assert result.home_paths["xml:first"] == "unknown_home"


def test_text_utg_context_includes_page_tags() -> None:
    """
    Input: StateGraph node annotated with page_tags.
    Output: text UTG includes the tags on that node.
    Function: lets the navigation/router LLM see semantic page markers such as home or policy.
    """
    graph = StateGraph()
    graph.record_observation("xml:home", page_kind="stable", meta={"page_summary": "Main home screen."})
    graph.annotate("xml:home", page_tags=["home"])

    result = graph.build_text_utg_context(
        current_sig="xml:home",
        home_sig="xml:home",
    )

    assert "tags=home" in result.text
