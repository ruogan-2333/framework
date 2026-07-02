"""
Offline LLM test for transition/replay/return mismatch state matching.

Input:
- An observed trace state directory captured after a transition/replay/return action.
- An expected trace state directory that the workflow intended to reach.
- Optional nearest known trace state directory selected by local similarity or manual choice.
- Optional transition context JSON file describing the action/task that led here.

Output:
- test_debug/state_family_compare_outputs/<timestamp>_transition_mismatch/
  - input.json: compact payload used by the LLM method.
  - result.json: TransitionMismatchResult.
  - report.md: human-readable summary.

Function:
- Reuses GPTClient.match_state_after_transition so this offline experiment can
  validate state-family matching before connecting it to workflow replay/return.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_config import load_project_env
from gpt_cls import GPTClient


def parse_args() -> argparse.Namespace:
    """
    Input: command-line arguments.
    Output: parsed arguments for one transition mismatch comparison.
    Function: defines the offline test interface.
    """
    parser = argparse.ArgumentParser(description="Compare observed transition result against expected/nearest states.")
    parser.add_argument("--observed-state", required=True, help="State directory actually observed after transition.")
    parser.add_argument("--expected-state", required=True, help="State directory expected by UTG/replay/return.")
    parser.add_argument("--nearest-state", default="", help="Optional nearest known state directory.")
    parser.add_argument("--transition-context-json", default="", help="Optional JSON file with action/task context.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "test_debug" / "state_family_compare_outputs"),
        help="Directory where the debug run folder is written.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name; default matches current workflow use.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--timeout", type=float, default=120.0, help="LLM request timeout seconds.")
    return parser.parse_args()


def read_json(path: Path, default: Any) -> Any:
    """
    Input: JSON path and default value.
    Output: parsed JSON object or default when the file is missing/invalid.
    Function: keeps offline experiments tolerant of partially populated state dirs.
    """
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def image_to_b64(path: Path) -> str:
    """
    Input: image file path.
    Output: base64 string, or empty string when missing.
    Function: prepares screenshots for GPTClient image input.
    """
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def state_sig_from_dir(state_dir: Path) -> str:
    """
    Input: trace state directory path.
    Output: state signature inferred from directory name.
    Function: converts names such as UI000039_xml_<hash> to xml:<hash>.
    """
    name = state_dir.name
    if "_xml_" in name:
        return "xml:" + name.rsplit("_xml_", 1)[1]
    if "_phash_" in name:
        return "phash:" + name.rsplit("_phash_", 1)[1]
    return name


def load_state_payload(state_dir: Path) -> Dict[str, Any]:
    """
    Input: trace state directory.
    Output: compact state payload with screenshot, snap/uist, and signature.
    Function: centralizes state-dir parsing for this offline test.
    """
    if not state_dir:
        return {"state_dir": "", "state_sig": "", "xml_reliable": None, "screenshot_b64": "", "ui_json": {}}
    snap = read_json(state_dir / "snap.json", {})
    uist = snap.get("uist") if isinstance(snap, dict) else {}
    if not uist:
        uist = read_json(state_dir / "uist.json", {})
    xml_reliable = snap.get("xml_reliable") if isinstance(snap, dict) else None
    return {
        "state_dir": str(state_dir),
        "state_sig": state_sig_from_dir(state_dir),
        "xml_reliable": xml_reliable,
        "screenshot_b64": image_to_b64(state_dir / "screenshot.png"),
        "ui_json": uist or snap or {},
    }


def write_report(
    output_dir: Path,
    result: Dict[str, Any],
    observed_state: Dict[str, Any],
    expected_state: Dict[str, Any],
    nearest_state: Dict[str, Any],
) -> None:
    """
    Input: output directory, LLM result, and loaded state metadata.
    Output: report.md in output directory.
    Function: writes a short human-readable summary for manual review.
    """
    lines = [
        "# Transition Mismatch Comparison",
        "",
        f"- observed: `{observed_state['state_sig']}`",
        f"- expected: `{expected_state['state_sig']}`",
        f"- nearest: `{nearest_state.get('state_sig', '')}`",
        f"- match_type: `{result.get('match_type')}`",
        f"- matched_state_sig: `{result.get('matched_state_sig')}`",
        f"- should_reanalyze_observed: `{result.get('should_reanalyze_observed')}`",
        f"- recommended_next_step: `{result.get('recommended_next_step')}`",
        f"- confidence: `{result.get('confidence')}`",
        "",
        "## Visible Change",
        "",
        result.get("visible_change_summary", ""),
        "",
        "## Reason",
        "",
        result.get("reason", ""),
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """
    Input: CLI args and project .env.
    Output: process exit code plus JSON/Markdown artifacts.
    Function: runs one transition mismatch LLM comparison.
    """
    load_project_env(ROOT / ".env")
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[TRANSITION] OPENAI_API_KEY not set")
        return 2

    observed_state = load_state_payload(Path(args.observed_state))
    expected_state = load_state_payload(Path(args.expected_state))
    nearest_state = load_state_payload(Path(args.nearest_state)) if args.nearest_state else {
        "state_dir": "",
        "state_sig": "",
        "xml_reliable": None,
        "screenshot_b64": "",
        "ui_json": {},
    }
    transition_context = read_json(Path(args.transition_context_json), {}) if args.transition_context_json else {}

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_transition_mismatch"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt = GPTClient(api_key=api_key, model=args.model, temperature=args.temperature, timeout_s=args.timeout)
    result = gpt.match_state_after_transition(
        observed_screenshot_b64=observed_state["screenshot_b64"],
        observed_ui_json=observed_state["ui_json"],
        expected_screenshot_b64=expected_state["screenshot_b64"],
        expected_ui_json=expected_state["ui_json"],
        nearest_screenshot_b64=nearest_state["screenshot_b64"],
        nearest_ui_json=nearest_state["ui_json"],
        transition_context=transition_context,
        expected_state_sig=expected_state["state_sig"],
        observed_state_sig=observed_state["state_sig"],
        nearest_state_sig=nearest_state["state_sig"],
        expected_xml_reliable=expected_state["xml_reliable"],
        observed_xml_reliable=observed_state["xml_reliable"],
        nearest_xml_reliable=nearest_state["xml_reliable"],
        debug_payload_path=str(output_dir / "input.json"),
    )

    result_json = result.model_dump(mode="json")
    (output_dir / "result.json").write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, result_json, observed_state, expected_state, nearest_state)
    print(f"[TRANSITION] output={output_dir}")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
