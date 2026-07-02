"""
Offline LLM test for pre-action dynamic page drift.

Input:
- A reference trace state directory that was already analyzed by navigation/router.
- An observed trace state directory captured before executing the planned action.
- Optional action JSON file; otherwise the script uses the first candidate action
  in the reference state's navigation_router_result.json.

Output:
- test_debug/state_family_compare_outputs/<timestamp>_pre_action_drift/
  - input.json: compact payload used by the LLM method.
  - result.json: DriftBeforeActionResult.
  - report.md: human-readable summary.

Function:
- Reuses GPTClient.compare_drift_before_action so this offline experiment can
  validate the exact LLM schema before it is connected to the main workflow.
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
    Output: parsed arguments for one pre-action drift comparison.
    Function: defines the offline test interface.
    """
    parser = argparse.ArgumentParser(description="Compare reference and observed states before executing an action.")
    parser.add_argument("--reference-state", required=True, help="State directory originally analyzed by LLM.")
    parser.add_argument("--observed-state", required=True, help="Current state directory captured before action execution.")
    parser.add_argument("--action-json", default="", help="Optional JSON file containing the planned action/candidate.")
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
    Output: compact state payload with screenshot, snap/uist, sig, and LLM result.
    Function: centralizes state-dir parsing for this offline test.
    """
    snap = read_json(state_dir / "snap.json", {})
    uist = snap.get("uist") if isinstance(snap, dict) else {}
    if not uist:
        uist = read_json(state_dir / "uist.json", {})
    xml_reliable = snap.get("xml_reliable") if isinstance(snap, dict) else None
    llm_result = read_json(state_dir / "llm" / "navigation_router_result.json", {})
    return {
        "state_dir": str(state_dir),
        "state_sig": state_sig_from_dir(state_dir),
        "xml_reliable": xml_reliable,
        "screenshot_b64": image_to_b64(state_dir / "screenshot.png"),
        "ui_json": uist or snap or {},
        "llm_result": llm_result,
    }


def first_candidate_action(llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: saved navigation_router_result.json object.
    Output: first navigation candidate action, or empty dict.
    Function: provides a default planned action for quick experiments.
    """
    result = llm_result.get("result") if isinstance(llm_result, dict) else {}
    if not isinstance(result, dict):
        result = llm_result
    navigation = result.get("navigation") if isinstance(result, dict) else {}
    candidates = navigation.get("candidate_actions") if isinstance(navigation, dict) else []
    if isinstance(candidates, list) and candidates:
        return candidates[0]
    return {}


def write_report(output_dir: Path, result: Dict[str, Any], reference_state: Dict[str, Any], observed_state: Dict[str, Any]) -> None:
    """
    Input: output directory, LLM result, and loaded state metadata.
    Output: report.md in output directory.
    Function: writes a short human-readable summary for manual review.
    """
    lines = [
        "# Pre-action Drift Comparison",
        "",
        f"- reference: `{reference_state['state_sig']}`",
        f"- observed: `{observed_state['state_sig']}`",
        f"- same_page: `{result.get('same_page')}`",
        f"- recommended_next_step: `{result.get('recommended_next_step')}`",
        f"- confidence: `{result.get('confidence')}`",
        "",
        "## Visible Change",
        "",
        result.get("visible_change_summary", ""),
        "",
        "## Reuse Reason",
        "",
        result.get("reuse_reason", ""),
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """
    Input: CLI args and project .env.
    Output: process exit code plus JSON/Markdown artifacts.
    Function: runs one drift-before-action LLM comparison.
    """
    load_project_env(ROOT / ".env")
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[DRIFT] OPENAI_API_KEY not set")
        return 2

    reference_state = load_state_payload(Path(args.reference_state))
    observed_state = load_state_payload(Path(args.observed_state))
    planned_action = read_json(Path(args.action_json), {}) if args.action_json else first_candidate_action(reference_state["llm_result"])

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_pre_action_drift"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt = GPTClient(api_key=api_key, model=args.model, temperature=args.temperature, timeout_s=args.timeout)
    result = gpt.compare_drift_before_action(
        reference_screenshot_b64=reference_state["screenshot_b64"],
        observed_screenshot_b64=observed_state["screenshot_b64"],
        reference_ui_json=reference_state["ui_json"],
        observed_ui_json=observed_state["ui_json"],
        planned_action=planned_action,
        reference_analysis=reference_state["llm_result"].get("result", reference_state["llm_result"]),
        reference_state_sig=reference_state["state_sig"],
        observed_state_sig=observed_state["state_sig"],
        reference_xml_reliable=reference_state["xml_reliable"],
        observed_xml_reliable=observed_state["xml_reliable"],
        debug_payload_path=str(output_dir / "input.json"),
    )

    result_json = result.model_dump(mode="json")
    (output_dir / "result.json").write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, result_json, reference_state, observed_state)
    print(f"[DRIFT] output={output_dir}")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
