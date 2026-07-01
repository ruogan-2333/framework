"""
Standalone metadata analysis debug runner.

Input:
- Android package id.
- Metadata CSV path.
- Optional output directory.

Output:
- metadata_result.json containing only AppMetadataSummary LLM output.
- Console summary showing app_type, app_intro/focus_hints presence, and output path.

Function:
- Reuses main.py metadata row lookup, field selection, app_type reading, and metadata result writer.
- Lets developers test metadata analysis without running Appium exploration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_config import load_project_env
from gpt_cls import GPTClient
from main import (
    _find_metadata_row_by_app_id,
    _pick_metadata_fields,
    _read_csv_rows,
    _read_questionnaire_type_from_metadata_row,
    _write_metadata_result_json,
)


def parse_args() -> argparse.Namespace:
    """
    Input: command-line arguments.
    Output: parsed arguments for metadata debug run.
    Function: provides a small CLI for testing metadata analysis outside Appium.
    """
    parser = argparse.ArgumentParser(description="Run metadata LLM analysis for one app.")
    parser.add_argument("--package", required=True, help="Android package id, matching CSV field APID.")
    parser.add_argument(
        "--metadata-csv-path",
        default=str(ROOT / "APP_csv" / "known_dataset_200_metadata.csv"),
        help="Metadata CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "test_debug" / "metadata_analysis_outputs"),
        help="Directory where debug output run folder is written.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--timeout", type=float, default=120.0, help="LLM request timeout seconds.")
    return parser.parse_args()


def main() -> int:
    """
    Input: CLI args and project .env.
    Output: process exit code; writes metadata_result.json on success.
    Function: runs the same metadata summary logic used by the main exploration command.
    """
    load_project_env(ROOT / ".env")
    args = parse_args()

    rows = _read_csv_rows(Path(args.metadata_csv_path))
    row = _find_metadata_row_by_app_id(rows, args.package)
    if row is None:
        print(f"[METADATA] package not found in CSV: {args.package}")
        return 2

    app_type = _read_questionnaire_type_from_metadata_row(row)
    selected_meta = _pick_metadata_fields(row)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[METADATA] OPENAI_API_KEY not set")
        return 2

    gpt = GPTClient(api_key=api_key, model=args.model, temperature=args.temperature, timeout_s=args.timeout)
    result = gpt.analyze_app_metadata(app_id=args.package, app_metadata=selected_meta)

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + args.package
    output_path = _write_metadata_result_json(result, args.output_dir, run_id)

    print(f"[METADATA] package={args.package}")
    print(f"[METADATA] app_type={app_type}")
    print(f"[METADATA] app_intro={bool(result.app_intro)} focus_hints={bool(result.focus_hints)}")
    print(f"[METADATA] result={output_path}")
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
