"""Entrypoint to run the Android UI exploration workflow."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from env_config import load_project_env

# Load the project environment file as the source of truth for API and proxy settings.
load_project_env(Path(__file__).resolve().parent / ".env")

from appium_android import AndroidAppiumClient
from gpt_cls import GPTClient
from questionnaire_state2 import QuestionnaireState as QuestionnaireState2
from workflow import BudgetConfig, WorkflowRunner
from trace_callbacks import InteractiveDebugCallbacks, JsonlTraceCallbacks, NoOpCallbacks


_META_SELECTED_FIELDS = [
    "app_name",
    "content_descriptors",
    "age_rating_descriptors",
    "category_name",
    "category_code",
    "details_full_description",
    "application_category",
    "data_safety_summary",
    "security_practices_text",
    "permissions_text",
    "details_interactive_elements",
    "details_in_app_purchases",
]


def _row_first_value(row: Dict[str, Any], keys: list[str]) -> Any:
    """
    Input: one CSV row and an ordered list of possible column names.
    Output: first non-empty column value, or an empty string when none exists.
    Function: lets metadata loading tolerate CSVs with different field names.
    """
    for key in keys:
        value = row.get(key, "")
        if str(value or "").strip():
            return value
    return ""


class MainlineConsoleFilter(logging.Filter):
    """
    Input: logging records from all project and dependency loggers.
    Output: True only for concise timeline records and warnings/errors.
    Function: keeps the console readable while app.log still receives full debug details.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "run_timeline":
            return True
        return record.levelno >= logging.WARNING


def setup_logging(debug: bool, level: str, *, quiet_console: bool = False, verbose_console: bool = False):
    # Console + file logging. The file keeps full logs; the console defaults to the human timeline.
    root_level = logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
    console_formatter = logging.Formatter('%(message)s') if not verbose_console else formatter

    root = logging.getLogger()
    root.setLevel(root_level)
    root.handlers.clear()

    file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
    file_handler.setLevel(root_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_level if verbose_console else (logging.INFO if not quiet_console else logging.WARNING))
    console_handler.setFormatter(console_formatter)
    if not verbose_console:
        console_handler.addFilter(MainlineConsoleFilter())

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)


def build_restart(appium: AndroidAppiumClient, package: str, activity: str, wait: float = 1.0):
    # Factory used in other tools: stops and relaunches the target package safely.
    def restart():
        try:
            appium.force_stop(package)
        except Exception:
            logging.getLogger(__name__).debug("force_stop failed; continuing")
        time.sleep(wait)
        appium.launch_app(package, activity)
    return restart


def _read_csv_rows(csv_path: Path) -> list[Dict[str, Any]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {csv_path} ({last_err})")


def _find_metadata_row_by_app_id(rows: list[Dict[str, Any]], app_id: str) -> Optional[Dict[str, Any]]:
    """
    Input: CSV rows and the Android package id requested from CLI.
    Output: matching metadata row, or None when the app is absent.
    Function: supports both old metadata exports (`appId`) and downloaded-app CSVs (`APID`).
    """
    target = str(app_id or "").strip()
    package_keys = ["appId", "APID", "package", "package_name", "packageName", "app_id"]
    for row in rows:
        package_id = str(_row_first_value(row, package_keys) or "").strip()
        if package_id == target:
            return row
    return None


def _pick_metadata_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: one row from known_dataset_200_metadata.csv or the final large metadata CSV.
    Output: selected metadata fields consumed by GPTClient.analyze_app_metadata.
    Function: keeps only fields useful for app-level exploration context and avoids answer-leaking fields.
    """
    return {key: str(row.get(key, "") or "").strip() for key in _META_SELECTED_FIELDS}


def _read_questionnaire_type_from_metadata_row(row: Dict[str, Any]) -> str:
    """
    Input: one row from the final metadata CSV.
    Output: questionnaire type string from CSV field `app_type`.
    Function: uses the dataset's curated questionnaire type instead of asking LLM to infer it.
    """
    return str(row.get("app_type", "") or "").strip()


def _normalize_questionnaire_type(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"games", "game"}:
        return "games"
    if raw in {"social", "social_apps", "social-apps", "socialapps"}:
        return "social_apps"
    if raw in {"others", "other"}:
        return "others"
    return ""


def _maybe_build_interactive_visualization(args: argparse.Namespace, logger: logging.Logger, actual_run_id: str = "") -> None:
    """
    Input: parsed CLI args, the main logger, and optional runtime-generated run id.
    Output: writes interactive HTML artifacts when enabled, otherwise returns without side effects.
    Function: optionally runs the post-run interactive UTG/timeline visualization for the current trace run.
    When --run-id is omitted, JsonlTraceCallbacks generates the run id; actual_run_id lets visualization find that directory.
    """
    if not bool(getattr(args, "auto_visualize_interactive", False)):
        return

    trace_dir = str(getattr(args, "trace_dir", "") or "").strip()
    run_id = str(actual_run_id or getattr(args, "run_id", "") or "").strip()
    if not trace_dir:
        logger.warning("--auto-visualize-interactive requires --trace-dir; skip visualization.")
        return
    if not run_id:
        logger.warning("--auto-visualize-interactive could not determine run id; skip visualization.")
        return

    run_dir = Path(trace_dir) / run_id
    try:
        from visualize_run_interactive import build_interactive_html

        out_html = build_interactive_html(run_dir)
        logger.info("Interactive visualization generated: %s", out_html)
        print(f"[VIS] interactive_html={out_html}")
    except Exception:
        logger.warning("Auto interactive visualization failed for run_dir=%s", run_dir, exc_info=True)


def _safe_report_name(value: str) -> str:
    """
    Input: arbitrary identifier such as a package name or run id.
    Output: filesystem-safe name fragment.
    Function: keeps install report paths readable without allowing path separators.
    """
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "").strip()) or "unknown"


def _write_install_json_report(result: Any, output_dir: Path, file_name: str) -> Path:
    """
    Input: an InstallResult-like object, output directory, and report file name.
    Output: path to the JSON report written on disk.
    Function: stores structured install/preflight results for later debugging.
    """
    from device_utils import install_result_to_dict

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / file_name
    report_path.write_text(
        json.dumps(install_result_to_dict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def _ensure_trace_run_id(args: argparse.Namespace) -> None:
    """
    Input: parsed CLI args for a workflow run.
    Output: mutates args.run_id when trace output is enabled and no run id was provided.
    Function: makes pre-workflow artifacts share the same run directory as trace.jsonl and index.html.
    """
    if str(getattr(args, "trace_dir", "") or "").strip() and not str(getattr(args, "run_id", "") or "").strip():
        args.run_id = time.strftime("%Y%m%d_%H%M%S")


def _write_metadata_result_json(result: Any, trace_dir: str, run_id: str) -> Optional[Path]:
    """
    Input: metadata analysis result object plus trace root and run id.
    Output: path to metadata_result.json, or None when trace output is disabled.
    Function: stores only the metadata LLM output in the current run directory for later inspection.
    """
    if not str(trace_dir or "").strip() or not str(run_id or "").strip():
        return None

    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    elif hasattr(result, "dict"):
        payload = result.dict()
    else:
        payload = dict(getattr(result, "__dict__", {})) or {"result": str(result)}

    output_dir = Path(trace_dir) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metadata_result.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _discover_install_package_files(apk_dir: Path, package_filter: str = "") -> list[Path]:
    """
    Input: directory containing APK/XAPK files and optional package-name substring filter.
    Output: sorted APK/XAPK file list.
    Function: provides deterministic input order for batch install smoke tests.
    """
    package_filter = str(package_filter or "").strip().lower()
    files: list[Path] = []
    for path in sorted(apk_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in {".apk", ".xapk"}:
            continue
        if package_filter and package_filter not in path.stem.lower():
            continue
        files.append(path)
    return files


def _write_batch_install_reports(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """
    Input: batch install row dictionaries and output directory.
    Output: paths to the CSV and JSONL reports.
    Function: writes incremental-friendly batch smoke test reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "install_smoke_summary.csv"
    jsonl_path = output_dir / "install_smoke_results.jsonl"
    fieldnames = [
        "package",
        "app_file",
        "file_type",
        "install_status",
        "successful_method",
        "verify_installed_status",
        "launch_status",
        "launch_foreground_package",
        "launch_error",
        "uninstall_before_status",
        "uninstall_after_status",
        "uninstall_error",
        "obb_push_status",
        "extraction_error",
        "duration_seconds",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def parse_args(argv) -> argparse.Namespace:
    # CLI supports the main exploration run plus lightweight device utilities.
    p = argparse.ArgumentParser(description="Run Appium + LLM exploration workflow or device utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run exploration workflow")
    run.add_argument("--appium-url", type=str, default="http://127.0.0.1:4723", help="Appium server URL")
    run.add_argument("--device-name", type=str, default=None, help="Optional device name (caps deviceName)")
    run.add_argument("--package", type=str, required=True, help="Target app package name")
    run.add_argument("--activity", type=str, default=None, help="Optional launch activity")
    run.add_argument("--app-file", type=str, default="", help="Optional APK/XAPK file used by --install-before-run")
    run.add_argument("--install-before-run", action="store_true", help="Install --app-file before starting Appium/LLM exploration")
    run.add_argument("--uninstall-before-install", action="store_true", help="Uninstall target package before --install-before-run")
    run.add_argument("--validate-launch-before-run", action="store_true", help="Launch and foreground-check the app after installing")
    run.add_argument("--uninstall-after-run", action="store_true", help="Uninstall target package after the exploration run finishes")
    run.add_argument("--install-timeout", type=int, default=180, help="Timeout seconds for each adb install/uninstall command")
    run.add_argument("--install-launch-wait-seconds", type=float, default=5.0, help="Seconds to wait before checking foreground package")
    run.add_argument("--install-temp-dir", type=str, default="", help="Optional temp directory for XAPK extraction")
    run.add_argument("--keep-install-temp", action="store_true", help="Keep extracted XAPK temp files for debugging")
    run.add_argument("--adb-path", type=str, default="adb", help="adb executable used for install/uninstall helpers")
    run.add_argument("--questionnaire-dir", type=str, required=True, help="Directory containing questionnaire JSON files")
    run.add_argument(
        "--questionnaire-type-source",
        type=str,
        default="manual",
        choices=["manual", "metadata", "auto"],
        help=(
            "Choose questionnaire type source. "
            "manual: always use --questionnaire-dir; "
            "metadata: prefer metadata inferred type then fallback manual; "
            "auto: same fallback behavior but intended as recommended default."
        ),
    )
    run.add_argument(
        "--metadata-csv-path",
        type=str,
        default="",
        help="Optional metadata CSV path (must contain appId). If set, metadata summary runs before exploration.",
    )

    run.add_argument("--task", type=str, default="Please explore this app with the goal of covering as many distinct major UI screens and flows as possible while using the fewest reasonable number of steps.Focus on the app’s main functions and representative interfaces.Please also look for and collect any Terms of Service, Privacy Policy, or similar legal/policy documents available within the app. In addition, please explore the app’s settings pages and any purchase-related pages, especially any screens related to subscriptions, in-app purchases, paid items, or “random loot box” / randomized reward mechanics.Please pay particular attention to UI related to user-generated content (UGC), social features, user interaction, sharing, messaging, profiles, comments, communities, or similar functions.Do not spend much effort on minor, repetitive, or highly detailed sub-features. The priority is to efficiently identify and capture the main categories of UI", help="Exploration goal string")
    run.add_argument("--relaunch", action="store_true", help="Relaunch app each time and kill once finished")

    # Budgets
    run.add_argument("--time-budget", type=float, default=300.0, help="Total run time budget (seconds)")
    run.add_argument("--max-actions", type=int, default=300, help="Max number of actions (click/back/etc.)")
    run.add_argument("--probe-cap", type=int, default=10, help="Max candidate probes per page")
    run.add_argument("--disable-probe-return", action="store_true", help="Skip probe-return exploration and commit forward directly")
    run.add_argument("--min-candidate-score", type=float, default=-1.0, help="Filter out LLM1 candidates with score below this threshold before probe/forward")
    run.add_argument("--workers", type=int, default=1, help="Thread pool workers (LLM overlap)")

    # Model
    run.add_argument("--model", type=str, default="gemini-2.5-flash", help="OpenAI-compatible model name")
    run.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    run.add_argument("--timeout", type=int, default=60, help="LLM request timeout seconds")
    run.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or use OPENAI_API_KEY env)")
    run.add_argument("--trace-dir", type=str, default=None, help="If set, writes JSONL trace + assets to this directory")
    run.add_argument("--run-id", type=str, default=None, help="Optional run id (defaults to timestamp)")
    run.add_argument(
        "--auto-visualize-interactive",
        action="store_true",
        help="After run finishes, build index.html for this trace run.",
    )

    # Logging
    run.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING)")
    run.add_argument("--debug", action="store_true")
    run.add_argument("--verbose-console", action="store_true", help="Show full log stream on console instead of concise timeline only")
    run.add_argument("--pause", action="store_true")
    run.add_argument(
        "--interactive-debug",
        action="store_true",
        help="Use concise Chinese step-by-step console output and wait for a key between major steps",
    )

    lp = sub.add_parser("list-packages", help="List installed packages")
    lp.add_argument("--device-name", default=None)
    lp.add_argument("--debug", action="store_true")

    ps = sub.add_parser("list-processes", help="List processes")
    ps.add_argument("--device-name", default=None)
    ps.add_argument("--debug", action="store_true")

    ss = sub.add_parser("screenshot", help="Capture screenshot to file")
    ss.add_argument("--device-name", default=None)
    ss.add_argument("--out", required=True)
    ss.add_argument("--debug", action="store_true")

    ui = sub.add_parser("dump-ui", help="Dump UI XML to file")
    ui.add_argument("--device-name", default=None)
    ui.add_argument("--out", required=True)
    ui.add_argument("--debug", action="store_true")

    ins = sub.add_parser("install-apk", help="Install APK via adbutils")
    ins.add_argument("--device-name", default=None)
    ins.add_argument("--apk", required=True)
    ins.add_argument("--debug", action="store_true")

    pkg = sub.add_parser("install-package", help="Install and optionally launch-test an APK/XAPK")
    pkg.add_argument("--device-name", default=None)
    pkg.add_argument("--app-file", required=True)
    pkg.add_argument("--package", required=True)
    pkg.add_argument("--adb-path", default="adb")
    pkg.add_argument("--temp-dir", default="")
    pkg.add_argument("--keep-temp", action="store_true")
    pkg.add_argument("--timeout-seconds", type=int, default=180)
    pkg.add_argument("--launch-after-install", action="store_true")
    pkg.add_argument("--no-foreground-check", action="store_true")
    pkg.add_argument("--launch-wait-seconds", type=float, default=5.0)
    pkg.add_argument("--uninstall-after-test", action="store_true")
    pkg.add_argument("--output-dir", default="test_debug/install_test_outputs")
    pkg.add_argument("--debug", action="store_true")

    batch = sub.add_parser("batch-install-smoke", help="Batch install/launch smoke-test APK/XAPK files")
    batch.add_argument("--device-name", default=None)
    batch.add_argument("--apk-dir", required=True)
    batch.add_argument("--limit", type=int, default=0, help="Max files to test; 0 means all")
    batch.add_argument("--package-filter", default="")
    batch.add_argument("--adb-path", default="adb")
    batch.add_argument("--temp-dir", default="")
    batch.add_argument("--keep-temp", action="store_true")
    batch.add_argument("--timeout-seconds", type=int, default=180)
    batch.add_argument("--uninstall-before-install", action="store_true")
    batch.add_argument("--launch-after-install", action="store_true")
    batch.add_argument("--no-foreground-check", action="store_true")
    batch.add_argument("--launch-wait-seconds", type=float, default=5.0)
    batch.add_argument("--uninstall-after-test", action="store_true")
    batch.add_argument("--output-dir", default="test_debug/install_test_outputs")
    batch.add_argument("--debug", action="store_true")

    return p.parse_args(argv)

# #============================  # Dev-default args for quick local run; keep disabled for CLI-driven runs.
# # package="bim.app"
# # package="com.lemonpiggy.drinkwater"
# package = "com.maimemo.android.momo"
#
#
# sys.argv = [sys.argv[0],
#             "run",
#             # "--appium-url", "http://localhost:4723",
#             "--appium-url", "http://127.0.0.1:4723",
#             "--device-name", "emulator-5554",
#             "--package", package,
#             # "--questionnaire-dir", "./questionnaire-v2/others",
#             "--questionnaire-dir", "./questionnaire-v2/games",
#             "--trace-dir", "./traces/",
#             "--run-id", time.strftime("%Y%m%d_%H%M%S") + "_" + package,
#             "--time-budget", "300",
#             # "--pause",
#             # "--interactive-debug",
#             # "--auto-visualize-interactive",
#             "--disable-probe-return",
#             "--min-candidate-score", "0.0",
#             "--relaunch", "--debug"]
# #============================
def main(argv=None):
    # WHEN CALLED: process entry; sets up logging, loads questionnaires, initializes Appium/GPT, and calls WorkflowRunner.run.
    # POSITION: only place run() is invoked; other subcommands bypass workflow and perform utility actions.
    args = parse_args(argv or sys.argv[1:])
    setup_logging(
        getattr(args, "debug", False),
        getattr(args, "log_level", "INFO"),
        quiet_console=bool(getattr(args, "interactive_debug", False)),
        verbose_console=bool(getattr(args, "verbose_console", False)),
    )
    logger = logging.getLogger(__name__)

    if args.cmd == "run":
        _ensure_trace_run_id(args)
        install_preflight_result = None
        install_preflight_report = None
        if args.install_before_run:
            if not str(getattr(args, "app_file", "") or "").strip():
                logger.error("--install-before-run requires --app-file")
                print("[INSTALL] failed: --install-before-run requires --app-file")
                return 2
            from device_utils import adb_success, install_package_file, install_result_to_dict, uninstall_package

            if args.uninstall_before_install:
                uninstall_first = uninstall_package(
                    args.package,
                    device_name=args.device_name,
                    adb_path=args.adb_path,
                    timeout_seconds=int(args.install_timeout),
                )
                print(
                    "[INSTALL][pre-uninstall] package={pkg} status={status}".format(
                        pkg=args.package,
                        status="ok" if adb_success(uninstall_first) else "failed",
                    )
                )

            install_preflight_result = install_package_file(
                args.app_file,
                args.package,
                device_name=args.device_name,
                adb_path=args.adb_path,
                temp_dir=str(args.install_temp_dir or "") or None,
                keep_temp=bool(args.keep_install_temp),
                timeout_seconds=int(args.install_timeout),
                launch_after_install=bool(args.validate_launch_before_run),
                validate_launch_foreground=True,
                launch_wait_seconds=float(args.install_launch_wait_seconds),
                uninstall_after_test=False,
            )
            report_base = Path(args.trace_dir or "test_debug/install_test_outputs")
            if args.run_id:
                report_dir = report_base / _safe_report_name(args.run_id)
            else:
                report_dir = report_base / ("install_preflight_" + time.strftime("%Y%m%d_%H%M%S") + "_" + _safe_report_name(args.package))
            install_preflight_report = _write_install_json_report(install_preflight_result, report_dir, "install_result.json")
            install_payload = install_result_to_dict(install_preflight_result)
            print(
                "[INSTALL] package={pkg} install={install} verify={verify} launch={launch} foreground={fg} report={report}".format(
                    pkg=args.package,
                    install=install_payload.get("install_status") or "-",
                    verify=install_payload.get("verify_installed_status") or "-",
                    launch=install_payload.get("launch_status") or "-",
                    fg=install_payload.get("launch_foreground_package") or "-",
                    report=install_preflight_report,
                )
            )
            install_failed = install_preflight_result.install_status != "ok"
            verify_failed_raw = bool(
                install_preflight_result.verify_installed_status
                and install_preflight_result.verify_installed_status != "ok"
            )
            launch_required = bool(args.validate_launch_before_run)
            launch_foreground_package = str(install_preflight_result.launch_foreground_package or "").strip()
            launch_foreground_mismatch = bool(
                launch_required and launch_foreground_package and launch_foreground_package != args.package
            )
            launch_failed = bool(
                launch_required
                and (install_preflight_result.launch_status != "ok" or launch_foreground_mismatch)
            )
            verify_failed = verify_failed_raw and not (launch_required and not launch_failed)
            if verify_failed_raw and not verify_failed:
                logger.warning(
                    "Install pm-path verification failed, but launch validation succeeded; continuing. package=%s foreground=%s",
                    args.package,
                    install_preflight_result.launch_foreground_package or "-",
                )
            if install_failed or verify_failed or launch_failed:
                logger.error("Install preflight failed: %s", json.dumps(install_payload, ensure_ascii=False))
                return 2

        api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logging.getLogger(__name__).warning("OPENAI_API_KEY not set; GPT calls will fail.")

        # Init GPT client first because metadata summarization also uses LLM.
        gpt = GPTClient(api_key=api_key, model=args.model, temperature=args.temperature, timeout_s=args.timeout)

        manual_questionnaire_dir = str(args.questionnaire_dir)
        questionnaire_type_source = str(getattr(args, "questionnaire_type_source", "manual") or "manual").strip().lower()
        metadata_csv_path = str(getattr(args, "metadata_csv_path", "") or "").strip()
        metadata_entry_found = False
        metadata_notes = ""
        app_intro: Optional[str] = None
        focus_hints: Optional[str] = None
        metadata_questionnaire_type = ""

        # Metadata pre-analysis (before workflow run).
        if metadata_csv_path:
            try:
                csv_path = Path(metadata_csv_path).resolve()
                if not csv_path.exists():
                    metadata_notes = f"metadata_csv_not_found:{csv_path}"
                    logger.warning("Metadata CSV not found: %s", csv_path)
                else:
                    rows = _read_csv_rows(csv_path)
                    row = _find_metadata_row_by_app_id(rows, args.package)
                    if row is None:
                        metadata_notes = f"appId_not_found_in_csv:{args.package}"
                        logger.warning("Metadata CSV has no appId=%s; fallback to manual questionnaire type.", args.package)
                    else:
                        metadata_entry_found = True
                        metadata_questionnaire_type = _read_questionnaire_type_from_metadata_row(row)
                        selected_meta = _pick_metadata_fields(row)
                        meta_result = gpt.analyze_app_metadata(app_id=args.package, app_metadata=selected_meta)
                        app_intro = (str(meta_result.app_intro or "").strip() or None)
                        focus_hints = (str(meta_result.focus_hints or "").strip() or None)
                        metadata_notes = str(meta_result.notes or "").strip()
                        metadata_result_path = _write_metadata_result_json(meta_result, args.trace_dir, args.run_id)
                        logger.info(
                            "Metadata analyzed app=%s questionnaire_type=%s intro=%s hints=%s result=%s",
                            args.package,
                            metadata_questionnaire_type or "-",
                            bool(app_intro),
                            bool(focus_hints),
                            metadata_result_path or "-",
                        )
            except Exception as e:
                metadata_notes = f"metadata_analysis_failed:{type(e).__name__}:{e}"
                logger.warning("Metadata analysis failed: %s", e)
        else:
            metadata_notes = "metadata_csv_path_not_set"

        selected_questionnaire_type = _normalize_questionnaire_type(Path(manual_questionnaire_dir).resolve().name)
        selected_questionnaire_dir = manual_questionnaire_dir

        # questionnaire_type source switch:
        # - manual: always manual dir/type
        # - metadata/auto: try metadata type, fallback manual if unavailable
        if questionnaire_type_source in {"metadata", "auto"}:
            if metadata_questionnaire_type:
                candidate_dir = Path(manual_questionnaire_dir).resolve().parent / metadata_questionnaire_type
                if candidate_dir.exists():
                    selected_questionnaire_dir = str(candidate_dir)
                    selected_questionnaire_type = metadata_questionnaire_type
                else:
                    logger.warning(
                        "Metadata questionnaire dir not found: %s. Fallback manual dir=%s",
                        candidate_dir,
                        manual_questionnaire_dir,
                    )
            else:
                logger.warning(
                    "Metadata questionnaire_type unavailable. Fallback manual dir=%s",
                    manual_questionnaire_dir,
                )

        # Load the UI-level router/block questionnaire state.
        # This replaces the old tree-state workflow; block_status is the main
        # runtime questionnaire signal.
        q = QuestionnaireState2.load_from_questionnaire_dir(selected_questionnaire_dir)
        logger.info(
            "Loaded questionnaire_state2: dir=%s type=%s source=%s routers=%d blocks=%d block_status=%d",
            selected_questionnaire_dir,
            selected_questionnaire_type,
            questionnaire_type_source,
            len(q.routers),
            len(q.blocks),
            len(q.block_status),
        )

        # Init appium
        appium = AndroidAppiumClient(server_url=args.appium_url, device_name=args.device_name).init_connection()

        budget = BudgetConfig(
            time_budget_s=float(args.time_budget),
            max_actions=int(args.max_actions),
            enable_probe_return=not bool(args.disable_probe_return),
            min_candidate_score=float(args.min_candidate_score),
            per_page_probe_cap=int(args.probe_cap),
            max_workers=int(args.workers),
        )

        if args.interactive_debug:
            callbacks = InteractiveDebugCallbacks()
        elif args.trace_dir:
            callbacks = JsonlTraceCallbacks(args.trace_dir, args.run_id)
        else:
            callbacks = NoOpCallbacks()

        runner = WorkflowRunner(
            appium=appium,
            gpt=gpt,
            questionnaires=q,
            budget=budget,
            target_package=args.package,
            target_activity=args.activity,
            pause=args.pause,
            app_intro=app_intro,
            focus_hints=focus_hints,
            questionnaire_type=selected_questionnaire_type,
            questionnaire_type_source=questionnaire_type_source,
            metadata_csv_path=metadata_csv_path,
            metadata_entry_found=metadata_entry_found,
            metadata_notes=metadata_notes,
            callbacks=callbacks,
            run_id=getattr(callbacks, "run_id", args.run_id or ""),
        )

        if args.relaunch and args.package:
            appium.force_stop(args.package)

        run_started_at = time.time()
        run_elapsed_s = 0.0
        try:
            runner.run(task=args.task)
        finally:
            run_elapsed_s = float(time.time() - run_started_at)
            usage_summary = {}
            try:
                usage_summary = gpt.usage_summary() if hasattr(gpt, "usage_summary") else {}
            except Exception:
                usage_summary = {}

            logger.info("Run finished: elapsed_s=%.3f", run_elapsed_s)
            logger.info("LLM token usage summary: %s", json.dumps(usage_summary, ensure_ascii=False))
            print(f"[RUN] elapsed_seconds={run_elapsed_s:.3f}")
            if usage_summary:
                print(
                    "[TOKENS] calls={calls} prompt={prompt} completion={completion} total={total}".format(
                        calls=int(usage_summary.get("calls", 0) or 0),
                        prompt=int(usage_summary.get("prompt_tokens", 0) or 0),
                        completion=int(usage_summary.get("completion_tokens", 0) or 0),
                        total=int(usage_summary.get("total_tokens", 0) or 0),
                    )
                )
                by_op = usage_summary.get("by_op", {}) or {}
                for op_name in sorted(by_op.keys()):
                    row = by_op.get(op_name, {}) or {}
                    print(
                        "[TOKENS][{op}] calls={calls} prompt={prompt} completion={completion} total={total}".format(
                            op=op_name,
                            calls=int(row.get("calls", 0) or 0),
                            prompt=int(row.get("prompt_tokens", 0) or 0),
                            completion=int(row.get("completion_tokens", 0) or 0),
                            total=int(row.get("total_tokens", 0) or 0),
                        )
                    )

            if args.relaunch and args.package:
                appium.force_stop(args.package)
            appium.quit()
            if args.uninstall_after_run and args.package:
                from device_utils import adb_success, uninstall_package

                uninstall_result = uninstall_package(
                    args.package,
                    device_name=args.device_name,
                    adb_path=args.adb_path,
                    timeout_seconds=int(args.install_timeout),
                )
                uninstall_status = "ok" if adb_success(uninstall_result) else "failed"
                print(f"[INSTALL][post-uninstall] package={args.package} status={uninstall_status}")
            _maybe_build_interactive_visualization(args, logger, actual_run_id=getattr(runner, "run_id", "") or getattr(callbacks, "run_id", ""))

        # New workflow stores per-UI observations for later merge; there is no
        # old-style final answer export at runtime yet.
        logger.info("Final block_status:\n%s", q.block_status)

    else:
        from device_utils import (
            adb_success,
            install_apk,
            install_package_file,
            install_result_to_dict,
            list_packages,
            list_processes,
            screenshot,
            dump_ui,
            uninstall_package,
        )
        if args.cmd == "list-packages":
            for pkg in list_packages(args.device_name):
                print(pkg)
        elif args.cmd == "list-processes":
            for line in list_processes(args.device_name):
                print(line)
        elif args.cmd == "screenshot":
            screenshot(args.out, args.device_name)
            print(f"Saved screenshot to {args.out}")
        elif args.cmd == "dump-ui":
            dump_ui(args.out, args.device_name)
            print(f"Saved UI XML to {args.out}")
        elif args.cmd == "install-apk":
            install_apk(args.apk, args.device_name)
            print("APK installed")
        elif args.cmd == "install-package":
            package_result = install_package_file(
                args.app_file,
                args.package,
                device_name=args.device_name,
                adb_path=args.adb_path,
                temp_dir=str(args.temp_dir or "") or None,
                keep_temp=bool(args.keep_temp),
                timeout_seconds=int(args.timeout_seconds),
                launch_after_install=bool(args.launch_after_install),
                validate_launch_foreground=not bool(args.no_foreground_check),
                launch_wait_seconds=float(args.launch_wait_seconds),
                uninstall_after_test=bool(args.uninstall_after_test),
            )
            output_dir = Path(args.output_dir) / (time.strftime("%Y%m%d_%H%M%S") + "_" + _safe_report_name(args.package))
            report_path = _write_install_json_report(package_result, output_dir, "install_result.json")
            payload = install_result_to_dict(package_result)
            print(
                "[INSTALL] package={pkg} install={install} verify={verify} launch={launch} foreground={fg} uninstall={uninstall}".format(
                    pkg=args.package,
                    install=payload.get("install_status") or "-",
                    verify=payload.get("verify_installed_status") or "-",
                    launch=payload.get("launch_status") or "-",
                    fg=payload.get("launch_foreground_package") or "-",
                    uninstall=payload.get("uninstall_status") or "-",
                )
            )
            print(f"[REPORT] {report_path}")
            if package_result.install_status != "ok" or (
                args.launch_after_install and package_result.launch_status != "ok"
            ):
                return 1
        elif args.cmd == "batch-install-smoke":
            apk_dir = Path(args.apk_dir)
            if not apk_dir.exists():
                print(f"[BATCH] failed: apk dir not found: {apk_dir}")
                return 2
            files = _discover_install_package_files(apk_dir, args.package_filter)
            if args.limit and int(args.limit) > 0:
                files = files[: int(args.limit)]
            output_dir = Path(args.output_dir) / ("batch_" + time.strftime("%Y%m%d_%H%M%S"))
            rows: list[dict[str, Any]] = []
            ok_count = 0
            for index, app_file in enumerate(files, start=1):
                package = app_file.stem
                row_prefix: dict[str, Any] = {
                    "package": package,
                    "app_file": str(app_file),
                    "file_type": app_file.suffix.lower().lstrip("."),
                    "uninstall_before_status": "",
                    "uninstall_after_status": "",
                }
                if args.uninstall_before_install:
                    pre_uninstall = uninstall_package(
                        package,
                        device_name=args.device_name,
                        adb_path=args.adb_path,
                        timeout_seconds=int(args.timeout_seconds),
                    )
                    row_prefix["uninstall_before_status"] = "ok" if adb_success(pre_uninstall) else "failed"

                package_result = install_package_file(
                    str(app_file),
                    package,
                    device_name=args.device_name,
                    adb_path=args.adb_path,
                    temp_dir=str(args.temp_dir or "") or None,
                    keep_temp=bool(args.keep_temp),
                    timeout_seconds=int(args.timeout_seconds),
                    launch_after_install=bool(args.launch_after_install),
                    validate_launch_foreground=not bool(args.no_foreground_check),
                    launch_wait_seconds=float(args.launch_wait_seconds),
                    uninstall_after_test=bool(args.uninstall_after_test),
                )
                result_row = {**install_result_to_dict(package_result), **row_prefix}
                result_row["uninstall_after_status"] = package_result.uninstall_status
                rows.append(result_row)
                if package_result.install_status == "ok" and (not args.launch_after_install or package_result.launch_status == "ok"):
                    ok_count += 1
                csv_path, jsonl_path = _write_batch_install_reports(rows, output_dir)
                print(
                    "[BATCH] {idx}/{total} package={pkg} install={install} launch={launch} report={report}".format(
                        idx=index,
                        total=len(files),
                        pkg=package,
                        install=package_result.install_status,
                        launch=package_result.launch_status or "-",
                        report=csv_path,
                    )
                )
            csv_path, jsonl_path = _write_batch_install_reports(rows, output_dir)
            failed_count = len(rows) - ok_count
            print(f"[BATCH] done total={len(rows)} ok={ok_count} failed={failed_count}")
            print(f"[REPORT] csv={csv_path}")
            print(f"[REPORT] jsonl={jsonl_path}")
            if failed_count:
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
