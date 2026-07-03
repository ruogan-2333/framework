"""Sequential benchmark runner for APK/XAPK app datasets.

Inputs:
- An app-list CSV containing package ids such as ``APID``.
- A metadata CSV consumed by ``main.py --metadata-csv-path``.
- An APK directory containing ``<package>.apk`` or ``<package>.xapk`` files.

Outputs:
- A batch directory under ``test_debug/batch_runs`` containing ``apps.csv``,
  ``batch_summary.csv``, ``batch_summary.json``, and per-app logs.
- Per-app workflow traces under the configured trace root.

Function:
- Run ``main.py run`` sequentially for reproducible long-running experiments,
  with install-before-run and uninstall-after-run enabled by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_LIST_CSV = PROJECT_ROOT / "APP_csv" / "known_dataset_200.csv"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "APP_csv" / "known_dataset_200_metadata.csv"
DEFAULT_APK_DIR = Path(r"F:\workplace\data_fetch_5_18\apkdownload_known200\APK")
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "traces"
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "test_debug" / "batch_runs"
DEFAULT_QUESTIONNAIRE_ROOT = PROJECT_ROOT / "questionnaire-UI"


@dataclass
class BenchApp:
    """One app target selected for a benchmark batch.

    Inputs:
    - package: Android package id.
    - app_name: Human-readable app name.
    - questionnaire_type: Optional known questionnaire type, otherwise auto.
    - app_file: Local APK/XAPK path.
    - global_index: 1-based index in the sorted app-list CSV.
    - row: Original CSV row for audit output.

    Output:
    - Structured target consumed by command construction.
    """

    package: str
    app_name: str
    questionnaire_type: str
    app_file: Path
    global_index: int
    row: Dict[str, str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner.

    Inputs:
    - argv: Optional argument list. Uses process argv when omitted.

    Output:
    - argparse Namespace with runner settings.
    """

    parser = argparse.ArgumentParser(description="Run a reproducible APK/XAPK benchmark batch through main.py.")
    parser.add_argument("--app-list-csv", default=str(DEFAULT_APP_LIST_CSV), help="CSV containing APID/app rows.")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV), help="Metadata CSV passed to main.py.")
    parser.add_argument("--apk-dir", default=str(DEFAULT_APK_DIR), help="Directory containing APK/XAPK files.")
    parser.add_argument("--device-name", default="127.0.0.1:7555", help="ADB/Appium device name.")
    parser.add_argument("--appium-url", default="http://127.0.0.1:4723", help="Appium server URL.")
    parser.add_argument("--questionnaire-root", default=str(DEFAULT_QUESTIONNAIRE_ROOT), help="Questionnaire root directory.")
    parser.add_argument("--questionnaire-type", default="auto", choices=["auto", "games", "social_apps", "others"], help="Questionnaire strategy.")
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT), help="Trace output root.")
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT), help="Batch report output root.")
    parser.add_argument("--main-path", default=str(PROJECT_ROOT / "main.py"), help="Path to main.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run main.py.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based app index in app-list CSV after sorting.")
    parser.add_argument("--limit", type=int, default=50, help="Number of apps to run; 0 means all remaining.")
    parser.add_argument("--batch-label", default="bench_known200", help="Batch directory name prefix.")
    parser.add_argument("--time-budget", type=float, default=600.0, help="Seconds per app.")
    parser.add_argument("--max-actions", type=int, default=9999, help="Max actions per app.")
    parser.add_argument("--workers", type=int, default=1, help="Forwarded LLM worker count.")
    parser.add_argument("--timeout", type=int, default=120, help="LLM request timeout forwarded to main.py.")
    parser.add_argument("--probe-cap", type=int, default=10, help="Forwarded probe cap.")
    parser.add_argument("--min-candidate-score", type=float, default=0.0, help="Forwarded min candidate score.")
    parser.add_argument("--model", default="", help="Optional model override.")
    parser.add_argument("--no-install", action="store_true", help="Do not pass install-before-run flags.")
    parser.add_argument("--keep-installed", action="store_true", help="Do not uninstall after each app run.")
    parser.add_argument("--no-auto-visualize", action="store_true", help="Do not build interactive HTML after each app.")
    parser.add_argument("--enable-probe-return", action="store_true", help="Do not pass --disable-probe-return.")
    parser.add_argument("--verbose-console", action="store_true", help="Forward verbose console logging.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands and summaries without running apps.")
    return parser.parse_args(argv)


def safe_token(text: str, max_len: int = 160) -> str:
    """Make a filesystem-safe token.

    Inputs:
    - text: Arbitrary package/run label text.
    - max_len: Maximum returned length.

    Output:
    - String containing only alphanumeric, dot, underscore, and dash characters.
    """

    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text or "").strip())
    return (token or "unknown")[:max_len]


def safe_console_text(value: object) -> str:
    """Convert text to the current console encoding safely.

    Inputs:
    - value: Any object intended for console output.

    Output:
    - Printable string with unsupported characters replaced.
    """

    text = str(value or "")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def read_csv_rows(path: Path) -> list[Dict[str, str]]:
    """Read a CSV with tolerant UTF encodings.

    Inputs:
    - path: CSV file path.

    Output:
    - List of row dictionaries.
    """

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV: {path} ({last_error})")


def row_package(row: Dict[str, str]) -> str:
    """Extract package id from an app-list row.

    Inputs:
    - row: CSV row with APID/package-like columns.

    Output:
    - Android package id, or an empty string.
    """

    for key in ("APID", "appId", "package", "package_name", "packageName", "app_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def infer_questionnaire_type(row: Dict[str, str], requested_type: str) -> str:
    """Choose the questionnaire directory name for one app.

    Inputs:
    - row: App-list CSV row.
    - requested_type: CLI setting, either auto or a fixed questionnaire type.

    Output:
    - One of games/social_apps/others.
    """

    if requested_type in {"games", "social_apps", "others"}:
        return requested_type
    is_game = str(row.get("is_game") or "").strip().lower()
    app_category = str(row.get("application_category") or row.get("category_code") or "").strip().upper()
    category_name = str(row.get("category_name") or "").strip().lower()
    if is_game == "true" or app_category.startswith("GAME_"):
        return "games"
    if any(word in category_name for word in ("social", "dating")):
        return "social_apps"
    return "others"


def find_app_file(apk_dir: Path, package: str) -> Optional[Path]:
    """Find the local APK/XAPK file for a package.

    Inputs:
    - apk_dir: Directory containing downloaded package files.
    - package: Android package id.

    Output:
    - Path to ``package.apk`` or ``package.xapk``. None if absent.
    """

    for suffix in (".xapk", ".apk"):
        candidate = apk_dir / f"{package}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_bench_apps(args: argparse.Namespace) -> list[BenchApp]:
    """Load and slice benchmark app targets.

    Inputs:
    - args: Parsed CLI settings.

    Output:
    - List of BenchApp objects selected by start-index and limit.
    """

    app_list_csv = Path(args.app_list_csv).resolve()
    apk_dir = Path(args.apk_dir).resolve()
    if not app_list_csv.exists():
        raise SystemExit(f"--app-list-csv not found: {app_list_csv}")
    if not apk_dir.exists():
        raise SystemExit(f"--apk-dir not found: {apk_dir}")

    rows = read_csv_rows(app_list_csv)
    rows.sort(key=lambda row: int(float(row.get("sample_rank") or row.get("index") or "999999")))
    start = max(1, int(args.start_index))
    indexed_rows = list(enumerate(rows, start=1))
    selected_rows = indexed_rows[start - 1 :]
    if int(args.limit) > 0:
        selected_rows = selected_rows[: int(args.limit)]

    apps: list[BenchApp] = []
    missing_files: list[str] = []
    for global_index, row in selected_rows:
        package = row_package(row)
        if not package:
            continue
        app_file = find_app_file(apk_dir, package)
        if app_file is None:
            missing_files.append(package)
            continue
        apps.append(
            BenchApp(
                package=package,
                app_name=str(row.get("app_name") or package).strip(),
                questionnaire_type=infer_questionnaire_type(row, str(args.questionnaire_type)),
                app_file=app_file,
                global_index=global_index,
                row=row,
            )
        )

    if missing_files:
        preview = ", ".join(missing_files[:10])
        print(f"[BENCH][WARN] missing APK/XAPK for {len(missing_files)} apps: {preview}")
    if not apps:
        raise SystemExit("No runnable apps selected.")
    return apps


def build_main_command(args: argparse.Namespace, app: BenchApp, run_id: str) -> list[str]:
    """Build one main.py run command.

    Inputs:
    - args: Runner CLI settings.
    - app: Selected app target.
    - run_id: Trace run id.

    Output:
    - Subprocess argv list.
    """

    questionnaire_dir = Path(args.questionnaire_root).resolve() / app.questionnaire_type
    cmd = [
        str(args.python),
        str(Path(args.main_path).resolve()),
        "run",
        "--appium-url",
        str(args.appium_url),
        "--device-name",
        str(args.device_name),
        "--package",
        app.package,
        "--questionnaire-dir",
        str(questionnaire_dir),
        "--questionnaire-type-source",
        "auto",
        "--metadata-csv-path",
        str(Path(args.metadata_csv).resolve()),
        "--trace-dir",
        str(Path(args.trace_root).resolve()),
        "--run-id",
        run_id,
        "--time-budget",
        str(float(args.time_budget)),
        "--max-actions",
        str(int(args.max_actions)),
        "--probe-cap",
        str(int(args.probe_cap)),
        "--workers",
        str(int(args.workers)),
        "--min-candidate-score",
        str(float(args.min_candidate_score)),
        "--timeout",
        str(int(args.timeout)),
        "--debug",
    ]
    if not args.no_install:
        cmd.extend(
            [
                "--app-file",
                str(app.app_file),
                "--install-before-run",
                "--uninstall-before-install",
            ]
        )
        if not args.keep_installed:
            cmd.append("--uninstall-after-run")
    if args.model:
        cmd.extend(["--model", str(args.model)])
    if not args.enable_probe_return:
        cmd.append("--disable-probe-return")
    if not args.no_auto_visualize:
        cmd.append("--auto-visualize-interactive")
    if args.verbose_console:
        cmd.append("--verbose-console")
    return cmd


def command_to_text(cmd: Sequence[str]) -> str:
    """Format a subprocess argv list for logs.

    Inputs:
    - cmd: Command tokens.

    Output:
    - Readable command string.
    """

    return " ".join(f'"{item}"' if " " in str(item) else str(item) for item in cmd)


def run_and_tee(cmd: Sequence[str], log_path: Path) -> int:
    """Run one command while teeing output to console and log file.

    Inputs:
    - cmd: Subprocess argv tokens.
    - log_path: File path receiving combined stdout/stderr.

    Output:
    - Child process exit code.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(safe_console_text(line), end="")
        return int(process.wait())


def load_analysis_summary(trace_dir: Path) -> Dict[str, object]:
    """Load compact post-run analysis metadata.

    Inputs:
    - trace_dir: One run trace directory.

    Output:
    - Dict with stop reason and graph/action counts when available.
    """

    for relative in ("graph/run_analysis_summary.json", "analysis/run_analysis_summary.json"):
        summary_path = trace_dir / relative
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"analysis_exists": True, "analysis_error": f"{type(exc).__name__}: {exc}"}
            return {
                "analysis_exists": True,
                "stop_reason": str(data.get("stop_reason") or ""),
                "node_count": data.get("graph_node_count", data.get("node_count")),
                "edge_count": data.get("graph_edge_count", data.get("edge_count")),
                "action_count": data.get("action_count"),
            }
    return {"analysis_exists": False, "stop_reason": "", "node_count": None, "edge_count": None, "action_count": None}


def write_apps_csv(path: Path, apps: Sequence[BenchApp]) -> None:
    """Write the selected benchmark app list.

    Inputs:
    - path: Output CSV path.
    - apps: Selected app targets.

    Output:
    - None. Writes CSV to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "global_index",
        "package",
        "app_name",
        "questionnaire_type",
        "app_file",
        "sample_rank",
        "category_name",
        "category_code",
        "application_category",
        "is_game",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, app in enumerate(apps, start=1):
            writer.writerow(
                {
                    "index": index,
                    "global_index": app.global_index,
                    "package": app.package,
                    "app_name": app.app_name,
                    "questionnaire_type": app.questionnaire_type,
                    "app_file": str(app.app_file),
                    "sample_rank": app.row.get("sample_rank", ""),
                    "category_name": app.row.get("category_name", ""),
                    "category_code": app.row.get("category_code", ""),
                    "application_category": app.row.get("application_category", ""),
                    "is_game": app.row.get("is_game", ""),
                }
            )


def write_summary(batch_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write batch summary files.

    Inputs:
    - batch_dir: Batch output directory.
    - rows: Per-app result rows.

    Output:
    - None. Writes JSON, CSV, and Markdown summaries.
    """

    batch_dir.mkdir(parents=True, exist_ok=True)
    json_path = batch_dir / "batch_summary.json"
    csv_path = batch_dir / "batch_summary.csv"
    md_path = batch_dir / "batch_summary.md"
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "index",
        "global_index",
        "package",
        "app_name",
        "questionnaire_type",
        "exit_code",
        "elapsed_s",
        "analysis_exists",
        "stop_reason",
        "action_count",
        "run_id",
        "trace_dir",
        "log_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Bench Run Summary",
        "",
        "| # | global# | package | app | q | exit | elapsed_s | analysis | stop_reason | actions | run_id |",
        "|---:|---:|---|---|---|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {index} | {global_index} | `{package}` | {app_name} | `{questionnaire_type}` | {exit_code} | {elapsed_s:.1f} | {analysis_exists} | {stop_reason} | {action_count} | `{run_id}` |".format(
                index=int(row.get("index") or 0),
                global_index=int(row.get("global_index") or 0),
                package=str(row.get("package") or ""),
                app_name=str(row.get("app_name") or "").replace("|", "\\|"),
                questionnaire_type=str(row.get("questionnaire_type") or ""),
                exit_code=int(row.get("exit_code") if row.get("exit_code") is not None else -999),
                elapsed_s=float(row.get("elapsed_s") or 0.0),
                analysis_exists="yes" if row.get("analysis_exists") else "no",
                stop_reason=str(row.get("stop_reason") or "").replace("|", "\\|"),
                action_count="" if row.get("action_count") is None else row.get("action_count"),
                run_id=str(row.get("run_id") or ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected benchmark batch.

    Inputs:
    - argv: Optional command-line arguments.

    Output:
    - Process exit code. The script returns 0 after completing the loop even if
      some app runs fail; failures are recorded in batch_summary files.
    """

    args = parse_args(argv)
    metadata_csv = Path(args.metadata_csv).resolve()
    if not metadata_csv.exists():
        raise SystemExit(f"--metadata-csv not found: {metadata_csv}")

    apps = load_bench_apps(args)
    batch_id = f"{safe_token(args.batch_label)}_{time.strftime('%Y%m%d_%H%M%S')}"
    batch_dir = Path(args.batch_root).resolve() / batch_id
    logs_dir = batch_dir / "logs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    write_apps_csv(batch_dir / "apps.csv", apps)

    print(f"[BENCH] apps={len(apps)} time_budget={args.time_budget}s batch_dir={batch_dir}")
    print(f"[BENCH] metadata_csv={metadata_csv}")
    print(f"[BENCH] apk_dir={Path(args.apk_dir).resolve()}")

    rows: list[Dict[str, object]] = []
    for index, app in enumerate(apps, start=1):
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_bench_i{index:03d}_g{app.global_index:03d}_{safe_token(app.package)}"
        trace_dir = Path(args.trace_root).resolve() / run_id
        log_path = logs_dir / f"{index:03d}_g{app.global_index:03d}_{safe_token(app.package)}.log"
        cmd = build_main_command(args, app, run_id)
        command_text = command_to_text(cmd)
        row: Dict[str, object] = {
            "index": index,
            "global_index": app.global_index,
            "package": app.package,
            "app_name": app.app_name,
            "questionnaire_type": app.questionnaire_type,
            "app_file": str(app.app_file),
            "run_id": run_id,
            "trace_dir": str(trace_dir),
            "log_path": str(log_path),
            "command": command_text,
            "dry_run": bool(args.dry_run),
            "exit_code": None,
            "elapsed_s": 0.0,
        }

        print(f"[BENCH][{index}/{len(apps)}] package={safe_console_text(app.package)} app={safe_console_text(app.app_name)}")
        print(f"[BENCH][CMD] {safe_console_text(command_text)}")
        start = time.time()
        if args.dry_run:
            row["exit_code"] = 0
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(command_text + "\n", encoding="utf-8")
        else:
            try:
                row["exit_code"] = run_and_tee(cmd, log_path)
            except Exception as exc:
                row["exit_code"] = -1
                row["error"] = f"{type(exc).__name__}: {exc}"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(str(row["error"]) + "\n", encoding="utf-8")
                print(f"[BENCH][ERROR] {safe_console_text(row['error'])}")
        row["elapsed_s"] = round(time.time() - start, 3)
        row.update(load_analysis_summary(trace_dir))
        rows.append(row)
        write_summary(batch_dir, rows)
        print(
            f"[BENCH][DONE] package={safe_console_text(app.package)} exit={row['exit_code']} "
            f"elapsed_s={row['elapsed_s']} analysis={row.get('analysis_exists')} stop={safe_console_text(row.get('stop_reason'))}"
        )

    failures = [row for row in rows if int(row.get("exit_code") if row.get("exit_code") is not None else -1) != 0]
    missing_analysis = [row for row in rows if not row.get("analysis_exists")]
    print(f"[BENCH] finished total={len(rows)} failures={len(failures)} missing_analysis={len(missing_analysis)}")
    print(f"[BENCH] summary={batch_dir / 'batch_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
