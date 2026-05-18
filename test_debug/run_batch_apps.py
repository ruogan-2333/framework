"""
Batch-run sampled Android apps through the main workflow.

Inputs:
  - metadata CSV containing APID, app_name, and category columns.
  - Appium/device settings and workflow budget settings from CLI flags.

Outputs:
  - test_debug/batch_runs/<batch_id>/sampled_apps.csv
  - test_debug/batch_runs/<batch_id>/batch_summary.json
  - test_debug/batch_runs/<batch_id>/batch_summary.md
  - One normal workflow trace directory per app under --trace-root.

Function:
  Deterministically samples a small set of apps, maps each app category to one
  questionnaire directory, and sequentially invokes `main.py run` for each app.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "metadata_downloaded_apps.csv"
DEFAULT_QUESTIONNAIRE_ROOT = PROJECT_ROOT / "questionnaire-UI"
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "traces"
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "test_debug" / "batch_runs"

GAMES_CATEGORIES = {
    "Action",
    "Puzzle",
    "Racing",
    "Word",
    "Music",
}

SOCIAL_CATEGORIES = {
    "Social",
    "Communication",
    "Dating",
}


@dataclass
class AppRow:
    """
    Input: one CSV row normalized by `load_app_rows`.
    Output: typed app metadata used by sampler and command builder.
    Function: keeps package, display name, category, and questionnaire type together.
    """

    package: str
    app_name: str
    category: str
    questionnaire_type: str
    source_row: Dict[str, str]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """
    Input: command-line argv.
    Output: parsed batch-run configuration.
    Function: defines stable defaults for the 10-app, 300-second smoke test.
    """

    parser = argparse.ArgumentParser(description="Sample apps from metadata CSV and run main.py sequentially.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Metadata CSV path containing APID/category.")
    parser.add_argument("--sample-size", type=int, default=10, help="Total sampled app count.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--per-app-time-budget", type=float, default=300.0, help="Seconds per app workflow run.")
    parser.add_argument("--appium-url", default="http://127.0.0.1:4723", help="Appium server URL.")
    parser.add_argument("--device-name", default="emulator-5554", help="Appium deviceName capability.")
    parser.add_argument("--questionnaire-root", default=str(DEFAULT_QUESTIONNAIRE_ROOT), help="Questionnaire root containing games/social_apps/others and optional addition.")
    parser.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT), help="Workflow trace root.")
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT), help="Batch summary output root.")
    parser.add_argument("--main-path", default=str(PROJECT_ROOT / "main.py"), help="Path to main.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to invoke main.py.")
    parser.add_argument("--min-candidate-score", type=float, default=0.0, help="Forwarded to main.py.")
    parser.add_argument("--probe-cap", type=int, default=10, help="Forwarded to main.py.")
    parser.add_argument("--workers", type=int, default=4, help="Forwarded to main.py.")
    parser.add_argument("--model", default="", help="Optional model name forwarded to main.py when set.")
    parser.add_argument("--timeout", type=int, default=60, help="LLM request timeout forwarded to main.py.")
    parser.add_argument("--no-relaunch", action="store_true", help="Do not pass --relaunch to main.py.")
    parser.add_argument("--enable-probe-return", action="store_true", help="Do not pass --disable-probe-return.")
    parser.add_argument("--auto-visualize-interactive", action="store_true", help="Forward visualization flag.")
    parser.add_argument("--verbose-console", action="store_true", help="Forward verbose console flag to main.py.")
    parser.add_argument("--dry-run", action="store_true", help="Write sample/commands without executing apps.")
    return parser.parse_args(argv)


def questionnaire_type_for_category(category: str) -> str:
    """
    Input: category string from metadata CSV.
    Output: questionnaire type directory name: games, social_apps, or others.
    Function: maps app-store categories onto the three local questionnaire packs.
    """

    clean = str(category or "").strip()
    if clean in GAMES_CATEGORIES:
        return "games"
    if clean in SOCIAL_CATEGORIES:
        return "social_apps"
    return "others"


def load_app_rows(csv_path: Path) -> List[AppRow]:
    """
    Input: metadata CSV path.
    Output: normalized AppRow list with non-empty package IDs.
    Function: reads UTF-8-SIG CSV and prepares category-to-questionnaire mapping.
    """

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    out: List[AppRow] = []
    for row in rows:
        package = str(row.get("APID") or "").strip()
        if not package:
            continue
        category = str(row.get("category") or "").strip()
        out.append(
            AppRow(
                package=package,
                app_name=str(row.get("app_name") or "").strip(),
                category=category,
                questionnaire_type=questionnaire_type_for_category(category),
                source_row=row,
            )
        )
    return out


def stratified_sample(rows: Sequence[AppRow], sample_size: int, seed: int) -> List[AppRow]:
    """
    Input: all candidate apps, desired sample size, deterministic random seed.
    Output: sampled apps in execution order.
    Function: balances games/social/others before filling any shortage from remaining apps.
    """

    rng = random.Random(int(seed))
    buckets: Dict[str, List[AppRow]] = {"games": [], "social_apps": [], "others": []}
    for row in rows:
        buckets.setdefault(row.questionnaire_type, []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    if sample_size >= 10:
        plan = {"games": 3, "social_apps": 2, "others": sample_size - 5}
    elif sample_size >= 7:
        plan = {"games": 2, "social_apps": 2, "others": sample_size - 4}
    else:
        plan = {"games": max(1, sample_size // 3), "social_apps": max(1, sample_size // 3), "others": sample_size}

    selected: List[AppRow] = []
    selected_packages = set()

    def take_from(kind: str, count: int) -> None:
        """Move up to count apps from one bucket into selected."""
        nonlocal selected
        bucket = buckets.get(kind, [])
        while bucket and count > 0 and len(selected) < sample_size:
            app = bucket.pop(0)
            if app.package in selected_packages:
                continue
            selected.append(app)
            selected_packages.add(app.package)
            count -= 1

    for kind in ("games", "social_apps", "others"):
        take_from(kind, int(plan.get(kind, 0)))

    remaining: List[AppRow] = []
    for kind in ("games", "social_apps", "others"):
        remaining.extend(buckets.get(kind, []))
    rng.shuffle(remaining)
    for app in remaining:
        if len(selected) >= sample_size:
            break
        if app.package in selected_packages:
            continue
        selected.append(app)
        selected_packages.add(app.package)

    return selected[:sample_size]


def make_safe_run_token(text: str) -> str:
    """
    Input: arbitrary package or batch text.
    Output: filesystem-safe token.
    Function: prevents package names or timestamps from producing invalid path names.
    """

    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text or ""))[:160]


def build_main_command(args: argparse.Namespace, app: AppRow, run_id: str) -> List[str]:
    """
    Input: batch args, one sampled app, and run_id.
    Output: argv list for subprocess.run.
    Function: translates batch config into one `main.py run` invocation.
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
        "--trace-dir",
        str(Path(args.trace_root).resolve()),
        "--run-id",
        run_id,
        "--time-budget",
        str(float(args.per_app_time_budget)),
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
    if args.model:
        cmd.extend(["--model", str(args.model)])
    if not args.enable_probe_return:
        cmd.append("--disable-probe-return")
    if not args.no_relaunch:
        cmd.append("--relaunch")
    if args.auto_visualize_interactive:
        cmd.append("--auto-visualize-interactive")
    if args.verbose_console:
        cmd.append("--verbose-console")
    return cmd


def write_sample_csv(path: Path, sampled: Sequence[AppRow]) -> None:
    """
    Input: output path and sampled app rows.
    Output: CSV file listing the selected apps and questionnaire mapping.
    Function: records the exact deterministic sample used for this batch.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["package", "app_name", "category", "questionnaire_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for app in sampled:
            writer.writerow(
                {
                    "package": app.package,
                    "app_name": app.app_name,
                    "category": app.category,
                    "questionnaire_type": app.questionnaire_type,
                }
            )


def write_summary(batch_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    """
    Input: batch output directory and per-app result rows.
    Output: JSON and Markdown summaries.
    Function: makes batch results readable without opening every trace directory.
    """

    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_summary.json").write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Batch App Run Summary",
        "",
        "| # | package | app | category | questionnaire | exit | elapsed_s | run_id |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {idx} | `{package}` | {app_name} | {category} | `{questionnaire_type}` | {exit_code} | {elapsed_s:.1f} | `{run_id}` |".format(
                idx=idx,
                package=str(row.get("package") or ""),
                app_name=str(row.get("app_name") or "").replace("|", "\\|"),
                category=str(row.get("category") or "").replace("|", "\\|"),
                questionnaire_type=str(row.get("questionnaire_type") or ""),
                exit_code=int(row.get("exit_code") or 0),
                elapsed_s=float(row.get("elapsed_s") or 0.0),
                run_id=str(row.get("run_id") or ""),
            )
        )
    lines.append("")
    (batch_dir / "batch_summary.md").write_text("\n".join(lines), encoding="utf-8")


def iter_commands_preview(commands: Iterable[str]) -> str:
    """
    Input: command argv tokens.
    Output: a readable one-line command string.
    Function: stores/debugs subprocess commands without shell quoting side effects.
    """

    return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in commands)


def safe_console_text(value: object) -> str:
    """
    Input: any value destined for Windows console output.
    Output: text that can be printed even when the console encoding is GBK.
    Function: prevents batch execution from crashing on app names containing emoji or trademark symbols.
    """

    text = str(value or "")
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """
    Input: optional CLI argv.
    Output: process exit code; 0 when the batch script itself completed.
    Function: samples apps and runs them sequentially through the main workflow.
    """

    args = parse_args(argv or sys.argv[1:])
    csv_path = Path(args.csv_path).resolve()
    questionnaire_root = Path(args.questionnaire_root).resolve()
    batch_id = time.strftime("%Y%m%d_%H%M%S") + f"_sample{int(args.sample_size)}_seed{int(args.seed)}"
    batch_dir = Path(args.batch_root).resolve() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    apps = load_app_rows(csv_path)
    sampled = stratified_sample(apps, int(args.sample_size), int(args.seed))
    write_sample_csv(batch_dir / "sampled_apps.csv", sampled)

    print(f"[BATCH] csv={csv_path}")
    print(f"[BATCH] apps_total={len(apps)} sampled={len(sampled)} seed={args.seed}")
    print(f"[BATCH] batch_dir={batch_dir}")

    results: List[Dict[str, object]] = []
    for idx, app in enumerate(sampled, start=1):
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{make_safe_run_token(app.package)}"
        questionnaire_dir = questionnaire_root / app.questionnaire_type
        cmd = build_main_command(args, app, run_id)
        command_text = iter_commands_preview(cmd)

        row: Dict[str, object] = {
            "index": idx,
            "package": app.package,
            "app_name": app.app_name,
            "category": app.category,
            "questionnaire_type": app.questionnaire_type,
            "questionnaire_dir": str(questionnaire_dir),
            "run_id": run_id,
            "trace_dir": str(Path(args.trace_root).resolve() / run_id),
            "command": command_text,
            "dry_run": bool(args.dry_run),
            "exit_code": None,
            "elapsed_s": 0.0,
        }
        print(
            f"[BATCH][{idx}/{len(sampled)}] "
            f"package={safe_console_text(app.package)} "
            f"app={safe_console_text(app.app_name)} "
            f"q={safe_console_text(app.questionnaire_type)}"
        )
        print(f"[BATCH][CMD] {safe_console_text(command_text)}")

        start = time.time()
        if args.dry_run:
            row["exit_code"] = 0
        else:
            try:
                completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
                row["exit_code"] = int(completed.returncode)
            except Exception as exc:
                row["exit_code"] = -1
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[BATCH][ERROR] package={safe_console_text(app.package)} error={safe_console_text(row['error'])}")
        row["elapsed_s"] = round(time.time() - start, 3)
        results.append(row)
        write_summary(batch_dir, results)
        print(f"[BATCH][DONE] package={safe_console_text(app.package)} exit={row['exit_code']} elapsed_s={row['elapsed_s']}")

    write_summary(batch_dir, results)
    failures = [r for r in results if int(r.get("exit_code") or 0) != 0]
    print(f"[BATCH] finished total={len(results)} failures={len(failures)} summary={batch_dir / 'batch_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
