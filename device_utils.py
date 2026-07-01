"""ADB device utilities for install, launch, uninstall, and capture workflows.

Inputs:
- Optional Android device serials such as ``127.0.0.1:7555``.
- Local APK/XAPK files and Android package names.

Outputs:
- Lightweight capture files for screenshot/UI XML helpers.
- Structured command and install results for CLI reporting.

Function:
- Keep all low-level ADB package management helpers in one existing module so
  main.py and workflow code do not need to duplicate install logic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import adbutils


@dataclass
class CommandResult:
    """Result of one adb command.

    Inputs:
    - method: Human-readable operation name.
    - command: Full subprocess command list.
    - returncode/stdout/stderr: Raw subprocess outcome.

    Output:
    - Structured record used by install reports and diagnostics.
    """

    method: str
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class InstallResult:
    """Result of one APK/XAPK install verification workflow.

    Inputs:
    - package: Android package id being installed.
    - app_file: Local APK/XAPK file path.

    Output:
    - Summary fields plus raw adb attempts for CSV/JSON reporting.
    """

    package: str
    app_file: str
    file_type: str
    install_status: str = "failed"
    successful_method: str = ""
    verify_installed_status: str = ""
    launch_status: str = ""
    launch_foreground_package: str = ""
    launch_error: str = ""
    uninstall_status: str = ""
    uninstall_error: str = ""
    obb_push_status: str = ""
    extraction_error: str = ""
    duration_seconds: float = 0.0
    attempts: list[CommandResult] = field(default_factory=list)


def _get_device(device_name: Optional[str] = None):
    """Return an adbutils device object.

    Inputs:
    - device_name: Optional adb serial. If omitted, adbutils selects default.

    Output:
    - adbutils device object.
    """

    return adbutils.AdbClient().device(device_name) if device_name else adbutils.adb.device()


def list_packages(device_name: Optional[str] = None) -> List[str]:
    """List installed Android packages on a device.

    Inputs:
    - device_name: Optional adb serial.

    Output:
    - Sorted package name list.
    """

    dev = _get_device(device_name)
    return sorted(dev.list_packages())


def list_processes(device_name: Optional[str] = None) -> List[str]:
    """List running Android processes.

    Inputs:
    - device_name: Optional adb serial.

    Output:
    - Lines from ``ps -A``.
    """

    dev = _get_device(device_name)
    out = dev.shell("ps -A")
    return out.strip().splitlines()


def install_apk(apk_path: str, device_name: Optional[str] = None):
    """Install a regular APK with adbutils for backward compatibility.

    Inputs:
    - apk_path: Local APK path.
    - device_name: Optional adb serial.

    Output:
    - None. Raises if adbutils install fails.
    """

    if not os.path.exists(apk_path):
        raise FileNotFoundError(apk_path)
    dev = _get_device(device_name)
    dev.install(apk_path, clean=True)


def screenshot(path: str, device_name: Optional[str] = None):
    """Capture a screenshot to disk.

    Inputs:
    - path: Output image path.
    - device_name: Optional adb serial.

    Output:
    - None. Writes an image file.
    """

    dev = _get_device(device_name)
    img = dev.screenshot()
    img.save(path)


def dump_ui(path: str, device_name: Optional[str] = None):
    """Dump current Android UI XML to disk.

    Inputs:
    - path: Output XML path.
    - device_name: Optional adb serial.

    Output:
    - None. Writes an XML file.
    """

    dev = _get_device(device_name)
    if hasattr(dev, "dump_hierarchy"):
        xml = dev.dump_hierarchy()
    else:
        xml = dev.shell(["uiautomator", "dump", "/dev/tty"])
        marker = "<?xml"
        if marker in xml:
            xml = xml[xml.index(marker) :]
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def adb_command_prefix(adb_path: str = "adb", device_name: Optional[str] = None) -> list[str]:
    """Build the adb command prefix.

    Inputs:
    - adb_path: adb executable path or command name.
    - device_name: Optional adb serial.

    Output:
    - Command prefix such as ``["adb", "-s", "127.0.0.1:7555"]``.
    """

    prefix = [adb_path]
    if device_name:
        prefix.extend(["-s", str(device_name)])
    return prefix


def run_adb_command(
    method: str,
    args: list[str],
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 180,
) -> CommandResult:
    """Run one adb command and capture output.

    Inputs:
    - method: Short label for this operation.
    - args: Arguments after adb and optional ``-s`` serial.
    - device_name: Optional adb serial.
    - adb_path: adb executable.
    - timeout_seconds: Maximum wait time.

    Output:
    - CommandResult containing return code, stdout, and stderr.
    """

    command = adb_command_prefix(adb_path, device_name) + list(args)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout_seconds),
        )
        return CommandResult(method, command, completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(method, command, 124, stdout, stderr + f"\nTIMEOUT after {timeout_seconds}s")
    except OSError as exc:
        return CommandResult(method, command, 127, "", str(exc))


def adb_success(result: CommandResult) -> bool:
    """Return whether an adb command looks successful.

    Inputs:
    - result: CommandResult from run_adb_command().

    Output:
    - True when return code is zero and output has no common install failure marker.
    """

    combined = f"{result.stdout}\n{result.stderr}".lower()
    failure_markers = ["failure [", "failed to install", "error:"]
    return result.returncode == 0 and not any(marker in combined for marker in failure_markers)


def command_to_text(command: list[str]) -> str:
    """Format a command list as readable text.

    Inputs:
    - command: Command argument list.

    Output:
    - Space-joined command text with simple quotes around paths containing spaces.
    """

    parts: list[str] = []
    for item in command:
        text = str(item)
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


def clip_text(value: str | None, max_chars: int = 1200) -> str:
    """Trim long diagnostic text.

    Inputs:
    - value: Raw text.
    - max_chars: Maximum characters to keep.

    Output:
    - Trimmed string, optionally suffixed with a truncation marker.
    """

    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def command_result_to_dict(result: CommandResult) -> dict[str, Any]:
    """Convert one CommandResult to a JSON-serializable dictionary.

    Inputs:
    - result: CommandResult instance.

    Output:
    - Dict safe for json/csv report fields.
    """

    return {
        "method": result.method,
        "command": command_to_text(result.command),
        "returncode": result.returncode,
        "stdout": clip_text(result.stdout),
        "stderr": clip_text(result.stderr),
    }


def install_result_to_dict(result: InstallResult) -> dict[str, Any]:
    """Convert InstallResult to a JSON-serializable dictionary.

    Inputs:
    - result: InstallResult instance.

    Output:
    - Dict with summary fields and compact command attempts.
    """

    return {
        "package": result.package,
        "app_file": result.app_file,
        "file_type": result.file_type,
        "install_status": result.install_status,
        "successful_method": result.successful_method,
        "verify_installed_status": result.verify_installed_status,
        "launch_status": result.launch_status,
        "launch_foreground_package": result.launch_foreground_package,
        "launch_error": result.launch_error,
        "uninstall_status": result.uninstall_status,
        "uninstall_error": result.uninstall_error,
        "obb_push_status": result.obb_push_status,
        "extraction_error": result.extraction_error,
        "duration_seconds": f"{result.duration_seconds:.2f}",
        "attempts_json": json.dumps(
            [command_result_to_dict(item) for item in result.attempts],
            ensure_ascii=False,
        ),
    }


def find_base_apk(apk_files: list[Path]) -> Path:
    """Choose the most likely base APK from extracted XAPK files.

    Inputs:
    - apk_files: APK files discovered inside an XAPK.

    Output:
    - Path to the likely base APK.
    """

    for apk_file in apk_files:
        lower_name = apk_file.name.lower()
        if lower_name == "base.apk" or lower_name.endswith(".base.apk"):
            return apk_file
    non_config = [path for path in apk_files if not path.name.lower().startswith("config.")]
    return sorted(non_config or apk_files, key=lambda path: path.name.lower())[0]


def order_split_apks(apk_files: list[Path]) -> list[Path]:
    """Order split APKs with base APK first.

    Inputs:
    - apk_files: APK paths discovered inside an XAPK.

    Output:
    - Ordered APK list suitable for adb install-multiple.
    """

    base_apk = find_base_apk(apk_files)
    rest = sorted([path for path in apk_files if path != base_apk], key=lambda path: path.name.lower())
    return [base_apk] + rest


def extract_xapk(xapk_path: Path, temp_root: Path) -> Path:
    """Extract one XAPK archive.

    Inputs:
    - xapk_path: Local XAPK file.
    - temp_root: Directory where a temporary extraction directory is created.

    Output:
    - Extraction directory path.
    """

    extract_dir = Path(tempfile.mkdtemp(prefix=f"{xapk_path.stem}_", dir=temp_root))
    with zipfile.ZipFile(xapk_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


def discover_obb_files(extract_dir: Path) -> list[Path]:
    """Find OBB files under an extracted XAPK directory.

    Inputs:
    - extract_dir: XAPK extraction directory.

    Output:
    - Sorted list of OBB file paths.
    """

    return sorted(extract_dir.rglob("*.obb"), key=lambda path: str(path).lower())


def push_obb_files(
    package: str,
    extract_dir: Path,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 180,
) -> str:
    """Push XAPK OBB files to the Android device.

    Inputs:
    - package: Android package id used as OBB target directory.
    - extract_dir: XAPK extraction directory.
    - device_name/adb_path/timeout_seconds: ADB execution settings.

    Output:
    - Compact status string such as ``no_obb`` or ``obb_pushed:1``.
    """

    obb_files = discover_obb_files(extract_dir)
    if not obb_files:
        return "no_obb"

    target_dir = f"/sdcard/Android/obb/{package}/"
    mkdir_result = run_adb_command(
        "obb_mkdir",
        ["shell", "mkdir", "-p", target_dir],
        device_name=device_name,
        adb_path=adb_path,
        timeout_seconds=timeout_seconds,
    )
    if not adb_success(mkdir_result):
        return f"obb_mkdir_failed:{clip_text(mkdir_result.stderr or mkdir_result.stdout, 300)}"

    pushed = 0
    failures: list[str] = []
    for obb_file in obb_files:
        push_result = run_adb_command(
            "obb_push",
            ["push", str(obb_file), target_dir],
            device_name=device_name,
            adb_path=adb_path,
            timeout_seconds=timeout_seconds,
        )
        if adb_success(push_result):
            pushed += 1
        else:
            failures.append(clip_text(push_result.stderr or push_result.stdout, 300))
    if failures:
        return f"obb_push_partial:pushed={pushed},failed={len(failures)},first_error={failures[0]}"
    return f"obb_pushed:{pushed}"


def install_apk_file(
    apk_path: Path,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 180,
) -> tuple[bool, str, list[CommandResult]]:
    """Install a regular APK with several adb variants.

    Inputs:
    - apk_path: Local APK file.
    - device_name/adb_path/timeout_seconds: ADB execution settings.

    Output:
    - Tuple of success flag, successful method name, and all attempts.
    """

    attempts: list[CommandResult] = []
    install_variants = [
        ("apk_install", ["install", "-r", str(apk_path)]),
        ("apk_install_allow_downgrade", ["install", "-r", "-d", str(apk_path)]),
        ("apk_install_grant_permissions", ["install", "-r", "-g", str(apk_path)]),
        ("apk_install_downgrade_grant_permissions", ["install", "-r", "-d", "-g", str(apk_path)]),
    ]
    for method, args in install_variants:
        result = run_adb_command(method, args, device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
        attempts.append(result)
        if adb_success(result):
            return True, method, attempts
    return False, "", attempts


def install_xapk_file(
    xapk_path: Path,
    package: str,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    temp_dir: str | None = None,
    keep_temp: bool = False,
    timeout_seconds: int = 180,
) -> tuple[bool, str, list[CommandResult], str, str]:
    """Install an XAPK by extracting split APKs and optional OBB files.

    Inputs:
    - xapk_path: Local XAPK file.
    - package: Android package id.
    - device_name/adb_path/temp_dir/keep_temp/timeout_seconds: Runtime settings.

    Output:
    - Tuple of success flag, method, attempts, OBB status, extraction error.
    """

    attempts: list[CommandResult] = []
    temp_root = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    extract_dir: Path | None = None
    obb_push_status = ""
    extraction_error = ""
    try:
        extract_dir = extract_xapk(xapk_path, temp_root)
        apk_files = sorted(extract_dir.rglob("*.apk"), key=lambda path: str(path).lower())
        if not apk_files:
            return False, "", attempts, "", "no APK files found inside XAPK"
        ordered_apks = order_split_apks(apk_files)
        base_apk = ordered_apks[0]
        install_variants = [
            ("xapk_install_multiple", ["install-multiple", "-r"] + [str(path) for path in ordered_apks]),
            ("xapk_install_multiple_allow_downgrade", ["install-multiple", "-r", "-d"] + [str(path) for path in ordered_apks]),
            ("xapk_install_multiple_grant_permissions", ["install-multiple", "-r", "-g"] + [str(path) for path in ordered_apks]),
            ("xapk_base_apk_only", ["install", "-r", str(base_apk)]),
            ("xapk_base_apk_only_allow_downgrade", ["install", "-r", "-d", str(base_apk)]),
        ]
        for method, args in install_variants:
            result = run_adb_command(method, args, device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
            attempts.append(result)
            if adb_success(result):
                obb_push_status = push_obb_files(
                    package,
                    extract_dir,
                    device_name=device_name,
                    adb_path=adb_path,
                    timeout_seconds=timeout_seconds,
                )
                return True, method, attempts, obb_push_status, extraction_error
        return False, "", attempts, obb_push_status, extraction_error
    except zipfile.BadZipFile as exc:
        extraction_error = f"bad XAPK zip: {exc}"
        return False, "", attempts, obb_push_status, extraction_error
    finally:
        if extract_dir and extract_dir.exists() and not keep_temp:
            shutil.rmtree(extract_dir, ignore_errors=True)


def verify_package_installed(
    package: str,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 60,
) -> CommandResult:
    """Check whether a package is installed with pm path.

    Inputs:
    - package: Android package id.
    - device_name/adb_path/timeout_seconds: ADB execution settings.

    Output:
    - CommandResult from ``adb shell pm path``.
    """

    return run_adb_command(
        "verify_pm_path",
        ["shell", "pm", "path", package],
        device_name=device_name,
        adb_path=adb_path,
        timeout_seconds=timeout_seconds,
    )


def uninstall_package(
    package: str,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 120,
) -> CommandResult:
    """Uninstall one package from a device.

    Inputs:
    - package: Android package id.
    - device_name/adb_path/timeout_seconds: ADB execution settings.

    Output:
    - CommandResult from ``adb uninstall``.
    """

    return run_adb_command(
        "uninstall",
        ["uninstall", package],
        device_name=device_name,
        adb_path=adb_path,
        timeout_seconds=timeout_seconds,
    )


def launch_package(
    package: str,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 60,
) -> CommandResult:
    """Launch a package through Android launcher intent.

    Inputs:
    - package: Android package id.
    - device_name/adb_path/timeout_seconds: ADB execution settings.

    Output:
    - CommandResult from ``adb shell monkey``.
    """

    return run_adb_command(
        "launch_monkey",
        ["shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1"],
        device_name=device_name,
        adb_path=adb_path,
        timeout_seconds=timeout_seconds,
    )


def _extract_package_from_dumpsys(text: str) -> str:
    """Extract a likely foreground package from dumpsys text.

    Inputs:
    - text: Raw dumpsys output.

    Output:
    - Package name string, or empty string if not found.
    """

    for line in text.splitlines():
        if any(marker in line for marker in ("mCurrentFocus", "mFocusedApp", "topResumedActivity", "ResumedActivity")):
            match = re.search(r"\b([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)/", line)
            if match:
                return match.group(1)
    return ""


def get_foreground_package(
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    timeout_seconds: int = 60,
) -> str:
    """Best-effort query of the current foreground package.

    Inputs:
    - device_name/adb_path/timeout_seconds: ADB execution settings.

    Output:
    - Foreground package name, or empty string if it cannot be parsed.
    """

    window_result = run_adb_command(
        "foreground_dumpsys_window",
        ["shell", "dumpsys", "window"],
        device_name=device_name,
        adb_path=adb_path,
        timeout_seconds=timeout_seconds,
    )
    package = _extract_package_from_dumpsys(f"{window_result.stdout}\n{window_result.stderr}")
    if package:
        return package

    activity_result = run_adb_command(
        "foreground_dumpsys_activity",
        ["shell", "dumpsys", "activity", "activities"],
        device_name=device_name,
        adb_path=adb_path,
        timeout_seconds=timeout_seconds,
    )
    return _extract_package_from_dumpsys(f"{activity_result.stdout}\n{activity_result.stderr}")


def install_package_file(
    app_file: str,
    package: str,
    device_name: Optional[str] = None,
    adb_path: str = "adb",
    temp_dir: str | None = None,
    keep_temp: bool = False,
    timeout_seconds: int = 180,
    launch_after_install: bool = False,
    validate_launch_foreground: bool = True,
    launch_wait_seconds: float = 5.0,
    uninstall_after_test: bool = False,
) -> InstallResult:
    """Install one APK/XAPK and optionally verify launch/uninstall.

    Inputs:
    - app_file: Local APK/XAPK path.
    - package: Android package id.
    - device_name: Optional adb serial.
    - adb_path/temp_dir/keep_temp/timeout_seconds: ADB and XAPK settings.
    - launch_after_install: Whether to launch after installing.
    - validate_launch_foreground: Whether foreground package must match target package.
    - launch_wait_seconds: Seconds to wait after launch before foreground check.
    - uninstall_after_test: Whether to uninstall after verification.

    Output:
    - InstallResult containing summary and adb command attempts.
    """

    started = time.perf_counter()
    path = Path(app_file)
    suffix = path.suffix.lower()
    result = InstallResult(package=package, app_file=str(path), file_type=suffix.lstrip("."))
    if not path.exists():
        result.extraction_error = f"file_not_found:{path}"
        result.duration_seconds = time.perf_counter() - started
        return result
    if suffix not in {".apk", ".xapk"}:
        result.extraction_error = f"unsupported_file_type:{suffix}"
        result.duration_seconds = time.perf_counter() - started
        return result

    if suffix == ".apk":
        success, method, attempts = install_apk_file(path, device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
        result.attempts.extend(attempts)
    else:
        success, method, attempts, obb_status, extraction_error = install_xapk_file(
            path,
            package,
            device_name=device_name,
            adb_path=adb_path,
            temp_dir=temp_dir,
            keep_temp=keep_temp,
            timeout_seconds=timeout_seconds,
        )
        result.attempts.extend(attempts)
        result.obb_push_status = obb_status
        result.extraction_error = extraction_error

    result.install_status = "ok" if success else "failed"
    result.successful_method = method
    if success:
        verify_result = verify_package_installed(package, device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
        result.attempts.append(verify_result)
        result.verify_installed_status = "ok" if adb_success(verify_result) and verify_result.stdout.strip() else "failed"

    if success and launch_after_install:
        launch_result = launch_package(package, device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
        result.attempts.append(launch_result)
        if adb_success(launch_result):
            time.sleep(max(0.0, float(launch_wait_seconds)))
            if validate_launch_foreground:
                foreground_package = get_foreground_package(device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
                result.launch_foreground_package = foreground_package
                result.launch_status = "ok" if foreground_package == package else "failed"
                if result.launch_status != "ok":
                    result.launch_error = f"foreground_package_mismatch:{foreground_package or '-'}"
            else:
                result.launch_status = "ok"
        else:
            result.launch_status = "failed"
            result.launch_error = clip_text(launch_result.stderr or launch_result.stdout)

    if success and uninstall_after_test:
        uninstall_result = uninstall_package(package, device_name=device_name, adb_path=adb_path, timeout_seconds=timeout_seconds)
        result.attempts.append(uninstall_result)
        result.uninstall_status = "ok" if adb_success(uninstall_result) else "failed"
        if result.uninstall_status != "ok":
            result.uninstall_error = clip_text(uninstall_result.stderr or uninstall_result.stdout)

    result.duration_seconds = time.perf_counter() - started
    return result


__all__ = [
    "CommandResult",
    "InstallResult",
    "adb_success",
    "clip_text",
    "command_result_to_dict",
    "command_to_text",
    "dump_ui",
    "get_foreground_package",
    "install_apk",
    "install_package_file",
    "install_result_to_dict",
    "launch_package",
    "list_packages",
    "list_processes",
    "run_adb_command",
    "screenshot",
    "uninstall_package",
    "verify_package_installed",
]
