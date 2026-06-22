#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_policy_document_capture.py

Input:
- The emulator is already showing a policy / terms document page.
- Appium server is running and can attach to the current Android device.
- PAGE_KIND, CAPTURE_METHOD, and COPY_STRATEGY can be configured near the top of
  this file or overridden from CLI for quick experiments.

Output:
- test_debug/policy_document_capture/<timestamp>_<page_kind>_<method>/metadata.json
- test_debug/policy_document_capture/<timestamp>_<page_kind>_<method>/screenshot_before.png
- test_debug/policy_document_capture/<timestamp>_<page_kind>_<method>/xml_before.xml
- xml_text.txt when CAPTURE_METHOD="xml"
- page.html/page_text.txt when external_web+xml finds a browser URL and downloads it
- clipboard_text.txt when CAPTURE_METHOD="copy" successfully reads clipboard

Function:
- Provides a small manual probe for policy / TOS text extraction.
- It intentionally does not launch apps, click policy entrances, classify packages, or return to the app.
- It lets us compare four manual cases:
  external_web + xml, external_web + copy, in_app_document + xml, in_app_document + copy.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from appium_android import AndroidAppiumClient


# ====================== Manual Test Config ======================
# PAGE_KIND = "external_web"
PAGE_KIND = "in_app_document"
# PAGE_KIND describes the page carrier that you manually opened in the emulator.
# Allowed values:
# - "external_web": external browser page, such as Chrome or emulator browser.
# - "in_app_document": document page rendered inside the target app.

# CAPTURE_METHOD = "xml"
CAPTURE_METHOD = "copy"
# CAPTURE_METHOD selects exactly one extraction method for this run.
# Allowed values:
# - "xml": only read Appium XML and extract text-like attributes.
# - "copy": copy text by COPY_STRATEGY and read clipboard.

COPY_STRATEGY = "keyboard_shortcut"
# COPY_STRATEGY is used only when CAPTURE_METHOD="copy".
# Allowed values:
# - "keyboard_shortcut": tap document body, send Ctrl+A, send Ctrl+C, then read clipboard.
# - "context_menu": long-press document body, tap Select all, tap Copy, then read clipboard.

BODY_X_RATIO = 0.50
BODY_Y_RATIO = 0.45
# BODY_* controls where the keyboard shortcut strategy taps before Ctrl+A / Ctrl+C.

LONG_PRESS_X_RATIO = 0.50
LONG_PRESS_Y_RATIO = 0.45
LONG_PRESS_MS = 1200
# LONG_PRESS_* controls where and how long the context-menu copy strategy presses.

SELECT_ALL_TEXTS = ["\u5168\u9009", "Select all", "SELECT ALL", "Select All"]
COPY_TEXTS = ["\u590d\u5236", "Copy", "COPY"]
# These labels are searched in Appium XML after long press to locate Android text action menu buttons.

KEYCODE_A = 29
KEYCODE_C = 31
META_CTRL_ON = 4096
# Android keycode constants used for keyboard shortcut copy probing.

REQUEST_TIMEOUT_S = 20
# REQUEST_TIMEOUT_S bounds external webpage download time in external_web+xml mode.
# ================================================================


logger = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    """
    Input: debug flag.
    Output: configures process-wide logging.
    Function: keeps Appium/Selenium logs concise while making this probe's steps visible.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)


def write_text(path: Path, text: str) -> None:
    """
    Input: output path and UTF-8 text.
    Output: writes text to disk.
    Function: centralizes text output and creates parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Input: output path and JSON-serializable dictionary.
    Output: writes pretty UTF-8 JSON to disk.
    Function: centralizes metadata output and creates parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    """
    Input: output path and binary data.
    Output: writes bytes to disk.
    Function: centralizes screenshot output and creates parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data or b"")


def make_run_dir(output_root: Path, page_kind: str, capture_method: str) -> Path:
    """
    Input: output root, page kind, and capture method.
    Output: unique timestamped run directory.
    Function: isolates each manual probe run for later comparison.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_{page_kind}_{capture_method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def normalize_text(value: str) -> str:
    """
    Input: raw UI text value.
    Output: whitespace-normalized text.
    Function: removes XML/UI whitespace noise before deduplication and saving.
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def dedupe_preserve_order(lines: Iterable[str]) -> List[str]:
    """
    Input: iterable of text lines.
    Output: de-duplicated list preserving first-seen order.
    Function: keeps extracted document text readable without sorting away page order.
    """
    seen = set()
    out: List[str] = []
    for line in lines:
        cleaned = normalize_text(line)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def extract_text_from_xml(xml_text: str) -> List[str]:
    """
    Input: UiAutomator2 XML string from Appium page_source.
    Output: de-duplicated text lines from text/content-desc/hint attributes.
    Function: checks whether policy/TOS body text is directly exposed through XML.
    """
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        logger.warning("Failed to parse XML while extracting text", exc_info=True)
        return []

    values: List[str] = []
    for elem in root.iter():
        attrs = elem.attrib or {}
        for key in ("text", "content-desc", "hint"):
            value = normalize_text(attrs.get(key, ""))
            if value:
                values.append(value)
    return dedupe_preserve_order(values)


def extract_urls_from_text(text: str) -> List[str]:
    """
    Input: arbitrary XML-derived or clipboard text.
    Output: de-duplicated http/https URLs.
    Function: records useful URL evidence when full URLs appear in text.
    """
    urls = re.findall(r"https?://[^\s\"'<>]+", text or "", flags=re.IGNORECASE)
    return dedupe_preserve_order(urls)


def extract_browser_address_from_xml(xml_text: str) -> str:
    """
    Input: UiAutomator2 XML from an external browser page.
    Output: raw address bar text, without adding scheme or normalization.
    Function: finds browser URL/address fields such as Chromium's com.android.chromium:id/url_bar.
    """
    if not xml_text:
        return ""

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        logger.warning("Failed to parse XML while extracting browser address", exc_info=True)
        return ""

    fallback = ""
    for elem in root.iter():
        attrs = elem.attrib or {}
        text = normalize_text(attrs.get("text", ""))
        hint = normalize_text(attrs.get("hint", ""))
        rid = normalize_text(attrs.get("resource-id", ""))
        cls = normalize_text(attrs.get("class", ""))

        if text and ("url_bar" in rid or "address" in hint.casefold() or "web address" in hint.casefold()):
            return text
        if text and not fallback and cls.endswith("EditText") and ("." in text or "/" in text):
            fallback = text
    return fallback


def make_download_url(raw_address: str) -> str:
    """
    Input: raw browser address, possibly without protocol.
    Output: URL usable by requests, or empty string if the address is not URL-like.
    Function: preserves raw address in metadata while adding https:// only for download execution.
    """
    value = normalize_text(raw_address)
    if not value:
        return ""
    if re.match(r"^https?://", value, flags=re.IGNORECASE):
        return value
    if " " in value or "." not in value:
        return ""
    return f"https://{value}"


def extract_text_from_html(html_text: str) -> str:
    """
    Input: raw HTML string.
    Output: readable plain text.
    Function: converts downloaded policy/TOS HTML into text for offline inspection.
    """
    if not html_text:
        return ""

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        lines = [normalize_text(line) for line in soup.get_text("\n").splitlines()]
        return "\n".join(dedupe_preserve_order(lines))

    no_script = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html_text)
    no_tags = re.sub(r"(?s)<[^>]+>", "\n", no_script)
    lines = [html.unescape(normalize_text(line)) for line in no_tags.splitlines()]
    return "\n".join(dedupe_preserve_order(lines))


def download_external_page(raw_address: str, run_dir: Path) -> Dict[str, Any]:
    """
    Input: raw browser address and output directory.
    Output: metadata dictionary for the download attempt.
    Function: downloads external policy/TOS webpage HTML and extracts text when a browser URL is available.
    """
    download_url = make_download_url(raw_address)
    result: Dict[str, Any] = {
        "browser_url_raw": raw_address,
        "download_url": download_url,
        "attempted": bool(download_url),
        "success": False,
    }
    if not download_url:
        result["error"] = "No URL-like browser address found"
        return result

    try:
        response = requests.get(
            download_url,
            timeout=REQUEST_TIMEOUT_S,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                )
            },
        )
        result["status_code"] = response.status_code
        result["final_url"] = response.url
        result["content_type"] = response.headers.get("content-type", "")
        response.raise_for_status()

        html_text = response.text or ""
        page_text = extract_text_from_html(html_text)
        html_path = run_dir / "page.html"
        text_path = run_dir / "page_text.txt"
        write_text(html_path, html_text)
        write_text(text_path, page_text)
        result.update(
            {
                "success": True,
                "html_file": str(html_path),
                "page_text_file": str(text_path),
                "html_char_count": len(html_text),
                "page_text_char_count": len(page_text),
            }
        )
    except Exception as exc:
        logger.warning("External webpage download failed: %s", download_url, exc_info=True)
        result["error"] = str(exc)
    return result


def parse_bounds(bounds: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Input: Android bounds string in the form [x1,y1][x2,y2].
    Output: tuple (x1, y1, x2, y2) or None.
    Function: converts XML action-menu node bounds into clickable screen coordinates.
    """
    matches = re.findall(r"\[(\d+),(\d+)\]", bounds or "")
    if len(matches) != 2:
        return None
    x1, y1 = map(int, matches[0])
    x2, y2 = map(int, matches[1])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def find_node_center_by_text(xml_text: str, labels: Sequence[str]) -> Optional[Tuple[int, int, str]]:
    """
    Input: current XML and acceptable visible labels.
    Output: center coordinate plus matched label, or None.
    Function: finds Android action menu items such as Select all / Copy after long press.
    """
    if not xml_text:
        return None
    wanted = {normalize_text(label).casefold() for label in labels}

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        logger.warning("Failed to parse XML while locating action menu", exc_info=True)
        return None

    for elem in root.iter():
        attrs = elem.attrib or {}
        candidates = [
            normalize_text(attrs.get("text", "")),
            normalize_text(attrs.get("content-desc", "")),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            if candidate.casefold() not in wanted:
                continue
            parsed = parse_bounds(attrs.get("bounds", ""))
            if not parsed:
                continue
            x1, y1, x2, y2 = parsed
            return int((x1 + x2) / 2), int((y1 + y2) / 2), candidate
    return None


def save_current_snapshot(appium: AndroidAppiumClient, run_dir: Path, prefix: str) -> Dict[str, Any]:
    """
    Input: live Appium client, output directory, and filename prefix.
    Output: dictionary with XML, screenshot path, XML path, package, and activity.
    Function: captures current page evidence before/after copy-related actions.
    """
    xml_text = appium.page_source_once()
    screenshot = appium.screenshot_png_once()
    screenshot_path = run_dir / f"{prefix}.png"
    xml_path = run_dir / f"{prefix}.xml"
    write_bytes(screenshot_path, screenshot)
    write_text(xml_path, xml_text)
    return {
        "xml": xml_text,
        "screenshot_path": str(screenshot_path),
        "xml_path": str(xml_path),
        "foreground_package": appium.foreground_package(),
        "foreground_activity": appium.foreground_activity(),
    }


def get_ratio_point(appium: AndroidAppiumClient, x_ratio: float, y_ratio: float) -> Tuple[int, int, Dict[str, Any]]:
    """
    Input: live Appium client and x/y screen ratios.
    Output: concrete x/y coordinates and raw window-size dictionary.
    Function: converts configured screen ratios into device coordinates.
    """
    appium._require_driver()
    size = appium.driver.get_window_size()
    width = int(size.get("width", 0) or 0)
    height = int(size.get("height", 0) or 0)
    x = int(width * float(x_ratio))
    y = int(height * float(y_ratio))
    return x, y, dict(size)


def clear_clipboard_text(appium: AndroidAppiumClient) -> Dict[str, Any]:
    """
    Input: live Appium client.
    Output: result dictionary for the clear attempt.
    Function: clears clipboard before copy probing so stale clipboard content is easier to detect.
    """
    appium._require_driver()
    try:
        appium.driver.set_clipboard_text("")
        return {"success": True, "strategy": "driver.set_clipboard_text"}
    except Exception as exc:
        logger.warning("Failed to clear clipboard before copy probe", exc_info=True)
        return {"success": False, "error": str(exc)}


def press_ctrl_shortcut(appium: AndroidAppiumClient, keycode: int, name: str) -> Dict[str, Any]:
    """
    Input: live Appium client, Android keycode, and human-readable key name.
    Output: shortcut result dictionary.
    Function: sends Ctrl+<key> to the focused Android app through Appium.
    """
    appium._require_driver()
    try:
        appium.driver.press_keycode(int(keycode), metastate=int(META_CTRL_ON))
        return {"success": True, "key": name, "strategy": "driver.press_keycode"}
    except TypeError:
        try:
            appium.driver.press_keycode(int(keycode), int(META_CTRL_ON))
            return {"success": True, "key": name, "strategy": "driver.press_keycode_positional"}
        except Exception as exc:
            logger.warning("Ctrl+%s failed", name, exc_info=True)
            return {"success": False, "key": name, "error": str(exc)}
    except Exception as exc:
        logger.warning("Ctrl+%s failed", name, exc_info=True)
        return {"success": False, "key": name, "error": str(exc)}


def perform_long_press(appium: AndroidAppiumClient, x: int, y: int) -> str:
    """
    Input: live Appium client and target coordinate.
    Output: name of the strategy that succeeded.
    Function: opens the Android text selection/action menu at the configured document body point.
    """
    appium._require_driver()
    try:
        appium.driver.execute_script(
            "mobile: longClickGesture",
            {"x": int(x), "y": int(y), "duration": int(LONG_PRESS_MS)},
        )
        return "mobile:longClickGesture"
    except Exception:
        logger.warning("mobile: longClickGesture failed; falling back to driver.tap(duration)", exc_info=True)

    appium.driver.tap([(int(x), int(y))], int(LONG_PRESS_MS))
    return "driver.tap_duration"


def tap_first_matching_label(appium: AndroidAppiumClient, xml_text: str, labels: Sequence[str]) -> Dict[str, Any]:
    """
    Input: live Appium client, current XML, and acceptable labels.
    Output: action result dictionary with found/clicked/matched label fields.
    Function: clicks a text action menu item such as Select all or Copy if it is visible in XML.
    """
    match = find_node_center_by_text(xml_text, labels)
    if not match:
        return {"found": False, "clicked": False, "matched_label": ""}

    x, y, matched_label = match
    try:
        appium.tap(x, y)
        return {"found": True, "clicked": True, "matched_label": matched_label, "x": x, "y": y}
    except Exception as exc:
        logger.warning("Failed to tap matched label %s", matched_label, exc_info=True)
        return {
            "found": True,
            "clicked": False,
            "matched_label": matched_label,
            "x": x,
            "y": y,
            "error": str(exc),
        }


def read_clipboard_text(appium: AndroidAppiumClient) -> Tuple[str, str]:
    """
    Input: live Appium client.
    Output: clipboard text and strategy name.
    Function: reads copied document text from Android clipboard through Appium.
    """
    appium._require_driver()
    text = appium.driver.get_clipboard_text()
    return text or "", "driver.get_clipboard_text"


def capture_xml_method(appium: AndroidAppiumClient, run_dir: Path, page_kind: str) -> Dict[str, Any]:
    """
    Input: live Appium client, output directory, and manually configured page kind.
    Output: metadata dictionary describing XML extraction result.
    Function: saves current page XML/screenshot and extracts document-like text from XML only.
    """
    before = save_current_snapshot(appium, run_dir, "screenshot_before")
    xml_text = before["xml"]
    lines = extract_text_from_xml(xml_text)
    joined = "\n".join(lines)
    urls = extract_urls_from_text(xml_text + "\n" + joined)
    write_text(run_dir / "xml_text.txt", joined)

    browser_address = ""
    download_result: Dict[str, Any] = {}
    if page_kind == "external_web":
        browser_address = extract_browser_address_from_xml(xml_text)
        download_result = download_external_page(browser_address, run_dir)

    return {
        "page_kind": page_kind,
        "capture_method": "xml",
        "foreground_package": before["foreground_package"],
        "foreground_activity": before["foreground_activity"],
        "xml_path": before["xml_path"],
        "screenshot_path": before["screenshot_path"],
        "line_count": len(lines),
        "text_char_count": len(joined),
        "urls": urls,
        "browser_url_raw": browser_address,
        "external_download": download_result,
        "final_text_file": str(run_dir / "xml_text.txt"),
    }


def capture_copy_keyboard_shortcut(appium: AndroidAppiumClient, run_dir: Path, page_kind: str) -> Dict[str, Any]:
    """
    Input: live Appium client, output directory, and manually configured page kind.
    Output: metadata dictionary describing keyboard shortcut copy result.
    Function: taps document body, sends Ctrl+A and Ctrl+C, then reads clipboard.
    """
    before = save_current_snapshot(appium, run_dir, "screenshot_before")
    x, y, window_size = get_ratio_point(appium, BODY_X_RATIO, BODY_Y_RATIO)

    metadata: Dict[str, Any] = {
        "page_kind": page_kind,
        "capture_method": "copy",
        "copy_strategy": "keyboard_shortcut",
        "foreground_package": before["foreground_package"],
        "foreground_activity": before["foreground_activity"],
        "xml_before_path": before["xml_path"],
        "screenshot_before_path": before["screenshot_path"],
        "window_size": window_size,
        "body_tap": {"x": x, "y": y},
    }

    try:
        appium.tap(x, y)
        time.sleep(0.5)
        after_tap = save_current_snapshot(appium, run_dir, "screenshot_after_body_tap")
        metadata["after_body_tap_xml_path"] = after_tap["xml_path"]
        metadata["after_body_tap_screenshot_path"] = after_tap["screenshot_path"]
    except Exception as exc:
        metadata["body_tap"]["error"] = str(exc)
        return metadata

    metadata["clipboard_clear"] = clear_clipboard_text(appium)
    metadata["ctrl_a"] = press_ctrl_shortcut(appium, KEYCODE_A, "A")
    time.sleep(0.8)
    after_ctrl_a = save_current_snapshot(appium, run_dir, "screenshot_after_ctrl_a")
    metadata["after_ctrl_a_xml_path"] = after_ctrl_a["xml_path"]
    metadata["after_ctrl_a_screenshot_path"] = after_ctrl_a["screenshot_path"]

    metadata["ctrl_c"] = press_ctrl_shortcut(appium, KEYCODE_C, "C")
    time.sleep(0.8)
    after_ctrl_c = save_current_snapshot(appium, run_dir, "screenshot_after_ctrl_c")
    metadata["after_ctrl_c_xml_path"] = after_ctrl_c["xml_path"]
    metadata["after_ctrl_c_screenshot_path"] = after_ctrl_c["screenshot_path"]

    try:
        clipboard_text, clipboard_strategy = read_clipboard_text(appium)
        write_text(run_dir / "clipboard_text.txt", clipboard_text)
        metadata["clipboard"] = {
            "read_success": True,
            "strategy": clipboard_strategy,
            "text_char_count": len(clipboard_text or ""),
            "text_file": str(run_dir / "clipboard_text.txt"),
            "urls": extract_urls_from_text(clipboard_text),
        }
    except Exception as exc:
        logger.warning("Failed to read clipboard text", exc_info=True)
        metadata["clipboard"] = {"read_success": False, "error": str(exc)}
    return metadata


def capture_copy_context_menu(appium: AndroidAppiumClient, run_dir: Path, page_kind: str) -> Dict[str, Any]:
    """
    Input: live Appium client, output directory, and manually configured page kind.
    Output: metadata dictionary describing context-menu copy result.
    Function: tries long-press, Select all, Copy, and clipboard read without XML text fallback.
    """
    before = save_current_snapshot(appium, run_dir, "screenshot_before")
    x, y, window_size = get_ratio_point(appium, LONG_PRESS_X_RATIO, LONG_PRESS_Y_RATIO)

    metadata: Dict[str, Any] = {
        "page_kind": page_kind,
        "capture_method": "copy",
        "copy_strategy": "context_menu",
        "foreground_package": before["foreground_package"],
        "foreground_activity": before["foreground_activity"],
        "xml_before_path": before["xml_path"],
        "screenshot_before_path": before["screenshot_path"],
        "window_size": window_size,
        "long_press": {"x": x, "y": y, "duration_ms": LONG_PRESS_MS},
    }

    metadata["clipboard_clear"] = clear_clipboard_text(appium)

    try:
        strategy = perform_long_press(appium, x, y)
        metadata["long_press"]["strategy"] = strategy
    except Exception as exc:
        metadata["long_press"]["error"] = str(exc)
        return metadata

    time.sleep(0.8)
    after_press = save_current_snapshot(appium, run_dir, "screenshot_after_long_press")
    metadata["after_long_press_xml_path"] = after_press["xml_path"]
    metadata["after_long_press_screenshot_path"] = after_press["screenshot_path"]

    select_result = tap_first_matching_label(appium, after_press["xml"], SELECT_ALL_TEXTS)
    metadata["select_all"] = select_result
    if not select_result.get("clicked"):
        metadata["clipboard"] = {
            "read_success": False,
            "skipped": True,
            "reason": "Select all was not found or not clicked; clipboard not read to avoid stale content.",
        }
        return metadata

    time.sleep(0.8)
    after_select = save_current_snapshot(appium, run_dir, "screenshot_after_select_all")
    metadata["after_select_all_xml_path"] = after_select["xml_path"]
    metadata["after_select_all_screenshot_path"] = after_select["screenshot_path"]

    copy_result = tap_first_matching_label(appium, after_select["xml"], COPY_TEXTS)
    metadata["copy"] = copy_result
    if not copy_result.get("clicked"):
        metadata["clipboard"] = {
            "read_success": False,
            "skipped": True,
            "reason": "Copy was not found or not clicked; clipboard not read to avoid stale content.",
        }
        return metadata

    time.sleep(0.8)
    try:
        clipboard_text, clipboard_strategy = read_clipboard_text(appium)
        write_text(run_dir / "clipboard_text.txt", clipboard_text)
        metadata["clipboard"] = {
            "read_success": True,
            "strategy": clipboard_strategy,
            "text_char_count": len(clipboard_text or ""),
            "text_file": str(run_dir / "clipboard_text.txt"),
            "urls": extract_urls_from_text(clipboard_text),
        }
    except Exception as exc:
        logger.warning("Failed to read clipboard text", exc_info=True)
        metadata["clipboard"] = {"read_success": False, "error": str(exc)}
    return metadata


def capture_copy_method(appium: AndroidAppiumClient, run_dir: Path, page_kind: str, copy_strategy: str) -> Dict[str, Any]:
    """
    Input: live Appium client, output directory, page kind, and copy strategy.
    Output: metadata dictionary describing copy extraction result.
    Function: dispatches copy probing to keyboard shortcut or context-menu implementation.
    """
    if copy_strategy == "keyboard_shortcut":
        return capture_copy_keyboard_shortcut(appium, run_dir, page_kind)
    if copy_strategy == "context_menu":
        return capture_copy_context_menu(appium, run_dir, page_kind)
    raise ValueError(f"Unsupported copy_strategy={copy_strategy!r}")


def validate_config(page_kind: str, capture_method: str, copy_strategy: str) -> None:
    """
    Input: page kind, capture method, and copy strategy.
    Output: raises ValueError on invalid config.
    Function: fails early when the test script is configured with unsupported modes.
    """
    if page_kind not in {"external_web", "in_app_document"}:
        raise ValueError(f"Unsupported page_kind={page_kind!r}; use external_web or in_app_document")
    if capture_method not in {"xml", "copy"}:
        raise ValueError(f"Unsupported capture_method={capture_method!r}; use xml or copy")
    if copy_strategy not in {"keyboard_shortcut", "context_menu"}:
        raise ValueError(f"Unsupported copy_strategy={copy_strategy!r}; use keyboard_shortcut or context_menu")


def parse_args() -> argparse.Namespace:
    """
    Input: command-line arguments.
    Output: argparse namespace.
    Function: reads device/output settings and optional mode overrides for quick experiments.
    """
    parser = argparse.ArgumentParser(description="Manual policy/TOS document text capture probe.")
    parser.add_argument("--appium-url", default="http://127.0.0.1:4723", help="Appium server URL.")
    parser.add_argument("--device-name", default=None, help="Optional Appium deviceName/udid.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "test_debug" / "policy_document_capture"),
        help="Root directory for timestamped capture outputs.",
    )
    parser.add_argument("--page-kind", choices=["external_web", "in_app_document"], default=None, help="Override PAGE_KIND.")
    parser.add_argument("--capture-method", choices=["xml", "copy"], default=None, help="Override CAPTURE_METHOD.")
    parser.add_argument("--copy-strategy", choices=["keyboard_shortcut", "context_menu"], default=None, help="Override COPY_STRATEGY.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    """
    Input: CLI args plus hardcoded defaults.
    Output: process exit code.
    Function: connects to Appium, runs exactly one configured document capture method, and saves results.
    """
    args = parse_args()
    setup_logging(args.debug)
    page_kind = args.page_kind or PAGE_KIND
    capture_method = args.capture_method or CAPTURE_METHOD
    copy_strategy = args.copy_strategy or COPY_STRATEGY
    validate_config(page_kind, capture_method, copy_strategy)

    output_root = Path(args.output_dir)
    run_dir = make_run_dir(output_root, page_kind, capture_method)
    logger.info("Policy document capture output: %s", run_dir)
    logger.info("Config: page_kind=%s capture_method=%s copy_strategy=%s", page_kind, capture_method, copy_strategy)

    appium = AndroidAppiumClient(server_url=args.appium_url, device_name=args.device_name)
    try:
        appium.init_connection()
        if capture_method == "xml":
            metadata = capture_xml_method(appium, run_dir, page_kind)
        else:
            metadata = capture_copy_method(appium, run_dir, page_kind, copy_strategy)
        metadata["run_dir"] = str(run_dir)
        metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(run_dir / "metadata.json", metadata)
        logger.info("Capture finished. metadata=%s", run_dir / "metadata.json")
        return 0
    except Exception as exc:
        logger.exception("Policy document capture failed")
        write_json(
            run_dir / "metadata.json",
            {
                "page_kind": page_kind,
                "capture_method": capture_method,
                "copy_strategy": copy_strategy,
                "run_dir": str(run_dir),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
            },
        )
        return 1
    finally:
        appium.quit()


if __name__ == "__main__":
    raise SystemExit(main())
