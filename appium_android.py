"""
appium_android.py

Android Appium client wrapper with stability guards and small surface.

Design goals (empirical best-practices from flaky mobile automation):
- **Stable capture**: hash-based page stabilization + single-call snapshot helper.
- **Bounded retries**: tiny retry wrapper for transient driver hiccups.
- **Minimal API**: only primitives needed by higher layers (capture, tap, input,
  nav, start/stop). Keeps tests easy via driver injection.

Key additions for the updated workflow:
- NEW #A5: `parse_xml_to_uist(xml, pixel_ratio)` converts UiAutomator2 XML into a
  consistent UI tree structure used by ui_cls.BaseUI.post_process_ui.
- NEW #A6: `type_text(text)` types into the currently focused element, with
  robust fallbacks (active_element.send_keys -> mobile:shell input text).

Important:
- We do NOT touch your utils.py or cnn_cls.*; this file is self-contained.
- This file assumes Appium Python client is installed and you use UiAutomator2.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

import adbutils
from appium.options.android.uiautomator2.base import UiAutomator2Options
from appium.webdriver.webdriver import WebDriver

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("selenium").setLevel(logging.WARNING)

# --------- Configuration ---------
DEFAULT_CAPS: Dict[str, str | bool | int] = {
    "platformName": "Android",
    "automationName": "UiAutomator2",
    "autoGrantPermissions": True,
    "newCommandTimeout": 3600,
}

R = TypeVar("R")


def _with_retry(fn: Callable[[], R], attempts: int = 3, delay: float = 0.25) -> R:
    """Tiny retry helper for flaky driver calls."""
    last_err = None
    for i in range(max(1, int(attempts))):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - depends on real driver
            last_err = exc
            remaining = int(attempts) - i - 1
            if "NoSuchElement" in exc.__class__.__name__:
                logger.debug("Retryable NoSuchElement failure, remaining=%s", max(0, remaining))
            else:
                logger.debug("Retryable failure: %s, remaining=%s", exc, max(0, remaining))
            if remaining > 0:
                time.sleep(delay)
    if last_err:
        raise last_err
    raise RuntimeError("retry failed without exception")  # defensive


def _parse_bounds(bounds: str) -> Dict[str, int]:
    """
    UiAutomator2 bounds string: "[x1,y1][x2,y2]"
    """
    if not bounds:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    m = re.findall(r"\[(\d+),(\d+)\]", bounds)
    if len(m) != 2:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    x1, y1 = map(int, m[0])
    x2, y2 = map(int, m[1])
    return {"x": x1, "y": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1)}


def _scale_frame(frame: Dict[str, int], pixel_ratio: float) -> Dict[str, int]:
    """
    Some devices expose pixelRatio; we keep coordinates in *pixel* space for
    screenshot alignment (tap uses pixel coords).
    If your XML is already in pixels, pixel_ratio will be ~1.0 and no change.
    """
    pr = float(pixel_ratio or 1.0)
    if pr <= 0:
        pr = 1.0
    # We scale only if pr != 1 and values look like dp (heuristic would be risky),
    # so we simply trust caller: pass the ratio you want.
    return {
        "x": int(frame["x"] * pr),
        "y": int(frame["y"] * pr),
        "width": int(frame["width"] * pr),
        "height": int(frame["height"] * pr),
    }


def _adb_input_escape(text: str) -> str:
    """
    adb shell input text has special handling:
    - spaces must be encoded as %s
    - many symbols need escaping; keep it conservative
    """
    if text is None:
        return ""
    s = str(text)
    s = s.replace(" ", "%s")
    # escape shell-sensitive characters
    s = s.replace("&", r"\&").replace("|", r"\|").replace("<", r"\<").replace(">", r"\>")
    s = s.replace("(", r"\(").replace(")", r"\)").replace(";", r"\;")
    s = s.replace('"', r"\"").replace("'", r"\'")
    return s


@dataclass
class AndroidAppiumClient:
    """Thin wrapper for Appium driver lifecycle and simple actions.

    WHEN USED: instantiated once in main.py and passed into WorkflowRunner; all device IO flows through here.
    IMPORTANCE: provides the authoritative capture (xml+screenshot) and low-level taps/back/input that higher layers orchestrate.
    """

    server_url: str
    device_name: Optional[str] = None
    driver: Optional[WebDriver] = field(default=None, init=False)
    capabilities: Optional[UiAutomator2Options] = field(default=None, init=False)

    # ------------- Lifecycle -------------
    def init_connection(self, extra_caps: Optional[Dict] = None) -> "AndroidAppiumClient":
        merged = {**DEFAULT_CAPS, **(extra_caps or {})}
        if self.device_name:
            merged.setdefault("deviceName", self.device_name)

        logger.info("Initializing Appium connection at %s", self.server_url)
        options = UiAutomator2Options().load_capabilities(merged)
        self.capabilities = options
        try:
            self.driver = WebDriver(command_executor=self.server_url, options=options)
            logger.info("Appium driver created")
        except Exception:  # pragma: no cover - real device only
            logger.exception("Failed to create Appium driver")
            raise
        return self

    def quit(self) -> None:
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                logger.debug("Driver quit failed", exc_info=True)
        self.driver = None

    def launch_app(self, package: str, activity: str, wait: float = 0.5) -> None:
        self._require_driver()
        logger.info("Launching app %s/%s", package, activity)
        _with_retry(lambda: self.driver.start_activity(package, activity))
        time.sleep(wait)

    def start_activity(self, package: str, activity: str, wait: float = 0.5) -> None:
        self.launch_app(package, activity, wait=wait)

    def is_foreground(self, package: str) -> bool:
        self._require_driver()
        try:
            return getattr(self.driver, "current_package", None) == package
        except Exception:
            return False

    def foreground_package(self) -> str:
        """
        Best-effort current foreground package name.
        """
        self._require_driver()
        try:
            return str(getattr(self.driver, "current_package", "") or "")
        except Exception:
            return ""

    def foreground_activity(self) -> str:
        """
        Best-effort current foreground activity name (driver-specific).
        """
        self._require_driver()
        try:
            return str(getattr(self.driver, "current_activity", "") or "")
        except Exception:
            return ""

    def ensure_foreground(self, package: str, activity: Optional[str] = None, wait: float = 0.5):
        """Bring an app to foreground if needed."""
        self._require_driver()
        if not self.is_foreground(package):
            if activity:
                self.launch_app(package, activity, wait=wait)
            else:
                _with_retry(lambda: self.driver.activate_app(package))
            time.sleep(wait)

    def force_stop(self, package: str) -> None:
        """Stop one package on this client's device.

        Input:
        - package: Android package name to stop.

        Output:
        - None. Failures are logged and ignored because cleanup should not hide
          the original workflow result.

        Function:
        - Uses self.device_name when present so multi-device runs do not stop
          the package on the wrong adb device.
        """
        try:
            device = adbutils.AdbClient().device(self.device_name) if self.device_name else adbutils.adb.device()
            logger.info("Force-stop %s via adb device=%s", package, self.device_name or "<default>")
            device.shell(["am", "force-stop", package])
        except Exception:
            logger.debug("force_stop failed (ignored)", exc_info=True)

    # ------------- Stability / Snapshot -------------
    def wait_for_stable_page(self, timeout: float = 5.0, interval: float = 0.6) -> str:
        """Poll page_source until hash is stable across two consecutive reads."""
        self._require_driver()
        deadline = time.time() + timeout
        last_hash: Optional[str] = None
        last_xml: Optional[str] = None
        while time.time() < deadline:
            xml = _with_retry(lambda: self.driver.page_source)
            h = hashlib.md5(xml.encode("utf-8")).hexdigest()
            if h == last_hash:
                logger.debug("Page stable with hash=%s", h)
                return xml
            last_hash, last_xml = h, xml
            time.sleep(interval)
        logger.debug("Page not stable within timeout; returning last snapshot")
        return last_xml or ""

    def page_source_hash(self) -> str:
        """
        Fast, best-effort hash of current page_source (NO stabilization loop).
        Used for preflight change detection.
        """
        self._require_driver()
        # Best-effort: avoid retries to keep preflight fast.
        xml = _with_retry(lambda: self.driver.page_source, attempts=1)
        return hashlib.md5((xml or "").encode("utf-8")).hexdigest()

    def page_source_once(self) -> str:
        """
        Best-effort page_source fetch (single attempt, no stabilization loop).
        Used by workflow preflight when it needs to hash a processed/normalized XML string.
        """
        self._require_driver()
        return _with_retry(lambda: self.driver.page_source, attempts=1) or ""

    def screenshot_png_hash(self) -> str:
        """
        Fast, best-effort hash of the current screenshot PNG bytes (NO base64 encoding).
        Used for preflight change detection when XML is unreliable (e.g., Unity/canvas UIs).
        """
        self._require_driver()
        # Best-effort: avoid retries to keep preflight fast.
        png = _with_retry(lambda: self.driver.get_screenshot_as_png(), attempts=1)
        return hashlib.md5(png).hexdigest()

    def screenshot_png_once(self) -> bytes:
        """
        Best-effort screenshot fetch (single attempt).
        Used by workflow preflight when it needs to hash a processed/normalized screenshot.
        """
        self._require_driver()
        return _with_retry(lambda: self.driver.get_screenshot_as_png(), attempts=1)

    def capture_snapshot(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Single-call snapshot used by pipeline.

        - waits for stable page source once
        - captures screenshot once
        - returns both with deviceInfo (pixelRatio useful for scaling)
        WHEN: every loop in WorkflowRunner._capture_and_process (before LLMs); defines the authoritative UI state.
        IMPORTANCE: ensures xml and screenshot are time-aligned to avoid stale or inconsistent UI analysis.
        """
        logger.debug("Capturing snapshot (xml + screenshot + device_info)")
        xml = self.wait_for_stable_page(timeout=timeout)
        xml_hash = hashlib.md5((xml or "").encode("utf-8")).hexdigest()

        # Capture screenshot ONCE, but keep both base64 (for OCR/LLM) and a cheap hash (for caching/preflight).
        png = _with_retry(lambda: self.driver.get_screenshot_as_png())
        screenshot_hash = hashlib.md5(png).hexdigest()
        screenshot_b64 = base64.b64encode(png).decode("utf-8")
        device_info = self.get_device_info()
        return {
            "xml": xml,
            "xml_hash": xml_hash,
            "screenshot": screenshot_b64,
            "screenshot_hash": screenshot_hash,
            "device_info": device_info,
        }

    def get_device_info(self) -> Dict[str, Any]:
        self._require_driver()
        try:
            return self.driver.execute_script("mobile: deviceInfo") or {}
        except Exception:  # pragma: no cover - driver specific
            return {}

    def screenshot_base64(self) -> str:
        self._require_driver()
        png = _with_retry(lambda: self.driver.get_screenshot_as_png())
        return base64.b64encode(png).decode("utf-8")

    # ------------- XML -> UIST -------------
    def parse_xml_to_uist(self, xml: str, pixel_ratio: float = 1.0) -> Dict[str, Any]:
        """
        Convert UiAutomator2 page_source XML to the internal uist shape used by ui_cls.

        Output shape:
          {
            "elements": [node, ...],
            "screenscale": pixel_ratio
          }

        Each node:
          {
            "class": str,
            "text": str,
            "content_desc": str,
            "resource_id": str,
            "clickable": bool,
            "enabled": bool,
            "absolute_frame": {"x","y","width","height"},
            "subviews": [...]
          }

        Notes:
        - We treat only real UI nodes (<node ...>) as elements.
        - We preserve tree structure for dedupe + OCR attachment.
        WHEN: immediately after capture_snapshot inside WorkflowRunner._capture_and_process.
        IMPORTANCE: bridges raw Appium XML to the enriched UI tree expected by BaseUI.post_process_ui and LLM digests.
        """
        if not xml:
            return {"elements": [], "screenscale": float(pixel_ratio or 1.0)}

        try:
            root = ET.fromstring(xml)
        except Exception:
            logger.debug("Failed to parse XML; returning empty uist", exc_info=True)
            return {"elements": [], "screenscale": float(pixel_ratio or 1.0)}

        def conv(elem: ET.Element) -> Dict[str, Any]:
            attrs = elem.attrib or {}
            cls = attrs.get("class", "") or ""
            text = attrs.get("text", "") or ""
            cdesc = attrs.get("content-desc", "") or ""
            rid = attrs.get("resource-id", "") or ""
            clickable = (attrs.get("clickable", "false") == "true")
            enabled = (attrs.get("enabled", "true") == "true")
            selected = (attrs.get("selected", "false") == "true")
            focused = (attrs.get("focused", "false") == "true")
            focusable = (attrs.get("focusable", "false") == "true")
            scrollable = (attrs.get("scrollable", "false") == "true")
            checkable = (attrs.get("checkable", "false") == "true")
            checked = (attrs.get("checked", "false") == "true")
            long_clickable = (attrs.get("long-clickable", "false") == "true")
            password = (attrs.get("password", "false") == "true")

            frame = _parse_bounds(attrs.get("bounds", ""))
            frame = _scale_frame(frame, pixel_ratio)

            node = {
                "class": cls,
                "text": text,
                "content_desc": cdesc,
                "resource_id": rid,
                "clickable": bool(clickable),
                "enabled": bool(enabled),
                "selected": bool(selected),
                "focused": bool(focused),
                "focusable": bool(focusable),
                "scrollable": bool(scrollable),
                "checkable": bool(checkable),
                "checked": bool(checked),
                "long_clickable": bool(long_clickable),
                "password": bool(password),
                "absolute_frame": frame,
                "subviews": [],
            }

            # Recursively convert children nodes
            subs = []
            for ch in list(elem):
                # UiAutomator2 uses <node> tags; we accept any element but treat similarly.
                subs.append(conv(ch))
            node["subviews"] = subs
            return node

        # Common root is <hierarchy> -> children nodes.
        # We return its children as top-level elements (ui_cls can unwrap further).
        children = [conv(ch) for ch in list(root)]
        return {"elements": children, "screenscale": float(pixel_ratio or 1.0)}

    # ------------- Interactions -------------
    def tap(self, x: int, y: int) -> None:
        self._require_driver()
        logger.debug("Tap at (%s,%s)", x, y)
        _with_retry(lambda: self.driver.tap([(int(x), int(y))]))

    def back(self) -> None:
        if self.driver:
            logger.debug("Press back")
            _with_retry(lambda: self.driver.back())

    def click(self, by: str, value: str) -> None:
        self._require_driver()
        logger.debug("Click by=%s value=%s", by, value)
        _with_retry(lambda: self.driver.find_element(by, value).click())

    def send_keys(self, by: str, value: str, text: str) -> None:
        self._require_driver()
        logger.debug("Send keys by=%s value=%s text_len=%d", by, value, len(text or ""))

        def _do():
            el = self.driver.find_element(by, value)
            el.click()
            try:
                el.clear()
            except Exception:
                pass
            el.send_keys(text)

        _with_retry(_do)

    def send_keys_by_element_id(self, element_id: int, text: str) -> None:
        """
        Compatibility helper: send keys to an element located by id.
        """
        self.send_keys("id", str(element_id), text)

    def type_text(self, text: str) -> None:
        """
        Type text into the currently focused element.

        Strategy:
        1) active_element.send_keys(text) (best integration with IME)
        2) driver.execute_script("mobile: shell", {"command":"input","args":["text", escaped]})
        """
        self._require_driver()
        s = "" if text is None else str(text)

        # 1) active element typing
        try:
            el = self.driver.switch_to.active_element
            if el is not None:
                el.send_keys(s)
                return
        except Exception:
            pass

        # 2) adb input fallback via Appium "mobile: shell"
        try:
            escaped = _adb_input_escape(s)
            self.driver.execute_script(
                "mobile: shell",
                {
                    "command": "input",
                    "args": ["text", escaped],
                    "includeStderr": True,
                    "timeout": 5000,
                },
            )
            return
        except Exception:
            logger.debug("mobile:shell input text failed", exc_info=True)

        # 3) last resort: paste via key events is device-dependent; skip to avoid flakiness
        raise RuntimeError("type_text failed: no typing strategy succeeded")

    # ------------- Helpers -------------
    def _require_driver(self):
        if not self.driver:
            raise RuntimeError("Driver not initialized")

    # context manager sugar
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.quit()


__all__ = ["AndroidAppiumClient", "DEFAULT_CAPS"]
