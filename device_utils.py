"""ADB utility helpers built on adbutils for CLI and workflow reuse."""

from __future__ import annotations

import os
from typing import List, Optional

import adbutils


def _get_device(device_name: Optional[str] = None):
    return adbutils.AdbClient().device(device_name) if device_name else adbutils.adb.device()


def list_packages(device_name: Optional[str] = None) -> List[str]:
    dev = _get_device(device_name)
    return sorted(dev.list_packages())


def list_processes(device_name: Optional[str] = None) -> List[str]:
    dev = _get_device(device_name)
    out = dev.shell("ps -A")
    return out.strip().splitlines()


def install_apk(apk_path: str, device_name: Optional[str] = None):
    if not os.path.exists(apk_path):
        raise FileNotFoundError(apk_path)
    dev = _get_device(device_name)
    dev.install(apk_path, clean=True)


def screenshot(path: str, device_name: Optional[str] = None):
    dev = _get_device(device_name)
    img = dev.screenshot()
    img.save(path)


def dump_ui(path: str, device_name: Optional[str] = None):
    dev = _get_device(device_name)
    if hasattr(dev, "dump_hierarchy"):
        xml = dev.dump_hierarchy()
    else:
        xml = dev.shell(["uiautomator", "dump", "/dev/tty"])
        marker = "<?xml"
        if marker in xml:
            xml = xml[xml.index(marker):]
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


__all__ = ["list_packages", "list_processes", "install_apk", "screenshot", "dump_ui"]
