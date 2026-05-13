from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from appium_android import AndroidAppiumClient
from gpt_cls import GPTClient
from questionnaire_state2 import QuestionnaireState as QuestionnaireState2
from trace_callbacks import NoOpCallbacks
from workflow import BudgetConfig, WorkflowRunner


load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# 调试配置（你只需要改这里，不用命令行）
# ============================================================================
DEBUG_CONFIG: Dict[str, Any] = {
    # 必填：本地截图路径（支持 png/jpg/webp，脚本内部会统一转成 PNG）
    "image_path": r"F:\workplace\framework\test_5_12\test\test\test.png",

    # 可选：本地 XML 路径。为空字符串则按“无 XML”处理。
    "xml_path": "",

    # 目标 app 信息（用于 _capture_and_process_debug 内部的部分启发式逻辑）
    "package": "com.maimemo.android.momo",
    "activity": None,

    # 问卷目录（仅用于构造 WorkflowRunner，沿用主流程初始化）
    "questionnaire_dir": r"F:\workplace\framework\questionnaire-v2\games",

    # 本地样本对应的设备像素比（会写入 raw["device_info"]["pixelRatio"]）
    "pixel_ratio": 1.0,

    # 调试函数超时（秒）
    "capture_timeout": 20.0,

    # 输出目录
    "output_root": r"F:\workplace\framework\mytest2\capture_and_process_debug_local",

    # run_id 为空时自动使用时间戳
    "run_id": "",

    # 日志开关
    "debug_log": True,

    # 以下参数仅用于构建 Runner 对象（本地调试不依赖实时 Appium 连接）
    "appium_url": "http://127.0.0.1:4723",
    "device_name": None,
    "api_key": "",
    "model": "gpt-4o",
}


def setup_logging(debug: bool) -> None:
    """
    输入:
    - debug: 是否开启 debug 级别日志。

    输出:
    - 无（就地配置全局 logging）。
    """
    level = logging.DEBUG if debug else logging.INFO
    fmt = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)


def _safe_token(value: Any, default: str = "unknown") -> str:
    """
    输入:
    - value: 任意值（通常用于路径命名）。
    - default: 清洗后为空时的兜底字符串。

    输出:
    - 文件名安全字符串（仅保留 A-Za-z0-9_.-）。
    """
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or default


def _md5_bytes(data: bytes) -> str:
    """
    输入:
    - data: 字节串。

    输出:
    - data 的 MD5 十六进制字符串。
    """
    return hashlib.md5(data).hexdigest()


def _read_image_as_png_bytes(path: Path) -> bytes:
    """
    输入:
    - path: 本地图片路径（png/jpg/webp 等）。

    输出:
    - 统一转换后的 PNG 字节。
    """
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def _read_xml_text(path: Optional[Path]) -> str:
    """
    输入:
    - path: XML 文件路径；None 表示不提供 XML。

    输出:
    - 读取到的 XML 文本；path 为 None 时返回空字符串。
    """
    if path is None:
        return ""
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read xml file: {path} ({last_err})")


def _draw_vid_map_overlay(screenshot_b64: str, vid_map: Dict[str, Any], out_path: Path) -> bool:
    """
    输入:
    - screenshot_b64: base64 编码截图（处理后截图）。
    - vid_map: id -> node 的映射。
    - out_path: 叠框图输出路径。

    输出:
    - bool: 是否成功写出叠框图。
    """
    if not screenshot_b64 or not vid_map:
        return False
    try:
        raw = base64.b64decode(screenshot_b64 + "==")
        image = Image.open(BytesIO(raw)).convert("RGB")
        draw = ImageDraw.Draw(image)

        for element_id, node in (vid_map or {}).items():
            frame = node.get("absolute_frame") or node.get("frame") or {}
            x = int(frame.get("x", 0))
            y = int(frame.get("y", 0))
            w = int(frame.get("width", 0))
            h = int(frame.get("height", 0))
            if w <= 1 or h <= 1:
                continue

            clickable = bool(node.get("clickable"))
            color = (220, 50, 47) if clickable else (46, 160, 67)
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3)

            label = str(element_id)
            box = draw.textbbox((x, y), label)
            tw = box[2] - box[0]
            th = box[3] - box[1]
            left = max(0, x)
            top = max(0, y - th - 8)
            if top == 0 and y + h + th + 8 < image.height:
                top = y + h + 2
            draw.rectangle((left, top, left + tw + 10, top + th + 8), fill=color)
            draw.text((left + 5, top + 4), label, fill=(255, 255, 255))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return True
    except Exception:
        logger.debug("Failed to draw vid_map overlay", exc_info=True)
        return False


def _count_uist_nodes(uist: Dict[str, Any]) -> int:
    """
    输入:
    - uist: UI 树字典（期望包含 elements/subviews 层级）。
    输出:
    - int: UI 树中总节点数（包含根节点与所有子节点）。
    """
    stack = list((uist or {}).get("elements", []) or [])
    total = 0
    while stack:
        node = stack.pop()
        total += 1
        subs = node.get("subviews", []) or []
        if subs:
            stack.extend(subs)
    return total


def _save_snap_outputs(run_dir: Path, snap: Dict[str, Any], input_payload: Dict[str, Any]) -> None:
    """
    输入:
    - run_dir: 本次输出目录。
    - snap: _capture_and_process_debug 返回的完整快照字典。
    - input_payload: 本次调试输入信息（原始图片/XML/hash等）。

    输出:
    - 无（把 snap 和拆分文件写入磁盘）。
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "input_payload.json").write_text(
        json.dumps(input_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "snap.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    xml = str(snap.get("xml") or "")
    xml_raw = str(snap.get("xml_raw") or "")
    (run_dir / "xml.xml").write_text(xml, encoding="utf-8")
    (run_dir / "xml_raw.xml").write_text(xml_raw, encoding="utf-8")

    screenshot = str(snap.get("screenshot") or "")
    screenshot_raw = str(snap.get("screenshot_raw") or "")
    if screenshot:
        (run_dir / "screenshot.png").write_bytes(base64.b64decode(screenshot + "=="))
    if screenshot_raw:
        (run_dir / "screenshot_raw.png").write_bytes(base64.b64decode(screenshot_raw + "=="))

    uist = snap.get("uist") or {}
    vid_map = snap.get("vid_map") or {}
    (run_dir / "uist.json").write_text(json.dumps(uist, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "vid_map.json").write_text(json.dumps(vid_map, ensure_ascii=False, indent=2), encoding="utf-8")

    overlay_ok = _draw_vid_map_overlay(screenshot, vid_map, run_dir / "vid_map_overlay.png")
    uist_total_nodes = _count_uist_nodes(uist)
    vid_map_count = int(len(vid_map or {}))

    meta = snap.get("meta") or {}
    summary = {
        "saved_at_ms": int(time.time() * 1000),
        "state_sig": str(snap.get("state_sig") or ""),
        "xml_reliable": bool(snap.get("xml_reliable")),
        "uist_root_count": int(len((uist or {}).get("elements") or [])),
        "uist_total_nodes": int(uist_total_nodes),
        "vid_map_count": int(vid_map_count),
        "vid_map_matches_all_uist_nodes": bool(vid_map_count == uist_total_nodes),
        "vid_map_missing_nodes": int(max(0, uist_total_nodes - vid_map_count)),
        "overlay_written": bool(overlay_ok),
        "identity_source": str(meta.get("identity_source") or ""),
        "coord_scale": float(meta.get("coord_scale") or 0.0),
        "foreground_package": str(meta.get("foreground_package") or ""),
        "foreground_activity": str(meta.get("foreground_activity") or ""),
        "xml_hash": str(meta.get("xml_hash") or ""),
        "xml_hash_raw": str(meta.get("xml_hash_raw") or ""),
        "screenshot_hash": str(meta.get("screenshot_hash") or ""),
        "screenshot_hash_raw": str(meta.get("screenshot_hash_raw") or ""),
        "screenshot_phash": str(meta.get("screenshot_phash") or ""),
        "cache_hit": bool(meta.get("cache_hit")),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_runner(cfg: Dict[str, Any]) -> WorkflowRunner:
    """
    输入:
    - cfg: 调试配置字典（来自 DEBUG_CONFIG）。

    输出:
    - WorkflowRunner: 与主流程一致的 Runner（但此脚本不做实时 Appium 连接）。
    """
    questionnaires = QuestionnaireState2.load_from_questionnaire_dir(str(cfg["questionnaire_dir"]))

    api_key = str(cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "")
    gpt = GPTClient(api_key=api_key, model=str(cfg.get("model") or "gpt-4o"), temperature=0.0, timeout_s=30)

    # 本地样本调试不走实时采样，因此这里只构造对象，不 init_connection。
    appium = AndroidAppiumClient(
        server_url=str(cfg.get("appium_url") or "http://127.0.0.1:4723"),
        device_name=cfg.get("device_name"),
    )

    budget = BudgetConfig(
        time_budget_s=30.0,
        max_actions=1,
        per_page_probe_cap=1,
        max_workers=1,
    )

    return WorkflowRunner(
        appium=appium,
        gpt=gpt,
        questionnaires=questionnaires,
        budget=budget,
        target_package=str(cfg.get("package") or ""),
        target_activity=cfg.get("activity"),
        pause=False,
        callbacks=NoOpCallbacks(),
        run_id=str(cfg.get("run_id") or ""),
    )


def run_debug_capture(cfg: Dict[str, Any]) -> int:
    """
    输入:
    - cfg: 调试配置字典。

    输出:
    - int: 进程退出码。0=成功，1=失败。
    """
    setup_logging(bool(cfg.get("debug_log")))

    image_path = Path(str(cfg["image_path"])).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    xml_path_value = str(cfg.get("xml_path") or "").strip()
    xml_path: Optional[Path] = Path(xml_path_value).resolve() if xml_path_value else None
    if xml_path is not None and not xml_path.exists():
        raise FileNotFoundError(f"xml not found: {xml_path}")

    xml_text = _read_xml_text(xml_path)
    screenshot_png_bytes = _read_image_as_png_bytes(image_path)
    screenshot_b64 = base64.b64encode(screenshot_png_bytes).decode("utf-8")

    raw_payload = {
        "xml": xml_text,
        "xml_hash": _md5_bytes((xml_text or "").encode("utf-8")),
        "screenshot": screenshot_b64,
        "screenshot_hash": _md5_bytes(screenshot_png_bytes),
        "device_info": {"pixelRatio": float(cfg.get("pixel_ratio", 1.0))},
    }

    run_stamp = str(cfg.get("run_id") or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_stamp}_{_safe_token(image_path.stem)}"
    run_dir = Path(str(cfg.get("output_root"))).resolve() / run_name

    runner = _build_runner(cfg)

    t0 = time.perf_counter()
    snap = runner._capture_and_process_debug(
        timeout=float(cfg.get("capture_timeout", 10.0)),
        debug=True,
        raw=raw_payload,
        xml_raw=xml_text,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not snap:
        fail_payload = {
            "ok": False,
            "error": "_capture_and_process_debug returned None",
            "elapsed_ms": elapsed_ms,
            "input": {
                "image_path": str(image_path),
                "xml_path": str(xml_path) if xml_path else "",
                "xml_len": len(xml_text or ""),
                "screenshot_png_bytes": len(screenshot_png_bytes),
            },
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "failure.json").write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[FAILED] output={run_dir}")
        return 1

    input_payload = {
        "image_path": str(image_path),
        "xml_path": str(xml_path) if xml_path else "",
        "elapsed_ms": round(elapsed_ms, 3),
        "raw_payload_meta": {
            "xml_hash": raw_payload["xml_hash"],
            "screenshot_hash": raw_payload["screenshot_hash"],
            "xml_len": len(xml_text or ""),
            "screenshot_png_bytes": len(screenshot_png_bytes),
            "pixel_ratio": float(cfg.get("pixel_ratio", 1.0)),
        },
    }
    _save_snap_outputs(run_dir, snap, input_payload)

    meta = snap.get("meta") or {}
    print("=== capture_and_process_debug done ===")
    print(f"output_dir={run_dir}")
    print(f"elapsed_ms={elapsed_ms:.3f}")
    print(f"state_sig={snap.get('state_sig')}")
    print(f"xml_reliable={snap.get('xml_reliable')}")
    uist = snap.get("uist") or {}
    uist_total_nodes = _count_uist_nodes(uist)
    vid_map_count = len(snap.get("vid_map") or {})
    print(f"uist_total_nodes={uist_total_nodes}")
    print(f"vid_map_count={vid_map_count}")
    print(f"vid_map_matches_all_uist_nodes={vid_map_count == uist_total_nodes}")
    print(f"identity_source={meta.get('identity_source')}")
    print(f"cache_hit={meta.get('cache_hit')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_debug_capture(DEBUG_CONFIG))
