"""
Standalone probe for validating LLM-directed coordinate clicks on Android screenshots.

Why this exists:
- The main workflow primarily clicks `vid_map` ids.
- Some Unity/game/canvas surfaces expose visible controls that never enter `vid_map`.
- This script lets us validate whether a VLM can still point to those controls directly.

What it does:
1. Capture current XML + screenshot from Appium.
2. Build the current `vid_map` using the same UIED-first post-processing path.
3. Ask the model for a direct coordinate target on the screenshot.
4. Save a preview image with the proposed point/bbox and diagnostics.
5. Optionally execute the tap after mapping screenshot pixels to tap-space.

Default behavior is SAFE:
- It does not tap unless `--execute` is provided.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from appium_android import AndroidAppiumClient
from gpt_cls import GPTClient, _b64_image_url
from ui_cls import BaseUI


load_dotenv()

logger = logging.getLogger(__name__)


class CoordinateClickProposal(BaseModel):
    decision: Literal["coordinate_click", "use_vid_map", "not_found"] = "not_found"
    target_summary: str = Field("", description="Short human-readable description of the intended target")
    suggested_element_id: Optional[int] = Field(None, description="Use when the target is already well represented in vid_map")
    x: Optional[int] = Field(None, description="Screenshot-pixel x coordinate")
    y: Optional[int] = Field(None, description="Screenshot-pixel y coordinate")
    bbox: Optional[List[int]] = Field(None, description="Optional screenshot bbox [x1,y1,x2,y2]")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="One short sentence")


class RegionOfInterestProposal(BaseModel):
    decision: Literal["roi_found", "not_found"] = "not_found"
    region_summary: str = Field("", description="Short description of the region containing the target")
    bbox: Optional[List[int]] = Field(None, description="Screenshot bbox [x1,y1,x2,y2] covering a useful region that contains the target")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="One short sentence")


PROBE_SYSTEM = """You are validating direct coordinate tapping on an Android screenshot.

You must choose ONE best visible target for the user's task on THIS exact screenshot.

Return rules:
- Coordinates are in screenshot pixel space for this exact image.
- Origin is top-left. x grows right, y grows down.
- If the target is visible but absent or poorly represented in ui_digest, return decision="coordinate_click".
- If the target is already clearly represented in ui_digest, return decision="use_vid_map" and suggested_element_id.
- If you cannot find the target, return decision="not_found".
- Never invent element ids.
- Prefer a tight bbox around the intended tap region when possible.
- Keep reason short and concrete.
"""


ROI_SYSTEM = """You are localizing a useful region of interest on an Android screenshot.

Goal:
- Find a region that CONTAINS the user's intended target control.

Return rules:
- Bounding boxes are in screenshot pixel space for this exact image.
- Return decision="roi_found" only if you can identify a concrete visible region that contains the target.
- If the target is on a popup/dialog/card/panel, prefer the popup/dialog/card itself as the ROI rather than the whole screen.
- The ROI should include enough local context to refine a small target inside it later.
- Do not return the entire screen unless the target truly spans most of the screen.
- Never invent element ids.
"""


REFINE_SYSTEM = """You are refining a click target inside a cropped Android screenshot.

Goal:
- Return the final target location INSIDE THIS CROP.

Return rules:
- Coordinates and bbox are in THIS crop's pixel space, not the original screen.
- If the target is visible in this crop, return decision="coordinate_click".
- Prefer a tight bbox around the visible target control.
- For tiny controls like close/X buttons, the bbox must tightly wrap the actual icon/button, not nearby empty space.
- For close/X buttons on a popup/dialog/card, interpret "top right" relative to the popup/dialog/card itself, not the whole crop.
- If you cannot clearly find the target in this crop, return decision="not_found".
- Never invent element ids.
- Keep reason short and concrete.
"""


def build_probe_system(force_coordinate: bool) -> str:
    if not force_coordinate:
        return PROBE_SYSTEM
    return (
        PROBE_SYSTEM
        + """

IMPORTANT OVERRIDE FOR THIS RUN:
- This is a forced coordinate-only validation.
- If the target is visible on the screenshot, return decision="coordinate_click".
- Do NOT return decision="use_vid_map" even if ui_digest contains a matching element.
- suggested_element_id must be null in this mode.
"""
    )


def task_suggests_close_icon(task: str) -> bool:
    t = str(task or "").lower()
    keywords = [
        "close",
        "dismiss",
        "cancel",
        "关闭",
        "关掉",
        "叉",
        "叉叉",
        "x",
    ]
    return any(k in t for k in keywords)


def build_roi_system(force_container: bool = False) -> str:
    if not force_container:
        return ROI_SYSTEM
    return (
        ROI_SYSTEM
        + """

IMPORTANT OVERRIDE FOR THIS RUN:
- The target is likely a tiny close/dismiss control.
- DO NOT return a tiny bbox around the close icon itself.
- Return the FULL visible popup/dialog/card/panel that contains the target.
- The ROI must include the title/body area plus the corner control.
- If you return only the icon-sized box, that is considered wrong.
"""
    )


def build_refine_system() -> str:
    return REFINE_SYSTEM


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone LLM coordinate-click feasibility probe.")
    p.add_argument("--appium-url", type=str, default="http://127.0.0.1:4723")
    p.add_argument("--device-name", type=str, default=None)
    p.add_argument("--package", type=str, default=None, help="Optional target package to foreground before probing")
    p.add_argument("--activity", type=str, default=None, help="Optional activity used with --package")
    p.add_argument("--task", type=str, required=True, help="What visible target to click, e.g. 'Click the red X close button'")
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="./traces/llm_coordinate_click_probe")
    p.add_argument("--execute", action="store_true", help="Actually tap after generating the coordinate proposal")
    p.add_argument("--settle", type=float, default=1.0, help="Seconds to wait after tap before capturing after-screenshot")
    p.add_argument("--ui-limit", type=int, default=80, help="Max vid_map elements summarized to the model")
    p.add_argument("--force-coordinate", action="store_true", help="Force coordinate_click output even when target is already in vid_map")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)

def _png_dimensions(png: bytes) -> Tuple[int, int]:
    try:
        with Image.open(io.BytesIO(png)) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def _xml_bounds_p95(xml: str) -> Tuple[int, int, int]:
    import re

    try:
        xs: List[int] = []
        ys: List[int] = []
        for x1, y1, x2, y2 in re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", xml or ""):
            xs.append(int(x2))
            ys.append(int(y2))
        if not xs or not ys:
            return 0, 0, 0
        xs.sort()
        ys.sort()
        idx = max(0, min(len(xs) - 1, int(round(0.95 * (len(xs) - 1)))))
        return int(xs[idx]), int(ys[idx]), int(len(xs))
    except Exception:
        return 0, 0, 0


def infer_coord_scale(xml: str, png: bytes, device_pixel_ratio: float) -> Tuple[float, Dict[str, Any]]:
    """
    Same philosophy as workflow._infer_coord_scale, copied here to keep this probe standalone.
    """
    meta: Dict[str, Any] = {"device_pixel_ratio": float(device_pixel_ratio or 1.0)}
    png_w, png_h = _png_dimensions(png)
    xml_w, xml_h, n_bounds = _xml_bounds_p95(xml)
    meta.update({"png_w": png_w, "png_h": png_h, "xml_w": xml_w, "xml_h": xml_h, "xml_bounds_n": n_bounds})

    if png_w <= 0 or png_h <= 0 or xml_w <= 0 or xml_h <= 0:
        meta["coord_scale_reason"] = "missing_dims"
        return 1.0, meta

    rw = float(png_w) / float(max(1, xml_w))
    rh = float(png_h) / float(max(1, xml_h))
    meta.update({"ratio_w": rw, "ratio_h": rh})

    if max(rw, rh) > 0:
        rel = abs(rw - rh) / max(rw, rh)
        meta["ratio_rel_diff"] = rel
        if rel > 0.12:
            meta["coord_scale_reason"] = "ratio_inconsistent"
            return 1.0, meta

    ratio = 0.5 * (rw + rh)
    meta["ratio_avg"] = ratio

    if 0.90 <= ratio <= 1.10:
        meta["coord_scale_reason"] = "ratio_near_1"
        return 1.0, meta

    if ratio < 1.15 or ratio > 6.0:
        meta["coord_scale_reason"] = "default_no_scale"
        return 1.0, meta

    pr = float(device_pixel_ratio or 0.0)
    if pr > 0 and abs(ratio - pr) <= 0.20:
        meta["coord_scale_reason"] = "snapped_to_device_pixel_ratio"
        return float(pr), meta

    meta["coord_scale_reason"] = "inferred"
    return float(round(ratio, 4)), meta


def summarize_vid_map(vid_map: Dict[int, Dict[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for eid, node in sorted((vid_map or {}).items(), key=lambda kv: int(kv[0])):
        frame = BaseUI.get_frame(node)
        label = (
            str(node.get("text") or "")
            or str(node.get("content_desc") or "")
            or str(node.get("ocr_text") or "")
            or str(node.get("icon_label") or "")
        ).strip()
        items.append(
            {
                "id": int(eid),
                "class": str(node.get("class") or ""),
                "label": label,
                "clickable": bool(node.get("clickable")),
                "bounds": [int(frame["x"]), int(frame["y"]), int(frame["x"] + frame["width"]), int(frame["y"] + frame["height"])],
            }
        )
    return items[: max(1, int(limit))]


def point_from_proposal(proposal: CoordinateClickProposal) -> Optional[Tuple[int, int]]:
    if proposal.x is not None and proposal.y is not None:
        return int(proposal.x), int(proposal.y)
    bbox = proposal.bbox or []
    if len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
    return None


def sanitize_bbox(bbox: Optional[List[int]], max_w: int, max_h: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(max_w - 1, x1))
    y1 = max(0, min(max_h - 1, y1))
    x2 = max(0, min(max_w, x2))
    y2 = max(0, min(max_h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    *,
    img_w: int,
    img_h: int,
    pad_frac: float = 0.20,
    min_size: int = 420,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    out_w = max(min_size, int(round(bw * (1.0 + 2.0 * pad_frac))))
    out_h = max(min_size, int(round(bh * (1.0 + 2.0 * pad_frac))))
    half_w = int(round(out_w / 2.0))
    half_h = int(round(out_h / 2.0))
    nx1 = max(0, cx - half_w)
    ny1 = max(0, cy - half_h)
    nx2 = min(img_w, cx + half_w)
    ny2 = min(img_h, cy + half_h)
    if nx2 - nx1 < out_w:
        if nx1 == 0:
            nx2 = min(img_w, out_w)
        elif nx2 == img_w:
            nx1 = max(0, img_w - out_w)
    if ny2 - ny1 < out_h:
        if ny1 == 0:
            ny2 = min(img_h, out_h)
        elif ny2 == img_h:
            ny1 = max(0, img_h - out_h)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def crop_png(png: bytes, crop_box: Tuple[int, int, int, int]) -> bytes:
    x1, y1, x2, y2 = crop_box
    with Image.open(io.BytesIO(png)).convert("RGB") as im:
        cropped = im.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()


def globalize_point(local_point: Tuple[int, int], crop_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, _x2, _y2 = crop_box
    return int(x1 + local_point[0]), int(y1 + local_point[1])


def globalize_bbox(local_bbox: Optional[List[int]], crop_box: Tuple[int, int, int, int]) -> Optional[List[int]]:
    if not isinstance(local_bbox, (list, tuple)) or len(local_bbox) != 4:
        return None
    x1, y1, _x2, _y2 = crop_box
    try:
        lx1, ly1, lx2, ly2 = [int(round(float(v))) for v in local_bbox]
    except Exception:
        return None
    return [int(x1 + lx1), int(y1 + ly1), int(x1 + lx2), int(y1 + ly2)]


def compute_tap_mapping(appium: AndroidAppiumClient, screenshot_w: int, screenshot_h: int) -> Dict[str, Any]:
    window_w = screenshot_w
    window_h = screenshot_h
    source = "screenshot_dims"
    try:
        if appium.driver:
            ws = appium.driver.get_window_size() or {}
            ww = int(ws.get("width") or 0)
            wh = int(ws.get("height") or 0)
            if ww > 0 and wh > 0:
                window_w, window_h = ww, wh
                source = "driver.get_window_size"
    except Exception:
        logger.debug("get_window_size failed; falling back to screenshot dims", exc_info=True)

    scale_x = float(window_w) / float(max(1, screenshot_w))
    scale_y = float(window_h) / float(max(1, screenshot_h))
    return {
        "window_w": int(window_w),
        "window_h": int(window_h),
        "screenshot_w": int(screenshot_w),
        "screenshot_h": int(screenshot_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "source": source,
    }


def screenshot_to_tap(point: Tuple[int, int], mapping: Dict[str, Any]) -> Tuple[int, int]:
    x, y = point
    tx = int(round(float(x) * float(mapping["scale_x"])))
    ty = int(round(float(y) * float(mapping["scale_y"])))
    return tx, ty


def point_hits_vid_map(point: Tuple[int, int], vid_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    x, y = point
    hits: List[Dict[str, Any]] = []
    for eid, node in (vid_map or {}).items():
        frame = BaseUI.get_frame(node)
        x1 = int(frame["x"])
        y1 = int(frame["y"])
        x2 = int(frame["x"] + frame["width"])
        y2 = int(frame["y"] + frame["height"])
        if x1 <= x <= x2 and y1 <= y <= y2:
            label = (
                str(node.get("text") or "")
                or str(node.get("content_desc") or "")
                or str(node.get("ocr_text") or "")
                or str(node.get("icon_label") or "")
            ).strip()
            hits.append(
                {
                    "id": int(eid),
                    "class": str(node.get("class") or ""),
                    "label": label,
                    "bounds": [x1, y1, x2, y2],
                    "area": max(1, (x2 - x1) * (y2 - y1)),
                }
            )
    hits.sort(key=lambda it: (int(it["area"]), int(it["id"])))
    return hits


def save_png(path: Path, png: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def save_annotated_preview(
    path: Path,
    png: bytes,
    proposal: CoordinateClickProposal,
    vid_hits: List[Dict[str, Any]],
    suggested_vid_node: Optional[Dict[str, Any]] = None,
    tap_point: Optional[Tuple[int, int]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(png)).convert("RGB") as im:
        draw = ImageDraw.Draw(im)
        bbox = proposal.bbox or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        pt = point_from_proposal(proposal)
        if pt:
            x, y = pt
            r = 14
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 0), width=4)
            draw.line([x - 24, y, x + 24, y], fill=(255, 255, 0), width=3)
            draw.line([x, y - 24, x, y + 24], fill=(255, 255, 0), width=3)
        for idx, hit in enumerate((vid_hits or [])[:3]):
            x1, y1, x2, y2 = [int(v) for v in hit["bounds"]]
            color = (0, 255, 255) if idx == 0 else (0, 180, 255)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if suggested_vid_node:
            x1, y1, x2, y2 = [int(v) for v in suggested_vid_node["bounds"]]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=4)
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            draw.line([cx - 18, cy, cx + 18, cy], fill=(255, 0, 255), width=3)
            draw.line([cx, cy - 18, cx, cy + 18], fill=(255, 0, 255), width=3)
        if tap_point and pt and tap_point != pt:
            tx, ty = tap_point
            r = 10
            draw.ellipse([tx - r, ty - r, tx + r, ty + r], outline=(0, 255, 0), width=3)
        im.save(path)


def build_probe_messages(
    *,
    screenshot_b64: str,
    screenshot_w: int,
    screenshot_h: int,
    task: str,
    ui_digest: List[Dict[str, Any]],
    force_coordinate: bool,
) -> List[Dict[str, Any]]:
    payload = {
        "task": task,
        "screenshot_size": {"width": screenshot_w, "height": screenshot_h},
        "ui_digest": ui_digest,
    }
    return [
        {"role": "system", "content": build_probe_system(force_coordinate)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}},
            ],
        },
    ]


def build_roi_messages(
    *,
    screenshot_b64: str,
    screenshot_w: int,
    screenshot_h: int,
    task: str,
    ui_digest: List[Dict[str, Any]],
    force_container: bool,
) -> List[Dict[str, Any]]:
    payload = {
        "task": task,
        "screenshot_size": {"width": screenshot_w, "height": screenshot_h},
        "ui_digest": ui_digest,
    }
    return [
        {"role": "system", "content": build_roi_system(force_container)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _b64_image_url(screenshot_b64)}},
            ],
        },
    ]


def build_refine_messages(
    *,
    crop_b64: str,
    crop_w: int,
    crop_h: int,
    task: str,
    roi_summary: str,
    focus_hint: str = "",
) -> List[Dict[str, Any]]:
    payload = {
        "task": task,
        "crop_size": {"width": crop_w, "height": crop_h},
        "roi_summary": roi_summary,
    }
    if focus_hint:
        payload["focus_hint"] = focus_hint
    return [
        {"role": "system", "content": build_refine_system()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _b64_image_url(crop_b64)}},
            ],
        },
    ]


def expand_bbox_for_corner_close(
    bbox: Tuple[int, int, int, int],
    *,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Close / dismiss icons often sit near the top-right corner of a larger popup.
    If the ROI stage returns a box near that corner, we want a crop that extends
    much farther to the left and downward so the refine stage can see the whole
    popup instead of a tiny or empty patch.
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    left_pad = max(900, bw * 8)
    right_pad = max(180, bw * 2)
    up_pad = max(260, bh * 3)
    down_pad = max(1200, bh * 10)
    nx1 = max(0, x1 - left_pad)
    ny1 = max(0, y1 - up_pad)
    nx2 = min(img_w, x2 + right_pad)
    ny2 = min(img_h, y2 + down_pad)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def roi_looks_like_container(
    bbox: Tuple[int, int, int, int],
    *,
    img_w: int,
    img_h: int,
) -> bool:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = float(bw * bh) / float(max(1, img_w * img_h))
    width_ratio = float(bw) / float(max(1, img_w))
    height_ratio = float(bh) / float(max(1, img_h))
    return area_ratio >= 0.10 or width_ratio >= 0.45 or height_ratio >= 0.22


def corner_focus_crop_from_container(
    bbox: Tuple[int, int, int, int],
    *,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Build a focused crop around the top-right corner of a popup/container, which is
    where close / dismiss icons usually live.
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    crop_w = max(280, int(round(bw * 0.26)))
    crop_h = max(280, int(round(bh * 0.26)))
    nx1 = max(0, int(round(x2 - crop_w * 1.10)))
    ny1 = max(0, int(round(y1 - crop_h * 0.15)))
    nx2 = min(img_w, int(round(x2 + crop_w * 0.10)))
    ny2 = min(img_h, int(round(y1 + crop_h * 1.05)))
    return int(nx1), int(ny1), int(nx2), int(ny2)


def detect_red_close_icon_in_crop(
    png: bytes,
) -> Tuple[Optional[Tuple[int, int]], Optional[List[int]]]:
    """
    In the focused top-right crop, try to snap onto a red close/dismiss button.
    This is a light-weight geometric post-process that complements the VLM.
    """
    with Image.open(io.BytesIO(png)).convert("RGB") as im:
        w, h = im.size
        px = im.load()
        x_start = max(0, int(w * 0.35))
        x_end = w
        y_start = 0
        y_end = max(1, int(h * 0.45))

        mask: List[List[bool]] = [[False] * w for _ in range(h)]
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                r, g, b = px[x, y]
                if r >= 150 and g <= 110 and b <= 110 and (r - g) >= 45 and (r - b) >= 45:
                    mask[y][x] = True

        visited = [[False] * w for _ in range(h)]
        best = None
        best_score = float("-inf")
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for sy in range(y_start, y_end):
            for sx in range(x_start, x_end):
                if not mask[sy][sx] or visited[sy][sx]:
                    continue
                stack = [(sx, sy)]
                visited[sy][sx] = True
                area = 0
                min_x = max_x = sx
                min_y = max_y = sy
                sum_x = 0
                sum_y = 0
                while stack:
                    x, y = stack.pop()
                    area += 1
                    sum_x += x
                    sum_y += y
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if nx < x_start or nx >= x_end or ny < y_start or ny >= y_end:
                            continue
                        if visited[ny][nx] or not mask[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        stack.append((nx, ny))

                bw = max_x - min_x + 1
                bh = max_y - min_y + 1
                if area < 120 or bw < 18 or bh < 18:
                    continue
                aspect = float(bw) / float(max(1, bh))
                if aspect < 0.55 or aspect > 1.8:
                    continue
                cx = float(sum_x) / float(max(1, area))
                cy = float(sum_y) / float(max(1, area))
                rightness = cx / float(max(1, w))
                topness = 1.0 - (cy / float(max(1, h)))
                squareness = 1.0 - min(1.0, abs(aspect - 1.0))
                score = (area * 1.0) + (rightness * 220.0) + (topness * 220.0) + (squareness * 160.0)
                if score > best_score:
                    best_score = score
                    best = {
                        "point": (int(round(cx)), int(round(cy))),
                        "bbox": [int(min_x), int(min_y), int(max_x + 1), int(max_y + 1)],
                    }

        if not best:
            return None, None
        return best["point"], best["bbox"]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(bool(args.debug))

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; LLM call may fail.")

    out_root = Path(args.out_dir).expanduser().resolve()
    run_dir = out_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    appium = AndroidAppiumClient(server_url=args.appium_url, device_name=args.device_name).init_connection()
    gpt = GPTClient(api_key=api_key, model=args.model, temperature=float(args.temperature), timeout_s=int(args.timeout))

    try:
        if args.package:
            appium.ensure_foreground(args.package, args.activity, wait=0.8)

        snap = appium.capture_snapshot(timeout=5.0)
        screenshot_b64 = str(snap.get("screenshot") or "")
        png = base64.b64decode(screenshot_b64) if screenshot_b64 else b""
        xml = str(snap.get("xml") or "")
        device_info = snap.get("device_info") or {}
        screenshot_w, screenshot_h = _png_dimensions(png)
        save_png(run_dir / "before.png", png)

        device_pixel_ratio = float(device_info.get("pixelRatio", 1.0) or 1.0)
        coord_scale, coord_meta = infer_coord_scale(xml, png, device_pixel_ratio)
        uist = appium.parse_xml_to_uist(xml, pixel_ratio=coord_scale)
        uist2, vid_map = BaseUI.post_process_ui_uied_first(uist, screenshot_b64, device_info=device_info)
        ui_digest = summarize_vid_map(vid_map, limit=int(args.ui_limit))

        roi_proposal = None
        wants_corner_close = task_suggests_close_icon(str(args.task))
        if args.force_coordinate:
            roi_messages = build_roi_messages(
                screenshot_b64=screenshot_b64,
                screenshot_w=screenshot_w,
                screenshot_h=screenshot_h,
                task=str(args.task),
                ui_digest=ui_digest,
                force_container=wants_corner_close,
            )
            roi_proposal = gpt._call_structured(roi_messages, RegionOfInterestProposal, opname="coordinate_click_probe_roi")
            roi_bbox = sanitize_bbox(roi_proposal.bbox, screenshot_w, screenshot_h)
            if roi_proposal.decision == "roi_found" and roi_bbox is not None:
                focus_hint = ""
                if wants_corner_close:
                    if roi_looks_like_container(roi_bbox, img_w=screenshot_w, img_h=screenshot_h):
                        container_box = expand_bbox(
                            roi_bbox,
                            img_w=screenshot_w,
                            img_h=screenshot_h,
                            pad_frac=0.06,
                            min_size=max(420, min(screenshot_w, screenshot_h) // 4),
                        )
                        save_png(run_dir / "roi_container.png", crop_png(png, container_box))
                        crop_box = corner_focus_crop_from_container(roi_bbox, img_w=screenshot_w, img_h=screenshot_h)
                        focus_hint = "This crop is focused on the popup dialog's top-right corner where the close/X button should appear."
                    else:
                        crop_box = expand_bbox_for_corner_close(roi_bbox, img_w=screenshot_w, img_h=screenshot_h)
                        focus_hint = "This crop should contain the close/X button region."
                else:
                    crop_box = expand_bbox(roi_bbox, img_w=screenshot_w, img_h=screenshot_h)
                crop_png_bytes = crop_png(png, crop_box)
                save_png(run_dir / "roi_crop.png", crop_png_bytes)
                crop_w, crop_h = _png_dimensions(crop_png_bytes)
                crop_b64 = base64.b64encode(crop_png_bytes).decode("utf-8")
                refine_messages = build_refine_messages(
                    crop_b64=crop_b64,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    task=str(args.task),
                    roi_summary=str(roi_proposal.region_summary or ""),
                    focus_hint=focus_hint,
                )
                refined = gpt._call_structured(refine_messages, CoordinateClickProposal, opname="coordinate_click_probe_refine")
                snapped_local_point = None
                snapped_local_bbox = None
                if wants_corner_close:
                    snapped_local_point, snapped_local_bbox = detect_red_close_icon_in_crop(crop_png_bytes)

                if refined.decision == "coordinate_click" or snapped_local_point is not None:
                    local_bbox = snapped_local_bbox if snapped_local_bbox is not None else refined.bbox
                    global_bbox = globalize_bbox(local_bbox, crop_box)
                    global_point = snapped_local_point if snapped_local_point is not None else point_from_proposal(refined)
                    if global_point:
                        gx, gy = globalize_point(global_point, crop_box)
                    else:
                        gx = gy = None
                    reason = refined.reason or roi_proposal.reason or ""
                    if snapped_local_point is not None:
                        reason = (reason + " | snapped_to_red_close_icon").strip(" |")
                    proposal = CoordinateClickProposal(
                        decision="coordinate_click",
                        target_summary=refined.target_summary or roi_proposal.region_summary or "",
                        suggested_element_id=None,
                        x=gx,
                        y=gy,
                        bbox=global_bbox,
                        confidence=min(1.0, float(refined.confidence or 0.0)),
                        reason=reason,
                    )
                    preview_proposal = CoordinateClickProposal(
                        decision="coordinate_click",
                        target_summary=refined.target_summary or roi_proposal.region_summary or "",
                        suggested_element_id=None,
                        x=snapped_local_point[0] if snapped_local_point is not None else refined.x,
                        y=snapped_local_point[1] if snapped_local_point is not None else refined.y,
                        bbox=snapped_local_bbox if snapped_local_bbox is not None else refined.bbox,
                        confidence=min(1.0, float(refined.confidence or 0.0)),
                        reason=reason,
                    )
                    save_annotated_preview(
                        run_dir / "roi_crop_annotated.png",
                        crop_png_bytes,
                        preview_proposal,
                        [],
                    )
                else:
                    proposal = CoordinateClickProposal(
                        decision="not_found",
                        target_summary=refined.target_summary or "",
                        suggested_element_id=None,
                        x=None,
                        y=None,
                        bbox=None,
                        confidence=float(refined.confidence or 0.0),
                        reason=refined.reason or "refine_stage_not_found",
                    )
            else:
                proposal = CoordinateClickProposal(
                    decision="not_found",
                    target_summary="",
                    suggested_element_id=None,
                    x=None,
                    y=None,
                    bbox=None,
                    confidence=float(getattr(roi_proposal, "confidence", 0.0) or 0.0),
                    reason=getattr(roi_proposal, "reason", "") or "roi_stage_not_found",
                )
        else:
            messages = build_probe_messages(
                screenshot_b64=screenshot_b64,
                screenshot_w=screenshot_w,
                screenshot_h=screenshot_h,
                task=str(args.task),
                ui_digest=ui_digest,
                force_coordinate=bool(args.force_coordinate),
            )
            proposal = gpt._call_structured(messages, CoordinateClickProposal, opname="coordinate_click_probe")

        shot_point = point_from_proposal(proposal)
        mapping = compute_tap_mapping(appium, screenshot_w, screenshot_h)
        tap_point = screenshot_to_tap(shot_point, mapping) if shot_point else None
        vid_hits = point_hits_vid_map(shot_point, vid_map) if shot_point else []
        suggested_vid_node = None
        if proposal.suggested_element_id is not None:
            node = (vid_map or {}).get(int(proposal.suggested_element_id))
            if node:
                frame = BaseUI.get_frame(node)
                label = (
                    str(node.get("text") or "")
                    or str(node.get("content_desc") or "")
                    or str(node.get("ocr_text") or "")
                    or str(node.get("icon_label") or "")
                ).strip()
                suggested_vid_node = {
                    "id": int(proposal.suggested_element_id),
                    "label": label,
                    "bounds": [
                        int(frame["x"]),
                        int(frame["y"]),
                        int(frame["x"] + frame["width"]),
                        int(frame["y"] + frame["height"]),
                    ],
                }

        save_annotated_preview(
            run_dir / "before_annotated.png",
            png,
            proposal,
            vid_hits,
            suggested_vid_node=suggested_vid_node,
            tap_point=tap_point,
        )

        result: Dict[str, Any] = {
            "task": args.task,
            "model": args.model,
            "proposal": proposal.model_dump(),
            "force_coordinate": bool(args.force_coordinate),
            "screenshot_size": {"width": screenshot_w, "height": screenshot_h},
            "tap_mapping": mapping,
            "coord_scale_for_vid_map": coord_scale,
            "coord_scale_meta": coord_meta,
            "vid_map_count": len(vid_map),
            "vid_map_hits_at_point": vid_hits,
            "suggested_vid_map_node": suggested_vid_node,
            "artifacts": {
                "before_png": str((run_dir / "before.png").resolve()),
                "before_annotated_png": str((run_dir / "before_annotated.png").resolve()),
            },
        }
        if roi_proposal is not None:
            result["roi_proposal"] = roi_proposal.model_dump()
            if (run_dir / "roi_container.png").exists():
                result["artifacts"]["roi_container_png"] = str((run_dir / "roi_container.png").resolve())
            if (run_dir / "roi_crop.png").exists():
                result["artifacts"]["roi_crop_png"] = str((run_dir / "roi_crop.png").resolve())
            if (run_dir / "roi_crop_annotated.png").exists():
                result["artifacts"]["roi_crop_annotated_png"] = str((run_dir / "roi_crop_annotated.png").resolve())

        if shot_point:
            result["screenshot_point"] = {"x": int(shot_point[0]), "y": int(shot_point[1])}
            result["normalized_point"] = {
                "x": round(float(shot_point[0]) / float(max(1, screenshot_w)), 6),
                "y": round(float(shot_point[1]) / float(max(1, screenshot_h)), 6),
            }
        if tap_point:
            result["tap_point"] = {"x": int(tap_point[0]), "y": int(tap_point[1])}

        if args.execute and proposal.decision == "coordinate_click" and tap_point:
            logger.info("Executing tap at screenshot_point=%s tap_point=%s", shot_point, tap_point)
            appium.tap(int(tap_point[0]), int(tap_point[1]))
            time.sleep(float(args.settle))
            after_png = appium.screenshot_png_once()
            save_png(run_dir / "after.png", after_png)
            result["artifacts"]["after_png"] = str((run_dir / "after.png").resolve())
        else:
            if args.execute:
                logger.info("No tap executed because proposal.decision=%s tap_point=%s", proposal.decision, tap_point)
            else:
                logger.info("Dry run only. Use --execute to actually tap.")

        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(json.dumps(result, ensure_ascii=False, indent=2))
        logger.info("Saved probe artifacts to %s", str(run_dir))
        logger.info("before: %s", str((run_dir / "before_annotated.png").resolve()))
        if result["artifacts"].get("after_png"):
            logger.info("after: %s", str((run_dir / "after.png").resolve()))
        return 0
    finally:
        appium.quit()


if __name__ == "__main__":
    raise SystemExit(main())
