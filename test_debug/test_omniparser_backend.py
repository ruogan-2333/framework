"""Test the OmniParser OCR backend against one screenshot or trace state directory.

Inputs:
- ``--image``: direct PNG/JPG screenshot path, or
- ``--state-dir``: trace state directory containing ``screenshot.png``.
- ``--out-dir``: output directory for converted framework artifacts.

Outputs:
- ``uist.json``: framework UI tree converted from OmniParser.
- ``vid_map.json``: element id to node mapping used by LLM/action execution.
- ``vid_map_overlay.png``: framework-style overlay for inspecting clickable nodes.
- ``summary.json``: compact run summary.

Function:
- Exercises the same ``BaseUI.post_process_ui_omniparser_ocr`` entrypoint that
  the main workflow uses when ``--visual-detector-backend omniparser_ocr`` is
  selected.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from ui_cls import BaseUI  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Input: command-line arguments. Output: parsed namespace. Function: defines the backend test interface."""
    parser = argparse.ArgumentParser(description="Run the framework OmniParser OCR backend on one screenshot.")
    parser.add_argument("--image", default="", help="Screenshot image path.")
    parser.add_argument("--state-dir", default="", help="Trace state directory containing screenshot.png.")
    parser.add_argument("--out-dir", required=True, help="Directory where converted outputs are written.")
    return parser.parse_args()


def resolve_image(args: argparse.Namespace) -> Path:
    """Input: parsed CLI args. Output: screenshot path. Function: selects a direct image or a state-dir screenshot."""
    if args.image:
        image = Path(args.image).resolve()
        if not image.exists():
            raise FileNotFoundError(f"--image not found: {image}")
        return image
    if args.state_dir:
        state_dir = Path(args.state_dir).resolve()
        candidates = [
            state_dir / "screenshot.png",
            state_dir / "screenshot_raw.png",
            state_dir / "raw_screenshot.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        pngs = sorted(state_dir.rglob("*.png"))
        if pngs:
            return pngs[0]
        raise FileNotFoundError(f"No screenshot PNG found in state dir: {state_dir}")
    raise ValueError("Provide either --image or --state-dir")


def image_to_b64(path: Path) -> str:
    """Input: image path. Output: base64 screenshot string. Function: matches the workflow screenshot representation."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def draw_overlay(image_path: Path, vid_map: Dict[int, Dict[str, Any]], out_path: Path) -> None:
    """Input: screenshot path, vid_map, output path. Output: overlay PNG. Function: visualizes converted nodes."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for element_id, node in sorted((vid_map or {}).items(), key=lambda item: int(item[0])):
        frame = BaseUI.get_frame(node)
        x = int(frame.get("x", 0))
        y = int(frame.get("y", 0))
        w = int(frame.get("width", 0))
        h = int(frame.get("height", 0))
        draw.rectangle([x, y, x + w, y + h], outline=(220, 50, 47), width=3)
        draw.text((x + 3, y + 3), str(element_id), fill=(255, 255, 0))
    image.save(out_path)


def main() -> None:
    """Input: CLI args. Output: JSON and overlay files. Function: runs one backend conversion and saves artifacts."""
    args = parse_args()
    image_path = resolve_image(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshot_b64 = image_to_b64(image_path)
    uist, vid_map = BaseUI.post_process_ui_omniparser_ocr({}, screenshot_b64, device_info=None)

    (out_dir / "uist.json").write_text(json.dumps(uist, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "vid_map.json").write_text(json.dumps(vid_map, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_overlay(image_path, vid_map, out_dir / "vid_map_overlay.png")

    summary = {
        "image": str(image_path),
        "node_count": len(vid_map),
        "out_dir": str(out_dir),
        "semantic_sources": sorted(
            set(str(node.get("semantic_source") or "") for node in (vid_map or {}).values())
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
