"""Run OmniParser on one screenshot or trace state directory and save overlay outputs.

Inputs:
- ``--image``: direct path to a PNG/JPG screenshot, or
- ``--state-dir``: a framework trace state directory containing a screenshot.
- ``--omniparser-root``: local OmniParser checkout path.
- ``--out-dir``: output directory for overlay and parsed JSON files.

Outputs:
- ``overlay.png``: screenshot with OmniParser numbered boxes.
- ``parsed_elements.json``: raw parsed element list returned by OmniParser.
- ``label_coordinates.json``: label-to-box coordinates returned by OmniParser.
- ``summary.json``: timing and input/output metadata.

Function:
- Provides an offline comparison harness before OmniParser is considered for
  integration into ``ui_cls.py`` or the main workflow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Input:
    - Process command-line arguments.

    Output:
    - argparse Namespace containing input image/state and OmniParser settings.

    Function:
    - Keeps the smoke-test script reproducible from PowerShell commands.
    """

    parser = argparse.ArgumentParser(description="Run OmniParser on a framework screenshot/state directory.")
    parser.add_argument("--image", default="", help="Path to a screenshot image.")
    parser.add_argument("--state-dir", default="", help="Path to a trace state directory.")
    parser.add_argument(
        "--omniparser-root",
        default=r"F:\workplace\external_tools\OmniParser",
        help="Local OmniParser repository path.",
    )
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults under test_debug/omniparser_outputs.")
    parser.add_argument("--box-threshold", type=float, default=0.05, help="YOLO icon detection confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Overlap filtering threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Florence icon caption batch size.")
    parser.add_argument("--use-ocr", action="store_true", help="Enable OCR boxes. Disabled by default for stable smoke tests.")
    return parser.parse_args()


def find_state_screenshot(state_dir: Path) -> Path:
    """Find a screenshot inside one framework state directory.

    Input:
    - state_dir: directory produced under ``traces/<run>/states/<UI...>``.

    Output:
    - Path to the selected screenshot image.

    Function:
    - Supports current and historical state directory layouts by checking common
      names first, then falling back to the first PNG found recursively.
    """

    candidates = [
        state_dir / "screenshot.png",
        state_dir / "screenshot_raw.png",
        state_dir / "raw_screenshot.png",
        state_dir / "screen.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pngs = sorted(state_dir.rglob("*.png"))
    if pngs:
        return pngs[0]
    raise FileNotFoundError(f"No screenshot PNG found in state dir: {state_dir}")


def resolve_input_image(args: argparse.Namespace) -> Path:
    """Resolve the input screenshot path.

    Input:
    - args.image or args.state_dir from CLI.

    Output:
    - Existing screenshot path.

    Function:
    - Accepts either direct screenshots or framework state directories.
    """

    if args.image:
        image = Path(args.image).resolve()
        if not image.exists():
            raise FileNotFoundError(f"--image not found: {image}")
        return image
    if args.state_dir:
        state_dir = Path(args.state_dir).resolve()
        if not state_dir.exists():
            raise FileNotFoundError(f"--state-dir not found: {state_dir}")
        return find_state_screenshot(state_dir)
    raise ValueError("Provide either --image or --state-dir")


def default_output_dir(image_path: Path) -> Path:
    """Build a default output directory for one input image.

    Input:
    - image_path: selected screenshot file.

    Output:
    - Directory path under ``test_debug/omniparser_outputs``.

    Function:
    - Keeps outputs grouped by image stem and timestamp.
    """

    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in image_path.stem)[:80]
    return Path(__file__).resolve().parent / "omniparser_outputs" / f"{stamp}_{safe_name}"


def run_omniparser(args: argparse.Namespace) -> dict[str, Any]:
    """Run OmniParser and write overlay/JSON outputs.

    Input:
    - args: parsed CLI arguments with OmniParser paths and thresholds.

    Output:
    - Summary dictionary also saved as ``summary.json``.

    Function:
    - Imports OmniParser utilities from the external checkout, loads local V2
      weights, runs detection/caption, and saves inspectable artifacts.
    """

    image_path = resolve_input_image(args)
    omniparser_root = Path(args.omniparser_root).resolve()
    if not omniparser_root.exists():
        raise FileNotFoundError(f"OmniParser root not found: {omniparser_root}")
    sys.path.insert(0, str(omniparser_root))

    from util.utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model

    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_output_dir(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    yolo_model = get_yolo_model(model_path=str(omniparser_root / "weights" / "icon_detect" / "model.pt"))
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=str(omniparser_root / "weights" / "icon_caption_florence"),
    )

    image = Image.open(image_path).convert("RGB")
    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    ocr_text: list[str] = []
    ocr_bbox: list[Any] = []
    ocr_seconds = 0.0
    if args.use_ocr:
        ocr_start = time.perf_counter()
        (ocr_text, ocr_bbox), _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            easyocr_args={"paragraph": False, "text_threshold": 0.8},
            use_paddleocr=False,
        )
        ocr_seconds = time.perf_counter() - ocr_start

    parse_start = time.perf_counter()
    overlay_b64, label_coordinates, parsed_content_list = get_som_labeled_img(
        image,
        yolo_model,
        BOX_TRESHOLD=float(args.box_threshold),
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=ocr_text,
        iou_threshold=float(args.iou_threshold),
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
    )
    parse_seconds = time.perf_counter() - parse_start

    overlay = Image.open(io.BytesIO(base64.b64decode(overlay_b64)))
    overlay.save(out_dir / "overlay.png")
    (out_dir / "parsed_elements.json").write_text(
        json.dumps(parsed_content_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "label_coordinates.json").write_text(
        json.dumps(label_coordinates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "image": str(image_path),
        "omniparser_root": str(omniparser_root),
        "out_dir": str(out_dir),
        "use_ocr": bool(args.use_ocr),
        "ocr_count": len(ocr_text),
        "parsed_count": len(parsed_content_list),
        "ocr_seconds": round(ocr_seconds, 3),
        "parse_seconds": round(parse_seconds, 3),
        "total_seconds": round(time.perf_counter() - start, 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint for the OmniParser smoke-test script.

    Input:
    - Command-line arguments parsed by ``parse_args``.

    Output:
    - Prints a JSON summary to stdout and writes output files to disk.

    Function:
    - Allows the script to be called from PowerShell and batch experiments.
    """

    summary = run_omniparser(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
