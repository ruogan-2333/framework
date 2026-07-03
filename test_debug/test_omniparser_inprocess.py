r"""Smoke-test the framework in-process OmniParser client.

Inputs:
- ``--state-dir``: one or more framework trace state directories containing screenshots.
- ``--image``: one or more direct PNG/JPG screenshot paths.
- ``--out-dir``: output directory for per-image OmniParser artifacts and run summary.

Outputs:
- One subdirectory per input image containing ``omniparser_input.png``,
  ``overlay.png``, ``parsed_elements.json``, ``label_coordinates.json``, and
  ``summary.json``.
- A top-level ``run_summary.json`` with elapsed time and parsed element counts.

Function:
- Simulates the future main workflow call path by reading screenshots as base64
  and invoking ``OmniParserClient.parse_base64(...)`` from ``omniparser_cls.py``.
- Verifies whether OmniParser models load once and are reused for later images in
  the same Python process.

Example:
```
.\.venv\Scripts\python.exe .\test_debug\test_omniparser_inprocess.py `
  --state-dir "F:\workplace\framework\traces\20260702_232311\states\UI000036_phash_6cec74c309ea73b382ae2558ace5e763" `
  --out-dir "F:\workplace\framework\test_debug\omniparser_inprocess_outputs\manual_check"
```

Input directories:
- Trace state directories usually live under ``F:\workplace\framework\traces\<run_id>\states\<UI...>``.
- Each state directory should contain ``screenshot.png`` or another PNG screenshot.

Output directories:
- If ``--out-dir`` is omitted, outputs are written under
  ``F:\workplace\framework\test_debug\omniparser_inprocess_outputs``.
- The script creates a timestamped child directory and writes ``run_summary.json``
  plus per-image artifact directories.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
"""Absolute path to the framework repository root used for local imports."""

if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from omniparser_cls import OmniParserClient


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the in-process OmniParser smoke test.

    Input:
    - Process command-line arguments.

    Output:
    - Namespace containing input state directories, direct images, output path,
      and parser settings.

    Function:
    - Provides a reproducible CLI for testing multiple screenshots in one Python
      process, which is required to verify model reuse.
    """

    parser = argparse.ArgumentParser(description="Test in-process OmniParser on framework screenshots.")
    parser.add_argument("--state-dir", action="append", default=[], help="Trace state directory containing screenshot.")
    parser.add_argument("--image", action="append", default=[], help="Direct screenshot image path.")
    parser.add_argument(
        "--out-dir",
        default=str(FRAMEWORK_ROOT / "test_debug" / "omniparser_inprocess_outputs"),
        help="Output directory for parser artifacts.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.05, help="YOLO icon detection confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Overlap filtering threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Florence icon caption batch size.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR to isolate icon detection/caption timing.")
    return parser.parse_args()


def find_state_screenshot(state_dir: Path) -> Path:
    """Find a screenshot image inside one framework state directory.

    Input:
    - ``state_dir``: directory under ``traces/<run>/states/<UI...>``.

    Output:
    - Path to the selected screenshot PNG.

    Function:
    - Supports current and historical trace layouts by checking common screenshot
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


def collect_input_images(args: argparse.Namespace) -> list[Path]:
    """Collect and validate all requested screenshot inputs.

    Input:
    - Parsed CLI arguments with ``--state-dir`` and/or ``--image`` values.

    Output:
    - Ordered list of screenshot paths.

    Function:
    - Converts state directories to screenshot paths and rejects missing inputs
      before any model loading occurs.
    """

    images: list[Path] = []
    for state in args.state_dir:
        state_dir = Path(state).resolve()
        if not state_dir.exists():
            raise FileNotFoundError(f"--state-dir not found: {state_dir}")
        images.append(find_state_screenshot(state_dir))

    for image in args.image:
        image_path = Path(image).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"--image not found: {image_path}")
        images.append(image_path)

    if not images:
        raise ValueError("Provide at least one --state-dir or --image")
    return images


def image_to_base64(image_path: Path) -> str:
    """Read an image file as base64 text.

    Input:
    - ``image_path``: existing screenshot path.

    Output:
    - Base64-encoded screenshot string.

    Function:
    - Mirrors the main workflow's in-memory screenshot format before calling
      ``OmniParserClient.parse_base64``.
    """

    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def safe_name_for_image(index: int, image_path: Path) -> str:
    """Build a stable output subdirectory name for one image.

    Input:
    - ``index``: one-based image index in the current run.
    - ``image_path``: screenshot path.

    Output:
    - Filesystem-safe directory name.

    Function:
    - Makes it easy to compare first, second, and later images when checking
      cold-start versus warm-start timings.
    """

    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in image_path.stem)[:80]
    return f"{index:03d}_{safe_stem}"


def run_one_image(index: int, image_path: Path, out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run in-process OmniParser on one screenshot.

    Input:
    - ``index``: one-based image order.
    - ``image_path``: screenshot path.
    - ``out_root``: root directory for this smoke test.
    - ``args``: parser settings from CLI.

    Output:
    - Summary dictionary for the image.

    Function:
    - Converts the screenshot to base64, calls the formal framework client, and
      returns the timing fields needed to verify warm model reuse.
    """

    image_out = out_root / safe_name_for_image(index, image_path)
    start = time.perf_counter()
    result = OmniParserClient.parse_base64(
        image_to_base64(image_path),
        str(image_out),
        use_ocr=not bool(args.no_ocr),
        box_threshold=float(args.box_threshold),
        iou_threshold=float(args.iou_threshold),
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
    )
    elapsed = time.perf_counter() - start
    summary = result.get("summary") or {}
    return {
        "index": index,
        "image": str(image_path),
        "out_dir": str(image_out),
        "elapsed_seconds": round(elapsed, 3),
        "model_seconds": summary.get("model_seconds"),
        "ocr_seconds": summary.get("ocr_seconds"),
        "parse_seconds": summary.get("parse_seconds"),
        "parsed_count": summary.get("parsed_count"),
        "summary_path": result.get("summary_path"),
        "overlay_path": result.get("overlay_path"),
    }


def main() -> None:
    """CLI entrypoint for in-process OmniParser smoke tests.

    Input:
    - Command-line arguments parsed by ``parse_args``.

    Output:
    - Prints a JSON run summary and writes ``run_summary.json``.

    Function:
    - Runs all requested screenshots in a single Python process so model loading
      should be paid once and visible in the first image timing.
    """

    args = parse_args()
    images = collect_input_images(args)
    out_root = Path(args.out_dir).resolve() / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    results = [run_one_image(i, image, out_root, args) for i, image in enumerate(images, start=1)]
    payload = {
        "total_images": len(images),
        "total_seconds": round(time.perf_counter() - run_start, 3),
        "out_dir": str(out_root),
        "use_ocr": not bool(args.no_ocr),
        "results": results,
    }
    (out_root / "run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
