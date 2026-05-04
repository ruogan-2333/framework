"""
ui_cls.py

UI parsing + post-processing utilities.

This module is responsible for:
- parsing raw UI tree into a consistent structure (already done upstream in appium_android.getui() typically)
- post-processing:
    - unwrap "hierarchy" / placeholder roots
    - dedupe noisy nodes
    - assign stable incremental ids to interactables
    - integrate OCR results into the UI (for icon-heavy UIs / missing text)
    - keep optional icon classification hooks (cnn_cls) WITHOUT depending on it

Key fixes for your debug log failure:
- BUGFIX: "After dedupe/sort: 0 root nodes" was caused by treating the top placeholder as root
  and then deduping it away, losing actual content.
   We now "salvage roots": if dedupe produces empty roots but the original has children,
     we fall back to children as roots.
- BUGFIX: Assigned 0 ids: happened when clickables were in children but roots got removed.
   id assignment now traverses the entire tree and assigns ids to interactables
     (clickable OR has OCR/icon text useful for LLM).
- PERFORMANCE: OCR is kept but called conditionally and cached by screenshot hash.
   The workflow should call post_process_ui with screenshot_b64 so OCR can be applied.
- ROBUSTNESS: Dedupe is conservative; never deletes all roots.

We do NOT touch your utils.py or cnn_cls.* files. We only *optionally* import and use them.

WHEN USED IN WORKFLOW:
- WorkflowRunner._capture_and_process calls BaseUI.post_process_ui on every snapshot to enrich UI, attach OCR/icon labels, and assign stable ids for LLM.
IMPORTANCE: Stable ids/bounds are required so LLM proposals click the intended elements; OCR/icon labels compensate for missing XML text.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


from utils import time_consumed, crop_image

# OCR is required; keep paddle_cls
from paddle_cls import PaddleOCRClient


# Optional icon classifier hook (do not require it)
try:
    from cnn_cls import EfficientNetClient  # type: ignore
except Exception:  # pragma: no cover
    EfficientNetClient = None  # type: ignore[assignment]


@dataclass
class OCRItem:
    x: int
    y: int
    w: int
    h: int
    text: str
    conf: float


@dataclass
class SemanticDetection:
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    description: str = ""
    element_type: str = ""
    confidence: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalSemanticConfig:
    endpoint: str
    api_key: str
    model_version: str
    timeout_s: float
    max_calls_per_run: int
    min_interval_s: float
    iou_threshold: float
    min_area_px: int
    max_area_fraction: float
    quantize_px: int
    clickable_confidence: float
    force: bool

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint.strip())

    @staticmethod
    def from_env() -> "ExternalSemanticConfig":
        def f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, default))
            except Exception:
                return float(default)

        def i(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, default))
            except Exception:
                return int(default)

        def b(name: str, default: bool = False) -> bool:
            raw = os.getenv(name, "1" if default else "0").strip().lower()
            return raw in ("1", "true", "yes", "y", "on")

        endpoint = os.getenv("UI_SEMANTIC_ENDPOINT", "").strip()
        api_key = os.getenv("UI_SEMANTIC_API_KEY", "").strip()
        model_version = (os.getenv("UI_SEMANTIC_MODEL_VERSION", "").strip() or "v1")

        return ExternalSemanticConfig(
            endpoint=endpoint,
            api_key=api_key,
            model_version=model_version,
            timeout_s=f("UI_SEMANTIC_TIMEOUT_S", 8.0),
            max_calls_per_run=i("UI_SEMANTIC_BUDGET", 12),
            min_interval_s=f("UI_SEMANTIC_MIN_INTERVAL_S", 1.0),
            iou_threshold=f("UI_SEMANTIC_MERGE_IOU", 0.70),
            min_area_px=i("UI_SEMANTIC_MIN_AREA_PX", 14 * 14),
            max_area_fraction=f("UI_SEMANTIC_MAX_AREA_FRACTION", 0.92),
            quantize_px=i("UI_SEMANTIC_QUANTIZE_PX", 4),
            clickable_confidence=f("UI_SEMANTIC_CLICKABLE_CONF", 0.72),
            force=b("UI_SEMANTIC_FORCE", default=False),
        )


class BaseUI:
    """
    Minimal stable API used by workflow + actions:
    - post_process_ui(uist, screenshot_b64) -> (uist2, vid_map)
    - get_text_list(uist) -> list[str]
    - get_center(node) -> {"x","y"}
    - get_frame(node) -> frame dict
    - to_digest(uist) -> compact for LLM
    """

    # In-memory OCR cache: screenshot_hash -> list[OCRItem]
    _ocr_cache: Dict[str, List[OCRItem]] = {}
    _icon_cache: Dict[str, Dict[Tuple[int, int, int, int], Tuple[str, float]]] = {}
    _uied_cache: Dict[str, Dict[str, Any]] = {}

    # Post-process bundle versions for cache coherence.
    # NOTE: bump these when changing merge/threshold logic or provider outputs.
    _POSTPROCESS_BUNDLE_VERSION: str = "v1"
    _OCR_PROVIDER_VERSION: str = "paddleocr:v1"
    _ICON_PROVIDER_VERSION: str = "efficientnet:v1"
    _EXTERNAL_SEMANTIC_PROVIDER_VERSION: str = "external_semantic_provider:v1"
    _UIED_PROVIDER_VERSION: str = "uied_merge:v1"

    # External semantic caches and budgets (process-local; reset per run invocation).
    _semantic_cache: Dict[Tuple[str, str, str], List[SemanticDetection]] = {}
    _semantic_calls_made: int = 0
    _semantic_last_call_ts: float = 0.0
    _semantic_recent_hashes: List[Tuple[float, str]] = []

    @staticmethod
    def postprocess_version_bundle() -> str:
        """
        Stable version bundle string for anything derived from post_process_ui outputs.
        This should be included in cache keys that store uist/vid_map/state_sig.
        """
        cfg = ExternalSemanticConfig.from_env()
        if cfg.enabled:
            ext_part = (
                f"external_semantic:{BaseUI._EXTERNAL_SEMANTIC_PROVIDER_VERSION}:{cfg.model_version}"
                f":iou={cfg.iou_threshold:.2f}:min_area={cfg.min_area_px}:q={cfg.quantize_px}"
                f":click_conf={cfg.clickable_confidence:.2f}:max_area={cfg.max_area_fraction:.2f}"
            )
        else:
            ext_part = "external_semantic:off"
        return "|".join(
            [
                f"bundle:{BaseUI._POSTPROCESS_BUNDLE_VERSION}",
                f"ocr:{BaseUI._OCR_PROVIDER_VERSION}",
                f"icon:{BaseUI._ICON_PROVIDER_VERSION}",
                f"uied:{BaseUI._UIED_PROVIDER_VERSION}",
                ext_part,
            ]
        )

    # ---------------------------
    # Geometry helpers
    # ---------------------------

    @staticmethod
    def get_frame(node: Dict[str, Any]) -> Dict[str, int]:
        f = node.get("absolute_frame") or node.get("frame") or {}
        return {
            "x": int(f.get("x", 0)),
            "y": int(f.get("y", 0)),
            "width": int(f.get("width", 0)),
            "height": int(f.get("height", 0)),
        }

    @staticmethod
    def get_center(node: Dict[str, Any]) -> Dict[str, int]:
        f = BaseUI.get_frame(node)
        return {
            "x": int(f["x"] + max(1, f["width"]) / 2),
            "y": int(f["y"] + max(1, f["height"]) / 2),
        }

    # ---------------------------
    # Tree traversal
    # ---------------------------

    @staticmethod
    def iter_nodes(uist: Dict[str, Any]):
        stack = list(uist.get("elements", []) or [])
        while stack:
            n = stack.pop()
            yield n
            subs = n.get("subviews", []) or []
            if subs:
                stack.extend(reversed(subs))

    @staticmethod
    def iter_nodes_with_parent(uist: Dict[str, Any]):
        stack = [(None, n) for n in (uist.get("elements", []) or [])]
        while stack:
            parent, n = stack.pop()
            yield parent, n
            subs = n.get("subviews", []) or []
            for ch in reversed(subs):
                stack.append((n, ch))

    # ---------------------------
    # OCR + icon hooks
    # ---------------------------

    @staticmethod
    def _hash_screenshot(screenshot_b64: str) -> str:
        if not screenshot_b64:
            return ""
        try:
            raw = base64.b64decode(screenshot_b64[:20000] + "==", validate=False)
        except Exception:
            raw = screenshot_b64.encode("utf-8", errors="ignore")
        return hashlib.md5(raw).hexdigest()

    @staticmethod
    def _hash_screenshot_full(screenshot_b64: str) -> str:
        if not screenshot_b64:
            return ""
        return hashlib.md5((screenshot_b64 or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _needs_ocr(uist: Dict[str, Any]) -> bool:
        """
        OCR is expensive; only run it when likely to add value.

        Overrides:
          - UI_OCR_FORCE=1 enables OCR on every snapshot.
          - UI_OCR_DISABLE=1 disables OCR entirely.
        """
        try:
            if os.getenv("UI_OCR_DISABLE", "0").strip().lower() in ("1", "true", "yes", "y", "on"):
                return False
            if os.getenv("UI_OCR_FORCE", "0").strip().lower() in ("1", "true", "yes", "y", "on"):
                return True
        except Exception:
            pass

        clickables = 0
        unlabeled = 0
        any_text = 0
        for n in BaseUI.iter_nodes(uist):
            txt = (n.get("text") or "").strip()
            cdesc = (n.get("content_desc") or "").strip()
            rid = (n.get("resource_id") or "").strip()
            if txt or cdesc:
                any_text += 1
            if n.get("clickable"):
                clickables += 1
                if not (txt or cdesc or rid):
                    unlabeled += 1

        # Very sparse trees or icon-heavy pages benefit most.
        if clickables <= 1:
            return True
        if clickables >= 5 and (unlabeled / max(1, clickables)) >= 0.5:
            return True
        if any_text <= 3:
            return True
        return False

    @staticmethod
    def _run_ocr_cached(screenshot_b64: str, force: bool = False) -> List[OCRItem]:
        if not screenshot_b64:
            return []
        key = BaseUI._hash_screenshot_full(screenshot_b64)
        if not force and key in BaseUI._ocr_cache:
            return BaseUI._ocr_cache[key]
        raw = PaddleOCRClient.ocr(screenshot_b64, debug_output=False)
        items = [OCRItem(*r) for r in raw]
        BaseUI._ocr_cache[key] = items
        return items

    @staticmethod
    def _attach_ocr(uist: Dict[str, Any], ocr_items: List[OCRItem]) -> None:
        """
        Attach OCR text to nearest UI node by overlap.
        For icon-only controls, OCR text often appears near them; we attach only if overlap is meaningful.
        """
        if not ocr_items:
            return

        nodes = list(BaseUI.iter_nodes(uist))
        if not nodes:
            return

        # Precompute frames
        frames = []
        for n in nodes:
            f = BaseUI.get_frame(n)
            frames.append((n, f["x"], f["y"], f["x"] + f["width"], f["y"] + f["height"]))

        def iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a = (ax2 - ax1) * (ay2 - ay1)
            b = (bx2 - bx1) * (by2 - by1)
            return inter / max(1.0, (a + b - inter))

        for o in ocr_items:
            if not o.text or o.conf < 0.5:
                continue
            ox1, oy1, ox2, oy2 = o.x, o.y, o.x + o.w, o.y + o.h
            best = None
            best_iou = 0.0
            for n, x1, y1, x2, y2 in frames:
                s = iou(ox1, oy1, ox2, oy2, x1, y1, x2, y2)
                if s > best_iou:
                    best, best_iou = n, s
            if best is not None and best_iou >= 0.15:
                # don't overwrite real text; attach as ocr_text
                if not (best.get("text") or best.get("content_desc")):
                    best["ocr_text"] = o.text.strip()
                    best["ocr_conf"] = float(o.conf)

    @staticmethod
    def _run_uied_cached(screenshot_b64: str) -> Dict[str, Any]:
        # 作用：
        # 1. 以当前截图为输入，调用 UIED 跑 OCR / IP / MERGE
        # 2. 把中间结果落盘到 mytests/outputs/uied_runtime/<截图hash>/
        # 3. 用内存缓存避免同一张截图重复跑 UIED
        #
        # 为什么放在 ui_cls.py：
        # - workflow 只负责调度；真正“如何补树”属于 UI 后处理逻辑
        # - 这样后面无论从 _capture_and_process 还是 _capture_and_process2 调用，都可以复用
        if not screenshot_b64:
            return {"ocr": {"texts": []}, "merge": {"compos": []}, "ip": {"compos": []}}

        # 用整张截图的 base64 算一个稳定 hash，作为缓存 key 和落盘目录名。
        # 同一张截图重复进入这里时，直接复用结果，不再重复执行 UIED。
        key = BaseUI._hash_screenshot_full(screenshot_b64)
        cached = BaseUI._uied_cache.get(key)
        if cached is not None:
            return cached

        # 约定 UIED 的运行时输出目录。
        # 这里落盘的内容方便你后续人工检查：
        # - 原始截图
        # - ocr/*.json
        # - ip/*.json
        # - merge/*.json
        project_root = Path(__file__).resolve().parent
        mytests_dir = project_root / "mytests"
        runtime_root = mytests_dir / "outputs" / "uied_runtime" / key
        runtime_root.mkdir(parents=True, exist_ok=True)

        # 先把 base64 截图写成 png 文件，因为 UIED 当前入口要求传入图片路径。
        img_path = runtime_root / f"{key}.png"
        img_bytes = base64.b64decode(screenshot_b64)
        img_path.write_bytes(img_bytes)

        import sys

        # 运行时补 sys.path，确保可以从 framework 根目录导入项目级的 LayoutCoder_develop。
        # 你现在已经把工具目录放到了 framework/LayoutCoder_develop，
        # 所以后续统一按顶级包 LayoutCoder_develop 来导入，避免再依赖 mytests 下的副本路径。
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from LayoutCoder_develop.run_single import uied

        # 下面开始真正执行 UIED。
        # 这里额外记录 key 和 output_root，方便主流程日志里快速定位问题页面。
        logger.debug(
            "UIED start: key=%s output_root=%s image=%s",
            key,
            str(runtime_root),
            str(img_path),
        )

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            # 这一步内部会依次尝试执行 OCR -> IP -> MERGE。
            # UIED 脚本内部会打印大量阶段日志，这里默认静默，避免调试主流程时刷屏。
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                uied(input_path_img=str(img_path), output_root=str(runtime_root))
        except Exception as exc:
            # 如果 UIED 主入口自己抛异常，在这里补齐定位信息后继续向外抛。
            logger.exception(
                "UIED execution failed: key=%s output_root=%s error=%s stdout_tail=%s stderr_tail=%s",
                key,
                str(runtime_root),
                repr(exc),
                (stdout_buffer.getvalue() or "")[-600:],
                (stderr_buffer.getvalue() or "")[-600:],
            )
            raise

        def load_json(path: Path, default: Any) -> Any:
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return default

        # 分阶段检查 UIED 的产物是否真的生成成功。
        # 这样即使 UIED 内部只跑到 OCR、没跑完 IP/MERGE，我们也能准确知道卡在哪一步。
        ocr_json = runtime_root / "ocr" / f"{key}.json"
        ip_json = runtime_root / "ip" / f"{key}.json"
        merge_json = runtime_root / "merge" / f"{key}.json"

        ocr_ok = ocr_json.exists()
        ip_ok = ip_json.exists()
        merge_ok = merge_json.exists()

        logger.debug(
            "UIED outputs: key=%s ocr_ok=%s ip_ok=%s merge_ok=%s output_root=%s",
            key,
            ocr_ok,
            ip_ok,
            merge_ok,
            str(runtime_root),
        )

        if not ocr_ok:
            raise FileNotFoundError(
                f"UIED OCR output missing: key={key} path={ocr_json}"
            )
        if not ip_ok:
            raise FileNotFoundError(
                f"UIED IP output missing: key={key} path={ip_json}"
            )
        if not merge_ok:
            raise FileNotFoundError(
                f"UIED MERGE output missing: key={key} path={merge_json}"
            )

        # 把 UIED 原始结果统一组织成一个 dict 返回，便于后续：
        # - OCR 挂载到已有 XML 节点
        # - MERGE 结果补成 synthetic node
        result = {
            "image_path": str(img_path),
            "output_root": str(runtime_root),
            "ocr": load_json(ocr_json, {"img_shape": None, "texts": []}),
            "ip": load_json(ip_json, {"img_shape": None, "compos": []}),
            "merge": load_json(merge_json, {"img_shape": None, "compos": []}),
        }

        logger.debug(
            "UIED success: key=%s ocr_texts=%d ip_compos=%d merge_compos=%d",
            key,
            len((result.get("ocr") or {}).get("texts", []) or []),
            len((result.get("ip") or {}).get("compos", []) or []),
            len((result.get("merge") or {}).get("compos", []) or []),
        )
        BaseUI._uied_cache[key] = result
        return result

    @staticmethod
    def _uied_ocr_items(uied_result: Dict[str, Any]) -> List[OCRItem]:
        # 作用：
        # - 把 UIED OCR 的原始 json 转成框架已有的 OCRItem 结构
        #
        # 为什么要做这一步：
        # - 这样可以直接复用你们原来的 _attach_ocr() 逻辑
        # - 不需要重写“如何把 OCR 文本贴回现有节点”这部分代码
        out: List[OCRItem] = []
        ocr_data = (uied_result or {}).get("ocr") or {}
        for item in ocr_data.get("texts", []) or []:
            try:
                x1 = int(item.get("column_min", 0))
                y1 = int(item.get("row_min", 0))
                x2 = int(item.get("column_max", 0))
                y2 = int(item.get("row_max", 0))
                text = str(item.get("content") or "").strip()
                if not text:
                    continue
                conf = float(item.get("score", 1.0) or 1.0)
                out.append(OCRItem(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1), text=text, conf=conf))
            except Exception:
                continue
        return out

    @staticmethod
    def _enrich_uied_merge_with_ocr(uied_result: Dict[str, Any]) -> Dict[str, Any]:
        # 作用：
        # - 先在 UIED 内部把 OCR 结果并入 merge 节点
        # - 后续补树时，以“已经带文字的 merge 节点”为准
        #
        # 为什么这样做：
        # - 这条 UIED-first 链里，我们希望以 merge 结果为中心，而不是先把 OCR 单独挂回 XML
        # - 这样可以避免“两次独立对齐”造成的文字丢失
        # - 也更接近 darkcode 那种“以 merge 结果补树”的思路
        #
        # 注意：
        # - 这里只增强 merge 结果本身
        # - 不会在这里把 OCR 直接写回 XML 节点
        if not uied_result:
            return {"ocr": {"texts": []}, "merge": {"compos": []}, "ip": {"compos": []}}

        merge_data = (uied_result.get("merge") or {})
        ocr_data = (uied_result.get("ocr") or {})
        compos = merge_data.get("compos", []) or []
        ocr_items = ocr_data.get("texts", []) or []

        def frame_from_compo(compo: Dict[str, Any]) -> Dict[str, int]:
            pos = compo.get("position") or {}
            x1 = int(pos.get("column_min", 0))
            y1 = int(pos.get("row_min", 0))
            x2 = int(pos.get("column_max", x1))
            y2 = int(pos.get("row_max", y1))
            w = int(compo.get("width", max(1, x2 - x1)))
            h = int(compo.get("height", max(1, y2 - y1)))
            return {"x": x1, "y": y1, "width": max(1, w), "height": max(1, h)}

        def frame_from_ocr(item: Dict[str, Any]) -> Dict[str, int]:
            x1 = int(item.get("column_min", 0))
            y1 = int(item.get("row_min", 0))
            x2 = int(item.get("column_max", x1))
            y2 = int(item.get("row_max", y1))
            return {"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)}

        def center_in(inner: Dict[str, int], outer: Dict[str, int]) -> bool:
            cx = int(inner["x"] + inner["width"] / 2)
            cy = int(inner["y"] + inner["height"] / 2)
            return (
                outer["x"] <= cx <= outer["x"] + outer["width"]
                and outer["y"] <= cy <= outer["y"] + outer["height"]
            )

        def overlap_ratio(inner: Dict[str, int], outer: Dict[str, int]) -> float:
            ix1 = max(inner["x"], outer["x"])
            iy1 = max(inner["y"], outer["y"])
            ix2 = min(inner["x"] + inner["width"], outer["x"] + outer["width"])
            iy2 = min(inner["y"] + inner["height"], outer["y"] + outer["height"])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            inner_area = max(1, inner["width"] * inner["height"])
            return float(inter) / float(inner_area)

        for compo in compos:
            compo_frame = frame_from_compo(compo)
            matched_texts: List[Dict[str, Any]] = []
            for item in ocr_items:
                text = str(item.get("content") or "").strip()
                if not text:
                    continue
                ocr_frame = frame_from_ocr(item)
                # 匹配策略复用原先“按位置归属元素”的思路：
                # - OCR 框中心点落在 merge 节点里，直接认为属于它
                # - 或者 OCR 框与 merge 节点有明显重叠
                # 这样可以避免把相邻元素的文字误并进来。
                if center_in(ocr_frame, compo_frame) or overlap_ratio(ocr_frame, compo_frame) >= 0.50:
                    matched_texts.append(item)

            # 为了保证字符串稳定，多个 OCR 文本框按页面阅读顺序排序：
            # 1. 先按 row_min（从上到下）
            # 2. 再按 column_min（从左到右）
            matched_texts.sort(key=lambda x: (int(x.get("row_min", 0)), int(x.get("column_min", 0))))

            merged_text = "\n".join(
                str(item.get("content") or "").strip()
                for item in matched_texts
                if str(item.get("content") or "").strip()
            ).strip()

            compo["matched_ocr_items"] = matched_texts
            compo["matched_ocr_text"] = merged_text

            # 如果 merge 自己没有 text_content，就用 OCR 文本补上；
            # 如果 merge 自己已有 text_content，则保留它，同时额外保存 matched_ocr_text。
            if merged_text and not str(compo.get("text_content") or "").strip():
                compo["text_content"] = merged_text

        return uied_result

    @staticmethod
    def _frame_iou(a: Dict[str, int], b: Dict[str, int]) -> float:
        # 计算两个矩形框的 IoU（Intersection over Union）。
        # 这个值越大，说明两个框越像在描述同一个视觉区域。
        ax1, ay1 = int(a.get("x", 0)), int(a.get("y", 0))
        ax2, ay2 = ax1 + int(a.get("width", 0)), ay1 + int(a.get("height", 0))
        bx1, by1 = int(b.get("x", 0)), int(b.get("y", 0))
        bx2, by2 = bx1 + int(b.get("width", 0)), by1 + int(b.get("height", 0))
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return float(inter) / float(max(1, area_a + area_b - inter))

    @staticmethod
    def _frames_almost_overlap(a: Dict[str, int], b: Dict[str, int], tolerance: float = 0.70) -> bool:
        # 这个函数比 IoU 更“保守”一点，专门用来判断：
        # “这两个框是不是几乎就是同一个元素”
        #
        # 规则：
        # - 交集面积 / A面积 >= tolerance
        # - 且 交集面积 / B面积 >= tolerance
        #
        # 这样做是参考 darkcode 的 almost_overlap 思路。
        # 好处是：当两个框彼此高度覆盖时，就认为它们应该合并，而不是重复插入。
        ax1, ay1 = int(a.get("x", 0)), int(a.get("y", 0))
        ax2, ay2 = ax1 + int(a.get("width", 0)), ay1 + int(a.get("height", 0))
        bx1, by1 = int(b.get("x", 0)), int(b.get("y", 0))
        bx2, by2 = bx1 + int(b.get("width", 0)), by1 + int(b.get("height", 0))
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return False
        area_a = max(1, int(a.get("width", 0)) * int(a.get("height", 0)))
        area_b = max(1, int(b.get("width", 0)) * int(b.get("height", 0)))
        return (inter / area_a) >= tolerance and (inter / area_b) >= tolerance

    @staticmethod
    def _find_best_uied_host(
        uist: Dict[str, Any],
        frame: Dict[str, int],
        tolerance: float = 0.20,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        # 作用：
        # - 给一个 UIED 检出的框，尝试在当前 uist 里找到“最适合挂靠”的节点
        #
        # 思路参考 darkcode 的 find_suitable_element：
        # - 先找所有“能包住这个框”的节点
        # - 再从中选面积最小、层级最深的那个
        #
        # 这样做的原因：
        # - 如果直接把 UIED 节点都挂到 root，树会很乱
        # - 更合理的是尽量挂到它视觉上所属的父节点下面
        target_x = int(frame.get("x", 0))
        target_y = int(frame.get("y", 0))
        target_w = int(frame.get("width", 0))
        target_h = int(frame.get("height", 0))

        best_parent: Optional[Dict[str, Any]] = None
        best_node: Optional[Dict[str, Any]] = None
        best_depth = -1
        min_area = float("inf")

        def contains(outer: Dict[str, int], inner: Dict[str, int]) -> bool:
            ox, oy = int(outer.get("x", 0)), int(outer.get("y", 0))
            ow, oh = int(outer.get("width", 0)), int(outer.get("height", 0))
            ix, iy = int(inner.get("x", 0)), int(inner.get("y", 0))
            iw, ih = int(inner.get("width", 0)), int(inner.get("height", 0))
            return (
                ox - ow * tolerance <= ix
                and ix + iw <= ox + ow + ow * tolerance
                and oy - oh * tolerance <= iy
                and iy + ih <= oy + oh + oh * tolerance
            )

        def walk(parent: Optional[Dict[str, Any]], node: Dict[str, Any], depth: int) -> None:
            nonlocal best_parent, best_node, best_depth, min_area
            node_frame = BaseUI.get_frame(node)
            if contains(node_frame, frame):
                area = max(1, node_frame["width"] * node_frame["height"])
                if area < min_area or (area == min_area and depth > best_depth):
                    best_parent = parent
                    best_node = node
                    best_depth = depth
                    min_area = area
            for child in node.get("subviews", []) or []:
                walk(node, child, depth + 1)

        for root in uist.get("elements", []) or []:
            walk(None, root, 0)
        return best_parent, best_node

    @staticmethod
    def _make_uied_node(compo: Dict[str, Any], frame: Dict[str, int]) -> Dict[str, Any]:
        # 把 UIED merge 里的一个 compo 转成框架自己的节点格式。
        #
        # 注意：
        # - 第一版先保守处理，默认 clickable=False
        # - 这里的目标是“先补结构”，不是立刻把所有 UIED 节点都纳入动作候选
        # - 后面如果要让它参与点击，可以在这里或后续流程里加 clickable 推断
        raw_class = str(compo.get("class") or "").strip()
        text = str(compo.get("text_content") or compo.get("matched_ocr_text") or "").strip()
        node_class = "uied_text" if raw_class == "Text" else "uied_compo"
        node: Dict[str, Any] = {
            "class": node_class,
            "text": text if raw_class == "Text" else "",
            "content_desc": "",
            "resource_id": "",
            # "clickable": False,
            "clickable": True,
            "enabled": True,
            "absolute_frame": frame,
            "subviews": [],
            "uied_class": raw_class,
            "semantic_source": "uied_merge",
        }
        if text and raw_class != "Text":
            node["ocr_text"] = text
        return node

    @staticmethod
    def _is_uied_ocr_action_text(text: str) -> bool:
        norm = re.sub(r"\s+", " ", str(text or "").strip()).lower()
        if not norm:
            return False
        safe_labels = {
            "no",
            "yes",
            "ok",
            "okay",
            "cancel",
            "close",
            "skip",
            "deny",
            "allow",
            "accept",
            "decline",
            "later",
            "continue",
            "confirm",
            "not now",
            "got it",
            "dismiss",
            "x",
            "否",
            "是",
            "取消",
            "确定",
            "关闭",
            "跳过",
            "稍后",
            "允许",
            "拒绝",
            "同意",
            "不同意",
        }
        return norm in safe_labels

    @staticmethod
    def _add_uied_ocr_action_nodes(uist: Dict[str, Any], uied_result: Dict[str, Any]) -> None:
        # UIED merge 有时会把按钮文字吞进大 Block，导致 NO/YES 这种按钮不进 vid_map。
        # 这里不重新 OCR，只复用 UIED 已经落盘的 ocr/*.json，把短按钮文本补成可点击节点。
        ocr_data = (uied_result or {}).get("ocr") or {}
        added = 0

        def existing_same_label(frame: Dict[str, int], text: str) -> bool:
            for n in BaseUI.iter_nodes(uist):
                label = str(n.get("text") or n.get("content_desc") or n.get("ocr_text") or "").strip()
                if label.lower() != text.lower():
                    continue
                if BaseUI._frames_almost_overlap(BaseUI.get_frame(n), frame, tolerance=0.45):
                    return True
            return False

        for item in ocr_data.get("texts", []) or []:
            text = str(item.get("content") or "").strip()
            if not BaseUI._is_uied_ocr_action_text(text):
                continue
            try:
                x1 = int(item.get("column_min", 0))
                y1 = int(item.get("row_min", 0))
                x2 = int(item.get("column_max", x1))
                y2 = int(item.get("row_max", y1))
            except Exception:
                continue
            frame = {"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)}
            if frame["width"] <= 1 or frame["height"] <= 1:
                continue
            if existing_same_label(frame, text):
                continue

            node: Dict[str, Any] = {
                "class": "uied_ocr_action",
                "text": text,
                "content_desc": text,
                "resource_id": "",
                "clickable": True,
                "enabled": True,
                "absolute_frame": frame,
                "subviews": [],
                "uied_class": "OCRText",
                "semantic_source": "uied_ocr_action",
                "ocr_text": text,
            }
            _, host = BaseUI._find_best_uied_host(uist, frame, tolerance=0.20)
            if host is not None:
                host.setdefault("subviews", []).append(node)
            else:
                uist.setdefault("elements", []).append(node)
            added += 1

        if added:
            logger.debug("Injected %d UIED OCR action nodes", added)

    @staticmethod
    def _merge_uied_into_uist(uist: Dict[str, Any], uied_result: Dict[str, Any]) -> None:
        # 这是 UIED 补树的核心逻辑。
        #
        # 输入：
        # - 现有的 XML 解析树 uist
        # - UIED 的原始结果（重点使用 merge.compos）
        #
        # 输出：
        # - 直接原地修改 uist
        #
        # 总体策略：
        # 1. 先遍历 UIED merge 结果
        # 2. 跳过明显噪声（例如 Block）
        # 3. 对每个 UIED 元素，先找最合适的宿主节点
        # 4. 如果和现有节点高度重合 -> 认为是同一个元素，优先合并信息
        # 5. 如果不重合 -> 新建 synthetic node 插回树里
        #
        # 这就是“保留 XML 为主干，UIED 作为补充”的实现。
        merge_data = (uied_result or {}).get("merge") or {}
        compos = merge_data.get("compos", []) or []
        if not compos:
            return

        synthetic_roots: List[Dict[str, Any]] = []
        seen_frames: List[Dict[str, int]] = []

        for compo in compos:
            raw_class = str(compo.get("class") or "").strip()
            # Block 往往是大容器/背景块，第一版先不引入，避免噪声太多。
            if raw_class == "Block":
                continue

            pos = compo.get("position") or {}
            frame = {
                "x": int(pos.get("column_min", 0)),
                "y": int(pos.get("row_min", 0)),
                "width": int(compo.get("width", max(1, int(pos.get("column_max", 0)) - int(pos.get("column_min", 0))))),
                "height": int(compo.get("height", max(1, int(pos.get("row_max", 0)) - int(pos.get("row_min", 0))))),
            }
            if frame["width"] <= 1 or frame["height"] <= 1:
                continue

            # 先做一次本轮 UIED 检测结果内部去重：
            # 如果前面已经处理过一个几乎完全相同的框，就跳过，避免重复插入。
            if any(BaseUI._frames_almost_overlap(frame, old, tolerance=0.92) for old in seen_frames):
                continue

            # 找这个 UIED 框在现有 XML 树里最可能属于哪个节点。
            _, host = BaseUI._find_best_uied_host(uist, frame, tolerance=0.20)
            if host is not None and BaseUI._frames_almost_overlap(BaseUI.get_frame(host), frame, tolerance=0.70):
                # 如果和现有节点高度重合，认为它们其实是“同一个元素”。
                # 按当前你的要求：
                # - 这里不再把 OCR 文本补到 XML 节点上
                # - 只记录“它和 UIED merge 对应上了”
                # 这样可以避免原 XML 节点被 UIED OCR 二次覆盖。
                host["uied_source"] = raw_class or "merge"
                seen_frames.append(frame)
                continue

            # 走到这里说明：
            # - 没找到合适宿主；或者
            # - 找到了宿主，但它并不是同一个元素
            # 此时把它作为新的 synthetic node 插回树里。
            new_node = BaseUI._make_uied_node(compo, frame)
            if host is not None:
                host.setdefault("subviews", []).append(new_node)
            else:
                # 连宿主都找不到时，先挂到 root，方便后续调试观察。
                synthetic_roots.append(new_node)
            seen_frames.append(frame)

        if synthetic_roots:
            uist.setdefault("elements", [])
            uist["elements"].extend(synthetic_roots)

    @staticmethod
    def _attach_icon_labels(uist: Dict[str, Any], screenshot_b64: str) -> None:
        """
        FIXED: integrate with your EfficientNetClient correctly.

        Your classifier:
            EfficientNetClient.classify(image: io.BytesIO) -> (label, confidence)

        We:
        - decode screenshot once to PIL.Image
        - crop candidate icon boxes
        - call EfficientNetClient.classify(BytesIO(cropped_png))
        - cache results by screenshot hash + box
        Icon labels help recover semantics when XML lacks text (e.g., toolbar buttons).
        """
        if EfficientNetClient is None or not screenshot_b64:
            return

        # decode screenshot to PIL image once
        try:
            img_bytes = base64.b64decode(screenshot_b64)
            base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            W, H = base_img.size
        except Exception:
            logger.debug("Icon classification skipped: screenshot decode failed", exc_info=True)
            return

        key = BaseUI._hash_screenshot_full(screenshot_b64)
        cached = BaseUI._icon_cache.get(key, {})
        updated_cache = dict(cached)

        # limits/thresholds to keep runtime bounded
        MAX_ICONS_PER_PAGE = 24
        MIN_CONF = 0.55

        # collect candidates: small clickables with no textual labels
        candidates: List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = []
        for n in BaseUI.iter_nodes(uist):
            if not n.get("clickable"):
                continue
            if (n.get("text") or n.get("content_desc") or n.get("resource_id") or n.get("ocr_text")):
                continue
            f = BaseUI.get_frame(n)
            x, y, w, h = f["x"], f["y"], f["width"], f["height"]
            if w <= 0 or h <= 0:
                continue
            area = w * h
            # conservative: likely icons only
            if area > 140000:  # too large to be an icon
                continue
            if w < 12 or h < 12:  # too tiny / noise
                continue
            box = (x, y, w, h)
            candidates.append((box, n))

        # prioritize smaller ones first (more icon-like)
        candidates.sort(key=lambda t: (t[0][2] * t[0][3]))

        classified = 0
        for box, n in candidates:
            if classified >= MAX_ICONS_PER_PAGE:
                break

            if box in updated_cache:
                label, conf = updated_cache[box]
            else:
                x, y, w, h = box
                # clamp to image bounds
                x1 = max(0, min(W, x))
                y1 = max(0, min(H, y))
                x2 = max(0, min(W, x + w))
                y2 = max(0, min(H, y + h))
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue

                crop = base_img.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                buf.seek(0)

                try:
                    pred = EfficientNetClient.classify(buf)  # returns (label, conf) in your code
                    if isinstance(pred, tuple) and len(pred) >= 2:
                        label, conf = str(pred[0]), float(pred[1])
                    else:
                        label, conf = str(pred), 0.0
                except Exception:
                    logger.debug("Icon classify failed for box=%s", box, exc_info=True)
                    continue

                updated_cache[box] = (label, conf)

            classified += 1

            # attach only if high enough confidence
            if label and conf >= MIN_CONF:
                n["icon_label"] = label
                n["icon_conf"] = conf

        BaseUI._icon_cache[key] = updated_cache

    # ---------------------------
    # External semantic detection (optional provider)
    # ---------------------------

    @staticmethod
    def _screen_size(screenshot_b64: str, device_info: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
        # Prefer device_info if it contains plausible dimensions.
        di = device_info or {}
        for kx, ky in (("width", "height"), ("screenWidth", "screenHeight"), ("displayWidth", "displayHeight")):
            try:
                w = int(di.get(kx, 0) or 0)
                h = int(di.get(ky, 0) or 0)
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass

        try:
            if not screenshot_b64:
                return 0, 0
            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            return int(w), int(h)
        except Exception:
            return 0, 0

    @staticmethod
    def _should_run_external_semantic(cfg: ExternalSemanticConfig, uist: Dict[str, Any]) -> bool:
        if not cfg.enabled:
            return False
        if cfg.force:
            return True

        nodes = list(BaseUI.iter_nodes(uist))
        if not nodes:
            return True

        clickables = 0
        meaningful = 0
        for n in nodes:
            if n.get("clickable"):
                clickables += 1
            if (n.get("text") or n.get("content_desc") or n.get("ocr_text") or n.get("icon_label") or n.get("semantic_label")):
                meaningful += 1

        # Triggers: empty/sparse hierarchy, few clickables, few labels.
        if clickables <= 1:
            return True
        if clickables <= 3 and meaningful <= 5:
            return True
        return False

    @staticmethod
    def _external_device_bucket(screen_w: int, screen_h: int, device_info: Optional[Dict[str, Any]] = None) -> str:
        # Bucket sizes to avoid over-fragmenting caches across tiny variations.
        bw = int(screen_w // 100) * 100 if screen_w > 0 else 0
        bh = int(screen_h // 100) * 100 if screen_h > 0 else 0
        orient = ""
        try:
            orient = str((device_info or {}).get("orientation") or "").strip().lower()
        except Exception:
            orient = ""
        return f"{bw}x{bh}:{orient or 'na'}"

    @staticmethod
    def _external_semantic_parse(raw: Any) -> List[SemanticDetection]:
        if not raw:
            return []
        items = raw
        if isinstance(raw, dict):
            items = raw.get("elements") or raw.get("detections") or raw.get("items") or []
        if not isinstance(items, list):
            return []

        out: List[SemanticDetection] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            b = it.get("bounds") or it.get("bbox") or it.get("rect")
            x = y = w = h = None
            if isinstance(b, dict):
                x = b.get("x")
                y = b.get("y")
                w = b.get("w", None if "w" not in b else b.get("w"))
                if w is None:
                    w = b.get("width")
                h = b.get("h", None if "h" not in b else b.get("h"))
                if h is None:
                    h = b.get("height")
                if (w is None or h is None) and ("x2" in b and "y2" in b):
                    try:
                        w = float(b["x2"]) - float(x or 0)
                        h = float(b["y2"]) - float(y or 0)
                    except Exception:
                        pass
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    x, y, w, h = b
                except Exception:
                    x = y = w = h = None

            try:
                xi = int(float(x or 0))
                yi = int(float(y or 0))
                wi = int(float(w or 0))
                hi = int(float(h or 0))
            except Exception:
                continue

            out.append(
                SemanticDetection(
                    x=xi,
                    y=yi,
                    w=wi,
                    h=hi,
                    label=str(it.get("label") or "").strip(),
                    description=str(it.get("description") or it.get("desc") or "").strip(),
                    element_type=str(it.get("type") or it.get("role") or "").strip(),
                    confidence=float(it.get("confidence") or 0.0),
                    attributes=dict(it.get("attributes") or {}),
                )
            )
        return out

    @staticmethod
    def _merge_external_semantic(
        uist: Dict[str, Any],
        detections: List[SemanticDetection],
        *,
        screen_w: int,
        screen_h: int,
        cfg: ExternalSemanticConfig,
        provider_tag: str,
        pixel_ratio: float,
    ) -> None:
        if not detections:
            return
        if screen_w <= 0 or screen_h <= 0:
            return

        def clamp(v: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, v))

        def quant(v: int, q: int) -> int:
            if q <= 1:
                return int(v)
            return int(round(float(v) / float(q)) * q)

        def norm_label(s: str) -> str:
            t = (s or "").strip()
            t = re.sub(r"\d+", "", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        max_area = int(float(screen_w * screen_h) * float(cfg.max_area_fraction))
        min_area = int(cfg.min_area_px)
        tol = int(max(2.0, 4.0 * float(pixel_ratio or 1.0)))

        # Precompute node frames for matching.
        nodes = list(BaseUI.iter_nodes(uist))
        frames: List[Tuple[Dict[str, Any], int, int, int, int]] = []
        for n in nodes:
            f = BaseUI.get_frame(n)
            x1 = int(f["x"])
            y1 = int(f["y"])
            x2 = int(f["x"] + f["width"])
            y2 = int(f["y"] + f["height"])
            frames.append((n, x1, y1, x2, y2))

        def iou(ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> float:
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a = (ax2 - ax1) * (ay2 - ay1)
            b = (bx2 - bx1) * (by2 - by1)
            return float(inter) / float(max(1.0, (a + b - inter)))

        def is_near(ax1: int, ay1: int, aw: int, ah: int, bx1: int, by1: int, bx2: int, by2: int) -> bool:
            bw = bx2 - bx1
            bh = by2 - by1
            if abs(ax1 - bx1) <= tol and abs(ay1 - by1) <= tol and abs(aw - bw) <= tol and abs(ah - bh) <= tol:
                return True
            return False

        normed: List[Tuple[int, int, int, int, str, SemanticDetection]] = []
        seen_norm: set[Tuple[int, int, int, int, str]] = set()
        for d in detections:
            if d.w <= 1 or d.h <= 1:
                continue
            x = clamp(int(d.x), 0, max(0, screen_w - 1))
            y = clamp(int(d.y), 0, max(0, screen_h - 1))
            w = clamp(int(d.w), 1, screen_w)
            h = clamp(int(d.h), 1, screen_h)
            # Clamp width/height to stay on-screen.
            w = min(w, max(1, screen_w - x))
            h = min(h, max(1, screen_h - y))

            area = int(w * h)
            if area < min_area or area > max_area:
                continue

            q = int(cfg.quantize_px)
            xq = quant(x, q)
            yq = quant(y, q)
            wq = max(1, quant(w, q))
            hq = max(1, quant(h, q))

            lbl = norm_label(d.label)
            key = (xq, yq, wq, hq, lbl.lower())
            if key in seen_norm:
                continue
            seen_norm.add(key)
            d2 = SemanticDetection(
                x=xq,
                y=yq,
                w=wq,
                h=hq,
                label=lbl,
                description=(d.description or "").strip(),
                element_type=(d.element_type or "").strip(),
                confidence=float(d.confidence or 0.0),
                attributes=dict(d.attributes or {}),
            )
            normed.append((yq, xq, wq, hq, lbl.lower(), d2))

        # Deterministic ordering for merge stability.
        normed.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))

        inserted: List[Dict[str, Any]] = []
        for _, _, _, _, _, d in normed:
            ax1, ay1 = int(d.x), int(d.y)
            ax2, ay2 = int(d.x + d.w), int(d.y + d.h)

            best_node: Optional[Dict[str, Any]] = None
            best_frame: Optional[Tuple[int, int, int, int]] = None
            best_score = 0.0
            for n, bx1, by1, bx2, by2 in frames:
                s = iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
                if s > best_score:
                    best_node, best_score = n, s
                    best_frame = (bx1, by1, bx2, by2)

            if best_node is not None and (
                best_score >= float(cfg.iou_threshold)
                or (best_frame is not None and is_near(ax1, ay1, d.w, d.h, *best_frame))
            ):
                prev_conf = float(best_node.get("semantic_conf") or 0.0)
                if not best_node.get("semantic_label") or float(d.confidence or 0.0) >= prev_conf:
                    if d.label:
                        best_node["semantic_label"] = d.label
                    if d.description:
                        best_node["semantic_desc"] = d.description
                    if d.element_type:
                        best_node["semantic_type"] = d.element_type
                    best_node["semantic_conf"] = float(d.confidence or 0.0)
                    best_node["semantic_source"] = provider_tag
                continue

            # Insert new synthetic node.
            clickable_guess = False
            et = (d.element_type or "").lower()
            if d.confidence >= float(cfg.clickable_confidence):
                if et in ("button", "tab", "icon", "icon_button", "textfield", "toggle", "checkbox", "radio", "dialog"):
                    clickable_guess = True
                if bool(d.attributes.get("clickable_guess")):
                    clickable_guess = True

            node = {
                "class": "semantic_detected",
                "text": "",
                "content_desc": "",
                "resource_id": "",
                "clickable": bool(clickable_guess),
                "enabled": True,
                "absolute_frame": {"x": int(d.x), "y": int(d.y), "width": int(d.w), "height": int(d.h)},
                "subviews": [],
                "semantic_label": d.label,
                "semantic_desc": d.description,
                "semantic_type": d.element_type,
                "semantic_conf": float(d.confidence or 0.0),
                "semantic_source": provider_tag,
            }
            inserted.append(node)

        if not inserted:
            return

        # Place inserted nodes under a stable container to avoid polluting root ordering.
        container = {
            "class": "semantic_layer",
            "text": "",
            "content_desc": "",
            "resource_id": "",
            "clickable": False,
            "enabled": True,
            "absolute_frame": {"x": 0, "y": 0, "width": int(screen_w), "height": int(screen_h)},
            "subviews": inserted,
            "semantic_source": provider_tag,
        }
        uist.setdefault("elements", [])
        uist["elements"].append(container)

    @staticmethod
    def _attach_external_semantic(uist: Dict[str, Any], screenshot_b64: str, device_info: Optional[Dict[str, Any]] = None) -> None:
        cfg = ExternalSemanticConfig.from_env()
        if not cfg.enabled or not screenshot_b64:
            return
        if not BaseUI._should_run_external_semantic(cfg, uist):
            return

        screen_w, screen_h = BaseUI._screen_size(screenshot_b64, device_info=device_info)
        if screen_w <= 0 or screen_h <= 0:
            return

        bucket = BaseUI._external_device_bucket(screen_w, screen_h, device_info=device_info)
        screenshot_hash = BaseUI._hash_screenshot_full(screenshot_b64)
        cache_key = (screenshot_hash, cfg.model_version, bucket)
        cached = BaseUI._semantic_cache.get(cache_key)
        if cached is not None:
            BaseUI._merge_external_semantic(
                uist,
                cached,
                screen_w=screen_w,
                screen_h=screen_h,
                cfg=cfg,
                provider_tag=f"external_semantic:{cfg.model_version}",
                pixel_ratio=float((device_info or {}).get("pixelRatio", 1.0) or 1.0),
            )
            return

        now = time.time()
        if BaseUI._semantic_calls_made >= int(cfg.max_calls_per_run):
            return
        if (now - BaseUI._semantic_last_call_ts) < float(cfg.min_interval_s):
            return

        # Churn guard: if the screenshot hash is changing rapidly, avoid spamming the API.
        BaseUI._semantic_recent_hashes.append((now, screenshot_hash))
        BaseUI._semantic_recent_hashes = [(t, h) for (t, h) in BaseUI._semantic_recent_hashes if now - t <= 6.0]
        distinct = len({h for _, h in BaseUI._semantic_recent_hashes})
        if distinct >= 6 and not cfg.force:
            return

        payload = {
            "screenshot": screenshot_b64,
            "device_info": device_info or {},
            "screen_hash": screenshot_hash,
            "hint": "no_xml" if not (uist.get("elements") or []) else "",
            "model_version": cfg.model_version,
        }
        headers: Dict[str, str] = {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        try:
            import httpx  # local import to keep base path light

            with httpx.Client(timeout=float(cfg.timeout_s)) as client:
                resp = client.post(cfg.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.debug("External semantic inference failed (continuing)", exc_info=True)
            return

        detections = BaseUI._external_semantic_parse(data)
        BaseUI._semantic_cache[cache_key] = detections
        BaseUI._semantic_calls_made += 1
        BaseUI._semantic_last_call_ts = now

        BaseUI._merge_external_semantic(
            uist,
            detections,
            screen_w=screen_w,
            screen_h=screen_h,
            cfg=cfg,
            provider_tag=f"external_semantic:{cfg.model_version}",
            pixel_ratio=float((device_info or {}).get("pixelRatio", 1.0) or 1.0),
        )

    # ---------------------------
    # Root salvage + dedupe + id assignment
    # ---------------------------

    @staticmethod
    def _unwrap_hierarchy_roots(uist: Dict[str, Any]) -> None:
        """
        Many android dumps have one placeholder root that contains the real nodes.
        Example: elements=[{class="hierarchy", subviews=[...]}]
        We unwrap it to avoid losing children during dedupe.
        """
        roots = uist.get("elements", []) or []
        if len(roots) == 1:
            r = roots[0]
            cls = (r.get("class") or "").lower()
            subs = r.get("subviews", []) or []
            if not subs:
                return

            # Always unwrap explicit placeholder root.
            if cls == "hierarchy":
                uist["elements"] = subs
                return

            # Be conservative: FrameLayout/View can be real content roots.
            # Only unwrap when it looks like a pure wrapper around a single child.
            if cls in ("android.widget.framelayout", "android.view.view"):
                if (r.get("text") or r.get("content_desc") or r.get("resource_id")):
                    return
                if len(subs) != 1:
                    return
                rf = BaseUI.get_frame(r)
                cf = BaseUI.get_frame(subs[0])
                rw, rh = int(rf.get("width", 0)), int(rf.get("height", 0))
                cw, ch = int(cf.get("width", 0)), int(cf.get("height", 0))
                if rw <= 1 or rh <= 1:
                    uist["elements"] = subs
                    return
                r_area = float(max(1, rw * rh))
                c_area = float(max(1, cw * ch))
                if (c_area / r_area) >= 0.85:
                    uist["elements"] = subs
                    return

    @staticmethod
    def _dedupe_and_sort_roots(roots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Conservative dedupe:
        - remove exact duplicates by (class, resource_id, frame, text/content_desc)
        - keep order by y then x
        """
        seen = set()
        out = []
        for n in roots:
            f = BaseUI.get_frame(n)
            key = (
                str(n.get("class") or ""),
                str(n.get("resource_id") or ""),
                str(n.get("text") or ""),
                str(n.get("content_desc") or ""),
                int(f["x"]),
                int(f["y"]),
                int(f["width"]),
                int(f["height"]),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(n)

        out.sort(key=lambda x: (BaseUI.get_frame(x)["y"], BaseUI.get_frame(x)["x"]))
        return out

    @staticmethod
    def _assign_ids(uist: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Assign stable incremental ids for interactable or useful nodes.
        Rules:
        - clickable nodes ALWAYS get id
        - nodes with text/content_desc/ocr/icon label get id if nearby clickables are scarce
        - id is stored as node["id"]
        Deterministic ids are critical so LLM proposals click the intended bounds within a snapshot.
        Returns vid_map: id -> node
        """
        vid_map: Dict[int, Dict[str, Any]] = {}
        next_id = 1

        # Determine whether we should also label non-clickable text nodes
        clickables = 0
        for n in BaseUI.iter_nodes(uist):
            if n.get("clickable"):
                clickables += 1

        for n in BaseUI.iter_nodes(uist):
            clickable = bool(n.get("clickable"))
            label = (
                n.get("text")
                or n.get("content_desc")
                or n.get("semantic_label")
                or n.get("ocr_text")
                or n.get("icon_label")
                or ""
            ).strip()

            should_id = clickable
            if not should_id and clickables <= 3 and label:
                # extremely sparse clickables: keep text anchors too
                should_id = True

            if should_id:
                n["id"] = next_id
                vid_map[next_id] = n
                next_id += 1

        logger.debug("Assigned %d ids", len(vid_map))
        return vid_map

    # ---------------------------
    # Public API
    # ---------------------------

    @staticmethod
    @time_consumed
    def post_process_ui(
        uist: Dict[str, Any],
        screenshot_b64: str,
        device_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """
        Main post-process:
        - unwrap placeholder roots
        - dedupe/sort roots safely (salvage children if roots become empty)
        - conditional OCR + icon labels
        - assign ids

        Returns:
          (new_uist, vid_map)
        This is the single entry point the workflow relies on for stable ids and enriched labels per snapshot.
        WHEN CALLED: every snapshot inside WorkflowRunner._capture_and_process before computing state_sig.
        POSITION IN FLOW: raw Appium XML -> parse_xml_to_uist -> post_process_ui -> compute_state_signature -> LLM digests.
        """
        if not uist:
            return {"elements": [], "screenscale": 1.0}, {}

        roots = uist.get("elements", []) or []
        if logger.isEnabledFor(logging.DEBUG):
            node_ct = sum(1 for _ in BaseUI.iter_nodes(uist))
            logger.debug("Post-process start: %d root nodes; %d nodes", len(roots), node_ct)

        # unwrap placeholders
        BaseUI._unwrap_hierarchy_roots(uist)
        roots = uist.get("elements", []) or []

        # dedupe roots
        deduped = BaseUI._dedupe_and_sort_roots(roots)

        # SALVAGE: never lose everything
        if not deduped:
            # if original roots had children, use them
            salvage: List[Dict[str, Any]] = []
            for r in roots:
                salvage.extend(r.get("subviews", []) or [])
            if salvage:
                deduped = BaseUI._dedupe_and_sort_roots(salvage)
                logger.debug("Salvaged %d roots from children", len(deduped))
            else:
                # last resort: keep original roots
                deduped = roots

        uist["elements"] = deduped
        if logger.isEnabledFor(logging.DEBUG):
            node_ct2 = sum(1 for _ in BaseUI.iter_nodes(uist))
            logger.debug("After dedupe/sort: %d root nodes, %d nodes", len(uist["elements"]), node_ct2)

        # OCR (conditional)
        try:
            if screenshot_b64 and BaseUI._needs_ocr(uist):
                ocr_items = BaseUI._run_ocr_cached(screenshot_b64, force=False)
                BaseUI._attach_ocr(uist, ocr_items)
        except Exception:
            logger.debug("OCR attach failed (continuing)", exc_info=True)

        # Optional icon classification (CNN)
        try:
            if screenshot_b64:
                BaseUI._attach_icon_labels(uist, screenshot_b64)
        except Exception:
            logger.debug("Icon classification failed (continuing)", exc_info=True)

        # Optional external semantic detection (API)
        try:
            if screenshot_b64:
                BaseUI._attach_external_semantic(uist, screenshot_b64, device_info=device_info)
        except Exception:
            logger.debug("External semantic attach failed (continuing)", exc_info=True)

        # Assign ids
        vid_map = BaseUI._assign_ids(uist)
        return uist, vid_map

    @staticmethod
    @time_consumed
    def post_process_ui_uied_first(
        uist: Dict[str, Any],
        screenshot_b64: str,
        device_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """
        Experimental UIED-first path for debugging:
        - keep XML-derived uist as the backbone
        - attach UIED OCR text onto overlapping existing nodes
        - add UIED merge components as synthetic nodes when XML lacks them
        - then run the regular OCR/icon/external-semantic enrichers as a second pass
        """
        # 这是专门给调试阶段准备的“UIED-first”实验入口。
        #
        # 和原 post_process_ui() 的区别：
        # - 原逻辑：优先走框架自带 OCR
        # - 这里：直接改成走 UIED OCR + UIED MERGE
        #
        # 但它依然保留原框架的几件关键事：
        # - 根节点 unwrap / dedupe
        # - icon label attach
        # - external semantic attach
        # - 最终统一 assign ids
        if not uist:
            return {"elements": [], "screenscale": 1.0}, {}

        roots = uist.get("elements", []) or []
        if logger.isEnabledFor(logging.DEBUG):
            node_ct = sum(1 for _ in BaseUI.iter_nodes(uist))
            logger.debug("UIED-first post-process start: %d root nodes; %d nodes", len(roots), node_ct)

        BaseUI._unwrap_hierarchy_roots(uist)
        roots = uist.get("elements", []) or []
        deduped = BaseUI._dedupe_and_sort_roots(roots)
        if not deduped:
            salvage: List[Dict[str, Any]] = []
            for r in roots:
                salvage.extend(r.get("subviews", []) or [])
            deduped = BaseUI._dedupe_and_sort_roots(salvage) if salvage else roots
        uist["elements"] = deduped

        try:
            if screenshot_b64:
                # 1. 跑 UIED，拿到 OCR / IP / MERGE 原始结果
                uied_result = BaseUI._run_uied_cached(screenshot_b64)
                # 2. 先把 OCR 文本并入 merge 节点，而不是先回填 XML
                uied_result = BaseUI._enrich_uied_merge_with_ocr(uied_result)
                # 3. 再把已经“自带 OCR 文本”的 MERGE 节点补成 synthetic node，插回树里
                BaseUI._merge_uied_into_uist(uist, uied_result)
                # 4. 对 UIED merge 吞掉的短按钮 OCR 文本，补成可点击节点。
                BaseUI._add_uied_ocr_action_nodes(uist, uied_result)
        except Exception:
            logger.exception(
                "UIED attach failed (continuing): screenshot_key=%s",
                BaseUI._hash_screenshot_full(screenshot_b64),
            )

        try:
            if screenshot_b64 and BaseUI._needs_ocr(uist):
                # UIED 先补结构/文本；若页面仍然稀疏，再用常规 OCR 做第二次兜底补全。
                ocr_items = BaseUI._run_ocr_cached(screenshot_b64, force=False)
                BaseUI._attach_ocr(uist, ocr_items)
        except Exception:
            logger.debug("Fallback OCR attach failed after UIED (continuing)", exc_info=True)

        try:
            if screenshot_b64:
                # 保留原有的图标语义增强逻辑。
                BaseUI._attach_icon_labels(uist, screenshot_b64)
        except Exception:
            logger.debug("Icon classification failed (continuing)", exc_info=True)

        try:
            if screenshot_b64:
                # 保留原有的外部语义检测逻辑。
                BaseUI._attach_external_semantic(uist, screenshot_b64, device_info=device_info)
        except Exception:
            logger.debug("External semantic attach failed (continuing)", exc_info=True)

        # 最后统一给树里的可用节点分配 id，生成 vid_map。
        # 注意：当前第一版 synthetic UIED 节点默认 clickable=False，
        # 所以很多节点会进入 uist，但不一定进入 vid_map。
        vid_map = BaseUI._assign_ids(uist)
        return uist, vid_map

    @staticmethod
    def get_text_list(uist: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for n in BaseUI.iter_nodes(uist):
            for k in ("text", "content_desc", "semantic_label", "semantic_desc", "ocr_text", "icon_label"):
                v = n.get(k)
                if v and str(v).strip():
                    out.append(str(v).strip())
        return out

    @staticmethod
    def to_digest(uist: Dict[str, Any], limit: int = 180) -> Dict[str, Any]:
        """
        Compact digest for LLM.
        """
        elements = []
        for n in BaseUI.iter_nodes(uist):
            if len(elements) >= limit:
                break
            f = BaseUI.get_frame(n)
            label = n.get("text") or n.get("content_desc") or n.get("semantic_label") or n.get("ocr_text") or n.get("icon_label")
            elements.append(
                {
                    "id": n.get("id"),
                    "class": (n.get("class") or "")[:60],
                    "label": (str(label)[:90] if label else None),
                    "text": (str(n.get("text"))[:90] if n.get("text") else None),
                    "content_desc": (str(n.get("content_desc"))[:90] if n.get("content_desc") else None),
                    "resource_id": (str(n.get("resource_id"))[:90] if n.get("resource_id") else None),
                    "semantic_label": (str(n.get("semantic_label"))[:90] if n.get("semantic_label") else None),
                    "semantic_type": (str(n.get("semantic_type"))[:40] if n.get("semantic_type") else None),
                    "ocr_text": (str(n.get("ocr_text"))[:90] if n.get("ocr_text") else None),
                    "icon_label": (str(n.get("icon_label"))[:60] if n.get("icon_label") else None),
                    "clickable": bool(n.get("clickable")),
                    "enabled": bool(n.get("enabled", True)),
                    "bounds": [f["x"], f["y"], f["width"], f["height"]],
                }
            )
        return {"elements": elements, "screenscale": uist.get("screenscale", 1.0)}
    
    @staticmethod
    def compute_state_signature(uist: Dict[str, Any]) -> str:
        """
        Stable signature for a screen state.
        Robust against tiny text changes and dynamic counters by hashing:
        - class + resource_id suffix + clickable + quantized bounds
        """
        items: List[Tuple] = []

        def q(v: int) -> int:
            return int(v // 10)

        def walk(n: Dict[str, Any]):
            f = BaseUI.get_frame(n)
            items.append(
                (
                    str(n.get("class") or ""),
                    str(n.get("resource_id") or "")[-40:],
                    bool(n.get("clickable")),
                    q(int(f["x"])),
                    q(int(f["y"])),
                    q(int(f["width"])),
                    q(int(f["height"])),
                )
            )
            for ch in n.get("subviews", []) or []:
                walk(ch)

        for r in uist.get("elements", []) or []:
            walk(r)

        blob = json.dumps(items[:2000], ensure_ascii=False)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()
