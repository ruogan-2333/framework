"""
questionnaire_state2.py

Small execution-state helper for the new UI-level router/block workflow.

This file intentionally keeps only three public runtime structures:
1. `routers`
   A flat list of router questions loaded from `questionnaire_routers.json`.
2. `blocks`
   A flat list of blocks loaded from `questionnaire_blocks.json`.
   Each block contains its own `questions` dict, so the block still preserves
   the implicit question tree through each question's internal `show_if`.
3. `block_status`
   Runtime counters keyed by block id.

The loader reads the questionnaire directory passed by the caller directly. If
the selected main questionnaire has a sibling `addition` directory, the loader
also merges that supplemental UI questionnaire for the same run.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class QuestionnaireState:
    """
    Minimal state container for the UI-level split questionnaire.

    Public fields after loading:
    - routers:
      List[dict]. Each item is one router question:
      {
        "id": "sexuality_includes",
        "full_id": "sexuality.sexuality_includes",
        "module": "sexuality",
        "category": "multiple_class",
        "question": "...",
        "type": "multiple",
        "options": [...],
        "show_if": ...
      }
    - blocks:
      List[dict]. Each item is one fillable block:
      {
        "id": "sexuality.sexuality_includes.suggestive...",
        "module": "sexuality",
        "topic": "",
        "block_show_if": [{"id": "sexuality_includes", "option_id": "..."}],
        "questions": {"question_id": {...UI-level question payload...}}
      }
    - block_status:
      Dict[block_id, dict]. Runtime counters and block metadata:
      {
        "<block_id>": {
          "id": "...",
          "module": "...",
          "topic": "",
          "visit_count": 0,
          "hit_count": 0
        }
      }
    """

    def __init__(self) -> None:
        self.routers: List[Dict[str, Any]] = []
        self.blocks: List[Dict[str, Any]] = []
        self.block_status: Dict[str, Dict[str, Any]] = {}
        self.loaded_dirs: List[str] = []

    @staticmethod
    def load_from_questionnaire_dir(questionnaire_dir: str) -> "QuestionnaireState":
        """
        Load router/block files from the exact questionnaire directory passed in.

        Input:
        - questionnaire_dir:
          Directory containing questionnaire_routers.json/questionnaire_blocks.json.
          If a sibling `addition` directory exists, it is merged automatically
          unless `questionnaire_dir` itself is `addition`.

        Processing:
        - Read the provided directory directly.
        - Optionally read sibling `addition`.
        - Check duplicate router/block ids before initializing runtime status.

        Output:
        - QuestionnaireState
          A loaded instance with `routers`, `blocks`, and `block_status`.
        """
        qdir = Path(questionnaire_dir).resolve()
        inst = QuestionnaireState()
        # Old behavior kept for reference only. It guessed a separate generated
        # split path from the collection name, which made `--questionnaire-dir`
        # not behave like an actual directory input:
        # collection_name = qdir.name
        # project_root = Path(__file__).resolve().parent
        # split_dir = project_root / "mytest2" / "questionnaire_handler" / "chain_debug" / f"{collection_name}_split"
        # inst.load_routers_and_blocks(str(split_dir))

        dirs = [qdir]
        addition_dir = qdir.parent / "addition"
        if qdir.name != "addition" and inst._has_questionnaire_files(addition_dir):
            dirs.append(addition_dir)
        inst.load_routers_and_blocks_from_dirs([str(path) for path in dirs])
        return inst

    def load_routers_and_blocks(self, split_dir: str) -> None:
        """
        Read generated router/block JSON files.

        Input:
        - split_dir:
          Directory containing:
          - questionnaire_routers.json
          - questionnaire_blocks.json

        Processing:
        - Read `{"routers": [...]}` into `self.routers`.
        - Read `{"blocks": [...]}` into `self.blocks`.
        - Initialize `self.block_status` with zero counters for every block.

        Output:
        - None. The instance fields are updated in-place.
        """
        root = Path(split_dir).resolve()
        routers_path = root / "questionnaire_routers.json"
        blocks_path = root / "questionnaire_blocks.json"

        self.routers = self._read_routers_json(routers_path) if routers_path.exists() else []
        self.blocks = self._read_blocks_json(blocks_path) if blocks_path.exists() else []
        self._init_block_status()

    def load_routers_and_blocks_from_dirs(self, questionnaire_dirs: List[str]) -> None:
        """
        Read and merge router/block JSON files from one or more questionnaire directories.

        Input:
        - questionnaire_dirs:
          Ordered directories. The first is the selected main questionnaire;
          later directories such as `addition` are supplemental.

        Processing:
        - Read questionnaire_routers.json/questionnaire_blocks.json directly
          from each directory.
        - Attach `source_collection` and `source_dir` metadata to every loaded
          router/block for later debugging.
        - Reject duplicate router ids/full_ids and duplicate block ids.
        - Initialize block_status from the merged block list.

        Output:
        - None. The instance fields are replaced in-place.
        """
        self.routers = []
        self.blocks = []
        self.loaded_dirs = []

        for raw_dir in questionnaire_dirs:
            root = Path(raw_dir).resolve()
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"questionnaire directory not found: {root}")
            routers_path = root / "questionnaire_routers.json"
            blocks_path = root / "questionnaire_blocks.json"
            if not routers_path.exists() and not blocks_path.exists():
                raise FileNotFoundError(f"questionnaire directory has no router/block JSON files: {root}")

            source_collection = root.name
            self.loaded_dirs.append(str(root))

            for router in (self._read_routers_json(routers_path) if routers_path.exists() else []):
                item = dict(router)
                item.setdefault("source_collection", source_collection)
                item.setdefault("source_dir", str(root))
                self.routers.append(item)

            for block in (self._read_blocks_json(blocks_path) if blocks_path.exists() else []):
                item = dict(block)
                item.setdefault("source_collection", source_collection)
                item.setdefault("source_dir", str(root))
                self.blocks.append(item)

        self._validate_unique_ids()
        self._init_block_status()

    def match_blocks_from_router_answers(self, router_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return all blocks whose `block_show_if` conditions are satisfied.

        Input:
        - router_answers:
          List of LLM2-1 router answers. Each item should look like:
          {
            "question_id": "sexuality.sexuality_includes",
            "new_answer": ["nudity_or_revealing_outfits"]
          }
          `question_id` may also be local, e.g. `sexuality_includes`.

        Processing:
        - Build an answer lookup using both full ids and local ids.
        - A block with empty `block_show_if` is not matched. All blocks must be
          selected by explicit router evidence.
        - A block with conditions is matched only when every condition's
          option_id appears in the corresponding router answer.

        Output:
        - List[dict]
          Full block payloads from `self.blocks`.
        """
        answer_lookup = self._normalize_router_answers(router_answers)
        matched: List[Dict[str, Any]] = []

        for block in self.blocks:
            conditions = list(block.get("block_show_if") or [])
            if not conditions:
                continue
            if all(self._condition_is_satisfied(cond, answer_lookup) for cond in conditions):
                matched.append(block)
        return matched

    def mark_blocks_hit(self, matched_blocks: List[Dict[str, Any]]) -> None:
        """
        Increment hit counters for blocks selected by router answers.

        Input:
        - matched_blocks:
          Usually the output of `match_blocks_from_router_answers(...)`.

        Processing:
        - For each block, read `block["id"]`.
        - Increase `block_status[id]["hit_count"]`.

        Output:
        - None. `block_status` is updated in-place.
        """
        for block in matched_blocks:
            block_id = str(block.get("id") or "").strip()
            if block_id in self.block_status:
                self.block_status[block_id]["hit_count"] += 1

    def mark_blocks_visited(self, block_ids: List[str]) -> None:
        """
        Increment visit counters for blocks actually sent to LLM2-2.

        Input:
        - block_ids:
          List of block ids, usually pulled from matched blocks.

        Processing:
        - Increase `visit_count` for each known block id.

        Output:
        - None. `block_status` is updated in-place.
        """
        for raw_id in block_ids:
            block_id = str(raw_id or "").strip()
            if block_id in self.block_status:
                self.block_status[block_id]["visit_count"] += 1

    def get_block_payload(self, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch one complete block payload by id.

        Input:
        - block_id:
          The block id from `questionnaire_blocks.json`.

        Processing:
        - Scan `self.blocks` and return the block with matching `id`.

        Output:
        - dict if found, otherwise None.
        """
        target = str(block_id or "").strip()
        for block in self.blocks:
            if str(block.get("id") or "").strip() == target:
                return block
        return None

    def save_observation(
        self,
        out_dir: str,
        state_sig: str,
        router_answers: List[Dict[str, Any]],
        matched_blocks: List[Dict[str, Any]],
        block_fill_results: List[Dict[str, Any]],
        screenshot_path: str = "",
    ) -> Path:
        """
        Persist one UI observation for later inspection/merge.

        Input:
        - out_dir:
          Directory where the observation JSON will be saved.
        - state_sig:
          Current UI state signature.
        - router_answers:
          Raw LLM2-1 answers.
        - matched_blocks:
          Blocks selected locally from router answers.
        - block_fill_results:
          LLM2-2 results for visited blocks. This can be empty during state
          debugging.
        - screenshot_path:
          Optional path to the screenshot used for this observation.

        Processing:
        - Store compact ids plus raw answers/results.
        - File name includes timestamp and state_sig for easy browsing.

        Output:
        - Path to the written JSON file.
        """
        root = Path(out_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        # Windows treats ":" as an alternate-data-stream separator, so replace
        # every filename-unsafe character, not only slashes.
        safe_sig = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(state_sig or "unknown")).strip("_") or "unknown"
        out_path = root / f"{stamp}_{safe_sig}.json"

        payload = {
            "state_sig": state_sig,
            "screenshot_path": screenshot_path,
            "router_answers": router_answers,
            "matched_block_ids": [block.get("id") for block in matched_blocks],
            "block_fill_results": block_fill_results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def _read_routers_json(self, path: Path) -> List[Dict[str, Any]]:
        """
        Read `questionnaire_routers.json`.

        Input:
        - path: JSON file with shape `{"routers": [...]}`.

        Output:
        - List[dict]. The router objects are kept as generated.
        """
        data = self._read_json(path)
        return list(data.get("routers") or [])

    def _read_blocks_json(self, path: Path) -> List[Dict[str, Any]]:
        """
        Read `questionnaire_blocks.json`.

        Input:
        - path: JSON file with shape `{"blocks": [...]}`.

        Output:
        - List[dict]. The block objects are kept as generated.
        """
        data = self._read_json(path)
        return list(data.get("blocks") or [])

    def _init_block_status(self) -> None:
        """
        Initialize runtime counters for all loaded blocks.

        Input:
        - self.blocks:
          The generated block catalog.

        Processing:
        - Use each block's `id` as the key.
        - Copy `topic` so later topic-generation can populate it without
          changing the counter schema.

        Output:
        - None. `self.block_status` becomes:
          {block_id: {"id": ..., "module": ..., "topic": "", "visit_count": 0, "hit_count": 0}}
        """
        self.block_status = {}
        for block in self.blocks:
            block_id = str(block.get("id") or "").strip()
            if not block_id:
                continue
            self.block_status[block_id] = {
                "id": block_id,
                "module": str(block.get("module") or ""),
                "topic": str(block.get("topic") or ""),
                "source_collection": str(block.get("source_collection") or ""),
                "source_dir": str(block.get("source_dir") or ""),
                "visit_count": 0,
                "hit_count": 0,
            }

    def _has_questionnaire_files(self, questionnaire_dir: Path) -> bool:
        """
        Check whether a directory looks like a router/block questionnaire folder.

        Input:
        - questionnaire_dir: candidate directory path.

        Output:
        - True when the directory exists and contains at least one expected JSON file.
        """
        root = Path(questionnaire_dir)
        if not root.exists() or not root.is_dir():
            return False
        return (root / "questionnaire_routers.json").exists() or (root / "questionnaire_blocks.json").exists()

    def _validate_unique_ids(self) -> None:
        """
        Validate merged router/block identifiers.

        Input:
        - self.routers and self.blocks after loading one or more directories.

        Output:
        - None on success.
        - Raises ValueError when duplicate router id/full_id or block id exists.
        """
        seen_router_ids: Dict[str, str] = {}
        for router in self.routers:
            source = str(router.get("source_collection") or "")
            for key in ("id", "full_id"):
                rid = str(router.get(key) or "").strip()
                if not rid:
                    continue
                prev = seen_router_ids.get(rid)
                if prev is not None:
                    raise ValueError(f"duplicate router {key}={rid!r}: {prev} vs {source}")
                seen_router_ids[rid] = source

        seen_block_ids: Dict[str, str] = {}
        for block in self.blocks:
            block_id = str(block.get("id") or "").strip()
            if not block_id:
                continue
            source = str(block.get("source_collection") or "")
            prev = seen_block_ids.get(block_id)
            if prev is not None:
                raise ValueError(f"duplicate block id={block_id!r}: {prev} vs {source}")
            seen_block_ids[block_id] = source

    def _normalize_router_answers(self, router_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a router-answer lookup using both local and full ids.

        Input:
        - router_answers:
          Items from `RouterResult.router_updates`.

        Processing:
        - If answer id is `module.question`, also add `question`.
        - If answer id is local, add local id and, when possible, the matching
          full id from `self.routers`.

        Output:
        - Dict[str, Any] where keys may include both `id` and `full_id`.
        """
        lookup: Dict[str, Any] = {}
        local_to_full = {
            str(router.get("id") or ""): str(router.get("full_id") or "")
            for router in self.routers
            if router.get("id") and router.get("full_id")
        }

        for item in router_answers:
            raw_qid = str(item.get("question_id") or "").strip()
            if not raw_qid:
                continue
            answer = item.get("new_answer")
            lookup[raw_qid] = answer

            if "." in raw_qid:
                local = raw_qid.rsplit(".", 1)[-1]
                lookup[local] = answer
            elif raw_qid in local_to_full:
                lookup[local_to_full[raw_qid]] = answer
        return lookup

    def _condition_is_satisfied(self, condition: Dict[str, Any], answer_lookup: Dict[str, Any]) -> bool:
        """
        Check one `block_show_if` condition against normalized answers.

        Input:
        - condition:
          {"id": "sexuality_includes", "option_id": "nudity_or_revealing_outfits"}
        - answer_lookup:
          Output of `_normalize_router_answers(...)`.

        Output:
        - bool. True when the router answer includes the required option.
        """
        question_id = str(condition.get("id") or "").strip()
        option_id = str(condition.get("option_id") or "").strip()
        if not question_id or not option_id or question_id not in answer_lookup:
            return False

        answer = answer_lookup.get(question_id)
        values = answer if isinstance(answer, list) else ([] if answer is None else [answer])
        return option_id in {str(value) for value in values}

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)


__all__ = ["QuestionnaireState"]
