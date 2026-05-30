"""
summarize_batch_runs.py

基于已有 APP 批量运行记录生成中文统计表。

输入：
  - 原始批次目录，例如 test_debug/batch_runs/20260518_082912_sample47_seed42。
  - 可选 rerun 批次目录，例如 test_debug/batch_runs/rerun_failed_10_20260518_153504。
  - 可选 app.log，用于给没有单独日志的旧批次恢复 LLM token 用量。

输出：
  - app_summary.csv/json/md：每个 APP 的运行时间、UI 数、动作数、token、估算费用等。
  - token_by_op.csv：按 APP 和 LLM 操作拆分的 token 用量。
  - router_hit_summary.csv：按 router question 汇总命中次数。
  - block_hit_summary.csv：按 block id 汇总命中次数。
  - 字段说明.md：中文说明每个字段来自哪里、怎么计算。

功能：
  - 将原始 47 个 APP 批次和后续 rerun 批次合并。
  - 同 package 有 rerun 时，用 rerun 结果覆盖原始结果。
  - 从 trace/observations 和 trace.jsonl 中统计完成 router 分析的 UI 数。
  - 从 per-app log 或全局 app.log 中统计 token，并同时换算 gpt-4o / gemini-2.5-flash 成本。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = PROJECT_ROOT / "test_debug" / "batch_runs"
DEFAULT_BASE_BATCH_DIR = DEFAULT_BATCH_ROOT / "20260518_082912_sample47_seed42"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_debug" / "analysis_outputs"
DEFAULT_APP_LOG = PROJECT_ROOT / "app.log"

GPT4O_INPUT_PER_M = 2.2
GPT4O_OUTPUT_PER_M = 8.8
GEMINI_FLASH_INPUT_PER_M = 0.27
GEMINI_FLASH_OUTPUT_PER_M = 2.25


@dataclass
class TokenUsage:
    """
    输入：prompt/completion token 计数。
    输出：可累加的 token 统计对象。
    功能：统一保存总量和按 op 拆分的 token 用量。
    """

    calls: int = 0
    prompt: int = 0
    completion: int = 0
    by_op: Dict[str, Dict[str, int]] | None = None

    def __post_init__(self) -> None:
        """
        输入：dataclass 初始化后的字段。
        输出：确保 by_op 是可写字典。
        功能：避免默认可变对象问题。
        """
        if self.by_op is None:
            self.by_op = {}

    @property
    def total(self) -> int:
        """
        输入：当前 prompt/completion 计数。
        输出：总 token 数。
        功能：提供统一的 total 计算字段。
        """
        return int(self.prompt) + int(self.completion)

    def add(self, op: str, prompt: int, completion: int, calls: int = 1) -> None:
        """
        输入：一次或多次 LLM 调用的 op、输入 token、输出 token、调用次数。
        输出：无返回值，原地累加。
        功能：同时更新总量和 by_op 拆分。
        """
        op_key = str(op or "unknown")
        self.calls += int(calls)
        self.prompt += int(prompt)
        self.completion += int(completion)
        row = self.by_op.setdefault(op_key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        row["calls"] += int(calls)
        row["prompt_tokens"] += int(prompt)
        row["completion_tokens"] += int(completion)
        row["total_tokens"] += int(prompt) + int(completion)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """
    输入：命令行参数。
    输出：argparse.Namespace。
    功能：定义批次统计脚本的输入目录、输出目录和价格参数。
    """

    parser = argparse.ArgumentParser(description="汇总已有 APP 批量运行结果，输出中文统计表。")
    parser.add_argument("--base-batch-dir", default=str(DEFAULT_BASE_BATCH_DIR), help="原始 47 APP 批次目录。")
    parser.add_argument("--rerun-batch-dir", default="auto", help="rerun 批次目录；auto 表示自动选择最新 rerun_failed_10_*；空字符串表示不合并 rerun。")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="分析结果输出根目录。")
    parser.add_argument("--app-log", default=str(DEFAULT_APP_LOG), help="全局 app.log 路径，用于旧批次 token 恢复。")
    parser.add_argument("--questionnaire-root", default=str(PROJECT_ROOT / "questionnaire-UI"), help="问卷根目录，用于 block topic 反查。")
    parser.add_argument("--gpt4o-input-per-m", type=float, default=GPT4O_INPUT_PER_M, help="gpt-4o 输入 token 每百万价格。")
    parser.add_argument("--gpt4o-output-per-m", type=float, default=GPT4O_OUTPUT_PER_M, help="gpt-4o 输出 token 每百万价格。")
    parser.add_argument("--gemini-input-per-m", type=float, default=GEMINI_FLASH_INPUT_PER_M, help="gemini-2.5-flash 输入 token 每百万价格。")
    parser.add_argument("--gemini-output-per-m", type=float, default=GEMINI_FLASH_OUTPUT_PER_M, help="gemini-2.5-flash 输出 token 每百万价格。")
    parser.add_argument("--run-label", default="batch_47_summary", help="输出目录名前缀。")
    return parser.parse_args(argv)


def read_json(path: Path, default: Any) -> Any:
    """
    输入：JSON 文件路径和默认值。
    输出：解析后的 JSON；失败时返回默认值。
    功能：让统计脚本可以容忍部分 trace 缺文件或损坏。
    """
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    """
    输入：目标路径和任意可 JSON 序列化对象。
    输出：写入 UTF-8 JSON 文件。
    功能：统一 JSON 输出格式。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_text(value: Any) -> str:
    """
    输入：任意值。
    输出：适合 CSV/Markdown 的字符串。
    功能：避免 None 和竖线破坏 Markdown 表格。
    """
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def safe_token(value: Any, default: str = "item") -> str:
    """
    输入：任意文本。
    输出：Windows 文件名安全 token。
    功能：生成输出目录名或文件名时规避非法字符。
    """
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or default


def parse_run_datetime(run_id: str) -> Optional[datetime]:
    """
    输入：run_id，例如 20260518_082912_com.xxx。
    输出：run_id 前缀对应的 datetime；无法解析返回 None。
    功能：为旧批次从全局 app.log 按时间窗口恢复 token。
    """
    m = re.match(r"^(\d{8}_\d{6})", str(run_id or ""))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def find_latest_rerun_batch(base_batch_dir: Path) -> Optional[Path]:
    """
    输入：原始 batch 目录。
    输出：同级目录中最新 rerun_failed_10_* 目录；没有则返回 None。
    功能：让脚本默认使用最近一次后 10 APP rerun 结果。
    """
    root = base_batch_dir.parent
    candidates = [p for p in root.glob("rerun_failed_10_*") if p.is_dir() and (p / "rerun_summary.json").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_base_rows(base_batch_dir: Path) -> List[Dict[str, Any]]:
    """
    输入：原始批次目录。
    输出：batch_summary.json 中的 APP 行列表。
    功能：读取 47 APP 原始运行索引和基础元数据。
    """
    summary_path = base_batch_dir / "batch_summary.json"
    rows = read_json(summary_path, [])
    if not isinstance(rows, list):
        raise SystemExit(f"batch_summary.json 格式不是列表: {summary_path}")
    return [dict(row) for row in rows]


def load_rerun_rows(rerun_batch_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """
    输入：rerun 批次目录，可为空。
    输出：rerun_summary.json 中的 APP 行列表。
    功能：读取后 10 APP 的补跑结果。
    """
    if not rerun_batch_dir:
        return []
    summary_path = rerun_batch_dir / "rerun_summary.json"
    rows = read_json(summary_path, [])
    if not isinstance(rows, list):
        raise SystemExit(f"rerun_summary.json 格式不是列表: {summary_path}")
    return [dict(row) for row in rows]


def merge_base_and_rerun(base_rows: Sequence[Dict[str, Any]], rerun_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    输入：原始批次行和 rerun 批次行。
    输出：按原始 47 APP 顺序合并后的行。
    功能：同 package 有 rerun 时，用 rerun 的 run_id/trace/log/结果覆盖原始行，同时保留原始索引。
    """
    rerun_by_package = {str(row.get("package") or ""): dict(row) for row in rerun_rows if row.get("package")}
    merged: List[Dict[str, Any]] = []
    for base in base_rows:
        package = str(base.get("package") or "")
        row = dict(base)
        row["original_run_id"] = str(base.get("run_id") or "")
        row["record_source"] = "base"
        if package in rerun_by_package:
            rerun = rerun_by_package[package]
            preserved = {
                "index": base.get("index"),
                "package": package,
                "app_name": base.get("app_name") or rerun.get("app_name"),
                "category": base.get("category") or rerun.get("category"),
                "questionnaire_type": base.get("questionnaire_type") or rerun.get("questionnaire_type"),
                "original_run_id": base.get("run_id"),
                "record_source": "rerun",
            }
            row.update(rerun)
            row.update(preserved)
        merged.append(row)
    return merged


def parse_tokens_from_text(text: str) -> TokenUsage:
    """
    输入：单个 APP 日志文本。
    输出：TokenUsage。
    功能：优先解析 [TOKENS][op] 汇总；没有汇总时回退解析逐条 LLM_USAGE。
    """
    usage = TokenUsage()
    by_op_seen = False
    op_re = re.compile(r"\[TOKENS\]\[(?P<op>[^\]]+)\]\s+calls=(?P<calls>\d+)\s+prompt=(?P<prompt>\d+)\s+completion=(?P<completion>\d+)\s+total=(?P<total>\d+)")
    for match in op_re.finditer(text):
        by_op_seen = True
        usage.add(match.group("op"), int(match.group("prompt")), int(match.group("completion")), int(match.group("calls")))
    if by_op_seen:
        return usage

    llm_re = re.compile(r"LLM_USAGE\s+stage=\S+\s+op=(?P<op>\S+)\s+sig=\S+\s+prompt=(?P<prompt>\d+)\s+completion=(?P<completion>\d+)\s+total=(?P<total>\d+)")
    for match in llm_re.finditer(text):
        usage.add(match.group("op"), int(match.group("prompt")), int(match.group("completion")), 1)
    return usage


def parse_global_llm_usage(app_log: Path) -> List[Dict[str, Any]]:
    """
    输入：全局 app.log。
    输出：按时间排序的 LLM_USAGE 记录列表。
    功能：为没有 per-app log 的旧批次恢复 token 用量。
    """
    if not app_log.exists():
        return []
    out: List[Dict[str, Any]] = []
    line_re = re.compile(
        r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3})\].*?LLM_USAGE\s+stage=(?P<stage>\S+)\s+op=(?P<op>\S+)\s+sig=\S+\s+prompt=(?P<prompt>\d+)\s+completion=(?P<completion>\d+)\s+total=(?P<total>\d+)"
    )
    try:
        with app_log.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = line_re.search(line)
                if not m:
                    continue
                ts = datetime.strptime(m.group("ts") + "." + m.group("ms"), "%Y-%m-%d %H:%M:%S.%f")
                out.append(
                    {
                        "ts": ts,
                        "op": m.group("op"),
                        "prompt": int(m.group("prompt")),
                        "completion": int(m.group("completion")),
                    }
                )
    except Exception:
        return out
    return out


def token_usage_from_global_window(records: Sequence[Dict[str, Any]], start: Optional[datetime], end: Optional[datetime]) -> TokenUsage:
    """
    输入：全局 LLM_USAGE 记录、窗口开始、窗口结束。
    输出：该时间窗口内的 TokenUsage。
    功能：用 run_id 时间窗口近似恢复旧批次每个 APP 的 token。
    """
    usage = TokenUsage()
    if start is None:
        return usage
    if end is None:
        end = start + timedelta(minutes=10)
    for rec in records:
        ts = rec.get("ts")
        if isinstance(ts, datetime) and start <= ts < end:
            usage.add(str(rec.get("op") or "unknown"), int(rec.get("prompt") or 0), int(rec.get("completion") or 0), 1)
    return usage


def load_trace_analysis(trace_dir: Path) -> Dict[str, Any]:
    """
    输入：单个 APP trace 目录。
    输出：analysis/run_analysis_summary.json 的关键字段。
    功能：读取停止原因、UI 图节点数、边数、动作数等主流程统计。
    """
    summary = read_json(trace_dir / "graph" / "run_analysis_summary.json", {})
    if not summary:
        summary = read_json(trace_dir / "analysis" / "run_analysis_summary.json", {})
    if not isinstance(summary, dict):
        summary = {}
    return {
        "analysis_exists": bool(summary),
        "stop_reason": str(summary.get("stop_reason") or ""),
        "graph_node_count": summary.get("graph_node_count"),
        "graph_edge_count": summary.get("graph_edge_count"),
        "action_count": summary.get("action_count"),
        "unfinished_state_count": summary.get("unfinished_state_count"),
        "no_progress_loops": summary.get("no_progress_loops"),
        "no_new_state_count": summary.get("no_new_state_count"),
    }


def answer_is_nonempty(answer: Any) -> bool:
    """
    输入：router answer 的 new_answer 值。
    输出：是否可视为非空命中。
    功能：定义分析脚本中的 router 命中标准。
    """
    if answer is None:
        return False
    if isinstance(answer, list):
        return any(str(x).strip() for x in answer)
    if isinstance(answer, dict):
        return bool(answer)
    return bool(str(answer).strip())


def iter_observation_files(trace_dir: Path, stage: str) -> Iterable[Path]:
    """
    输入：trace 目录和 observation stage。
    输出：对应 observation JSON 文件迭代器。
    功能：统一访问 observations/router 与 observations/blocks_fill。
    """
    root = trace_dir / "observations" / stage
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def iter_state_llm_files(trace_dir: Path, filename: str) -> Iterable[Path]:
    """
    输入：trace 目录和 states/<UI>/llm 下的文件名。
    输出：匹配到的 LLM JSON 文件迭代器。
    功能：读取新结构里按 UI 聚合的大模型调试结果。
    """
    root = trace_dir / "states"
    if not root.exists():
        return []
    return sorted(root.glob(f"UI*/llm/{filename}"))


def count_router_states_from_trace(trace_dir: Path) -> set[str]:
    """
    输入：单个 APP trace 目录。
    输出：trace.jsonl 中完成 block_router LLM 结果的 state_sig 集合。
    功能：当 observations/router 不存在时，作为 router UI 数的回退来源。
    """
    trace_path = trace_dir / "trace.jsonl"
    states: set[str] = set()
    if not trace_path.exists():
        return states
    try:
        with trace_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                data = obj.get("data") or {}
                if obj.get("event") == "llm_result" and data.get("kind") in {"block_router", "navigation_router"} and not data.get("error"):
                    sig = str(data.get("state_sig") or "")
                    if sig:
                        states.add(sig)
    except Exception:
        return states
    return states


def summarize_router_block_observations(trace_dir: Path) -> Tuple[Dict[str, Any], Counter, Counter]:
    """
    输入：单个 APP trace 目录。
    输出：APP 级 router/block 统计，以及全局 router/block 计数器。
    功能：统计完成 router 分析的去重 UI 数、router 命中、block 命中和 block fill UI 数。
    """
    router_states = set()
    block_fill_states = set()
    router_obs_count = 0
    block_fill_obs_count = 0
    router_hit_counter: Counter = Counter()
    block_hit_counter: Counter = Counter()

    for path in iter_state_llm_files(trace_dir, "navigation_router_result.json"):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        router = result.get("router") if isinstance(result.get("router"), dict) else {}
        router_answers = list(router.get("router_updates") or [])
        router_obs_count += 1
        state_sig = str(payload.get("state_sig") or result.get("state_sig") or "")
        if state_sig:
            router_states.add(state_sig)
        for item in router_answers:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id") or item.get("id") or "").strip()
            if qid and answer_is_nonempty(item.get("new_answer")):
                router_hit_counter[qid] += 1
        for block_id in list(payload.get("matched_block_ids") or []):
            bid = str(block_id or "").strip()
            if bid:
                block_hit_counter[bid] += 1

    for path in iter_state_llm_files(trace_dir, "blocks_fill_result.json"):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        block_fill_obs_count += 1
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        state_sig = str(payload.get("state_sig") or result.get("state_sig") or "")
        if state_sig:
            block_fill_states.add(state_sig)

    for path in iter_observation_files(trace_dir, "router"):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        router_obs_count += 1
        state_sig = str(payload.get("state_sig") or "")
        if state_sig:
            router_states.add(state_sig)
        for item in list(payload.get("router_answers") or []):
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id") or item.get("id") or "").strip()
            if qid and answer_is_nonempty(item.get("new_answer")):
                router_hit_counter[qid] += 1
        for block_id in list(payload.get("matched_block_ids") or []):
            bid = str(block_id or "").strip()
            if bid:
                block_hit_counter[bid] += 1

    for path in iter_observation_files(trace_dir, "blocks_fill"):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        block_fill_obs_count += 1
        state_sig = str(payload.get("state_sig") or "")
        if state_sig:
            block_fill_states.add(state_sig)

    if not router_states:
        router_states.update(count_router_states_from_trace(trace_dir))

    summary = {
        "router_observation_count": router_obs_count,
        "router_analyzed_ui_count": len(router_states),
        "router_duplicate_count": max(0, router_obs_count - len(router_states)),
        "router_hit_question_count": len(router_hit_counter),
        "router_hit_total_count": sum(router_hit_counter.values()),
        "matched_block_unique_count": len(block_hit_counter),
        "matched_block_total_count": sum(block_hit_counter.values()),
        "block_fill_observation_count": block_fill_obs_count,
        "block_fill_ui_count": len(block_fill_states),
    }
    return summary, router_hit_counter, block_hit_counter


def calculate_cost(prompt_tokens: int, completion_tokens: int, input_price: float, output_price: float) -> float:
    """
    输入：prompt/completion token 和每百万 token 单价。
    输出：美元估算费用。
    功能：统一按百万 token 价格换算成本。
    """
    return (float(prompt_tokens) / 1_000_000.0 * float(input_price)) + (float(completion_tokens) / 1_000_000.0 * float(output_price))


def build_run_windows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[Optional[datetime], Optional[datetime]]]:
    """
    输入：按执行顺序排列的原始 batch 行。
    输出：run_id 到时间窗口的映射。
    功能：给没有 per-app log 的旧批次从 app.log 中切分 token。
    """
    starts: List[Tuple[str, Optional[datetime], float]] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        starts.append((run_id, parse_run_datetime(run_id), float(row.get("elapsed_s") or 0.0)))

    windows: Dict[str, Tuple[Optional[datetime], Optional[datetime]]] = {}
    for idx, (run_id, start, elapsed_s) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else None
        fallback_end = start + timedelta(seconds=max(elapsed_s + 90.0, 120.0)) if start else None
        windows[run_id] = (start, next_start or fallback_end)
    return windows


def usage_for_row(row: Dict[str, Any], global_usage_records: Sequence[Dict[str, Any]], windows: Dict[str, Tuple[Optional[datetime], Optional[datetime]]]) -> Tuple[TokenUsage, str]:
    """
    输入：合并后的 APP 行、全局 LLM_USAGE 记录、旧批次时间窗口。
    输出：TokenUsage 和 token 来源说明。
    功能：优先使用 per-app log；没有时从 app.log 时间窗口恢复。
    """
    raw_log_path = str(row.get("log_path") or "").strip()
    log_path = Path(raw_log_path) if raw_log_path else None
    if log_path is not None and log_path.exists() and log_path.is_file():
        usage = parse_tokens_from_text(log_path.read_text(encoding="utf-8", errors="replace"))
        return usage, "per_app_log"

    original_run_id = str(row.get("original_run_id") or row.get("run_id") or "")
    start, end = windows.get(original_run_id, (None, None))
    usage = token_usage_from_global_window(global_usage_records, start, end)
    return usage, "app_log_window" if usage.total else "missing"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """
    输入：CSV 路径、行列表、字段名。
    输出：UTF-8-SIG CSV 文件。
    功能：保证 Excel 打开中文字段时不乱码。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_summary(path: Path, rows: Sequence[Dict[str, Any]], totals: Dict[str, Any]) -> None:
    """
    输入：Markdown 路径、APP 统计行、总计信息。
    输出：中文 Markdown 总览。
    功能：生成适合人直接阅读的统计表。
    """
    lines = [
        "# 47 个 APP 运行统计汇总",
        "",
        "## 总览",
        "",
        f"- APP 数量：{totals.get('app_count', 0)}",
        f"- 成功退出数量：{totals.get('exit0_count', 0)}",
        f"- 有 analysis 结果数量：{totals.get('analysis_count', 0)}",
        f"- 完成 router 分析 UI 数：{totals.get('router_analyzed_ui_count', 0)}",
        f"- 总动作数：{totals.get('action_count', 0)}",
        f"- 总耗时秒数：{totals.get('elapsed_s', 0):.1f}",
        f"- 总 token：{totals.get('total_tokens', 0)}",
        f"- 按 gpt-4o 估算费用：${totals.get('cost_gpt4o_usd', 0):.4f}",
        f"- 按 gemini-2.5-flash 估算费用：${totals.get('cost_gemini_25_flash_usd', 0):.4f}",
        "",
        "## APP 明细表",
        "",
        "| # | 来源 | APP | package | 问卷 | 退出码 | 停止原因 | 耗时(s) | Router UI | 动作数 | Token | 4o费用($) | Gemini费用($) |",
        "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {index} | {record_source} | {app_name} | `{package}` | {questionnaire_type} | {exit_code} | {stop_reason} | {elapsed_s:.1f} | {router_analyzed_ui_count} | {action_count} | {total_tokens} | {cost_gpt4o_usd:.4f} | {cost_gemini_25_flash_usd:.4f} |".format(
                index=int(row.get("index") or 0),
                record_source=safe_text(row.get("record_source")),
                app_name=safe_text(row.get("app_name")),
                package=safe_text(row.get("package")),
                questionnaire_type=safe_text(row.get("questionnaire_type")),
                exit_code=int(row.get("exit_code") if row.get("exit_code") is not None else -999),
                stop_reason=safe_text(row.get("stop_reason")),
                elapsed_s=float(row.get("elapsed_s") or 0.0),
                router_analyzed_ui_count=int(row.get("router_analyzed_ui_count") or 0),
                action_count=int(row.get("action_count") or 0),
                total_tokens=int(row.get("total_tokens") or 0),
                cost_gpt4o_usd=float(row.get("cost_gpt4o_usd") or 0.0),
                cost_gemini_25_flash_usd=float(row.get("cost_gemini_25_flash_usd") or 0.0),
            )
        )
    avg_elapsed = (float(totals.get("elapsed_s") or 0.0) / max(1, int(totals.get("app_count") or 0)))
    avg_router_ui = (float(totals.get("router_analyzed_ui_count") or 0.0) / max(1, int(totals.get("app_count") or 0)))
    avg_actions = (float(totals.get("action_count") or 0.0) / max(1, int(totals.get("app_count") or 0)))
    avg_tokens = (float(totals.get("total_tokens") or 0.0) / max(1, int(totals.get("app_count") or 0)))
    lines.extend(
        [
            "|  | 总计 |  |  |  |  |  | {elapsed_s:.1f} | {router_ui} | {actions} | {tokens} | {cost4o:.4f} | {costgemini:.4f} |".format(
                elapsed_s=float(totals.get("elapsed_s") or 0.0),
                router_ui=int(totals.get("router_analyzed_ui_count") or 0),
                actions=int(totals.get("action_count") or 0),
                tokens=int(totals.get("total_tokens") or 0),
                cost4o=float(totals.get("cost_gpt4o_usd") or 0.0),
                costgemini=float(totals.get("cost_gemini_25_flash_usd") or 0.0),
            ),
            "|  | 平均 |  |  |  |  |  | {elapsed_s:.1f} | {router_ui:.2f} | {actions:.2f} | {tokens:.1f} | {cost4o:.4f} | {costgemini:.4f} |".format(
                elapsed_s=avg_elapsed,
                router_ui=avg_router_ui,
                actions=avg_actions,
                tokens=avg_tokens,
                cost4o=float(totals.get("cost_gpt4o_usd") or 0.0) / max(1, int(totals.get("app_count") or 0)),
                costgemini=float(totals.get("cost_gemini_25_flash_usd") or 0.0) / max(1, int(totals.get("app_count") or 0)),
            ),
            "",
            "## 表格字段说明",
            "",
            "- 来源：`base` 表示使用原始 47 批次结果；`rerun` 表示使用后续补跑结果覆盖。",
            "- 问卷：本 APP 使用的问卷目录类型，包括 `games`、`social_apps`、`others`。",
            "- 退出码：主流程进程退出码，`0` 表示脚本层面正常结束，非 0 表示运行报错或启动失败。",
            "- 停止原因：主流程停止分支，具体含义见下方“停止原因说明”。",
            "- Router UI：完成 block_router/router 分析的去重 UI 数，按 `state_sig` 去重；这是你当前主要关注的“经过 router 分析的 UI 数”。",
            "- 动作数：主流程内部实际执行动作计数，来自 `run_analysis_summary.json`，不是 trace 行数。",
            "- Token：prompt token 与 completion token 之和。",
            "- 4o费用/Gemini费用：分别按 gpt-4o 和 gemini-2.5-flash 价格估算的美元成本。",
            "",
            "## 停止原因说明",
            "",
            "- `time_budget_reached`：达到本次运行时间预算后停止。",
            "- `root_exhausted`：根页面及其可执行候选基本探索完毕，没有新的可执行动作。",
            "- `recovery_route_exhausted`：恢复/回源路径尝试耗尽，系统判断继续恢复收益不高，停止当前 APP。",
            "- `overlay_recovery_exhausted`：弹窗/遮罩恢复次数达到上限，避免卡死循环后停止。",
            "- `strong_stall_timeout`：长时间没有强进展，触发停滞保护后停止。",
            "- `state_saturation_reached`：状态数量或状态重复程度达到饱和阈值，继续探索收益较低。",
            "- 空值：没有成功导出 analysis summary，通常表示启动失败、早期异常或 trace 被删除。",
        ]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_field_doc(path: Path) -> None:
    """
    输入：Markdown 路径。
    输出：中文字段说明文档。
    功能：说明统计脚本读取哪些本地文件、输出哪些字段、字段含义是什么。
    """
    text = """# 批量运行统计字段说明

## 输入来源

- `batch_summary.json`：读取原始批次 APP 列表、package、APP 名称、类别、问卷类型、run_id、trace_dir、elapsed_s、exit_code。
- `rerun_summary.json`：读取补跑批次结果；同 package 存在 rerun 时，统计表使用 rerun 的 run_id、trace_dir、elapsed_s、exit_code。
- `trace/analysis/run_analysis_summary.json`：读取 stop_reason、graph_node_count、graph_edge_count、action_count、unfinished_state_count、no_progress_loops。
- `trace/observations/router/*.json`：读取每个 UI 的 router_answers 和 matched_block_ids，用于统计完成 router 分析的 UI 数、router 命中和 block 命中。
- `trace/observations/blocks_fill/*.json`：读取完成 blocks_fill 的 UI 数。
- `logs/*.log`：rerun 脚本保存的单 APP 控制台日志，优先用于解析 `[TOKENS]` token 汇总。
- `app.log`：旧批次没有单 APP 日志时，用 run_id 时间窗口近似恢复 `LLM_USAGE` token。

## 关键字段

- `record_source`：`base` 表示使用原始 47 批次结果；`rerun` 表示该 APP 使用补跑结果覆盖。
- `router_analyzed_ui_count`：完成 block_router 分析的去重 UI 数，按 `state_sig` 去重。优先来自 `observations/router`，缺失时回退到 `trace.jsonl` 的 `llm_result kind=block_router`。
- `router_observation_count`：router observation 文件数量，可能大于去重 UI 数。
- `router_duplicate_count`：`router_observation_count - router_analyzed_ui_count`，用于观察重复 observation。
- `router_hit_question_count`：出现非空 `new_answer` 的去重 router question 数。
- `router_hit_total_count`：非空 router answer 的总次数。
- `matched_block_unique_count`：被 router 条件命中的去重 block 数。
- `matched_block_total_count`：block 命中总次数。
- `block_fill_ui_count`：完成 blocks_fill observation 的去重 UI 数。
- `action_count`：主流程内部实际执行动作计数，来自 `run_analysis_summary.json`，不是 trace 行数。
- `prompt_tokens` / `completion_tokens` / `total_tokens`：LLM token 用量。
- `token_source`：`per_app_log` 表示从单 APP 日志解析；`app_log_window` 表示从全局 app.log 按时间窗口恢复；`missing` 表示没有恢复到 token。
- `cost_gpt4o_usd`：按 gpt-4o 输入 `$2.2/M`、输出 `$8.8/M` 估算。
- `cost_gemini_25_flash_usd`：按 gemini-2.5-flash 输入 `$0.27/M`、输出 `$2.25/M` 估算。

## 注意事项

- 旧批次 token 没有单 APP 日志，只能通过 `app.log` 时间窗口恢复；如果 app.log 被截断或混入其他运行，旧批次 token 会不完整。
- router 命中在本统计脚本中的定义是：`router_answers` 中某个 question 的 `new_answer` 非空。
- block 命中沿用代码逻辑：`QuestionnaireState.match_blocks_from_router_answers` 根据 `block_show_if` 从 router answer 本地匹配得到。
"""
    path.write_text(text, encoding="utf-8")


def load_block_catalog(questionnaire_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    输入：问卷根目录，例如 questionnaire-UI。
    输出：block_id 到 block 元数据的映射。
    功能：为 block 命中截图目录提供 topic/module/source 信息。
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    if not questionnaire_root.exists():
        return catalog
    for blocks_path in sorted(questionnaire_root.glob("*/questionnaire_blocks.json")):
        data = read_json(blocks_path, {})
        if not isinstance(data, dict):
            continue
        source = blocks_path.parent.name
        for block in list(data.get("blocks") or []):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id") or "").strip()
            if not block_id:
                continue
            item = dict(block)
            item.setdefault("source_collection", source)
            item.setdefault("source_dir", str(blocks_path.parent))
            catalog[block_id] = item
    return catalog


def state_sig_to_debug_suffix(state_sig: str) -> str:
    """
    输入：state_sig，例如 phash:abc 或 xml:abc。
    输出：debug_pages 目录名里使用的安全后缀。
    功能：把 observation 的 state_sig 对应回 debug_pages/*_<suffix>。
    """
    return safe_token(state_sig)[:48]


def find_state_screenshot_raw(trace_dir: Path, state_sig: str) -> Optional[Path]:
    """
    输入：trace 目录和 state_sig。
    输出：该 state 对应的 screenshot_raw.png 路径；找不到返回 None。
    功能：根据 state_sig 反查 debug_pages 中的原始截图。
    """
    suffix = state_sig_to_debug_suffix(state_sig)
    roots = [trace_dir / "states", trace_dir / "debug_pages"]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob(f"*_{suffix}")))
    for page_dir in candidates:
        shot = page_dir / "screenshot_raw.png"
        if shot.exists():
            return shot
    return None


def iter_router_hit_payloads(trace_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    """
    输入：单个 APP trace 目录。
    输出：包含 state_sig 和 matched_block_ids 的 router 命中记录。
    功能：同时支持新 states/<UI>/llm 和旧 observations/router 结构。
    """
    for path in iter_state_llm_files(trace_dir, "navigation_router_result.json"):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        yield path, {
            "state_sig": str(payload.get("state_sig") or result.get("state_sig") or ""),
            "matched_block_ids": list(payload.get("matched_block_ids") or []),
        }
    for path in iter_observation_files(trace_dir, "router"):
        payload = read_json(path, {})
        if isinstance(payload, dict):
            yield path, payload


def collect_block_hit_images(output_dir: Path, app_rows: Sequence[Dict[str, Any]], questionnaire_root: Path) -> Dict[str, Any]:
    """
    输入：分析输出目录、APP 统计行、问卷根目录。
    输出：block 图片聚合统计。
    功能：按 block topic/id 建目录，复制每个命中 block 对应 UI 的 screenshot_raw.png 供人工审查。
    """
    catalog = load_block_catalog(questionnaire_root)
    root = output_dir / "block_hit_images"
    manifest_rows: List[Dict[str, Any]] = []
    copied = 0
    missing = 0

    for app in app_rows:
        trace_dir = Path(str(app.get("trace_dir") or ""))
        if not trace_dir.exists():
            continue
        for obs_path, payload in iter_router_hit_payloads(trace_dir):
            state_sig = str(payload.get("state_sig") or "")
            block_ids = [str(x or "").strip() for x in list(payload.get("matched_block_ids") or []) if str(x or "").strip()]
            if not state_sig or not block_ids:
                continue
            screenshot = find_state_screenshot_raw(trace_dir, state_sig)
            for block_id in block_ids:
                block = catalog.get(block_id, {})
                topic = str(block.get("topic") or block_id).strip() or block_id
                dir_name = f"{safe_token(topic)[:80]}__{safe_token(block_id)[:80]}"
                block_dir = root / dir_name
                block_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{int(app.get('index') or 0):02d}_{safe_token(app.get('package'))}_{state_sig_to_debug_suffix(state_sig)}.png"
                out_path = block_dir / out_name
                if screenshot and screenshot.exists():
                    shutil.copy2(screenshot, out_path)
                    copied += 1
                    copied_path = str(out_path)
                else:
                    missing += 1
                    copied_path = ""
                manifest_rows.append(
                    {
                        "block_id": block_id,
                        "block_topic": topic,
                        "block_module": str(block.get("module") or ""),
                        "source_collection": str(block.get("source_collection") or ""),
                        "package": str(app.get("package") or ""),
                        "app_name": str(app.get("app_name") or ""),
                        "run_id": str(app.get("run_id") or ""),
                        "state_sig": state_sig,
                        "observation_path": str(obs_path),
                        "source_screenshot_raw": str(screenshot or ""),
                        "copied_screenshot": copied_path,
                    }
                )

    fields = [
        "block_id", "block_topic", "block_module", "source_collection", "package", "app_name",
        "run_id", "state_sig", "observation_path", "source_screenshot_raw", "copied_screenshot",
    ]
    write_csv(output_dir / "block_hit_images_manifest.csv", manifest_rows, fields)
    write_json(output_dir / "block_hit_images_manifest.json", manifest_rows)
    return {"block_hit_image_count": copied, "block_hit_image_missing": missing, "block_hit_manifest_count": len(manifest_rows), "block_hit_image_root": str(root)}


def build_summary_rows(args: argparse.Namespace, output_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    输入：解析后的参数和输出目录。
    输出：APP 行、token by op 行、router 汇总行、block 汇总行、总计。
    功能：执行所有统计计算，但不负责写文件。
    """
    base_batch_dir = Path(args.base_batch_dir).resolve()
    rerun_arg = str(args.rerun_batch_dir or "").strip()
    if rerun_arg.lower() == "auto":
        rerun_batch_dir = find_latest_rerun_batch(base_batch_dir)
    elif rerun_arg:
        rerun_batch_dir = Path(rerun_arg).resolve()
    else:
        rerun_batch_dir = None

    base_rows = load_base_rows(base_batch_dir)
    rerun_rows = load_rerun_rows(rerun_batch_dir)
    merged_rows = merge_base_and_rerun(base_rows, rerun_rows)
    windows = build_run_windows(base_rows)
    global_usage_records = parse_global_llm_usage(Path(args.app_log).resolve())

    app_rows: List[Dict[str, Any]] = []
    token_by_op_rows: List[Dict[str, Any]] = []
    router_global: Counter = Counter()
    block_global: Counter = Counter()

    for row in merged_rows:
        trace_dir = Path(str(row.get("trace_dir") or ""))
        trace_exists = trace_dir.exists()
        analysis = load_trace_analysis(trace_dir) if trace_exists else {
            "analysis_exists": False,
            "stop_reason": "",
            "graph_node_count": None,
            "graph_edge_count": None,
            "action_count": row.get("action_count"),
            "unfinished_state_count": None,
            "no_progress_loops": None,
            "no_new_state_count": None,
        }
        obs_summary, router_hits, block_hits = summarize_router_block_observations(trace_dir) if trace_exists else ({}, Counter(), Counter())
        router_global.update(router_hits)
        block_global.update(block_hits)

        usage, token_source = usage_for_row(row, global_usage_records, windows)
        cost_gpt4o = calculate_cost(usage.prompt, usage.completion, args.gpt4o_input_per_m, args.gpt4o_output_per_m)
        cost_gemini = calculate_cost(usage.prompt, usage.completion, args.gemini_input_per_m, args.gemini_output_per_m)

        app_row = {
            "index": row.get("index"),
            "record_source": row.get("record_source"),
            "package": row.get("package"),
            "app_name": row.get("app_name"),
            "category": row.get("category"),
            "questionnaire_type": row.get("questionnaire_type"),
            "run_id": row.get("run_id"),
            "original_run_id": row.get("original_run_id"),
            "trace_dir": str(trace_dir),
            "trace_exists": trace_exists,
            "exit_code": row.get("exit_code"),
            "elapsed_s": row.get("elapsed_s"),
            "analysis_exists": analysis.get("analysis_exists"),
            "stop_reason": analysis.get("stop_reason") or row.get("stop_reason") or "",
            "graph_node_count": analysis.get("graph_node_count"),
            "graph_edge_count": analysis.get("graph_edge_count"),
            "action_count": analysis.get("action_count") if analysis.get("action_count") is not None else row.get("action_count"),
            "unfinished_state_count": analysis.get("unfinished_state_count"),
            "no_progress_loops": analysis.get("no_progress_loops"),
            "no_new_state_count": analysis.get("no_new_state_count"),
            "router_observation_count": obs_summary.get("router_observation_count", 0),
            "router_analyzed_ui_count": obs_summary.get("router_analyzed_ui_count", 0),
            "router_duplicate_count": obs_summary.get("router_duplicate_count", 0),
            "router_hit_question_count": obs_summary.get("router_hit_question_count", 0),
            "router_hit_total_count": obs_summary.get("router_hit_total_count", 0),
            "matched_block_unique_count": obs_summary.get("matched_block_unique_count", 0),
            "matched_block_total_count": obs_summary.get("matched_block_total_count", 0),
            "block_fill_observation_count": obs_summary.get("block_fill_observation_count", 0),
            "block_fill_ui_count": obs_summary.get("block_fill_ui_count", 0),
            "token_source": token_source,
            "llm_calls": usage.calls,
            "prompt_tokens": usage.prompt,
            "completion_tokens": usage.completion,
            "total_tokens": usage.total,
            "cost_gpt4o_usd": round(cost_gpt4o, 6),
            "cost_gemini_25_flash_usd": round(cost_gemini, 6),
        }
        app_rows.append(app_row)

        for op, op_usage in sorted((usage.by_op or {}).items()):
            token_by_op_rows.append(
                {
                    "index": row.get("index"),
                    "record_source": row.get("record_source"),
                    "package": row.get("package"),
                    "app_name": row.get("app_name"),
                    "run_id": row.get("run_id"),
                    "op": op,
                    "calls": op_usage.get("calls", 0),
                    "prompt_tokens": op_usage.get("prompt_tokens", 0),
                    "completion_tokens": op_usage.get("completion_tokens", 0),
                    "total_tokens": op_usage.get("total_tokens", 0),
                }
            )

    router_rows = [
        {"router_question_id": key, "hit_count": count}
        for key, count in sorted(router_global.items(), key=lambda item: (-item[1], item[0]))
    ]
    block_rows = [
        {"block_id": key, "hit_count": count}
        for key, count in sorted(block_global.items(), key=lambda item: (-item[1], item[0]))
    ]

    totals = {
        "base_batch_dir": str(base_batch_dir),
        "rerun_batch_dir": str(rerun_batch_dir or ""),
        "output_dir": str(output_dir),
        "app_count": len(app_rows),
        "exit0_count": sum(1 for row in app_rows if int(row.get("exit_code") if row.get("exit_code") is not None else -1) == 0),
        "analysis_count": sum(1 for row in app_rows if row.get("analysis_exists")),
        "elapsed_s": sum(float(row.get("elapsed_s") or 0.0) for row in app_rows),
        "router_analyzed_ui_count": sum(int(row.get("router_analyzed_ui_count") or 0) for row in app_rows),
        "action_count": sum(int(row.get("action_count") or 0) for row in app_rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in app_rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in app_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in app_rows),
        "cost_gpt4o_usd": round(sum(float(row.get("cost_gpt4o_usd") or 0.0) for row in app_rows), 6),
        "cost_gemini_25_flash_usd": round(sum(float(row.get("cost_gemini_25_flash_usd") or 0.0) for row in app_rows), 6),
        "router_hit_question_count": len(router_rows),
        "matched_block_count": len(block_rows),
    }
    write_json(output_dir / "run_config.json", {"args": vars(args), "totals": totals})
    return app_rows, token_by_op_rows, router_rows, block_rows, totals


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    输入：可选命令行参数。
    输出：进程退出码。
    功能：生成 APP 统计、token 拆分、router/block 命中汇总和字段说明文档。
    """
    args = parse_args(argv or sys.argv[1:])
    base_name = safe_token(Path(args.base_batch_dir).resolve().name)
    output_dir = Path(args.output_root).resolve() / f"{safe_token(args.run_label)}_{base_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    app_rows, token_by_op_rows, router_rows, block_rows, totals = build_summary_rows(args, output_dir)

    app_fields = [
        "index", "record_source", "package", "app_name", "category", "questionnaire_type", "run_id", "original_run_id",
        "trace_dir", "trace_exists", "exit_code", "elapsed_s", "analysis_exists", "stop_reason", "graph_node_count",
        "graph_edge_count", "action_count", "unfinished_state_count", "no_progress_loops", "no_new_state_count",
        "router_observation_count", "router_analyzed_ui_count", "router_duplicate_count", "router_hit_question_count",
        "router_hit_total_count", "matched_block_unique_count", "matched_block_total_count", "block_fill_observation_count",
        "block_fill_ui_count", "token_source", "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
        "cost_gpt4o_usd", "cost_gemini_25_flash_usd",
    ]
    token_fields = ["index", "record_source", "package", "app_name", "run_id", "op", "calls", "prompt_tokens", "completion_tokens", "total_tokens"]

    write_csv(output_dir / "app_summary.csv", app_rows, app_fields)
    write_json(output_dir / "app_summary.json", app_rows)
    write_csv(output_dir / "token_by_op.csv", token_by_op_rows, token_fields)
    write_csv(output_dir / "router_hit_summary.csv", router_rows, ["router_question_id", "hit_count"])
    write_csv(output_dir / "block_hit_summary.csv", block_rows, ["block_id", "hit_count"])
    image_summary = collect_block_hit_images(output_dir, app_rows, Path(args.questionnaire_root).resolve())
    totals.update(image_summary)
    write_json(output_dir / "totals.json", totals)
    write_markdown_summary(output_dir / "app_summary.md", app_rows, totals)
    write_field_doc(output_dir / "字段说明.md")

    print(f"[SUMMARY] output_dir={output_dir}")
    print(f"[SUMMARY] app_count={totals['app_count']} analysis_count={totals['analysis_count']} router_ui={totals['router_analyzed_ui_count']}")
    print(f"[SUMMARY] total_tokens={totals['total_tokens']} gpt4o=${totals['cost_gpt4o_usd']:.4f} gemini=${totals['cost_gemini_25_flash_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
