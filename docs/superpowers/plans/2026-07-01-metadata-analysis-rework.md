# Metadata Analysis Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 metadata 分析流程：LLM 只生成 `app_intro` 和 `focus_hints`，问卷类型直接读取 CSV 的 `app_type` 字段，并增加一个可单独运行的 metadata 调试脚本。

**Architecture:** metadata 流程拆成两条清晰路径：`app_type` 直接决定问卷目录，LLM metadata summary 只负责生成后续 UI 探索提示。主流程函数放在 `main.py`，测试脚本复用这些函数，避免主流程和调试脚本逻辑分叉。

**Tech Stack:** Python 3.12, Pydantic, OpenAI-compatible SDK via `GPTClient`, CSV metadata file, JSON debug output.

---

## 现状和目标

当前 metadata 分析输入仍是旧版十字段：

```text
description
descriptionHTML
summary
contentRating
contentRatingDescription
offersIAP
inAppProductPrice
genre
genreId
categories
```

当前 LLM 输出包含：

```text
app_id
app_intro
focus_hints
questionnaire_type
notes
```

问题：

- `questionnaire_type` 现在不应由 LLM 判断，应该直接读取 CSV 的 `app_type` 字段。
- `questionnaire_type: Literal["games", "social_apps", "others", ""]` 会导致 Gemini/UniAPI structured parse 报错，因为 enum 里不能包含空字符串。
- metadata 输入字段应换成最终版 CSV 中对探索有用的字段。

目标：

- `app_type`：CSV 中的问卷类型字段，取值已经确认固定为 `games` / `social_apps` / `others`，直接读取，不做 LLM 推断。
- `app_intro`：一句话全局 APP 背景，必须利用 `application_category`。
- `focus_hints`：给后续 UI 探索的关注点提示。
- `metadata_result.json`：只保存 LLM 输出，不保存输入，不包含 `questionnaire_type`。

---

## 文件结构

### 修改文件

- `F:\workplace\framework\gpt_cls.py`
  - 修改 `AppMetadataSummary` 输出 schema，删除 `questionnaire_type`。
  - 修改 `_APP_METADATA_SYSTEM`，删除问卷类型推断步骤，改成最终版 CSV 字段说明。
  - 修改 `GPTClient.analyze_app_metadata()`，不再读取或清洗 `questionnaire_type`。

- `F:\workplace\framework\main.py`
  - 替换 `_META_SELECTED_FIELDS` 为最终 metadata LLM 输入字段。
  - 修改 `_pick_metadata_fields()`，直接读取最终 CSV 字段。
  - 新增 `_read_questionnaire_type_from_metadata_row()`，直接从 `app_type` 读取问卷类型。
  - 修改 `run` 主流程，让 `metadata_questionnaire_type` 来自 `app_type`，而不是 LLM 输出。
  - 保留 `_write_metadata_result_json()`，但写入内容不再包含 `questionnaire_type`。

- `F:\workplace\framework\更新日志.md`
  - 代码完成并测试通过后追加本次更新记录。

### 新增文件

- `F:\workplace\framework\test_debug\test_metadata_analysis.py`
  - 单独测试 metadata 分析。
  - 输入 APP 包名和 metadata CSV。
  - 复用 `main.py` 的 CSV 查找、字段抽取、`app_type` 读取、metadata result 写入逻辑。
  - 调用 `GPTClient.analyze_app_metadata()`。
  - 输出 JSON 到 `F:\workplace\framework\test_debug\metadata_analysis_outputs\`。

---

## Task 1: 修改 metadata LLM 输出 schema 和 prompt

**Files:**
- Modify: `F:\workplace\framework\gpt_cls.py`

- [ ] **Step 1: 修改 `AppMetadataSummary`**

把当前字段：

```python
questionnaire_type: Literal["games", "social_apps", "others", ""] = Field(
    "",
    description="Questionnaire bucket inferred from genre fields. Empty when insufficient evidence.",
)
```

删除。保留：

```python
class AppMetadataSummary(BaseModel):
    """
    One-shot summary generated from app-level metadata.
    Input: selected final-dataset metadata fields.
    Output: compact reusable app context for downstream UI exploration.
    Function: stores app-level exploration hints without deciding questionnaire type.
    """
    app_id: str = Field("", description="App package id (e.g., com.example.app)")
    app_intro: str = Field(
        "",
        description=(
            "One concise sentence describing the app's core purpose and category context. "
            "It must consider application_category when available."
        ),
    )
    focus_hints: str = Field(
        "",
        description=(
            "Potential UI exploration focus hints, written as short phrases separated by semicolons. "
            "Leave empty only when metadata evidence is insufficient."
        ),
    )
    notes: str = Field(
        "",
        description="Brief reason when any output field is empty or metadata is incomplete/garbled.",
    )
```

- [ ] **Step 2: 修改 `_APP_METADATA_SYSTEM`**

将 prompt 改成只说明最终版 CSV 字段。核心内容：

```python
_APP_METADATA_SYSTEM = """You are an assistant that summarizes Android app metadata for downstream UI exploration.

GOAL:
- Convert selected app metadata fields into compact reusable context for later Android UI exploration and UI analysis.
- Keep the output factual and concise; do not invent details not supported by the metadata.
- Do not infer questionnaire type. Questionnaire type is provided by a separate CSV field outside this LLM call.

INPUTS:
- app_id:
  - Android package id.
- app_metadata: contains selected fields from the final benchmark metadata CSV:
  - app_name:
    - App display name. Use it only to understand the app; do not simply repeat the name in app_intro.
  - content_descriptors:
    - Store-provided content descriptors such as ads or in-app purchases.
  - age_rating_descriptors:
    - Store-provided descriptors related to age/content considerations. Do not infer or output the age rating itself.
  - category_name:
    - Human-readable app category.
  - category_code:
    - Store category code.
  - application_category:
    - Stable normalized category. IMPORTANT: incorporate this signal into app_intro when present.
  - details_full_description:
    - Main app-store description text; primary source for app functionality/content.
  - details_interactive_elements:
    - Store-provided interactive element hints, such as in-app purchases or user interaction.
  - details_in_app_purchases:
    - In-app purchase price/range or purchase availability note.
  - data_safety_summary:
    - Summary of collected/shared data and safety practices.
  - security_practices_text:
    - Store-provided data security practices.
  - permissions_text:
    - Permission information useful for UI exploration.

WORKFLOW:
1) Build `app_intro`.
   - Output one concise sentence that explains what the app appears to do and what broad category/context it belongs to.
   - Use details_full_description as the main functionality source.
   - Use application_category as an important category signal.
   - Do not mention the app's exact name unless necessary for clarity.

2) Build `focus_hints`.
   - Output short natural-language hints for downstream UI exploration.
   - Focus on UI-relevant signals such as purchases, subscriptions, ads, social/user interaction, content risks, permissions, data safety, account/settings/policy areas.
   - Format as semicolon-separated phrases.
   - Do not output or infer the age rating.

3) Fill `notes`.
   - If app_intro or focus_hints is weak because source fields are missing, empty, garbled, or insufficient, briefly explain why.
   - If both fields are confidently filled, notes should be empty.

OUTPUT (strict JSON matching AppMetadataSummary):
- app_id
- app_intro
- focus_hints
- notes

RULES:
- Use only provided metadata.
- Do not infer questionnaire_type.
- Do not use age_rating, teacher_approved, play_families_policy_committed, or privacy_policy_url; those fields are not provided to this LLM call.
- Follow output format strictly.
"""
```

- [ ] **Step 3: 修改 `analyze_app_metadata()` 收尾逻辑**

删除：

```python
if out.questionnaire_type not in ("games", "social_apps", "others", ""):
    out.questionnaire_type = ""
```

保留并更新截断：

```python
out.app_id = app_id or out.app_id
out.app_intro = str(out.app_intro or "")[:280]
out.focus_hints = str(out.focus_hints or "")[:500]
out.notes = str(out.notes or "")[:500]
return out
```

---

## Task 2: 修改主流程 metadata 字段抽取和问卷类型读取

**Files:**
- Modify: `F:\workplace\framework\main.py`

- [ ] **Step 1: 替换 `_META_SELECTED_FIELDS`**

改成最终版 LLM 输入字段：

```python
_META_SELECTED_FIELDS = [
    "app_name",
    "content_descriptors",
    "age_rating_descriptors",
    "category_name",
    "category_code",
    "details_full_description",
    "application_category",
    "data_safety_summary",
    "security_practices_text",
    "permissions_text",
    "details_interactive_elements",
    "details_in_app_purchases",
]
```

字段说明：

- `application_category`：稳定类别字段，必须给 LLM 用于 `app_intro`。
- `details_full_description`：APP 功能描述主来源。
- `content_descriptors` / `age_rating_descriptors` / `details_interactive_elements` / `details_in_app_purchases`：用于生成探索关注点。
- `data_safety_summary` / `security_practices_text` / `permissions_text`：用于生成数据安全、权限、隐私相关探索关注点。

明确不传：

```text
age_rating
teacher_approved
play_families_policy_committed
privacy_policy_url
```

- [ ] **Step 2: 简化 `_pick_metadata_fields()`**

改成直接按最终字段名读取，不再映射旧字段：

```python
def _pick_metadata_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: one row from known_dataset_200_metadata.csv.
    Output: selected metadata fields consumed by GPTClient.analyze_app_metadata.
    Function: keeps only fields useful for app-level exploration context and avoids answer-leaking fields.
    """
    return {key: str(row.get(key, "") or "").strip() for key in _META_SELECTED_FIELDS}
```

- [ ] **Step 3: 新增 `_read_questionnaire_type_from_metadata_row()`**

`app_type` 是 CSV 里已经确认无空值、取值固定的问卷类型字段。直接读取即可。

```python
def _read_questionnaire_type_from_metadata_row(row: Dict[str, Any]) -> str:
    """
    Input: one row from known_dataset_200_metadata.csv.
    Output: questionnaire type string from CSV field `app_type`.
    Function: uses the benchmark dataset's curated questionnaire type instead of asking LLM to infer it.
    """
    return str(row.get("app_type", "") or "").strip()
```

变量说明：

- `app_type`：CSV 中人工/规则确认后的问卷类型，直接对应问卷目录名。
- `metadata_questionnaire_type`：主流程中实际用于选择问卷目录的变量。

- [ ] **Step 4: 修改 metadata 分析主流程**

当前逻辑中：

```python
metadata_questionnaire_type = _normalize_questionnaire_type(str(meta_result.questionnaire_type or ""))
```

改成在找到 CSV 行后直接读取：

```python
metadata_questionnaire_type = _read_questionnaire_type_from_metadata_row(row)
selected_meta = _pick_metadata_fields(row)
meta_result = gpt.analyze_app_metadata(app_id=args.package, app_metadata=selected_meta)
```

日志中保留：

```python
logger.info(
    "Metadata analyzed app=%s questionnaire_type=%s intro=%s hints=%s result=%s",
    args.package,
    metadata_questionnaire_type or "-",
    bool(app_intro),
    bool(focus_hints),
    metadata_result_path or "-",
)
```

- [ ] **Step 5: 问卷目录选择逻辑保持不变**

`questionnaire_type_source=metadata/auto` 时，继续使用：

```python
candidate_dir = Path(manual_questionnaire_dir).resolve().parent / metadata_questionnaire_type
```

这里 `metadata_questionnaire_type` 已经来自 CSV `app_type`，不是 LLM。

---

## Task 3: 增加 metadata 单独测试脚本

**Files:**
- Create: `F:\workplace\framework\test_debug\test_metadata_analysis.py`

- [ ] **Step 1: 创建脚本文件**

脚本需要复用主流程函数：

```python
"""
Standalone metadata analysis debug runner.

Input:
- Android package id.
- Metadata CSV path.
- Optional output directory.

Output:
- metadata_result.json containing only AppMetadataSummary LLM output.
- console summary showing app_type, app_intro/focus_hints presence, and output path.

Function:
- Reuses main.py metadata row lookup, field selection, app_type reading, and metadata result writer.
- Lets developers test metadata analysis without running Appium exploration.
"""
```

核心代码结构：

```python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_config import load_project_env
from gpt_cls import GPTClient
from main import (
    _find_metadata_row_by_app_id,
    _pick_metadata_fields,
    _read_csv_rows,
    _read_questionnaire_type_from_metadata_row,
    _write_metadata_result_json,
)


def parse_args() -> argparse.Namespace:
    """
    Input: command-line arguments.
    Output: parsed arguments for metadata debug run.
    Function: provides a small CLI for testing metadata analysis outside Appium.
    """
    parser = argparse.ArgumentParser(description="Run metadata LLM analysis for one app.")
    parser.add_argument("--package", required=True, help="Android package id, matching CSV field APID.")
    parser.add_argument(
        "--metadata-csv-path",
        default=str(ROOT / "APP_csv" / "known_dataset_200_metadata.csv"),
        help="Metadata CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "test_debug" / "metadata_analysis_outputs"),
        help="Directory where debug output run folder is written.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--timeout", type=float, default=120.0, help="LLM request timeout seconds.")
    return parser.parse_args()


def main() -> int:
    """
    Input: CLI args and project .env.
    Output: process exit code; writes metadata_result.json on success.
    Function: runs the same metadata summary logic used by the main exploration command.
    """
    args = parse_args()
    load_project_env(ROOT / ".env")

    rows = _read_csv_rows(Path(args.metadata_csv_path))
    row = _find_metadata_row_by_app_id(rows, args.package)
    if row is None:
        print(f"[METADATA] package not found in CSV: {args.package}")
        return 2

    app_type = _read_questionnaire_type_from_metadata_row(row)
    selected_meta = _pick_metadata_fields(row)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[METADATA] OPENAI_API_KEY not set")
        return 2

    gpt = GPTClient(api_key=api_key, model=args.model, temperature=args.temperature, timeout_s=args.timeout)
    result = gpt.analyze_app_metadata(app_id=args.package, app_metadata=selected_meta)

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + args.package
    output_path = _write_metadata_result_json(result, args.output_dir, run_id)

    print(f"[METADATA] package={args.package}")
    print(f"[METADATA] app_type={app_type}")
    print(f"[METADATA] app_intro={bool(result.app_intro)} focus_hints={bool(result.focus_hints)}")
    print(f"[METADATA] result={output_path}")
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行脚本语法检查**

Run:

```powershell
cd F:\workplace\framework
.\.venv\Scripts\python.exe -m py_compile .\main.py .\gpt_cls.py .\test_debug\test_metadata_analysis.py
```

Expected:

```text
无输出，退出码为 0
```

- [ ] **Step 3: 运行一个 metadata 单独测试**

这个测试会调用 LLM，预计几十秒。

Run:

```powershell
cd F:\workplace\framework
.\.venv\Scripts\python.exe .\test_debug\test_metadata_analysis.py `
  --package com.pazugames.avatarworld `
  --metadata-csv-path F:\workplace\framework\APP_csv\known_dataset_200_metadata.csv
```

Expected:

```text
[METADATA] package=com.pazugames.avatarworld
[METADATA] app_type=games
[METADATA] app_intro=True focus_hints=True
[METADATA] result=F:\workplace\framework\test_debug\metadata_analysis_outputs\<run_id>\metadata_result.json
```

检查输出 JSON：

```json
{
  "app_id": "com.pazugames.avatarworld",
  "app_intro": "...",
  "focus_hints": "...",
  "notes": "..."
}
```

确认不应包含：

```text
questionnaire_type
```

---

## Task 4: 更新主流程验证

**Files:**
- Modify only if needed: `F:\workplace\framework\test_debug\bench_run.py`

- [ ] **Step 1: 确认 `bench_run.py` 已传入 metadata CSV**

检查命令中应包含：

```python
"--metadata-csv-path", str(metadata_csv_path)
```

如果已有，不改。

- [ ] **Step 2: 用 dry-run 确认命令仍包含 metadata 参数**

Run:

```powershell
cd F:\workplace\framework
.\.venv\Scripts\python.exe .\test_debug\bench_run.py `
  --limit 1 `
  --start-index 1 `
  --time-budget 60 `
  --device-name 127.0.0.1:7555 `
  --batch-label metadata_rework_dryrun `
  --dry-run
```

Expected:

```text
生成 batch dry-run 记录
命令里包含 --metadata-csv-path F:\workplace\framework\APP_csv\known_dataset_200_metadata.csv
命令里包含 --questionnaire-type-source auto
```

---

## Task 5: 更新文档和更新日志

**Files:**
- Modify: `F:\workplace\framework\更新日志.md`

- [ ] **Step 1: 追加更新记录**

在 `更新日志.md` 末尾追加一节：

```markdown
## 2026-07-01 Metadata 分析流程调整

- 调整 metadata LLM 分析职责：LLM 不再推断问卷类型，只生成 `app_intro`、`focus_hints`、`notes`。
- 问卷类型改为直接读取最终 metadata CSV 的 `app_type` 字段，取值对应 `games` / `social_apps` / `others`。
- metadata LLM 输入字段切换为最终数据集字段，移除 `age_rating`、`teacher_approved`、`play_families_policy_committed`、`privacy_policy_url` 等不应进入分析的问题字段。
- 保留 `metadata_result.json`，仅保存 metadata LLM 输出结果。
- 新增 `test_debug/test_metadata_analysis.py`，用于单独验证 metadata 分析流程。

验证：

- `python -m py_compile main.py gpt_cls.py test_debug/test_metadata_analysis.py` 通过。
- 单 APP metadata 测试可生成不含 `questionnaire_type` 的 `metadata_result.json`。
```

---

## 测试策略

最小测试顺序：

1. 语法检查：

```powershell
cd F:\workplace\framework
.\.venv\Scripts\python.exe -m py_compile .\main.py .\gpt_cls.py .\test_debug\test_metadata_analysis.py
```

2. 单独 metadata 测试：

```powershell
.\.venv\Scripts\python.exe .\test_debug\test_metadata_analysis.py `
  --package com.pazugames.avatarworld `
  --metadata-csv-path F:\workplace\framework\APP_csv\known_dataset_200_metadata.csv
```

3. 批量 dry-run：

```powershell
.\.venv\Scripts\python.exe .\test_debug\bench_run.py `
  --limit 1 `
  --start-index 1 `
  --time-budget 60 `
  --device-name 127.0.0.1:7555 `
  --batch-label metadata_rework_dryrun `
  --dry-run
```

4. 用户确认后再运行真实 5 APP 测试。真实测试会调用 Appium、ADB、LLM，耗时较长，不在代码修改阶段自动跑。

---

## 风险和注意事项

- `app_type` 已由用户确认无空值且取值固定，因此第一版直接读取，不做复杂推断。
- 如果未来 CSV 字段名变化，`_read_questionnaire_type_from_metadata_row()` 会是唯一需要调整的位置。
- 删除 `questionnaire_type` 后，`metadata_result.json` 的结构会变化；下游如有脚本读取该字段，需要改为读取运行参数或 `app_metadata.json` 中的问卷类型信息。
- metadata 单独测试会调用 LLM，可能受代理、API key、UniAPI/Gemini 兼容性影响。

---

## Self-Review

- Spec coverage:
  - 已覆盖 metadata 输入字段重做。
  - 已覆盖 `app_type` 直接决定问卷类型。
  - 已覆盖删除 LLM 输出里的 `questionnaire_type`。
  - 已覆盖新增单独测试脚本。
  - 已覆盖更新日志。
- Placeholder scan:
  - 没有保留 TBD/TODO/稍后实现等占位内容。
- Type consistency:
  - `app_type` 是 CSV 字段。
  - `metadata_questionnaire_type` 是主流程变量。
  - `AppMetadataSummary` 输出只含 `app_id/app_intro/focus_hints/notes`。

