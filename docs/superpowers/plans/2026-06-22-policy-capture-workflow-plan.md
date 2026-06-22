# Policy Capture Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将隐私政策 / 服务条款文档采集接入主流程：点击 policy/TOS 入口后，policy 页面进入 UTG、保存文档文本和证据、标记任务完成，但不再送交 LLM 做普通导航分析。

**Architecture:** 在 `WorkflowRunner` 内增加一个小的 policy capture 分支。普通动作执行后，如果该动作疑似打开 policy/TOS 文档，则设置 `policy_capture_pending`；下一轮 capture 仍正常生成 state、记录 transition、给节点打 `policy` tag，然后根据当前前台包判断外部浏览器或 APP 内文档，并用 Ctrl+A/C 复制正文。采集完成后保存到 run 目录、结束当前 `explore_policy` 任务、返回目标 APP。

**Tech Stack:** Python, Appium UiAutomator2, existing `AndroidAppiumClient`, existing `WorkflowRunner`, existing `StateGraph`, JSON trace artifacts.

---

## 0. 当前分支和范围

当前分支：

```powershell
git branch --show-current
# 0622_policy_capture_workflow
```

本计划只覆盖 policy/TOS 文档采集接入主流程。暂不处理：

- URL 下载网页全文；
- OCR fallback；
- 滚动采集全文；
- policy 文本自动填问卷；
- 多 policy 入口拆成多个 proposed tasks 的 prompt 修改；
- 多 policy 入口的完整调度策略改造。

本阶段第一版规则：

- 外部浏览器页面：保存浏览器地址栏原始 URL，并执行 Ctrl+A / Ctrl+C 获取正文文本。
- APP 内文档页面：执行 Ctrl+A / Ctrl+C 获取正文文本。
- policy 页面进入 UTG，并打 `policy` tag。
- policy 页面不调用 navigation/router LLM。
- 成功采集后，当前 `explore_policy` 任务标记为完成。
- 失败时保存失败记录，清理 pending，并执行返回 / recovery 逻辑。

---

## 1. 变量和数据结构约定

新增或使用这些运行时字段：

- `policy_capture_pending`：布尔值，表示上一步动作可能打开了隐私政策 / 服务条款页面，下一轮 capture 后优先走 policy capture 分支。
- `policy_capture_context`：字典，保存本次 pending 的上下文。
- `policy_capture_source_task_id`：触发 policy capture 的任务 ID，例如 `task_0007`。
- `policy_capture_source_state_sig`：触发动作前所在 UI 的状态签名。
- `policy_capture_source_action`：触发动作的简化描述，例如 `click Privacy Policy`。
- `policy_capture_document_title`：文档标题，优先取上一步动作按钮名，例如 `Privacy Policy`、`Terms of Service`、`User Service Agreement`。
- `policy_capture_source_candidate_key`：触发动作对应的候选动作 key，用来回溯动作缓存和报告。

采集结果字段：

- `capture_location`：采集位置，`external_browser` 或 `in_app_document`。
- `url_raw`：外部浏览器地址栏原始文本，不补 `https://`。
- `text_char_count`：复制得到的文本字符数。
- `status`：采集状态，`success` 或 `failed`。
- `failure_reason`：失败原因。
- `output_dir`：本次文档采集目录。
- `document_text_path`：保存的文档文本路径。
- `screenshot_path`：保存的截图路径。

### 文档标题

文档保存名称以入口动作按钮文本为主，而不是只按 `privacy` / `terms` 粗分类。

优先级：

1. `source_action_label`，也就是上一步点击的按钮文本。

2. 复制文本前几行里的标题。

3. fallback 为 `Policy Document`。

本阶段不做 `privacy_policy` / `terms_of_service` / `policy_unknown` 这类粗分类，也不在 metadata 中保存 `document_kind`。原因是该信息对第一版抓取流程没有直接作用，且容易增加输出冗余。

---

## 2. 文件结构

### 不新增独立 `policy_capture.py`

本阶段不新增 `policy_capture.py`。原因：policy capture 是 workflow 的一个特殊分支，依赖 run 输出目录、任务状态、目标包、UTG 标记和 recovery 行为，直接放在 `WorkflowRunner` 内更直观。

新增 helper 方法放在 `workflow.py`：

- `_should_enter_policy_capture(...)`
- `_build_policy_capture_context(...)`
- `_safe_policy_document_title(...)`
- `_next_policy_capture_dir(...)`
- `_extract_browser_address_from_xml(...)`
- `_copy_all_policy_document_text(...)`
- `_capture_policy_document(...)`
- `_handle_policy_capture_pending(...)`
- `_return_from_policy_document(...)`

变量说明：

- `_safe_policy_document_title`：把文档标题转换成可用作文件夹 / 文件名的安全字符串。
- `_next_policy_capture_dir`：如果同名文档已存在，自动生成 `_2`、`_3` 目录。
- `_handle_policy_capture_pending`：主流程 capture 后、LLM 前的 policy 分支入口。

### 修改文件

- `workflow.py`
  - 增加 pending 状态字段。
  - 在执行疑似 policy 入口动作成功后设置 pending。
  - 在 capture 后、LLM 前拦截 policy capture。
  - 将 policy 页面加入 UTG 并打 `policy` tag。
  - 采集完成后结束当前 policy 任务并返回目标 APP。

- `trace_callbacks.py`
  - 如现有事件接口足够，可只通过 workflow 的 `_log_event(...)` 记录。
  - 如果 `debug_summary.md` 需要专门展示，再补充 policy capture 事件展示。

- `visualize_run_interactive.py`
  - 在 HTML 节点详情中展示 policy capture 结果。

- `task_report.py`
  - 在任务报告中展示 policy capture 结果路径、文档标题、文本长度。

- `test_debug/run_policy_document_capture.py`
  - 继续保留为手动实验脚本。
  - 不强制和主流程 helper 复用，避免为了测试脚本额外抽象主流程代码。

### 输出目录

新增 run 内目录：

```text
traces/<run_id>/
  policy_captures/
    User Service Agreement/
      metadata.json
      User Service Agreement.txt
      screenshot.png

    Privacy Policy/
      metadata.json
      Privacy Policy.txt
      screenshot.png

    User Service Agreement_2/
      metadata.json
      User Service Agreement_2.txt
      screenshot.png
```

命名规则：

- 保留空格。
- 将 Windows 文件名非法字符替换为 `_`：`< > : " / \ | ? *`。
- 去掉首尾空格和句点。
- 文档名过长时截断，例如 80 字符。
- 同名冲突时追加 `_2`、`_3`。
- 成功路径默认不保存 XML。
- 失败路径可以额外保存 `xml.xml`，用于排查。

---

## Task 1: 在 workflow.py 中增加 policy capture helper

**Files:**
- Modify: `F:\workplace\framework\workflow.py`

- [ ] **Step 1: 增加常量和字段**

在 `WorkflowRunner` 初始化中增加：

```python
self.policy_capture_pending = False
self.policy_capture_context = {}
```

增加常量：

```python
POLICY_CAPTURE_MIN_TEXT_CHARS = 500
POLICY_CAPTURE_BODY_X_RATIO = 0.50
POLICY_CAPTURE_BODY_Y_RATIO = 0.45
POLICY_CAPTURE_RETURN_MAX_BACKS = 3
POLICY_CAPTURE_RETURN_WAIT_S = 2.0
```

变量说明：

- `POLICY_CAPTURE_MIN_TEXT_CHARS`：policy 文本最少字符数，小于该值认为没有可靠拿到正文。
- `POLICY_CAPTURE_BODY_X_RATIO` / `POLICY_CAPTURE_BODY_Y_RATIO`：复制前点击正文区域的位置。
- `POLICY_CAPTURE_RETURN_MAX_BACKS`：从 APP 内文档页返回时最多按几次 back。
- `POLICY_CAPTURE_RETURN_WAIT_S`：每次 back 后等待多少秒。

- [ ] **Step 2: 增加 `_should_enter_policy_capture(...)`**

```python
def _should_enter_policy_capture(self, task_type: str = "") -> bool:
    """
    Input: current task type.
    Output: True when the current task is explore_policy.
    Function: decides whether the next captured page should enter policy_capture branch.
    """
```

触发规则：

- 当前任务类型必须是 `explore_policy`。
- 不额外使用关键词命中规则判断动作是否是 policy 入口。

注意：本阶段不改 prompt。LLM 是否把多个 policy 入口拆成多个任务，后续再处理。采集是否成功只看文本长度，不用关键词验证。

- [ ] **Step 3: 增加 `_safe_policy_document_title(...)`**

```python
def _safe_policy_document_title(self, title: str) -> str:
    """
    Input: raw document title from source action or copied text.
    Output: Windows-safe file/directory title.
    Function: preserves readable document names while replacing invalid path characters.
    """
```

规则：

- 空标题 fallback 为 `Policy Document`。
- 替换非法字符为 `_`。
- 连续空白压成单空格。
- 截断到 80 字符。

- [ ] **Step 4: 增加 `_next_policy_capture_dir(...)`**

```python
def _next_policy_capture_dir(self, base_title: str) -> tuple[Path, str]:
    """
    Input: desired policy document title.
    Output: unique output directory and final title.
    Function: creates a policy_captures/<title> directory, adding _2/_3 on conflicts.
    """
```

输出示例：

```text
policy_captures/User Service Agreement/User Service Agreement.txt
policy_captures/User Service Agreement_2/User Service Agreement_2.txt
```

- [ ] **Step 5: 增加 `_extract_browser_address_from_xml(...)`**

```python
def _extract_browser_address_from_xml(self, xml_text: str) -> str:
    """
    Input: UiAutomator2 XML from an external browser page.
    Output: raw address bar text, without adding scheme.
    Function: extracts Chromium/browser URL field such as com.android.chromium:id/url_bar.
    """
```

不补 `https://`，只保存原始值。

- [ ] **Step 6: 增加 `_copy_all_policy_document_text(...)`**

```python
def _copy_all_policy_document_text(self, out_dir: Path) -> dict:
    """
    Input: output directory for evidence files.
    Output: metadata containing clipboard text path and char count.
    Function: clears clipboard, taps document body, sends Ctrl+A/C, reads clipboard, and saves document text.
    """
```

流程：

```text
clear clipboard
save screenshot.png
tap body area
send Ctrl+A
send Ctrl+C
read clipboard
save <document_title>.txt later in caller
```

注意：复制前必须清空剪贴板，避免读取旧内容。

---

## Task 2: 设置 pending，并在下一轮 capture 后处理 policy 页面

**Files:**
- Modify: `F:\workplace\framework\workflow.py`

- [ ] **Step 1: 在动作执行成功后设置 pending**

在候选动作执行成功、准备进入下一轮 capture 的位置，基于当前任务和动作文本判断：

```python
if self._should_enter_policy_capture(current_task_type):
    self.policy_capture_pending = True
    self.policy_capture_context = {...}
```

`policy_capture_context` 至少包含：

```python
{
    "source_task_id": task_id,
    "source_state_sig": src_sig,
    "source_action": action_label,
    "document_title": safe_title_from_action,
    "source_candidate_key": candidate_key,
}
```

- [ ] **Step 2: capture 后、LLM 前检查 pending**

主流程中找到：

```text
capture current page
-> compute state_sig
-> record transition/observation
-> normal LLM navigation/router
```

在 normal LLM 前插入：

```python
if self.policy_capture_pending:
    handled = self._handle_policy_capture_pending(sig, snap)
    if handled:
        continue_or_return_to_loop
```

要求：policy 页面必须已经进入 UTG 后再处理。

- [ ] **Step 3: policy 页面打 tag**

在 `_handle_policy_capture_pending(...)` 内：

```python
self._graph_annotate(
    sig,
    page_tags=["policy"],
    meta={"policy_capture_pending": True, ...},
)
```

policy 页面不送 LLM，不生成普通 candidate actions。

本阶段 `policy` tag 只由 policy capture 分支写入，表示该节点是实际抓取过 policy/TOS 文档的页面。普通 LLM 页面分析返回的 `policy` tag 不写入 UTG；如果 LLM 仍返回该 tag，写图前过滤掉。

---

## Task 3: 采集并保存 policy 文档

**Files:**
- Modify: `F:\workplace\framework\workflow.py`

- [ ] **Step 1: 判断外部浏览器还是 APP 内文档**

```python
foreground_package = self.appium.foreground_package()
capture_location = (
    "external_browser"
    if self.target_package and foreground_package != self.target_package
    else "in_app_document"
)
```

变量说明：

- `foreground_package`：当前前台包名。
- `target_package`：被测 APP 包名。
- `capture_location`：policy 文档页面位置。

- [ ] **Step 2: 外部浏览器保存 URL**

如果 `capture_location == "external_browser"`：

```python
url_raw = self._extract_browser_address_from_xml(xml_text)
```

只保存 `url_raw`，不下载。

- [ ] **Step 3: 两种页面都执行 Ctrl+A/C**

调用：

```python
copy_result = self._copy_all_policy_document_text(out_dir)
```

生成：

```text
<document_title>.txt
screenshot.png
metadata.json
```

- [ ] **Step 4: 保存 metadata**

metadata 示例：

```json
{
  "status": "success",
  "document_title": "Terms of Service",
  "capture_location": "external_browser",
  "url_raw": "newplg.dev/terms-of-service",
  "text_char_count": 41749,
  "source_task_id": "task_0007",
  "source_state_sig": "xml:...",
  "source_action": "Terms of Service",
  "screenshot_path": "policy_captures/Terms of Service/screenshot.png",
  "document_text_path": "policy_captures/Terms of Service/Terms of Service.txt"
}
```

如果失败，metadata 写：

```json
{
  "status": "failed",
  "failure_reason": "clipboard text too short",
  "text_char_count": 0
}
```

失败时可以额外保存 `xml.xml`。

---

## Task 4: 完成任务、返回 APP、清理状态

**Files:**
- Modify: `F:\workplace\framework\workflow.py`

- [ ] **Step 1: 成功时完成任务**

成功条件：

```python
text_char_count >= POLICY_CAPTURE_MIN_TEXT_CHARS
```

成功后：

```python
self.task_manager.finish_current_task("done", "policy document captured", state_sig=sig)
```

- [ ] **Step 2: 失败时记录失败**

失败后：

```python
self.task_manager.finish_current_task("failed", "policy capture failed: <reason>", state_sig=sig)
```

- [ ] **Step 3: 清理 pending**

无论成功失败都执行：

```python
self.policy_capture_pending = False
self.policy_capture_context = {}
```

- [ ] **Step 4: 返回目标 APP**

外部浏览器页面：

```python
self.appium.ensure_foreground(self.target_package, self.target_activity)
```

APP 内文档页：

```text
最多 3 次：
  back
  wait 2s
  如果不在目标 APP：ensure_foreground(target_package)
  如果页面已变化或回到 source 附近：成功
3 次仍失败：记录 policy_return_failed，进入已有 recovery/restart 流程
```

实现 helper：

```python
def _return_from_policy_document(self, source_state_sig: str, capture_location: str) -> dict:
    """
    Input: source state signature and policy capture location.
    Output: return result metadata.
    Function: returns from policy document page to target app with bounded back/foreground recovery attempts.
    """
```

---

## Task 5: trace、HTML、任务报告展示

**Files:**
- Modify: `F:\workplace\framework\workflow.py`
- Modify: `F:\workplace\framework\visualize_run_interactive.py`
- Modify: `F:\workplace\framework\task_report.py`

- [ ] **Step 1: trace 记录 policy capture 事件**

通过 `_log_event(...)` 记录：

```json
{
  "type": "policy_capture_done",
  "state_sig": "xml:...",
  "task_id": "task_0007",
  "document_title": "Privacy Policy",
  "capture_location": "external_browser",
  "url_raw": "example.com/privacy",
  "text_char_count": 41749,
  "status": "success",
  "output_dir": "policy_captures/Privacy Policy"
}
```

- [ ] **Step 2: StateGraph node meta 保存 capture 结果**

在 policy 节点 meta 中保存：

```python
{
    "policy_capture": metadata
}
```

HTML 可以直接从 graph node meta 读取。

- [ ] **Step 3: HTML 节点详情展示**

节点 detail 增加：

- `page_tags: policy`
- `policy_capture.status`
- `policy_capture.document_title`
- `policy_capture.capture_location`
- `policy_capture.url_raw`
- `policy_capture.text_char_count`
- `policy_capture.output_dir`

- [ ] **Step 4: 任务报告展示**

`task_report.md` 中对 `explore_policy` 任务展示：

```text
Policy capture:
- document_title: Privacy Policy
- location: external_browser
- url_raw: ...
- text_chars: ...
- output: ...
```

---

## Task 6: 测试和验证

**Files:**
- Modify or create tests only if pure helper functions can be tested without device.
- Output: `F:\workplace\framework\traces\<run_id>\policy_captures\...`

- [ ] **Step 1: 纯函数测试**

如果 `_safe_policy_document_title` 等函数容易单测，可以增加轻量测试。否则先用真实设备验证。

建议测试点：

- `Privacy Policy` 文件名保持可读。
- `User Service Agreement` 保留空格。
- `A/B:C*D?` 替换非法字符。
- 同名目录生成 `_2`。

- [ ] **Step 2: 手动短跑验证**

选择已知包含 policy/TOS 的 APP，运行 5 分钟以内。

命令示例：

```powershell
F:\workplace\framework\.venv\Scripts\python.exe main.py run `
  --appium-url http://127.0.0.1:4723 `
  --device-name 127.0.0.1:7555 `
  --package com.bd.nproject `
  --questionnaire-dir questionnaire-UI\games `
  --time-budget 300 `
  --trace-dir traces `
  --run-id 20260622_policy_capture_probe `
  --workers 1 `
  --auto-visualize-interactive `
  --log-level INFO
```

- [ ] **Step 3: 验证输出目录**

检查：

```text
traces/20260622_policy_capture_probe/policy_captures/
```

期望：

```text
Privacy Policy/
  Privacy Policy.txt
  metadata.json
  screenshot.png

Terms of Service/
  Terms of Service.txt
  metadata.json
  screenshot.png
```

实际是否两个都有，取决于当前 LLM 是否创建并执行了两个 policy 任务。本阶段不修改 prompt，所以可能只采到一个。

- [ ] **Step 4: 验证 HTML**

打开：

```text
traces/20260622_policy_capture_probe/index.html
```

检查：

- policy 页面节点存在；
- 节点带 `policy` tag；
- 节点详情显示 capture 结果；
- policy 页面没有普通 LLM candidate actions。

- [ ] **Step 5: 验证任务状态**

检查：

```text
traces/20260622_policy_capture_probe/tasks.json
traces/20260622_policy_capture_probe/task_report.md
```

期望：

- 对应 `explore_policy` 任务为 `done`；
- 报告中能看到文档标题、文本长度、输出目录。

---

## 风险和待确认点

1. **多个 policy 入口只执行一个的问题本阶段不处理。**
   - 本计划处理“点击 policy 入口后如何采集文档”。
   - 如果 LLM 没有把 `Privacy Policy` 和 `Terms of Service` 拆成两个 `proposed_tasks`，仍可能只采集一个。
   - 后续再单独改 prompt 和任务拆分规则。

2. **返回原入口页存在工程风险，但有明确处理策略。**
   - 风险不是理论上不能返回，而是返回后 APP 可能刷新、activity 重建、WebView 内部历史先变化，导致不一定回到原入口页。
   - 第一版用最多 3 次 back + 每次等待 2 秒 + 前台检查。
   - 失败时记录 `policy_return_failed`，再进入现有 recovery/restart。

3. **Ctrl+A/C 依赖页面焦点。**
   - 已在外部浏览器和 APP 内 WebView 实测成功。
   - 接入主流程后必须先点击正文区域，再发送 Ctrl+A/C。

4. **外部 URL 暂时只记录，不下载。**
   - 主流程不做 URL 下载，避免运行中网络耗时。
   - 后续只考虑对“文本获取失败且拿到了 url_raw”的文档做离线下载。

5. **policy 页面不送 LLM。**
   - 这是有意设计。
   - 成功采集文本后该页面只作为证据节点进入 UTG，不生成探索动作。
   - 后续可以在 APP 探索完成后统一做 policy 文档 LLM 分析，用于问卷填写。

---

## Commit 建议

实现完成并通过测试后提交：

```powershell
git add workflow.py visualize_run_interactive.py task_report.py test_debug/run_policy_document_capture.py docs/superpowers/plans/2026-06-22-policy-capture-workflow-plan.md
git commit -m "feat: capture policy documents in workflow"
```
