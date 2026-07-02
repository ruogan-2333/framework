# DFS 分支补齐当前能力代码修改计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 5 月 31 日任务调度之前的 DFS 基线分支更新到接近当前分支的基础设施能力，同时保留 DFS 动作选择与遍历策略，便于后续公平对比 DFS 策略和任务调度策略。

**Architecture:** 以 `debug_5_28` 作为 DFS 基线，不直接覆盖当前 `workflow.py`。迁移通用能力，包括 metadata、APK/XAPK 安装批测、UTG 文本上下文、page_kind/page_return_status、policy 文档抓取、dynamic state family 和 trace/html 输出；不迁移 TaskManager、任务栈、任务类型 quota、任务目标/history、task-bound action 等任务调度专属逻辑。

**Tech Stack:** Python, Appium, ADB, Pydantic schema, 本地 trace/UTG/HTML 调试系统。

---

## 0. 当前结论

- DFS 基线分支：`debug_5_28`。
- 任务调度分支起点：`debug_5_31`，其中 `e6bdd03 feat: add task stack guided exploration` 引入 Task Stack。
- 当前能力分支：`0701_dynamic_state_family`。
- 当前工作区不是干净状态，已有改动：
  - `.gitignore`
  - `gpt_cls.py`
  - `test_debug/bench_run.py`
  - `workflow.py`
  - `test_debug/test_drift_before_action.py`
  - `test_debug/test_transition_mismatch.py`
  - `test_debug/install_recheck_outputs/`
- 实施前必须先提交或暂存当前工作区，否则不能安全切分 DFS 更新分支。

## 1. 迁移边界

### 必须迁移到 DFS 分支的通用能力

- `main.py`
  - metadata CSV 读取。
  - 从 metadata 的 `app_type` 字段读取问卷类型。
  - `metadata_result.json` 输出。
  - APK/XAPK 安装参数和预安装流程。
  - `install-package` / `batch-install-smoke` 子命令。
  - 运行结束后自动 HTML 可视化。

- `device_utils.py`
  - APK 安装。
  - XAPK 解包、split APK 安装。
  - OBB 推送。
  - package 安装校验、卸载、启动辅助函数。

- `appium_android.py`
  - `force_stop(package)` 必须带 `device_name`。
  - Appium driver 创建时遇到 ADB daemon 异常后，自动 `adb kill-server` / `adb start-server` 并重试一次。

- `gpt_cls.py`
  - `AppMetadataSummary` 删除 `questionnaire_type`。
  - metadata prompt 使用当前 200/2000 数据集字段。
  - `NavigationProposal` 从旧 `overlay_kind` 改为 `page_kind` / `page_kind_reason`。
  - 增加 `page_tags`。
  - 增加 `page_return_status` / `page_return_reason`。
  - 增加 `ActionCandidate.action_intent`。
  - 增加 screenshot-first 规则，尤其是 `xml_reliable=False` 时优先相信截图。
  - 增加 loading 页面判断规则。
  - 保留 DFS 需要的 `candidate_actions` 和 `page_return_actions`。

- `state_graph.py`
  - 节点属性从 `overlay_kind` 更新为 `page_kind`。
  - 增加 `page_tags`。
  - 增加节点 `meta` 中的 `page_summary` 等摘要维护。
  - 增加 `build_text_utg_context(...)`，生成文本 UTG。
  - 增加 `from_snapshot(...)`，支持离线恢复图。
  - 保留 `shortest_action_path(...)` / replay 所需边结构。

- `workflow.py`
  - 保留 DFS 主循环。
  - 移除旧 overlay dismiss 专用分支，改为统一候选动作处理。
  - 接入 `page_kind`。
  - 接入 `page_return_status`。
  - 接入文本 UTG 输入到 LLM。
  - 接入 policy/TOS 文档抓取。
  - 接入 dynamic state family 判断。
  - 保留 DFS 的 `dfs_stack`、`dfs_via`、`state_candidates`、`state_tried_actions`、`page_return_actions` 回退逻辑。

- `trace_callbacks.py`
  - 输出 `page_kind`、`page_kind_reason`、`page_return_status`、`page_tags`。
  - 输出 `action_intent`。
  - 输出 policy capture 事件。
  - 输出 `utg_context.txt`。

- `visualize_run_interactive.py`
  - HTML 节点详情展示 `page_kind`、`page_tags`、`page_return_status`、candidate actions、return actions、policy capture。
  - 不展示 task stack 卡片。
  - 默认使用 LLM action overlay 或截图，减少冗余信息。

- `test_debug/bench_run.py`
  - 批量 runner。
  - 支持 `start-index` / `limit`。
  - 支持 metadata CSV。
  - 支持自动安装/卸载。
  - 默认不要传 `--validate-launch-before-run`。

### 不迁移到 DFS 分支的任务调度专属能力

- `task_manager.py` 不作为 DFS 主流程依赖。
- 不迁移 `TaskManager`。
- 不迁移 `TaskType` / `TaskTypeSpec` / task quota / task priority。
- 不迁移 `current_task`、`task_stack`、`task_decision`、`proposed_tasks` 到 DFS 的 LLM 输入输出。
- 不迁移 task goal / current_goal / history。
- 不迁移 task-bound action 绑定表。
- 不迁移 `task_report.py`、`task_analysis_report.py`、`task_router_brief_report.py` 作为 DFS 必需输出。

---

## 2. 推荐分支操作

- [ ] **Step 1: 保存当前工作区**

运行：

```powershell
git -C F:\workplace\framework status --short --branch
```

如果当前改动需要保留，先提交：

```powershell
git -C F:\workplace\framework add .gitignore gpt_cls.py workflow.py test_debug\bench_run.py test_debug\test_drift_before_action.py test_debug\test_transition_mismatch.py
git -C F:\workplace\framework commit -m "wip: save dynamic state family batch updates"
```

如果 `test_debug/install_recheck_outputs/` 是临时产物，不提交，加入 `.gitignore` 后删除或保留为未跟踪。

- [ ] **Step 2: 从 DFS 基线创建更新分支**

运行：

```powershell
git -C F:\workplace\framework checkout debug_5_28
git -C F:\workplace\framework checkout -b 0702_dfs_updated_workflow
```

- [ ] **Step 3: 建立对比清单**

运行：

```powershell
git -C F:\workplace\framework diff --name-status debug_5_28..0701_dynamic_state_family -- main.py workflow.py gpt_cls.py state_graph.py trace_callbacks.py visualize_run_interactive.py device_utils.py appium_android.py test_debug\bench_run.py
```

预期：看到这些核心文件存在大差异。

---

## 3. 主流程入口与批量能力迁移

### Task 1: 迁移 `appium_android.py` 的设备控制修复

**Files:**
- Modify: `F:\workplace\framework\appium_android.py`

- [ ] **Step 1: 从当前分支提取 ADB daemon 重试逻辑**

对比：

```powershell
git -C F:\workplace\framework diff debug_5_28..0701_dynamic_state_family -- appium_android.py
```

迁移内容：

- `_ADB_DAEMON_ERROR_MARKERS`
- `_looks_like_adb_daemon_failure(exc)`
- `_restart_adb_server()`
- `AndroidAppiumClient.init_connection(...)` 内的 ADB daemon retry。
- `AndroidAppiumClient.force_stop(...)` 使用 `self.device_name` 指定设备。

- [ ] **Step 2: 验证语法**

运行：

```powershell
.\.venv\Scripts\python.exe -m py_compile .\appium_android.py
```

预期：无输出，退出码为 0。

### Task 2: 迁移 `device_utils.py` 的安装/卸载工具

**Files:**
- Modify: `F:\workplace\framework\device_utils.py`

- [ ] **Step 1: 直接迁移当前分支的安装工具函数**

从当前分支迁移以下函数和数据结构：

- `CommandResult`
- `InstallResult`
- `run_adb_command(...)`
- `adb_success(...)`
- `install_apk_file(...)`
- `install_xapk_file(...)`
- `install_package_file(...)`
- `uninstall_package(...)`
- `verify_package_installed(...)`
- `launch_package(...)`
- `get_foreground_package(...)`
- `install_result_to_dict(...)`
- `write_install_reports(...)`

注意：保留 DFS 分支已有的其他工具函数；如果同名函数存在，用当前分支版本替换。

- [ ] **Step 2: 验证单个安装工具**

运行：

```powershell
.\.venv\Scripts\python.exe .\main.py install-package `
  --device-name 127.0.0.1:7555 `
  --app-file "F:\workplace\data_fetch_5_18\apkdownload_known200\APK\com.dramaton.slime.xapk" `
  --package com.dramaton.slime `
  --output-dir test_debug\install_smoke_dfs_updated
```

预期：

- 生成 `install_result.json`。
- 至少 `install_status=ok` 或明确输出安装失败原因。

### Task 3: 迁移 `main.py` 的 metadata、安装、批量入口能力

**Files:**
- Modify: `F:\workplace\framework\main.py`

- [ ] **Step 1: 迁移 metadata 字段选择逻辑**

迁移：

- `_META_SELECTED_FIELDS`
- `_row_first_value(...)`
- `_pick_metadata_fields(...)`
- `_read_questionnaire_type_from_metadata_row(...)`
- `_normalize_questionnaire_type(...)`
- `_write_metadata_result_json(...)`
- metadata CSV 读取后调用 `GPTClient.analyze_app_metadata(...)`
- 使用 CSV 的 `app_type` 决定问卷类型。

- [ ] **Step 2: 迁移安装相关 CLI 参数**

在 `run` 子命令中加入：

```text
--app-file
--install-before-run
--uninstall-before-install
--validate-launch-before-run
--uninstall-after-run
--install-timeout
--install-launch-wait-seconds
--install-temp-dir
--keep-install-temp
--adb-path
```

说明：`--validate-launch-before-run` 保留为可选功能，但 `bench_run.py` 默认不使用。

- [ ] **Step 3: 迁移安装预处理流程**

运行前：

- 如果 `--install-before-run`，调用 `install_package_file(...)`。
- 如果 `--uninstall-before-install`，先调用 `uninstall_package(...)`。
- 如果安装失败，写出 `install_result.json`。

注意：建议在 DFS 更新分支中顺手修复 cleanup 问题：只要传了 `--uninstall-after-run`，即使安装预检失败，也要尝试卸载。

- [ ] **Step 4: 迁移 `install-package` 和 `batch-install-smoke` 子命令**

从当前 `main.py` 迁移两个子命令，用于单独调试安装能力。

- [ ] **Step 5: 验证 metadata 单测脚本**

如果迁移 `test_debug/test_metadata_analysis.py`，运行：

```powershell
.\.venv\Scripts\python.exe .\test_debug\test_metadata_analysis.py `
  --package com.tocaboca.tocalifeworld `
  --metadata-csv "F:\workplace\framework\APP_csv\known_dataset_200_metadata.csv"
```

预期：输出 metadata LLM 结果，并写入测试输出目录。

---

## 4. LLM schema 与 prompt 迁移

### Task 4: 在 DFS 分支迁移通用 Navigation schema

**Files:**
- Modify: `F:\workplace\framework\gpt_cls.py`

- [ ] **Step 1: 保留 DFS 兼容字段，迁移通用字段**

从当前分支迁移：

- `PageKind`
- `PageTag`
- `PageReturnStatus`
- `ActionCandidate.action_intent`
- `NavigationProposal.page_kind`
- `NavigationProposal.page_kind_reason`
- `NavigationProposal.page_tags`
- `NavigationProposal.page_return_status`
- `NavigationProposal.page_return_reason`
- `AppMetadataSummary` 新结构。

不要迁移：

- `TaskDecision`
- `ProposedTask`
- `TaskUpdate`
- `NavigationRouterResult.task_id`
- `NavigationRouterResult.task_progress`
- `NavigationRouterResult.task_decision`
- `NavigationRouterResult.proposed_tasks`

- [ ] **Step 2: 调整 `NavigationRouterResult` 为 DFS 版**

DFS 版应保留：

```text
state_sig
navigation
router
```

可选保留：

```text
task_progress
```

不建议保留 task 字段，避免 DFS 对比时混入任务策略。

- [ ] **Step 3: 调整 prompt**

迁移通用 prompt 规则：

- screenshot-first。
- `xml_reliable=False` 时不要过度相信 UI tree。
- `page_kind=stable|popup|loading`。
- `page_kind` 只是页面描述，不触发单独 dismiss 分支。
- loading 页面优先 wait。
- return action 规则使用 `page_return_status`。
- page tags 规则。
- router 只在有证据时输出 matched block。

不要迁移：

- current_task。
- task_stack。
- proposed_tasks。
- action_role。
- starts_task_type。
- starts_task_depth。
- task type 列表。

- [ ] **Step 4: 验证 schema**

运行：

```powershell
.\.venv\Scripts\python.exe -m py_compile .\gpt_cls.py
```

预期：无输出，退出码为 0。

---

## 5. UTG / StateGraph 迁移

### Task 5: 迁移 `state_graph.py` 的文本 UTG 和页面元数据

**Files:**
- Modify: `F:\workplace\framework\state_graph.py`

- [ ] **Step 1: 迁移 Node 元数据**

将 DFS 基线中的 `overlay_kind` 替换或兼容为：

```text
page_kind
page_tags
meta
```

兼容策略：

- 旧 trace 中如果还有 `overlay_kind`，读取时转为 `page_kind`：
  - `dismiss` -> `popup`
  - `loading` -> `loading`
  - 其他 -> `stable`

- [ ] **Step 2: 迁移文本 UTG**

迁移：

- `TextUTGContext`
- `StateGraph.build_text_utg_context(...)`
- action label 生成函数。
- home path 生成逻辑。

- [ ] **Step 3: 验证 UTG 测试**

迁移 `test_debug/test_utg_context.py` 后运行：

```powershell
.\.venv\Scripts\python.exe .\test_debug\test_utg_context.py
```

预期：通过。

---

## 6. DFS `workflow.py` 主体迁移

### Task 6: 以 `debug_5_28` 的 DFS 主循环为底座

**Files:**
- Modify: `F:\workplace\framework\workflow.py`

- [ ] **Step 1: 明确保留 DFS 核心结构**

必须保留：

```text
dfs_stack
dfs_via
state_candidates
state_tried_actions
state_return_exhausted
_next_unexplored_candidate(...)
_execute_candidate_branch(...)
_return_from_exhausted_state(...)
_restart_and_replay(...)
_pick_global_frontier(...)
```

这些结构是 DFS 策略对比的核心。

- [ ] **Step 2: 移除旧 overlay dismiss 专用流程**

从当前分支迁移统一逻辑：

- 不再根据 `overlay_kind=dismiss` 进入独立 dismiss 分支。
- `popup` 只是 `page_kind`。
- close/back/cancel 等动作由 `candidate_actions` 或 `page_return_actions` 正常执行。

注意：如果直接删除旧函数风险大，可以第一版保留旧函数但不调用。

- [ ] **Step 3: 接入 `page_kind` 和 `page_return_status`**

替换：

```text
_overlay_kind_value(...)
_overlay_kind_for_sig(...)
overlay_kind graph annotation
```

为：

```text
_page_kind_value(...)
_page_kind_for_sig(...)
page_kind graph annotation
```

`page_return_status` 用于 HTML/debug 展示，不直接替代 DFS 回退逻辑。

- [ ] **Step 4: 接入文本 UTG 到 LLM**

在调用 `GPTClient.propose_navigation_and_router(...)` 前生成：

```text
utg_context = self.graph.build_text_utg_context(
    current_sig=cur_sig,
    parent_sig=parent_sig,
    home_sig=self.entry_sig 或 page_tags 中的 home
)
```

传入 LLM。

说明：DFS 版不传 `current_task` / `task_stack`。

- [ ] **Step 5: 接入 dynamic state family 判断**

迁移当前分支的两个 LLM 调用：

- pre-action drift compare。
- transition mismatch compare。

DFS 使用方式：

- 动作执行前如果发现当前 UI 与计划 UI 不一致，调用 pre-action drift compare。
- 沿 UTG/replay/return 后如果落到未知状态，调用 transition mismatch compare。
- 如果 LLM 判断同一 family，则复用旧状态。
- 如果不是同一 family，则作为新状态进入 DFS 正常分析。

- [ ] **Step 6: 接入 policy capture，但不依赖 task type**

当前 policy capture 依赖 `current_task.task_type == explore_policy`。DFS 版没有任务类型，因此触发条件改为：

```text
selected ActionCandidate 或 ActionStep 的 label/reasoning/action_intent 包含 policy/TOS 语义
```

触发词包括：

```text
privacy policy
privacy
terms of service
terms
user agreement
data policy
child privacy
```

流程：

1. 执行 policy 入口动作。
2. 下一轮 capture 生成 UI 节点并写入 UTG。
3. 不送 navigation/router。
4. 进入 policy capture。
5. 保存文档。
6. 尝试回到目标 APP。

- [ ] **Step 7: 不迁移任务栈事件**

不要调用：

```text
_emit_task_ui_observation(...)
_emit_task_action_selected(...)
task_manager.*
```

如果 HTML 需要 action intent，直接从 `ActionCandidate.action_intent` 展示即可。

- [ ] **Step 8: 验证语法**

运行：

```powershell
.\.venv\Scripts\python.exe -m py_compile .\workflow.py
```

预期：无输出，退出码为 0。

---

## 7. Trace / HTML 迁移

### Task 7: 更新 `trace_callbacks.py`

**Files:**
- Modify: `F:\workplace\framework\trace_callbacks.py`

- [ ] **Step 1: 展示 DFS 版 LLM 字段**

保留展示：

```text
page_kind
page_kind_reason
page_tags
page_return_status
page_return_reason
candidate_actions
action_intent
page_return_actions
utg_context.txt
policy_capture
```

删除或隐藏：

```text
current_task
task_stack
task_decision
proposed_tasks
task_progress
```

### Task 8: 更新 `visualize_run_interactive.py`

**Files:**
- Modify: `F:\workplace\framework\visualize_run_interactive.py`

- [ ] **Step 1: 迁移当前 HTML 的简洁版详情布局**

节点详情保留：

- 截图/LLM overlay。
- UI 文件夹路径复制。
- LLM result。
- candidate actions。
- page return actions。
- page kind / tags。
- policy capture。
- 可折叠 UTG context。

不展示：

- Task 状态卡片。
- Proposed Tasks 表格。
- VID map 摘要。
- raw JSON 大块内容。

- [ ] **Step 2: 生成 HTML 验证**

对一个旧 trace 或新 trace 运行：

```powershell
.\.venv\Scripts\python.exe .\visualize_run_interactive.py `
  --trace-dir "F:\workplace\framework\traces\<run_id>"
```

预期：生成 `index.html`。

---

## 8. 批量测试脚本迁移

### Task 9: 迁移 `test_debug/bench_run.py`

**Files:**
- Create or Modify: `F:\workplace\framework\test_debug\bench_run.py`

- [ ] **Step 1: 迁移当前 batch runner**

功能要求：

- 读取 `APP_csv/known_dataset_200.csv`。
- 读取 `APP_csv/known_dataset_200_metadata.csv`。
- 支持 `--start-index`。
- 支持 `--limit`。
- 支持 `--time-budget`。
- 支持 `--device-name`。
- 支持 `--batch-label`。
- 为每个 APP 传入：
  - `--app-file`
  - `--install-before-run`
  - `--uninstall-before-install`
  - `--uninstall-after-run`
  - `--metadata-csv-path`
  - `--questionnaire-type-source auto`
  - `--auto-visualize-interactive`

不要默认传：

```text
--validate-launch-before-run
```

- [ ] **Step 2: dry-run 验证命令**

运行：

```powershell
.\.venv\Scripts\python.exe .\test_debug\bench_run.py `
  --limit 1 `
  --start-index 3 `
  --time-budget 1 `
  --device-name 127.0.0.1:7555 `
  --batch-label dfs_dry_check `
  --dry-run
```

预期：生成的命令不包含 `--validate-launch-before-run`。

---

## 9. 测试计划

### Task 10: 静态测试

- [ ] **Step 1: 语法检查**

运行：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  .\appium_android.py `
  .\device_utils.py `
  .\gpt_cls.py `
  .\main.py `
  .\state_graph.py `
  .\workflow.py `
  .\trace_callbacks.py `
  .\visualize_run_interactive.py `
  .\test_debug\bench_run.py
```

预期：无输出，退出码为 0。

- [ ] **Step 2: 单元/离线测试**

运行：

```powershell
.\.venv\Scripts\python.exe .\test_debug\test_utg_context.py
.\.venv\Scripts\python.exe .\test_debug\test_metadata_analysis.py --package com.tocaboca.tocalifeworld
```

预期：UTG 测试通过；metadata 输出正常。

### Task 11: 单 APP 在线验证

- [ ] **Step 1: 跑 Toca Boca World**

运行：

```powershell
.\.venv\Scripts\python.exe .\main.py run `
  --package com.tocaboca.tocalifeworld `
  --app-file "F:\workplace\data_fetch_5_18\apkdownload_known200\APK\com.tocaboca.tocalifeworld.apk" `
  --install-before-run `
  --uninstall-before-install `
  --uninstall-after-run `
  --device-name 127.0.0.1:7555 `
  --trace-dir traces `
  --run-id dfs_updated_tocaboca_10min `
  --time-budget 600 `
  --max-actions 9999 `
  --workers 1 `
  --metadata-csv-path "F:\workplace\framework\APP_csv\known_dataset_200_metadata.csv" `
  --questionnaire-type-source auto `
  --questionnaire-dir "F:\workplace\framework\questionnaire-UI\games" `
  --disable-probe-return `
  --auto-visualize-interactive `
  --debug
```

观察：

- 是否进入 workflow。
- 是否生成 `states/`。
- 是否生成 `timeline.md`。
- 是否生成 `index.html`。
- DFS 是否能继续探索多个 UI。
- 动态页面是否减少重复状态。

### Task 12: 小批量验证

- [ ] **Step 1: 跑前 5 个 APP**

运行：

```powershell
.\.venv\Scripts\python.exe .\test_debug\bench_run.py `
  --limit 5 `
  --start-index 1 `
  --time-budget 600 `
  --device-name 127.0.0.1:7555 `
  --batch-label dfs_updated_known200_test5
```

验收：

- 不因 `validate-launch-before-run` 大面积失败。
- 每个成功进入 workflow 的 APP 都生成 HTML。
- 失败 APP 的日志能明确区分安装失败、ADB offline、Appium session 失败、workflow stop。

---

## 10. 阶段验收标准

本次 DFS 分支补齐完成的标准：

- DFS 主流程仍然是候选动作 DFS：
  - 一个 UI 中按候选动作逐个探索。
  - 分支耗尽后通过 `page_return_actions` / back / restart replay 回退。
  - 不引入 task stack。

- 基础设施与当前版本对齐：
  - metadata 可用。
  - APK/XAPK 自动安装卸载可用。
  - 文本 UTG 可生成并输入 LLM。
  - `page_kind` / `page_return_status` 可用。
  - dynamic state family 可用。
  - policy 文档抓取可用。
  - HTML 可读。
  - batch runner 可用。

- 能跑通至少一个 10 分钟单 APP 和一个 5 APP 小批量。

---

## 11. 风险与处理

- `workflow.py` 冲突最大，不建议 cherry-pick 整个文件。
  - 处理：以 `debug_5_28` 为底座，逐段迁移函数。

- `gpt_cls.py` 当前 schema 混入任务字段。
  - 处理：只迁移通用 schema，删除 task schema。

- policy capture 当前依赖 `explore_policy` 任务类型。
  - 处理：DFS 版改成基于 action label / intent / reasoning 的 policy 语义触发。

- HTML 当前展示 task 卡片。
  - 处理：DFS 版隐藏 task 部分，保留通用 LLM/action/UTG/policy 信息。

- 当前工作区不干净。
  - 处理：实施前必须提交或 stash，否则不能安全切换分支。

