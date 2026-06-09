# Task Quota And Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有任务系统加入第一版按 `task_type` 控制任务创建数量和任务步长预算的调度规则，并保留被拒绝任务记录用于后续分析。

**Architecture:** 第一版不引入 beam search、不做全局任务池、不做连续无 block 命中提前结束。调度规则集中放在 `task_manager.py` 的任务类型配置中，由 `TaskManager.push_child_task(...)` 在任务创建时统一执行；`workflow.py` 继续按现有 DFS 任务栈执行。

**Tech Stack:** Python, pytest, existing `TaskManager`, existing `Workflow`, JSON run artifacts.

---

## 1. 本次改动范围

本次只做第一版简单调度：

| task_type | max_created | step_budget |
|---|---:|---:|
| `enter_main_page` | 1 | 8 |
| `explore_main_function` | 8 | 5 |
| `explore_payment` | 3 | 3 |
| `explore_policy` | 3 | 3 |
| `explore_settings` | 3 | 3 |
| `generic` | 3 | 4 |

变量说明：

- `task_type`：任务类型，比如 `explore_payment` 表示探索支付相关页面。
- `max_created`：某一种任务类型在一次 APP 运行中最多创建多少个任务。
- `step_budget`：某一种任务类型的每个任务最多执行多少步。
- `ignored_proposed_tasks`：被拒绝创建的任务列表，用于调试为什么某个 LLM 提议没有变成真实任务。

本次不做：

- 连续无 `matched_block_ids` 的提前结束。
- 根据 `priority` 动态调整 `step_budget`。
- 根据 `starts_task_depth` / `exploration_depth` 动态调整 `step_budget`。
- home 页面总体任务规划。
- 外部隐私政策网页采集。
- 动态页面状态合并。

---

## 2. 文件结构

### 修改文件

- `F:/workplace/framework/task_manager.py`
  - 在 `TaskTypeSpec` 中增加 `max_created` 和 `step_budget`。
  - 在 `TASK_TYPE_SPECS` 中写入每类任务的数量上限和步长预算。
  - 在 `task_type_prompt_rows()` 中输出 `max_created` 和 `step_budget`，让 LLM 知道当前调度约束。
  - 在 `TaskManager.push_child_task(...)` 中执行任务数量上限检查。
  - 在 `TaskManager.push_child_task(...)` 中用 `task_type` 配置覆盖 LLM 的 `initial_steps`。
  - 在 `ignored_proposed_tasks` 中记录因为超出 `max_created` 被拒绝的任务。

- `F:/workplace/framework/test_debug/test_task_manager.py`
  - 增加任务类型配置测试。
  - 增加任务步长预算测试。
  - 增加任务数量上限测试。
  - 更新 `task_type_prompt_rows()` 的字段断言。

### 观察文件，不一定修改

- `F:/workplace/framework/workflow.py`
  - 当前 `workflow.py` 已经把 LLM proposed task 传入 `TaskManager.push_child_task(...)`。
  - 本次优先让 `TaskManager` 统一控制预算和上限，因此 `workflow.py` 理论上不需要改。
  - 如果测试发现 `workflow.py` 侧需要额外记录 ignored reason，再补最小改动。

- `F:/workplace/framework/task_router_brief_report.py`
  - 当前已经能统计 `matched_block_count` 和 `unique_matched_block_count`。
  - 本次调度改动后继续用它对比改前改后的任务数量和 block 命中。

---

## 3. 任务拆分

### Task 1: 扩展任务类型配置

**Files:**

- Modify: `F:/workplace/framework/task_manager.py`
- Test: `F:/workplace/framework/test_debug/test_task_manager.py`

- [ ] **Step 1: 先写失败测试，确认 `TaskTypeSpec` 暴露数量和步长字段**

在 `F:/workplace/framework/test_debug/test_task_manager.py` 中新增测试：

```python
def test_task_type_specs_include_quota_and_budget() -> None:
    """
    Input: predefined task type specs.
    Output: expected max_created and step_budget for each task type.
    Function: protects the first-version task scheduling quota table.
    """
    expected = {
        TaskType.ENTER_MAIN_PAGE: (1, 8),
        TaskType.EXPLORE_MAIN_FUNCTION: (8, 5),
        TaskType.EXPLORE_PAYMENT: (3, 3),
        TaskType.EXPLORE_POLICY: (3, 3),
        TaskType.EXPLORE_SETTINGS: (3, 3),
        TaskType.GENERIC: (3, 4),
    }

    for task_type, (max_created, step_budget) in expected.items():
        spec = TASK_TYPE_SPECS[task_type]
        assert spec.max_created == max_created
        assert spec.step_budget == step_budget
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_task_type_specs_include_quota_and_budget -q
```

Expected:

```text
FAILED
AttributeError: 'TaskTypeSpec' object has no attribute 'max_created'
```

- [ ] **Step 3: 修改 `TaskTypeSpec` 和 `TASK_TYPE_SPECS`**

在 `F:/workplace/framework/task_manager.py` 中把 `TaskTypeSpec` 改成：

```python
@dataclass(frozen=True)
class TaskTypeSpec:
    """
    Configuration for one predefined task type.

    Input:
    - Task type enum plus human-authored description, completion goal, and defaults.

    Output:
    - Prompt-facing task metadata and workflow scheduling defaults.

    Function:
    - Separates task taxonomy, task creation quota, and step budget from task-stack runtime state.
    """

    task_type: TaskType
    description: str
    completion_goal: str
    default_priority: float
    default_depth: str
    max_created: int
    step_budget: int
```

然后在 `TASK_TYPE_SPECS` 每个 `TaskTypeSpec(...)` 中补字段：

```python
TaskType.ENTER_MAIN_PAGE: TaskTypeSpec(
    task_type=TaskType.ENTER_MAIN_PAGE,
    description="...",
    completion_goal="...",
    default_priority=1.0,
    default_depth="normal",
    max_created=1,
    step_budget=8,
),
TaskType.EXPLORE_MAIN_FUNCTION: TaskTypeSpec(
    task_type=TaskType.EXPLORE_MAIN_FUNCTION,
    description="...",
    completion_goal="...",
    default_priority=0.75,
    default_depth="normal",
    max_created=8,
    step_budget=5,
),
TaskType.EXPLORE_PAYMENT: TaskTypeSpec(
    task_type=TaskType.EXPLORE_PAYMENT,
    description="...",
    completion_goal="...",
    default_priority=0.90,
    default_depth="normal",
    max_created=3,
    step_budget=3,
),
TaskType.EXPLORE_POLICY: TaskTypeSpec(
    task_type=TaskType.EXPLORE_POLICY,
    description="...",
    completion_goal="...",
    default_priority=0.70,
    default_depth="shallow",
    max_created=3,
    step_budget=3,
),
TaskType.EXPLORE_SETTINGS: TaskTypeSpec(
    task_type=TaskType.EXPLORE_SETTINGS,
    description="...",
    completion_goal="...",
    default_priority=0.80,
    default_depth="normal",
    max_created=3,
    step_budget=3,
),
TaskType.GENERIC: TaskTypeSpec(
    task_type=TaskType.GENERIC,
    description="...",
    completion_goal="...",
    default_priority=0.20,
    default_depth="shallow",
    max_created=3,
    step_budget=4,
),
```

注意：上面代码块里的 `description` 和 `completion_goal` 保留当前文件已有内容，不要重写成省略号。

- [ ] **Step 4: 运行测试，确认通过**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_task_type_specs_include_quota_and_budget -q
```

Expected:

```text
1 passed
```

---

### Task 2: 让 prompt-facing 任务类型表包含调度配置

**Files:**

- Modify: `F:/workplace/framework/task_manager.py`
- Test: `F:/workplace/framework/test_debug/test_task_manager.py`

- [ ] **Step 1: 更新现有 `task_type_prompt_rows` 测试**

把 `test_task_type_prompt_rows_are_minimal_and_json_safe` 里的字段集合从：

```python
assert set(row) == {
    "task_type",
    "description",
    "completion_goal",
    "default_priority",
    "default_depth",
}
```

改成：

```python
assert set(row) == {
    "task_type",
    "description",
    "completion_goal",
    "default_priority",
    "default_depth",
    "max_created",
    "step_budget",
}
assert isinstance(row["max_created"], int)
assert isinstance(row["step_budget"], int)
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_task_type_prompt_rows_are_minimal_and_json_safe -q
```

Expected:

```text
FAILED
```

失败原因应该是 `max_created` 和 `step_budget` 还没有输出。

- [ ] **Step 3: 修改 `task_type_prompt_rows()`**

在 `F:/workplace/framework/task_manager.py` 中修改 `task_type_prompt_rows()` 的 row：

```python
rows.append(
    {
        "task_type": spec.task_type.value,
        "description": spec.description,
        "completion_goal": spec.completion_goal,
        "default_priority": clamp_priority(spec.default_priority),
        "default_depth": normalize_depth(spec.default_depth),
        "max_created": max(0, int(spec.max_created)),
        "step_budget": max(1, int(spec.step_budget)),
    }
)
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_task_type_prompt_rows_are_minimal_and_json_safe -q
```

Expected:

```text
1 passed
```

---

### Task 3: 用任务类型配置覆盖 LLM initial_steps

**Files:**

- Modify: `F:/workplace/framework/task_manager.py`
- Test: `F:/workplace/framework/test_debug/test_task_manager.py`

- [ ] **Step 1: 写失败测试**

在 `F:/workplace/framework/test_debug/test_task_manager.py` 中新增测试：

```python
def test_child_task_step_budget_uses_task_type_spec() -> None:
    """
    Input: child task proposal with an exaggerated LLM initial step count.
    Output: created task uses the task-type configured step budget.
    Function: keeps runtime budgets controlled by local scheduling config instead of raw LLM output.
    """
    manager = TaskManager(default_steps=8)
    manager.ensure_initial_task("xml:root")

    child = manager.push_child_task(
        prompt="探索支付入口",
        task_type="explore_payment",
        initial_steps=20,
        entry_action={"action": "click", "element_id": 9},
        origin_state_sig="xml:root",
        reason="store entry is visible",
    )

    assert child is not None
    assert child.step_budget == 3
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_child_task_step_budget_uses_task_type_spec -q
```

Expected:

```text
FAILED
assert 20 == 3
```

- [ ] **Step 3: 修改 `push_child_task(...)` 的预算逻辑**

在 `F:/workplace/framework/task_manager.py` 的 `push_child_task(...)` 中，在创建 task 前增加：

```python
        normalized_task_type = normalize_task_type(task_type)
        spec = get_task_type_spec(normalized_task_type)
        configured_step_budget = max(1, int(spec.step_budget))
```

然后把 `_new_task(...)` 参数从：

```python
            task_type=normalize_task_type(task_type).value,
            step_budget=int(initial_steps if initial_steps is not None else self.default_steps),
```

改成：

```python
            task_type=normalized_task_type.value,
            step_budget=configured_step_budget,
```

说明：

- `initial_steps` 是 LLM 建议步长，本次第一版不再直接使用。
- `configured_step_budget` 是根据 `task_type` 本地配置得到的真实步长预算。

- [ ] **Step 4: 运行测试，确认通过**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_child_task_step_budget_uses_task_type_spec -q
```

Expected:

```text
1 passed
```

---

### Task 4: 按 task type 限制任务创建数量

**Files:**

- Modify: `F:/workplace/framework/task_manager.py`
- Test: `F:/workplace/framework/test_debug/test_task_manager.py`

- [ ] **Step 1: 写失败测试**

在 `F:/workplace/framework/test_debug/test_task_manager.py` 中新增测试：

```python
def test_child_task_creation_respects_max_created_per_type() -> None:
    """
    Input: four proposed explore_payment child tasks.
    Output: only three are created and the fourth is recorded as ignored.
    Function: verifies first-version task-type creation quota.
    """
    manager = TaskManager(default_steps=8)
    manager.ensure_initial_task("xml:root")

    created = []
    for idx in range(4):
        created.append(
            manager.push_child_task(
                prompt=f"探索支付入口 {idx}",
                task_type="explore_payment",
                entry_action={"action": "click", "element_id": idx + 1},
                origin_state_sig="xml:root",
                reason="payment entry is visible",
            )
        )

    assert [task is not None for task in created] == [True, True, True, False]
    assert len([task for task in manager.tasks_by_id.values() if task.task_type == "explore_payment"]) == 3
    assert manager.ignored_proposed_tasks[-1]["ignored_reason"] == "max_created_per_type_exceeded"
    assert manager.ignored_proposed_tasks[-1]["task_type"] == "explore_payment"
    assert manager.ignored_proposed_tasks[-1]["max_created"] == 3
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_child_task_creation_respects_max_created_per_type -q
```

Expected:

```text
FAILED
```

当前第 4 个任务仍会被创建。

- [ ] **Step 3: 在 `TaskManager` 中增加已创建任务计数函数**

在 `F:/workplace/framework/task_manager.py` 的 `TaskManager` 类中新增函数：

```python
    def created_count_by_type(self, task_type: str) -> int:
        """
        Input: task type string.
        Output: number of tasks already created for this normalized task type.
        Function: supports first-version per-task-type creation quotas.
        """
        normalized = normalize_task_type(task_type).value
        return sum(1 for task in self.tasks_by_id.values() if task.task_type == normalized)
```

- [ ] **Step 4: 在 `push_child_task(...)` 中增加数量上限检查**

在 `push_child_task(...)` 里完成 `normalized_task_type` 和 `spec` 后，创建 task 前加入：

```python
        created_count = self.created_count_by_type(normalized_task_type.value)
        max_created = max(0, int(spec.max_created))
        if max_created > 0 and created_count >= max_created:
            ignored = {
                "prompt": clean_prompt,
                "task_type": normalized_task_type.value,
                "exploration_depth": self._normalize_depth(exploration_depth),
                "entry_action": clean_action,
                "reason": reason,
                "origin_state_sig": origin_state_sig,
                "ignored_reason": "max_created_per_type_exceeded",
                "created_count": created_count,
                "max_created": max_created,
                "ts_ms": self._now_ms(),
            }
            self.ignored_proposed_tasks.append(ignored)
            return None
```

注意：这个检查要放在 `missing_prompt_or_entry_action` 检查之后。原因是无效任务应该先按“缺少 prompt 或 entry_action”记录，不应该算成 quota 拒绝。

- [ ] **Step 5: 运行测试，确认通过**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_child_task_creation_respects_max_created_per_type -q
```

Expected:

```text
1 passed
```

---

### Task 5: 更新快照和报告验证

**Files:**

- Modify: `F:/workplace/framework/test_debug/test_task_manager.py`

- [ ] **Step 1: 扩展 snapshot 测试，确认 ignored quota 会进入 `tasks.json` 数据结构**

在 `test_snapshot_is_json_safe()` 后新增：

```python
def test_snapshot_records_quota_ignored_tasks() -> None:
    """
    Input: task manager where one proposed task exceeds per-type quota.
    Output: snapshot contains the quota rejection entry.
    Function: ensures run-level tasks.json can explain why LLM proposals were dropped.
    """
    manager = TaskManager(default_steps=8)
    manager.ensure_initial_task("xml:root")

    for idx in range(4):
        manager.push_child_task(
            prompt=f"探索 policy 入口 {idx}",
            task_type="explore_policy",
            entry_action={"action": "click", "element_id": idx + 1},
            origin_state_sig="xml:root",
            reason="policy link is visible",
        )

    snapshot = manager.snapshot()

    assert len([task for task in snapshot["tasks"] if task["task_type"] == "explore_policy"]) == 3
    assert snapshot["ignored_proposed_tasks"][-1]["ignored_reason"] == "max_created_per_type_exceeded"
    assert snapshot["ignored_proposed_tasks"][-1]["task_type"] == "explore_policy"
```

- [ ] **Step 2: 运行测试，确认通过**

Run:

```powershell
python -m pytest F:\workplace\framework\test_debug\test_task_manager.py::test_snapshot_records_quota_ignored_tasks -q
```

Expected:

```text
1 passed
```

---

### Task 6: 跑完整单元测试

**Files:**

- Test: `F:/workplace/framework/test_debug/test_task_manager.py`
- Test: `F:/workplace/framework/test_debug/test_task_report.py`
- Test: `F:/workplace/framework/test_debug/test_page_kind_planning.py`
- Test: `F:/workplace/framework/test_debug/test_utg_context.py`

- [ ] **Step 1: 运行核心测试**

Run:

```powershell
python -m pytest `
  F:\workplace\framework\test_debug\test_task_manager.py `
  F:\workplace\framework\test_debug\test_task_report.py `
  F:\workplace\framework\test_debug\test_page_kind_planning.py `
  F:\workplace\framework\test_debug\test_utg_context.py `
  -q
```

Expected:

```text
passed
```

如果出现历史 warning，例如 Pydantic `min_items` 或 requests dependency warning，只记录，不作为本次失败。

- [ ] **Step 2: 运行语法检查**

Run:

```powershell
python -m py_compile `
  F:\workplace\framework\task_manager.py `
  F:\workplace\framework\workflow.py `
  F:\workplace\framework\task_router_brief_report.py
```

Expected:

```text
no output and exit code 0
```

---

### Task 7: 用一个 APP 做短时在线验证

**Files:**

- Read output: `F:/workplace/framework/traces/<new_run>/tasks.json`
- Read output: `F:/workplace/framework/traces/<new_run>/task_router_brief.md`
- Read output: `F:/workplace/framework/traces/<new_run>/index.html`

- [ ] **Step 1: 选择一个已验证可跑的 APP**

建议先用 Sudoku：

```text
easy.sudoku.puzzle.solver.free
```

原因：

- 之前已有多次运行记录。
- 任务结构相对可比较。
- 能快速观察 `explore_payment`、`explore_policy`、`explore_settings` 是否被限制。

- [ ] **Step 2: 跑 5 分钟验证**

具体命令以当前项目实际 batch 脚本参数为准。原则是：

```powershell
python F:\workplace\framework\test_debug\run_batch_apps.py `
  --packages easy.sudoku.puzzle.solver.free `
  --time-budget-s 300 `
  --workers 1 `
  --trace-root F:\workplace\framework\traces
```

Expected:

```text
生成一个新的 traces/<run_id> 目录
```

如果实际脚本参数不是 `--packages` 或 `--time-budget-s`，执行前先用：

```powershell
python F:\workplace\framework\test_debug\run_batch_apps.py --help
```

确认参数。

- [ ] **Step 3: 生成任务 block 简报**

Run:

```powershell
python F:\workplace\framework\task_analysis_report.py --run-dir F:\workplace\framework\traces\<new_run>
python F:\workplace\framework\task_router_brief_report.py --run-dir F:\workplace\framework\traces\<new_run>
```

Expected:

```text
F:\workplace\framework\traces\<new_run>\task_analysis_report.md
F:\workplace\framework\traces\<new_run>\task_router_brief.md
```

- [ ] **Step 4: 人工检查调度结果**

检查 `F:/workplace/framework/traces/<new_run>/tasks.json`：

```text
explore_main_function <= 8
explore_payment <= 3
explore_policy <= 3
explore_settings <= 3
generic <= 3
```

检查 `ignored_proposed_tasks`：

```text
如果出现 ignored_reason=max_created_per_type_exceeded，需要确认被拒绝任务是否合理。
```

检查 `task_router_brief.md`：

```text
确认任务数量收敛后，matched_block_count 没有明显变差。
```

---

## 4. 后续改进思路：home 页面总体任务规划

这部分本次不实现，但需要记录为下一阶段方向。

当前简单 quota 的风险：

- 如果一个 APP 有 5 个主要功能入口，LLM 可能反复在前 2 个入口上创建任务。
- `max_created` 会把同类任务截断，可能导致后面 3 个入口没有探索机会。
- 单纯按 `task_type` 限制数量，不能保证功能覆盖面。

后续可考虑增加 home-level planner：

1. 当系统确认 `home` 页面后，触发一次总体任务规划。
2. 输入：
   - APP metadata，例如包名、应用名、类别、问卷类型。
   - home 页面截图和 UI digest。
   - 当前文本 UTG。
   - 任务类型定义表。
   - 问卷关注点摘要。
3. 输出：
   - 一组覆盖全面的高层任务。
   - 每个任务绑定入口动作或入口区域。
   - 每个任务包含 `task_type`、`priority`、`exploration_depth`。
4. 后续页面仍允许提出子任务，但需要和总体计划去重或合并。

这个方向的目标是：

```text
先保证覆盖 APP 主要入口，再在每个入口下做有限深入。
```

---

## 5. 验收标准

本次代码改动完成后，需要满足：

- `TaskTypeSpec` 中存在 `max_created` 和 `step_budget`。
- LLM 提出的 child task 创建时，真实 `step_budget` 来自任务类型配置，不再直接听 LLM `initial_steps`。
- 同一 `task_type` 超过 `max_created` 后，新任务不会创建。
- 被拒绝任务会写入 `ignored_proposed_tasks`，并包含：
  - `ignored_reason`
  - `task_type`
  - `created_count`
  - `max_created`
  - `origin_state_sig`
- 现有核心 pytest 通过。
- 至少一个 5 分钟在线运行可以生成：
  - `tasks.json`
  - `task_analysis_report.md`
  - `task_router_brief.md`
  - `index.html`
- 新运行中任务数量符合配置上限。

---

## 6. 执行前注意

- 当前工作区已有未提交改动。执行前先确认 `git status`，避免把报告脚本、调度改动、文档改动混在一起无法回退。
- 在线跑 APP 前确认：
  - 模拟器已启动。
  - Appium 已启动。
  - `.env` 中 LLM base URL 和 key 正确。
  - 代理设置正确。
- 如果运行中出现 401、Appium 连接失败、ADB 找不到设备、PaddleOCR 报错，需要先停止并报告，不要继续堆运行结果。

---

## 7. 执行方式建议

Plan complete and saved to `F:/workplace/framework/docs/superpowers/plans/2026-06-09-task-quota-budget-code-plan-zh.md`.

建议执行方式：

1. 先按 Task 1-6 做纯代码和单元测试。
2. 通过后再跑 Task 7 的一个 APP 在线验证。
3. 验证结果给用户看，确认后再写更新日志并提交。

