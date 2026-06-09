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

---

## 8. 本阶段执行结果

本阶段已经完成第一版任务数量和步长预算控制。

代码改动：

- `F:/workplace/framework/task_manager.py`
  - `TaskTypeSpec` 增加 `max_created`。
  - `TaskTypeSpec` 增加 `step_budget`。
  - `task_type_prompt_rows()` 输出 `max_created` 和 `step_budget`。
  - `TaskManager.created_count_by_type(...)` 用于统计某类任务已经创建了多少个。
  - `TaskManager.push_child_task(...)` 在创建任务前检查 `max_created`。
  - 超过数量上限的任务写入 `ignored_proposed_tasks`，不进入任务栈。
  - child task 的真实 `step_budget` 改为使用本地任务类型配置，不再直接使用 LLM 的 `initial_steps`。

- `F:/workplace/framework/test_debug/test_task_manager.py`
  - 增加任务类型 quota / budget 配置测试。
  - 增加 `task_type_prompt_rows()` 输出字段测试。
  - 增加按任务类型覆盖步长预算测试。
  - 增加同类型任务创建数量上限测试。
  - 增加 `ignored_proposed_tasks` 快照记录测试。

当前配置：

| task_type | max_created | step_budget |
|---|---:|---:|
| `enter_main_page` | 1 | 8 |
| `explore_main_function` | 8 | 5 |
| `explore_payment` | 3 | 3 |
| `explore_policy` | 3 | 3 |
| `explore_settings` | 3 | 3 |
| `generic` | 3 | 4 |

变量说明：

- `max_created`：某类任务在一次 APP 运行中最多创建多少个。
- `step_budget`：某类任务每个任务最多执行多少步。
- `created_count`：当前已经创建的同类型任务数量。
- `ignored_proposed_tasks`：被拒绝创建的任务记录列表。
- `initial_steps`：LLM 给出的建议步数；当前第一版不再直接作为真实任务步长。

---

## 9. 验证结果

本阶段使用项目虚拟环境执行测试：

```text
F:\workplace\framework\.venv\Scripts\python.exe
```

单元测试：

```text
test_debug/test_task_manager.py: 14 passed
```

核心测试集合：

```text
test_debug/test_task_manager.py
test_debug/test_task_report.py
test_debug/test_page_kind_planning.py
test_debug/test_utg_context.py

结果：25 passed
```

语法检查：

```text
task_manager.py
workflow.py
task_router_brief_report.py

结果：通过
```

非阻塞 warning：

- Pydantic `min_items` deprecated warning。
- requests dependency version warning。

在线验证：

- APP: `com.bd.nproject`
- run: `F:/workplace/framework/traces/20260609_180918_com.bd.nproject`
- HTML: `F:/workplace/framework/traces/20260609_180918_com.bd.nproject/index.html`
- `exit=0`
- `stop_reason=root_exhausted`
- 实际耗时约 `242s`
- LLM calls: `10`
- total tokens: `190956`

任务统计：

```text
总任务数: 4
ignored_proposed_tasks: 0

enter_main_page:
- count=1
- status=running 1
- step_budget=8
- used_steps=4

explore_main_function:
- count=2
- status=expired 2
- step_budget=5
- used_steps=5, 5

explore_settings:
- count=1
- status=expired 1
- step_budget=3
- used_steps=3
```

结论：

- `step_budget` 已确认按任务类型配置生效。
- 本次 run 没有触发 `max_created`，因为创建任务数量较少，所以 `ignored_proposed_tasks=0`。
- `max_created` 逻辑已由单元测试覆盖。

---

## 10. 当前保留问题

### 10.1 简单 quota 可能误杀有价值任务

当前第一版逻辑是：

```text
同类任务数量达到 max_created
    -> 新任务直接拒绝
    -> 记录到 ignored_proposed_tasks
```

这个逻辑可控，但比较粗糙。

风险：

- 如果一个 APP 有多个主要功能入口，LLM 可能先在前几个入口上创建同类任务。
- 后续更有价值的同类入口可能因为数量上限被拒绝。
- 单纯按 `task_type` 限制数量不能保证功能覆盖面。

### 10.2 同类型未执行任务替换策略

后续可以考虑替换策略。

思路：

```text
新任务超过 max_created
    -> 查找同 task_type 的未执行任务
    -> 如果旧任务 used_steps == 0 且 priority 更低
    -> 用新任务替换旧任务
```

变量说明：

- `used_steps`：任务已经消耗的步数。`used_steps == 0` 可以近似表示任务还没真正执行。
- `priority`：任务最终优先级，由 `type_priority` 和 `llm_priority` 合成。
- `type_priority`：任务类型默认重要性。
- `llm_priority`：LLM 对当前具体入口给出的局部重要性。

第一版没有实现该策略，原因：

- 需要改任务栈删除和替换逻辑。
- 需要定义被替换任务的状态，例如 `replaced`。
- 需要避免替换已经执行过、已有历史路径的任务。

### 10.3 home 页面总体任务规划

这是后续更重要的方向。

当前流程偏局部反应式：

```text
看到一个 UI
    -> LLM 分析当前 UI
    -> 提出当前可见子任务
    -> DFS 执行
```

后续希望加入 home-level planner：

```text
确认 home 页面
    -> 结合 app metadata、home 截图、UI digest、文本 UTG、问卷目标和人工探索模板
    -> 规划一组覆盖全面的高层任务
    -> 后续探索围绕这些任务执行
```

人工探索模板可以按 APP 类型整理，例如：

- 游戏类 APP：
  - 先确认主玩法入口。
  - 再看商店、订阅、虚拟货币、随机奖励。
  - 再看设置、隐私、家长控制。
  - 如果有社交、排行榜、聊天，再轻度探索。

- 社交类 APP：
  - 先确认内容流和发布入口。
  - 再看聊天、好友、关注、评论。
  - 再看账号、隐私、安全设置。
  - 再看付费会员或订阅。

- 内容/生活方式类 APP：
  - 先看首页推荐流。
  - 再看搜索、分类、内容详情页。
  - 再看发布、互动、账号页。
  - 再看设置、隐私、广告、订阅。

home-level planner 输入建议：

- `app_metadata`：包名、应用名、类别、问卷类型等。
- home 页面截图。
- home 页面 UI digest。
- 当前文本 UTG。
- 任务类型定义表。
- 问卷关注点摘要。
- 人工探索模板。

home-level planner 输出建议：

- 一组覆盖主要功能入口的高层任务。
- 每个任务包含：
  - `task_type`
  - `prompt`
  - `entry_action`
  - `priority`
  - `exploration_depth`
  - `reason`

后续页面仍可以提出新任务，但需要和总体计划去重或合并。

### 10.4 外部政策页面采集

隐私政策或服务条款可能打开外部浏览器或外部 WebView。

当前问题：

- 工作流检测到前台包名不是目标 APP 时，可能会把流程拉回 APP。
- 因此外部政策页内容可能无法采集。

后续方向：

- 对 `explore_policy` 类型任务增加特殊处理。
- 允许进入可信外部政策页面。
- 或者在进入外部页面前记录 URL / 页面文本。
- 后续可能需要单独实现政策页文本抽取，例如复制全文、读 WebView 文本或读取浏览器 URL。

### 10.5 动态页面状态膨胀

部分 APP 页面内容会持续变化。

当前问题：

- 动态内容可能导致多个相似但不同的 state。
- 任务步数可能被动态页面快速耗尽。

后续方向：

- 给动态页面增加更粗粒度 state 合并策略。
- 或者对动态 feed 页面减少深挖，只记录代表性证据。

### 10.6 多设备 force_stop warning

本次在线验证中出现非致命 warning：

```text
adbutils.errors.AdbError: more than one device/emulator, please specify the serial number
```

原因：

- 当前机器同时存在多个 ADB 设备。
- `force_stop` 使用 `adbutils.adb.device()` 时没有传 serial。

当前影响：

- Appium 主流程已完成。
- trace 和报告正常生成。
- 结束时 force_stop 失败被忽略。

后续方向：

- 在 `AndroidAppiumClient` 中保存 device serial。
- `force_stop` 使用指定 serial 获取 device。

### 10.7 run 结束时任务状态语义

当前问题：

- 因 run 结束未显式关闭的任务仍可能显示 `running`。
- 语义上它们更接近“运行结束时未完成”，不是还在真实执行。

后续方向：

- 区分 `pending`、`interrupted`、`running`。
- 在 run 结束时对未完成任务做一次终态整理。

