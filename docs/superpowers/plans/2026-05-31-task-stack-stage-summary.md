# 2026-05-31 Task Stack 阶段归档

## 1. 阶段目标

1. 引入 task stack，让探索过程从单纯 DFS 改为任务指导下的 DFS。
2. 合并 navigation 和 router，让同一个 LLM 调用同时看到导航目标和问卷关注点。
3. 支持 LLM 在当前 UI 上创建子任务，并给出进入子任务的动作。
4. 区分候选动作是继续当前任务，还是启动新子任务。
5. 增加探索深度字段，用于提示 LLM 控制探索细致程度。
6. 调整任务步数字段，只保留 `step_budget` 和 `used_steps`。
7. 让 trace、debug summary、interactive HTML 能显示 task 相关信息，方便人工检查。

## 2. 已完成修改

1. 新增 `task_manager.py`。
2. 新增 `Task` 数据结构。
3. 新增 `TaskManager`，维护当前任务栈。
4. 初始任务为 `enter_main_page`，目标是进入稳定主界面。
5. `gpt_cls.py` 中扩展 `NavigationRouterResult`。
6. 新增 `TaskDecision`。
7. 新增 `ProposedTask`。
8. `ActionCandidate` 新增任务相关字段。
9. `workflow.py` 接入任务创建、任务完成、任务步数消耗和任务快照保存。
10. `trace_callbacks.py` 将 task context 写入 `trace.jsonl` 和 `debug_summary.md`。
11. `visualize_run_interactive.py` 展示当前任务、任务栈、候选动作角色、子任务类型和探索深度。
12. `test_debug/run_llm_on_fetch_snap.py` 适配单 UI replay，用于测试一个 snap 的 navigation/router/task 返回结果。
13. `test_debug/test_task_manager.py` 覆盖 task stack 的基础行为。
14. `main.py` 和 `test_debug/run_batch_apps.py` 默认 `workers=1`，减少并发导致的调试干扰。

## 3. 关键字段

1. `task_id`：任务编号，例如 `task_0001`。
2. `task_type`：任务类型，例如 `enter_main_page`、`explore_payment`、`explore_settings`。
3. `priority`：任务优先级，范围 0 到 1，用于后续全局调度。
4. `exploration_depth`：探索深度。
   - `shallow`：轻度探索，只确认功能类型和基本相关性。
   - `normal`：中度探索，允许进入关键入口。
   - `deep`：重度探索，主要用于问卷/router 强相关任务。
5. `step_budget`：任务总步数。
6. `used_steps`：任务已经使用的步数。
7. `action_role`：候选动作角色。
   - `continue_current_task`：继续当前任务。
   - `start_child_task`：启动子任务。
8. `starts_task_type`：`start_child_task` 动作会进入的子任务类型。
9. `starts_task_depth`：`start_child_task` 动作会进入的子任务探索深度。

## 4. 验证记录

1. `py_compile` 通过。
2. `test_debug/test_task_manager.py` 通过，结果为 5 passed。
3. 单 UI replay 脚本已适配 task context。
4. Sudoku 运行结果：
   - trace: `traces/20260531_sudoku_depth_steps_schemafix_10min`
   - HTML: `traces/20260531_sudoku_depth_steps_schemafix_10min/index.html`
   - stop_reason: `strong_stall_timeout`
5. 本次 Sudoku 运行没有再出现 `starts_task_depth.enum[0]: cannot be empty` schema 报错。
6. 新 trace 中 task context 已按 `step_budget / used_steps` 输出，不再主动输出 `remaining_steps`。

## 5. 已知问题

1. LLM 当前没有 UTG 上下文，所以不能稳定判断怎样返回父任务页面。
2. 因为缺少 UTG，LLM 给出的 `page_return_actions` 有时不足，系统会退化到 fallback back 或 restart。
3. fallback back 和 restart 增多后，容易出现 replay path 找不到目标状态的问题。
4. dismiss 逻辑还没有完全并入 task-aware navigation/router，目前仍有独立 dismiss/recovery 流程。
5. task stack 和返回路径恢复还没有真正联动。
6. 当前仍是任务栈 DFS，没有实现基于 `priority` 和 `exploration_depth` 的全局调度。
7. prompt 只做了轻量规则，后续还需要结合问卷类型细化任务创建和停止条件。

## 6. 下一阶段建议

1. 将压缩后的 UTG 传给 LLM。
2. 基于 UTG 改造返回动作策略，让 LLM 能看到当前 UI、父任务 UI 和已知路径。
3. 减少 fallback back 和 restart 的使用频率。
4. 将 dismiss 处理逐步并入 task-aware navigation/router。
5. 设计任务调度策略，后续可结合 `priority` 和 `exploration_depth` 做任务切换。
6. 根据问卷/router 问题整理任务类型和对应 prompt 规则。
