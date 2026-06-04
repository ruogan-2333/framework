# 0601 UTG 工作记录

## 1. 文档用途

本文记录 UTG 接入 LLM 相关工作的过程信息。

`0601-utg.md` 是计划文档，记录方案和后续实现路径。

本文是工作记录，记录：

1. 每一步做了什么小实验。
2. 实验结果是什么。
3. 讨论中确认了什么口径。
4. 哪些想法暂时不做，但后续可能有价值。

## 2. 当前阶段目标

当前阶段不是直接接入主流程，而是先验证：

```text
LLM 是否能利用 UTG 信息生成更合理的返回动作。
```

这里的返回动作不是普通探索动作，而是用于补充当前 UTG 中缺失的回退能力。

## 3. 已确认的基本思路

### 3.1 不做全图任意补边

最开始讨论过让 LLM 针对当前 UI 找出尽可能多的“当前 UI 到图中已有 UI”的虚边。

后来确认这个方案容易变复杂：

1. 可能产生大量低价值虚边。
2. 很多 UI 在视觉上相似，LLM 可能误判目标。
3. 主流程真正需要的是回退能力，不是任意节点之间的连接。

因此当前收敛为：

```text
每次 parent UI 通过 forward action 到达 child UI 后，
让 LLM 为 child UI 生成少量 return hints。
```

### 3.2 return hints 的目标优先级

返回目标按优先级考虑：

1. home。
2. parent。
3. nearest useful ancestor。
4. no_return_needed。

`no_return_needed` 用于 loading、splash、一次性启动页等单向流程。

### 3.3 return hints 是预测结果，不是真实 UTG 边

LLM 给出的 return hints 只是预测。

它们不能直接当成已验证 UTG 边。

只有主流程实际执行该动作，并确认到达目标 UI 后，才可以升级为真实 UTG 边。

## 4. 已实现的小实验

### 4.1 新增离线测试脚本

新增脚本：

```text
test_debug/run_utg_return_eval.py
```

脚本功能：

1. 读取已有 trace。
2. 读取 `graph/state_graph_snapshot.json`。
3. 为每个 UI 生成 home path。
4. 根据 parent UI 和 child UI 构造文本 UTG。
5. 把 parent screenshot、child screenshot、child UI digest 一起发给 LLM。
6. 让 LLM 输出 child UI 的 return hints。

### 4.2 使用的 trace

第一轮实验使用：

```text
F:\workplace\framework\traces\20260531_1845_sudoku_dismiss_fix_10min
```

HTML：

```text
file:///F:/workplace/framework/traces/20260531_1845_sudoku_dismiss_fix_10min/index.html
```

### 4.3 使用的测试 case

测试边：

```text
UI2 -> UI3
```

对应 state：

```text
parent UI:
  UI2
  xml:779861dd5ffe359163c6587da5124a46

child UI:
  UI3
  xml:55461b663fc7d4dfce6570bfecbecc08
```

已有 forward action：

```text
click Personal
```

这个 case 的意义：

1. UI2 是 home/tab 主页面。
2. UI3 是点击底部 Personal tab 后进入的页面。
3. 旧逻辑只看当前 UI 时，不稳定知道应该如何回到 UI2。
4. 给 LLM UTG 后，可以测试它是否知道应该点击 Home tab，而不是系统 back。

### 4.4 生成的输出目录

无 LLM 调用版本：

```text
F:\workplace\framework\test_debug\utg_eval\sudoku_return_hints_ui2_ui3
```

LLM 调用版本：

```text
F:\workplace\framework\test_debug\utg_eval\sudoku_return_hints_ui2_ui3_llm
```

主要输出文件：

```text
home_paths.md
home_paths.json
utg.txt
prompt.md
input.json
result.json
result.md
```

## 5. 实验结果

LLM 返回结果：

```text
parent_ui: UI2
child_ui: UI3
no_return_needed: false
reason: The child UI is a tab screen. Returning to the parent UI involves switching back to the original tab.
```

返回动作：

```text
target: UI2
target_type: parent
action_type: click
element_id: 17
action_summary: Click the Home tab to return to the Home screen.
confidence: 1.0
```

LLM 使用的证据：

```text
The forward action was a tab switch to 'Personal'.
The child UI has a 'Home' tab (element_id 17) which corresponds to the parent UI.
```

结论：

```text
压缩后的文本 UTG + parent screenshot + child screenshot + child UI digest，
可以让 LLM 在该 case 中生成合理的 return hint。
```

## 6. 过程中发现的问题

### 6.1 .env 加载问题

第一次运行 LLM 调用时出现：

```text
401 无效的令牌
```

原因：

```text
run_utg_return_eval.py 把项目目录传给 load_project_env，
而不是传 .env 文件路径。
```

修正：

```text
load_project_env(PROJECT_ROOT / ".env")
```

修正后 LLM 调用成功。

### 6.2 当前 forward action 仍然偏文本摘要

当前 prompt 中明确告诉 LLM：

```text
Forward action: click Personal
```

但还没有把 forward action 的完整结构重点展开给 LLM。

完整结构包括：

```text
action_type
element_id
anchor_label
anchor_class
anchor_frame
anchor_center
fingerprint
```

这个暂时不做，作为后续改进。

### 6.3 当前没有传 parent UI digest

当前输入包含：

1. 文本 UTG。
2. parent screenshot。
3. child screenshot。
4. child UI digest。

没有传：

```text
parent UI digest
```

因为 return hint 的实际点击发生在 child UI 上，第一版优先让 LLM 看 child UI digest。

后续如果要让 LLM 更准确理解 parent 上发生了什么，可以补 parent UI digest。

## 7. 后续改进方向

### 7.1 增强 forward action 输入

后续可以把 forward action 从文本摘要升级为完整结构：

```json
{
  "action_type": "click",
  "element_id": 11,
  "text": "Personal",
  "anchor_label": "Personal",
  "anchor_class": "android.view.ViewGroup",
  "anchor_frame": [1080, 2336, 360, 224],
  "anchor_center": [1260.0, 2448.0]
}
```

这样 LLM 可以更明确知道：

1. parent UI 上点了哪个元素。
2. 该元素位于什么区域。
3. 它是不是底部 tab。

### 7.2 增加 parent UI digest

后续可以把 parent UI digest 加入 prompt。

这样 LLM 能同时看到：

1. parent UI 的可操作元素。
2. child UI 的可操作元素。
3. forward action 在 parent UI 上对应哪个元素。
4. child UI 上哪个元素可能是反向动作。

### 7.3 扩展到更多 case

下一步可以让用户从 HTML 中挑选更多 UI 边进行测试。

建议类型：

1. tab 切换。
2. 设置子页面返回。
3. 弹窗关闭。
4. 支付页面返回。
5. loading/splash 到正常页面，预期 `no_return_needed=true`。

### 7.4 接入主流程

如果后续接入主流程，建议流程是：

1. 主流程产生新真实边：`parent -> child`。
2. 判断 child 是否需要 return hints。
3. 调用 UTG return hints LLM。
4. 把 hints 保存到 trace 和图结构中，但标记为未验证。
5. 真正需要回退时，优先走真实 UTG 边。
6. 真实边不可达时，尝试 return hints。
7. 执行成功后升级为真实边。
8. 执行失败后标记该 hint 失效。

## 8. 当前阶段结论

当前小实验说明：

```text
把压缩后的 UTG 放入 LLM 交互是可行的。
LLM 能在至少一个真实 trace case 中，结合 UTG 和截图生成更合理的返回动作。
```

但当前还只是离线验证。

还没有完成：

1. 多 case 验证。
2. contact sheet / hybrid 图像 UTG 输入验证。
3. 主流程接入。
4. return hints 的持久化和验证升级机制。
## 9. 主流程接入与在线验证记录

日期：2026-06-04。

本节记录文本版 UTG 接入主流程后的代码改动、调试输出变化和 Sudoku 在线验证结果。

### 9.1 本阶段目标

本阶段目标是把离线实验确认可用的文本版 UTG 接入主流程，让 `propose_navigation_and_router(...)` 在生成 `page_return_actions` 时能够看到当前 UI 在已知 UI 转换图中的位置。

这里几个变量的含义如下：

1. `utg_context`：传给 LLM 的文本版 UI 转换图上下文。
2. `page_return_actions`：当前页面候选动作耗尽后，用于返回、关闭或切回已知页面的动作列表。
3. `page_return_status`：LLM 对当前页面是否需要返回、是否找到返回控件的状态判断。
4. `page_return_reason`：LLM 对 `page_return_status` 的自然语言解释。

### 9.2 代码层面改动

1. `state_graph.py`
   - 在现有 `StateGraph` 上增加文本 UTG 构造接口。
   - 新增 `TextUTGContext`，作为文本 UTG 构造结果的容器。
   - 支持从运行时图或 graph snapshot 生成压缩 UTG 文本。
   - 文本内容包含 `HOME`、`CURRENT`、`PARENT`、nodes、edges 和 home paths。
   - 节点文本优先使用 LLM 返回的 `page_summary`。

2. `workflow.py`
   - 在调度 `propose_navigation_and_router(...)` 前，为当前 UI 构造 `utg_context`。
   - 将 `utg_context` 写入每个 UI 的 `llm/utg_context.txt`。
   - 将 `utg_context` 放入 `navigation_router_input.json`。
   - 在 LLM 返回后把 `page_summary` 写回 StateGraph 节点元数据。
   - 增加 home UI 识别和 home path 导出。

3. `gpt_cls.py`
   - 更新 navigation/router prompt，要求 LLM 使用 `utg_context` 判断返回动作。
   - 继续复用现有 `page_return_actions`，不新增独立 `return_hints`。
   - 新增 `PageReturnStatus` 枚举。
   - `NavigationProposal` 新增 `page_return_status` 和 `page_return_reason`。

4. `trace_callbacks.py`
   - `debug_summary.md` 增加 UTG context 文件路径。
   - `debug_summary.md` 展示 `page_return_status` 和 `page_return_reason`。
   - `debug_summary.md` 展示 `page_return_actions` 明细。

5. `visualize_run_interactive.py`
   - HTML 节点详情增加 `UTG Context` 区块。
   - HTML 节点详情增加 `Page Return Actions` 表格。
   - HTML 节点详情展示 `page_return_status` 和 `page_return_reason`。
   - 当没有 `page_return_actions` 时，同时展示 status，方便区分“无需返回”和“需要返回但没找到控件”。

6. `test_debug/run_utg_return_eval.py`
   - 离线 UTG return eval 脚本改为复用 `state_graph.py` 的公共 UTG 构造接口。
   - 避免离线脚本和主流程各自维护一套 UTG 文本生成逻辑。

7. `test_debug/test_utg_context.py`
   - 新增 UTG context 单元测试。
   - 覆盖文本 UTG 的 nodes、edges、current、home path 等基础输出。

### 9.3 返回状态字段

新增 `page_return_status` 的原因是：仅看 `page_return_actions=[]` 无法区分不同情况。

现在状态分三类：

1. `has_visible_return`
   - 当前页面需要返回。
   - LLM 找到了可靠的可见返回控件。
   - 此时 `page_return_actions` 应该非空。

2. `no_return_needed`
   - 当前页面不需要返回。
   - 典型情况包括 home 页面、loading 页面、splash 页面、启动页。
   - 此时 `page_return_actions=[]` 是合理结果。

3. `needs_return_but_no_visible_control`
   - 当前页面理论上需要返回到 parent、ancestor 或 home。
   - 但 LLM 没有找到可靠的可见返回控件。
   - 此时 `page_return_actions=[]` 表示后续可能需要系统 back、fallback 或 recovery。

`page_return_reason` 用来解释为什么选择这个状态，方便人工检查。

### 9.4 legacy_unknown 说明

HTML 中可能出现：

```text
No page_return_actions. status=legacy_unknown
```

这个值不是 LLM 判断结果，而是 HTML 展示层的兼容兜底。

含义是：当前节点没有可用的 `page_return_status` 字段。

常见原因有两种：

1. 旧 trace 的 `navigation_router_result.json` 没有这个字段。
2. 当前 UI 刚被发现，LLM 请求已入队但运行预算结束，尚未生成 `navigation_router_result.json`。

后续如果需要更清楚，可以把展示层进一步区分为：

1. `not_analyzed`：没有 LLM 返回结果。
2. `legacy_unknown`：有 LLM 返回结果，但结果是旧 schema。

本阶段暂不改，因为当前主验证 trace 中新字段已经正常写入。

### 9.5 验证记录

基础检查：

```text
python -m py_compile gpt_cls.py state_graph.py workflow.py trace_callbacks.py visualize_run_interactive.py test_debug/run_utg_return_eval.py
python -m pytest test_debug/test_utg_context.py -q
```

结果：

```text
test_debug/test_utg_context.py: 3 passed
```

在线验证使用 Sudoku：

```text
package: easy.sudoku.puzzle.solver.free
trace: F:\workplace\framework\traces\20260604_sudoku_return_status_5min_retry
HTML: F:\workplace\framework\traces\20260604_sudoku_return_status_5min_retry\index.html
```

运行结果：

```text
time_budget_s = 300
elapsed_seconds = 303.184
stop_reason = time_budget_reached
LLM calls = 16
propose_navigation_and_router calls = 10
propose_blocks_fill calls = 6
```

`page_return_status` 分布：

```text
no_return_needed = 4
has_visible_return = 5
needs_return_but_no_visible_control = 1
```

这说明新字段已经能在真实主流程中生成，并进入 trace、debug summary 和 HTML。

### 9.6 当前结论

1. 文本版 UTG 可以接入主流程。
2. LLM 能基于文本 UTG 生成更合理的 `page_return_actions`。
3. `page_return_status` 可以解决 `page_return_actions=[]` 语义不清的问题。
4. HTML 现在能支持人工检查每个 UI 的 UTG context、返回动作和返回状态。

### 9.7 后续仍需关注

1. 当前 task 调度仍是任务栈 DFS，尚未使用更复杂的全局调度策略。
2. dismiss 逻辑仍然保留独立 overlay dismiss / recovery 流程，后续可以并入 task-aware navigation/router。
3. 如果后续遇到动作语义不足，再考虑增加 `semantic_action`。
4. 如果文本 UTG 不足以支撑复杂 case，再考虑 parent screenshot、parent UI digest 或 contact sheet。
5. 如果 HTML 中 `legacy_unknown` 影响阅读，再把 `not_analyzed` 和 `legacy_unknown` 分开展示。
