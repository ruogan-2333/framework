1. **最终按这个方向改**

   1. 完全关闭 probe-return。
   2. 候选动作按 LLM 输出顺序执行。
   3. 每执行一个候选动作，就立刻抓新 snap，判断落点。
   4. 如果进入新 UI，就 DFS 入栈。
   5. 如果当前页面候选动作都执行完，就执行 page_return_actions 退栈。
   6. page_return_actions 失败后允许 fallback 系统 BACK。
   7. 系统 BACK 后立刻检查前台包，不对就拉回目标 APP。
   8. 外部浏览器/外部页面只记录，不深入探索。
   9. state_return_actions 用 family_id 存。

   1. 第一版加深度限制字段，但默认设大一点，基本不限制探索。

   **你批注里问的几个点**
   DFS commit 是什么意思：

   就是“不再试探性点击再回来评估”，而是直接把 LLM 给出的候选动作当成 DFS 分支执行。

   也就是说原来是：

   ```
   点候选动作 -> 看看去哪 -> 尝试返回原页面 -> 再决定正式走哪个
   ```

   现在改成：

   ```
   取一个候选动作 -> 直接执行 -> 去哪就在哪继续探索
   ```

   所以“commit”就是“正式执行这个分支”。

   (没问题)

   **候选动作状态**
   你理解得对，probe_outcomes / probe_novelty 这套基本就是旧 probe-return 逻辑里的东西。新流程里可以先不用。

   保留这几个就够：

   ```
   state_candidates attempted_actions explored_actions
   ```

   第一版语义简单处理：

   - 动作执行过，就记 attempted。

   - 动作分支已经处理完，就记 explored。

   - 失败、无变化、跳外部页面，都可以先记成 explored，不做复杂区分。

   - 后面如果你想看失败原因，再单独加 trace 字段，不必一开始就设计很细。

     (这里面三个有啥区别.attempted_actions和explored有啥区别啊??)

   **跳到祖先页面怎么处理**
   你批注里问得对。这里第一版就按你说的处理：

   ```
   执行候选动作 -> 发现 new_sig 在 dfs_stack 里 -> 记录状态图边：cur_sig -> ancestor_sig -> 标记这个候选 explored -> DFS 栈 reconcile 到 ancestor_sig -> 在 ancestor_sig 继续探索
   ```

   不再想办法回到刚才那个子页面。

   原因很简单：真实 UI 已经跳到祖先页了，主流程应该服从真实当前位置。刚才那个子页面如果还有未探索动作，以后可以通过状态图/replay 再回来，但第一版不强求立刻返回。

   (这个逻辑不好,如果回到祖先页面,应该按照UI图中动作再回到刚才探索的页面.因为一般这个过程只需要一个动作.然后继续探索之前探索的那个页面.然后可以把刚才执行了返回祖先页面的那个动作记录到返回动作里面,不要覆盖原有的返回动作,可以放在原有返回动作后面执行.你觉得这个策略如何??)

   **外部页面 meta 是什么**
   我说的外部 meta 指：

   ```
   {  "src_sig": "...",  "candidate_key": "...",  "foreground_package": "...",  "foreground_activity": "...",  "target_package": "...",  "action": {...},  "timestamp": ... }
   ```

   你说的截图和 XML 也可以加。第一版我建议记录这些：

   ```
   external_meta.json external_screenshot.png external_xml.xml
   ```

   如果 Appium 能抓到外部页面，就保存；抓不到就只保存 meta，不中断主流程。

   (你说的外部meta是咋获取的..用哪个函数啥的??)

   **主循环中文伪代码**
   按你的批注，我重写成中文版：

   ```
   循环开始：
   
   1. 检查是否超时、动作数超限。
      如果达到停止条件，结束。
   
   2. 回收已经完成的 LLM 任务。
   
   3. 检查当前前台包。
      如果不在目标 APP：
          记录外部/前台异常
          拉回目标 APP
          重新抓 snap(这里是不是要比对一下之前送交LLM分析的UI和拉回后的UI是否是同一个UI,不同再重新抓)
          修正 DFS 栈
          继续下一轮
   
   4. 检查页面是否自己变化了。
      如果页面 drift：
          记录状态变化
          更新当前 snap 和 cur_sig
          重新安排 LLM 分析
          继续下一轮
   
   5. 等待当前页面的 LLM navigation/router 结果。
   
   6. 如果当前页面有 dismiss overlay：
          执行 overlay_dismiss_actions
          重新抓 snap
          继续下一轮
          (这个dismiss的页面信息是不是目前不会保存到本地<以及UI图中??后续可以考虑想办法保存一下.这个不是主要任务.先不重点研究..)
   
   7. 保存当前页面的候选动作和 page_return_actions：
          state_candidates[family_id] = nav.candidate_actions
          state_return_actions[family_id] = nav.page_return_actions
   
   8. 从当前页面候选动作里找第一个未 explored 的动作。
   
   9. 如果找到了候选动作：
          执行动作
          抓执行后的 snap
   
          如果动作失败：
              记录失败
              标记该候选 explored
              重新抓当前页面(这里是不是要比对一下现在的UI和之前的UI是否是同一个UI,不同再重新抓)
              继续下一轮
   
          如果执行后还在同一个 state_sig：
              记录 cur -> cur
              标记该候选 explored
              继续当前页面
   
          如果执行后跳到外部页面：
              记录 external meta/xml/screenshot
              标记该候选 explored
              拉回目标 APP
              重新抓 snap(这里是不是要比对一下现在的UI和之前的UI是否是同一个UI,不同再重新抓)
              修正 DFS 栈
              继续下一轮
   
          如果执行后进入新的 APP 内页面：
              记录 cur -> new 的状态图边
              标记该候选 attempted
              new 入 DFS 栈
              安排 new 页面 LLM 分析
              当前页面切换到 new
              继续下一轮
   
          如果执行后回到祖先页面：
              记录 cur -> ancestor 的状态图边
              标记该候选 explored
              DFS 栈退到 ancestor
              当前页面切换到 ancestor
              继续下一轮
              (这一部分我有点意见,看我上面的批注)
   
   10. 如果没有未 explored 的候选动作：
          当前页面视作探索完
   
          如果当前页面是根页面：
              可以结束，或者之后再做全局 frontier
              第一版先结束/等待停止条件
   
          如果当前页面不是根页面：
              优先执行 page_return_actions
              如果失败，就执行系统 BACK
              BACK 后检查前台包，不对就拉回目标 APP
              抓新 snap(这里是不是要比对一下现在的UI和之前的UI是否是同一个UI,不同再重新抓)
              修正 DFS 栈
              继续下一轮
   ```

   **最小改造范围**
   如果接下来开始改代码，我建议只做这几个点：

   1. 修 workflow.py 对新 gpt_cls.py schema 的兼容。
   2. 新增 state_return_actions。
   3. 新增 _next_unexplored_candidate(...)。
   4. 新增 _execute_candidate_branch(...)。
   5. 新增 _classify_after_candidate(...)。
   6. 新增 _handle_external_after_candidate(...)。
   7. 新增 _return_from_current_state(...)。
   8. 主循环绕开 _probe_candidates(...) 和 _choose_forward(...)。
   9. 旧函数先保留，不删。

   **我现在理解你的最终决定**

   - 关闭 probe-return。
   - 外部页面只记录并拉回 APP。
   - 候选动作按 LLM 输出顺序执行。
   - page_return_actions 失败后 fallback BACK。
   - BACK 后检查前台包，不对就拉回。
   - 暂时不严格限制 DFS 深度，字段可以加，但默认设大一点。
   - 不搞复杂 scoring。
   - failed/no_change 先简单标 explored，后面再细分。