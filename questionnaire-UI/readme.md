# questionnaire-UI 修改记录

## games 问卷：新增执行用 router

`games` 问卷已经新增 7 个执行用 router，并写入
`games/questionnaire_routers.json`。

这些 router 不是原始问卷里的正式问题，只用于运行时判断某个 UI
是否需要进入对应 block 继续填写。后续导出最终问卷答案时，不应把这些
router 的答案当成正式问卷答案。当前通过下面字段标记：

```json
"export_answer": false
```

新增 router 与目标 block 对应关系：

| Router ID | 目标 block |
| --- | --- |
| `exec_games_crude_humor_visible` | `crude.crude_humor_content` |
| `exec_games_user_interaction_visible` | `misc.misc_user_interaction` |
| `exec_games_realistic_crime_visible` | `misc.realistic_crime_descriptions` |
| `exec_games_precise_location_visible` | `misc.share_precise_location` |
| `exec_games_nazi_symbols_visible` | `misc.nazi_symbols` |
| `exec_games_south_korea_identity_visible` | `misc.south_korea_national_identity` |
| `exec_games_terrorism_advocacy_visible` | `misc.advocate_terrorism` |

`games/questionnaire_blocks.json` 中上述 7 个 block 已经从无条件
`block_show_if: []` 改为对应 router 的 `yes` 条件，例如：

```json
"block_show_if": [
  {
    "id": "exec_games_crude_humor_visible",
    "option_id": "yes"
  }
]
```

修改后，`games/questionnaire_blocks.json` 中已经没有无条件 block。

## social_apps 问卷：新增执行用 router

`social_apps` 问卷原本没有 router。现在已经按“一个 block 对应一个
执行用 router”的方式新增 8 个 router。每个 router 的问题都是对其目标
block 内部问题的概括。

新增 router 与目标 block 对应关系：

| Router ID | 目标 block |
| --- | --- |
| `exec_social_app_type_visible` | `social_app.app_type` |
| `exec_social_digital_purchase_visible` | `social_app.digital_purchase` |
| `exec_social_graphic_violence_visible` | `social_app.graphic_violence` |
| `exec_social_precise_location_visible` | `social_app.share_location` |
| `exec_social_block_users_visible` | `social_app.block_users` |
| `exec_social_report_users_visible` | `social_app.report_users` |
| `exec_social_chat_moderation_visible` | `social_app.chat_moderation` |
| `exec_social_invited_friends_visible` | `social_app.invited_friends` |

`social_apps/questionnaire_blocks.json` 中上述 8 个 block 已经从无条件
`block_show_if: []` 改为对应 router 的 `yes` 条件。

设计备注：

`social_app.app_type` 比普通 UI 功能 block 更偏 APP 整体判断。它涉及
APP 是 communication 还是 social、是否用于 dating 或 sexual relationship、
是否允许公开分享 nudity、nudity sharing 是否是 primary focus。这些内容
不一定能从单个 UI 截图中完整判断。当前先保留为一个执行用 router，
目标 block 为 `social_app.app_type`，后续需要结合真实 trace 再确认是否
拆分或调整。

## others 问卷：新增执行用 router

`others` 问卷已经新增 7 个执行用 router。每个 router 对应一个原本无条件
命中的 block。

新增 router 与目标 block 对应关系：

| Router ID | 目标 block |
| --- | --- |
| `exec_others_crude_humor_visible` | `down.crude_humor_content` |
| `exec_others_digital_purchase_visible` | `miscell.misc_purchase_digital_goods` |
| `exec_others_precise_location_visible` | `miscell.misc_shares_location` |
| `exec_others_browser_search_visible` | `miscell.is_web_browser_search` |
| `exec_others_news_educational_visible` | `miscell.is_news_or_educational` |
| `exec_others_age_restricted_promotion_visible` | `promotion.promotion_age_restricted` |
| `exec_others_user_content_interaction_visible` | `user.user_content_interaction` |

`others/questionnaire_blocks.json` 中上述 7 个 block 已经从无条件
`block_show_if: []` 改为对应 router 的 `yes` 条件。

## addition 补充检测任务设计

`addition` 问卷用于存放原始评级问卷之外的补充检测任务。

当前设计方向：

- `addition` 只放运行时 UI 截图能够判断的检测任务。
- 文档、metadata、隐私政策、服务条款、Google Play 商店信息、APK 权限等
  不放进 `addition`，后续单独设计一套非 UI 分析任务。
- UI 类补充检测任务继续沿用 router/block 结构。
- router 负责判断当前 UI 是否存在相关证据。
- block 负责记录更具体的检测结果。
- 补充检测任务的 router 也使用 `"export_answer": false`。
- 最终补充检测结论应来自 block 的答案，而不是 router 的答案。

初步适合放入 `addition` 的 UI 截图检测方向：

| 检测方向 | 初步处理方式 |
| --- | --- |
| 年龄验证 | UI 截图检测，例如年龄门槛、生日输入、18+ 确认、家长同意 |
| 广告 | UI 截图检测，例如广告内容、广告标识、广告落地页 |
| 防沉迷时间限制 | UI 截图检测，例如时间限制、休息提醒、冷却时间、使用时长提示 |
| 成瘾性设计、推送等风险 | UI 截图检测，例如连续奖励、限时刺激、强引导留存、推送诱导 |
| 家长控制 | UI 截图检测，例如 parental controls、guardian controls、family settings |
| 儿童心理与生理健康风险 | UI 截图检测，例如自伤、极端节食、身体焦虑、危险挑战等内容 |
| 儿童剥削、网络接触风险 | UI 截图检测，例如陌生人聊天、私信、匹配、直播、送礼、线下接触诱导 |
| LGBT+ | UI 截图检测，判断是否存在相关可见内容或主题 |
| AI | UI 截图检测，例如 AI chat、AI companion、AI-generated text/image、自动生成内容 |

不适合直接放入 `addition` 的方向：

| 检测方向 | 建议处理方式 |
| --- | --- |
| 数据安全 | APK 权限、manifest、SDK 或静态分析 |
| 隐私政策、隐私风险 | 隐私政策文档分析 |
| 服务条款 TOS | 服务条款文档分析 |
| 儿童适用性 | metadata、商店信息、TOS/PP 与 UI 结果综合判断 |
| Google Play 层面的家长控制或年龄政策 | 商店信息或平台层面分析，不依赖 APP 截图 |

后续讨论重点：

逐个确定上述 UI 截图检测方向是否需要进入 `addition`，并为每个方向设计
具体的 router 问题和 block 问题。

当前草稿状态：

`addition/questionnaire_routers.json` 和 `addition/questionnaire_blocks.json`
已经写入第一版 UI 截图补充检测草稿。当前包含 8 个检测 block：

| Block ID | 检测方向 |
| --- | --- |
| `addition.age_verification` | 年龄验证 |
| `addition.unsafe_ad_content` | 广告中的儿童不适宜内容 |
| `addition.anti_addiction_notice` | 防沉迷、健康使用、时长限制或休息提醒 |
| `addition.parental_controls` | 家长控制或家庭安全设置 |
| `addition.child_health_risk` | 儿童心理或生理健康风险 |
| `addition.child_contact_risk` | 儿童网络接触或剥削风险 |
| `addition.lgbt_content` | LGBT+ 相关可见内容 |
| `addition.ai_features` | AI 功能或 AI 生成内容 |

每个 block 当前都有一个对应的执行用 router，且 `block_show_if` 均绑定到
对应 router 的 `yes` 条件。下一步需要人工检查每个 router 问题和 block
问题的具体措辞。

## block topic 字段规范

所有集合的 `questionnaire_blocks.json` 都已经补齐 `topic` 字段：

- `addition/questionnaire_blocks.json`
- `games/questionnaire_blocks.json`
- `others/questionnaire_blocks.json`
- `social_apps/questionnaire_blocks.json`

`topic` 字段用于描述 block 关注的主题，应使用英文自然语言短语，而不是
变量名形式。示例：

```json
"topic": "Precise location sharing"
```

不要写成：

```json
"topic": "precise_location_sharing"
```

当前校验结果：

- 所有 block 都有非空 `topic`。
- 所有 `topic` 都不包含下划线。
- 四个集合的 JSON 均可正常解析。
- 现有 `block_show_if` 引用的 router 均存在。

后续人工审查后，已经对部分语义不够清晰的 topic 做了进一步调整，例如：

- `South Korea national identity` 改为 `South Korea national identity harm`
- `App type and social purpose` 改为 `Social app type, dating, and nudity focus`
- `Child health risks` 改为 `Child psychological and physical health risks`
- `Child online contact risks` 改为 `Child stranger contact and exploitation risks`
- `Transferable digital assets` 改为 `Transferable digital assets and NFT marketplace`
