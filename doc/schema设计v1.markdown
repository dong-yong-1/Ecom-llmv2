# 《中文电商客服微调项目：schema 设计 v1》

## 1. 设计目标

`schema v1` 的目标，不是一次性覆盖所有电商客服场景，而是先为当前首期范围建立一套：

- 可标注
- 可校验
- 可扩展
- 可直接用于 SFT 数据构造

的数据结构。

本版本遵循前文已经确定的原则：

- 模型训练输入：`系统指令 + 上下文 + 对话历史`
- 模型训练输出：`当前轮客服回复`
- 数据底层保留辅助标注字段，便于规则一致性控制、质量检查和评测

---

## 2. schema 设计原则

### 2.1 训练字段与标注字段分离

训练时只需要模型学习“这一轮客服该怎么回复”。  
但在数据层，必须额外保存决策信息，否则后续难以控制一致性。

因此 `schema v1` 分成两层：

- `train_payload`：直接给模型训练使用
- `annotation_meta`：标注、质检、评测使用

### 2.2 先覆盖高频场景

本版本优先支持：

- 查询订单状态
- 修改地址
- 催发货
- 取消订单
- 查询物流进度
- 物流异常解释
- 退款流程说明
- 退货条件判断
- 转人工

### 2.3 不在 v1 里做过度设计

当前版本先不纳入：

- 特殊商品细分类规则
- 图片举证字段
- 多订单复杂关联结构
- 多语言字段
- 真正的系统动作执行结果

这些都可以在 `schema v2` 再扩展。

---

## 3. schema 总体结构

推荐一条样本的标准结构如下：

```json
{
  "id": "sample_000001",
  "version": "schema_v1",
  "train_payload": {
    "system": "...",
    "context": {},
    "history": [],
    "target": "..."
  },
  "annotation_meta": {
    "scenario": "...",
    "intent": "...",
    "decision_type": "...",
    "should_transfer": false,
    "missing_slots": [],
    "policy_basis": [],
    "risk_flags": [],
    "quality_check": {}
  }
}
```

这是 `schema v1` 的核心框架。

---

## 4. 顶层字段定义

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 样本唯一标识 |
| `version` | string | 是 | 当前 schema 版本，固定为 `schema_v1` |
| `train_payload` | object | 是 | 模型训练直接使用的数据 |
| `annotation_meta` | object | 是 | 标注辅助信息 |

### 4.1 `id`

建议规则：
- 全局唯一
- 稳定可追踪
- 不要包含业务隐私

推荐格式：

```text
sample_000001
```

或：

```text
cancel_order_000123
```

### 4.2 `version`

固定值：

```json
"version": "schema_v1"
```

后续升级时可改为：
- `schema_v2`
- `schema_v3`

---

## 5. train_payload 结构

```json
{
  "system": "string",
  "context": {},
  "history": [],
  "target": "string"
}
```

### 5.1 `system`

类型：`string`  
是否必填：是

推荐固定模板：

```text
你是电商客服助手，需要礼貌、专业、基于已知信息回答，不编造订单状态、物流进度、处理结果或承诺时间；信息不足时先追问，超出权限或高风险场景时转人工。
```

说明：
- 建议保持高度统一
- 不建议每条样本都改写 system
- system 负责统一边界，不负责承载具体业务事实

### 5.2 `context`

类型：`object`  
是否必填：是，但允许为空对象

说明：
- 用于存放本轮允许模型使用的显式事实
- 建议优先结构化字段
- 未知值要显式写 `未知`

### 5.3 `history`

类型：`array<object>`  
是否必填：是

说明：
- 表示本轮对话历史
- 最少包含当前用户输入
- 多轮场景下可包含之前的 assistant 回复

### 5.4 `target`

类型：`string`  
是否必填：是

说明：
- 当前轮客服标准回复
- 是模型训练目标
- 必须符合业务规则和承诺边界

---

## 6. context 字段定义 v1

`context` 推荐采用固定主字段 + 场景扩展字段的方式。

### 6.1 核心字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `order_status` | string | 否 | 订单状态 |
| `shipping_status` | string | 否 | 发货状态 |
| `logistics_status` | string | 否 | 物流状态 |
| `is_signed` | string | 否 | 是否签收 |
| `is_opened` | string | 否 | 是否拆封 |
| `has_quality_issue` | string | 否 | 是否存在质量问题 |
| `order_id_provided` | string | 否 | 是否已提供定位订单信息 |
| `user_emotion` | string | 否 | 用户情绪状态 |
| `user_request` | string | 否 | 当前主诉求 |

说明：
- `context` 顶层字段建议尽量稳定
- 某条样本不需要的字段可以不填
- 但一旦某字段对决策关键，建议明确填写具体值或 `未知`

### 6.2 枚举建议

#### `order_status`

推荐值：

- `未知`
- `待支付`
- `已支付`
- `已取消`
- `已完成`

说明：
- `order_status` 不等于发货状态
- 如果业务侧后续有更细状态，可在 v2 扩充

#### `shipping_status`

推荐值：

- `未知`
- `未发货`
- `已发货`

说明：
- 这个字段直接服务于“取消订单”“修改地址”“催发货”等判断
- 不建议和 `logistics_status` 混用

#### `logistics_status`

推荐值：

- `未知`
- `待揽收`
- `已揽收`
- `运输中`
- `派送中`
- `已签收`

#### `is_signed`

推荐值：

- `未知`
- `是`
- `否`

#### `is_opened`

推荐值：

- `未知`
- `是`
- `否`

#### `has_quality_issue`

推荐值：

- `未知`
- `是`
- `否`

#### `order_id_provided`

推荐值：

- `未知`
- `是`
- `否`

#### `user_emotion`

推荐值：

- `正常`
- `焦虑`
- `生气`
- `投诉倾向`

#### `user_request`

推荐值：

- `查询订单状态`
- `修改地址`
- `催发货`
- `取消订单`
- `查询物流`
- `物流异常`
- `退款`
- `退货退款`
- `售后`
- `转人工`

### 6.3 场景扩展字段

这些字段只在有需要的场景中出现：

| 字段 | 类型 | 说明 |
|---|---|---|
| `new_address_provided` | string | 是否已提供新地址 |
| `logistics_delay_days` | integer/string | 物流停滞天数，未知可填 `未知` |
| `after_sales_requested` | string | 是否已明确提出售后请求 |

推荐值：

#### `new_address_provided`
- `未知`
- `是`
- `否`

#### `after_sales_requested`
- `未知`
- `是`
- `否`

---

## 7. history 结构定义

`history` 是一个按时间顺序排列的数组。

单条消息结构如下：

```json
{
  "role": "user",
  "content": "怎么还没到？"
}
```

字段定义如下：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `role` | string | 是 | 说话方 |
| `content` | string | 是 | 发言内容 |

### 7.1 `role` 枚举

推荐值：
- `user`
- `assistant`

### 7.2 history 约束

- 至少包含 1 条消息
- 最后一条通常应为 `user`
- 多轮样本中允许出现若干条 `assistant` 历史回复
- 不建议放入与当前判断无关的长历史

---

## 8. target 定义

`target` 是当前轮客服标准回复。

要求如下：

- 必须礼貌、专业、清晰
- 必须基于输入中显式信息
- 不编造订单状态、物流进度、处理结果、承诺时间
- 信息不足时体现追问
- 超权限、高风险或投诉场景下体现转人工
- 不能假装已经执行系统操作

### 8.1 建议的 target 质量标准

一条合格的 `target` 最好同时满足：

1. 回答用户当前问题
2. 与上下文一致
3. 不越权
4. 有下一步建议
5. 语气符合用户情绪状态

---

## 9. annotation_meta 结构

```json
{
  "scenario": "取消订单",
  "intent": "取消订单",
  "decision_type": "direct_answer",
  "should_transfer": false,
  "missing_slots": [],
  "policy_basis": [],
  "risk_flags": [],
  "quality_check": {}
}
```

---

## 10. annotation_meta 字段定义

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `scenario` | string | 是 | 场景归类 |
| `intent` | string | 是 | 当前轮用户意图 |
| `decision_type` | string | 是 | 本轮处理策略 |
| `should_transfer` | boolean | 是 | 是否应转人工 |
| `missing_slots` | array<string> | 是 | 当前缺失的关键信息 |
| `policy_basis` | array<string> | 是 | 本轮依据的规则 |
| `risk_flags` | array<string> | 是 | 风险标记 |
| `quality_check` | object | 否 | 质检辅助字段 |

### 10.1 `scenario`

推荐值：

- `查询订单状态`
- `修改地址`
- `催发货`
- `取消订单`
- `查询物流进度`
- `物流异常`
- `退款流程说明`
- `退货条件判断`
- `转人工`

### 10.2 `intent`

`intent` 可以和 `scenario` 一致，也可以更细。

例如：
- `询问是否可取消`
- `要求立即发货`
- `询问退货条件`
- `投诉物流过慢`

建议：
- `scenario` 负责粗分类
- `intent` 负责当前轮更细粒度表达

### 10.3 `decision_type`

推荐枚举：

- `direct_answer`
- `ask_followup`
- `rule_explanation`
- `transfer_human`

说明：
- `direct_answer`：信息足够，直接处理
- `ask_followup`：关键信息不足，先追问
- `rule_explanation`：重点在解释规则边界
- `transfer_human`：应转人工

### 10.4 `should_transfer`

类型：`boolean`

取值：
- `true`
- `false`

说明：
- 这是规则判断结果
- 即使 `target` 中没有出现“转人工”字样，也要根据标注规则真实填写

### 10.5 `missing_slots`

类型：`array<string>`

推荐值示例：
- `order_id`
- `shipping_status`
- `logistics_status`
- `is_opened`
- `has_quality_issue`
- `new_address`

说明：
- 如果信息充分，则填空数组 `[]`
- 这个字段是后续分析“为什么该追问”的关键依据

### 10.6 `policy_basis`

类型：`array<string>`

推荐写法：
- 使用简短规则短语，不要写成长段自然语言

示例：
- `未发货可取消`
- `已发货不可取消应引导退货`
- `已发货已揽收不可修改地址`
- `催发货只能反馈不能承诺时间`
- `拆封后仅质量问题可售后`

### 10.7 `risk_flags`

类型：`array<string>`

推荐枚举：
- `权限限制`
- `资金风险`
- `投诉风险`
- `规则未确认`
- `信息不足`
- `情绪升级`
- `疑似异常订单`

### 10.8 `quality_check`

类型：`object`

可选子字段建议：

```json
{
  "is_consistent_with_context": true,
  "contains_forbidden_promise": false,
  "needs_revision": false
}
```

说明：
- 这是质检字段，不参与模型训练
- 可在人工审核或脚本审核阶段使用

---

## 11. 必填与可选规则

### 11.1 顶层必填

以下字段必须存在：

- `id`
- `version`
- `train_payload`
- `annotation_meta`

### 11.2 train_payload 必填

以下字段必须存在：

- `system`
- `context`
- `history`
- `target`

### 11.3 annotation_meta 必填

以下字段必须存在：

- `scenario`
- `intent`
- `decision_type`
- `should_transfer`
- `missing_slots`
- `policy_basis`
- `risk_flags`

### 11.4 context 可选但建议显式

`context` 里的字段允许按场景省略。  
但如果某字段直接影响当前判断，建议：

- 已知就填具体值
- 不知道就填 `未知`

不要让关键判断依赖“字段没写”。

---

## 12. 样本示例 v1

### 12.1 示例一：取消订单，可直接回答

```json
{
  "id": "cancel_order_000001",
  "version": "schema_v1",
  "train_payload": {
    "system": "你是电商客服助手，需要礼貌、专业、基于已知信息回答，不编造订单状态、物流进度、处理结果或承诺时间；信息不足时先追问，超出权限或高风险场景时转人工。",
    "context": {
      "user_request": "取消订单",
      "shipping_status": "未发货",
      "order_id_provided": "是",
      "user_emotion": "正常"
    },
    "history": [
      {
        "role": "user",
        "content": "帮我取消一下订单"
      }
    ],
    "target": "您好，如果订单目前还未发货，一般是可以取消的。您可以先在订单详情页查看是否有取消入口并按页面提示操作；如果页面暂时没有相关入口，再联系人工客服进一步处理。"
  },
  "annotation_meta": {
    "scenario": "取消订单",
    "intent": "询问是否可取消订单",
    "decision_type": "direct_answer",
    "should_transfer": false,
    "missing_slots": [],
    "policy_basis": [
      "未发货可取消"
    ],
    "risk_flags": [],
    "quality_check": {
      "is_consistent_with_context": true,
      "contains_forbidden_promise": false,
      "needs_revision": false
    }
  }
}
```

### 12.2 示例二：修改地址，需要明确限制

```json
{
  "id": "change_address_000001",
  "version": "schema_v1",
  "train_payload": {
    "system": "你是电商客服助手，需要礼貌、专业、基于已知信息回答，不编造订单状态、物流进度、处理结果或承诺时间；信息不足时先追问，超出权限或高风险场景时转人工。",
    "context": {
      "user_request": "修改地址",
      "shipping_status": "已发货",
      "logistics_status": "已揽收",
      "new_address_provided": "是",
      "order_id_provided": "是",
      "user_emotion": "焦虑"
    },
    "history": [
      {
        "role": "user",
        "content": "我地址填错了，现在能改吗？"
      }
    ],
    "target": "您好，这边根据当前信息看，订单已经发货且物流已揽收，这种情况下通常无法直接修改收货地址。建议您尽快联系承运物流尝试协调修改，具体以物流方处理结果为准。"
  },
  "annotation_meta": {
    "scenario": "修改地址",
    "intent": "询问是否可修改地址",
    "decision_type": "rule_explanation",
    "should_transfer": false,
    "missing_slots": [],
    "policy_basis": [
      "已发货已揽收不可修改地址"
    ],
    "risk_flags": [],
    "quality_check": {
      "is_consistent_with_context": true,
      "contains_forbidden_promise": false,
      "needs_revision": false
    }
  }
}
```

### 12.3 示例三：退货条件判断，需要追问

```json
{
  "id": "return_judge_000001",
  "version": "schema_v1",
  "train_payload": {
    "system": "你是电商客服助手，需要礼貌、专业、基于已知信息回答，不编造订单状态、物流进度、处理结果或承诺时间；信息不足时先追问，超出权限或高风险场景时转人工。",
    "context": {
      "user_request": "退货退款",
      "is_signed": "是",
      "is_opened": "未知",
      "has_quality_issue": "未知",
      "user_emotion": "正常"
    },
    "history": [
      {
        "role": "user",
        "content": "这个商品我想退掉"
      }
    ],
    "target": "您好，我先帮您判断是否符合退货条件。请问商品目前是否已经拆封，另外是否存在质量问题？您补充这两个信息后，我继续帮您判断对应的处理方式。"
  },
  "annotation_meta": {
    "scenario": "退货条件判断",
    "intent": "询问是否可退货",
    "decision_type": "ask_followup",
    "should_transfer": false,
    "missing_slots": [
      "is_opened",
      "has_quality_issue"
    ],
    "policy_basis": [
      "退货条件判断需先确认是否拆封和是否质量问题"
    ],
    "risk_flags": [
      "信息不足"
    ],
    "quality_check": {
      "is_consistent_with_context": true,
      "contains_forbidden_promise": false,
      "needs_revision": false
    }
  }
}
```

---

## 13. 校验规则建议

为了保证样本质量，建议对 `schema v1` 增加以下校验：

### 13.1 结构校验
- 顶层必填字段不能为空
- `history` 至少 1 条
- `target` 不能为空字符串
- `decision_type` 必须在枚举中

### 13.2 一致性校验
- 若 `decision_type = ask_followup`，则 `missing_slots` 不应为空
- 若 `should_transfer = true`，则 `decision_type` 通常应为 `transfer_human`
- 若 `shipping_status = 已发货` 且 `intent = 询问是否可取消订单`，则 `policy_basis` 不应出现“未发货可取消”
- 若 `is_opened = 是` 且 `has_quality_issue = 否`，则 `target` 不应表达“可直接退货退款”

### 13.3 禁止承诺校验

建议脚本或人工审核以下高风险表达：
- `今天一定发`
- `明天就到`
- `已经帮您取消成功`
- `已经为您改好了`
- `已经退款成功`
- `已经加急处理完成`

---

## 14. v1 的边界与后续升级方向

本版 `schema v1` 的特点是：

- 训练接口简单
- 数据结构清晰
- 已足够支持第一版 SFT

但它仍有边界：

- 还没有纳入特殊商品规则
- 还没有纳入售后时效规则
- 还没有纳入图片、凭证、金额等字段
- 还没有纳入更细粒度情绪标签

后续如需扩展，可在 `schema v2` 增加：

- `is_special_product`
- `after_sales_within_time_limit`
- `evidence_provided`
- `refund_amount_related`
- `complaint_level`

---

## 15. 最终建议

如果你现在要正式开始做样本，我建议直接采用这版 `schema v1`，原因是：

- 足够简单，不会拖慢起步
- 足够结构化，能控制标注一致性
- 和你当前 `input/output v1` 完全兼容
- 可以直接用于后续样本构造和自动校验

下一步最自然的是继续做：

1. `字段枚举值表 v1`
2. `标注规范 v1`
3. `高质量样例集 v1`

其中最建议先做的是 `标注规范 v1`，因为 schema 只是骨架，真正决定数据质量的是标注规则是否统一。
