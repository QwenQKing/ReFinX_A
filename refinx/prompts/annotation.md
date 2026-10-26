你是金融评测基准的标注员。下面给出一道金融问答任务、它的信息截止时间、参考答案，以及旧版标注和标注时检索到的证据。请把旧版标注**重写为精炼、可核验的新版标注**，只输出一个 JSON 对象。

## 新版标注的字段（只输出这些字段）
```
{
  "context_constraints": [{"kind": "...", "requirement": "..."}],
  "facts": [{"id": "F1", "kind": "fact|arithmetic", "statement": "..."}],
  "dependencies": [{"id": "D1", "relation": "...", "prerequisites": ["F1"], "requirement": "..."}],
  "answer_points": [{"id": "A1", "statement": "..."}],
  "numeric_targets": [{"description": "...", "value": 0, "unit": "...", "tolerance": 0}]
}
```

## 规则
1. **context_constraints**：只保留"弄错会改变结论"的金融语境约束，通常 0–3 条，最多 5 条。`kind` 取 `entity`、`time`、`period`、`metric`、`accounting_scope`、`unit_currency`、`assumption` 之一。不要写信息截止时间（评测框架已强制执行），不要写"不涉及……"之类的占位约束，不要为了凑类别而写，不要重复 answer_points 的内容。
2. **facts**：回答这道题所需的最小**经验性**前提集合。每条必须是**脱离上下文也能核验**的事实陈述：写明主体全称或代码、日期或报告期、数值及单位。
   - 分析判断、估值观点、"不能据此推断……"之类的推理不是 fact，放进 answer_points。
   - 不得引用"检索结果、工具轨迹、参考材料、公开资料未显示、supplied reference"等标注过程；把"检索未发现某事"改写为"截至 YYYY-MM-DD，某公司未公告某事"。
   - 不得描述"题设、所给材料、证据是否包含某信息"。只有澄清类任务可以把"用户问题未提供某项信息"写成 fact。
   - `arithmetic` 写明输入数值和计算结果。
   - 只用下面给出的参考答案、旧版标注和证据中已有的信息，不要编造。
3. **dependencies**：只保留实质影响结论的跨阶段依赖。`relation` 只能取 `entity`、`time`、`metric_scope`（指标定义与会计口径，包括 accounting scope）、`unit_currency`、`input_result_use`、`evidential_support` 之一，不要使用其他取值；`prerequisites` 只引用本标注中的 fact id；`requirement` 写清下游应如何正确使用这些前提（例如"不得把可转债增持当作股票增持"）。
4. **answer_points**：最终回答必须给出或支持的实质结论，来自参考答案，最多 6 条。澄清类任务把"应向用户询问的关键缺失信息"也列为要点。
5. **numeric_targets**：只填用户问题**要求得出**的数值结论（例如增速、估值、收益率），每个结论一条；中间数据和支撑性数字不要放这里（它们属于 facts）。参考答案没有给出确定数值结论时为空数组——大多数任务应为空。`value` 和 `tolerance` 必须是单个数字（不能是区间、文字或 null）；区间或非数值结论不要放在这里。
6. **所有字段的文字内容一律使用{output_language}**，即使用户问题、参考答案或本说明是中文——这是要随论文发布的英文数据集。
   - 公司写英文全称或通用英文名并附证券代码，例如 "Shanxi Meijin Energy Co., Ltd. (000723.SZ)"；没有通用英文名时用拼音，并在括号中保留中文原名。
   - 中文特有的术语（如"主力资金""扣非净利润"）第一次出现时给出英文译名，并在括号中保留中文原文；金额保留原币种与单位（如 "RMB 1.2 billion"）。
7. 只输出 JSON，不要输出任何解释。

## 任务
- 用户问题：{query}
- 信息截止时间（UTC）：{as_of}
- 期望行为：{terminal_action}
- 输出语言：{output_language}

## 参考答案
{gold_answer}

## 可接受的替代答案
{accepted_alternatives}

## 需要澄清的信息
{required_clarifications}

## 旧版标注（供改写，不要照抄其中的问题）
{old_annotation}

## 标注时检索到的证据（节选）
{evidence}
