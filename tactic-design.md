# Tactic 搜索设计

## 定位

`tactic` 是搜索层模块。它不是训练标签，不是教师机，也不是强制覆盖走法的规则。

搜索层包含两个平级组件：

- `mcts`：基于模型 policy/value 的通用搜索。
- `tactic`：基于模型/MCTS 候选空间的目标导向选择性搜索。

模型本体只提供 policy/value。离线训练阶段的监督信号来自教师机 regret。`tactic` 的作用是解释、分析和可选的运行时软排序，不负责制造训练真值/标签。

## 前置要求

未来实现 `tactic` 前，需要先让 MCTS 的搜索树信息可以被 `tactic` 复用。

具体要求：

- MCTS root node 需要暴露给后续模块。
- `tactic` 在递归进入子局面时，应尽量能找到对应的 MCTS node。
- MCTS node 至少需要暴露：
  - `q`
  - `visits`
  - `prior`
  - `children`
- 如果当前路径在 MCTS tree 中存在，`tactic` 应优先使用对应节点统计进行排序。
- 如果当前路径不在 MCTS tree 中，`tactic` 回退到模型 policy/value 与规则启发式排序。

也就是说，未来 `tactic` 不是只在根节点借用 MCTS，而是要在非根节点也尽量复用 MCTS 已经花过的搜索预算。

MCTS 信息只作为排序和解释信号，不能直接当成战术证明。

## 目标

`tactic` 要回答的问题是：

在当前模型 policy/value 和 MCTS 支持的 top-k 候选空间里，某个候选走法是否可能通向一个可解释的战术或战略目标？

目标局面包括：

- 将杀或将杀网
- 长将和局
- 赢子
- 弃子后获得具体补偿
- 唯一保优手
- 大劣势下强行转入均势
- 王攻
- 开放线压力
- 空间或控制格优势
- 危险区域控制
- 防守资源
- 陷阱或复杂化意图

输出语义应当是“意图可能是”，而不是“客观证明为最佳”。

示例：

```text
这步可能意图制造王翼攻击。
这步可能在搜索候选空间内导向长将和局。
这步可能是当前候选里唯一保持优势的走法。
```

## 目标检测器

模型本身不会说话。它只会给出 policy/value，并通过 MCTS 形成候选分布。

因此 `tactic` 不能直接询问模型“你的意图是什么”，而是要做两件事：

```text
1. 选择性搜索器：
   沿着模型/MCTS 支持的候选空间展开局面线。

2. 目标检测器：
   对搜索得到的路径、终点局面和中间局面进行可计算检测。
```

不同目标需要不同检测器。它们统一在 `tactic` 框架下，但不共享同一个判断公式。

### 静态目标

静态目标可以通过某个局面本身判断，通常比较起点与终点特征差值。

例子：

- 将杀：`board.is_checkmate()`
- 子力优势：material balance、悬挂子、受攻击高价值子
- 空间优势：安全控制格、敌方半场控制、前哨格、开放线、危险区域控制
- 王安全：王区控制、攻击子数量、防守子数量、开放线指向王
- 兵形：通路兵、孤兵、叠兵、弱格

形式：

```text
feature_delta = features(position_after_line) - features(start_position)
```

例如空间优势不应只输出一个抽象分数，而应拆成可解释特征：

```text
safe_controlled_squares_delta
enemy_territory_control_delta
outpost_delta
open_file_pressure_delta
king_zone_control_delta
mobility_delta
```

### 过程目标

过程目标必须观察一条路径，不能只看终点。

例子：

- 长将
- perpetual
- 重复局面
- 连续将军追王
- 强制换子
- 弃子后追回
- 对方唯一应手序列

长将可以通过路径特征检测：

```text
line 中己方连续给将次数
line 中对方王可走区域是否持续缩小
line 中是否出现重复局面 key
line 中是否出现三次重复趋势
对方 top-k 应手后是否仍持续被将
终点是否回到相同或相似局面且轮到同一方
```

示例输出：

```json
{
  "target": "perpetual_check",
  "checks_by_attacker": 5,
  "repetition_keys": 2,
  "king_escape_squares_trend": [3, 2, 2, 1],
  "line": ["Qh5+", "Kg8", "Qe8+"],
  "confidence": "selective"
}
```

### 赢子与弃子

赢子不能只看即时 material delta，因为战术中经常会先弃子。

建议拆成两种：

```text
immediate_material_delta:
  当前局面立刻变化的子力差。

stable_material_delta:
  搜索线结束后，并且对方 top-k 反击已经检查过，仍然保留的子力差。
```

例如：

```text
当前亏一子，但 4 ply 后在 top-k 线里赢回车，
stable_material_delta = +500cp。
```

这类输出可以解释为：

```text
这步可能是临时弃子，意图是在后续强制线中赢回更多子力。
```

## 非目标

`tactic` 不应该：

- 替代 MCTS
- 独自覆盖最终走法
- 充当 Stockfish 或外部教师机
- 生成 regret 标签
- 在当前阶段参与 offline RL 训练
- 在使用 top-k 剪枝时声称自己给出了绝对证明
- 假设对手会故意下让自己变劣的招法

## 与训练的关系

当前训练阶段刻意关闭运行时搜索：

- 模型直接训练 policy/value。
- 搜索不是模型参数的一部分。
- 教师机 regret 由 UCI 教师机给出。
- MCTS 和 tactic 都属于运行时搜索组件。

未来当模型足够强，进入更成熟的自对战训练阶段时，`mcts` 和 `tactic` 可以成为 feedback 链路的一部分。

在当前阶段，`tactic` 主要用于：

- board 分析
- search trace
- arena 可解释性
- 人类阅读模型意图
- 运行时软排序信号

## 与 MCTS 的关系

MCTS 是通用决策搜索，它问：

```text
在模型引导下，哪个走法整体上更值得选择？
```

`tactic` 是目标导向搜索，它问：

```text
某个候选走法是否能在模型/MCTS 支持的候选空间内导向一个可解释目标？
```

`tactic` 应尽量复用 MCTS 信息：

- 根节点候选排序
- 根节点访问次数
- 根节点 Q 值
- 根节点 prior
- 已展开子节点
- MCTS tree 中已有的更深层子节点统计

当 MCTS 信息缺失时，`tactic` 回退到模型 policy/value 和规则启发式。

MCTS 统计可以指导排序，但不是证明。`tactic` 不能因为某个 MCTS Q 高就直接认为战术成立。

## 选择性证明语义

`tactic` 不是绝对证明搜索，而是在指定候选空间内的选择性证明/判断。

绝对证明：

```text
对方每一个合法应手都无法阻止目标成立。
```

选择性证明：

```text
对方每一个被搜索到的 top-k 合理应手都无法阻止目标成立。
```

第二种更便宜，也更符合模型引导搜索，但可能漏掉低 policy 的冷门防守。因此输出必须标明作用范围。

示例：

```json
{
  "confidence": "selective",
  "scope": "model_mcts_topk",
  "searched_all_legal_replies": false
}
```

## 候选空间

根候选来自模型/MCTS top-k。

内部候选排序应综合：

- 已有 MCTS node 统计
- 模型 policy
- 模型 value
- 将军
- 吃子
- 升变
- 对王的威胁
- 受保护/未保护子力
- 子力变化
- 控制格变化
- 线、列、斜线压力
- 重复、长将等和棋资源

对手也应被假设会从合理 top-k 里选招，而不是故意下坏棋。如果某条线只被一个低排序防守推翻，输出应标记风险，而不是把它当成绝对成立。

## 搜索规模与概率语义

如果简单展开 `topk=4`、`depth=10 ply`，叶子数约为：

```text
4^10 = 1,048,576
```

这个数量级在抽象算法题里不算大，但在实际工程里仍然偏贵：

- 每个节点需要生成合法走法。
- 每个节点可能需要模型 policy/value。
- 每条路径都要维护局面、重复、将军、子力、控制格等特征。
- board/arena 中可能要对多个 root candidate 同时分析。
- Python 层 chess move generation 与对象复制成本不可忽略。
- 如果还要输出解释线，路径记录和排序也会产生额外开销。

因此 `tactic` 不应盲目完整展开 `topk^ply`。它需要类似 MCTS 的预算分配思想，但目标和 MCTS 不同。

MCTS 关心：

```text
当前最好下什么？
```

`tactic` 关心：

```text
某个候选走法是否可能导向某个特定目标？
对方合理应手下，这个目标有多稳？
```

所以 tactic 的判断天然带概率语义。一个目标线成立，不只是因为存在某条路径，还因为这条路径在模型/MCTS 分布里有足够概率质量，且对方 top-k 合理防守没有轻易破坏目标。

建议维护以下量：

- `path_prob`：沿路径累乘或累加 log policy 得到的路径概率质量。
- `mcts_support`：路径中有多少节点能复用 MCTS 统计。
- `target_score`：目标检测器给出的收益分数。
- `target_confidence`：目标在搜索候选空间内的稳定程度。
- `refutation_mass`：对方合理应手中破坏目标的概率质量。
- `coverage_mass`：本次搜索覆盖了多少模型/MCTS 概率质量。

候选线不应只有二值结论，而应类似：

```json
{
  "target": "king_attack",
  "path_prob": 0.18,
  "coverage_mass": 0.72,
  "refutation_mass": 0.11,
  "target_score": 0.34,
  "confidence": "selective"
}
```

### 搜索策略

`tactic` 可以采用 beam / best-first 风格：

```text
priority = path_prob * target_potential * mcts_support_adjustment
```

候选展开顺序优先考虑：

- MCTS visits 高的节点
- MCTS Q 高的节点
- model policy 高的走法
- model value 改善明显的走法
- 目标检测器认为有潜力的走法
- 将军、吃子、升变等强战术候选
- 过程目标相关候选，例如连续将军、重复局面、王区限制

这样 tactic 不需要像当前 mate 那样确定式地穷举所有防守，也不需要像纯 MCTS 那样只优化通用胜率。它应当是：

```text
目标导向的概率式选择性搜索。
```

### 对方应手

对方不会被假设故意下坏棋。

对方节点也应按模型/MCTS top-k 搜索，并统计破坏目标的质量：

```text
如果对方 top-k 高概率应手都无法破坏目标：
  目标较稳。

如果只有一个低概率应手能破坏目标：
  标记为有隐藏防守风险。

如果对方高概率应手能直接破坏目标：
  目标不成立或置信度低。
```

这比 absolute proof 弱，但更符合模型/MCTS 搜索视角。

## 输出

`tactic` 应返回每个候选走法的结构化分析，而不是只返回最佳走法。

示例：

```json
{
  "move": "Nxd3",
  "intent": "可能制造王翼攻击机会",
  "targets": ["king_attack", "mate_net"],
  "confidence": "selective",
  "scope": "model_mcts_topk",
  "uses_mcts": true,
  "score_delta": 0.18,
  "risk": "可能存在未搜索到的防守资源",
  "pv": ["Nxd3", "Bh6", "Qh5"],
  "searched_nodes": 4210
}
```

建议字段：

- `move`
- `intent`
- `targets`
- `confidence`
- `scope`
- `uses_mcts`
- `score_delta`
- `risk`
- `pv`
- `searched_nodes`
- `candidate_rank`
- `mcts_visits`
- `mcts_q`
- `model_prior`
- `model_value`

## 运行时软排序

未来 `tactic` 可以作为运行时软信号：

```text
final_score = mcts_score + tactic_weight * tactic_score
```

它不应该独自强制决定走法。

适合用途：

- 在 MCTS 候选接近时辅助排序
- 解释低排序走法的实际战术意图
- 标记表面诱人的风险走法
- 高亮唯一保优手

不适合用途：

- 用选择性线覆盖 MCTS
- 把选择性将杀当成绝对将杀
- 把 tactic 输出当成教师机真值

## 关于将杀

将杀只是 `tactic` 的一个目标：

```text
target = mate_net
```

其他目标可以共享同一套框架：

```text
target = material_win
target = perpetual
target = only_move
target = king_attack
target = space_gain
```

最终结构应保持统一：

```text
model policy/value
  -> mcts 通用搜索
  -> tactic 目标导向搜索
  -> 解释输出与软排序
```
