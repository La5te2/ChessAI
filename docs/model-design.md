# Model 设计方向

## 目标

Gadidae 当前已经是一个可以下棋的神经网络引擎原型，但如果继续追求“真正有新东西”的方向，核心不应是给国际象棋补更多领域规则，而应是探索更通用的模型与搜索关系。

国际象棋在这里更适合作为实验环境：

- 规则明确。
- 状态和动作空间复杂。
- 有成熟教师机可以提供外部评价。
- 可以通过 arena、regret、value validation 做闭环测试。

真正值得推进的主题应当满足：

- 可以迁移到其他博弈或决策任务。
- 能改善模型本身，而不是只给搜索器打补丁。
- 能在单 GPU 或 CPU 可承受的资源下验证。
- 能解释为什么模型在 `search=closed` 和 `MCTS` 下都变强。

## 近期取舍

当前默认实现是 `resnet_pv_linear`，`resnet_pva_gad` 是仍需实验验证的半 Chessformer 候选：

```text
ResNet trunk
  -> 6 geometry self-attention blocks
  -> source-destination policy / dueling advantage head
  -> token value head
```

对应代码架构名：

```text
resnet_pva_gad
```

这个候选把三个低风险改动合在一起：

- `geometry attention`：把同行、同列、斜线、马步、王步和距离关系作为 attention bias。
- `source-destination policy head`：使用 `64x64 + 64x9` action space，贴近棋子从源格到目标格的动作结构。
- `dueling advantage head`：单独输出 `A(s,a)`，表达同一局面内某个动作的相对优势。

暂缓方向：

- `Search-Consistency Training` 需要额外采样 closed / low-sim / high-sim 输出，成本高，并且当前无法证明高 sims 决策稳定更好。
- `Search-Aware Model` 与 `search-gain-head` 很理想，但训练信号难定义，先不作为近期目标。
- `Piece-Token Transformer` 作为第三类模型路线，等输入、mask、policy head 重新设计后再做。
- `NNUE-inspired` 作为独立 CPU 模型路线，目标是 CPU 高速评估和增量更新。
- 基于高 sims 选择的训练目标暂不推进。

## 当前基线

当前模型是残差 CNN：

```text
18x8x8 board features
  -> conv stem
  -> residual blocks
  -> policy head: 4672 actions
  -> value head: [-1, 1]
```

当前训练大致分两层：

- `train.py`：从人类棋谱监督学习 policy，value 来自棋局结果。
- `offline_pv.py`：从 PGN/H5/FEN 抽局面，让 `resnet_pv_linear` 当前模型提出候选走法，再由 Stockfish 评价候选，生成 reward / teacher policy / teacher value，用离线 actor-critic 方式训练。

当前搜索大致是：

- `closed`：直接使用模型 policy/value。
- `only-mcts`：使用模型 policy/value 引导 MCTS。

决策层按 checkpoint `arch.type` 分派。`resnet_pv_linear` 的 profile 使用 `alphazero_64x73` policy/value 与 FPU，`resnet_pva_gad` 的 profile 使用 `sd_64x64_underpromo9` policy/value/advantage，并用 FPU + advantage 初始化未访问动作。

这说明当前瓶颈不只是搜索参数，而是模型对“某一步棋到底好不好”的内部表示还不够强。

## 方向一：Action-Conditioned Advantage Head

这是最值得优先考虑的方向。

当前模型只有：

```text
policy(s): 这个局面下哪些走法像好棋
value(s): 这个局面对当前方整体有多好
```

但搜索真正需要的是：

```text
q(s, a): 当前局面下走某一步之后，这一步本身有多好
```

也就是说，模型需要学会“评价动作”，而不是只学会“选择动作”和“评价局面”。

### 为什么它更通用

`A(s, a)` 不是国际象棋专用概念。任何决策系统都可以使用：

- 当前状态 `s`
- 可选动作 `a`
- 动作价值 `A(s, a)`

这比 tactic 更接近通用方法。

### 怎么落到当前项目

可以给模型增加一个 `advantage_head`：

```text
shared trunk
  -> policy logits: 4672
  -> value: 1
  -> advantage values: 4672
```

其中 `advantage_head[index(move)]` 直接估计该走法相对同局面候选动作的质量，可以用 Stockfish regret 转换成目标：

```text
advantage_target = f(regret_cp)
```

例如：

```text
advantage_target = -tanh(regret_cp / scale)
```

或者保持和当前 reward 一致：

```text
reward = 1 - clamp(regret_cp / reward_scale_cp, 0, 2)
```

只对合法候选和被教师机评价过的 top-k 位置计算 loss。

### 对 MCTS 的作用

当前 MCTS 中，未访问子节点的初始价值主要依赖 FPU 和父节点估计。加入 `advantage_head` 后，可以改成：

```text
child_initial_q = FPU(s) + advantage_head(s, a)
```

这会直接改善：

- 低 sims 下的走法质量。
- 高 sims 时的早期探索方向。
- policy 相近但真实价值不同的候选排序。
- `closed` 模式下模型自身的实战能力。

### 主要风险

- `advantage_head` 输出 4672 个动作值，会增加参数和显存。
- 只有 top-k 被教师机评价，未评价动作必须 mask。
- 如果 reward scale 设计不好，advantage 会变成粗糙二分类。

### 优先级

高。

这是当前最像“模型能力升级”的方向。

## 方向二：Dueling Policy-Value-Q Architecture

可以把 `A(s, a)` 拆成：

```text
A(s, a) = V(s) + A(s, a)
```

其中：

- `V(s)`：局面本身价值。
- `A(s, a)`：某个动作相对其他动作的优势。

这类结构常见于强化学习，叫 dueling architecture。

### 为什么它适合当前项目

国际象棋里很多局面本身已经很差或很好。模型如果直接学 `A(s, a)`，可能会把局面价值和动作质量混在一起。

例如：

```text
当前局面已经 -800cp。
某步棋虽然还是很差，但已经是唯一防守。
```

这时 `V(s)` 应该低，但 `A(s, a)` 应该高。

这种拆分可以帮助模型理解：

- 局面有多好。
- 某步棋在当前局面中是否相对正确。

### 可能结构

```text
shared trunk
  -> value head V(s)
  -> value head V(s)
  -> advantage head A(s, a)
  -> policy head
```

### 对训练的要求

需要让 `value` 学教师机局面评价，让 `advantage` 学候选动作 regret。

这比单纯 policy distillation 更清楚：

- policy 学“该下什么”。
- value 学“局面怎样”。
- advantage 学“这一步相对怎样”。

### 优先级

高。

## 方向三：Search-Consistency Training

当前有一个明显现象：

```text
同一个模型在 sim=0、低 sims、高 sims 下会给出不同甚至更差的走法。
```

这说明模型和搜索之间没有形成稳定关系。

Search-consistency 的目标是让模型理解：

```text
搜索前 policy
搜索后 policy
教师机评价
```

三者之间应当尽量一致。

### 可训练目标

对同一批局面，分别记录：

```text
policy_closed(s)
policy_mcts_100(s)
policy_mcts_500(s)
teacher_scores(top-k)
```

然后训练模型减少：

```text
closed policy 与高预算搜索结果的分布差异
closed top-k 与教师机评分排序的差异
value 与教师机局面评价的差异
q 与教师机动作评价的差异
```

### 关键点

不能无脑蒸馏 MCTS，因为 MCTS 自己也可能被弱 value 带偏。

更合理的是：

```text
MCTS 提供候选空间和访问分布。
Stockfish/evaluator 提供质量评价。
模型学习“哪些搜索结果真的值得相信”。
```

### 通用意义

这不是 chess-specific。任何使用模型 + 搜索的系统都会遇到：

- 搜索预算越大是否一定更好。
- 模型如何从昂贵搜索中吸收能力。
- 低预算模型如何逼近高预算决策。

### 优先级

中高。

它和 `advantage_head` 可以结合，但单独做可能仍然被弱 value 限制。

## 方向四：Search-Aware Model

当前模型只回答：

```text
走什么？
局面怎样？
```

但一个更适合搜索的模型还可以回答：

```text
这个局面是否复杂？
policy 是否可靠？
value 是否不确定？
是否值得分配更多搜索预算？
```

### 可能输出

```text
uncertainty_head(s)
complexity_head(s)
policy_entropy_target(s)
search_gain_head(s)
```

其中 `search_gain_head` 可以学习：

```text
高 sims 相比 closed 在该局面的收益有多大
```

如果某个局面 closed 和高 sims 结果一致，说明模型可以快走。

如果某个局面搜索后发生巨大变化，说明它是复杂局面，需要更多预算。

### 对 CPU 引擎的意义

这对 CPU 很重要，因为 CPU 无法长期高 sims。模型如果能判断“哪里值得想”，就可以更聪明地用有限时间。

### 优先级

中。

它更像搜索预算控制层，最好在 policy/value/advantage 稳定后再做。

## 方向五：更强的 Board Network

当前 CNN + ResNet 是合理基线，但可以继续增强。

### 5.1 更大 ResNet

直接增加：

```text
channels
blocks
```

优点：

- 最简单。
- 对当前代码改动范围小。
- 容易验证。

缺点：

- CPU 变慢。
- 训练数据和训练步数不足时，可能只是更容易过拟合。
- 不是方法上的新东西。

### 5.2 Policy Plane Head

当前 policy head 是：

```text
conv -> flatten -> linear 4672
```

可以改成更空间化的：

```text
conv -> 73 policy planes -> flatten 64x73
```

这更贴近 move encoder 的 `64x73` 结构，也保留了“每个起点格有哪些动作类型”的空间归纳偏置。

这不是国际象棋完全专用，而是 board-game action-space 的结构化建模。

优先级：中高。

### 5.3 SE / Channel Attention

在 ResNet block 中加入 squeeze-excitation：

```text
board features -> channel weights -> rescale channels
```

它可以让模型动态强调：

- 王安全通道。
- 攻击通道。
- 子力通道。
- 走子方相关通道。

这比完整 Transformer 便宜得多。

优先级：中。

### 5.4 Axial Attention / Lightweight Attention

在 8x8 棋盘上，完整 attention 的规模不大，可以考虑少量 attention block：

```text
CNN trunk
  -> axial attention or small self-attention
  -> heads
```

意义：

- CNN 擅长局部模式。
- attention 擅长长距离关系。

国际象棋有大量长距离关系：

- 车/象/后沿线攻击。
- 王翼与后翼联动。
- 远距离牵制。

风险：

- CPU 推理变慢。
- 小数据下不一定比 ResNet 稳。

优先级：中低，作为实验分支。

### 5.5 Piece-Token Transformer

把棋盘转成 piece tokens：

```text
piece type
color
square
side to move
castling/en-passant
```

然后用 Transformer 建模棋子间关系。

优点：

- 更接近通用 attention 方法。
- 自然处理长距离关系。

缺点：

- 需要重新设计输入、mask、policy head。
- CPU 上可能不友好。
- 与当前 4672 policy 输出的连接需要额外设计。

优先级：低到中。

不建议作为近期主线。

## 方向六：NNUE-Inspired CPU Network

Stockfish 的 NNUE 重要之处不是“更聪明”，而是：

```text
适合 CPU。
适合增量更新。
适合高频局面评价。
```

Gadidae 可以借鉴这个思想，但不必复刻 Stockfish NNUE。

### 可借鉴点

- 用稀疏特征表示局面。
- 每走一步只更新变化部分。
- 让 value 评估极快。
- CPU 上大量节点搜索成为可能。

### 与当前 CNN 的区别

CNN 每次评估都要完整跑一遍 18x8x8。

NNUE-style 网络可以维护 accumulator：

```text
position accumulator
  -> small MLP
  -> value / q
```

这对 CPU 引擎非常关键。

### 可能路线

保留 CNN 作为训练/研究模型，同时设计一个轻量 CPU 模型：

```text
Gadidae-ResNet: GPU/训练/分析
Gadidae-CPU: NNUE-inspired value/q evaluator
```

然后探索：

```text
ResNet teacher -> CPU model distillation
Stockfish regret -> CPU model advantage/value training
```

### 通用意义

这不是国际象棋 tactic，而是“增量神经网络评估器”的工程路线。它可以迁移到其他局面变化稀疏的任务。

### 优先级

中高。

如果目标包括 lichess bot 和 CPU 可用性，这条很重要。

## 方向七：Offline Evaluator-Guided RL

这是当前 `offline_pv.py` 的路线。

抽象成通用方法就是：

```text
已有模型提出候选动作。
外部 evaluator 给每个候选动作打分。
模型用这些反馈更新 policy/value/advantage。
```

Stockfish 只是当前 evaluator。

### 优点

- 不需要大规模自对战。
- 单 GPU 可跑。
- 能直接用强 evaluator 指出错误。

### 问题

- 很容易退化成 evaluator distillation。
- 如果只有 policy/value，没有 advantage，模型可能学不到“某步棋为什么错”。
- 如果候选 top-k 太窄，模型看不到真正好棋。
- 如果 reward scale 不合适，学习信号会过粗或过饱和。

### 改进重点

这条路线继续走时，应优先配合：

- `advantage_head`
- teacher value
- teacher policy soft distribution
- search consistency validation
- action-level regret validation

而不是只继续调 learning rate、steps、arena gate。

### 优先级

高。

但它更像实验框架，不是最终答案。

## 方向八：Model-Based Policy Improvement

当前模型依赖 Stockfish 做 evaluator。未来如果模型足够强，可以让模型自己产生改进目标。

类似 AlphaZero：

```text
self-play
  -> MCTS policy improvement
  -> train policy/value
  -> stronger model
```

但现阶段模型还不够强，直接走这条风险很高。

更现实的中间路线是：

```text
Stockfish 继续做验证器。
模型/MCTS 生成候选和分布。
advantage/value 学会更稳定后，逐步减少对 Stockfish 的训练依赖。
```

### 优先级

长期方向。

当前不是最优先。

## 方向九：Tactic 作为领域增强

`tactic` 仍然有价值，但它的位置应当明确：

```text
领域层搜索增强
```

它可以：

- 帮人类理解模型候选。
- 在运行时提供软排序信号。
- 标记某些候选的战术意图和风险。
- 未来替代窄化的 mate 搜索。

但它不应被当成：

- 通用模型方法。
- 核心研究创新。
- 训练真值来源。
- MCTS/AB 同级的通用搜索范式。

### 优先级

中低。

等模型本身更强后再做更合理。

## 不建议作为主线的方向

### 只堆更大模型

更大模型可能提高上限，但不自动解决：

- policy 错误自信。
- value 不准。
- 高 sims 反而变差。
- CPU 运行慢。

### 只调 MCTS 参数

MCTS 参数可以改善表现，但不会让模型参数变聪明。

如果 `closed` 很弱，高 sims 又不稳定，根因仍然是 policy/value/advantage 表示不足。

### 纯 Stockfish 蒸馏

可以提高棋力，但方法上容易变成：

```text
为什么不直接用 Stockfish？
```

Stockfish 可以作为 evaluator，但不应成为项目的全部意义。

### 过早上完整 Transformer

Transformer 是通用方法，但在当前资源下不一定划算。

如果没有足够数据、训练预算、消融实验，它可能只是更贵的模型。

## 推荐路线

### 阶段 1：增加 action-value 能力

目标：

```text
让模型知道每个候选动作本身有多好。
```

改动：

- 增加 `advantage_head`。
- offline-pv 数据保存 `advantage_target` 或 action reward。
- loss 中加入 masked advantage loss。
- validation 加入 advantage/ranking/regret 指标。
- MCTS 使用 FPU + advantage 初始化未访问子节点。

预期收益：

- `closed` 更强。
- 低 sims 更强。
- 高 sims 更稳定。
- policy 相近时更少选明显坏棋。

### 阶段 2：改 policy head 结构

目标：

```text
让 4672 action space 保留 64x73 的空间结构。
```

改动：

- 将 policy head 改为 `73x8x8` plane 输出。
- 保持 move encoder 不变，只改变 head 形状。
- 对比 `resnet_pv_linear` 的 linear policy head。

预期收益：

- 更好的空间归纳偏置。
- 更少参数浪费。
- 可能提高训练效率。

### 阶段 3：search consistency

目标：

```text
减少 sim=0、低 sims、高 sims 下选择互相打架。
```

改动：

- 对同一局面采集 closed / low-sim / high-sim 输出。
- 教师机评价候选。
- 训练模型吸收“高质量搜索结果”，而不是无脑蒸馏搜索分布。

预期收益：

- MCTS 更像模型能力放大器。
- 高 sims 不再频繁暴露 value 偏差。

### 阶段 4：CPU-oriented model

目标：

```text
让 Gadidae 在 CPU 上也能高频评估。
```

改动：

- 设计 NNUE-inspired value/q evaluator。
- 尝试从 ResNet 或 Stockfish regret 蒸馏。
- UCI/lichess 使用 CPU 模型或混合模型。

预期收益：

- 更适合常开服务器。
- 更适合 lichess bot。
- 减少对 GPU 的依赖。

## 最小实验矩阵

每次只改一个大方向，至少记录：

- `closed` arena。
- `only-mcts` arena。
- top1 regret。
- composite regret。
- teacher best in top-k。
- value MAE / RMSE / sign accuracy。
- 如果有 advantage head，记录 advantage MAE、best-adv move regret、policy-adv disagreement。
- CPU 单步延迟。
- GPU 训练吞吐。

推荐比较：

```text
baseline ResNet policy/value
baseline + advantage_head
baseline + advantage_head + plane policy head
baseline + advantage_head + search consistency
```

## 当前结论

近期最值得做的不是 tactic，也不是直接换 Transformer。

更合理的主线是：

```text
policy/value 模型
  -> policy/value/advantage 模型
  -> search-aware policy/value/advantage 模型
  -> CPU-friendly evaluator
```

这条线保留了国际象棋作为实验场，但真正研究的是更通用的问题：

```text
一个神经网络如何学习动作价值，
以及如何和搜索稳定协作。
```
