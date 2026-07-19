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
- `dueling advantage head`：输出 minimax `A(s,a)=Q(s,a)-V(s)`；状态包含当前行棋方，因此 `V(s)=max_{a in legal(s)} Q(s,a)` 且合法动作的 A 始终非正。

暂缓方向：

- `Search-Consistency Training` 需要额外采样 closed / low-sim / high-sim 输出，成本高，并且当前无法证明高 sims 决策稳定更好。
- `Search-Aware Model` 与 `search-gain-head` 很理想，但训练信号难定义，先不作为近期目标。
- `Piece-Token Transformer` 作为独立模型路线，等输入、mask、policy head 重新设计后再做。
- `NNUE-inspired` 作为独立 CPU 模型路线，目标是 CPU 高速评估和增量更新。
- 基于高 sims 选择的训练目标暂不推进。

## 第三架构候选：History-Aware Model

当前两个架构都把单个 `chess.Board` 编码为无历史状态：

- `resnet_pv_linear` 的 18 planes 包含棋子、行棋方、王车易位权和吃过路兵。
- `resnet_pva_gad` 的 square tokens 包含同类当前局面信息。
- 两者都不编码 `move_stack`、重复次数、halfmove clock 或过去局面。

因此，棋盘、行棋方、王车易位权和吃过路兵状态完全相同时，第一次、第二次和第三次出现的局面对当前模型是同一个输入。模型可以学习“胜势时通常应减少循环”，但无法严格判断某个候选是否正在接受或完成三次重复。

IMF 与 RPP 在这里具有不同性质：

- 一步将死完全由当前棋盘决定。Policy/Value、终局回报、反事实评价和 MCTS 都有可能逐步学会它；IMF 是确定性的决策层保障。
- 三次重复依赖历史。当前模型缺少区分这些状态的信息，RPP 补充的是不可观测状态，而不只是棋力不足。

逼和也需要进一步区分：

- 三次重复、五十回合和依赖既往循环次数的长将属于历史问题。
- 无合法着法导致的逼和由当前棋盘即可确定；“唯一序列获胜，否则逼和”主要属于多步推演和 Value 判断问题。

第三架构的目标不是把手写 RPP 换成一个更不可靠的近似，而是让模型获得当前架构缺失的时间状态，并学习历史如何影响 Policy 和 Value。

### 将棋局恢复为完整状态

最直接的输入可以写成：

```text
current board
+ side to move
+ castling rights
+ en-passant state
+ halfmove clock
+ reversible history
```

这使模型看到的状态更接近 Markov 状态。显式规则状态负责准确表达规则条件，历史编码器负责学习循环、长将、节奏和历史战术模式。

### 固定历史窗口

最简单的实现是堆叠最近 `N` 个棋盘或走法：

```text
position(t), position(t-1), ..., position(t-N+1)
```

模型仍可以是普通 CNN、ResNet 或 attention 网络，不需要保存外部隐藏状态。该方案容易批处理，但窗口之外的长周期重复不可见，因此属于近似记忆。

### 可逆历史序列

与重复和五十回合有关的历史可以从最近一次不可逆变化之后开始记录。不可逆变化包括兵移动、吃子、王车易位权变化及其他使旧状态无法重新出现的变化。

在项目采用实际达到五十回合条件后判和的规则下，序列可以限制在约 100 ply。这个规模远小于语言模型上下文，可以由较小的时间编码器处理：

- 一层 GRU；
- 少量 temporal attention blocks；
- 小型 move-token Transformer。

Move token 可以包含：

```text
from square
to square
moving piece
captured piece
promotion
check
castling
```

候选结构为：

```text
current-board encoder
        +
reversible-history encoder
        -> gated fusion / cross attention
        -> policy / value / optional uncertainty heads
```

该结构把当前棋盘保持为主要信息源，历史只提供当前状态无法表达的时间信息。

### 不采用隐式持久隐藏状态作为起点

让 RNN 隐藏状态跟随整盘棋持续存在，会增加以下复杂度：

- Undo 必须恢复对应隐藏状态。
- 只给 FEN 时无法恢复隐藏状态。
- MCTS 每条分支必须复制并更新独立隐藏状态。
- 同一棋盘经不同历史到达时具有不同隐藏状态。
- Arena 的批量局面更难合批。

因此，第三架构应该优先接收显式历史并在每次推理时生成 history embedding。它具备记忆能力，同时没有脱离输入数据的隐式状态。

### 数据与运行链路

第三架构需要独立的 per-arch 数据 schema：

- `preprocess` 输出当前状态、历史序列、有效长度和 mask，不能把历史关系完全打散。
- `train` 对当前局面目标和对应历史窗口联合训练。
- FCPI 从每条 trajectory 构造历史输入，反事实分支继承并追加各自的历史。
- Arena、Stadium 和 UCI 从完整 move stack 构造输入。
- PGN 可以提供完整历史。
- 只有 FEN 时必须使用 `history unknown` 标记或空历史，不能伪造重复次数。
- 从开局书 FEN 启动时应明确把该位置视作新的历史起点。

### 与 RPP/IMF 的关系

当前 FCPI 采样保持不接入 RPP/IMF。验收时启用它们评估的是“模型 + 决策层组件”的完整引擎行为。

第三架构跑通后，可以分别测试：

- 无 RPP 时，模型能否依据历史降低主动重复概率。
- 有 RPP 时，规则组件还能带来多少额外收益。
- IMF 关闭时，模型对一步杀的 Top1 命中率是否随训练提高。

RPP 在任何阶段都可以继续作为严格规则保障；模型记忆能力的价值在于理解历史，而不是要求概率模型取代所有确定性规则。

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

- `train.py`：按架构读取对应 H5 schema，监督训练该架构拥有的网络头。
- `fcpi.py`：按架构执行各自的行为分布、反事实目标构造与损失函数，再通过 Arena 验收候选模型。

当前搜索大致是：

- `closed`：直接使用模型 policy/value。
- `only-mcts`：使用模型 policy/value 引导 MCTS。

决策层按 checkpoint `arch.type` 分派。`resnet_pv_linear` 的 profile 使用 `alphazero_64x73` policy/value 与 FPU；`resnet_pva_gad` 的 profile 使用 `sd_64x64_underpromo9` policy/value/advantage，以 `V+A` 构造边 Q 先验，并在已访问节点上让该先验随 visits 衰减。

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

可以把 `Q(s, a)` 拆成：

```text
Q(s, a) = V(s) + A(s, a)
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

它更像搜索预算控制层，应该在 policy/value/advantage 稳定后再做。

### 4.1 Bayesian Epistemic Uncertainty / PVUA

当前 `search.py` 里的 uncertainty 是搜索后的启发式根节点模糊度，主要来自 visits 熵、top2 visits 接近程度和 top2 Q 接近程度。它能帮助分配 MCTS 预算，但它不是模型自己输出的认知不确定性。

一个更系统的方向是让模型显式输出：

```text
policy(s)
value(s)
advantage(s, a)
uncertainty(s)
```

也就是可以称为 `PVUA` 的结构。其中 `U` 应该拆清楚：

```text
U_a(s): aleatoric uncertainty，局面评价本身的噪声或多解性
U_e(s): epistemic uncertainty，模型因为数据不足、容量不足或分布外局面而产生的不确定
```

如果只用一个 `log_variance` 并通过 Gaussian NLL 训练 value：

```text
loss_value = 0.5 * exp(-log_sigma2) * (value - target)^2 + 0.5 * log_sigma2
```

它主要学到的是 `U_a`，也就是“这个 target 本身有多嘈杂”。这有用，但还不是严格意义上的认知不确定性。真正更接近 `U_e` 的实现可以考虑：

- bootstrap value heads：共享 trunk，挂多个 value heads，用 heads 间方差表示认知不确定。
- lightweight ensemble：多个小 evaluator 或多个 checkpoint 的预测方差。
- MC dropout：推理时多次 dropout 采样，成本较低但稳定性需要验证。
- Laplace / SWAG 类近似：在训练后估计权重后验，工程复杂度更高。

对 Gadidae 来说，最现实的版本是：

```text
shared trunk
  -> policy head
  -> value heads: V_1 ... V_k
  -> log_sigma2 head
  -> advantage head

value_mean = mean(V_i)
U_e = variance(V_i)
U_a = exp(log_sigma2)
```

这样它不是把国际象棋变成非完全信息游戏。棋盘仍然是完全信息的；“战争迷雾”来自模型自身看不见的计算空间：有限容量、有限训练数据、有限搜索预算让许多后继局面在模型内部近似不可见。因此 `U_e` 可以被看作模型对自己盲区的估计。

### 对搜索的作用

`U` 不应直接当作走法好坏。更合理的用法是：

- 高 `U_e` 局面分配更多 MCTS 预算。
- 高 `U_e` 时提高 FPU 保守性，避免 policy 自信地冲进盲区。
- 训练时把高 `U_e` 且高 regret 的局面加入优先采样。
- arena 或 simulator 中显示“模型对此局面不确定”，帮助人类理解模型输出。

这能解释一个现实问题：小模型在棋盘上看到的是完整状态，但它未必真的“理解”完整后果。显式不确定性可以让搜索知道哪些地方需要多看，哪些地方可以快走。

### 风险

- 如果 value target 本身来自噪声很大的自对战，`U` 可能学成“哪里数据脏”。
- 如果只训练 NLL 而没有多头或 ensemble，`U` 更像 aleatoric，不足以表达模型盲区。
- 如果把 `U` 直接加进走法排序，可能让模型偏向保守或偏向怪招。

### 优先级

中高。

它适合作为后续架构候选，但应放在 `resnet_pva_gad` 跑通监督学习和基础 arena 后再做。推荐名称可以是 `resnet_pvua_gad`，表示 geometry attention + policy/value/uncertainty/advantage。

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

这是一条可独立研究的外部评价器引导路线，并非当前训练链路。

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

## 方向八：Folded Counterfactual Policy Iteration

该方法简称 `FCPI`，中文名为“折叠式反事实策略迭代”。目标是在自对战采集阶段关闭 MCTS，再把少量反事实后继评价形成的策略改进折叠回模型。

```text
closed self-play
  -> TD(lambda) value targets from game outcomes
  -> adaptive multi-ply value expansion for a small action set
  -> target-network counterfactual Q estimates mixed across depths
  -> KL-regularized improved policy
  -> train policy/value/minimax advantage through dueling Q
  -> stronger closed model
```

FCPI 的策略迭代原理与 head 类型无关。入口读取 checkpoint 的架构后，由架构实现定义 rollout 排序，以及通用 policy/value/Q 目标到自身 heads 的投影；缺少某个 head 时不生成该 head 的空目标。

反事实 value expansion 在深度 `h` 的估计为：

```text
q_h(s,a) = (-1)^h * V_ref(s_h)
```

若 `s_h` 已终局，则使用精确 WDL。最终使用截断 lambda 混合：

```text
Q_expand = sum[h=1..H-1] (1-lambda) * lambda^(h-1) * q_h
           + lambda^(H-1) * q_H
```

所有候选先达到最小深度。每批候选具有目标平均展开深度，即这一批允许使用的 Value 评价预算。达到最小深度后，分支优先级由相邻 Value 的 Bellman residual、不同深度 Q 的变化和根节点竞争性共同决定；预算优先分配给高优先级分支，单个分支受最大深度约束。

该预算控制器用目标平均深度替代固定 residual/change 阈值。目标平均深度控制总成本，分支优先级控制成本分布，因此模型变强或局面分布改变时无需重新猜测绝对阈值。终局分支和最大深度可能使实际平均深度低于目标，运行汇总会报告预算利用率。它只执行批量 principal rollout，不维护 tree、visits 或 UCB，因此属于自适应 value expansion，而不是 MCTS。

### resnet_pv_linear FCPI

该架构只更新已有的 policy/value：

- closed self-play 通过 TD(lambda) 生成 value target。
- TD(lambda) 先作用于完整轨迹；随后每局按模型 state 去重，并对超长轨迹执行固定上限的均匀无放回 position 采样，避免错误长局按长度获得线性权重。
- policy 给出合法招法先验与反事实候选。
- 小候选集执行自适应多步 value expansion，后续 rollout 按冻结 Policy 排序。
- 实际走出的动作额外融合轨迹 return。
- 未执行反事实评价的合法动作以 `V_ref(s)` 为基线，避免从 policy target 中消失。
- KL 正则策略目标训练 policy，TD(lambda) target 训练 value。

该架构不存在 advantage head，因此 FCPI 不生成或猜测 advantage。

### resnet_pva_gad FCPI

该架构的三个 head 分工为：

- policy 给出当前候选分布。
- value 学习当前局面的 side-to-move return。
- minimax advantage 学习 `Q(s,a)-V(s)∈[-2,0]`，与 Value 组合成边 Q 先验，并以一个伪访问进入 MCTS 利用项。

反事实 Q 独立融合：

```text
Q_dueling(s,a) = clip(V_ref(s) + A_ref(s,a), -1, 1)
Q_target(s,a) = eta * Q_expand(s,a) + (1-eta) * Q_dueling(s,a)
V_target(s) = max_a Q_target(s,a)
A_target(s,a) = clip(Q_target(s,a) - V_target(s), -2, 0)
```

策略改进目标使用：

```text
pi_plus(a|s) proportional to
pi_ref(a|s)^rho * exp(Q_target(s,a) / temperature)
```

冻结的 current target、policy KL 和 candidate/current arena 共同限制单轮变化。arena 与 acceptance 使用 paired-game 结果；Stockfish comment 负责监督初始模型，FCPI 的新增训练信号来自自对战终局、TD return 和模型反事实后继评价。

### 优先级

先用已有 `resnet_pv_linear` checkpoint 验证 P/V 路径；`resnet_pva_gad` 完成大规模 commented-PGN 监督训练后，再独立验证 P/V/A 路径。

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
- 监督数据或自学习轨迹保存与架构定义一致的 action-value target。
- loss 中通过 `Q=V+A` 加入 masked dueling Q loss。
- 独立诊断统计 advantage、ranking 与 action-value 校准误差。
- MCTS 使用 `V+A` 初始化边 Q，并在已访问节点上按一个伪访问与 MCTS_Q 平滑融合。

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
- 如果有 uncertainty head，记录 value NLL、校准曲线、U_e 与 regret / search gain 的相关性。
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
