# ChessAI / Gadidae

Gadidae 是一个实验性国际象棋神经网络引擎项目。当前主线由三部分组成：

- `resnet_pv_linear`：ResNet + linear policy/value，使用 `resnet_pv_linear_18_planes` state encoding 和 `alphazero_64x73` move encoding。
- `resnet_pva_gad`：residual geometry transformer + source-destination policy/value/advantage，使用 `resnet_pva_gad_square_tokens` state encoding 和 `sd_64x64_underpromo9` move encoding。
- `search.py`：模型 policy/value 加 MCTS，支持 `closed` 和 `only-mcts` 两种 search type。

---

## 1. 目录结构

```text
ChessAI/
  data/
    games.pgn
    games-pv-linear.h5
    games-pva-gad.h5
    openings.bin
    openings.gen.bin
    openings.bal.bin
    user-pgn/
    lichess/
    runs/fcpi_YYYYMMDD_HHMMSS_id/
  models/
    chessnet.pth
    candidate.pth
    champion.pth
    runs/fcpi_YYYYMMDD_HHMMSS_id/
    stockfish/
  docs/
    model-design.md
    tactic-design.md
  src/
    acceptance.py
    analyze.py
    architectures.py
    arena.py
    checkpoint_io.py
    config.py
    data.py
    decision.py
    evaluator.py
    fcpi.py
    game_rules.py
    gui.py
    inspection.py
    model.py
    move_codecs.py
    opening_book.py
    preprocess.py
    search.py
    simulator.py
    standardize.py
    state_codecs.py
    stadium.py
    teacher.py
    train.py
    uci_engine.py
  run_opening.sh
  run_fcpi.sh
  run_simulator.vbs
  run_stadium.vbs
  .bot/
    setup_lichess_bot.sh
    run_lichess_bot.sh
    stop_lichess_bot.sh
  requirements.txt
```

主要目录：

- `data/games.pgn`：原始 PGN 训练来源。
- `data/*.h5`：`preprocess.py` 生成的监督训练数据，按架构区分。
- `data/openings.bin`：外部 Polyglot 源书。
- `data/openings.gen.bin`：从 PGN 生成的均势开局书。
- `data/openings.bal.bin`：验证后保留的均势开局书。
- `data/user-pgn/`：手工导入、分析、评注 PGN 的工作目录。
- `.bot/`：Lichess Bot 管理脚本，以及安装后生成的 lichess-bot 源码、运行配置、PID、PGN 和日志。
- `data/runs/<run-id>/`：一次训练 run 的日志、PID、标注 HDF5、arena FEN、trace 和 summary。
- `models/*.pth`：可加载的模型 checkpoint。
- `models/runs/<run-id>/`：一次训练 run 的 `current.pth` 和 `candidate_iter_*.pth`。
- `models/stockfish/`：Linux / Windows 的 UCI 教师机。
- `docs/`：模型和 tactic 的设计文档。
- `src/`：训练、搜索、对战、数据处理和 UCI/GUI 入口。

`src` 文件：

- `acceptance.py`：根据 arena paired-game 结果执行 candidate gate。
- `analyze.py`：用 UCI 引擎分析 PGN，生成 `.cmt` 和带评价批注的 `_cmt.pgn`。
- `architectures.py`：架构注册表，定义每个架构的 state encoding、move encoding 和 HDF5 schema。
- `arena.py`：两个模型 paired games 对战，输出胜负统计、PGN 和 trace。
- `checkpoint_io.py`：checkpoint 原子写回和备份工具。
- `config.py`：默认路径、设备、训练参数、搜索参数和平台化 UCI 路径。
- `data.py`：supervised HDF5 schema 校验和 PyTorch Dataset。
- `decision.py`：模型到 search 的决策适配层；按 `arch_type` 选择 state codec、move codec、MCTS profile 和架构特有搜索信号。
- `evaluator.py`：batched neural inference，供 MCTS 批量评估叶子局面。
- `fcpi.py`：按 checkpoint 架构分派独立 FCPI 进化公式，执行 closed self-play、反事实目标训练与 arena gate。
- `game_rules.py`：统一实际终局、三次重复、五十回合、胜负结果和终止原因的判定。
- `gui.py`：Simulator 与 Stadium 共用的 Tk 棋盘绘制、坐标映射、翻转、局面文本和 PGN 保存界面。
- `inspection.py`：检查 `preprocess.py` 生成的 supervised HDF5 是否符合对应架构。
- `model.py`：模型结构、架构识别、checkpoint 加载和保存。
- `move_codecs.py`：招法编码、解码、合法招法概率映射和 action size。
- `opening_book.py`：从 PGN 生成开局书、验证开局书均势性、为 arena 生成 paired opening specs。
- `preprocess.py`：从 PGN 生成指定架构的 supervised HDF5。
- `search.py`：单局面模型直出和 MCTS 搜索。
- `simulator.py`：局面模拟与模型分析 GUI，支持双方走子、实时候选、FEN 和 PGN 操作。
- `standardize.py`：规范化 checkpoint 结构，并按权重结构识别架构。
- `state_codecs.py`：棋盘状态编码，按架构注册。
- `stadium.py`：单盘 UCI 引擎对战观察 GUI，支持任意两个标准 UCI 引擎。
- `teacher.py`：UCI 教师机封装，提供走法评分、regret 计算和 sqlite cache。
- `train.py`：指定架构的监督训练入口。
- `uci_engine.py`：Gadidae 的 UCI 引擎外壳，供 lichess-bot 或 GUI 引擎前端调用。

Stockfish 默认路径按平台选择：

```text
Linux:   models/stockfish/stockfish
Windows: models/stockfish/stockfish.exe
```

---

## 2. 环境准备

Python 依赖：

```bash
python -m pip install -r requirements.txt
```

Linux CPU 环境示例：

```bash
apt update
apt install -y python3.10-venv python3-pip git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU 版 PyTorch 可单独安装：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Windows PowerShell 可把 README 里的多行 bash 命令改成单行，并把 `--device cuda` 改成 `--device cpu` 做轻量测试。

---

## 3. 架构与数据格式

| 架构 | state encoding | move encoding | HDF5 datasets | 训练入口 | search profile |
| --- | --- | --- | --- | --- | --- |
| `resnet_pv_linear` | `resnet_pv_linear_18_planes` | `alphazero_64x73` | `states`, `moves`, `values` | supervised + FCPI | policy/value + MCTS |
| `resnet_pva_gad` | `resnet_pva_gad_square_tokens` | `sd_64x64_underpromo9` | `states`, `moves`, `values`, `adv_moves`, `adv_values` | supervised + FCPI | policy/value/advantage + MCTS |

每个 HDF5 都带有：

```text
arch_type
state_encoding
move_encoding
target_schema
has_cmt
```

训练前按目标架构重新 preprocess。`resnet_pv_linear` 也需要显式使用 `--arch-type resnet_pv_linear` 生成 HDF5。

`resnet_pva_gad` 的 `gad` 表示 geometry / attention / dueling。该架构使用：

- square-token board state：64 个格子 token 加 side-to-move / castling / en-passant 全局特征
- residual geometry transformer trunk
- static geometry relation bias
- global-context dynamic relation bias
- source-destination policy head
- global-token value head
- non-positive minimax dueling advantage head

架构相关训练参数：

- `--channels` 默认值为 `128`。对 `resnet_pv_linear`，它表示卷积通道数；对 `resnet_pva_gad`，它表示 square/global token embedding 与 attention hidden size。
- `--blocks` 默认值为 `10`。对 `resnet_pv_linear`，它表示 ResNet residual block 数；对 `resnet_pva_gad`，它表示 geometry attention block 数。

`resnet_pv_linear` 使用 18-plane packbits state。`resnet_pva_gad` 使用紧凑 square-token state，HDF5 中 `states` 形状为 `(N, 67)`。

`standardize.py` 用于规范化 checkpoint 结构，并按权重结构自动识别模型类型：

```bash
python src/standardize.py --model models/chessnet.pth
```

---

## 4. Opening Book

从 PGN 生成均势开局书：

```bash
bash run_opening.sh data/games.pgn 50000 data/openings.gen.bin
```

等价展开：

```bash
python src/opening_book.py \
  --pgn data/games.pgn \
  --uci models/stockfish/stockfish \
  --output data/openings.gen.bin \
  --max-abs-cp 80 \
  --book-plies 8 \
  --min-fens 50000 \
  --uci-depth 10 \
  --uci-movetime-ms 0 \
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --log-every 1000
```

验证已有 Polyglot book：

```bash
python src/opening_book.py \
  --verify data/openings.bin \
  --uci models/stockfish/stockfish \
  --output data/openings.bal.bin \
  --max-abs-cp 80 \
  --book-plies 8 \
  --min-fens 50000 \
  --uci-depth 12 \
  --uci-movetime-ms 0 \
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --log-every 1000
```

核心参数：

- `--book-plies`：开局展开到第几个半回合。
- `--min-fens`：输出 `.bin` 在该深度可展开出的 unique opening state 下限。
- `--max-abs-cp`：Stockfish 白方视角评价绝对值上限。
- `--uci-depth` / `--uci-movetime-ms` / `--uci-threads` / `--uci-hash-mb`：传给 Stockfish。

`--opening-book ""` 在 arena 中表示从标准初始局面开始。

---

## 5. 监督数据预处理

### 5.1 resnet_pv_linear

```bash
python src/preprocess.py \
  --input data/games.pgn \
  --output data/games-pv-linear.h5 \
  --arch-type resnet_pv_linear \
  --has-cmt 1 \
  --chunk-size 32768 \
  --compression lzf \
  --max-games 2000000 \
  --random-select \
  --log-every 10000
```

生成内容：

```text
states
moves
values
```

`resnet_pv_linear` 的 `--has-cmt` 默认值为 `1`。在该架构下：

- `--has-cmt 1`：`policy` 来自 PGN 主线走法，`value` 来自 PGN 局面评价批注并转换为 side-to-move 视角。
- `--has-cmt 0`：`policy` 来自 PGN 主线走法，`value` 来自对局结果并转换为 side-to-move 视角。

### 5.2 resnet_pva_gad

```bash
python src/preprocess.py \
  --input data/ccrl.pgn \
  --output data/games-pva-gad.h5 \
  --arch-type resnet_pva_gad \
  --has-cmt 1 \
  --chunk-size 32768 \
  --compression lzf \
  --max-games 2000000 \
  --random-select \
  --log-every 10000
```

`resnet_pva_gad` 的 `--has-cmt` 默认值为 `1`。在该架构下：

- `--has-cmt 1`：`policy` 来自 PGN 主线走法，`value` 来自 PGN 局面评价批注，`advantage` 来自走前/走后评价差值。
- `--has-cmt 0`：`policy` 来自 PGN 主线走法，`value` 来自对局结果，`advantage` head 不参与监督。

`--has-cmt 1` 读取 PGN 主线走法前后节点的 CCRL / Stockfish 风格评价批注：

```text
{+0.60/16 193s}
{(Rd8) +0.71/14 171s}
```

批注数值按白方视角解析。`values` 使用当前节点评价并转换为 side-to-move 视角；缺少 `+/-` 数值的节点按 $0$ 处理。`adv_values` 先把走前、走后评价分别转换到 value head 的 $[-1,1]$ 尺度，再计算当前走子方视角的变化：

$$
\begin{aligned}
V_{before} &= \tanh(score_{before}/3) \\
Q_{after} &= \tanh(score_{after}/3) \\
A_{target} &= \operatorname{clip}(\min(0,Q_{after}-V_{before}),-2,0)
\end{aligned}
$$

状态包含当前行棋方，$V$、$Q$ 与 $A$ 都采用该行棋方视角。确定性零和转移满足 $Q(s,a)=-V(T(s,a))$，最优性定义为 $V(s)=\max_{a\in\mathcal L(s)}Q(s,a)$，因此 $A(s,a)=Q(s,a)-V(s)$ 必为非正数。正差值按评估噪声处理并写为 $0$；$V,Q\in[-1,1]$，所以完整 $A$ 范围为 $[-2,0]$。

使用 `--has-cmt 1` 时，整盘主线至少需要一个可解析评价批注；整盘没有评价批注的 game 会被跳过。使用 `analyze.py --pgn-comments` 可以把普通 PGN 转成带 UCI 评价批注的 `<name>_cmt.pgn`，再生成 `resnet_pva_gad` 的 HDF5。

---

## 6. 监督训练

### 6.1 resnet_pv_linear

```bash
python src/train.py \
  --data data/games-pv-linear.h5 \
  --out models/chessnet.pth \
  --arch-type resnet_pv_linear \
  --device cuda \
  --channels 128 \
  --blocks 10 \
  --epochs 10 \
  --batch-size 512 \
  --workers 4 \
  --max-steps 80000 \
  --save-every 5000 \
  --log-every 100
```

### 6.2 resnet_pva_gad

```bash
python src/train.py \
  --data data/games-pva-gad.h5 \
  --out models/chessnet-pva-gad.pth \
  --arch-type resnet_pva_gad \
  --device cuda \
  --channels 128 \
  --blocks 10 \
  --epochs 10 \
  --batch-size 512 \
  --workers 4 \
  --max-steps 80000 \
  --save-every 5000 \
  --log-every 100
```

训练 loss 参数：

- `resnet_pv_linear`：`--value-weight` 控制 Value loss；该架构只注册 Policy/Value 训练参数。
- `resnet_pva_gad`：`--value-weight` 控制 Value loss；`--dueling-q-weight` 控制组合 Dueling Q loss。该项使用 $Q_{pred}=\operatorname{clip}(V_{pred}+A_{pred},-1,1)$ 对走后局面价值进行监督，使 V 与 A 通过同一动作价值联合训练；`has_cmt=0` 时该项关闭。

本地 CPU smoke test：

```powershell
python src/train.py --data data/games-pv-linear.h5 --out models/test-train.pth --arch-type resnet_pv_linear --device cpu --channels 16 --blocks 1 --epochs 1 --batch-size 8 --workers 0 --max-steps 1 --save-every 0 --log-every 1
```

---

## 7. 单局面 Search

模型直出：

```bash
python src/search.py \
  --model models/champion.pth \
  --fen startpos \
  --device cpu \
  --search-type closed \
  --root-topn 16
```

MCTS：

```bash
python src/search.py \
  --model models/champion.pth \
  --fen startpos \
  --device cuda \
  --search-type only-mcts \
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 64 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --virtual-loss 0.0 \
  --repetition-policy-penalty 0.15 \
  --instant-mate-first \
  --root-topn 16
```

### 7.1 共同搜索流程

`decision.py` 根据 checkpoint 的 `arch.type` 选择架构对应的 state codec、move codec 和 decision profile。`closed` 只执行一次网络推理；`only-mcts` 在每个展开节点读取模型输出，使用 PUCT 选择叶子并把 leaf value 逐层变号回传。`resnet_pv_linear` profile 使用该架构的 Policy/Value，`resnet_pva_gad` profile 使用该架构的 Policy/Value/Advantage；搜索结果按所选 profile 呈现对应字段。

对已经访问的子节点，父节点视角的利用项为：

$$
Q_{select}(s,a)=-Q(child)
$$

$$
Q(child)=\frac{value\_sum(child)}{visits(child)}
$$

负号来自 value 与 $Q$ 始终采用当前局面 side-to-move 视角。PUCT 选择分数为：

$$
\operatorname{PUCT}(s,a)=Q_{select}(s,a)
+C(s)P(s,a)\frac{\sqrt{N(s)+N_{virtual}(s)+1}}
{1+N(s,a)+N_{virtual}(s,a)}
-virtual\_loss\,N_{virtual}(s,a)
$$

其中 $P(s,a)$ 是模型 policy 在合法招法上的归一化先验。搜索结束后，根节点策略由 $visits+prior$ 归一化得到；实际 top1 先按该策略确定，再经过可选的 IMF/RPP 决策层。$Q_{MCTS}$ 来自叶子 value 回传，决策组件不会修改树内 prior、visits 或 $Q$。

搜索先运行 `--mcts-min-sims`；该值为 $0$ 时使用 $\max(mcts\_batch\_size,mcts\_sims/4)$。之后根据根节点访问分布熵、前两名 visits 接近程度和 $Q$ 接近程度得到 `uncertainty`，在最小模拟数与 `--mcts-sims` 软上限之间动态确定目标。`--movetime-ms` 是独立的硬时间预算。

### 7.2 resnet_pv_linear

- `closed`：Policy 头给出合法招法排序；Value 头给出根局面的 side-to-move 估值，并供输出及 RPP 优势判断使用。
- `only-mcts`：Policy 头提供每个节点的 $P(s,a)$；Value 头评价叶子并形成回传的 $Q_{MCTS}$。
- 未访问子节点采用父节点 FPU：

$$
\operatorname{FPU}(s)=\operatorname{clip}
\left(Q(s)-fpu\_reduction\sqrt{visited\_policy\_mass},-1,1\right)
$$

$$
Q_{select,unvisited}(s,a)=\operatorname{FPU}(s)
$$

### 7.3 resnet_pva_gad

- `closed`：与 `resnet_pv_linear` 的直出路径一样，由 Policy 排序、Value 评价根局面；Advantage 头不参与 closed 排序。
- `only-mcts`：Policy 与 Value 的职责不变；节点展开时按 move codec 为每条边保存非正 $A(s,a)$，并构造固定网络先验：

$$
Q_{prior}(s,a)=\operatorname{clip}\left(V_{net}(s)+A_{net}(s,a),-1,1\right)
$$

- 未访问边使用 $Q_{prior}-fpu\_penalty$。已访问边通过 Q 先验权重 $\lambda$ 与实际叶子回传均值平滑融合：

$$
Q_{select}(s,a)=
\frac{N(s,a)Q_{mcts}(s,a)+\lambda Q_{prior}(s,a)}
{N(s,a)+\lambda}
$$

**项目内部定义：伪访问（pseudo-visit）**是从统计学 pseudocount / prior sample weight 迁移来的名称，表示 $Q_{prior}$ 在上述加权平均中占有 $\lambda$ 个样本单位的先验权重。它不是 MCTS 真正遍历过的边，不增加 `visits`，不写入 `value_sum`，也不代表一次叶子观测；该名称是项目内部定义，不把它视为 MCTS 领域的公认学术术语。

- 当前 $\lambda=1$，即一个伪访问。网络 $Q$ 的实际占比为 $1/(N+1)$：未访问和低访问边更多使用 $V+A$，随后平滑收敛到 $Q_{MCTS}$。固定的小权重不假设当前模型已经具备可靠的置信度校准。
- $A$ 不进入 PUCT 的探索项、MCTS `value_sum` 或叶子回传。它通过低访问量下的利用项影响后续 visits，而不是永久叠加到搜索 $Q$，也不会被当作新的叶子观测重复回传。

若模型能够输出经过校准的动作 Q 认知不确定性，并且 MCTS 节点记录叶子回传方差，可以改用精度加权：

$$
Q_{select}(s,a)=
\frac{
Q_{prior}(s,a)/\sigma^2_{net}(s,a)
+N(s,a)Q_{mcts}(s,a)/\sigma^2_{mcts}(s,a)
}{
1/\sigma^2_{net}(s,a)
+N(s,a)/\sigma^2_{mcts}(s,a)
}
$$

它等价于状态和动作相关的先验权重：

$$
\lambda(s,a)=\frac{\sigma^2_{mcts}(s,a)}{\sigma^2_{net}(s,a)}
$$

网络越确定，$Q$ 先验保留越久；MCTS 回传越稳定，搜索均值接管越快。Policy entropy、Advantage 差距和 $|Q_{MCTS}-Q_{prior}|$ 都不直接等价于认知不确定性，当前实现不使用这些量替代 $\sigma_{net}$。

### 7.4 参数含义

- `--search-type closed`：执行模型直出路径。
- `--search-type only-mcts`：执行架构对应的 MCTS profile；`--mcts-sims 0` 时退化为模型直出。
- `--mcts-sims`：MCTS simulations 软上限，表示模拟次数。
- `--mcts-min-sims`：动态预算下的最小模拟次数。
- `--movetime-ms`：完整 search 的时间预算。
- `--c-puct`：PUCT 初始探索常数。
- `--c-puct-base` / `--c-puct-factor`：随访问数增长的 C-PUCT schedule。
- `--fpu-reduction`：未访问节点的 FPU 折减。
- `--virtual-loss`：batched MCTS 中对同批已选路径施加的额外临时分数惩罚。搜索始终使用 virtual visits 做路径占位；设为 `0` 只关闭额外惩罚，负数与非有限值统一归一化为 `0`。
- `--repetition-policy-penalty`：RPP（Repetition Policy Penalty）。己方优势时，软惩罚己方当前招法直接完成第三次重复的候选，以及会让对手存在一手合法回复完成第三次重复的候选；对手是否申请和棋不属于 RPP。取值范围为 $[0,1]$，默认 $0$。搜索结束后，决策层复制搜索 policy 作为排序分数，从目标招法的分数中直接减去 $penalty\cdot\max(value,0)$ 并钳到 $0$；其他招法保持原值，决策层不重新归一化，也不修改搜索 policy。该构件依赖完整走棋历史，只有 FEN 时无法识别此前重复次数。
- `--instant-mate-first`：IMF（Instant Mate First）。搜索结束后若根节点存在一步将杀，在所有一步杀中采用搜索 policy 最高的一手，并把该手的决策排序分数设为 `1`；其他招法保持原值，决策层不重新归一化，也不修改搜索 policy。默认关闭，可用 `--no-instant-mate-first` 显式关闭。IMF 与 RPP 独立执行；合法对局中的一步将杀既不会完成历史重复，也不会留下对手回复，因此两者自然不会冲突。

搜索输出中的 `p`/`mcts_p` 始终表示组件介入前的搜索概率，`decision_score` 表示 IMF/RPP 处理后的临时排序分数；实际走法按 `decision_score` 选择。

动态 C-PUCT：

$$
c_{puct}+c_{puct\_factor}
\log\left(\frac{parent\_visits+c_{puct\_base}+1}{c_{puct\_base}}\right)
$$

---

## 8. Arena 模型比较

```bash
python src/arena.py \
  --candidate models/candidate.pth \
  --baseline models/champion.pth \
  --device cuda \
  --games 100 \
  --games-in-flight 100 \
  --max-plies 240 \
  --opening-book data/openings.gen.bin \
  --book-plies 8 \
  --max-book-positions 50000 \
  --search-type closed \
  --sims 0 \
  --movetime-ms 0 \
  --pgn-output data/runs/arena.pgn \
  --log-every 1
```

arena 行为：

- paired openings：同一个起始局面交换颜色各下一局。
- candidate 与 baseline 可使用不同架构；arena 给双方同一 search budget；每个模型按 checkpoint 的 `arch_type` 通过 `decision.py` 选择 decision profile、state codec、move codec 和架构特有的搜索信号。
- `--search-type closed` 使用模型直出的 policy/value。
- `--search-type only-mcts` 使用模型 policy/value 加 MCTS。
- `--games 100` 使用 50 个 unique start positions。
- `--games-in-flight` 控制同时驻留的棋局数。arena 只加载一份 candidate 和一份 baseline，并按当前行棋模型合并多个棋局的神经网络推理。
- 每盘棋保有独立的棋盘、MCTS tree、visits、Q、dynamic target 和时间预算；`--mcts-batch-size` 仍表示每盘 MCTS 单轮选择的叶子数量，随后再跨棋局合并推理。
- candidate 与 baseline 始终使用相同的 search type、sims、movetime、C-PUCT、FPU 和 MCTS batch 参数。
- 输出 W/D/L、net wins、score、score confidence interval 和 Elo diff。
- arena 与 acceptance 使用模型对战结果作为独立比较与 gate 信号。
- `--pgn-output` 保存棋谱。
- `--trace-output` 保存逐手 search 细节。
- Arena 在当前位置实际达到三次重复或五十回合条件后判和；仅存在下一手可申请和棋的走法时继续对局，使 RPP 能在决策层处理该候选。
- 从标准初始局面开始时使用 `--opening-book ""`。
- 查看逐手 search 时加入 `--trace-output data/runs/arena.trace.jsonl --pgn-comments --trace-root-topn 12`。

Windows CPU smoke test：

```powershell
python src/arena.py --candidate models/candidate.pth --baseline models/champion.pth --device cpu --games 2 --games-in-flight 2 --max-plies 20 --opening-book "" --search-type closed --sims 0 --movetime-ms 0 --pgn-output data/user-pgn/arena_smoke.pgn --log-every 1
```

---

## 9. FCPI

FCPI（Folded Counterfactual Policy Iteration，折叠式反事实策略迭代）在采样和验收时均可使用 `closed` 模型。它通过自对战回报与少量自适应多步反事实评价生成训练信号，运行过程不调用 UCI 教师机。

启动后台 run：

```bash
bash run_fcpi.sh
```

查看最新 run：

```bash
RUN_ID="$(ls -td data/runs/fcpi_* | head -1 | xargs basename)"
tail -f "data/runs/$RUN_ID/info.log"
```

`fcpi.py` 先读取 `--model` checkpoint 的 `arch.type`，再选择该架构的 FCPI 实现与参数表。FCPI 的 TD、反事实 value expansion、KL policy improvement 和 arena gate 原理通用；每个架构单独定义 rollout 排序及目标到自身 heads 的投影：

- `resnet_pv_linear`：rollout 按 Policy 排序；多深度 Value 混合形成反事实 Q；训练 policy/value。
- `resnet_pva_gad`：rollout 按 Policy+Advantage 排序；多深度 Value 与 $V(s)+A(s,a)$ 融合形成候选 Q；$V_{target}$ 取已评价候选 Q 的最大值，$A_{target}=Q_{target}-V_{target}\in[-2,0]$，并通过组合 Dueling Q loss 训练 policy/value/advantage。

FCPI 一次只训练一个 checkpoint。入口先识别 `arch.type`，再由对应架构注册同一组用户参数、提供自己的默认值，并把例如 `--counterfactual-topk` 映射为该架构内部的独立参数变量。`resnet_pva_gad` 还会注册仅供其 Advantage 路径使用的参数；这些参数不会出现在 `resnet_pv_linear` 命令中。

### 9.1 共同目标

记本轮开始时冻结的 `current.pth` 参数为 $\theta_0$，从它初始化并接受梯度更新的 candidate 参数为 $\theta$。$\pi_0$、$V_0$、$A_0$ 均来自冻结模型；Value、$Q$ 和 Advantage 使用 side-to-move 视角。

完整轨迹的终点目标为：

$$
G_T =
\begin{cases}
z_T, & \text{正式终局} \\
V_0(s_T), & \text{达到 max-plies}
\end{cases}
$$

其中 $z_T$ 取值为 $-1$、$0$ 或 $1$。TD($\lambda$) 从轨迹末端反向计算：

$$
G_t = \operatorname{clip}\left(
-\left[(1-\lambda_{TD})V_0(s_{t+1})+\lambda_{TD}G_{t+1}\right],
-1,1
\right)
$$

Value 监督目标为：

$$
V^*(s_t)=G_t
$$

对根局面 $s$ 的候选招法 $a$，先走出该招法，再沿冻结模型在后续局面的确定性 top1 继续 rollout。深度 $d$ 的 Value 转换回根节点行棋方视角：

$$
e_d(s,a)=(-1)^dV_0(s_d)
$$

若 $s_d$ 已经终局，则使用真实终局值。不同深度的结果按 `counterfactual-lambda` 混合：

$$
Q_{CF}(s,a)=
(1-\lambda_{CF})
\sum_{d=1}^{D-1}\lambda_{CF}^{d-1}e_d(s,a)
+\lambda_{CF}^{D-1}e_D(s,a)
$$

得到架构对应的改进值 $\widehat Q$ 后，Policy 目标统一为：

$$
\pi^*(a\mid s)=
\frac{
\pi_0(a\mid s)^\rho
\exp\left((\widehat Q(s,a)-V_0(s))/\tau_\pi\right)
}{
\sum_b \pi_0(b\mid s)^\rho
\exp\left((\widehat Q(s,b)-V_0(s))/\tau_\pi\right)
}
$$

`prior-power` 对应 $\rho$，`policy-temperature` 对应 $\tau_\pi$。当前脚本中二者分别为 $1.0$ 和 $0.25$。

### 9.2 resnet_pv_linear

该架构输出 Policy 与 Value：

$$
f_\theta(s)=\left(\pi_\theta(s),V_\theta(s)\right)
$$

自对局 behavior 覆盖全部合法招法：

$$
\mu(a\mid s)=
(1-\epsilon)
\frac{\pi_0(a\mid s)^{1/T_b}}
{\sum_b\pi_0(b\mid s)^{1/T_b}}
+\frac{\epsilon}{|\mathcal A(s)|}
$$

当前脚本使用 $T_b=1.0$、$\epsilon=0.03$。反事实候选取 Policy 排名前 $K=6$ 的招法，并始终包含实际走出的招法。候选初值为：

$$
Q_c(s,a)=Q_{CF}(s,a)
$$

实际走出的招法再混入整盘 TD 回报：

$$
Q'_c(s,a_t)=(1-w_p)Q_c(s,a_t)+w_pG_t
$$

当前 `played-return-weight` 为 $w_p=0.5$。最终改进值为：

$$
\widehat Q(s,a)=
\begin{cases}
Q'_c(s,a), & a\text{ 是反事实候选} \\
V_0(s), & a\text{ 未展开}
\end{cases}
$$

令 $p_\theta$ 为 candidate 在合法招法上的 Policy softmax：

$$
L_{policy}=-\mathbb E_s\sum_a\pi^*(a\mid s)\log p_\theta(a\mid s)
$$

$$
L_V=\operatorname{SmoothL1}\left(V_\theta(s),G_t\right)
$$

代码使用反向 KL，约束 candidate 不要一次偏离冻结 Policy 过远：

$$
L_{KL}=D_{KL}\left(p_\theta(\cdot\mid s)\parallel\pi_0(\cdot\mid s)\right)
$$

$$
H(p_\theta)=-\sum_a p_\theta(a\mid s)\log p_\theta(a\mid s)
$$

总损失为：

$$
L_{PV}=L_{policy}+L_V+0.05L_{KL}-0.001H(p_\theta)
$$

### 9.3 resnet_pva_gad

该架构输出 Policy、Value 与 Advantage：

$$
f_\theta(s)=\left(\pi_\theta(s),V_\theta(s),A_\theta(s,a)\right)
$$

状态包含当前行棋方，PVA 使用 minimax Dueling 定义：

$$
Q(s,a)=-V(T(s,a))
$$

$$
V(s)=\max_{a\in\mathcal L(s)} Q(s,a)
$$

$$
A(s,a)=Q(s,a)-V(s)\in[-2,0]
$$

模型动作价值为：

$$
Q_0(s,a)=\operatorname{clip}\left(V_0(s)+A_0(s,a),-1,1\right)
$$

自对局排序分数与 behavior 为：

$$
R_0(s,a)=\log\pi_0(a\mid s)+\beta A_0(s,a)
$$

$$
\mu(a\mid s)=
(1-\epsilon)\operatorname{softmax}\left(R_0(s,a)/T_b\right)
+\frac{\epsilon}{|\mathcal A(s)|}
$$

当前脚本使用 $\beta=0.5$、$T_b=1.0$、$\epsilon=0.03$。反事实候选取 $R_0$ 排名前 $K=8$ 的招法，并包含实际走法；后续 rollout 同样选择 $R_0$ 的 top1。

反事实展开结果与冻结模型的 dueling Q 混合：

$$
Q_c(s,a)=\eta Q_{CF}(s,a)+(1-\eta)Q_0(s,a)
$$

当前 `successor-weight` 为 $\eta=0.75$。实际走法继续混入 TD 回报：

$$
Q'_c(s,a_t)=(1-w_p)Q_c(s,a_t)+w_pG_t
$$

反事实候选形成最终动作值，轨迹 TD 回报只通过实际走法的 $Q'_c$ 进入。PVA 的改进 Value 取已评价候选动作最大值：

$$
V^*(s)=\max_{a\in C(s)}Q'_c(s,a)
$$

候选招法的非正 Advantage 目标为：

$$
A^*(s,a)=\operatorname{clip}\left(Q'_c(s,a)-V^*(s),-2,0\right)
$$

未展开动作使用冻结模型的 $V_0+A_0$，并截断到不高于 $V^*(s)$，作为 Policy soft target 的保守背景值。训练不单独回归 A，而是组合动作价值：

$$
Q_\theta(s,a)=\operatorname{clip}\left(V_\theta(s)+A_\theta(s,a),-1,1\right)
$$

$$
L_Q=\operatorname{SmoothL1}\left(Q_\theta(s,a),Q'_c(s,a)\right)
$$

总损失为：

$$
L_{PVA}=L_{policy}+L_V+0.5L_Q+0.05L_{KL}-0.001H(p_\theta)
$$

$L_Q$ 只在反事实候选上计算，并通过 $V+A$ 同时更新 Value head、Advantage head 与共享 trunk。Policy、Value、Q 和 Advantage 目标由冻结的 $\theta_0$ 与反事实展开预先生成，训练时作为常量。

### 9.4 参数更新与重复 state 聚合

两个架构都使用 AdamW，当前脚本学习率为 $10^{-5}$、weight decay 为 $10^{-4}$，梯度范数裁剪为 $1.0$：

$$
\theta\leftarrow\operatorname{AdamW}\left(\theta,\nabla_\theta L,10^{-5},10^{-4}\right)
$$

反事实目标生成后，相同编码 state 在本轮训练数据内统一聚合。Value 与 Policy 目标取平均：

$$
\overline G(s)=\frac{1}{N_s}\sum_{i=1}^{N_s}G_i(s)
$$

$$
\overline\pi^*(a\mid s)=\frac{1}{N_s}\sum_{i=1}^{N_s}\pi_i^*(a\mid s)
$$

同一候选招法的 $Q$ 只在实际展开过该候选的记录之间取平均；`resnet_pva_gad` 随后使用聚合后的 $V$ 和 $Q$ 重新计算 $A=Q-V$，保持 Dueling 定义。Arena 与 acceptance 不进入梯度方程，只负责 candidate 晋升 gate。

`run_fcpi.sh` 默认使用 50% startpos 与 50% 均势开局书起点。开局书数量不足时，每轮完整洗牌后循环使用全部开局，使复用次数均匀；arena 继续使用唯一 paired openings。常用参数：

```text
MODEL=models/chessnet.pth
ITERATIONS=10
GAMES_PER_ITER=1000
GAMES_IN_FLIGHT=100
MAX_PLIES=240
POSITIONS_PER_GAME=200
STARTPOS_FRACTION=0.50
EPOCHS=15
TRAIN_MAX_STEPS=2000
EVAL_GAMES=400
EVAL_SEARCH_TYPE=closed
EVAL_SIMS=0
EVAL_MIN_NET_WINS=4
EVAL_REPETITION_POLICY_PENALTY=0.15
EVAL_INSTANT_MATE_FIRST=true
EVAL_HISTORY_GAMES=100
EVAL_HISTORY_POOL_SIZE=3
```

自适应多步参数只输入一次。以下为 `resnet_pv_linear` 默认值；加载 `resnet_pva_gad` 时，同名参数会采用该架构在脚本中定义的默认值：

```text
COUNTERFACTUAL_TOPK=6
COUNTERFACTUAL_MIN_PLIES=2
COUNTERFACTUAL_MAX_PLIES=6
COUNTERFACTUAL_TARGET_AVERAGE_PLIES=4.0
COUNTERFACTUAL_LAMBDA=0.80
```

所有候选至少展开 `MIN_PLIES`。之后根据相邻局面的 Bellman residual、不同深度的 $Q$ 变化和候选在根节点的竞争性计算优先级，将剩余评价预算集中到高优先级分支，单个分支最多展开到 `MAX_PLIES`。`TARGET_AVERAGE_PLIES` 直接控制每批候选的目标平均展开深度；终局分支或最大深度限制可能使实际值略低。`LAMBDA` 以几何权重混合各深度 $Q$；该过程不维护 visits 和搜索树。

`info.log` 中的 `counterfactual summary` 会输出 `average_depth`、`depth_histogram`、`target_average_plies` 和 `budget_utilization`，用于确认实际计算量与目标预算。

自对战先在完整轨迹上计算 TD($\lambda$)，再按局处理训练 position：同一局中编码后完全相同的模型 state 只保留一次；超过 `POSITIONS_PER_GAME` 时均匀无放回采样，并保持选中 position 的时间顺序。生成反事实目标后，本轮全部训练记录按编码 state 聚合，平均 policy/value 及对应候选目标，避免常见开局前缀按重复次数主导 loss。日志与 `summary.json` 会记录采样、起点复用和聚合数量。

达到 `MAX_PLIES` 时，正式规则已经判定的将死、逼和、子力不足、重复或五十回合结果照常使用；仍未终局的轨迹使用当前模型对尾部局面的连续 Value bootstrap，不强制改写为和棋。

输出：

```text
data/runs/<run-id>/info.log
data/runs/<run-id>/pid
data/runs/<run-id>/fcpi_iter_*.h5
data/runs/<run-id>/summary.json
models/runs/<run-id>/current.pth
models/runs/<run-id>/initial.pth
models/runs/<run-id>/candidate_iter_*.pth
```

每轮 candidate 训练完成后直接与本 run 的 `current.pth` 进行 paired arena。主赛达到 `EVAL_MIN_NET_WINS` 后才运行历史稳定赛；主赛失败时跳过历史赛。

`initial.pth` 保存本 run 的输入模型。历史池只收录 `initial.pth` 和真正晋升过的 `candidate_iter_*.pth`，并排除当前模型与重复 checkpoint。池超过 `EVAL_HISTORY_POOL_SIZE` 时，在时间轴上均匀保留里程碑。Candidate 分别与每个选中的历史模型使用成对开局和相同搜索参数比赛，每组都必须满足 `net_wins >= EVAL_MIN_NET_WINS`。主赛和历史赛复用同一个门槛；所有历史项通过后，candidate 原子写入本 run 的 `current.pth`。`EVAL_HISTORY_GAMES=0` 或 `EVAL_HISTORY_POOL_SIZE=0` 可关闭历史稳定赛。

历史列表、去重、时间轴抽样和稳定性判断均由 `fcpi.py` 管理。`arena.py` 只执行指定的两模型对战并返回统计；`acceptance.py` 根据 `net_wins` 和同一个 `EVAL_MIN_NET_WINS` 计算无状态 gate，不保存历史池。

`EVAL_REPETITION_POLICY_PENALTY` 与 `EVAL_INSTANT_MATE_FIRST` 同时作用于主 Arena 和历史稳定赛，并适用于所有已注册模型架构以及 `closed`、`only-mcts` 两种 search type。它们不会进入 FCPI 自对战采样或梯度目标。

FCPI 提供 policy/value/advantage 的可学习改进信号；实际棋力提升仍由 arena 结果确认。目标网络误差、自对战分布偏移和有限反事实候选都会影响结果，因此单轮 loss 下降不等价于 Elo 上升。

---

## 10. Simulator 与 Stadium

启动 Simulator：

```bash
python src/simulator.py
```

带模型启动：

```bash
python src/simulator.py \
  --model models/champion.pth \
  --device cpu \
  --search-type only-mcts \
  --mcts-sims 10000 \
  --mcts-min-sims 0 \
  --mcts-batch-size 32 \
  --movetime-ms 0 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --repetition-policy-penalty 0.15 \
  --instant-mate-first \
  --progress-interval-ms 750 \
  --root-topn 8
```

Windows 一键启动：

```text
run_simulator.vbs
```

Simulator 功能：

- `Settings`：选择模型、设备和 search 参数。
- 棋盘：双方轮流走子，自动显示当前局面候选。
- `Close / Open`：暂停或恢复自动候选。
- `Reset FEN`：载入 FEN；空输入恢复 `startpos`。
- `Import PGN` / `Save PGN`：导入或保存主线棋谱。
- `Board state`：复制 FEN 和 PGN。

启动 Stadium：

```bash
python src/stadium.py \
  --white-uci "python src/uci_engine.py --model models/candidate0.pth --device cpu --search-type only-mcts --mcts-sims 1000 --mcts-min-sims 1000 --mcts-batch-size 32 --progress-interval-ms 750" \
  --white-options '{}' \
  --white-movetime-ms 10000 \
  --black-uci "models/stockfish/stockfish" \
  --black-options '{"Threads": 4, "Hash": 512}' \
  --black-movetime-ms 1000 \
  --delay-ms 300 \
  --max-plies 240
```

Windows 一键启动：

```text
run_stadium.vbs
```

Stadium 的 `Settings` 分别配置白方与黑方的 UCI command、UCI options JSON 和每步思考时间，也可配置显示间隔与最大 ply；`Start FEN` 设置单盘起始局面。双方资源可以不同，例如为 Gadidae 分配更长思考时间、为 Stockfish 设置独立的 `Threads` 与 `Hash`。`Moves` 在当前行棋方尚未落子时实时显示其最新 MultiPV，`*` 标记当前暂定首选；`UCI analysis` 同步显示该候选的 score、深度、nodes 和 PV。收到 `bestmove` 后棋盘落子并清空旧分析，面板随后切换到下一行棋方。Stadium 始终运行一盘可视化对局，并保留人工启动、暂停和停止。

---

## 11. UCI Engine

启动：

```bash
python src/uci_engine.py \
  --model models/champion.pth \
  --device cpu \
  --search-type only-mcts \
  --mcts-sims 100 \
  --mcts-batch-size 64 \
  --movetime-ms 1000 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --repetition-policy-penalty 0.15 \
  --instant-mate-first \
  --progress-interval-ms 750 \
  --multipv 1 \
  --score-scale 1000
```

手动协议测试：

```text
uci
isready
position startpos moves e2e4 e7e5
go movetime 1000
quit
```

常用 UCI option：

```text
ModelPath
Device
SearchType
MCTSSims
MCTSMinSims
MCTSBatchSize
MoveTimeMS
MoveOverheadMS
MinMoveTimeMS
MaxMoveTimeMS
TimeDivisor
IncrementFraction
CPuct
CPuctBase
CPuctFactor
FPUReduction
VirtualLoss
RepetitionPolicyPenalty
InstantMateFirst
ProgressIntervalMS
MultiPV
RootTopN
ScoreScale
LogSearch
```

UCI 输出使用标准 `info ... score cp ... multipv ... pv ...` 与 `bestmove ...`。`nodes` 表示 MCTS simulations，`depth` 表示本次搜索使用的 NN batch 数。`ProgressIntervalMS` 控制 Gadidae 搜索期间发送实时 MultiPV `info` 的最小时间间隔。

---

## 12. Lichess Bot

安装 lichess-bot：

```bash
bash .bot/setup_lichess_bot.sh
```

导入 token：

```bash
read -rsp "LICHESS_TOKEN: " LICHESS_TOKEN
echo
export LICHESS_TOKEN
test -n "$LICHESS_TOKEN" && echo token_ok
```

首次升级 BOT 账号：

```bash
UPGRADE_BOT=1 MODEL=models/candidate.pth DEVICE=cpu SEARCH_TYPE=closed MCTS_SIMS=0 bash .bot/run_lichess_bot.sh
```

常驻运行：

```bash
MODEL=models/candidate.pth \
DEVICE=cpu \
SEARCH_TYPE=closed \
MCTS_SIMS=0 \
MOVETIME_MS=1000 \
MAX_MOVETIME_MS=3000 \
CHALLENGE_ONLY_BOT=true \
ALLOW_MATCHMAKING=false \
RATE_LIMITING_DELAY_MS=1000 \
bash .bot/run_lichess_bot.sh
```

允许人类挑战：

```bash
MODEL=models/candidate.pth DEVICE=cpu SEARCH_TYPE=only-mcts MCTS_SIMS=100 MOVETIME_MS=0 MAX_MOVETIME_MS=5000 CHALLENGE_ONLY_BOT=false RATE_LIMITING_DELAY_MS=1000 bash .bot/run_lichess_bot.sh
```

查看最新 run：

```bash
ls -lt .bot/runs
RUN_ID="$(ls -td .bot/runs/lichess_* | head -1 | xargs basename)"
echo "$RUN_ID"
tail -f ".bot/runs/$RUN_ID/info.log"
```

停止：

```bash
bash .bot/stop_lichess_bot.sh "$RUN_ID"
```

启动脚本通过 `.bot/lichess-bot.lock` 保证同一项目同时只有一个 lichess-bot 实例。`RATE_LIMITING_DELAY_MS` 写入 lichess-bot 的 `engine.rate_limiting_delay`，表示提交一步棋后等待的毫秒数；默认 `1000`，主要用于降低短时间内连续 move 请求的概率。

Lichess API 要求请求串行执行。收到 HTTP `429` 后至少等待完整一分钟，再继续 API 活动；部分 endpoint 的限制会持续更久。等待期间保留当前进程，反复停止和重启会重新建立 event stream，并不会缩短服务端限制。event/game stream 的 `timeout=15` 是连接健康检查时间，保持该值即可。持续遇到 `429` 时检查后台进程数量，并确认代理出口 IP 没有被其他程序共享使用。官方说明：<https://lichess.org/page/api-tips>。

云端使用本地代理时，先保持 SSH 反向隧道：

```bash
ssh -N -R 127.0.0.1:10090:127.0.0.1:10090 MS
```

云端 shell 导入代理：

```bash
export http_proxy=http://127.0.0.1:10090
export https_proxy=http://127.0.0.1:10090
export HTTP_PROXY=http://127.0.0.1:10090
export HTTPS_PROXY=http://127.0.0.1:10090
export no_proxy=localhost,127.0.0.1,::1
```

---

## 13. PGN 分析

```bash
python src/analyze.py \
  --input data/user-pgn/1.pgn \
  --uci models/stockfish/stockfish \
  --uci-depth 14 \
  --uci-movetime-ms 0 \
  --uci-multipv 5 \
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --critical-threshold-cp 50 \
  --top-moves 3
```

输出同目录同名 `.cmt`，例如：

```text
data/user-pgn/1.cmt
```

报告包含 Summary、Main Reading、Critical Moves、Full Move Table 和关键候选行。

生成可供 `resnet_pva_gad` 预处理读取的评注版 PGN：

```bash
python src/analyze.py \
  --input data/user-pgn/1.pgn \
  --uci models/stockfish/stockfish \
  --uci-depth 14 \
  --uci-movetime-ms 0 \
  --uci-multipv 5 \
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --pgn-comments \
  --pgn-columns 88
```

默认 PGN 输出：

```text
data/user-pgn/1_cmt.pgn
```

评注中的 `{+0.23}` 使用白方视角，单位为 pawn。继续生成 HDF5：

```bash
python src/preprocess.py \
  --input data/user-pgn/1_cmt.pgn \
  --output data/games-pva-gad.h5 \
  --arch-type resnet_pva_gad \
  --chunk-size 32768 \
  --compression lzf \
  --log-every 10000
```

---

## 14. 数据检查

监督 HDF5：

```bash
python src/inspection.py --path data/games-pv-linear.h5
python src/inspection.py --path data/games-pva-gad.h5
```

`inspection.py` 只检查 `preprocess.py` 生成的 supervised HDF5；FCPI 的架构专属数据由 `fcpi.py` 在生成和读取时校验。

---

## 15. 控制台输出

`--log-every` 控制 step / game / labeling 进度行：

```text
preprocess progress: ...
train step: ...
fcpi self-play start: ...
fcpi game ...
fcpi train: ...
arena game ...
```

阶段汇总使用 JSON：

```text
preprocess summary:
fcpi self-play finished: ...
arena game summary:
arena: finished ...
```

---

## 16. 后台进程管理

查看 FCPI：

```bash
ps -ef | grep -E "fcpi.py" | grep -v grep
```

查看某个 run：

```bash
RUN_DIR=data/runs/<run-id>
PID="$(cat "$RUN_DIR/pid")"
ps -fp "$PID"
tail -f "$RUN_DIR/info.log"
```

停止某个 run：

```bash
RUN_DIR=data/runs/<run-id>
PID="$(cat "$RUN_DIR/pid")"
kill -TERM "$PID"
sleep 5
ps -fp "$PID"
```

结束指定残留 PID：

```bash
kill -TERM <pid>
sleep 3
kill -KILL <pid>
```

结束全部 Stockfish：

```bash
pkill -KILL -f "models/stockfish/stockfish"
```

查看 Lichess bot：

```bash
ps -ef | grep -E "lichess-bot.py|uci_engine.py" | grep -v grep
```

停止 Lichess bot：

```bash
bash .bot/stop_lichess_bot.sh <run-id>
```

`tail -f` 只是日志查看进程，`Ctrl+C` 退出日志查看。

---

## 17. 空间维护

查看占用：

```bash
du -h --max-depth=2 data models | sort -h
```

清理 Python 缓存：

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

清理 FCPI run：

```bash
rm -rf data/runs/fcpi_*
rm -rf models/runs/fcpi_*
```

清理模型备份：

```bash
rm -f models/*.bak_*
```

清理 AutoDL 回收站：

```bash
rm -rf ~/.local/share/Trash/files/*
rm -rf ~/.local/share/Trash/info/*
rm -rf /root/autodl-tmp/.Trash-0/files/*
rm -rf /root/autodl-tmp/.Trash-0/*
```
