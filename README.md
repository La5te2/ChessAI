# ChessAI / Gadidae

Gadidae 是一个实验性国际象棋神经网络引擎项目。当前主线由三部分组成：

- `resnet_pv_linear`：ResNet + linear policy/value，使用 `resnet_pv_linear_18_planes` state encoding 和 `alphazero_64x73` move encoding。
- `resnet_pva_gad`：residual geometry transformer + source-destination policy/value/advantage，使用 `square_tokens` state encoding 和 `sd_64x64_underpromo9` move encoding。
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
    runs/offline_pv_YYYYMMDD_HHMMSS_pid/
  models/
    chessnet.pth
    candidate.pth
    champion.pth
    runs/offline_pv_YYYYMMDD_HHMMSS_pid/
    stockfish/
  docs/
    model-design.md
    tactic-design.md
  src/
    acceptance.py
    analyze.py
    architectures.py
    arena.py
    board.py
    checkpoint_io.py
    config.py
    data.py
    decision.py
    evaluator.py
    inspection.py
    model.py
    move_codecs.py
    offline_pv.py
    opening_book.py
    preprocess.py
    search.py
    standardize.py
    state_codecs.py
    teacher.py
    train.py
    uci_engine.py
  run_opening.sh
  run_offline_pv.sh
  run_board.vbs
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
- `data/lichess/`：lichess-bot 源码、配置和运行日志。
- `data/runs/<run-id>/`：一次训练 run 的日志、PID、标注 HDF5、arena FEN、trace 和 summary。
- `models/*.pth`：可加载的模型 checkpoint。
- `models/runs/<run-id>/`：一次训练 run 的 `current.pth` 和 `candidate_iter_*.pth`。
- `models/stockfish/`：Linux / Windows 的 UCI 教师机。
- `docs/`：模型和 tactic 的设计文档。
- `src/`：训练、搜索、对战、数据处理和 UCI/GUI 入口。

`src` 文件：

- `acceptance.py`：arena 结果 gate 和 move-quality gate。
- `analyze.py`：用 UCI 引擎分析 PGN，生成 `.cmt` 和带评价批注的 `_cmt.pgn`。
- `architectures.py`：架构注册表，定义每个架构的 state encoding、move encoding 和 HDF5 schema。
- `arena.py`：两个模型 paired games 对战，输出胜负统计、PGN、trace 和 UCI move-quality。
- `board.py`：Tk GUI 棋盘，支持加载模型、下棋、翻转棋盘和实时搜索显示。
- `checkpoint_io.py`：checkpoint 原子写回和备份工具。
- `config.py`：默认路径、设备、训练参数、搜索参数和平台化 UCI 路径。
- `data.py`：supervised HDF5 schema 校验和 PyTorch Dataset。
- `decision.py`：模型到 search 的决策适配层；按 `arch_type` 选择 state codec、move codec、MCTS profile 和架构特有搜索信号。
- `evaluator.py`：batched neural inference，供 MCTS 批量评估叶子局面。
- `inspection.py`：检查 `preprocess.py` 生成的 supervised HDF5 是否符合对应架构。
- `model.py`：模型结构、架构识别、checkpoint 加载和保存。
- `move_codecs.py`：招法编码、解码、合法招法概率映射和 action size。
- `offline_pv.py`：`resnet_pv_linear` 的 offline teacher-guided 训练流程和 arena gate。
- `opening_book.py`：从 PGN 生成开局书、验证开局书均势性、为 arena 生成 paired opening specs。
- `preprocess.py`：从 PGN 生成指定架构的 supervised HDF5。
- `search.py`：单局面模型直出和 MCTS 搜索。
- `standardize.py`：规范化 checkpoint 结构，并按权重结构识别架构。
- `state_codecs.py`：棋盘状态编码，按架构注册。
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
| `resnet_pv_linear` | `resnet_pv_linear_18_planes` | `alphazero_64x73` | `states`, `moves`, `values` | supervised + offline-pv | policy/value + MCTS |
| `resnet_pva_gad` | `square_tokens` | `sd_64x64_underpromo9` | `states`, `moves`, `values`, `adv_moves`, `adv_values` | supervised | policy/value/advantage + MCTS |

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
- dueling advantage head

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

`--opening-book ""` 在 arena / offline-pv eval 中表示从标准初始局面开始。

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

批注数值按白方视角解析。`values` 使用当前节点评价并转换为 side-to-move 视角；缺少 `+/-` 数值的节点按 `0` 处理。`adv_values` 使用当前节点评价与当前走法完成后的评价计算当前走子方视角的变化：

```text
白方: after - before
黑方: before - after
target: tanh(min(0, delta) / 3)
```

正差值写为 `0`，表示该走法保住了当前走子方的评价；负差值进入 advantage 监督目标。

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

- `resnet_pv_linear`：`--value-weight` 控制 value loss 权重；该架构没有 advantage head，`--advantage-weight` 不参与该架构训练。
- `resnet_pva_gad`：`--value-weight` 控制 value loss 权重；`--advantage-weight` 控制 advantage loss 权重。训练数据为 `has_cmt=0` 时，advantage loss 关闭。

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
  --root-topn 16
```

参数含义：

- `--search-type closed`：使用模型 policy/value 输出候选。
- `--search-type only-mcts`：使用模型 policy/value 加 MCTS。
- `resnet_pv_linear`：search 使用 policy/value。
- `resnet_pva_gad`：search 使用 policy/value，并把 model advantage 作为架构特有搜索信号用于未访问节点估值。
- `--mcts-sims`：MCTS simulations 软上限，表示模拟次数。
- `--mcts-min-sims`：动态预算下的最小模拟次数。
- `--movetime-ms`：完整 search 的时间预算。
- `--c-puct`：PUCT 初始探索常数。
- `--c-puct-base` / `--c-puct-factor`：随访问数增长的 C-PUCT schedule。
- `--fpu-reduction`：未访问节点的 FPU 折减。
- `--virtual-loss`：batched MCTS 中同批路径的临时占位。

动态 C-PUCT：

```text
c_puct + c_puct_factor * log((parent_visits + c_puct_base + 1) / c_puct_base)
```

---

## 8. Arena 模型比较

```bash
python src/arena.py \
  --candidate models/candidate.pth \
  --baseline models/champion.pth \
  --device cuda \
  --games 100 \
  --workers 10 \
  --max-plies 240 \
  --opening-book data/openings.gen.bin \
  --book-plies 8 \
  --max-book-positions 50000 \
  --search-type closed \
  --sims 0 \
  --movetime-ms 0 \
  --uci models/stockfish/stockfish \
  --uci-depth 16 \
  --uci-movetime-ms 0 \
  --uci-multipv 1 \
  --uci-threads 1 \
  --uci-hash-mb 512 \
  --teacher-cache data/runs/arena_teacher_cache.sqlite \
  --quality-loss-cap-cp 1000 \
  --pgn-output data/runs/arena.pgn \
  --log-every 1
```

arena 行为：

- paired openings：同一个起始局面交换颜色各下一局。
- candidate 与 baseline 可使用不同架构；arena 给双方同一 search budget；每个模型按 checkpoint 的 `arch_type` 通过 `decision.py` 选择 decision profile、state codec、move codec 和架构特有的搜索信号。
- `--search-type closed` 使用模型直出的 policy/value。
- `--search-type only-mcts` 使用模型 policy/value 加 MCTS。
- `--games 100` 使用 50 个 unique start positions。
- `--workers` 控制并行对局和 Stockfish move-quality 分析。
- 输出 W/D/L、net wins、score、Elo diff、ACPL、accuracy、inaccuracy、mistake、blunder。
- `--pgn-output` 保存棋谱。
- `--trace-output` 保存逐手 search 细节。
- 从标准初始局面开始时使用 `--opening-book ""`。
- 查看逐手 search 时加入 `--trace-output data/runs/arena.trace.jsonl --pgn-comments --trace-root-topn 12`。

Windows CPU smoke test：

```powershell
python src/arena.py --candidate models/candidate.pth --baseline models/champion.pth --device cpu --games 1 --workers 1 --max-plies 20 --opening-book "" --search-type closed --sims 0 --movetime-ms 0 --uci models/stockfish/stockfish.exe --uci-depth 8 --uci-multipv 1 --uci-threads 1 --teacher-cache data/user-pgn/arena_cache.sqlite --pgn-output data/user-pgn/arena_smoke.pgn --log-every 1
```

---

## 9. Offline-PV

`offline_pv.py` 当前目标架构为 `resnet_pv_linear`。输入模型使用 `models/chessnet.pth`，每次运行创建独立 run 目录：

```text
data/runs/offline_pv_YYYYMMDD_HHMMSS_pid/
models/runs/offline_pv_YYYYMMDD_HHMMSS_pid/
```

启动：

```bash
bash run_offline_pv.sh
```

查看日志：

```bash
ls -lt data/runs
RUN_ID="$(ls -td data/runs/offline_pv_* | head -1 | xargs basename)"
tail -f "data/runs/$RUN_ID/info.log"
```

脚本核心参数：

```text
MODEL=models/chessnet.pth
FEN_SOURCE=data/games.pgn
ITERATIONS=5
POSITIONS_PER_ITER=10000
PARALLEL=10
SAMPLE_TOPK=6
REWARD_SCALE_CP=600
TEACHER_POLICY_WEIGHT=0.10
TEACHER_RANK_WEIGHT=0.10
TEACHER_VALUE_WEIGHT=0.50
TEACHER_POLICY_TEMP_CP=150
UCI_DEPTH=16
UCI_MULTIPV=1
EPOCHS=30
TRAIN_MAX_STEPS=2000
LR=0.00003
VALIDATION_POSITIONS=1000
EVAL_GAMES=200
EVAL_SEARCH_TYPE=closed
EVAL_SIMS=0
EVAL_MIN_NET_WINS=0
```

流程：

1. 从 `--fen-source` 按顺序抽取 FEN。
2. 当前模型用 sim=0 policy top-k 提出 action。
3. Stockfish 为 action 生成 centipawn score。
4. offline-pv 使用 score 生成 `tanh(score_cp / reward_scale_cp)` reward。
5. offline-pv 使用 score 生成候选动作 soft policy。
6. offline-pv 使用 score 排序生成 pairwise ranking loss。
7. actor 使用 `reward - value` advantage 更新 policy。
8. critic 学习当前候选策略的期望 reward。
9. teacher validation 检查 policy regret 与 value 误差。
10. arena gate 比较 `candidate_iter_*.pth` 与本 run 的 `current.pth`。
11. 通过 gate 后写回本 run 的 `current.pth`。

replay 参数：

```text
ARENA_REPLAY_WINDOW=1
ARENA_REPLAY_POSITIONS=-1
ARENA_REPLAY_POSITIONS_PER_ITER=10000
```

例如 iter5 使用前四轮各最多 10000 个 arena FEN：

```bash
ARENA_REPLAY_WINDOW=4
ARENA_REPLAY_POSITIONS=-1
ARENA_REPLAY_POSITIONS_PER_ITER=10000
```

主要输出：

```text
data/runs/<run-id>/info.log
data/runs/<run-id>/pid
data/runs/<run-id>/offline_iter_*.h5
data/runs/<run-id>/arena_fens_iter_*.txt
data/runs/<run-id>/arena_trace_iter_*.jsonl
data/runs/<run-id>/summary.json
models/runs/<run-id>/current.pth
models/runs/<run-id>/candidate_iter_*.pth
```

---

## 10. GUI Board

启动 GUI：

```bash
python src/board.py
```

带模型启动：

```bash
python src/board.py \
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
  --progress-interval-ms 750 \
  --root-topn 8
```

Windows 一键启动：

```text
run_board.vbs
```

GUI 功能：

- `Settings`：选择模型、设备和 search 参数。
- `Simulator`：双方轮流走子，自动显示当前局面候选。
- `Close / Open`：暂停或恢复自动候选。
- `Play`：选择人类执白或执黑，AI 自动回复。
- `Reset FEN`：载入 FEN；空输入恢复 `startpos`。
- `Import PGN` / `Save PGN`：导入或保存主线棋谱。
- `Board state`：复制 FEN 和 PGN。

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
MultiPV
RootTopN
ScoreScale
LogSearch
```

UCI 输出使用标准 `info ... score cp ... multipv ... pv ...` 与 `bestmove ...`。`nodes` 表示 MCTS simulations，`depth` 表示本次搜索使用的 NN batch 数。

---

## 12. Lichess Bot

安装 lichess-bot：

```bash
bash setup_lichess_bot.sh
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
UPGRADE_BOT=1 MODEL=models/candidate.pth DEVICE=cpu SEARCH_TYPE=closed MCTS_SIMS=0 bash run_lichess_bot.sh
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
bash run_lichess_bot.sh
```

允许人类挑战：

```bash
MODEL=models/candidate.pth DEVICE=cpu SEARCH_TYPE=only-mcts MCTS_SIMS=100 MOVETIME_MS=0 MAX_MOVETIME_MS=5000 CHALLENGE_ONLY_BOT=false bash run_lichess_bot.sh
```

查看最新 run：

```bash
ls -lt data/lichess/runs
RUN_ID="$(ls -td data/lichess/runs/lichess_* | head -1 | xargs basename)"
echo "$RUN_ID"
tail -f "data/lichess/runs/$RUN_ID/info.log"
```

停止：

```bash
bash stop_lichess_bot.sh "$RUN_ID"
```

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

`inspection.py` 只检查 `preprocess.py` 生成的 supervised HDF5。Offline-PV 标注 HDF5 由 `offline_pv.py` 自身在生成、训练和 summary 阶段验证。

---

## 15. 控制台输出

`--log-every` 控制 step / game / labeling 进度行：

```text
preprocess progress: ...
train step: ...
offline label: ...
offline-pv train step: ...
arena worker ... game ...
arena quality worker ...:
```

阶段汇总使用 JSON：

```text
preprocess summary:
offline-pv label summary:
offline-pv teacher validation summary:
offline-pv arena summary:
arena game summary:
arena: finished ...
```

---

## 16. 后台进程管理

查看 offline-pv：

```bash
ps -ef | grep -E "offline_pv.py|stockfish" | grep -v grep
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
bash stop_lichess_bot.sh <run-id>
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

清理 offline-pv run：

```bash
rm -rf data/runs/offline_pv_*
rm -rf models/runs/offline_pv_*
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
