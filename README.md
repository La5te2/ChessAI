## 0. 项目目录约定

```text
ChessAI/
├── data/
│   ├── games.pgn
│   ├── games.h5
│   ├── openings.bin
│   ├── openings.bal.bin
│   ├── openings.gen.bin
│   └── runs/
│       └── reinforce_YYYYMMDD_HHMMSS_pid/
│           ├── info.log
│           ├── pid
│           ├── offline_iter_*.h5
│           └── summary.json
├── models/
│   ├── chessnet.pth
│   ├── runs/
│   │   └── reinforce_YYYYMMDD_HHMMSS_pid/
│   │       ├── current.pth
│   │       └── candidate_iter_*.pth
│   └── stockfish/
│       ├── stockfish      # Linux / 云端
│       └── stockfish.exe  # Windows / 本地
├── src/
│   ├── config.py
│   ├── chess_env.py
│   ├── move_encoder.py
│   ├── data.py
│   ├── model.py
│   ├── evaluator.py
│   ├── search.py
│   ├── uci_engine.py
│   ├── opening_book.py
│   ├── checkpoint_io.py
│   ├── preprocess.py
│   ├── inspection.py
│   ├── standardize.py
│   ├── train.py
│   ├── arena.py
│   ├── teacher.py
│   ├── reinforce.py
│   ├── analyze.py
│   └── board.py
├── run_reinforce.sh
├── setup_lichess_bot.sh
├── run_lichess_bot.sh
├── stop_lichess_bot.sh
└── requirements.txt
```

项目数据由监督训练 HDF5、actor-critic run 目录、教师缓存和模型 checkpoint 组成。

---

## 1. 功能概览

### 1.1 监督训练

`preprocess.py` 将 PGN 转换为监督训练 HDF5：

```text
states
moves
values
```

`train.py` 完成 policy/value 监督训练，并支持从已有模型权重继续训练。续训时 `--max-steps` 表示本轮追加训练步数，保存的 `global_step` 会在原 checkpoint 的基础上累加。

### 1.2 模型格式

`standardize.py` 将 checkpoint 整理为统一结构：

```text
model
arch
epoch
global_step
extra
```

模型写回采用临时文件、历史备份和原子替换。

### 1.3 Search

`search.py` 组合模型 policy、动态预算 MCTS、FPU、动态 C-PUCT 与主动 mate search 确定性选棋。

MCTS 根据根节点访问分布熵、候选访问差距和 Q 值差距计算不确定性，并在软上限范围内分配模拟次数。FPU 为未访问节点提供父局面相关的初始估值，动态 C-PUCT 随父节点访问量提高探索强度。主动 mate search 对根候选执行短深度强制将杀证明，可在预算内直接选择己方强制将杀首步。

### 1.4 模型验收

`arena.py` 使用 Polyglot opening book 生成配对局面，交换双方颜色完成 candidate 与 baseline 对局。`--workers` 对配对对局和 Stockfish move-quality 分析同时生效。

`opening_book.py` 使用 Stockfish 验证 Polyglot opening book，按局面分值筛选均势开局并写出新的 `.bin`。

Stockfish 分析双方实际落子并统计：

```text
W / D / L
net wins
ACPL
Accuracy
inaccuracy
mistake
blunder
```

candidate 的对局结果与走法质量共同构成验收条件。

### 1.5 Offline Actor-Critic

`reinforce.py` 是独立的 offline actor-critic 实验入口。脚本从 PGN 或带 `fens` dataset 的 HDF5 中按顺序抽取局面，模型用 sim=0 policy top-k 提出候选走法，Stockfish 为各 action 生成连续 reward。critic 学习当前候选策略的期望 reward，actor 使用 action reward 与 critic value 形成 advantage 并执行策略梯度更新。训练使用 entropy 与 KL reference；teacher validation 使用 Stockfish 在人类棋谱局面上验证 baseline 与 candidate 的 policy regret 与 value 误差。候选模型通过 arena 验收后写回本次 run 的 `current.pth`。

归档文件 `data/selflearn.py.bak_20260713_134805` 与 `data/regression.py.bak_20260713_134805` 保存此前的教师约束自学习和动态回归实现。

### 1.6 棋盘模拟器

`board.py` 启动后进入 `Simulator` 模式，双方按照当前行棋方轮流走子。

模型文件与搜索参数通过 `Settings` 统一应用。每次应用参数都会重新加载模型。处于 `Simulator` 模式且模型已载入时，当前局面会自动生成候选走法，并受 `movetime`、`mcts_sims`、mate 节点数等搜索预算约束。

`Play` 启动人机对弈，并提示选择：

```text
用户执白 / 用户执黑
沿用当前局面 / 使用 startpos
```

进入 Play 模式后，AI 在轮到自身颜色时自动行棋。每次自动回复提交一手，回合随后交还给用户。

`Simulator` 将棋盘恢复为双人模拟状态。

`Reset FEN` 输入框接收完整 FEN。空输入恢复 `startpos`。

### 1.7 PGN 棋谱分析

`analyze.py` 使用 Stockfish 按 PGN 主线逐手分析，输出同目录同名 `.cmt` 报告。报告包含 Summary、Main Reading、Critical Moves、Full Move Table 和关键行候选走法。`--critical-threshold-cp` 默认 50，贴合 `?!` 起点；mate 编码行会在摘要中单独说明。

### 1.8 UCI 引擎

`uci_engine.py` 是 ChessAI checkpoint 的 UCI 协议外壳。进程通过 stdin/stdout 接收 `uci`、`isready`、`setoption`、`position`、`go` 和 `quit`，加载指定 `.pth` 后调用当前 `search.py` 逻辑输出 `bestmove`。模型文件保持 `.pth` checkpoint 格式。

---

## 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

README 命令默认按 Linux / 云端 bash 写法展示。Windows PowerShell 可将多行命令改写为单行，并把设备设为 `--device cpu` 做本地轻量测试。

Stockfish 可执行文件按平台放置于：

```text
Linux:   models/stockfish/stockfish
Windows: models/stockfish/stockfish.exe
```

代码按当前平台选择默认文件名。云端 Linux 使用 `stockfish`，本地 Windows 使用 `stockfish.exe`。

Polyglot opening book 放置于：

```text
data/openings.bin
data/openings.bal.bin
data/openings.gen.bin
```

### 2.1 Opening Book 验证

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

`--verify` 使用 Stockfish 评估 Polyglot book 的可达开局。`abs(white_cp) <= --max-abs-cp` 的路径会写入输出 book。`--book-plies` 控制开局深度，`--min-fens` 控制输出 `.bin` 按同一开局深度可展开出的 unique opening state 下限；reinforce eval 和 arena 读取 book 时使用同一类 state 去分配起始局面。

Windows 本地 Stockfish 示例：

```powershell
python src/opening_book.py --verify data/openings.bin --uci models/stockfish/stockfish.exe --output data/openings.bal.local.bin --min-fens 200 --uci-depth 8 --uci-movetime-ms 0 --log-every 20
```

`--in-place` 会备份原 book 并写回原路径。

PGN 生成开局书：

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

`--pgn` 从 PGN 主线开局中筛选均势路径，并写入到达 endpoint FEN 的 Polyglot entry。默认输出路径为 `data/openings.gen.bin`。`--uci-depth`、`--uci-movetime-ms`、`--uci-threads` 和 `--uci-hash-mb` 传给 UCI 引擎；当前脚本按 Stockfish 常用选项配置。

脚本入口：

```bash
bash run_opening.sh data/games.pgn 50000 data/openings.gen.bin
```

`run_opening.sh` 也读取环境变量：`PGN`、`MIN_FENS`、`OUTPUT`、`UCI`、`MAX_ABS_CP`、`BOOK_PLIES`、`UCI_DEPTH`、`UCI_MOVETIME_MS`、`UCI_THREADS`、`UCI_HASH_MB` 和 `LOG_EVERY`。

Opening book 生成和验证会打印 `accepted_fens`、`readable_fens`、`unique_entries` 和拒绝统计。写出 `.bin` 后会重新展开校验，`readable_fens` 达到 `--min-fens` 后完成。`--opening-book ""` 在 reinforce eval 和 arena 中表示从标准初始局面开始。

### 2.2 控制台输出

训练、预处理、offline reinforce 和 arena 的进度信息直接打印到控制台。

`--log-every` 控制 step / move / preprocess game 进度行：

```text
train step: ...
offline reinforce train step: ...
arena quality: ...
arena quality worker ...:
preprocess progress: ...
```

`--log-every 0` 关闭这类进度行。

arena 每局结束打印一行；offline reinforce 每轮标注和训练打印进度行：

```text
arena worker ... game ...
```

阶段汇总使用 JSON：

```text
preprocess summary:
offline reinforce label summary:
offline reinforce arena summary:
arena game summary:
arena: finished 后输出最终 metrics JSON
resume validation metrics
```

---

## 3. PGN 预处理

```bash
python src/preprocess.py \
  --input data/games.pgn \
  --output data/games.h5 \
  --chunk-size 32768 \
  --compression lzf \
  --max-games 2000000 \
  --random-select \
  --log-every 10000
```

`--random-select` 从整个 PGN 文件范围内选择棋局。

---

## 4. 数据检查与 PGN 分析

监督训练数据：

```bash
python src/inspection.py \
  data/games.h5
```

Offline reinforce 标注数据：

```bash
python src/inspection.py \
  data/runs/<run-id>/offline_iter_001.h5
```

PGN 主线逐手分析：

```bash
python src/analyze.py \
  --input data/user-pgn/1.pgn \
  --uci models/stockfish/stockfish \
  --uci-depth 14 \
  --uci-multipv 5 \
  --uci-threads 4
```

Windows 本地 Stockfish 示例：

```powershell
python src/analyze.py --input data/user-pgn/1.pgn --uci models/stockfish/stockfish.exe --uci-depth 14 --uci-multipv 5 --uci-threads 4
```

输出文件使用输入文件同路径同名 `.cmt`，例如 `data/user-pgn/1.cmt`。报告包含开局信息、全局摘要、关键问题手、完整逐手表格和关键行候选走法。`--critical-threshold-cp` 控制进入 Critical Moves 的 regret 下限，默认 50；`--top-moves` 控制每个关键行展示的 Stockfish 候选数量，默认 3。

---

## 5. 模型标准化

```bash
python src/standardize.py \
  --model models/chessnet.pth
```

执行过程保留原模型备份，并将主文件写回统一 checkpoint 结构。

---

## 6. 基础监督训练

```bash
python src/train.py \
  --data data/games.h5 \
  --out models/chessnet.pth \
  --device cuda \
  --epochs 10 \
  --batch-size 512 \
  --workers 4 \
  --max-steps 80000 \
  --save-every 5000 \
  --log-every 100
```

---

## 7. 模型续训与自动验收

```bash
python src/train.py \
  --data data/games.h5 \
  --resume models/chessnet.pth \
  --device cuda \
  --max-steps 20000 \
  --batch-size 512 \
  --workers 4 \
  --save-every 5000 \
  --eval-games 100 \
  --eval-sims 200 \
  --eval-workers 1 \
  --eval-max-plies 240 \
  --eval-min-net-wins 5 \
  --eval-mcts-batch-size 64 \
  --eval-movetime-ms 10000 \
  --eval-search-type closed \
  --eval-c-puct 0.5 \
  --eval-c-puct-base 19652 \
  --eval-c-puct-factor 1.0 \
  --eval-fpu-reduction 0.15 \
  --eval-mcts-time-fraction 0.90 \
  --eval-mate-plies 0 \
  --eval-mate-topk 4 \
  --eval-mate-nodes 20000 \
  --eval-mate-hash-mb 16 \
  --eval-opening-book data/openings.bal.bin \
  --eval-book-plies 8 \
  --eval-max-book-positions 50000 \
  --uci models/stockfish/stockfish \
  --eval-uci-depth 10 \
  --eval-uci-multipv 8 \
  --eval-uci-threads 4 \
  --eval-uci-hash-mb 512 \
  --log-every 100
```

续训生成临时候选模型。candidate 同时满足对局结果、ACPL 和 Accuracy 条件后写回 `--resume` 指定路径。

`--max-steps` 表示本轮追加步数。例如 checkpoint 中 `global_step=80000`，续训传入 `--max-steps 100000`，候选模型保存时 `global_step=180000`。

本地 CPU 入口 smoke test 示例：`--device cpu --workers 0 --max-steps 1 --eval-games 0`。`--eval-games 0` 跳过 resume 验收，候选模型保存在输出路径。

---

## 8. 单局面 Search

```bash
python src/search.py \
  --model models/champion.pth \
  --fen startpos \
  --device cpu \
  --search-type mcts-mate \
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 32 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --mcts-time-fraction 0.90 \
  --mate-plies 3 \
  --mate-topk 4 \
  --mate-nodes 20000 \
  --mate-hash-mb 16 \
  --root-topn 16
```

`--search-type closed` 使用模型 policy top1。`--search-type only-mcts` 使用模型 policy/value 与 MCTS。`--search-type mcts-mate` 在 MCTS 后追加主动 mate search。`--mcts-sims` 表示 MCTS 软上限。`--movetime-ms` 表示完整 Search 的时间上限。动态探索常数为 `c_puct + c_puct_factor * log((parent_visits + c_puct_base + 1) / c_puct_base)`。`--fpu-reduction` 控制未访问节点相对父节点 Q 的初始折减。`--virtual-loss` 控制 batched MCTS 同一批次内已选路径的额外临时扣分，默认 `0.0`，保留 virtual visits 的轻度占位效果。`--mcts-time-fraction` 分配完整 Search 时间中供 MCTS 使用的比例。`--mate-plies` 控制走子方在几个己方走子内寻找强制将杀，`--mate-topk` 控制每个己方节点保留的候选数，`--mate-nodes` 控制 mate search 节点预算，`--mate-hash-mb` 控制单次 mate search 的 proof / transposition cache 内存预算。

### 8.1 UCI 引擎入口

```bash
python src/uci_engine.py \
  --model models/champion.pth \
  --device cuda \
  --search-type only-mcts \
  --mcts-sims 100 \
  --mcts-batch-size 64 \
  --movetime-ms 1000 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --mcts-time-fraction 1.0 \
  --mate-plies 0 \
  --mate-topk 4 \
  --mate-nodes 20000 \
  --mate-hash-mb 16
```

最小协议手测：

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
MCTSBatchSize
MoveTimeMS
CPuct
CPuctBase
CPuctFactor
FPUReduction
MCTSTimeFraction
MatePlies
MateTopK
MateNodes
MateHashMB
LogSearch
```

外部 GUI 或 lichess-bot 可通过 `setoption name ModelPath value models/champion.pth` 和 `go wtime ... btime ... winc ... binc ...` 控制模型与时钟预算。

### 8.2 Lichess Bot

`setup_lichess_bot.sh` 和 `run_lichess_bot.sh` 面向云端 Linux 使用。`data/lichess/` 存放官方 lichess-bot 仓库、虚拟环境、生成配置、日志、PID 和 PGN。

云端需要走本地代理时，先在本地保持反向隧道：

```bash
ssh -N -R 127.0.0.1:10090:127.0.0.1:10090 MS
```

然后在云端 shell 导入代理：

```bash
export http_proxy=http://127.0.0.1:10090
export https_proxy=http://127.0.0.1:10090
export HTTP_PROXY=http://127.0.0.1:10090
export HTTPS_PROXY=http://127.0.0.1:10090
export no_proxy=localhost,127.0.0.1,::1
```

创建并激活项目 Python 环境，供 `src/uci_engine.py` 加载 `.pth` 模型：

```bash
apt update
apt install -y python3.10-venv python3-pip git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install python-chess numpy h5py tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

安装或更新 lichess-bot：

```bash
bash setup_lichess_bot.sh
```

准备 Lichess OAuth token 后导入环境变量。输入提示出现后粘贴完整的 `lip_...`，再回车：

```bash
read -rsp "LICHESS_TOKEN: " LICHESS_TOKEN
echo
export LICHESS_TOKEN
test -n "$LICHESS_TOKEN" && echo token_ok
```

首次把 Lichess 账号升级为 BOT 账号：

```bash
UPGRADE_BOT=1 \
MODEL=models/candidate.pth \
DEVICE=cpu \
MCTS_SIMS=0 \
bash run_lichess_bot.sh
```

常驻云服务器 CPU 启动 bot：

```bash
MODEL=models/candidate.pth \
DEVICE=cpu \
MCTS_SIMS=0 \
MOVETIME_MS=1000 \
MAX_MOVETIME_MS=3000 \
CHALLENGE_ONLY_BOT=true \
ALLOW_MATCHMAKING=false \
bash run_lichess_bot.sh
```

允许人类账号挑战：

```bash
MODEL=models/candidate.pth DEVICE=cpu SEARCH_TYPE=only-mcts MCTS_SIMS=100 MOVETIME_MS=0 MAX_MOVETIME_MS=5000 MATE_HASH_MB=16 CHALLENGE_ONLY_BOT=false bash run_lichess_bot.sh
```

脚本会生成：

```text
data/lichess/lichess-bot/
data/lichess/runs/<run-id>/config.yml
data/lichess/runs/<run-id>/chessai_uci.sh
data/lichess/runs/<run-id>/info.log
data/lichess/runs/<run-id>/pid
data/lichess/runs/<run-id>/pgn/
```

默认挑战策略为 standard / casual / bot-only / 并发 1。常用覆盖参数：

```bash
CHALLENGE_CONCURRENCY=1
CHALLENGE_VARIANTS=standard
CHALLENGE_TIME_CONTROLS=blitz,rapid,classical
CHALLENGE_MODES=casual
CHALLENGE_ONLY_BOT=true
ALLOW_MATCHMAKING=false
```

查看日志：

```bash
ls -lt data/lichess/runs
RUN_ID="$(ls -td data/lichess/runs/lichess_* | head -1 | xargs basename)"
echo "$RUN_ID"
tail -f "data/lichess/runs/$RUN_ID/info.log"
```

停止 bot：

```bash
ps -ef | grep -E "lichess-bot.py|uci_engine.py" | grep -v grep
bash stop_lichess_bot.sh "$RUN_ID"
```

本地 Windows 只用于代码修改和 UCI 烟测；持续在线的 Lichess bot 进程放在云端运行。


---

## 9. 手动模型比较

```bash
python src/arena.py \
  --candidate models/chessnet2.pth \
  --baseline models/chessnet1.pth \
  --games 100 \
  --sims 200 \
  --workers 10 \
  --device cuda \
  --max-plies 240 \
  --opening-book data/openings.bal.bin \
  --book-plies 8 \
  --max-book-positions 50000 \
  --mcts-batch-size 64 \
  --movetime-ms 10000 \
  --search-type closed \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --mcts-time-fraction 0.90 \
  --mate-plies 0 \
  --mate-topk 4 \
  --mate-nodes 20000 \
  --mate-hash-mb 16 \
  --uci models/stockfish/stockfish \
  --uci-depth 10 \
  --uci-multipv 8 \
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --teacher-cache data/runs/arena_teacher_cache.sqlite \
  --quality-loss-cap-cp 1000 \
  --pgn-output data/runs/chessnet2_vs_chessnet1.pgn \
  --log-every 1000
```

arena 使用 paired openings：同一个起始局面会交换双方颜色各下一局；`--games 100` 会消耗 50 个 unique start positions。`--opening-book ""` 使用标准初始局面并轮换 candidate 执白/执黑。`--workers` 同时作用于模型对局和 Stockfish move-quality 分析；每个 worker 使用独立的 teacher cache 分片。`--pgn-output` 保存 arena 对局棋谱。命令输出配对对局结果、双方 ACPL / accuracy / blunder 质量指标，以及 `accepted` 所需的验收字段。

Windows:

```bash
python src/arena.py   --candidate models/candidate.pth   --baseline models/champion.pth   --device cpu   --games 1   --workers 1   --max-plies 240   --opening-book ""   --sims 10   --mcts-batch-size 32   --movetime-ms 0   --search-type closed   --c-puct 0.8   --c-puct-base 19652   --c-puct-factor 1.0   --fpu-reduction 0.15   --mcts-time-fraction 1   --mate-plies 0   --mate-topk 4   --mate-nodes 20000   --mate-hash-mb 16   --uci models/stockfish/stockfish.exe   --uci-depth 16   --uci-multipv 8   --uci-threads 1   --uci-hash-mb 256   --teacher-cache data/user-pgn/test4_cache.sqlite   --pgn-output data/user-pgn/test5.pgn   --log-every 1
```



---

## 10. Offline Actor-Critic 实验

推荐使用脚本启动：

```bash
bash run_reinforce.sh
```

脚本默认后台运行，并把输出写入本次 run 的日志：

```bash
tail -f data/runs/<run-id>/info.log
```

`run_reinforce.sh` 当前展开命令：

```bash
python src/reinforce.py \
  --run-id <reinforce-run-id> \
  --model models/chessnet.pth \
  --fen-source data/games.pgn \
  --uci models/stockfish/stockfish \
  --device cuda \
  --iterations 5 \
  --positions-per-iter 10000 \
  --parallel 10 \
  --source-min-ply 0 \
  --source-max-ply 160 \
  --arena-replay-window 1 \
  --arena-replay-positions -1 \
  --arena-replay-positions-per-iter 10000 \
  --sample-topk 6 \
  --reward-scale-cp 600 \
  --teacher-policy-weight 0.10 \
  --teacher-policy-temp-cp 150 \
  --actor-exploration-mix 0.05 \
  --advantage-clip 1.0 \
  --uci-depth 16 \
  --uci-movetime-ms 0 \
  --uci-multipv 1 \
  --uci-threads 1 \
  --uci-hash-mb 512 \
  --epochs 30 \
  --train-max-steps 2000 \
  --batch-size 256 \
  --train-workers 4 \
  --lr 0.00003 \
  --actor-weight 1.0 \
  --critic-weight 0.50 \
  --entropy-weight 0.003 \
  --kl-weight 0.05 \
  --validation-source data/games.pgn \
  --validation-positions 1000 \
  --validation-offset 100000 \
  --validation-min-ply 0 \
  --validation-max-ply 160 \
  --validation-topk 4 \
  --validation-workers 10 \
  --validation-uci-depth 16 \
  --validation-uci-movetime-ms 0 \
  --validation-uci-multipv 1 \
  --validation-uci-threads 1 \
  --validation-uci-hash-mb 512 \
  --eval-games 200 \
  --eval-sims 0 \
  --eval-workers 10 \
  --eval-max-plies 160 \
  --eval-opening-book data/openings.gen.bin \
  --eval-movetime-ms 0 \
  --eval-search-type closed \
  --eval-c-puct 0.5 \
  --eval-c-puct-base 19652 \
  --eval-c-puct-factor 1.0 \
  --eval-fpu-reduction 0.15 \
  --eval-mcts-time-fraction 1.0 \
  --eval-mate-plies 0 \
  --eval-mate-topk 4 \
  --eval-mate-nodes 20000 \
  --eval-mate-hash-mb 16 \
  --eval-uci-depth 16 \
  --eval-uci-multipv 1 \
  --eval-min-net-wins 0 \
  --eval-min-acpl-improvement 0.0 \
  --eval-min-accuracy-improvement 0.0 \
  --log-every 50 \
  --seed 2026
```

Offline reinforce 的状态由 `--fen-source` 提供：PGN 或带 `fens` dataset 的 HDF5。`--positions-per-iter` 表示每轮按顺序抽取的 FEN 数量，`--source-min-ply` 与 `--source-max-ply` 控制 PGN 局面范围。模型用 sim=0 policy top-k 提出 action，`--sample-topk` 控制候选数量，`--include-teacher-best` 把 Stockfish 最佳招加入 action 集合。教师机为每个 action 生成 `tanh(score_cp / reward_scale_cp)` 连续 reward。`--teacher-policy-temp-cp` 将候选 action 的 Stockfish score 转换为 softmax teacher policy；`--teacher-policy-weight` 控制该 soft policy 交叉熵对 policy head 的辅助牵引。critic 学习候选行为策略的期望 reward；actor 使用 `reward - value` advantage 执行策略梯度。`--actor-exploration-mix` 为已评价 action 分配均匀探索权重，`--advantage-clip` 控制 advantage 范围。训练使用 entropy 与 KL reference。`--arena-replay-window` 控制读取最近几轮 arena FEN，`--arena-replay-positions` 是窗口内总量上限，`--arena-replay-positions-per-iter` 是每个历史 iter 的读取上限；`-1` 表示总量不截断。`--validation-*` 从人类棋谱局面抽取验证集，并由 Stockfish 统计 baseline 与 candidate 的 top1 regret、max regret、teacher-best top-k 命中率、value MAE/RMSE/correlation/sign accuracy。验收对局通过 `--eval-opening-book` 使用开局书，并由 arena 调用 search 参数完成对战。

例如 iter5 使用前四轮各最多 10000 个 replay FEN：

```bash
ARENA_REPLAY_WINDOW=4
ARENA_REPLAY_POSITIONS=-1
ARENA_REPLAY_POSITIONS_PER_ITER=10000
```

`--seed` 控制 offline RL DataLoader 的 batch shuffle 与 arena opening book 洗牌。FEN 标注保持源文件顺序，固定 seed 便于复现实验的数据顺序和验收开局。

reinforce 的 arena gate 使用当前 run 的 `current.pth` 作为 baseline，`candidate_iter_*.pth` 作为 candidate。`result_ok` 要求 `net_wins >= --eval-min-net-wins`；`quality_ok` 要求 candidate 的 ACPL 低于 baseline 且 accuracy 高于 baseline，阈值由 `--eval-min-acpl-improvement` 和 `--eval-min-accuracy-improvement` 控制，脚本当前使用默认 `0.0`。arena trace 会写入 `data/runs/<run-id>/arena_trace_iter_*.jsonl`，去重后的 FEN 写入 `data/runs/<run-id>/arena_fens_iter_*.txt`。通过 gate 后，candidate 写回本次 run 的 `models/runs/<run-id>/current.pth`。

offline reinforce 每次运行都会按当前参数生成本次 run 的 `offline_iter_*.h5`；中断后的 run 目录保留为历史产物。

每次 offline reinforce 运行都会生成配对的独立 run 目录：

```text
data/runs/reinforce_YYYYMMDD_HHMMSS_pid/
models/runs/reinforce_YYYYMMDD_HHMMSS_pid/
```

固定 run-id 时：

```bash
RUN_ID=reinforce_experiment_001 bash run_reinforce.sh
```

主要输出：

```text
data/runs/<run-id>/info.log
data/runs/<run-id>/pid
data/runs/<run-id>/offline_iter_*.h5
data/runs/<run-id>/summary.json
models/runs/<run-id>/current.pth
models/runs/<run-id>/candidate_iter_*.pth
```

候选通过 arena 验收后写回本次 run 的 `current.pth`。

---

## 11. 后台进程管理

`run_reinforce.sh` 默认使用 `nohup` 后台运行，并把主进程 PID 写入：

```text
data/runs/<run-id>/pid
```

查看日志：

```bash
RUN_DIR=data/runs/<run-id>
tail -f "$RUN_DIR/info.log"
```

查看主进程：

```bash
RUN_DIR=data/runs/<run-id>
PID="$(cat "$RUN_DIR/pid")"
ps -fp "$PID"
```

停止当前 run：

```bash
RUN_DIR=data/runs/<run-id>
PID="$(cat "$RUN_DIR/pid")"
kill -TERM "$PID"
sleep 5
ps -fp "$PID"
```

查看仍在运行的 reinforce / Stockfish 进程：

```bash
ps -ef | grep -E "reinforce.py|stockfish" | grep -v grep
```

清理该 run 的残留子进程时，优先按 `ps` 输出中的 PID 精确结束：

```bash
kill -TERM <pid>
sleep 3
kill -KILL <pid>
pkill -KILL -f "models/stockfish/stockfish"
```

同一台云端机器同时跑多个实验时，先用 `ps -ef` 确认命令行里的 `--run-id` 和路径，再结束对应 PID。`tail -f` 只是在查看日志，按 `Ctrl+C` 只会退出日志查看。

查看 Lichess bot 进程：

```bash
ps -ef | grep -E "lichess-bot.py|uci_engine.py" | grep -v grep
```

停止 Lichess bot：

```bash
bash stop_lichess_bot.sh <run-id>
```

---

## 12. GUI 棋盘模拟器

```bash
python src/board.py \
  --gui 1
```

主要操作：

```text
棋盘点击 / Move 输入框:
    按当前行棋方执行合法走法。

Reset FEN:
    恢复输入局面；空输入恢复 startpos。

Settings:
    选择模型文件，设置搜索参数，应用并重新加载模型。
    Simulator 模式会自动分析当前局面并显示候选走法。

Close / Open:
    暂停或恢复 Simulator 自动候选走法。
    Open 会分析当前局面。

Play:
    选择用户颜色和起始局面，进入人机对弈。
    AI 自动回复一手并将回合交还给用户。

Simulator:
    进入双方轮流走子的模拟状态。

Undo:
    模拟器中撤销一手；人机模式中撤销最近一轮。

Import PGN:
    载入 PGN 主线的最终局面。

Save PGN:
    保存当前棋盘历史。
```

启动时加载指定模型：

```bash
python src/board.py \
  --gui 1 \
  --model models/champion.pth \
  --device cpu \
  --search-type mcts-mate \
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 32 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --mcts-time-fraction 0.90 \
  --mate-plies 3 \
  --mate-topk 4 \
  --mate-nodes 20000 \
  --mate-hash-mb 16 \
  --root-topn 16
```


---

## 13. CLI 棋盘模拟器

```bash
python src/board.py \
  --gui 0 \
  --model models/champion.pth \
  --device cpu \
  --search-type mcts-mate \
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 32 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --c-puct-base 19652 \
  --c-puct-factor 1.0 \
  --fpu-reduction 0.15 \
  --mcts-time-fraction 0.90 \
  --mate-plies 3 \
  --mate-topk 4 \
  --mate-nodes 20000 \
  --mate-hash-mb 16 \
  --root-topn 16
```

常用命令：

```text
e4
Nf3
e2e4

model models/chessnet.pth
params
set mcts_sims 30000

close
open

play
simulator

reset
reset "2k1rr2/pp4R1/2p1NQ2/3p1P2/8/1PP2n1P/P1K3Bq/R7 w - - 7 30"

undo
pgn data/game.pgn
save data/output.pgn
state
board
help
quit
```

`set <name> <value>` 应用单个搜索参数并重新加载当前模型。CLI 在 `Simulator` 模式且模型已载入时，会在局面或模型参数变化后自动输出候选走法。`close` 暂停自动候选输出，`open` 恢复自动候选输出并分析当前局面。

CLI 自动候选同样受 `movetime` 约束。`movetime=0` 表示时间上限关闭，由模拟次数和节点数等预算控制。

---

## 14. 空间维护

查看主要文件：

```bash
du -h --max-depth=2 data models | sort -h
```

清理 Python 缓存：

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

清理 run 目录：

```bash
rm -rf data/runs/reinforce_*
rm -rf models/runs/reinforce_*
```

清理临时候选模型：

```bash
rm -f models/tmp-*.pth
```

清理模型备份：

```bash
rm -f models/*.bak_*
```

清理 Linux / AutoDL 回收站：

```bash
rm -rf ~/.local/share/Trash/files/*
rm -rf ~/.local/share/Trash/info/*
rm -rf /root/autodl-tmp/.Trash-0/files/*
rm -rf /root/autodl-tmp/.Trash-0/*
```
