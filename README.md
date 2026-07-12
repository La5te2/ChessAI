## 0. 项目目录约定

```text
ChessAI/
├── data/
│   ├── games.pgn
│   ├── games.h5
│   ├── openings.bin
│   ├── openings.bal.bin
│   ├── openings.gen.bin
│   ├── selflearn/
│   │   └── regression.json
│   └── runs/
│       └── run_YYYYMMDD_HHMMSS_pid_xxxxxxxx/
│           ├── selflearn_iter_*.h5
│           ├── regression.json
│           ├── teacher_cache.sqlite
│           └── teacher_cache.worker*.sqlite
├── models/
│   ├── chessnet.pth
│   ├── runs/
│   │   └── run_YYYYMMDD_HHMMSS_pid_xxxxxxxx/
│   │       ├── current.pth
│   │       └── selflearn_candidate_iter_*.pth
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
│   ├── opening_book.py
│   ├── checkpoint_io.py
│   ├── preprocess.py
│   ├── inspection.py
│   ├── standardize.py
│   ├── train.py
│   ├── arena.py
│   ├── teacher.py
│   ├── regression.py
│   ├── selflearn.py
│   └── board.py
├── run_selflearn.sh
└── requirements.txt
```

项目数据由监督训练 HDF5、自学习 run 目录、动态回归集、教师缓存和模型 checkpoint 组成。

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

`search.py` 组合动态预算 MCTS、Alpha-Beta 根候选验证与 Q tiebreak 确定性选棋。

MCTS 根据根节点访问分布熵、候选访问差距和 Q 值差距计算不确定性，并在软上限范围内分配模拟次数。Alpha-Beta 对根候选执行统一验证，搜索叶节点由 policy/value 网络估值。

Q tiebreak 在确定性选棋中扫描根候选：当候选的访问数与概率接近当前首选时，按接近程度动态降低所需 Q 领先幅度，并由排序更高的候选接管首选。低预算局面会按首选访问数动态收缩最小访问数门槛。

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

### 1.5 教师约束自学习

`selflearn.py` 运行教师约束的 AlphaZero 式闭环：

```text
champion 自博弈
↓
Search 生成策略分布
↓
Stockfish 评价 root top-k 候选并生成 policy、WDL value、regret 和可接受答案集合
↓
teacher veto 修正高 regret 自博弈落子
↓
top1 policy、teacher-labeled top-k policy、terminal value、teacher value、KL、监督 replay
↓
candidate
↓
arena、监督集、自学习目标、动态回归集验收
↓
原子写回 champion
```

达到最大步数的棋谱由 Stockfish 按当前局面分值裁定为 `1-0`、`0-1` 或 `1/2-1/2`，并通过 `terminal_valid=1` 参与 policy、terminal value、teacher value 和 KL 训练。

### 1.6 动态回归集

`regression.py` 管理教师标注的动态回归集。

每个局面保存：

```text
FEN
可接受走法集合
教师最佳分数
学生 regret
教师权重
出现次数
```

可接受走法由 Stockfish MultiPV 分值与容差共同确定。candidate 通过完整验收后，当前轮样例进入长期回归集。

### 1.7 棋盘模拟器

`board.py` 启动后进入 `Simulator` 模式，双方按照当前行棋方轮流走子。

模型文件与搜索参数通过 `Settings` 统一应用。每次应用参数都会重新加载模型。处于 `Simulator` 模式且模型已载入时，当前局面会自动生成候选走法，并受 `movetime`、`mcts_sims`、Alpha-Beta 节点数等搜索预算约束。

`Play` 启动人机对弈，并提示选择：

```text
用户执白 / 用户执黑
沿用当前局面 / 使用 startpos
```

进入 Play 模式后，AI 在轮到自身颜色时自动行棋。每次自动回复提交一手，回合随后交还给用户。

`Simulator` 将棋盘恢复为双人模拟状态。

`Reset FEN` 输入框接收完整 FEN。空输入恢复 `startpos`。

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
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --log-every 1000
```

`--verify` 使用 Stockfish 评估 Polyglot book 的可达开局。`abs(white_cp) <= --max-abs-cp` 的路径会写入输出 book。`--book-plies` 控制开局深度，`--min-fens` 控制输出 `.bin` 按同一开局深度可展开出的 selflearn unique opening state 下限。

Windows 本地 Stockfish 示例：

```powershell
python src/opening_book.py --verify data/openings.bin --uci models/stockfish/stockfish.exe --output data/openings.bal.local.bin --min-fens 200 --uci-depth 8 --log-every 20
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
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --log-every 1000
```

`--pgn` 从 PGN 主线开局中筛选均势路径，并写入到达 endpoint FEN 的 Polyglot entry。省略 `--output` 时写入 `data/openings.gen.bin`。

脚本入口：

```bash
bash run_opening.sh data/games.pgn 50000 data/openings.gen.bin
```

### 2.2 控制台输出

训练、预处理、自学习和 arena 的进度信息直接打印到控制台。

`--log-every` 控制 step / move / preprocess game 进度行：

```text
train step: ...
selflearn train step: ...
regression validation: ...
arena quality: ...
arena quality worker ...:
preprocess progress: ...
```

`--log-every 0` 关闭这类进度行。

自学习和 arena 每局结束打印一行：

```text
selflearn worker ... game ...
arena worker ... game ...
```

阶段汇总使用 JSON：

```text
preprocess summary:
selflearn games summary:
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

## 4. HDF5 检查

监督训练数据：

```bash
python src/inspection.py \
  data/games.h5
```

自学习数据与概率分布：

```bash
python src/inspection.py \
  data/runs/<run-id>/selflearn_iter_1.h5 \
  --check-probabilities
```

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
  --eval-c-puct 0.5 \
  --eval-alpha-beta-depth 5 \
  --eval-alpha-beta-topk 16 \
  --eval-alpha-beta-nodes 100000 \
  --eval-alpha-beta-quiescence 4 \
  --eval-alpha-beta-margin 0.02 \
  --eval-alpha-beta-time-fraction 0.45 \
  --eval-q-tiebreak-min-visits 32 \
  --eval-q-tiebreak-p-ratio 0.70 \
  --eval-q-tiebreak-visit-ratio 0.70 \
  --eval-q-tiebreak-margin 0.02 \
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
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 32 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --alpha-beta-depth 0 \
  --alpha-beta-topk 1 \
  --alpha-beta-nodes 0 \
  --alpha-beta-quiescence 0 \
  --alpha-beta-margin 0.02 \
  --alpha-beta-time-fraction 0.00 \
  --mate-guard-plies 5 \
  --q-tiebreak-min-visits 32 \
  --q-tiebreak-p-ratio 0.70 \
  --q-tiebreak-visit-ratio 0.70 \
  --q-tiebreak-margin 0.02 \
  --root-topn 16
```

`--mcts-sims` 表示 MCTS 软上限。`--movetime-ms` 表示完整 Search 的时间上限。`--c-puct` 控制 MCTS 探索强度，手动局面分析使用 `0.4~0.5` 更集中。`--mate-guard-plies` 对根候选执行短深度强制将杀检查。`--q-tiebreak-*` 控制 Q 值接管：动态最小访问数、概率比例、访问比例和动态 Q 领先上限；候选带有 Alpha-Beta 分数时，Q 接管候选的 Alpha-Beta 分数需要达到当前首选水平。

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
  --c-puct 0.5 \
  --alpha-beta-depth 5 \
  --alpha-beta-topk 16 \
  --alpha-beta-nodes 100000 \
  --alpha-beta-quiescence 4 \
  --alpha-beta-margin 0.02 \
  --alpha-beta-time-fraction 0.25 \
  --mate-guard-plies 5 \
  --q-tiebreak-min-visits 32 \
  --q-tiebreak-p-ratio 0.70 \
  --q-tiebreak-visit-ratio 0.70 \
  --q-tiebreak-margin 0.02 \
  --uci models/stockfish/stockfish \
  --uci-depth 10 \
  --uci-multipv 8 \
  --uci-threads 4 \
  --uci-hash-mb 512 \
  --log-every 1000
```

命令输出配对对局结果和双方走法质量指标。

---

## 10. 教师约束自学习

推荐使用脚本启动：

```bash
bash run_selflearn.sh
```

保留控制台输出并写入日志：

```bash
mkdir -p logs
LOG="logs/selflearn_$(date +%Y%m%d_%H%M%S).log"
set -o pipefail
bash run_selflearn.sh 2>&1 | tee -a "$LOG"
```

实验参数直接写在 `run_selflearn.sh` 里。调整采样、教师、训练或验收配置时，编辑脚本中的对应数字。脚本启动时会打印本次模型、数据、worker、search、teacher、train 和 eval 摘要。

`run_selflearn.sh` 当前展开命令：

```bash
python src/selflearn.py \
  --model models/chessnet.pth \
  --supervised-data data/games.h5 \
  --uci models/stockfish/stockfish \
  --device cuda \
  --iterations 1 \
  --games-per-iter 1000 \
  --parallel 10 \
  --max-plies 150 \
  --opening-book data/openings.gen.bin \
  --book-plies 8 \
  --max-book-positions 1000 \
  --sims 32 \
  --mcts-batch-size 64 \
  --movetime-ms 1000 \
  --c-puct 0.5 \
  --alpha-beta-depth 3 \
  --alpha-beta-topk 4 \
  --alpha-beta-nodes 20000 \
  --alpha-beta-quiescence 2 \
  --alpha-beta-margin 0.02 \
  --alpha-beta-time-fraction 0.20 \
  --mate-guard-plies 3 \
  --q-tiebreak-min-visits 32 \
  --q-tiebreak-p-ratio 0.85 \
  --q-tiebreak-visit-ratio 0.85 \
  --q-tiebreak-margin 0.03 \
  --uci-depth 12 \
  --uci-multipv 6 \
  --uci-threads 2 \
  --uci-hash-mb 512 \
  --teacher-start-ply 0 \
  --teacher-every 1 \
  --teacher-sample-rate 1 \
  --teacher-label-topk 4 \
  --teacher-label-min-weight 0.20 \
  --teacher-veto-regret-cp 200 \
  --teacher-veto-min-weight 0.70 \
  --epochs-per-iter 64 \
  --train-max-steps 2500 \
  --batch-size 256 \
  --train-workers 4 \
  --replay-window 5 \
  --lr 2e-5 \
  --supervised-weight 0.50 \
  --kl-weight 0.20 \
  --max-supervised-loss-increase 0.25 \
  --max-target-ce-increase 0.02 \
  --regression-sims 200 \
  --regression-movetime-ms 1000 \
  --min-regression-accuracy 0.0 \
  --max-regression-drop 0 \
  --eval-games 100 \
  --eval-sims 32 \
  --eval-max-plies 180 \
  --eval-mcts-batch-size 64 \
  --eval-movetime-ms 1000 \
  --eval-c-puct 0.5 \
  --eval-alpha-beta-depth 3 \
  --eval-alpha-beta-topk 4 \
  --eval-alpha-beta-nodes 20000 \
  --eval-alpha-beta-quiescence 2 \
  --eval-alpha-beta-margin 0.02 \
  --eval-alpha-beta-time-fraction 0.20 \
  --eval-mate-guard-plies 3 \
  --eval-q-tiebreak-min-visits 32 \
  --eval-q-tiebreak-p-ratio 0.85 \
  --eval-q-tiebreak-visit-ratio 0.85 \
  --eval-q-tiebreak-margin 0.03 \
  --eval-opening-book data/openings.gen.bin \
  --eval-book-plies 8 \
  --eval-max-book-positions 500 \
  --eval-min-net-wins 0 \
  --eval-min-acpl-improvement 0.0 \
  --eval-min-accuracy-improvement 0.0 \
  --eval-uci-depth 12 \
  --eval-uci-multipv 6 \
  --log-every 50
```

自学习每个 iteration 会从 `--opening-book` 展开的起始局面中分配 `--games-per-iter` 个唯一开局，走子使用与 board 和 arena 一致的 deterministic top1。`--parallel` 同时设置 selfplay worker 与 eval worker；采样对局、候选验收对局和 arena move-quality 分析按 worker 分片执行。采样与候选验收使用同一组 MCTS/search 预算。训练阶段以模型 top1 one-hot 作为实际落子基底，并用 Stockfish 评价 root top-k 候选生成 teacher-labeled policy；`--teacher-label-topk` 控制候选数量，`--teacher-label-min-weight` 控制 teacher-labeled policy 的最小混入权重。teacher analyse 每步执行，teacher veto 按 `regret_cp` 与 `teacher_weight` 修正高风险自博弈落子。达到 `--max-plies` 的棋局由 Stockfish 对当前局面裁定结果。

脚本默认采用 `--min-regression-accuracy 0.0 --max-regression-drop 0`，回归验收以 champion 的当前回归正确数作为基准。candidate 的回归正确数达到 champion 水平时通过该项验收。

每次自学习运行都会生成配对的独立 run 目录：

```text
data/runs/run_YYYYMMDD_HHMMSS_pid_xxxxxxxx/
models/runs/run_YYYYMMDD_HHMMSS_pid_xxxxxxxx/
```

run-id 默认由程序自动生成，格式为：

```text
run_YYYYMMDD_HHMMSS_pid_xxxxxxxx
```

其中 `pid` 来自当前 Python 进程，末尾 8 位来自随机 UUID。相同 run-id 同时用于 `data/runs/<run-id>/` 和 `models/runs/<run-id>/`。

固定 run-id 时，在 `run_selflearn.sh` 的 `python src/selflearn.py` 参数中加入：

```bash
  --run-id experiment_001 \
```

`--model` 指定初始模型。运行开始后会复制为本次 run 的：

```text
models/runs/<run-id>/current.pth
```

候选通过验收后写回本次 run 的 `current.pth`。自学习 HDF5、教师缓存和本次回归集保存在数据 run 目录；候选模型和 `current.pth` 保存在模型 run 目录：

```text
data/runs/<run-id>/selflearn_iter_*.h5
data/runs/<run-id>/teacher_cache.sqlite
data/runs/<run-id>/teacher_cache.worker*.sqlite
data/runs/<run-id>/regression.json
models/runs/<run-id>/current.pth
models/runs/<run-id>/selflearn_candidate_iter_*.pth
```

---

## 11. GUI 棋盘模拟器

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
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 32 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --alpha-beta-depth 0 \
  --alpha-beta-topk 1 \
  --alpha-beta-nodes 0 \
  --alpha-beta-quiescence 0 \
  --alpha-beta-margin 0.02 \
  --alpha-beta-time-fraction 0.00 \
  --mate-guard-plies 5 \
  --q-tiebreak-min-visits 32 \
  --q-tiebreak-p-ratio 0.70 \
  --q-tiebreak-visit-ratio 0.70 \
  --q-tiebreak-margin 0.02 \
  --root-topn 16
```

在 `Simulator` 模式中，加载模型后会自动分析当前局面；之后每次走子、撤销、重置、导入 PGN 或修改模型参数，都会重新分析当前局面。`Close` 暂停自动候选走法，按钮文字变为 `Open`；`Open` 恢复自动候选走法并分析当前局面。`Play` 模式由 AI 回合触发自动行棋，用户回合显示当前对局信息。`Settings` 中可调整 `c_puct`、Alpha-Beta 参数、`q_tiebreak`、`q_tiebreak_min_visits`、`q_tiebreak_p_ratio`、`q_tiebreak_visit_ratio` 和 `q_tiebreak_margin`。

---

## 12. CLI 棋盘模拟器

```bash
python src/board.py \
  --gui 0 \
  --model models/champion.pth \
  --device cpu \
  --mcts-sims 30000 \
  --mcts-min-sims 6000 \
  --mcts-batch-size 32 \
  --movetime-ms 30000 \
  --c-puct 0.5 \
  --alpha-beta-depth 0 \
  --alpha-beta-topk 1 \
  --alpha-beta-nodes 0 \
  --alpha-beta-quiescence 0 \
  --alpha-beta-margin 0.02 \
  --alpha-beta-time-fraction 0.00 \
  --mate-guard-plies 5 \
  --q-tiebreak-min-visits 32 \
  --q-tiebreak-p-ratio 0.70 \
  --q-tiebreak-visit-ratio 0.70 \
  --q-tiebreak-margin 0.02 \
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

## 13. 空间维护

查看主要文件：

```bash
du -h --max-depth=2 data models | sort -h
```

清理 Python 缓存：

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

清理自学习 run 目录：

```bash
rm -rf data/runs/run_*
rm -rf models/runs/run_*
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
