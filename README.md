# Gadidae

Gadidae 是一个实验性国际象棋神经网络引擎项目。当前有两套彼此独立的 C++ 架构：

- `Gadus`：ResNet + linear policy/value，使用 `gadus_18_planes` state encoding 和 `alphazero_64x73` move encoding。
- `Melano`：geometry attention + source-destination policy/value/advantage，使用 `melano_square_tokens` state encoding 和 `sd_64x64_underpromo9` move encoding。

两套架构分别实现 preprocess、train、search、arena、FCPI 和 UCI。它们共享 LibTorch、HDF5、chess-library、nlohmann-json、zlib 与构建基础设施。

## 1. 目录

```text
api/
	setup.bat
	setup.sh
	libtorch/
	hdf5/
	zlib/
	nlohmann/
	chess/
include/
	gadus/
	melano/
src/
	gadus/
	melano/
scripts/
	analyze.py
	teacher.py
	opening_book.py
	gui.py
	rules.py
	simulator.py
	stadium.py
	transit.py
	uci.py
	build.bat
	build.sh
	run_opening.sh
	run_simulator.vbs
	run_stadium.vbs
build/
	gadus/
	melano/
data/
models/
docs/
```

- `include/gadus/`、`src/gadus/`：Gadus 数据、模型、搜索、对战和 FCPI 实现。
- `include/melano/`、`src/melano/`：Melano 数据、模型、搜索、对战和 FCPI 实现。
- `scripts/`：通用 UCI 工具、图形界面与构建启动脚本。
- `api/`：仓库本地 C++ 依赖与安装脚本。
- `build/gadus/`、`build/melano/`：可直接运行的架构程序与运行 DLL。
- `data/`：PGN、HDF5、开局书、分析结果和运行数据。
- `models/`：LibTorch checkpoint 与 UCI 教师机。

架构细节与公式见 [Gadus C++](src/gadus/README.md) 和 [Melano C++](src/melano/README.md)。

LibTorch checkpoint 的逻辑顶层固定为：

```text
model
arch
```

`model` 保存网络参数，`arch` 保存架构标识及构造网络所需的形状信息。训练轮次与步数只作为运行期日志，不写入模型。

## 2. 依赖

Python 公共脚本：

```bash
python -m pip install -r requirements.txt
```

C++ 依赖安装到 `api/`。安装脚本根据 NVIDIA CUDA 环境选择 LibTorch 包，也可通过 `GADIDAE_TORCH_VARIANT=cpu|cu126` 指定。

Windows：

```powershell
$env:GADIDAE_TORCH_VARIANT = "cpu"
.\api\setup.bat
```

Linux：

```bash
GADIDAE_TORCH_VARIANT=cu126 bash api/setup.sh
```

## 3. 构建

Windows：

```powershell
.\scripts\build.bat
```

Linux：

```bash
bash scripts/build.sh
```

构建脚本在临时目录完成 CMake、Ninja 和 CTest。成功后发布以下程序，并清理 CMake 元数据、测试程序、链接库和日志：

```text
build/gadus/preprocess
build/gadus/train
build/gadus/search
build/gadus/arena
build/gadus/fcpi
build/gadus/uci

build/melano/preprocess
build/melano/train
build/melano/search
build/melano/arena
build/melano/fcpi
build/melano/uci
```

Windows 程序带 `.exe` 后缀。`build/` 顶层固定为 `gadus/` 与 `melano/`。

### 3.1 历史模型迁移

`scripts/transit.py` 将历史 Python `state_dict` 或带 `model` 字段的 `.pth` 转换为当前 LibTorch checkpoint。脚本根据参数名称和张量形状识别架构，并校验完整参数集合：

```bash
python scripts/transit.py \
	--input models/candidate0.pth \
	--output models/candidate0.gadus.pth
```

迁移脚本需要 Python PyTorch。生成的模型由对应架构的 C++ search、arena、FCPI、UCI 及图形界面直接读取。当前迁移器覆盖历史 `resnet_pv_linear` 参数，即当前 Gadus 架构。

## 4. Gadus

```bash
build/gadus/preprocess \
	--input data/games.pgn \
	--output data/games.gadus.h5 \
	--has-cmt 1 \
	--log-every 10000

build/gadus/train \
	--data data/games.gadus.h5 \
	--out models/gadus.pth \
	--channels 128 \
	--blocks 10 \
	--epochs 3 \
	--batch-size 256 \
	--max-steps 80000 \
	--device cuda

build/gadus/search \
	--model models/gadus.pth \
	--fen "startpos" \
	--device cuda \
	--search-type only-mcts \
	--mcts-sims 1000 \
	--mcts-batch-size 64 \
	--root-topn 8
```

Arena：

```bash
build/gadus/arena \
	--candidate models/candidate.pth \
	--baseline models/champion.pth \
	--device cuda \
	--games 400 \
	--games-in-flight 32 \
	--search-type closed \
	--sims 0 \
	--opening-book data/openings.gen.bin \
	--pgn-output data/gadus-arena.pgn
```

`--games` 使用正偶数，使每个开局都能交换双方颜色完成配对。

FCPI：

```bash
build/gadus/fcpi \
	--model models/gadus.pth \
	--device cuda \
	--iterations 1 \
	--games-per-iter 1000 \
	--games-in-flight 64 \
	--counterfactual-topk 6 \
	--epochs 15 \
	--train-max-steps 2000 \
	--eval-games 400 \
	--eval-search-type closed \
	--eval-min-net-wins 4
```

## 5. Melano

Melano 使用独立 HDF5 schema 与 checkpoint。入口形式与 Gadus 对称，架构目标参数由 Melano 程序自身定义。

```bash
build/melano/preprocess \
	--input data/games.cmt.pgn \
	--output data/games.melano.h5 \
	--has-cmt 1 \
	--log-every 10000

build/melano/train \
	--data data/games.melano.h5 \
	--out models/melano.pth \
	--channels 128 \
	--blocks 10 \
	--epochs 3 \
	--batch-size 256 \
	--max-steps 80000 \
	--device cuda

build/melano/search \
	--model models/melano.pth \
	--fen "startpos" \
	--device cuda \
	--search-type only-mcts \
	--mcts-sims 1000 \
	--mcts-batch-size 64 \
	--root-topn 8
```

## 6. UCI

两个 C++ UCI 程序直接加载对应 LibTorch checkpoint。公共 launcher 根据架构选择已构建程序：

```bash
python scripts/uci.py \
	--arch gadus \
	--model models/gadus.pth \
	--device cpu \
	--search-type only-mcts \
	--mcts-sims 100
```

```bash
python scripts/uci.py \
	--arch melano \
	--model models/melano.pth \
	--device cuda \
	--search-type only-mcts \
	--mcts-sims 1000
```

UCI 输出包含 MultiPV、side-to-move `score cp`、节点数、NPS、耗时和 PV。搜索开始时先发布模型直觉结果，MCTS 期间按 `ProgressIntervalMS` 发布中间结果。

## 7. Simulator

Simulator 接收任意完整 UCI 命令：

```powershell
python scripts\simulator.py `
	--uci "python scripts\uci.py --arch gadus --model models\gadus.pth" `
	--device cpu `
	--search-type only-mcts `
	--mcts-sims 1000 `
	--movetime-ms 0 `
	--progress-interval-ms 750 `
	--root-topn 8
```

Windows 隐藏控制台启动：

```powershell
wscript.exe scripts\run_simulator.vbs
```

## 8. Stadium

Stadium 让两个任意 UCI 引擎进行一盘可视化对局。双方拥有独立命令、UCI options 和 movetime。

```powershell
python scripts\stadium.py `
	--white-uci "python scripts\uci.py --arch gadus --model models\gadus.pth" `
	--black-uci "python scripts\uci.py --arch melano --model models\melano.pth" `
	--white-movetime-ms 3000 `
	--black-movetime-ms 3000 `
	--max-plies 240
```

Windows 隐藏控制台启动：

```powershell
wscript.exe scripts\run_stadium.vbs
```

## 9. UCI 教师机工具

`teacher.py` 提供 UCI 招法评分、regret 与 SQLite cache。`analyze.py` 使用它生成 `.cmt` 分析和带局面评价的 PGN。

```bash
python scripts/analyze.py \
	--input data/user-pgn/game.pgn \
	--uci models/stockfish/stockfish \
	--uci-depth 16 \
	--uci-multipv 8 \
	--pgn-comments
```

Windows 教师机路径使用 `models/stockfish/stockfish.exe`。

## 10. Opening Book

```bash
bash scripts/run_opening.sh data/games.pgn 50000 data/openings.gen.bin
```

```bash
python scripts/opening_book.py \
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
