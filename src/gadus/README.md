# Gadus C++

Gadus C++ 覆盖以下链路：

```text
PGN -> preprocess -> HDF5 -> train -> checkpoint
checkpoint -> search
candidate + baseline -> arena
checkpoint -> FCPI self-play -> train -> arena gate -> current checkpoint
```

Gadus 的数据、模型、搜索公式和入口均位于自身模块中。Melano 采用对称目录和独立 library 实现自身链路。

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
include/gadus/
	arena.hpp
	args.hpp
	checkpoint.hpp
	dataset.hpp
	fcpi.hpp
	game.hpp
	model.hpp
	search.hpp
src/gadus/
	args.cpp
	checkpoint.cpp
	dataset.cpp
	evolution.cpp
	game.cpp
	match.cpp
	model.cpp
	searcher.cpp
	preprocess.cpp
	train.cpp
	search.cpp
	arena.cpp
	fcpi.cpp
	uci.cpp
	tests.cpp
CMakeLists.txt
```

- `api/`：项目本地依赖与安装脚本。
- `include/gadus/`：Gadus 对外类型和函数声明。
- `src/gadus/dataset.cpp`：PGN 预处理、HDF5 读取和监督训练。
- `src/gadus/model.cpp`：Gadus ResNet policy/value 网络。
- `src/gadus/searcher.cpp`：closed 决策、batched MCTS、PUCT、FPU、IMF 和 RPP。
- `src/gadus/match.cpp`：paired-opening batched arena 和 PGN 输出。
- `src/gadus/evolution.cpp`：Gadus FCPI 采样、反事实目标、训练、arena gate 和晋升。
- `src/gadus/*.cpp` 中的六个入口：`preprocess`、`train`、`search`、`arena`、`fcpi`、`uci`。
- `src/gadus/tests.cpp`：状态编码、招法编码、模型前反向和 checkpoint 测试。

## 2. 依赖

依赖安装到 `api/`：

- LibTorch
- HDF5
- zlib
- nlohmann-json
- chess-library
- Ninja

脚本仅在 `nvidia-smi` 能成功查询实际 GPU compute capability 时选择 LibTorch CUDA `cu126`，其余情况选择 CPU 包。可使用 `GADIDAE_TORCH_VARIANT` 显式选择。Windows 使用 `curl.exe` 下载并显示进度、速度和预计剩余时间。

依赖安装成功后，脚本清理下载包、HDF5/zlib 源码和中间构建目录，仅保留运行与开发所需的安装目录、头文件和脚本。

### Windows

```powershell
$env:GADIDAE_TORCH_VARIANT = "cpu"
.\api\setup.bat
```

CUDA：

```powershell
$env:GADIDAE_TORCH_VARIANT = "cu126"
.\api\setup.bat
```

### Linux

```bash
GADIDAE_TORCH_VARIANT=cpu bash api/setup.sh
```

CUDA：

```bash
GADIDAE_TORCH_VARIANT=cu126 bash api/setup.sh
```

## 3. 构建

### Windows

官方 Windows LibTorch 使用 MSVC ABI。完整 Visual Studio IDE 可由 Microsoft C++ Build Tools 与 Windows SDK 代替。`scripts/build.bat` 通过 `vswhere` 初始化编译环境，并让 CMake 生成 Ninja 构建文件。

Gadus library 使用 LibTorch/chess-library 预编译头，MSVC 使用 `/MP` 并行编译，第三方依赖作为 system headers 处理。构建脚本在临时目录完成编译与 CTest，再将运行文件发布到 `build/gadus/`。

```powershell
.\scripts\build.bat
```

`scripts/build.bat` 默认读取 `api/libtorch`。使用已有 LibTorch 安装时，可将 `GADIDAE_TORCH_DIR` 指向其根目录。

可执行文件位于：

```text
build/gadus/preprocess.exe
build/gadus/train.exe
build/gadus/search.exe
build/gadus/arena.exe
build/gadus/fcpi.exe
build/gadus/uci.exe
```

### Linux

```bash
bash scripts/build.sh
```

可执行文件位于 `build/gadus/`。

每个入口支持 `--help`：

```bash
build/gadus/search --help
```

## 4. Gadus 数据与模型

Gadus 使用：

- state encoding：`gadus_18_planes`
- move encoding：`alphazero_64x73`
- action size：`4672`
- 网络：ResNet trunk + linear policy head + MLP value head
- value 范围：`[-1, 1]`

HDF5 datasets：

```text
states: uint8, (N, 18, 8), 逐行 packbits
moves:  uint16, (N,)
values: float32, (N,)
```

HDF5 metadata：

```text
arch_type=gadus
state_encoding=gadus_18_planes
move_encoding=alphazero_64x73
target_schema=policy_value
has_cmt=0|1
```

C++ Gadus checkpoint 使用 LibTorch archive，逻辑顶层仅包含 `model` 与 `arch`。`model` 保存网络参数，`arch` 保存 Gadus 标识、`channels`、`blocks` 与 `action_size`。监督训练与 FCPI 的轮次和步数只写入运行日志。

监督训练损失：

$$
L_{supervised}=L_{policy\ CE}+w_vL_{value\ MSE}
$$

## 5. Preprocess

有局面评价批注的 PGN：

```bash
build/gadus/preprocess \
	--input data/games.pgn \
	--output data/games.gadus.h5 \
	--has-cmt 1 \
	--chunk-size 16384 \
	--compression-level 1 \
	--log-every 10000
```

普通 PGN 使用对局结果生成 value：

```bash
build/gadus/preprocess \
	--input data/games.pgn \
	--output data/games.gadus.h5 \
	--has-cmt 0 \
	--chunk-size 16384 \
	--compression-level 1 \
	--log-every 10000
```

- `--max-games`：读取对局上限，默认遍历输入文件。
- `--has-cmt 1`：读取 PGN 局面评价并转换成 side-to-move value。
- `--has-cmt 0`：按最终胜负生成 side-to-move value。
- `--chunk-size`：HDF5 扩展和压缩 chunk 大小。
- `--compression-level`：HDF5 deflate 等级。
- `--log-every`：每处理指定对局数打印一行进度。

## 6. Train

```bash
build/gadus/train \
	--data data/games.gadus.h5 \
	--out models/gadus.pth \
	--device cuda \
	--channels 128 \
	--blocks 10 \
	--epochs 10 \
	--batch-size 512 \
	--max-steps 80000 \
	--lr 0.001 \
	--weight-decay 0.0001 \
	--value-weight 0.25 \
	--save-every 5000 \
	--log-every 100 \
	--seed 2026
```

`train` 创建新的 Gadus 模型并执行一次监督训练。`--channels` 和 `--blocks` 决定模型结构。checkpoint 采用临时文件加 rename 的原子写回。

## 7. Search

模型直觉输出：

```bash
build/gadus/search \
	--model models/gadus.pth \
	--fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
	--device cuda \
	--search-type closed \
	--mcts-sims 0 \
	--root-topn 10
```

MCTS：

```bash
build/gadus/search \
	--model models/gadus.pth \
	--fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
	--device cuda \
	--search-type only-mcts \
	--mcts-sims 1000 \
	--mcts-min-sims 100 \
	--mcts-batch-size 64 \
	--movetime-ms 5000 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--root-topn 10
```

PUCT 选择分数：

$$
S(s,a)=Q(s,a)+c_{puct}(s)P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}
$$

未访问节点使用 FPU：

$$
Q_{FPU}=Q(s)-r_{FPU}\sqrt{\sum_{a:N(s,a)>0}P(s,a)}
$$

`closed` 使用模型合法走法 policy 排序。`only-mcts` 使用 batched leaf inference、动态模拟预算、PUCT 和 FPU。IMF 与 RPP 位于最终决策层，保持 MCTS prior 和访问统计原值。

## 8. Arena

```bash
build/gadus/arena \
	--candidate models/candidate.pth \
	--baseline models/gadus.pth \
	--device cuda \
	--games 400 \
	--games-in-flight 32 \
	--max-plies 240 \
	--opening-book data/openings.gen.bin \
	--book-plies 8 \
	--max-book-positions 50000 \
	--search-type closed \
	--sims 0 \
	--mcts-batch-size 64 \
	--movetime-ms 0 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--min-net-wins 4 \
	--pgn-output data/gadus-arena.pgn \
	--log-every 1
```

Arena 使用 paired openings，同一开局交换双方颜色。`games` 使用正偶数。`games-in-flight` 表示同时推进的对局数，两个模型各加载一次并批量评估轮到自己走子的局面。结果包含 wins、draws、losses、net wins、score、置信区间和 Elo 差估计。

从标准初始局面开始：

```bash
build/gadus/arena --candidate models/candidate.pth --baseline models/gadus.pth --opening-book=
```

## 9. FCPI

```bash
build/gadus/fcpi \
	--model models/gadus.pth \
	--device cuda \
	--iterations 10 \
	--games-per-iter 1000 \
	--games-in-flight 64 \
	--max-plies 240 \
	--positions-per-game 200 \
	--opening-book data/openings.gen.bin \
	--startpos-fraction 0.5 \
	--book-plies 8 \
	--max-book-positions 50000 \
	--inference-batch-size 64 \
	--target-records-per-batch 256 \
	--counterfactual-topk 6 \
	--counterfactual-min-plies 2 \
	--counterfactual-max-plies 6 \
	--counterfactual-target-average-plies 4 \
	--counterfactual-lambda 0.8 \
	--td-lambda 0.8 \
	--behavior-temperature 1.0 \
	--uniform-mix 0.03 \
	--policy-temperature 0.25 \
	--prior-power 1.0 \
	--played-return-weight 0.5 \
	--policy-weight 1.0 \
	--value-weight 1.0 \
	--kl-weight 0.05 \
	--entropy-weight 0.001 \
	--epochs 15 \
	--train-max-steps 2000 \
	--batch-size 256 \
	--lr 0.00002 \
	--weight-decay 0.0001 \
	--grad-clip 1.0 \
	--eval-games 400 \
	--eval-games-in-flight 32 \
	--eval-max-plies 240 \
	--eval-opening-book data/openings.gen.bin \
	--eval-book-plies 8 \
	--eval-max-book-positions 50000 \
	--eval-search-type closed \
	--eval-sims 0 \
	--eval-mcts-batch-size 64 \
	--eval-movetime-ms 0 \
	--eval-c-puct 0.5 \
	--eval-c-puct-base 19652 \
	--eval-c-puct-factor 1.0 \
	--eval-fpu-reduction 0.15 \
	--eval-repetition-policy-penalty 0.0 \
	--eval-instant-mate-first 0 \
	--eval-min-net-wins 4 \
	--log-every 50 \
	--seed 2026
```

每次运行由系统生成 `fcpi_YYYYMMDD_HHMMSS_id`：

```text
data/runs/<run-id>/
	fcpi_iter_001.h5
	summary.json
models/runs/<run-id>/
	initial.pth
	current.pth
	candidate_iter_001.pth
```

每轮执行：

1. 使用 `current.pth` 进行 closed self-play。
2. 对采样局面执行自适应多步反事实展开。
3. 用 TD($\lambda$) 回报和反事实动作价值构造 policy/value 目标。
4. 训练 `candidate_iter_NNN.pth`。
5. 通过 paired-game arena 比较 candidate 与 current。
6. 当 `net_wins >= eval_min_net_wins` 时原子写回 `current.pth`。

Gadus FCPI policy 目标：

$$
z_a=\alpha\log(\pi_{old}(a)+\epsilon)+\frac{q_a-V(s)}{\tau}
$$

$$
\pi_{target}(a)=\operatorname{softmax}(z_a)
$$

训练损失：

$$
L=w_pL_{policy}+w_vL_{value}+w_{KL}D_{KL}(\pi_{old}\|\pi_{new})-w_HH(\pi_{new})
$$

## 10. 已验证链路

本地 Windows CPU 烟测覆盖：

- 单盘 PGN 生成 Gadus HDF5
- 一步监督训练和 checkpoint 原子写回
- closed 搜索
- batched MCTS 搜索
- 两盘 paired arena 与 PGN 输出
- 一轮 FCPI 采样、反事实展开、训练、arena gate 与 current 晋升
- `gadustests` 全部通过
