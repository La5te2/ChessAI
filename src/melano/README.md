# Melano C++

Melano 是 Gadidae 的独立 PVA 架构。C++ 实现覆盖：

- PGN 预处理与 Melano HDF5
- 监督训练与 LibTorch checkpoint
- 模型 Policy 和 PVA-MCTS 搜索
- batched arena
- Melano FCPI 采样、反事实目标、训练、arena gate 与晋升

Gadus 和 Melano 只共享 LibTorch、HDF5、chess-library、nlohmann-json、zlib 与构建基础设施。状态编码、招法编码、模型、搜索公式、训练目标和 FCPI 公式分别位于各自命名空间和 library 中。

## 1. 目录

```text
include/melano/
	args.hpp
	arena.hpp
	checkpoint.hpp
	dataset.hpp
	fcpi.hpp
	game.hpp
	model.hpp
	search.hpp

src/melano/
	args.cpp
	arena.cpp
	checkpoint.cpp
	dataset.cpp
	evolution.cpp
	fcpi.cpp
	game.cpp
	match.cpp
	model.cpp
	preprocess.cpp
	search.cpp
	searcher.cpp
	tests.cpp
	train.cpp
```

## 2. 依赖与构建

依赖安装到仓库的 `api/`。脚本通过可用的 NVIDIA GPU 自动选择 LibTorch CUDA `cu126` 或 CPU 包，并使用 `curl` 显示下载进度、速度和 ETA。

Windows：

```powershell
.\api\setup.bat
.\scripts\build.bat
```

Linux：

```bash
bash api/setup.sh
bash scripts/build.sh
```

显式选择 LibTorch 包：

```powershell
$env:GADIDAE_TORCH_VARIANT = "cpu"
.\api\setup.bat
```

```bash
GADIDAE_TORCH_VARIANT=cu126 bash api/setup.sh
```

使用已有 LibTorch 时设置 `GADIDAE_TORCH_DIR`。构建采用 CMake、Ninja 和预编译头，在临时目录通过 CTest 后发布到：

```text
build/melano/preprocess
build/melano/train
build/melano/search
build/melano/arena
build/melano/fcpi
build/melano/uci
```

Windows 可执行文件带 `.exe` 后缀。

## 3. 架构

Melano HDF5 schema：

```text
arch_type=melano
state_encoding=melano_square_tokens
move_encoding=sd_64x64_underpromo9
target_schema=pva_minimax_dueling
value_perspective=side_to_move
```

### 3.1 State

每个局面编码为 67 个整数：

- 64 个棋盘格 token：空格为 0，白方六种棋子为 1 到 6，黑方六种棋子为 7 到 12
- 1 个行棋方 token
- 1 个王车易位 bitmask
- 1 个 en passant 文件 token

模型将棋盘编码为 64 个 square token 和 1 个 global token。piece、square、side、castling、en passant embedding 共同构成初始表示。

### 3.2 Move

动作空间大小为：

$$
64\times64+64\times9=4672
$$

普通走法与后升变采用 source-destination 编码。马、象、车升变采用每个起点 3 个方向乘 3 个升变棋子的独立编码。

### 3.3 Network

默认 `channels=128`、`blocks=10`。每个 geometry attention block 包含：

- pre-norm multi-head self-attention
- 32 类静态棋盘几何关系 bias
- 由 global token 生成的动态关系 bias
- residual connection
- pre-norm feed-forward network

Policy head 使用 source projection 与 destination projection 的缩放点积生成 $64\times64$ logits，并拼接 576 个 underpromotion logits。Value head 从 global token 输出 $V(s)\in[-1,1]$。Advantage head输出：

$$
A(s,a)=-2\tanh^2(z_{s,a})\in[-2,0]
$$

动作价值定义为：

$$
Q(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

## 4. Preprocess

带 `{+x}` 或 `{-x}` 局面评注的 PGN：

```bash
build/melano/preprocess \
	--input data/games.cmt.pgn \
	--output data/games.melano.h5 \
	--has-cmt 1 \
	--chunk-size 4096 \
	--compression 1 \
	--log-every 10000
```

评注分数先从白方视角转换到行棋方视角，再映射为：

$$
V(s)=\tanh(\operatorname{score}_{stm}(s)/3)
$$

连续两个有评注局面之间的动作目标为：

$$
A_{target}(s,a)=\operatorname{clip}(V(s')_{mover}-V(s)_{mover},-2,0)
$$

每盘棋的首个评分用于建立局面 Value 基准，其动作 Advantage 记为 0。`--has-cmt 0` 使用终局结果训练 Value，并关闭监督训练中的 dueling-Q loss。

HDF5 数据集为 `states`、`moves`、`values`、`adv_moves` 和 `adv_values`。

## 5. Train

```bash
build/melano/train \
	--data data/games.melano.h5 \
	--out models/melano.pth \
	--channels 128 \
	--blocks 10 \
	--epochs 3 \
	--batch-size 256 \
	--max-steps 80000 \
	--lr 0.0002 \
	--weight-decay 0.0001 \
	--value-weight 1.0 \
	--dueling-q-weight 0.5 \
	--device cuda \
	--log-every 50
```

监督损失为：

$$
L=L_{policy}+w_VL_V+w_QL_Q
$$

其中 $L_Q$ 比较 `clip(V + selected A)` 与 `clip(value_target + advantage_target)`。checkpoint 的逻辑顶层仅包含 `model` 与 `arch`，其中 `arch` 保存 Melano 标识、`channels`、`blocks` 与 `action_size`。checkpoint 通过临时文件和 rename 原子写回。

## 6. Search

模型 Policy：

```bash
build/melano/search \
	--model models/melano.pth \
	--fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
	--device cuda \
	--search-type closed \
	--mcts-sims 0 \
	--root-topn 8
```

PVA-MCTS：

```bash
build/melano/search \
	--model models/melano.pth \
	--device cuda \
	--search-type only-mcts \
	--mcts-sims 1000 \
	--mcts-min-sims 250 \
	--mcts-batch-size 64 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--root-topn 8
```

Melano 为每条边建立一份伪访问：

$$
Q_{prior}(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

定义：伪访问是由网络 $V+A$ 提供的动作价值先验。它以一个统计样本的权重参与边价值估计，同时保持 MCTS 的真实 visits 与叶节点回传独立。

已访问边的利用项为：

$$
Q_{edge}=\frac{N\cdot Q_{mcts}+Q_{prior}}{N+1}
$$

未访问边从 $Q_{prior}$ 应用 FPU reduction。`closed` 按模型 Policy 排序，`only-mcts` 按根节点 visits 与 prior 形成的分布排序。

## 7. Arena

```bash
build/melano/arena \
	--candidate models/candidate.pth \
	--baseline models/melano.pth \
	--device cuda \
	--games 400 \
	--games-in-flight 32 \
	--max-plies 240 \
	--opening-book data/openings.gen.bin \
	--book-plies 8 \
	--search-type only-mcts \
	--sims 64 \
	--mcts-batch-size 32 \
	--min-net-wins 4 \
	--pgn-output data/melano-arena.pgn \
	--log-every 1
```

arena 使用 paired openings，让 candidate 与 baseline 在相同局面交换颜色。`games` 使用正偶数。多盘棋共享两份已加载模型，通过 `games-in-flight` 批量推理。gate 条件为 `net_wins >= min_net_wins`。

## 8. FCPI

```bash
build/melano/fcpi \
	--model models/melano.pth \
	--device cuda \
	--iterations 5 \
	--games-per-iter 1000 \
	--games-in-flight 64 \
	--max-plies 240 \
	--positions-per-game 200 \
	--opening-book data/openings.gen.bin \
	--startpos-fraction 0.5 \
	--counterfactual-topk 8 \
	--counterfactual-min-plies 2 \
	--counterfactual-max-plies 6 \
	--counterfactual-target-average-plies 4 \
	--counterfactual-lambda 0.85 \
	--td-lambda 0.85 \
	--behavior-temperature 0.85 \
	--behavior-advantage-weight 0.5 \
	--uniform-mix 0.02 \
	--policy-temperature 0.25 \
	--successor-weight 0.75 \
	--played-return-weight 0.5 \
	--dueling-q-weight 0.5 \
	--epochs 15 \
	--train-max-steps 2000 \
	--batch-size 256 \
	--eval-games 400 \
	--eval-games-in-flight 32 \
	--eval-search-type closed \
	--eval-sims 0 \
	--eval-min-net-wins 4 \
	--log-every 50
```

行为分布使用：

$$
\ell_{behavior}(a)=\frac{\log P(a)+w_AA(s,a)}{T_b}
$$

反事实分支的多深度估计通过 $\lambda$ 混合，并与网络 $V+A$ 初值按 `successor-weight` 合并。已实际走出的动作再按 `played-return-weight` 混合 TD($\lambda$) 回报。

改进目标为：

$$
V^*(s)=\max_{a\in C}Q^*(s,a)
$$

$$
A^*(s,a)=\operatorname{clip}(Q^*(s,a)-V^*(s),-2,0)
$$

$$
\pi^*(a)\propto P(a)^{\alpha}\exp\left(\frac{Q^*(s,a)-V(s)}{T_\pi}\right)
$$

训练损失包含 policy、Value、dueling-Q、KL 与 entropy。每次运行创建 `data/runs/fcpi_<id>/` 和 `models/runs/fcpi_<id>/`。arena 接受 candidate 后，checkpoint 原子复制到该 run 的 `current.pth`。

## 9. 验证

```bash
ctest --test-dir build --output-on-failure
```

`melanotests` 覆盖 state codec、普通走法与升变 move codec、PVA 输出形状、Advantage 范围、反向传播和 checkpoint 往返。
