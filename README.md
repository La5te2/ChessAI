# Gadidae

Gadidae 是一个实验性国际象棋神经网络引擎项目。当前架构如下：

- `Gadus`：ResNet + linear policy/value，使用 `gadus_18_planes` state encoding 和 `alphazero_64x73` move encoding。
- `Melano`：residual geometry attention + action-conditioned latent dynamics + source-destination policy/value/advantage，使用 `melano_square_tokens` state encoding 和 `sd_64x64_underpromo9` move encoding。

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
	glfw/
	glad/
	glm/
	imgui/
	freetype/
include/
	gadus/
	melano/
	graphics/
src/
	gadus/
	melano/
	graphics/
scripts/
	analyze.py
	opening_book.py
	check.py
	build.bat
	build.sh
	package_engine.bat
	package_engine.sh
	run_opening.bat
	run_opening.sh
build/
	gadus/
	melano/
	graphics/
data/
models/
	gadus/
	melano/
	stockfish/
```

- `include/gadus/`、`src/gadus/`：Gadus 独立实现。`dataset` 负责 PGN、HDF5 与监督训练，`game` 负责状态、动作和棋规，`model` 负责 ResNet Policy/Value，`searcher` 负责 closed 与 MCTS，`match` 负责 arena，`evolution` 负责 FCPI。
- `include/melano/`、`src/melano/`：Melano 独立实现。文件职责与入口形式和 Gadus 对称，状态编码、动作编码、网络、搜索与 FCPI 方程均由 Melano 自身实现。
- `preprocess.cpp`、`train.cpp`、`search.cpp`、`arena.cpp`、`fcpi.cpp`、`uci.cpp`：每套架构的六个命令入口。
- `tests.cpp`：每套架构的状态编码、特殊走法、棋规、网络前反向、数值范围与 checkpoint 往返测试。
- `scripts/`：通用 UCI 工具、模型检查与构建启动脚本。
- `api/`：仓库本地 C++ 依赖与安装脚本。
- `include/graphics/`、`src/graphics/`：架构无关的原生 OpenGL 图形界面，通过 UCI 与引擎通信。
- `build/gadus/`、`build/melano/`：可直接运行的架构程序与运行 DLL。
- `build/graphics/`：原生 Gadidae 图形程序与图形运行库。
- `data/`：PGN、HDF5、开局书、分析结果和运行数据。
- `models/gadus/`、`models/melano/`：各架构的 LibTorch checkpoint，以及可直接注册到 UCI 客户端的引擎与运行库。
- `models/stockfish/`：UCI 引擎示例。

Gadus 与 Melano 的数据定义、数学公式和运行方法分别写在第 4、5 节。

LibTorch checkpoint 的逻辑顶层固定为：

```text
model
arch
```

`model` 保存网络参数，`arch` 保存架构标识及构造网络所需的形状信息。训练轮次与步数记录在运行期日志中。

## 2. 依赖

Python 公共脚本：

```bash
python -m pip install -r scripts/requirements.txt
```

C++ 依赖安装到 `api/`：LibTorch、HDF5、zlib、nlohmann-json、chess-library 与 Ninja。图形构建同时安装 GLFW、GLM、Dear ImGui、FreeType 与 GLAD。`api/versions.env` 是 Windows 与 Linux 共用的依赖版本锁。安装脚本在 `nvidia-smi` 能查询 GPU compute capability 时选择 LibTorch CUDA `cu126`，其余环境选择 CPU 包，也可通过 `GADIDAE_TORCH_VARIANT=cpu|cu126` 指定。已有 LibTorch 安装可通过 `GADIDAE_TORCH_DIR` 指向其根目录。`setup.bat/.sh` 在安装结束时验证实际依赖，`build.bat/.sh` 在配置前再次验证，CMake 对 LibTorch、HDF5 与 zlib 使用精确版本约束。版本、variant 或 chess-library 校验和不匹配时立即停止。下载过程显示进度、速度和 ETA，成功后清理压缩包、源码和依赖构建目录。

Windows：

Windows LibTorch 使用 MSVC ABI。可安装 Microsoft C++ Build Tools 与 Windows SDK，`scripts/build.bat` 通过 `vswhere` 初始化 x64 编译环境，并由 CMake 生成 Ninja 构建文件。架构 library 使用 LibTorch/chess-library 预编译头，MSVC 使用 `/MP` 并行编译，第三方依赖作为 system headers 处理。

```powershell
$env:GADIDAE_TORCH_VARIANT = "cpu"
.\api\setup.bat
```

Linux：

```bash
sudo apt install build-essential cmake curl unzip tar python3 python3-pip
```

图形构建还需要 X11 开发环境：

```bash
sudo apt install xorg-dev
```

```bash
GADIDAE_TORCH_VARIANT=cu126 bash api/setup.sh
```

`setup.sh` 与 `build.sh` 默认根据 `DISPLAY` 或 `WAYLAND_DISPLAY` 判断当前 Linux 是否具有图形会话。Windows 的 `setup.bat` 与 `build.bat` 在 `auto` 模式下构建 Graphics。纯命令行服务器会跳过 GLFW、OpenGL 图形依赖和 `Gadidae` GUI，同时保留 Gadus、Melano 的完整命令行训练、搜索、竞技场、FCPI 与 UCI 链路。可使用 `GADIDAE_BUILD_GRAPHICS=0` 明确选择命令行构建，或使用 `GADIDAE_BUILD_GRAPHICS=1` 强制编译 GUI。

## 3. 构建

Windows：

```powershell
.\scripts\build.bat
# cmake -S <根目录> -B <根目录>\build\.build-work -G Ninja
```

Linux：

```bash
bash scripts/build.sh
```

纯命令行服务器也可显式运行：

```bash
GADIDAE_BUILD_GRAPHICS=0 bash api/setup.sh
GADIDAE_BUILD_GRAPHICS=0 bash scripts/build.sh
```

构建脚本通过 CMake、Ninja 和 CTest 生成并验证以下程序：

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

Windows 程序带 `.exe` 后缀。`build/gadus/` 与 `build/melano/` 保存发布程序。

`build/.build-work/` 保存 CMake 与 Ninja 的增量构建状态。后续构建复用未变化目标的对象文件；编译或 CTest 失败时，诊断信息保留在该目录，CTest 日志位于 `build/.build-work/Testing/Temporary/LastTest.log`。

每个命令入口支持 `--help`：

```bash
build/gadus/search --help
build/melano/fcpi --help
```

### 3.1 模型检查

`scripts/check.py` 只读检查现行 LibTorch checkpoint，输出架构、网络头、channels、blocks、动作空间、参数规模、张量类型、内存规模、有限性、文件大小与 SHA-256：

```bash
python scripts/check.py --model models/gadus/gadus.pth
```

模型逻辑顶层必须仅包含 `model` 与 `arch`。检查过程不会修改模型。

## 4. Gadus

### 4.1 数据与网络

Gadus 的状态编码为 `gadus_18_planes`：12 个棋子平面、1 个行棋方平面、4 个王车易位平面和 1 个 en passant 文件平面。每个平面按 8 个 rank bytes packbits，HDF5 中单个状态占 $18\times8$ bytes。

动作编码为 `alphazero_64x73`。每个起点对应 56 个八方向滑动动作、8 个马步动作和 9 个 underpromotion 动作：

$$
|\mathcal A|=64\times(56+8+9)=4672
$$

模型由 ResNet trunk、linear Policy head 和 MLP Value head 组成：

$$
(\ell(s),V(s))=f_{\theta}(s),\qquad V(s)\in[-1,1]
$$

$\ell(s)$ 是 4672 维 Policy logits。合法动作概率为：

$$
P(a\mid s)=\frac{\exp \ell_a(s)}{\sum_{b\in\mathcal A(s)}\exp \ell_b(s)}
$$

Gadus HDF5 schema：

```text
states: uint8,  (N, 18, 8)
moves:  uint16, (N,)
values: float32, (N,)

arch_type=gadus
state_encoding=gadus_18_planes
move_encoding=alphazero_64x73
target_schema=policy_value
has_cmt=0|1
```

`--has-cmt 1` 读取 `{+x}`、`{-x}` 形式的白方视角 pawn score。转换到 side-to-move 后：

$$
V_{target}(s)=\tanh\left(\frac{score_{stm}(s)}{3}\right)
$$

`--has-cmt 0` 使用最终胜负生成 $V_{target}\in\{-1,0,1\}$。监督损失为：

$$
L_{sup}=L_{CE}(\ell,a^*)+w_V\operatorname{MSE}(V,V_{target})
$$

### 4.2 Preprocess、Train 与 Search

```bash
build/gadus/preprocess \
	--input data/games.pgn \
	--output data/games.gadus.h5 \
	--has-cmt 1 \
	--chunk-size 16384 \
	--compression-level 1 \
	--log-every 10000

build/gadus/train \
	--data data/games.gadus.h5 \
	--out models/gadus/gadus.pth \
	--channels 128 \
	--blocks 10 \
	--epochs 10 \
	--batch-size 512 \
	--max-steps 80000 \
	--lr 0.001 \
	--weight-decay 0.0001 \
	--value-weight 0.25 \
	--save-every 5000 \
	--device cuda \
	--precision bf16 \
	--log-every 100 \
	--seed 2026

build/gadus/search \
	--model models/gadus/gadus.pth \
	--fen "startpos" \
	--device cuda \
	--precision bf16 \
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
	--root-topn 8
```

- `preprocess --max-games` 控制读取对局上限，`--chunk-size` 控制 HDF5 扩展单元，`--compression-level` 控制 deflate 等级，`--log-every` 控制对局进度输出。
- `train` 每次创建新的 Gadus 模型。`--channels` 和 `--blocks` 决定结构，`--max-steps` 是本次训练步数上限，`--save-every` 控制原子 checkpoint 写入周期。
- `search --fen startpos` 使用标准初始局面，也可传入完整 FEN。
- `--precision` 可取 `fp32` 或 `bf16`，默认 `fp32`。`bf16` 仅用于 CUDA 前向计算，Policy softmax、训练损失和指标累加使用 FP32，checkpoint 参数保持 FP32。CUDA 输入批次使用 pinned memory，Search 只把合法动作的 Policy 从 GPU 传回 CPU。

`closed` 直接按合法动作 Policy 排序。`only-mcts` 使用 batched leaf inference。设根视角下边 $(s,a)$ 的真实访问数为 $N(s,a)$，父节点访问数为 $N(s)$，平均回传价值为 $Q(s,a)$。动态探索系数为：

$$
c_{puct}(N)=c_0+f\log\left(\frac{N+b+1}{b}\right)
$$

PUCT 选择分数为：

$$
S(s,a)=Q(s,a)+c_{puct}(N(s))P(s,a)
\frac{\sqrt{N(s)+1}}{1+N(s,a)}-l_vN_v(s,a)
$$

$N_v$ 和 $l_v$ 分别是 virtual visits 与 virtual loss。未访问边使用 FPU：

$$
Q_{FPU}(s,a)=\operatorname{clip}\left(
Q(s)-r_{FPU}\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right)
$$

叶节点 Value 沿路径逐 ply 取反并回传。MCTS 根分布保留一份 prior 平滑：

$$
P_{root}(a\mid s)=\frac{N(s,a)+P(a\mid s)}
{\sum_{b\in\mathcal A(s)}(N(s,b)+P(b\mid s))}
$$

动态模拟预算使用根分布归一化熵 $H$、前两名访问差 $U_N$ 和前两名价值差 $U_Q$：

$$
H=-\frac{\sum_a p_a\log p_a}{\log|\mathcal A(s)|}
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},\qquad
U_Q=1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right)
$$

$$
u=\operatorname{clip}(0.5H+0.35U_N+0.15U_Q,0,1)
$$

$$
N_{target}=N_{min}+\left\lceil u(N_{cap}-N_{min})\right\rceil
$$

IMF 与 RPP 位于最终决策层，只调整走法排序。

### 4.3 Arena

```bash
build/gadus/arena \
	--candidate models/gadus/candidate.pth \
	--baseline models/gadus/champion.pth \
	--device cuda \
	--precision bf16 \
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

`--games` 使用正偶数，使每个开局都能交换双方颜色完成配对。`--games-in-flight` 控制同时推进的对局数，candidate 与 baseline 各加载一次，轮到同一模型行棋的局面组成 inference batch。

FCPI 完成后，可让最终 `current.pth` 与同一 run 开始时写入的 `initial.pth` 进行累计棋力测试：

```bash
python scripts/test.py
```

脚本默认读取 `data/runs/` 中最新的 FCPI `summary.json`，使用其中的 `current_model` 作为 candidate、`initial_model` 作为 baseline。第一阶段采用与 Gadus FCPI 验收一致的 2000 局 `closed` paired Arena；完成后从标准初始局面运行 2 局换边 `only-mcts` 对战，默认每步上限为 10000 simulations。`--summary <path>` 可选择指定 run，`--candidate` 与 `--baseline` 可覆盖模型路径，`--mcts-sims` 可调整第二阶段预算。每次测试在 `data/tests/gadus_vs_initial_<timestamp>/` 中写入 `summary.json`、`closed.pgn`、`startpos-mcts.pgn` 和 `info.log`。

从标准初始局面开始时，将开局书设为空：

```bash
build/gadus/arena \
	--candidate models/gadus/candidate.pth \
	--baseline models/gadus/gadus.pth \
	--opening-book=
```

设 candidate 的胜、和、负局数为 $W,D,L$，总局数为 $G$：

$$
score=\frac{W+\frac12D}{G},\qquad net\_wins=W-L
$$

每局得分 $x_i\in\{0,\frac12,1\}$，代码使用总体方差计算 95% 正态近似区间：

$$
\sigma^2=\frac1G\sum_{i=1}^{G}(x_i-score)^2
$$

$$
CI_{95\%}=\operatorname{clip}\left(score\pm1.96\sqrt{\frac{\sigma^2}{G}},0,1\right)
$$

显示用 Elo 差为：

$$
\Delta Elo=400\log_{10}\left(\frac{score}{1-score}\right)
$$

gate 条件为：

$$
W-L\geq min\_net\_wins
$$

### 4.4 FCPI

```bash
build/gadus/fcpi \
	--model models/gadus/gadus.pth \
	--device cuda \
	--precision bf16 \
	--iterations 10 \
	--games-per-iter 1000 \
	--games-in-flight 64 \
	--max-plies 240 \
	--opening-book data/openings.gen.bin \
	--startpos-fraction 0.2 \
	--book-plies 8 \
	--max-book-positions 50000 \
	--inference-batch-size 64 \
	--target-records-per-batch 256 \
	--counterfactual-budget 24 \
	--behavior-temperature 1.0 \
	--epochs 15 \
	--train-max-steps 2000 \
	--batch-size 256 \
	--lr 0.00002 \
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
	--eval-repetition-policy-penalty 1.0 \
	--eval-instant-mate-first 1 \
	--eval-min-net-wins 4 \
	--log-every 50 \
	--seed 2026
```

每轮 FCPI 使用冻结的 `current.pth` 进行模型自对战。Gadus 的行为分布只做温度变换：

$$
\mu(a\mid s)=
\frac{P(a\mid s)^{1/T_b}}{\sum_bP(b\mid s)^{1/T_b}}
$$

$T_b<1$ 使行为分布更集中于高 Policy 动作，$T_b=1$ 使用原 Policy 比例，$T_b>1$ 提高低 Policy 动作的采样概率。`behavior-temperature` 只控制真实自对弈，不控制反事实树的候选宽度。

同一编码局面第 $n$ 次出现时，FCPI 使用概率欠额分配动作：

$$
a_n=\arg\max_a\left[n\mu(a\mid s)-N_{n-1}(s,a)\right]
$$

$N_{n-1}(s,a)$ 是此前该局面选择动作 $a$ 的次数。该调度使实际动作频率跟随 $\mu$，同时避免独立随机抽样持续漏掉某个正概率动作。只要同一局面继续出现，任意满足 $\mu(a\mid s)>0$ 的动作都会在有限次访问内取得样本。

真实终局是无教师机自学习中直接来自棋规环境的事实信号。设终局 side-to-move 结果为 $z_T\in\{-1,0,1\}$，完整结束的对局对整条轨迹执行无 bootstrap 的 Monte Carlo 回传：

$$
G_T=z_T,\qquad G_t=-G_{t+1}
$$

负号对应行棋方在每个 ply 的切换。完整终局轨迹中所有真实局面的 Monte Carlo 权重为 $w_{MC}=1$。被 `max-plies` 截断的对局没有结果标签，因此 $w_{MC}=0$；其中的局面仍可作为反事实树根。

每局按完整 Gadus 编码对局面去重，每个局面建立一棵独立反事实树。根节点一次评价全部合法动作，使任何合法动作都不会被 top-k 排除。`--counterfactual-budget` 表示根节点之外可评价的动作边数，也是深层反事实树唯一的规模参数。非根节点的局部展开宽度由剩余预算自动确定：

$$
w(s)=\min\left(|\mathcal A(s)|,B_{remain},
\max\left(2,\left\lceil\sqrt{B_{remain}}\right\rceil\right)\right)
$$

非根节点始终选择 Policy top-1，其余位置通过 Gumbel top-k 无放回采样得到：

$$
\operatorname{key}(a)=\log(P(a\mid s)+\varepsilon)+g_a,
\qquad g_a\sim\operatorname{Gumbel}(0,1)
$$

树使用冻结的 `current.pth` 批量评价精确棋盘子局面。完整对局还会形成按局面与动作聚合的事实回报：

$$
\widehat G(s,a)=
\frac{1}{N(s,a)}
\sum_{i=1}^{N(s,a)}G_i(s,a)
$$

动作价值按信息强度依次确定：

$$
Q(s,a)=
\begin{cases}
z, & T(s,a)\text{ 是终局}\\
\widehat G(s,a), & N(s,a)>0\\
-\overline V(T(s,a)), & a\text{ 已展开}\\
V_{old}(s), & a\text{ 未展开}
\end{cases}
$$

终局边使用棋规给出的精确结果。事实回报是当前行为策略从该动作继续行棋至终局的 Monte Carlo 样本均值，它会取代该边错误的浅层 Value bootstrap。没有事实样本的已展开动作使用子树回传值，未展开动作使用冻结模型基线。

对只有模型回传信息的已展开动作：

$$
Q(s,a)=-\overline V(T(s,a))
$$

其中 $\overline V$ 是子树从叶到根回传后的值。

设冻结 Policy 下的局部均值为：

$$
m(s)=\sum_aP_{old}(a\mid s)Q(s,a)
$$

局部 Policy target 使用 KL 正则化 Policy improvement：

$$
\pi^+(a\mid s)=
\frac{P_{old}(a\mid s)\exp\left(Q(s,a)-m(s)\right)}
{\sum_bP_{old}(b\mid s)\exp\left(Q(s,b)-m(s)\right)}
$$

它等价于：

$$
\pi^+=\arg\max_\pi
\left[
\sum_a\pi(a\mid s)Q(s,a)
-D_{KL}\left(\pi(\cdot\mid s)\,\|\,P_{old}(\cdot\mid s)\right)
\right]
$$

KL 项只出现在 Policy target 的闭式构造中，用于限制一次有限预算规划对原 Policy 的偏离；训练总损失没有额外 KL 项。由于 $Q\in[-1,1]$，Policy improvement 直接使用 Value 原生尺度。

Gadus 节点回传值为：

$$
\overline V(s)=\sum_a\pi^+(a\mid s)Q(s,a)
$$

定义反事实残差：

$$
\delta_{CF}(s)=\overline V(s)-V_{old}(s)
$$

由于未展开动作满足 $Q(s,a)=V_{old}(s)$，只有实际展开的动作能够贡献非零残差：

$$
\delta_{CF}(s)=
\sum_{a\in E(s)}
\pi^+(a\mid s)\left(Q(s,a)-V_{old}(s)\right)
$$

因此树覆盖率已经进入残差本身。预算较小时，未展开概率质量把 $\overline V(s)$ 拉回冻结基线；预算增加后，更多反事实结论进入修正。完整对局结果同时训练局面 Value，并作为实际动作的事实 $Q$ 参与 Policy 比较。这一步把终局事实传入动作排序，避免冻结 Value 在各层形成自洽但错误的循环。

对同一局面的两个动作 $a$ 与 $b$，一次精确拟合后的目标赔率满足：

$$
\log\frac{\pi^+(a\mid s)}{\pi^+(b\mid s)}
=
\log\frac{P_{old}(a\mid s)}{P_{old}(b\mid s)}
+Q(s,a)-Q(s,b)
$$

若后续迭代持续得到 $Q(s,a)-Q(s,b)=\Delta>0$，第 $k$ 次局部改进后的对数赔率累计增加 $k\Delta$。因此排序翻转所需的理想拟合迭代数满足：

$$
k>
\frac{\log P_0(b\mid s)-\log P_0(a\mid s)}{\Delta}
$$

这证明了正回报差异能够在有限次局部改进中改变排序。它不等同于在固定浅层预算内证明某步棋的博弈论真值；真值仍来自终局样本或完整求解。根节点全动作评价与概率欠额采样消除了“动作从未进入训练”的结构性盲区。

待展开前沿按到达概率与 Bellman residual 排序。对深度为 $d$ 的子节点 $s'$：

$$
priority(s')=\rho(s)P(a\mid s)
\left(|-V(s')-V(s)|+\frac{1}{\sqrt{2+d}}\right)
$$

因此预算会自然分配给较可能到达、局部判断不一致且仍靠近根部的节点。设该节点展开边数为 $n(s)$，同一根树中的训练权重为：

$$
w_T(s)=w_P(s)=
\frac{n(s)}{\sum_{u\in\mathcal T_{root}}n(u)}
$$

每棵树的 Policy 总权重和反事实 Value 总权重均为 1：

$$
\sum_s w_P(s)=\sum_s w_T(s)=1
$$

真实 Monte Carlo 目标 $G_t$ 与冻结树目标 $\overline V(s)$ 作为独立 SmoothL1 项进入 Value 损失：

$$
L_V=
\frac{
\sum_s w_{MC}(s)\rho\left(V_{new}(s)-G_t(s)\right)+
\sum_s w_T(s)\rho\left(V_{new}(s)-\overline V(s)\right)
}{
\sum_s\left(w_{MC}(s)+w_T(s)\right)
}
$$

其中 $\rho$ 是 SmoothL1。$\overline V(s)$ 是冻结模型经过有限反事实规划得到的 detached target。它不是把当前 Value 原样训练回自身：当多步回传与原判断矛盾时，$\delta_{CF}\neq0$；未展开动作的冻结基线和 SmoothL1 的有界梯度共同使修正保持渐进。

Policy 损失为：

$$
L_P=
\frac{\sum_s w_P(s)L_{CE}\left(\pi_{new}(\cdot\mid s),\pi^+(\cdot\mid s)\right)}
{\sum_s w_P(s)}
$$

完整 FCPI 损失固定为：

$$
L=L_P+L_V
$$

Policy、Value 和共享 residual backbone 通过同一次反向传播联合更新。AdamW weight decay 与梯度裁剪使用固定训练常量，FCPI 命令行只提供学习率、batch、epoch 与 step 上限。

FCPI HDF5 保存 `policy_targets`、`policy_weights`、`mc_value_targets`、`mc_value_weights`、`tree_value_targets` 与 `tree_value_weights`。训练日志只输出 `policy`、`value_mc`、`value_tree`、合并后的 `value` 和 `loss`。反事实 summary 记录树数、决策节点数、评价边数、事实回报边数、终局边数、最大深度、平均 Bellman residual、平均覆盖率、平均 Value 修正和 Policy top-1 改变率。

每轮依次执行 `current.pth` 自对战、全局面去重采集、冻结反事实树展开、Monte Carlo/反事实 Value 目标构造、candidate 训练和 paired-game arena。每局按完整 Gadus 编码对全部局面去重。

每次运行由程序生成 `fcpi_YYYYMMDD_HHMMSS_id`，并创建：

```text
data/runs/<run-id>/
	fcpi_iter_001.h5
	summary.json
models/runs/<run-id>/
	initial.pth
	current.pth
	candidate_iter_001.pth
```

candidate 达到 arena gate 后原子写入该 run 的 `current.pth`。

针对 16 GiB RTX 4080 Super 的后台启动脚本为：

```bash
bash scripts/gadus_fcpi.sh
```

脚本默认使用 `PRECISION=bf16`、较大的批次与 batched games。可通过环境变量覆盖，例如 `PRECISION=fp32 BATCH_SIZE=512 bash scripts/gadus_fcpi.sh`。

## 5. Melano

### 5.1 数据与网络

Melano 将局面编码为 67 个整数：64 个棋盘格 piece tokens、1 个行棋方 token、1 个王车易位 bitmask 和 1 个 en passant 文件 token。空格编码为 0，白方六种棋子编码为 1 到 6，黑方六种棋子编码为 7 到 12。模型把它们展开为 64 个 square tokens 与 1 个 global token。piece、square、side、castling 和 en passant embeddings 共同形成初始表示。

动作编码为 `sd_64x64_underpromo9`：

$$
|\mathcal A|=64\times64+64\times9=4672
$$

普通走法和后升变使用 source-destination 编码。马、象、车升变使用每个起点的 3 个方向乘 3 个升变棋子编码。

每个 geometry attention block 包含 pre-norm multi-head self-attention、32 类静态棋盘几何关系 bias、global token 生成的动态关系 bias、residual connection 和 pre-norm feed-forward network。

Policy head 对 source 与 destination projections 做缩放点积，生成 $64\times64$ logits，并拼接 576 个 underpromotion logits。Value head 从 global token 输出：

$$
V(s)\in[-1,1]
$$

Advantage head 输出：

$$
A(s,a)=-2\tanh^2(z_{s,a})\in[-2,0]
$$

动作价值定义为：

$$
Q(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

Melano HDF5 schema：

```text
states:     uint8,  (N, 67)
next_states:uint8,  (N, 67)
moves:      uint16, (N,)
values:     float32, (N,)
next_values:float32, (N,)
adv_moves:  uint16, (N,)
adv_values: float32, (N,)

arch_type=melano
state_encoding=melano_square_tokens
move_encoding=sd_64x64_underpromo9
target_schema=pva_latent_dynamics
value_perspective=side_to_move
has_cmt=0|1
```

带评注 PGN 的 Value target 与 Gadus 使用相同映射：

$$
V_{target}(s)=\tanh\left(\frac{score_{stm}(s)}{3}\right)
$$

设动作前后都换算到该动作行棋方视角，监督 Advantage target 为：

$$
A_{target}(s,a)=\operatorname{clip}
\left(V_{mover}(s')-V_{mover}(s),-2,0\right)
$$

每盘棋的第一个有效评分建立 Value 基准，对应动作的 $A_{target}=0$。`--has-cmt 0` 使用终局结果生成 Value target，监督损失中的 dueling-Q 项权重取 0。

监督训练使用：

$$
\widehat Q(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

$$
Q_{target}(s,a)=\operatorname{clip}
\left(V_{target}(s)+A_{target}(s,a),-1,1\right)
$$

$$
z=E(s),\qquad \widehat z'=D(z,a),\qquad \bar z'=\operatorname{stopgrad}(E(s'))
$$

潜在转移使用动作 embedding 条件化一个 residual geometry-attention block：

$$
\widehat z'=\operatorname{LN}\left(z+\sigma(g(a))\odot
(T(z+c(a))-z)\right)
$$

其中 $c(a)$ 是动作条件，$g(a)$ 是逐通道更新门。潜在一致性损失逐 token 比较走后预测与精确棋规生成的走后状态编码：

$$
L_D=1-\frac{1}{65}\sum_i\cos(\widehat z'_i,\bar z'_i)
$$

`next_values` 独立保存走后局面在新行棋方视角下的 Value target。它直接来自当前走法后的评注，因此每盘棋第一条有效评注也能监督走后 latent：

$$
L_I=\operatorname{MSE}(V(\widehat z'),V_{target}(s'))
$$

完整监督损失为：

$$
L_{sup}=L_{CE}+w_V\operatorname{MSE}(V,V_{target})+
w_Q\operatorname{MSE}(\widehat Q,Q_{target})+w_D L_D+w_I L_I
$$

### 5.2 Preprocess、Train 与 Search

```bash
build/melano/preprocess \
	--input data/games.cmt.pgn \
	--output data/games.melano.h5 \
	--has-cmt 1 \
	--chunk-size 4096 \
	--compression-level 1 \
	--log-every 10000

build/melano/train \
	--data data/games.melano.h5 \
	--out models/melano/melano.pth \
	--channels 128 \
	--blocks 10 \
	--epochs 3 \
	--batch-size 256 \
	--max-steps 80000 \
	--lr 0.0002 \
	--weight-decay 0.0001 \
	--value-weight 1.0 \
	--dueling-q-weight 0.5 \
	--dynamics-weight 0.25 \
	--imagined-value-weight 0.25 \
	--device cuda \
	--precision bf16 \
	--log-every 50

build/melano/search \
	--model models/melano/melano.pth \
	--fen "startpos" \
	--device cuda \
	--precision bf16 \
	--search-type only-mcts \
	--mcts-sims 1000 \
	--mcts-min-sims 250 \
	--mcts-batch-size 64 \
	--movetime-ms 5000 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--root-topn 8
```

- `preprocess --max-games` 控制读取对局上限，`--chunk-size` 与 `--compression-level` 控制 Melano HDF5 写入。
- `train` 每次创建新的 Melano 模型。`--channels` 和 `--blocks` 决定 geometry attention 宽度与层数，checkpoint 通过临时文件和 rename 原子写回。
- Melano checkpoint 的逻辑顶层为 `model` 与 `arch`，其中 `arch` 保存架构标识、`channels`、`blocks` 和 `action_size`。
- `--precision` 可取 `fp32` 或 `bf16`，默认 `fp32`。`bf16` 用于 CUDA 前向计算，Policy softmax、P/V/A 与 latent dynamics 损失、指标累计和 checkpoint 参数保持 FP32。CUDA 输入批次使用 pinned memory，Search 只把合法动作的 Policy 与 Advantage 从 GPU 传回 CPU。

`closed` 按 Melano Policy 排序。`only-mcts` 使用 $K=2$ anchored latent MCTS，并为每条边建立一份伪访问。定义：伪访问是 $V+A$ 提供的动作价值先验，它以一个统计样本的权重参与边价值估计，同时保持真实 visits 与叶节点回传独立。

MCTS 的每个节点都保留精确 `chess::Board`，合法走法、将军、终局、重复局面与五十回合规则由棋规计算。网络评价在偶数深度重新建立精确 latent 锚点，在奇数深度使用动作条件 latent transition：

$$
z_d=
\begin{cases}
E(s_d),&d\bmod 2=0\\
D(z_{d-1},a_{d-1}),&d\bmod 2=1
\end{cases}
$$

$$
(P_d,V_d,A_d)=H(z_d)
$$

因此任意预测 latent 与最近的精确编码只相隔一个动作，运行时分布与当前一步 dynamics 训练目标一致。偶数深度节点缓存 $E(s_d)$，奇数深度 latent 只在批量评价期间存在，从而限制设备内存。`search` 输出中的 `exact_evaluations` 与 `latent_evaluations` 分别报告两类网络评价位置数。

$$
Q_{prior}(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

对已有 $N(s,a)$ 次真实访问、真实平均回传 $Q_{mcts}(s,a)$ 的边：

$$
Q_{edge}(s,a)=\frac{N(s,a)Q_{mcts}(s,a)+Q_{prior}(s,a)}{N(s,a)+1}
$$

未访问边使用：

$$
Q_{edge}(s,a)=\operatorname{clip}\left(
Q_{prior}(s,a)-r_{FPU}\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right)
$$

Melano 的动态探索系数与 PUCT 为：

$$
c_{puct}(N)=c_0+f\log\left(\frac{N+b+1}{b}\right)
$$

$$
S(s,a)=Q_{edge}(s,a)+c_{puct}(N(s))P(s,a)
\frac{\sqrt{N(s)+1}}{1+N(s,a)}-l_vN_v(s,a)
$$

Melano 根分布保留一份 prior 平滑：

$$
P_{root}(a\mid s)=\frac{N(s,a)+P(a\mid s)}
{\sum_{b\in\mathcal A(s)}(N(s,b)+P(b\mid s))}
$$

动态模拟预算使用根分布归一化熵 $H$、前两名访问差 $U_N$ 和前两名边价值差 $U_Q$：

$$
H=-\frac{\sum_a p_a\log p_a}{\log|\mathcal A(s)|}
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},\qquad
U_Q=1-\min\left(1,\frac{|Q_{edge,1}-Q_{edge,2}|}{0.5}\right)
$$

$$
u=\operatorname{clip}(0.5H+0.35U_N+0.15U_Q,0,1)
$$

$$
N_{target}=N_{min}+\left\lceil u(N_{cap}-N_{min})\right\rceil
$$

### 5.3 Arena

```bash
build/melano/arena \
	--candidate models/melano-candidate.pth \
	--baseline models/melano/melano.pth \
	--device cuda \
	--precision bf16 \
	--games 400 \
	--games-in-flight 32 \
	--max-plies 240 \
	--opening-book data/openings.gen.bin \
	--book-plies 8 \
	--max-book-positions 50000 \
	--search-type only-mcts \
	--sims 64 \
	--mcts-min-sims 32 \
	--mcts-batch-size 32 \
	--movetime-ms 0 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--min-net-wins 4 \
	--pgn-output data/melano-arena.pgn \
	--log-every 1
```

Melano arena 采用自身模型与搜索 backend。paired openings 让同一开局交换双方颜色，`--games-in-flight` 让两份已加载模型批量评估多盘棋。设 candidate 的胜、和、负局数为 $W,D,L$，总局数为 $G$：

$$
score=\frac{W+\frac12D}{G},\qquad net\_wins=W-L
$$

每局得分 $x_i\in\{0,\frac12,1\}$，95% 正态近似区间与显示用 Elo 差为：

$$
\sigma^2=\frac1G\sum_{i=1}^{G}(x_i-score)^2
$$

$$
CI_{95\%}=\operatorname{clip}\left(score\pm1.96\sqrt{\frac{\sigma^2}{G}},0,1\right)
$$

$$
\Delta Elo=400\log_{10}\left(\frac{score}{1-score}\right)
$$

gate 条件为：

$$
W-L\geq min\_net\_wins
$$

### 5.4 FCPI

```bash
build/melano/fcpi \
	--model models/melano/melano.pth \
	--device cuda \
	--precision bf16 \
	--iterations 5 \
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
	--counterfactual-budget 24 \
	--td-lambda 0.85 \
	--behavior-temperature 0.85 \
	--uniform-mix 0.02 \
	--policy-weight 1.0 \
	--value-weight 1.0 \
	--dueling-q-weight 0.5 \
	--dynamics-weight 0.25 \
	--imagined-value-weight 0.25 \
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

Melano 自对战行为分布同时使用 Policy 与非正 Advantage。设：

$$
\sigma_A(s)=\max_a|A(s,a)|
$$

当 $\sigma_A(s)<10^{-4}$ 时只使用 Policy。其余情况：

$$
\widetilde\mu(a\mid s)=\operatorname{softmax}_a\left(
\frac{\log(P(a\mid s)+\varepsilon)+A(s,a)/\sigma_A(s)}{T_b}
\right)
$$

$$
\mu(a\mid s)=(1-\epsilon)\widetilde\mu(a\mid s)+
\frac{\epsilon}{|\mathcal A(s)|}
$$

Melano `--counterfactual-budget` 同样是每个真实根最多评价的动作边数。局部宽度、Gumbel 无放回候选和前沿优先级由剩余预算自动产生。Melano 的候选 proposal 使用：

$$
\nu(a\mid s)=\operatorname{softmax}_a\left(
\log(P(a\mid s)+\varepsilon)+A(s,a)/\sigma_A(s)
\right)
$$

对未展开动作，反事实树保留 Melano 自身的 dueling 动作值：

$$
Q(s,a)=\operatorname{clip}(V_{old}(s)+A_{old}(s,a),-1,1)
$$

已展开动作由精确棋规生成子局面并覆盖为：

$$
Q(s,a)=-\overline V(T(s,a))
$$

Melano 在自己的 FCPI backend 中根据 PVA 动作值计算节点内均值：

$$
m(s)=\sum_aP_{old}(a\mid s)Q(s,a)
$$

$$
\pi^+(a\mid s)=
\frac{P_{old}(a\mid s)\exp(Q(s,a)-m(s))}
{\sum_bP_{old}(b\mid s)\exp(Q(s,b)-m(s))}
$$

该更新保留 Melano 动作值的原生尺度。summary 中的 KL、总变差距离、Advantage 幅度和 top-1 改变率只用于诊断，不进入训练损失。

Melano 的 $A(s,a)\leq0$ 表示动作相对局面能力上界的损失。反事实树使用所有合法动作的动作值上界：

$$
\overline V_T(s)=\max_{a\in\mathcal A(s)}Q_T(s,a)
$$

对应的非正 Advantage target 为：

$$
A_T(s,a)=\operatorname{clip}
\left(Q_T(s,a)-\overline V_T(s),-2,0\right)
$$

因此 $V$、$Q$ 与 $A$ 使用同一棵冻结反事实树的定义，并满足 $Q_T(s,a)\leq\overline V_T(s)$。动作无法提高真实局面价值，但冻结模型的原 Value 估计可能偏高或偏低，所以反事实证据允许修正方向为正或负。真实自对弈根仍按 TD($\lambda$) 生成独立 Value target：

$$
G_t=-\left[(1-\lambda_{TD})V(s_{t+1})+\lambda_{TD}G_{t+1}\right]
$$

设 $E(s)$ 为实际展开并通过精确棋规评价的动作集合，其在改进 Policy 下的覆盖质量为：

$$
c_T(s)=\sum_{a\in E(s)}\pi^+(a\mid s)
$$

每棵树的节点 Policy 权重仍为：

$$
w_P(s)=\frac{n(s)}{\sum_{u\in\mathcal T_{root}}n(u)}
$$

树 Value 权重为：

$$
w_T(s)=w_P(s)c_T(s)
$$

因此每棵树满足 $\sum_s w_T(s)\leq1$。所有展开节点都使用 $\overline V_T(s)$ 训练反事实 Value，真实自对弈根另外保留权重为 1 的 TD target。Value loss 为：

$$
L_V=
\frac{
\sum_s w_{TD}(s)\rho(V_{new}(s)-G_t(s))+
\sum_s w_T(s)\rho(V_{new}(s)-\overline V_T(s))
}{
\sum_s(w_{TD}(s)+w_T(s))
}
$$

每个已展开节点及其精确子状态同时训练 Policy、Value、Advantage 与 latent transition。Dueling-Q 和 imagined Value 直接拟合同一份 $Q_T$，避免通过另一组 Value target 重构动作值。FCPI 损失为：

$$
L=w_P L_{CE}(\pi_{new},\pi^+)+
w_VL_V+
w_Q\operatorname{SmoothL1}(\widehat Q,Q^*)+
w_D L_D+w_I\operatorname{SmoothL1}(-V(\widehat z'),Q^*)
$$

其中每条树边都通过精确棋规生成 $s'$，并写入 `candidate_next_states`。$L_D$ 使用与监督训练相同的 latent cosine consistency。动作条件 dynamics 对树中每个已展开节点学习一步转移 $E(s)\rightarrow E(s')$，与 $K=2$ anchored latent MCTS 的运行时假设保持一致。

Melano FCPI HDF5 分别保存 `td_value_targets`、`tree_value_targets`、`td_value_weights`、`tree_value_weights` 和 `candidate_q`。训练日志分别输出 `value_td`、`value_tree`、合并后的 `value`、`dueling_q`、`dynamics` 与 `imagined_value`。summary 使用与 Melano backend 对应的 `mean_abs_advantage`、`max_abs_advantage`、`mean_policy_kl`、`mean_policy_total_variation` 与 `policy_top1_change_rate` 描述策略更新。

Melano 每轮依次执行自身 `current.pth` 自对战、局面采样、树一致反事实展开、PVA 与 latent-dynamics 目标构造、candidate 训练和 paired-game arena。每次运行由程序生成 `fcpi_YYYYMMDD_HHMMSS_id`，创建对应的 `data/runs/<run-id>/` 与 `models/runs/<run-id>/`。其中 HDF5 schema、candidate 和 current checkpoint 均属于 Melano，candidate 达到 arena gate 后原子写入该 run 的 `current.pth`。`summary.json` 记录预算、决策节点数、评价边数、最大深度和 arena 结果。

## 6. UCI

两个 C++ UCI 程序直接加载对应 LibTorch checkpoint：

```powershell
build\gadus\uci.exe `
	--model models\gadus\gadus.pth `
	--device cpu `
	--search-type only-mcts `
	--mcts-sims 100
```

```powershell
build\melano\uci.exe `
	--model models\melano\melano.pth `
	--device cuda `
	--search-type only-mcts `
	--mcts-sims 1000
```

UCI 输出包含 MultiPV、side-to-move `score cp`、节点数、NPS、耗时和 PV。搜索开始时先发布模型直觉结果，MCTS 期间按 `ProgressIntervalMS` 发布中间结果。

### 6.1 Gadus UCI options

Gadus 在 `uci` 握手中公开以下 option：

- `ModelPath`：checkpoint 路径。打包引擎默认读取可执行文件同目录的 `gadus.pth`。
- `Device`：`auto`、`cpu` 或 `cuda`。
- `SearchType`：`closed` 表示只使用模型 Policy，`only-mcts` 表示使用 Gadus MCTS。
- `MCTSSims`：MCTS simulation 上限，默认 `100`。客户端发送 `go nodes <n>` 时以 `<n>` 为当前搜索上限。
- `MCTSMinSims`：时间预算结束前至少完成的 simulation 数，默认 `0`。
- `MCTSBatchSize`：一次神经网络叶子批量，默认 `32`。
- `MoveTimeMS`：客户端未在 `go` 中给出时间时使用的固定思考时间，默认 `0`。
- `MoveOverheadMS`：从棋钟预算中预留的通信与落子时间，默认 `50`。
- `MinMoveTimeMS`、`MaxMoveTimeMS`：棋钟模式下的单步时间边界，默认 `50` 与 `10000`。
- `TimeDivisor`：把剩余时间按该除数分配给当前步，默认 `30.0`。
- `IncrementFraction`：当前步可使用的加秒比例，默认 `0.75`。
- `CPuct`、`CPuctBase`、`CPuctFactor`：PUCT 探索系数及其访问数 schedule，默认 `0.5`、`19652`、`1.0`。
- `FPUReduction`：未访问 child 的 first-play urgency 折减，默认 `0.15`。
- `VirtualLoss`：同一批 MCTS selection 的重复路径惩罚，默认 `0.0`。
- `RepetitionPolicyPenalty`：决策层对己方优势时重复和棋走法的排序惩罚，范围 $[0,1]$，默认 `0.0`。
- `InstantMateFirst`：发现一步将杀时优先选择该走法，默认 `false`。
- `ProgressIntervalMS`：UCI 中间 `info` 行的发布间隔，默认 `750`。
- `MultiPV`：输出的分析行数，默认 `5`。
- `ScoreScale`：把 $[-1,1]$ Value/Q 映射为 `score cp` 的显示比例，默认 `1000`。

在 Gadidae GUI 中，`Device` 与 `MultiPV` 使用专用控件，思考时间和 simulation 上限使用 `Search` 区域。`UCI options` 可以直接填写 JSON object，例如：

```json
{
  "SearchType": "only-mcts",
  "MCTSSims": 1000,
  "MCTSMinSims": 0,
  "MCTSBatchSize": 64,
  "CPuct": 0.5,
  "CPuctBase": 19652,
  "CPuctFactor": 1.0,
  "FPUReduction": 0.15,
  "VirtualLoss": 0.0,
  "RepetitionPolicyPenalty": 1.0,
  "InstantMateFirst": true,
  "ProgressIntervalMS": 750,
  "ScoreScale": 1000
}
```

`MCTSSims` 设置 Gadus 在 UCI 客户端没有提供 `go nodes <n>` 时使用的默认 simulation 上限。GUI Search 区域的 `Node / simulation limit` 会为每次搜索发送 `go nodes <n>` 并覆盖该次搜索的 `MCTSSims`。设为 `0` 时不发送 nodes 限制。`Device` 与 `MultiPV` 使用 GUI 对应控件填写，`Launch arguments` 通常留空。

其他 UCI 客户端使用标准命令设置同一组选项：

```text
setoption name SearchType value only-mcts
setoption name MCTSSims value 1000
setoption name MCTSBatchSize value 64
setoption name CPuct value 0.5
setoption name RepetitionPolicyPenalty value 1.0
setoption name InstantMateFirst value true
```

### 6.2 Gadidae 引擎目录

Windows 可将指定 checkpoint 包装到对应的 `models/<architecture>/` UCI 引擎目录：

```powershell
scripts\package_engine.bat gadus models\gadus\candidate3.pth
scripts\package_engine.bat melano models\melano\candidate.pth
```

Linux 使用：

```bash
bash scripts/package_engine.sh gadus models/gadus/candidate3.pth
bash scripts/package_engine.sh melano models/melano/candidate.pth
```

Gadus UCI 在缺少 `--model` 时只读取自身目录中的 `gadus.pth`，Melano UCI 只读取 `melano.pth`。该规则不引用仓库默认模型，也不依赖 EXE 名称。Windows 目录结构为：

```text
models/
	gadus/
		gadus.exe
		gadus.pth
		LibTorch DLLs
	melano/
		melano.exe
		melano.pth
		LibTorch DLLs
```

Linux 在各架构目录内使用 `gadus`、`melano` launcher，架构二进制分别为 `gadus.bin` 与 `melano.bin`，运行库位于各自的 `lib/`。重复打包同一架构会更新对应 checkpoint、UCI 程序和运行库。

UCI 客户端可以直接注册 `models/gadus/gadus.exe` 或 `models/melano/melano.exe`，无需填写命令行参数。Linux 注册对应架构目录内的 `gadus` 或 `melano` launcher。

## 7. GUI

`Gadidae` 是基于 GLFW、OpenGL 3.3、GLAD、Dear ImGui 与 FreeType 的原生图形程序。Simulator 与 Stadium 共用一个窗口，并只通过 UCI 与 Gadus、Melano、Stockfish 或其他引擎通信。

先完成依赖安装和构建：

```powershell
api\setup.bat
scripts\build.bat
```

Linux 使用：

```bash
GADIDAE_BUILD_GRAPHICS=1 bash api/setup.sh
GADIDAE_BUILD_GRAPHICS=1 bash scripts/build.sh
```

Windows 双击 `build/graphics/Gadidae.exe`，具有 X11 或 Wayland 图形会话的 Linux 运行 `build/graphics/Gadidae`。程序默认进入 Simulator，顶部模式控件可切换到 Stadium。GUI 的 FreeType 压缩支持静态编译进可执行文件。SSH 服务器需要 X11 forwarding、远程桌面或其他可见显示服务才能实际操作 GUI；`xvfb-run` 适合自动化启动测试，无法提供可交互画面。

Simulator 用于局面分析。Settings 的 `Engine` 区域填写 UCI 可执行文件、显示名称和设备，`Launch` 区域填写启动进程时附加的命令行参数，`Search` 区域填写时间或节点预算、MultiPV 行数与显示刷新间隔。`UCI options` 在握手完成后发送对应的 `setoption` 命令。`Run` 同时显示 `Open` 与 `Close`，勾选项表示实时分析的当前状态。也可以从命令行指定常用参数：

```powershell
build\graphics\Gadidae.exe `
	--mode simulator `
	--uci "models\gadus\gadus.exe" `
	--device cpu `
	--movetime-ms 3000 `
	--node-limit 0 `
	--multipv 8 `
	--font-size 20 `
	--theme dark
```

Stadium 用于同时组织多盘独立对局。`Tools > Matches` 可新建、进入和关闭对局，切换到 Simulator 或进入另一盘棋时，其余对局继续在后台运行。默认名称为 `#<id> <white> vs. <black>`，名称留空时自动采用该格式。每个席位都必须填写参赛者名称。Human toggle 打开时 UCI 设置整体置灰，走棋由用户直接在棋盘上完成；关闭时还必须填写 UCI 可执行文件，并可分别设置 UCI options、启动参数、计算预算与 MultiPV。`Match` 区域设置双方共用的初始时间、每步加秒、显示延迟与最大 ply，初始时间设为 0 时关闭棋钟。`Run > Start/Pause/Stop` 只控制当前进入的对局，关闭程序会终止全部对局拥有的 UCI 子进程。可从命令行预填首盘对局的双方引擎：

```powershell
build\graphics\Gadidae.exe `
	--mode stadium `
	--white-uci "models\gadus\gadus.exe" `
	--white-name "Gadus" `
	--black-uci "models\stockfish\stockfish.exe" `
	--black-name "Stockfish"
```

Appearance 提供 `Dark` 与 `Light` 应用主题、基础字号、受限范围内随窗口缩放的字号调整、棋盘配色预设、颜色编辑、坐标开关以及 `Vector`、`RhosGFX`、`Chessnut`、`Spatial`、`Cburnett`、`Fantasy` 棋子样式。棋子均由编译进程序的预计算几何绘制，SVG 的填充、线性渐变与描边在导入阶段完成预三角化，外部样式经过顶点去重和 zlib 压缩后嵌入可执行文件，发布目录只包含可执行文件。点击 Apply 后设置写入 `Gadidae` 可执行文件同目录的 `gui.json`，后续启动会自动恢复。已有的用户目录配置会在首次启动时迁移到该位置。只修改外观时会保留已经加载的 UCI 进程与正在进行的 Stadium 对局。`--font-size <px>`、`--theme dark`、`--theme light` 与 `--piece-style vector|rhosgfx|chessnut|spatial|cburnett|fantasy` 可覆盖本次启动的外观。第三方棋子样式的来源与许可记录在根目录 `THIRD_PARTY.md`。

### 7.1 导入棋子样式

`scripts/import_pieces.py` 把一套 SVG 的填充、线性渐变与描边转换为 `src/graphics/pieces.gpack` 中的预三角化索引网格。先安装脚本依赖：

```powershell
python -m pip install -r scripts\requirements.txt
```

输入目录必须包含 `wK.svg`、`wQ.svg`、`wR.svg`、`wB.svg`、`wN.svg`、`wP.svg`、`bK.svg`、`bQ.svg`、`bR.svg`、`bB.svg`、`bN.svg` 和 `bP.svg`。`--name` 使用以小写字母开头、仅包含小写字母、数字和连字符的稳定名称。Windows 导入并构建：

```powershell
python scripts\import_pieces.py `
	--input data\pieces\my-style `
	--name my-style
scripts\build.bat
```

Linux 导入并构建：

```bash
python scripts/import_pieces.py \
	--input data/pieces/my-style \
	--name my-style
GADIDAE_BUILD_GRAPHICS=1 bash scripts/build.sh
```

导入器一次性读取 12 个 SVG，完成预三角化、逐棋子无损顶点去重与 zlib 压缩。导入结束后项目只保留 `pieces.gpack`，不复制或保留 SVG，输入目录也不会被脚本修改。新名称向几何包追加一个内置样式，同名导入原子覆盖该样式。`--curve-step` 控制曲线路径采样间距，默认值为 `1.5`。较小值产生更平滑且更大的几何数据，较大值减少几何包体积与导入成本。构建时，Windows 通过 `RCDATA`、Linux 通过只读数据段把几何包嵌入 `Gadidae`。

每个新增的第三方样式都必须在根目录 `THIRD_PARTY.md` 中独立声明。SVG 不随项目保留，因此该声明是样式来源与授权信息的长期记录。来源应链接到固定 commit、固定版本或其他固定页面，并记录原作者、许可证名称和许可证全文链接。声明格式：

```markdown
## <Style Name>

- Author: <author or project>
- Source: <permanent source URL>
- License: <license name and version>
- License text: <license text URL>
```

导入前应确认许可证允许复制、修改和随项目分发，并遵守署名、相同方式共享、源代码提供或用途限制等条款。经过修改的素材应在对应声明中注明修改内容。由多个来源组合的样式应分别列出各来源及其许可证。

`UCI options` 使用 JSON object。图形程序按引擎公开的 option 名称进行匹配，未知名称会报告错误。`Device` 与 `MultiPV` 由专用控件管理，其余 options 用于 `Hash`、`Threads` 等引擎自有设置。`Device` 在目标引擎公开该 option 时生效，因此同一界面可直接运行 Stockfish 等通用 UCI 引擎。

Simulator 首次打开分析时在后台启动并加载一次 UCI 引擎，后续局面复用同一子进程。切换局面时会异步停止上一轮计算、丢弃其后续输出，再把最新局面交给已经加载的引擎。Gadus 与 Melano 的 UCI 命令循环和搜索线程彼此分离，MCTS 搜索会在批次边界响应 `stop`。图形程序在窗口持续移动或缩放时暂停 OpenGL 提交，操作停止后清空 GPU 命令队列并按新尺寸交换完整帧。

## 8. UCI 分析

`analyze.py` 直接连接 UCI 引擎，计算招法评分与 regret，并可使用 SQLite cache。它生成 `.cmt` 分析和带局面评价的 PGN。

```bash
python scripts/analyze.py \
	--input data/user-pgn/game.pgn \
	--uci models/stockfish/stockfish \
	--uci-depth 16 \
	--uci-multipv 8 \
	--analysis-cache data/user-pgn/analysis.sqlite \
	--pgn-comments
```

Windows 路径示例： `models/stockfish/stockfish.exe`。

## 9. Opening Book

```bash
bash scripts/run_opening.sh data/games.pgn 50000 data/openings.gen.bin
```

```powershell
scripts\run_opening.bat data\games.pgn 50000 data\openings.gen.bin
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

## 10. 验证

`scripts/build.bat` 与 `scripts/build.sh` 在发布可执行文件前运行 CTest。测试失败时构建停止，现场保留在 `build/.build-work/`。

Gadus 测试覆盖 `gadus_18_planes`、普通走法与特殊走法编码、棋规、Policy/Value 输出形状、有限数值、反向传播和 checkpoint 往返。本地 Windows CPU 烟测覆盖单盘 PGN 生成 HDF5、一步监督训练、closed 搜索、batched MCTS、两盘 paired arena、PGN 输出以及一轮 FCPI 的采样、反事实展开、训练、arena gate 和 current 晋升。

Melano 测试覆盖 `melano_square_tokens`、普通走法与升变编码、棋规、Policy/Value/Advantage 输出形状、Advantage 范围、动作条件 latent successor、$K=2$ anchored latent MCTS 路径、有限数值、反向传播和 checkpoint 往返。本地 Windows CPU 烟测覆盖单盘 PGN 生成含 `next_states` 与 `next_values` 的 HDF5、一步监督训练以及一轮树一致 Melano FCPI 的候选后继训练与 arena gate。
