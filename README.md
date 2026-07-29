# Gadidae

Gadidae 是一个实验性国际象棋神经网络引擎项目。当前架构如下：

- `Gadus`：ResNet + linear policy/value，使用 `gadus_18_planes` state encoding 和 `alphazero_64x73` move encoding。
- `Melano`：residual geometry attention + action-conditioned latent dynamics + source-destination policy/value/advantage，使用 `melano_square_tokens` state encoding 和 `sd_64x64_underpromo9` move encoding。

两套架构分别实现 preprocess、train、search、arena、事实—反事实策略迭代（Factual-Counterfactual Policy Iteration，FCPI）和通用国际象棋接口（Universal Chess Interface，UCI）。Gadus 另有独立的 Bellman 受限反事实迭代（Bellman-Restricted Counterfactual Iteration，BRCI）实验链路。它们共享 LibTorch、第五版层次数据格式（Hierarchical Data Format 5，HDF5）、chess-library、nlohmann-json、zlib 与构建基础设施。

## 记号与缩写

全文采用行棋方视角。除非所在小节另行定义，数学记号具有以下含义：

- $\mathcal X$ 是完整棋规环境状态空间。$x\in\mathcal X$ 包含棋盘、行棋方、王车易位权、吃过路兵状态、半回合计数和重复局面历史。
- $\mathcal A(x)$ 是环境状态 $x$ 的合法动作集合，$a\in\mathcal A(x)$ 是一个动作，$T(x,a)$ 是执行动作后的确定性环境状态。
- $z(x)\in\{-1,0,1\}$ 是终局环境状态从当前行棋方视角观察的规则结果。
- $\phi:\mathcal X\rightarrow\mathcal S$ 是架构自身的状态编码，$s=\phi(x)$ 是网络可见状态。若编码省略部分棋规历史，不同的 $x$ 可能映射到同一个 $s$。
- $\pi_\theta(a\mid s)$ 或 $P_\theta(a\mid s)$ 是网络参数 $\theta$ 产生的合法动作 Policy，$V_\theta(s)$ 是状态 Value，$Q(s,a)$ 是动作 Value，$A(s,a)$ 是 Advantage。上下文明确冻结某个模型时可省略下标 $\theta$。
- $\ell(s)$ 是网络输出的 Policy logits，$\theta$ 是网络参数。下标 `old`、`new`、`raw` 分别表示冻结源模型、更新后模型和未经限制的优化器提案。上标 $+$ 表示由局部策略改进构造的目标。
- $\widehat x$ 表示有限样本或有限搜索得到的估计量，$\overline x$ 表示经过树回传或加权聚合得到的量。任何其他下标或上标均在首次使用处定义。
- $L_{\mathrm{CE}}(p,q)=-\sum_iq_i\log p_i$ 是目标分布 $q$ 与预测分布 $p$ 的交叉熵，$\operatorname{MSE}(u,v)$ 是对应元素平方误差的均值。

算法与数据缩写如下：

- **PGN**：Portable Game Notation，对局记录格式。
- **FEN**：Forsyth-Edwards Notation，单局面描述格式。
- **UCI**：Universal Chess Interface，图形客户端与国际象棋引擎之间的文本协议。
- **HDF5**：Hierarchical Data Format 5，训练数据容器格式。
- **FCPI**：Factual-Counterfactual Policy Iteration，事实—反事实策略迭代。
- **BRCI**：Bellman-Restricted Counterfactual Iteration，Bellman 受限反事实迭代。
- **MCTS**：Monte Carlo Tree Search，蒙特卡洛树搜索。
- **PUCT**：Predictor + Upper Confidence bounds applied to Trees，结合网络先验的树上置信上界选择规则。
- **FPU**：First Play Urgency，未访问边的初始动作价值。
- **MC**：Monte Carlo，本文特指完整轨迹的终局回报样本。
- **KL**：Kullback-Leibler divergence，Kullback-Leibler 散度。
- **CE**：cross entropy，交叉熵。
- **MSE**：mean squared error，均方误差。
- **IMF**：Instant Mate First，存在一步将杀时将该动作置于最终排序首位。
- **RPP**：Repetition Policy Penalty，优势方规避接受或允许三次重复和棋的最终排序惩罚。
- **PV**：principal variation，引擎报告的主要变化。**MultiPV** 表示同时报告多条主要变化。
- **NPS**：nodes per second，每秒处理的搜索节点数。

实现相关缩写如下：

- **ResNet**：residual network，残差网络。
- **MLP**：multilayer perceptron，多层感知机。
- **CPU** 与 **GPU**：central processing unit 与 graphics processing unit。
- **CUDA**：NVIDIA 的 GPU 并行计算平台。
- **FP32** 与 **BF16**：32 位浮点数与 bfloat16。
- **MSVC**：Microsoft Visual C++ 编译工具链。
- **GUI**：graphical user interface，图形用户界面。
- **JSON**：JavaScript Object Notation。
- **ABI**：application binary interface。
- **SDK**：software development kit。
- **DLL**：dynamic-link library。
- **EXE**：Windows executable，可执行文件。
- **PID**：process identifier，进程标识符。
- **SVG**：Scalable Vector Graphics。
- **SHA-256**：256 位 Secure Hash Algorithm。
- **ETA**：estimated time of arrival。
- **SSH**：Secure Shell。
- **URL**：Uniform Resource Locator。
- **X11**：X Window System 的第 11 版协议。
- **RCDATA**：Windows 资源系统中的 raw data 资源类型。
- **CI**：confidence interval，置信区间。
- **LN**：layer normalization，层归一化。

LibTorch、CMake、CTest、Ninja、OpenGL、GLFW、GLAD、GLM、Dear ImGui、FreeType、chess-library、nlohmann-json 与 zlib 均为库、工具或项目名称。

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
	gadus_fcpi.sh
	gadus_brci.sh
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

- `include/gadus/`、`src/gadus/`：Gadus 独立实现。`dataset` 负责 PGN、HDF5 与监督训练，`game` 负责状态、动作和棋规，`model` 负责 ResNet Policy/Value，`searcher` 负责 closed 与 MCTS，`match` 负责 arena，`evolution` 负责 FCPI，`bellman` 负责 BRCI 的终局受限图、精确 minimax、参数回溯和晋升链路。
- `include/melano/`、`src/melano/`：Melano 独立实现。文件职责与入口形式和 Gadus 对称，状态编码、动作编码、网络、搜索与 FCPI 方程均由 Melano 自身实现。
- `preprocess.cpp`、`train.cpp`、`search.cpp`、`arena.cpp`、`fcpi.cpp`、`uci.cpp`：每套架构的六个命令入口。Gadus 另有 `brci.cpp`。
- `tests.cpp`：每套架构的状态编码、特殊走法、棋规、网络前反向、数值范围与 checkpoint 往返测试。
- `scripts/`：通用 UCI 工具、模型检查与构建启动脚本。
- `api/`：仓库本地 C++ 依赖与安装脚本。
- `include/graphics/`、`src/graphics/`：架构无关的原生 OpenGL 图形界面，通过 UCI 与引擎通信。
- `build/gadus/`、`build/melano/`：可直接运行的架构程序与运行 DLL。
- `build/graphics/`：原生 Gadidae 图形程序与图形运行库。
- `data/`：PGN、HDF5、开局书、分析结果和运行数据。
- `models/gadus/`、`models/melano/`：各架构的 LibTorch checkpoint，以及可直接注册到 UCI 客户端的引擎与运行库。
- `models/stockfish/`：UCI 引擎示例。

LibTorch checkpoint 的逻辑顶层固定为：

```text
model
arch
```

`model` 保存网络参数，`arch` 保存架构标识及构造网络所需的形状信息。

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

`setup.sh` 与 `build.sh` 默认根据 `DISPLAY` 或 `WAYLAND_DISPLAY` 判断当前 Linux 是否具有图形会话。Windows 的 `setup.bat` 与 `build.bat` 在 `auto` 模式下构建 Graphics。纯命令行服务器会跳过 GLFW、OpenGL 图形依赖和 `Gadidae` GUI，同时保留 Gadus、Melano 的完整命令行训练、搜索、竞技场、FCPI 与 UCI 链路，以及 Gadus BRCI。可使用 `GADIDAE_BUILD_GRAPHICS=0` 明确选择命令行构建，或使用 `GADIDAE_BUILD_GRAPHICS=1` 强制编译 GUI。

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
build/gadus/brci
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

`build/.build-work/` 保存 CMake 与 Ninja 的增量构建状态。后续构建复用未变化目标的对象文件。编译或 CTest 失败时，诊断信息保留在该目录，CTest 日志位于 `build/.build-work/Testing/Temporary/LastTest.log`。

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

### 4.1 状态与动作编码

Gadus 的状态编码为 `gadus_18_planes`：12 个棋子平面、1 个行棋方平面、4 个王车易位平面和 1 个 en passant 文件平面。每个平面按 8 个 rank bytes packbits，HDF5 中单个状态占 $18\times8$ bytes。

动作编码为 `alphazero_64x73`。每个起点对应 56 个八方向滑动动作、8 个马步动作和 9 个 underpromotion 动作：

$$
|\mathcal A|=64\times(56+8+9)=4672
$$

### 4.2 网络

模型由 ResNet trunk、linear Policy head 和 MLP Value head 组成：

$$
(\ell(s),V(s))=f_{\theta}(s),\qquad V(s)\in[-1,1]
$$

$\ell(s)$ 是 4672 维 Policy logits。合法动作概率为：

$$
P(a\mid s)=\frac{\exp \ell_a(s)}{\sum_{b\in\mathcal A(s)}\exp \ell_b(s)}
$$

### 4.3 数据预处理

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

`--has-cmt 0` 使用最终胜负生成 $V_{target}\in\{-1,0,1\}$。

```bash
build/gadus/preprocess \
	--input data/games.pgn \
	--output data/games.gadus.h5 \
	--has-cmt 1 \
	--chunk-size 16384 \
	--compression-level 1 \
	--log-every 10000
```

`--max-games` 控制读取对局上限，`--chunk-size` 控制 HDF5 扩展单元，`--compression-level` 控制 deflate 等级，`--log-every` 控制对局进度输出。

### 4.4 监督训练

记 PGN 中的实际走法为 $a^*$，对应的 one-hot 分布为 $\delta_{a^*}$，Value loss 权重为 $w_V$。监督损失为：

$$
L_{\mathrm{sup}}=
L_{\mathrm{CE}}\left(P_\theta(\cdot\mid s),\delta_{a^*}\right)+
w_V\operatorname{MSE}(V,V_{\mathrm{target}})
$$

```bash
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
```

`train` 每次创建新的 Gadus 模型。`--channels` 和 `--blocks` 决定网络结构，`--max-steps` 是本次训练步数上限，`--save-every` 控制原子 checkpoint 写入周期。

`--precision` 可取 `fp32` 或 `bf16`，默认 `fp32`。`bf16` 仅用于 CUDA 前向计算，Policy softmax、训练损失和指标累加使用 FP32，checkpoint 参数保持 FP32。CUDA 训练批次使用 pinned memory。

### 4.5 搜索

`closed` 直接按合法动作 Policy 排序。`only-mcts` 使用 batched leaf inference。记根视角下边 $(s,a)$ 的实际访问数为 $N(s,a)$，节点访问数为 $N(s)=\sum_aN(s,a)$，平均叶节点回传为 $Q(s,a)$。记 $c_0$、$b$ 与 $f$ 分别为 PUCT 的初始系数、基数和增长系数。动态探索系数为：

$$
c_{\mathrm{puct}}(N)=c_0+f\log\left(\frac{N+b+1}{b}\right)
$$

PUCT 选择分数为：

$$
S(s,a)=Q(s,a)+c_{\mathrm{puct}}(N(s))P(s,a)
\frac{\sqrt{N(s)+1}}{1+N(s,a)}-l_vN_v(s,a)
$$

$N_v(s,a)$ 和 $l_v$ 分别是同批 selection 的 virtual visits 与每个 virtual visit 的 virtual loss。记 $Q(s)$ 为节点已有访问的平均回传，$r_{\mathrm{FPU}}\geq0$ 为 FPU 折减系数。未访问边使用：

$$
Q_{\mathrm{FPU}}(s,a)=\operatorname{clip}\left(
Q(s)-r_{\mathrm{FPU}}\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right)
$$

叶节点 Value 沿路径逐 ply 取反并回传。MCTS 根分布保留一份 prior 平滑：

$$
P_{\mathrm{root}}(a\mid s)=\frac{N(s,a)+P(a\mid s)}
{\sum_{b\in\mathcal A(s)}(N(s,b)+P(b\mid s))}
$$

记 $p_a=P_{\mathrm{root}}(a\mid s)$。动态模拟预算使用根分布归一化熵 $H_\pi$、前两名访问差 $U_N$ 和前两名价值差 $U_Q$：

$$
H_\pi=-\frac{\sum_a p_a\log p_a}{\log|\mathcal A(s)|}
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},\qquad
U_Q=1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right)
$$

$$
u=\operatorname{clip}(0.5H_\pi+0.35U_N+0.15U_Q,0,1)
$$

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil
$$

其中 $N_{\min}$ 与 $N_{\mathrm{cap}}$ 是当前搜索的最小和最大 simulation 数，$u\in[0,1]$ 是根节点不确定性。IMF 与 RPP 位于最终决策层，只调整走法排序。

```bash
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

`--fen startpos` 使用标准初始局面，也可传入完整 FEN。CUDA 搜索只把合法动作的 Policy 从 GPU 传回 CPU。

### 4.6 Arena

`--games` 使用正偶数，使每个开局都能交换双方颜色完成配对。`--games-in-flight` 控制同时推进的对局数，candidate 与 baseline 各加载一次，轮到同一模型行棋的局面组成 inference batch。

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

从标准初始局面开始时，将 `--opening-book` 设为空。

### 4.7 FCPI

每轮 FCPI 使用冻结的 `current.pth` 进行模型自对战。记 $\mu(a\mid s)$ 为实际自对弈的行为策略，$T_b>0$ 为行为温度。Gadus 只对冻结 Policy 做温度变换：

$$
\mu(a\mid s)=
\frac{P(a\mid s)^{1/T_b}}{\sum_bP(b\mid s)^{1/T_b}}
$$

$T_b<1$ 使行为分布更集中于高 Policy 动作，$T_b=1$ 使用原 Policy 比例，$T_b>1$ 提高低 Policy 动作的采样概率。`behavior-temperature` 的作用范围是实际自对弈，反事实树候选宽度由边预算决定。

同一编码局面第 $n$ 次出现时，记 $N_{n-1}^{\mu}(s,a)$ 为前 $n-1$ 次访问中实际选择动作 $a$ 的次数。FCPI 使用概率欠额分配动作：

$$
a_n=\arg\max_a\left[n\mu(a\mid s)-N_{n-1}^{\mu}(s,a)\right]
$$

该调度使实际动作频率跟随 $\mu$，同时避免独立随机抽样持续漏掉某个正概率动作。只要同一局面继续出现，任意满足 $\mu(a\mid s)>0$ 的动作都会在有限次访问内取得样本。

规则终局提供自学习的事实信号。设终局时刻为 $\tau$，终局状态从行棋方视角观察的结果为 $z_\tau\in\{-1,0,1\}$，$G_t$ 为时刻 $t$ 的 MC 回报。完整结束的对局对整条轨迹执行无 bootstrap 的 MC 回传：

$$
G_\tau=z_\tau,\qquad G_t=-G_{t+1}
$$

负号对应行棋方在每个 ply 的切换。完整终局轨迹中各局面的 MC 权重为 $w_{\mathrm{MC}}=1$。达到 `max-plies` 的轨迹只提供反事实树根，其 MC 权重为 $w_{\mathrm{MC}}=0$。

每局按完整 Gadus 编码对局面去重，每个局面建立一棵独立反事实树。根节点一次评价全部合法动作，因此根目标覆盖完整合法动作集合。记 $B_{\mathrm{remain}}$ 为当前树尚未使用的反事实边预算，$w(s)$ 为非根节点 $s$ 的局部展开宽度。`--counterfactual-budget` 给出根节点之外可评价的动作边数，也是深层反事实树唯一的规模参数：

$$
w(s)=\min\left(|\mathcal A(s)|,B_{remain},
\max\left(2,\left\lceil\sqrt{B_{remain}}\right\rceil\right)\right)
$$

非根节点始终选择 Policy top-1，其余位置通过 Gumbel top-k 无放回采样得到：

$$
\operatorname{key}(a)=\log(P(a\mid s)+\varepsilon)+g_a,
\qquad g_a\sim\operatorname{Gumbel}(0,1)
$$

树使用冻结的 `current.pth` 批量评价精确棋盘子局面。记 $N_G(s,a)$ 为完整轨迹实际经过状态—动作对 $(s,a)$ 的次数，$G_i(s,a)$ 为其中第 $i$ 个终局回报。完整对局形成事实动作回报均值：

$$
\widehat G(s,a)=
\frac{1}{N_G(s,a)}
\sum_{i=1}^{N_G(s,a)}G_i(s,a)
$$

对样本中的环境状态 $x$ 及动作 $a$，记环境后继为 $x_a'=T(x,a)$，网络后继为 $s_a'=\phi(x_a')$。动作价值按信息强度依次确定：

$$
Q(s,a)=
\begin{cases}
z_{s,a}, & x_a'\text{ 是终局}\\
\widehat G(s,a), & N_G(s,a)>0\\
-\overline V(s_a'), & a\text{ 已展开}\\
V_{old}(s), & a\text{ 未展开}
\end{cases}
$$

其中 $z_{s,a}=-z(x_a')$ 是终局结果转换到父状态 $s$ 的行棋方视角后的值。事实回报是当前行为策略从该动作继续行棋至终局的 MC 样本均值，它取代该边的浅层 Value bootstrap。其余已展开动作使用子树回传值，冻结模型基线负责评价其余合法动作。

对只有模型回传信息的已展开动作：

$$
Q(s,a)=-\overline V(s_a')
$$

其中 $\overline V(s_a')$ 是子树从叶节点回传到网络后继 $s_a'$ 的聚合值。

设冻结 Policy 下的局部均值为：

$$
m(s)=\sum_aP_{old}(a\mid s)Q(s,a)
$$

局部 Policy target 使用 KL 正则化的 Policy improvement。这里 $D_{\mathrm{KL}}(p\|q)$ 表示分布 $p$ 相对于分布 $q$ 的 KL 散度：

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

KL 项出现在 Policy target 的闭式构造中，用于限制一次有限预算规划对原 Policy 的偏离。训练总损失由 Policy 拟合项和 Value 拟合项组成。由于 $Q\in[-1,1]$，Policy improvement 直接使用 Value 原生尺度。

Gadus 节点在目标 Policy 下的回传值为：

$$
\overline V(s)=\sum_a\pi^+(a\mid s)Q(s,a)
$$

定义反事实残差（counterfactual residual）$\delta_{\mathrm{CF}}(s)$：

$$
\delta_{\mathrm{CF}}(s)=\overline V(s)-V_{old}(s)
$$

记 $E(s)\subseteq\mathcal A(s)$ 为状态 $s$ 中已经显式展开的动作集合。由于未展开动作满足 $Q(s,a)=V_{old}(s)$，只有 $E(s)$ 中的动作能够贡献非零残差：

$$
\delta_{\mathrm{CF}}(s)=
\sum_{a\in E(s)}
\pi^+(a\mid s)\left(Q(s,a)-V_{old}(s)\right)
$$

因此树覆盖率已经进入残差本身。预算较小时，未展开概率质量把 $\overline V(s)$ 拉回冻结基线。预算增加后，更多反事实结论进入修正。完整对局结果同时训练局面 Value，并作为实际动作的事实 $Q$ 参与 Policy 比较。这一步把终局事实传入动作排序，使各层冻结 Value 的自洽关系受到事实回报校正。

对同一局面的两个动作 $a$ 与 $b$，一次精确拟合后的目标赔率满足：

$$
\log\frac{\pi^+(a\mid s)}{\pi^+(b\mid s)}
=
\log\frac{P_{old}(a\mid s)}{P_{old}(b\mid s)}
+Q(s,a)-Q(s,b)
$$

若后续迭代持续得到同一正动作价值差 $\Delta=Q(s,a)-Q(s,b)>0$，第 $k$ 次局部改进后的对数赔率累计增加 $k\Delta$。因此排序翻转所需的理想拟合迭代数满足：

$$
k>
\frac{\log P_0(b\mid s)-\log P_0(a\mid s)}{\Delta}
$$

这证明了正回报差异能够在有限次局部改进中改变排序。该结论的范围是固定动作价值差下的局部 Policy 排序。动作的博弈论真值仍由终局样本或完整求解确定。根节点全动作评价与概率欠额采样为每个正概率动作提供有限访问机会。

待展开前沿按到达概率与 Bellman residual 排序。记 $p_{\mathrm{reach}}(s)$ 为冻结行为策略沿当前树路径到达状态 $s$ 的概率，$d$ 为子节点 $s'$ 的树深度：

$$
priority(s')=p_{\mathrm{reach}}(s)P(a\mid s)
\left(|-V(s')-V(s)|+\frac{1}{\sqrt{2+d}}\right)
$$

因此预算会自然分配给较可能到达、局部判断不一致且仍靠近根部的节点。设该节点展开边数为 $n(s)$，$\mathcal T_{\mathrm{root}}$ 为同一反事实根生成的决策节点集合，$w_T(s)$ 与 $w_P(s)$ 分别为树 Value 与 Policy 的训练权重：

$$
w_T(s)=w_P(s)=
\frac{n(s)}{\sum_{u\in\mathcal T_{root}}n(u)}
$$

每棵树的 Policy 总权重和反事实 Value 总权重均为 1：

$$
\sum_s w_P(s)=\sum_s w_T(s)=1
$$

定义 $\operatorname{SL1}(x)$ 为阈值等于 1 的 SmoothL1 函数：

$$
\operatorname{SL1}(x)=
\begin{cases}
\frac12x^2, & |x|<1\\
|x|-\frac12, & |x|\geq1
\end{cases}
$$

终局 MC 目标 $G_t$ 与冻结树目标 $\overline V(s)$ 作为两个独立的 SmoothL1 项进入 Value 损失：

$$
L_V=
\frac{
\sum_s w_{MC}(s)\operatorname{SL1}\left(V_{new}(s)-G_t(s)\right)+
\sum_s w_T(s)\operatorname{SL1}\left(V_{new}(s)-\overline V(s)\right)
}{
\sum_s\left(w_{MC}(s)+w_T(s)\right)
}
$$

其中 $\overline V(s)$ 是冻结模型经过有限反事实规划得到、在训练时停止梯度传播的目标。有限展开与根节点判断一致时，$\delta_{\mathrm{CF}}(s)=0$，树目标等于冻结的 $V_{old}(s)$。多步回传与根节点判断产生差异时，$\delta_{\mathrm{CF}}(s)\neq0$，关系 $\overline V(s)=V_{old}(s)+\delta_{\mathrm{CF}}(s)$ 为原 Value 提供修正方向。冻结基线约束有限树覆盖范围，SmoothL1 的有界梯度控制单次修正幅度。

记 $L_{\mathrm{CE}}(p,q)$ 为目标分布 $q$ 与预测分布 $p$ 的交叉熵。Policy 损失为：

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

后台启动脚本为：

```bash
bash scripts/gadus_fcpi.sh
```

脚本默认使用 `PRECISION=bf16`、较大的批次与 batched games。可通过环境变量覆盖，例如 `PRECISION=fp32 BATCH_SIZE=512 bash scripts/gadus_fcpi.sh`。

脚本会等待 FCPI 完成全部迭代，再调用独立 Arena 追加一次最终 `current.pth` 对同一 run 的 `initial.pth` 的 `closed` paired 对战。它复用 `EVAL_GAMES`、开局书、并发数和最大步数，并将 MCTS 关闭。结果写入 `summary.json` 的 `final_arena`，棋谱写入 `current_vs_initial.pgn`。该累计赛只报告整次运行的净变化，不参与 candidate 晋升。直接运行 `build/gadus/fcpi` 时，流程在 FCPI 迭代结束处终止。

### 4.8 BRCI

BRCI 从完整自对弈轨迹建立终局锚定的有限受限图，在该图上精确计算 minimax 目标，再用独立受限图限制参数更新。

#### 4.8.1 轨迹采样与受限图

冻结源模型在 `closed` 模式下给出合法动作 Policy。记行为温度为 $T_b>0$，行为分布为：

$$
\mu(a\mid s)=
\frac{\pi_{\mathrm{old}}(a\mid s)^{1/T_b}}
{\sum_{b\in\mathcal A(x)}\pi_{\mathrm{old}}(b\mid s)^{1/T_b}}
$$

同一网络可见状态第 $n+1$ 次出现时，代码选择概率缺口最大的动作：

$$
a_{n+1}=
\arg\max_a\left[(n+1)\mu(a\mid s)-N_n(s,a)\right]
$$

$N_n(s,a)$ 是前 $n$ 次出现中动作 $a$ 的实际选择次数。该确定性配额规则让正概率动作随重复采样获得覆盖，同时避免独立随机抽样造成的额外方差。

只有由棋规判定结束的完整轨迹进入受限图。达到 `max-plies` 的轨迹计入诊断，但不生成图节点或训练目标。代码以“起始 FEN + 完整动作序列”为轨迹签名，并对签名执行由 `seed` 固定的哈希分区。同一条轨迹在整个 run 中始终属于训练图或测试图。测试签名的期望比例约为 $1/\sqrt{G}$，其中 $G$ 是每轮计划对局数，因此每轮测试轨迹的期望数量约为 $\sqrt G$。两个图分别跨迭代追加。

受限图采用如下节点合并规则：两个轨迹前缀当且仅当起始 FEN 相同且动作序列逐项相同时表示同一个图节点。同一路径前缀可以合并，不同历史路径不会因棋盘摆放相同而合并。因此重复局面历史、半回合计数和其他棋规状态保留在图结构中。网络输入仍是 $\phi(x)$，所以状态编码省略的历史可能形成不可约拟合误差。

记第 $k$ 轮累积后的有限训练图或测试图为 $\mathcal G_k$，节点 $x$ 的已观察动作集合为 $\mathcal A_k(x)$。每个叶节点都是规则终局。BRCI 中的 Bellman 计算指受限图上的 Bellman 最优算子：

$$
(\mathcal B_k V)(x)=
\begin{cases}
z(x),&x\text{ 是终局}\\
\displaystyle\max_{a\in\mathcal A_k(x)}
\left[-V(T(x,a))\right],&x\text{ 是内部节点}
\end{cases}
$$

符号 $V_k^*$ 表示第 $k$ 个受限图上的最优 Value，即受限 Bellman 算子的不动点：

$$
V_k^*=\mathcal B_kV_k^*
$$

对应的动作 Value 为：

$$
Q_k^*(x,a)=-V_k^*(T(x,a))
$$

图按动作前缀建立，因此是有限无环图。终局节点提供边界值，`RestrictedGraph::solve()` 按节点深度逆序执行 Bellman 最优递推，唯一确定所有 $V_k^*$ 与 $Q_k^*$。`materialize_graph()` 再把这些解转换成 Policy 和 Value 训练目标。该计算只使用图结构与规则终局值，截断局面不会进入受限图。

#### 4.8.2 训练目标

模型 Policy 在受限动作集合上重新归一化，记为 $\pi_{\theta,k}(a\mid\phi(x))$。冻结源模型的受限 Policy 为 $\pi_{\mathrm{old},k}$。定义：

$$
m_k(x)=
\sum_{a\in\mathcal A_k(x)}
\pi_{\mathrm{old},k}(a\mid\phi(x))Q_k^*(x,a)
$$

Policy 目标采用单位温度的 KL 正则化改进：

$$
\pi_k^+(a\mid\phi(x))=
\frac{
\pi_{\mathrm{old},k}(a\mid\phi(x))
\exp\left(Q_k^*(x,a)-m_k(x)\right)
}{
\sum_{b\in\mathcal A_k(x)}
\pi_{\mathrm{old},k}(b\mid\phi(x))
\exp\left(Q_k^*(x,b)-m_k(x)\right)
}
$$

当节点只有一条已观察边，或所有已观察动作的 $Q_k^*$ 相等时，该节点不提供 Policy 梯度。每个内部节点都使用 $V_k^*(x)$ 监督 Value。记训练节点集合为 $\mathcal X_k^{\mathrm{train}}$，Policy 有效节点集合为 $\mathcal X_{P,k}^{\mathrm{train}}$，训练损失为：

$$
L_{\mathrm{BRCI}}=
\frac{1}{|\mathcal X_{P,k}^{\mathrm{train}}|}
\sum_{x\in\mathcal X_{P,k}^{\mathrm{train}}}
L_{\mathrm{CE}}\left(
\pi_{\theta,k}(\cdot\mid\phi(x)),
\pi_k^+(\cdot\mid\phi(x))
\right)
+
\frac{1}{|\mathcal X_k^{\mathrm{train}}|}
\sum_{x\in\mathcal X_k^{\mathrm{train}}}
\left[V_\theta(\phi(x))-V_k^*(x)\right]^2
$$

训练 HDF5 保存：

```text
states
legal_indices
legal_counts
policy_targets
policy_weights
value_targets
value_weights
components
```

`components` 标识起始 FEN 对应的图分量。具有不同完整历史但相同网络编码的节点保留为独立样本。

#### 4.8.3 独立图误差

记独立测试图的内部节点集合为 $\mathcal X_k^{\mathrm{test}}$。Value 均方误差定义为：

$$
E_{V,k}(\theta)=
\frac{1}{|\mathcal X_k^{\mathrm{test}}|}
\sum_{x\in\mathcal X_k^{\mathrm{test}}}
\left[V_\theta(\phi(x))-V_k^*(x)\right]^2
$$

测试节点上的 Policy 单节点遗憾定义为：

$$
g_{P,k}(\theta;x)=
V_k^*(x)-
\sum_{a\in\mathcal A_k(x)}
\pi_{\theta,k}(a\mid\phi(x))Q_k^*(x,a)
$$

由 $V_k^*(x)=\max_aQ_k^*(x,a)$ 可得 $g_{P,k}(\theta;x)\geq0$。它表示模型受限 Policy 的期望动作值与该节点最优动作值之间的差距。仅具有多个不同动作值的节点进入 Policy 评价，记这些节点组成 $\mathcal X_{P,k}^{\mathrm{test}}$。Policy 平均遗憾定义为：

$$
E_{P,k}(\theta)=
\frac{1}{|\mathcal X_{P,k}^{\mathrm{test}}|}
\sum_{x\in\mathcal X_{P,k}^{\mathrm{test}}}
g_{P,k}(\theta;x)
$$

从源模型 $\theta_{\mathrm{old}}$ 到候选模型 $\theta_{\mathrm{new}}$ 的两项误差减少量为：

$$
\Delta_{V,k}=
E_{V,k}(\theta_{\mathrm{old}})
-E_{V,k}(\theta_{\mathrm{new}})
$$

$$
\Delta_{P,k}=
E_{P,k}(\theta_{\mathrm{old}})
-E_{P,k}(\theta_{\mathrm{new}})
$$

代码要求：

$$
\Delta_{V,k}>10^{-6},
\qquad
\Delta_{P,k}>10^{-6}
$$

$10^{-6}$ 是 FP32/BF16 推理与参数插值的数值分辨容差。满足两项不等式表示 candidate 在固定独立受限图上更接近精确 minimax Value，并且其受限 Policy 的期望动作值更高。

#### 4.8.4 有限参数回溯

优化器先产生未经限制的提案 $\theta_{\mathrm{raw}}$。令：

$$
d=\theta_{\mathrm{raw}}-\theta_{\mathrm{old}}
$$

BRCI 检查 12 个有限步长：

$$
\theta_j=\theta_{\mathrm{old}}+2^{-j}d,
\qquad j\in\{0,1,\ldots,11\}
$$

每个步长都在同一独立测试图上重新计算 $E_{V,k}$ 与 $E_{P,k}$。只有 Value 均方误差和 Policy 平均遗憾的减少量都超过数值容差时，该步长才进入候选集合。

将测试图按起始 FEN 分成 $n$ 个能够同时评价 Policy 和 Value 的分量。第 $i$ 个分量上的联合改进指示量为：

$$
I_i(\theta_j)=
\mathbf 1\left[
\Delta_{V,k}^{(i)}(\theta_j)>10^{-6}
\land
\Delta_{P,k}^{(i)}(\theta_j)>10^{-6}
\right]
$$

经验联合改进率为：

$$
\widehat p_k(\theta_j)=
\frac{1}{n}\sum_{i=1}^{n}I_i(\theta_j)
$$

代码固定置信失败概率 $\beta=0.05$，并计算 Hoeffding 单侧下界：

$$
\underline p_k(\theta_j)=
\max\left(
0,
\widehat p_k(\theta_j)-
\sqrt{\frac{\log(1/\beta)}{2n}}
\right)
$$

候选集合优先选择 $\underline p_k$ 最大的步长。下界相同时，选择 Value 均方误差和 Policy 平均遗憾的相对降幅之和更大的步长。独立图缺少任一项可比较指标时，本轮拒绝。

通过受限图判别的 candidate 继续参加 `closed` paired Arena。Arena 达到 `eval-min-net-wins` 后，candidate 才会原子写入该 run 的 `current.pth`。因此受限图判别负责有限图上的数值性质，Arena 负责实际走棋分布上的最终晋升。

#### 4.8.5 受限 Bellman 解的收敛

设完整国际象棋博弈图为 $\mathcal G$。完整图上的 Bellman 最优算子定义为：

$$
(\mathcal B V)(x)=
\begin{cases}
z(x),&x\text{ 是终局}\\
\displaystyle\max_{a\in\mathcal A(x)}
\left[-V(T(x,a))\right],&x\text{ 是内部节点}
\end{cases}
$$

符号 $V^*$ 表示完整图上的最优 Value，$Q^*$ 表示对应的最优动作 Value：

$$
V^*=\mathcal BV^*,
\qquad
Q^*(x,a)=-V^*(T(x,a))
$$

若受限图序列满足以下条件：

1. 图节点区分所有影响棋规的环境历史。
2. 每个叶节点都是精确规则终局。
3. 已收录的状态动作边不会从后续图中删除。
4. 每条可达合法边都在有限阶段后被收录。

由第三项可得嵌套关系：

$$
\mathcal G_1\subseteq\mathcal G_2\subseteq\cdots\subseteq\mathcal G
$$

由第四项可得穷尽覆盖：

$$
\bigcup_{k=1}^{\infty}\mathcal G_k=\mathcal G
$$

**命题（穷尽覆盖下的有限阶段一致性）：** 存在有限整数 $K$，使得对所有 $k\geq K$、任意可达节点 $x$ 及其合法动作 $a$，受限 Bellman 解满足：

$$
V_k^*(x)=V^*(x),
\qquad
Q_k^*(x,a)=Q^*(x,a)
$$

因此：

$$
\lim_{k\to\infty}V_k^*(x)=V^*(x)
$$

$$
\lim_{k\to\infty}Q_k^*(x,a)=Q^*(x,a)
$$

**证明：** 按照第 4.8.1 节定义的节点合并规则，每个图节点由起始 FEN 与完整动作前缀唯一确定，重复局面历史和其他棋规状态不会在图合并过程中丢失。国际象棋的可达状态动作边集合有限。对每条边 $e$，记其首次进入受限图的阶段为 $k_e$。第四项保证每个 $k_e$ 都是有限整数。取：

$$
K=\max_{e\in\mathcal E(\mathcal G)}k_e
$$

则对所有 $k\geq K$ 都有 $\mathcal G_k=\mathcal G$。以下证明完整覆盖后的 Bellman 解相等。规则终局满足 $V_k^*(x)=V^*(x)=z(x)$。假设所有剩余合法长度小于 $d$ 的节点均满足 $V_k^*=V^*$，则对剩余合法长度为 $d$ 的节点有：

$$
\begin{aligned}
V_k^*(x)
&=\max_{a\in\mathcal A(x)}
\left[-V_k^*(T(x,a))\right]\\
&=\max_{a\in\mathcal A(x)}
\left[-V^*(T(x,a))\right]\\
&=V^*(x)
\end{aligned}
$$

由剩余合法长度上的逆向归纳，对所有 $k\geq K$ 都有 $V_k^*=V^*$。再由 $Q_k^*(x,a)=-V_k^*(T(x,a))$ 可得 $Q_k^*=Q^*$。两个序列从第 $K$ 阶段起保持不变，因此上述极限成立。换言之，在穷尽覆盖条件下，BRCI 的受限 Bellman 解具有有限阶段稳定性，极限等价是该性质的直接结果。

当前实现满足前两项，并在单次运行中满足第三项。第四项由行为 Policy、起始局面覆盖、运行迭代数和 `max-plies` 共同决定。当前实现能够精确求解已经构建的有限受限图，并在独立受限图上测量参数更新。完整国际象棋上的覆盖程度由实际采样决定，Arena 用于检验有限图之外的走棋表现。

#### 4.8.6 Value 误差与 Policy 遗憾的条件收敛

4.8.5 证明受限 Bellman 目标在穷尽覆盖条件下与完整博弈目标达到有限阶段一致。本节给出网络 Value 预测误差与 Policy 遗憾趋于零所需的附加条件。定义联合误差：

$$
E_k(\theta)=E_{V,k}(\theta)+E_{P,k}(\theta)
$$

记模型类在第 $k$ 个图上的最小可达误差为：

$$
E_k^{\inf}=\inf_\theta E_k(\theta)
$$

记一次训练与受限回溯组成的拟合算子为 $\mathcal F_k$。假设：

1. 存在有限的 $K$，使 $\mathcal G_k=\mathcal G$ 对所有 $k\geq K$ 成立。
2. 模型类能够表示 $V^*$ 和某个只选择最优动作的 Policy，因此 $E_K^{\inf}=0$。
3. 存在与拟合次数无关的常数 $\eta\in[0,1)$，使 $\mathcal F_K$ 满足：

$$
E_K(\mathcal F_K(\theta))-E_K^{\inf}
\leq
\eta\left[E_K(\theta)-E_K^{\inf}\right]
$$

连续执行 $m$ 次成功拟合后，由归纳法可得：

$$
E_K(\mathcal F_K^m(\theta))-E_K^{\inf}
\leq
\eta^m\left[E_K(\theta)-E_K^{\inf}\right]
$$

因为 $\eta<1$ 且 $E_K^{\inf}=0$，所以：

$$
\lim_{m\to\infty}
E_K(\mathcal F_K^m(\theta))=0
$$

当联合误差大于零时，每次成功拟合都会严格降低该误差。完整博弈图有限，并且每个测试节点具有正权重，因此 Value 均方误差趋于零可推出每个节点的网络 Value 收敛到 $V^*$。Policy 平均遗憾趋于零可推出网络 Policy 的期望动作值收敛到该节点的最优动作值。最优动作存在并列时，Policy 分布不需要收敛到唯一分布。

上述条件收敛结论只涉及完整博弈节点上的 Value 预测误差和 Policy 遗憾，不要求网络参数、Policy logits 或并列最优动作之间的概率分配收敛。神经网络存在通道置换和等价参数化，不同参数可以表示相同函数。

当前实现对 12 个有限回溯步长计算独立图误差，并只把误差严格下降的步长交给 Arena。这保证每个通过受限图判别的 candidate 在该独立图上变好。当前实现尚未证明对任意非最优参数都存在能够通过判别的步长，也未证明 Arena 会接受每个满足受限图条件的 candidate。因此，上述结论给出 BRCI 从 Bellman 目标通向 Value 误差与 Policy 遗憾收敛的充分条件，实际运行仍可能因拟合停滞或 Arena 拒绝而保持原模型。

`summary.json` 记录每轮新增终局轨迹、训练图和测试图规模、分支节点数、Policy 目标变化、训练损失、12 个回溯步长的 Value 均方误差与 Policy 平均遗憾、分量改进率、Arena 结果与最终晋升状态。

复现实验使用：

```bash
bash scripts/gadus_brci.sh
```

`build/gadus/brci --help` 列出可覆盖的采样、拟合与 Arena 参数。

每次运行创建独立的 `brci_<timestamp>_<id>` 数据目录和模型目录。受限图在该次运行的内存中跨迭代累积，candidate 是否通过晋升不影响已经取得的终局事实。

## 5. Melano

### 5.1 状态与动作编码

Melano 将局面编码为 67 个整数：64 个棋盘格 piece tokens、1 个行棋方 token、1 个王车易位 bitmask 和 1 个 en passant 文件 token。空格编码为 0，白方六种棋子编码为 1 到 6，黑方六种棋子编码为 7 到 12。模型把它们展开为 64 个 square tokens 与 1 个 global token。piece、square、side、castling 和 en passant embeddings 共同形成初始表示。

动作编码为 `sd_64x64_underpromo9`：

$$
|\mathcal A|=64\times64+64\times9=4672
$$

普通走法和后升变使用 source-destination 编码。马、象、车升变使用每个起点的 3 个方向乘 3 个升变棋子编码。

### 5.2 网络

每个 geometry attention block 包含 pre-norm multi-head self-attention、32 类静态棋盘几何关系 bias、global token 生成的动态关系 bias、residual connection 和 pre-norm feed-forward network。

Policy head 对 source 与 destination projections 做缩放点积，生成 $64\times64$ logits，并拼接 576 个 underpromotion logits。Value head 从 global token 输出：

$$
V(s)\in[-1,1]
$$

记 $u_A(s,a)$ 为 Advantage head 在动作 $a$ 上的无界标量输出。该 head 的最终输出为：

$$
A(s,a)=-2\tanh^2(u_A(s,a))\in[-2,0]
$$

动作价值定义为：

$$
Q(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

记精确状态 encoder 为 $E$，动作条件 latent dynamics 为 $D$。潜在转移使用动作 embedding 条件化一个 residual geometry-attention block。记 $\mathcal B$ 为该 block，$c(a)$ 为动作条件向量，$g(a)$ 为逐通道门控 logits，$\sigma$ 为 sigmoid，$\odot$ 为逐元素乘法：

$$
h=E(s)
$$

$$
\widehat h'=D(h,a)=\operatorname{LN}\left(h+\sigma(g(a))\odot
\left[\mathcal B(h+c(a))-h\right]\right)
$$

### 5.3 数据预处理

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

对带评注 PGN，记 $score_{\mathrm{stm}}(s)$ 为转成当前行棋方视角后的兵值评分。Value target 为：

$$
V_{target}(s)=\tanh\left(\frac{score_{stm}(s)}{3}\right)
$$

设 $s'=\phi(T(x,a))$ 为动作后的网络状态。$V_{\mathrm{mover}}(s)$ 与 $V_{\mathrm{mover}}(s')$ 分别表示动作前后换算到动作行棋方视角的 Value target。监督 Advantage target 为：

$$
A_{target}(s,a)=\operatorname{clip}
\left(V_{mover}(s')-V_{mover}(s),-2,0\right)
$$

每盘棋的第一个有效评分建立 Value 基准，对应动作的 $A_{target}=0$。`--has-cmt 0` 使用终局结果生成 Value target，监督损失中的 dueling-Q 项权重取 0。

`next_values` 独立保存走后局面在新行棋方视角下的 Value target。它直接来自当前走法后的评注，因此每盘棋第一条有效评注也能监督走后 latent。

```bash
build/melano/preprocess \
	--input data/games.cmt.pgn \
	--output data/games.melano.h5 \
	--has-cmt 1 \
	--chunk-size 4096 \
	--compression-level 1 \
	--log-every 10000
```

`--max-games` 控制读取对局上限，`--chunk-size` 与 `--compression-level` 控制 Melano HDF5 写入。

### 5.4 监督训练

监督训练的 dueling action Value 与目标为：

$$
\widehat Q(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

$$
Q_{target}(s,a)=\operatorname{clip}
\left(V_{target}(s)+A_{target}(s,a),-1,1\right)
$$

精确棋规生成动作后的状态 $s'$。预测 latent 与停止梯度的精确 successor latent 分别为：

$$
\widehat h'=D(E(s),a),
\qquad
\overline h'=\operatorname{stopgrad}(E(s'))
$$

潜在一致性损失 $L_D$ 逐 token 比较走后预测与精确棋规生成的走后状态编码：

$$
L_D=1-\frac{1}{65}\sum_{i=1}^{65}
\cos(\widehat h'_i,\overline h'_i)
$$

Imagined Value 损失 $L_I$ 定义为：

$$
L_I=\operatorname{MSE}(V(\widehat h'),V_{target}(s'))
$$

记 PGN 中实际走法的 one-hot 分布为 $\delta_{a^*}$。记 $\lambda_V$、$\lambda_Q$、$\lambda_D$ 与 $\lambda_I$ 分别为 Value、dueling action Value、latent dynamics 和 imagined Value 的损失系数。完整监督损失为：

$$
L_{\mathrm{sup}}=
L_{\mathrm{CE}}\left(P_\theta(\cdot\mid s),\delta_{a^*}\right)+
\lambda_V\operatorname{MSE}(V,V_{\mathrm{target}})+
\lambda_Q\operatorname{MSE}(\widehat Q,Q_{target})+
\lambda_D L_D+\lambda_I L_I
$$

```bash
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
```

`train` 每次创建新的 Melano 模型。`--channels` 和 `--blocks` 决定 geometry attention 宽度与层数，checkpoint 通过临时文件和 rename 原子写回。Melano checkpoint 的逻辑顶层为 `model` 与 `arch`，其中 `arch` 保存架构标识、`channels`、`blocks` 和 `action_size`。

`--precision` 可取 `fp32` 或 `bf16`，默认 `fp32`。`bf16` 用于 CUDA 前向计算，Policy softmax、P/V/A 与 latent dynamics 损失、指标累计和 checkpoint 参数保持 FP32。CUDA 训练批次使用 pinned memory。

### 5.5 搜索

`closed` 按 Melano Policy 排序。`only-mcts` 使用 anchored latent MCTS。锚定周期 $K=2$ 表示每经过两个 ply 重新从精确棋盘编码 latent，因此任意预测 latent 最多跨越一个动作。每条边还具有一份伪访问。本文所称的伪访问，是 $V+A$ 提供的动作价值先验以一个统计样本的权重参与边价值估计。实际访问次数与叶节点回传保持独立计数。

MCTS 的每个节点都保留精确 `chess::Board`，合法走法、将军、终局、重复局面与五十回合规则由棋规计算。网络评价在偶数深度重新建立精确 latent 锚点，在奇数深度使用动作条件 latent transition：

$$
h_d=
\begin{cases}
E(s_d),&d\bmod 2=0\\
D(h_{d-1},a_{d-1}),&d\bmod 2=1
\end{cases}
$$

$$
(P_d,V_d,A_d)=\mathcal H(h_d)
$$

其中 $d$ 是根节点到当前节点的 ply 深度，$h_d$ 是该节点的 latent，$\mathcal H$ 是共享的 Policy、Value 与 Advantage heads。因此任意预测 latent 与最近的精确编码只相隔一个动作，运行时分布与当前一步 dynamics 训练目标一致。偶数深度节点缓存 $E(s_d)$，奇数深度 latent 只在批量评价期间存在，从而限制设备内存。`search` 输出中的 `exact_evaluations` 与 `latent_evaluations` 分别报告两类网络评价位置数。

$$
Q_{\mathrm{prior}}(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

记 $N(s,a)$ 为边的实际访问次数，$N(s)=\sum_aN(s,a)$ 为节点访问次数，$Q_{\mathrm{MCTS}}(s,a)$ 为实际访问的平均叶节点回传。对已有实际访问的边：

$$
Q_{\mathrm{edge}}(s,a)=
\frac{N(s,a)Q_{\mathrm{MCTS}}(s,a)+Q_{\mathrm{prior}}(s,a)}
{N(s,a)+1}
$$

未访问边使用：

$$
Q_{\mathrm{edge}}(s,a)=\operatorname{clip}\left(
Q_{\mathrm{prior}}(s,a)-r_{\mathrm{FPU}}
\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right)
$$

其中 $r_{\mathrm{FPU}}\geq0$ 是 FPU 折减系数。记 $c_0$、$b$ 与 $f$ 分别为 PUCT 的初始系数、基数和增长系数。Melano 的动态探索系数为：

$$
c_{\mathrm{puct}}(N)=c_0+f\log\left(\frac{N+b+1}{b}\right)
$$

记 $N_v(s,a)$ 为同一批 selection 中施加在边上的 virtual visits，$l_v$ 为每个 virtual visit 的 virtual loss。PUCT 选择分数为：

$$
S(s,a)=Q_{\mathrm{edge}}(s,a)+c_{\mathrm{puct}}(N(s))P(s,a)
\frac{\sqrt{N(s)+1}}{1+N(s,a)}-l_vN_v(s,a)
$$

Melano 根分布保留一份 prior 平滑：

$$
P_{\mathrm{root}}(a\mid s)=\frac{N(s,a)+P(a\mid s)}
{\sum_{b\in\mathcal A(s)}(N(s,b)+P(b\mid s))}
$$

记 $p_a=P_{\mathrm{root}}(a\mid s)$。动态模拟预算使用根分布归一化熵 $H_\pi$、前两名访问差 $U_N$ 和前两名边价值差 $U_Q$：

$$
H_\pi=-\frac{\sum_a p_a\log p_a}{\log|\mathcal A(s)|}
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},\qquad
U_Q=1-\min\left(1,\frac{|Q_{\mathrm{edge},1}-Q_{\mathrm{edge},2}|}{0.5}\right)
$$

$$
u=\operatorname{clip}(0.5H_\pi+0.35U_N+0.15U_Q,0,1)
$$

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil
$$

其中 $N_{\min}$ 与 $N_{\mathrm{cap}}$ 是当前搜索的最小和最大 simulation 数，$u\in[0,1]$ 是根节点不确定性。

```bash
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

CUDA 搜索只把合法动作的 Policy 与 Advantage 从 GPU 传回 CPU。

### 5.6 Arena

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

```bash
build/melano/arena \
	--candidate models/melano/candidate.pth \
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

### 5.7 FCPI

Melano 自对战行为分布同时使用 Policy 与非正 Advantage。记 $\sigma_A(s)$ 为该状态合法动作上的最大 Advantage 绝对值：

$$
\sigma_A(s)=\max_a|A(s,a)|
$$

当 $\sigma_A(s)<10^{-4}$ 时，行为策略等于温度变换后的 Policy。其余情况先构造 Advantage 修正分布：

$$
\widetilde\mu(a\mid s)=\operatorname{softmax}_a\left(
\frac{\log(P(a\mid s)+\varepsilon)+A(s,a)/\sigma_A(s)}{T_b}
\right)
$$

$$
\mu(a\mid s)=(1-\epsilon_u)\widetilde\mu(a\mid s)+
\frac{\epsilon_u}{|\mathcal A(s)|}
$$

其中 $T_b>0$ 是行为温度，$\epsilon_u\in[0,1]$ 是 `uniform-mix`，$\varepsilon>0$ 只用于保证对数数值稳定。完整结束的自对弈使用终局事实生成 MC 回报。设终局时刻为 $\tau$，终局行棋方视角回报为 $z_\tau$：

$$
G_\tau=z_\tau,\qquad G_t=-G_{t+1}
$$

这些回报作为实际状态—动作对的事实 $Q$ 样本。Melano 的 $V(s)$ 表示合法动作价值的上界，而随机行为动作的 $G(s,a)$ 只描述该动作的结果。将两者作为独立目标可防止单个坏动作下拉整个局面 Value。达到 `max-plies` 的截断对局只提供反事实树根。每局先按 Melano state encoding 去重，再使用全部不同局面。

Melano 的反事实树在根节点评价全部合法动作。`--counterfactual-budget` 表示根节点之后可额外评价的动作边数。非根节点的局部宽度、Gumbel 无放回候选和前沿优先级由剩余预算自动产生。Melano 的候选分布 $\nu$ 使用：

$$
\nu(a\mid s)=\operatorname{softmax}_a\left(
\log(P(a\mid s)+\varepsilon)+A(s,a)/\sigma_A(s)
\right)
$$

记 $N_G(s,a)$ 为完整轨迹实际经过状态—动作对 $(s,a)$ 的次数，$G_i(s,a)$ 为第 $i$ 个对应终局回报。事实动作回报均值为：

$$
\widehat G(s,a)=
\frac{1}{N_G(s,a)}
\sum_{i=1}^{N_G(s,a)}G_i(s,a)
$$

对样本中的环境状态 $x$ 及动作 $a$，记 $x_a'=T(x,a)$、$s_a'=\phi(x_a')$。Melano 对每条动作边按以下优先级构造动作值：

1. 动作立即结束对局时，使用精确棋规给出的终局值。
2. 完整自对弈实际走过该 state-action 时，使用 $\widehat G(s,a)$。
3. 该动作已进入反事实子树时，使用递归回传值 $-\overline V(s_a')$。
4. 其余动作使用冻结 Melano 的 dueling 动作值。

最后一种情况为：

$$
Q(s,a)=\operatorname{clip}(V_{old}(s)+A_{old}(s,a),-1,1)
$$

Melano 在自己的 FCPI backend 中根据 Policy、Value 与 Advantage 动作值计算节点内均值：

$$
m(s)=\sum_aP_{old}(a\mid s)Q(s,a)
$$

$$
\pi^+(a\mid s)=
\frac{P_{old}(a\mid s)\exp(Q(s,a)-m(s))}
{\sum_bP_{old}(b\mid s)\exp(Q(s,b)-m(s))}
$$

该更新使用 Melano 动作值的原生尺度。summary 中的 KL、总变差距离、Advantage 幅度和 top-1 改变率均为诊断量。训练损失由下文定义的五项组成。

下标 $T$ 在本小节表示反事实树目标。Melano 的 $A(s,a)\leq0$ 表示动作相对局面能力上界的损失。反事实树使用所有合法动作的动作值上界：

$$
\overline V_T(s)=\max_{a\in\mathcal A(s)}Q_T(s,a)
$$

对应的非正 Advantage target 为：

$$
A_T(s,a)=\operatorname{clip}
\left(Q_T(s,a)-\overline V_T(s),-2,0\right)
$$

因此 $V$、$Q$ 与 $A$ 使用同一棵事实锚定的反事实树，并满足 $Q_T(s,a)\leq\overline V_T(s)$。冻结模型的原 Value 估计可能偏高或偏低，所以终局、实际动作回报和反事实证据都允许修正方向为正或负。

记 $\mathcal E_T(s)$ 为实际展开并通过精确棋规评价的动作集合，其在改进 Policy 下的覆盖质量为：

$$
c_T(s)=\sum_{a\in\mathcal E_T(s)}\pi^+(a\mid s)
$$

每棵树的节点 Policy 权重仍为：

$$
w_P(s)=\frac{n(s)}{\sum_{u\in\mathcal T_{root}}n(u)}
$$

树 Value 权重使用同一预算份额：

$$
w_T(s)=w_P(s)
$$

因此每棵树满足 $\sum_s w_T(s)=1$。覆盖率 $c_T(s)$ 用于诊断。其余动作通过冻结 $V+A$ 进入 $\overline V_T(s)$，有限树预算会自然缩小修正。所有展开节点使用 $\overline V_T(s)$ 训练 Value：

$$
L_V=
\frac{\sum_s w_T(s)\operatorname{SL1}(V_{new}(s)-\overline V_T(s))}
{\sum_s w_T(s)}
$$

每个已展开节点及其精确子状态同时训练 Policy、Value、Advantage 与 latent transition。Dueling action Value 和 imagined Value 直接拟合同一份 $Q_T$。记 $\lambda_P$、$\lambda_V$、$\lambda_Q$、$\lambda_D$ 与 $\lambda_I$ 为五项损失的命令行系数。FCPI 损失为：

$$
L=\lambda_P L_{\mathrm{CE}}(\pi_{\mathrm{new}},\pi^+)+
\lambda_VL_V+
\lambda_Q\operatorname{SL1}(\widehat Q,Q_T)+
\lambda_D L_D+
\lambda_I\operatorname{SL1}(-V(\widehat h'),Q_T)
$$

其中每条树边都通过精确棋规生成 $s'$，并写入 `candidate_next_states`。$L_D$ 使用与监督训练相同的 latent cosine consistency。动作条件 dynamics 对树中每个已展开节点学习一步转移 $E(s)\rightarrow E(s')$，与 $K=2$ anchored latent MCTS 的运行时假设保持一致。

Melano FCPI HDF5 保存 `tree_value_targets`、`tree_value_weights`、`candidate_q` 和逐动作的 `candidate_weights`。完整轨迹终局回报写入对应动作的 `candidate_q`，树目标与 Melano Value 定义保持一致。训练日志输出 `policy`、`value`、`dueling_q`、`dynamics`、`imagined_value` 与总损失。summary 另外记录终局边、事实动作边、反事实覆盖率、Advantage 幅度、Policy 总变差距离与 top-1 改变率。

Melano 每轮依次执行自身 `current.pth` 自对战、完整轨迹事实回传、局面去重、事实锚定反事实展开、Policy/Value/Advantage 与 latent-dynamics 目标构造、candidate 训练和 paired-game arena。每次运行由程序生成 `fcpi_YYYYMMDD_HHMMSS_id`，创建对应的 `data/runs/<run-id>/` 与 `models/runs/<run-id>/`。其中 HDF5 schema、candidate 和 current checkpoint 均属于 Melano，candidate 达到 arena gate 后原子写入该 run 的 `current.pth`。`summary.json` 记录预算、决策节点数、终局/事实边数、评价边数、最大深度和 arena 结果。

云端可用同一组默认参数后台启动：

```bash
bash scripts/melano_fcpi.sh
```

参数通过环境变量覆盖，例如 `GAMES_PER_ITER=2000 BATCH_SIZE=512 bash scripts/melano_fcpi.sh`。脚本会打印 run id、PID、日志路径、`tail` 命令和停止命令。

## 6. UCI

### 6.1 Gadus

```powershell
build\gadus\uci.exe `
	--model models\gadus\gadus.pth `
	--device cpu `
	--search-type only-mcts `
	--mcts-sims 100
```

Gadus UCI 加载 Gadus checkpoint。`SearchType=closed` 使用 Gadus Policy，`SearchType=only-mcts` 使用第 4.5 节定义的 Gadus MCTS。

### 6.2 Melano

```powershell
build\melano\uci.exe `
	--model models\melano\melano.pth `
	--device cuda `
	--search-type only-mcts `
	--mcts-sims 1000
```

Melano UCI 加载 Melano checkpoint。`SearchType=closed` 使用 Melano Policy，`SearchType=only-mcts` 使用第 5.5 节定义的 anchored latent MCTS。

两个程序都输出 MultiPV、side-to-move `score cp`、节点数、NPS、耗时和 PV。搜索开始时先发布模型 Policy 结果，MCTS 期间按 `ProgressIntervalMS` 发布中间结果。

### 6.3 UCI options

Gadus 与 Melano 当前公开相同的 UCI 控制项。选项作用于各自的模型与搜索实现：

- `ModelPath`：checkpoint 路径。打包后的 Gadus 与 Melano 分别读取可执行文件同目录的 `gadus.pth` 与 `melano.pth`。
- `Device`：`auto`、`cpu` 或 `cuda`。
- `SearchType`：`closed` 表示只使用对应架构的 Policy，`only-mcts` 表示使用对应架构的 MCTS。
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

`MCTSSims` 设置引擎在 UCI 客户端没有提供 `go nodes <n>` 时使用的默认 simulation 上限。GUI Search 区域的 `Node / simulation limit` 会为每次搜索发送 `go nodes <n>` 并覆盖该次搜索的 `MCTSSims`。设为 `0` 时不发送 nodes 限制。`Device` 与 `MultiPV` 使用 GUI 对应控件填写，`Launch arguments` 通常留空。

其他 UCI 客户端使用标准命令设置同一组选项：

```text
setoption name SearchType value only-mcts
setoption name MCTSSims value 1000
setoption name MCTSBatchSize value 64
setoption name CPuct value 0.5
setoption name RepetitionPolicyPenalty value 1.0
setoption name InstantMateFirst value true
```

### 6.4 Gadidae 引擎目录

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

Windows 双击 `build/graphics/Gadidae.exe`，具有 X11 或 Wayland 图形会话的 Linux 运行 `build/graphics/Gadidae`。程序默认进入 Simulator，顶部模式控件可切换到 Stadium。GUI 的 FreeType 压缩支持静态编译进可执行文件。SSH 服务器需要 X11 forwarding、远程桌面或其他可见显示服务才能实际操作 GUI。`xvfb-run` 适合自动化启动测试，交互操作则需要可见显示服务。

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

Stadium 用于同时组织多盘独立对局。`Tools > Matches` 可新建、进入和关闭对局，切换到 Simulator 或进入另一盘棋时，其余对局继续在后台运行。默认名称为 `#<id> <white> vs. <black>`，名称留空时自动采用该格式。每个席位都必须填写参赛者名称。Human toggle 打开时 UCI 设置整体置灰，走棋由用户直接在棋盘上完成。Human toggle 关闭时，席位还需要填写 UCI 可执行文件，并可分别设置 UCI options、启动参数、计算预算与 MultiPV。`Match` 区域设置双方共用的初始时间、每步加秒、显示延迟与最大 ply，初始时间设为 0 时关闭棋钟。`Run > Start/Pause/Stop` 只控制当前进入的对局，关闭程序会终止全部对局拥有的 UCI 子进程。可从命令行预填首盘对局的双方引擎：

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

### 10.1 Gadus

Gadus 测试覆盖 `gadus_18_planes`、普通走法与特殊走法编码、棋规、Policy/Value 输出形状、有限数值、反向传播和 checkpoint 往返。本地 Windows CPU 烟测覆盖单盘 PGN 生成 HDF5、一步监督训练、closed 搜索、batched MCTS、两盘 paired arena、PGN 输出以及一轮 FCPI 的采样、反事实展开、训练、arena gate 和 current 晋升。

### 10.2 Melano

Melano 测试覆盖 `melano_square_tokens`、普通走法与升变编码、棋规、Policy/Value/Advantage 输出形状、Advantage 范围、动作条件 latent successor、$K=2$ anchored latent MCTS 路径、有限数值、反向传播和 checkpoint 往返。本地 Windows CPU 烟测覆盖单盘 PGN 生成含 `next_states` 与 `next_values` 的 HDF5、一步监督训练以及一轮树一致 Melano FCPI 的候选后继训练与 arena gate。
