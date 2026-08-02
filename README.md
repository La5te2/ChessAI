# Gadidae

Gadidae 是一个实验性国际象棋神经网络引擎项目。当前架构如下：

- `Gadus`：ResNet + linear policy/value，使用 `gadus_18_planes` state encoding 和 `alphazero_64x73` move encoding。
- `Melano`：residual geometry attention + action-conditioned latent dynamics + source-destination policy/value/advantage，使用 `melano_square_tokens` state encoding 和 `sd_64x64_underpromo9` move encoding。

## 记号与缩写

全文采用行棋方视角。以下数学记号构成全文默认定义，各小节的局部定义会在首次使用前明确给出：

- $\mathcal X$ 是完整棋规环境状态空间。$x\in\mathcal X$ 包含棋盘、行棋方、王车易位权、吃过路兵状态、半回合计数和重复局面历史。
- $\mathcal A(x)$ 是环境状态 $x$ 的合法动作集合，$a\in\mathcal A(x)$ 是一个动作，$T(x,a)$ 是执行动作后的确定性环境状态。
- $z(x)\in\{-1,0,1\}$ 是终局环境状态从当前行棋方视角观察的规则结果。
- $\phi:\mathcal X\rightarrow\mathcal S$ 是架构自身的状态编码，$s=\phi(x)$ 是网络可见状态。状态编码保留的信息决定 $\mathcal X$ 在 $\mathcal S$ 中的等价类，具有相同编码特征的完整环境状态映射到同一个 $s$。
- $P_\theta(a\mid s)$ 是网络参数 $\theta$ 产生的合法动作 Policy，$V_\theta(s)$ 是状态 Value，$Q(s,a)$ 是动作 Value，$A(s,a)$ 是 Advantage。上下文明确冻结某个模型时可省略下标 $\theta$。
- 每套架构分别定义动作索引集合 $\mathcal I$ 与编码函数 $i(a)$。$\ell_\theta(s,i)$ 是动作索引 $i$ 的 Policy logit。logit 是 softmax 归一化前的实数分数。$\theta$ 是网络参数。下标 `old` 与 `new` 分别表示冻结源模型和训练中的 candidate。上标 $+$ 表示由局部策略改进构造的目标。
- 对任意待估计量 $y$，$\widehat y$ 表示有限样本或有限搜索产生的估计，$\overline y$ 表示树回传或加权聚合产生的结果。其他下标和上标均在首次使用前定义。
- $L_{\mathrm{CE}}(p,q)=-\sum_iq_i\log p_i$ 是目标分布 $q$ 与预测分布 $p$ 的交叉熵，$\operatorname{MSE}(u,v)$ 是对应元素平方误差的均值。
- target 是损失函数中的固定比较值或固定比较分布。自动微分把 target 视为常量，并计算损失对训练参数的梯度。

算法与数据缩写如下：

- **PGN**：Portable Game Notation，对局记录格式。
- **FEN**：Forsyth-Edwards Notation，单局面描述格式。
- **UCI**：Universal Chess Interface，图形客户端与国际象棋引擎之间的文本协议。
- **HDF5**：Hierarchical Data Format 5，训练数据容器格式。
- **FCPI**：Folded Counterfactual Policy Iteration，折叠反事实策略迭代。
- **MCTS**：Monte Carlo Tree Search，蒙特卡洛树搜索。
- **PUCT**：Predictor + Upper Confidence bounds applied to Trees，结合网络先验的树上置信上界选择规则。
- **FPU**：First Play Urgency，边在首次实际访问前使用的动作价值。
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
CMakeLists.txt
README.md
THIRD_PARTY.md
api/
	setup.bat
	setup.sh
	patch.cmake
	verify.cmake
	versions.env
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
		uci.cpp
	melano/
		args.cpp
		arena.cpp
		checkpoint.cpp
		dataset.cpp
		game.cpp
		match.cpp
		model.cpp
		preprocess.cpp
		search.cpp
		searcher.cpp
		tests.cpp
		train.cpp
		uci.cpp
	graphics/
		application.cpp
		archive.cpp
		game.cpp
		main.cpp
		piece.inc
		pieces.cpp
		pieces.gpack
		pieces.rc.in
		pieces.S.in
		registry.cpp
		simulator.cpp
		stadium.cpp
		uci.cpp
scripts/
	analyze.py
	opening_book.py
	check.py
	build.bat
	build.sh
	gadus_fcpi.sh
	import_pieces.py
	package_engine.bat
	package_engine.sh
	requirements.txt
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

- `include/gadus/` 与 `include/melano/`：两套架构各自的公开 C++ 接口。头文件与对应 `src/<architecture>/` 模块一一对应。
- `args.cpp`：解析该架构各命令入口的参数。
- `checkpoint.cpp`：校验架构元数据，并负责 LibTorch checkpoint 的加载与原子写入。
- `dataset.cpp`：实现该架构的 PGN 预处理、HDF5 读写和监督训练循环。
- `game.cpp`：实现该架构的状态编码、动作编码、合法走法转换、棋规终局判断和开局状态读取。
- `model.cpp`：实现该架构的网络结构与前向计算。
- `searcher.cpp`：实现可复用的 `closed` 与 MCTS 搜索器。`search.cpp` 提供单局面命令入口。
- `match.cpp`：实现 batched paired games 与 arena 统计。`arena.cpp` 提供竞技场命令入口。
- `preprocess.cpp`、`train.cpp` 与 `uci.cpp`：分别提供预处理、监督训练和 UCI 命令入口。
- `src/gadus/evolution.cpp`：实现 Gadus FCPI 的采样、反事实树、训练和晋升循环。`src/gadus/fcpi.cpp` 提供 FCPI 命令入口。
- `tests.cpp`：验证对应架构的状态与动作编码、棋规、网络前反向、数值范围、搜索和 checkpoint 往返。
- `include/graphics/` 与 `src/graphics/`：实现原生图形程序。`application` 管理窗口、菜单、配置和模式切换，`simulator` 管理单局面分析，`stadium` 管理并发对局，`registry` 管理引擎配置，`uci` 管理引擎子进程与协议，`game` 管理可视化棋局和 PGN，`pieces` 与 `archive` 管理内嵌棋子几何，`main` 提供程序入口。`piece.inc` 定义程序自带的 Vector 样式，`pieces.gpack` 保存导入样式的压缩几何，`pieces.rc.in` 与 `pieces.S.in` 分别把该几何包嵌入 Windows 和 Linux 可执行文件。
- `scripts/`：提供构建、模型检查、UCI 对局分析、开局书生成、棋子导入和引擎打包脚本。
- `api/`：仓库本地 C++ 依赖与安装脚本。
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

安装到 `api/` 的 C++ 依赖包括 LibTorch、HDF5、zlib、nlohmann-json、chess-library 与 Ninja。包含图形目标的构建还需要 GLFW、GLM、Dear ImGui、FreeType 与 GLAD，安装脚本会在项目构建前准备这些依赖。`api/versions.env` 是 Windows 与 Linux 共用的依赖版本锁。安装脚本在 `nvidia-smi` 能查询 GPU compute capability 时选择 LibTorch CUDA `cu126`，其余环境选择 CPU 包。环境变量 `GADIDAE_TORCH_VARIANT=cpu|cu126` 可以显式选择 LibTorch variant，`GADIDAE_TORCH_DIR` 可以指向已有 LibTorch 的根目录。

`setup.bat` 与 `setup.sh` 在安装结束时验证实际依赖。`build.bat` 与 `build.sh` 在 CMake 配置前再次验证。CMake 对 LibTorch、HDF5 与 zlib 使用精确版本约束。版本、LibTorch variant 和 chess-library 校验和全部符合 `api/versions.env` 时，验证程序接受该依赖组合。任一项目出现差异时，验证程序终止安装或构建并报告首个差异。下载过程显示进度、速度和 ETA。安装成功后，脚本清理压缩包、源码和依赖构建目录。

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

`setup.sh` 与 `build.sh` 默认根据 `DISPLAY` 或 `WAYLAND_DISPLAY` 判断当前 Linux 的构建类型。具有图形会话的 Linux 和 Windows `auto` 模式构建 Graphics。纯命令行 Linux 构建两套架构的监督训练、搜索、竞技场与 UCI 程序，以及 Gadus FCPI。`GADIDAE_BUILD_GRAPHICS=0` 明确选择命令行构建，`GADIDAE_BUILD_GRAPHICS=1` 明确选择包含 GUI 的构建。

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
build/melano/uci
```

Windows 程序带 `.exe` 后缀。`build/gadus/` 与 `build/melano/` 保存发布程序。

`build/.build-work/` 保存 CMake 与 Ninja 的增量构建状态。后续构建复用仍与当前源码和依赖一致的对象文件。编译或 CTest 失败时，诊断信息保留在该目录，CTest 日志位于 `build/.build-work/Testing/Temporary/LastTest.log`。

每个命令入口支持 `--help`：

```bash
build/gadus/search --help
build/melano/train --help
```

### 3.1 模型检查

`scripts/check.py` 检查现行 LibTorch checkpoint 的有效性（以只读方式加载模型），输出架构、网络头、channels、blocks、动作空间、参数规模、张量类型、内存规模、有限性、文件大小与 SHA-256：

```bash
python scripts/check.py --model models/gadus/gadus.pth
```

## 4. Gadus

### 4.1 状态与动作编码

Gadus 的状态编码为 `gadus_18_planes`：12 个棋子平面、1 个行棋方平面、4 个王车易位平面和 1 个 en passant 文件平面。每个平面按 8 个 rank bytes packbits，HDF5 中单个状态占 $18\times8$ bytes。

动作编码为 `alphazero_64x73`。记 $\mathcal I_G$ 为 Gadus 的动作索引集合，$i_G(a)\in\mathcal I_G$ 为合法动作 $a$ 的编码。每个起点对应 56 个八方向滑动动作、8 个马步动作和 9 个 underpromotion 动作：

$$
|\mathcal I_G|=64\times(56+8+9)=4672
$$

### 4.2 网络

模型由 ResNet trunk、linear Policy head 和 MLP Value head 组成。令 $C$ 为 `--channels` 所对应参数，$B$ 为 `--blocks` 所对应参数，trunk 的首层使用无偏置 $3\times3$ 卷积把 18 个输入平面映射为 $C$ 个通道，随后执行 BatchNorm 和 ReLU。记该输出为 $h_0$，第 $j$ 个残差块定义为：

$$
F_j(h)=\operatorname{BN}_{j,2}\left(
\operatorname{Conv}^{C\rightarrow C}_{3\times3,j,2}\left(
\operatorname{ReLU}\left(
\operatorname{BN}_{j,1}\left(
\operatorname{Conv}^{C\rightarrow C}_{3\times3,j,1}(h)
\right)\right)\right)\right)
$$

$$
h_{j+1}=\operatorname{ReLU}(h_j+F_j(h_j)),\qquad 0\leq j<B
$$

所有 $3\times3$ 卷积使用一格 padding，并保持 $8\times8$ 空间尺寸。Policy head 对 $h_B$ 依次执行无偏置 $1\times1$ 卷积 $C\rightarrow32$、BatchNorm、ReLU、展平和线性映射 $2048\rightarrow4672$。Value head 对 $h_B$ 依次执行无偏置 $1\times1$ 卷积 $C\rightarrow32$、BatchNorm、ReLU、展平、线性映射 $2048\rightarrow256$、ReLU、线性映射 $256\rightarrow1$ 和 $\tanh$ 变换。

对网络状态 $s$，完整前向计算为：

$$
(\ell_\theta(s),V_\theta(s))=f_{\theta}(s),\qquad
\ell_\theta(s)\in\mathbb R^{4672},\quad V_\theta(s)\in[-1,1]
$$

对于满足 $s=\phi(x)$ 的环境状态 $x$，搜索过程提取合法动作集合 $\mathcal A(x)$ 中各动作对应的 logit，并对这些 logits 应用 softmax，得到合法动作 Policy：

$$
P_\theta(a\mid s)=
\frac{\exp \ell_\theta(s,i_G(a))}
{\sum_{b\in\mathcal A(x)}\exp \ell_\theta(s,i_G(b))}
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

`--has-cmt` 的默认值为 `1`。`--has-cmt 1` 读取 `{+x}`、`{-x}` 形式的白方视角 centipawn score。PGN 中一手棋后的评注描述该手棋产生的局面，因此状态 $s_t$ 使用前一手棋后的评注。记 $q_t$ 为该评注中的白方视角评分，令 $\rho(x_t)=1$ 表示白方行棋，$\rho(x_t)=-1$ 表示黑方行棋。行棋方视角评分与 Value target 为：
$$
q_{\mathrm{stm}}(s_t)=\rho(x_t)q_t,\qquad
V_{\mathrm{target}}(s_t)=\tanh\left(\frac{q_{\mathrm{stm}}(s_t)}{3}\right)
$$

缺少可解析评分的局面使用 $q_{\mathrm{stm}}=0$。预处理器将至少含有一个可解析评分的对局写入 `--has-cmt 1` 数据集。`--has-cmt 0` 根据最终胜负生成 $V_{\mathrm{target}}\in\{-1,0,1\}$。

```bash
build/gadus/preprocess \
	--input data/ccrl.pgn \
	--output data/games.gadus.h5 \
	--has-cmt 1 \
	--chunk-size 4096 \
	--compression-level 1 \
	--log-every 10000 \
	--max-games 1000000
```

`--max-games` 控制读取对局上限，`--chunk-size` 控制 HDF5 扩展单元，`--compression-level` 控制 deflate 等级，`--log-every` 控制对局进度输出。

### 4.4 监督训练

记 PGN 中的实际走法为 $a^*$，其动作索引为 $i^*=i_G(a^*)$。监督训练在完整动作索引集合 $\mathcal I_G$ 上定义：

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\sum_{j\in\mathcal I_G}\exp\ell_\theta(s,j)}
$$

$R_\theta(\cdot\mid s)$ 是模型在完整动作索引集合 $\mathcal I_G$ 上产生的监督训练概率分布，它的 softmax 分母包含全部 4672 个动作索引。$R_\theta(i^*\mid s)$ 表示模型在状态 $s$ 下分配给动作索引 $i^*$ 的概率。

$\delta_{i^*}$ 是索引 $i^*$ 的 one-hot 分布，$w_V$ 是 `--value-weight`。监督损失为：

$$
L_{\mathrm{sup}}=
L_{\mathrm{CE}}\left(R_\theta(\cdot\mid s),\delta_{i^*}\right)+
w_V\operatorname{MSE}\left(V_\theta(s),V_{\mathrm{target}}(s)\right)
$$

```bash
build/gadus/train \
	--data data/games.gadus.h5 \
	--out models/gadus/gadus.pth \
	--channels 128 \
	--blocks 20 \
	--epochs 10 \
	--batch-size 512 \
	--max-steps 500000 \
	--lr 0.001 \
	--weight-decay 0.0001 \
	--value-weight 0.25 \
	--save-every 5000 \
	--device cuda \
	--precision bf16 \
	--log-every 50 \
	--seed 2026
```

`train` 每次创建新的 Gadus 模型。`--channels` 和 `--blocks` 决定网络结构，`--max-steps` 是本次训练步数上限，`--save-every` 控制原子 checkpoint 写入周期。

`--precision` 可取 `fp32` 或 `bf16`，默认 `fp32`。CUDA 前向计算在 `bf16` 模式下使用 BF16。Policy softmax、训练损失、指标累加和 checkpoint 参数使用 FP32。CUDA 训练批次使用 pinned memory。

### 4.5 搜索

`closed` 以合法动作 Policy 作为基础排序，再应用 IMF 与 RPP 决策组件。`only-mcts` 使用 batched leaf inference。对于每条边 $(s,a)$，$P(s,a)$ 表示模型在节点展开时为动作 $a$ 计算的 Policy 概率。该概率称为边 $(s,a)$ 的 Policy 先验概率，并在节点存续期间保持不变。记父节点行棋方视角下边 $(s,a)$ 的实际访问数为 $N(s,a)$，节点访问数为 $N(s)=\sum_aN(s,a)$，实际访问产生的平均叶节点回传为 $Q(s,a)$。$N_v(s,a)$ 是当前神经网络评价批次为边预留的 virtual visits，$N_v(s)=\sum_aN_v(s,a)$。选择阶段使用的访问数为：

$$
\widetilde N(s,a)=N(s,a)+N_v(s,a),\qquad
\widetilde N(s)=N(s)+N_v(s)
$$

记 $c_0$、$b_0$ 与 $f_0$ 分别为 PUCT 的初始系数、输入基数和输入增长系数，并定义 $b=\max(1,b_0)$ 与 $f=\max(0,f_0)$。动态探索系数为：

$$
c_{\mathrm{puct}}(\widetilde N)=
\max\left(0,c_0+f\log\left(\frac{\widetilde N+b+1}{b}\right)\right)
$$

PUCT 选择分数为：

$$
S(s,a)=Q_{\mathrm{sel}}(s,a)
+c_{\mathrm{puct}}(\widetilde N(s))P(s,a)
\frac{\sqrt{\widetilde N(s)+1}}{1+\widetilde N(s,a)}
-l_vN_v(s,a)
$$

其中 $l_v=\max(0,\texttt{--virtual-loss})$ 是每个 virtual visit 的 virtual loss。

当选择分数相同时，搜索器依次按 $P(s,a)$ 和选择值 $Q_{\mathrm{sel}}(s,a)$ 降序选择边。当 $N(s,a)>0$ 时，$Q_{\mathrm{sel}}(s,a)=Q(s,a)$。当 $N(s,a)=0$ 时，选择值由 FPU 公式确定：
$$
Q_{\mathrm{sel}}(s,a)=Q_{\mathrm{FPU}}(s)=\operatorname{clip}\left(
Q(s)-r_{\mathrm{FPU}}\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right)
$$

其中 $Q(s)$ 为节点已有访问的平均回传，$r_{\mathrm{FPU}}\geq0$ 为 FPU 折减系数。

每次 simulation 完成叶节点评价后，搜索器将叶节点 Value 沿所选路径逐 ply 取反并回传，同时更新路径上各边的 $N(s,a)$ 和 $Q(s,a)$。全部 simulation 结束后，搜索器使用 $N(s,a)+P(s,a)$ 作为根动作 $a$ 的分布权重，并将所有合法根动作的分布权重归一化为根 Policy：
$$
P_{\mathrm{root}}(a\mid s)=\frac{N(s,a)+P(s,a)}
{\sum_{b\in\mathcal A(x)}(N(s,b)+P(s,b))}
$$

记 $N_{\mathrm{cap}}=\max(0,\texttt{--mcts-sims})$，$B=\max(1,\texttt{--mcts-batch-size})$，$M=\texttt{--mcts-min-sims}$。搜索器按下式计算 simulation 下限：

$$
N_{\min}=
\begin{cases}
0,&N_{\mathrm{cap}}=0,\\
\max\!\left(1,\min\!\left(N_{\mathrm{cap}},M\right)\right),&N_{\mathrm{cap}}>0\ \text{且}\ M>0,\\
\max\!\left(1,\min\!\left(N_{\mathrm{cap}},\max\!\left(B,\left\lfloor\dfrac{N_{\mathrm{cap}}}{4}\right\rfloor\right)\right)\right),&N_{\mathrm{cap}}>0\ \text{且}\ M=0.
\end{cases}
$$

`movetime` 截止时间和 `stop` 请求具有更高终止优先级，并可在实际 simulation 数到达 $N_{\min}$ 前结束搜索。动态模拟预算在完成 $N_{\min}$ 次 simulation 后使用实际访问分布：

$$
v_a=\frac{N(s,a)}{\sum_{b\in\mathcal A(x)}N(s,b)}
$$

记 $a_1$ 与 $a_2$ 为按实际访问数排序的前两项，$N_i=N(s,a_i)$，$Q_i=Q(s,a_i)$。访问分布归一化熵 $H_N$、前两项访问接近度 $U_N$ 和前两项价值接近度 $U_Q$ 为：

$$
H_N=-\frac{\sum_{a\in\mathcal A(x)}v_a\log v_a}
{\log|\mathcal A(x)|}
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},\qquad
U_Q=1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right)
$$

$$
u=\operatorname{clip}(0.5H_N+0.35U_N+0.15U_Q,0,1)
$$

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil
$$

其中 $u\in[0,1]$ 是根节点不确定性。

IMF 与 RPP 位于最终决策层。记 $D_0(a)=P(a\mid s)$ 表示 `closed` 的基础排序分数，记 $D_0(a)=P_{\mathrm{root}}(a\mid s)$ 表示 `only-mcts` 的基础排序分数。令 $\mathcal M(x)$ 为执行后立即形成 checkmate 的合法动作集合。IMF 在 $\mathcal M(x)$ 中选择基础分数最高的动作 $a_M$，并令：

$$
D_I(a)=
\begin{cases}
1,&a=a_M\\
D_0(a),&a\in\mathcal A(x)\setminus\{a_M\}
\end{cases}
$$

$\mathcal M(x)=\varnothing$ 时定义 $D_I(a)=D_0(a)$。

令 $\lambda_R\in[0,1]$ 为 `--repetition-policy-penalty`，$V_R$ 为搜索返回的根 Value。集合 $\mathcal R_3(x)$ 包含两类动作：执行该动作后立即满足三次重复申和条件的动作，以及执行该动作后对手存在一步应手可满足该条件的动作。RPP 的扣分量与最终决策分数为：

$$
d_R=\lambda_R\operatorname{clip}(V_R,0,1)
$$

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),&a\in\mathcal R_3(x)\\
D_I(a),&a\in\mathcal A(x)\setminus\mathcal R_3(x)
\end{cases}
$$

最终走法先按 $D(a)$ 降序排列，再按基础 Policy 降序排列，最后按 UCI 走法字符串降序排列。

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

`--fen startpos` 使用标准初始局面，也可传入完整 FEN。CUDA 搜索向 CPU 返回根 Value，并从动作输出中选取合法动作的 Policy。

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

记 $score_b=\operatorname{clip}(score,10^{-6},1-10^{-6})$。显示用 Elo 差为：

$$
\Delta Elo=400\log_{10}\left(\frac{score_b}{1-score_b}\right)
$$

记 $M=\texttt{--min-net-wins}$。gate 条件为：

$$
W-L\geq M
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

#### 4.7.1 模型与训练目标

用 $r$ 表示 FCPI 轮次，用 $C_r$ 表示该轮开始时的 `current.pth`。程序把 $C_r$ 同时作为冻结目标生成器和 candidate 的初始参数。记其参数为 $\theta_{old}$，输出为 $P_{old}(a\mid s)$ 与 $V_{old}(s)$。训练中的 candidate 参数记为 $\theta_{new}$，输出为 $P_{new}(a\mid s)$ 与 $V_{new}(s)$。

用 $t$ 表示一盘自对弈中的 ply，记执行动作前的完整环境状态为 $x_t$，网络输入为 $s_t=\phi(x_t)$，实际动作是 $a_t$。完整对局的最终胜、和、负结果换算到 $x_t$ 的行棋方视角后，得到 MC 回报 $G_t\in\{-1,0,1\}$。第 4.7.3 节给出 $G_t$ 的逐 ply 递推。

candidate 对状态 $s$ 输出合法动作 Policy $P_{new}(\cdot\mid s)$ 和状态 Value $V_{new}(s)$。FCPI 同时训练这两个输出及其共享的网络参数。完整对局结果确定 MC Value target $G_t$，$G_t$ 与冻结 Value $V_{old}(s_t)$ 共同确定轨迹所选动作的 MC Advantage $A_{\mathrm{MC}}(s_t,a_t)$。有限反事实树产生树 Policy target $\pi^+(\cdot\mid s)$ 和树 Value target $\overline V_{\mathrm{CF}}(s)$。第 4.7.3 节定义两类 MC 监督量，第 4.7.5 节定义两类反事实树监督量，第 4.7.7 节定义这些监督量对应的损失函数及总损失。

一轮按以下顺序执行：唯一起始局面采样、`closed` 自对弈、终局 MC 回传、逐局去重、反事实树构造、全局编码状态聚合、candidate 训练、paired-game arena 和原子晋升。

#### 4.7.2 起始局面与行为策略

使用采样书时，程序从 `--opening-book` 的可达非终局状态中无放回选择 `--games-per-iter` 个不同 FEN。候选池最多读取 `--max-book-positions` 个状态。采样书允许任意 ply，标准初始局面可以作为普通可达状态出现。采样书为空时，每盘都从标准初始局面开始。第 $r$ 轮的起始局面读取和洗牌使用 `--seed + r`。

自对弈执行冻结 $C_r$ 的 `closed` Policy。对状态 $s$ 的合法动作集合 $\mathcal A(x)$，程序把网络 logits 在合法动作上归一化为 $P_{old}(a\mid s)$。定义实际使用的温度为：

$$
\widetilde T_b=\max(T_b,10^{-4})
$$

行为分布为：

$$
\mu_{old}(a\mid s)=
\frac{P_{old}(a\mid s)^{1/\widetilde T_b}}
{\sum_{b\in\mathcal A(x)}P_{old}(b\mid s)^{1/\widetilde T_b}}
$$

同一 FCPI 轮次内，程序按照 Gadus `PackedState` 对编码状态的访问进行分组，并为每个编码状态 $s$ 记录各合法动作的累计选择次数。当编码状态 $s$ 即将发生第 $n$ 次访问时，$N_{n-1}(s,a)$ 表示前 $n-1$ 次访问中动作 $a$ 的实际选择次数。$n\mu_{old}(a\mid s)$ 表示完成第 $n$ 次访问后动作 $a$ 应达到的累计选择次数。二者之差是动作 $a$ 在本次访问时的概率欠额，程序选择概率欠额最大的合法动作：

$$
a_n=\arg\max_{a\in\mathcal A(x)}
\left[n\mu_{old}(a\mid s)-N_{n-1}(s,a)\right]
$$

相同欠额按合法动作数组中的先后顺序打破平局。程序在走子前保存 $x_t$ 的 FEN、$s_t$、$V_{old}(s_t)$、合法动作索引、$P_{old}(\cdot\mid s_t)$ 和 $a_t$。规则终局结束完整轨迹。保存的走前状态数达到 `--max-plies` 时，程序结束截断轨迹。

每盘轨迹随后按 `PackedState` 去重。每个编码状态在本局中的第一次出现成为一棵反事实树的根。

#### 4.7.3 终局 MC 目标

设一盘完整对局保存了走前状态 $x_0,\ldots,x_{L-1}$，执行 $a_{L-1}$ 后到达规则终局 $x_L$。$z(x_L)$ 是终局状态从其当前行棋方视角观察的结果。各走前状态的 MC 回报定义为：

$$
G_{L-1}=-z(x_L),\qquad
G_t=-G_{t+1}\quad(0\leq t<L-1)
$$

因此 $G_t\in\{-1,0,1\}$ 始终采用 $x_t$ 的行棋方视角。完整终局轨迹中每个保留状态的 MC Value 权重为：

$$
w_{\mathrm{MC}}(s_t)=1
$$

截断轨迹中各状态的 MC Value 权重设置为 $w_{\mathrm{MC}}=0$。截断轨迹中的保留状态继续作为反事实树根。

完整终局轨迹中，轨迹所选动作 $a_t$ 的 MC Advantage 为：

$$
A_{\mathrm{MC}}(s_t,a_t)=
\operatorname{clip}_{[-1,1]}
\left(\frac{G_t-V_{old}(s_t)}{2}\right)
$$

MC Advantage 存储向量在动作 $a_t$ 的坐标写入 $A_{\mathrm{MC}}(s_t,a_t)$，其余动作坐标的权重写为 0。第 4.7.7 节定义 $G_t$ 与 $A_{\mathrm{MC}}$ 所对应损失项的具体计算。

#### 4.7.4 有限反事实树

对于第 4.7.2 节从实际自对弈轨迹中去重后保留的每个走前环境状态 $x_t$，程序分别以 $x_t$ 为根节点建立一棵独立的反事实树。`--counterfactual-budget` 为每棵树规定深层决策节点最多可以评价的动作边数，记为 $B$。评价一条动作边是指从一个深层决策节点执行一个合法动作，再以规则终局结果或冻结模型 Value 为后继状态赋值。各棵树分别获得 $B$ 条深层动作边预算。树的实际深度和节点数由动作边预算、局部分支宽度与终局位置共同决定。

根节点使用独立的全动作评价过程，使根节点的目标覆盖完整合法动作集合。$B$ 专门计量根节点以下的深层动作边。当 $B=0$ 时，每棵树只评价根节点的一步合法动作。增大 $B$ 后，程序继续评价前沿中深层决策节点的动作边。

若非根决策节点 $x$ 尚有 $B_{rem}>0$ 条深层边预算，它的展开宽度为：

$$
w(x)=\min\left(
|\mathcal A(x)|,
B_{rem},
\max\left(2,\left\lceil\sqrt{B_{rem}}\right\rceil\right)
\right)
$$

非根节点纳入 $P_{old}$ 的 top-1 动作，其余 $w(x)-1$ 个动作通过 Gumbel top-k 无放回选择。对其余动作定义：

$$
k(a)=\log\left(\operatorname{clip}(P_{old}(a\mid s),10^{-12},1)\right)+g_a,
\qquad g_a=-\log(-\log u_a),\quad u_a\sim U(0,1)
$$

程序按 $k(a)$ 降序补齐候选。反事实 Gumbel 随机数生成器使用 `--seed + 3000017`。

树节点保存精确环境状态。记 $\mathcal X_T$ 为规则终局状态集合，$\mathcal X_O$ 为具有合法后继的进行中状态集合。冻结 $C_r$ 对 $\mathcal X_O$ 中的状态执行 `closed` 推理。叶点评价定义为：

$$
v_{leaf}(x)=
\begin{cases}
z(x), & x\in\mathcal X_T\\
V_{old}(\phi(x)), & x\in\mathcal X_O
\end{cases}
$$

根的到达概率为 1。若边 $(x,a)$ 到达 $x'=T(x,a)$，则：

$$
p_{reach}(x')=p_{reach}(x)P_{old}(a\mid\phi(x))
$$

新子节点的前沿优先级为：

$$
priority(x')=p_{reach}(x')
\left(
\left|-v_{leaf}(x')-V_{old}(\phi(x))\right|
+\frac{1}{\sqrt{2+d(x')}}
\right)
$$

$d(x')$ 是子节点相对树根的 ply 深度。前沿由尚待展开的进行中节点组成。程序反复展开最高优先级节点，直至深层边预算耗尽或前沿为空。

记 $E(x)\subseteq\mathcal A(x)$ 为节点 $x$ 中已经显式评价的动作集合。树构造结束后按深度逆序回传。若 $s=\phi(x)$、$x_a'=T(x,a)$，反事实动作价值为：

$$
Q_{\mathrm{CF}}(s,a)=
\begin{cases}
-z(x_a'), & a\in E(x),\ x_a'\in\mathcal X_T\\
-\overline V_{\mathrm{CF}}(\phi(x_a')), & a\in E(x),\ x_a'\in\mathcal X_O\\
V_{old}(s), & a\in\mathcal A(x)\setminus E(x)
\end{cases}
$$

公式中的三个分支分别处理已评价终局边、已评价进行中边和未评价边。对于已评价边，程序先依据精确棋规执行动作并得到子节点。终局子节点使用规则终局结果 $z$，进行中子节点使用其回传值 $\overline V_{\mathrm{CF}}$。若进行中子节点 $x'$ 在树构造结束时仍位于前沿，则该节点没有更深的已评价后继，因此取 $\overline V_{\mathrm{CF}}(\phi(x'))=V_{old}(\phi(x'))$。对于 $\mathcal A(x)\setminus E(x)$ 中的未评价动作，程序使用父节点的冻结 Value $V_{old}(s)$ 作为其反事实动作价值。

#### 4.7.5 反事实 Policy target 与 Value target

对一个已展开决策节点，代码先计算冻结 Policy 下的中心量：

$$
m(s)=\sum_{a\in\mathcal A(x)}P_{old}(a\mid s)Q_{\mathrm{CF}}(s,a)
$$

为了给每个合法动作保留正的基准权重，定义截断 prior 权重 $\widetilde p(a\mid s)$：

$$
\widetilde p(a\mid s)=
\operatorname{clip}(P_{old}(a\mid s),10^{-12},1)
$$

$\widetilde p(a\mid s)$ 是将冻结 Policy $P_{old}(a\mid s)$ 限制在 $[10^{-12},1]$ 后得到的未归一化权重。树 Policy target 按下式构造：

$$
\pi^+(a\mid s)=
\frac{\widetilde p(a\mid s)
\exp\left(Q_{\mathrm{CF}}(s,a)-m(s)\right)}
{\sum_{b\in\mathcal A(x)}\widetilde p(b\mid s)
\exp\left(Q_{\mathrm{CF}}(s,b)-m(s)\right)}
$$

因子 $\exp[-m(s)]$ 同时出现在分子和分母中，并在归一化时约去。将截断 prior 权重归一化后得到分布 $p_\varepsilon$：

$$
p_\varepsilon(a\mid s)=
\frac{\widetilde p(a\mid s)}
{\sum_{b\in\mathcal A(x)}\widetilde p(b\mid s)}
$$

以 $p_\varepsilon$ 为基准分布时，$\pi^+$ 等价于单位 KL 系数下的局部优化解：

$$
\pi^+=\arg\max_\pi
\left[
\sum_a\pi(a\mid s)Q_{\mathrm{CF}}(s,a)
-D_{\mathrm{KL}}\left(\pi(\cdot\mid s)\,\|\,p_\varepsilon(\cdot\mid s)\right)
\right]
$$

树 Value target 是目标 Policy 下的反事实动作价值期望：

$$
\overline V_{\mathrm{CF}}(s)=
\operatorname{clip}_{[-1,1]}
\left(
\sum_{a\in\mathcal A(x)}
\pi^+(a\mid s)Q_{\mathrm{CF}}(s,a)
\right)
$$

定义反事实 Value 修正量：

$$
\delta_{\mathrm{CF}}(s)=
\overline V_{\mathrm{CF}}(s)-V_{old}(s)
$$

集合 $\mathcal A(x)\setminus E(x)$ 中的动作满足 $Q_{\mathrm{CF}}(s,a)=V_{old}(s)$，所以：

$$
\delta_{\mathrm{CF}}(s)=
\sum_{a\in E(x)}
\pi^+(a\mid s)
\left(Q_{\mathrm{CF}}(s,a)-V_{old}(s)\right)
$$

这条等式给出反事实树的 Value target：$\overline V_{\mathrm{CF}}=V_{old}+\delta_{\mathrm{CF}}$。修正量 $\delta_{\mathrm{CF}}$ 由 $P_{old}$、$Q_{\mathrm{CF}}$ 和 $V_{old}$ 确定。

若一棵树共评价 $N_E$ 条动作边，已展开节点 $x$ 自身评价了 $n(x)=|E(x)|$ 条边，则该节点的树 Policy 权重和树 Value 权重相同：

$$
w_P(x)=w_T(x)=\frac{n(x)}{N_E},
\qquad
N_E=\sum_{u\in\mathcal T_{root}}n(u)
$$

$\mathcal T_{root}$ 是这棵树产生的全部已展开决策节点，包括根。根的全部合法边计入 $N_E$，深层边也计入 $N_E$。因此每棵树满足：

$$
\sum_{x\in\mathcal T_{root}}w_P(x)
=\sum_{x\in\mathcal T_{root}}w_T(x)=1
$$

#### 4.7.6 编码状态聚合

树中每个已展开决策节点形成一条训练记录。树根记录携带对应自对弈走前状态的 MC Value 数据和 MC Policy 数据，深层反事实节点记录的 MC 权重为 0。程序在整轮数据中按 Gadus `PackedState` 合并具有相同网络输入的记录。程序还会验证相同编码状态的各条记录具有相同的合法动作列表。

记 $\mathcal R(s)$ 为聚合到编码状态 $s$ 的记录集合，$\mathcal A_s$ 为这些记录共同的合法动作集合。对于记录 $i\in\mathcal R(s)$，$w_{P,i}$、$w_{\mathrm{MC},i}$ 与 $w_{T,i}$ 分别表示树 Policy 权重、MC Value 权重和树 Value 权重。同一条记录中的 $\pi_i^+(\cdot\mid s)$、$G_i$ 与 $\overline V_{\mathrm{CF},i}(s)$ 分别表示树 Policy target、MC Value target 和树 Value target。

树 Policy 聚合先求出编码状态 $s$ 的总权重 $W_P(s)$，再按每条记录的树 Policy 权重对 $\pi_i^+$ 求加权平均：

$$
W_P(s)=\sum_{i\in\mathcal R(s)}w_{P,i},
\qquad
\Pi^+(a\mid s)=
\frac{\sum_{i\in\mathcal R(s)}w_{P,i}\pi_i^+(a\mid s)}{W_P(s)}
$$

$\Pi^+(\cdot\mid s)$ 是编码状态 $s$ 的聚合树 Policy target。程序在计算加权平均后对 $\Pi^+(\cdot\mid s)$ 再次归一化，以消除浮点计算产生的概率和偏差。

MC Value 聚合先求出编码状态 $s$ 的总权重 $W_{\mathrm{MC}}(s)$，再按每条记录的 MC Value 权重对 $G_i$ 求加权平均：

$$
W_{\mathrm{MC}}(s)=\sum_{i\in\mathcal R(s)}w_{\mathrm{MC},i},
\qquad
\overline G(s)=
\frac{\sum_{i\in\mathcal R(s)}w_{\mathrm{MC},i}G_i}{W_{\mathrm{MC}}(s)}
$$

$\overline G(s)$ 是编码状态 $s$ 的聚合 MC Value target。

树 Value 聚合先求出编码状态 $s$ 的总权重 $W_T(s)$，再按每条记录的树 Value 权重对 $\overline V_{\mathrm{CF},i}(s)$ 求加权平均：

$$
W_T(s)=\sum_{i\in\mathcal R(s)}w_{T,i},
\qquad
\overline V_T(s)=
\frac{\sum_{i\in\mathcal R(s)}w_{T,i}\overline V_{\mathrm{CF},i}(s)}{W_T(s)}
$$

$\overline V_T(s)$ 是编码状态 $s$ 的聚合树 Value target。

对任意 $s$，三个总权重都满足 $W_P(s)\geq 0$、$W_{\mathrm{MC}}(s)\geq 0$ 与 $W_T(s)\geq 0$。聚合树 Policy target $\Pi^+(\cdot\mid s)$ 在 $W_P(s)>0$ 的状态上计算，聚合 MC Value target $\overline G(s)$ 在 $W_{\mathrm{MC}}(s)>0$ 的状态上计算，聚合树 Value target $\overline V_T(s)$ 在 $W_T(s)>0$ 的状态上计算。某个总权重等于 0 时，程序不计算对应的商式，并将状态 $s$ 对应损失项的加权贡献设为 0。

前三组聚合分别产生一个 Policy 分布 target 和两个标量 Value target。MC Policy 损失需要区分同一编码状态上的不同实际动作，因此程序按编码状态 $s$ 和合法动作 $a$ 聚合 MC Policy 数据。指示函数 $\mathbf 1[a_i=a]$ 从记录 $i$ 中选出实际动作等于 $a$ 的样本。程序对这些样本的 MC Advantage 加权值和样本权重分别求和：

$$
S_A(s,a)=\sum_{i\in\mathcal R(s)}
w_{\mathrm{MC},i}\mathbf 1[a_i=a]A_{\mathrm{MC},i}
$$

$$
W_A(s,a)=\sum_{i\in\mathcal R(s)}
w_{\mathrm{MC},i}\mathbf 1[a_i=a]
$$

$S_A(s,a)$ 是编码状态 $s$ 上实际选择动作 $a$ 的 MC Advantage 加权和，$W_A(s,a)$ 是这些样本的总权重。$S_A(s,a)$ 和 $W_A(s,a)$ 分别对应 HDF5 中的 `mc_policy_advantage_sums` 和 `mc_policy_weights`。第 4.7.7 节使用这两个聚合量计算 MC Policy 损失。

#### 4.7.7 训练损失

反事实 Policy 损失中的预测分布 $P_{new}$ 由合法动作 logits 的标准 softmax 定义：

$$
P_{new}(a\mid s)=
\frac{\exp\ell_{new}(s,i_G(a))}
{\sum_{b\in\mathcal A_s}\exp\ell_{new}(s,i_G(b))}
$$

MC Policy 使用与自对弈相同的 $\widetilde T_b$：

$$
\mu_{new}(a\mid s)=
\frac{\exp(\ell_{new}(s,i_G(a))/\widetilde T_b)}
{\sum_{b\in\mathcal A_s}\exp(\ell_{new}(s,i_G(b))/\widetilde T_b)}
$$

记 $\mathcal S_{\mathrm{agg}}$ 为本轮聚合后的编码状态集合。反事实 Policy 损失为：

$$
L_{P,\mathrm{CF}}=
\frac{
\sum_{s\in\mathcal S_{\mathrm{agg}}}W_P(s)L_{\mathrm{CE}}
\left(P_{new}(\cdot\mid s),\Pi^+(\cdot\mid s)\right)
}
{\max\left(\sum_{s\in\mathcal S_{\mathrm{agg}}}W_P(s),10^{-8}\right)}
$$

由终局 MC 信号构造的 Policy loss 为：

$$
L_{P,\mathrm{MC}}=
-\frac{
\sum_{s\in\mathcal S_{\mathrm{agg}}}\sum_{a\in\mathcal A_s}
S_A(s,a)\log\mu_{new}(a\mid s)
}
{\max\left(\sum_{s\in\mathcal S_{\mathrm{agg}}}\sum_{a\in\mathcal A_s}W_A(s,a),1\right)}
$$

单条记录场景下，$S_A(s_t,a_t)=A_{\mathrm{MC}}(s_t,a_t)$ 且 $W_A(s_t,a_t)=1$。因此正 $A_{\mathrm{MC}}$ 对应提高轨迹所选动作概率的梯度方向，负 $A_{\mathrm{MC}}$ 对应降低该动作概率的梯度方向。下标 $\mathrm{MC}$ 表示该损失的监督信号来自终局 MC 回报。Policy 参数上的总梯度满足：

$$
\nabla_{\theta_{new}}\left(L_{P,\mathrm{CF}}+L_{P,\mathrm{MC}}\right)=
\nabla_{\theta_{new}}L_{P,\mathrm{CF}}+
\nabla_{\theta_{new}}L_{P,\mathrm{MC}}
$$

MC Value 与树 Value 使用阈值为 1 的 SmoothL1。对预测值与 target 之差 $e$，定义：

$$
\operatorname{SL1}(e)=
\begin{cases}
\frac12e^2, & |e|<1\\
|e|-\frac12, & |e|\geq1
\end{cases}
$$

联合 Value 损失使用一个加权分母：

$$
L_V=
\frac{
\sum_{s\in\mathcal S_{\mathrm{agg}}}W_{\mathrm{MC}}(s)
\operatorname{SL1}\left(V_{new}(s)-\overline G(s)\right)
+\sum_{s\in\mathcal S_{\mathrm{agg}}}W_T(s)
\operatorname{SL1}\left(V_{new}(s)-\overline V_T(s)\right)
}
{\max\left(\sum_{s\in\mathcal S_{\mathrm{agg}}}W_{\mathrm{MC}}(s)+\sum_{s\in\mathcal S_{\mathrm{agg}}}W_T(s),1\right)}
$$

三个顶层损失项的系数均为 1：

$$
L=L_{P,\mathrm{CF}}+L_{P,\mathrm{MC}}+L_V
$$

第 4.7.6 节定义的聚合量构成三个顶层损失项的输入。反事实 Policy 损失 $L_{P,\mathrm{CF}}$ 计算 $P_{new}(\cdot\mid s)$ 相对于 $\Pi^+(\cdot\mid s)$ 的交叉熵。MC Policy 损失 $L_{P,\mathrm{MC}}$ 使用 $S_A(s,a)$ 加权 $\log\mu_{new}(a\mid s)$，并使用 $W_A(s,a)$ 计算加权分母。联合 Value 损失 $L_V$ 使用 $W_{\mathrm{MC}}(s)$ 和 $W_T(s)$ 分别加权 $V_{new}(s)$ 相对于 $\overline G(s)$ 和 $\overline V_T(s)$ 的 SmoothL1 误差。

自动微分计算总损失 $L$ 对 candidate 参数 $\theta_{new}$ 的梯度。AdamW 根据该梯度更新 Policy head、Value head 和共享 ResNet backbone 中的可训练参数。

#### 4.7.8 局部 Policy 更新性质

令 $10^{-12}$ 概率下限趋于 0，并假设 candidate 精确拟合 $\pi^+$。对同一状态的两个动作 $a$ 与 $b$：

$$
\log\frac{\pi^+(a\mid s)}{\pi^+(b\mid s)}
=
\log\frac{P_{old}(a\mid s)}{P_{old}(b\mid s)}
+Q_{\mathrm{CF}}(s,a)-Q_{\mathrm{CF}}(s,b)
$$

若连续局部更新得到固定差值 $\Delta=Q_{\mathrm{CF}}(s,a)-Q_{\mathrm{CF}}(s,b)>0$，第 $k$ 次更新后动作 $a$ 的对数赔率累计增加 $k\Delta$。从初始 Policy $P_0$ 开始，理想拟合下的排序翻转条件为：

$$
k>
\frac{\log P_0(b\mid s)-\log P_0(a\mid s)}{\Delta}
$$

该式描述固定反事实动作价值差下的局部 Policy 更新。它的适用范围是给定状态、给定动作对与固定 $\Delta$ 的迭代拟合过程。

#### 4.7.9 优化、数据文件与晋升

candidate 从本轮 `current.pth` 参数开始训练。每轮创建新的 AdamW，weight decay 固定为 $10^{-4}$。训练开始时以 `--seed` 初始化一次随机数生成器，随后每个 epoch 使用该生成器洗牌记录。训练在 `--epochs` 或 `--train-max-steps` 先到达者处停止。模型使用 `eval()` 模式，使 BatchNorm running statistics 保持固定。模型参数继续参与自动微分。梯度范数裁剪上限固定为 1。`bf16` 作用于 CUDA 前向。logits、Value、损失和 checkpoint 参数使用 FP32。

FCPI HDF5 保存 `policy_targets`、`policy_weights`、`mc_policy_advantage_sums`、`mc_policy_weights`、`mc_value_targets`、`mc_value_weights`、`tree_value_targets` 与 `tree_value_weights`。HDF5 是本轮目标的持久化记录。candidate 训练直接使用完成聚合的内存记录。训练日志中的 `value_mc` 与 `value_tree` 分别用各自权重归一化，仅用于报告。反向传播使用上式定义的联合 $L_V$。

每轮 arena 令 candidate 与本轮 `current.pth` 使用成对开局并交换颜色。若 candidate 的胜、和、负局数分别为 $W$、$D$、$L$，唯一晋升条件为：

$$
W-L\geq\texttt{--min-net-wins}
$$

当 $W-L\geq\texttt{--min-net-wins}$ 时，candidate 原子写入该 run 的 `current.pth`。当 $W-L<\texttt{--min-net-wins}$ 时，本轮开始时的 current 成为下一轮的数据生成模型。`scripts/gadus_fcpi.sh` 在全部轮次结束后执行一次 `current.pth` 对 `initial.pth` 的 `closed` paired arena，并把结果写入 `summary.json` 的 `final_arena`。`final_arena` 记录整次运行的累计对战结果。每轮 candidate 对 current 的 arena 结果单独决定该轮晋升。

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

candidate 达到 arena gate 后，程序以原子替换方式更新该 run 的 `current.pth`。

后台启动脚本为：

```bash
bash scripts/gadus_fcpi.sh
```

脚本启动后台进程后会输出 `run_id`、PID 与日志路径。运行期间可按脚本输出的实际路径查看日志：

```bash
tail -n 100 -f data/runs/<run-id>/info.log
```

`data/runs/<run-id>/pid` 保存后台进程 PID。下列命令读取该文件并向进程发送终止信号：

```bash
kill "$(cat data/runs/<run-id>/pid)"
```

脚本默认使用 `PRECISION=bf16`、`GAMES_PER_ITER=6000`、`TRAIN_MAX_STEPS=6000`、`data/openings.sam.bin`、较大的批次与 batched games。每轮从采样书选择 6000 个不同起始局面进行 current-self 对局。可通过环境变量覆盖，例如 `PRECISION=fp32 BATCH_SIZE=512 bash scripts/gadus_fcpi.sh`。

脚本会等待 FCPI 完成全部迭代，再调用独立 Arena 执行最终 `current.pth` 对同一 run 的 `initial.pth` 的 `closed` paired 对战。它复用 `EVAL_GAMES`、开局书、并发数和最大步数，并设置 `search_type=closed` 与 `sims=0`。结果写入 `summary.json` 的 `final_arena`，棋谱写入 `current_vs_initial.pgn`。`final_arena` 负责报告整次运行的净变化。candidate 晋升由每轮 candidate 对 current 的 arena 结果决定。直接运行 `build/gadus/fcpi` 时，流程在 FCPI 迭代结束处终止。

## 5. Melano

### 5.1 状态与动作编码

Melano 将局面编码为 67 个整数：64 个棋盘格 piece tokens、1 个行棋方 token、1 个王车易位 bitmask 和 1 个 en passant 文件 token。空格编码为 0，白方六种棋子编码为 1 到 6，黑方六种棋子编码为 7 到 12。模型把它们展开为 64 个 square tokens 与 1 个 global token。

令 $e_{piece}$、$e_{square}$、$e_{side}$、$e_{castle}$ 与 $e_{ep}$ 表示对应 embedding，令 $g_0$ 表示可训练 global token。对状态中的行棋方 $u$、王车易位 bitmask $c$ 和吃过路兵文件 $e$，全局规则上下文为：

$$
\chi=e_{side}(u)+e_{castle}(c)+e_{ep}(e)
$$

棋盘格 $i$ 上的棋子类别为 $\tau_i$。初始 global token 与 square tokens 为：

$$
h_0=g_0+\chi,\qquad
h_{i+1}=e_{piece}(\tau_i)+e_{square}(i)+\chi,\quad 0\leq i<64
$$

记 $\mathcal I_M$ 为 Melano 的动作索引集合，$i_M(a)$ 为合法动作 $a$ 的编码。`sd_64x64_underpromo9` 编码满足：

$$
|\mathcal I_M|=64\times64+64\times9=4672
$$

普通走法和后升变使用 source-destination 编码。马、象、车升变使用每个起点的 3 个方向乘 3 个升变棋子编码。

### 5.2 网络

每个 geometry attention block 包含 pre-norm multi-head self-attention、32 类棋盘几何关系、global token 生成的动态关系 bias、residual connection 和 pre-norm feed-forward network。令 $C$ 为 `--channels`。attention head 数 $n_h$ 取集合 $\{8,4,2,1\}$ 中可以整除 $C$ 的最大值，单个 head 的通道数为 $d_h=C/n_h$。feed-forward network 的通道变化为 $C\rightarrow4C\rightarrow C$，中间激活函数为 GELU。

令 $h_i$ 为该层第 $i$ 个 token，$\zeta_i$ 为该层学习的位置向量，并定义 $\widetilde h_i=h_i+\zeta_i$。$\kappa_{ij}\in\{0,\ldots,31\}$ 是有序 token 对 $(i,j)$ 的几何关系类别。global token 相关的 token 对使用类别 0。棋盘格 token 对根据同格、同行、同列、对角线、马步、邻接和距离关系使用类别 1 至 31。对 attention head $m$，静态关系参数为 $b^{(m)}_{\kappa_{ij}}$。global token $\widetilde h_0$ 经过 LayerNorm 和两层 MLP 后产生动态关系参数 $d^{(m)}_{\kappa_{ij}}(\widetilde h_0)$。attention 权重为：

$$
\alpha^{(m)}_{ij}=\operatorname{softmax}_j\left(
\frac{q^{(m)}_i\left(k^{(m)}_j\right)^{\mathsf T}}{\sqrt{d_h}}
+b^{(m)}_{\kappa_{ij}}+d^{(m)}_{\kappa_{ij}}(\widetilde h_0)
\right)
$$

其中 $q^{(m)}_i$、$k^{(m)}_j$ 与 $v^{(m)}_j$ 分别是 $\operatorname{LN}(\widetilde h)$ 的 query、key 与 value 线性投影。每个 head 按 $\sum_j\alpha^{(m)}_{ij}v^{(m)}_j$ 汇总 value vectors。各 head 的结果拼接后经过输出投影并加回 $\widetilde h$，随后 pre-norm feed-forward network 形成第二次残差更新。这里的 value vector 属于 attention 运算，本节后续定义的 $V_\theta(s)$ 属于棋局状态 Value。

`--blocks` 决定上述 geometry attention block 的数量。Policy head 对 64 个 square tokens 执行 LayerNorm，再分别执行 $C\rightarrow C$ 的 source projection 与 destination projection。两组向量的缩放点积生成 $64\times64$ logits。独立的 $C\rightarrow9$ 线性映射为每个起点生成 9 个 underpromotion logits。记完整动作索引 logits 为 $\ell_\theta(s)\in\mathbb R^{4672}$。

Value head 对 global token 执行 LayerNorm、线性映射 $C\rightarrow256$、ReLU、线性映射 $256\rightarrow1$ 和 tanh。记 $\mathcal V_\theta(h)$ 为 Value head 对 latent token 序列 $h$ 的输出。精确状态 Value 为 $V_\theta(s)=\mathcal V_\theta(E_\theta(s))$，其范围为：

$$
V_\theta(s)\in[-1,1]
$$

记 $u_{A,\theta}(s,i_M(a))$ 为 Advantage head 在动作 $a$ 上的无界标量输出。该 head 的最终输出为：

$$
A_\theta(s,a)=-2\tanh^2(u_{A,\theta}(s,i_M(a)))\in[-2,0]
$$

动作价值定义为：

$$
Q_\theta(s,a)=\operatorname{clip}(V_\theta(s)+A_\theta(s,a),-1,1)
$$

记精确状态 encoder 为 $E$，动作条件 latent dynamics 为 $D$。潜在转移使用动作 embedding 条件化一个 residual geometry-attention block。记 $\mathcal B$ 为该 block，$c(a)$ 为动作条件向量，$\gamma(a)$ 为逐通道门控 logits，$\sigma$ 为 sigmoid，$\odot$ 为逐元素乘法：

$$
h=E(s)
$$

$$
\widehat h'=D(h,a)=\operatorname{LN}\left(h+\sigma(\gamma(a))\odot
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

`--has-cmt` 的默认值为 `1`。对第 $t$ 个 PGN 动作，记 $x_t$ 为走前环境状态，$a_t$ 为该动作，$x_{t+1}=T(x_t,a_t)$。$c_{t-1}$ 是前一动作后的评注，$c_t$ 是当前动作后的评注。定义 $q(c)$ 为评注中白方视角的兵值评分。定义 $\rho(x)=1$ 表示白方行棋，$\rho(x)=-1$ 表示黑方行棋。可解析评注对应的行棋方视角 Value 为：

$$
v(c,x)=\tanh\left(\frac{\rho(x)q(c)}{3}\right)
$$

使用 `--has-cmt 1` 时，当前状态与后继状态的 Value target 分别为：

$$
V_{target}(s_t)=
\begin{cases}
v(c_{t-1},x_t),&q(c_{t-1})\text{ 可解析}\\
0,&q(c_{t-1})\text{ 缺失}
\end{cases}
$$

$$
V_{target}(s_{t+1})=
\begin{cases}
v(c_t,x_{t+1}),&q(c_t)\text{ 可解析}\\
0,&q(c_t)\text{ 缺失}
\end{cases}
$$

当前动作前后的评分都可解析时，程序把走后评分换算到 $x_t$ 的行棋方视角，并定义监督 Advantage target：

$$
A_{target}(s_t,a_t)=\operatorname{clip}
\left(
\tanh\left(\frac{\rho(x_t)q(c_t)}{3}\right)
-v(c_{t-1},x_t),-2,0
\right)
$$

其余评注组合对应 $A_{target}(s_t,a_t)=0$。使用 `--has-cmt 1` 时，程序接收至少包含一个可解析评分的对局。使用 `--has-cmt 0` 时，程序按照各状态的行棋方视角把 PGN 终局结果写入当前状态和后继状态的 Value target，并将全部 Advantage target 写为 0。训练程序在该模式下将 dueling action Value 损失设为 0。

`next_values` 保存走后局面在新行棋方视角下的 Value target。当前动作后的评注 $c_t$ 决定这一 target，因此每盘棋的第一条可解析评注可以监督走后 latent。

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

监督训练的 dueling action Value 预测与 target 为：

$$
\widehat Q_\theta(s,a)=\operatorname{clip}(V_\theta(s)+A_\theta(s,a),-1,1)
$$

$$
Q_{target}(s,a)=\operatorname{clip}
\left(V_{target}(s)+A_{target}(s,a),-1,1\right)
$$

精确棋规生成动作后的状态 $s'$。在线 encoder 参数记为 $\theta_E$，缓慢移动的 target encoder 参数记为 $\bar\theta_E$。预测 latent 与停止梯度的精确 successor latent 分别为：

$$
\widehat h'=D_\theta(E_{\theta_E}(s),a),
\qquad
\overline h'=\operatorname{stopgrad}(E_{\bar\theta_E}(s'))
$$

每次在线模型更新后，target encoder 使用指数移动平均更新。记 `--target-decay` 为 $m$：

$$
\bar\theta_E\leftarrow m\bar\theta_E+(1-m)\theta_E
$$

潜在一致性损失 $L_D$ 逐 token 比较走后预测与精确棋规生成的走后状态编码：

$$
L_D=1-\frac{1}{65}\sum_{i=1}^{65}
\cos(\widehat h'_i,\overline h'_i)
$$

Imagined Value 损失 $L_I$ 定义为：

$$
L_I=\operatorname{MSE}(\mathcal V_\theta(\widehat h'),V_{target}(s'))
$$

监督 Policy 损失对全部 4672 个动作索引 logits 计算交叉熵。定义完整动作索引分布：

$$
R_\theta(i\mid s)=
\frac{\exp\ell_\theta(s,i)}
{\sum_{j\in\mathcal I_M}\exp\ell_\theta(s,j)}
$$

记 PGN 中的实际走法为 $a^*$，其动作索引 $i_M(a^*)$ 的 one-hot 分布为 $\delta_{i_M(a^*)}$。$\lambda_V$、$\lambda_Q$、$\lambda_D$ 与 $\lambda_I$ 分别对应 `--value-weight`、`--dueling-q-weight`、`--dynamics-weight` 与 `--imagined-value-weight`。完整监督损失为：

$$
L_{\mathrm{sup}}=
L_{\mathrm{CE}}\left(R_\theta(\cdot\mid s),\delta_{i_M(a^*)}\right)+
\lambda_V\operatorname{MSE}(V_\theta(s),V_{\mathrm{target}}(s))+
\lambda_Q\operatorname{MSE}(\widehat Q_\theta(s,a^*),Q_{target}(s,a^*))+
\lambda_D L_D+\lambda_I L_I
$$

记 $\mathbf g_\theta=\nabla_\theta L_{\mathrm{sup}}$ 为监督损失对在线模型参数的梯度，记 $c$ 为 `--grad-clip` 给出的全局范数上限。训练程序按下式裁剪梯度：

$$
\mathbf g_\theta\leftarrow
\mathbf g_\theta\min\left(1,\frac{c}{\lVert\mathbf g_\theta\rVert_2}\right)
$$

当 `--grad-clip` 大于 0 时，训练程序会在检测到非有限梯度范数时终止本次运行。target encoder 在训练进程中提供停止梯度的 successor latent。checkpoint 保存在线模型参数，Melano 推理程序加载该在线模型。

```bash
build/melano/train \
	--data data/games.melano.h5 \
	--out models/melano/melano.pth \
	--channels 128 \
	--blocks 12 \
	--epochs 3 \
	--batch-size 256 \
	--max-steps 500000 \
	--lr 0.0002 \
	--weight-decay 0.0001 \
	--value-weight 1.0 \
	--dueling-q-weight 0.5 \
	--dynamics-weight 0.25 \
	--imagined-value-weight 0.25 \
	--target-decay 0.995 \
	--grad-clip 1.0 \
	--device cuda \
	--precision bf16 \
	--log-every 50
```

`train` 每次创建新的 Melano 模型。`--channels` 和 `--blocks` 决定 geometry attention 宽度与层数，checkpoint 通过临时文件和 rename 原子写回。Melano checkpoint 的逻辑顶层为 `model` 与 `arch`，其中 `arch` 保存架构标识、`channels`、`blocks` 和 `action_size`。

`--precision` 可取 `fp32` 或 `bf16`，默认 `fp32`。CUDA 前向计算在 `bf16` 模式下使用 BF16。Policy softmax、P/V/A 与 latent dynamics 损失、指标累计和 checkpoint 参数使用 FP32。CUDA 训练批次使用 pinned memory。

`--lr` 表示 AdamW 的峰值学习率。设计划训练步数为 $T$，warmup 步数为

$$
T_w=\min\left(T,\ 2000,\ \max\left(100,\left\lfloor\frac{T}{100}\right\rfloor\right)\right)
$$

第 $t$ 步的学习率为

$$
\eta_t=
\begin{cases}
\eta_{\max}\dfrac{t}{T_w}, & t\le T_w,\\
\eta_{\max}\sqrt{\dfrac{T_w}{t}}, & t>T_w.
\end{cases}
$$

该轨迹在训练初期逐步建立 AdamW 统计量，随后降低 Transformer 参数更新幅度。latent dynamics 的 cosine 归一化使用固定正范数下限，使退化 latent 的除法保持有限。日志中的 `grad_norm_before_clip` 是裁剪前的全局梯度范数。裁剪后的参数梯度范数上限是 `--grad-clip`，AdamW 根据裁剪后的梯度更新参数。

### 5.5 搜索

`closed` 按 Melano 合法动作 Policy 排序。对 $a\in\mathcal A(x)$，该 Policy 为：

$$
P_\theta(a\mid s)=
\frac{\exp\ell_\theta(s,i_M(a))}
{\sum_{b\in\mathcal A(x)}\exp\ell_\theta(s,i_M(b))}
$$

对于每条边 $(s,a)$，$P(s,a)$ 表示模型在节点展开时为动作 $a$ 计算的 Policy 概率。该概率称为边 $(s,a)$ 的 Policy 先验概率，并在节点存续期间保持不变。

`only-mcts` 使用 anchored latent MCTS。锚定周期 $K=2$ 表示每经过两个 ply 重新从精确棋盘编码 latent，因此任意预测 latent 最多跨越一个动作。每条边还具有一份伪访问。定义：伪访问表示 $V+A$ 产生的动作价值先验以一个统计样本的权重参与边价值估计。实际访问次数由叶节点回传单独累计。

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

其中 $d$ 是根节点到当前节点的 ply 深度，$h_d$ 是该节点的 latent，$\mathcal H$ 表示作用于 $h_d$ 的 Policy head、Value head 与 Advantage head。因此任意预测 latent 与最近的精确编码只相隔一个动作，运行时分布与当前一步 dynamics 训练目标一致。偶数深度节点缓存 $E(s_d)$，奇数深度 latent 只在批量评价期间存在，从而限制设备内存。`search` 输出中的 `exact_evaluations` 与 `latent_evaluations` 分别报告两类网络评价位置数。

$$
Q_{\mathrm{prior}}(s,a)=\operatorname{clip}(V(s)+A(s,a),-1,1)
$$

记 $N(s,a)$ 为边的实际访问次数，$N(s)=\sum_aN(s,a)$ 为节点访问次数，$Q_{\mathrm{MCTS}}(s,a)$ 为父节点行棋方视角下实际访问的平均叶节点回传。对已有实际访问的边：

$$
Q_{\mathrm{edge}}(s,a)=
\frac{N(s,a)Q_{\mathrm{MCTS}}(s,a)+Q_{\mathrm{prior}}(s,a)}
{N(s,a)+1}
$$

实际访问数 $N(s,a)=0$ 的边使用：

$$
Q_{\mathrm{edge}}(s,a)=\operatorname{clip}\left(
Q_{\mathrm{prior}}(s,a)-r_{\mathrm{FPU}}
\sqrt{\sum_{b:N(s,b)>0}P(s,b)},-1,1
\right)
$$

其中 $r_{\mathrm{FPU}}\geq0$ 是 FPU 折减系数。记 $c_0$、$b_0$ 与 $f_0$ 分别为 PUCT 的初始系数、输入基数和输入增长系数，并定义 $b=\max(1,b_0)$ 与 $f=\max(0,f_0)$。Melano 的动态探索系数为：

$$
c_{\mathrm{puct}}(\widetilde N)=
\max\left(0,c_0+f\log\left(\frac{\widetilde N+b+1}{b}\right)\right)
$$

记 $N_v(s,a)$ 为同一批 selection 中施加在边上的 virtual visits，$N_v(s)=\sum_aN_v(s,a)$。选择阶段使用增广访问数：

$$
\widetilde N(s,a)=N(s,a)+N_v(s,a),\qquad
\widetilde N(s)=N(s)+N_v(s)
$$

令 $l_v=\max(0,\texttt{--virtual-loss})$ 为每个 virtual visit 的 virtual loss。PUCT 选择分数为：

$$
S(s,a)=Q_{\mathrm{edge}}(s,a)+c_{\mathrm{puct}}(\widetilde N(s))P(s,a)
\frac{\sqrt{\widetilde N(s)+1}}{1+\widetilde N(s,a)}-l_vN_v(s,a)
$$

选择分数相同时，搜索器依次按 $P(s,a)$ 和边价值 $Q_{\mathrm{edge}}(s,a)$ 降序选择边。

每次 simulation 完成叶节点评价后，搜索器将叶节点 Value 沿所选路径逐 ply 取反并回传，同时更新路径上各边的 $N(s,a)$ 和 $Q_{\mathrm{MCTS}}(s,a)$。全部 simulation 结束后，搜索器使用 $N(s,a)+P(s,a)$ 作为根动作 $a$ 的分布权重，并将所有合法根动作的分布权重归一化为根 Policy：

$$
P_{\mathrm{root}}(a\mid s)=\frac{N(s,a)+P(s,a)}
{\sum_{b\in\mathcal A(x)}(N(s,b)+P(s,b))}
$$

记 $N_{\mathrm{cap}}=\max(0,\texttt{--mcts-sims})$，$B=\max(1,\texttt{--mcts-batch-size})$，$M=\texttt{--mcts-min-sims}$。搜索器按下式计算 simulation 下限：

$$
N_{\min}=
\begin{cases}
0,&N_{\mathrm{cap}}=0,\\
\max\!\left(1,\min\!\left(N_{\mathrm{cap}},M\right)\right),&N_{\mathrm{cap}}>0\ \text{且}\ M>0,\\
\max\!\left(1,\min\!\left(N_{\mathrm{cap}},\max\!\left(B,\left\lfloor\dfrac{N_{\mathrm{cap}}}{4}\right\rfloor\right)\right)\right),&N_{\mathrm{cap}}>0\ \text{且}\ M=0.
\end{cases}
$$

`movetime` 截止时间和 `stop` 请求具有更高终止优先级，并可在实际 simulation 数到达 $N_{\min}$ 前结束搜索。动态模拟预算在完成 $N_{\min}$ 次 simulation 后使用实际访问分布：

$$
v_a=\frac{N(s,a)}{\sum_{b\in\mathcal A(x)}N(s,b)}
$$

记 $a_1$ 与 $a_2$ 为按实际访问数排序的前两项，$N_i=N(s,a_i)$，$Q_i=Q_{\mathrm{MCTS}}(s,a_i)$。访问分布归一化熵 $H_N$、前两项访问接近度 $U_N$ 和前两项实际回传价值接近度 $U_Q$ 为：

$$
H_N=-\frac{\sum_{a\in\mathcal A(x)}v_a\log v_a}
{\log|\mathcal A(x)|}
$$

$$
U_N=1-\frac{|N_1-N_2|}{\max(1,N_1+N_2)},\qquad
U_Q=1-\min\left(1,\frac{|Q_1-Q_2|}{0.5}\right)
$$

$$
u=\operatorname{clip}(0.5H_N+0.35U_N+0.15U_Q,0,1)
$$

$$
N_{\mathrm{target}}=
N_{\min}+\left\lceil u(N_{\mathrm{cap}}-N_{\min})\right\rceil
$$

其中 $u\in[0,1]$ 是根节点不确定性。

Melano 的 IMF 与 RPP 同样位于最终决策层。记 $D_0(a)=P_\theta(a\mid s)$ 表示 `closed` 的基础排序分数，记 $D_0(a)=P_{\mathrm{root}}(a\mid s)$ 表示 `only-mcts` 的基础排序分数。令 $\mathcal M(x)$ 为执行后立即形成 checkmate 的合法动作集合。IMF 在 $\mathcal M(x)$ 中选择基础分数最高的动作 $a_M$，并令：

$$
D_I(a)=
\begin{cases}
1,&a=a_M\\
D_0(a),&a\in\mathcal A(x)\setminus\{a_M\}
\end{cases}
$$

$\mathcal M(x)=\varnothing$ 时定义 $D_I(a)=D_0(a)$。

令 $\lambda_R\in[0,1]$ 为 `--repetition-policy-penalty`，$V_R$ 为搜索返回的根 Value。集合 $\mathcal R_3(x)$ 包含执行后立即满足三次重复申和条件的动作，以及执行后让对手获得一步重复申和应手的动作。RPP 定义为：

$$
d_R=\lambda_R\operatorname{clip}(V_R,0,1)
$$

$$
D(a)=
\begin{cases}
\max(0,D_I(a)-d_R),&a\in\mathcal R_3(x)\\
D_I(a),&a\in\mathcal A(x)\setminus\mathcal R_3(x)
\end{cases}
$$

最终走法先按 $D(a)$ 降序排列，再按基础 Policy 降序排列，最后按 UCI 走法字符串降序排列。

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

CUDA 搜索向 CPU 返回根 Value，并从动作输出中选取合法动作的 Policy 与 Advantage。

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
score_b=\operatorname{clip}(score,10^{-6},1-10^{-6})
$$

$$
\Delta Elo=400\log_{10}\left(\frac{score_b}{1-score_b}\right)
$$

记 $M=\texttt{--min-net-wins}$。gate 条件为：

$$
W-L\geq M
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

Gadus UCI 输出 MultiPV、行棋方视角的 `score cp`、节点数、NPS、耗时和单步 PV。具有实际访问的 MCTS 根边使用该边在根节点视角下的平均回传 $Q(s,a)$ 作为 $q_{line}$。`closed` 根边和零访问根边使用搜索结果的根 Value。令 $c_s$ 为 `ScoreScale`，显示分数为：

$$
score\_cp=\operatorname{round}\left(c_s\operatorname{clip}(q_{line},-0.999,0.999)\right)
$$

Gadus UCI 公开以下选项：

- `ModelPath`：Gadus checkpoint 路径。打包引擎默认读取可执行文件同目录的 `gadus.pth`。
- `Device`：`auto`、`cpu` 或 `cuda`。
- `SearchType`：`closed` 使用 Gadus Policy 与最终决策组件，`only-mcts` 使用 Gadus MCTS 与最终决策组件。
- `MCTSSims`：MCTS simulation 上限，默认 `100`。客户端发送 `go nodes <n>` 时以 `<n>` 为当前搜索上限。
- `MCTSMinSims`：动态模拟预算开始计算前完成的 simulation 下限，默认 `0`。值为 `0` 时，Gadus 搜索器根据 MCTS 上限与 batch size 计算该下限。
- `MCTSBatchSize`：一次 Gadus 神经网络叶子批量，默认 `32`。
- `MoveTimeMS`：`go` 命令同时缺少 `movetime` 与当前行棋方棋钟时采用的固定思考时间，默认 `0`。
- `MoveOverheadMS`：从棋钟预算中预留的通信与落子时间，默认 `50`。
- `MinMoveTimeMS`、`MaxMoveTimeMS`：棋钟模式下的单步时间边界，默认 `50` 与 `10000`。
- `TimeDivisor`：把剩余时间按该除数分配给当前步，默认 `30.0`。
- `IncrementFraction`：当前步可使用的加秒比例，默认 `0.75`。
- `CPuct`、`CPuctBase`、`CPuctFactor`：Gadus PUCT 探索系数及其访问数 schedule，默认 `0.5`、`19652`、`1.0`。
- `FPUReduction`：Gadus MCTS 边首次被实际访问前使用的 FPU 折减，默认 `0.15`。
- `VirtualLoss`：同一批 Gadus MCTS selection 的重复路径惩罚，默认 `0.0`。
- `RepetitionPolicyPenalty`：Gadus 决策层对己方优势时重复和棋走法的排序惩罚，范围 $[0,1]$，默认 `0.0`。
- `InstantMateFirst`：Gadus 决策层发现一步将杀时优先选择该走法，默认 `false`。
- `ProgressIntervalMS`：UCI 中间 `info` 行的发布间隔，默认 `750`。值为 `0` 时，中间结果发布次数为 0。
- `MultiPV`：输出的分析行数，默认 `5`。
- `ScoreScale`：把 Gadus 的 $[-1,1]$ Value/Q 映射为 `score cp` 的显示比例，默认 `1000`。

Gadus 在搜索开始时发布 Policy 结果，在 MCTS 期间按 `ProgressIntervalMS` 发布中间结果。该间隔控制可见更新频率。搜索参数和停止命令决定搜索预算、搜索顺序与最终结果。

### 6.2 Melano

```powershell
build\melano\uci.exe `
	--model models\melano\melano.pth `
	--device cuda `
	--search-type only-mcts `
	--mcts-sims 1000
```

Melano UCI 加载 Melano checkpoint。`SearchType=closed` 使用 Melano Policy，`SearchType=only-mcts` 使用第 5.5 节定义的 anchored latent MCTS。

Melano UCI 输出 MultiPV、行棋方视角的 `score cp`、节点数、NPS、耗时和单步 PV。具有实际访问的 MCTS 根边使用第 5.5 节定义的 $Q_{\mathrm{edge}}(s,a)$ 作为 $q_{line}$。`closed` 根边和零访问根边使用搜索结果的根 Value。令 $c_s$ 为 `ScoreScale`，显示分数为：

$$
score\_cp=\operatorname{round}\left(c_s\operatorname{clip}(q_{line},-0.999,0.999)\right)
$$

Melano UCI 公开以下选项：

- `ModelPath`：Melano checkpoint 路径。打包引擎默认读取可执行文件同目录的 `melano.pth`。
- `Device`：`auto`、`cpu` 或 `cuda`。
- `SearchType`：`closed` 使用 Melano Policy 与最终决策组件，`only-mcts` 使用 Melano anchored latent MCTS 与最终决策组件。
- `MCTSSims`：MCTS simulation 上限，默认 `100`。客户端发送 `go nodes <n>` 时以 `<n>` 为当前搜索上限。
- `MCTSMinSims`：动态模拟预算开始计算前完成的 simulation 下限，默认 `0`。值为 `0` 时，Melano 搜索器根据 MCTS 上限与 batch size 计算该下限。
- `MCTSBatchSize`：一次 Melano 神经网络叶子批量，默认 `32`。
- `MoveTimeMS`：`go` 命令同时缺少 `movetime` 与当前行棋方棋钟时采用的固定思考时间，默认 `0`。
- `MoveOverheadMS`：从棋钟预算中预留的通信与落子时间，默认 `50`。
- `MinMoveTimeMS`、`MaxMoveTimeMS`：棋钟模式下的单步时间边界，默认 `50` 与 `10000`。
- `TimeDivisor`：把剩余时间按该除数分配给当前步，默认 `30.0`。
- `IncrementFraction`：当前步可使用的加秒比例，默认 `0.75`。
- `CPuct`、`CPuctBase`、`CPuctFactor`：Melano PUCT 探索系数及其访问数 schedule，默认 `0.5`、`19652`、`1.0`。
- `FPUReduction`：Melano MCTS 边首次被实际访问前使用的 FPU 折减，默认 `0.15`。
- `VirtualLoss`：同一批 Melano MCTS selection 的重复路径惩罚，默认 `0.0`。
- `RepetitionPolicyPenalty`：Melano 决策层对己方优势时重复和棋走法的排序惩罚，范围 $[0,1]$，默认 `0.0`。
- `InstantMateFirst`：Melano 决策层发现一步将杀时优先选择该走法，默认 `false`。
- `ProgressIntervalMS`：UCI 中间 `info` 行的发布间隔，默认 `750`。值为 `0` 时，中间结果发布次数为 0。
- `MultiPV`：输出的分析行数，默认 `5`。
- `ScoreScale`：把 Melano 的 $[-1,1]$ Value/Q 映射为 `score cp` 的显示比例，默认 `1000`。

Melano 在搜索开始时发布 Policy 结果，在 MCTS 期间按 `ProgressIntervalMS` 发布中间结果。该间隔控制可见更新频率。搜索参数和停止命令决定搜索预算、搜索顺序与最终结果。

### 6.3 UCI 客户端配置

Gadidae GUI 在导入引擎时执行标准 UCI 握手，并读取全部 `option name ...` 声明。Settings 根据 option 类型自动生成 checkbox、整数输入、下拉框、文本框或命令按钮。Gadus、Melano、Stockfish 以及其他 UCI 引擎分别显示自身公开的选项。选项值随引擎配置保存，命令按钮在 Apply 后向对应引擎发送一次。`Launch arguments` 用于进程启动命令中的非 UCI 参数，通常留空。

其他 UCI 客户端使用标准命令设置引擎公开的选项。以下示例配置 Gadus：

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

Gadus UCI 的默认 checkpoint 路径是可执行文件目录中的 `gadus.pth`，Melano UCI 的默认 checkpoint 路径是可执行文件目录中的 `melano.pth`。显式 `--model` 参数覆盖默认路径。Windows 目录结构为：

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

Windows UCI 客户端直接注册 `models/gadus/gadus.exe` 或 `models/melano/melano.exe`，可执行文件会加载同目录 checkpoint。Linux 客户端注册对应架构目录内的 `gadus` 或 `melano` launcher。

## 7. GUI

`Gadidae` 是基于 GLFW、OpenGL 3.3、GLAD、Dear ImGui 与 FreeType 的原生图形程序。Simulator 与 Stadium 共用一个窗口。图形程序与 Gadus、Melano、Stockfish 或其他引擎之间的通信接口是 UCI。

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

Windows 双击 `build/graphics/Gadidae.exe`，具有 X11 或 Wayland 图形会话的 Linux 运行 `build/graphics/Gadidae`。程序默认进入 Simulator，顶部模式控件可切换到 Stadium。FreeType 作为 GUI 的构建目标参与链接。Windows GUI 静态链接 zlib，Linux GUI 链接依赖安装阶段验证过的 zlib。SSH 服务器需要 X11 forwarding、远程桌面或其他可见显示服务才能实际操作 GUI。`xvfb-run` 适合自动化启动测试，交互操作则需要可见显示服务。

Simulator 用于局面分析。Settings 的 `Engine` 区域填写 UCI 可执行文件和显示名称。程序完成 UCI 握手后，`Options` 区域自动列出该引擎公开的全部设置。例如 Gadus 会显示 `Device`、`SearchType`、`MCTSSims` 与 `MultiPV`，Stockfish 会显示其自身的 `Threads`、`Hash` 等选项。`Launch arguments` 用于进程启动参数，默认折叠。`Run` 同时显示 `Open` 与 `Close`，勾选项表示实时分析的当前状态。也可以从命令行指定常用参数：

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

Stadium 用于同时组织多盘独立对局。`Tools > Matches` 可新建、进入和关闭对局，切换到 Simulator 或进入另一盘棋时，其余对局继续在后台运行。默认名称为 `#<id> <white> vs. <black>`，名称留空时自动采用该格式。每个席位都必须填写参赛者名称。Human toggle 打开时引擎设置整体置灰，走棋由用户直接在棋盘上完成。Human toggle 关闭时，席位需要填写 UCI 可执行文件，并分别使用握手生成的 Options 与可选启动参数。`Match` 区域设置双方共用的初始时间、每步加秒、显示延迟与最大 ply，初始时间设为 0 时关闭棋钟。`Run > Start/Pause/Stop` 只控制当前进入的对局，关闭程序会终止全部对局拥有的 UCI 子进程。可从命令行预填首盘对局的双方引擎：

```powershell
build\graphics\Gadidae.exe `
	--mode stadium `
	--white-uci "models\gadus\gadus.exe" `
	--white-name "Gadus" `
	--black-uci "models\stockfish\stockfish.exe" `
	--black-name "Stockfish"
```

Appearance 提供 `Dark` 与 `Light` 应用主题、基础字号、受限范围内随窗口缩放的字号调整、棋盘配色预设、颜色编辑、坐标开关以及 `Vector`、`RhosGFX`、`Chessnut`、`Spatial`、`Cburnett`、`Fantasy` 棋子样式。棋子均由编译进程序的预计算几何绘制。SVG 的填充、线性渐变与描边在导入阶段完成预三角化，外部样式经过顶点去重和 zlib 压缩后嵌入可执行文件。运行时直接从可执行文件的数据资源读取棋子几何。点击 Apply 后设置写入 `Gadidae` 可执行文件同目录的 `gui.json`，后续启动会自动恢复。已有的用户目录配置会在首次启动时迁移到该位置。外观设置更新会保持已经加载的 UCI 进程与正在进行的 Stadium 对局。`--font-size <px>`、`--theme dark`、`--theme light` 与 `--piece-style vector|rhosgfx|chessnut|spatial|cburnett|fantasy` 可覆盖本次启动的外观。第三方棋子样式的来源与许可记录在根目录 `THIRD_PARTY.md`。

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

导入器一次性读取 12 个 SVG，完成预三角化、逐棋子无损顶点去重与 zlib 压缩。导入结果写入 `pieces.gpack`，SVG 输入目录保持原状并继续由用户管理。新名称向几何包追加一个内置样式，同名导入原子覆盖该样式。`--curve-step` 控制曲线路径采样间距，默认值为 `1.5`。较小值产生更平滑且更大的几何数据，较大值减少几何包体积与导入成本。构建时，Windows 通过 `RCDATA`、Linux 通过只读数据段把几何包嵌入 `Gadidae`。

每个新增的第三方样式都必须在根目录 `THIRD_PARTY.md` 中独立声明。仓库长期保存预三角化几何与第三方声明，用户自行管理 SVG 原文件。第三方声明长期记录样式来源与授权信息。来源应链接到固定 commit、固定版本或其他固定页面，并记录原作者、许可证名称和许可证全文链接。声明格式：

```markdown
## <Style Name>

- Author: <author or project>
- Source: <permanent source URL>
- License: <license name and version>
- License text: <license text URL>
```

导入前应确认许可证允许复制、修改和随项目分发，并遵守署名、相同方式共享、源代码提供或用途限制等条款。经过修改的素材应在对应声明中注明修改内容。由多个来源组合的样式应分别列出各来源及其许可证。

Simulator 首次打开分析时在后台启动并加载一次 UCI 引擎，后续局面复用同一子进程。切换局面时会异步停止上一轮计算、丢弃其后续输出，再把最新局面交给已经加载的引擎。Gadus 与 Melano 的 UCI 命令循环和搜索线程彼此分离，MCTS 搜索会在批次边界响应 `stop`。图形程序在窗口持续移动或缩放时暂停 OpenGL 提交，操作停止后清空 GPU 命令队列并按新尺寸交换完整帧。

## 8. UCI 分析

`analyze.py` 直接连接 UCI 引擎，计算招法评分与 regret，并可使用 SQLite cache。对一个走前局面，记 $c_{best}$ 为 UCI 引擎从当前行棋方视角给出的最高 `score cp`，$c_{played}$ 为实际走法的 `score cp`。脚本定义：

$$
regret_{cp}=\max(0,c_{best}-c_{played})
$$

默认 `.cmt` 输出路径与输入 PGN 同名。使用 `--pgn-comments` 时，脚本把每个走后局面的评分换算为白方视角评注，并输出 `<输入名>_cmt.pgn`。该评注格式可以作为 `--has-cmt 1` 的预处理输入。

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

## 9. 开局书

Gadus FCPI 的采样书直接采用 Polyglot book 或 PGN 中的可达状态。状态选择允许任意 ply 和任意局面评价。下列命令从现有 Polyglot book 的整张可达图中提取至少 10000 个唯一进行中状态，标准初始局面也属于该状态集合：

```powershell
python scripts/opening_book.py --sampling-source data/openings.bin --output data/openings.sam.bin --min-fens 10000 --log-every 1000
```

`--sampling-source` 也可指向 PGN。此时脚本读取所有合法主线，并将任意 ply 的可达状态直接写入同一 Polyglot 格式。FCPI 使用 `--opening-book data/openings.sam.bin --max-book-positions 10000` 读取该池。

Arena 验收使用固定第 8 ply 的 `openings.gen.bin`，同一个局面交换双方颜色组成 paired games。UCI 评分的绝对值上限负责筛选该书中的均衡局面：

```bash
bash scripts/run_opening.sh data/games.pgn 1000 data/openings.gen.bin
```

```powershell
scripts\run_opening.bat data\games.pgn 1000 data\openings.gen.bin
```

```bash
python scripts/opening_book.py \
	--pgn data/games.pgn \
	--uci models/stockfish/stockfish \
	--output data/openings.gen.bin \
	--max-abs-cp 80 \
	--book-plies 8 \
	--min-fens 1000 \
	--uci-depth 10 \
	--uci-threads 4 \
	--uci-hash-mb 512 \
	--log-every 1000
```

## 10. 验证

`scripts/build.bat` 与 `scripts/build.sh` 在发布可执行文件前运行 CTest。测试失败时构建停止，现场保留在 `build/.build-work/`。

### 10.1 Gadus

Gadus CTest 覆盖 `gadus_18_planes`、普通走法与特殊走法编码、棋规、Policy/Value 输出形状、有限数值、反向传播、checkpoint 往返、`closed` 搜索和 batched MCTS。

### 10.2 Melano

Melano CTest 覆盖 `melano_square_tokens`、普通走法与升变编码、棋规、Policy/Value/Advantage 输出形状、Advantage 范围、动作条件 latent successor、$K=2$ anchored latent MCTS 路径、有限数值、反向传播和 checkpoint 往返。
