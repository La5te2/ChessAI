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
	check.py
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

- `include/gadus/`、`src/gadus/`：Gadus 独立实现。`dataset` 负责 PGN、HDF5 与监督训练，`game` 负责状态、动作和棋规，`model` 负责 ResNet Policy/Value，`searcher` 负责 closed 与 MCTS，`match` 负责 arena，`evolution` 负责 FCPI。
- `include/melano/`、`src/melano/`：Melano 独立实现。文件职责与入口形式和 Gadus 对称，状态编码、动作编码、网络、搜索与 FCPI 方程均由 Melano 自身实现。
- `preprocess.cpp`、`train.cpp`、`search.cpp`、`arena.cpp`、`fcpi.cpp`、`uci.cpp`：每套架构的六个命令入口。
- `tests.cpp`：每套架构的状态编码、特殊走法、棋规、网络前反向、数值范围与 checkpoint 往返测试。
- `scripts/`：通用 UCI 工具、模型检查、图形界面与构建启动脚本。
- `api/`：仓库本地 C++ 依赖与安装脚本。
- `build/gadus/`、`build/melano/`：可直接运行的架构程序与运行 DLL。
- `data/`：PGN、HDF5、开局书、分析结果和运行数据。
- `models/`：LibTorch checkpoint 与 UCI 教师机。

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
python -m pip install -r requirements.txt
```

C++ 依赖安装到 `api/`：LibTorch、HDF5、zlib、nlohmann-json、chess-library 与 Ninja。`api/versions.env` 是 Windows 与 Linux 共用的依赖版本锁。安装脚本在 `nvidia-smi` 能查询 GPU compute capability 时选择 LibTorch CUDA `cu126`，其余环境选择 CPU 包，也可通过 `GADIDAE_TORCH_VARIANT=cpu|cu126` 指定。已有 LibTorch 安装可通过 `GADIDAE_TORCH_DIR` 指向其根目录。`setup.bat/.sh` 在安装结束时验证实际依赖，`build.bat/.sh` 在配置前再次验证，CMake 对 LibTorch、HDF5 与 zlib 使用精确版本约束。版本、variant 或 chess-library 校验和不匹配时立即停止。下载过程显示进度、速度和 ETA，成功后清理压缩包、源码和依赖构建目录。

Windows：

Windows LibTorch 使用 MSVC ABI。可安装 Microsoft C++ Build Tools 与 Windows SDK，`scripts/build.bat` 通过 `vswhere` 初始化 x64 编译环境，并由 CMake 生成 Ninja 构建文件。架构 library 使用 LibTorch/chess-library 预编译头，MSVC 使用 `/MP` 并行编译，第三方依赖作为 system headers 处理。

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
# cmake -S <根目录> -B <根目录>\.build-work -G Ninja
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

构建期间使用 `.build-work/`。成功后该目录被清理。编译或 CTest 失败时，构建现场改名为 `.crash/`，CTest 日志位于 `.crash/Testing/Temporary/LastTest.log`。下一次构建开始时清理旧 `.crash/`。

每个命令入口支持 `--help`：

```bash
build/gadus/search --help
build/melano/fcpi --help
```

### 3.1 模型检查

`scripts/check.py` 只读检查现行 LibTorch checkpoint，输出架构、网络头、channels、blocks、动作空间、参数规模、张量类型、内存规模、有限性、文件大小与 SHA-256：

```bash
python scripts/check.py --model models/gadus.pth
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
	--out models/gadus.pth \
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
	--log-every 100 \
	--seed 2026

build/gadus/search \
	--model models/gadus.pth \
	--fen "startpos" \
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
	--root-topn 8
```

- `preprocess --max-games` 控制读取对局上限，`--chunk-size` 控制 HDF5 扩展单元，`--compression-level` 控制 deflate 等级，`--log-every` 控制对局进度输出。
- `train` 每次创建新的 Gadus 模型。`--channels` 和 `--blocks` 决定结构，`--max-steps` 是本次训练步数上限，`--save-every` 控制原子 checkpoint 写入周期。
- `search --fen startpos` 使用标准初始局面，也可传入完整 FEN。

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
	--candidate models/candidate.pth \
	--baseline models/champion.pth \
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

`--games` 使用正偶数，使每个开局都能交换双方颜色完成配对。`--games-in-flight` 控制同时推进的对局数，candidate 与 baseline 各加载一次，轮到同一模型行棋的局面组成 inference batch。

从标准初始局面开始时，将开局书设为空：

```bash
build/gadus/arena \
	--candidate models/candidate.pth \
	--baseline models/gadus.pth \
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
	--opponent-reply-topk 4 \
	--opponent-reply-temperature 0.2 \
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

每轮 FCPI 使用 `current.pth` 进行模型自对战。Gadus 的行为分布先做温度变换，再混入均匀探索：

$$
\widetilde\mu(a\mid s)=
\frac{P(a\mid s)^{1/T_b}}{\sum_bP(b\mid s)^{1/T_b}}
$$

$$
\mu(a\mid s)=(1-\epsilon)\widetilde\mu(a\mid s)+
\frac{\epsilon}{|\mathcal A(s)|}
$$

终局取真实 side-to-move 结果，截断局面取冻结模型 Value 作为 bootstrap。代码从后向前计算 TD($\lambda$)：

$$
G_t=-\left[(1-\lambda_{TD})V(s_{t+1})+\lambda_{TD}G_{t+1}\right]
$$

负号对应行棋方在每个 ply 的切换。候选动作 $a$ 执行后，Gadus 取对手 Policy 的前 $K_r$ 个应手。设 $Q_b$ 是执行应手 $b$ 后恢复到根行棋方视角的 Value，则对手响应分布为：

$$
\rho(b\mid s,a)=\operatorname{softmax}_b\left(
\log(P(b\mid s_a)+\varepsilon)-\frac{Q_b}{T_r}
\right)
$$

二层反事实值为：

$$
q^{(2)}(s,a)=\sum_b\rho(b\mid s,a)Q_b
$$

$T_r$ 较小时接近对根行棋方不利的 minimax 应手，Policy 项保留对手真实会选择该应手的可能性。后续自适应展开沿 $\rho$ 最大的应手继续。候选动作由此得到不同深度的根视角估计 $q^{(1)},\ldots,q^{(D)}$，反事实深度混合为：

$$
Q_{cf}=(1-\lambda_{cf})\sum_{d=1}^{D-1}
\lambda_{cf}^{d-1}q^{(d)}+\lambda_{cf}^{D-1}q^{(D)}
$$

实际走出的动作再与轨迹回报混合：

$$
Q_{played}\leftarrow(1-w_r)Q_{cf}+w_rG_t
$$

未展开合法动作使用根 Value 作为基线。设 $Q_a$ 为最终动作估计，Policy target 为：

$$
z_a=\alpha\log(P_{old}(a\mid s)+\varepsilon)+
\frac{Q_a-V_{old}(s)}{T_\pi}
$$

$$
\pi_{target}(a\mid s)=\operatorname{softmax}(z)_a
$$

FCPI 训练损失为：

$$
L=w_P L_{CE}(\pi_{new},\pi_{target})+
w_V\operatorname{SmoothL1}(V_{new},G_t)+
w_{KL}D_{KL}(\pi_{new}\Vert P_{old})-
w_H H(\pi_{new})
$$

每轮依次执行 `current.pth` 自对战、局面采样、自适应多步反事实展开、TD($\lambda$) 目标构造、candidate 训练和 paired-game arena。每局先按编码状态去重，再按 `positions-per-game` 做均匀无放回采样。

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
	--dynamics-weight 0.25 \
	--imagined-value-weight 0.25 \
	--device cuda \
	--log-every 50

build/melano/search \
	--model models/melano.pth \
	--fen "startpos" \
	--device cuda \
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
	--candidate models/candidate.pth \
	--baseline models/melano.pth \
	--device cuda \
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
	--model models/melano.pth \
	--device cuda \
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
	--counterfactual-topk 8 \
	--opponent-reply-topk 4 \
	--opponent-reply-temperature 0.2 \
	--counterfactual-min-plies 2 \
	--counterfactual-max-plies 6 \
	--counterfactual-target-average-plies 4 \
	--counterfactual-lambda 0.85 \
	--td-lambda 0.85 \
	--behavior-temperature 0.85 \
	--behavior-advantage-weight 0.5 \
	--uniform-mix 0.02 \
	--policy-temperature 0.25 \
	--prior-power 1.0 \
	--successor-weight 0.75 \
	--played-return-weight 0.5 \
	--policy-weight 1.0 \
	--value-weight 1.0 \
	--dueling-q-weight 0.5 \
	--dynamics-weight 0.25 \
	--imagined-value-weight 0.25 \
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

Melano 自对战行为分布同时使用 Policy 与 Advantage：

$$
b_a=\frac{\log(P(a\mid s)+\varepsilon)+w_AA(s,a)}{T_b}
$$

$$
\widetilde\mu(a\mid s)=\operatorname{softmax}(b)_a
$$

$$
\mu(a\mid s)=(1-\epsilon)\widetilde\mu(a\mid s)+
\frac{\epsilon}{|\mathcal A(s)|}
$$

Melano 的反事实候选使用精确棋规生成走后局面。对候选动作 $a$ 的对手局面 $s_a$，先按 Melano 自身的 Policy 与 Advantage 选出 $R$ 个响应：

$$
B(r\mid s_a)=\log(P(r\mid s_a)+\varepsilon)+w_AA(s_a,r)
$$

每个响应 $r$ 都通过精确棋盘执行，再由冻结网络评价两步后的根方视角价值：

$$
q_r=V(s_{a,r})
$$

对手响应权重同时考虑 Melano 认为该响应的合理程度，以及它对根方造成的损失：

$$
\omega_r=
\frac{\exp\left(B(r\mid s_a)-q_r/T_{opp}\right)}
{\sum_{j=1}^{R}\exp\left(B(j\mid s_a)-q_j/T_{opp}\right)}
$$

$$
q^{(2)}(s,a)=\sum_{r=1}^{R}\omega_rq_r
$$

`--opponent-reply-topk` 控制 $R$，`--opponent-reply-temperature` 控制对低根方价值响应的集中程度。权重最大的响应作为后续反事实分支的连续状态。该响应层使用 Melano 的 $P+A$ 动作语义筛选可行响应，使用精确棋盘与冻结网络 Value 构造训练目标。

终局取真实 side-to-move 结果，截断局面取冻结模型 Value 作为 bootstrap。Melano 从后向前计算 TD($\lambda$)：

$$
G_t=-\left[(1-\lambda_{TD})V(s_{t+1})+\lambda_{TD}G_{t+1}\right]
$$

候选动作不同深度的根视角估计为 $q^{(1)},\ldots,q^{(D)}$，反事实深度混合为：

$$
Q_{cf}=(1-\lambda_{cf})\sum_{d=1}^{D-1}
\lambda_{cf}^{d-1}q^{(d)}+\lambda_{cf}^{D-1}q^{(D)}
$$

对每个候选动作，先把反事实后继估计与当前网络动作价值混合：

$$
Q^*(s,a)=(1-w_s)\operatorname{clip}(V(s)+A(s,a),-1,1)+w_sQ_{cf}(s,a)
$$

实际走出的动作再混入轨迹回报：

$$
Q^*(s,a_{played})\leftarrow(1-w_r)Q^*(s,a_{played})+w_rG_t
$$

改进后的 Value 与 Advantage targets 为：

$$
V^*(s)=\max_{a\in C}Q^*(s,a)
$$

$$
A^*(s,a)=\operatorname{clip}(Q^*(s,a)-V^*(s),-2,0)
$$

其余合法动作以当前 $V+A$ 为基线，并被上界约束到 $V^*(s)$。Policy target 为：

$$
z_a=\alpha\log(P_{old}(a\mid s)+\varepsilon)+
\frac{Q^*(s,a)-V_{old}(s)}{T_\pi}
$$

$$
\pi^*(a\mid s)=\operatorname{softmax}(z)_a
$$

FCPI 训练损失为：

$$
L=w_P L_{CE}+w_V\operatorname{SmoothL1}(V,V^*)+
w_Q\operatorname{SmoothL1}(\widehat Q,Q^*)+
w_D L_D+w_I\operatorname{SmoothL1}(-V(\widehat z'),Q^*)+
w_{KL}D_{KL}(\pi_{new}\Vert P_{old})-
w_H H(\pi_{new})
$$

其中每个 FCPI 候选动作都通过精确棋规生成 $s'$，并写入 `candidate_next_states`。$L_D$ 使用与监督训练相同的 latent cosine consistency。反事实价值可以沿精确棋盘展开 2 到 6 plies，动作条件 dynamics 仍只学习根候选动作的一步转移 $E(s)\rightarrow E(s')$，与 $K=2$ anchored latent MCTS 的运行时假设保持一致。

Melano 每轮依次执行自身 `current.pth` 自对战、局面采样、带对手响应的反事实展开、PVA 与 latent-dynamics 目标构造、candidate 训练和 paired-game arena。每次运行由程序生成 `fcpi_YYYYMMDD_HHMMSS_id`，创建对应的 `data/runs/<run-id>/` 与 `models/runs/<run-id>/`。其中 HDF5 schema、candidate 和 current checkpoint 均属于 Melano，candidate 达到 arena gate 后原子写入该 run 的 `current.pth`。`summary.json` 记录反事实分支数、平均深度、实际评价的对手响应数和 arena 结果。

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

Simulator 接收任意完整 UCI 命令。Settings 使用 `Engine`、`Budget`、`MCTS`、`Decisions` 四个标签页：

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

Stadium 让两个任意 UCI 引擎进行一盘可视化对局。双方拥有独立命令、UCI options、movetime 和 MultiPV 行数。Settings 使用 `White`、`Black`、`Match` 三个标签页。

```powershell
python scripts\stadium.py `
	--white-uci "python scripts\uci.py --arch gadus --model models\gadus.pth" `
	--black-uci "python scripts\uci.py --arch melano --model models\melano.pth" `
	--white-movetime-ms 3000 `
	--white-multipv 5 `
	--black-movetime-ms 3000 `
	--black-multipv 5 `
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

## 11. 验证

`scripts/build.bat` 与 `scripts/build.sh` 在发布可执行文件前运行 CTest。测试失败时，构建停止并将现场保存在 `.crash/`。

Gadus 测试覆盖 `gadus_18_planes`、普通走法与特殊走法编码、棋规、Policy/Value 输出形状、有限数值、反向传播和 checkpoint 往返。本地 Windows CPU 烟测覆盖单盘 PGN 生成 HDF5、一步监督训练、closed 搜索、batched MCTS、两盘 paired arena、PGN 输出以及一轮 FCPI 的采样、反事实展开、训练、arena gate 和 current 晋升。

Melano 测试覆盖 `melano_square_tokens`、普通走法与升变编码、棋规、Policy/Value/Advantage 输出形状、Advantage 范围、动作条件 latent successor、对手 $P+A$ 与精确 Value 响应聚合、$K=2$ anchored latent MCTS 路径、有限数值、反向传播和 checkpoint 往返。本地 Windows CPU 烟测覆盖单盘 PGN 生成含 `next_states` 与 `next_values` 的 HDF5、一步监督训练以及一轮 Melano FCPI 的候选后继训练与 arena gate。
