# Gadidae

Gadidae is a family of experimental chess engines.

This README explains installation, commands and user-facing interfaces. [Gadus.md](Gadus.md) and [Melano.md](Melano.md) specify their architectures, training methods and search algorithms.

## Dependencies

### Python

Install the Python dependencies used by the repository scripts with:

```bash
python -m pip install torch shapely svgelements
```

### C++

The Windows and Linux dependency installers place LibTorch, HDF5, zlib, nlohmann-json, chess-library and Ninja under `api/`. When graphics support is enabled, they additionally place GLFW, GLM, Dear ImGui, FreeType and GLAD under `api/`. Both installers read their pinned dependency versions from `api/versions.env`.

The installers select the LibTorch CUDA `cu126` package when `nvidia-smi` reports a GPU compute capability. They select the CPU package in other environments. Set `GADIDAE_TORCH_VARIANT=cpu` or `GADIDAE_TORCH_VARIANT=cu126` to select a package explicitly. Set `GADIDAE_TORCH_DIR` to use an existing LibTorch installation.

After installing the dependencies, `api/setup.bat` and `api/setup.sh` verify each installed package. The build scripts repeat this verification before configuring CMake. LibTorch, HDF5, zlib, nlohmann-json and Ninja must match the versions recorded in `api/versions.env`. Chess-library must match its recorded SHA-256 checksum. After verification succeeds, the installer removes downloaded archives, extracted sources and temporary dependency build directories.

Windows uses the MSVC-compatible LibTorch ABI. Install Microsoft C++ Build Tools and a Windows SDK, then run:

```powershell
$env:GADIDAE_TORCH_VARIANT = "cpu"
.\api\setup.bat
```

On Linux, install the base toolchain with:

```bash
sudo apt install build-essential cmake curl unzip tar python3 python3-pip
```

A graphics-enabled build also requires the X11 development packages:

```bash
sudo apt install xorg-dev
```

Install the repository dependencies with:

```bash
GADIDAE_TORCH_VARIANT=cu126 bash api/setup.sh
```

On Linux, `api/setup.sh` and `scripts/build.sh` inspect `DISPLAY` and `WAYLAND_DISPLAY` to select the default graphics mode. Set `GADIDAE_BUILD_GRAPHICS=0` to install and build only command-line targets. Set `GADIDAE_BUILD_GRAPHICS=1` to include the graphics dependencies and target.

## Build

Build on Windows with:

```powershell
.\scripts\build.bat
```

The Windows build script locates Visual Studio through `vswhere`, initializes an x64 compiler environment and invokes CMake with the Ninja generator.

Gadus, Melano, Eleginus and the graphical client can be enabled independently with `GADIDAE_BUILD_GADUS`, `GADIDAE_BUILD_MELANO`, `GADIDAE_BUILD_ELEGINUS` and `GADIDAE_BUILD_GRAPHICS`. Values `0` and `1` disable and enable a component.

Build on Linux with:

```bash
bash scripts/build.sh
```

A command-line-only Linux server selects `GADIDAE_BUILD_GRAPHICS=0` during installation and build:

```bash
GADIDAE_BUILD_GRAPHICS=0 bash api/setup.sh
GADIDAE_BUILD_GRAPHICS=0 bash scripts/build.sh
```

The same switches select any subset on Linux. Disabled components are not rebuilt, published, or removed from `build/`.

The build scripts use CMake and Ninja to produce the following command-line executables:

```text
build/gadus/preprocess
build/gadus/train
build/gadus/search
build/gadus/uci

build/melano/preprocess
build/melano/train
build/melano/search
build/melano/uci

build/eleginus/train
build/eleginus/search
build/eleginus/uci
```


A graphics-enabled build also produces `build/graphics/Gadidae`. Windows executables use the `.exe` suffix. The [Graphics](#graphics) section describes the graphical client, its operating modes and its piece-import pipeline.

The `build/.build-work/` directory stores the CMake and Ninja state used for incremental builds. Subsequent builds reuse compatible object files from this directory. A failed build leaves its diagnostic files in this directory.

The `preprocess`, `train` and `search` entry points provide their current argument lists through `--help`:

```bash
build/gadus/search --help
build/melano/train --help
build/eleginus/search --help
```

The Gadus, Melano and Eleginus UCI executables publish their configurable engine options in response to the UCI `uci` command rather than through `--help`.

## Commands

### Gadus

Preprocess a JSONL stream into the Gadus HDF5 schema with:

```bash
zstdcat data/positions.jsonl.zst | build/gadus/preprocess \
	--input - \
	--output data/positions.gadus.h5 \
	--chunk-size 16384 \
	--compression-level 1 \
	--max-positions 300000000 \
	--log-every 1000000
```

Each JSONL record contains a `fen` string and an `evals` array. The preprocessor selects the entry with the greatest `depth`, breaks equal-depth ties by `knodes` and uses its first principal variation. The first move in `line` supplies the Policy target, while the White-perspective `cp` or `mate` score supplies the side-to-move Value target. `--max-positions` limits accepted records. An uncompressed JSONL file may be passed directly through `--input`.

Train a new Gadus model with:

```bash
build/gadus/train \
	--data data/positions.gadus.h5 \
	--out models/gadus/gadus.pth \
	--channels 128 \
	--blocks 12 \
	--epochs 3 \
	--batch-size 512 \
	--max-steps 0 \
	--lr 0.001 \
	--weight-decay 0.0001 \
	--value-weight 0.5 \
	--save-every 5000 \
	--device cuda \
	--precision bf16 \
	--log-every 50 \
	--seed 2026
```

For supervised training, a positive `--max-steps` value caps optimizer updates. Setting it to `0` leaves the update count under the control of `--epochs`. `--save-every` controls periodic atomic checkpoint writes.

Gadus checkpoints contain learned parameters and normalization state. Deterministic displacement tables and compact action lookup tables are reconstructed when a model is created, so they do not occupy checkpoint storage.

Render the learned Gadus relation matrices from one trained checkpoint with:

```bash
python scripts/visual.py --model models/gadus/gadus.pth --source e4
```

`--source` selects the source square shown in every heatmap and defaults to `e4`. Rows in the generated contact sheet follow the model's relation blocks, columns follow their computational groups, and each panel uses its own color scale so that learned spatial structure remains visible.

Each run writes `data/<run-id>-<time>.zip`. The archive contains only `relations.png`; the script does not generate or modify model parameters.

Analyze one position with Gadus search using:

```bash
build/gadus/search \
	--model models/gadus/gadus.pth \
	--fen "startpos" \
	--device cuda \
	--precision bf16 \
	--mcts-sims 1000 \
	--mcts-batch-size 64 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--root-topn 8
```

`--fen` accepts a complete FEN or the value `startpos`, which selects the standard initial position.

Launch the Gadus UCI engine on Windows with:

```powershell
build\gadus\uci.exe --model models\gadus\gadus.pth
```

The corresponding Linux command is:

```bash
build/gadus/uci --model models/gadus/gadus.pth
```

The [UCI](#uci) section describes runtime options, output fields and time management.

### Melano

Preprocess a JSONL stream into the Melano HDF5 schema with:

```bash
zstdcat data/positions.jsonl.zst | build/melano/preprocess \
	--input - \
	--output data/positions.melano.h5 \
	--chunk-size 16384 \
	--compression-level 1 \
	--max-positions 100000000 \
	--log-every 1000000
```

Melano accepts the same JSONL record structure as Gadus. Its preprocessor applies the same evaluation and principal-variation selection rule, then encodes the resulting state and targets in the Melano HDF5 schema.

Train a new Melano model with:

```bash
build/melano/train \
	--data data/positions.melano.h5 \
	--out models/melano/melano.pth \
	--channels 128 \
	--blocks 16 \
	--epochs 2 \
	--batch-size 512 \
	--max-steps 0 \
	--lr 0.001 \
	--weight-decay 0.0001 \
	--value-weight 0.8 \
	--grad-clip 1.0 \
	--device cuda \
	--precision bf16 \
	--save-every 5000 \
	--log-every 50
```

For supervised training, a positive `--max-steps` value caps optimizer updates. Setting it to `0` leaves the update count under the control of `--epochs`. `--save-every` controls periodic atomic checkpoint writes.

Analyze one position with Melano search using:

```bash
build/melano/search \
	--model models/melano/melano.pth \
	--fen "startpos" \
	--device cuda \
	--precision bf16 \
	--mcts-sims 1000 \
	--mcts-min-sims 250 \
	--mcts-batch-size 64 \
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--root-topn 8
```

`--fen` accepts a complete FEN or the value `startpos`, which selects the standard initial position.

Launch the Melano UCI engine on Windows with:

```powershell
build\melano\uci.exe `
	--model models\melano\melano.pth
```

The corresponding Linux command is:

```bash
build/melano/uci \
	--model models/melano/melano.pth
```

The [UCI](#uci) section describes runtime options, output fields and time management.

### Eleginus

Export the initial Eleginus checkpoint manually from the repository root.

On Windows, run:

```powershell
build\eleginus\search.exe --export-initial models\eleginus\eleginus.pth
```

On Linux, run:

```bash
build/eleginus/search --export-initial models/eleginus/eleginus.pth
```

Both commands write the initial checkpoint directly to `models/eleginus/eleginus.pth`.

Run continuous self-play training from the repository root:

```powershell
build\eleginus\train.exe `
	--out models\eleginus\current.pth `
	--opening-book data\openings.gen.bin `
	--eval-every 1000 `
	--depth 2 `
	--eval-depth 4 `
	--workers 16
```

`--eval-every K` schedules an evaluation after every K completed training games. Training starts from the standard initial position and discards games that remain unfinished at `--max-plies` (320 by default). Discarded games do not count toward K.

Each evaluation freezes the candidate and plays 2000 games against the last accepted model. The Polyglot opening book must provide exactly 1000 distinct nonterminal leaf positions reachable from the standard initial position. Each position is played twice with the candidate's color reversed. Both models use `--eval-depth`, select their best move without training exploration, and play to a rule-defined terminal result.

The acceptance test uses a two-sided 95% Hoeffding interval for the mean score over the 1000 opening pairs and converts its endpoints to relative Elo. This interval treats opening pairs as independent samples, rather than treating both games within a pair as independent. A candidate replaces `--out` only when the lower endpoint exceeds zero Elo. The confidence level applies to one evaluation, not to the entire sequence of repeated evaluations.

An existing `--out` supplies the baseline and the starting training weights. `--init` overrides the starting training weights without replacing an existing baseline. When `--out` does not exist, the starting weights also supply the first baseline; no file is created until a candidate passes evaluation.

Ctrl+C stops training or evaluation without saving progress. The accepted checkpoint contains only graybox weights and is replaced atomically. Optimizer state and training samples remain in memory and are discarded on exit. Rejected candidates do not replace the checkpoint or reset the ongoing training process.

Analyze one position with Eleginus search using:

```bash
build/eleginus/search \
	--model models/eleginus/eleginus.pth \
	--fen "startpos" \
	--depth 10 \
	--hash 64 \
	--nodes 0 \
	--multipv 1
```

`--fen` accepts a complete FEN or the value `startpos`. A positive `--nodes` value stops the search at the requested node count, while `0` leaves the node count unbounded. `--multipv` selects the number of root lines to search and report.

Launch the Eleginus UCI engine on Windows with:

```powershell
build\eleginus\uci.exe --model models\eleginus\eleginus.pth
```

The corresponding Linux command is:

```bash
build/eleginus/uci --model models/eleginus/eleginus.pth
```

The [UCI](#uci) section describes runtime options, output fields and time management.

## UCI

Each architecture implements UCI time management within its own engine code. Gadus and Melano accept `go wtime`, `btime`, `winc`, `binc` and `movestogo`, and their allocation equations follow the optimum-time calculation in Stockfish's [Time Management](https://github.com/official-stockfish/Stockfish/blob/master/src/timeman.cpp). Let $t$ be the active side's remaining time in milliseconds, $i$ its increment, $o$ the `Move Overhead`, $p$ the number of plies played and $m$ the move horizon. An explicit `movestogo` sets $m$ up to a maximum of 50; otherwise $m=50$. When $t<1000$, the engines replace $m$ by $max(1,\lfloor0.05t\rfloor)$. They then compute

$$
T=\max\left(1,t+i(m-1)-o(2+m)\right).
$$

Without an explicit `movestogo`, the first clock-managed search after `ucinewgame` initializes

$$
r=0.3272\log_{10}T-0.4141,
\qquad
c=\min\left(0.0029869+0.00033554\log_{10}\frac{t}{1000},0.004905\right),
$$

and later moves in the game reuse $r$. The optimum-time scale is

$$
q=\min\left(0.012112+(p+3.22713)^{0.46866}c,\frac{0.19404t}{T}\right)r.
$$

With an explicit `movestogo`, the scale is

$$
q=\min\left(\frac{0.88+p/116.4}{m},\frac{0.88t}{T}\right).
$$

The resulting search budget is

$$
t_{\mathrm{search}}=
\min\left(\max(1,\lfloor qT\rfloor),\max(1,t-o)\right).
$$

`go movetime <milliseconds>` sets the internal budget to the requested duration minus `Move Overhead`, with a lower bound of one millisecond. `go infinite` disables the clock deadline. `Move Overhead` defaults to 10 milliseconds and reserves time for protocol communication and move submission.

### Gadus

The Gadus UCI executable loads its checkpoint when `isready` or `go` first requires neural evaluation. An explicit `--model` argument selects the checkpoint, while the default path is `gadus.pth` beside the executable. The process selects a CUDA device when one is available and otherwise uses CPU. It retains the loaded model across positions, evaluates it in FP32 and reloads it before the next search after `ModelPath` changes.

The UCI worker performs neural inference and tree search while the protocol loop remains available for `stop`. A `position`, `setoption` or `ucinewgame` command first stops and joins an active worker before changing engine state. Closing the process also joins the worker.

Gadus reports MultiPV rows containing `score cp`, nodes, NPS, elapsed time and a one-move principal variation. With `Sims=0`, the reported root evaluation is the network Value $V_\theta(s)$. After at least one simulation, it is the mean return backed up to the root. A visited root edge reports its backed-up $Q(s,a)$, while an unvisited edge reports the root evaluation. Let $q_{\mathrm{line}}\in[-1,1]$ denote the Value reported for one row. Gadus converts this bounded quantity to the centipawn display scale by

$$
\text{score cp}=\mathrm{round}\left(
90\tan\left(1.5637541897q_{\mathrm{line}}\right)
\right).
$$

The score is expressed from the root side-to-move perspective. The `nodes` field equals the number $n$ of completed simulations, including zero for direct Policy inference. Both `depth` and `seldepth` equal

$$
\max\left(1,\left\lfloor\log_2\max(1,n)\right\rfloor+1\right).
$$

These depth fields summarize simulation effort rather than maximum tree depth. NPS is $1000n/t$, where $t$ is elapsed time in milliseconds with a denominator of at least one millisecond. Gadus emits the initial root result, updates at fixed 300-millisecond intervals and one final result before `bestmove`.

`Sims` supplies the default simulation cap and the reference budget used by root allocation. `go nodes N` gives the current command a cap and reference budget of $N$, taking priority over `Sims`, `depth` and `infinite`; `N=0` selects direct Policy inference. In the absence of `nodes`, `go depth d` uses $2^{d-1}$ simulations for $d\geq1$. A clock or `movetime` deadline may stop a bounded search before it reaches its simulation cap. When neither `nodes` nor `depth` is present, `go infinite` removes a positive `Sims` cap and continues until the client sends `stop`.

Gadus exposes these options:

- `ModelPath` selects the checkpoint.
- `Threads` controls LibTorch CPU threads and defaults to `2`.
- `Hash` sets the evaluation-cache capacity in MiB and defaults to `256`.
- `Sims` sets the default simulation cap and defaults to `100`.
- `Move Overhead` reserves time for communication and move submission and defaults to `10`.
- `MultiPV` sets the number of reported root lines and defaults to `5`. It changes the report width while search allocation and move selection remain fixed.

The UCI adapter uses a neural batch capacity of 32, $c_{\mathrm{puct}}=1$, a PUCT base of 19652, a PUCT factor of 1, a First Play Urgency reduction of 0.15, zero virtual loss, a Repetition Policy Penalty coefficient of 1 and Instant Mate First. The standalone Gadus search command retains command-line controls for these quantities.

A positive `Hash` capacity retains compact Policy and Value evaluations across successive `go` commands. Gadus TLRU (trajectory-aware least-recently-used) records normalized tree-visit heat after each search, then restricts the retained trajectory region to the roots requested by the next `go` command before refreshing its eviction order. Cross-command reuse applies to neural-evaluation records, whereas each `go` command constructs a new MCTS tree and new tree statistics. Setting `Hash` to zero creates a call-local cache that deduplicates neural evaluations during one `go` command and releases its records when the command completes. `ucinewgame` and `ModelPath` changes clear retained cross-search evaluations.

A UCI client may configure Gadus with commands such as:

```text
setoption name Threads value 2
setoption name Hash value 256
setoption name Sims value 1000
setoption name Move Overhead value 10
setoption name MultiPV value 5
```

### Melano

The Melano UCI executable loads its checkpoint when `isready` or `go` first requires neural evaluation. An explicit `--model` argument selects the checkpoint, while the default path is `melano.pth` beside the executable. The process selects a CUDA device when one is available and otherwise uses CPU. It retains the loaded model across positions, evaluates it in FP32 and reloads it before the next search after `ModelPath` changes.

Search runs on a worker thread so the protocol loop can process `stop`. A `position`, `setoption` or `ucinewgame` command first stops and joins an active search before changing engine state. Closing the UCI process also joins the worker.

Melano reports MultiPV rows containing `score cp`, nodes, NPS, elapsed time and a one-move principal variation. With `Sims=0`, the reported root evaluation is the network Value $V_\theta(s)$. After at least one simulation, it is the mean return backed up to the root. A visited root edge reports its backed-up $Q(s,a)$, while an unvisited edge reports the root evaluation. The displayed score uses the fixed scale $c_s=1000$:

$$
\text{score cp}=\mathrm{round}\left(
c_s\,\mathrm{clip}(q_{\mathrm{line}},-0.999,0.999)
\right).
$$

The score is expressed from the root side-to-move perspective. The `nodes` field equals the number $n$ of completed simulations, including zero for direct Policy inference. Both `depth` and `seldepth` equal

$$
\max\left(1,\left\lfloor\log_2\max(1,n)\right\rfloor+1\right).
$$

These depth fields summarize search effort rather than maximum tree depth. NPS is $1000n/t$, where $t$ is elapsed time in milliseconds with a denominator of at least one millisecond. Melano emits the initial root result, updates at fixed 300-millisecond intervals and one final result before `bestmove`.

`Sims` supplies the default upper simulation cap and the reference budget used by Melano's uncertainty-controlled stopping rule. `go nodes N` gives the current command an upper cap and reference budget of $N$, taking priority over `Sims`, `depth` and `infinite`; `N=0` selects direct Policy inference. In the absence of `nodes`, `go depth d` uses $2^{d-1}$ simulations for $d\geq1$. A bounded search may finish below its upper cap when its dynamic target is reached, and a clock or `movetime` deadline may stop it earlier. When neither `nodes` nor `depth` is present, `go infinite` removes a positive `Sims` cap and continues until the client sends `stop`.

Melano exposes these options:

- `ModelPath` selects the checkpoint.
- `Threads` controls LibTorch CPU threads and defaults to `2`.
- `Hash` sets the evaluation-cache capacity in MiB and defaults to `256`.
- `Sims` sets the default upper simulation cap and defaults to `100`.
- `Move Overhead` reserves time for communication and move submission and defaults to `10`.
- `MultiPV` sets the number of reported root lines and defaults to `5`.

The UCI adapter uses a neural batch capacity of 32 and leaves Melano's automatic minimum-simulation rule active. It uses $c_{\mathrm{puct}}=1$, a PUCT base of 19652, a PUCT factor of 1, a First Play Urgency reduction of 0.15, zero virtual loss, a Repetition Policy Penalty coefficient of 1 and Instant Mate First. The standalone Melano search command retains command-line controls for these quantities.

A positive `Hash` capacity retains compact Policy and Value evaluations across successive `go` commands. Melano TLRU records normalized tree-visit heat after each search, then restricts the retained trajectory region to the roots requested by the next `go` command before refreshing its eviction order. The cache accounts for evaluation records, trajectory metadata and recorded parent-child links within the same `Hash` capacity. Cross-command reuse applies to neural-evaluation records, whereas each `go` command constructs a new MCTS tree and new tree statistics. Setting `Hash` to zero creates a call-local cache that deduplicates neural evaluations during one `go` command and releases its records when the command completes. `ucinewgame` and `ModelPath` changes clear retained cross-search evaluations.

A UCI client may configure Melano with commands such as:

```text
setoption name Threads value 2
setoption name Hash value 256
setoption name Sims value 1000
setoption name Move Overhead value 10
setoption name MultiPV value 5
```

### Eleginus

The Eleginus UCI executable loads its checkpoint when `isready` or `go` first requires evaluation. An explicit `--model` argument selects the checkpoint, while the default path is `eleginus.pth` beside the executable. The process retains the loaded model across positions and reloads it before the next search after `ModelPath` changes.

Search runs on a worker thread so the protocol loop can process `stop`. A `position`, `setoption` or `ucinewgame` command first stops and joins an active search before changing engine state. Closing the UCI process also joins the worker.

Eleginus reports MultiPV rows containing the completed iterative depth, the greatest visited ply as `seldepth`, the root-side score, visited nodes, NPS, elapsed time and a one-move principal variation. The static evaluator maps its dimensionless output $H(s)$ to centipawns by

$$
\operatorname{cp}(s)=
\operatorname{round}\left(\operatorname{clip}\left(400H(s),-25000,25000\right)\right).
$$

Search may replace this finite score with a mate score. NPS is $1000n/t$, where $n$ is the number of visited principal and quiescence nodes and $t$ is elapsed time in milliseconds with a denominator of at least one millisecond.

`go depth d` sets the maximum iterative depth. `go nodes N` sets a node limit, and `go movetime M` sets a deadline of $M-o$ milliseconds with a lower bound of one millisecond. In a clock-managed search, Eleginus uses

$$
t_{\mathrm{search}}=
\operatorname{clamp}\left(
\left\lfloor\frac{t}{30}\right\rfloor+
\left\lfloor\frac{i}{2}\right\rfloor-o,
1,
\max(1,t-o)
\right).
$$

When a command supplies more than one applicable limit, the search stops at the first reached limit. `go infinite` removes the clock deadline and leaves the search bounded by its maximum supported iterative depth or a subsequent `stop` command.

Eleginus exposes these options:

- `ModelPath` selects the checkpoint.
- `Hash` sets the combined cache budget in MiB and defaults to `64`. The engine rounds a requested value down to the nearest power of two. A value of `0` disables both caches, a value of `1` assigns one MiB to the transposition table, and every larger effective budget is divided equally between the transposition table and the static-evaluation cache.
- `Move Overhead` reserves time for communication and move submission and defaults to `10`.
- `MultiPV` sets the number of searched and reported root lines and defaults to `1`.

Both caches belong to one `go` command. The transposition table stores depth-qualified exact, lower and upper search bounds, while the static-evaluation cache reuses evaluations of repeated positions. `MultiPV` requires additional root searches or exhaustive root scoring, so increasing it increases search work.

A UCI client may configure Eleginus with commands such as:

```text
setoption name ModelPath value models/eleginus/eleginus.pth
setoption name Hash value 256
setoption name Move Overhead value 10
setoption name MultiPV value 4
```

## Scripts

The `scripts/` directory contains the Windows and Linux build launchers, checkpoint inspection and graphical piece preparation.

### Checkpoint Inspection

`scripts/check.py` performs a read-only inspection of a Gadus, Melano or Eleginus checkpoint. Neural checkpoint reports include heads, architecture dimensions, parameter counts, tensor data types, tensor memory, devices and finite-value status. Eleginus reports include its formula and active-relation counts. Every report includes the detected architecture, file size and SHA-256 digest.

```bash
python scripts/check.py models/gadus/gadus.pth
python scripts/check.py models/melano/melano.pth
python scripts/check.py models/eleginus/eleginus.pth
```

## Graphics

Gadidae provides a native graphical client for engines that implement the Universal Chess Interface (UCI). The client uses GLFW, OpenGL 3.3, GLAD, Dear ImGui and FreeType for its graphical interface. The UCI boundary allows the client to work with engines independently of their internal architecture.

A graphics-enabled build produces `build/graphics/Gadidae.exe` on Windows and `build/graphics/Gadidae` on Linux. Interactive use over Secure Shell (SSH) requires X11 forwarding, a remote desktop or another display service.

### Engine Interface

Importing an engine starts a temporary process and performs a standard UCI handshake. The client reads every `option name ...` declaration returned by the engine, generates a matching control and then closes the temporary process. The imported configuration enters a reusable registry stored in `gui.json`. Importing the same executable path updates its existing entry. UCI options configure a working engine after its handshake, whereas `Launch arguments` supplies arguments when the operating system starts an engine process.

Simulator starts a working engine when analysis opens, and Stadium starts the configured working engines when a match begins. Each working process remains alive for its analysis or match session. A position change sends `stop` to the active search, discards subsequent output associated with the previous position and submits the new position to the same process. The UCI `position` command reconstructs the game from its initial FEN and the complete move sequence up to the position being displayed, preserving the rule history required for repetition detection. The engine determines how quickly an active search responds to `stop`.

### Simulator

Simulator provides interactive position analysis. `Run > Open` starts live analysis, and `Run > Close` stops it. The following command opens Simulator with a Gadus engine and preconfigures its analysis controls:

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

Simulator accepts any executable that implements the UCI protocol. The `--uci` argument therefore may refer to Gadus, Melano, Stockfish or another UCI engine.

`File` imports engines and PGN files and saves the current PGN. `Board` sets the starting FEN, resets the position, undoes a Simulator move and flips the board. The board-state panel provides commands that copy the current FEN or PGN to the clipboard. The slider below the board selects a read-only historical position without changing the live game.

### Stadium

Stadium manages multiple independent games. `Tools > Matches` creates games, selects the game shown in the active view and closes games. Games outside the active view continue in the background, including while Simulator is visible. Each seat requires a participant name. A Human seat accepts moves from the board. An engine seat requires a UCI executable and stores its own UCI option values and launch arguments.

Match settings define the initial clock, increment, display delay, maximum ply count and starting position in Forsyth-Edwards Notation (FEN). An initial clock of zero selects untimed play. `Run > Start`, `Run > Pause` and `Run > Stop` control the active game. Closing Gadidae terminates every UCI subprocess owned by every open match.

The following command opens Stadium with a Gadus-versus-Stockfish game:

```powershell
build\graphics\Gadidae.exe `
	--mode stadium `
	--white-uci "models\gadus\gadus.exe" `
	--white-name "Gadus" `
	--black-uci "models\stockfish\stockfish.exe" `
	--black-name "Stockfish"
```

Replacing the White executable and name with `models\melano\melano.exe` and `Melano` creates a Melano-versus-Stockfish game.

### Appearance

Appearance settings include dark and light application themes, base font size, bounded font scaling, board-color presets, custom colors, coordinate labels and embedded piece styles. Applying a setting writes `gui.json` beside the Gadidae executable, and subsequent launches restore that configuration. Command-line overrides include `--font-size <px>`, `--theme dark`, `--theme light` and `--piece-style <name>`.

### Piece Import

`scripts/pieces.py` converts one Scalable Vector Graphics (SVG) set into pre-triangulated indexed meshes stored in `src/graphics/pieces.gpack`. The input directory must contain `wK.svg`, `wQ.svg`, `wR.svg`, `wB.svg`, `wN.svg`, `wP.svg`, `bK.svg`, `bQ.svg`, `bR.svg`, `bB.svg`, `bN.svg` and `bP.svg`. A valid style name begins with a lowercase letter and contains only lowercase letters, digits or hyphens.

Import a style and rebuild the client on Windows with:

```powershell
python -m pip install shapely svgelements
python scripts\pieces.py `
	--input data\pieces\my-style `
	--name my-style
scripts\build.bat
```

Import and rebuild on Linux with:

```bash
python -m pip install shapely svgelements
python scripts/pieces.py \
	--input data/pieces/my-style \
	--name my-style
GADIDAE_BUILD_GRAPHICS=1 bash scripts/build.sh
```

Importing a new style name adds that style to the embedded archive, whereas reimporting an existing name atomically replaces the stored style. The `--curve-step` argument controls the curve-sampling distance and defaults to `1.5`. Smaller values produce smoother, denser meshes and larger archives, whereas larger values reduce archive size and import time.

Every imported third-party style must be declared in the following subsection. A declaration identifies its author when known, permanent source, license and license text. A combined style must identify every source it uses.

### Third-Party Pieces

`src/graphics/piece.inc` contains the fixed built-in geometry, while `src/graphics/pieces.gpack` contains deduplicated and compressed indexed geometry for imported styles. Both files are generated from the SVG sources listed below. The source SVG files are one-time inputs and are not retained by the project. RhosGFX, Chessnut, Spatial and Fantasy use the Lichess repository at commit `c54596c9e1569d378fe20f80d4c790c6fdee54c4`.

#### RhosGFX

- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/rhosgfx
- License: CC0 1.0
- License text: https://creativecommons.org/publicdomain/zero/1.0/legalcode

#### Chessnut

- Author: Alexis Luengas
- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/chessnut
- License: Apache License 2.0
- License text: https://www.apache.org/licenses/LICENSE-2.0

#### Spatial

- Author: Maurizio Monge
- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/spatial
- License: MIT
- License text: https://opensource.org/license/mit

#### Fantasy

- Author: Maurizio Monge
- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/fantasy
- License: MIT
- License text: https://opensource.org/license/mit

#### Cburnett

- Author: Colin M.L. Burnett
- Source: https://commons.wikimedia.org/wiki/Template:SVG_chess_pieces
- License: Creative Commons Attribution-ShareAlike 3.0 Unported
- License text: https://creativecommons.org/licenses/by-sa/3.0/legalcode

The generated geometry is embedded in Gadidae. The application does not load the source SVG files at runtime.
