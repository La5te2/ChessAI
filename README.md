# Gadidae

Gadidae is a family of experimental chess engines.

This README and [Graphics.md](Graphics.md) explain installation, commands and user-facing interfaces. [Gadus.md](Gadus.md), [Melano.md](Melano.md) and [Eleginus.md](Eleginus.md) specify the corresponding architectures, training methods and search algorithms.

## Dependencies

### Python

Install the Python dependencies used by the repository scripts with:

```bash
python -m pip install -r scripts/requirements.txt
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

Each component can be enabled independently with `GADIDAE_BUILD_GADUS`,
`GADIDAE_BUILD_MELANO`, `GADIDAE_BUILD_ELEGINUS`, and `GADIDAE_BUILD_GRAPHICS`. Values `0` and `1`
disable and enable a component. Eleginus additionally provides
`GADIDAE_BUILD_ELEGINUS_TRAINING`; setting it to `0` builds only its statically linked, Torch-free
`search` and `uci` programs. For example, an Eleginus inference-only Windows build is:

```powershell
$env:GADIDAE_BUILD_GADUS = "0"
$env:GADIDAE_BUILD_MELANO = "0"
$env:GADIDAE_BUILD_ELEGINUS = "1"
$env:GADIDAE_BUILD_ELEGINUS_TRAINING = "0"
$env:GADIDAE_BUILD_GRAPHICS = "0"
.\scripts\build.bat
```

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
build/gadus/arena
build/gadus/fcpi
build/gadus/uci

build/melano/preprocess
build/melano/train
build/melano/search
build/melano/uci

build/eleginus/preprocess
build/eleginus/train
build/eleginus/embed
build/eleginus/search
build/eleginus/uci
build/eleginus/tests
```

The Eleginus `preprocess`, `train`, `embed` and `tests` programs are omitted when
`GADIDAE_BUILD_ELEGINUS_TRAINING=0`; `search` and `uci` remain available and do not depend on
LibTorch at runtime.

A graphics-enabled build also produces `build/graphics/Gadidae`. Windows executables use the `.exe` suffix. [Graphics.md](Graphics.md) describes the graphical client, its operating modes and its piece-import pipeline.

The `build/.build-work/` directory stores the CMake and Ninja state used for incremental builds. Subsequent builds reuse compatible object files from this directory. A failed build leaves its diagnostic files in this directory.

The `preprocess`, `train`, `embed`, `search`, `arena` and Gadus `fcpi` entry points provide their current argument lists through `--help`:

```bash
build/gadus/search --help
build/melano/train --help
```

The Gadus, Melano and Eleginus UCI executables publish their configurable engine options in response to the UCI `uci` command rather than through `--help`.

## Commands

### Gadus

Preprocess an annotated PGN into the Gadus HDF5 schema with:

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

`--has-cmt 1` derives Value targets from numerical PGN comments, while `--has-cmt 0` derives them from final game results. `--max-games` limits the number of PGN games read. `--chunk-size` controls HDF5 dataset extension units. `--compression-level` selects the deflate level. `--log-every` controls progress reporting by game count.

Train a new Gadus model with:

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

For supervised training, a positive `--max-steps` value caps optimizer updates. Setting it to `0` leaves the update count under the control of `--epochs`. `--save-every` controls periodic atomic checkpoint writes.

Analyze one position with Gadus search using:

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

Run a paired Gadus arena with:

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
	--c-puct 0.5 \
	--c-puct-base 19652 \
	--c-puct-factor 1.0 \
	--fpu-reduction 0.15 \
	--virtual-loss 0.0 \
	--repetition-policy-penalty 0.0 \
	--instant-mate-first 0 \
	--pgn-output data/gadus-arena.pgn \
	--log-every 1
```

Passing an empty value to `--opening-book` starts every game from the standard initial position.

Launch the Gadus UCI engine on Windows with:

```powershell
build\gadus\uci.exe `
	--model models\gadus\gadus.pth `
	--device cpu `
	--threads 2 `
	--eval-cache-mb 256 `
	--search-type only-mcts `
	--mcts-sims 100
```

The corresponding Linux command is:

```bash
build/gadus/uci \
	--model models/gadus/gadus.pth \
	--device cpu \
	--threads 2 \
	--eval-cache-mb 256 \
	--search-type only-mcts \
	--mcts-sims 100
```

The [UCI](#uci) section describes runtime options, output fields and time management.

### Melano

Preprocess an annotated PGN into the Melano HDF5 schema with:

```bash
build/melano/preprocess \
	--input data/ccrl.pgn \
	--output data/games.melano.h5 \
	--has-cmt 1 \
	--chunk-size 4096 \
	--compression-level 1 \
	--log-every 10000 \
	--max-games 1000000
```

`--has-cmt 1` derives Value targets from numerical PGN comments, while `--has-cmt 0` derives them from final game results. `--max-games` limits the number of PGN games read. `--chunk-size` controls HDF5 dataset extension units. `--compression-level` selects the deflate level. `--log-every` controls progress reporting by game count.

Train a new Melano model with:

```bash
build/melano/train \
	--data data/games.melano.h5 \
	--out models/melano/melano.pth \
	--channels 128 \
	--blocks 20 \
	--epochs 3 \
	--batch-size 256 \
	--max-steps 500000 \
	--lr 0.0002 \
	--weight-decay 0.0001 \
	--value-weight 1.0 \
	--grad-clip 1.0 \
	--device cuda \
	--precision bf16 \
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
	--search-type only-mcts \
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
	--model models\melano\melano.pth `
	--device cuda `
	--search-type only-mcts `
	--mcts-sims 1000
```

The corresponding Linux command is:

```bash
build/melano/uci \
	--model models/melano/melano.pth \
	--device cuda \
	--search-type only-mcts \
	--mcts-sims 1000
```

The [UCI](#uci) section describes runtime options, output fields and time management.

### Eleginus

Preprocess a PGN into the architecture-locked Eleginus Policy/Value schema with:

```bash
build/eleginus/preprocess \
	--input data/ccrl.pgn \
	--output data/games.eleginus.h5 \
	--has-cmt 1 \
	--compression-level 4 \
	--max-games 100000
```

The HDF5 file uses the same `states`, `moves` and `values` dataset names as Gadus and Melano, but
its root metadata identifies `arch_type=eleginus` together with Eleginus-specific state, move and
target encodings. Eleginus moves use side-to-move-relative coordinates, and the reader accepts only
`target_schema=policy_value_perspective_resolved`. Generate the HDF5 file with the Eleginus preprocessor
before training.
With `--has-cmt 1`, the preprocessor determines whether each game's numerical pawn evaluations use
White or mover perspective, converts them to side-to-move targets and maps them into `[0,1]`. With
`--has-cmt 0`, completed game results supply targets in `{0, 0.5, 1}`. A game with an unparseable move
is rejected as an `invalid_game`, while a game lacking the target required by the selected mode is
counted as a `missing_target_game`.

Train independent Eleginus Policy and Value networks with:

```bash
build/eleginus/train \
	--data data/games.eleginus.h5 \
	--out models/eleginus/eleginus.pth \
	--epochs 2 \
	--batch-size 512 \
	--lr 0.001 \
	--device cuda \
	--seed 2026
```

Eleginus uses the same `--max-steps` convention: a positive value caps optimizer updates, while `0` lets `--epochs` determine the training length. The trainer atomically updates the checkpoint selected by `--out` after every epoch.

The Policy network learns the recorded PGN moves, while the Value network learns the selected
comment- or result-derived targets. Their parameter sets and AdamW states are independent. Supplying
`--model existing.pth` initializes both networks from that checkpoint, while omitting it initializes
a new model. Every training invocation constructs fresh optimizer state.

The `.pth` checkpoint is the only standalone Eleginus weight file. Like the other architecture
checkpoints, its top level contains `model` and `arch`; the latter identifies Eleginus with
`type_id=3` and records every fixed network dimension. Inspect it with:

```bash
python scripts/check.py --model models/eleginus/eleginus.pth
```

Search and UCI are built first as weightless, Torch-free executable templates. Select the standalone search template while embedding the checkpoint, then analyze one position:

```bash
build/eleginus/embed \
	--model models/eleginus/eleginus.pth \
	--type search \
	--output models/eleginus/eleginus

models/eleginus/eleginus \
	--fen startpos \
	--depth 4
```

`--fen` accepts `startpos` or one quoted six-field FEN. The embedded search program uses statically
linked float32 Policy and Value evaluators. Policy orders legal actions, while Value supplies the
static scores used by iterative deepening, PVS and quiescence search. The resulting executable contains its model parameters and all inference code required for search.

Create and launch the Eleginus UCI engine on Windows with:

```powershell
build\eleginus\embed.exe `
	--model models\eleginus\eleginus.pth `
	--type uci `
	--output models\eleginus\eleginus.exe

models\eleginus\eleginus.exe --depth 4
```

The corresponding Linux command is:

```bash
build/eleginus/embed \
	--model models/eleginus/eleginus.pth \
	--type uci \
	--output models/eleginus/eleginus

models/eleginus/eleginus --depth 4
```

The final Eleginus UCI and standalone search executables are self-contained. The `train` and `embed` tools use LibTorch during model production.

## UCI

Each architecture implements UCI time management within its own engine code. The three engines accept `go wtime`, `btime`, `winc`, `binc` and `movestogo`, and their allocation equations follow the optimum-time calculation in Stockfish's [Time Management](https://github.com/official-stockfish/Stockfish/blob/master/src/timeman.cpp). Let $t$ be the active side's remaining time in milliseconds, $i$ its increment, $o$ the `Move Overhead`, $p$ the number of plies played and $m$ the move horizon. An explicit `movestogo` sets $m$ up to a maximum of 50; otherwise $m=50$. When $t<1000$, the engines replace $m$ by $max(1,\lfloor0.05t\rfloor)$. They then compute

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

### Gadus and Melano

The Gadus and Melano UCI executables load their checkpoints when `isready` or `go` first requires neural evaluation. An explicit `--model` argument selects the checkpoint. The default path is `gadus.pth` beside the Gadus executable or `melano.pth` beside the Melano executable. Each process retains the loaded model across positions and evaluates it in FP32. Changing `ModelPath` or `Device` reloads the model before the next search.

Search runs on a worker thread so the protocol loop can process `stop`. A `position`, `setoption` or `ucinewgame` command first stops and joins an active search before changing engine state. Closing the UCI process also joins the worker.

Both engines report MultiPV rows containing `score cp`, nodes, NPS, elapsed time and a one-move principal variation. In `closed` mode, the reported root evaluation is the network Value $V_\theta(s)$. After at least one MCTS simulation, it is the mean return backed up to the root. A visited root edge reports its backed-up $Q(s,a)$, while an unvisited edge reports the root evaluation. For `ScoreScale` value $c_s$, the displayed value is

$$
\text{score cp}=\mathrm{round}\left(
c_s\,\mathrm{clip}(q_{\mathrm{line}},-0.999,0.999)
\right).
$$

The score is expressed from the root side-to-move perspective. The `nodes` field reports completed simulations and represents the initial neural evaluation as one node until the first simulation completes. For reported node count $n$, both `depth` and `seldepth` equal

$$
\max\left(1,\left\lfloor\log_2\max(1,n)\right\rfloor+1\right).
$$

These depth fields summarize search effort rather than maximum tree depth. NPS is $1000n/t$, where $t$ is elapsed time in milliseconds with a denominator of at least one millisecond. The engines emit the initial root result, periodic MCTS updates and one final result before `bestmove`. `ProgressIntervalMS=0` suppresses periodic updates.

The MCTS deadline uses the time allocation defined above. `go nodes <n>` overrides the current simulation cap, and `stop` requests early termination.

Gadus and Melano expose these shared options:

- `ModelPath` selects the checkpoint.
- `Device` selects `auto`, `cpu` or `cuda` and defaults to `auto`.
- `SearchType` selects `closed` or `only-mcts` and defaults to `only-mcts`.
- `MCTSSims` sets the simulation cap and defaults to `100`.
- `MCTSMinSims` sets the nominal simulation floor and defaults to `0`, which activates the dynamic floor described in each architecture's search specification.
- `MCTSBatchSize` sets the neural leaf-batch capacity and defaults to `32`.
- `Move Overhead` reserves time for communication and move submission and defaults to `10`.
- `CPuct`, `CPuctBase` and `CPuctFactor` configure the visit-dependent exploration coefficient and default to `0.5`, `19652` and `1.0`.
- `FPUReduction` sets the First Play Urgency reduction and defaults to `0.15`.
- `VirtualLoss` sets the repeated-path penalty used during batched selection and defaults to `0.0`.
- `RepetitionPolicyPenalty` sets the RPP coefficient in $[0,1]$ and defaults to `0.0`.
- `InstantMateFirst` enables IMF and defaults to `false`.
- `ProgressIntervalMS` sets the periodic report interval and defaults to `750` milliseconds.
- `MultiPV` sets the number of reported root lines and defaults to `5`.
- `ScoreScale` sets $c_s$ in the displayed-score equation and defaults to `1000`.

Gadus and Melano expose `Threads`, which controls LibTorch CPU threads and defaults to `2`, and `EvalCacheMB`, which defaults to `256`. A positive `EvalCacheMB` retains compact Policy and Value evaluations across successive `go` commands in TLRU (trajectory-aware least-recently-used). TLRU records evaluated parent-child transitions, and the root of each search call promotes its retained descendants within two recorded plies before capacity-based eviction resumes. Every search call creates a separate MCTS tree. Setting the option to zero selects a per-search cache that removes duplicate evaluations within the call and discards its records when the call returns. `ucinewgame`, `ModelPath` changes and `Device` changes clear retained cross-search evaluations.

A UCI client may configure the engines with commands such as:

```text
setoption name Threads value 2
setoption name EvalCacheMB value 256
setoption name SearchType value only-mcts
setoption name MCTSSims value 1000
setoption name MCTSBatchSize value 64
setoption name CPuct value 0.5
setoption name RepetitionPolicyPenalty value 1.0
setoption name InstantMateFirst value true
```

All commands above apply to Gadus and Melano.

### Eleginus

An embedded Eleginus UCI executable loads its independent Policy and Value parameters when `isready` or `go` first requires them. `Depth` sets the default iterative-deepening limit, `Hash` sets the transposition-table capacity, `Threads` sets the number of root PVS workers and `MultiPV` sets the number of reported principal variations. Their defaults are 4 plies, 64 MiB, one worker and five variations. `Move Overhead` uses the common UCI time-allocation rule above.

Each `go` command starts iterative deepening on a worker thread, allowing the UCI loop to process `stop`, a replacement position or `quit` while PVS is running. `go depth <n>` overrides `Depth` for one search, `go nodes <n>` sets a cumulative node limit and the standard clock fields activate the deadline described above. Policy orders legal actions, and PVS uses the Value network to establish static centipawn scores at quiescent leaves. Exact terminal scores use the range near $\pm30000$, while ongoing static scores use $150v_{\theta_V}(s)$. The reported `score cp` is expressed from the root side-to-move perspective.

With `MultiPV=1`, the first root action establishes the principal-variation bound and the remaining actions use null-window probes followed by full-window re-search when they exceed that bound. With `MultiPV=k>1`, every root action receives a full-window score so the engine can identify and report the best $k$ variations. Root actions are distributed among at most `Threads` workers, and the configured `Hash` capacity is divided among their transposition tables.

The engine emits one numbered `info` row for each requested variation after every completed depth. `depth` reports the completed principal depth, `seldepth` reports the greatest ply reached by PVS or quiescence search and `nodes` includes visits made by all root workers during every completed iteration. Cancellation discards an incomplete iteration and returns the selected action from the latest completed depth.

```text
setoption name Depth value 6
setoption name Hash value 256
setoption name Threads value 2
setoption name MultiPV value 3
setoption name Move Overhead value 10
```

## Opening Books

Gadus Arena reads Polyglot books by traversing legal book moves breadth-first from the standard initial position. The effective depth is at least one ply. The reader treats each outgoing legal book move uniformly, randomizes their order from the selected seed and emits unique nonterminal frontier positions. Position identity is the tuple of piece placement, side to move, castling rights and the en-passant field.

An empty arena book path starts every game from the standard initial position and alternates the candidate's color. A nonempty book must provide at least one unique position for each game pair. The arena shuffles the available positions, selects one position per pair and plays both color assignments.

Gadus FCPI reads every reachable nonterminal position in a sampling book as a possible starting state, including the standard initial position. It removes transpositions using the same position identity, limits the resulting pool to the requested size and treats outgoing Polyglot moves uniformly.

`scripts/opening_book.py` creates two kinds of Polyglot books. A sampling book supplies positions from ply zero through ply eight to Gadus FCPI. The generator assigns each position to one ply, asks a UCI engine to evaluate every noninitial position and includes only positions whose absolute evaluation does not exceed 80 centipawns. The `--sampling-source` option accepts a PGN or an existing Polyglot book, and sampling generation requires at least 10,000 unique readable positions. The following command creates `openings.sam.bin` from a PGN:

```bash
python scripts/opening_book.py \
	--sampling-source data/games.pgn \
	--uci models/stockfish/stockfish \
	--output data/openings.sam.bin \
	--min-fens 10000 \
	--book-plies 8 \
	--max-abs-cp 80 \
	--log-every 1000
```

The same command accepts a Polyglot source by replacing `data/games.pgn` with its `.bin` path. Sampling generation accepts stricter evaluation bounds but rejects a bound above 80 centipawns or a ply limit above eight. Its final validation checks the minimum position count, the maximum reachable ply and the absence of outgoing edges from the final layer.

An arena book contains positions from one fixed ply whose absolute UCI evaluation lies within a selected bound. The following launcher invocations request 1,000 positions at ply eight with the default bound of 80 centipawns:

```bash
bash scripts/run_opening.sh data/games.pgn 1000 data/openings.gen.bin
```

```powershell
scripts\run_opening.bat data\games.pgn 1000 data\openings.gen.bin
```

The corresponding explicit command is:

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

## Scripts

The `scripts/` directory contains repository-level launchers and data-preparation tools. The architecture documents specify the models and algorithms implemented by the compiled executables. The following sections describe the scripts shared across those executables.

### PGN Analysis

`scripts/analyze.py` analyzes every mainline position in a PGN with a UCI engine. For each recorded move, the script compares the engine's best side-to-move centipawn score with the score assigned to the recorded move and reports their nonnegative difference as centipawn regret. Its primary output is a `.cmt` report beside the input PGN.

The `--pgn-comments` option also writes an annotated PGN. Each generated `{+x}` or `{-x}` comment gives the post-move evaluation from White's perspective. The default annotated path is `<input-name>_cmt.pgn`, and `--pgn-output` selects another path.

```bash
python scripts/analyze.py \
	--input data/user-pgn/game.pgn \
	--uci models/stockfish/stockfish \
	--uci-depth 16 \
	--uci-multipv 8 \
	--analysis-cache data/user-pgn/analysis.sqlite \
	--pgn-comments
```

On Windows, the UCI path commonly ends in `.exe`, as in `models\stockfish\stockfish.exe`.

### Gadus FCPI Launcher

`scripts/gadus_fcpi.sh` launches Gadus FCPI as a background process and then runs an informational closed-search match between the final `current.pth` and the run's `initial.pth`. The launcher records the background process ID in `data/runs/<run-id>/pid`, writes combined output to `data/runs/<run-id>/info.log` and merges the final match report into `summary.json` under `final_arena`.

FCPI stores the aggregated targets for iteration `<NNN>` in `data/runs/<run-id>/fcpi_iter_<NNN>.h5`. Under `models/runs/<run-id>/`, it stores `initial.pth`, `current.pth` and each `candidate_iter_<NNN>.pth`.

```bash
bash scripts/gadus_fcpi.sh
```

The launcher prints the run ID, process ID, log path, monitoring command and stop command. A running job can be monitored and stopped with:

```bash
tail -n 100 -f data/runs/<run-id>/info.log
kill "$(cat data/runs/<run-id>/pid)"
```

The launcher's production defaults are `bf16` precision, 10 iterations, 6,000 self-play games per iteration, 512 games in flight, 512 positions per inference batch, 512 target records per batch, 30 training epochs, a batch size of 1,024 and at most 6,000 training steps per iteration. Self-play uses at most 10,000 positions from `data/openings.sam.bin`. Arena evaluation uses at most 1,000 positions from `data/openings.gen.bin`, applies an RPP coefficient of `1.0` and enables IMF.

Environment variables provide per-run overrides for individual launcher settings:

```bash
MODEL=models/gadus/gadus.pth \
ITERATIONS=5 \
GAMES_PER_ITER=2000 \
TRAIN_MAX_STEPS=3000 \
EVAL_GAMES=400 \
bash scripts/gadus_fcpi.sh
```

Running `build/gadus/fcpi` directly executes FCPI and its per-iteration promotion matches in the foreground. The launcher adds background execution, a combined log, a PID file and the final current-versus-initial match.

### Engine Packaging

`scripts/package_engine.bat` and `scripts/package_engine.sh` package a checkpoint for its architecture. The first argument selects the architecture, and the second supplies the source checkpoint. For Eleginus, an optional third argument selects the embedded `uci` or `search` executable and defaults to `uci`.

```powershell
scripts\package_engine.bat gadus models\gadus\gadus.pth
scripts\package_engine.bat melano models\melano\melano.pth
scripts\package_engine.bat eleginus models\eleginus\eleginus.pth uci
```

```bash
bash scripts/package_engine.sh gadus models/gadus/candidate.pth
bash scripts/package_engine.sh melano models/melano/candidate.pth
bash scripts/package_engine.sh eleginus models/eleginus/eleginus.pth uci
```

Gadus and Melano packages contain the UCI executable, checkpoint and required runtime libraries under `models/<architecture>/`. An Eleginus package is a single self-contained executable named `eleginus` for UCI or `eleginus_search` for standalone analysis, with the platform executable suffix where applicable.

### Checkpoint Inspection

`scripts/check.py` performs a read-only inspection of a Gadus, Melano or Eleginus checkpoint. The report includes its architecture, heads, architecture dimensions, action-space size, parameter counts, tensor data types, tensor memory, devices, finite-value status, file size and SHA-256 digest.

```bash
python scripts/check.py --model models/gadus/gadus.pth
python scripts/check.py --model models/melano/melano.pth
python scripts/check.py --model models/eleginus/eleginus.pth
```

### Piece Import

`scripts/import_pieces.py` converts an SVG chess-piece set into the pre-triangulated mesh archive used by the graphical client. [Graphics.md](Graphics.md) specifies the required filenames, style naming rules and import commands.
