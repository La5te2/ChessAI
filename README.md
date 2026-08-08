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

A command-line-only Linux server can install and build with graphics disabled:

```bash
GADIDAE_BUILD_GRAPHICS=0 bash api/setup.sh
GADIDAE_BUILD_GRAPHICS=0 bash scripts/build.sh
```

The same switches select any subset on Linux. Disabled components are not rebuilt, published, or
removed from `build/`.

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
build/melano/arena
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
	--min-net-wins 4 \
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

Run a paired Melano arena with:

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

Passing an empty value to `--opening-book` starts every game from the standard initial position.

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

Preprocess a PGN into the architecture-locked Eleginus Value schema with:

```bash
build/eleginus/preprocess \
	--input data/games.cmt.pgn \
	--output data/games.eleginus.h5 \
	--has-cmt 1 \
	--compression-level 4
```

The HDF5 file uses the same `states`, `moves` and `values` dataset names as Gadus and Melano, but
its root metadata identifies `arch_type=eleginus` together with Eleginus-specific state, move and
target encodings. Every reader rejects a dataset produced for either of the other architectures.
With `--has-cmt 1`, numerical pawn evaluations in PGN comments become side-to-move targets in `[0,1]`;
with `--has-cmt 0`, completed game results supply targets in `{0, 0.5, 1}`.

Train a new Eleginus Value network with:

```bash
build/eleginus/train \
	--data data/games.eleginus.h5 \
	--out models/eleginus/eleginus.pth \
	--epochs 10 \
	--batch-size 512 \
	--lr 0.001 \
	--device cuda \
	--seed 2026
```

Eleginus uses the same `--max-steps` convention: a positive value caps optimizer updates, while `0` lets `--epochs` determine the training length.

This command trains the sole Eleginus Value network. Supplying `--model existing.pth` initializes
the training run from those parameters, while omitting it initializes a new Value model. Each run
constructs a new AdamW optimizer. Eleginus currently defines no self-learning procedure.

The `.pth` checkpoint is the only standalone Eleginus weight file. Like the other architecture
checkpoints, its top level contains `model` and `arch`; the latter identifies Eleginus with
`type_id=3` and records every fixed network dimension. Inspect it with:

```bash
python scripts/check.py --model models/eleginus/eleginus.pth
```

Search and UCI are built first as weightless, Torch-free executable templates. Embed the checkpoint
into a copy of the standalone search template and analyze one position:

```bash
build/eleginus/embed \
	--model models/eleginus/eleginus.pth \
	--input build/eleginus/search \
	--output models/eleginus/eleginus-search

models/eleginus/eleginus-search \
	--fen startpos \
	--expansions 32
```

`--fen` accepts `startpos` or one quoted six-field FEN. The embedded search program uses the
statically linked float32 NNUE evaluator and therefore has no model path, device argument or
LibTorch runtime dependency.

Create and launch the Eleginus UCI engine on Windows with:

```powershell
build\eleginus\embed.exe `
	--model models\eleginus\eleginus.pth `
	--input build\eleginus\uci.exe `
	--output models\eleginus\eleginus.exe

models\eleginus\eleginus.exe --expansions 32
```

The corresponding Linux command is:

```bash
build/eleginus/embed \
	--model models/eleginus/eleginus.pth \
	--input build/eleginus/uci \
	--output models/eleginus/eleginus

models/eleginus/eleginus --expansions 32
```

The final Eleginus UCI and standalone search executables require neither an external weight file nor
`torch.dll` or `c10.dll`. The `train` and `embed` tools remain training-side LibTorch programs.

## UCI

### Gadus and Melano

The Gadus and Melano UCI executables load their checkpoints when `isready` or `go` first requires neural evaluation. An explicit `--model` argument selects the checkpoint. Without that argument, Gadus reads `gadus.pth` and Melano reads `melano.pth` beside the executable. Each process retains the loaded model across positions and evaluates it in FP32. Changing `ModelPath` or `Device` reloads the model before the next search.

Search runs on a worker thread so the protocol loop can process `stop`. A `position`, `setoption` or `ucinewgame` command first stops and joins an active search before changing engine state. Closing the UCI process also joins the worker.

Both engines report MultiPV rows containing `score cp`, nodes, NPS, elapsed time and a one-move principal variation. In `closed` mode, the reported root evaluation is the network Value $V_\theta(s)$. After at least one MCTS simulation, it is the mean return backed up to the root. A visited root edge reports its backed-up $Q(s,a)$, while an unvisited edge reports the root evaluation. For `ScoreScale` value $c_s$, the displayed value is

$$
\text{score cp}=\mathrm{round}\left(
c_s\,\mathrm{clip}(q_{\mathrm{line}},-0.999,0.999)
\right).
$$

The score is expressed from the root side-to-move perspective. The `nodes` field reports completed simulations and represents the initial neural evaluation as one node when no simulation has completed. For reported node count $n$, both `depth` and `seldepth` equal

$$
\max\left(1,\left\lfloor\log_2\max(1,n)\right\rfloor+1\right).
$$

These depth fields summarize search effort rather than maximum tree depth. NPS is $1000n/t$, where $t$ is elapsed time in milliseconds with a denominator of at least one millisecond. The engines emit the initial root result, periodic MCTS updates and one final result before `bestmove`. `ProgressIntervalMS=0` suppresses periodic updates.

`go movetime <ms>` supplies the wall-clock budget directly. When `go` instead supplies the active side's remaining time $t_{\mathrm{remain}}$ and increment $t_{\mathrm{inc}}$, the engine computes

$$
t_0=\frac{t_{\mathrm{remain}}}{30}+0.75t_{\mathrm{inc}}-t_{\mathrm{overhead}}.
$$

It clamps $t_0$ to 50 through 10000 milliseconds and then limits the allocation to $\max(1,t_{\mathrm{remain}}-t_{\mathrm{overhead}})$. A `go` command without `movetime` or the active clock supplies no wall-clock deadline. `go nodes <n>` overrides the current MCTS simulation cap, and `stop` requests early termination.

Gadus and Melano expose these shared options:

- `ModelPath` selects the checkpoint.
- `Device` selects `auto`, `cpu` or `cuda` and defaults to `auto`.
- `SearchType` selects `closed` or `only-mcts` and defaults to `only-mcts`.
- `MCTSSims` sets the simulation cap and defaults to `100`.
- `MCTSMinSims` sets the nominal simulation floor and defaults to `0`, which activates the dynamic floor described in each architecture's search specification.
- `MCTSBatchSize` sets the neural leaf-batch capacity and defaults to `32`.
- `MoveOverheadMS` reserves time for communication and move submission and defaults to `50`.
- `CPuct`, `CPuctBase` and `CPuctFactor` configure the visit-dependent exploration coefficient and default to `0.5`, `19652` and `1.0`.
- `FPUReduction` sets the First Play Urgency reduction and defaults to `0.15`.
- `VirtualLoss` sets the repeated-path penalty used during batched selection and defaults to `0.0`.
- `RepetitionPolicyPenalty` sets the RPP coefficient in $[0,1]$ and defaults to `0.0`.
- `InstantMateFirst` enables IMF and defaults to `false`.
- `ProgressIntervalMS` sets the periodic report interval and defaults to `750` milliseconds.
- `MultiPV` sets the number of reported root lines and defaults to `5`.
- `ScoreScale` sets $c_s$ in the displayed-score equation and defaults to `1000`.

Gadus additionally exposes `Threads`, which controls LibTorch CPU threads and defaults to `2`, and `EvalCacheMB`, which defaults to `256`. A positive `EvalCacheMB` retains compact Policy and Value evaluations across successive `go` commands with least-recently-used eviction. Each search still builds a new MCTS tree. Setting the option to zero disables cross-search retention while preserving duplicate-evaluation removal within the current search. `ucinewgame`, `ModelPath` changes and `Device` changes clear the retained evaluations.

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

`Threads` and `EvalCacheMB` apply only to Gadus. The remaining commands apply to both engines.

### Eleginus

An embedded Eleginus UCI executable reads its Value parameters from its own file at startup. It exposes `BFMExpansions`, with default value `32` and range 1 through 1,000,000. Each `go` command performs one synchronous best-first minimax search. UCI time, depth and node fields do not alter the expansion budget.

For root value $\overline v(x)$, Eleginus reports

$$
\text{score cp}=\mathrm{round}\left(2000\left(\overline v(x)-\frac12\right)\right).
$$

The score uses the root side-to-move perspective. `depth` reports the number of expanded parents, `nodes` reports the number of generated and evaluated children and the one-move principal variation contains the selected root move. `bestmove` reports the same move.

```text
setoption name BFMExpansions value 128
```

## Opening Books

Gadus and Melano arenas read Polyglot books by traversing legal book moves breadth-first from the standard initial position. The effective depth is at least one ply. The reader ignores Polyglot weight and learn fields, randomizes outgoing move order from the selected seed and emits unique nonterminal frontier positions. Position identity includes piece placement, side to move, castling rights and the en-passant field but excludes move counters.

An empty arena book path starts every game from the standard initial position and alternates the candidate's color. A nonempty book must provide at least one unique position for each game pair. The arena shuffles the available positions, selects one position per pair and plays both color assignments.

Gadus FCPI reads a sampling book as a randomized traversal in which every reachable nonterminal position may enter the starting-state pool, including the standard initial position. It removes transpositions using the same position identity and limits the resulting pool to the requested size. Polyglot weights do not affect this traversal.

`scripts/opening_book.py` creates two kinds of Polyglot books. A sampling book supplies arbitrary-ply positions to Gadus FCPI. The `--sampling-source` option accepts a PGN or an existing Polyglot book. The following command creates a pool of at least 10,000 positions from an existing book:

```bash
python scripts/opening_book.py \
	--sampling-source data/openings.bin \
	--output data/openings.sam.bin \
	--min-fens 10000 \
	--log-every 1000
```

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

`scripts/package_engine.bat` and `scripts/package_engine.sh` package a Gadus or Melano checkpoint with the corresponding UCI executable and runtime libraries. The first argument selects the architecture, and the second supplies the source checkpoint.

```powershell
scripts\package_engine.bat gadus models\gadus\gadus.pth
scripts\package_engine.bat melano models\melano\melano.pth
```

```bash
bash scripts/package_engine.sh gadus models/gadus/candidate.pth
bash scripts/package_engine.sh melano models/melano/candidate.pth
```

On Windows, each package is written to `models/<architecture>/` as `<architecture>.exe`, `<architecture>.pth` and the required DLLs. On Linux, the same directory contains an `<architecture>` launcher, an `<architecture>.bin` executable, `<architecture>.pth` and a private `lib/` directory. Repackaging an architecture updates these files in place.

### Checkpoint Inspection

`scripts/check.py` performs a read-only inspection of a Gadus, Melano or Eleginus checkpoint. The report includes its architecture, heads, architecture dimensions, action-space size, parameter counts, tensor data types, tensor memory, devices, finite-value status, file size and SHA-256 digest.

```bash
python scripts/check.py --model models/gadus/gadus.pth
python scripts/check.py --model models/melano/melano.pth
python scripts/check.py --model models/eleginus/eleginus.pth
```

### Piece Import

`scripts/import_pieces.py` converts an SVG chess-piece set into the pre-triangulated mesh archive used by the graphical client. [Graphics.md](Graphics.md) specifies the required filenames, style naming rules and import commands.
