# Gadidae

Gadidae is a family of experimental chess engines.

## Dependencies

### Python

Install the Python dependencies used by the repository scripts with:

```bash
python -m pip install -r scripts/requirements.txt
```

### C++

The Windows and Linux dependency installers place LibTorch, HDF5, zlib, nlohmann-json, chess-library and Ninja under `api/`. When graphics support is enabled, they additionally place GLFW, GLM, Dear ImGui, FreeType and GLAD under `api/`. Both installers read their pinned dependency versions from `api/versions.env`.

The installers select the LibTorch CUDA `cu126` package when `nvidia-smi` reports a GPU compute capability. They select the CPU package in other environments. Set `GADIDAE_TORCH_VARIANT=cpu` or `GADIDAE_TORCH_VARIANT=cu126` to select a package explicitly. Set `GADIDAE_TORCH_DIR` to use an existing LibTorch installation.

After installing the dependencies, `api/setup.bat` and `api/setup.sh` verify each installed package. The build scripts repeat this verification before configuring CMake. LibTorch, HDF5, zlib and chess-library must match the versions and checksums recorded in `api/versions.env`. After verification succeeds, the installer removes downloaded archives, extracted sources and temporary dependency build directories.

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

Build on Linux with:

```bash
bash scripts/build.sh
```

A command-line-only Linux server can install and build with graphics disabled:

```bash
GADIDAE_BUILD_GRAPHICS=0 bash api/setup.sh
GADIDAE_BUILD_GRAPHICS=0 bash scripts/build.sh
```

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
```

A graphics-enabled build also produces `build/graphics/Gadidae`. Windows executables use the `.exe` suffix. [Graphics.md](Graphics.md) describes the graphical client, its operating modes and its piece-import pipeline.

The `build/.build-work/` directory stores the CMake and Ninja state used for incremental builds. Subsequent builds reuse compatible object files from this directory. Every build runs CTest before publishing the executables. A compilation or test failure leaves diagnostic files under `build/.build-work/`. CTest failures also produce `build/.build-work/Testing/Temporary/LastTest.log`.

The `preprocess`, `train`, `search`, `arena` and Gadus `fcpi` entry points provide their current argument lists through `--help`:

```bash
build/gadus/search --help
build/melano/train --help
```

The Gadus and Melano UCI executables publish their configurable engine options in response to the UCI `uci` command rather than through `--help`.

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

`--max-games` limits the number of PGN games read. `--chunk-size` controls HDF5 dataset extension units. `--compression-level` selects the deflate level. `--log-every` controls progress reporting by game count.

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

Passing an empty value to `--opening-book` starts every game from the standard initial position.

Launch the Gadus UCI engine on Windows with:

```powershell
build\gadus\uci.exe `
	--model models\gadus\gadus.pth `
	--device cpu `
	--search-type only-mcts `
	--mcts-sims 100
```

The corresponding Linux command is:

```bash
build/gadus/uci \
	--model models/gadus/gadus.pth \
	--device cpu \
	--search-type only-mcts \
	--mcts-sims 100
```

A UCI client can configure a running Gadus engine with commands such as:

```text
setoption name SearchType value only-mcts
setoption name MCTSSims value 1000
setoption name MCTSBatchSize value 64
setoption name CPuct value 0.5
setoption name RepetitionPolicyPenalty value 1.0
setoption name InstantMateFirst value true
```

### Melano

Preprocess an annotated PGN into the Melano HDF5 schema with:

```bash
build/melano/preprocess \
	--input data/games.cmt.pgn \
	--output data/games.melano.h5 \
	--has-cmt 1 \
	--chunk-size 4096 \
	--compression-level 1 \
	--log-every 10000 \
	--max-games 1000000
```

`--max-games` limits the number of PGN games read. `--chunk-size` controls HDF5 dataset extension units. `--compression-level` selects the deflate level. `--log-every` controls progress reporting by game count.

Train a new Melano model with:

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

A UCI client can configure a running Melano engine with commands such as:

```text
setoption name SearchType value only-mcts
setoption name MCTSSims value 1000
setoption name MCTSBatchSize value 64
setoption name CPuct value 0.5
setoption name RepetitionPolicyPenalty value 1.0
setoption name InstantMateFirst value true
```

## Scripts

The `scripts/` directory contains repository-level launchers and data-preparation tools. The architecture documents specify the models and algorithms implemented by the compiled executables. The following sections describe the scripts shared across those executables.

### PGN Analysis and Annotation

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

### Opening Books

`scripts/opening_book.py` creates two kinds of Polyglot opening books. A sampling book contains unique reachable nonterminal positions drawn from arbitrary plies and supplies varied starting states to Gadus FCPI. The `--sampling-source` option accepts either a PGN or an existing Polyglot book. The following command creates a pool of at least 10,000 positions from an existing book:

```bash
python scripts/opening_book.py \
	--sampling-source data/openings.bin \
	--output data/openings.sam.bin \
	--min-fens 10000 \
	--log-every 1000
```

An arena book contains positions from one fixed ply whose absolute UCI evaluation lies within a selected bound. The shell and batch launchers generate 1,000 positions at ply eight with a default bound of 80 centipawns:

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

### Gadus FCPI Launcher

`scripts/gadus_fcpi.sh` launches Gadus FCPI as a background process and then runs an informational closed-search match between the final `current.pth` and the run's `initial.pth`. The launcher records the background process ID in `data/runs/<run-id>/pid`, writes combined output to `data/runs/<run-id>/info.log` and merges the final match report into `summary.json` under `final_arena`.

```bash
bash scripts/gadus_fcpi.sh
```

The launcher prints the run ID, process ID, log path, monitoring command and stop command. A running job can be monitored and stopped with:

```bash
tail -n 100 -f data/runs/<run-id>/info.log
kill "$(cat data/runs/<run-id>/pid)"
```

The launcher's production defaults are `bf16` precision, 10 iterations, 6,000 self-play games per iteration, 512 games in flight, 512 positions per inference batch, 512 target records per batch, 30 training epochs, a batch size of 1,024 and at most 6,000 training steps per iteration. Self-play uses at most 10,000 positions from `data/openings.sam.bin`. Arena evaluation uses at most 1,000 positions from `data/openings.gen.bin`.

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
scripts\package_engine.bat gadus models\gadus\candidate.pth
scripts\package_engine.bat melano models\melano\candidate.pth
```

```bash
bash scripts/package_engine.sh gadus models/gadus/candidate.pth
bash scripts/package_engine.sh melano models/melano/candidate.pth
```

On Windows, each package is written to `models/<architecture>/` as `<architecture>.exe`, `<architecture>.pth` and the required DLLs. On Linux, the same directory contains an `<architecture>` launcher, an `<architecture>.bin` executable, `<architecture>.pth` and a private `lib/` directory. Repackaging an architecture updates these files in place.

### Checkpoint Inspection

`scripts/check.py` performs a read-only inspection of a Gadus or Melano checkpoint. The report includes its architecture, heads, channels, blocks, action-space size, parameter counts, tensor data types, tensor memory, devices, finite-value status, file size and SHA-256 digest.

```bash
python scripts/check.py --model models/gadus/gadus.pth
python scripts/check.py --model models/melano/melano.pth
```

### Piece Import

`scripts/import_pieces.py` converts an SVG chess-piece set into the pre-triangulated mesh archive used by the graphical client. [Graphics.md](Graphics.md) specifies the required filenames, style naming rules and import commands.
