# Gadidae Graphics

Gadidae provides a native graphical client for engines that implement the Universal Chess Interface (UCI). The client uses GLFW, OpenGL 3.3, GLAD, Dear ImGui and FreeType for its graphical interface. The UCI boundary allows the client to work with engines independently of their internal architecture.

## 1. Installation and Build

On Windows, install the graphics dependencies and build the client with

```powershell
api\setup.bat
scripts\build.bat
```

On Linux, enable the graphics targets explicitly with

```bash
GADIDAE_BUILD_GRAPHICS=1 bash api/setup.sh
GADIDAE_BUILD_GRAPHICS=1 bash scripts/build.sh
```

The build produces `build/graphics/Gadidae.exe` on Windows and `build/graphics/Gadidae` on Linux. Interactive use over Secure Shell (SSH) requires X11 forwarding, a remote desktop or another display service.

## 2. Engine Interface

Importing an engine starts a temporary process and performs a standard UCI handshake. The client reads every `option name ...` declaration returned by the engine, generates a matching control and then closes the temporary process. UCI options configure a working engine after its handshake, whereas `Launch arguments` supplies arguments when the operating system starts an engine process.

Simulator starts a working engine when analysis opens, and Stadium starts the configured working engines when a match begins. Each working process remains alive for its analysis or match session. A position change sends `stop` to the active search, discards subsequent output associated with the previous position and submits the new position to the same process. The UCI `position` command reconstructs the game from its initial FEN and the complete move sequence up to the position being displayed, preserving the rule history required for repetition detection. The engine determines how quickly an active search responds to `stop`.

## 3. Simulator

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

## 4. Stadium

Stadium manages multiple independent games. `Tools > Matches` creates games, selects the game shown in the active view and closes games. Games outside the active view continue in the background. Each seat requires a participant name. A Human seat accepts moves from the board. An engine seat requires a UCI executable and stores its own UCI option values and launch arguments.

Match settings define the initial clock, increment, display delay, maximum ply count and starting position in Forsyth-Edwards Notation (FEN). An initial clock of zero disables clock timing. `Run > Start`, `Run > Pause` and `Run > Stop` control the active game. Closing Gadidae terminates every UCI subprocess owned by every open match.

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

## 5. Appearance

Appearance settings include dark and light application themes, base font size, bounded font scaling, board-color presets, custom colors, coordinate labels and embedded piece styles. Applying a setting writes `gui.json` beside the Gadidae executable, and subsequent launches restore that configuration. Command-line overrides include `--font-size <px>`, `--theme dark`, `--theme light` and `--piece-style <name>`.

## 6. Piece Import

`scripts/import_pieces.py` converts one Scalable Vector Graphics (SVG) set into pre-triangulated indexed meshes stored in `src/graphics/pieces.gpack`. The input directory must contain `wK.svg`, `wQ.svg`, `wR.svg`, `wB.svg`, `wN.svg`, `wP.svg`, `bK.svg`, `bQ.svg`, `bR.svg`, `bB.svg`, `bN.svg` and `bP.svg`. A valid style name begins with a lowercase letter and contains only lowercase letters, digits or hyphens.

Import a style and rebuild the client on Windows with

```powershell
python -m pip install -r scripts\requirements.txt
python scripts\import_pieces.py `
	--input data\pieces\my-style `
	--name my-style
scripts\build.bat
```

Import and rebuild on Linux with

```bash
python -m pip install -r scripts/requirements.txt
python scripts/import_pieces.py \
	--input data/pieces/my-style \
	--name my-style
GADIDAE_BUILD_GRAPHICS=1 bash scripts/build.sh
```

Importing a new style name adds that style to the embedded archive, whereas reimporting an existing name atomically replaces the stored style. The `--curve-step` argument controls the curve-sampling distance and defaults to `1.5`. Smaller values produce smoother, denser meshes and larger archives, whereas larger values reduce archive size and import time.

Every imported third-party style requires a separate declaration in `THIRD_PARTY.md`. The declaration records the author, permanent source URL, license name and version, license-text URL, modifications and every source used by a combined style.
