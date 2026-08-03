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

After installation, `api/setup.bat` and `api/setup.sh` verify every dependency that they installed. The build scripts repeat this verification before configuring CMake. LibTorch, HDF5, zlib and chess-library must match the versions and checksums recorded in `api/versions.env`. A successful installation removes its downloaded archives, extracted sources and temporary dependency build directories.

Windows uses the MSVC-compatible LibTorch ABI. Install Microsoft C++ Build Tools and a Windows SDK, then run:

```powershell
$env:GADIDAE_TORCH_VARIANT = "cpu"
.\api\setup.bat
```

The Windows build script `scripts/build.bat` locates Visual Studio through `vswhere`, initializes an x64 compiler environment and invokes CMake with the Ninja generator.

If the OS is Linux, install the base toolchain with:

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

Build on Linux with:

```bash
bash scripts/build.sh
```

A command-line Linux server can install and build explicitly with:

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

Directory `build/.build-work/` stores the CMake and Ninja state used for incremental builds. Subsequent builds reuse compatible object files from this directory. Every build runs CTest before publishing the executables. A compilation or test failure leaves diagnostic files under `build/.build-work/`. CTest failures also produce `build/.build-work/Testing/Temporary/LastTest.log`.

Each command entry point provides its current argument list through `--help`:

```bash
build/gadus/search --help
build/melano/train --help
```
