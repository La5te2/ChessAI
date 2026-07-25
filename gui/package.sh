#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$ROOT/build/.pyinstaller-gui"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

if ! "$PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
	echo "PyInstaller is required."
	echo "Install it with: $PYTHON -m pip install pyinstaller"
	exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK"

echo "Packaging gui/Gadidae..."
if ! "$PYTHON" -m PyInstaller \
	--noconfirm \
	--clean \
	--onefile \
	--windowed \
	--name Gadidae \
	--exclude-module pygame \
	--exclude-module numpy \
	--exclude-module torch \
	--paths "$ROOT/gui" \
	--distpath "$ROOT/gui" \
	--workpath "$WORK/work" \
	--specpath "$WORK/spec" \
	"$ROOT/gui/gadidae.py"; then
	echo "GUI packaging failed. Diagnostics retained in: $WORK"
	exit 1
fi

rm -rf "$WORK"
echo "Gadidae GUI finished: $ROOT/gui/Gadidae"
