#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
	echo "Usage: bash scripts/package_engine.sh <gadus|melano> <model.pth>"
	echo "Example: bash scripts/package_engine.sh gadus models/gadus/candidate3.pth"
	exit 1
fi

ARCH="$1"
MODEL_ARG="$2"

case "$ARCH" in
	gadus|melano)
		;;
	*)
		echo "Unsupported architecture: $ARCH" >&2
		exit 2
		;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$MODEL_ARG" ]]; then
	echo "Model not found: $MODEL_ARG" >&2
	exit 3
fi
MODEL="$(realpath "$MODEL_ARG")"
UCI="$ROOT/build/$ARCH/uci"
OUTPUT="$ROOT/models/gadidae"
LIB_OUTPUT="$OUTPUT/lib"
BINARY="$OUTPUT/$ARCH.bin"
LAUNCHER="$OUTPUT/$ARCH"

if [[ ! -x "$UCI" ]]; then
	echo "UCI executable not found: $UCI" >&2
	echo "Build first with: bash scripts/build.sh" >&2
	exit 4
fi

mkdir -p "$OUTPUT" "$LIB_OUTPUT"
cp -f "$UCI" "$BINARY"
cp -f "$MODEL" "$OUTPUT/$ARCH.pth"

for library_dir in \
	"$ROOT/api/libtorch/lib" \
	"$ROOT/api/hdf5/lib" \
	"$ROOT/api/zlib/lib"; do
	if [[ -d "$library_dir" ]]; then
		find "$library_dir" -maxdepth 1 -name '*.so*' -exec cp -Lf {} "$LIB_OUTPUT/" \;
	fi
done

{
	echo '#!/usr/bin/env bash'
	echo 'set -euo pipefail'
	echo 'ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
	echo 'export LD_LIBRARY_PATH="$ENGINE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
	echo "exec \"\$ENGINE_DIR/$ARCH.bin\" \"\$@\""
} >"$LAUNCHER"
chmod +x "$LAUNCHER" "$BINARY"

echo "Gadidae UCI engine packaged"
echo "architecture=$ARCH"
echo "executable=$LAUNCHER"
echo "checkpoint=$OUTPUT/$ARCH.pth"
echo "Cute Chess command=$LAUNCHER"
