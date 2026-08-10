#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
	echo "Usage: bash scripts/package_engine.sh <gadus|melano|eleginus> <model.pth> [uci|search]"
	echo "Example: bash scripts/package_engine.sh gadus models/gadus/candidate3.pth"
	echo "Example: bash scripts/package_engine.sh eleginus models/eleginus/eleginus.pth uci"
	exit 1
fi

ARCH="$1"
MODEL_ARG="$2"

case "$ARCH" in
	gadus|melano|eleginus)
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
if [[ "$ARCH" == "eleginus" ]]; then
	TYPE="${3:-uci}"
	if [[ "$TYPE" != "uci" && "$TYPE" != "search" ]]; then
		echo "Eleginus package type must be uci or search: $TYPE" >&2
		exit 2
	fi
	EMBED="$ROOT/build/eleginus/embed"
	if [[ ! -x "$EMBED" ]]; then
		echo "Eleginus embed executable not found: $EMBED" >&2
		echo "Build first with: bash scripts/build.sh" >&2
		exit 4
	fi
	if [[ "$TYPE" == "uci" ]]; then
		ENGINE="$ROOT/models/eleginus/eleginus"
	else
		ENGINE="$ROOT/models/eleginus/eleginus_search"
	fi
	mkdir -p "$ROOT/models/eleginus"
	"$EMBED" --model "$MODEL" --type "$TYPE" --output "$ENGINE"
	chmod +x "$ENGINE"
	echo "Gadidae Eleginus executable packaged"
	echo "type=$TYPE"
	echo "executable=$ENGINE"
	exit 0
fi
UCI="$ROOT/build/$ARCH/uci"
OUTPUT="$ROOT/models/$ARCH"
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
if [[ "$MODEL" != "$OUTPUT/$ARCH.pth" ]]; then
	cp -f "$MODEL" "$OUTPUT/$ARCH.pth"
fi

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
