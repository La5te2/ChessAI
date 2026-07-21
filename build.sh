#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NINJA_DIR="${ROOT_DIR}/api/ninja"
BUILD_DIR="${GADUS_BUILD_DIR:-${ROOT_DIR}/build}"

if [[ ! -x "${NINJA_DIR}/ninja" ]]; then
	echo "Ninja is missing. Run bash api/setup.sh first." >&2
	exit 1
fi

export PATH="${NINJA_DIR}:${PATH}"
CMAKE_ARGS=(
	-S "${ROOT_DIR}"
	-B "${BUILD_DIR}"
	-G Ninja
	-DCMAKE_BUILD_TYPE=Release
)
if [[ -n "${GADUS_TORCH_DIR:-}" ]]; then
	CMAKE_ARGS+=("-DGADIDAE_TORCH_DIR=${GADUS_TORCH_DIR}")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 2)"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "Gadus build finished: ${BUILD_DIR}/gadus"
