#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NINJA_DIR="${ROOT_DIR}/api/ninja"
TORCH_DIR="${GADIDAE_TORCH_DIR:-${ROOT_DIR}/api/libtorch}"
PUBLISH_DIR="${ROOT_DIR}/build"
WORK_DIR="${PUBLISH_DIR}/.build-work"

resolve_graphics_mode() {
	case "${GADIDAE_BUILD_GRAPHICS:-auto}" in
		1|ON|on|true)
			echo "ON"
			;;
		0|OFF|off|false)
			echo "OFF"
			;;
		auto)
			if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
				echo "ON"
			else
				echo "OFF"
			fi
			;;
		*)
			echo "GADIDAE_BUILD_GRAPHICS must be auto, 0, or 1." >&2
			exit 1
			;;
	esac
}

BUILD_GRAPHICS="$(resolve_graphics_mode)"

report_failure() {
	local status=$?
	trap - EXIT
	if [[ ${status} -ne 0 && -d "${WORK_DIR}" ]]; then
		echo "Build failed. Diagnostic files retained in: ${WORK_DIR}" >&2
		if [[ -f "${WORK_DIR}/Testing/Temporary/LastTest.log" ]]; then
			echo "CTest log: ${WORK_DIR}/Testing/Temporary/LastTest.log" >&2
		fi
	fi
	exit "${status}"
}
trap report_failure EXIT

if [[ ! -x "${NINJA_DIR}/ninja" ]]; then
	echo "Ninja is missing. Run bash api/setup.sh first." >&2
	exit 1
fi
if [[ ! -f "${TORCH_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
	echo "LibTorch is missing or GADIDAE_TORCH_DIR is invalid." >&2
	exit 1
fi
cmake \
	"-DAPI_DIR=${ROOT_DIR}/api" \
	"-DTORCH_DIR=${TORCH_DIR}" \
	"-DVERIFY_GUI=${BUILD_GRAPHICS}" \
	-P "${ROOT_DIR}/api/verify.cmake"
if [[ "${PUBLISH_DIR}" != "${ROOT_DIR}/build" ]]; then
	exit 1
fi
if [[ "${WORK_DIR}" != "${ROOT_DIR}/build/.build-work" ]]; then
	exit 1
fi

mkdir -p "${PUBLISH_DIR}"
export PATH="${NINJA_DIR}:${PATH}"
cmake \
	-S "${ROOT_DIR}" \
	-B "${WORK_DIR}" \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	"-DGADIDAE_BUILD_GRAPHICS=${BUILD_GRAPHICS}" \
	"-DGADIDAE_TORCH_DIR=${TORCH_DIR}"
cmake --build "${WORK_DIR}" --parallel "$(nproc 2>/dev/null || echo 2)"
ctest --test-dir "${WORK_DIR}" --output-on-failure

rm -rf -- "${PUBLISH_DIR}/gadus" "${PUBLISH_DIR}/melano" "${PUBLISH_DIR}/graphics"
mkdir -p "${PUBLISH_DIR}/gadus" "${PUBLISH_DIR}/melano"
for executable in preprocess train search arena fcpi uci; do
	test -x "${WORK_DIR}/gadus/${executable}"
	cp "${WORK_DIR}/gadus/${executable}" "${PUBLISH_DIR}/gadus/"
done
for executable in preprocess train search arena uci; do
	test -x "${WORK_DIR}/melano/${executable}"
	cp "${WORK_DIR}/melano/${executable}" "${PUBLISH_DIR}/melano/"
done
if [[ "${BUILD_GRAPHICS}" == "ON" ]]; then
	mkdir -p "${PUBLISH_DIR}/graphics"
	test -x "${WORK_DIR}/graphics/Gadidae"
	cp "${WORK_DIR}/graphics/Gadidae" "${PUBLISH_DIR}/graphics/"
fi

echo "Gadus build finished: ${PUBLISH_DIR}/gadus"
echo "Melano build finished: ${PUBLISH_DIR}/melano"
if [[ "${BUILD_GRAPHICS}" == "ON" ]]; then
	echo "Gadidae graphics finished: ${PUBLISH_DIR}/graphics"
else
	echo "Gadidae graphics skipped for this headless build."
fi
echo "Incremental build cache: ${WORK_DIR}"
