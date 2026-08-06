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

resolve_architecture_mode() {
	local name="$1"
	local value="$2"
	case "${value}" in
		1|ON|on|true) echo "ON" ;;
		0|OFF|off|false) echo "OFF" ;;
		*)
			echo "${name} must be 0 or 1." >&2
			exit 1
			;;
	esac
}

BUILD_GADUS="$(resolve_architecture_mode GADIDAE_BUILD_GADUS "${GADIDAE_BUILD_GADUS:-1}")"
BUILD_MELANO="$(resolve_architecture_mode GADIDAE_BUILD_MELANO "${GADIDAE_BUILD_MELANO:-1}")"
BUILD_ELEGINUS="$(resolve_architecture_mode GADIDAE_BUILD_ELEGINUS "${GADIDAE_BUILD_ELEGINUS:-1}")"
BUILD_ELEGINUS_TRAINING="$(resolve_architecture_mode GADIDAE_BUILD_ELEGINUS_TRAINING "${GADIDAE_BUILD_ELEGINUS_TRAINING:-1}")"
BUILD_GRAPHICS="$(resolve_graphics_mode)"
if [[ "${BUILD_ELEGINUS}" == "OFF" ]]; then
	BUILD_ELEGINUS_TRAINING=OFF
fi
if [[ "${BUILD_GADUS}" == "OFF" && "${BUILD_MELANO}" == "OFF" &&
	  "${BUILD_ELEGINUS}" == "OFF" && "${BUILD_GRAPHICS}" == "OFF" ]]; then
	echo "At least one Gadidae architecture must be enabled." >&2
	exit 1
fi
VERIFY_TORCH=OFF
VERIFY_HDF5=OFF
VERIFY_ZLIB=OFF
VERIFY_JSON=OFF
if [[ "${BUILD_GADUS}" == "ON" || "${BUILD_MELANO}" == "ON" ||
	  ("${BUILD_ELEGINUS}" == "ON" && "${BUILD_ELEGINUS_TRAINING}" == "ON") ]]; then
	VERIFY_TORCH=ON
fi
if [[ "${BUILD_GADUS}" == "ON" || "${BUILD_MELANO}" == "ON" ]]; then
	VERIFY_HDF5=ON
	VERIFY_ZLIB=ON
	VERIFY_JSON=ON
fi
if [[ "${BUILD_ELEGINUS}" == "ON" && "${BUILD_ELEGINUS_TRAINING}" == "ON" ]]; then
	VERIFY_HDF5=ON
	VERIFY_ZLIB=ON
fi
if [[ "${BUILD_GRAPHICS}" == "ON" ]]; then
	VERIFY_ZLIB=ON
	VERIFY_JSON=ON
fi

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
if [[ "${VERIFY_TORCH}" == "ON" && ! -f "${TORCH_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
	echo "LibTorch is missing or GADIDAE_TORCH_DIR is invalid." >&2
	exit 1
fi
cmake \
	"-DAPI_DIR=${ROOT_DIR}/api" \
	"-DTORCH_DIR=${TORCH_DIR}" \
	"-DVERIFY_TORCH=${VERIFY_TORCH}" \
	"-DVERIFY_HDF5=${VERIFY_HDF5}" \
	"-DVERIFY_ZLIB=${VERIFY_ZLIB}" \
	"-DVERIFY_JSON=${VERIFY_JSON}" \
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
	"-DGADIDAE_BUILD_GADUS=${BUILD_GADUS}" \
	"-DGADIDAE_BUILD_MELANO=${BUILD_MELANO}" \
	"-DGADIDAE_BUILD_ELEGINUS=${BUILD_ELEGINUS}" \
	"-DGADIDAE_BUILD_ELEGINUS_TRAINING=${BUILD_ELEGINUS_TRAINING}" \
	"-DGADIDAE_BUILD_GRAPHICS=${BUILD_GRAPHICS}" \
	"-DGADIDAE_TORCH_DIR=${TORCH_DIR}"
cmake --build "${WORK_DIR}" --parallel "$(nproc 2>/dev/null || echo 2)"
ctest --test-dir "${WORK_DIR}" --output-on-failure

if [[ "${BUILD_GADUS}" == "ON" ]]; then
	rm -rf -- "${PUBLISH_DIR}/gadus"
	mkdir -p "${PUBLISH_DIR}/gadus"
	for executable in preprocess train search arena fcpi uci; do
		test -x "${WORK_DIR}/gadus/${executable}"
		cp "${WORK_DIR}/gadus/${executable}" "${PUBLISH_DIR}/gadus/"
	done
fi
if [[ "${BUILD_MELANO}" == "ON" ]]; then
	rm -rf -- "${PUBLISH_DIR}/melano"
	mkdir -p "${PUBLISH_DIR}/melano"
	for executable in preprocess train search arena uci; do
		test -x "${WORK_DIR}/melano/${executable}"
		cp "${WORK_DIR}/melano/${executable}" "${PUBLISH_DIR}/melano/"
	done
fi
if [[ "${BUILD_ELEGINUS}" == "ON" ]]; then
	rm -rf -- "${PUBLISH_DIR}/eleginus"
	mkdir -p "${PUBLISH_DIR}/eleginus"
	for executable in search uci; do
		test -x "${WORK_DIR}/eleginus/${executable}"
		cp "${WORK_DIR}/eleginus/${executable}" "${PUBLISH_DIR}/eleginus/"
	done
	if [[ "${BUILD_ELEGINUS_TRAINING}" == "ON" ]]; then
		for executable in preprocess train embed tests; do
			test -x "${WORK_DIR}/eleginus/${executable}"
			cp "${WORK_DIR}/eleginus/${executable}" "${PUBLISH_DIR}/eleginus/"
		done
	fi
fi
if [[ "${BUILD_GRAPHICS}" == "ON" ]]; then
	rm -rf -- "${PUBLISH_DIR}/graphics"
	mkdir -p "${PUBLISH_DIR}/graphics"
	test -x "${WORK_DIR}/graphics/Gadidae"
	cp "${WORK_DIR}/graphics/Gadidae" "${PUBLISH_DIR}/graphics/"
fi

if [[ "${BUILD_GADUS}" == "ON" ]]; then echo "Gadus build finished: ${PUBLISH_DIR}/gadus"; fi
if [[ "${BUILD_MELANO}" == "ON" ]]; then echo "Melano build finished: ${PUBLISH_DIR}/melano"; fi
if [[ "${BUILD_ELEGINUS}" == "ON" ]]; then echo "Eleginus build finished: ${PUBLISH_DIR}/eleginus"; fi
if [[ "${BUILD_ELEGINUS}" == "ON" && "${BUILD_ELEGINUS_TRAINING}" == "OFF" ]]; then
	echo "Eleginus LibTorch training tools skipped."
fi
if [[ "${BUILD_GRAPHICS}" == "ON" ]]; then
	echo "Gadidae graphics finished: ${PUBLISH_DIR}/graphics"
else
	echo "Gadidae graphics skipped for this headless build."
fi
echo "Incremental build cache: ${WORK_DIR}"
