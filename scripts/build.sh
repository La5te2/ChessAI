#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NINJA_DIR="${ROOT_DIR}/api/ninja"
TORCH_DIR="${GADIDAE_TORCH_DIR:-${ROOT_DIR}/api/libtorch}"
WORK_DIR="${ROOT_DIR}/.build-work"
CRASH_DIR="${ROOT_DIR}/.crash"
PUBLISH_DIR="${ROOT_DIR}/build"

cleanup_transient_directories() {
	if [[ "${WORK_DIR}" == "${ROOT_DIR}/.build-work" ]]; then
		rm -rf -- "${WORK_DIR}"
	fi
	if [[ "${CRASH_DIR}" == "${ROOT_DIR}/.crash" ]]; then
		rm -rf -- "${CRASH_DIR}"
	fi
}

report_failure() {
	local status=$?
	trap - EXIT
	if [[ ${status} -ne 0 && -d "${WORK_DIR}" ]]; then
		rm -rf -- "${CRASH_DIR}"
		if mv -- "${WORK_DIR}" "${CRASH_DIR}"; then
			echo "Build failed. Diagnostic files retained in: ${CRASH_DIR}" >&2
			if [[ -f "${CRASH_DIR}/Testing/Temporary/LastTest.log" ]]; then
				echo "CTest log: ${CRASH_DIR}/Testing/Temporary/LastTest.log" >&2
			fi
		else
			echo "Build failed. Diagnostic files retained in: ${WORK_DIR}" >&2
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
if [[ "${PUBLISH_DIR}" != "${ROOT_DIR}/build" ]]; then
	exit 1
fi

cleanup_transient_directories
export PATH="${NINJA_DIR}:${PATH}"
cmake \
	-S "${ROOT_DIR}" \
	-B "${WORK_DIR}" \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	"-DGADIDAE_TORCH_DIR=${TORCH_DIR}"
cmake --build "${WORK_DIR}" --parallel "$(nproc 2>/dev/null || echo 2)"
ctest --test-dir "${WORK_DIR}" --output-on-failure

rm -rf -- "${PUBLISH_DIR}"
mkdir -p "${PUBLISH_DIR}/gadus" "${PUBLISH_DIR}/melano"
for architecture in gadus melano; do
	for executable in preprocess train search arena fcpi uci; do
		test -x "${WORK_DIR}/${architecture}/${executable}"
		cp "${WORK_DIR}/${architecture}/${executable}" "${PUBLISH_DIR}/${architecture}/"
	done
done

rm -rf -- "${WORK_DIR}"
echo "Gadus build finished: ${PUBLISH_DIR}/gadus"
echo "Melano build finished: ${PUBLISH_DIR}/melano"
