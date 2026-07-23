#!/usr/bin/env bash
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${API_DIR}/.." && pwd)"
DOWNLOADS="${API_DIR}/downloads"
VERSION_FILE="${API_DIR}/versions.env"

cleanup() {
	rm -rf -- \
		"${API_DIR}/zlib-src" \
		"${API_DIR}/zlib-build" \
		"${API_DIR}/zlib-unpack" \
		"${API_DIR}/hdf5-src" \
		"${API_DIR}/hdf5-build" \
		"${API_DIR}/hdf5-unpack" \
		"${DOWNLOADS}"
}
trap cleanup EXIT

if [[ ! -f "${VERSION_FILE}" ]]; then
	echo "Dependency lock is missing: ${VERSION_FILE}" >&2
	exit 1
fi
source "${VERSION_FILE}"
mkdir -p "${DOWNLOADS}"

TORCH_VARIANT="cpu"
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits >/dev/null 2>&1; then
	TORCH_VARIANT="${TORCH_GPU_VARIANT}"
fi
TORCH_VARIANT="${GADIDAE_TORCH_VARIANT:-${TORCH_VARIANT}}"
case ",${TORCH_VARIANTS}," in
	*,"${TORCH_VARIANT}",*) ;;
	*)
		echo "Unsupported LibTorch variant ${TORCH_VARIANT}; allowed: ${TORCH_VARIANTS}" >&2
		exit 1
		;;
esac

download() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 "${url}" -o "${output}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
}

if [[ "${GADIDAE_SKIP_TORCH:-0}" == "1" ]]; then
  echo "Skipping LibTorch setup."
elif [[ ! -d "${API_DIR}/libtorch/share/cmake/Torch" ]]; then
  TORCH_ZIP="${DOWNLOADS}/libtorch-${TORCH_VERSION}-${TORCH_VARIANT}-linux.zip"
  TORCH_URL="https://download.pytorch.org/libtorch/${TORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_VARIANT}.zip"
  echo "Downloading LibTorch ${TORCH_VERSION} ${TORCH_VARIANT}..."
  download "${TORCH_URL}" "${TORCH_ZIP}"
  unzip -q -o "${TORCH_ZIP}" -d "${API_DIR}"
else
  echo "LibTorch already installed."
fi
cmake \
	"-DAPI_DIR=${API_DIR}" \
	"-DTORCH_DIR=${API_DIR}/libtorch" \
	-P "${API_DIR}/patch.cmake"

if [[ ! -x "${API_DIR}/ninja/ninja" ]]; then
  NINJA_ZIP="${DOWNLOADS}/ninja-${NINJA_VERSION}-linux.zip"
  echo "Downloading Ninja ${NINJA_VERSION}..."
  download "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip" \
    "${NINJA_ZIP}"
  mkdir -p "${API_DIR}/ninja"
  unzip -q -o "${NINJA_ZIP}" -d "${API_DIR}/ninja"
  chmod +x "${API_DIR}/ninja/ninja"
else
  echo "Ninja already installed."
fi

if [[ ! -f "${API_DIR}/nlohmann/include/nlohmann/json.hpp" ]]; then
  echo "Downloading nlohmann-json ${JSON_VERSION}..."
  mkdir -p "${API_DIR}/nlohmann/include/nlohmann"
  download "https://github.com/nlohmann/json/releases/download/v${JSON_VERSION}/json.hpp" \
    "${API_DIR}/nlohmann/include/nlohmann/json.hpp"
fi

if [[ ! -f "${API_DIR}/chess/chess.hpp" ]]; then
  echo "Downloading chess-library 0.9.4..."
  mkdir -p "${API_DIR}/chess"
  download "https://raw.githubusercontent.com/Disservin/chess-library/${CHESS_REF}/include/chess.hpp" \
    "${API_DIR}/chess/chess.hpp"
  echo "${CHESS_SHA256}  ${API_DIR}/chess/chess.hpp" | sha256sum --check --status
fi

if [[ ! -d "${API_DIR}/zlib/lib" && ! -d "${API_DIR}/zlib/lib64" ]]; then
  ZLIB_ARCHIVE="${DOWNLOADS}/zlib-${ZLIB_VERSION}.tar.gz"
  echo "Downloading and building zlib ${ZLIB_VERSION}..."
  if [[ ! -f "${ZLIB_ARCHIVE}" ]]; then
    download "https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VERSION}.tar.gz" "${ZLIB_ARCHIVE}"
  fi
  rm -rf "${API_DIR}/zlib-src" "${API_DIR}/zlib-build"
  mkdir -p "${API_DIR}/zlib-src"
  tar -xzf "${ZLIB_ARCHIVE}" --strip-components=1 -C "${API_DIR}/zlib-src"
  cmake -S "${API_DIR}/zlib-src" -B "${API_DIR}/zlib-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DZLIB_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX="${API_DIR}/zlib"
  cmake --build "${API_DIR}/zlib-build" --parallel "$(nproc 2>/dev/null || echo 2)"
  cmake --install "${API_DIR}/zlib-build"
else
  echo "zlib already installed."
fi

if [[ ! -d "${API_DIR}/hdf5/lib" && ! -d "${API_DIR}/hdf5/lib64" ]]; then
  HDF5_ARCHIVE="${DOWNLOADS}/hdf5-${HDF5_VERSION}.tar.gz"
  echo "Downloading and building HDF5 ${HDF5_VERSION}..."
  if [[ ! -f "${HDF5_ARCHIVE}" ]]; then
    download "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_${HDF5_VERSION}.tar.gz" "${HDF5_ARCHIVE}"
  fi
  rm -rf "${API_DIR}/hdf5-src" "${API_DIR}/hdf5-build"
  mkdir -p "${API_DIR}/hdf5-src"
  tar -xzf "${HDF5_ARCHIVE}" --strip-components=1 -C "${API_DIR}/hdf5-src"
  cmake -S "${API_DIR}/hdf5-src" -B "${API_DIR}/hdf5-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DHDF5_BUILD_TOOLS=OFF \
    -DHDF5_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DZLIB_ROOT="${API_DIR}/zlib" \
    -DHDF5_USE_ZLIB_STATIC=ON \
    -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
    -DCMAKE_INSTALL_PREFIX="${API_DIR}/hdf5"
  cmake --build "${API_DIR}/hdf5-build" --parallel "$(nproc 2>/dev/null || echo 2)"
  cmake --install "${API_DIR}/hdf5-build"
else
  echo "HDF5 already installed."
fi

cmake \
	"-DAPI_DIR=${API_DIR}" \
	"-DTORCH_DIR=${API_DIR}/libtorch" \
	"-DEXPECTED_TORCH_VARIANT=${TORCH_VARIANT}" \
	-P "${API_DIR}/verify.cmake"

echo
echo "Gadus dependencies ready."
echo "LibTorch variant: ${TORCH_VARIANT}"
echo "Build: bash \"${ROOT_DIR}/scripts/build.sh\""
