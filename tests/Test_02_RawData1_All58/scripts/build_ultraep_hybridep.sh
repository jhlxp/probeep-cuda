#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${ULTRAEP_SOURCE_ROOT:-${PROBEEP_ROOT}/src/ultraep}"
RUNTIME_ROOT="${ULTRAEP_HYBRIDEP_ROOT:-${PROBEEP_ROOT}/build/ultraep-hybridep-e0a5b1d9}"
HYBRID_ROOT="${RUNTIME_ROOT}/HybridEP"
RDMA_CORE_HOME="${RDMA_CORE_HOME:-${RUNTIME_ROOT}/rdma-core-container}"
RDMA_INCLUDE_DIR="${HYBRIDEP_RDMA_INCLUDE_DIR:-/usr/include}"
RDMA_LIBRARY_DIR="${HYBRIDEP_RDMA_LIBRARY_DIR:-/opt/probeep-host-rdma/lib}"
NVIDIA_LIBRARY_DIR="${HYBRIDEP_NVIDIA_LIBRARY_DIR:-/.singularity.d/libs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NCCL_COMMIT=1e0c869c39bb33f1034cb9920bd2a8a8406f04a3

if [[ ! -f "${RUNTIME_ROOT}/setup.py" ]]; then
  mkdir -p "${RUNTIME_ROOT}"
  cp -a "${SOURCE_ROOT}/." "${RUNTIME_ROOT}/"
  patch --directory="${HYBRID_ROOT}" --strip=1 < "${SCRIPT_DIR}/hybridep_setup.patch"
  patch --directory="${HYBRID_ROOT}" --strip=1 < "${SCRIPT_DIR}/hybridep_cuda12.patch"
  sed -i \
    's/cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,/cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster,/g' \
    "${HYBRID_ROOT}/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh"
fi

if [[ ! -f "${HYBRID_ROOT}/third-party/nccl/Makefile" ]]; then
  mkdir -p "${HYBRID_ROOT}/third-party/nccl"
  NCCL_INDEX="$(mktemp /tmp/probeep-clean-main-nccl-index.XXXXXX)"
  trap 'rm -f "${NCCL_INDEX}"' EXIT
  rm -f "${NCCL_INDEX}"
  GIT_INDEX_FILE="${NCCL_INDEX}" git \
    --git-dir="${HYBRIDEP_NCCL_GIT_DIR}" \
    --work-tree="${HYBRID_ROOT}/third-party/nccl" \
    read-tree "${NCCL_COMMIT}"
  GIT_INDEX_FILE="${NCCL_INDEX}" git \
    --git-dir="${HYBRIDEP_NCCL_GIT_DIR}" \
    --work-tree="${HYBRID_ROOT}/third-party/nccl" \
    checkout-index --all --force \
    --prefix="${HYBRID_ROOT}/third-party/nccl/"
fi

mkdir -p "${RDMA_CORE_HOME}/include" "${RDMA_CORE_HOME}/lib"
ln -sfn "${RDMA_INCLUDE_DIR}/infiniband" "${RDMA_CORE_HOME}/include/infiniband"
ln -sfn "${RDMA_INCLUDE_DIR}/rdma" "${RDMA_CORE_HOME}/include/rdma"
for library in libibverbs libmlx5; do
  ln -sfn "${RDMA_LIBRARY_DIR}/${library}.so.1" "${RDMA_CORE_HOME}/lib/${library}.so"
  ln -sfn "${RDMA_LIBRARY_DIR}/${library}.so.1" "${RDMA_CORE_HOME}/lib/${library}.so.1"
done
ln -sfn "${NVIDIA_LIBRARY_DIR}/libnvidia-ml.so.1" \
  "${RDMA_CORE_HOME}/lib/libnvidia-ml.so.1"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=9.0
export HYBRID_EP_MULTINODE=1
export USE_NIXL=0
export RDMA_CORE_HOME
export MAX_JOBS="${PROBEEP_BUILD_MAX_JOBS:-32}"

if [[ -n "${ULTRAEP_REUSE_EXTENSION:-}" ]]; then
  cp -p "${ULTRAEP_REUSE_EXTENSION}" "${RUNTIME_ROOT}/ultra_ep/"
fi
if [[ -n "${HYBRIDEP_REUSE_EXTENSION:-}" ]]; then
  cp -p "${HYBRIDEP_REUSE_EXTENSION}" "${HYBRID_ROOT}/"
fi

if ! compgen -G "${RUNTIME_ROOT}/ultra_ep/_C*.so" >/dev/null; then
  (cd "${RUNTIME_ROOT}" && "${PYTHON_BIN}" setup.py build_ext --inplace)
fi
if [[ "${PROBEEP_FORCE_HYBRIDEP_BUILD:-0}" == "1" ]] || \
    ! compgen -G "${HYBRID_ROOT}/hybrid_ep_cpp*.so" >/dev/null; then
  HYBRID_BUILD_ARGS=(setup.py build_ext --inplace)
  if [[ "${PROBEEP_FORCE_HYBRIDEP_BUILD:-0}" == "1" ]]; then
    HYBRID_BUILD_ARGS+=(--force)
  fi
  (cd "${HYBRID_ROOT}" && "${PYTHON_BIN}" "${HYBRID_BUILD_ARGS[@]}")
fi

printf 'UltraEP runtime: %s\nHybridEP runtime: %s\nNCCL commit: %s\n' \
  "${RUNTIME_ROOT}" "${HYBRID_ROOT}" "${NCCL_COMMIT}"
