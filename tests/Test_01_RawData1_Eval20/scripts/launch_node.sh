#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 ENV_FILE SUITE NODE_RANK MASTER_ADDR [RUNNER_ARGS...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBEEP_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROBEEP_TEST_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="$(realpath "$1")"
SUITE="$2"
if [[ "$3" == "auto" ]]; then
  NODE_RANK="${SLURM_PROCID:?SLURM_PROCID is required for automatic node rank}"
else
  NODE_RANK="$3"
fi
MASTER_ADDR="$4"
shift 4

source "${ENV_FILE}"
export PROBEEP_ROOT PROBEEP_TEST_DIR NODE_RANK MASTER_ADDR
export PROBEEP_RUN_ID="${PROBEEP_RUN_ID:?PROBEEP_RUN_ID is required}"
export PROBEEP_RUN_DIR="${PROBEEP_RUN_DIR:?PROBEEP_RUN_DIR is required}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -n "${HYBRID_EP_CACHE_DIR:-}" ]]; then
  export HYBRID_EP_CACHE_DIR="${HYBRID_EP_CACHE_DIR%/}/${SLURMD_NODENAME:-node-${NODE_RANK}}"
fi

if [[ -n "${PROBEEP_APPTAINER_IMAGE:-}" && "${PROBEEP_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  APPTAINER_ARGS=(
    exec
    --nv
    --containall
    --bind "${PROBEEP_ROOT}:${PROBEEP_ROOT}:rw"
    --bind "${NVSHMEM_HOST_DIR}:${NVSHMEM_CONTAINER_DIR}:ro"
    --bind /usr/sbin/ibdev2netdev:/usr/local/bin/ibdev2netdev:ro
    --bind /dev/infiniband:/dev/infiniband
    --cwd "${PROBEEP_ROOT}"
    --env PROBEEP_CONTAINER_ACTIVE=1
    --env "NVSHMEM_DIR=${NVSHMEM_CONTAINER_DIR}"
  )
  if [[ -n "${HYBRIDEP_NCCL_GIT_DIR:-}" ]]; then
    APPTAINER_ARGS+=(--bind "${HYBRIDEP_NCCL_GIT_DIR}:${HYBRIDEP_NCCL_GIT_DIR}:ro")
  fi
  if [[ -n "${HYBRIDEP_HOST_LIBRARY_DIR:-}" ]]; then
    APPTAINER_ARGS+=(
      --bind "${HYBRIDEP_HOST_LIBRARY_DIR}/libmlx5.so.1:${HYBRIDEP_CONTAINER_LIBRARY_DIR}/libmlx5.so.1:ro"
      --bind "${HYBRIDEP_HOST_LIBRARY_DIR}/libmlx5.so.1:${HYBRIDEP_CONTAINER_LIBRARY_DIR}/libmlx5.so:ro"
      --bind "${HYBRIDEP_HOST_LIBRARY_DIR}/libibverbs.so.1:${HYBRIDEP_CONTAINER_LIBRARY_DIR}/libibverbs.so.1:ro"
      --bind "${HYBRIDEP_HOST_LIBRARY_DIR}/libibverbs.so.1:${HYBRIDEP_CONTAINER_LIBRARY_DIR}/libibverbs.so:ro"
    )
  fi
  while IFS= read -r env_name; do
    case "${env_name}" in
      PROBEEP_*|NCCL_*|NVSHMEM_*|CUDA_*|TORCH_*|SLURM_*|ULTRAEP_*|HYBRIDEP_*|HYBRID_EP_*|NIXL_*|UCX_*|FI_*|MASTER_*|NODE_RANK|NNODES|GPUS_PER_NODE|WORLD_SIZE|LOCAL_WORLD_SIZE|NUM_*|TOKENS_PER_RANK|TOPK|HIDDEN|FFN_INTERMEDIATE|LOCAL_EXPERTS|REPLICA_SLOTS|WARMUP_ITERS|MEASURE_ITERS|REPEATS|MAX_JOBS|MAX_NUM_NVL_PEERS|OMP_NUM_THREADS|RDMA_CORE_HOME|USE_NIXL)
        APPTAINER_ARGS+=(--env "${env_name}=${!env_name}")
        ;;
    esac
  done < <(compgen -e)
  exec apptainer "${APPTAINER_ARGS[@]}" "${PROBEEP_APPTAINER_IMAGE}" \
    bash "${SCRIPT_DIR}/launch_node.sh" \
      "${ENV_FILE}" "${SUITE}" "${NODE_RANK}" "${MASTER_ADDR}" "$@"
fi

if [[ "${PROBEEP_CONTAINER_ACTIVE:-0}" == "1" ]]; then
  export NVSHMEM_DIR="${NVSHMEM_CONTAINER_DIR}"
  export PATH="/usr/local/cuda/bin:${PATH}"
  RUNTIME_LIBRARY_PATH="${NVSHMEM_DIR}/lib:/usr/local/cuda/lib64"
  export LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTHONNOUSERSITE=1
elif [[ -n "${VENV_DIR:-}" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "missing venv python: ${VENV_DIR}/bin/python" >&2
    exit 2
  fi
  export PATH="${VENV_DIR}/bin:${PATH}"
fi

exec python3 "${SCRIPT_DIR}/runner.py" "${SUITE}" "$@"
