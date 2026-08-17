#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 ENV_FILE SUITE [RUNNER_ARGS...]" >&2
  echo "suites: readiness preflight build-extensions build-ultraep-hybridep import-smoke deepep-smoke deepep-moonep-smoke probeep-forward probeep-feedback-synthetic probeep-expert-io probeep-backward probeep-full ultraep-smoke formal-performance nsys" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBEEP_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROBEEP_TEST_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="$(realpath "$1")"
SUITE="$2"
shift 2
source "${ENV_FILE}"

NUM_EXPERTS="${NUM_EXPERTS:-256}"
PROBEEP_TOPOLOGY_TAG="${PROBEEP_TOPOLOGY_TAG:-${NNODES}n${GPUS_PER_NODE}g_h20}"

if [[ "${NUM_EXPERTS:-256}" -ne 256 ]]; then
  echo "DSV3 contract requires NUM_EXPERTS=256" >&2
  exit 2
fi
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
if (( WORLD_SIZE == 0 || NUM_EXPERTS % WORLD_SIZE != 0 )); then
  echo "NUM_EXPERTS=${NUM_EXPERTS} must be divisible by world size ${WORLD_SIZE}" >&2
  exit 2
fi
if [[ "${GPUS_PER_NODE}" -ne 8 ]]; then
  echo "current H20 multi-node implementation requires GPUS_PER_NODE=8" >&2
  exit 2
fi
if [[ "${SUITE}" == "formal-performance" ]]; then
  if [[ "${PROBEEP_DYNAMIC_OBSERVATION_MODE:-}" != "benchmark_cuda_events" && "${PROBEEP_DYNAMIC_OBSERVATION_MODE:-}" != "real" ]]; then
    echo "formal-performance requires measured benchmark CUDA-event observations" >&2
    exit 2
  fi
  if [[ -z "${PROBEEP_FORMAL_ENTRYPOINT:-}" || ! -f "${PROBEEP_FORMAL_ENTRYPOINT}" ]]; then
    echo "formal-performance requires an existing PROBEEP_FORMAL_ENTRYPOINT" >&2
    exit 2
  fi
  if [[ -z "${PROBEEP_WORKLOAD_MANIFEST:-}" || ! -f "${PROBEEP_WORKLOAD_MANIFEST}" ]]; then
    echo "formal-performance requires an existing PROBEEP_WORKLOAD_MANIFEST" >&2
    exit 2
  fi
  if [[ -z "${PROBEEP_BENCHMARK_PLAN:-}" || ! -f "${PROBEEP_BENCHMARK_PLAN}" ]]; then
    echo "formal-performance requires an existing PROBEEP_BENCHMARK_PLAN" >&2
    exit 2
  fi
  if [[ "${PROBEEP_FORMAL_PAPER_MODE:-1}" == "1" && "${PROBEEP_WORKLOAD_SELECTOR:-raw_data1_all}" != "raw_data1_all" ]]; then
    echo "paper-mode formal-performance requires PROBEEP_WORKLOAD_SELECTOR=raw_data1_all" >&2
    exit 2
  fi
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S_%6N)}"
case "${SUITE}" in
  formal-performance) RUN_SLUG="${PROBEEP_TOPOLOGY_TAG}_5algo" ;;
  nsys) RUN_SLUG="${PROBEEP_TOPOLOGY_TAG}_nsys" ;;
  *) RUN_SLUG="${PROBEEP_TOPOLOGY_TAG}_${SUITE//[^A-Za-z0-9_]/_}" ;;
esac
PROBEEP_RUN_ID="${PROBEEP_RUN_ID:-run_${RUN_STAMP}_${RUN_SLUG}}"
PROBEEP_RESULTS_ROOT="${PROBEEP_RESULTS_ROOT:-${PROBEEP_ROOT}/test_logs}"
PROBEEP_RUN_DIR="${PROBEEP_RUN_DIR:-${PROBEEP_RESULTS_ROOT}/${PROBEEP_RUN_ID}}"
if [[ -e "${PROBEEP_RUN_DIR}" ]]; then
  echo "run directory already exists; use a new PROBEEP_RUN_ID: ${PROBEEP_RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${PROBEEP_RUN_DIR}/logs" "${PROBEEP_RUN_DIR}/raw" "${PROBEEP_RUN_DIR}/preflight" "${PROBEEP_RUN_DIR}/nsys"
cp "${ENV_FILE}" "${PROBEEP_RUN_DIR}/launch.env"

export PROBEEP_ROOT PROBEEP_TEST_DIR PROBEEP_RUN_ID PROBEEP_RUN_DIR PROBEEP_ENV_FILE="${ENV_FILE}"

ALLOCATION_NODES="${NNODES}"
ALLOCATION_GPUS="${GPUS_PER_NODE}"
ALLOCATION_CPUS="${SLURM_CPUS_PER_TASK:-192}"
ALLOCATION_MEM="${SLURM_MEM_MB:-1937152}"
ALLOCATION_TIME="${SLURM_TIME:-24:00:00}"
ALLOCATION_NODELIST="${SLURM_NODELIST:-}"
if [[ "${SUITE}" == "build-extensions" || "${SUITE}" == "build-ultraep-hybridep" ]]; then
  ALLOCATION_NODES=1
  ALLOCATION_GPUS="${BUILD_GPUS_PER_NODE:-1}"
  ALLOCATION_CPUS="${BUILD_CPUS_PER_TASK:-32}"
  ALLOCATION_MEM="${BUILD_MEM_MB:-262144}"
  ALLOCATION_TIME="${BUILD_TIME:-01:00:00}"
  if [[ -n "${ALLOCATION_NODELIST}" ]]; then
    mapfile -t BUILD_HOSTS < <(scontrol show hostnames "${ALLOCATION_NODELIST}")
    ALLOCATION_NODELIST="${BUILD_HOSTS[0]}"
  fi
fi

SBATCH_ARGS=(
  --wait --parsable
  --partition="${SLURM_PARTITION:-long}"
  --nodes="${ALLOCATION_NODES}"
  --ntasks="${ALLOCATION_NODES}"
  --ntasks-per-node=1
  --gres="gpu:${ALLOCATION_GPUS}"
  --cpus-per-task="${ALLOCATION_CPUS}"
  --mem="${ALLOCATION_MEM}"
  --time="${ALLOCATION_TIME}"
  --job-name="probeep-${SUITE}"
  --output="${PROBEEP_RUN_DIR}/logs/slurm-%j.out"
  --error="${PROBEEP_RUN_DIR}/logs/slurm-%j.err"
  --export=ALL
)
if [[ -n "${ALLOCATION_NODELIST}" ]]; then
  SBATCH_ARGS+=(--nodelist="${ALLOCATION_NODELIST}")
fi
if [[ -n "${SLURM_QOS:-}" && "${SLURM_QOS}" != "normal" ]]; then
  SBATCH_ARGS+=(--qos="${SLURM_QOS}")
fi

set +e
sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/slurm_job.sh" "${SUITE}" "$@"
STATUS=$?
set -e
echo "results: ${PROBEEP_RUN_DIR}"
exit "${STATUS}"
