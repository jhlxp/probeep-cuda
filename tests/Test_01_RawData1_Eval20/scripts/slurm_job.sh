#!/usr/bin/env bash
set -euo pipefail

SUITE="$1"
shift

source "${PROBEEP_ENV_FILE:?PROBEEP_ENV_FILE is required}"
SCRIPT_DIR="${PROBEEP_TEST_DIR:?PROBEEP_TEST_DIR is required}/scripts"
if [[ "${SUITE}" == "build-extensions" || "${SUITE}" == "build-ultraep-hybridep" ]]; then
  export NNODES=1
  export GPUS_PER_NODE="${BUILD_GPUS_PER_NODE:-1}"
  export SLURM_CPUS_PER_TASK="${BUILD_CPUS_PER_TASK:-32}"
fi
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#HOSTS[@]}" -ne "${NNODES}" ]]; then
  echo "allocation has ${#HOSTS[@]} nodes, expected NNODES=${NNODES}" >&2
  exit 2
fi
export MASTER_ADDR="${HOSTS[0]}"
export PROBEEP_RENDEZVOUS_ADDR="${MASTER_ADDR}"

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-192}" \
  --kill-on-bad-exit=1 \
  --output="${PROBEEP_RUN_DIR}/logs/${SUITE}-%N.log" \
  --error="${PROBEEP_RUN_DIR}/logs/${SUITE}-%N.err" \
  bash "${SCRIPT_DIR}/launch_node.sh" \
    "${PROBEEP_ENV_FILE}" "${SUITE}" auto "${MASTER_ADDR}" "$@"
