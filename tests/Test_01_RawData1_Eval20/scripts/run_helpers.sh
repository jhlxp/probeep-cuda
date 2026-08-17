#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TYPE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TYPE_DIR}/../.." && pwd)"
PROBEEP_TEST_DIR="${TYPE_DIR}"
PLAN_ONLY="${PLAN_ONLY:-0}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/configs/h20_multinode.env}"
LAUNCHER="${SCRIPT_DIR}/launch_slurm.sh"

if [[ "${PLAN_ONLY}" != "0" && "${PLAN_ONLY}" != "1" ]]; then
  echo "PLAN_ONLY must be 0 or 1" >&2
  exit 2
fi

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

require_environment() {
  if [[ "${PLAN_ONLY}" == "0" && ! -f "${ENV_FILE}" ]]; then
    echo "missing target-machine env: ${ENV_FILE}" >&2
    echo "copy ${SCRIPT_DIR}/configs/h20_multinode.env.example to ${ENV_FILE} first" >&2
    exit 2
  fi
}

begin_test_run() {
  local slug="$1"
  local stamp="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S_%6N)}"
  TEST_RUN_ID="${TEST_RUN_ID:-run_${stamp}_${slug}}"
  TEST_RUN_DIR="${TEST_RUN_DIR:-${REPO_ROOT}/test_logs/${TEST_RUN_ID}}"
  export TEST_RUN_ID TEST_RUN_DIR
  echo "[test-run] ${TEST_RUN_DIR}"
}

run_suite_at() {
  local phase="$1"
  local label="$2"
  local suite="$3"
  shift 3
  local phase_dir="${TEST_RUN_DIR:?begin_test_run must run first}/${phase}"
  local phase_tag="${phase//\//_}"
  local phase_id="${TEST_RUN_ID}_${phase_tag}"
  echo "[${label}]"
  print_command env \
    "PROBEEP_RUN_ID=${phase_id}" \
    "PROBEEP_RUN_DIR=${phase_dir}" \
    "PROBEEP_RESULTS_ROOT=${TEST_RUN_DIR}" \
    bash "${LAUNCHER}" "${ENV_FILE}" "${suite}" "$@"
  if [[ "${PLAN_ONLY}" == "0" ]]; then
    require_environment
    env \
      "PROBEEP_RUN_ID=${phase_id}" \
      "PROBEEP_RUN_DIR=${phase_dir}" \
      "PROBEEP_RESULTS_ROOT=${TEST_RUN_DIR}" \
      bash "${LAUNCHER}" "${ENV_FILE}" "${suite}" "$@"
  fi
}

run_local() {
  local label="$1"
  shift
  echo "[${label}]"
  print_command "$@"
  if [[ "${PLAN_ONLY}" == "0" ]]; then
    if [[ -n "${PROBEEP_APPTAINER_IMAGE:-}" && "${PROBEEP_CONTAINER_ACTIVE:-0}" != "1" ]]; then
      apptainer exec --containall \
        --bind "${REPO_ROOT}:${REPO_ROOT}:rw" \
        --cwd "${REPO_ROOT}" \
        "${PROBEEP_APPTAINER_IMAGE}" "$@"
    else
      "$@"
    fi
  fi
}

source_topology() {
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
  fi
  NNODES="${NNODES:-2}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  NUM_EXPERTS="${NUM_EXPERTS:-256}"
  TOKENS_PER_RANK="${TOKENS_PER_RANK:-4096}"
  TOPK="${TOPK:-8}"
  WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
  export NNODES GPUS_PER_NODE NUM_EXPERTS TOKENS_PER_RANK TOPK WORLD_SIZE
}

run_correctness_log() {
  if [[ "${RUN_CORRECTNESS:-1}" != "1" ]]; then
    return
  fi
  run_suite_at correctness/deepep deepep-correctness deepep-smoke
  run_suite_at correctness/deepep-moonep deepep-moonep-correctness deepep-moonep-smoke
  run_suite_at correctness/probeep probeep-correctness probeep-full
  run_suite_at correctness/observation-synthetic observation-consumer probeep-feedback-synthetic
  run_suite_at correctness/ultraep-hybridep ultraep-correctness ultraep-smoke
}

run_setup_log() {
  if [[ "${RUN_SETUP:-1}" != "1" ]]; then
    return
  fi
  run_suite_at setup/preflight preflight preflight
  run_suite_at setup/build extensions build-extensions
  run_suite_at setup/import imports import-smoke
}

run_nsys_log() {
  if [[ "${RUN_NSYS:-1}" != "1" ]]; then
    return
  fi
  local targets=(
    deepep-smoke
    deepep-moonep-smoke
    probeep-forward
    probeep-expert-io
    probeep-backward
    ultraep-smoke
    formal-performance
  )
  local target
  for target in "${targets[@]}"; do
    if [[ "${target}" == "formal-performance" ]]; then
      run_suite_at "nsys/formal-pipeline" "nsys-formal-pipeline" nsys \
        --target-suite formal-performance --include-layer 0 \
        --max-cases 5 --profile-worker
    else
      run_suite_at "nsys/${target}" "nsys-${target}" nsys --target-suite "${target}"
    fi
  done
}

build_test_artifacts() {
  if [[ "${RUN_ARTIFACTS:-1}" != "1" ]]; then
    return
  fi
  local benchmark_dir="${TEST_RUN_DIR}/benchmark"
  local artifact_dir="${TEST_RUN_DIR}/artifacts"
  local report_name="raw_data1_layers_00_19_by_algorithm_rounds_11_20_mean"
  run_local algorithm-layer-report \
    python3 "${SCRIPT_DIR}/reprocess_layers_by_algorithm.py" \
    "${benchmark_dir}" \
    --layers 0-19 \
    --output-dir "${artifact_dir}/${report_name}"
  run_local validate-test-run \
    python3 "${SCRIPT_DIR}/validate_run_layout.py" \
    "${TEST_RUN_DIR}" --kind test
}
