#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/scripts/run_helpers.sh"
source_topology

SELECTOR=raw_data1_eval20
begin_test_run "test01_rawdata1_eval20_${WORLD_SIZE}r"
CACHE_DIR="${CACHE_DIR:-${TEST_RUN_DIR}/workload}"
MANIFEST="${CACHE_DIR}/manifest.json"
PLAN="${CACHE_DIR}/benchmark_plan.json"

run_local workload-materialize \
  python3 "${TYPE_DIR}/scripts/raw_data1.py" materialize \
  --selector "${SELECTOR}" --world-size "${WORLD_SIZE}" \
  --tokens-per-rank "${TOKENS_PER_RANK}" --topk "${TOPK}" \
  --output-dir "${CACHE_DIR}"

run_local benchmark-plan \
  python3 "${TYPE_DIR}/benchmark.py" \
  --manifest "${MANIFEST}" --output-plan "${PLAN}" \
  --warmup-iters "${WARMUP_ITERS:-10}" \
  --measure-iters "${MEASURE_ITERS:-10}" \
  --repeats "${REPEATS:-1}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

export PROBEEP_WORKLOAD_SELECTOR="${SELECTOR}"
export PROBEEP_WORKLOAD_MANIFEST="${MANIFEST}"
export PROBEEP_BENCHMARK_PLAN="${PLAN}"
export PROBEEP_FORMAL_PAPER_MODE=0
export PROBEEP_DYNAMIC_OBSERVATION_MODE=benchmark_cuda_events
export PROBEEP_WEIGHT_CACHE_MODE="${PROBEEP_WEIGHT_CACHE_MODE:-cold}"
export PROBEEP_FORMAL_ENTRYPOINT="${TYPE_DIR}/scripts/formal_entrypoint.py"
run_setup_log
run_correctness_log
run_suite_at benchmark raw-data1-eval20 formal-performance --benchmark-plan "${PLAN}"
run_nsys_log
build_test_artifacts
