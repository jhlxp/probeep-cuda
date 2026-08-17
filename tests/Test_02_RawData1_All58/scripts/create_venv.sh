#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBEEP_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROBEEP_TEST_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROBEEP_TEST_DIR}/.venv-h20}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:?Set TORCH_INDEX_URL for the target H20 CUDA stack}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install ninja packaging numpy pytest pandas matplotlib
"${VENV_DIR}/bin/python" -m pip install --index-url "${TORCH_INDEX_URL}" torch
"${VENV_DIR}/bin/python" -m pip install 'nvidia-nvshmem-cu12>=3.3.9'

"${VENV_DIR}/bin/python" - <<'PY'
import importlib.util
import torch

assert torch.cuda.is_available(), "CUDA is unavailable in the H20 venv"
assert torch.cuda.device_count() == 8, "each target node must expose exactly 8 GPUs"
devices = [torch.cuda.get_device_properties(i) for i in range(8)]
assert all("H20" in item.name for item in devices), [item.name for item in devices]
assert all((item.major, item.minor) == (9, 0) for item in devices)
assert importlib.util.find_spec("nvidia.nvshmem"), "nvidia.nvshmem is unavailable"
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpus={len(devices)} model={devices[0].name} sm={devices[0].major}{devices[0].minor}"
)
PY

echo "venv: ${VENV_DIR}"
