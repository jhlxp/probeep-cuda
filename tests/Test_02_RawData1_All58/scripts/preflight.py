#!/usr/bin/env python3
"""Fail-closed H20/RDMA/source preflight, executed once per global GPU rank."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def capture(argv: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {"argv": argv, "returncode": result.returncode, "output": result.stdout.strip()}
    except FileNotFoundError as error:
        return {"argv": argv, "returncode": 127, "output": str(error)}


def git_tree(root: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "tree": capture(["git", "rev-parse", f"HEAD:{relative}"], cwd=root),
        "status": capture(["git", "status", "--short", "--", relative], cwd=root),
    }


def inspect_nvshmem() -> dict[str, Any]:
    root_value = os.environ.get("NVSHMEM_DIR")
    if root_value:
        root = Path(root_value)
    else:
        spec = importlib.util.find_spec("nvidia.nvshmem")
        locations = list(spec.submodule_search_locations or []) if spec else []
        root = Path(locations[0]) if locations else Path("/missing/nvshmem")
    header = root / "include/nvshmem.h"
    host_library = root / "lib/libnvshmem_host.so"
    transports = sorted(path.name for path in (root / "lib").glob("nvshmem_transport_*.so*"))
    payload = {
        "root": str(root),
        "header": str(header),
        "host_library": str(host_library),
        "transports": transports,
    }
    return {
        "argv": ["inspect-nvshmem", str(root)],
        "returncode": 0 if header.is_file() and host_library.is_file() else 1,
        "output": json.dumps(payload, sort_keys=True),
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    node_rank = rank // local_world_size
    expected_nodes = int(os.environ["NNODES"])
    expected_local = int(os.environ["GPUS_PER_NODE"])
    num_experts = int(os.environ.get("NUM_EXPERTS", "256"))
    root = Path(os.environ["PROBEEP_ROOT"]).resolve()
    run_dir = Path(os.environ["PROBEEP_RUN_DIR"]).resolve()
    failures: list[str] = []

    if world_size != expected_nodes * expected_local:
        failures.append(f"world_size={world_size}, expected {expected_nodes * expected_local}")
    if local_world_size != expected_local:
        failures.append(f"LOCAL_WORLD_SIZE={local_world_size}, expected {expected_local}")
    if torch.cuda.device_count() != expected_local:
        failures.append(f"visible CUDA devices={torch.cuda.device_count()}, expected {expected_local}")
    if expected_local != 8:
        failures.append(f"GPUS_PER_NODE={expected_local}, current kernels require NVL8")
    if num_experts != 256 or num_experts % world_size:
        failures.append(f"DSV3 requires E=256 divisible by world_size, got E={num_experts}, R={world_size}")
    if os.environ.get("NVSHMEM_IB_ENABLE_IBGDA") != "1":
        failures.append("NVSHMEM_IB_ENABLE_IBGDA must be 1")
    if os.environ.get("NVSHMEM_DISABLE_CUDA_VMM") != "1":
        failures.append("NVSHMEM_DISABLE_CUDA_VMM must be 1 for this DeepEP setup")
    if os.environ.get("NVSHMEM_REMOTE_TRANSPORT") != "ibrc":
        failures.append("NVSHMEM_REMOTE_TRANSPORT must be ibrc")
    if not os.environ.get("NVSHMEM_HCA_LIST"):
        failures.append("NVSHMEM_HCA_LIST must be set from the target topology")
    if not os.environ.get("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"):
        failures.append("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME must be set")
    if os.environ.get("NCCL_IB_DISABLE") == "1":
        failures.append("NCCL_IB_DISABLE=1 is invalid for the multi-node run")
    if not os.environ.get("NCCL_SOCKET_IFNAME"):
        failures.append("NCCL_SOCKET_IFNAME must be set from the target topology")
    if not os.environ.get("NCCL_IB_HCA"):
        failures.append("NCCL_IB_HCA must be set from the target topology")

    physical_nics = int(os.environ.get("PROBEEP_PHYSICAL_NICS_PER_SERVER", "4"))
    rails_per_nic = int(os.environ.get("PROBEEP_RAILS_PER_PHYSICAL_NIC", "2"))
    physical_nic_gbps = float(
        os.environ.get("PROBEEP_PHYSICAL_NIC_BANDWIDTH_GBPS", "400")
    )
    rail_gbps = float(
        os.environ.get(
            "PROBEEP_RDMA_PATH_BANDWIDTH_GBPS",
            os.environ.get("RDMA_PATH_BANDWIDTH_GBPS", "200"),
        )
    )
    if physical_nics != 4 or rails_per_nic != 2:
        failures.append(
            "target H20 topology requires 4 physical NICs x 2 rails per NIC"
        )
    if physical_nics * rails_per_nic != expected_local:
        failures.append(
            "physical NIC split does not produce one logical rail per local GPU"
        )
    if abs(physical_nic_gbps - 400.0) > 1e-6 or abs(rail_gbps - 200.0) > 1e-6:
        failures.append(
            "target H20 topology requires 400 Gbps/physical NIC and 200 Gbps/rail"
        )
    if abs(rail_gbps * rails_per_nic - physical_nic_gbps) > 1e-6:
        failures.append("logical rail bandwidth exceeds physical NIC capacity")
    nvshmem_hcas = [
        item.strip()
        for item in os.environ.get("NVSHMEM_HCA_LIST", "").split(",")
        if item.strip()
    ]
    nccl_hcas = [
        item.strip()
        for item in os.environ.get("NCCL_IB_HCA", "").split(",")
        if item.strip()
    ]
    if len(nvshmem_hcas) != physical_nics:
        failures.append(
            f"NVSHMEM_HCA_LIST exposes {len(nvshmem_hcas)} HCA ports, "
            f"expected {physical_nics}"
        )
    if len(nccl_hcas) != physical_nics:
        failures.append(
            f"NCCL_IB_HCA exposes {len(nccl_hcas)} HCA devices, "
            f"expected {physical_nics}"
        )

    properties = torch.cuda.get_device_properties(local_rank)
    free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
    minimum_free_bytes = int(
        float(os.environ.get("PROBEEP_MIN_FREE_GPU_GIB", "48")) * (1024 ** 3)
    )
    if free_bytes < minimum_free_bytes:
        failures.append(
            f"rank {rank}: free GPU memory={free_bytes / (1024 ** 3):.1f} GiB, "
            f"required>={minimum_free_bytes / (1024 ** 3):.1f} GiB"
        )
    required_name = os.environ.get("PROBEEP_REQUIRE_GPU_NAME", "H20")
    if required_name not in properties.name:
        failures.append(f"rank {rank}: GPU {properties.name!r} does not contain {required_name!r}")
    if (properties.major, properties.minor) != (9, 0):
        failures.append(f"rank {rank}: compute capability is {properties.major}.{properties.minor}, expected 9.0")

    reduced = torch.tensor(rank + 1, dtype=torch.int64, device="cuda")
    dist.all_reduce(reduced)
    expected_sum = world_size * (world_size + 1) // 2
    if int(reduced.item()) != expected_sum:
        failures.append(f"NCCL all-reduce={int(reduced.item())}, expected {expected_sum}")

    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None:
        uuid_result = capture([
            "nvidia-smi", "-i", str(local_rank), "--query-gpu=uuid", "--format=csv,noheader"
        ])
        device_uuid = uuid_result["output"] if uuid_result["returncode"] == 0 else None
    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "hostname": socket.gethostname(),
        "gpu_uuid": str(device_uuid) if device_uuid is not None else None,
        "free_memory_bytes": free_bytes,
        "total_memory_bytes": total_bytes,
    }
    identities: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(identities, identity)
    if rank == 0:
        valid_identities = [item for item in identities if item is not None]
        uuids = [item["gpu_uuid"] for item in valid_identities]
        if any(value is None for value in uuids) or len(set(uuids)) != world_size:
            failures.append("global GPU UUIDs are missing or not unique")
        for expected_node_rank in range(expected_nodes):
            chunk = valid_identities[
                expected_node_rank * expected_local : (expected_node_rank + 1) * expected_local
            ]
            if len({item["hostname"] for item in chunk}) != 1:
                failures.append(f"global ranks for node {expected_node_rank} are not node-major")
            if sorted(item["local_rank"] for item in chunk) != list(range(expected_local)):
                failures.append(f"node {expected_node_rank} local ranks are not 0..{expected_local - 1}")
            if {item["node_rank"] for item in chunk} != {expected_node_rank}:
                failures.append(f"node-rank metadata is wrong for global-rank chunk {expected_node_rank}")

    node_record: dict[str, Any] | None = None
    if local_rank == 0:
        commands = {
            "nvidia_smi": capture(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
            "nvidia_topology": capture(["nvidia-smi", "topo", "-m"]),
            "nvidia_p2p": capture(["nvidia-smi", "topo", "-p2p", "r"]),
            "nvcc": capture(["nvcc", "--version"]),
            "ibv_devices": capture(["ibv_devices"]),
            "ibv_devinfo": capture(["ibv_devinfo", "-v"]),
            "ibdev2netdev": capture(["ibdev2netdev"]),
            "nvidia_peermem": capture(["bash", "-lc", "lsmod | rg '^nvidia_peermem' || true"]),
            "nvshmem": inspect_nvshmem(),
            "nsys": capture(["nsys", "--version"]),
        }
        for name in (
            "nvidia_smi", "nvidia_topology", "nvidia_p2p", "nvcc",
            "ibv_devices", "ibv_devinfo", "ibdev2netdev", "nvshmem", "nsys",
        ):
            if commands[name]["returncode"] != 0:
                failures.append(f"{name} failed: {commands[name]['output']}")
        if "PORT_ACTIVE" not in commands["ibv_devinfo"]["output"]:
            failures.append("ibv_devinfo reports no PORT_ACTIVE device")
        if "Up" not in commands["ibdev2netdev"]["output"]:
            failures.append("ibdev2netdev reports no Up netdev")

        source_lock = json.loads(Path(__file__).with_name("source_lock.json").read_text())
        sources = {
            name: git_tree(root, relative)
            for name, relative in {
                "deepep": "src/deepep",
                "deepep_moonep": "src/deepep-moonep",
                "deepep_probeep": "src/deepep-probeep",
                "ultraep": "src/ultraep",
                "hybridep": "src/ultraep/HybridEP",
            }.items()
        }
        for name, source in sources.items():
            if source["tree"]["returncode"] != 0:
                failures.append(f"cannot resolve locked source tree {name}: {source['tree']['output']}")
        actual_hybrid_tree = sources["hybridep"]["tree"]["output"]
        expected_hybrid_tree = source_lock["ultraep_hybridep"]["vendored_git_tree"]
        if actual_hybrid_tree != expected_hybrid_tree:
            failures.append(f"HybridEP tree={actual_hybrid_tree}, locked={expected_hybrid_tree}")

        # Local env files, build products and test_logs are allowed to be
        # untracked. Formal runs reject modifications to tracked source.
        repo_status = capture(
            ["git", "status", "--short", "--untracked-files=no"], cwd=root
        )
        if os.environ.get("PROBEEP_REQUIRE_CLEAN_TREE", "0") == "1" and repo_status["output"]:
            failures.append("repository is dirty while PROBEEP_REQUIRE_CLEAN_TREE=1")

        devices = []
        for index in range(torch.cuda.device_count()):
            item = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": item.name,
                "compute_capability": [item.major, item.minor],
                "total_memory": item.total_memory,
                "multi_processor_count": item.multi_processor_count,
            })
        selected_environment = {
            key: value for key, value in os.environ.items()
            if key.startswith(("PROBEEP_", "NCCL_", "NVSHMEM_", "CUDA_"))
            or key in {"MASTER_ADDR", "MASTER_PORT", "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_STEP_GPUS"}
        }
        node_record = {
            "schema": "probeep.h20.preflight.node.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": os.environ["PROBEEP_RUN_ID"],
            "node_rank": node_rank,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "devices": devices,
            "nccl_collective": {"value": int(reduced.item()), "expected": expected_sum},
            "rank_identities": identities,
            "environment": selected_environment,
            "commands": commands,
            "repository": {
                "head": capture(["git", "rev-parse", "HEAD"], cwd=root),
                "status": repo_status,
                "sources": sources,
                "source_lock": source_lock,
            },
            "failures": failures,
            "status": "PASS" if not failures else "FAIL",
        }
        (run_dir / "preflight").mkdir(parents=True, exist_ok=True)
        (run_dir / "preflight" / f"node-{node_rank}.json").write_text(
            json.dumps(node_record, ensure_ascii=False, indent=2) + "\n"
        )

    gathered_failures: list[list[str] | None] = [None] * world_size
    gathered_nodes: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered_failures, failures)
    dist.all_gather_object(gathered_nodes, node_record)
    all_failures = [
        f"rank {index}: {message}"
        for index, messages in enumerate(gathered_failures)
        for message in (messages or [])
    ]
    if rank == 0:
        summary = {
            "schema": "probeep.h20.preflight.summary.v1",
            "run_id": os.environ["PROBEEP_RUN_ID"],
            "status": "PASS" if not all_failures else "FAIL",
            "topology": {
                "num_nodes": expected_nodes,
                "gpus_per_node": expected_local,
                "world_size": world_size,
                "num_experts": num_experts,
                "experts_per_rank": num_experts // world_size if num_experts % world_size == 0 else None,
            },
            "nodes": [record for record in gathered_nodes if record is not None],
            "failures": all_failures,
            "note": "NCCL collective proves distributed reachability, not ProbeEP IBGDA data-path use; run deepep-smoke next.",
        }
        (run_dir / "preflight" / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    dist.destroy_process_group()
    if all_failures:
        raise RuntimeError("preflight failed:\n" + "\n".join(all_failures))


if __name__ == "__main__":
    main()
