"""Load the exact 58-layer DSV3 MoE receive-count trace used by ProbeEP."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


class RawReceiveError(ValueError):
    """Raised when raw receive data or a requested lowering is invalid."""


def runtime_tree_sha256(directory: Path | str) -> str:
    """Hash the JSON/CSV runtime inputs using the generator's wire format."""

    root = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(
        (*root.glob("*.json"), *root.glob("*.csv")),
        key=lambda item: item.name,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RawReceiveDataset:
    placement_path: Path
    csv_pattern: str
    num_raw_ranks: int
    num_layers: int
    physical_slots_per_raw_rank: int
    num_logical_experts: int
    logical_loads: tuple[tuple[int, ...], ...]
    total_receive_by_layer: tuple[int, ...]

    @classmethod
    def load(
        cls,
        placement_path: Path | str,
        *,
        csv_pattern: str = "decode_{rank}.csv",
    ) -> "RawReceiveDataset":
        placement_path = Path(placement_path).resolve()
        if not placement_path.is_file():
            raise RawReceiveError(f"missing raw placement JSON: {placement_path}")
        try:
            root = json.loads(placement_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RawReceiveError(
                f"invalid raw placement JSON: {placement_path}"
            ) from exc

        layers = root.get("layer_list") if isinstance(root, dict) else None
        if not isinstance(layers, list) or not layers:
            raise RawReceiveError("raw placement JSON needs a non-empty layer_list")
        if root.get("moe_layer_count") != len(layers):
            raise RawReceiveError(
                "raw placement moe_layer_count disagrees with layer_list"
            )

        first_devices = layers[0].get("device_list")
        if not isinstance(first_devices, list) or not first_devices:
            raise RawReceiveError("raw placement layer needs a device_list")
        num_raw_ranks = len(first_devices)
        first_slots = first_devices[0].get("device_expert")
        if not isinstance(first_slots, list) or not first_slots:
            raise RawReceiveError("raw placement device needs device_expert slots")
        slots_per_rank = len(first_slots)

        mappings: list[list[list[int]]] = []
        expert_ids: set[int] = set()
        for expected_layer, layer in enumerate(layers):
            if layer.get("layer_id") != expected_layer:
                raise RawReceiveError("raw placement layer IDs must be contiguous")
            devices = layer.get("device_list")
            if not isinstance(devices, list) or len(devices) != num_raw_ranks:
                raise RawReceiveError(
                    "raw placement device count changed across layers"
                )
            by_rank: list[list[int] | None] = [None] * num_raw_ranks
            for device in devices:
                rank = device.get("device_id") if isinstance(device, dict) else None
                slots = (
                    device.get("device_expert")
                    if isinstance(device, dict)
                    else None
                )
                if not isinstance(rank, int) or not 0 <= rank < num_raw_ranks:
                    raise RawReceiveError("raw placement has invalid device_id")
                if by_rank[rank] is not None:
                    raise RawReceiveError("raw placement has duplicate device_id")
                if not isinstance(slots, list) or len(slots) != slots_per_rank:
                    raise RawReceiveError(
                        "raw placement slot count changed across devices"
                    )
                if any(
                    not isinstance(expert, int) or expert < 0 for expert in slots
                ):
                    raise RawReceiveError(
                        "raw placement expert IDs must be non-negative integers"
                    )
                by_rank[rank] = list(slots)
                expert_ids.update(slots)
            if any(slots is None for slots in by_rank):
                raise RawReceiveError("raw placement is missing a device")
            mappings.append([slots for slots in by_rank if slots is not None])

        if not expert_ids or expert_ids != set(range(max(expert_ids) + 1)):
            raise RawReceiveError(
                "raw placement logical expert IDs must be contiguous"
            )
        num_logical_experts = max(expert_ids) + 1

        receives: list[list[list[int]]] = []
        for rank in range(num_raw_ranks):
            csv_path = placement_path.parent / csv_pattern.format(rank=rank)
            if not csv_path.is_file():
                raise RawReceiveError(f"missing raw receive CSV: {csv_path}")
            rows: list[list[int]] = []
            try:
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    for row_index, row in enumerate(csv.reader(handle), start=1):
                        if len(row) != slots_per_rank:
                            raise RawReceiveError(
                                f"{csv_path}:{row_index}: expected "
                                f"{slots_per_rank} columns"
                            )
                        try:
                            values = [int(value) for value in row]
                        except ValueError as exc:
                            raise RawReceiveError(
                                f"{csv_path}:{row_index}: counts must be integers"
                            ) from exc
                        if any(value < 0 for value in values):
                            raise RawReceiveError(
                                f"{csv_path}:{row_index}: counts must be non-negative"
                            )
                        rows.append(values)
            except OSError as exc:
                raise RawReceiveError(
                    f"failed to read raw receive CSV: {csv_path}"
                ) from exc
            if len(rows) != len(layers):
                raise RawReceiveError(
                    f"{csv_path}: expected {len(layers)} rows, found {len(rows)}"
                )
            receives.append(rows)

        logical_loads: list[tuple[int, ...]] = []
        totals: list[int] = []
        for layer in range(len(layers)):
            loads = [0] * num_logical_experts
            physical_total = 0
            for rank in range(num_raw_ranks):
                for slot in range(slots_per_rank):
                    count = receives[rank][layer][slot]
                    expert = mappings[layer][rank][slot]
                    loads[expert] += count
                    physical_total += count
            if sum(loads) != physical_total:
                raise RawReceiveError(
                    "raw physical-to-logical folding lost receive counts"
                )
            logical_loads.append(tuple(loads))
            totals.append(physical_total)

        return cls(
            placement_path=placement_path,
            csv_pattern=csv_pattern,
            num_raw_ranks=num_raw_ranks,
            num_layers=len(layers),
            physical_slots_per_raw_rank=slots_per_rank,
            num_logical_experts=num_logical_experts,
            logical_loads=tuple(logical_loads),
            total_receive_by_layer=tuple(totals),
        )

@dataclass(frozen=True, slots=True)
class FullDSV3MoETrace:
    """Deterministically scaled expert rows for all 58 MoE layers."""

    expert_rows: np.ndarray
    static_rank_rows: np.ndarray
    static_server_rows: np.ndarray
    moonep_local_target_rows: np.ndarray
    probeep_global_target_rows: np.ndarray
    probeep_cross_server_rows: np.ndarray
    fidelity: str = "largest_remainder_scaled_receive_distribution"

    @property
    def num_layers(self) -> int:
        return int(self.expert_rows.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.expert_rows.shape[1])

    @property
    def total_rows(self) -> int:
        return int(self.expert_rows.sum())

    def summary(self) -> dict[str, object]:
        return {
            "fidelity": self.fidelity,
            "num_moe_layers": self.num_layers,
            "num_experts": self.num_experts,
            "rows_per_layer": self.expert_rows.sum(axis=1).tolist(),
            "total_rows_all_layers": self.total_rows,
            "static_rank_max_mean_by_layer": (
                self.static_rank_rows.max(axis=1)
                / self.static_rank_rows.mean(axis=1)
            ).tolist(),
            "static_server_max_mean_by_layer": (
                self.static_server_rows.max(axis=1)
                / self.static_server_rows.mean(axis=1)
            ).tolist(),
            "cross_server_rows_by_layer": self.probeep_cross_server_rows.tolist(),
        }


def _equal_partition(total: int, parts: int) -> np.ndarray:
    quotient, remainder = divmod(total, parts)
    result = np.full(parts, quotient, dtype=np.int64)
    result[:remainder] += 1
    return result


def _largest_remainder_scale(values: tuple[int, ...], total: int) -> np.ndarray:
    source_total = sum(values)
    if source_total <= 0 or total <= 0:
        raise RawReceiveError("cannot scale an empty receive histogram")
    numerators = [int(value) * total for value in values]
    scaled = [numerator // source_total for numerator in numerators]
    remainder = total - sum(scaled)
    order = sorted(
        range(len(values)),
        key=lambda expert: (-(numerators[expert] % source_total), expert),
    )
    for expert in order[:remainder]:
        scaled[expert] += 1
    return np.asarray(scaled, dtype=np.int64)


def load_full_dsv3_moe_trace(
    dataset: RawReceiveDataset,
    *,
    num_model_ranks: int = 16,
    experts_per_rank: int | None = None,
    ranks_per_logical_server: int = 8,
    tokens_per_rank: int = 4096,
    topk: int = 8,
) -> FullDSV3MoETrace:
    """Scale and map the 58 256-expert histograms onto H20 EP ranks."""
    if num_model_ranks not in (16, 32) or ranks_per_logical_server != 8:
        raise RawReceiveError("supported topologies are 2x8/EP16 and 4x8/EP32")
    if min(tokens_per_rank, topk) <= 0:
        raise RawReceiveError("tokens_per_rank and topk must be positive")
    if experts_per_rank is None:
        experts_per_rank = dataset.num_logical_experts // num_model_ranks
    if dataset.num_logical_experts != num_model_ranks * experts_per_rank:
        raise RawReceiveError("256 experts must divide evenly over model ranks")
    if dataset.num_layers != 58:
        raise RawReceiveError("the full DSV3 trace must contain 58 MoE layers")

    target_rows = num_model_ranks * tokens_per_rank * topk
    expert_rows = np.stack(
        [
            _largest_remainder_scale(layer, target_rows)
            for layer in dataset.logical_loads
        ]
    )
    static_rank_rows = expert_rows.reshape(
        dataset.num_layers, num_model_ranks, experts_per_rank
    ).sum(axis=2)
    static_server_rows = static_rank_rows.reshape(
        dataset.num_layers,
        num_model_ranks // ranks_per_logical_server,
        ranks_per_logical_server,
    ).sum(axis=2)

    local_target = np.empty_like(static_rank_rows)
    global_target = np.empty_like(static_rank_rows)
    cross_server = np.empty(dataset.num_layers, dtype=np.int64)
    num_servers = num_model_ranks // ranks_per_logical_server
    for layer in range(dataset.num_layers):
        for server in range(num_servers):
            begin = server * ranks_per_logical_server
            end = begin + ranks_per_logical_server
            local_target[layer, begin:end] = _equal_partition(
                int(static_server_rows[layer, server]), ranks_per_logical_server
            )
        global_target[layer] = _equal_partition(
            int(expert_rows[layer].sum()), num_model_ranks
        )
        server_target = _equal_partition(
            int(expert_rows[layer].sum()), num_servers
        )
        cross_server[layer] = sum(
            max(
                0,
                int(static_server_rows[layer, server])
                - int(server_target[server]),
            )
            for server in range(num_servers)
        )

    for array in (
        expert_rows,
        static_rank_rows,
        static_server_rows,
        local_target,
        global_target,
        cross_server,
    ):
        array.setflags(write=False)
    return FullDSV3MoETrace(
        expert_rows=expert_rows,
        static_rank_rows=static_rank_rows,
        static_server_rows=static_server_rows,
        moonep_local_target_rows=local_target,
        probeep_global_target_rows=global_target,
        probeep_cross_server_rows=cross_server,
    )


__all__ = [
    "FullDSV3MoETrace",
    "RawReceiveDataset",
    "RawReceiveError",
    "load_full_dsv3_moe_trace",
    "runtime_tree_sha256",
]
