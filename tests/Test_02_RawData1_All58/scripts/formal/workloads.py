"""Deterministic routing workloads shared by correctness and performance tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch

from workload.gate.raw_receive import RawReceiveDataset, load_full_dsv3_moe_trace


@dataclass(frozen=True)
class RoutingWorkload:
    topk_experts: torch.Tensor
    topk_weights: torch.Tensor
    mode: str
    bias_ratio: float
    seed: int
    sha256: str


def routing_sha256(topk_experts: torch.Tensor, topk_weights: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(bytes(topk_experts.cpu().contiguous().untyped_storage()))
    digest.update(bytes(topk_weights.cpu().contiguous().untyped_storage()))
    return digest.hexdigest()


def _normalized_weights(shape: tuple[int, int, int], seed: int) -> torch.Tensor:
    if os.environ.get("PROBEEP_ROUTE_FILE"):
        return torch.full(shape, 1.0 / shape[-1], dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 0x5EED)
    weights = torch.rand(shape, generator=generator, dtype=torch.float32)
    return weights / weights.sum(dim=-1, keepdim=True)


def balanced_topk(
    *,
    world_size: int = 16,
    num_tokens: int = 4096,
    topk: int = 8,
    local_experts: int = 16,
    ranks_per_server: int = 8,
) -> torch.Tensor:
    """Round-robin routes with equal expert and server totals."""

    num_servers = world_size // ranks_per_server
    if topk % num_servers:
        raise ValueError("topk must be divisible by the number of servers")
    routes_per_server = topk // num_servers
    experts_per_server = ranks_per_server * local_experts

    token_number = torch.arange(
        world_size * num_tokens, dtype=torch.int64
    ).view(world_size, num_tokens, 1)
    within_server = (
        token_number * routes_per_server
        + torch.arange(routes_per_server, dtype=torch.int64).view(1, 1, -1)
    ) % experts_per_server
    server_offsets = (
        torch.arange(num_servers, dtype=torch.int64) * experts_per_server
    ).view(1, 1, num_servers, 1)
    return (within_server.unsqueeze(2) + server_offsets).reshape(
        world_size, num_tokens, topk
    )


def server_preserving_skew_topk(
    *,
    world_size: int = 16,
    num_tokens: int = 4096,
    topk: int = 8,
    local_experts: int = 16,
    ranks_per_server: int = 8,
    bias_ratio: float = 1.0,
    seed: int = 1234,
) -> torch.Tensor:
    """Sample without replacement while keeping equal routes per server.

    Gumbel top-k gives a vectorized Plackett-Luce sample and avoids a Python
    loop over 65,536 tokens.
    """

    num_servers = world_size // ranks_per_server
    if topk % num_servers:
        raise ValueError("topk must be divisible by the number of servers")
    routes_per_server = topk // num_servers
    experts_per_server = ranks_per_server * local_experts
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    result = []
    for server in range(num_servers):
        log_popularity = torch.randn(
            experts_per_server, generator=generator, dtype=torch.float32
        ) * float(bias_ratio)
        uniform = torch.rand(
            (world_size, num_tokens, experts_per_server),
            generator=generator,
            dtype=torch.float32,
        ).clamp_(1e-7, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(uniform))
        chosen = torch.topk(
            log_popularity.view(1, 1, -1) + gumbel,
            k=routes_per_server,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        result.append(chosen + server * experts_per_server)
    return torch.cat(result, dim=-1).to(torch.int64)


def server_imbalanced_topk(
    *,
    world_size: int = 16,
    num_tokens: int = 4096,
    topk: int = 8,
    local_experts: int = 16,
    ranks_per_server: int = 8,
    bias_ratio: float = 0.75,
    seed: int = 1234,
) -> torch.Tensor:
    """Create deterministic server imbalance without duplicate experts.

    ``bias_ratio`` is the fraction of every token's top-k routed to server 0.
    For the EP16/top-k=8 target, 0.75 is an exact 6/2 split.  The expert
    popularity inside each server is also skewed, so the same input exercises
    ProbeEP's server-first and MoonEP's server-local stages.
    """

    num_servers = world_size // ranks_per_server
    if num_servers != 2:
        raise ValueError("server_imbalanced currently requires exactly two servers")
    routes_server0 = int(round(topk * float(bias_ratio)))
    routes_server0 = max(0, min(topk, routes_server0))
    routes_by_server = (routes_server0, topk - routes_server0)
    experts_per_server = ranks_per_server * local_experts
    if any(routes > experts_per_server for routes in routes_by_server):
        raise ValueError("top-k server share exceeds the server expert count")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = []
    for server, routes in enumerate(routes_by_server):
        if routes == 0:
            continue
        log_popularity = torch.randn(
            experts_per_server, generator=generator, dtype=torch.float32
        )
        uniform = torch.rand(
            (world_size, num_tokens, experts_per_server),
            generator=generator,
            dtype=torch.float32,
        ).clamp_(1e-7, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(uniform))
        chosen = torch.topk(
            log_popularity.view(1, 1, -1) + gumbel,
            k=routes,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        result.append(chosen + server * experts_per_server)
    return torch.cat(result, dim=-1).to(torch.int64)


def raw_data1_layer_topk(
    layer: int,
    *,
    world_size: int = 16,
    num_tokens: int = 4096,
    topk: int = 8,
    local_experts: int = 16,
    ranks_per_server: int = 8,
    seed: int = 1234,
) -> torch.Tensor:
    """Realize one exact raw_data1 histogram as unique per-token TopK routes.

    The source trace contains receive counts, not the original token-to-expert
    matrix. This deterministic bipartite realization preserves every expert
    count exactly while making no source-token fidelity claim. It runs only
    during test-data preparation, outside the timed CUDA path.
    """

    materialized = os.environ.get("PROBEEP_ROUTE_FILE")
    if materialized:
        route_path = Path(materialized).resolve()
        routes = np.load(route_path, allow_pickle=False)
        expected = (world_size, num_tokens, topk)
        if routes.shape != expected or routes.dtype != np.int16:
            raise ValueError(
                f"materialized route {route_path} is {routes.shape}/{routes.dtype}, "
                f"expected {expected}/int16"
            )
        digest = hashlib.sha256(route_path.read_bytes()).hexdigest()
        locked = os.environ.get("PROBEEP_ROUTING_SHA256")
        if locked and digest != locked:
            raise ValueError(
                f"materialized route SHA-256 {digest} does not match plan {locked}"
            )
        return torch.from_numpy(routes.astype(np.int64, copy=True))

    project_root = Path(
        os.environ.get("PROBEEP_ROOT", Path(__file__).resolve().parents[4])
    ).resolve()
    placement = project_root / "workload/raw_data1/DSV3_32x8_256_unique.json"
    dataset = RawReceiveDataset.load(placement)
    if not 0 <= layer < dataset.num_layers:
        raise ValueError(f"raw_data1 layer must be in [0,{dataset.num_layers})")
    if dataset.num_logical_experts != world_size * local_experts:
        raise ValueError("raw_data1 E256 does not match world_size*local_experts")
    trace = load_full_dsv3_moe_trace(
        dataset,
        num_model_ranks=world_size,
        experts_per_rank=local_experts,
        ranks_per_logical_server=ranks_per_server,
        tokens_per_rank=num_tokens,
        topk=topk,
    )
    counts = trace.expert_rows[layer].tolist()
    num_global_tokens = world_size * num_tokens
    if max(counts) > num_global_tokens:
        raise ValueError("an expert count exceeds the unique-token capacity")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer * 0x9E3779B1)
    buckets: list[list[int]] = [
        torch.randperm(num_global_tokens, generator=generator).tolist()
    ] + [[] for _ in range(topk)]
    cursors = [0] * (topk + 1)
    routes = torch.empty((num_global_tokens, topk), dtype=torch.int64)

    # Bipartite Havel-Hakimi: each expert consumes tokens with the largest
    # remaining capacity. Newly selected tokens are withheld until that expert
    # is complete, so one token cannot contain the same expert twice.
    for expert in sorted(range(len(counts)), key=lambda item: (-counts[item], item)):
        remaining = int(counts[expert])
        moved: list[tuple[int, list[int]]] = []
        for current_load in range(topk):
            available = len(buckets[current_load]) - cursors[current_load]
            take = min(remaining, available)
            if take:
                begin = cursors[current_load]
                selected = buckets[current_load][begin : begin + take]
                cursors[current_load] += take
                routes[selected, current_load] = expert
                moved.append((current_load + 1, selected))
                remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise ValueError(
                f"raw_data1 layer {layer} cannot be realized without duplicate TopK"
            )
        for next_load, selected in moved:
            buckets[next_load].extend(selected)

    if len(buckets[topk]) - cursors[topk] != num_global_tokens:
        raise AssertionError("raw_data1 realization did not fill every TopK row")
    return routes.view(world_size, num_tokens, topk)


def make_routing_workload(
    mode: str,
    *,
    world_size: int = 16,
    num_tokens: int = 4096,
    topk: int = 8,
    local_experts: int = 16,
    ranks_per_server: int = 8,
    bias_ratio: float = 1.0,
    seed: int = 1234,
) -> RoutingWorkload:
    kwargs = dict(
        world_size=world_size,
        num_tokens=num_tokens,
        topk=topk,
        local_experts=local_experts,
        ranks_per_server=ranks_per_server,
    )
    if mode == "balanced":
        topk_experts = balanced_topk(**kwargs)
        effective_bias = 0.0
    elif mode == "server_preserving_skew":
        topk_experts = server_preserving_skew_topk(
            **kwargs, bias_ratio=bias_ratio, seed=seed
        )
        effective_bias = float(bias_ratio)
    elif mode == "server_imbalanced":
        topk_experts = server_imbalanced_topk(
            **kwargs, bias_ratio=bias_ratio, seed=seed
        )
        effective_bias = float(bias_ratio)
    elif mode.startswith("raw_data1_layer_"):
        try:
            layer = int(mode.removeprefix("raw_data1_layer_"))
        except ValueError as exc:
            raise ValueError(f"invalid raw_data1 workload: {mode}") from exc
        topk_experts = raw_data1_layer_topk(layer, **kwargs, seed=seed)
        effective_bias = 0.0
        mode = f"raw_data1_layer_{layer:02d}"
    else:
        raise ValueError(f"unknown routing mode: {mode}")

    topk_weights = _normalized_weights(tuple(topk_experts.shape), seed)
    workload_digest = (
        os.environ.get("PROBEEP_ROUTING_SHA256")
        if os.environ.get("PROBEEP_ROUTE_FILE")
        else routing_sha256(topk_experts, topk_weights)
    )
    return RoutingWorkload(
        topk_experts=topk_experts,
        topk_weights=topk_weights,
        mode=mode,
        bias_ratio=effective_bias,
        seed=seed,
        sha256=str(workload_digest),
    )
