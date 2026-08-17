#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

#include "../moonep_runtime.hpp"
#include "api.cuh"
#include "probeep_plan.cuh"
#include "probeep_weight_transport.cuh"

namespace deep_ep::probeep {
namespace {

constexpr int kThreads = 256;
constexpr int kAdmissionThreads = 32;
constexpr int kHistogramMaxBlocks = 128;
constexpr int kHistogramSegmentRoutes = 256;
constexpr int kPackedOrdinalBits = 16;
constexpr int kPackedOrdinalMask = (1 << kPackedOrdinalBits) - 1;
constexpr int kIntentFields = 6;

struct PlanConfig {
    int world_size;
    int num_servers;
    int num_tokens_per_rank;
    int local_experts;
    int execution_slots;
    int token_padding;
    std::int64_t dispatch_bytes_per_route;
    std::int64_t expert_weight_bytes;
    std::int64_t weight_chunk_bytes;
};

__device__ __forceinline__ std::int64_t bounded_dispatch_bytes(
        std::int64_t occurrence_bytes, const PlanConfig& cfg) {
    // The histogram does not expose TopK co-occurrence, but one source token is
    // sent at most once to each remote server.  Likewise, one destination relay
    // endpoint receives at most S tokens from the matching lane of each remote
    // server.  Both the occurrence sum and this topology ceiling are safe upper
    // bounds, so their minimum remains conservative without an expanded-route
    // exchange.
    const auto topology_ceiling =
            static_cast<std::int64_t>(cfg.num_tokens_per_rank) *
            max(0, cfg.num_servers - 1) * cfg.dispatch_bytes_per_route;
    return min(max(static_cast<std::int64_t>(0), occurrence_bytes),
               topology_ceiling);
}

struct IntentCandidate {
    int expert = -1;
    int donor = -1;
    int receiver = -1;
    int source_rank = -1;
    int seed_rank = -1;
    int moved = 0;
    int surplus = -1;
    int deficit = -1;
    int reuse = -1;
    int hot_rows = -1;
    int objective_max = INT_MAX;
    int objective_spread = INT_MAX;
    std::int64_t objective_energy = INT64_MAX;
};

__device__ __forceinline__ bool candidate_better(
        const IntentCandidate& candidate,
        const IntentCandidate& incumbent) {
    if (candidate.expert < 0)
        return false;
    if (incumbent.expert < 0)
        return true;
    if (candidate.hot_rows != incumbent.hot_rows)
        return candidate.hot_rows > incumbent.hot_rows;
    if (candidate.deficit != incumbent.deficit)
        return candidate.deficit > incumbent.deficit;
    if (candidate.reuse != incumbent.reuse)
        return candidate.reuse > incumbent.reuse;
    if (candidate.surplus != incumbent.surplus)
        return candidate.surplus > incumbent.surplus;
    if (candidate.objective_max != incumbent.objective_max)
        return candidate.objective_max < incumbent.objective_max;
    if (candidate.objective_spread != incumbent.objective_spread)
        return candidate.objective_spread < incumbent.objective_spread;
    if (candidate.objective_energy != incumbent.objective_energy)
        return candidate.objective_energy < incumbent.objective_energy;
    if (candidate.expert != incumbent.expert)
        return candidate.expert < incumbent.expert;
    if (candidate.donor != incumbent.donor)
        return candidate.donor < incumbent.donor;
    return candidate.receiver < incumbent.receiver;
}

// Padding refinement is a second compute-only objective after the raw quota.
// It deliberately does not share the quota candidate ordering: the primary
// key is the resulting global padded objective, then the extra padding created
// at the receiver.  Reusing an already-open replica is preferred only after
// those compute keys are equal.
__device__ __forceinline__ bool refinement_candidate_better(
        const IntentCandidate& candidate,
        const IntentCandidate& incumbent) {
    if (candidate.expert < 0)
        return false;
    if (incumbent.expert < 0)
        return true;
    if (candidate.objective_max != incumbent.objective_max)
        return candidate.objective_max < incumbent.objective_max;
    if (candidate.objective_spread != incumbent.objective_spread)
        return candidate.objective_spread < incumbent.objective_spread;
    if (candidate.objective_energy != incumbent.objective_energy)
        return candidate.objective_energy < incumbent.objective_energy;
    // ``surplus`` stores receiver padded growth in this phase.
    if (candidate.surplus != incumbent.surplus)
        return candidate.surplus < incumbent.surplus;
    if (candidate.reuse != incumbent.reuse)
        return candidate.reuse > incumbent.reuse;
    if (candidate.moved != incumbent.moved)
        return candidate.moved < incumbent.moved;
    if (candidate.hot_rows != incumbent.hot_rows)
        return candidate.hot_rows > incumbent.hot_rows;
    if (candidate.expert != incumbent.expert)
        return candidate.expert < incumbent.expert;
    if (candidate.donor != incumbent.donor)
        return candidate.donor < incumbent.donor;
    return candidate.receiver < incumbent.receiver;
}

__device__ __forceinline__ int owner_rank(int expert,
                                           const PlanConfig& cfg) {
    return expert / cfg.local_experts;
}

__device__ __forceinline__ int owner_server(int expert,
                                             const PlanConfig& cfg) {
    return owner_rank(expert, cfg) / kRanksPerServer;
}

__device__ __forceinline__ int padded_rows(int rows, int padding) {
    return rows <= 0 ? 0 : ((rows + padding - 1) / padding) * padding;
}

__device__ __forceinline__ bool objective_less(
        int candidate_max, int candidate_spread,
        std::int64_t candidate_energy,
        int current_max, int current_spread,
        std::int64_t current_energy) {
    return candidate_max < current_max ||
           (candidate_max == current_max &&
            (candidate_spread < current_spread ||
             (candidate_spread == current_spread &&
              candidate_energy < current_energy)));
}

__device__ void server_objective(const int* padded_load,
                                 int num_servers,
                                 int& maximum,
                                 int& spread,
                                 std::int64_t& energy) {
    maximum = 0;
    int minimum = INT_MAX;
    energy = 0;
    for (int server = 0; server < num_servers; ++server) {
        maximum = max(maximum, padded_load[server]);
        minimum = min(minimum, padded_load[server]);
        energy += static_cast<std::int64_t>(padded_load[server]) *
                  padded_load[server];
    }
    spread = maximum - minimum;
}

__global__ void serial_histogram_and_ordinal(
        const std::int64_t* topk_idx,
        int* counts,
        int* ordinal,
        int routes_per_rank,
        int world_size) {
    const int source = static_cast<int>(blockIdx.x);
    if (source >= world_size || threadIdx.x != 0)
        return;
    int running[kNumExperts];
    for (int expert = 0; expert < kNumExperts; ++expert)
        running[expert] = 0;
    const auto offset = static_cast<std::int64_t>(source) * routes_per_rank;
    for (int route = 0; route < routes_per_rank; ++route) {
        const int expert = static_cast<int>(topk_idx[offset + route]);
        ordinal[offset + route] = running[expert]++;
    }
    for (int expert = 0; expert < kNumExperts; ++expert)
        counts[source * kNumExperts + expert] = running[expert];
}

__global__ void gather_ipc_counts(
        void** buffer_ptrs,
        std::int64_t plan_reserve_offset,
        int* global_counts,
        int world_size) {
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < world_size * kNumExperts;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const int source = index / kNumExperts;
        const int expert = index % kNumExperts;
        const int lane = source % kRanksPerServer;
        const int server = source / kRanksPerServer;
        const auto* reserve = reinterpret_cast<const moonep::IpcPlanReserve*>(
                static_cast<const std::uint8_t*>(buffer_ptrs[lane]) +
                plan_reserve_offset);
        global_counts[index] = reserve->source_counts[server][expert];
    }
}

// Recompute the selected expert's conservative route contribution after every
// tentative placement.  The compact [R,E] histogram cannot encode TopK
// co-occurrence, so expert occurrences sharing one destination rank may be
// counted more than once.  This never over-admits network traffic; measured
// de-duplicated DeepEP bytes close the next controller window.
template <bool kAtomic>
__device__ void accumulate_expert_dispatch(
        const int* counts,
        const int* alloc,
        int expert,
        int sign,
        const PlanConfig& cfg,
        std::int64_t* tx,
        std::int64_t* rx) {
    // Source counts and destination allocation describe two sorted
    // partitions of the same expert-occurrence interval.  A two-pointer
    // intersection is exact and O(W); the former nested scan was O(W^2) for
    // every admission trial even though at most 2W-1 pairs can overlap.
    int source = 0;
    int destination = 0;
    int source_begin = 0;
    int destination_begin = 0;
    int source_end = counts[expert];
    int destination_end = alloc[expert * cfg.world_size];
    while (source < cfg.world_size && destination < cfg.world_size) {
        if (source_end == source_begin) {
            ++source;
            source_begin = source_end;
            if (source < cfg.world_size)
                source_end += counts[source * kNumExperts + expert];
            continue;
        }
        if (destination_end == destination_begin) {
            ++destination;
            destination_begin = destination_end;
            if (destination < cfg.world_size)
                destination_end +=
                        alloc[expert * cfg.world_size + destination];
            continue;
        }

        const int overlap = max(0, min(source_end, destination_end) -
                                   max(source_begin, destination_begin));
        const int source_server = source / kRanksPerServer;
        const int destination_server = destination / kRanksPerServer;
        if (overlap > 0 && source_server != destination_server) {
            const auto bytes = static_cast<std::int64_t>(overlap) *
                               cfg.dispatch_bytes_per_route * sign;
            const int destination_endpoint =
                    destination_server * kRanksPerServer +
                    source % kRanksPerServer;
            if constexpr (kAtomic) {
                atomicAdd(reinterpret_cast<unsigned long long*>(tx + source),
                          static_cast<unsigned long long>(bytes));
                atomicAdd(reinterpret_cast<unsigned long long*>(
                                  rx + destination_endpoint),
                          static_cast<unsigned long long>(bytes));
            } else {
                tx[source] += bytes;
                rx[destination_endpoint] += bytes;
            }
        }

        const bool advance_source = source_end <= destination_end;
        const bool advance_destination = destination_end <= source_end;
        if (advance_source) {
            ++source;
            source_begin = source_end;
            if (source < cfg.world_size)
                source_end += counts[source * kNumExperts + expert];
        }
        if (advance_destination) {
            ++destination;
            destination_begin = destination_end;
            if (destination < cfg.world_size)
                destination_end +=
                        alloc[expert * cfg.world_size + destination];
        }
    }
}

__device__ __forceinline__ bool rail_better(
        bool valid,
        std::int64_t pair,
        std::int64_t endpoint,
        std::int64_t endpoint_sum,
        int rail,
        bool other_valid,
        std::int64_t other_pair,
        std::int64_t other_endpoint,
        std::int64_t other_endpoint_sum,
        int other_rail) {
    if (!other_valid)
        return false;
    if (!valid)
        return true;
    return other_pair < pair ||
           (other_pair == pair &&
            (other_endpoint < endpoint ||
             (other_endpoint == endpoint &&
              (other_endpoint_sum < endpoint_sum ||
               (other_endpoint_sum == endpoint_sum &&
                other_rail < rail)))));
}

// One warp trial-schedules a complete expert. Lanes 0..7 evaluate the eight
// rails concurrently for each chunk; lane 0 commits only after every chunk
// has found capacity. This preserves atomic expert admission while removing
// the serial Cx8 rail scan from the control thread.
__device__ bool schedule_complete_expert_warp(
        int expert,
        int replica_id,
        int source_server,
        int destination_server,
        int seed_rank,
        const PlanConfig& cfg,
        const std::int64_t* budgets,
        const std::int64_t* dispatch_tx,
        const std::int64_t* dispatch_rx,
        std::int64_t* assigned_tx,
        std::int64_t* assigned_rx,
        std::int64_t* pair_load,
        std::int64_t* chunk_table,
        int* chunk_count,
        std::int64_t* next_tx,
        std::int64_t* next_rx,
        std::int64_t* next_pair,
        int* selected_rail,
        std::int64_t* source_offset,
        std::int64_t* destination_offset) {
    const int chunks = static_cast<int>(
            (cfg.expert_weight_bytes + cfg.weight_chunk_bytes - 1) /
            cfg.weight_chunk_bytes);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    bool valid = chunks > 0 && chunks <= kProbeMaxChunksPerExpert &&
                 *chunk_count + chunks <= kProbeMaxChunks;
    valid = __shfl_sync(0xffffffffu, valid, 0);
    if (!valid)
        return false;

    for (int rank = lane; rank < cfg.world_size; rank += 32) {
        next_tx[rank] = assigned_tx[rank];
        next_rx[rank] = assigned_rx[rank];
    }
    const int pair_base =
            (source_server * cfg.num_servers + destination_server) *
            kRanksPerServer;
    if (lane < kRanksPerServer)
        next_pair[lane] = pair_load[pair_base + lane];
    __syncwarp();

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const auto expert_offset =
                static_cast<std::int64_t>(chunk) * cfg.weight_chunk_bytes;
        const auto bytes = min(
                cfg.weight_chunk_bytes,
                cfg.expert_weight_bytes - expert_offset);
        int best_rail = lane;
        std::int64_t best_pair = INT64_MAX;
        std::int64_t best_endpoint = INT64_MAX;
        std::int64_t best_sum = INT64_MAX;
        bool rail_valid = lane < kRanksPerServer;
        if (rail_valid) {
            const int source_rank =
                    source_server * kRanksPerServer + lane;
            const int destination_rank =
                    destination_server * kRanksPerServer + lane;
            const auto projected_tx = next_tx[source_rank] + bytes +
                    bounded_dispatch_bytes(dispatch_tx[source_rank], cfg);
            const auto projected_rx = next_rx[destination_rank] + bytes +
                    bounded_dispatch_bytes(dispatch_rx[destination_rank], cfg);
            rail_valid = projected_tx <= budgets[source_rank] &&
                         projected_rx <= budgets[destination_rank];
            best_pair = next_pair[lane] + bytes;
            best_endpoint = max(projected_tx, projected_rx);
            best_sum = projected_tx + projected_rx;
        }
        for (int offset = 4; offset > 0; offset >>= 1) {
            const bool other_valid = __shfl_down_sync(
                    0xffffffffu, rail_valid, offset, kRanksPerServer);
            const auto other_pair = __shfl_down_sync(
                    0xffffffffu, best_pair, offset, kRanksPerServer);
            const auto other_endpoint = __shfl_down_sync(
                    0xffffffffu, best_endpoint, offset, kRanksPerServer);
            const auto other_sum = __shfl_down_sync(
                    0xffffffffu, best_sum, offset, kRanksPerServer);
            const int other_rail = __shfl_down_sync(
                    0xffffffffu, best_rail, offset, kRanksPerServer);
            if (lane + offset < kRanksPerServer && rail_better(
                    rail_valid, best_pair, best_endpoint, best_sum,
                    best_rail, other_valid, other_pair, other_endpoint,
                    other_sum, other_rail)) {
                rail_valid = other_valid;
                best_pair = other_pair;
                best_endpoint = other_endpoint;
                best_sum = other_sum;
                best_rail = other_rail;
            }
        }
        valid = __shfl_sync(0xffffffffu, rail_valid, 0);
        best_rail = __shfl_sync(0xffffffffu, best_rail, 0);
        if (!valid)
            return false;
        if (lane == 0) {
            const int source_rank =
                    source_server * kRanksPerServer + best_rail;
            const int destination_rank =
                    destination_server * kRanksPerServer + best_rail;
            selected_rail[chunk] = best_rail;
            source_offset[chunk] = next_tx[source_rank];
            destination_offset[chunk] = next_rx[destination_rank];
            next_tx[source_rank] += bytes;
            next_rx[destination_rank] += bytes;
            next_pair[best_rail] += bytes;
        }
        __syncwarp();
    }

    for (int chunk = lane; chunk < chunks; chunk += 32) {
        const int rail = selected_rail[chunk];
        const auto expert_offset =
                static_cast<std::int64_t>(chunk) * cfg.weight_chunk_bytes;
        const auto bytes = min(
                cfg.weight_chunk_bytes,
                cfg.expert_weight_bytes - expert_offset);
        auto* row = chunk_table +
                static_cast<std::int64_t>(*chunk_count + chunk) *
                kProbeChunkFields;
        row[0] = expert;
        row[1] = replica_id;
        row[2] = seed_rank;
        row[3] = chunk;
        row[4] = source_server;
        row[5] = destination_server;
        row[6] = source_server * kRanksPerServer + rail;
        row[7] = destination_server * kRanksPerServer + rail;
        row[8] = expert_offset;
        row[9] = bytes;
        row[10] = rail;
        row[11] = source_offset[chunk];
        row[12] = destination_offset[chunk];
    }
    for (int rank = lane; rank < cfg.world_size; rank += 32) {
        assigned_tx[rank] = next_tx[rank];
        assigned_rx[rank] = next_rx[rank];
    }
    if (lane < kRanksPerServer)
        pair_load[pair_base + lane] = next_pair[lane];
    if (lane == 0)
        *chunk_count += chunks;
    __syncwarp();
    return true;
}

__device__ int choose_seed_rank(const int* alloc,
                                const int* rank_load,
                                const int* remote_expert_count,
                                int expert,
                                int server,
                                const PlanConfig& cfg,
                                bool& resident) {
    resident = false;
    for (int local = 0; local < kRanksPerServer; ++local) {
        const int rank = server * kRanksPerServer + local;
        if (alloc[expert * cfg.world_size + rank] > 0) {
            resident = true;
            return rank;
        }
    }
    int seed = -1;
    for (int local = 0; local < kRanksPerServer; ++local) {
        const int rank = server * kRanksPerServer + local;
        if (remote_expert_count[rank] >= kReplicaSlots)
            continue;
        if (seed < 0 || rank_load[rank] < rank_load[seed])
            seed = rank;
    }
    return seed;
}

// One block owns the compute-only state transition. The cross-server greedy
// uses one fixed-width device radix ordering of E=256 hot groups and never
// returns metadata to the host. Admission, server-local packing and
// finalization remain phases of the same public operator and CUDA stream.
__global__ void build_probeep_plan(
        const int* counts,
        const std::int64_t* budgets,
        const std::int64_t* controller_summary,
        PlanConfig cfg,
        int* count_prefix,
        int* ideal_alloc,
        int* alloc,
        int* expert_slot,
        int* slot_count,
        int* slot_begin,
        int* replica_expert,
        int* slot_expert,
        int* server_load_before,
        int* server_load_after,
        int* server_padded_load_before,
        int* server_padded_load_after,
        std::int64_t* assigned_tx,
        std::int64_t* assigned_rx,
        std::int64_t* dispatch_tx,
        std::int64_t* dispatch_rx,
        std::int64_t* pair_load,
        int* server_expert_rows,
        int* compute_intents,
        std::int64_t* budget_snapshot,
        std::int64_t* endpoint_total_cap_output,
        int* admitted_experts,
        bool* deferred_experts,
        std::int64_t* chunk_table,
        int* plan_counts) {
    using ExpertSort = cub::BlockRadixSort<
            unsigned long long, kThreads, 1, int>;
    __shared__ int ideal_rank_load[kMaxWorldSize];
    __shared__ int ideal_server_load[kMaxServers];
    __shared__ int ideal_server_padded[kMaxServers];
    __shared__ typename ExpertSort::TempStorage expert_sort_storage;
    __shared__ int ordered_expert[kNumExperts];
    __shared__ int ordered_rows[kNumExperts];
    __shared__ std::int64_t endpoint_total_cap[kMaxWorldSize];
    __shared__ int target_low;
    __shared__ int target_high;
    __shared__ bool planning_done;
    __shared__ int compute_intent_count;
    __shared__ unsigned long long planner_stage_clock;
    __shared__ unsigned long long planner_phase_cycles;

    if (threadIdx.x == 0)
        planner_stage_clock = clock64();

    for (int index = threadIdx.x;
         index < kNumExperts * cfg.world_size;
         index += blockDim.x) {
        ideal_alloc[index] = 0;
        alloc[index] = 0;
        expert_slot[index] = -1;
    }
    for (int index = threadIdx.x;
         index < cfg.world_size * cfg.execution_slots;
         index += blockDim.x) {
        slot_count[index] = 0;
        slot_begin[index] = 0;
        slot_expert[index] = -1;
    }
    for (int index = threadIdx.x;
         index < cfg.world_size * kReplicaSlots;
         index += blockDim.x)
        replica_expert[index] = -1;
    for (int index = threadIdx.x;
         index < kNumExperts * cfg.num_servers;
         index += blockDim.x) {
        server_expert_rows[index] = 0;
    }
    for (int index = threadIdx.x;
         index < cfg.num_servers * cfg.num_servers * kRanksPerServer;
         index += blockDim.x)
        pair_load[index] = 0;
    for (int rank = threadIdx.x; rank < cfg.world_size;
         rank += blockDim.x) {
        ideal_rank_load[rank] = 0;
        assigned_tx[rank] = 0;
        assigned_rx[rank] = 0;
        dispatch_tx[rank] = 0;
        dispatch_rx[rank] = 0;
        budget_snapshot[rank] = budgets[rank];
    }
    for (int server = threadIdx.x; server < cfg.num_servers;
         server += blockDim.x) {
        ideal_server_load[server] = 0;
        ideal_server_padded[server] = 0;
    }
    for (int expert = threadIdx.x; expert < kNumExperts;
         expert += blockDim.x) {
        int prefix = 0;
        for (int source = 0; source < cfg.world_size; ++source) {
            prefix += counts[source * kNumExperts + expert];
            count_prefix[source * kNumExperts + expert] = prefix;
        }
        deferred_experts[expert] = false;
        const int owner = owner_rank(expert, cfg);
        const int server = owner / kRanksPerServer;
        ideal_alloc[expert * cfg.world_size + owner] = prefix;
        alloc[expert * cfg.world_size + owner] = prefix;
        server_expert_rows[expert * cfg.num_servers + server] = prefix;
        atomicAdd(ideal_rank_load + owner, prefix);
        atomicAdd(ideal_server_load + server, prefix);
        const int padded = padded_rows(prefix, cfg.token_padding);
        atomicAdd(ideal_server_padded + server, padded);
        ordered_rows[expert] = prefix;
    }
    if (threadIdx.x == 0) {
        compute_intent_count = 0;
        for (int counter = 0; counter < kProbePlanCounterFields; ++counter)
            plan_counts[counter] = 0;
    }
    __syncthreads();

    const int sort_rows = ordered_rows[threadIdx.x];
    unsigned long long sort_key[1] = {
            (static_cast<unsigned long long>(
                     static_cast<unsigned int>(sort_rows)) << 32) |
            static_cast<unsigned int>(UINT_MAX - threadIdx.x)};
    int sort_value[1] = {static_cast<int>(threadIdx.x)};
    ExpertSort(expert_sort_storage).SortDescending(sort_key, sort_value);
    ordered_expert[threadIdx.x] = sort_value[0];
    ordered_rows[threadIdx.x] = sort_key[0] >> 32;
    __syncthreads();

    // Materialize all experts' current Dispatch footprint in parallel.  The
    // former thread-0 E-by-W scan dominated the small metadata planner even
    // though expert intervals are independent at this point.
    accumulate_expert_dispatch<true>(
            counts, alloc, static_cast<int>(threadIdx.x), 1, cfg,
            dispatch_tx, dispatch_rx);
    __syncthreads();

    if (threadIdx.x == 0) {
        // The controller learns a total network window from the previous
        // completed same-kind observation; current Dispatch is non-croppable
        // and is re-derived on every invocation.
        const std::int64_t learned_total = controller_summary == nullptr
                ? 0 : controller_summary[5];
        for (int rank = 0; rank < cfg.world_size; ++rank) {
            const std::int64_t baseline = max(
                    bounded_dispatch_bytes(dispatch_tx[rank], cfg),
                    bounded_dispatch_bytes(dispatch_rx[rank], cfg));
            const std::int64_t budget_cap = baseline + max(
                    static_cast<std::int64_t>(0), budgets[rank]);
            // Current Dispatch is non-croppable.  A stale learned window may
            // be smaller than this invocation's baseline (for example after
            // the route distribution changes between layers).  In that case
            // the only valid migration budget is zero; the endpoint cap must
            // never make the baseline plan itself infeasible.
            endpoint_total_cap[rank] = learned_total > 0
                    ? max(baseline, min(learned_total, budget_cap))
                    : budget_cap;
            endpoint_total_cap_output[rank] = endpoint_total_cap[rank];
        }
        for (int server = 0; server < cfg.num_servers; ++server) {
            server_load_before[server] = ideal_server_load[server];
            server_padded_load_before[server] =
                    ideal_server_padded[server];
        }
        int total_routes = 0;
        for (int server = 0; server < cfg.num_servers; ++server)
            total_routes += ideal_server_load[server];
        target_low = total_routes / cfg.num_servers;
        target_high =
                (total_routes + cfg.num_servers - 1) / cfg.num_servers;
        planning_done = false;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Global quota negotiation consumes one fixed hot-first ordering of
        // the original (home-server, expert) groups.  This follows the
        // simulator contract and avoids re-scanning all E experts after each
        // accepted move.  Each move exhausts an expert group, a donor surplus
        // or a receiver deficit, so I remains bounded by E+2P+1.
        bool intent_overflow = false;
        for (int ordered = 0; ordered < kNumExperts; ++ordered) {
            const int expert = ordered_expert[ordered];
            if (ordered_rows[ordered] <= 0)
                break;
            const int donor = owner_server(expert, cfg);
            while (ideal_server_load[donor] > target_high &&
                   server_expert_rows[
                           expert * cfg.num_servers + donor] > 0) {
                if (compute_intent_count >= kProbeMaxComputeIntents) {
                    intent_overflow = true;
                    break;
                }
                const int server_rows = server_expert_rows[
                        expert * cfg.num_servers + donor];
                const int surplus = max(
                        0, ideal_server_load[donor] - target_high);
                int source_rank = donor * kRanksPerServer;
                for (int local = 1; local < kRanksPerServer; ++local) {
                    const int rank = donor * kRanksPerServer + local;
                    if (ideal_alloc[expert * cfg.world_size + rank] >
                        ideal_alloc[expert * cfg.world_size + source_rank])
                        source_rank = rank;
                }

                int current_max, current_spread;
                std::int64_t current_energy;
                server_objective(ideal_server_padded, cfg.num_servers,
                                 current_max, current_spread, current_energy);
                IntentCandidate best;
                for (int receiver = 0; receiver < cfg.num_servers;
                     ++receiver) {
                    if (receiver == donor)
                        continue;
                    const int deficit = max(
                            0, target_low - ideal_server_load[receiver]);
                    if (deficit == 0)
                        continue;
                    const int moved = min(
                            ideal_alloc[expert * cfg.world_size + source_rank],
                            min(server_rows, min(surplus, deficit)));
                    if (moved <= 0)
                        continue;
                    const int receiver_rows = server_expert_rows[
                            expert * cfg.num_servers + receiver];
                    const int donor_padded = ideal_server_padded[donor] -
                            padded_rows(server_rows, cfg.token_padding) +
                            padded_rows(server_rows - moved,
                                        cfg.token_padding);
                    const int receiver_padded =
                            ideal_server_padded[receiver] -
                            padded_rows(receiver_rows, cfg.token_padding) +
                            padded_rows(receiver_rows + moved,
                                        cfg.token_padding);
                    int candidate_max = 0;
                    int candidate_min = INT_MAX;
                    for (int server = 0; server < cfg.num_servers; ++server) {
                        int value = ideal_server_padded[server];
                        if (server == donor)
                            value = donor_padded;
                        else if (server == receiver)
                            value = receiver_padded;
                        candidate_max = max(candidate_max, value);
                        candidate_min = min(candidate_min, value);
                    }
                    const std::int64_t candidate_energy = current_energy -
                            static_cast<std::int64_t>(
                                    ideal_server_padded[donor]) *
                                    ideal_server_padded[donor] -
                            static_cast<std::int64_t>(
                                    ideal_server_padded[receiver]) *
                                    ideal_server_padded[receiver] +
                            static_cast<std::int64_t>(donor_padded) *
                                    donor_padded +
                            static_cast<std::int64_t>(receiver_padded) *
                                    receiver_padded;
                    const int candidate_spread =
                            candidate_max - candidate_min;
                    if (!objective_less(
                                candidate_max, candidate_spread,
                                candidate_energy, current_max, current_spread,
                                current_energy))
                        continue;
                    int seed_rank = receiver * kRanksPerServer;
                    for (int local = 1; local < kRanksPerServer; ++local) {
                        const int rank = receiver * kRanksPerServer + local;
                        if (ideal_rank_load[rank] < ideal_rank_load[seed_rank])
                            seed_rank = rank;
                    }
                    IntentCandidate candidate{
                        expert, donor, receiver, source_rank, seed_rank, moved,
                        surplus, deficit, receiver_rows > 0, server_rows,
                        candidate_max, candidate_spread, candidate_energy};
                    if (candidate_better(candidate, best))
                        best = candidate;
                }
                if (best.expert < 0 || best.moved <= 0)
                    break;

                auto* intent = compute_intents +
                        compute_intent_count * kIntentFields;
                intent[0] = best.expert;
                intent[1] = best.donor;
                intent[2] = best.receiver;
                intent[3] = best.source_rank;
                intent[4] = best.seed_rank;
                intent[5] = best.moved;
                ++compute_intent_count;
                ideal_alloc[best.expert * cfg.world_size +
                            best.source_rank] -= best.moved;
                ideal_alloc[best.expert * cfg.world_size +
                            best.seed_rank] += best.moved;
                ideal_rank_load[best.source_rank] -= best.moved;
                ideal_rank_load[best.seed_rank] += best.moved;
                const int donor_rows = server_expert_rows[
                        best.expert * cfg.num_servers + best.donor];
                const int receiver_rows = server_expert_rows[
                        best.expert * cfg.num_servers + best.receiver];
                ideal_server_padded[best.donor] =
                        ideal_server_padded[best.donor] -
                        padded_rows(donor_rows, cfg.token_padding) +
                        padded_rows(donor_rows - best.moved,
                                    cfg.token_padding);
                ideal_server_padded[best.receiver] =
                        ideal_server_padded[best.receiver] -
                        padded_rows(receiver_rows, cfg.token_padding) +
                        padded_rows(receiver_rows + best.moved,
                                    cfg.token_padding);
                server_expert_rows[
                        best.expert * cfg.num_servers + best.donor] -=
                        best.moved;
                server_expert_rows[
                        best.expert * cfg.num_servers + best.receiver] +=
                        best.moved;
                ideal_server_load[best.donor] -= best.moved;
                ideal_server_load[best.receiver] += best.moved;
            }
            if (intent_overflow)
                break;
        }

        // Raw quota equality is not sufficient when grouped GEMM rounds every
        // non-empty (server, expert) group to token_padding.  Reuse the same
        // fixed expert order for one bounded refinement pass.  Each expert can
        // contribute at most one additional intent, keeping the metadata path
        // O(E*P^2), while every accepted move strictly improves the global
        // (padded maximum, padded spread, sum-of-squares) objective.  The
        // energy term is only a strict tie-break: it lets one hot server move
        // toward one cold server when another equally-hot/equally-cold server
        // keeps max and spread unchanged.  Network state remains absent here;
        // admission replays these immutable intents later.
        if (!intent_overflow && cfg.token_padding > 1) {
            for (int ordered = 0; ordered < kNumExperts; ++ordered) {
                const int expert = ordered_expert[ordered];
                if (ordered_rows[ordered] <= 0)
                    break;
                int current_max, current_spread;
                std::int64_t current_energy;
                server_objective(ideal_server_padded, cfg.num_servers,
                                 current_max, current_spread, current_energy);
                if (current_spread <= cfg.token_padding)
                    break;

                IntentCandidate best;
                for (int donor = 0; donor < cfg.num_servers; ++donor) {
                    const int donor_rows = server_expert_rows[
                            expert * cfg.num_servers + donor];
                    if (donor_rows <= 0)
                        continue;
                    int source_rank = donor * kRanksPerServer;
                    for (int local = 1; local < kRanksPerServer; ++local) {
                        const int rank = donor * kRanksPerServer + local;
                        if (ideal_alloc[expert * cfg.world_size + rank] >
                            ideal_alloc[expert * cfg.world_size + source_rank])
                            source_rank = rank;
                    }
                    // Smallest number of routes that removes one padded block
                    // from this donor's (server,expert) group.
                    const int boundary = donor_rows -
                            ((donor_rows - 1) / cfg.token_padding) *
                                    cfg.token_padding;
                    if (ideal_alloc[expert * cfg.world_size + source_rank] <
                        boundary)
                        continue;
                    for (int receiver = 0; receiver < cfg.num_servers;
                         ++receiver) {
                        if (receiver == donor)
                            continue;
                        const int receiver_rows = server_expert_rows[
                                expert * cfg.num_servers + receiver];
                        const int receiver_before = padded_rows(
                                receiver_rows, cfg.token_padding);
                        const int receiver_after = padded_rows(
                                receiver_rows + boundary, cfg.token_padding);
                        const int receiver_increase =
                                receiver_after - receiver_before;
                        const int donor_padded =
                                ideal_server_padded[donor] -
                                padded_rows(donor_rows, cfg.token_padding) +
                                padded_rows(donor_rows - boundary,
                                            cfg.token_padding);
                        const int receiver_padded =
                                ideal_server_padded[receiver] +
                                receiver_increase;
                        int candidate_max = 0;
                        int candidate_min = INT_MAX;
                        for (int server = 0; server < cfg.num_servers;
                             ++server) {
                            int value = ideal_server_padded[server];
                            if (server == donor)
                                value = donor_padded;
                            else if (server == receiver)
                                value = receiver_padded;
                            candidate_max = max(candidate_max, value);
                            candidate_min = min(candidate_min, value);
                        }
                        const std::int64_t candidate_energy = current_energy -
                                static_cast<std::int64_t>(
                                        ideal_server_padded[donor]) *
                                        ideal_server_padded[donor] -
                                static_cast<std::int64_t>(
                                        ideal_server_padded[receiver]) *
                                        ideal_server_padded[receiver] +
                                static_cast<std::int64_t>(donor_padded) *
                                        donor_padded +
                                static_cast<std::int64_t>(receiver_padded) *
                                        receiver_padded;
                        const int candidate_spread =
                                candidate_max - candidate_min;
                        if (!objective_less(
                                    candidate_max, candidate_spread,
                                    candidate_energy, current_max,
                                    current_spread, current_energy))
                            continue;
                        int seed_rank = receiver * kRanksPerServer;
                        bool resident = false;
                        for (int local = 0; local < kRanksPerServer; ++local) {
                            const int rank = receiver * kRanksPerServer + local;
                            if (ideal_alloc[expert * cfg.world_size + rank] > 0) {
                                seed_rank = rank;
                                resident = true;
                                break;
                            }
                            if (ideal_rank_load[rank] <
                                ideal_rank_load[seed_rank])
                                seed_rank = rank;
                        }
                        IntentCandidate candidate{
                            expert, donor, receiver, source_rank, seed_rank,
                            boundary,
                            // Reused as receiver padded growth for refinement.
                            receiver_increase,
                            0, resident ? 1 : 0, donor_rows,
                            candidate_max, candidate_spread,
                            candidate_energy};
                        if (refinement_candidate_better(candidate, best))
                            best = candidate;
                    }
                }
                if (best.expert < 0)
                    continue;
                if (compute_intent_count >= kProbeMaxComputeIntents) {
                    intent_overflow = true;
                    break;
                }
                auto* intent = compute_intents +
                        compute_intent_count * kIntentFields;
                intent[0] = best.expert;
                intent[1] = best.donor;
                intent[2] = best.receiver;
                intent[3] = best.source_rank;
                intent[4] = best.seed_rank;
                intent[5] = best.moved;
                ++compute_intent_count;

                const int donor_rows = server_expert_rows[
                        best.expert * cfg.num_servers + best.donor];
                const int receiver_rows = server_expert_rows[
                        best.expert * cfg.num_servers + best.receiver];
                ideal_alloc[best.expert * cfg.world_size +
                            best.source_rank] -= best.moved;
                ideal_alloc[best.expert * cfg.world_size +
                            best.seed_rank] += best.moved;
                ideal_rank_load[best.source_rank] -= best.moved;
                ideal_rank_load[best.seed_rank] += best.moved;
                ideal_server_padded[best.donor] =
                        ideal_server_padded[best.donor] -
                        padded_rows(donor_rows, cfg.token_padding) +
                        padded_rows(donor_rows - best.moved,
                                    cfg.token_padding);
                ideal_server_padded[best.receiver] =
                        ideal_server_padded[best.receiver] -
                        padded_rows(receiver_rows, cfg.token_padding) +
                        padded_rows(receiver_rows + best.moved,
                                    cfg.token_padding);
                server_expert_rows[
                        best.expert * cfg.num_servers + best.donor] -=
                        best.moved;
                server_expert_rows[
                        best.expert * cfg.num_servers + best.receiver] +=
                        best.moved;
                ideal_server_load[best.donor] -= best.moved;
                ideal_server_load[best.receiver] += best.moved;
            }
        }
        // The quota/refinement loops above mutate an internal route mapping.
        // Their transition steps are not transport work: a route may move
        // through an intermediate server while the padded objective is being
        // refined.  Lower only the final mapping into unique home->destination
        // intents, otherwise an intermediate server can receive a full expert
        // whose final route count is zero.  This also guarantees one Weight
        // RDMA replica per (expert,destination_server); destination-local
        // packing may subsequently fan it out over NVLink.
        if (!intent_overflow) {
            int final_intent_count = 0;
            for (int ordered = 0; ordered < kNumExperts; ++ordered) {
                const int expert = ordered_expert[ordered];
                if (ordered_rows[ordered] <= 0)
                    break;
                const int home = owner_server(expert, cfg);
                for (int receiver = 0; receiver < cfg.num_servers;
                     ++receiver) {
                    if (receiver == home)
                        continue;
                    const int moved = server_expert_rows[
                            expert * cfg.num_servers + receiver];
                    if (moved <= 0)
                        continue;
                    if (final_intent_count >= kProbeMaxComputeIntents) {
                        intent_overflow = true;
                        break;
                    }
                    int seed_rank = receiver * kRanksPerServer;
                    for (int local = 1; local < kRanksPerServer; ++local) {
                        const int rank = receiver * kRanksPerServer + local;
                        if (ideal_alloc[expert * cfg.world_size + rank] >
                            ideal_alloc[expert * cfg.world_size + seed_rank])
                            seed_rank = rank;
                    }
                    auto* intent = compute_intents +
                            final_intent_count * kIntentFields;
                    intent[0] = expert;
                    intent[1] = home;
                    intent[2] = receiver;
                    intent[3] = owner_rank(expert, cfg);
                    intent[4] = seed_rank;
                    intent[5] = moved;
                    ++final_intent_count;
                }
                if (intent_overflow)
                    break;
            }
            compute_intent_count = final_intent_count;
        }
        planning_done = !intent_overflow;
        planner_phase_cycles = clock64() - planner_stage_clock;
        plan_counts[3] = compute_intent_count;
        plan_counts[8] = planning_done ? 1 : 0;
        plan_counts[9] = static_cast<int>(min(
                planner_phase_cycles,
                static_cast<unsigned long long>(INT_MAX)));
    }
}

// Replay the immutable compute-only intent sequence through the network
// controller in its own kernel.  Separating this serial dependency chain from
// candidate generation drops the candidate kernel's register/shared-memory
// footprint and lets later server packing occupy independent SMs.
__global__ void admit_probeep_intents(
        const int* counts,
        const int* count_prefix,
        const std::int64_t* endpoint_total_cap,
        const int* compute_intents,
        int* alloc,
        std::int64_t* assigned_tx,
        std::int64_t* assigned_rx,
        std::int64_t* dispatch_tx,
        std::int64_t* dispatch_rx,
        std::int64_t* pair_load,
        int* server_expert_rows,
        int* admitted_experts,
        bool* deferred_experts,
        std::int64_t* chunk_table,
        int* plan_counts,
        PlanConfig cfg) {
    const auto phase_start = clock64();
    __shared__ int actual_rank_load[kMaxWorldSize];
    __shared__ int remote_expert_count[kMaxWorldSize];
    __shared__ int actual_server_load[kMaxServers];
    __shared__ int actual_server_padded[kMaxServers];
    __shared__ std::int64_t trial_dispatch_tx[kMaxWorldSize];
    __shared__ std::int64_t trial_dispatch_rx[kMaxWorldSize];
    __shared__ std::int64_t trial_weight_tx[kMaxWorldSize];
    __shared__ std::int64_t trial_weight_rx[kMaxWorldSize];
    __shared__ std::int64_t trial_pair[kRanksPerServer];
    __shared__ int trial_selected_rail[kProbeMaxChunksPerExpert];
    __shared__ std::int64_t trial_source_offset[kProbeMaxChunksPerExpert];
    __shared__ std::int64_t trial_destination_offset[kProbeMaxChunksPerExpert];
    __shared__ int admitted_count;
    __shared__ int chunk_count;
    __shared__ int deferred_count;
    __shared__ int current_expert;
    __shared__ int current_donor;
    __shared__ int current_receiver;
    __shared__ int current_source_rank;
    __shared__ int current_seed_rank;
    __shared__ int current_moved;
    __shared__ int current_donor_padded;
    __shared__ int current_receiver_padded;
    __shared__ bool current_resident;
    __shared__ bool current_valid;
    __shared__ bool no_more_intents;
    __shared__ bool processed_intents[kProbeMaxComputeIntents];
    __shared__ int current_scheduled;

    for (int rank = threadIdx.x; rank < cfg.world_size;
         rank += blockDim.x) {
        actual_rank_load[rank] = 0;
        remote_expert_count[rank] = 0;
    }
    for (int server = threadIdx.x; server < cfg.num_servers;
         server += blockDim.x) {
        actual_server_load[server] = 0;
        actual_server_padded[server] = 0;
    }
    for (int index = threadIdx.x;
         index < kNumExperts * cfg.num_servers;
         index += blockDim.x)
        server_expert_rows[index] = 0;
    __syncthreads();

    for (int expert = threadIdx.x; expert < kNumExperts;
         expert += blockDim.x) {
        const int total = count_prefix[
                (cfg.world_size - 1) * kNumExperts + expert];
        const int owner = owner_rank(expert, cfg);
        const int server = owner / kRanksPerServer;
        server_expert_rows[expert * cfg.num_servers + server] = total;
        atomicAdd(actual_rank_load + owner, total);
        atomicAdd(actual_server_load + server, total);
        atomicAdd(actual_server_padded + server,
                  padded_rows(total, cfg.token_padding));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        admitted_count = 0;
        chunk_count = 0;
        deferred_count = 0;
        no_more_intents = false;
    }
    for (int intent = threadIdx.x; intent < kProbeMaxComputeIntents;
         intent += blockDim.x)
        processed_intents[intent] = false;
    __syncthreads();

    const int compute_intent_count = plan_counts[3];
    // Final home->destination intents are coalesced from a monotonic internal
    // transition path.  A fixed one-pass replay can encounter an intent whose
    // direct move is not beneficial *yet*, even though a later intent first
    // lowers another maximum and makes it beneficial.  Select the first
    // currently improving unprocessed intent and revisit skipped entries.
    // This preserves the compute planner's stable hot-first priority without a
    // sort. I is bounded metadata (<=2E+2P+1), and rail/chunk admission remains
    // the dominant per-intent work.
    for (int admission_step = 0;
         admission_step < compute_intent_count; ++admission_step) {
        if (threadIdx.x == 0) {
            current_valid = false;
            int current_max, current_spread;
            std::int64_t current_energy;
            server_objective(actual_server_padded, cfg.num_servers,
                             current_max, current_spread, current_energy);
            for (int intent_index = 0;
                 intent_index < compute_intent_count; ++intent_index) {
                if (processed_intents[intent_index])
                    continue;
                const auto* intent = compute_intents +
                        intent_index * kIntentFields;
                const int expert = intent[0];
                const int donor = intent[1];
                const int receiver = intent[2];
                int source_rank = donor * kRanksPerServer;
                for (int local = 1; local < kRanksPerServer; ++local) {
                    const int rank = donor * kRanksPerServer + local;
                    if (alloc[expert * cfg.world_size + rank] >
                        alloc[expert * cfg.world_size + source_rank])
                        source_rank = rank;
                }
                const int moved = min(
                        intent[5],
                        alloc[expert * cfg.world_size + source_rank]);
                bool resident = false;
                const int seed_rank = moved > 0 ? choose_seed_rank(
                        alloc, actual_rank_load, remote_expert_count,
                        expert, receiver, cfg, resident) : -1;
                if (moved <= 0 || seed_rank < 0) {
                    processed_intents[intent_index] = true;
                    deferred_experts[expert] = true;
                    ++deferred_count;
                    continue;
                }
                const int donor_rows = server_expert_rows[
                        expert * cfg.num_servers + donor];
                const int receiver_rows = server_expert_rows[
                        expert * cfg.num_servers + receiver];
                const int donor_padded = actual_server_padded[donor] -
                        padded_rows(donor_rows, cfg.token_padding) +
                        padded_rows(donor_rows - moved, cfg.token_padding);
                const int receiver_padded =
                        actual_server_padded[receiver] -
                        padded_rows(receiver_rows, cfg.token_padding) +
                        padded_rows(receiver_rows + moved,
                                    cfg.token_padding);
                int candidate_max = 0;
                int candidate_min = INT_MAX;
                for (int server = 0; server < cfg.num_servers; ++server) {
                    int value = actual_server_padded[server];
                    if (server == donor)
                        value = donor_padded;
                    else if (server == receiver)
                        value = receiver_padded;
                    candidate_max = max(candidate_max, value);
                    candidate_min = min(candidate_min, value);
                }
                const std::int64_t candidate_energy = current_energy -
                        static_cast<std::int64_t>(
                                actual_server_padded[donor]) *
                                actual_server_padded[donor] -
                        static_cast<std::int64_t>(
                                actual_server_padded[receiver]) *
                                actual_server_padded[receiver] +
                        static_cast<std::int64_t>(donor_padded) * donor_padded +
                        static_cast<std::int64_t>(receiver_padded) *
                                receiver_padded;
                if (!objective_less(
                            candidate_max, candidate_max - candidate_min,
                            candidate_energy, current_max, current_spread,
                            current_energy))
                    continue;

                current_expert = expert;
                current_donor = donor;
                current_receiver = receiver;
                current_source_rank = source_rank;
                current_seed_rank = seed_rank;
                current_moved = moved;
                current_donor_padded = donor_padded;
                current_receiver_padded = receiver_padded;
                current_resident = resident;
                current_valid = true;
                processed_intents[intent_index] = true;
                break;
            }
            if (!current_valid) {
                for (int intent_index = 0;
                     intent_index < compute_intent_count; ++intent_index) {
                    if (processed_intents[intent_index])
                        continue;
                    processed_intents[intent_index] = true;
                    const int expert = compute_intents[
                            intent_index * kIntentFields];
                    deferred_experts[expert] = true;
                    ++deferred_count;
                }
                no_more_intents = true;
            }
        }
        __syncthreads();
        if (no_more_intents)
            break;

        for (int rank = threadIdx.x; rank < cfg.world_size;
             rank += blockDim.x) {
            trial_dispatch_tx[rank] = dispatch_tx[rank];
            trial_dispatch_rx[rank] = dispatch_rx[rank];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            accumulate_expert_dispatch<false>(
                    counts, alloc, current_expert, -1, cfg,
                    trial_dispatch_tx, trial_dispatch_rx);
            alloc[current_expert * cfg.world_size + current_source_rank] -=
                    current_moved;
            alloc[current_expert * cfg.world_size + current_seed_rank] +=
                    current_moved;
            accumulate_expert_dispatch<false>(
                    counts, alloc, current_expert, 1, cfg,
                    trial_dispatch_tx, trial_dispatch_rx);
            current_scheduled = true;
        }
        __syncthreads();

        for (int rank = threadIdx.x; rank < cfg.world_size;
             rank += blockDim.x) {
            const auto token_tx = bounded_dispatch_bytes(
                    trial_dispatch_tx[rank], cfg);
            const auto token_rx = bounded_dispatch_bytes(
                    trial_dispatch_rx[rank], cfg);
            if (assigned_tx[rank] + token_tx > endpoint_total_cap[rank] ||
                assigned_rx[rank] + token_rx > endpoint_total_cap[rank])
                atomicExch(&current_scheduled, 0);
        }
        __syncthreads();
        if (current_scheduled && !current_resident) {
            const bool scheduled = schedule_complete_expert_warp(
                    current_expert, admitted_count,
                    owner_server(current_expert, cfg), current_receiver,
                    current_seed_rank, cfg, endpoint_total_cap,
                    trial_dispatch_tx, trial_dispatch_rx,
                    assigned_tx, assigned_rx, pair_load,
                    chunk_table, &chunk_count,
                    trial_weight_tx, trial_weight_rx, trial_pair,
                    trial_selected_rail, trial_source_offset,
                    trial_destination_offset);
            if (threadIdx.x == 0)
                current_scheduled = scheduled;
        }
        __syncthreads();
        if (!current_scheduled) {
            if (threadIdx.x == 0) {
                alloc[current_expert * cfg.world_size +
                      current_source_rank] += current_moved;
                alloc[current_expert * cfg.world_size +
                      current_seed_rank] -= current_moved;
                deferred_experts[current_expert] = true;
                ++deferred_count;
            }
            __syncthreads();
            continue;
        }

        for (int rank = threadIdx.x; rank < cfg.world_size;
             rank += blockDim.x) {
            dispatch_tx[rank] = trial_dispatch_tx[rank];
            dispatch_rx[rank] = trial_dispatch_rx[rank];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            admitted_experts[admitted_count++] =
                    current_expert * kMaxServers + current_receiver;
            actual_rank_load[current_source_rank] -= current_moved;
            actual_rank_load[current_seed_rank] += current_moved;
            if (!current_resident)
                ++remote_expert_count[current_seed_rank];
            actual_server_load[current_donor] -= current_moved;
            actual_server_load[current_receiver] += current_moved;
            actual_server_padded[current_donor] = current_donor_padded;
            actual_server_padded[current_receiver] =
                    current_receiver_padded;
            server_expert_rows[
                    current_expert * cfg.num_servers + current_donor] -=
                    current_moved;
            server_expert_rows[
                    current_expert * cfg.num_servers + current_receiver] +=
                    current_moved;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        plan_counts[0] = admitted_count;
        plan_counts[1] = chunk_count;
        plan_counts[4] = deferred_count;
        plan_counts[10] = static_cast<int>(min(
                clock64() - phase_start,
                static_cast<unsigned long long>(INT_MAX)));
    }
    __syncthreads();
    // Publish the conservative server-deduplicated bound rather than the raw
    // occurrence sum.  The raw sum is only scratch for incremental replay.
    for (int rank = threadIdx.x; rank < cfg.world_size;
         rank += blockDim.x) {
        dispatch_tx[rank] = bounded_dispatch_bytes(dispatch_tx[rank], cfg);
        dispatch_rx[rank] = bounded_dispatch_bytes(dispatch_rx[rank], cfg);
    }
}

// The old fused block selected every server's hot experts with P independent
// O(E^2) scalar scans.  That left almost every warp parked behind a block
// barrier for several milliseconds.  A dedicated block per server performs
// one stable radix ordering, then preserves the same hot-first packing and
// deterministic tie rules.  This is still an internal phase of the single
// public ProbeEP operator and does not add a host round-trip.
__global__ void pack_server_local(
        const int* server_expert_rows,
        int* alloc,
        int* plan_counts,
        PlanConfig cfg) {
    const auto phase_start = clock64();
    const int server = static_cast<int>(blockIdx.x);
    const int expert = static_cast<int>(threadIdx.x);
    if (server >= cfg.num_servers || expert >= kNumExperts)
        return;

    using BlockSort = cub::BlockRadixSort<
            unsigned long long, kThreads, 1, int>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ int ordered_expert[kNumExperts];
    __shared__ int ordered_rows[kNumExperts];

    const int rows = server_expert_rows[
            expert * cfg.num_servers + server];
    unsigned long long key[1] = {
            (static_cast<unsigned long long>(static_cast<unsigned int>(rows))
             << 32) |
            static_cast<unsigned int>(UINT_MAX - expert)};
    int value[1] = {expert};
    BlockSort(sort_storage).SortDescending(key, value);
    ordered_expert[expert] = value[0];
    ordered_rows[expert] = server_expert_rows[
            value[0] * cfg.num_servers + server];

    const int rank_begin = server * kRanksPerServer;
    for (int destination = expert;
         destination < kNumExperts * kRanksPerServer;
         destination += blockDim.x) {
        const int item_expert = destination / kRanksPerServer;
        const int local = destination % kRanksPerServer;
        alloc[item_expert * cfg.world_size + rank_begin + local] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int rank_blocks[kRanksPerServer] = {0};
        int remote_experts[kRanksPerServer] = {0};
        int target_blocks[kRanksPerServer];
        int total_blocks = 0;
        for (int ordered = 0; ordered < kNumExperts; ++ordered)
            total_blocks += padded_rows(
                    ordered_rows[ordered], cfg.token_padding) /
                    cfg.token_padding;
        const int base_capacity = total_blocks / kRanksPerServer;
        const int extra_capacity = total_blocks % kRanksPerServer;
        for (int local = 0; local < kRanksPerServer; ++local)
            target_blocks[local] = base_capacity +
                    (local < extra_capacity ? 1 : 0);

        int local_placements = 0;
        for (int ordered = 0; ordered < kNumExperts; ++ordered) {
            const int selected_expert = ordered_expert[ordered];
            const int selected_rows = ordered_rows[ordered];
            if (selected_rows <= 0)
                break;
            int remaining_rows = selected_rows;
            int remaining_blocks = padded_rows(
                    selected_rows, cfg.token_padding) / cfg.token_padding;
            const int home_rank = owner_rank(selected_expert, cfg);
            while (remaining_blocks > 0) {
                int best_local = -1;
                int best_capacity = -1;
                int best_home = -1;
                for (int local = 0; local < kRanksPerServer; ++local) {
                    const int rank = rank_begin + local;
                    const bool home = rank == home_rank;
                    if (alloc[selected_expert * cfg.world_size + rank] > 0)
                        continue;
                    if (!home && remote_experts[local] >= kReplicaSlots)
                        continue;
                    const int capacity = max(
                            0, target_blocks[local] - rank_blocks[local]);
                    if (capacity > best_capacity ||
                        (capacity == best_capacity && home > best_home) ||
                        (capacity == best_capacity && home == best_home &&
                         (best_local < 0 || local < best_local))) {
                        best_local = local;
                        best_capacity = capacity;
                        best_home = home;
                    }
                }
                if (best_local < 0) {
                    atomicExch(plan_counts + 2, 1);
                    break;
                }
                const int assigned_blocks = min(
                        remaining_blocks, max(1, best_capacity));
                const int assigned_rows = min(
                        remaining_rows,
                        assigned_blocks * cfg.token_padding);
                const int rank = rank_begin + best_local;
                alloc[selected_expert * cfg.world_size + rank] = assigned_rows;
                rank_blocks[best_local] += padded_rows(
                        assigned_rows, cfg.token_padding) /
                        cfg.token_padding;
                if (rank != home_rank)
                    ++remote_experts[best_local];
                remaining_rows -= assigned_rows;
                remaining_blocks -= assigned_blocks;
                ++local_placements;
            }
            if (remaining_rows != 0)
                atomicExch(plan_counts + 2, 1);
        }
        atomicMax(plan_counts + 5, local_placements);
        atomicMax(plan_counts + 11, static_cast<int>(min(
                clock64() - phase_start,
                static_cast<unsigned long long>(INT_MAX))));
    }
}

__global__ void finalize_probeep_plan(
        const int* count_prefix,
        const int* server_expert_rows,
        int* alloc,
        int* alloc_prefix,
        int* expert_slot,
        int* slot_count,
        int* slot_begin,
        int* replica_expert,
        int* slot_expert,
        int* server_load_after,
        int* server_padded_load_after,
        int* plan_counts,
        PlanConfig cfg) {
    const auto phase_start = clock64();

    for (int rank = threadIdx.x; rank < cfg.world_size;
         rank += blockDim.x) {
        const int local_begin = rank * cfg.local_experts;
        for (int local = 0; local < cfg.local_experts; ++local) {
            const int expert = local_begin + local;
            expert_slot[rank * kNumExperts + expert] = local;
            slot_expert[rank * cfg.execution_slots + local] = expert;
        }

        // Packing already guarantees at most kReplicaSlots remote experts.
        // Their physical slot order has no algorithmic meaning, so one stable
        // expert-ID pass replaces the former kReplicaSlots x E selection.
        int replicas = 0;
        for (int expert = 0; expert < kNumExperts; ++expert) {
            const bool local = expert >= local_begin &&
                               expert < local_begin + cfg.local_experts;
            if (local || alloc[expert * cfg.world_size + rank] <= 0)
                continue;
            if (replicas >= kReplicaSlots) {
                atomicExch(plan_counts + 2, 1);
                break;
            }
            const int slot = cfg.local_experts + replicas;
            replica_expert[rank * kReplicaSlots + replicas] = expert;
            slot_expert[rank * cfg.execution_slots + slot] = expert;
            expert_slot[rank * kNumExperts + expert] = slot;
            ++replicas;
        }

        int begin = 0;
        for (int slot = 0; slot < cfg.execution_slots; ++slot) {
            const int expert = slot_expert[
                    rank * cfg.execution_slots + slot];
            const int rows = expert >= 0 ?
                    alloc[expert * cfg.world_size + rank] : 0;
            slot_begin[rank * cfg.execution_slots + slot] = begin;
            slot_count[rank * cfg.execution_slots + slot] = rows;
            begin += padded_rows(rows, cfg.token_padding);
        }
    }
    __syncthreads();

    for (int expert = threadIdx.x; expert < kNumExperts;
         expert += blockDim.x) {
        int prefix = 0;
        int negative = 0;
        for (int rank = 0; rank < cfg.world_size; ++rank) {
            const int rows = alloc[expert * cfg.world_size + rank];
            negative += rows < 0;
            prefix += rows;
            alloc_prefix[expert * cfg.world_size + rank] = prefix;
        }
        if (negative)
            atomicAdd(plan_counts + 6, negative);
        const int expected = count_prefix[
                (cfg.world_size - 1) * kNumExperts + expert];
        if (prefix != expected)
            atomicAdd(plan_counts + 7, 1);
    }
    for (int server = threadIdx.x; server < cfg.num_servers;
         server += blockDim.x) {
        int raw = 0;
        int padded = 0;
        for (int expert = 0; expert < kNumExperts; ++expert) {
            const int rows = server_expert_rows[
                    expert * cfg.num_servers + server];
            raw += rows;
            padded += padded_rows(rows, cfg.token_padding);
        }
        server_load_after[server] = raw;
        server_padded_load_after[server] = padded;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        plan_counts[12] = static_cast<int>(min(
                clock64() - phase_start,
                static_cast<unsigned long long>(INT_MAX)));
}

__global__ void clear_materialized_outputs(
        bool* is_token_in_rank,
        int* num_tokens_per_rank,
        int* num_tokens_per_rdma_rank,
        int* num_tokens_per_exec_expert,
        int total_tokens,
        int num_sources,
        PlanConfig cfg) {
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < total_tokens * cfg.world_size;
         index += static_cast<int>(blockDim.x * gridDim.x))
        is_token_in_rank[index] = false;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < num_sources * cfg.world_size;
         index += static_cast<int>(blockDim.x * gridDim.x))
        num_tokens_per_rank[index] = 0;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < num_sources * cfg.num_servers;
         index += static_cast<int>(blockDim.x * gridDim.x))
        num_tokens_per_rdma_rank[index] = 0;
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < num_sources * cfg.world_size * cfg.execution_slots;
         index += static_cast<int>(blockDim.x * gridDim.x))
        num_tokens_per_exec_expert[index] = 0;
}

__global__ void materialize_exec_counts(
        const int* count_prefix,
        const int* alloc,
        const int* expert_slot,
        int* num_tokens_per_exec_expert,
        int num_sources,
        int source_rank_base,
        PlanConfig cfg) {
    const int expert = static_cast<int>(threadIdx.x);
    const int input_source = static_cast<int>(blockIdx.x);
    if (expert >= kNumExperts || input_source >= num_sources)
        return;
    const int source = source_rank_base < 0 ? input_source : source_rank_base;
    const int source_begin = source == 0 ? 0 :
            count_prefix[(source - 1) * kNumExperts + expert];
    const int source_end = count_prefix[source * kNumExperts + expert];
    int destination_begin = 0;
    for (int destination = 0; destination < cfg.world_size; ++destination) {
        const int destination_end = destination_begin +
                alloc[expert * cfg.world_size + destination];
        const int overlap = max(0, min(source_end, destination_end) -
                                   max(source_begin, destination_begin));
        if (overlap > 0) {
            const int slot = expert_slot[
                    destination * kNumExperts + expert];
            num_tokens_per_exec_expert[
                    input_source * (cfg.world_size * cfg.execution_slots) +
                    destination * cfg.execution_slots + slot] = overlap;
        }
        destination_begin = destination_end;
    }
}

template <bool kSingleSource>
__global__ void materialize_probe_routes(
        const std::int64_t* topk_idx,
        const int* local_ordinal,
        const int* count_prefix,
        const int* alloc_prefix,
        const int* expert_slot,
        const int* slot_begin,
        int* route_dst,
        int* exec_rank,
        int* exec_slot,
        bool* is_token_in_rank,
        int* num_tokens_per_rank,
        int* num_tokens_per_rdma_rank,
        int routes_per_rank,
        int total_routes,
        int source_rank_base,
        int nvs,
        bool segmented_ordinals,
        PlanConfig cfg) {
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < total_routes;
         index += static_cast<int>(blockDim.x * gridDim.x)) {
        const int input_source = index / routes_per_rank;
        const int source = source_rank_base + input_source;
        const int local_route = index - input_source * routes_per_rank;
        const int expert = static_cast<int>(topk_idx[index]);
        const int previous_sources = source == 0 ? 0 :
                count_prefix[(source - 1) * kNumExperts + expert];
        int ordinal = local_ordinal[index];
        if (segmented_ordinals) {
            const int histogram_blocks = min(
                    routes_per_rank / kHistogramSegmentRoutes,
                    kHistogramMaxBlocks);
            const int segment = min(
                    local_route / kHistogramSegmentRoutes,
                    histogram_blocks - 1);
            const int segment_begin = segment * kHistogramSegmentRoutes;
            const int packed_prefix = local_ordinal[
                    input_source * routes_per_rank + segment_begin + expert];
            if (local_route - segment_begin < kNumExperts)
                ordinal &= kPackedOrdinalMask;
            ordinal += packed_prefix >> kPackedOrdinalBits;
        }
        const int global_ordinal = previous_sources + ordinal;
        int lo = 0;
        int hi = cfg.world_size;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (alloc_prefix[expert * cfg.world_size + mid] <=
                global_ordinal)
                lo = mid + 1;
            else
                hi = mid;
        }
        const int destination = lo;
        const int previous_destination = destination == 0 ? 0 :
                alloc_prefix[expert * cfg.world_size + destination - 1];
        const int slot = expert_slot[
                destination * kNumExperts + expert];
        const int row = slot_begin[
                destination * cfg.execution_slots + slot] +
                global_ordinal - previous_destination;
        route_dst[index] = destination * nvs + row;
        exec_rank[index] = destination;
        exec_slot[index] = slot;

        const int warp_lane = threadIdx.x & 31;
        const int subgroup_begin = warp_lane & ~(kTopK - 1);
        const unsigned subgroup_mask =
                ((1u << kTopK) - 1u) << subgroup_begin;
        const unsigned rank_peers = __match_any_sync(
                subgroup_mask, destination);
        const unsigned server_peers = __match_any_sync(
                subgroup_mask, destination / kRanksPerServer);
        const bool primary_rank = warp_lane == __ffs(rank_peers) - 1;
        const bool primary_server = warp_lane == __ffs(server_peers) - 1;
        auto* token_layout = is_token_in_rank +
                static_cast<std::int64_t>(index / kTopK) * cfg.world_size;
        token_layout[destination] = true;
        if (primary_rank)
            atomicAdd(num_tokens_per_rank +
                      (kSingleSource ? 0 : input_source * cfg.world_size) +
                      destination, 1);
        if (primary_server)
            atomicAdd(num_tokens_per_rdma_rank +
                      (kSingleSource ? 0 : input_source * cfg.num_servers) +
                      destination / kRanksPerServer, 1);
    }
}

PlanConfig make_config(int world_size,
                       int num_tokens_per_rank,
                       int token_padding,
                       std::int64_t dispatch_bytes_per_route,
                       std::int64_t expert_weight_bytes,
                       std::int64_t weight_chunk_bytes) {
    const auto topology = make_topology(world_size);
    return {topology.world_size, topology.num_servers, num_tokens_per_rank,
            topology.local_experts, topology.execution_slots,
            token_padding, dispatch_bytes_per_route,
            expert_weight_bytes, weight_chunk_bytes};
}

void validate_topk(const torch::Tensor& topk_idx) {
    TORCH_CHECK(topk_idx.is_cuda() && topk_idx.is_contiguous(),
                "topk_idx must be contiguous CUDA");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64,
                "topk_idx must be int64");
    TORCH_CHECK(topk_idx.dim() == 3 && topk_idx.size(2) == kTopK,
                "topk_idx must have shape [R,S,8]");
    TORCH_CHECK(supported_world_size(static_cast<int>(topk_idx.size(0))),
                "unsupported ProbeEP world size");
}

}  // namespace

void launch_probeep_plan_from_ipc_counts(
        const std::int64_t* local_topk_idx,
        void** buffer_ptrs,
        std::int64_t plan_reserve_offset,
        int source_rank,
        int world_size,
        int num_tokens,
        int token_padding,
        const std::int64_t* migration_budget_bytes,
        const std::int64_t* controller_summary,
        std::int64_t dispatch_bytes_per_route,
        std::int64_t expert_weight_bytes,
        std::int64_t weight_chunk_bytes,
        const ProbePlanWorkspace& workspace,
        cudaStream_t stream) {
    const auto cfg = make_config(
            world_size, num_tokens, token_padding, dispatch_bytes_per_route,
            expert_weight_bytes, weight_chunk_bytes);
    gather_ipc_counts<<<min(32, cfg.world_size), kThreads, 0, stream>>>(
            buffer_ptrs, plan_reserve_offset, workspace.global_counts,
            cfg.world_size);
    build_probeep_plan<<<1, kThreads, 0, stream>>>(
            workspace.global_counts, migration_budget_bytes,
            controller_summary, cfg,
            workspace.count_prefix, workspace.alloc_prefix, workspace.alloc,
            workspace.expert_slot, workspace.slot_count, workspace.slot_begin,
            workspace.replica_expert, workspace.slot_expert,
            workspace.server_load_before, workspace.server_load_after,
            workspace.server_padded_load_before,
            workspace.server_padded_load_after,
            workspace.assigned_tx_bytes, workspace.assigned_rx_bytes,
            workspace.dispatch_tx_bytes,
            workspace.dispatch_rx_bytes,
            workspace.pair_load_bytes, workspace.server_expert_rows,
            workspace.compute_intents,
            workspace.migration_budget_snapshot,
            workspace.endpoint_total_cap_bytes,
            workspace.admitted_experts, workspace.deferred_experts,
            workspace.chunk_table, workspace.plan_counts);
    admit_probeep_intents<<<1, kAdmissionThreads, 0, stream>>>(
            workspace.global_counts, workspace.count_prefix,
            workspace.endpoint_total_cap_bytes,
            workspace.compute_intents, workspace.alloc,
            workspace.assigned_tx_bytes, workspace.assigned_rx_bytes,
            workspace.dispatch_tx_bytes, workspace.dispatch_rx_bytes,
            workspace.pair_load_bytes, workspace.server_expert_rows,
            workspace.admitted_experts, workspace.deferred_experts,
            workspace.chunk_table, workspace.plan_counts, cfg);
    pack_server_local<<<cfg.num_servers, kThreads, 0, stream>>>(
            workspace.server_expert_rows, workspace.alloc,
            workspace.plan_counts, cfg);
    finalize_probeep_plan<<<1, kThreads, 0, stream>>>(
            workspace.count_prefix, workspace.server_expert_rows,
            workspace.alloc, workspace.alloc_prefix,
            workspace.expert_slot, workspace.slot_count,
            workspace.slot_begin, workspace.replica_expert,
            workspace.slot_expert, workspace.server_load_after,
            workspace.server_padded_load_after,
            workspace.plan_counts, cfg);
    clear_materialized_outputs<<<64, kThreads, 0, stream>>>(
            workspace.is_token_in_rank, workspace.num_tokens_per_rank,
            workspace.num_tokens_per_rdma_rank,
            workspace.num_tokens_per_exec_expert, num_tokens, 1, cfg);
    materialize_exec_counts<<<1, kNumExperts, 0, stream>>>(
            workspace.count_prefix, workspace.alloc,
            workspace.expert_slot, workspace.num_tokens_per_exec_expert,
            1, source_rank, cfg);
    const int routes = num_tokens * kTopK;
    const int nvs = cfg.num_servers * routes +
                    (token_padding - 1) * cfg.execution_slots;
    const bool segmented_ordinals =
            routes / kHistogramSegmentRoutes > 1;
    materialize_probe_routes<true><<<64, kThreads, 0, stream>>>(
            local_topk_idx, workspace.local_ordinal,
            workspace.count_prefix, workspace.alloc_prefix,
            workspace.expert_slot, workspace.slot_begin,
            workspace.route_dst, workspace.exec_rank, workspace.exec_slot,
            workspace.is_token_in_rank, workspace.num_tokens_per_rank,
            workspace.num_tokens_per_rdma_rank, routes, routes,
            source_rank, nvs, segmented_ordinals, cfg);
}

ProbePlanCuda plan_probeep_cuda(
        const torch::Tensor& topk_idx,
        const torch::Tensor& migration_budget_bytes,
        std::int64_t expert_weight_bytes,
        std::int64_t weight_chunk_bytes,
        int ranks_per_server,
        int local_experts,
        int replica_slots,
        int token_padding,
        std::int64_t learned_total_bytes) {
    validate_topk(topk_idx);
    const int world_size = static_cast<int>(topk_idx.size(0));
    const auto topology = make_topology(world_size);
    TORCH_CHECK(migration_budget_bytes.is_cuda() &&
                migration_budget_bytes.is_contiguous() &&
                migration_budget_bytes.scalar_type() == torch::kInt64 &&
                migration_budget_bytes.numel() == world_size,
                "migration_budget_bytes must be CUDA int64 [R]");
    TORCH_CHECK(ranks_per_server == kRanksPerServer &&
                (local_experts < 0 ||
                 local_experts == topology.local_experts) &&
                replica_slots == kReplicaSlots,
                "planner topology arguments do not match R/E256");
    TORCH_CHECK(token_padding > 0 && expert_weight_bytes > 0 &&
                weight_chunk_bytes > 0 && learned_total_bytes >= 0,
                "padding and weight sizes must be positive");
    TORCH_CHECK((expert_weight_bytes + weight_chunk_bytes - 1) /
                        weight_chunk_bytes <= kProbeMaxChunksPerExpert,
                "expert requires too many chunks");

    const int num_tokens = static_cast<int>(topk_idx.size(1));
    const int routes_per_rank = num_tokens * kTopK;
    const int total_routes = world_size * routes_per_rank;
    const int nvs = topology.num_servers * routes_per_rank +
                    (token_padding - 1) * topology.execution_slots;
    const auto ints = topk_idx.options().dtype(torch::kInt32);
    const auto int64s = topk_idx.options().dtype(torch::kInt64);
    const auto bools = topk_idx.options().dtype(torch::kBool);
    auto counts = torch::empty({world_size, kNumExperts}, ints);
    auto ordinal = torch::empty(topk_idx.sizes(), ints);
    auto count_prefix = torch::empty({world_size, kNumExperts}, ints);
    auto alloc_prefix = torch::empty({kNumExperts, world_size}, ints);
    auto alloc = torch::empty({kNumExperts, world_size}, ints);
    auto expert_slot = torch::empty({world_size, kNumExperts}, ints);
    auto route_dst = torch::empty(topk_idx.sizes(), ints);
    auto exec_rank = torch::empty(topk_idx.sizes(), ints);
    auto exec_slot = torch::empty(topk_idx.sizes(), ints);
    auto token_layout = torch::empty(
            {world_size, num_tokens, world_size}, bools);
    auto slot_count = torch::empty(
            {world_size, topology.execution_slots}, ints);
    auto slot_begin = torch::empty(
            {world_size, topology.execution_slots}, ints);
    auto replicas = torch::empty({world_size, kReplicaSlots}, ints);
    auto slot_expert = torch::empty(
            {world_size, topology.execution_slots}, ints);
    auto tokens_per_rank = torch::empty({world_size, world_size}, ints);
    auto tokens_per_server = torch::empty(
            {world_size, topology.num_servers}, ints);
    auto tokens_per_slot = torch::empty(
            {world_size, world_size * topology.execution_slots}, ints);
    auto server_before = torch::empty({topology.num_servers}, ints);
    auto server_after = torch::empty({topology.num_servers}, ints);
    auto padded_before = torch::empty({topology.num_servers}, ints);
    auto padded_after = torch::empty({topology.num_servers}, ints);
    auto assigned_tx = torch::empty({world_size}, int64s);
    auto assigned_rx = torch::empty({world_size}, int64s);
    auto dispatch_tx = torch::empty({world_size}, int64s);
    auto dispatch_rx = torch::empty({world_size}, int64s);
    auto pair_load = torch::empty(
            {topology.num_servers, topology.num_servers, kRanksPerServer},
            int64s);
    auto server_expert_rows = torch::empty(
            {kNumExperts, topology.num_servers}, ints);
    auto compute_intents = torch::empty(
            {kProbeMaxComputeIntents, kIntentFields}, ints);
    auto budget_snapshot = torch::empty({world_size}, int64s);
    auto endpoint_total_cap = torch::empty({world_size}, int64s);
    // Diagnostic-only input for exercising the production controller/planner
    // boundary on one GPU.  Only summary[5] is consumed by the planner.
    auto controller_summary = torch::full(
            {6}, learned_total_bytes, int64s);
    auto admitted = torch::full({kProbeMaxRemoteReplicas}, -1, ints);
    auto deferred = torch::zeros({kNumExperts}, bools);
    auto chunks = torch::full(
            {kProbeMaxChunks, kProbeChunkFields}, -1, int64s);
    auto plan_counts = torch::zeros({kProbePlanCounterFields}, ints);

    const auto diagnostic_wire_bytes =
            (kHidden + kFp8Scales * static_cast<int>(sizeof(float)) +
             internode::get_source_meta_bytes() +
             kTopK * static_cast<int>(sizeof(int) + sizeof(float)) +
             static_cast<int>(sizeof(int4)) - 1) /
            static_cast<int>(sizeof(int4)) *
            static_cast<int>(sizeof(int4));
    const auto cfg = make_config(
            world_size, num_tokens, token_padding, diagnostic_wire_bytes,
            expert_weight_bytes, weight_chunk_bytes);
    const auto stream = at::cuda::getCurrentCUDAStream();
    serial_histogram_and_ordinal<<<world_size, 32, 0, stream>>>(
            topk_idx.data_ptr<std::int64_t>(), counts.data_ptr<int>(),
            ordinal.data_ptr<int>(), routes_per_rank, world_size);
    build_probeep_plan<<<1, kThreads, 0, stream>>>(
            counts.data_ptr<int>(),
            migration_budget_bytes.data_ptr<std::int64_t>(),
            learned_total_bytes > 0
                    ? controller_summary.data_ptr<std::int64_t>() : nullptr,
            cfg,
            count_prefix.data_ptr<int>(), alloc_prefix.data_ptr<int>(),
            alloc.data_ptr<int>(), expert_slot.data_ptr<int>(),
            slot_count.data_ptr<int>(), slot_begin.data_ptr<int>(),
            replicas.data_ptr<int>(), slot_expert.data_ptr<int>(),
            server_before.data_ptr<int>(), server_after.data_ptr<int>(),
            padded_before.data_ptr<int>(), padded_after.data_ptr<int>(),
            assigned_tx.data_ptr<std::int64_t>(),
            assigned_rx.data_ptr<std::int64_t>(),
            dispatch_tx.data_ptr<std::int64_t>(),
            dispatch_rx.data_ptr<std::int64_t>(),
            pair_load.data_ptr<std::int64_t>(),
            server_expert_rows.data_ptr<int>(),
            compute_intents.data_ptr<int>(),
            budget_snapshot.data_ptr<std::int64_t>(),
            endpoint_total_cap.data_ptr<std::int64_t>(),
            admitted.data_ptr<int>(), deferred.data_ptr<bool>(),
            chunks.data_ptr<std::int64_t>(), plan_counts.data_ptr<int>());
    admit_probeep_intents<<<1, kAdmissionThreads, 0, stream>>>(
            counts.data_ptr<int>(), count_prefix.data_ptr<int>(),
            endpoint_total_cap.data_ptr<std::int64_t>(),
            compute_intents.data_ptr<int>(), alloc.data_ptr<int>(),
            assigned_tx.data_ptr<std::int64_t>(),
            assigned_rx.data_ptr<std::int64_t>(),
            dispatch_tx.data_ptr<std::int64_t>(),
            dispatch_rx.data_ptr<std::int64_t>(),
            pair_load.data_ptr<std::int64_t>(),
            server_expert_rows.data_ptr<int>(), admitted.data_ptr<int>(),
            deferred.data_ptr<bool>(), chunks.data_ptr<std::int64_t>(),
            plan_counts.data_ptr<int>(), cfg);
    pack_server_local<<<cfg.num_servers, kThreads, 0, stream>>>(
            server_expert_rows.data_ptr<int>(), alloc.data_ptr<int>(),
            plan_counts.data_ptr<int>(), cfg);
    finalize_probeep_plan<<<1, kThreads, 0, stream>>>(
            count_prefix.data_ptr<int>(), server_expert_rows.data_ptr<int>(),
            alloc.data_ptr<int>(), alloc_prefix.data_ptr<int>(),
            expert_slot.data_ptr<int>(), slot_count.data_ptr<int>(),
            slot_begin.data_ptr<int>(), replicas.data_ptr<int>(),
            slot_expert.data_ptr<int>(), server_after.data_ptr<int>(),
            padded_after.data_ptr<int>(), plan_counts.data_ptr<int>(), cfg);
    clear_materialized_outputs<<<256, kThreads, 0, stream>>>(
            token_layout.data_ptr<bool>(), tokens_per_rank.data_ptr<int>(),
            tokens_per_server.data_ptr<int>(), tokens_per_slot.data_ptr<int>(),
            world_size * num_tokens, world_size, cfg);
    materialize_exec_counts<<<world_size, kNumExperts, 0, stream>>>(
            count_prefix.data_ptr<int>(), alloc.data_ptr<int>(),
            expert_slot.data_ptr<int>(), tokens_per_slot.data_ptr<int>(),
            world_size, -1, cfg);
    materialize_probe_routes<false><<<256, kThreads, 0, stream>>>(
            topk_idx.data_ptr<std::int64_t>(), ordinal.data_ptr<int>(),
            count_prefix.data_ptr<int>(), alloc_prefix.data_ptr<int>(),
            expert_slot.data_ptr<int>(), slot_begin.data_ptr<int>(),
            route_dst.data_ptr<int>(), exec_rank.data_ptr<int>(),
            exec_slot.data_ptr<int>(), token_layout.data_ptr<bool>(),
            tokens_per_rank.data_ptr<int>(), tokens_per_server.data_ptr<int>(),
            routes_per_rank, total_routes, 0, nvs, false, cfg);
    return {route_dst, exec_rank, exec_slot, token_layout,
            slot_count, slot_begin, replicas, slot_expert, alloc,
            tokens_per_rank, tokens_per_server, tokens_per_slot,
            server_before, server_after, padded_before, padded_after,
            assigned_tx, assigned_rx, dispatch_tx, dispatch_rx,
            pair_load, compute_intents, budget_snapshot, endpoint_total_cap,
            admitted, deferred, chunks,
            plan_counts, nvs};
}

}  // namespace deep_ep::probeep
