#pragma once

#include <cstdint>

namespace deep_ep::probeep {

// ProbeEP keeps DeepEP's physical topology: every NVSHMEM PE represents one
// eight-GPU NVLink domain and equal local GPU indices form one RDMA rail.
// DSV3 contributes 256 routed experts per MoE layer; world size is runtime
// state so the same operator can run on every DeepEP-supported server count.
inline constexpr int kRanksPerServer = 8;
inline constexpr int kPhysicalNicsPerServer = 4;
inline constexpr int kRailsPerPhysicalNic = 2;
inline constexpr int kLogicalRailGbps = 200;
inline constexpr int kPhysicalNicGbps = 400;
static_assert(kPhysicalNicsPerServer * kRailsPerPhysicalNic ==
              kRanksPerServer);
static_assert(kLogicalRailGbps * kRailsPerPhysicalNic == kPhysicalNicGbps);
inline constexpr int kNumExperts = 256;
inline constexpr int kTopK = 8;
inline constexpr int kMaxServers = 16;
inline constexpr int kMaxWorldSize = kRanksPerServer * kMaxServers;
inline constexpr int kMaxLocalExperts = 16;  // multi-server minimum is EP16

// Match the ProbeEP reference/Test01 contract: one replica slot per local
// expert.  The previous 32-slot over-allocation doubled the physical grouped
// FFN and expert-pool footprint even though the corrected Eval20 plans use at
// most nine remote experts on any rank.
inline constexpr int kReplicaSlots = 16;
inline constexpr int kMaxExecutionSlots =
        kMaxLocalExperts + kReplicaSlots;
inline constexpr int kPlanRingSlots = 3;
inline constexpr int kMaxTokensPerRank = 4096;
inline constexpr int kTokenPadding = 8;
inline constexpr int kHidden = 7168;
inline constexpr int kFp8ScaleBlock = 128;
inline constexpr int kFp8Scales = kHidden / kFp8ScaleBlock;
inline constexpr int kNumSms = 24;
inline constexpr int kNumChannels = kNumSms / 2;

struct Topology {
    int world_size;
    int num_servers;
    int local_experts;
    int execution_slots;

    constexpr int server(int rank) const {
        return rank / kRanksPerServer;
    }
    constexpr int lane(int rank) const {
        return rank % kRanksPerServer;
    }
    constexpr int owner_rank(int expert) const {
        return expert / local_experts;
    }
    constexpr int owner_server(int expert) const {
        return server(owner_rank(expert));
    }
};

inline constexpr bool supported_world_size(int world_size) {
    if (world_size < 16 || world_size > kMaxWorldSize ||
        world_size % kRanksPerServer != 0 ||
        kNumExperts % world_size != 0)
        return false;
    const int servers = world_size / kRanksPerServer;
    return servers == 2 || servers == 4 || servers == 8 || servers == 16;
}

inline constexpr Topology make_topology(int world_size) {
    const int local = kNumExperts / world_size;
    return {world_size, world_size / kRanksPerServer, local,
            local + kReplicaSlots};
}

inline constexpr int max_execution_rows(const Topology& topology) {
    return topology.num_servers * kMaxTokensPerRank * kTopK +
           (kTokenPadding - 1) * topology.execution_slots;
}

inline constexpr int max_transport_rows(const Topology& topology) {
    return topology.world_size * kMaxTokensPerRank;
}

}  // namespace deep_ep::probeep
