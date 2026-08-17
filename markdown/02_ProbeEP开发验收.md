# ProbeEP 开发验收

## 1. 目标

ProbeEP 的唯一实现目录是 `src/deepep-probeep`。生产热路径以 DeepEP normal HT CUDA/RDMA 数据面为底座；UltraEP + 官方 HybridEP 只属于独立 baseline，不参与 ProbeEP 实现。

正式性能比较固定为 5 个算法：

| 类别 | 算法 | 代码路径 | 均衡范围 |
|---|---|---|---|
| baseline | NCCL | `torch.distributed`/benchmark adapter | 不做专家均衡 |
| baseline | DeepEP | `src/deepep` | 不做专家均衡 |
| baseline | DeepEP-MoonEP | `src/deepep-moonep` | server 内均衡 |
| baseline | UltraEP + HybridEP | `src/ultraep` + `src/ultraep/HybridEP` | server 内均衡 |
| ours | ProbeEP | `src/deepep-probeep` | 跨 server + server 内均衡 |

| 目标 | 要求 |
|---|---|
| 跨 server 均衡 | server padded load 必须下降 |
| server 内均衡 | 复用 MoonEP 思想，rank padded load 必须下降 |
| 网络预算 | sampled controller 控制 expert replica，不让 RDMA 成为新瓶颈 |
| chunk 调度 | expert weight 分 chunk 到 RDMA path，优先均衡有向 `src_server -> dst_server` 的 rail 水位 |
| 热路径 | CUDA/C++/NVSHMEM/IBGDA，不走 Python planner |
| 融合 | 全部 ProbeEP 功能由一个主算子承载；controller/plan/admission/chunk/packing/weight/dispatch 只是其内部阶段 |

## 2. ProbeEP 核心算法

ProbeEP 是一个完整的反馈闭环算法，不是若干独立算法的组合。独立探测提供控制样本，跨 server compute planner 产生迁移意图，`0.90` controller 决定本轮接纳量，server 内 planner 完成二次均衡，pair-aware chunk scheduler 决定已接纳权重使用哪条 rail；本轮实测结果再反馈给下一次同类窗口。

```text
Attention/MoE 独立观测
        -> 跨 server compute planning
        -> 0.90 NIC budget 与完整 replica admission
        -> server 内二次 compute packing
        -> 按 (src_server,dst_server) rail 水位发送 weight chunks
        -> Weight+Dispatch 实测反馈到下一次同类窗口
        = one ProbeEP closed-loop state transition
```

| 闭环环节 | 算法语义 |
|---|---|
| Attention/MoE 独立探测 | 两类观测和预算状态互不污染，当前通信只能使用已完成的同类计算窗口 |
| 计算规划与网络调度解耦 | compute planner 不读取 NIC 水位；已接纳 weight chunk 优先均衡对应有向 server pair 的 rail 水位 |
| 跨 server 优先、server 内二次均衡 | 先降低全局 server padded load，再在确定的 server mapping 内降低全局最慢 rank |
| `alpha=0.90` 反馈式探测 | 根据同一 overlap window 的实测 `Cmax/Nmax` 和瓶颈 endpoint 实际字节更新下一次同类 migration budget |

只有这些环节连成闭环后才构成 ProbeEP。排序方法、kernel 划分、通信原语和融合边界不属于核心算法定义。

### 2.1 Attention/MoE 独立探测

一次 MoE invocation 是：

```text
Router/Gate -> [Weight Migration + Dispatch] -> Expert FFN -> Combine
```

ProbeEP 的控制样本是方括号内通信阶段与计算 stream 在相邻同步边界之间形成的 overlap window，不是整个 invocation。双 microbatch 的两条控制链为：

```text
Layer L: A1 || W0+D0 -> Attention observation A[L]
Layer L: E0 || W1+D1 -> MoE observation M[L]

Layer L+1 / MB0 Dispatch 只消费 A[L]
Layer L+1 / MB1 Dispatch 只消费 M[L]
```

控制状态按计算类型划分：

```text
B_attention[rank]
B_moe[rank]
```

每条 Dispatch observation 必须显式携带 `compute_kind=attention|moe`，并且只更新对应状态。Attention 和 MoE 的 history、budget、producer/consumer ID 完全独立，任何一条链缺样本时都不能拿另一条链代替。只能使用决策时已经完成的上一层、同 round、同类观测，不能使用当前层尚未完成的窗口或未来 MoE 时间作为 oracle。Layer 0 没有上一 MoE Layer，使用 runtime 初始化的明确 bootstrap budget，并在 plan log 标为 `feedback_source=bootstrap`；不得伪造实测 observation。持久 worker 开始新的 repeat/序列时必须同时清空 A/M summary 并恢复 bootstrap budget，禁止继承上一 repeat 最后一层的状态。Combine 完整执行并记录 telemetry，但不更新 migration budget。

物理 overlap 会跨 microbatch，但控制状态不会交叉：

| 控制链 | observation producer | 下一层 consumer | 唯一状态 |
|---|---|---|---|
| Attention | `MB1 Attention || MB0 W+D` | `MB0 Dispatch` | `B_attention` |
| MoE | `MB0 Expert || MB1 W+D` | `MB1 Dispatch` | `B_moe` |

每个 rank 的网络时间定义为完整 Weight+Dispatch stage elapsed：

```text
release_i = 与该 Dispatch observation 配对的 compute start
done_i    = max(last Weight/Dispatch TX done,
                last Weight/Dispatch RX done)
N_i       = done_i - release_i
```

TX/RX 可以 full-duplex，不能把两个方向的 duration 相加。NIC active-time union 只用于利用率诊断，不能替代 `N_i`。plan 每个 invocation 根据当前 Gate 重新计算；Attention/MoE budget 分别跨 layer 保留，因为它们学习的是不同计算窗口下、共享 NIC endpoint 的可掩盖通信量。

#### 2.1.1 真实 Observation Producer 与论文 Benchmark 接入

真实探测需要执行 scheduler 提供计算边界。当前论文阶段由 dual-microbatch benchmark 自己承担 scheduler 角色，不要求接入 vLLM/SGLang；未来业务接入复用同一接口。benchmark 只能使用刚完成的真实窗口，不能预填合成 observation，也不能让 ProbeEP 猜测 Attention/MoE 边界。

| 信息/动作 | 唯一责任方 |
|---|---|
| `compute_kind=attention|moe`、microbatch slot | 当前 benchmark scheduler；未来业务 scheduler |
| Attention/MoE compute start/done event | 当前 benchmark compute stream；未来业务 compute stream |
| Weight+Dispatch TX/RX done 与实际完成字节 | ProbeEP CUDA/RDMA runtime |
| compute/network event 因果配对 | 当前 benchmark adapter；未来下沉到业务 runtime |
| 上一层、同 round、同类 observation | 当前 benchmark adapter 的 A/M 分离 device ring；未来由业务 runtime ring 持有 |
| A/M budget 更新与下一轮消费 | ProbeEP controller/runtime |
| `A1 || W0+D0`、`E0 || W1+D1` 调度 | dual-microbatch benchmark scheduler |

当前论文 benchmark 在 Layer L 的 timed window 结束后读取已完成的 CUDA event、实际 Dispatch destination rows 和 cache-miss Weight chunk bytes，按 `(phase, round, compute_kind)` 分键保存 device feedback；Layer L+1 的同 round 调用只取同一 `compute_kind`。所需 device collectives 计入 Layer L 的 `plan_ms` 与 `e2e_ms`，不能藏在测量外。adapter 不扫描 routes，也不执行 controller、排序、admission 或 packing。下一层 `balanced_dispatch` 在 CUDA/C++ 内融合消费 feedback，并只更新对应 A 或 M controller state。合成 `completed_observation` 只用于 consumer 正确性测试，不能进入正式 benchmark。

将 event 配对和 observation ring 完全下沉到 vLLM/SGLang/训练 runtime 是未来业务接入项，不是当前论文 prototype 的完成门禁；文档与结果必须区分“benchmark 闭环已实现”和“业务 runtime producer 未接入”。

### 2.2 计算规划与网络调度解耦

#### 2.2.1 Compute planner 只优化计算

Compute planner 的层级目标固定为：

```text
1. minimize max_server(padded_expert_compute_proxy)
2. minimize max_rank(expert_compute) under the admitted server mapping
```

planner 只读取 Gate histogram、raw/padded compute load 和 placement，不读取 NIC load、RDMA bandwidth、migration budget 或 weight bytes。它输出跨 server migration intents；NIC controller 只决定本轮能接纳哪些 intents；chunk scheduler 只决定已接纳 replica 的 chunks 走哪些 rail。网络状态不能回头改变 expert 的 source server、destination server 或计算优先级。

迁移只允许按完整 expert replica 原子接纳：全部 weight chunks 可调度才提交，否则整个 intent deferred。每接纳一个 intent 都会改变 route-to-server layout 和 Token Dispatch baseline，因此必须基于新 layout 重新计算逐 rail Dispatch TX/RX，并重新验证已经接纳的 source TX、destination RX 预算。compute planner 的单调内部迁移会合并成唯一的最终 `home_server -> destination_server` intent；admission 按 hot-first 顺序选择第一个“在当前已接纳 mapping 上严格改善”的 intent，并回访先前暂不可改善的 intent。不能用一次固定顺序回放就永久 defer，否则 4 台以上 server 会丢失可行的 compute plan。

#### 2.2.2 Chunk 首先均衡目标 server 对应的 rail 水位

weight chunk 的第一调度水位不是全局 NIC 总负载，而是有向 server pair 的 rail 负载：

```text
pair_load[src_server, dst_server, rail]
```

例如 `server0 -> server1` 的 chunk 只先比较该 pair 的各 rail 水位，不允许 `server0 -> server2` 的历史流量改变它的主 water-filling 顺序。一个大小为 `x` 的 chunk 只有在同一 rail 的 source TX 和 destination RX 都有足够预算时才是合法候选。

确定性选择顺序为：

```text
1. 最小化 pair_load[src,dst,rail] + x
2. pair 水位相同时，避开 Token+Weight endpoint 总量更高的 rail
3. 再按 rail id 打破平局
```

因此 endpoint 总负载只负责物理容量检查和同 pair 水位下的 tie-break，不能压过 `pair_load[src,dst,rail]` 成为首要目标。每放一个 chunk 同时更新：

```text
pair_load[src,dst,rail]
assigned_tx[src,rail]
assigned_rx[dst,rail]
```

### 2.3 跨 server 优先、server 内二次均衡

这里的“优先 RDMA 均衡”是先通过跨 server replica 降低 server 间计算不平衡，不是让 NIC 字节本身成为 compute planner 的目标。

#### 2.3.1 第一阶段：跨 server 计算均衡

```text
total_routes              = num_ranks * tokens_per_rank * topk
target_routes_per_rank    = total_routes / num_ranks
target_routes_per_server  = target_routes_per_rank * gpus_per_server
```

第一阶段同时处理所有 server：

1. 计算所有 server 的 `surplus=max(load-target,0)` 和 `deficit=max(target-load,0)`。
2. 从 donor expert 中产生有确定性优先级的迁移候选，优先处理能用较少 replica 承接更多 routes 的热点 expert。
3. 优先填补 deficit 最大的 receiver，并复用已经打开的 `(expert,destination_server)` replica。
4. 达到 floor/ceil raw quota 后，以 `sum_e ceil(routes[s,e]/token_padding)*token_padding` 做 padding refinement。
5. 只保留能严格降低词典序目标 `(global server padded max, global server padded spread, Σ server padded load²)` 的 intent；平方和仅在前两项完全相等时破除“多台同热/同冷服务器”的单步停滞，不能为了填满 NIC 窗口产生无计算收益迁移。

这里要求的是候选语义和目标一致，不要求使用比较排序。CUDA 实现可使用 bucket、radix、selection 或融合扫描，只要结果满足目标和确定性约束。

#### 2.3.2 第二阶段：server 内 rank 均衡

NIC admission 确定最终 route-to-server mapping 后，每个 server 再以 `token_padding` block 做 capacity-aware packing：

1. 以 padded-block capacity 为 rank 目标，而不是强制 real routes 完全相等。
2. 优先填充剩余容量最大的 rank，tie 时优先 expert home rank。
3. 同一 expert 尽量连续放在较少 rank，避免为追求绝对均匀创建大量本地副本。
4. remote expert 先选择 seed rank；需要 server 内 fan-out 时再通过本地高速互连复制。

验收顺序固定为：先检查 server padded maximum/spread；两者暂时并列时只允许平方和严格下降，再检查最终 global max-rank padded compute 是否下降。

### 2.4 `0.90` 反馈式探测与预算划分

#### 2.4.1 状态划分

| 维度 | 划分方法 |
|---|---|
| 计算类型 | Attention 和 MoE 两套独立状态 |
| 物理 endpoint | 每个 rank/rail 单独维护预算 |
| 方向 | source TX 与 destination RX 分别检查，full-duplex footprint 取两者最大值 |
| 字节类型 | Token Dispatch 是不可裁剪 baseline；controller 只控制额外 Expert Weight bytes |
| server pair | `pair_load[src,dst,rail]` 只用于该有向 pair 的 chunk water-filling |

对当前 `compute_kind`：

```text
Cmax = max_rank(attention_compute_ns) or max_rank(moe_compute_ns)
Nmax = max_rank(weight_dispatch_stage_ns)

P_i = max(
  dispatch_tx_i + migration_tx_i,
  dispatch_rx_i + migration_rx_i
)

Bsample = max(P_i for i where N_i == Nmax)
scale   = alpha * Cmax / Nmax

Bprobe_total  = floor(scale * Bsample)
Btheory_total = floor(Rnic_bytes_per_ns * Cmax)
Btotal        = min(Bprobe_total, Btheory_total)

Bhard_tx_i = max(0, Btheory_total - dispatch_tx_i)
Bhard_rx_i = max(0, Btheory_total - dispatch_rx_i)

Bnext_i = min(
  max(0, Btotal - max(dispatch_tx_i, dispatch_rx_i)),
  Bhard_tx_i,
  Bhard_rx_i
)
```

默认 `alpha=0.90`，表示目标是让瓶颈 Weight+Dispatch stage 接近对应计算窗口的 90%，留下 10% 工程余量。目标 H20 每台服务器有 4 个 400-Gbps 物理 NIC，每个物理口一分二映射成两个 GPU logical rail；因此 controller 的 endpoint 速率是 `Rrail=200 Gbps=25 bytes/ns`，8 个 logical rail 的总线速仍为 `4×400 Gbps`。不得把物理口 400 Gbps 错填成每个 logical rail 的速率。

该控制器使用同一个全局 `scale` 决定增窗或减窗方向，但每个 rank 因 Dispatch baseline 不同可得到不同 migration budget。若 Token Dispatch 本身已经超过目标，migration budget 降为 0，不能裁剪 token。测量无效或没有实际通信字节时必须保持完整原状态，包括逐 endpoint migration budget 和 `summary/learned_total`；时间戳为正但 `Bsample=0` 仍不是有效带宽样本，不得将该 A/M 状态行缩到 0。首次运行只能使用受理论线速约束的 fallback。

`0.90` 是反馈目标而非逐轮精确等式。完整 expert admission、离散 chunk、Gate 变化和网络排队都可能使观测值在目标附近波动；正确性要求是方向正确、预算有硬上限、A/M 状态隔离，而不是每轮恰好得到 `Nmax/Cmax=0.90`。

### 2.5 核心算法验收条件

| 条件 | 验收要求 |
|---|---|
| 计算收益 | 每个接纳 intent 在实际提交时必须严格改善 server padded objective；暂不可改善的 final intent 先跳过并回访，NIC 空闲不能产生无收益迁移 |
| 完整接纳 | 完整 expert 的全部 chunks 可调度才提交，不允许部分权重执行 |
| 动态重验 | 每次接纳后按新 placement 重算 Dispatch baseline，并重验 source TX/destination RX 预算 |
| A/M 隔离 | 当前 observation 只更新对应计算类型的状态，Combine 不更新 budget |
| pair-aware rail | chunk 第一水位必须是 `pair_load[src,dst,rail]`，总 endpoint 水位只能做容量检查和 tie-break |
| 两阶段顺序 | 先改善 server 间 padded load，再在 admitted server mapping 内改善全局最慢 rank |

## 3. 非核心功能与正确性契约

这些功能用于承载 ProbeEP，但不是核心算法本身；具体实现可以为性能修改：

| 项 | 算法契约 |
|---|---|
| histogram | 在线决策使用 `(server,expert)` counts，不对全部 expanded routes 做全局排序 |
| admission | 相对当前已接纳 mapping 仍有计算收益，且完整 expert 的全部 chunks 可调度时才原子提交 |
| replica reuse | 已存在或本轮已打开的 `(expert,dst_server)` replica 不重复传权重 |
| weight size | 从当前模型 shape/dtype 动态计算，不硬编码 84 MiB 或固定 chunk 数 |
| consumer ready | remote replica 的全部 chunks 和所需 tokens 到齐后才能执行 Expert FFN |
| per-rail dependency | 每条 source rail 完成自己的 remote Weight TX 后立即推进自己的 Dispatch TX |
| full-duplex staging | 每个 plan slot 的 TX 与 RX 使用独立 symmetric staging bank；两套 offset 均从 0 分配，禁止别名 |
| barrier | 禁止全局 Weight completion barrier；不同 rail、不同 server 不互相等待 |
| conservation | routes 守恒，migration TX bytes 等于 migration RX bytes |
| capacity | 每个 chunk 同时满足 source TX 与 destination RX 的逐 rail budget |
| determinism | 相同 Gate、controller state 和配置产生确定性 plan；tie 由稳定 ID 解决 |
| lifecycle | plan 每 invocation 重算；A/M state 只在同一序列内跨 layer 保存；每个 repeat 的 Layer 0 显式 reset 到 bootstrap |

记 server 数为 `P`、每 server GPU 数为 `G=8`、world size 为 `W=P×G`、expert 数为 `E=256`、本 rank route 数为 `Q=S×K`、compute intent 数为 `I`、完整 expert chunk 数为 `C`。当前 CUDA 实现的上界为：

| 阶段 | 当前复杂度 | 说明 |
|---|---:|---|
| local histogram + ordinal | `O(Q)` | warp match + segmented prefix；无临时 allocation |
| compact exchange | 每 rail `O(P×E)` bytes | expanded TopK 不跨机 |
| compute intent generation | `O(E×W + I×P²)` | 固定位宽 radix 只排一次 hot groups；quota 与 padding refinement 后 `I≤2E+2P+1`；初始 Dispatch footprint 由 256 expert threads 并行求交 |
| admission + Dispatch 重验 | `O(I²×P + I×(W+G+C×G))` | 稳定扫描回访暂不可改善的 final intent，不做排序；source/destination 区间用双指针 `O(W)` 求交；一个 warp 的 lanes 0..7 并行评估 rail；完整 expert 事务试排 |
| server-local packing | `O(P×(E+L×G))` | `P` 个 CUDA block 并行；固定趟 radix 得到 hot-first 顺序，`L` 是实际 local placement 数；不做 `E²` comparison scan |
| local route lowering | `O(Q log W)` | 每 rank 本地展开；不 all-gather route |
| controller | `O(W)` | 单个 device kernel，A/M 两行状态独立 |

`E=256`、`G=8` 和 `P≤16` 均为有界 metadata；route 数只进入 histogram 和最终 lowering，不进入 compute candidate search。Packing 不是零开销阶段，必须计入 planner 总时间。当前 `plan_counts[9:13]` 分别记录 init+intent、admission、server-local packing、finalization 的 device clock cycles；它们用于阶段归因，正式性能仍以 CUDA event/nsys 的完整 operator critical path 为准。

## 4. 单一 ProbeEP 主算子的高性能实现原则

算法回答“做什么”，主算子实现决定“如何以最低开销整体完成”。下表各项都是同一个 ProbeEP operator 的内部阶段，不是单独的公开算子；内部实现均可持续替换和融合：

```text
Buffer::balanced_dispatch(inputs, topk, expert_state,
                          completed_same_kind_observation, persistent_state,
                          expert_weight_version)
  internal: histogram + A/M state selection
  internal: inter-server plan + 0.90 admission
  internal: server-local packing + route lowering
  internal: pair-aware weight chunk schedule
  internal: weight transport + token dispatch
  output: grouped FFN input + opaque replay handle + updated persistent state

framework grouped FFN(handle.slot_count)
Buffer::balanced_combine(expert_output, handle)
optional backward entries reuse the same handle
```

“一个算子整体”指一个公开 runtime/operator 入口、一个持久状态所有者和一条统一的异步事件链。内部可以为了跨 rank 同步、RDMA progress 和 kernel occupancy 拆成少量 CUDA/C++ 阶段，但这些阶段不得成为 Python 逐项调用的独立算子。训练 backward 可以有框架要求的 backward entry，但必须复用同一个 opaque plan/state，不能重新运行另一套 planner。

| 可优化部分 | 高性能方向 | 不得破坏的算法语义 |
|---|---|---|
| histogram | warp/block histogram，只交换紧凑 `P×E` counts | 当前 Gate 负载统计正确 |
| 候选生成 | counting bucket、radix、select、固定 metadata scan | 跨 server 计算目标和确定性 |
| controller + admission | warp 内 rail 角色特化，批量试排全部 chunks | A/M 独立、完整 expert 原子接纳 |
| chunk scheduler | 批量生成 chunk table，不做 per-chunk host launch | `pair_load[src,dst,rail]` 是第一水位，双端预算合法 |
| local packing + route lowering | 融合 prefix/slot/row 生成 | 先 server、后 rank 的两阶段结果 |
| weight + dispatch | per-rail event chain，尽早 overlap | 本 rail Weight TX 在本 rail Dispatch TX 前完成，无全局 barrier |
| runtime state | 持久 device buffer、graph-friendly 固定 workspace | plan 生命周期与反馈因果正确 |

DeepEP 的大数据面使用 `num_sms` 限定的 CTA 和 sender/coordinator/forwarder/receiver warp 角色特化；ProbeEP 直接复用这条 Dispatch/Combine 路径。Planner 不长期常驻空转 CTA：常驻的是三槽 plan ring、workspace 和 controller state，metadata kernels 在同一 CUDA stream 内短时执行。长期占用 planner SM 会和双 microbatch 的 FFN/DeepEP stream 抢占资源，因此不是当前设计。

主算子热路径禁止 Python route scan、CPU polling、动态 host allocation、逐 chunk launch、子阶段间 host round-trip 和 Weight completion world barrier。全局 plan 必须等待紧凑 histogram 到齐，因此只保留同 rail NVSHMEM communicator 内的一次 device-side count synchronization；它必须单独计时。controller/planner 若成为端到端瓶颈，应继续融合或更换数据结构，而不是为了逐行复刻参考算法保留低效实现。性能计时必须以完整 ProbeEP 主算子及端到端 overlap 为主，内部阶段计时只用于诊断。

## 5. 功能开发表

状态定义：`已实现（代码）` 表示逻辑已进入生产 CUDA/C++ 路径并通过 SM90 编译；`目标机待验收` 表示代码已完成但当前没有 H20 多机 RDMA 环境，不能宣称运行 PASS；`待实现` 表示代码仍缺失。实现完成和目标机验收必须分开记录。

| ID | 功能 | 当前入口 | 状态 | 当前结论/剩余工作 |
|---|---|---|---|---|
| F01 | runtime topology | `probeep_topology.hpp` | 已实现（代码） | DSV3 E256/TopK8；8 GPU/server；每 server `4×400G physical NIC → 8×200G logical rail`；EP16/32/64/128 runtime topology |
| F02 | compact histogram 与 route ordinal | `moonep_exchange.cu`、`probeep_plan.cu` | 已实现（代码） | 每条 rail 只交换 E256 histogram；expanded TopK 不跨机 |
| F03 | Attention/MoE 独立 feedback state | `BalancedRuntime::probe_migration_budget_`、`probe_controller_summary_` | 已实现（代码） | `[2,R]` budget 与 `[2,6]` summary；本次调用只更新指定 `compute_kind` |
| F04 | `alpha=0.90` 实测比例 controller | `probeep_controller.cu` | 已实现（代码） | 有 bottleneck endpoint sample、line-rate cap、Dispatch baseline 扣减和 invalid hold |
| F05 | 跨 server compute planner | `probeep_plan.cu::build_probeep_plan` | 已实现（代码） | 同时处理 2/4/8/16 server；不读 bandwidth、budget、weight bytes 或 rail 水位；一次固定位宽 device radix，不重复全 E 选择 |
| F06 | padded objective 与 server 内二次 packing | `build_probeep_plan`、`pack_server_local` | 已实现（代码） | 跨 server 只提交严格 padded 改善；admission 后每 server 一个 CUDA block，以固定趟 radix hot-first 排序做 padding-block capacity packing |
| F07 | 完整 expert 原子 admission | `admit_probeep_intents`、`schedule_complete_expert_warp` | 已实现（代码） | 稳定回访暂不可改善 final intent；选中后先试排全部 chunks，任一 rail 超预算则整 expert deferred |
| F08 | directed server-pair rail water-filling | `schedule_complete_expert_warp` | 已实现（代码） | lanes 0..7 并行评估独立 `[src_server,dst_server,rail]` 水位；endpoint 只做容量和 tie-break |
| F09 | placement 变化后的 Dispatch baseline 动态重算与重验 | `accumulate_expert_dispatch`、`admit_probeep_intents` | 已实现（代码） | 每次 candidate 先移除旧 expert footprint、试放新 placement，再重验全部 endpoint；区间匹配为严格等价 `O(W)` 双指针；公开预算使用 `min(occurrence_bytes, S×(P−1)×wire_bytes)` 的 server-deduplicated 安全上界，下一层再由实测 `num_tokens_per_rdma_rank` 闭环 |
| F10 | route-to-rank/slot/physical expert lowering | `materialize_exec_counts`、`materialize_probe_routes` | 已实现（代码） | 输出 `exec_rank/exec_slot/route_dst/slot_begin`；TopK8 warp 去重；rank 搜索为二分 |
| F11 | 双 microbatch plan ring 与 generation guard | `BalancedRuntime`、`BalancedHandle` | 已实现（代码） | 三个固定 plan slot、generation guard、forward/backward 生命周期和 CUDA event 依赖 |
| F12 | 单一 ProbeEP 主算子 | `Buffer::balanced_dispatch` | 已实现（代码） | 一次 C++ 调用完成反馈更新、histogram、plan、admission、packing、Weight 与 DeepEP Dispatch；Python binding 只转发已完成 device feedback 与输入 tensor |
| F13 | DeepEP HT Dispatch/Combine | `internode::balanced_*` | 已实现（代码） | ProbeEP 直接使用 DeepEP normal RDMA+NVLink 数据面；UltraEP/HybridEP 不在本路径 |
| F14 | grouped FFN 边界 | `bf16_grouped_mm_out`、`slot_count` | 已实现（代码） | 主算子返回固定 grouped buffer 与 device counts；FFN 后以 opaque handle combine，不重跑 planner |
| F15 | registered dynamic weight materialize | `register_expert_pools`、`probeep_weight_transport.cu` | 已实现（代码） | 从注册 shard shape/dtype 推导完整 expert bytes；同 server IPC copy、跨 server IBGDA chunk；cache 同时校验 layout 与 optimizer weight version，`-1` 默认强制刷新；forward Weight 与 backward Grad 使用独立 expected-signal counter |
| F16 | 真跨机 expert-weight RDMA | `launch_probeep_weight_send/receive` | 目标机待验收 | source rail Weight TX 完成后立即进入该 rank Dispatch；RX/materialize 在私有 progress stream，Expert consumer 才 join；每个 ring slot 的 full-duplex TX/RX 使用两个不重叠的 symmetric staging bank |
| F17 | backward plan replay 与 remote grad reduce | `launch_probeep_grad_transport` | 目标机待验收 | 同一 handle replay；destination 先合并 FP32 replica grad，再按反向 rail 回 owner 并清空 replica |
| F18 | telemetry/result schema | `BalancedHandle` | 已实现（代码） | 暴露 compute intents、server before/after、Dispatch/Weight endpoint、pair load、budget/cap、chunks、invariant counters 和四段 planner device cycles |
| F19 | 分层自包含 HTML ZIP 与独立 nsys 目录 | Test 01/02 各自的 `scripts/` | 已实现（产物工具） | 根页按算法进入 Layer 页面；严格 timeline/DAG 与两个 microbatch 独立区块；nsys 不进入正式时延目录 |
| F20 | 无 host 热路径 | `Buffer::balanced_dispatch` + persistent workspace | 已实现（代码） | 无 Python planner、CPU polling、iteration allocation、per-chunk launch 或 plan `.item()`；compute/local packing 只做固定趟 device radix |
| F21 | 实测 observation producer 与跨层 A/M device ring | Test 01/02 私有 `formal_entrypoint.py`、`formal/run_benchmark.py` + ProbeEP runtime | 已实现（benchmark 代码），目标机待验收 | 固定 `A0 → (A1 ∥ W+D0) → (E0 ∥ W+D1) → E1`；A/M 分别使用 `A1.start→W+D0.done` 与 `E0.start→W+D1.done`；按 `(phase,round,compute_kind)` 分键，Layer L+1 只消费 Layer L 同 round、同类 feedback；layer>0 的 untimed correctness forward 也消费上一层 A/M，不再只验 bootstrap |
| F23 | 专家身份可辨 correctness | Test 01/02 `formal/run_benchmark.py` | 已实现（benchmark 代码），目标机待验收 | grouped gate/up 保持同 shape；down 的 16 个输出列以精确 BF16 `±128` 编码 global expert ID，单次公共 FFN oracle 即可识别错误 expert/replica/owner mapping |
| F22 | raw_data1 selector、精确 TopK materializer 与 benchmark case plan | Test 01/02 各自的 `scripts/` | 已实现（CPU 工具） | Test 01=前 20 层 debug；Test 02=完整 58 层论文输入；确定性缩放、精确 expert histogram、token 内 TopK 唯一 |

## 6. 算子优化红线

| 红线 | 处理 |
|---|---|
| timed path Python 扫 routes | 直接修 |
| controller+planner > 2% E2E | 合并 kernel / 降低同步 |
| per chunk host launch | 合并为批量 CUDA kernel |
| world weight barrier | 拆成 per-path/per-server event |
| weight 与 dispatch 串行 | 调整 event chain，前移 materialize |
| tail/max 冷启动污染 | warmup/cache lifecycle 修正 |

当前 `sm_90` cubin 静态资源门禁：

| kernel | registers/thread | stack/thread | shared/block | 结论 |
|---|---:|---:|---:|---|
| histogram serial/segmented | 28/30 | 0 B | 10,240 B | 无 local-memory spill |
| controller | 40 | 0 B | 0 B | metadata kernel |
| intent planner | 56 | 0 B | 13,021 B | initial Dispatch 按 expert 并行；固定位宽 radix；无 local-memory spill |
| admission | 64 | 0 B | 8,212 B | 稳定 intent 回访；Dispatch 区间线性求交；8 rail warp 特化；无 local-memory spill |
| server-local packing | 38 | 64 B | 12,352 B | 每 server 独立 block；固定趟 radix 替代 `E²` scan |
| plan finalization | 32 | 0 B | 0 B | remote slot 单次 expert-ID scan；prefix/invariant 并行 |
| Weight sender/receiver | 40/32 | 384/0 B | 0 B | 一个批量 grid 处理全部 chunks；sender stack 来自 NVSHMEM device transport 路径，目标机需用 nsys 检查代价 |
| gradient return | 56 | 384 B | 0 B | 一个批量 grid 处理全部 subchunks；目标机需用 nsys 检查 NVSHMEM device transport 的 stack/local-memory 代价 |

这些数值只证明编译资源形态，没有替代 nsys 延迟、occupancy、RDMA goodput 和端到端 overlap。每次修改 planner/transport 后必须重查 cubin；目标机上若 planner 超过 E2E 的 2%，优先减少 kernel launch、global metadata pass 和同步，再考虑改变 selection 数据结构。

单 GPU `sm_90` 诊断测量固定为 EP16、E256、TopK8、4096 tokens/rank、64 MiB/endpoint；30 次 CUDA-event 测量的可复现入口为：

```bash
python tests/bench_probeep_plan.py \
  --world 16 --tokens-per-rank 4096 --warmup 10 --iterations 30
```

| 范围 | 时间 | 计时口径 |
|---|---:|---|
| compute intent | 0.202 ms | `plan_counts[9]` cycles / advertised SM clock |
| network admission + chunk packing | 0.575 ms | `plan_counts[10]`；包含稳定 intent 回访、完整 expert 试排和 8-rail water-filling |
| server-local packing | 0.151 ms | `plan_counts[11]`；明确计入，不当作零开销 |
| finalization | 0.105 ms | `plan_counts[12]`；slot/prefix/invariant materialization |
| compute + packing + finalization | 0.458 ms | 不含 ProbeEP 特有的 network admission |
| 四阶段合计 | 1.033 ms | intent + admission + packing + finalization |
| 完整 diagnostic operator | p50 3.521 ms，p95 3.530 ms | 还包含单 GPU 串行模拟全 rank histogram、全 rank route lowering 与诊断输出；不是生产分布式路径 |

结论只限本机算子诊断：已经同时计时 packing，不能用 0.129 ms 的 intent 单段数字代表 planner。正式 H20 多机结果必须计入 compact histogram exchange、controller、四阶段 planner、Weight/Dispatch 和 overlap critical path。

## 7. 功能测试表

本轮没有 H20 多机 RDMA 运行证据。DeepEP RDMA、weight/gradient transport、A/M benchmark 闭环与性能实验必须在目标机重新执行；静态检查、计划预览和单 GPU planner 结果都不能替代真机 PASS。

| ID | 测试功能 | 测试入口 | 当前状态 | PASS 条件 |
|---|---|---|---|---|
| T01 | `alpha=0.90` controller 公式 | `src/deepep-probeep/tests/test_probeep_plan.py` | 单 GPU CUDA PASS | bottleneck sample、理论 cap、Dispatch baseline 扣减正确；有时间戳但零通信字节时保留/fallback，不将 A/M 预算错误缩到 0；runtime invalid hold 同时保留 budget 与 learned total |
| T02 | server-first 与两阶段 padded balance | 同上 `test_two_stage_*`、4096tpr case | 单 GPU CUDA PASS | server padded max/spread 优先下降；并列时平方和单调下降；route 守恒且 local packing 无 overflow |
| T03 | 完整 expert 原子 admission | 同上 `test_compute_plan_*` | 单 GPU CUDA PASS | 每个 `(expert,dst_server)` 恰有完整 weight bytes/chunk ids，或零 chunk |
| T04 | A/M 独立状态 | `test_balanced_internode.py` feedback rounds | 目标机待执行 | 只更新传入 `compute_kind`；另一行 bitwise 不变；Combine 不更新 |
| T05 | 计算规划与网络状态解耦 | `test_compute_plan_is_network_independent_*` | 单 GPU CUDA PASS | 零预算/宽预算的 `compute_intents` 相同，admitted/chunks 不同 |
| T06 | directed server-pair rail 水位 | two/four-server pair table tests | 单 GPU CUDA PASS | chunk table 聚合等于 `pair_load`；各 pair 单独 water-fill，endpoint 只做 tie-break |
| T07 | admission 后动态 Dispatch 重验 | endpoint cap assertions | 单 GPU CUDA PASS | 最终 `Weight+Dispatch TX/RX <= endpoint_total_cap`；每次 commit 使用新 footprint |
| T08 | compact histogram exchange | `test_balanced_internode.py` | 目标机待执行 | 分布式 `[server,expert]` counts 与 source Gate 一致；不传 expanded TopK |
| T09 | planner invariants | exact-balanced + EP16/32/64/128 + EP16 4096tpr + 32-seed unique-TopK fuzz | 单 GPU CUDA PASS | balanced Gate 不产生迁移；convergence=1、negative=0、conservation mismatch=0；19 项 test 全部通过 |
| T09b | server-local packing 与 lowering | `test_second_stage_packing_and_route_lowering_are_exact` | 单 GPU CUDA PASS | 每 server rank padded spread≤1 block；placement 确定；expert/slot/row、token layout、rank/server/slot counts 全部逐项重算一致；四段 planner cycles 均被记录 |
| T09c | planner + packing 完整开销 | `tests/bench_probeep_plan.py` | 单 GPU CUDA PASS | CUDA event 包含 histogram/intent/admission/packing/finalization/lowering；JSON 同时输出四阶段 cycles，不允许漏计 packing |
| T09d | 多 server final-intent 回访 | `test_multiserver_admission_revisits_temporarily_blocked_intents` + EP32/EP64 160-case 审计 | 单 GPU CUDA PASS | 四/八 server compute-final mapping 不因固定回放顺序误 defer；修复后 `actual_worse_than_compute=0/160` |
| T09e | RawData1 真实分布 planner | `raw_data1_all` 的 full/MB0/MB1 CUDA audit | 单 GPU CUDA PASS | 58 层×3=174 case 全部守恒、收敛、无负数/overflow、端点不过 cap；再以 0/1/4/8/16/32/64 MiB 预算扫描 1,218 case，compute intents 与网络预算无关，defer/rollback 后仍守恒 |
| T09f | 多服务器并列极值 | `test_multiserver_tied_hot_and_cold_servers_do_not_stall` | 单 GPU CUDA PASS | 两台同热、六台同冷时不得生成零 intent；平方和 tie-break 启动迁移并降低 max/spread |
| T09g | repeat 边界 bootstrap | Test01/Test02 source gate + 多 repeat 目标机 runner | 代码已实现；目标机待执行 | 每个 repeat 的 Layer 0 清空两行 controller summary、恢复 32 MiB bootstrap，不继承上一 repeat |
| T10 | raw_data1 E256/TopK8 workload | Test 01/02 私有 `scripts/test_raw_data1.py` | CPU contract PASS | 58 层总量守恒、256 logical experts、精确 histogram 与 token 内 TopK 唯一；真机仍须验证五算法 routing SHA 相同 |
| T11 | plan workspace 与双 microbatch ring | `test_balanced_internode.py` multi-round | 目标机待执行 | 三 slot overlap、generation/wrap、forward/backward handle 生命周期不串状态 |
| T12 | 单一 ProbeEP 主算子 | nsys + source gate | 目标机待执行 | 一次 `balanced_dispatch` 内完成 controller/plan/Weight/Dispatch，无 host round-trip |
| T13 | DeepEP token Dispatch/Combine | `test_balanced_internode.py` | 目标机待执行 | physical route/slot、FP8 dispatch、BF16 weighted combine 正确 |
| T14 | grouped FFN | grouped DSV3 case + expert fingerprint oracle | 目标机待执行 | valid prefix、padding、route weight 与 gate/up/down grouped-MM 正确；错误 expert/replica/owner mapping 必须产生 mismatch，不能用同权重专家掩盖 |
| T15 | registered local/remote weight | `test_balanced_expert_io.py` | 目标机待执行 | 动态 shard size、layout+weight-version cache、optimizer 后失效、checksum、consumer-ready 正确 |
| T16 | 真跨机 DeepEP RDMA baseline | upstream `test_internode.py` | 目标机待执行 | official normal HT RDMA forward/combine 通过 |
| T17 | 真跨机 ProbeEP weight RDMA | `test_balanced_expert_io.py` | 目标机待执行 | 完整 chunks 跨机到达、source-rail 依赖正确、无 Weight completion world barrier；Weight/Grad completion expected counter 独立；同一 rank 同时 TX/RX 时 staging 地址区间不相交且 checksum 正确 |
| T18 | 真跨机 ProbeEP backward/grad | `test_balanced_backward.py` + expert I/O | 目标机待执行 | plan replay、FP32 grad 回 owner、replica clear、ring release 正确；反向 TX/RX staging 同样不别名 |
| T19 | 多机五算法 raw_data1 性能 | Test 01/02 `formal-performance` | 目标机待执行 | 五 backend runner、同 route SHA、dual-microbatch HT、CSV/JSON/ZIP 已接线；目标机完成全 case plan 后才 PASS |
| T20 | 多机 nsys overlap | `03_Nsight-Systems使用.md` | 目标机待执行 | 保存 `.nsys-rep`；确认 operator 内无 host sync/Weight world barrier/新增关键路径瓶颈，窗口末端计时 event 同步单独识别 |
| T21 | 结果 schema、目录与分层 HTML ZIP | Test 01/02 私有 `scripts/validate_run_layout.py` | CPU 合成链路通过 | 单个 Test run、逐算法/逐 Layer 页面、成员完全镜像与 `testzip()` 均通过 |
| T22 | benchmark 跨层 A/M observation 闭环 | Test 01/02 dual-microbatch adapter + formal-pipeline nsys | 目标机待执行 | Layer 0 只用明确 bootstrap；Layer L/MB0 只消费 Layer L−1 Attention，Layer L/MB1 只消费 Layer L−1 MoE；producer/consumer layer、round、microbatch 全部 fail-closed |
| T23 | cold-layer Weight 与 4×400G/8×200G rail | Test 01/02 raw schema v4 + preflight | 目标机待执行 | 两个 microbatch 都使用正 weight version；有 admission 时两者均有真实 Weight chunks；`physical_nic=rail//2`、`subrail=rail%2`、每 rail 200 Gbps，两个 subrail 合计不超过对应 400-Gbps 物理口 |

当前门禁：目标机测试不得用本机编译、静态代码检查、历史日志或同机 IPC/NVLink 代替。任一 `plan_counts[2|6|7] != 0` 或 `plan_counts[8] != 1` 必须 fail-fast；未注册完整 expert weight/grad pools 时，只允许 planner/identity transport 诊断，不得标记为正式 grouped FFN。

## 8. 没提升时检查顺序

| 顺序 | 看什么 |
|---:|---|
| 1 | raw_data1 当前 layer 是否真的 server 倾斜 |
| 2 | ProbeEP server padded load 是否下降 |
| 3 | server-local rank padded load 是否下降 |
| 4 | admitted/deferred 是否按新 placement 重算 Dispatch baseline 并通过双端 budget |
| 5 | 每个有向 `(src_server,dst_server)` 内的 rail bytes 是否均衡；不能只看 NIC 总量 |
| 6 | weight transport 是否串行 |
| 7 | controller/planner 是否成瓶颈 |
| 8 | baseline 是否漏计 pack/unpack/weight sync |
