# Test 01：RawData1 Eval20 H20 多机调试测试

## 1. 测试身份与边界

本目录是一个完整、独立、可单独搬运的多机测试。唯一入口是 `bash/run.sh`；一次入口
只生成一个 Test 01 顶层日志。环境准备、正确性、动态探测 consumer、五算法 benchmark、
Nsight Systems 和可视化都是该日志的内部阶段，不拆成其他 Test 或平级 run。

| 项 | 固定值 |
|---|---|
| 测试目的 | 快速验证完整多机执行链、接口和性能趋势 |
| workload selector | `raw_data1_eval20` |
| MoE layer | 0–19，共 20 层 |
| 论文资格 | 否；不得作为完整 DSV3 主结果 |
| 默认 warmup | 每层 10 次 |
| 默认 measure | 每层 10 次 |
| 默认 repeat | 1 次 |
| 日志 | 一个 `test_logs/run_*_test01_rawdata1_eval20_*r/` |

本目录不调用其他 Test、其他实验目录或 `tests/` 下的共享执行代码。所需控制面工具均在
本目录 `scripts/` 内。项目级 `src/` 和只读 `workload/raw_data1/` 是被测源码与数据，
不是共享测试实现。

## 2. 目录与文件职责

| 路径 | 唯一职责 |
|---|---|
| `README.md` | 本测试的完整设计、执行和验收合同 |
| `benchmark.py` | 固定 eval20 selector 和 0–19 层，生成不可变 case plan |
| `bash/run.sh` | 唯一高层入口，串联本测试全部阶段 |
| `scripts/configs/h20_multinode.env.example` | 本测试独立的 H20/Slurm/RDMA 配置模板 |
| `scripts/create_venv.sh` | 在本测试目录创建独立 `.venv-h20` |
| `scripts/run_helpers.sh` | 本测试内部阶段编排；只调用本目录脚本 |
| `scripts/launch_slurm.sh` | 创建不可覆盖的阶段目录并申请 Slurm allocation |
| `scripts/slurm_job.sh` | 每节点一个 Slurm task，解析 host list |
| `scripts/launch_node.sh` | 激活本测试 venv，设置 node rank 并进入 runner |
| `scripts/runner.py` | build/import/smoke/formal/nsys 的控制面命令 |
| `scripts/preflight.py` | H20、SM90、rank、NCCL、IB、NVSHMEM 和源码锁验收 |
| `scripts/raw_data1.py` | 校验只读 trace、按本次 world size 精确实现 TopK routes |
| `scripts/benchmark_plan.py` | 固化五算法、层、route SHA、拓扑和计时参数 |
| `scripts/source_lock.json` | 锁定 UltraEP 所用 HybridEP vendored Git tree |
| `scripts/formal_entrypoint.py` | 校验不可变 case plan；每个 case 用独立多机进程加载唯一 backend |
| `scripts/formal/` | 本测试私有的五 backend、双 microbatch HT、grouped FFN、oracle 与 schema |
| `scripts/analyze_nsys_overlap.py` | 从 formal-pipeline SQLite 生成每节点 overlap JSON/TXT |
| `scripts/reprocess_layers_by_algorithm.py` | 从正式 raw 数据生成“算法入口→20 个独立 layer”的离线 HTML/CSV/JSON ZIP；只在同层十轮内统计 |
| `scripts/validate_run_layout.py` | 验收本测试唯一日志的目录和产物完整性 |
| `scripts/test_*.py` | 不占 GPU 的本测试静态合同测试 |

控制面可以使用 Python；禁止的是 ProbeEP timed operator 内出现 Python route scan、host
polling、`.item()` 同步、iteration 动态分配或逐 chunk host launch。算法热路径必须位于
`src/deepep-probeep` 的 CUDA/C++ 实现。

## 3. 被测源码与五算法

| 方法 | 唯一源码 | 均衡范围 | 本测试要求 |
|---|---|---|---|
| NCCL | benchmark 内 collective adapter | 不做专家均衡 | 与其他方法使用同一 routes/FFN/计时边界 |
| DeepEP | `src/deepep` | 不做专家均衡 | official normal HT internode 数据面 |
| DeepEP-MoonEP | `src/deepep-moonep` | server 内均衡 | 不允许混入 ProbeEP 跨 server 策略 |
| UltraEP + HybridEP | `src/ultraep` | server 内均衡 | 必须使用锁定的 HybridEP tree |
| ProbeEP | `src/deepep-probeep` | 跨 server 后 server 内二次均衡 | ours；planner/packing/weight/dispatch 全计时 |

UltraEP 数据面固定使用 HybridEP upstream commit
`e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8`。HybridEP 没有独立 `.git`，preflight
必须用 `scripts/source_lock.json` 核对父仓库中的 vendored tree。

三个 DeepEP 派生目录提供同名 Python package。构建可以依次 `build_ext --inplace`，运行
时必须为每个 backend 子进程设置独立 `PYTHONPATH`；不能把同名 wheel 混装后依赖 import
顺序。

## 4. 硬件、拓扑和模型合同

| 项 | 当前主测 | 规则 |
|---|---:|---|
| GPU | H20，SM90 | 每张卡必须是 compute capability 9.0 |
| servers | 2 | 当前真机 correctness oracle 固定 2 台 |
| GPUs/server | 8 | 固定 NVL8 |
| world size | 16 | node-major global rank |
| routed experts | 256 | 完整 DSV3 固定 E256 |
| experts/rank | 16 | `256/world_size` |
| tokens/rank | 4096 | 正式 eval20 测量固定 |
| TopK | 8 | 每 rank 32,768 expert-route rows |
| routes/layer | 524,288 | `16×4096×8` |
| hidden | 7168 | 五算法一致 |
| execution | dual microbatch HT | sync 模式只用于 debug |
| inter-server | 真 RDMA/NVSHMEM/IBGDA | 不用 NVLink 限速模拟 |
| physical NIC/server | 4 × 400 Gbps | 每个物理口服务两个 logical rail |
| logical rails/server | 8 × 200 Gbps | Rail 0/1→NIC0，2/3→NIC1，4/5→NIC2，6/7→NIC3 |
| rank mapping | node-major | 每台服务器连续 8 个 global rank |
| free memory | 每卡至少 48 GiB | preflight fail-closed；容纳 ProbeEP persistent pool 与 grouped FFN scratch |

算法接口允许未来扩到更多 H20 服务器，但必须满足 `world_size=8P`、
`256 % world_size == 0`，并先参数化当前 EP16 source oracle。只修改 `NNODES` 不能视为
扩容验收通过。

## 5. RawData1 Eval20 合同

`workload/raw_data/` 不修改；`workload/raw_data1/` 保存完整 58 层、256 routed experts 的
比例。本测试只选择 Layer 0–19，并按当前 world size 缩放每层总量：

```text
routes_per_layer = world_size × 4096 × 8
```

`scripts/raw_data1.py` 使用 largest-remainder 保持每层总量，随后实现精确、token 内 expert
不重复的 `[world_size,4096,8] int16` route tensor。每层保存 SHA-256；五算法必须读取同一
文件，不能各自采样或重新生成 routing。

| 必查项 | PASS |
|---|---|
| selector | 恰为 `raw_data1_eval20` |
| selected layers | 恰为 `[0,1,...,19]` |
| experts | 256 |
| tensor shape | `[world_size,4096,8]` |
| histogram | 与 materialized manifest 中 `expert_counts` 完全一致 |
| route total | 每层等于 `world_size×4096×8` |
| algorithm consistency | 每个算法记录相同 routing SHA |
| case count | 默认 `20 layers × 5 methods × 1 repeat = 100` |

这是调试 workload。它可以验证趋势、内存、闭环和五算法接口，但不代表完整 58 层 DSV3
均值。

## 6. 双 microbatch HT 执行模型

训练/prefill benchmark 固定两个 microbatch、一个 communication stream 和一个 compute
stream。4096 tokens/rank 沿 token 维切成 MB0/MB1，各为 2048 tokens/rank。目标流水线：

```text
wavefront:       A0 → (A1 || W+D0) → (E0 || W+D1) → E1
compute/default: A/G[MB0] ─ A/G[MB1] ─ E[MB0] ─ E[MB1] ─ wait(C1) ─ feedback_prepare(A[L,r],M[L,r])
communication:              P/W/D[MB0] ─ P/W/D[MB1] ─ C[MB0] ─ C[MB1]
cross layer:     A[L-1,r] → Layer L/MB0；M[L-1,r] → Layer L/MB1；A/M 状态互不覆盖
```

同 stream 的箭头是提交顺序。跨 stream 必须显式等待：`A0→D0`、`D0→E0`、`A1→D1`、
`D1→E1`、`E0→C0`、`E1→C1`。ProbeEP 的 `P/W/D0` 与 `A1` overlap，`P/W/D1` 与 `E0` overlap；observation producer 由 caller 在
default compute stream 上、当前 iteration 完成后执行，不能误标成 communication-stream
阶段。物理 overlap 跨 microbatch，但反馈链不交叉：`MB1 Attention || MB0 W+D` 只产生 A 链样本，`MB0 Expert || MB1 W+D` 只产生 M 链样本。

DeepEP-MoonEP 与 UltraEP + HybridEP 的 replica-weight bank 尚未双缓冲时，`W+D[1]` 必须等待
`E[0]` 释放 bank；报告 DAG 单独表达该依赖。每个 Layer 先展示逐 Round/逐 rank、带严格
`start_ms/end_ms` 的双 stream 时间线和事件依赖 DAG，再展示两个完全独立的 microbatch
区块。每个区块各自包含原始/调度后 rank 负载、directed server-pair 和 per-rail/NIC
负载；不画 MB0+MB1 合计柱图。

两个 stream 共用同一个 `e2e_start` CUDA event 作为时间原点；跨 stream 先建立 device-side
wait，再记录各阶段起止 event。禁止把独立 duration 累加成伪时间线。

| 组件 | 计时要求 |
|---|---|
| router/materialize | 五算法相同；不得隐藏 ProbeEP 特有准备 |
| controller | 计入 ProbeEP critical path |
| histogram/plan/admission/packing | 全部计入，不能只报 intent kernel |
| weight transport | 论文主实验固定 `PROBEEP_WEIGHT_CACHE_MODE=cold`，每 invocation 新 weight version；steady cache hit 只能作为补充；不能用 world completion barrier |
| dispatch/combine | 使用 DeepEP normal HT 的真实路径 |
| grouped FFN | 五算法同 shape、dtype 和有效 rows |
| backward/grad | 使用 forward handle replay，FP32 回 owner |
| overlap wait | 计入端到端 max-rank 时延 |

正式指标使用 world-rank critical path，报告 p50/p95/p99/max；不得只报 rank 平均或单个
transport kernel。

## 7. Attention/MoE 动态探测边界

ProbeEP controller 维护 Attention 与 MoE 两行独立状态，目标比例为 `alpha=0.90`。计算
规划只处理 token/padded compute；网络 admission 再根据真实 endpoint observation 决定
完整 expert 是否迁移，并在有向 `(src_server,dst_server)` 内把 weight chunks 优先放到
目的 server 当前水位更低的 rail。

| 项 | 当前能力 | 本测试要求 |
|---|---|---|
| A/M consumer | CUDA controller 已实现 | 合成输入验证两行隔离、invalid hold 和预算方向 |
| compute planner | 已实现 | 不读取 NIC budget/rail 水位 |
| network admission | 已实现 | 完整 expert 原子接纳；稳定回访暂不可改善的 final intent；TX/RX endpoint 均不过载 |
| server 内 packing | 已实现 | 跨 server placement 固定后再做 rank 二次均衡 |
| 实测 observation producer | 已接入 benchmark，目标机待验收 | A/M 各自使用已完成 CUDA-event 窗口与对应 W+D/Weight bytes；Layer L+1 同 round、同类 dispatch 融合消费 |

正式 benchmark scheduler 为两个 microbatch 分别记录 Attention/MoE compute 与对应 W+D
CUDA event，读取已完成 handle 的实际 destination rows 和 cache-miss Weight chunks，按
`(phase,round,compute_kind)` 保存两条互不替代的 device feedback。Layer L/MB0 只消费 Layer
L−1 的 Attention 链，Layer L/MB1 只消费 Layer L−1 的 MoE 链；Layer 0 使用明确 bootstrap。
若 `REPEATS>1`，每个 repeat 的 Layer 0 都会清空持久 A/M summary 并恢复 32 MiB bootstrap，
不得继承上一 repeat 最后一层的 controller 状态。
所需 device collectives 计入当前 ProbeEP 的 `plan_ms/e2e_ms`；Python adapter
不执行 controller、排序、admission 或 packing，这些步骤仍在下一次 `balanced_dispatch` 的融合
CUDA 路径内。layer>0 的专用 untimed correctness forward 也消费上一层 warmup round 0 的
A/M 双链，避免只验证 bootstrap。合成 tensor 只能进入 `correctness/observation-synthetic/`。

grouped correctness 不再让所有专家使用不可区分的权重：down projection 的前 16 个输出列
用精确 BF16 `±128` 编码 global expert ID。该指纹不改变 GEMM shape、kernel 数量或 timed
热路径，但会使错误 logical expert、replica slot 或 owner mapping 明确产生 mismatch。

## 8. 独立环境配置

每台目标服务器执行：

```bash
cd /absolute/path/to/ProbeEP_cuda
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
bash tests/Test_01_RawData1_Eval20/scripts/create_venv.sh

cp tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env.example \
   tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env
```

只修改本测试配置中的目标集群真实值：

| 类别 | 必填/核对 |
|---|---|
| Slurm | partition、node list、QoS、CPU、memory、time |
| bootstrap | `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` |
| RDMA | `NVSHMEM_HCA_LIST`、active port、`NVSHMEM_REMOTE_TRANSPORT=ibrc` |
| NCCL | `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`NCCL_IB_DISABLE!=1` |
| IBGDA | `NVSHMEM_IB_ENABLE_IBGDA=1`、`NVSHMEM_DISABLE_CUDA_VMM=1` |
| HybridEP | `HYBRID_EP_TRANSPORT=nixl|doca` 及对应依赖路径 |
| GPU | `PROBEEP_REQUIRE_GPU_NAME=H20`、`TORCH_CUDA_ARCH_LIST=9.0` |

HCA 和网卡名必须由目标机 `nvidia-smi topo -m`、`ibdev2netdev`、`ibv_devinfo` 获取，
禁止照抄其他集群。目录存在不代表 venv 可用；脚本必须重新核对 torch/CUDA/NVSHMEM、
8 张 H20 和 SM90。

## 9. 静态检查与命令预览

不占 GPU 的检查：

```bash
bash -n \
  tests/Test_01_RawData1_Eval20/bash/run.sh \
  tests/Test_01_RawData1_Eval20/scripts/create_venv.sh \
  tests/Test_01_RawData1_Eval20/scripts/launch_node.sh \
  tests/Test_01_RawData1_Eval20/scripts/slurm_job.sh \
  tests/Test_01_RawData1_Eval20/scripts/launch_slurm.sh

python3 -m pytest -q \
  tests/Test_01_RawData1_Eval20/scripts/test_benchmark_plan.py \
  tests/Test_01_RawData1_Eval20/scripts/test_raw_data1.py \
  tests/Test_01_RawData1_Eval20/scripts/test_execution_system.py

PLAN_ONLY=1 bash tests/Test_01_RawData1_Eval20/bash/run.sh
```

`PLAN_ONLY=1` 只打印所有阶段命令，不创建 route、不申请节点、不产生性能证据。

## 10. 唯一正式入口与阶段顺序

从仓库根目录执行。显式固定 `TEST_RUN_ID/TEST_RUN_DIR`，这样运行结束后无需猜测哪个目录
属于本次测试；目录必须尚不存在：

```bash
cd /absolute/path/to/ProbeEP_cuda

export TEST_RUN_ID="run_$(date -u +%Y%m%d_%H%M%S_%6N)_test01_rawdata1_eval20_16r"
export TEST_RUN_DIR="$(pwd)/test_logs/${TEST_RUN_ID}"
test ! -e "${TEST_RUN_DIR}"

ENV_FILE=tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env \
  bash tests/Test_01_RawData1_Eval20/bash/run.sh
```

入口第一行会打印 `[test-run] ${TEST_RUN_DIR}`。只有命令退出码为 0，且最后的
`validate-test-run` 为 PASS，才能进入归档步骤。

| 顺序 | 日志阶段 | PASS 条件 |
|---:|---|---|
| 0 | `workload/` | 20 层 manifest、routes 和 case plan 固定 |
| 1 | `setup/preflight/` | H20/SM90/rank/NCCL/IB/NVSHMEM/source 全 PASS |
| 2 | `setup/build/` | 五套源码在 SM90 环境构建成功 |
| 3 | `setup/import/` | 每个 backend 从指定源码树加载 |
| 4 | `correctness/deepep/` | official DeepEP internode smoke PASS |
| 5 | `correctness/deepep-moonep/` | server-local balance 路径 PASS |
| 6 | `correctness/probeep/` | forward、expert-I/O、backward PASS |
| 7 | `correctness/observation-synthetic/` | A/M consumer 合同 PASS；不冒充真实闭环 |
| 8 | `correctness/ultraep-hybridep/` | 固定 HybridEP 数据面 PASS |
| 9 | `benchmark/` | 五算法、20 层、同 route、dual-microbatch HT 完成 |
| 10 | `nsys/` | 七个 target 的 rep/sqlite/stats 完成；formal-pipeline 固定 Layer 00、五算法各 1 case，并有逐节点 overlap 摘要 |
| 11 | `artifacts/` | 五算法逐层 HTML/CSV/JSON ZIP 和目录验收 PASS |
| 12 | `${TEST_RUN_DIR}.tar.zst` | 入口退出后手工归档整个顶层 Test log，zstd/tar/sha256 均通过 |

任一阶段失败立即停止；不得跳过失败阶段后拼接其他 run 的产物。

## 11. Slurm 与手工多机模型

Slurm 模型固定为：

```text
sbatch: P nodes
  -> srun: 1 task/node
       -> node task: 8 local GPU workers
```

DeepEP 派生 source test 的外层 `WORLD_SIZE/RANK` 是 server 数/server rank；内部 communicator
才是 global GPU ranks。UltraEP 使用 torchrun global rank，`scripts/runner.py` 分开处理。

没有 Slurm 时，两个节点必须使用相同 `PROBEEP_RUN_ID`、`PROBEEP_RUN_DIR`、master address
和端口，并发调用本目录 `scripts/launch_node.sh`。任一节点失败后整轮作废，使用新 run id
重跑；不能复用半成品阶段目录。

## 12. 唯一 Test log 合同

```text
test_logs/run_<UTC>_test01_rawdata1_eval20_<world>r/
├── workload/
│   ├── manifest.json
│   ├── benchmark_plan.json
│   └── layer_00_topk_idx.npy ... layer_19_topk_idx.npy
├── setup/
│   ├── preflight/
│   ├── build/
│   └── import/
├── correctness/
│   ├── deepep/
│   ├── deepep-moonep/
│   ├── probeep/
│   ├── observation-synthetic/
│   └── ultraep-hybridep/
├── benchmark/
│   ├── launch.env
│   ├── readiness.json
│   ├── logs/
│   └── raw/
├── nsys/
│   ├── deepep-smoke/
│   ├── deepep-moonep-smoke/
│   ├── probeep-forward/
│   ├── probeep-expert-io/
│   ├── probeep-backward/
│   ├── ultraep-smoke/
│   └── formal-pipeline/
└── artifacts/
    ├── raw_data1_layers_00_19_by_algorithm_rounds_11_20_mean.zip
    └── raw_data1_layers_00_19_by_algorithm_rounds_11_20_mean/
        ├── algorithm_comparison.html
        ├── README.md
        ├── manifest.json
        └── algorithms/
            ├── nccl/
            ├── deepep/
            ├── deepep_moonep/
            ├── ultraep_hybridep/
            └── probeep/
```

一个 Test 入口对应一个顶层 log。correctness、benchmark、nsys 和 artifacts 不能变成
平级 `run_*`，也不能从其他执行复制进来。

## 13. Benchmark 原始数据 schema

下一次 H20 run 使用 raw schema v4；v1/v2/v3 历史数据只允许离线查看，不能通过 fresh Test gate。

| 文件 | 必须记录 |
|---|---|
| `raw/benchmark_status.jsonl` | method/layer/repeat 的最终 PASS/FAIL/BACKEND_UNAVAILABLE；STARTED/PASS/FAIL 调度状态另见逐节点 schedule JSONL |
| `raw/correctness.jsonl` | timed 前的数值正确性 |
| `raw/iterations.csv` | iteration、layer、method、rank、critical-path latency |
| `raw/rank_samples.csv` | home/raw/padding/execute load |
| `raw/expert_samples.csv` | 256 experts 的 receive rows |
| `raw/rank_expert_samples.csv` | `(rank,expert)` grouped-FFN rows |
| `raw/microbatch_rank_samples.csv` | 每轮 MB0/MB1 的逐 rank home/execution/padding rows |
| `raw/microbatch_timeline.csv` | 每轮、每 rank、每 logical stream 的阶段 CUDA-event 起止时间；W+D 使用真实 communication-stream active start/done，A/M observation 另用 release-to-done 窗口；ProbeEP 额外含 post-combine observation producer，不伪造独立 controller interval |
| `raw/rdma_path_load.csv` | 五算法、两 microbatch、每轮全部 directed logical rail；保存 physical NIC/subrail、200-Gbps rail cap、weight version/cache mode、runtime Dispatch、cache-miss Weight 与总 TX/RX，零流量 rail也保留 |
| `raw/probeep_observation_samples.csv` | 当前 Layer/round 真正消费的上一层同类 A/M observation；记录 producer/consumer layer、repeat、round、dispatch/compute microbatch、逐 rank compute/network ns 与 Dispatch/Weight TX/RX |
| `raw/probeep_plan_summary.jsonl` | production CUDA plan 的 feedback source/bootstrap、server/rank load、budget、intent、admission、完整 chunk table、invariants 与四段 device clock cycles |
| `raw/probeep_weight_chunks.csv` | 每个 ProbeEP expert-weight chunk 的 expert、offset、bytes、source/destination server/rank、rail 与 path offset |
| `artifacts/.../manifest.json` | 只记录五算法与 20 个独立 Layer 页面，不保存跨层性能汇总 |
| `artifacts/.../algorithms/*/layers/data/layer_XX/` | 当前算法、当前 Layer 独占的 JSON/CSV；禁止混入其他 Layer |

每行带 run id、workload/layer identity、routing SHA、runner mode 与计时 scope；world size、
dtype、拓扑和源码 provenance 由同 run 的 immutable manifest/launch/preflight 记录，不能跨
run 拼接。

下一次正式 Test 01 的固定 raw 行数如下；少一行即使 benchmark status 为 PASS，也不能生成
可引用报告：

| 文件 | 固定行数 | 计算 |
|---|---:|---|
| `iterations.csv` | 1,000 | 100 cases × 10 measured rounds |
| `rank_samples.csv` | 16,000 | 100 × 10 × 16 ranks |
| `microbatch_rank_samples.csv` | 32,000 | 100 × 10 × 16 × 2 MB |
| `microbatch_timeline.csv` | 131,200 | 四 baseline × 20 layers × 10 × 16 × 8 stages + ProbeEP × 20 × 10 × 16 × 9 stages |
| `rdma_path_load.csv` | 32,000 | 100 × 10 × 2 MB × 16 directed rails；当前为 2 servers |
| `probeep_observation_samples.csv` | 6,080 | 19 consuming layers × 10 × A/M × 16 ranks；Layer 00 只有 bootstrap、没有伪 observation |
| `probeep_plan_summary.jsonl` | 400 | 20 ProbeEP layers × 10 × A/M plans |
| `probeep_weight_chunks.csv` | 动态 | production plan 的逐 expert/chunk/rail 明细；正式 cold run 有 admission 时必须 `transfer_required=1`，MB0/MB1 都不能被旧 cache 隐藏；steady 补充实验才允许 cache-hit actual bytes=0 |

`rdma_path_load.csv` 是由各 backend 已完成的 runtime routing/count tensor 生成的逻辑数据面
payload 账本，不是 mlx5 端口硬件计数。NCCL 以 route occurrence 计数；DeepEP 系列以
`num_tokens_per_rdma_rank` 的 node-to-node 去重 token 计数；UltraEP+HybridEP 以 reroute 后
destination-server membership 计数。采集发生在 timed CUDA-event 区间之后，不重新执行
routing，也不进入 E2E latency。硬件链路利用率只能由 nsys/IB 计数另行佐证。
Dispatch activation 已按 destination server 去重，一个 token 可能同时服务多个 expert，
因此不能伪造单一 expert 归属；只有语义明确的 Weight chunks 按 `expert_id` 独立分色。

## 14. Nsight Systems 验收

本入口自动 profile：DeepEP、DeepEP-MoonEP、ProbeEP forward、ProbeEP expert-I/O、ProbeEP
backward、UltraEP+HybridEP，以及 Layer 00 的五算法 formal pipeline。UltraEP 不再使用
`--trace=none`；全部 target 统一采集 `cuda,nvtx,osrt`。每个 target 独立
保存在本 Test log 的 `nsys/<target>/`；formal pipeline 还生成逐节点 overlap JSON/TXT。

| 检查项 | PASS |
|---|---|
| two-stream overlap | `W+D[k+1]` 与 `compute[k]` 有实际交叠 |
| host sync | operator/overlap 窗口无 `cudaDeviceSynchronize`、隐式 D2H、`.item()`；只允许 benchmark 在窗口末端同步计时 event |
| allocation | steady iteration 无 `cudaMalloc` |
| kernel count | controller/planner/packing 未拆成大量小 kernel |
| RDMA order | data-before-signal，consumer event 正确；每个 plan slot 的 full-duplex TX/RX symmetric staging 不重叠 |
| barrier | 无 weight completion world barrier |
| planner cost | controller+planner+packing 接近 MoonEP，目标不超过 E2E 2% |

必须保留 `.nsys-rep`、`.sqlite`、NVTX projection、GPU kernel、CUDA API 和 GPU memory stats。
nsys 时延只用于诊断，不进入 benchmark latency 汇总。

## 15. 可视化与离线 ZIP

`artifacts/raw_data1_layers_00_19_by_algorithm_rounds_11_20_mean.zip` 必须完整镜像同名目录，
且无 CDN/外链脚本。根页只提供五个算法入口；每个算法入口进入 20 层目录，Layer 00–19
各自拥有单独 HTML 页面。单层页面的时延、rank load 仅在该层 Round 11–20 内求 mean/min/max，
禁止先混合 layer 再求均值或 P99。页面至少展示：

| 图/表 | 内容 |
|---|---|
| strict timeline / DAG | 当前算法、当前 layer、当前 Round/rank 的 CUDA-event `start_ms/end_ms`；DAG 区分同 stream 顺序、跨 stream event wait、host 完成边界和跨轮 feedback，不用估算时长 |
| E2E latency | 当前算法、当前 layer 的 Round 11–20 明细与 mean/min/max |
| Microbatch 1 rank load | 只包含 runtime MB0 的原始与调度后 16-rank rows |
| Microbatch 2 rank load | 只包含 runtime MB1 的原始与调度后 16-rank rows |
| server load | 当前算法、当前 layer 的 server max/mean |
| per-microbatch rail load | MB0、MB1 各自的有向 server-pair 和 rail TX/RX bytes；每条 rail 严格堆叠 `[Token Dispatch bytes][Expert Weight bytes]`，两段不覆盖 |
| ProbeEP plan | budget、admitted/deferred experts、chunks、planner stage |
| observation | Attention/MoE 分行样本、validity 和 budget update |

没有实测 active time 时只报告 bytes，不能用标称带宽伪造 goodput。

fresh run 禁止出现“历史 raw 缺失”空面板。`validate_run_layout.py --kind test` 强制检查：

| 页面数据 | fresh run 门禁 |
|---|---|
| strict timeline | 五算法、20 层、10 rounds、16 ranks 均有真实 CUDA-event interval |
| MB rank load | 五算法每层 MB0/MB1 均有 home、execution、padding |
| rail/NIC | 五算法每层 MB0/MB1 都有 16 条 directed rail × 10 rounds，包含零流量 rail |
| ProbeEP A/M | Layer 00 明确 bootstrap；其余每个 measured round 都能追到上一层、同 round、同 compute kind producer，A/M 不得互相替代 |
| Expert placement / Weight | A/M 两类 plan 都必须出现实际 admission 与跨 server compute 改善；formal cold run 有 admission 时必须真实传 Weight；aggregate rail bytes 与逐 expert chunk bytes 守恒 |
| nsys | 七个 target 每节点均有 `.nsys-rep`、`.sqlite`；formal pipeline 含五算法 |

此外，报告生成前必须逐记录通过：`combined rank=MB0+MB1`、
`rail TX=Dispatch+Weight=RX`、`plan assigned bytes=实际 chunk bytes=rail Weight bytes`，以及
`Layer L observation=Layer L−1 同 Round/同 A/M chain 的 rail producer counters`。导出的
rail/chunk CSV 必须保留 Round、A/M、NIC/subrail、cache/version 字段；HTML 只消费同一
算法、同一 Layer、同一 microbatch 的记录。

## 16. 正确性和性能门禁

| 门禁 | 失败处理 |
|---|---|
| E256 route/weight/count 守恒 | 整个 run 作废 |
| 五算法 routing SHA 不一致 | 整个 benchmark 作废 |
| ProbeEP planner invariants 非零 | fail-fast |
| 完整 expert 被部分 admission | fail-fast |
| Weight+Dispatch 超 endpoint budget | fail-fast |
| 有向 pair rail bytes 不守恒 | fail-fast |
| grouped FFN 输入或 padding 不一致 | 作废 |
| backward 未复用原 plan/grad 未回 owner | 作废 |
| synthetic observation 冒充真实 producer | 作废 |
| nsys profile 时间混入正式 latency | 作废 |
| planner/packing 未计时 | 作废 |
| 缺任一算法、时间线、microbatch、rail、A/M、chunk 或 raw schema | 顶层目录验收失败 |

`bash/run.sh` 会自动锁定并导出以下四项；任一项缺失或 SHA/拓扑不匹配时
`formal-performance` 必须主动停止：

```text
PROBEEP_DYNAMIC_OBSERVATION_MODE=benchmark_cuda_events
PROBEEP_FORMAL_ENTRYPOINT=<本 Test scripts/formal_entrypoint.py>
PROBEEP_WORKLOAD_MANIFEST=<本 Test workload/manifest.json>
PROBEEP_BENCHMARK_PLAN=<本 Test workload/benchmark_plan.json>
```

当前工作树已补齐下一次 run 的时间线、五算法 rail、A/M observation、ProbeEP chunk 与
五算法 nsys 采集合同；这些新增字段尚未在 H20 多机上重跑，不能回填到历史 raw，也不能
用静态计划替代真机结果。

## 17. 没提升时的诊断顺序

| 顺序 | 检查 |
|---:|---|
| 1 | 当前 layer 是否存在跨 server padded compute 倾斜 |
| 2 | ProbeEP server padded max/spread 是否实际下降 |
| 3 | server 内 rank padded max 是否二次下降 |
| 4 | admission 是否使用迁移后 Dispatch footprint 重算双端 budget |
| 5 | 每个 `(src,dst)` 的 chunks 是否优先走该目的 server 的低水位 rail |
| 6 | weight cache miss、version refresh、checksum 是否导致额外关键路径 |
| 7 | Weight 与 Dispatch/compute 是否错误串行或存在 world barrier |
| 8 | controller、planner、packing、lowering 是否成为新瓶颈 |
| 9 | baseline 是否漏计 pack/unpack、weight sync、grouped FFN 或 overlap wait |
| 10 | p95/p99 是否被 cold start、allocator 或个别 rank 卡住 |

先修正确性和完成语义，再调 CUDA kernel、warp specialization、persistent workspace、event
chain 和 SM 配额；不能为了得到提升修改 ProbeEP 算法核心。

## 18. 运行结束后的完整日志归档

### 18.1 得到的日志目录

正式入口退出后，本次测试的唯一完整结果就是前面固定的 `${TEST_RUN_DIR}`：

```text
test_logs/
└── run_<UTC>_test01_rawdata1_eval20_16r/   # ${TEST_RUN_DIR}
    ├── workload/
    ├── setup/
    ├── correctness/
    ├── benchmark/
    ├── nsys/
    └── artifacts/
```

`artifacts/*.zip` 只包含离线 HTML 报告，不等于完整实验日志。最终下载包必须归档整个
`${TEST_RUN_DIR}`，包括 workload、配置 provenance、原始 CSV/JSONL、节点日志、nsys 和
可视化产物；不能只打包 `artifacts/`，也不能从多个 run 拼接。

先重新执行顶层验收：

```bash
cd /absolute/path/to/ProbeEP_cuda
test -n "${TEST_RUN_DIR:-}"
test -d "${TEST_RUN_DIR}"

python3 tests/Test_01_RawData1_Eval20/scripts/validate_run_layout.py \
  "${TEST_RUN_DIR}" --kind test
```

验收失败时禁止压缩，应修复问题并使用新的 run id 完整重跑。

### 18.2 生成 `.tar.zst`

目标机必须有 GNU tar 和 `zstd`。缺少 `zstd` 时先用目标系统包管理器安装，例如
Debian/Ubuntu 为 `sudo apt-get install zstd`。压缩包必须放在 run 目录旁边，不能放进待
压缩的 run 目录内部：

```bash
LOG_DIR="$(realpath "${TEST_RUN_DIR}")"
LOG_PARENT="$(dirname "${LOG_DIR}")"
LOG_NAME="$(basename "${LOG_DIR}")"
ARCHIVE="${LOG_PARENT}/${LOG_NAME}.tar.zst"
CHECKSUM="${ARCHIVE}.sha256"

command -v tar
command -v zstd
test ! -e "${ARCHIVE}"
test ! -e "${CHECKSUM}"

tar -I 'zstd -T0 -10' -cf "${ARCHIVE}" \
  -C "${LOG_PARENT}" "${LOG_NAME}"
```

不要使用 `--exclude`：build/import 日志、raw、nsys 和内部 HTML ZIP 都属于完整证据。

### 18.3 压缩包完整性验收

归档后依次检查 zstd 数据、tar 成员、与原目录的一致性，并生成 SHA-256：

```bash
zstd -t "${ARCHIVE}"
tar --zstd -tf "${ARCHIVE}" >/dev/null
tar --zstd --compare --file "${ARCHIVE}" --directory "${LOG_PARENT}"

(
  cd "${LOG_PARENT}"
  sha256sum "${LOG_NAME}.tar.zst" > "${LOG_NAME}.tar.zst.sha256"
  sha256sum -c "${LOG_NAME}.tar.zst.sha256"
)

ls -lh "${ARCHIVE}" "${CHECKSUM}"
```

`tar --compare` 无输出且退出码为 0、`sha256sum` 显示 `OK` 才算归档完成。最终需要下载
两个相邻文件：

```text
test_logs/<TEST_RUN_ID>.tar.zst
test_logs/<TEST_RUN_ID>.tar.zst.sha256
```

下载到另一台机器后，在压缩包所在目录验证并解压：

```bash
sha256sum -c "${TEST_RUN_ID}.tar.zst.sha256"
tar --zstd -xf "${TEST_RUN_ID}.tar.zst"
```

解压后必须得到一个顶层 `${TEST_RUN_ID}/`，再用同版本仓库中的
`validate_run_layout.py <解压目录> --kind test` 复验。生成 checksum 后不要再修改原 log；
若确需修改，必须重新生成 tar 和 checksum。

## 19. 完成记录

| Gate | 状态 | run id / 证据 | 问题与修复 |
|---|---|---|---|
| 本测试静态合同 | PASS | local CPU static tests | 27/27 PASS；含 fresh raw-v4 schema/byte merge、logger、五算法 nsys 和 fail-closed raw 门禁 |
| H20/SM90/RDMA preflight |  |  |  |
| five-source build/import |  |  |  |
| DeepEP internode smoke |  |  |  |
| DeepEP-MoonEP smoke |  |  |  |
| ProbeEP forward |  |  |  |
| ProbeEP weight/grad I/O |  |  |  |
| ProbeEP backward |  |  |  |
| UltraEP + locked HybridEP |  |  |  |
| real A/M observation producer |  |  |  |
| eval20 five-algorithm benchmark |  |  |  |
| nsys overlap |  |  |  |
| visualization ZIP |  |  |  |
| 顶层 Test log 验收 |  |  |  |
| 完整 Test log `.tar.zst` + SHA-256 |  |  |  |
