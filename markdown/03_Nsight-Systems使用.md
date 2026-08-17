# Nsight Systems 使用

nsys 只用于诊断，不进入正式 latency summary。

## 1. 命令

```bash
cd /home/chen/workspace/infra/ProbeEP_cuda
ENV_FILE=tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env \
RUN_NSYS=1 bash tests/Test_01_RawData1_Eval20/bash/run.sh
```

该入口会在 Test 01 的同一个顶层 run 中依次保存下列 target：

```text
deepep-smoke
deepep-moonep-smoke
probeep-forward
probeep-expert-io
probeep-backward
ultraep-smoke
formal-pipeline
```

`formal-pipeline` 使用本 Test 内置入口，固定选择 RawData1 Layer 00，依次 profile NCCL、
DeepEP、DeepEP-MoonEP、UltraEP + HybridEP 和 ProbeEP。全部 target 使用
`cuda,nvtx,osrt`；不再接受 `trace=none` 空报告。历史 DOCA 环境曾在 profiler 注入时返回
QP status 21，因此目标机若复现该问题，整个 nsys gate 保持 FAIL，先适配 transport/驱动，
不能用空 trace 冒充 UltraEP overlap 证据。正式时延仍只来自未注入 profiler 的 benchmark。

## 2. 看什么

| 项 | PASS |
|---|---|
| 2-stream | `dual_microbatch_ht` 下 `W+D[k+1]` 与 compute[k] 有交叠 |
| host sync | operator/overlap 窗口中无 host wait；只允许 benchmark 在窗口末端同步计时 event |
| allocator | iteration 内无 `cudaMalloc` |
| kernel count | ProbeEP 不引入大量小 kernel |
| RDMA order | data-before-signal |
| barrier | 无 world-rank weight completion barrier |
| tail/max | 无明显 cold-start 污染 |

## 3. NVTX 范围

| Range | 含义 |
|---|---|
| `<variant>/measurement_iteration` | 一个正式双 microbatch iteration |
| `ubatch0/ht_dispatch` | ubatch0 Weight+Dispatch |
| `ubatch1/ht_dispatch` | ubatch1 Weight+Dispatch |
| `probeep/feedback_bind` | host 侧只绑定已完成的 A/M device observation；没有独立 controller kernel，真实 controller 融合在对应 `ubatch*/ht_dispatch` 内 |
| `probeep/feedback_prepare` | Layer L timed pipeline 后分别形成 A[L,r]/M[L,r] device feedback；Layer L+1 同 round 只消费同类链，其 GPU collective 时间已并入 Layer L 的 `plan_ms/e2e_ms` |
| `attention_or_gate/ubatch0` | Attention 独立 compute window |
| `ubatch0/expert_mlp`、`ubatch1/expert_mlp` | grouped FFN compute |
| `ubatch0/ht_combine`、`ubatch1/ht_combine` | HT combine |

ProbeEP 的 controller、histogram、server-first plan、admission、chunk packing、Weight 和
Dispatch 在 production CUDA 入口内融合；Python 侧不能为了细分计时再次执行这些阶段。
具体 kernel 以 CUDA kernel 表和 `src/deepep-probeep` 的 NVTX/CUDA 名称定位。

严格双流因果必须看到 `A0 → (A1 ∥ W+D0) → (E0 ∥ W+D1) → E1`。两个 W+D 的
active interval 分别取 communication-stream start/done；controller observation 的网络窗口
分别取 `A1.start→W+D0.done` 和 `E0.start→W+D1.done`，两种口径不能混写。

## 4. 产物

| 文件 | 用途 |
|---|---|
| `test_logs/run_*_test0X_*/nsys/<suite>/nsys/<suite>-node-<rank>.nsys-rep` | 逐节点原始 trace |
| `test_logs/run_*_test0X_*/nsys/<suite>/nsys/<suite>-node-<rank>.sqlite` | 逐节点 query 数据库 |
| `*-nvtx_gpu_proj_sum*` | NVTX GPU projection stats |
| `*-cuda_gpu_kern_sum*` | CUDA kernel stats |
| `*-cuda_api_sum*` | CUDA API stats |
| `*-cuda_gpu_mem_time_sum*` | GPU memory stats |
| `formal-performance-node-<rank>-overlap.json` | 五算法 network/Attention/FFN overlap 结构化摘要 |
| `formal-performance-node-<rank>-overlap.txt` | 同一摘要的可读表格 |

`formal-pipeline` 会自动从每节点 SQLite 生成 overlap JSON/TXT。目录验收要求每节点都有
摘要且至少匹配一个 measured NVTX iteration；摘要额外报告 `feedback_prepare_gpu_ms`。摘要是 NVTX 与 CUDA kernel 区间的交集，
最终 stream/event 依赖仍以 `.nsys-rep` timeline 为准。

Test 02 使用自身 `bash/run.sh` 与 `scripts/`，不调用 Test 01。禁止为了 nsys 创建 Test
03 或独立平级实验；全部 target 都写入本次 Test 顶层 run 的 `nsys/`。
