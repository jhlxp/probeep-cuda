# Test 01 私有执行工具

本目录只服务 Test 01，不被其他 Test 调用。`bash/run.sh` 通过这里完成 workload 构造、
H20/SM90/RDMA preflight、源码构建、五算法正确性与性能、nsys、绘图和目录验收。

`configs/h20_multinode.env.example` 是唯一配置模板；`runner.py` 只负责控制面，ProbeEP
timed 热路径仍必须位于 `src/deepep-probeep` 的 CUDA/C++ 实现中。

`formal_entrypoint.py` 校验不可变 case plan，并逐 case 启动隔离的 16-rank worker group（每节点 8 rank）；
`formal/` 内含本 Test 私有的五 backend adapter、dual-microbatch HT scheduler、grouped FFN、
CPU oracle 和结果 schema。它们不调用其他 Test，也不承载 ProbeEP planner 热路径。

`analyze_nsys_overlap.py` 只处理 `formal-pipeline` 导出的 SQLite，生成逐节点 JSON/TXT；
正式 latency 仍来自未开启 profiler 的 `benchmark/raw/iterations.csv`。

`reprocess_layers_by_algorithm.py` 用于对已有正式 raw log 做按算法、按层的复核。它固定丢弃
物理 Round 1–10 warmup，把 CSV 中 measured iteration 0–9 显示为物理 Round 11–20。输出根页
只放五个算法入口；进入算法后是 20 层目录，再点击进入单独 Layer 页面。单层页面只包含
该层的十轮时延、双 microbatch DAG/时间线和 rail 数据；统计只在同层内部求
mean/min/max。页面先展示严格时间戳时间线与事件依赖 DAG，随后依次展示 Microbatch 1、
Microbatch 2 的完整独立区块；不生成 MB0+MB1 合计 rank/rail 柱图。

新正式 run 必须写出 `microbatch_rank_samples.csv` 和 `microbatch_timeline.csv`：前者保存
每个 Round 的 MB0/MB1 逐 rank home/execution rows，后者保存逐 rank、逐 stream 的 CUDA-event
起止时间。DAG 固定表达 `A0 → (A1 ∥ W+D0) → (E0 ∥ W+D1) → E1`，并分开表示同 stream 顺序、event wait、host 完成边界与跨轮 feedback；实测时间线直接使用 `start_ms/end_ms`，不从 phase 时长拼接；
kernel 级结论仍以 nsys 为准。每个 microbatch 的 directed server-pair 与 per-rail/NIC 图
独立渲染；每条 rail 严格堆叠 `[Token Dispatch bytes][Expert Weight bytes]`，两段不覆盖。fresh raw 的 Expert 段再按 expert id 着色，cache-hit placement 单独列出且 actual bytes=0。
ProbeEP controller 融合在下一次 `balanced_dispatch` 内，不伪造独立 timeline 阶段。本 iteration
结束后的 observation producer 使用同一 CUDA-event 时间原点。A/M feedback 按
`(phase,round,compute_kind)` 分键：Layer L/MB0 只消费 Layer L−1 Attention，Layer L/MB1
只消费 Layer L−1 MoE；Layer 00 使用明确 bootstrap。目标拓扑固定为每 server
`4×400G physical NIC → 8×200G logical rail`，raw schema v4 明确记录跨层 provenance、
NIC/subrail、rail cap 与 cold weight version。

同一 run 还必须写出：

| raw 文件 | 数据来源 |
|---|---|
| `rdma_path_load.csv` | 五 backend 已完成的 runtime route/count tensor；timed 区间后只读取，不重跑 routing |
| `probeep_observation_samples.csv` | 本层 controller 实际消费的上一层、同 round、同类 A/M CUDA-event 与 byte observation |
| `probeep_plan_summary.jsonl` | production CUDA handle 的 intent/admission/invariant/phase-cycle/chunk table |
| `probeep_weight_chunks.csv` | production chunk table 的逐 expert、逐 offset、逐 rail 行 |

`validate_run_layout.py --kind test` 对这些文件 fail-closed：时间线 131,200 行、rail 32,000
行、A/M observation 6,080 行，并要求五算法、两个 microbatch、20 层都完整。下一次 H20
run 的 nsys formal pipeline 固定 profile Layer 00 的五个算法；所有 target 使用
`cuda,nvtx,osrt`，不再给 UltraEP 生成空 trace。

```bash
python3 tests/Test_01_RawData1_Eval20/scripts/reprocess_layers_by_algorithm.py \
  test_logs/Test01_20260817/run_20260817_test01_persistent_warm_full
```
