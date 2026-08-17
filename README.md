# ProbeEP CUDA

ProbeEP CUDA 基于 DeepEP normal HT 的 CUDA/NVSHMEM/IBGDA 数据面实现跨服务器 MoE 负载均衡。算法先降低服务器间 padded compute，再完成服务器内 GPU 二次均衡；完整专家权重按动态 endpoint budget 分 chunk，并按有向 server pair 的 rail 水位调度。

## 文件夹职责

```text
ProbeEP_cuda/
├── src/                    # 五套实现/基线源码
├── workload/               # 原始 trace、派生 workload 与转换工具
├── tests/Test_01_*/        # 自包含的 RawData1 前 20 层多机测试
├── tests/Test_02_*/        # 自包含的 RawData1 完整 58 层多机测试
├── test_logs/              # 每次真实运行的唯一落盘位置
├── visualization/          # 离线绘图与自包含 HTML ZIP
├── experiment/             # 每个 Expe 独立的论文问题与执行入口
├── experiment_logs/        # 每次 Expe 运行的唯一落盘位置
└── markdown/               # 算法、实现和验收合同
```

### 源码

| 路径 | 唯一职责 | 修改边界 |
|---|---|---|
| `src/deepep/` | 官方 DeepEP baseline | 固定版本；不在目录内实现 ProbeEP |
| `src/moonep/` | 官方 MoonEP 参考源码 | 固定版本；只用于算法与实现对照 |
| `src/ultraep/` | 官方 UltraEP + 指定 HybridEP baseline | 固定版本；HybridEP tree 必须通过 source lock |
| `src/deepep-moonep/` | 基于 DeepEP 数据面的 server-local MoonEP baseline | 独立基线；不得混入 ProbeEP 跨服务器策略 |
| `src/deepep-probeep/` | ProbeEP 算法、融合 CUDA planner、RDMA 数据面和 PyTorch binding | ProbeEP 唯一开发目录；热路径不得外置到 Python |
| `src/VENDORED_VERSIONS.md` | 上述 vendored 源码的版本记录 | 源码版本变化时同步更新 |

### Workload、测试与产物

| 路径 | 唯一职责 | 不应放入 |
|---|---|---|
| `workload/raw_data/` | 保存原始 Gate trace | 修改后的 JSON/CSV、测试结果 |
| `workload/raw_data1/` | 保存去冗余并缩放后的 E256、58 层 DSV3 论文主 workload | 运行时生成的 route tensor、其他 workload 图片 |
| `workload/raw_data2/` | 保存 94 个样本、`max/mean=8..14` 的辅助压力 workload | 冒充真实 DSV3 58 层主结果 |
| `workload/build_raw_data*.py` | 从只读 `raw_data/` 重建派生 workload | 原地覆盖 `raw_data/` |
| `workload/gate/` | 读取原始 Gate 数据的公共解析逻辑 | CUDA planner 或通信实现 |
| `tests/Test_01_RawData1_Eval20/` | 前 20 层五算法快速多机测试；自带 env/venv/Slurm/runner/workload/nsys/绘图/验收 | 论文正式数字、对 Test 02 或共享测试目录的调用 |
| `tests/Test_02_RawData1_All58/` | 完整 58 层五算法正式多机测试；自带完整独立执行栈 | smoke、单层结果、对 Test 01 或共享测试目录的调用 |
| `test_logs/` | 一个 Test 入口对应一个不可覆盖的 Test log | Expe 日志、源代码、手工填写的性能数字 |
| `visualization/` | 记录正式日志的离线逐算法、逐层报告合同 | benchmark timed path、伪造输入 |
| `experiment/` | 按 `Expe_XX_*` 保存完全独立的论文实验目录、README、入口和私有 scripts | `common/`、对 Test/其他 Expe 的调用、预填数字 |
| `experiment_logs/` | 一个 Expe 入口对应一个不可覆盖的 Expe log | Test 日志、不同 Expe 混合数据 |
| `markdown/` | 记录不可变算法语义、实现状态、测试门禁和产物 schema | 与当前代码不一致的历史方案 |

只有 Test 01 和 Test 02。correctness、observation、nsys、图和 HTML 都是这两个 Test
各自 run 目录中的阶段产物，不再单独编号。两个 Test 不共享执行代码。每个 Expe 也必须
自带完整 README、入口和执行脚本，不调用 Test 或其他 Expe。

数据和结果只能按以下方向流动：

```text
raw_data（只读） → raw_data1/raw_data2 → tests/Test_01 或 Test_02
    → test_logs/run_*

src + workload（只读） → experiment/Expe_XX → experiment_logs/run_*
```

Vendored 版本见 [src/VENDORED_VERSIONS.md](src/VENDORED_VERSIONS.md)。UltraEP 使用的
HybridEP 版本由每个 Test 私有的 `scripts/source_lock.json` 校验 vendored Git tree。

## 实验合同

| 项 | 主测配置 |
|---|---:|
| GPU | H20，SM90 |
| GPUs/server | 8 |
| routed experts | 256 |
| tokens/rank | 4096 |
| TopK | 8 |
| hidden | 7168 |
| execution | dual microbatch HT |
| inter-server | 真 RDMA/NVSHMEM/IBGDA |
| workload | raw_data1，完整 DSV3 58 层分布 |

当前首先验收 2 台 H20 服务器、16 个 global rank。扩到更多服务器时仍固定 E256，并要求 `256 % world_size == 0`；当前 EP16 correctness oracle 参数化完成之前，不能只修改 `NNODES` 就声明更大拓扑已通过。

正式比较五个方法：

| 方法 | 代码 | 均衡范围 |
|---|---|---|
| NCCL | benchmark adapter | 不做专家均衡 |
| DeepEP | `src/deepep` | 不做专家均衡 |
| DeepEP-MoonEP | `src/deepep-moonep` | server 内均衡 |
| UltraEP + HybridEP | `src/ultraep` | server 内均衡 |
| ProbeEP | `src/deepep-probeep` | 跨 server + server 内均衡 |

## 当前实现边界

| 功能 | 当前状态 |
|---|---|
| fused CUDA histogram、controller、server-first planner、local packing | 已实现并有单 GPU CUDA 测试 |
| sampled `alpha=0.90` controller | 已实现 consumer 和两行 A/M 状态 |
| complete-expert admission、pair-aware chunk scheduling | 已实现 |
| DeepEP normal HT dispatch/combine | 已接入 ProbeEP 主路径 |
| weight/gradient transport、forward/backward oracle | 已有多机测试入口，等待 H20 真机执行 |
| Attention/MoE benchmark observation producer | 已接入双 microbatch CUDA-event 实测窗口；业务框架 producer 不属于当前论文 benchmark |
| raw_data1 eval20/all selector、精确 TopK materializer 与 case plan | 已分别内置于 Test 01/02 |
| raw_data1 五算法正式 backend entrypoint | 已内置于 Test 01/02；NCCL、DeepEP、DeepEP-MoonEP、UltraEP+HybridEP、ProbeEP 分进程隔离 |
| H20 真 RDMA 性能结果 | 尚未执行 |

静态检查、单卡 planner 数字和合成 observation 都不能作为 H20 多机结果。

## 文档

| 文档 | 内容 |
|---|---|
| [01_DeepEP-MoonEP开发验收.md](markdown/01_DeepEP-MoonEP开发验收.md) | DeepEP-MoonEP baseline 设计与验收 |
| [02_ProbeEP开发验收.md](markdown/02_ProbeEP开发验收.md) | ProbeEP 核心算法、复杂度、融合实现和功能测试表 |
| [03_Nsight-Systems使用.md](markdown/03_Nsight-Systems使用.md) | nsys profile 与 overlap 检查 |
| [04_日志与可视化产物.md](markdown/04_日志与可视化产物.md) | run 目录、CSV/JSON、HTML ZIP 和图表合同 |
| [Test 01 README](tests/Test_01_RawData1_Eval20/README.md) | RawData1 前 20 层多机调试测试的完整独立任务书 |
| [Test 02 README](tests/Test_02_RawData1_All58/README.md) | RawData1 完整 58 层多机正式测试的完整独立任务书 |

## H20 多机入口

```bash
cd /home/chen/workspace/infra/ProbeEP_cuda

export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
bash tests/Test_01_RawData1_Eval20/scripts/create_venv.sh

cp tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env.example \
   tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env
```

先根据目标集群修改 NIC/HCA、Slurm 和 HybridEP transport。Test 01 的一次入口会在同一
run 内依次执行 setup、correctness、benchmark、nsys 和 artifacts：

```bash
ENV_FILE=tests/Test_01_RawData1_Eval20/scripts/configs/h20_multinode.env \
  bash tests/Test_01_RawData1_Eval20/bash/run.sh
```

Test 02 使用自己的脚本和配置，不调用 Test 01：

```bash
bash tests/Test_02_RawData1_All58/scripts/create_venv.sh
cp tests/Test_02_RawData1_All58/scripts/configs/h20_multinode.env.example \
   tests/Test_02_RawData1_All58/scripts/configs/h20_multinode.env
ENV_FILE=tests/Test_02_RawData1_All58/scripts/configs/h20_multinode.env \
  bash tests/Test_02_RawData1_All58/bash/run.sh
```

每次 run 使用独立目录；五算法必须来自同一个 job，并共享 routing SHA、workload、runner mode 和计时边界。

分类测试入口见 [tests/README.md](tests/README.md)。论文主 workload 是
`raw_data1_all`，完整 58 层全部进入统计；`raw_data1_eval20` 只用于快速调试。
