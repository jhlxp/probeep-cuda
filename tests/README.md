# ProbeEP CUDA 多机测试

本文件只负责分类索引，不是 Test 的执行合同。正式测试只有两个目录，而且完全独立：

| 目录 | workload | 论文资格 |
|---|---|---:|
| [Test_01_RawData1_Eval20](Test_01_RawData1_Eval20/README.md) | DSV3 Layer 0–19，五算法快速多机测试 | 否 |
| [Test_02_RawData1_All58](Test_02_RawData1_All58/README.md) | DSV3 Layer 0–57，五算法完整多机测试 | 是 |

每个 Test 都自行保存 `benchmark.py`、`bash/run.sh`、env 模板、venv 创建脚本、Slurm/
手工多机 launcher、preflight、runner、RawData1 materializer、case-plan、nsys、绘图和
目录校验工具。两个 Test 不互相调用，也不依赖 `tests/` 下的共享执行目录。环境、拓扑、
算法、workload、执行顺序、日志、nsys 和作废条件都在各自 README 重复写全。

每次执行只产生一个顶层目录：

```text
test_logs/run_<UTC>_test0X_<selector>_<world>r/
├── workload/                         # manifest、routes、benchmark_plan
├── setup/                            # preflight、build、import provenance
├── correctness/
│   ├── deepep/
│   ├── deepep-moonep/
│   ├── probeep/
│   ├── observation-synthetic/
│   └── ultraep-hybridep/
├── benchmark/                        # 五算法 timed raw/result/logs
├── nsys/                             # 各 backend 的 report/sqlite/stats
└── artifacts/                        # result、PDF/PNG、单 HTML ZIP、metadata
```

正确性、动态探测 consumer、性能、Nsight 和图表都是 Test 01/02 日志的一部分，不再
单独编号。真实 observation 证据写入 `benchmark/raw/`；合成输入检查只写入
`correctness/observation-synthetic/`。静态 plan 和 `PLAN_ONLY=1` 都不是实验结果。

```bash
PLAN_ONLY=1 bash tests/Test_01_RawData1_Eval20/bash/run.sh
PLAN_ONLY=1 bash tests/Test_02_RawData1_All58/bash/run.sh
```
