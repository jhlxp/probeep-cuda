# ProbeEP CUDA 论文实验分类合同

## 1. 一一对应关系

一个 Expe 目录表示一个明确的论文问题；一次 Expe 入口只生成一个顶层 Expe log。任何
Expe 都必须独立，不调用 Test、其他 Expe 或共享实验执行目录。

```text
experiment/Expe_XX_Question_Name/
├── README.md
├── bash/
│   └── run.sh
└── scripts/
    ├── configs/<experiment>.env.example
    ├── create_venv.sh
    ├── launch_slurm.sh
    ├── slurm_job.sh
    ├── launch_node.sh
    ├── runner.py
    ├── preflight.py
    ├── workload.py
    ├── collect.py
    ├── plot.py
    ├── validate_run_layout.py
    └── test_*.py
```

禁止建立 `experiment/common/`。可以复用项目级 `src/`、只读 `workload/` 和算法定义，
但执行代码必须复制到本 Expe 的 `scripts/` 后由本 Expe 维护。Expe 不能通过 shell、
Python import 或 symlink 调用 `tests/`、其他 `Expe_*` 或它们的日志。

## 2. Expe README 强制内容

每个 `Expe_XX_*/README.md` 必须是完整任务书，不能只链接中央文档。至少包含：

| 章节 | 必须写清楚 |
|---|---|
| 实验身份 | 编号、论文问题、假设、对应 figure/table、非目标 |
| 独立边界 | 唯一入口、唯一 Expe log、禁止调用的目录 |
| 文件职责 | 本 Expe 每个脚本、配置、数据和产物的责任 |
| 源码版本 | 五算法或消融版本、source tree/commit、HybridEP lock |
| 硬件拓扑 | server/GPU/NIC/rank mapping/RDMA/软件栈 |
| workload | 数据来源、层、tokens/rank、TopK、experts、route SHA |
| 自变量 | 本实验主动改变的唯一因素或参数集合 |
| 控制变量 | 必须保持完全一致的环境、shape、调度和计时边界 |
| 指标 | 主指标、次指标、world-rank reduction、percentile/CI |
| 执行模型 | dual microbatch HT、stream/event、warmup/measure/repeat |
| 执行顺序 | setup、sanity、measurement、nsys、artifacts、validation |
| 正确性门禁 | 数值、守恒、planner、RDMA、baseline parity |
| 原始 schema | 每个 CSV/JSONL/trace 字段和来源 |
| 作图合同 | 每个 figure 的输入、轴、聚合方法和误差线 |
| 作废条件 | 缺失算法、route 不同、失败样本、异常值处理、禁止补数据 |
| 排错顺序 | correctness→bytes→rail→stream→kernel→E2E |
| 完成表 | run id、证据路径、状态、问题和修复 |

Test README 中已经写过的环境、五算法、nsys 或日志规则，Expe README 仍要完整重写，
不能用“同 Test”或“见公共文档”省略。这样单独复制某个 Expe 目录时，仍能知道它为何
存在、如何运行、如何验收。

## 3. 唯一 Expe log

```text
experiment_logs/run_<UTC>_expeXX_<question>_<topology>/
├── setup/                 # env、preflight、build、import provenance
├── workload/              # immutable input、manifest、SHA、case plan
├── sanity/                # 本实验运行前的最小正确性
├── measurement/           # 本实验唯一正式 raw 数据
├── nsys/                  # 诊断 trace；不进入正式时延
└── artifacts/             # summary、figure、table、metadata、validation
```

一个 `bash/run.sh` 只能创建一个上述顶层目录。repeat、参数 sweep 和所有方法都放在同一
Expe log 的 immutable case plan 中；不能为有利 repeat 单独创建日志后再挑选。一个
Expe log 也不能混入其他 Expe 或 Test 的 raw/result。

## 4. Expe 与 Test 的区别

| 分类 | Test | Expe |
|---|---|---|
| 回答问题 | 实现/环境是否正确可用 | 某个论文假设是否成立 |
| 顶层目录 | `tests/Test_XX_*` | `experiment/Expe_XX_*` |
| 日志根 | `test_logs/` | `experiment_logs/` |
| 入口 | 自身 `bash/run.sh` | 自身 `bash/run.sh` |
| 执行代码 | 自身 `scripts/` | 自身 `scripts/` |
| README | 完整测试任务书 | 完整实验任务书 |
| 相互调用 | 禁止 | 禁止 |

Test 通过只说明源码具备实验资格。Expe 必须在自己的日志中重新记录 preflight、源码
provenance、workload SHA、所有方法和测量环境，不能把 Test log 重新包装成论文图。

## 5. 建立新 Expe 的门禁

新建 Expe 前逐项确认：

- 论文问题和主指标唯一且明确；
- 目录名、README、入口、scripts 和日志 slug 一致；
- 没有 import、source、symlink 或 shell 调用指向其他 Test/Expe；
- 配置模板不包含当前开发机的 NIC/HCA 假值；
- case plan 覆盖全部方法、参数、repeat，并在运行前固定；
- 五算法使用相同 route、shape、dtype、FFN 和 dual-microbatch HT 边界；
- 原始数据足够从零重建每张图和每个表；
- nsys 与正式 latency 隔离；
- validator 能拒绝缺文件、混日志和损坏 ZIP；
- README 的完成表为空，直到真机证据产生。

当前尚未建立具体 Expe 目录，也不预填论文数字。
