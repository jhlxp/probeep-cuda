# DeepEP-MoonEP 开发验收

## 1. 目标

DeepEP-MoonEP 是 ProbeEP 的直接 baseline：先把 MoonEP 的 server-local 均衡思想落到 DeepEP RDMA 数据面里，再在 ProbeEP 里增加跨 server admission。

| 项 | 要求 |
|---|---|
| 代码目录 | `src/deepep-moonep` |
| 参考语义 | `src/moonep`，只参考，不修改 |
| 数据面 | DeepEP normal-mode CUDA/NVSHMEM/IBGDA |
| 均衡范围 | 每台 server 内 8 个 ranks |
| 不做内容 | 不做跨 server 均衡 |
| 热路径 | CUDA/C++，不走 Python route planner |
| 正式角色 | MoonEP baseline，必须进入五算法 performance |

## 2. 算法语义

| 阶段 | 输入 | 输出 | 验收 |
|---|---|---|---|
| rank/expert histogram | `topk_idx` | server-local expert load | 与 CPU reference 对齐 |
| server-local balance | 同 server 内 8 ranks | replica placement | rank padded load 下降 |
| route materialize | top-k routes | balanced execution map | valid prefix 正确 |
| weight sync | server-local replica | local replica slots | checksum/version PASS |
| dispatch | DeepEP RDMA dispatch | expert input buffer | 无额外 host sync |
| grouped FFN | balanced rows | expert output | padding 不参与有效结果 |
| combine/backward | balanced handle | output/grad | replay plan PASS |

## 3. 和 official MoonEP 的关系

| 内容 | official MoonEP | DeepEP-MoonEP |
|---|---|---|
| 目标 | 单机 NVLink/VMM 场景 | 多机 DeepEP RDMA 工程底座 |
| 均衡范围 | 单机内部 ranks | 每台 server 内 EP8 |
| planner 语义 | 参考 | 保留 server-local 思想 |
| 数据传输 | MoonEP 自己的数据面 | DeepEP normal-mode |
| Python wrapper | 有 | timed path 不依赖 Python planner |
| 作用 | 算法参考 | 正式 MoonEP baseline |

## 4. 开发表

| ID | 模块 | 文件/接口 | 验收 |
|---|---|---|---|
| M01 | DeepEP base | `src/deepep-moonep/csrc/kernels/*` | official dispatch/combine PASS |
| M02 | histogram | `moonep_plan.cu` | CPU/CUDA load match |
| M03 | local planner | `moonep_plan.cu` | server-local padded load 下降 |
| M04 | route layout | `BalancedHandle` | valid rows/padding 正确 |
| M05 | expert I/O | `moonep_expert_io.cu` | replica checksum PASS |
| M06 | transport | `moonep_transport.cu` | no host poll |
| M07 | backward | balanced handle replay | grad reduce PASS |
| M08 | benchmark | `deepep_moonep_on` | 五算法 runner PASS |

## 5. 测试矩阵

| 测试 | 内容 | PASS |
|---|---|---|
| CPU reference | server-local placement | exact/invariant PASS |
| build | `src/deepep-moonep` extension | import PASS |
| forward | balanced/server skew/raw_data1 layer | tolerance PASS |
| expert I/O | local replica weight sync | checksum PASS |
| backward | replay plan + grad reduce | tolerance PASS |
| performance | `deepep_moonep_on` | 生成完整 CSV/JSON |
| nsys | no Python planner/no allocator | 无新增瓶颈 |

## 6. 作废条件

| 情况 | 处理 |
|---|---|
| 跨 server 移动 rows | 不是 MoonEP baseline，作废 |
| 用 official MoonEP Python 热路径直接计时 | 作废 |
| 不走 DeepEP RDMA 数据面 | 作废 |
| 和 ProbeEP 共用可变全局状态 | 作废 |
| baseline 漏计 weight sync/layout/materialize | 作废 |

