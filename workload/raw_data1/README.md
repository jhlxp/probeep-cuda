# raw_data1

由 `workload/raw_data` 通过 `workload/build_raw_data1.py` 生成。它不是原始物理
placement 的复制品，而是去除 32 个冗余物理槽后的 256-专家 workload。

| 项目 | 输入 | raw_data1 |
|---|---:|---:|
| layers | 58 | 58 |
| devices | 32 | 32 |
| slots/device | 9 | 8 |
| physical slots/layer | 288 | 256 |
| unique logical experts/layer | 256 | 256 |
| duplicate expert IDs/layer | 32 extra slots | 0 |

每层先为每个 logical expert 保留接收量最大的一个物理副本；其余 32 个物理槽的
token count 汇成池，再按 256 个保留主槽的接收量比例用最大余数法回填。随后把该层
分布缩放到 `16 ranks × 4096 tokens/rank × TopK
8 = 524,288` expert rows。输出 JSON 的 32 个 storage ranks
固定保存专家 `[8r, 8r+8)`；这只是 canonical 文件分块。运行时按
`256 / model_ranks` 专家/GPU 重新分组；EP16 为 16 experts/GPU。

| 转换项 | 值 |
|---|---|
| primary policy | `max-receive` |
| redistribution | `proportional_to_retained_primary_load` |
| integer apportionment | `largest_remainder_stable_expert_id` |
| scaling | deterministic largest remainder；无随机采样 |
| model ranks | 16 |
| tokens/rank | 4,096 |
| TopK | 8 |
| output rows/layer | 524,288 |
| Layer 0 retained primary | 2,493,953 |
| Layer 0 redistributed replicas | 374,175 |
| Layer 0 pre-scale total | 2,868,128 |
| Layer 0 output total | 524,288 |

原始 33 个运行时文件（JSON+CSV）tree SHA-256：

```text
7c2119fb81bc13bf87d4073941fd00080f5e47a766dee7564d387267b311cff9
```

输出 33 个运行时文件（JSON+CSV，不含 README/plot）tree SHA-256：

```text
5dd06345102493f76d7b5172f615a77055a0c6db18160e042e8cd90071d2fafb
```

`plot/` 只保存这份 `raw_data1` 派生出来的负载图，不混放其他 workload 的图片。
