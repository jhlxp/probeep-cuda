# raw_data

该目录从 `Poseidon/workload/raw_data` 按字节复制，是不可改写的原始输入。复制时共
33 个运行时输入文件、817,211 bytes；按“文件名 + NUL
+ 文件内容”排序计算的目录 SHA-256 为：

```text
7c2119fb81bc13bf87d4073941fd00080f5e47a766dee7564d387267b311cff9
```

| 输入 | Shape | 语义 |
|---|---|---|
| `decode_{rank}.csv` | 32 files × 58 layers × 9 slots | physical expert slot 接收 route 数 |
| `ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json` | 58 layers × 32 ranks × 9 slots | physical slot → logical expert 映射 |

两类文件必须一起使用。每层有 288 个 physical slots、256 个 logical experts 和 32 个
冗余物理槽。`workload/build_raw_data1.py` 读取本目录并生成无冗余的
`workload/raw_data1`；任何实验都不得原地修改本目录。

原数据不包含逐 token trace 或 source-rank 关联，因此只能恢复全局 expert receive
histogram，不能声称复现原始 token routing trace。
