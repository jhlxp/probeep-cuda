# raw_data2

由 `workload/raw_data` 通过 `workload/build_raw_data2.py` 生成。它先执行
`raw_data1` 同款去冗余与缩放，再把每个 prefill-batch sample 的 expert-level
`max/mean` 控制到 `8..14`，最后按固定
seed 打散 sample 顺序。

注意：底层 JSON 仍使用 `layer_list` 字段，这是为了复用现有 loader 和 benchmark
代码；对 `raw_data2` 来说，`decode_*` 的每一行语义是一个 Qwen3-235B-style
MoE layer slot / prefill batch hotspot sample，不是 DSV3 的第 N 个 MoE layer。
本地只有 58 个经验模板；输出 94 个样本时，额外样本是固定 seed 的模板复用，不声称
是真实 Qwen3 routing trace。

| 项目 | raw_data2 |
|---|---:|
| prefill batch samples | 94 |
| target model layer count | 94 |
| logical experts | 256 |
| storage layout | 32 × 8 |
| model ranks | 16 |
| tokens/rank | 4,096 |
| TopK | 8 |
| output rows/sample | 524,288 |
| primary policy | `max-receive` |
| tail cap range | `8..14` |
| shuffled sample seed | 20260815 |
| empirical source templates | 58 |
| capped source templates | 14 |
| redistributed rows across all samples | 396,576 |

| max/mean 指标 | min | mean | max |
|---|---:|---:|---:|
| before tail cap | 10.7900 | 14.0998 | 23.3916 |
| after tail cap + shuffle | 10.7900 | 12.4096 | 14.0000 |

原始 33 个运行时文件（JSON+CSV）tree SHA-256：

```text
7c2119fb81bc13bf87d4073941fd00080f5e47a766dee7564d387267b311cff9
```

输出 33 个运行时文件（JSON+CSV，不含 README/plot）tree SHA-256：

```text
f1be140954e6fc1feb18092f45bb0f1243f52ff030c20654209004fa115ec874
```

Sample template map (`new_sample -> source_template`):

```text
0->38, 1->42, 2->53, 3->13, 4->55, 5->21, 6->46, 7->23, 8->15, 9->20, 10->50, 11->7, 12->11, 13->49, 14->54, 15->14, 16->12, 17->29, 18->51, 19->44, 20->30, 21->22, 22->19, 23->36, 24->37, 25->39, 26->17, 27->24, 28->10, 29->56, 30->0, 31->27, 32->8, 33->25, 34->6, 35->41, 36->32, 37->9, 38->3, 39->26, 40->57, 41->34, 42->2, 43->47, 44->52, 45->16, 46->40, 47->43, 48->31, 49->4, 50->18, 51->28, 52->35, 53->33, 54->45, 55->48, 56->1, 57->5, 58->41, 59->5, 60->50, 61->28, 62->40, 63->22, 64->43, 65->11, 66->10, 67->3, 68->46, 69->32, 70->8, 71->31, 72->52, 73->7, 74->19, 75->55, 76->35, 77->37, 78->14, 79->57, 80->26, 81->16, 82->44, 83->24, 84->51, 85->18, 86->17, 87->29, 88->36, 89->33, 90->23, 91->2, 92->9, 93->4
```

`plot/` 只保存这份 `raw_data2` 派生出来的 prefill-batch 负载图，不混放其他
workload 的图片。
