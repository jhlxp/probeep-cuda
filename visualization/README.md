# 多机可视化

Test 01 和 Test 02 各自的 `scripts/load_profile.py` 在本次 Test 顶层 run 内生成：

```text
artifacts/
├── result.json
├── visualization_bundle.zip
└── figures/
    ├── end_to_end_latency.{pdf,png}
    ├── rank_load.{pdf,png}
    └── rank_expert_load.{pdf,png}
```

ZIP 只包含一个自包含的 `load_profile.html`，不依赖 CDN。页面同时展示 NCCL、DeepEP、
DeepEP-MoonEP、UltraEP + 固定 HybridEP、ProbeEP 的 rank/专家/server 负载、真实 RDMA
path、CUDA stage 和 ProbeEP plan。Test 01/02 的 `bash/run.sh` 会自动生成并校验这些产物，
无需调用共享聚合脚本。

Nsight Systems report 保存在同一 Test run 的 `nsys/<target>/`，不参与正式时延聚合。
