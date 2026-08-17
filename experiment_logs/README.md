# Experiment Logs

本目录只保存 `experiment/Expe_XX_*` 产生的真实运行日志。

```text
experiment_logs/run_<UTC>_expeXX_<question>_<topology>/
```

一个 Expe 入口对应一个顶层 log。禁止写入 Test 日志、源码、手工补写的性能数字，禁止
合并不同 Expe 的 raw 数据。具体目录 schema、有效性和作废条件必须由对应 Expe 自己的
`README.md` 与 `scripts/validate_run_layout.py` 完整定义。
