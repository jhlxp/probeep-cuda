# Test Logs

本目录只保存 `tests/Test_XX_*` 产生的真实测试日志：

```text
test_logs/run_<UTC>_testXX_<selector>_<world>r/
```

一个 Test 入口对应一个顶层 log；setup、workload、correctness、benchmark、nsys 和
artifacts 都必须位于该 log 内。禁止写入 Expe 日志、源码、手工性能数字，禁止混合不同
Test 或不同 run 的 raw 数据。具体 schema 和作废条件由对应 Test 自己的 `README.md` 与
`scripts/validate_run_layout.py` 完整定义。
