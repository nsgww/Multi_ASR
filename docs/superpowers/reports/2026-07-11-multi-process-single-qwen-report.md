# 多进程 Zipformer + 单共享 Qwen3-ASR 验证报告

## 验证环境

- 开发板：K3，`bianbu@10.0.91.51`
- Python：项目本地 `.venv313`，Python 3.13.12
- sherpa-onnx：1.13.4
- 输入：`~/Music/wav16k` 排序后的前 4 个 16 kHz 单声道 WAV
- 固定配置：q40、`--n-parallel 4`、`--n-ctx 8192`、encoder `8;9`、decoder
  `10,11,12,13,14,15`

板端 `compileall`、完整 pytest（197 passed）和四路 fake headless 均通过。真实测试前的
10 秒单请求冒烟输出非空中文，RTF 0.162329，且返回全部五类共享 timing。

## 拓扑

1 个主 Python 进程、4 个 lane Python 子进程、1 个 llama-multi-asr。

运行中 `pstree` 和 `ps` 均确认只有一个 `llama-multi-asr`，其父进程是主 Python；四个
`spawn_main` lane 同样直接属于主 Python。启动命令包含 q40 模型、`--n-parallel 4` 和
`--n-ctx 8192`。三轮结束后均未残留共享服务进程。

## 性能结论

- 验收：FAIL
- 三次合并 offline RTF：0.478326
- offline 请求数：472

| Run | lane0 drops | lane1 drops | lane2 drops | lane3 drops | online avg/max ms | RTF avg/max | loop lag avg/max ms |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| process-1783753922935 | 6 | 4 | 5 | 5 | 41.412/627.983 | 0.506155/1.983423 | 1.830/52.710 |
| process-1783754219344 | 4 | 5 | 5 | 5 | 41.568/630.299 | 0.485843/2.960517 | 1.823/101.584 |
| process-1783754521323 | 5 | 5 | 4 | 4 | 40.846/610.259 | 0.441413/2.111065 | 1.828/148.873 |

## 共享服务 timings

- `process-1783753922935`: queue_ms=101.331/1437.000ms, encode_ms=389.369/1432.000ms, prefill_ms=291.175/544.000ms, decode_ms=488.050/1762.000ms, total_ms=1594.312/4095.000ms
- `process-1783754219344`: queue_ms=92.057/1561.000ms, encode_ms=377.390/1400.000ms, prefill_ms=292.264/552.000ms, decode_ms=488.113/1766.000ms, total_ms=1534.415/3728.000ms
- `process-1783754521323`: queue_ms=84.431/1270.000ms, encode_ms=373.895/1301.000ms, prefill_ms=298.745/548.000ms, decode_ms=509.510/1709.000ms, total_ms=1533.706/4041.000ms

## 功能检查

四路结果非空、model_outputs 六字段、Zipformer/Qwen 句 ID 对齐。

三轮共完成 472 个离线请求，共享 timing 计数同为 472；12 个 lane 全部以 `finished`
结束。每个 lane 独立写入 `results/laneX.txt` 和 `model_outputs/laneX.jsonl`，主进程输出
schema v3 的 `metrics.json`。完整 run 产物、headless/Qwen 日志和进程树已同步回本机
`runs/2026-07-11-shared-qwen-validation/`，30 个 run 文件的 SHA-256 与开发板逐一一致。

## 失败项

- combined offline_rtf=0.478326 exceeds 0.471

丢帧门槛全部通过，唯一失败项是三轮合并 offline RTF，比目标高 0.007326。按照本任务
约束，未调整核心、线程数、`n_parallel` 或 Zipformer EP 配置。
