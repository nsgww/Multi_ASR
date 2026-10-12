# verbose 模式细化性能指标记录设计

## 问题

`metrics.json` 只保存 `online_frame_latency`、`offline_rtf`、`offline_shared_timings`、`event_loop_lag` 四类指标的聚合值（`count/total/average/max`），单帧/单句的原始数值和发生时间从未落盘，无法回放某次 run 中延迟或 RTF 随时间的变化曲线。

`AppConfig.verbose`（`config.py:186`）和 CLI `--verbose`（`main.py:23`）已经存在，但目前完全没有被消费。

同时，`loop_audio`（循环识别）打开时单个 lane 的 WAV 输入会无限循环、run 可以无限长时间运行；`LaneResultWriter`/`LaneModelOutputJsonlRecorder` 因此在 `loop_audio=True` 时整体 `enabled=False`（`lane_runtime.py:316-323`），避免文件无限增长。任何新增的详细记录都要遵守同一约束。

## 设计

### 记录粒度与开关

- 复用已有的 `AppConfig.verbose`，不新增命令行参数。
- 详细记录仅在 `config.verbose and not loop_audio` 时启用；循环识别时不写详细文件，与现有 `results/`、`model_outputs/` 行为一致。
- 普通模式（非 verbose）下行为完全不变：只有 `metrics.json` 的聚合值，`metrics.py` 中的 `LatencyTracker`/`RtfTracker`/`SharedTimingsTracker`/`EventLoopLagTracker` 及其聚合 schema 不做任何改动。

### 文件布局

按指标类型分目录，每个 lane 一个 jsonl 文件，流式追加写入、不在内存中累积（与 `LaneModelOutputJsonlRecorder` 同构：open("w") → 每条 `write` 后 `flush()` → run 结束 `close()`）：

```
runs/<run>/metrics_detail/
  frame_latency/lane<N>.jsonl
  event_loop_lag/lane<N>.jsonl
  rtf/lane<N>.jsonl
  shared_timing/lane<N>.jsonl
```

每个文件内 schema 单一，不需要 `kind` 字段区分。

`frame_latency/lane<N>.jsonl`：
```json
{"t_ms": 1234.5, "latency_ms": 0.31}
```

`event_loop_lag/lane<N>.jsonl`：
```json
{"t_ms": 1334.5, "lag_ms": 1.2}
```

`rtf/lane<N>.jsonl`：
```json
{"t_ms": 4210.0, "sentence_id": 3, "audio_seconds": 4.12, "elapsed_ms": 2007.0, "rtf": 0.487}
```

`shared_timing/lane<N>.jsonl`：
```json
{"t_ms": 4210.0, "sentence_id": 3, "queue_ms": 12.0, "encode_ms": 340.0, "prefill_ms": 90.0, "decode_ms": 1500.0, "total_ms": 2007.0}
```

`t_ms`：该 lane run 开始（`_run_started` 记录起点 `time.monotonic()`）到本次事件的毫秒偏移，不用挂钟时间，与现有 tracker 内部计时方式一致、不受系统时钟调整影响。

`rtf` 与 `shared_timing` 共享同一个 `sentence_id`，可与 `model_outputs/lane<N>.jsonl` 中的记录关联。

### 代码改动点

- `lane_artifacts.py`：新增一个通用 `DetailJsonlWriter` 类（接口与 `LaneModelOutputJsonlRecorder` 一致：`record(dict) -> None` / `close()`），按路径实例化 4 次（`frame_latency`/`event_loop_lag`/`rtf`/`shared_timing` 各一个），不为每种指标单独建类。
- `lane_runtime.py:_run_started`：
  - 按 `config.verbose and not loop_audio` 构造上述 4 个 writer（关闭时为 `None`）。
  - 记录 run 起始 `time.monotonic()`，用于计算 `t_ms`。
  - 将 writer 传入 `LanePipeline` 构造参数（新增 4 个可选字段，均默认 `None`）。
- `pipeline.py` 四处埋点（写入前判空跳过）：
  - `latency_tracker.add(...)`（:349）旁调用 `frame_latency_writer`。
  - `rtf_tracker.add(...)`（:472）旁调用 `rtf_writer`，带 `sentence_id`。
  - `shared_timings_tracker.add(...)`（:474）旁调用 `shared_timing_writer`，带 `sentence_id`。
- `lane_runtime.py:_track_event_loop_lag`（:608）：`event_loop_lag_tracker.add(...)` 旁调用 `event_loop_lag_writer`。
- `metrics.py`、`metrics.json` schema、`recorder.py` 的聚合/合并逻辑均不改动。

### 多进程场景

`AppConfig`（含 `verbose`）已经整体通过 `multiprocessing.Process` 传给每个 lane 子进程（`process_manager.py`），子进程内 `LaneRuntime._run_started` 直接读取同一个 `verbose` 值，无需额外 IPC 改造。

### 明确不做的事

- Web UI 的实时 `metrics_snapshot` 推送（0.1s 间隔，`lane_runtime.py:611-613`）不改动，仍是瞬时聚合值、不落盘。
- 不移除 Tracker 内部现有的、上限 1024 条但从未导出的环形缓冲（`samples_ms`/`values`）——现有单测覆盖其行为，改动收益低、风险不必要。
- 不新增命令行参数。

## 验证

- 单元测试：
  - `config.verbose=False`（默认）或 `loop_audio=True` 时，`metrics_detail/` 目录不产生；`metrics.json` 输出与改动前完全一致（回归测试）。
  - `config.verbose=True` 且 `loop_audio=False` 时，4 个子目录各自产生对应 lane 的 jsonl 文件，逐行 schema 校验（字段名、类型）。
  - `t_ms` 单调不减；`rtf`/`shared_timing` 文件中同一事件的 `sentence_id` 一致，且能在 `model_outputs/lane<N>.jsonl` 中找到相同 `sentence_id`。
  - `verbose=True` 且 `loop_audio=True` 时，详细文件不产生（与现有 `model_outputs`/`results` 行为一致的回归测试）。
- 多进程测试：`process_manager` 场景下确认每个 lane 子进程按各自 `stream_id` 写入独立文件，互不覆盖。
