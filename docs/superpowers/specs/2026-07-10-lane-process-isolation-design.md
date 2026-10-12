# Lane 进程隔离设计

## 背景

当前项目在一个主 Python 进程内加载多路 Zipformer recognizer。`OnlineRecognizer.from_transducer()`
对 Zipformer 会分别创建 encoder、decoder、joiner 三个 ORT session。多路 Zipformer
同时位于同一进程时，SpaceMiT EP 的 session/thread-pool 资源会互相影响；加入
`SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1` 后，4 路初始化可以通过，但该模式要求共享
global intra thread 的 session 不要同时推理。

目标是把 Zipformer 的计算资源集中到各自 lane 进程内：每一路独占音频输入线程、
Zipformer、Qwen3，并单独统计该路性能数据。主进程不再直接执行模型推理，只负责编排、
状态管理、IPC 事件汇总和全局 metrics 落盘。

## 目标

- 每一路 lane 使用独立 Python 子进程。
- 每个 lane 子进程独占该路音频输入、Zipformer recognizer、Qwen3 `llama-server`、
  Qwen3 client、性能统计和本路结果落盘。
- 主进程并发启动所有 lane 子进程，使多路模型能够并行加载。
- `results/laneX.txt` 和 `model_outputs/laneX.jsonl` 由对应 lane 子进程写入。
- 主进程汇总每路上报的摘要信息，写入全局 `config.json` 和 `metrics.json`。
- 不通过 IPC 传输 PCM、segment cache、Qwen raw response 或大块结果内容。

## 非目标

- 不在本项目中显式绑定 Linux 线程到 A100/X100 核。绑核仍由 sherpa-onnx、SpaceMiT EP
  和 `llama-server` 配置完成。
- 不改变 Zipformer、Qwen3 的模型调用语义。
- 不引入跨进程共享内存传输音频。
- 不让主进程重新解析 lane 写出的 model output 内容来计算 metrics；metrics 由 lane
  子进程直接上报摘要。

## 推荐架构

采用方案 A：每路一个 lane 子进程，子进程内管理音频、Zipformer、Qwen3，主进程只汇总事件。

```text
main process
  ├─ LaneProcessController[0]
  ├─ LaneProcessController[1]
  ├─ LaneProcessController[2]
  ├─ LaneProcessController[3]
  └─ async IPC reader tasks
       ├─ asr_event -> FrontendEventBus.publish(...)
       ├─ metrics_snapshot -> LaneMetrics cache
       ├─ metrics_final -> final metrics aggregator
       ├─ paths -> run artifact registry
       ├─ lifecycle -> state aggregation
       └─ fatal/exit -> stop all lanes, mark FATAL
```

主进程职责：

- 创建本次 run 目录。
- 为每路构造 `LaneConfig` 和输入音频路径。
- 并发启动所有 lane 子进程。
- 为每个 lane 创建 IPC reader task。
- 将 lane 返回的 `asr_event` 转成现有 `ASREvent` 并发布到 `FrontendEventBus`。
- 维护每路 lifecycle、metrics snapshot、final metrics、artifact paths 和错误信息。
- 在全部 lane loaded 后把整体状态置为 `LOADED`。
- 在全部 lane finished/stopped 后把整体状态置为 `STOPPED`。
- 任一路 fatal 时通知其他 lane 停止，并把整体状态置为 `FATAL`。
- run 结束后写全局 `config.json` 和 `metrics.json`。

lane 子进程职责：

- 启动并健康检查本路 `llama-server`。
- 加载本路 Zipformer recognizer。
- 创建本路 `Qwen3AsrClient`。
- 启动本路音频生产者线程。
- 启动本路 Zipformer 消费者线程。
- 在进程内使用 `OnlineFrameQueue(maxsize=1)` 连接音频生产和 Zipformer 消费。
- 在本进程 event loop 中调度并接收本路 Qwen3 请求。
- 维护本路 `LatencyTracker`、`RtfTracker`、`EventLoopLagTracker` 和 `LaneMetrics`。
- 直接写 `results/laneX.txt`。
- 直接写 `model_outputs/laneX.jsonl`。
- 结束时向主进程发送 `metrics_final` 和 `paths`。
- 异常时发送 `fatal`，并 best-effort 停止本路 `llama-server`。

## 模型加载流程

主进程启动所有 lane 子进程后不等待某一路完成再启动下一路。每个 lane 子进程独立执行：

```text
laneX process
  send lifecycle: lane_loading
  start own llama-server
  wait own llama-server health
  load own Zipformer
  create own Qwen3 client
  send lifecycle: lane_loaded
```

主进程收到所有 lane 的 `lane_loaded` 后，整体进入 `LOADED`。因此 4 路模型加载日志应该交错出现，
而不是按 lane 0、lane 1、lane 2、lane 3 串行完成。

## Lane 内部线程模型

每个 lane 子进程内部保留当前已经验证过的 lane 线程结构：

```text
laneX process
  ├─ lane event loop thread
  ├─ asr-lane-X-audio-producer
  ├─ asr-lane-X-zipformer-consumer
  └─ llama-server child process
```

音频生产者线程只负责按 100 ms 一帧向 `OnlineFrameQueue(maxsize=1)` 写入音频帧。
Zipformer 消费者线程阻塞读取最新帧并同步调用 sherpa-onnx。Qwen3 请求仍由 lane 进程内
asyncio task 发出和接收。同一路离线请求保持串行；不同 lane 之间天然并行，因为它们位于不同
lane 子进程并连接不同 `llama-server` 端口。

Zipformer 的 `zipformer.config` 可以继续包含：

```text
SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064
SPACEMIT_EP_INTRA_THREAD_NUM = 1
SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1
```

该配置只在单个 lane 进程内影响该路 Zipformer 的 encoder、decoder、joiner sessions。一个 lane
进程只有一个 Zipformer recognizer，且该 recognizer 的推理由本路 Zipformer 消费者线程串行调用，
满足 shared global intra thread 不并发推理的约束。

## IPC 协议

每个 lane 子进程和主进程之间使用两条控制通道：

```text
main -> laneX: command queue
laneX -> main: event queue
```

IPC payload 必须是 JSON-able `dict`。不直接传 dataclass、numpy array、PCM、segment cache、
raw response 或文件内容。

lane 到主进程事件统一格式：

```python
{
    "type": "asr_event" | "lifecycle" | "metrics_snapshot" | "metrics_final" | "fatal" | "paths",
    "stream_id": 0,
    "seq": 123,
    "time_ms": 1720000000000,
    "payload": {},
}
```

主进程到 lane 命令格式：

```python
{"type": "start"}
{"type": "stop", "reason": "user_requested"}
{"type": "shutdown"}
```

初版实现中，lane 子进程启动后可以自动进入 load/run；主进程至少需要支持 `stop` 和 `shutdown`。

## IPC 事件类型

`asr_event` payload 使用现有 `ASREvent` 字段的 JSON-able 表示：

```python
{
    "event_type": "asr_result",
    "stream_id": 0,
    "sentence_id": 0,
    "revision": 2,
    "status": "final",
    "result_type": "offline",
    "is_final": True,
    "text": "识别结果",
    "timestamp_ms": 1200,
}
```

`lifecycle` payload：

```python
{
    "state": "lane_loading" | "lane_loaded" | "lane_started" | "lane_finished" | "lane_stopped" | "lane_failed",
    "message": "",
}
```

`metrics_snapshot` payload 用于运行中展示。字段是每路当前摘要：

```python
{
    "lane_metrics": {
        "stream_id": 0,
        "processed_frames": 100,
        "dropped_frames": 0,
        "offline_requests": 3,
        "final_state": "stopped",
    },
    "online_frame_latency": {"count": 100, "total_ms": 5000.0, "average_ms": 50.0, "max_ms": 88.0},
    "offline_rtf": {"count": 3, "total": 1.2, "average": 0.4, "max": 0.46},
    "event_loop_lag": {"count": 250, "total_ms": 500.0, "average_ms": 2.0, "max_ms": 9.0},
}
```

`metrics_final` payload 与 `metrics_snapshot` 同结构，但它是该路最终统计来源。

`paths` payload：

```python
{
    "results_path": "results/lane0.txt",
    "model_outputs_path": "model_outputs/lane0.jsonl",
}
```

`fatal` payload：

```python
{
    "phase": "load" | "run" | "stop",
    "error_type": "RuntimeError",
    "message": "Not enough available AI cores for the thread pool",
    "traceback": "...",
    "exit_code": 1,
}
```

## 主进程事件处理

主进程为每路创建一个 reader task。由于 `multiprocessing.Queue.get()` 是阻塞调用，reader task
通过 `asyncio.to_thread(queue.get)` 接入主 event loop。

处理规则：

- 每个 lane 事件带 `seq`，同一路只接受 `seq` 大于已处理 seq 的事件。
- `asr_event` 转换为 `ASREvent` 后调用 `FrontendEventBus.publish(event)`。
- `lifecycle` 更新 lane 状态表。
- `metrics_snapshot` 更新主进程内 lane metrics cache。
- `metrics_final` 保存到 final metrics 聚合器，并优先于 snapshot。
- `paths` 登记该路结果文件和模型输出文件路径。
- `fatal` 记录错误，发布 `fatal_error`，向其他仍活跃的 lane 发送 `stop`。
- 同一路出现 `fatal` 后，除 `metrics_final`、`paths`、`lane_stopped` 外的后续事件可以忽略。

整体状态聚合规则：

```text
all lanes loaded      -> PipelineState.LOADED
any lane fatal        -> PipelineState.FATAL
all lanes finished    -> PipelineState.STOPPED
user stop requested   -> send stop to all lanes, wait exit, then STOPPED
```

## 落盘职责

run 目录结构：

```text
runs/<run-id>/
  config.json
  metrics.json
  results/
    lane0.txt
    lane1.txt
    lane2.txt
    lane3.txt
  model_outputs/
    lane0.jsonl
    lane1.jsonl
    lane2.jsonl
    lane3.jsonl
```

主进程写：

- `config.json`
- `metrics.json`

lane 子进程写：

- `results/laneX.txt`
- `model_outputs/laneX.jsonl`

主进程不再从 `FrontendEventBus.snapshot_results()` 反推 `results/laneX.txt`。前端状态仍由
`asr_event` 实时驱动，但最终文本文件以 lane 子进程写出的文件为准。

## Metrics Schema v3

全局 `metrics.json` 使用 `metrics_schema_version = 3`。顶层保存所有 lane 的聚合摘要，
`lanes` 保存每路详细摘要和 artifact path。

```json
{
  "metrics_schema_version": 3,
  "online_frame_latency": {"count": 400, "total_ms": 20800.0, "average_ms": 52.0, "max_ms": 88.0},
  "offline_rtf": {"count": 20, "total": 8.2, "average": 0.41, "max": 0.53},
  "event_loop_lag": {"count": 1000, "total_ms": 2100.0, "average_ms": 2.1, "max_ms": 12.0},
  "lanes": [
    {
      "stream_id": 0,
      "processed_frames": 100,
      "dropped_frames": 0,
      "offline_requests": 5,
      "final_state": "finished",
      "online_frame_latency": {"count": 100, "total_ms": 5000.0, "average_ms": 50.0, "max_ms": 80.0},
      "offline_rtf": {"count": 5, "total": 2.0, "average": 0.4, "max": 0.48},
      "event_loop_lag": {"count": 250, "total_ms": 450.0, "average_ms": 1.8, "max_ms": 7.0},
      "results_path": "results/lane0.txt",
      "model_outputs_path": "model_outputs/lane0.jsonl"
    }
  ]
}
```

全局 summary 通过每路 summary 的 `count` 和 `total` 计算：

- latency/event-loop-lag：汇总 `count`、`total_ms`，`average_ms = total_ms / count`，
  `max_ms = max(lane.max_ms)`。
- RTF：汇总 `count`、`total`，`average = total / count`，`max = max(lane.max)`。

## 错误处理

lane 子进程错误分三类：

- `load_failed`：Qwen3 启动失败、Zipformer 初始化失败、模型文件缺失。lane 发送 `fatal`，
  主进程进入 `FATAL`，停止其他 lane。
- `run_failed`：音频读取失败、Zipformer 推理异常、Qwen 请求异常。lane 先写已有结果和本路
  metrics，再发送 `fatal`，主进程停止其他 lane。
- `stop_timeout`：主进程请求停止后 lane 没有退出。主进程先 `terminate()`，再超时 `kill()`，
  并在 metrics 中把该路标为 fatal。

fatal 场景下，lane 尽量落盘已有结果，但不为了等待长时间 Qwen 请求无限卡住。等待上限使用
现有 `request_timeout_ms`。

## 停止流程

```text
user stop / run complete / fatal
  main sends stop to all active lanes
  lane stops audio source
  lane wakes OnlineFrameQueue
  lane waits current qwen request / offline tail up to request_timeout_ms
  lane writes lane results + model_outputs
  lane sends metrics_final + paths + lane_stopped
  lane stops its llama-server
  lane exits
```

主进程等待子进程退出。超过 graceful timeout 时先 `terminate()`，再次超时后 `kill()`。

## Headless 与 Web 行为

Headless：

- `HeadlessRunner` 创建 run 目录并使用 lane 进程化 manager。
- 运行结束后，主进程写 `config.json` 和 `metrics.json`。
- `results/laneX.txt` 和 `model_outputs/laneX.jsonl` 已由 lane 子进程写入。

Web：

- WebSocket 广播仍由主进程 `FrontendEventBus.broadcast_loop()` 完成。
- 前端继续接收 `partial`、`pending`、`final`、`fatal_error`、`lane_finished`。
- 模型加载按钮触发主进程并发启动 lane 子进程。
- 如果 lane 数变化，仍要求先 unload，再按新配置创建 lane 子进程。

## 测试策略

单元测试使用 fake audio/online/offline，不依赖真实模型：

- IPC message 序列化/反序列化测试。
- 主进程 event aggregator 测试：`asr_event` 能进入 `FrontendEventBus`，`metrics_final` 能汇总。
- lane fake runner 测试：验证每路独立写 `results/laneX.txt` 和 `model_outputs/laneX.jsonl`。
- failure 测试：一个 lane 发送 `fatal` 后，主进程给其他 lane 发送 `stop`。
- timeout 测试：lane 不退出时，主进程 escalate terminate/kill。
- recorder 测试：`metrics_schema_version = 3`，包含 per-lane summaries，主进程不再写 results 和
  model_outputs 内容。
- headless fake 端到端测试：启动 2 路 fake lane 子进程，确认最终状态、metrics、results、
  model_outputs 都存在。

真实开发板验证：

- 4 路 headless，确认存在 4 个 lane Python 子进程。
- 每个 lane 子进程各自启动 1 个 Qwen3 `llama-server`。
- 每个 lane 子进程内加载自己的 Zipformer。
- 多路模型加载日志交错出现，证明并行加载。
- `results/laneX.txt` 由对应 lane 写出。
- `model_outputs/laneX.jsonl` 由对应 lane 写出。
- `metrics.json` 由主进程汇总写出。
- 检查在线延迟、丢帧和 RTF：目标仍是在线延迟约 50 ms，整段识别 RTF 约 0.4。

## 验收标准

- 4 路 headless fake 测试通过，并生成每路独立结果文件和模型输出文件。
- 4 路真实开发板测试可以并行加载 lane 模型。
- 任一路 lane load/run 失败时，主进程能停止其他 lane，并写出可诊断的 fatal metrics。
- `metrics.json` 使用 schema v3，包含全局 summary 和每路 summary。
- 主进程不加载 Zipformer，不创建 Qwen3 client，不直接启动所有 `llama-server`。
- lane 子进程退出后不会残留本路 `llama-server`。
