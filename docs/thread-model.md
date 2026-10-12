# 线程与进程模型

本项目采用“一个主控进程、每路一个 lane 进程、一个共享 Qwen3-ASR 服务”的结构。
主进程负责服务和 lane 生命周期、状态机、IPC 事件汇总、WebSocket 广播以及全局指标落盘；
每个 lane 进程独立负责本路音频输入、Zipformer、Qwen HTTP 请求、结果文件和本路指标。

PCM、NumPy 数组、segment cache、Qwen 原始响应和结果文件内容都不会经过主进程 IPC。
因此主进程处理控制事件或汇总指标的速度，不会反向阻塞每路 100 ms 音频生产。

项目不直接给 Python 线程调用 Linux 绑核接口。Zipformer 的 SpaceMiT EP 配置和共享 Qwen
服务启动参数负责声明推理资源；实际线程位置应通过 `/proc/<pid>/task/*/status` 和运行时
工具验证。

## 进程与线程数量

设识别路数为 `N`：

- 1 个主 Python 进程，运行主 asyncio event loop 和 `ProcessPipelineManager`。
- `N` 个 lane Python 子进程，由 `multiprocessing` 的 `spawn` 上下文创建。
- 1 个 `llama-multi-asr` 进程，由主 Python 进程启动、监控和停止。
- 每个 lane 有 1 个 Python 主线程，承载该 lane 的 asyncio event loop。
- 每个 lane 创建 1 个 `asr-lane-X-audio-producer` 音频生产线程。
- 每个 lane 创建 1 个 `asr-lane-X-zipformer-consumer` Zipformer 消费线程。

默认四路运行时共有 1 个主 Python 进程、4 个 lane Python 子进程和 1 个共享 Qwen
服务进程；4 个 lane 合计创建 8 个专用 pipeline 工作线程。

asyncio 默认 executor、sherpa-onnx、ONNX Runtime、SpaceMiT EP、HTTP 库和
`llama-multi-asr` 还可能按需创建内部线程。因此 `ps -eLf` 或 `/proc/<pid>/task` 中的
Linux task 总数会高于上述数量，不能把 task 总数等同于项目配置的推理线程数。

## Pipeline 顺序

### 1. 主进程初始化

Headless 模式进入 `asyncio.run(...)`；Web 模式由 uvicorn 提供 asyncio event loop。
应用创建 `AppConfig`、`FrontendEventBus`、`ProcessPipelineManager` 和 `RunRecorder`，
并建立唯一的 `process-<timestamp>` run 目录。

Web 模式的 `FrontendEventBus.broadcast_loop()` 运行在主 event loop 中，以 asyncio task
向 WebSocket 客户端广播事件，不额外占用一个固定 OS 线程。

### 2. 启动共享 Qwen 服务

真实模式先由主进程生成共享 Qwen 配置并启动唯一的 `llama-multi-asr`。固定关键参数为：

- q40 文本模型；
- encoder 线程数 2，亲和配置 `8;9`；
- decoder cores `10,11,12,13,14,15`，线程数 6；
- `--n-parallel 4`；
- `--n-ctx 8192`。

主进程必须等待健康检查成功，才会创建 lane 子进程。服务启动失败时不会启动任何 lane。
运行期间 watchdog 周期检查共享服务；若服务异常退出，主进程记录 fatal，并向所有存活
lane 发送停止命令。Fake 模式不启动共享服务。

### 3. 并行加载 lane 与 Zipformer

共享服务就绪后，主进程并行 spawn `N` 个 lane。每个 lane 独立执行：

1. 在长期存活的 Zipformer 消费线程中创建本路 recognizer。
2. 在 lane event loop 中创建指向共享地址的 `Qwen3AsrClient`。
3. 通过 IPC 向主进程发送 `lane_model_loaded`，等待预热命令。

各路 Zipformer 可并行加载，不等待其他 lane。SpaceMiT EP session 的创建和后续 encoder
推理始终位于同一个消费线程。任一路加载失败会触发全局 fatal 和有界停止。

每路 recognizer 创建完成后先发送 `lane_model_loaded`。主进程等所有 lane 完成加载，
再同时发送固定 WAV 的 `warmup` 命令。预热 action 与正式 consumer 在同一长期
Zipformer 线程串行执行；每路 reset 成功后发送 `lane_loaded`。四路全部预热完成前，
主进程保持 `loading`，不会启动任何正式音频 producer。固定预热输入与用户选择的默认
文件、显式 WAV 或麦克风输入无关，也不写入正式识别结果和性能指标。

### 4. 开始识别

全部 lane 加载完成后，主进程向每路发送 `start`。lane 创建 `LanePipeline`、音频源和
event-loop lag 采样任务，启动音频生产线程，并让已有的 Zipformer 消费线程开始取帧。

IPC 只传递 JSON 可序列化的命令、ASR 事件、生命周期状态、指标摘要和文件路径。

### 5. 音频生产

每路音频生产线程独立读取音频，按 100 ms 一帧写入本路
`OnlineFrameQueue(maxsize=10)`。默认文件和显式 WAV 输入由该线程读取；麦克风输入由
PortAudio/sounddevice 回调线程采集，回调仅把 16 kHz 单声道 float32 block 安全投递到本路
音频生产线程内部的 asyncio loop，再由该线程消费并写入队列。该线程不运行 Zipformer 或 Qwen，因此
推理耗时不会直接阻塞下一次音频读取。

消费者落后但队列未满时，帧按 FIFO 保留并处理；队列满后才以最新普通帧替换最早的
尚未消费普通帧。`dropped_frames` 仅统计这类替换或拒绝的帧。`backlogged_frames` 统计
成功入队时已有未处理帧的次数，表示 Zipformer 消费落后但该音频仍会被处理。finish
帧必须送达：它可替换普通帧，入队后不再接受后续普通帧。

### 6. Zipformer 消费

每路 Zipformer 消费线程阻塞取帧，依次完成：

1. 把普通帧写入 `SegmentCache`。
2. 同步调用 sherpa-onnx Zipformer。
3. 记录在线帧延迟。
4. 文本变化时把 `partial` 事件交回 lane event loop。
5. endpoint 时发布 `pending`，记录 Zipformer model output，并复制本句 PCM。
6. 线程安全地在 lane event loop 中提交 Qwen 请求，然后推进 `sentence_id`。

Zipformer 计算不会运行在音频生产线程，也不会运行在主进程。

### 7. 共享 Qwen 请求

Qwen 请求由各 lane event loop 中的协程直接发往同一共享 HTTP 服务，并由协程接收响应。
每路维护一个 offline tail，保证同一路句子严格串行；不同 lane 可并发请求，共享服务最多
按 `--n-parallel 4` 处理。

响应返回后，lane 记录文本、音频时长、RTF 及 `queue_ms`、`encode_ms`、`prefill_ms`、
`decode_ms`、`total_ms`，写入本路 Qwen model output，并向主进程发送 `final`。每个 lane
还独立采样 event-loop lag，用于发现协程调度拥塞。

### 8. 主进程汇总事件与指标

lane 把 `partial`、`pending`、`final`、`fatal` 和 `lane_finished` 转为 JSON IPC 消息。
主进程的每路 reader 校验 lane 身份后发布到统一 `FrontendEventBus`：

- `partial` 为可替换事件，同一句只保留最新 revision；
- `pending` 表示在线切句完成、离线结果待返回；
- `final` 以更高 revision 替换对应 `pending`；
- `fatal` 和 `lane_finished` 驱动全局生命周期。

在线延迟、offline RTF、event-loop lag、丢帧和共享服务 timing 都在 lane 内统计。lane 只把
摘要发送给主进程，主进程按 count 和 total 加权合并，不接收逐请求原始数据。

### 9. 停止顺序

finish 帧到达后，lane 重置在线 stream，等待本路 offline tail 完成，发送最终指标和
`lane_finished`。主进程等待所有 lane 结束后进入 `STOPPED`。

卸载时主进程先取消共享服务 watchdog，再向 lane 发送停止命令并有界等待；超时进程先
terminate，必要时再 kill。只有确认全部 lane 已退出，主进程才停止唯一的共享 Qwen
服务。这样不会在仍有 lane 等待 Qwen 响应时提前回收服务。

## 文件归属

每次运行使用一个共享 run 目录，写入责任固定：

- lane 写 `results/laneX.txt`，保存该路最终完整识别结果；
- lane 写 `model_outputs/laneX.jsonl`，每条严格只含模型、音频时长、RTF、路 ID、句 ID
  和识别结果六个字段；
- 主进程写 `config.json`；
- 主进程合并各 lane 指标摘要并写 `metrics.json`。

`metrics.json` 使用 `metrics_schema_version = 3`，包含全局及逐 lane 的共享 Qwen timing
摘要。主进程不会重写 lane-owned 结果文件。

### verbose 详细指标记录

`--verbose` 开启且该路输入未循环（`loop_audio=False`）时，每个 lane 额外流式写入
`metrics_detail/{frame_latency,event_loop_lag,rtf,shared_timing}/laneX.jsonl`，
每行一条原始记录，字段含 `t_ms`（该路 run 开始到本次事件的毫秒偏移）及对应指标的
原始值（`rtf`/`shared_timing` 额外带 `sentence_id`，可与 `model_outputs/laneX.jsonl`
关联）。普通模式或循环识别时不产生这些文件，`metrics.json` 的聚合值不受影响。
