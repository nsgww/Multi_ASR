# Zipformer 加载阶段预热设计

## 目标

在模型加载完成、正式音频采集开始之前，让每一路真实 Zipformer recognizer 使用固定内置音频完成一次预热推理，把首次推理的模型权重访问、kernel 初始化和 cache 预热开销移动到 `loading` 阶段。

本次只实现 Zipformer 预热。共享 Qwen3-ASR 继续使用 `llama-multi-asr --warmup`，不增加手动 Qwen 请求；如果 Zipformer 预热后真实运行仍存在明显冷启动尖峰，再单独设计 Qwen 手动预热。

## 已知基线

现有四路真实运行中，Zipformer 平均在线帧延迟约 41 ms，但最大值稳定出现 610–630 ms，每路通常丢 4–6 帧。独立对照实验中，即使 Qwen 空闲，四路首次推理也会同步出现约 0.54–0.62 秒尖峰；完成一次推理后，Qwen 空闲阶段最大值恢复到约 93–112 ms。

`OnlineFrameQueue(maxsize=1)` 在 consumer 落后时只保留最新帧，因此一次约 600 ms 的首次推理阻塞会直接造成约 4–5 个丢帧。预热的首要目标是让这次阻塞发生在正式 producer 启动之前。

## 范围

包含：

- 将 `/home/liangff/Music/wavs/junzhao_nihao.wav` 原样复制进项目并重命名为 `zipformer_warmup.wav`。
- 四路 Zipformer 加载完成后并行预热。
- 预热在每路长期存活的 Zipformer consumer 线程中执行。
- 主进程增加模型加载与 Zipformer 预热两个可查询子阶段。
- 前端显示真实的全局预热阶段，不显示逐路进度。
- 自动化测试、fake headless 回归和开发板三轮真实验收。

不包含：

- Qwen3-ASR 手动 HTTP 预热。
- 修改 Zipformer 或 Qwen 的核分配、线程数和 provider 配置。
- 修改 `OnlineFrameQueue(maxsize=1)`。
- 修改 endpoint 默认规则、正式输入选择或结果落盘格式。
- 在正式 `metrics.json` 中记录预热帧或预热延迟。

## 固定预热资源

目标文件为：

`src/multi_stream_asr/assets/zipformer_warmup.wav`

文件属性固定为：

- 内容：你好吗？
- SHA-256：`d29d16456d9c7662135fbd696605391745ed56ce47eee3385a37c4b120a9691f`
- 采样率：16000 Hz
- 声道：1
- 采样宽度：16 bit signed PCM
- 样本数：22080
- 时长：约 1.38 秒

该文件作为 `multi_stream_asr` Python package data 发布。运行时通过 package resource 定位，不依赖进程当前工作目录。预热路径由主进程解析并随 `warmup` IPC 命令传给 lane。

`config.py` 暴露固定常量：

```python
ZIPFORMER_WARMUP_AUDIO_PATH = (
    Path(__file__).resolve().parent / "assets" / "zipformer_warmup.wav"
)
```

预热固定使用该资源，与以下配置完全无关：

- `AppConfig.input_dir`
- 每路 `default_files`
- 每路 `local_wav_path`
- 每路 `microphone_device`
- 前端当前选择的输入模式

## 生命周期与 IPC

Pipeline 主状态保持现有枚举，不增加新的 `PipelineState`。预热期间主状态仍为 `loading`。

`ProcessPipelineManager` 增加只读 `load_phase`，取值为：

- `loading_models`：共享 Qwen 启动中，或 lane Zipformer recognizer 加载中。
- `warming_zipformer`：所有 lane recognizer 已加载，四路预热进行中。
- `None`：不处于模型加载子阶段。

加载顺序如下：

1. 主进程验证内置预热 WAV 存在且格式正确。
2. 主进程把 `load_phase` 设为 `loading_models`。
3. 主进程启动共享 Qwen，保留现有 `--warmup`。
4. 主进程并行启动全部 lane 进程。
5. 每个 lane 在长期存活的 Zipformer consumer 线程中创建真实 recognizer。
6. recognizer 创建完成后，lane 发送 lifecycle `lane_model_loaded`，然后等待命令。
7. 主进程等待所有 lane 都发送 `lane_model_loaded`。
8. 主进程把 `load_phase` 设为 `warming_zipformer`，同时向全部 lane 发送 `{"type": "warmup", "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH)}`。
9. 每个 lane 在其 Zipformer consumer 线程中执行预热。
10. 预热完成并 reset stream 后，lane 发送现有 lifecycle `lane_loaded`。
11. 主进程等待所有 lane 都发送 `lane_loaded`。
12. 主进程把 Pipeline 状态切换为 `loaded`，并把 `load_phase` 清为 `None`。

fake 模式保留相同的 `lane_model_loaded → warmup → lane_loaded` IPC 顺序，但不读取 WAV，也不调用真实模型。

## Consumer 线程执行模型

`OnlineConsumerWorker` 当前只允许 factory 之后提交一个长期 consumer。它需要扩展为两类串行命令：

- 短生命周期 action：在线程中执行并返回，执行后继续等待下一条命令。
- 长期 consumer：正式识别 consumer，只允许提交一次；执行结束后 worker 退出。

预热作为一个 action 执行，正式 `LanePipeline` consumer 仍是唯一长期 consumer。factory、预热 action 和正式 consumer 必须具有相同的 OS thread ID。任何时刻都只有一个命令访问 recognizer。

每路预热 action：

1. 使用 `LocalWavSource(realtime=False)` 读取固定预热 WAV。
2. 按项目既有 100 ms frame duration 生成普通 `AudioFrame`。
3. 对每个普通帧调用真实 `ZipformerRecognizer.accept()`。
4. 忽略全部识别文本，不发布 `partial`、`pending` 或 `final`。
5. 遇到 finish frame 后调用 `ZipformerRecognizer.reset()`。
6. reset 成功后才返回 action future，并发送 `lane_loaded`。

预热不创建 `LanePipeline`、`SegmentCache`、run 目录、结果 writer 或 model-output recorder，因此不会改变 `sentence_id`，也不会污染正式识别状态和文件。

## 前端状态

`GET /health` 增加 `load_phase` 字段。前端不新增 Pipeline 主状态，也不通过识别结果 WebSocket 发送 lifecycle 事件。

点击“加载模型”后的提示顺序为：

1. `Validating model files...`
2. `Loading models...`
3. `Warming up Zipformer...`
4. `Models loaded.`

由于 `POST /api/models/load` 会等待整个加载流程完成，浏览器在该请求 pending 期间每 250 ms 查询 `/health`，根据 `load_phase` 更新提示。轮询在加载成功、失败或请求取消时停止。

前端只显示一个全局预热阶段，不显示 `1/4`、`2/4` 或 lane 独立状态。必须等四路全部预热成功后才显示 `Models loaded.`。加载期间沿用现有控制禁用规则。

## 错误与清理

- 主进程在启动共享 Qwen 前验证内置 WAV，避免资源缺失时产生昂贵的部分加载。
- lane 仍会使用 `LocalWavSource` 验证收到的文件，防止 IPC 路径或文件内容异常。
- WAV 读取、帧处理、Zipformer `accept()`、`reset()` 或 worker action 任一步失败，都由该 lane 上报 fatal。
- 任一路预热失败即视为整体模型加载失败：停止全部 lane、结束 reader、停止共享 Qwen，并让 load API 返回错误。
- `load_phase` 在成功和所有失败清理路径中都必须恢复为 `None`。
- 等待所有 `lane_model_loaded` 和等待所有 `lane_loaded` 分别使用现有 lifecycle timeout，避免某一路加载或预热永久阻塞。
- unload 或 shutdown 与 load 的并发行为继续由现有 lifecycle lock 串行化。

## 自动化测试

需要覆盖：

- 内置 WAV 的 SHA-256、格式、样本数和 package data 配置。
- `OnlineConsumerWorker` 的 factory、action、正式 consumer 顺序和同线程约束。
- action 失败向调用方传播，且 worker 可以安全关闭。
- lane 在 warmup 前只发送 `lane_model_loaded`；全部固定音频帧被 accept 且 reset 完成后才发送 `lane_loaded`。
- lane 预热不使用用户输入配置，不创建 run artifacts，不发送 ASR 事件。
- 主进程等待所有 `lane_model_loaded` 后才向全部 lane 广播 warmup。
- 主进程等待所有 `lane_loaded` 后才进入 `loaded`。
- 任一路预热失败触发全局 fatal 和完整清理。
- fake 模式执行相同 IPC 屏障但不调用真实 WAV 或模型。
- `/health.load_phase`、250 ms 前端轮询、四段提示和加载期间按钮禁用。
- 完整 pytest 和四路 fake headless 回归。

## 开发板真实验收

开发板测试遵守项目的 Python 3.13 和本地 `.venv313` 规则。代码通过既有 Git/rsync 流程同步，预热 WAV 必须来自项目自身，不使用板上外部路径。

连续执行 3 次四路真实 headless 测试，正式输入保持 `~/Music/wav16k`。每次必须满足：

- 四路最终结果非空。
- Zipformer 与 Qwen sentence ID 对齐。
- lane、共享 Qwen 和主进程正常结束，无残留服务。
- 每一路 `dropped_frames < 10`。
- 全局 `online_frame_latency.max_ms < 300`。
- 不再出现现有约 610–630 ms 的正式运行首次冷启动尖峰。

offline RTF 继续记录并与当前结果比较，但本次不设置新的 offline RTF 优化门槛。若三轮中仍存在明显冷启动尖峰，先分析尖峰发生时间与 Qwen 请求重叠关系，再为 Qwen 手动预热创建独立设计，不扩展本次实现。
