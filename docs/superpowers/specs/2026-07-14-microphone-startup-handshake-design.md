# 麦克风输入串行就绪握手设计

## 问题

多路 lane 已并行加载并预热 Zipformer 后，同时初始化两个 PulseAudio `InputStream`，会在
libpulse 中触发 `pa_operation_get_state()` 断言并导致 lane 以 `SIGABRT` 退出。设备 15/16
可以被独立调试脚本同时采集，单麦克风服务也正常；仅模型 lane 并发打开输入时稳定复现。
临时将两个麦克风启动错开 1 秒后故障消失，确认竞态位于输入初始化窗口。

## 方案

- 模型 lane 进程、Zipformer 加载和 Zipformer 预热继续并行，不改变模型生命周期。
- `LanePipeline` 在音频生产线程中完成 `source.prepare()` 后，通过线程安全回调通知 lane
  runtime。
- `lane_started` 改为输入真正就绪后发送，不再在 pipeline task 创建前提前发送。
- 主进程按 lane 顺序启动麦克风输入；每发送一路麦克风 `start` 后等待该路
  `lane_started`，再发送下一路麦克风命令。
- 本地文件 lane 不参与麦克风串行门控，可先统一发送并并行启动。
- 输入准备失败时 pipeline 原有异常链路上报 fatal，主进程等待立即失败，不启动后续麦克风。
- 不使用固定 sleep、重试或设备特判。

## 验证

- 单元测试证明第二路麦克风必须等待第一路 `lane_started`，文件 lane 无需等待。
- 单元测试证明 `lane_started` 只在 `source.prepare()` 成功后出现，prepare 失败不会误报就绪。
- 完整测试通过后，在 K3 使用设备 15/16 连续执行多轮开始、停止识别，确认无 PulseAudio
  abort、两路有音频帧、服务每轮回到 `loaded`。
