# 独立多麦克风调试脚本设计

## 目标

新增 `scripts/debug_microphones.py`，从识别服务中剥离麦克风采集能力，直接验证
PortAudio、PipeWire/ALSA 和 USB 麦克风在多路并行时是否持续输出有效 PCM。脚本不导入
`multi_stream_asr`，不启动模型、识别 pipeline 或 Web 服务。

## 使用方式

脚本使用项目 Python 3.13 虚拟环境中已有的 `sounddevice` 和 `numpy`。

未传 `--devices` 时，脚本列出所有具有输入声道的 PortAudio 设备，并提示用户输入一个或
多个逗号分隔的设备编号。传入 `--devices 22,24` 时跳过交互，直接打开指定设备。

可选参数：

- `--devices`：逗号分隔的 PortAudio 设备编号。
- `--sample-rate`：采样率，默认 `16000`。
- `--frame-ms`：回调块长度，默认 `100` 毫秒。
- `--stall-seconds`：无回调告警阈值，默认 `2.0` 秒。

设备编号必须存在、具有输入声道，并支持请求的采样格式。参数或设备无效时，脚本输出具体
错误并以非零状态退出。

## 采集模型

脚本在单个 Python 进程内为每个选中设备创建一个 `sounddevice.InputStream`。所有 stream
使用相同采样率、单声道 `float32` 和固定 blocksize。这与当前项目的
`MicrophoneSource` 输入参数一致，同时排除识别和多进程生命周期对采集的影响。

每个 PortAudio 回调只完成轻量统计并更新该设备的线程安全状态：

- 回调块数和最后回调时间；
- 当前块 RMS、峰值；
- 运行期间最大峰值；
- 最近一次 PortAudio callback status。

回调不执行终端输出，避免打印阻塞音频线程。主线程定期读取快照并刷新终端。

## 输出

每路显示设备编号、设备名、当前 RMS dBFS、当前 peak dBFS、线性 PCM peak、回调块数、
距最后回调的时间和状态。dBFS 以 `float32` 满幅 `1.0` 为 0 dBFS，零输入显示为
`-inf`。

交互终端使用 ANSI 原地刷新形成稳定表格；输出被重定向时逐次打印带时间戳的快照，便于
保存日志。`Ctrl+C` 正常关闭所有 stream 并返回 0。

## 异常处理

- 任一设备打开失败：关闭此前已经打开的 stream，打印设备编号、名称和原始异常，返回非零。
- callback status 非空：立即保留并显示，不在回调线程抛异常。
- stream 启动后超过 `--stall-seconds` 没有回调：持续显示 `STALLED` 和无回调时长，帮助识别
  USB 断开后 PortAudio 静默停止的问题；其他设备继续采集。
- 终端退出和异常路径均保证 stop/close 已打开的 stream。

## 测试

自动化测试通过 fake `sounddevice` 验证：

- 设备枚举和交互式多选解析；
- 命令行设备列表解析与非法编号拒绝；
- dBFS 对满幅、已知幅值和零输入的计算；
- 多个 stream 使用一致的 16 kHz、单声道、100ms 参数；
- 多路回调状态彼此隔离；
- 超时设备标记为 `STALLED`；
- 部分打开失败时关闭已打开 stream。

开发板验收同时打开 HM5 和 SPV，至少运行 20 秒，观察每路回调数持续增长，并在说话时确认
dBFS 明显上升。验收仅运行此脚本，不启动 ASR 服务。
