# 前端配置与麦克风采集设计

日期：2026-07-12

## 目标

完成现有前端未实现的配置、控制状态、独立结果滚动和麦克风输入能力，使 Web 模式能够在
未加载状态配置每路输入及离线模型，并在加载后以冻结的配置运行多路 ASR。

## 范围

- 顶部控制状态机、加载进度与前端错误展示。
- 全局路数、离线模型与模型目录配置。
- 每路默认文件、本地 WAV 或麦克风输入配置。
- 麦克风设备枚举、自动分配与 16 kHz/100 ms 采集。
- 0.6B q40 与 1.7B q4km 的共享 Qwen 服务配置。
- 固定 lane 网格、每路独立结果滚动与底部跟随。
- 配置/API、音频和前端逻辑测试，以及板端短音频验证。

不改变 Qwen prompt、核心/线程分配、Zipformer 配置、100 ms 生产语义、
`OnlineFrameQueue` 容量或所有 lane finish 后的自动停止规则。

## 配置模型

前端维护未应用的配置草稿，仅在 `unloaded` 状态可编辑：

- 全局 `lanes`：2、4、6 或 8。
- 全局 `offline_model`：`qwen3-asr-0.6b-q40` 或 `qwen3-asr-1.7b-q4km`。
- 全局 `offline_model_dir`：所选模型目录的绝对路径。
- 每路输入：`default_files`、`local_wav` 或 `microphone`。
- `local_wav` 需要绝对 WAV 路径。
- `microphone` 可传设备索引；省略时服务端从可用输入设备中循环分配。

点击加载模型前，前端向只读预检接口提交模型选择与目录。预检接口返回所选模型的完整
必需文件列表和缺失路径。存在缺失项时，前端持久显示错误且不发送加载请求。加载接口仍在
服务端重新校验文件，避免直接 API 调用或检查后文件变化绕过校验。

加载成功后，服务端将配置转换为不可变运行配置；开始、停止和卸载期间不允许修改。必须
卸载后才能重新配置。

## 模型定义

`qwen3-asr-0.6b-q40` 必需：

- `Qwen3-ASR-0.6B-text-q40.gguf`
- `Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx`
- `Qwen3-ASR-0.6B-encoder-backend.dynq.onnx`

它保留当前共享服务参数：`d_model=896`、`hidden_size=1024`、`output_dim=1024`。

`qwen3-asr-1.7b-q4km` 必需：

- `Qwen3-ASR-1.7B-text-q4km.gguf`
- `Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx`
- `Qwen3-ASR-1.7B-encoder-backend.dynq.onnx`

它使用独立参数：`d_model=1024`、`hidden_size=2048`、`output_dim=2048`。共享服务启动命令
和写入的 SMT config 必须使用所选模型的文件名与参数，不能混用 0.6B 值。

## Web 交互

控制状态：

| 状态 | 可用操作 |
| --- | --- |
| `unloaded` | 编辑配置、加载模型 |
| `loading`、`unloading` | 无操作，显示进行中 |
| `loaded`、`stopped` | 开始识别、卸载模型 |
| `starting`、`running` | 停止识别 |
| `stopping` | 无操作，等待结束 |
| `fatal` | 卸载模型 |

前端通过页面内持久状态/错误区域展示请求失败，所有控制请求由 async handler 捕获，不允许
未处理 Promise。开始识别只在后端开始成功后清空结果；失败时保留原结果。

未加载时显示全局配置和每路输入配置。输入类型为默认文件时隐藏路径与设备控件；本地 WAV
时显示路径；麦克风时显示设备选择。每路卡片在运行期间仅展示识别结果和指标。

页面为固定视口高度。lane 网格固定为：2 路 `2x1`、4 路 `2x2`、6 路 `3x2`、8 路 `4x2`。
每张卡片只有结果区域可滚动。新增结果前若该区域处于底部，渲染后保持底部；用户上滚查看
历史时不强制改变滚动位置。

`stream_id + sentence_id + revision` 事件语义不变：partial/pending 灰色，final 黑色，
final 覆盖同句 pending，fatal 为红色。

## 麦克风采集

服务端提供输入设备枚举接口。`MicrophoneSource` 用 `sounddevice.InputStream` 读取指定或
自动分配的输入设备，转换为 mono float32 16 kHz，并以现有 `frame_duration_ms` 生成连续
序号的 `AudioFrame`。每个正常/主动停止的输入流只发送一个 finish 帧。设备初始化、回调或
采集错误按 lane fatal 处理。

默认文件输入保持当前 `~/Music/wav16k` 循环分配规则；本地 WAV 继续使用现有格式校验。

## 验证

- 配置/API 测试覆盖模型选择、文件预检、缺失文件、服务端二次校验和每路输入解析。
- 音频测试覆盖设备枚举、自动分配、单声道/16 kHz 转换、100 ms 帧、连续序号、唯一 finish
  帧和错误传播。
- 前端测试覆盖控制禁用、条件配置、错误显示、固定网格、独立滚动和底部跟随。
- 开发板分别以本地 0.6B q40 和 1.7B q4km 发出短 WAV 请求，确认模型路径、非空结果、
  正常关闭的共享 Qwen 服务。

不要求重跑多轮性能基准。
