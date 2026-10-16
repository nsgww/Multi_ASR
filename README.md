# Multi-Stream ASR 实时语音转写服务

运行在 SpacemiT K3/A100（RISC-V）开发板上的多路实时语音转写 Web 服务。支持浏览器麦克风、UDP 组播、开发板本机麦克风三种音频输入，提供 Web 界面、WebSocket 实时协议和 HTTP 管理接口。

核心入口：`src/multi_stream_asr/main.py`，服务端口 `8000`。

---

## 1. 功能特性

- **三种音频输入**：
  - 浏览器本机麦克风（`/ws/stream`）
  - UDP 组播输入（`/ws/multicast`，默认 `224.2.1.168:6020`）
  - 开发板本机麦克风（`/ws/local_mic`，自动扫描有效输入设备）
- **识别引擎（在线，二选一，EdgeModelHub 切换）**：
  - **Zipformer**（sherpa-onnx 流式，自带端点检测断句；生产默认 zipformer-pro-q 进迭定制量化加速版）
  - **Paraformer**（funasr_onnx 流式 + Silero VAD 断句）
  - 注意：Paraformer 必须每路独立模型实例，不可多路共用（前端缓存会串扰，表现为高频重复词）
- **离线校正**：Qwen3-ASR 0.6B/1.7B（dynq 量化 encoder + GGUF text，进迭定制 runtime），对在线结果做二次校正
- **文件转写**：SenseVoice（量化 ONNX）
- **VAD**：Silero VAD v5（Paraformer 链路断句 + 唤醒门控）
- **模型管理**：全部模型由 EdgeModelHub（:8100）托管，页面可下载/切换/删除版本，切换后自动重启生效
- **热词增强**：自定义热词表，提升专有名词识别率（`config/hotwords.txt`）
- **敏感词过滤**：识别结果自动过滤敏感词（`config/sensitive_words.txt`）
- **唤醒词检测**：基于 sherpa-onnx KWS 的关键词命中提示（`config/keywords.txt`）
- **文件转写**：通过 HTTP 接口上传音频文件离线转写
- **识别纪要**：每路转写可生成摘要（summary），支持重新生成
- **模型预热**：Zipformer 加载完成后并行预热所有识别器，降低首句延迟

---

## 2. 快速开始

### 2.1 一键安装

```bash
cd /opt/multi-stream-asr
scripts/setup.sh
```

自动安装 Python 3.13、PortAudio 及硬件依赖，并下载 Zipformer 和 Qwen3-ASR 模型到 `models/asr`（校验 MD5，断网重跑自动续传）。K3/RISC-V 的 `llama-multi-asr` runtime 已内置在 `multi-asr-runtime/`，无需额外安装。

常用参数：`--skip-apt`（系统包已装时跳过）、`--skip-python`（只装模型）、`--models-dir PATH`（自定义模型目录）。

### 2.2 手动安装

```bash
python3.13 -m venv .venv313
. .venv313/bin/activate
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export SPACEMIT_PIP_INDEX_URL="https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple"
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pip install sounddevice
python -m pip install --no-cache-dir --only-binary=:all: --no-deps sherpa-onnx \
  --index-url "$SPACEMIT_PIP_INDEX_URL"
```

### 2.3 启动

```bash
python -m multi_stream_asr.main --host 0.0.0.0 --port 8000
```

打开 `http://<板子IP>:8000` 使用 Web 界面：选择 lane 数（2/4/6/8）→ 选择 Qwen 模型及目录 → Load → 为每路选择 WAV 文件或麦克风 → Start。

无界面验证：

```bash
pytest -v
python -m multi_stream_asr.main --headless --fake --lanes 4   # 模拟数据
python -m multi_stream_asr.main --headless --lanes 4          # 真实模型
```

---

## 3. WebSocket 接口

三种输入路径协议一致：连接后发 `start` 开始，支持 `pause`/`resume`/`finish`/`new_transcription` 控制，`done` 同 `finish`。

| 路径 | 输入源 | 说明 |
|------|--------|------|
| `/ws/stream` | 客户端推流 | 前端持续推送 16kHz PCM16 单声道二进制帧 |
| `/ws/multicast` | 组播音频 | 服务端从组播组收音频，前端无需推流 |
| `/ws/local_mic` | 板载麦克风 | 服务端自动扫描并打开有效输入设备，前端无需推流 |

### 3.1 控制消息（JSON 文本帧）

| 消息 | 作用 | 服务端响应 |
|------|------|-----------|
| `{"cmd":"start"}` | 开始转写 | `status: ready` → `status: started` |
| `{"cmd":"pause"}` | 暂停转写 | `status: paused` |
| `{"cmd":"resume"}` | 继续转写 | `status: resumed` |
| `{"cmd":"finish"}` / `{"cmd":"done"}` | 结束当前段 | `status: ready_for_new_transcription` |
| `{"cmd":"new_transcription"}` | 新建转写 | `status: ready` → `status: started` |

### 3.2 下行消息

- `status`：状态变更（ready / started / paused / resumed / ready_for_new_transcription）
- 转写文本：按句推送识别结果（含热词替换、敏感词过滤后的文本）
- 摘要：段结束后生成 summary，可重新生成

### 3.3 组播输入（/ws/multicast）

服务端主动加入组播组收音频，前端只收转写结果、无需推流。默认组播 `224.2.1.168:6020`。

**音频格式**：UDP 组播、48 kHz PCM16 单声道、每包 1280 字节（前 640 字节有效音频，320 采样）。

**URL 查询参数**（均可选）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `group` | `224.2.1.168` | 组播 IP |
| `port` | `6020` | 组播端口 |
| `local_ip` | `0.0.0.0` | 本地绑定 IP（多网卡时指定） |
| `packet_bytes` | `1280` | 每个 UDP 包字节数 |
| `use_full_packet` | `false` | 是否使用整包音频 |
| `heartbeat_interval` | `5.0` | 心跳发送间隔（秒） |
| `wakeup` | `true` | 是否启用唤醒词门控 |

示例：`ws://<板子IP>:8000/ws/multicast?group=224.2.1.168&port=6020`

**运行时切换组播地址**（不断开连接）：

```json
{"type": "set_multicast_group", "group": "224.2.1.168"}
```

限制：**必须先 `finish` 结束当前转写**才能切组，否则返回 `cannot change group while job running; finish first`。

完整协议见 `docs/组播地址修改接口说明.md`。

### 3.4 组播重采样（48 kHz → 16 kHz）

组播输入是 48 kHz，ASR 需要 16 kHz，重采样由 **libspeexdsp**（SpeexDSP 重采样器，quality=5）完成，通过 `ctypes` 动态加载，不依赖 Python 包。

**安装方式一：apt（推荐，板端镜像已预装）**：

```bash
sudo apt install -y libspeexdsp1
```

**安装方式二：源码编译**（无 apt 源或需要定制时）：

```bash
sudo apt update
sudo apt install -y build-essential autoconf automake libtool pkg-config git
git clone https://github.com/xiph/speexdsp.git
cd speexdsp
./autogen.sh
./configure --prefix=/usr/local     # 不要加 --disable-shared：代码用 ctypes 动态加载 .so
make -j$(nproc)
sudo make install
sudo ldconfig
```

**库查找顺序**：`speexdsp_lib_path` 参数 → 环境变量 `LIBSPEEXDSP_PATH` → 常见系统路径（含 `/usr/lib/riscv64-linux-gnu/libspeexdsp.so.1`）。都找不到时组播转写报 `SpeexResamplerError`，设置环境变量指向实际路径即可：

```bash
Environment=LIBSPEEXDSP_PATH=/usr/local/lib/libspeexdsp.so.1
```

注意：`scripts/setup.sh` 不会自动安装 libspeexdsp，新部署需单独执行上面的 apt 命令。

---

## 4. HTTP 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/audio/devices` | GET | 列出可用音频输入设备 |
| `/v1/audio/transcriptions` | POST | 上传音频文件离线转写 |
| `/v1/config/wakeup` | GET/POST | 查询/修改唤醒词配置 |
| `/v1/config/words` | GET/POST | 查询/修改热词、敏感词表 |
| `/v1/config/words/upload` | POST | 上传词表文件 |
| `/v1/transcripts/{filename}` | GET | 获取转写结果文本/摘要 |

---

## 5. 词表配置

| 文件 | 用途 | 格式 |
|------|------|------|
| `config/hotwords.txt` | 热词（提升识别率） | 一行一词，`#` 注释 |
| `config/sensitive_words.txt` | 敏感词（结果过滤） | 一行一词，`#` 注释 |
| `config/keywords.txt` | 唤醒词（sherpa 格式） | 每行 `拼音 @中文词`，由程序自动生成 |

热词/敏感词可经 `/v1/config/words` 接口动态更新，无需重启；修改后即时生效。

---

## 6. 模型准备

板端生产环境的模型全部由 **EdgeModelHub**（`http://<板子IP>:8100`）托管，存储在 `/var/lib/models/hf/scenarios/<角色>/current`：

| 角色 | 当前模型 | Hub 路径 |
|------|---------|---------|
| `streaming-asr`（在线识别） | zipformer-pro-q / paraformer-funasr | `/var/lib/models/hf/scenarios/streaming-asr/current` |
| `asr-correction`（离线校正） | Qwen3-ASR-0.6B（dynq+q40） | `/var/lib/models/hf/scenarios/asr-correction/current` |
| `vad`（断句） | silero-vad-v5 | `/var/lib/models/hf/scenarios/vad/current` |
| `offline-asr`（文件转写） | SenseVoice | `/var/lib/models/hf/scenarios/offline-asr/current` |

代码里的默认路径（`/var/lib/sm-sdk/models/...`）通过**符号链接**桥接到 Hub 的 current，例如：

```
/var/lib/sm-sdk/models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
  → /var/lib/models/hf/scenarios/streaming-asr/current
/var/lib/sm-sdk/models/vad/silero-vad/v5 → /var/lib/models/hf/scenarios/vad/current
```

所以在 Hub 页面切换模型版本后重启服务即生效，代码无需改动。

> 全新部署也可用 `scripts/setup.sh` 下载模型到 `models/asr`（校验 MD5，断点续传），再用 `--models-dir` 指定；与 Hub 方式二选一。

### 6.1 VAD 断句参数（Paraformer 链路）

| 环境变量 | 说明 | 代码默认 | 板上当前 |
|---------|------|---------|---------|
| `VAD_THRESHOLD` | 语音激活概率阈值 | `0.2` | `0.2` |
| `VAD_LOUD_TRIGGER` | 连续超阈值窗口数（判定开口） | `3` | `3` |
| `VAD_MIN_SILENCE` | 静音时长（秒），超过则断句 | `0.5` | `0.5` |
| `VAD_MIN_SPEECH` | 最短语音（秒），短于则丢弃 | `0.3` | `1.5`（systemd 设置） |
| `VAD_PRE_BUFFER` | 开口前预缓冲（秒） | `0.3` | `0.3` |
| `VAD_MAX_UTT` | 单句强制截断上限（秒） | `25` | `25` |

Zipformer 链路走 sherpa-onnx 自带端点检测（rule1/rule2/rule3），不受以上参数影响。

---

## 7. systemd 部署

```bash
sudo cp deploy/multi-stream-asr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now multi-stream-asr
```

服务配置：`WorkingDirectory=/opt/multi-stream-asr`，venv 位于 `/opt/envs/multi-stream-asr`，日志 `journalctl -u multi-stream-asr -f`。

板上另有 drop-in `/etc/systemd/system/multi-stream-asr.service.d/vad.conf` 设置 `VAD_MIN_SPEECH=1.5`（最短语音限制）。改 VAD 参数优先加在这里，`daemon-reload` + 重启生效，不用改代码。

> 服务重启约需 2.5 分钟（模型加载 + 预热），期间接口不可用。

---

## 8. 目录结构

```
multi-stream-asr/
├── src/multi_stream_asr/
│   ├── main.py            # 服务入口
│   ├── asr/               # 识别引擎（online/offline/vad/kws/summary）
│   ├── audio/             # 音频输入（设备/组播）
│   ├── web/               # Web UI 与 WS/HTTP 接口
│   ├── pipeline/          # 转写流水线
│   ├── session/           # 会话管理
│   ├── buffer/            # 音频缓冲
│   └── headless/          # 无界面模式
├── multi-asr-runtime/     # K3/RISC-V llama-multi-asr 运行库
├── config/                # 热词/敏感词/唤醒词
├── deploy/                # systemd 服务文件
├── scripts/               # setup.sh、组播收发测试、评测脚本
├── tests/                 # 测试
├── docs/                  # 设计与调优文档
├── output/                # 运行产物（会话、日志、摘要、转写）
└── models/asr/            # 模型（setup.sh 下载）
```

---

## 9. 性能指标

- 在线帧级延迟 `online_frame_latency.average_ms` ≈ 50 ms
- 离线 RTF `offline_rtf.average` ≤ 0.4
- 运行产物：headless 模式写 `runs/headless-*`，Web 模式写 `runs/web-*`；最终转写 `results/lane<N>.txt`，诊断 `model_outputs/lane<N>.jsonl`，Unload/关机时自动停止 App 拉起的 `llama-server` 进程

---

## 10. 相关文档

- `docs/thread-model.md`：asyncio 任务、Python 线程、sherpa-onnx 与 llama-server 线程模型
- `docs/web-ui-tutorial.html`：Web 界面交互式教程（无需后端，浏览器直接打开）
- `docs/reports/`：性能与问题排查报告
- `组播地址修改接口说明.md`：组播地址动态切换接口
