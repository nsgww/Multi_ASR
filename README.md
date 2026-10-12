# Multi-Stream ASR 实时语音转写服务

运行在 SpacemiT K3/A100（RISC-V）开发板上的多路实时语音转写 Web 服务。支持浏览器麦克风、UDP 组播、开发板本机麦克风三种音频输入，提供 Web 界面、WebSocket 实时协议和 HTTP 管理接口。

核心入口：`src/multi_stream_asr/main.py`，服务端口 `8000`。

---

## 1. 功能特性

- **多路并发转写**：支持 2/4/6/8 路 lane 同时识别，每路独立会话，模型资源共享
- **三种音频输入**：
  - 浏览器本机麦克风（`/ws/stream`）
  - UDP 组播输入（`/ws/multicast`，默认 `224.2.1.168:6020`）
  - 开发板本机麦克风（`/ws/local_mic`，自动扫描有效输入设备）
- **识别引擎**：K3/RISC-V 专用 `llama-multi-asr` runtime（Qwen3-ASR 0.6B/1.7B）+ SenseVoice，在线帧级延迟约 50ms，离线 RTF ≤ 0.4
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

`/ws/multicast` 连接时可通过 query 指定组播地址：`ws://<板子IP>:8000/ws/multicast?group=224.2.1.168&port=6020`。组播格式：UDP、48 kHz PCM16 单声道、每包 1280 字节（前 640 字节有效）。

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

默认模型目录 `models/asr`（可用 `--models-dir` 或环境变量调整）：

- **Zipformer**（在线识别，多路共享）
- **Qwen3-ASR 0.6B**（`qwen3-asr-0.6b-q40`）：`Qwen3-ASR-0.6B-text-q40.gguf` + 两个 0.6B dynq encoder ONNX
- **Qwen3-ASR 1.7B**（`qwen3-asr-1.7b-q4km`）：`Qwen3-ASR-1.7B-text-q4km.gguf` + 两个 1.7B dynq encoder ONNX
- **SenseVoice**（离线/文件转写）：量化 ONNX 模型

`setup.sh` 自动下载并校验；Web 界面 Load 前会检查全部必需模型文件，缺失时列出缺失路径。

---

## 7. systemd 部署

```bash
sudo cp deploy/multi-stream-asr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now multi-stream-asr
```

服务配置：`WorkingDirectory=/opt/multi-stream-asr`，venv 位于 `/opt/envs/multi-stream-asr`，日志 `journalctl -u multi-stream-asr -f`。

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
