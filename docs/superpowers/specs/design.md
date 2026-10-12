# 多路ASR

本项目提供2/4/6/8路并发语音识别
并将各路实时识别结果渲染到前端
语音识别包括流式识别和根据流式识别模型分段的整段纠正识别
每一路相互独立

## 音频输入（生产者）

使用16KHZ
各路音频输入可选麦克风/本地音频
以100ms（默认，可配参数）分帧输入到pipeline中
多路输入处理线程应不受后续处理阻塞，不断发送音频帧。
多路输入应全部初始化（麦克风准备、本地文件加载等）完成后等待100ms从各buffer处读取数据、发送音频帧。
不同类型音频输入抽象成统一的接口，对ASR Pipeline透明
音频流结束（文件读取完毕/麦克风输入受到结束消息）时发送一个finish音频帧
音频帧需登记序号

## 流式识别（消费者）

维护一个1.5*rule3_min_utterance_length的队列缓存音频帧，用于发送整段音频识别
维护一个slot=1消费音频帧，用于流式识别
使用sherpa onnx
当识别结果更新时，发送ws前端更新消息
当is_endpoint时，将缓存的音频发送到整段ASR进行异步处理，将流式识别的结果放入prompt作为参考，然后清空缓存
收到finish帧时reset

## 整段识别

使用llama server
异步处理完请求后发送ws前端更新消息

## 模型配置

模型在前端手动加载、释放，或者在服务关闭时释放
避免llama server僵尸进程

### 流式识别

使用zipformer：~/Workspace/sm-sdk/models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
每路`sherpa-onnx`流式识别配置`num_threads=1`

### 整段识别

使用qwen3-asr 0.6B:~/Workspace/sm-sdk/models/asr/qwen3asr
每一路启动一个独立`llama-server`，端口从`8063`递增
每个`llama-server`绑定一组A100 CPU，模型进程、GGUF文本模型线程和SMT音频ONNX线程使用同一组核心

默认使用A100 CPU：`8,9,10,11,12,13,14,15`

按路数分配：

| 路数 | 核心分配 |
| --- | --- |
| 2 | `8,9,10,11` / `12,13,14,15` |
| 4 | `8,9` / `10,11` / `12,13` / `14,15` |
| 6 | `10` / `11` / `12` / `13` / `14` / `15` |
| 8 | `8` / `9` / `10` / `11` / `12` / `13` / `14` / `15` |

绑核规则：

- GGUF文本模型使用环境变量`SPACEMIT_PERFER_CORE_ID`，核心列表用逗号分隔，例如`8,9`。
- `llama-server -t`和`-tb`必须等于该路分配的核心数。
- SMT音频ONNX模型在临时`config.json`中配置`ep_config`。
- `SPACEMIT_EP_INTRA_THREAD_NUM`必须等于该路分配的核心数。
- `SPACEMIT_EP_INTER_THREAD_NUM`固定为`1`。
- `SPACEMIT_EP_INTRA_THREAD_AFFINITY`核心列表用分号分隔，例如`8;9`。
- 启动`llama-server`时必须指定`--smt-config-dir`到临时配置目录，不能直接改模型目录内的原始`config.json`。

启动前清理残留进程和TCM状态：

```bash
pkill -x llama-server || true
sleep 1
pgrep -x llama-server && pkill -9 -x llama-server || true
spacemit-tcm-smi -c
```

启动命令由Python按当前路数生成，等价于以下脚本：

```bash
BASE=/home/bianbu/Workspace/sm-sdk/models/asr/qwen3asr
MODEL=$BASE/Qwen3-ASR-0.6B-text-q40.gguf

# 4路示例。2/6/8路时只需替换SPECS中的端口和核心分配。
SPECS='8063:8,9:8;9 8064:10,11:10;11 8065:12,13:12;13 8066:14,15:14;15'

for SPEC in $SPECS; do
  IFS=: read -r PORT GGUF_CORES ONNX_CORES <<EOF
$SPEC
EOF
  THREADS=$(awk -F, '{print NF}' <<EOF
$GGUF_CORES
EOF
)
  CFG=/tmp/spacemit-qwen3asr-$PORT
  mkdir -p "$CFG"
  cat > "$CFG/config.json" <<EOF
{
  "architectures": ["Qwen3ASRForConditionalGeneration"],
  "audio_model": {
    "frontend_model_path": "$BASE/Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
    "backend_model_path": "$BASE/Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
    "num_mel_bins": 128,
    "sample_rate": 16000,
    "n_fft": 400,
    "window_len": 400,
    "hop_len": 160,
    "chunk_frames": 100,
    "chunk_tokens": 13,
    "d_model": 896,
    "hidden_size": 1024,
    "output_dim": 1024,
    "ep_config": {
      "SPACEMIT_EP_INTRA_THREAD_NUM": "$THREADS",
      "SPACEMIT_EP_INTER_THREAD_NUM": "1",
      "SPACEMIT_EP_INTRA_THREAD_AFFINITY": "$ONNX_CORES"
    },
    "special_tokens": {
      "audio_start_token": "<|audio_start|>",
      "audio_end_token": "<|audio_end|>",
      "audio_token": "<|audio_pad|>",
      "audio_start_token_id": 151669,
      "audio_end_token_id": 151670,
      "audio_token_id": 151676
    }
  },
  "text_model": {
    "model_path": "$MODEL",
    "context_size": 65536,
    "logical_batch_size": 2048,
    "predict_token": 256,
    "vocab_size": 151936,
    "hidden_size": 1024
  },
  "prompt": {
    "system_prefix_token": "<|im_start|>system",
    "user_prefix_token": "<|im_start|>user",
    "assistant_prefix_token": "<|im_start|>assistant",
    "message_suffix_token": "<|im_end|>",
    "asr_text_tag": "<asr_text>"
  }
}
EOF

  LOG=/tmp/spacemit-qwen-$PORT.log
  SPACEMIT_PERFER_CORE_ID=$GGUF_CORES \
    nohup llama-server -m "$MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      -t "$THREADS" -tb "$THREADS" -c 4096 \
      --media-backend smt \
      --smt-config-dir "$CFG" \
      --slot-save-path /tmp \
      --warmup > "$LOG" 2>&1 &
  echo "$!" > /tmp/spacemit-qwen-$PORT.pid
done
```

启动后就绪测试：

```bash
for PORT in 8063 8064 8065 8066; do
  echo "wait llama-server health OK: $PORT"
  for I in $(seq 1 120); do
    if curl -fsS --max-time 2 http://127.0.0.1:$PORT/health | grep -q '"status":"ok"'; then
      echo "llama-server ready: $PORT"
      break
    fi
    sleep 1
    if [ "$I" = 120 ]; then
      echo "llama-server health timeout: $PORT" >&2
      exit 1
    fi
  done
done
```

Python实现中，`LlamaServerManager`启动每个进程后必须轮询对应`/health`接口，直到返回OK或超时。所有路的`llama-server`都就绪后，模型加载状态才允许切换为已加载；任一路超时则停止已启动进程并向前端返回加载失败。

### Qwen3-ASR请求和内存清理

每一路offline识别使用该路独占的`llama-server`，因此每个服务固定使用`id_slot=0`。

请求流程：

1. 将endpoint切出的`float32` mono 16k PCM编码为16kHz mono WAV字节。
2. 将WAV字节base64编码。
3. `POST /v1/chat/completions`提交Qwen3-ASR请求。
4. 请求结束后调用`POST /slots/0?action=erase`清理该服务的slot状态。
5. 解析响应文本，发送offline最终结果事件。

请求体格式：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a speech recognition model."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-wav>",
            "format": "wav"
          }
        },
        {
          "type": "text",
          "text": ""
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "id_slot": 0
}
```

HTTP接口：

- 识别请求：`POST http://127.0.0.1:<port>/v1/chat/completions`
- slot清理：`POST http://127.0.0.1:<port>/slots/0?action=erase`
- 请求超时默认`300000ms`。
- slot清理超时建议`30000ms`。

slot清理规则：

- 每次offline请求完成后必须清理slot，避免prompt/KV状态在同一路后续请求中累积。
- 只要请求已经发送到`llama-server`，即使响应解析失败，也应尝试清理slot。
- slot清理失败视为该路offline请求失败，记录error并向前端发送错误事件。
- 停止识别和释放模型前，对每个已启动的`llama-server`执行一次best-effort slot清理。

响应解析规则：

- 优先读取OpenAI兼容响应中的`choices[0].message.content`。
- 兼容读取顶层`content`或`text`字段。
- Qwen3-ASR可能返回`language Chinese<asr_text>...`，需要删除`<asr_text>`之前的内容。
- 如果存在`</asr_text>`，删除该结束标签及其后的内容。
- 最终文本需要去除首尾空白。
- 清洗后为空字符串是合法识别结果，应作为offline最终结果覆盖对应online结果。

每路offline worker串行处理请求：

- 同一路只允许一个Qwen3-ASR请求在执行。
- 不同路请求并发执行，各自访问独立端口和独立`llama-server`。
- 每个offline结果记录请求耗时、输入音频时长和RTF。

### 识别运行规则

分段提交规则：

- endpoint触发后，如果当前段没有产生过非空流式识别文本，则清空当前段音频缓存，不提交Qwen3-ASR。
- endpoint触发后，如果当前段产生过非空流式识别文本，则先发布pending结果，再异步提交Qwen3-ASR。
- Qwen3-ASR返回后，使用同一路、同一句ID的final结果覆盖pending结果。
- Qwen3-ASR返回空字符串是合法final结果，也需要覆盖对应pending结果。

分段边界规则：

- offline整段识别使用上一endpoint之后已经送入流式识别的PCM帧。
- 不额外添加前置padding或后置padding。
- endpoint作为句级触发信号使用，不要求达到采样级VAD边界精度。
- 允许offline音频中包含少量前后静音。

本地文件输入规则：

- 本地音频文件必须是16kHz、单声道、PCM WAV。
- 不满足格式要求的本地文件视为输入错误。
- 无头模式在开始识别前完成全部本地输入文件校验。
- 文件读取结束时必须发送finish帧。

事件规则：

- 流式中间结果为partial。
- endpoint后的流式句子结果为pending。
- Qwen3-ASR整段纠正结果为final。
- 运行错误为fatal error。
- 单路输入结束为lane finished。
- 性能统计更新为stats updated。
- 前端、无头模式和日志使用同一套事件语义。

正常结束规则：

- 所有音频输入结束。
- 所有已提交offline任务处理完成。
- 所有pipeline和offline worker停止。
- 落盘最终识别结果、配置和性能统计。
- 释放online模型并停止所有`llama-server`。

异常结束规则：

- 任一路发生fatal error后，系统进入全局停止流程。
- 停止继续读取新的音频输入。
- 未提交的当前段音频缓存可以丢弃。
- 已经发出的Qwen3-ASR请求等待返回或超时。
- 停止流程结束后记录错误来源和已收集的性能统计。

Qwen3-ASR hard failure规则：

- HTTP连接失败。
- HTTP非2xx响应。
- 请求超时。
- JSON解析失败。
- 响应缺少可识别文本字段。
- slot清理失败。

统计规则：

- 每路记录处理音频帧数。
- 每路记录流式识别平均帧处理延迟。
- 每路记录Qwen3-ASR请求数、平均延迟和RTF。
- 每路记录丢帧数。
- 每路记录最终状态：finished、stopped或fatal。

## 技术方案：Python

本项目采用Python方案实现，Python负责主控流程、配置管理、音频输入抽象、pipeline调度、FastAPI/WebSocket前端服务、日志和模型进程生命周期管理。

模型内部实现不属于本项目范围，本项目只负责调用和管理：

- 流式识别调用`sherpa-onnx`。
- 整段识别调用外部`llama server`。
- Python侧只做数据编排、任务调度、结果合并和前端事件广播。

### Python运行环境

开发、验证、推理脚本在远端开发板`bianbu@10.0.91.51`运行时，统一使用以下Python环境流程。

硬性要求：

- 只使用Python 3.13。
- 只使用项目本地虚拟环境`.venv313`。
- 安装Python依赖前必须先激活`.venv313`。
- 安装Python依赖前必须先导出Spacemit和Aliyun包索引相关环境变量。
- 不允许在系统Python环境或未激活虚拟环境时安装依赖。

初始化命令：

```bash
python3.13 -m venv .venv313
. .venv313/bin/activate
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

后续每次运行服务、开发脚本、验证脚本或推理脚本前，都必须先激活环境：

```bash
. .venv313/bin/activate
```

如果需要新增依赖，也必须在激活`.venv313`并导出包索引变量后再执行`python -m pip install ...`。

### Python并发模型

- 主服务使用`FastAPI + uvicorn`。
- WebSocket、控制接口和事件广播使用`asyncio`。
- 每一路音频输入使用独立生产者线程或异步任务，保证100ms分帧不被后续识别阻塞。
- 每一路online识别使用独立消费者线程，调用`sherpa-onnx`并向前端事件队列投递流式结果。
- offline识别通过独立异步任务、线程池或外部`llama server`请求处理，pipeline只提交任务，不等待整段识别完成。
- 日志落盘和结果保存不得阻塞音频生产者和online识别消费者。

### 队列和背压

- 音频生产者到online消费者使用有界队列，默认`slot=1`。
- 发送offline整段识别的音频缓存独立维护，大小为`1.5 * rule3_min_utterance_length`。
- 当前端事件队列堆积时，优先合并同一路同一句的online中间结果，只保留最新revision。
- offline最终结果不丢弃，必须投递到前端和日志。
- 队列满、丢弃online中间结果、offline请求超时都需要记录warning。

### 资源和生命周期

- 模型加载、卸载、开始识别、停止识别由统一`PipelineManager`串行化处理，避免重复启动或并发释放资源。
- `llama server`进程由Python统一启动、健康检查、停止和兜底kill，避免僵尸进程。
- 停止识别时先停止音频生产者，再向各路pipeline发送finish帧，等待online reset和offline任务收尾，最后落盘结果。
- 服务退出时必须执行统一清理流程：停止识别、释放online模型、停止offline服务、关闭WebSocket连接、flush日志。

### 项目边界

- 本项目只实现Python应用层，包括音频输入编排、ASR调用、pipeline调度、Web服务、WebSocket更新、日志和模型进程管理。
- 不实现`sherpa-onnx`内部推理逻辑。
- 不实现`llama server`内部推理逻辑。

## 前端

使用FastAPI+WebSocket
使用HTML，默认风格
整体页面不滚动
未选的配置隐藏
前端更新消息以句为单位，具有路ID、句ID

### 多路WebSocket更新请求处理

多路识别结果更新不应由各识别线程直接操作WebSocket连接，而应采用统一事件队列：

- 每一路pipeline只负责产生识别事件，并投递到全局`frontend_event_queue`。
- WebSocket广播由单独的异步任务负责，从`frontend_event_queue`顺序取事件并发送给所有前端连接。
- 识别链路和前端发送链路解耦，避免浏览器慢、网络慢或WebSocket异常阻塞音频输入和ASR推理。
- 前端按`stream_id + sentence_id`定位显示区域，按`result_type`区分流式结果和整段纠正结果。
- 同一句的offline结果到达后覆盖对应online结果；如果offline结果晚到，也只覆盖同一`sentence_id`，不影响后续句子。
- 每个事件携带`seq`或`revision`，前端只接受比当前显示版本更新的事件，避免乱序消息覆盖新结果。
- WebSocket连接断开时只移除该连接，不影响pipeline运行。

事件结构建议：

```json
{
  "type": "asr_result",
  "stream_id": 0,
  "sentence_id": 12,
  "revision": 3,
  "result_type": "online",
  "is_final": false,
  "text": "当前流式识别文本",
  "timestamp_ms": 123456
}
```

其中：

- `stream_id`：第几路输入。
- `sentence_id`：当前路内的句子编号，由endpoint切分时递增。
- `revision`：同一句结果的更新版本，online每次变化递增，offline最终结果使用更高版本。
- `result_type`：`online`或`offline`。
- `is_final`：offline纠正结果或finish结果为`true`。

服务端结构：

- `PipelineManager`：管理多路pipeline生命周期。
- `Pipeline(stream_id)`：独立处理一路音频、online识别和offline任务提交。
- `FrontendEventBus`：持有全局异步队列和WebSocket连接集合。
- `WebSocketEndpoint`：只负责连接注册、注销和接收前端控制请求。

推荐使用单写多读模型：

- ASR结果事件只有pipeline写入队列。
- WebSocket广播任务是唯一发送者。
- 前端DOM按`stream_id + sentence_id`局部更新。

这样可以把并发问题限制在服务端队列和连接集合中，避免多个识别线程同时发送WebSocket消息导致锁竞争、乱序和异常传播。

### 顶部控制栏

选择路数2/4（默认）/6/8
加载/卸载模型按钮：必须先停止识别，加载后模型配置禁用;加载/卸载完成前卸载/加载按钮禁用
开始/停止识别按钮：必须先加载模型，开始后配置禁用;初始化完成前停止识别按钮禁用，reset完成前开始识别按钮禁用
开始识别清空识别信息
停止识别终止各路音频流

### 主要显示

根据路数平均划分显示区域（h/w）：2*1 2*2 3*2 4*2
根据选择路数显示各路配置：
音频输入：麦克风设备/本地文件（默认）
麦克风设备：读取设备列表并自动分配
本地文件：默认~/Music/wav16k下前四个文件，循环分配
online模型和模型文件夹路径（默认）
offline模型和模型文件夹路径（默认）
流式识别输出以灰色文本显示，整段识别以黑色文本显示，并覆盖对应的流式识别输出

## 无头模式

无头模式用于开发板验证和批量运行，不打开浏览器，但仍启动与Web模式相同的服务对象：

- 启动`FastAPI/uvicorn`服务。
- 启动`FrontendEventBus`，识别事件仍进入前端事件队列。
- 启动`PipelineManager`和`LlamaServerManager`。
- 启动并等待各路`llama-server` health OK。
- 通过内部控制接口模拟前端操作：加载模型、开始识别、等待结束、停止识别、释放资源。

无头模式固定使用本地文件输入，不使用麦克风：

- 默认路数为4路。
- 支持通过命令行选择`--lanes 2|4|6|8`。
- 默认输入目录为`~/Music/wav16k`。
- 默认按文件名排序选择前N个音频文件；文件不足N个时循环分配。
- 每个输入文件按16kHz音频以100ms帧送入对应pipeline。

命令行建议：

```bash
python -m multi_stream_asr.main --headless
python -m multi_stream_asr.main --headless --lanes 8 --input-dir ~/Music/wav16k
```

无头模式状态流程：

1. 创建与Web模式相同的应用对象。
2. 根据`--lanes`和本地音频文件生成配置。
3. 调用`PipelineManager.load_models()`，内部启动online模型和各路`llama-server`。
4. 等待所有`llama-server` health OK。
5. 调用`PipelineManager.start_recognition()`开始识别。
6. 所有本地文件输入结束并完成offline纠正后，自动停止识别。
7. 落盘识别结果、性能数据和日志。
8. 释放online模型，停止`llama-server`，关闭事件总线和Web服务。

无头模式的识别事件处理：

- 事件格式与WebSocket模式完全一致。
- 如果没有浏览器连接，`FrontendEventBus`仍消费事件并更新内部结果状态。
- 日志和最终结果以内部状态为准，不依赖前端DOM。

## 日志

记录流式识别丢帧信息并打印warning
记录流式识别每帧处理延迟（verbose模式下落盘）
记录整段识别每请求处理延迟和RTF（verbose模式下落盘）
结束识别落盘各路结果：包括配置信息、性能数据和识别结果
打印每路性能数据：流式识别每帧处理延迟+整段识别每请求处理延迟和RTF

## 主要代码框架

```text
multi-stream-asr/
    pyproject.toml
    README.md
    design.md
    src/
        multi_stream_asr/
            __init__.py
            main.py                     FastAPI/uvicorn入口
            config.py                   全局配置和默认路径
            models.py                   AudioFrame、ASRResult、事件模型
            pipeline/
                manager.py              多路pipeline生命周期管理
                pipeline.py             单路pipeline编排
                state.py                运行状态和状态切换
            audio/                      音频输入
                base.py                 音频输入抽象接口
                mic.py                  麦克风输入实现
                local_file.py           本地文件输入实现
                device.py               麦克风设备枚举
            asr/
                online/
                    base.py             在线识别抽象接口
                    zipformer.py        sherpa-onnx zipformer实现
                offline/
                    base.py             整段识别抽象接口
                    qwen3asr.py         llama server/qwen3-asr实现
                    llama_server.py     llama server进程管理
            buffer/
                audio_queue.py          online消费队列
                segment_cache.py        offline整段音频缓存
            web/
                server.py               FastAPI路由和WebSocket入口
                event_bus.py            前端事件队列和广播
                static/
                    index.html
            headless/
                runner.py               无头模式运行器
            log/
                recorder.py             识别结果和性能数据落盘
                metrics.py              延迟、RTF、丢帧统计
            utils/
                paths.py
                time.py
                process.py
```

## 其他

在开发板进行开发：bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr，以上路径皆位于开发板
生产者消费者模型原子化
不需要test
