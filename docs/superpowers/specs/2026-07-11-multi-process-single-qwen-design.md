# 多进程 Zipformer + 单共享 Qwen3-ASR 设计

日期：2026-07-11

基线分支：`main`

开发分支：`multi-process-single-qwen`

本机与开发板目录：`~/Workspace/projects/multi-process-single-qwen`

## 背景

`main` 已将每路音频输入、Zipformer、lane 文件和 lane 指标隔离到独立 Python 子进程，
但每个 lane 仍会启动一份独占 `llama-server`。4 路真实运行需要 4 份 Qwen3-ASR 权重与
运行时资源，且最新基线 `runs/process-1783746225054` 的每路丢帧为 31-32、全局
`offline_rtf.average` 为 0.499。

`single-process-share-qwen` 分支已经在旧的单 Python 进程架构上验证
`llama-multi-asr`：单份 q40 权重、服务端 orchestrator 队列、encoder/decode 错峰流水、
`--n-parallel 4` 和共享 timing 指标。该分支的真实四路报告 RTF 为 0.471，但它基于旧
pipeline，不能直接合并，否则会删除 `main` 的 lane 进程、IPC、lane-owned 文件和
schema v3 聚合。

本设计只移植共享 Qwen 服务能力，保留 `main` 的多进程 Zipformer 架构。

## 目标

- 保留 4 个独立 lane Python 子进程，每路独占音频输入、Zipformer、结果文件和在线指标。
- 用 1 个全局 `llama-multi-asr` 替换 4 个 lane-owned `llama-server`。
- 由主 Python 进程管理共享 Qwen 的启动、健康检查、监控和停止。
- lane 子进程通过 HTTP 直接并发请求共享服务，不经主进程传输 PCM。
- 保持 Web、Headless、IPC、fake 模式、lane-owned 文件和 schema v3 功能正常。
- 共享服务固定使用 q40、`--n-ctx 8192`，避免 65536 context 引发 KV cache OOM。
- 连续 3 次真实四路测试中，每一次的每一路 `dropped_frames < 10`。
- 3 次所有 lane 合并后的 `offline_rtf.average <= 0.471`。

## 非目标

- 不把 PCM、NumPy 数组、segment cache、Qwen raw response 或文件内容传给主进程。
- 不增加主进程 Qwen broker 或额外 Qwen supervisor Python 进程。
- 不保留 per-lane `llama-server` 与共享服务的运行时切换开关。
- 不在项目代码中直接绑定 A100/X100 Linux 线程；绑核由 SpaceMiT EP、sherpa-onnx 和
  `llama-multi-asr` 配置完成。
- 首轮真实性能未达标时，不调整 Qwen 核分配、线程数、`n_parallel` 或 Zipformer EP
  参数；保留数据并报告验收失败。
- 不复制或提交 Zipformer、Qwen 模型、共享 runtime 和动态库。

## 方案选择

采用方案 A：主进程拥有共享服务，lane 直接请求。

```text
main Python process
  ├─ SharedQwenServerManager
  │    └─ llama-multi-asr :8090
  ├─ lane0 Python process -> Zipformer + Qwen HTTP client ─┐
  ├─ lane1 Python process -> Zipformer + Qwen HTTP client  ├─> :8090
  ├─ lane2 Python process -> Zipformer + Qwen HTTP client  │
  └─ lane3 Python process -> Zipformer + Qwen HTTP client ─┘
```

未采用的方案：

- 主进程 Qwen broker：要求 IPC 传整句 PCM，破坏现有进程边界并增加复制与排队。
- 独立 Qwen supervisor Python 进程：增加一层控制协议和故障状态，但单服务没有对应收益。

## 组件设计

### 1. 全局共享服务管理器

新增 `SharedQwenServerManager`，只由 `ProcessPipelineManager` 所在的主进程创建。它负责：

1. 清理本项目遗留的共享服务。
2. 校验 binary、runtime library 和模型文件存在。
3. 生成共享 SMT `config.json`。
4. 启动一个 `llama-multi-asr` 并等待 `/health`。
5. 暴露进程存活状态供主进程 watchdog 采样。
6. 正常退出时 terminate，超时后 kill，并关闭日志句柄。

启动命令固定为：

```text
<runtime>/bin/llama-multi-asr
  -m <qwen-model-dir>/Qwen3-ASR-0.6B-text-q40.gguf
  --smt-config-dir /tmp/multi-stream-asr/shared-qwen
  --host 127.0.0.1
  --port 8090
  --decoder-cores 10,11,12,13,14,15
  --decoder-n-threads 6
  --n-parallel 4
  --n-ctx 8192
  --warmup
```

环境中的 `LD_LIBRARY_PATH` 前置：

```text
<runtime>/lib:<runtime>/ort_lib
```

共享 SMT config 使用参考实现要求的平铺小写字段：

```json
{
  "audio_model": {
    "spacemit_ep_intra_thread_num": 2,
    "spacemit_ep_inter_thread_num": 1,
    "spacemit_ep_intra_thread_affinity": "8;9"
  }
}
```

config 同时保留完整 `audio_model`、`text_model`、`prompt` 和
`support_languages`，模型路径使用开发板绝对路径。

stale cleanup 必须限定为本 runtime binary、端口 8090 和 pid 文件。不得使用无范围的
`pkill -f llama-multi-asr`，避免误杀其他工程实例。

### 2. 配置模型

用全局 `SharedQwenConfig` 替换 `OfflineServerConfig` 中的 per-lane 服务语义，默认值：

```text
host=127.0.0.1
port=8090
runtime_dir=~/Workspace/projects/multi-asr-runtime
encoder_thread_num=2
encoder_affinity=8;9
decoder_cores=10,11,12,13,14,15
decoder_threads=6
n_parallel=4
n_ctx=8192
request_timeout_ms=120000
```

`LaneConfig` 不再持有 Qwen 端口、gguf core 或 Qwen thread count。Zipformer 继续使用
`main` 当前的 `zipformer.config`，模型创建和推理始终位于同一个长期消费者线程。

### 3. lane runtime

真实模式的 lane 加载流程改为：

```text
send lane_loading
start persistent Zipformer consumer thread
load Zipformer on that thread
create Qwen3AsrClient(:8090)
send lane_loaded
```

lane 不再创建、清理或持有 `LlamaServerManager`。Fake 模式继续使用现有 fake online/offline
recognizer，不启动共享服务。

lane-owned 文件职责不变：

- `results/laneX.txt`：该路最终完整结果。
- `model_outputs/laneX.jsonl`：严格保留 `model`、`audio_seconds`、`rtf`、
  `stream_id`、`sentence_id`、`text` 六个字段。

### 4. Qwen HTTP client

每个 lane 子进程创建一个指向共享端口的 client。请求协议按参考分支移植：

- endpoint：`POST http://127.0.0.1:8090/v1/chat/completions`。
- user text：`language Chinese<asr_text>`。
- 删除 `id_slot`。
- 删除 `/slots/0?action=erase`。
- `temperature=0`，`max_tokens=256`。

同一路继续通过 `_offline_tail` 保证句子顺序；不同 lane 并发请求，由服务端
`n_parallel=4` orchestrator 调度。客户端不增加全局锁、重试或额外排队。HTTP 非成功状态、
busy 或超时沿用 lane fatal 路径。

### 5. shared timings 与 schema v3

`OfflineRecognitionResult` 增加可选 `timings`。客户端解析响应中的
`multi_asr_timings`：

```text
queue_ms
encode_ms
prefill_ms
decode_ms
total_ms
```

每个 lane 维护上述字段的 count/total/average/max 摘要。`metrics_snapshot` 和
`metrics_final` 增加 `offline_shared_timings`，主进程合并后写入 schema v3：

- 全局 `offline_shared_timings`。
- 每个 `lanes[]` 元素内的 `offline_shared_timings`。

`offline_rtf` 继续使用客户端从发请求到收到并解析响应的墙钟时间，不改历史口径。
主进程不读取 lane model-output 文件计算指标。

## 生命周期与故障传播

### 加载

```text
main cleanup stale shared service
main start shared Qwen
main wait /health
main spawn all lane processes concurrently
main wait all lane_loaded
state -> LOADED
```

共享服务启动或健康检查失败时不 spawn lane，主进程停止已创建服务并进入 `FATAL`。

### 运行

主进程在 loaded/running 状态启动共享服务 watchdog。若服务进程意外退出：

1. 记录包含 exit code 的 fatal。
2. 发布前端 fatal event。
3. 通知全部活跃 lane 停止。
4. 等待 lane 有界退出并执行现有 terminate/kill 升级。
5. 清理共享服务资源。

lane fatal 时沿用现有 sibling-stop 规则；所有 lane 回收后再停止共享 Qwen。

### 正常停止

```text
request lane stop
wait each lane offline tail
close lane result/model-output files
join all lane processes
stop shared Qwen
close IPC/process resources
state -> UNLOADED
```

正常和失败退出后都必须确认新 worktree 对应的 lane 与 `llama-multi-asr` 无残留。

## 资产复用与部署

开发分支在执行阶段从最新 `main` 创建：

```text
branch:   multi-process-single-qwen
worktree: ~/Workspace/projects/multi-process-single-qwen
```

本机和开发板都创建：

```text
multi-process-single-qwen/models/asr
  -> ../multi-stream-asr/models/asr
```

开发板复用：

```text
Qwen model dir: ~/Workspace/sm-sdk/models/asr/qwen3asr
q40 model:       Qwen3-ASR-0.6B-text-q40.gguf
runtime:         ~/Workspace/projects/multi-asr-runtime
```

代码同步排除 `.git`、`.venv313`、`runs`、`models`、`__pycache__` 和其他缓存。开发板使用
项目本地 `.venv313` 和 Python 3.13；任何安装都必须先激活 venv，并设置
`~/.codex/docs/k3/remote-device-python.md` 指定的 Spacemit/Aliyun index 变量。

## 测试策略

### 本机自动测试

- 共享 config 的完整字段、绝对模型路径和小写 EP key。
- 启动命令包含 q40、`--n-parallel 4`、`--n-ctx 8192`、decoder 核与线程数。
- 健康检查成功、超时、启动失败、terminate/kill 和 scoped stale cleanup。
- Qwen payload 无 slot，prompt 正确，timings 正确解析。
- shared timing tracker 的单 lane 摘要、IPC round trip 和跨 lane 合并。
- 主进程 Qwen-first 加载顺序、watchdog fatal、lane fatal、正常停止和强制回收。
- 现有 IPC、lane runtime、Headless、Web、文件落盘和 schema v3 回归。
- Python 3.13 `compileall`、完整 `pytest`、4 路 fake headless smoke。

### 开发板功能测试

1. Python 3.13 完整测试。
2. 单请求真实 Qwen 冒烟：q40、context 8192、非空文本、timings 完整、无 OOM/exit 137。
3. 连续 3 次相同输入的真实四路 headless 运行。
4. 每次采样进程树，确认 `1 main + 4 lane + 1 llama-multi-asr`。
5. 检查四路结果非空、Zipformer/Qwen sentence ID 对齐、model-output 六字段、schema v3
   shared timings、无进程残留。
6. 将 3 个完整 run 同步回本机并生成中文性能与功能报告。

## 性能验收

使用与基线相同的 `~/Music/wav16k` 排序后前 4 个 WAV、100 ms 帧、
`OnlineFrameQueue(maxsize=1)` 和 endpoint 默认值。

每次运行必须满足：

```text
lane0.dropped_frames < 10
lane1.dropped_frames < 10
lane2.dropped_frames < 10
lane3.dropped_frames < 10
```

3 次合并 RTF 定义为：

```text
sum(3 次所有 lane 的 offline_rtf.total)
------------------------------------------------ <= 0.471
sum(3 次所有 lane 的 offline_rtf.count)
```

报告同时记录：

- online frame latency average/max。
- 每次每路 dropped frames。
- offline RTF average/max/count。
- queue/encode/prefill/decode/total timing average/max。
- event-loop lag average/max。
- 每路结果行数、字节数和 sentence ID 对齐情况。
- 服务进程 RSS、整机内存和 OOM 检查。

如果任一次任一路丢帧达到 10，或三次合并 RTF 大于 0.471，则性能验收失败。不得在本任务
内修改核分配、线程数、`n_parallel` 或 Zipformer EP 参数重跑以改变结论；保留三次真实
产物并在报告中说明原因。

## 风险与缓解

- **共享服务成为单点故障**：主进程 watchdog 监控 exit code，并统一停止 lane。
- **默认 context 导致 OOM**：启动命令和测试都固定断言 `--n-ctx 8192`。
- **EP config key 静默失效**：测试断言平铺小写 key，真实日志和 `/proc` 抽查 affinity。
- **共享队列造成 RTF 尾延迟**：记录 shared timings；验收使用三次合并平均，max 单独报告。
- **共享 Qwen 释放资源晚于 lane**：停止顺序固定为 lane 全部退出后再停止服务。
- **误杀其他项目服务**：stale cleanup 限定 binary、端口和 pid，不做全局模糊 pkill。
- **模型重复占用磁盘**：Zipformer 使用符号链接，Qwen/runtime 使用板上已有绝对路径。
