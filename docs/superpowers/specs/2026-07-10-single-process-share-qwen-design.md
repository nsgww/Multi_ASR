# 单进程共享 Qwen（single-process-share-qwen）设计

日期：2026-07-10
基线提交：`8522bf4240f5af9f3fb42d5240e2bb738a51d4b5`（单进程；音频输入与 zipformer 分生产/消费线程；各路独占 Qwen）
开发分支：`single-process-share-qwen`（自 8522bf4 新建）

## 目标

将 `multi-asr-test` 的共享 Qwen 服务（`llama-multi-asr`，单份权重 + 服务端
orchestrator 队列 + E/D 错峰流水）接入 multi-stream-asr，**直接替换**原有的
per-lane 独占 llama-server 后端。在开发板上完成 4 路真实无头测试，产出功能与
性能报告（与独占方案基线 runs 对比）。

预期收益：offline 权重内存从 4×(483MB gguf + 180MB onnx)≈2.6GB 降到
1×≈660MB；推理侧 rtf 有望优于独占基线（multi-asr-test 实测 rtf_infer≈0.16，
独占基线 offline_rtf avg≈0.50）。

## 非目标

- 不保留独占后端、不做双后端切换（用户已确认直接替换）。
- 不测 2/6/8 路矩阵（仅 4 路）。
- 不改 online 通路（zipformer 线程模型沿用 8522bf4）。
- 不上传 q41 gguf（复用板上 q40）。

## 部署布局

```
本机（liangff）
  multi-stream-asr 仓库 main
    └─ git worktree add ../multi-stream-asr-share-qwen single-process-share-qwen（基于 8522bf4）
         │  rsync（排除 .venv313/runs/models/.git）
         ▼
开发板 bianbu@10.0.91.51
  ~/Workspace/projects/multi-stream-asr              ← 不碰（main 的运行目录）
  ~/Workspace/projects/multi-stream-asr-share-qwen   ← 新建：rsync 目标 + .venv313 + 无头测试
  ~/Workspace/sm-sdk/models/asr/qwen3asr/            ← 复用：q40 gguf + dynq onnx（无新增大文件）
  ~/Workspace/projects/multi-asr-runtime/            ← 新建（~40MB）：bin/llama-multi-asr + lib/ + ort_lib/
                                                        （从本机 multi-asr-test rsync，不含模型和音频）
```

运行时进程模型（4 路）：

```
python 主进程（单 asyncio loop + 各路生产/消费线程，沿用 8522bf4 线程模型）
  ├─ 4× 音频输入线程 + 4× zipformer 消费线程（CPU provider）
  └─ 1× llama-multi-asr 子进程（替代原 4× llama-server）
       encoder 亲和 8,9 / decoder 亲和 10-15、6 线程 / --n-parallel 4 / 127.0.0.1:8090
```

板上 CPU present 0-15，8-15 为 A100 大核，分核采用 multi-asr-test 实测最优
的 2:6（encoder "8,9" : decoder "10,11,12,13,14,15"，decoder 6 线程）。

## 组件设计

### 1. `shared_server.py`（由 `asr/offline/llama_server.py` 改造）

**config.json 生成**：运行时写入 `/tmp/multi-stream-asr/shared-qwen/config.json`，
结构与 `multi-asr-test/qwen3-asr-0.6B/config.json` 逐字段对齐，仅改三处：

- `audio_model.frontend_model_path` / `backend_model_path` → 板上绝对路径
  `~/Workspace/sm-sdk/models/asr/qwen3asr/Qwen3-ASR-0.6B-encoder-{frontend,backend}.dynq.onnx`
- `text_model.model_path` → 同目录 `Qwen3-ASR-0.6B-text-q40.gguf`（绝对路径）
- 亲和字段可由 AppConfig 配置，默认 `spacemit_ep_intra_thread_num: 2`、
  `spacemit_ep_inter_thread_num: 1`、`spacemit_ep_intra_thread_affinity: "8;9"`

关键差异警示：`llama-multi-asr` 读取的是 `audio_model` 下**平铺小写**的三个
`spacemit_ep_*` 字段（亲和用分号分隔），不是现有代码的嵌套大写 `ep_config`
块；三字段缺一则 encoder 分核**静默失效**（回退 4 线程）。config 还必须包含
`text_model`、`prompt`、`support_languages` 块。

**启动命令**（对齐 run_server.sh）：

```
LD_LIBRARY_PATH=<runtime>/lib:<runtime>/ort_lib \
  <runtime>/bin/llama-multi-asr \
  -m <models>/Qwen3-ASR-0.6B-text-q40.gguf \
  --smt-config-dir /tmp/multi-stream-asr/shared-qwen \
  --host 127.0.0.1 --port 8090 \
  --decoder-cores 10,11,12,13,14,15 --decoder-n-threads 6 \
  --n-parallel 4 --warmup
```

`<runtime>` = `~/Workspace/projects/multi-asr-runtime`（AppConfig 新增字段）。

**生命周期**：应用启动时拉起单个子进程；就绪判定轮询 `GET /health` 至 200
（超时预算 ≥60s，warmup 约 25s）；退出时按现有 `run_best_effort` 模式
terminate→kill。日志与 pid 文件沿用现有 per-run 目录约定。

**AppConfig 变更**：删除 `LaneConfig` 中 per-lane 的 llama 端口/核号字段；
新增全局 `SharedQwenConfig`（port=8090、encoder_cores="8,9"、
decoder_cores="10,11,12,13,14,15"、decoder_threads=6、n_parallel=4、
runtime_dir、health 超时）。

### 2. `qwen3asr.py` 客户端改造

- 所有 lane 共用 `http://127.0.0.1:8090/v1/chat/completions`。
- payload 结构不变（input_audio base64 WAV + `"language Chinese<asr_text>"`
  prompt、temperature 0），删除 `id_slot: 0` 字段与请求后的
  `POST /slots/0?action=erase` 清理（slot 由服务端 orchestrator 接管）。
- 解析响应中的 `multi_asr_timings` {queue_ms, encode_ms, prefill_ms,
  decode_ms, total_ms} 并上报 metrics。
- 单请求超时放宽到 120s（可配）。

**并发语义**：8522bf4 的 per-lane `_offline_tail` 串行链原样保留（保证每路
final 顺序）；lane 间并发请求同时打到共享服务，由服务端队列
（n_parallel=4，与 lane 数一致）调度。客户端不新增排队逻辑。

**错误处理**：服务端过载返回 `server busy: queue full`（非 200）。4 路 +
n_parallel=4 + queue-max 64 理论上打不满；客户端按现有失败路径处理（记日志、
该句 final 标记失败，不重试、不崩 pipeline）。

### 3. metrics 扩展

逐请求记录 `multi_asr_timings`，metrics.json 新增 `offline_shared_timings`
聚合段（queue/encode/prefill/decode/total 各 avg/max/count），与现有
`offline_rtf` 并列。`offline_rtf` 计算口径不变（client 侧计时），保证与历史
runs 可比。由此报告可区分"排队慢"（queue_ms 大）与"计算慢"
（encode/prefill/decode_ms 大）。

## 数据流

```
lane N 音频线程 → OnlineFrameQueue → zipformer 线程 → endpoint
  → 整句 PCM → _schedule_offline（per-lane tail 串行）
  → Qwen3AsrClient.recognize()（base64 WAV → POST :8090）
  → orchestrator 队列 → encode_loop/decode_loop（E/D 错峰）
  → 响应（text + multi_asr_timings）→ final 事件 + metrics
```

## 测试策略

1. **单元测试**（本机/板上均可，mock HTTP）：
   - `test_qwen3asr.py`：payload 无 id_slot、无 erase 调用、timings 解析。
   - shared_server 测试：config.json 生成断言（三个小写 `spacemit_ep_*` 字段
     必须存在；`text_model`/`prompt` 块存在；模型绝对路径正确）、启动命令拼
     装、health 轮询与超时。
   - `pytest -v` 全绿。
2. **板上假源冒烟**：`python -m multi_stream_asr.main --headless --fake
   --lanes 4`（若 fake 路径不起共享服务则跳过，实施计划中落实）。
3. **板上真实无头测试**：`python -m multi_stream_asr.main --headless
   --lanes 4 --input-dir ~/Music/wav16k`；检查 metrics.json：offline_rtf、
   online_frame_latency、`offline_shared_timings`、识别文本非空。
4. **性能报告**：`docs/` 下新增报告，对比 4 路 shared run 与独占基线
   （runs/headless-1783590515514、-1783600625335，offline_rtf avg≈0.50）：
   offline_rtf、timings 分解、online 延迟是否受影响、内存对比
   （free / 进程 RSS）。

## 环境与依赖

- 板上新目录建 `.venv313`（Python 3.13），流程遵循 `~/.codex/AGENTS.md`。
- sherpa-onnx 装现成 wheel，不编译；如遇问题：
  `pip install sherpa-onnx --index-url
  https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple`
- 模型版本注记：性能与 multi-asr-test 的 q41 基线比较时需注明本方案用 q40。

## 风险

- q40 与 llama-multi-asr 的兼容性未验证（multi-asr-test 自带 q41）。缓解：
  实施第一步先在板上用 q40 手动拉起服务打通一条请求，失败则回退上传 q41
  （506MB，板上剩 6GB 可容纳）。
- config.json 亲和字段静默失效风险。缓解：单元测试断言字段存在 + 真实测试
  时用 `/proc/<pid>/task/*/status` 抽查线程绑核。
- 4 路突发同时打满 n_parallel=4 时 queue_ms 上升属预期行为，不视为缺陷，
  在报告中呈现。
