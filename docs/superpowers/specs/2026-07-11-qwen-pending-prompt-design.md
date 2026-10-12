# Qwen3-ASR Pending Prompt 注入设计

日期：2026-07-11

## 目标

把 Zipformer endpoint 后发布为 `pending` 的识别文本传入同一句的 Qwen3-ASR 请求，作为
Qwen 的辅助文本上下文。

## 范围

只修改 `Qwen3AsrClient` 的 chat request 构造和对应单元测试。`LanePipeline` 已将
endpoint 文本作为 `prompt` 传给 `recognize()`，不修改 pipeline、IPC、模型配置、并发配置
或指标口径。

## 请求格式

现有 user content 的文本项从固定值：

```text
language Chinese<asr_text>
```

改为：

```text
language Chinese<asr_text>{prompt}
```

其中 `{prompt}` 为 Zipformer pending 文本。不会新增 system/user message、提示说明、JSON
字段或额外 HTTP 请求。空字符串 prompt 仍产生原有固定文本。

## 验证

1. 单元测试断言非空 prompt 被精确附加到文本项，空 prompt 保持原格式。
2. 开发板在项目本地 Python 3.13 `.venv313` 中启动共享 q40 Qwen3-ASR 服务，对 10 秒
   16 kHz WAV 发出带明确 pending 文本的单请求，并确认 HTTP 成功、识别文本非空、返回
   timing。
3. 不重跑完整 pytest 或四路真实测试。
