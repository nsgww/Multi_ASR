# 基于 Token 时间戳的音频段切分设计

## 目标

当前 pipeline 在 Zipformer 触发 endpoint 时，将 `SegmentCache` 中的全部音频作为当前句提交给 Qwen3-ASR，然后清空缓存。endpoint 触发时刻晚于上一句末 token 的发射时刻，缓存尾部还可能比解码器已处理位置更靠后，因此当前句可能包含下一句字头，下一句则丢失字头。

本设计将 endpoint 仅作为“现在需要切分”的信号，实际切点使用 Zipformer 最后一个 token 的流内时间加固定尾部余量。切点之前的音频提交给当前句，切点之后的音频原样保留给下一句。目标是消除相邻段的重叠和缺口，降低 Qwen3-ASR 首字纠错失败，同时保持现有多进程、线程亲和、事件、日志和性能统计结构不变。

## 原因与约束

当前切点是 endpoint 被观察到时的缓存尾部。它相对末 token 发射时刻还包含 rule2 连续 blank 等待、chunk/下采样量化，以及 `put()` 先于 Zipformer `accept()` 带来的缓存前置。自然句间停顿可能短于这些延迟之和，因此 endpoint 到来时下一句已经进入缓存。

Sherpa 的 `timestamps[i]` 表示 token 的解码或发射时刻，不是字的声学结束时刻。`last_token_time + margin` 是经验性尾部保护，不能视为严格的声学结束边界。默认 margin 先取 `0.1s`，再通过真实音频测试校准。

当前应用还会复用已加载的 Zipformer recognizer：warmup 和多次识别使用同一个 recognizer，而每次识别新建 `SegmentCache`。Sherpa stream 的 `start_time` 在普通 reset 后继续累计，所以不能直接将 recognizer 生命周期内的绝对时间与新缓存从 0 开始的时间比较。每次识别必须建立独立 stream 和独立时间原点。

## 方案比较

### 固定回退窗口

endpoint 时从缓存尾减去固定时长。实现简单，但无法适应 token 发射延迟和 endpoint 量化变化，仍可能切入下一句或截断当前句。

### 直接使用 Sherpa 绝对时间

使用 `start_time + timestamps[-1]` 切分。单次 stream 内坐标正确，但当前项目的 warmup、模型复用和每次新建缓存会造成原点不一致；异常时间戳还可能生成空离线段。

### 受保护的 run 内 token 时间切分

每次识别创建新的 Sherpa stream，使 token 时间和该次识别的新缓存都从 0 开始。正常路径按最后 token 时间加 margin 切分；缺失或异常时间戳退回整缓存提交。该方案同时隔离 warmup、前一次自然结束和手动停止留下的状态，是本项目采用的方案。

## 架构与职责

### Zipformer 适配层

在线 recognizer 接口增加 `start_run()`。`LanePipeline` 的 consumer 线程在读取首帧前调用它；`ZipformerRecognizer.start_run()` 只创建新的 Sherpa online stream，不重新加载模型或创建新的 ONNX session。该调用必须留在现有 Zipformer consumer 线程中，以保持 SpaceMIT EP 线程归属和亲和性不变。

普通 endpoint 继续调用 Sherpa 对当前 stream 的 `reset()`。这种 reset 保留尚未处理的特征，供下一句继续解码；它不会创建新 run，也不会改变 run 时间原点。

`ZipformerRecognizer.accept()` 优先调用 `get_result_all()`，并在 endpoint reset 前读取：

- `text`
- `start_time`
- `timestamps[-1]`

输出模型增加：

```python
@dataclass(frozen=True)
class OnlineRecognition:
    # existing fields unchanged
    last_token_stream_s: float | None = None
```

当时间戳存在且有限时：

```python
last_token_stream_s = raw.start_time + raw.timestamps[-1]
```

新 stream 的原点是本次识别开始，因此该值可直接与当前 `SegmentCache` 的时间坐标比较。若当前 Sherpa Python API 不提供 `get_result_all()`，或结果没有可用时间戳，则保留现有文本识别兼容路径并返回 `None`。

### SegmentCache

`SegmentCache` 继续存储 `float32` mono PCM，并增加绝对样本坐标：

```python
self._start_sample = 0
self._end_sample = 0
```

`put()` 每接收一个非 finish chunk 都推进 `_end_sample`。容量裁剪从左侧删除或截短 `n` 个样本时，同时推进 `_start_sample += n`。

缓存公开以下接口：

```python
@property
def start_time_s(self) -> float: ...

@property
def end_time_s(self) -> float: ...

def cut(self, stream_time_s: float) -> np.ndarray:
    """返回 [缓存头, 切点)；将 [切点, 缓存尾) 原样留在缓存。"""
```

`cut()` 使用 `round(stream_time_s * sample_rate)` 得到样本索引，并 clamp 到当前 `[_start_sample, _end_sample]`。切分后 `_start_sample` 等于实际切点，`_end_sample` 不变，剩余 chunks 只包含尾部。切分可以跨 chunk，连续切分后各段拼接必须与原缓存逐样本一致。

`cut()` 拒绝 NaN 和 Inf。`reset()` 清空 chunks 和样本计数，并将 `_start_sample` 推进到 `_end_sample`，避免同一 run 内清空后坐标倒退。每次 run 会新建 cache，因此 run 起点仍为 0。

### LanePipeline

consumer 开始时先调用 `online_recognizer.start_run()`，再读取音频帧。每个普通音频帧仍按现有顺序先 `segment_cache.put(frame)`，后 `online_recognizer.accept(frame)`，两侧接收完全相同的样本流。

非空 endpoint 的处理顺序为：

1. 保存切分前的完整缓存，用于异常降级。
2. 若 `last_token_stream_s` 可用，计算 `cut_at = last_token_stream_s + cut_margin_seconds`。
3. 若 `cut_at` 是有限值且严格晚于 `segment_cache.start_time_s`，调用 `cut(cut_at)`。
4. 若时间戳缺失、非有限、切点不晚于缓存头，或切分结果为空但切分前缓存非空，则提交切分前的完整缓存并清空当前缓存。
5. 发布现有 `pending` 事件，记录 Zipformer endpoint 输出，异步提交 Qwen3-ASR，然后递增 `sentence_id`。

切点晚于缓存尾时由 `cut()` clamp 到尾部，行为等价于提交整个缓存。正常切分会将尾部保留为下一句的前导音频，不增加重叠、静音填充或文本去重。

空文本 endpoint 不提交 Qwen。它也不再清空全部缓存，而是执行：

```python
segment_cache.cut(segment_cache.end_time_s - rule2_min_trailing_silence)
```

返回的旧音频被丢弃，缓存最后 `rule2` 秒保留。若缓存不足 `rule2` 秒，clamp 后保留全部缓存。这可以保护 endpoint 时刚进入缓存但尚未发射 token 的下一句字头。

## 结束与停止语义

自然 finish 与手动停止采用不同策略：

- 自然 finish：若当前在线文本非空，发布该句 `pending`，将缓存全部剩余 PCM 提交 Qwen，等待现有 offline 队列完成并发布 `final`，最后发布 `LANE_FINISHED`。自然 finish 没有 token 切点，因为流已结束且不存在需要保留给下一句的音频。
- 自然 finish 且当前在线文本为空：不提交离线请求，清空缓存并直接完成。
- 手动停止或 shutdown：保持快速停止语义，不为未 endpoint 的残段新增 pending/final，也不等待一项新 Qwen 请求。下次识别的 `start_run()` 创建新 Sherpa stream，新建的 pipeline 使用新 cache，因此残余状态不会进入下一次识别。

已在 endpoint 前提交的 Qwen 请求继续按现有超时和收尾规则处理。

## 配置

`EndpointConfig` 增加：

```python
cut_margin_seconds: float = 0.1
```

配置创建时必须满足：

```text
0 <= cut_margin_seconds < rule2_min_trailing_silence
```

非法配置在模型加载前抛出 `ValueError`。该字段不增加前端控件，沿用默认值并通过现有 `AppConfig` 序列化写入 run 的 `config.json`。缓存容量仍为 `1.5 * rule3_min_utterance_length`。

## 错误处理与兼容

以下情况不使 lane 进入 fatal，而是退回旧的整缓存提交行为：

- Sherpa Python API 没有 `get_result_all()`。
- 结果没有 token 时间戳。
- `start_time` 或最后 token 时间戳不是有限数。
- 计算出的切点不晚于当前缓存头。
- 有非空缓存和非空文本，但时间切分意外得到空 PCM。

只有缓存本身为空时才跳过空 offline 请求。模型文本、事件 revision、sentence ID、Qwen prompt、落盘格式和 IPC payload 不增加 token 时间字段；该字段只在 lane 进程内部用于切分。

## 测试设计

### SegmentCache 单元测试

- 在单 chunk 内按样本边界切分。
- 使用 `round()` 处理非整数样本时间。
- 切点低于缓存头和高于缓存尾时正确 clamp。
- 跨多个 chunk 切分并保留尾部。
- 容量裁剪完整 chunk 和部分 chunk 时同步推进 `_start_sample`。
- 连续两次 `cut()` 后，各返回段与最终尾部拼接后逐样本等于原音频。
- `_end_sample` 等于所有 `put()` 样本累计值，且不因左裁剪或 cut 倒退。
- NaN/Inf 切点抛出 `ValueError`。

### Zipformer 单元测试

- `accept()` 使用 `get_result_all()`，在 endpoint reset 前读取结果。
- `last_token_stream_s` 等于 `start_time + timestamps[-1]`。
- 空 timestamps 和非有限时间返回 `None`。
- 缺少 `get_result_all()` 时仍能返回文本，时间字段为 `None`。
- 每次 `start_run()` 创建新 stream，但 recognizer/model 实例不变。
- endpoint 只 reset 当前 stream，不创建新 stream。

### Pipeline 集成测试

- 两个 endpoint 使用已知 token 时间，精确断言两次 Qwen PCM 无重叠、无缺口，第二句包含第一次切分保留的尾部。
- 空 endpoint 丢弃旧音频并保留最后 `rule2` 秒。
- 缺失、非有限、过旧时间戳提交完整缓存。
- 模拟 rule3 强切，断言最后 token 后的音频进入下一段。
- 自然 finish 对非空残句发布 pending/final 后再发布 `LANE_FINISHED`。
- 自然 finish 的空残句不提交 Qwen。
- 手动停止不提交未 endpoint 残段。
- 连续运行两次时每次调用 `start_run()`，第二次的切点不受 warmup 或第一次运行时长影响。
- 现有 partial/pending/final revision、sentence ID 和模型输出测试保持通过。

## 真实设备验收

本机完整测试通过后，按项目 K3 规范使用 Python 3.13 和项目 `.venv313` 在开发板验证：

1. 运行完整测试集。
2. 使用包含多个汉语句子和自然短停顿的本地 WAV 连续识别至少两次。
3. 检查每句首字不丢、相邻句不重复、Zipformer endpoint 文本与 Qwen final 句意对齐。
4. 检查第二次运行没有 warmup 或第一次运行造成的切点漂移。
5. 记录 dropped frames、online frame latency 的平均值和最大值、offline RTF 的平均值和最大值，与最近可比 run 对照。

正确性优先于性能调优。目标仍是 online 流式延迟约 `50ms`、整段 offline RTF 约 `0.4`，且不因切分改动产生额外丢帧。默认 `0.1s` margin 仅在真实切段显示尾字被截断或下一句字头混入时调整，调整后仍必须小于 rule2。

## 不在范围内

- 不修改 sherpa-onnx。
- 不引入 VAD、音频重叠、文本去重或应用层句子合并。
- 不解决拖长音导致的假 endpoint，也不解决噪声推迟 endpoint。
- 不改变 Zipformer/Qwen 模型、provider、核心亲和、进程模型、在线队列、前端布局或结果文件格式。
