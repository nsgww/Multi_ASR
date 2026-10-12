# 4 路在线识别丢帧回归调查

## 结论

丢帧回归由提交 `9b9dd18`（`feat: scale zipformer EP threads by lane count`）引入。该提交把路数小于等于 4 时每个 Zipformer 进程的 `SPACEMIT_EP_INTRA_THREAD_NUM` 从 1 改为 2。

在提交 `9ce077b` 上进行单变量真实测试，只改变该配置：

| 配置 | Online avg/max | Offline RTF | 每路丢帧 | 总丢帧 |
| --- | --- | --- | --- | ---: |
| intra=1 | 41.038/225.708 ms | 0.457224 | 2,1,2,1 | 6 |
| intra=2 | 39.781/387.264 ms | 0.471607 | 32,27,25,28 | 112 |

线程数 2 略微降低平均在线延迟，但显著放大尾延迟。`OnlineFrameQueue(maxsize=1)` 每 100 ms 接收一帧；当 Zipformer 消费者发生 300 ms 以上延迟尖峰时，未消费帧会被连续覆盖，因此出现大量丢帧。

建议恢复 `SPACEMIT_EP_INTRA_THREAD_NUM=1`。当前 6/8 路本来就是 1，因此实际改动应是所有支持路数统一使用 1；保留每路独立 config 和 `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1`。

## 已知正常基线

`docs/reports/2026-07-12-qwen3-lane-matrix-performance.md` 记录的测试提交为 `4787eca`：

- Qwen3-ASR 0.6B q40，4 路，连续三次。
- Online 平均 41.627 ms，最大值均值 208.492 ms。
- 三次总丢帧分别为 1、1、0，平均 0.667。
- 当时模型目录中的 `zipformer.config` 使用 `SPACEMIT_EP_INTRA_THREAD_NUM=1`。

当前 token 时间戳实现的两轮测试使用 intra=2，总丢帧为 174 和 150，Online 最大延迟为 375.326 ms 和 587.714 ms。该现象与严格 A/B 中 intra=2 的结果一致。

## 今日提交影响整理

### 直接根因

| 提交 | 时间 | 影响 |
| --- | --- | --- |
| `9b9dd18` | 15:20 | 为每路生成 provider config；4 路时将 intra 从 1 改为 2。严格 A/B 已确认这是丢帧回归来源。 |

### 曾有可能影响，但已被同提交 A/B 排除

| 提交 | 时间 | 变更 | 判断 |
| --- | --- | --- | --- |
| `853eff4` | 10:04 | 完成后回到 loaded 状态 | 不改变首轮在线消费者热路径。 |
| `dc83367` | 10:08 | lane 模型跨 run 复用；命令 watcher 改为 10 ms 轮询 | 可能增加少量事件循环负载，但 intra=1 时总丢帧为 6，不能解释 112 帧回归。 |
| `f1fbff9` | 10:13 | lane 进程跨 run 保留 | 主要改变 run 结束后的生命周期。 |
| `c1da6bb` | 10:38 | Zipformer worker 跨 run 复用 | worker 执行线程未改变；intra 单变量测试已排除。 |

### 不影响本地 WAV 在线热路径

| 提交 | 时间 | 判断 |
| --- | --- | --- |
| `2720a58`, `f611822` | 01:05, 09:13 | 仅性能报告。 |
| `b196d53`, `310a4a3` | 09:59, 10:03 | 仅 spec/plan。 |
| `41603d5` | 10:17 | 前端 lane 输入解锁。 |
| `5fb7900` | 10:37 | 麦克风停止路径。 |
| `cd28259`, `e6f44c4` | 11:00, 11:13 | 模型路径 UI 和麦克风布局。 |
| `013c3da` | 11:29 | 开始前验证麦克风；本次使用本地 WAV。 |
| `496679a` | 14:16 | setup 脚本；模板中的 intra 仍为 1。 |
| `1195953`, `9ce077b` | 16:22, 16:29 | 仅 token 切分 spec/plan。 |
| `8faa69c` | 16:32 | 配置和内部字段契约，不增加每帧计算。 |
| `2785c7a` | 16:35 | SegmentCache 切分，仅 endpoint 时执行。 |
| `e9179a2` | 16:39 | 读取 token 时间戳；位于已出现回归的 `9ce077b` 之后。 |
| `29f7ffd` | 16:41 | 删除误跟踪的任务报告。 |
| `ad61539` | 16:47 | endpoint 音频切分和离线提交。 |
| `9d87041`, `5b58353` | 16:58, 17:04 | 自然结束和手动停止路径。 |
| `a6e991c` | 17:22 | endpoint 时间戳溢出保护和测试补强。 |

## 资源竞争证据

开发板运行期间采样 `/proc/<pid>/task/*/status` 和线程 CPU 使用情况：

- 四个 lane 进程的全部线程均允许运行在 CPU `0-7`。
- Qwen 服务采样时有 33 个线程允许运行在 `0-7`，另有各 2 个线程固定在 `8` 和 `9`。
- intra=1 时，每路观察到一个主要 Zipformer 执行线程。
- intra=2 时，每路观察到主执行线程之外的第二个活跃 EP worker。
- Event loop lag 平均值几乎不变：intra=1 为 1.792 ms，intra=2 为 1.820 ms；异常主要体现在 Zipformer online 最大延迟，而不是 Python event loop 平均负载。

SpaceMIT EP 的 `SpineThreadPool::enqueueInternal()` 在主线程提交任务后等待 worker 完成。增加到两个 intra 执行线程后，四路共增加四个活跃 worker；这些 worker 与 Qwen host 线程、音频生产者和事件循环竞争 CPU `0-7`，放大调度尾延迟。主要竞争资源是宿主 CPU 调度时间；缓存和内存带宽可能是次要因素，本次没有独立测量，不能作为已确认根因。

Zipformer 指定 X100、Qwen 指定 A100 可以避免 AI 计算核心直接冲突，但不能消除两者 host/control 线程在 Linux CPU `0-7` 上的竞争。

## 测试方法

- 开发板 K3，Python 3.13 项目 `.venv313`。
- 固定提交 `9ce077b`、4 路、本地 `~/Music/wav16k`、Qwen3-ASR 0.6B q40、`n_parallel=4`、`n_ctx=8192`。
- 两轮之间清理所有测试父进程、multiprocessing 子进程和 Qwen 服务。
- 使用绝对 `PYTHONPATH=<临时源码>/src`，并检查四份运行时 `zipformer.config` 后再采信数据。
- intra=1 artifact：`/tmp/token-timestamp-threads1-runs/process-1783936878240`。
- intra=2 artifact：`/tmp/token-timestamp-baseline-runs/process-1783937168913`。

调查过程中有两轮数据被明确作废：第一次临时源码使用 `PYTHONPATH=.`，实际导入了主仓库 editable install；第二次受上一轮孤儿 lane 进程干扰。这两轮均未用于结论。

## 后续验证建议

1. 将运行时生成配置恢复为 intra=1。
2. 在当前 token 时间戳版本重新执行至少三次 4 路真实测试。
3. 验收每轮总丢帧小于 10，同时记录 Online 最大延迟；仅看平均延迟无法发现本次回归。
4. 若后续仍要尝试 intra=2，应先隔离 Qwen host/control 线程与 Zipformer worker 的 Linux CPU 集合，再重新测试尾延迟。
