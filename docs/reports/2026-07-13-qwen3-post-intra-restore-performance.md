# Qwen3-ASR Intra Thread Restore Performance

## Scope

- 开发板：K3/A100，Python 3.13.12，当前分支提交 `fe0e22a`。
- Zipformer provider config：`SPACEMIT_EP_INTRA_THREAD_NUM = 1`、`SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064`、`SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1`。
- 输入：`~/Music/wav16k`，100 ms frame，rule1/rule2/rule3 为 1.0/0.5/10.0 s，token cut margin 为 0.1 s。
- 共享 Qwen：`n_parallel=4`、`n_ctx=8192`；每次 run 重新加载 Zipformer 和对应 Qwen 模型。
- 每个组合连续运行 3 次。每次验证全部 lane 为 `finished`、每路 Zipformer/Qwen sentence ID 集合一致、results 与 model_outputs 文件存在。

## 恢复验收

0.6B q40 的 4 路首轮使用 intra=1，丢帧为 `0,1,2,0`，总计 3，小于 10。三次完整平均总丢帧为 2.0。此前 intra=2 的严格 A/B 对照总丢帧为 112，因此恢复已消除该回归。

## 三次平均

| Model | Lanes | Wall s | Online avg ms | Online max ms | Offline RTF | Offline RTF max | Drops total | Drops per lane |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-ASR 0.6B q40 | 2 | 263.036 | 33.277 | 171.618 | 0.252686 | 1.068053 | 0.000 | 0.000, 0.000 |
| Qwen3-ASR 0.6B q40 | 4 | 264.976 | 40.931 | 209.743 | 0.496261 | 4.718754 | 2.000 | 0.333, 0.333, 0.667, 0.667 |
| Qwen3-ASR 1.7B q4km | 2 | 287.354 | 41.414 | 223.226 | 0.776473 | 4.417104 | 2.000 | 1.333, 0.667 |

`Online max ms` 和 `Offline RTF max` 是三次 run 最大值的算术平均；其余数值也为三次 run 的算术平均。`Drops total` 是全部 lane 总丢帧的三次平均。

## 原始样本

| Model | Lanes | Rep | Wall s | Online avg/max ms | Offline RTF avg/max | Drops per lane | Artifact |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0.6B q40 | 4 | 1 | 265.081 | 40.501/211.085 | 0.546332/8.583429 | 0,1,2,0 | `process-1783940207632` |
| 0.6B q40 | 4 | 2 | 264.869 | 40.382/205.185 | 0.451747/1.888319 | 1,0,0,1 | `process-1783940472657` |
| 0.6B q40 | 4 | 3 | 264.980 | 41.910/212.959 | 0.490705/3.684513 | 0,0,0,1 | `process-1783940737531` |
| 0.6B q40 | 2 | 1 | 263.119 | 33.214/174.769 | 0.248955/1.049332 | 0,0 | `process-1783941001497` |
| 0.6B q40 | 2 | 2 | 263.046 | 33.304/168.942 | 0.254505/1.075819 | 0,0 | `process-1783941264629` |
| 0.6B q40 | 2 | 3 | 262.943 | 33.315/171.143 | 0.254598/1.079008 | 0,0 | `process-1783941527561` |
| 1.7B q4km | 2 | 1 | 288.056 | 41.851/233.504 | 0.860206/8.353417 | 0,2 | `process-1783941814558` |
| 1.7B q4km | 2 | 2 | 286.924 | 41.038/211.538 | 0.728315/2.506514 | 3,0 | `process-1783942101553` |
| 1.7B q4km | 2 | 3 | 287.082 | 41.352/224.637 | 0.740898/2.391380 | 1,0 | `process-1783942388573` |

所有 artifact 位于开发板：`~/Workspace/projects/multi-stream-asr/runs/qwen3-post-intra-restore-20260713/`；机器可读汇总为该目录的 `summary.json`。

## Interpretation

- 0.6B 4 路恢复到低丢帧状态，online 平均 40.931 ms，和恢复前正常基线的 41.627 ms 同一量级。
- 0.6B 2 路在三次运行中零丢帧，offline RTF 为 0.252686。
- 1.7B 2 路保持低丢帧，但 offline RTF 为 0.776473，显著高于 0.6B；这是离线模型规模导致的吞吐代价，不是 Zipformer intra 配置回归。
