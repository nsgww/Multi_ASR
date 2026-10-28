"""说话人分离（diarization）与声纹识别（speaker recognition）。

设计约束（来自现场：单通道全向麦、8 核 RISC-V 板、已有 4 个 LLM 常驻）：

* **句级归属，不做帧级分离**。pipeline 的 VAD 已经切好句，直接复用句段 PCM。
* **与离线 ASR 并行**。离线精修 1.5~5s，声纹只有 0.1~0.3s，净增延迟≈0。
* **两遍聚类**。第一遍逐句在线聚类给实时体验；第二遍在会话末用全部句向量
  全局重聚类纠正早期错分。
* **可降级**。模型缺失、音频过短、推理异常都不允许影响转写主链路。

``SpeakerInfo`` 等数据模型定义在 :mod:`multi_stream_asr.models`，这里只做
转出，避免与 ``models.ASREvent`` 形成循环依赖。
"""

from multi_stream_asr.models import SpeakerInfo, speaker_id, speaker_label
from multi_stream_asr.speaker.cluster import (
    ClusterAssignment,
    OnlineSpeakerClusterer,
    agglomerative_cluster,
)
from multi_stream_asr.speaker.embedder import (
    SherpaSpeakerEmbedder,
    SpeakerEmbedder,
    trim_silence,
)
from multi_stream_asr.speaker.preflight import (
    PreflightResult,
    clear_preflight_cache,
    verify_speaker_model,
)
from multi_stream_asr.speaker.registry import VoiceprintMatch, VoiceprintRegistry
from multi_stream_asr.speaker.tracker import SpeakerTracker

__all__ = [
    "ClusterAssignment",
    "OnlineSpeakerClusterer",
    "PreflightResult",
    "SherpaSpeakerEmbedder",
    "SpeakerEmbedder",
    "SpeakerInfo",
    "SpeakerTracker",
    "VoiceprintMatch",
    "VoiceprintRegistry",
    "agglomerative_cluster",
    "clear_preflight_cache",
    "speaker_id",
    "speaker_label",
    "trim_silence",
    "verify_speaker_model",
]
