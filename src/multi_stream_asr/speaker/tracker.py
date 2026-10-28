"""每路（lane）的说话人跟踪器：把 embedder / clusterer / registry 组装起来。

调用约定：

* **每路一个实例**，内部状态（簇质心、上一句说话人、句向量池）不跨路共享。
* **必须在单线程里串行调用 ``assign``**。pipeline 里离线任务是按 lane 串行
  排队的，天然满足；不要并发调用同一个 tracker。
* 任何异常都必须降级成"沿用上一句"或 ``None``，绝不能冒泡打断转写。
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from multi_stream_asr.config import SpeakerConfig
from multi_stream_asr.models import SpeakerInfo, speaker_id, speaker_label
from multi_stream_asr.speaker.cluster import (
    ClusterAssignment,
    OnlineSpeakerClusterer,
    agglomerative_cluster,
)
from multi_stream_asr.speaker.embedder import SpeakerEmbedder
from multi_stream_asr.speaker.registry import VoiceprintRegistry

_logger = logging.getLogger(__name__)


class SpeakerTracker:
    """一条音频流的说话人跟踪状态机。"""

    def __init__(
        self,
        *,
        embedder: SpeakerEmbedder | None,
        config: SpeakerConfig,
        registry: VoiceprintRegistry | None = None,
        stream_id: int = 0,
    ) -> None:
        self.stream_id = stream_id
        self._embedder = embedder
        self._config = config
        self._registry = registry
        self._clusterer = OnlineSpeakerClusterer(
            match_threshold=config.match_threshold,
            new_threshold=config.new_speaker_threshold,
            merge_threshold=config.merge_threshold,
            max_speakers=config.max_speakers,
            sticky_bonus=config.sticky_bonus,
        )
        self._embeddings: list[np.ndarray | None] = []
        self._cluster_of: list[int] = []
        self._last: SpeakerInfo | None = None
        self._inherited = 0
        self._failed = 0

    # -- 状态 -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """模型可用才为 True；不可用时上层应完全跳过声纹。"""
        if self._embedder is None:
            return False
        return bool(getattr(self._embedder, "available", True))

    @property
    def speaker_count(self) -> int:
        return self._clusterer.cluster_count

    @property
    def last_speaker(self) -> SpeakerInfo | None:
        return self._last

    # -- 主入口 -----------------------------------------------------------

    def assign(self, pcm: np.ndarray) -> SpeakerInfo | None:
        """给一个 VAD 句段判定说话人。

        返回 ``None`` 表示无法判定（声纹关闭 / 首次调用就提不出特征 /
        异常）。上层应把 ``None`` 当作"未知说话人"处理，不要当成错误。
        """
        if self._embedder is None:
            return None

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        min_samples = int(self._config.min_duration_seconds * self._config.sample_rate)
        if audio.size < min_samples:
            return self._inherit(reason="too_short")

        try:
            embedding = self._embedder.embed(audio)
        except Exception as exc:  # noqa: BLE001 - 声纹失败不能影响转写
            self._failed += 1
            _logger.debug("声纹提取异常 lane=%d: %s", self.stream_id, exc)
            return self._inherit(reason="embed_error")

        if embedding is None:
            self._failed += 1
            return self._inherit(reason="embed_empty")

        try:
            assignment = self._clusterer.assign(embedding)
        except Exception as exc:  # noqa: BLE001
            self._failed += 1
            _logger.debug("说话人聚类异常 lane=%d: %s", self.stream_id, exc)
            return self._inherit(reason="cluster_error")

        self._embeddings.append(embedding)
        self._cluster_of.append(assignment.index)
        info = self._build_info(assignment, embedding)
        self._last = info
        return info

    # -- 二遍重聚类 -------------------------------------------------------

    def resegment(self) -> dict[str, str]:
        """会话末用全部句向量做全局重聚类，返回 ``旧标签 -> 新标签``。

        只返回**发生变化**的映射。调用方必须**原子地**一次性套用整张映射表
        （不能逐条链式替换），否则 ``S1->S2`` 与 ``S2->S1`` 这类互换会互相
        覆盖。
        """
        if not self._config.enable_second_pass:
            return {}

        indexed = [(pos, emb) for pos, emb in enumerate(self._embeddings) if emb is not None]
        if len(indexed) < 3:
            return {}

        labels = agglomerative_cluster(
            [emb for _, emb in indexed],
            self._config.merge_threshold,
        )

        votes: dict[int, Counter[int]] = {}
        for k, (pos, _) in enumerate(indexed):
            old = self._cluster_of[pos]
            votes.setdefault(old, Counter())[labels[k]] += 1
        if not votes:
            return {}

        old_to_new = {
            old: counter.most_common(1)[0][0] for old, counter in votes.items()
        }

        mapping: dict[str, str] = {}
        for old, new in sorted(old_to_new.items()):
            old_label, new_label = speaker_label(old), speaker_label(new)
            if old_label != new_label:
                mapping[old_label] = new_label

        if mapping:
            _logger.info(
                "二遍重聚类 lane=%d: %d 句 -> %d 人, 重标 %s",
                self.stream_id,
                len(indexed),
                len(set(old_to_new.values())),
                mapping,
            )
        return mapping

    def summary(self) -> dict[str, object]:
        """给日志/指标用的摘要。"""
        return {
            "stream_id": self.stream_id,
            "enabled": self.enabled,
            "sentences": len(self._cluster_of),
            "speakers": self._clusterer.cluster_count,
            "inherited": self._inherited,
            "failed": self._failed,
        }

    # -- 内部 -------------------------------------------------------------

    def _build_info(
        self, assignment: ClusterAssignment, embedding: np.ndarray
    ) -> SpeakerInfo:
        index = assignment.index
        name: str | None = None
        confidence = (
            1.0 if assignment.created else max(0.0, min(1.0, assignment.similarity))
        )
        provisional = assignment.created

        registry = self._registry
        if registry is not None and len(registry) > 0:
            match = registry.identify(
                embedding,
                threshold=self._config.identify_threshold,
                margin=self._config.identify_margin,
            )
            if match is not None:
                name = match.name
                confidence = match.score
                provisional = False

        return SpeakerInfo(
            id=speaker_id(index),
            label=speaker_label(index),
            name=name,
            confidence=confidence,
            is_provisional=provisional,
        )

    def _inherit(self, *, reason: str) -> SpeakerInfo | None:
        assignment = self._clusterer.inherit_last()
        if assignment is None:
            return None
        self._inherited += 1
        self._embeddings.append(None)
        self._cluster_of.append(assignment.index)
        _logger.debug(
            "说话人沿用上一句 lane=%d reason=%s -> %s",
            self.stream_id,
            reason,
            speaker_label(assignment.index),
        )
        inherited = self._last
        if inherited is None:
            return SpeakerInfo(
                id=speaker_id(assignment.index),
                label=speaker_label(assignment.index),
                confidence=max(0.0, min(1.0, assignment.similarity)),
                is_provisional=True,
            )
        return SpeakerInfo(
            id=inherited.id,
            label=inherited.label,
            name=inherited.name,
            confidence=inherited.confidence,
            is_provisional=True,
        )
