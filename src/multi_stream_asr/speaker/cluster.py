"""说话人聚类：在线逐句聚类 + 会话末全局重聚类。

**第一遍（在线）**：每来一句就取 embedding，与已有簇质心算余弦，命中则并入并
更新质心，明显不像则开新簇。为了让标签不来回跳，做了两件事：

1. **粘滞奖励**：给"上一句的说话人"加一个小的相似度加成，避免两个人轮流说
   且相似度接近时标签高频翻转。
2. **双阈值滞回**：``>= match_threshold`` 直接归入；``< new_threshold`` 才开
   新簇；落在中间的灰区归入最优簇但置信度偏低。

**第二遍（离线）**：会话结束时把整场每句的 embedding 拿去做平均链接 AHC，
纠正开头几句质心还没立稳造成的错分。纯 numpy，不需要额外模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from multi_stream_asr.speaker.embedder import l2_normalize


@dataclass(frozen=True)
class ClusterAssignment:
    """一次在线归属的结果。

    Attributes:
        index: 簇下标（0 起）。
        similarity: 与命中的簇质心的余弦相似度。新建簇记 1.0。
        created: 是否新建了簇。
    """

    index: int
    similarity: float
    created: bool


@dataclass
class _Cluster:
    centroid: np.ndarray
    count: int = 1


@dataclass
class OnlineSpeakerClusterer:
    """增量式说话人聚类。

    Args:
        match_threshold: 高于此相似度直接归入最优簇。
        new_threshold: 低于此相似度才允许开新簇。
        merge_threshold: 会话末合并簇的相似度门限。
        max_speakers: 最多允许的说话人数，超出后强制归入最优簇。
        sticky_bonus: 上一句说话人的相似度加成，抑制标签抖动。
    """

    match_threshold: float = 0.55
    new_threshold: float = 0.40
    merge_threshold: float = 0.70
    max_speakers: int = 8
    sticky_bonus: float = 0.05

    _clusters: list[_Cluster] = field(default_factory=list, init=False, repr=False)
    _last_index: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.new_threshold > self.match_threshold:
            raise ValueError("new_threshold must not exceed match_threshold")
        if self.max_speakers < 1:
            raise ValueError("max_speakers must be >= 1")

    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    @property
    def last_index(self) -> int:
        return self._last_index

    def centroid(self, index: int) -> np.ndarray:
        return self._clusters[index].centroid

    def assign(self, embedding: np.ndarray) -> ClusterAssignment:
        """把一个 embedding 归到某个簇（必要时新建）。"""
        raw = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if raw.size == 0 or float(np.linalg.norm(raw)) <= 1e-8:
            raise ValueError("embedding must not be empty or all-zero")
        emb = l2_normalize(raw)

        if not self._clusters:
            self._clusters.append(_Cluster(centroid=emb))
            self._last_index = 0
            return ClusterAssignment(index=0, similarity=1.0, created=True)

        sims = np.array(
            [float(np.dot(emb, c.centroid)) for c in self._clusters],
            dtype=np.float32,
        )
        scores = sims.copy()
        if 0 <= self._last_index < scores.size:
            scores[self._last_index] += self.sticky_bonus
        best = int(np.argmax(scores))
        best_sim = float(sims[best])

        if best_sim < self.new_threshold and len(self._clusters) < self.max_speakers:
            self._clusters.append(_Cluster(centroid=emb))
            self._last_index = len(self._clusters) - 1
            return ClusterAssignment(index=self._last_index, similarity=1.0, created=True)

        self._absorb(best, emb)
        self._last_index = best
        return ClusterAssignment(index=best, similarity=best_sim, created=False)

    def inherit_last(self) -> ClusterAssignment | None:
        """短句/提特征失败时沿用上一句的说话人。"""
        if self._last_index < 0 or not self._clusters:
            return None
        cluster = self._clusters[self._last_index]
        return ClusterAssignment(
            index=self._last_index,
            similarity=max(0.0, 1.0 - 1.0 / (cluster.count + 1)),
            created=False,
        )

    def merge_similar(self, threshold: float | None = None) -> dict[int, int]:
        """返回 ``旧簇下标 -> 代表簇下标`` 的映射（不做状态变更）。

        仅在会话末统计/重标用；在线阶段不调用，避免质心被破坏。
        """
        thr = self.merge_threshold if threshold is None else float(threshold)
        n = len(self._clusters)
        if n < 2:
            return {i: i for i in range(n)}

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(self._clusters[i].centroid, self._clusters[j].centroid))
                if sim >= thr:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

        return {i: find(i) for i in range(n)}

    def _absorb(self, index: int, emb: np.ndarray) -> None:
        cluster = self._clusters[index]
        merged = cluster.centroid * cluster.count + emb
        cluster.centroid = l2_normalize(merged)
        cluster.count += 1


def agglomerative_cluster(
    embeddings: list[np.ndarray],
    threshold: float,
    *,
    max_iterations: int = 512,
) -> list[int]:
    """平均链接层次聚类（会话末第二遍用）。

    Args:
        embeddings: 每句一个向量，任意维度但需一致。
        threshold: 簇间平均余弦相似度高于此值即合并。
        max_iterations: 安全上限，防止异常输入长时间占用。

    Returns:
        与 ``embeddings`` 等长的簇标签（0 起，按首次出现顺序编号）。
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    matrix = np.stack([l2_normalize(e) for e in embeddings])
    sim = matrix @ matrix.T

    groups: list[list[int]] = [[i] for i in range(n)]
    active: list[int] = list(range(n))

    for _ in range(max_iterations):
        if len(active) < 2:
            break
        best_sim = -np.inf
        best_pair: tuple[int, int] | None = None
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                ia, ib = active[a], active[b]
                block = sim[np.ix_(groups[ia], groups[ib])]
                score = float(block.mean())
                if score > best_sim:
                    best_sim = score
                    best_pair = (a, b)
        if best_pair is None or best_sim < threshold:
            break
        a, b = best_pair
        ia, ib = active[a], active[b]
        groups[ia].extend(groups[ib])
        groups[ib] = []
        active.pop(b)

    raw = [0] * n
    for label, idx in enumerate(active):
        for item in groups[idx]:
            raw[item] = label

    # 按首次出现顺序重新编号，保证 S1 总是第一个开口的人
    remap: dict[int, int] = {}
    labels: list[int] = []
    for label in raw:
        if label not in remap:
            remap[label] = len(remap)
        labels.append(remap[label])
    return labels
