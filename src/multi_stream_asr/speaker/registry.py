"""声纹库：注册、1:N 识别、持久化。

存储布局（目录可配）::

    <registry_dir>/
        index.json        # 姓名 -> 文件名 + 条数
        spk_0.npy         # 该姓名的全部 embedding，shape=(k, dim)

用``spk_N.npy``而不是拿姓名当文件名，避免中文/特殊字符与路径穿越问题；
``index.json`` 负责映射。

识别用**均值质心 + 双门限**：既要超过 ``threshold``，又要领先第二名
``margin``，否则判为未注册。单门限在会议室里会把相似音色的人认错。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multi_stream_asr.speaker.embedder import l2_normalize

_logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
INDEX_VERSION = 1
MAX_PRINTS_PER_NAME = 8


@dataclass(frozen=True)
class VoiceprintMatch:
    """一次 1:N 命中的结果。"""

    name: str
    score: float
    margin: float


class VoiceprintRegistry:
    """线程安全的声纹库（多 lane 共享同一个实例）。"""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        max_prints_per_name: int = MAX_PRINTS_PER_NAME,
    ) -> None:
        self._directory = Path(directory) if directory is not None else None
        self._max_prints = max(int(max_prints_per_name), 1)
        self._lock = threading.RLock()
        self._prints: dict[str, list[np.ndarray]] = {}
        self._files: dict[str, str] = {}
        if self._directory is not None:
            self._load()

    # -- 查询 -------------------------------------------------------------

    def reload(self) -> None:
        """从磁盘重新加载（REST 注册/删除后由各 lane 热重载）。"""
        if self._directory is None:
            return
        with self._lock:
            self._prints = {}
            self._files = {}
            self._load()

    @property
    def directory(self) -> Path | None:
        return self._directory

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._prints)

    def count(self, name: str) -> int:
        with self._lock:
            return len(self._prints.get(name, ()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._prints)

    # -- 注册 / 删除 -------------------------------------------------------

    def enroll(self, name: str, embedding: np.ndarray) -> int:
        """追加一条声纹，返回该姓名当前的条数。"""
        key = name.strip()
        if not key:
            raise ValueError("name must not be empty")
        raw = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if raw.size == 0 or float(np.linalg.norm(raw)) <= 1e-8:
            raise ValueError("embedding must not be empty or all-zero")
        vec = l2_normalize(raw)

        with self._lock:
            prints = self._prints.setdefault(key, [])
            if prints and prints[0].shape != vec.shape:
                raise ValueError(
                    f"embedding dim mismatch for {key!r}: "
                    f"{vec.shape} vs {prints[0].shape}"
                )
            prints.append(vec)
            if len(prints) > self._max_prints:
                del prints[: len(prints) - self._max_prints]
            if key not in self._files:
                self._files[key] = self._next_filename_locked()
            self._save_locked()
            _logger.info("声纹注册: %s 共 %d 条", key, len(prints))
            return len(prints)

    def _next_filename_locked(self) -> str:
        """取最小的未占用文件名。不能用 ``len(self._files)``——删过人之后
        会与已存在的 ``spk_N.npy`` 撞名。"""
        used = set(self._files.values())
        index = 0
        while f"spk_{index}.npy" in used:
            index += 1
        return f"spk_{index}.npy"

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self._prints:
                return False
            self._prints.pop(name, None)
            filename = self._files.pop(name, None)
            if filename and self._directory is not None:
                try:
                    (self._directory / filename).unlink(missing_ok=True)
                except OSError:
                    _logger.warning("删除声纹文件失败: %s", filename)
            self._save_locked()
            return True

    # -- 识别 -------------------------------------------------------------

    def score_all(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """返回每个已注册姓名的余弦得分（降序）。诊断用，不做门限判断。"""
        raw = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if raw.size == 0 or float(np.linalg.norm(raw)) <= 1e-8:
            return []
        vec = l2_normalize(raw)

        with self._lock:
            gallery = {k: list(v) for k, v in self._prints.items()}

        scored: list[tuple[str, float]] = []
        for name, prints in gallery.items():
            if not prints or prints[0].shape != vec.shape:
                continue
            centroid = l2_normalize(np.mean(np.stack(prints), axis=0))
            scored.append((name, float(np.dot(vec, centroid))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def identify(
        self,
        embedding: np.ndarray,
        *,
        threshold: float = 0.55,
        margin: float = 0.05,
    ) -> VoiceprintMatch | None:
        """1:N 识别。未达门限或领先优势不足时返回 ``None``。"""
        scored = self.score_all(embedding)
        if not scored:
            return None

        best_name, best_score = scored[0]
        second = scored[1][1] if len(scored) > 1 else -1.0
        gap = best_score - second

        if best_score < threshold:
            return None
        if gap < margin:
            return None
        return VoiceprintMatch(name=best_name, score=best_score, margin=gap)

    # -- 持久化 -----------------------------------------------------------

    def _load(self) -> None:
        assert self._directory is not None
        index_path = self._directory / INDEX_FILENAME
        if not index_path.is_file():
            return
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _logger.exception("声纹索引读取失败，按空库处理: %s", index_path)
            return

        entries = payload.get("speakers") or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            filename = str(entry.get("file") or "").strip()
            if not name or not filename:
                continue
            # 防止索引被改写后指向目录外
            if Path(filename).name != filename:
                _logger.warning("声纹索引含非法文件名，跳过: %r", filename)
                continue
            array_path = self._directory / filename
            try:
                array = np.load(array_path, allow_pickle=False)
            except (OSError, ValueError):
                _logger.warning("声纹文件读取失败，跳过: %s", array_path)
                continue
            matrix = np.atleast_2d(array).astype(np.float32)
            self._prints[name] = [l2_normalize(row) for row in matrix]
            self._files[name] = filename
        if self._prints:
            _logger.info("声纹库载入 %d 人: %s", len(self._prints), ", ".join(sorted(self._prints)))

    def _save_locked(self) -> None:
        if self._directory is None:
            return
        assert self._directory is not None
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            for name, prints in self._prints.items():
                if not prints:
                    continue
                filename = self._files[name]
                np.save(self._directory / filename, np.stack(prints))
            payload = {
                "version": INDEX_VERSION,
                "speakers": [
                    {"name": name, "file": self._files[name], "count": len(prints)}
                    for name, prints in sorted(self._prints.items())
                    if prints
                ],
            }
            tmp = self._directory / f"{INDEX_FILENAME}.tmp"
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self._directory / INDEX_FILENAME)
        except OSError:
            _logger.exception("声纹库保存失败: %s", self._directory)
