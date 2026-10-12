from __future__ import annotations

import logging
import re
from pathlib import Path

_logger = logging.getLogger(__name__)

REPLACEMENT = "***"


class SensitiveWordFilter:
    """敏感词过滤器：把命中词替换为固定 ``***``。

    词表文件为 UTF-8 文本，一行一词，``#`` 开头为注释，空文件 = 不过滤直通。
    中文按子串精确匹配；纯 ASCII 词按大小写不敏感匹配。
    """

    def __init__(self, words: list[str] | None = None) -> None:
        self._words = [w.strip() for w in (words or []) if w and w.strip()]
        self._patterns: list[re.Pattern[str]] = []
        for w in self._words:
            if w.isascii():
                self._patterns.append(re.compile(re.escape(w), re.IGNORECASE))
            else:
                self._patterns.append(re.compile(re.escape(w)))

    @classmethod
    def from_file(cls, path: Path | str) -> "SensitiveWordFilter":
        path = Path(path)
        if not path.is_file():
            _logger.info("sensitive words file not found: %s; filtering disabled", path)
            return cls([])
        words: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line)
        _logger.info("sensitive words loaded: %d entries from %s", len(words), path)
        return cls(words)

    @property
    def enabled(self) -> bool:
        return bool(self._patterns)

    @property
    def words(self) -> list[str]:
        return list(self._words)

    def update_words(self, words: list[str]) -> None:
        """运行时热更新词表（持有同一对象的各 handler 立即生效）。"""
        updated = SensitiveWordFilter(words)
        self._words = updated._words
        self._patterns = updated._patterns
        _logger.info("sensitive words updated at runtime: %d entries", len(self._words))

    def filter(self, text: str) -> str:
        if not text or not self._patterns:
            return text
        for pattern in self._patterns:
            text = pattern.sub(REPLACEMENT, text)
        return text
