from __future__ import annotations

import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

_WORD_BEGIN = "▁"  # U+2581, sentencepiece 词首标记（英文 BPE 词带此前缀）


def load_hotwords(path: Path | str) -> list[str]:
    """读取人类可读热词表：UTF-8 一行一词，``#`` 开头为注释。

    文件不存在或为空时返回空列表 = 热词功能整体关闭。
    """
    path = Path(path)
    if not path.is_file():
        _logger.info("hotwords file not found: %s; hotwords disabled", path)
        return []
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)
    _logger.info("hotwords loaded: %d entries from %s", len(words), path)
    return words


def _load_vocab(tokens_path: Path | str) -> set[str]:
    vocab: set[str] = set()
    for line in Path(tokens_path).read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line:
            continue
        # tokens.txt 每行 "<token> <id>"，token 本身不含空白
        token = line.rsplit(maxsplit=1)[0] if len(line.split()) > 1 else line
        vocab.add(token)
    return vocab


def _tokenize_ascii(segment: str, vocab: set[str]) -> list[str] | None:
    """纯 ASCII 词按 BPE 贪心最长匹配切分（模型英文词表为大写、词首带 ▁）。"""
    text = _WORD_BEGIN + segment.upper()
    tokens: list[str] = []
    i = 0
    while i < len(text):
        match = None
        for j in range(len(text), i, -1):
            piece = text[i:j]
            if piece in vocab:
                match = piece
                break
        if match is None:
            return None
        tokens.append(match)
        i += len(match)
    return tokens


def _tokenize_cjk(word: str, vocab: set[str], *, at_word_start: bool) -> list[str] | None:
    """中文按字切分（该双语模型的中文词表为单字 token）。"""
    tokens: list[str] = []
    for index, ch in enumerate(word):
        if ch in vocab:
            tokens.append(ch)
        elif index == 0 and at_word_start and (_WORD_BEGIN + ch) in vocab:
            tokens.append(_WORD_BEGIN + ch)
        else:
            return None
    return tokens


def tokenize_hotword(word: str, vocab: set[str]) -> list[str] | None:
    """把一条热词切成模型 token 序列；无法被词表覆盖时返回 None。"""
    tokens: list[str] = []
    for segment in word.split():
        if not segment:
            continue
        if segment.isascii():
            piece_tokens = _tokenize_ascii(segment, vocab)
        else:
            piece_tokens = _tokenize_cjk(segment, vocab, at_word_start=not tokens)
        if piece_tokens is None:
            return None
        tokens.extend(piece_tokens)
    return tokens or None


def build_sherpa_hotwords_file(
    words: list[str],
    tokens_path: Path | str,
    out_path: Path | str,
) -> Path | None:
    """把人类可读热词表转成 sherpa-onnx 要求的 token 化热词文件。

    sherpa-onnx 热词格式：一行一条热词，每条为其 BPE/CJK token 以空格分隔。
    无有效热词时返回 None（调用方应保持 greedy_search 直通）。
    """
    if not words:
        return None
    vocab = _load_vocab(tokens_path)
    lines: list[str] = []
    for word in words:
        tokens = tokenize_hotword(word, vocab)
        if tokens is None:
            _logger.warning(
                "hotword %r not covered by model vocab; skipped", word
            )
            continue
        lines.append(" ".join(tokens))
    if not lines:
        return None
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _logger.info("sherpa hotwords written: %d entries -> %s", len(lines), out_path)
    return out_path


def resolve_sherpa_hotwords(
    hotwords_file: Path | str,
    tokens_path: Path | str,
) -> tuple[Path | None, list[str]]:
    """读取热词表并生成 token 化文件，返回 (token化文件路径或None, 原始热词列表)。"""
    words = load_hotwords(hotwords_file)
    if not words:
        return None, []
    source = Path(hotwords_file)
    out_path = source.with_name(f"{source.stem}_bpe{source.suffix}")
    built = build_sherpa_hotwords_file(words, tokens_path, out_path)
    return built, words
