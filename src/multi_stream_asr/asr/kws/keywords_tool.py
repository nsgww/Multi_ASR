from __future__ import annotations

import logging
from pathlib import Path

_logger = logging.getLogger(__name__)


def words_to_keywords_lines(words: list[str]) -> list[str]:
    """把中文唤醒词转成 sherpa KWS keywords.txt 的行格式（声母 韵母 @词）。

    已含 ``@`` 的条目视为已完成 token 化，原样保留。
    多音字取第一个读音；无法转换的词抛 ValueError。
    """
    lines: list[str] = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if "@" in word:
            lines.append(word)
            continue
        lines.append(f"{_word_to_tokens(word)} @{word}")
    return lines


def _word_to_tokens(word: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise RuntimeError("pypinyin 未安装，无法自动转换中文唤醒词") from exc

    import re

    syllables = lazy_pinyin(word, style=Style.TONE, errors="keep")
    initial_re = re.compile(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])")
    tokens: list[str] = []
    for char, syllable in zip(word, syllables, strict=True):
        if not ("\u4e00" <= char <= "\u9fff"):
            raise ValueError(f"唤醒词 {word!r} 含非中文字符 {char!r}，请用 tokens@词 原始格式")
        match = initial_re.match(syllable)
        if match is None:
            # 零声母（如"爱 ài"）：整韵母单 token
            tokens.append(syllable)
            continue
        initial = match.group(1)
        final = syllable[match.end() :]
        if initial in {"j", "q", "x", "y"}:
            # j/q/x/y 后的 ü 按标准拼写简化为 u（军 j+ūn、学 x+ué、杨 y+áng）
            final = final.translate(str.maketrans("üǖǘǚǜ", "uūúǔù"))
        tokens.append(initial)
        tokens.append(final)
    return " ".join(tokens)


def keywords_file_to_words(path: Path | str) -> list[str]:
    """从 keywords.txt 提取可展示的词（取 @ 后面的文本；无 @ 的行原样保留）。"""
    path = Path(path)
    if not path.is_file():
        return []
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line.split("@", 1)[1].strip() if "@" in line else line)
    return words


def write_word_list(path: Path | str, lines: list[str], header: str = "") -> None:
    """写词表文件（带可选注释头），供运行时装载。"""
    content = (header + "\n" if header else "") + "\n".join(lines) + "\n"
    Path(path).write_text(content, encoding="utf-8")


def find_invalid_keyword_tokens(
    lines: list[str], tokens_file: Path | str
) -> dict[str, list[str]]:
    """校验 keywords 行的 token 是否都在 KWS 模型的 tokens.txt 里。

    sherpa 在 C++ 层遇到未知 token 会直接终止进程（``Encode keywords
    failed`` 后 abort），所以写文件/热重载前必须先在 Python 侧拦住。
    返回 ``{原始行: [缺失的 token, ...]}``，全部合法时为空 dict。
    """
    tokens_path = Path(tokens_file)
    if not tokens_path.is_file():
        _logger.warning("KWS tokens.txt 不存在，跳过校验: %s", tokens_path)
        return {}
    valid = {
        line.split(" ", 1)[0]
        for line in tokens_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    invalid: dict[str, list[str]] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        left = line.split("@", 1)[0].strip()
        missing = [tok for tok in left.split() if tok not in valid]
        if missing:
            invalid[line] = missing
    return invalid
