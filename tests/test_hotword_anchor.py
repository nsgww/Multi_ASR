"""热词锚定：校正结果不得把在线文本已识别出的热词改错（2026-11-26 朗吉案）。"""

from multi_stream_asr.asr.offline.qwen3asr import anchor_hotwords


def test_anchor_replaces_homophone_mistake():
    online = "广州市朗吉电子科技有限公司。"
    corrected = "广州市朗极电子科技有限公司。"
    assert anchor_hotwords(online, corrected, ("朗吉",)) == online


def test_anchor_replaces_another_homophone():
    online = "广州市朗吉电子科技有限公司。"
    corrected = "广州市榄吉电子科技有限公司。"
    assert anchor_hotwords(online, corrected, ("朗吉",)) == online


def test_anchor_keeps_correct_result_unchanged():
    online = "广州市朗吉电子科技有限公司。"
    assert anchor_hotwords(online, online, ("朗吉",)) == online


def test_anchor_skips_when_length_differs():
    # 校正增删了字，位置对齐不可靠，保守跳过
    online = "广州市朗吉电子科技有限公司。"
    corrected = "广州朗吉电子科技公司。"
    assert anchor_hotwords(online, corrected, ("朗吉",)) == corrected


def test_anchor_without_hotword_in_online():
    online = "今天我们开会讨论方案。"
    corrected = "今天我们开会讨论方案。"
    assert anchor_hotwords(online, corrected, ("朗吉",)) == corrected


def test_anchor_multiple_occurrences():
    online = "朗吉公司说朗吉的产品好。"
    corrected = "朗极公司说榄吉的产品好。"
    assert anchor_hotwords(online, corrected, ("朗吉",)) == online


def test_anchor_single_char_hotword_ignored():
    # 单字热词误锚定风险大（同音字遍地），不处理
    online = "他说好的。"
    corrected = "他说好德。"
    assert anchor_hotwords(online, corrected, ("好",)) == corrected
