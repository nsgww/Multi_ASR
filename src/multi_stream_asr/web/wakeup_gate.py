from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class WakeupSettings:
    """唤醒门控的运行时共享配置（接口可调，所有会话即时生效）。

    window_seconds: 唤醒后限时转写秒数；0 = 不限时（唤醒后一直转写）。
    """

    window_seconds: float = 30.0


class WakeupGate:
    """语音唤醒门控：KWS 命中前抑制所有 transcript 消息。

    状态机::

        WAITING --(收到 kws)--> ACTIVE --(倒计时到期)--> SLEEP(回到 WAITING)

    - WAITING: 音频照常进 ASR/KWS, 但 transcript 一律不下发/不累积
    - 切到 ACTIVE 时, 抑制 ``sentence_id <= 切换时已见最大值`` 的 transcript,
      使唤醒词所在句不会出现在转写结果里
    - ACTIVE 期间每次 KWS 命中都会刷新倒计时（默认 30 秒, 见 WakeupSettings）
    - 倒计时到期回到 WAITING 并下发 status=sleep 提示（不进转写/落盘/纪要）;
      到期时刻正在进行的句子允许收尾(final 正常下发), 新句子继续被抑制
    - ``process()`` 返回应转发的消息列表
    """

    def __init__(
        self,
        enabled: bool = True,
        active: bool = False,
        settings: WakeupSettings | None = None,
    ) -> None:
        self._enabled = enabled
        self._active = active or not enabled
        self._settings = settings or WakeupSettings()
        self._max_seen_sentence_id = -1
        self._suppress_until_sentence_id = -1
        self._deadline: float | None = None
        self._grace_sentence_id = -1  # 到期后允许收尾的最大 sentence_id

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """外部同步激活(如组播重连挂回到已唤醒的 job)。"""
        self._active = True
        self._arm_deadline()

    def reset(self) -> None:
        self._active = not self._enabled
        self._max_seen_sentence_id = -1
        self._suppress_until_sentence_id = -1
        self._deadline = None
        self._grace_sentence_id = -1

    def _arm_deadline(self) -> None:
        window = self._settings.window_seconds
        self._deadline = time.monotonic() + window if window > 0 else None

    def _expire(self, message: dict[str, Any]) -> dict[str, Any]:
        """倒计时到期：回待唤醒状态, 记录允许收尾的句子。"""
        self._active = False
        self._deadline = None
        self._grace_sentence_id = self._max_seen_sentence_id
        sleep_status: dict[str, Any] = {
            "type": "status",
            "status": "sleep",
            "message": "转写时间已到，回到待唤醒状态",
        }
        if message.get("lane_id") is not None:
            sleep_status["lane_id"] = message["lane_id"]
        return sleep_status

    def process(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._enabled:
            return [message]

        # 倒计时到期检查（消息驱动, 先到期先回 WAITING）
        prefix: list[dict[str, Any]] = []
        if self._active and self._deadline is not None and time.monotonic() >= self._deadline:
            prefix.append(self._expire(message))

        msg_type = message.get("type")
        if msg_type == "kws":
            if not self._active:
                self._active = True
                self._grace_sentence_id = -1
                self._suppress_until_sentence_id = self._max_seen_sentence_id
                self._arm_deadline()
                # 唤醒提示只作为 status 消息下发, 不进入转写文本/落盘/纪要
                wakeup_status = {
                    "type": "status",
                    "status": "wakeup",
                    "message": "现在开始接收数据转写",
                    "keyword": message.get("keyword", ""),
                }
                if message.get("lane_id") is not None:
                    wakeup_status["lane_id"] = message["lane_id"]
                return prefix + [wakeup_status, message]
            # 已激活时再次命中: 刷新倒计时
            self._arm_deadline()
            return prefix + [message]
        if msg_type == "transcript":
            sid = message.get("sentence_id")
            if isinstance(sid, int) and sid > self._max_seen_sentence_id:
                self._max_seen_sentence_id = sid
            if not self._active:
                # 到期时刻正在进行的句子允许收尾; 之后的句子继续抑制
                if isinstance(sid, int) and sid <= self._grace_sentence_id:
                    return prefix + [message]
                return prefix
            if isinstance(sid, int) and sid <= self._suppress_until_sentence_id:
                return prefix  # 唤醒词所在句(命中时刻正在进行的句子)不下发
            return prefix + [message]
        return prefix + [message]
