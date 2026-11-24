"""WebSocket 空闲心跳、发送超时与慢客户端保护。

**为什么需要应用层心跳**：uvicorn 的 ``--ws-ping-interval`` 发的是 WebSocket
协议级控制帧，浏览器 JS 拿不到 ping/pong 事件。链路半开时（笔记本休眠、Wi-Fi
切换、代理提前断开），页面既收不到数据也收不到 ``onclose``，会一直停在
"转写中"——组播任务其实还在跑，只是前端不知道连接已经死了。

这里在**没有业务消息**时下发 ``{"type": "heartbeat", "ts": ...}``，前端可以用
它做看门狗：超过 N 秒没收到任何消息就主动关闭并重连。心跳同时给中间代理制造
流量，避免代理的空闲超时把连接掐掉。

**为什么还要发送超时**：半开连接时 ``send_json`` 可能永远不返回（TCP 缓冲写满，
内核还在重传）。只靠"队列积压"发现不了——纯静音时队列根本不增长，发送任务会
一直挂在那儿，心跳也发不出去。超过 ``WS_SEND_TIMEOUT_SECONDS`` 就抛
``WebSocketSendTimeout``，由 handler 断开连接，让客户端重连。

**慢客户端保护**：出站队列不设物理上限（``put`` 永远不阻塞，避免卡住 pipeline
回调），但逻辑上超过 ``WS_OUTGOING_QUEUE_MAXSIZE`` 就判定客户端已长期不消费，
主动关闭连接。组播任务不会因此停止，客户端重连后由 ``replay_messages`` 补齐。

环境变量：
    WS_HEARTBEAT_INTERVAL_SECONDS  空闲心跳间隔，默认 15；<=0 关闭
    WS_SEND_TIMEOUT_SECONDS        单次发送超时，默认 10；<=0 不限制
    WS_OUTGOING_QUEUE_MAXSIZE      出站积压上限，默认 2000
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from multi_stream_asr.web.messages import heartbeat_message

_logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_SEND_TIMEOUT_SECONDS = 10.0
DEFAULT_OUTGOING_QUEUE_MAXSIZE = 2000


class WebSocketSendTimeout(RuntimeError):
    """单次 ``send_json`` 超过阈值：对端已不消费，连接应被断开。"""


def _env_number(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        _logger.warning("%s 不是数字，使用默认 %s: %r", name, default, raw)
        return default
    return max(minimum, value)


def heartbeat_interval_seconds() -> float:
    return _env_number(
        "WS_HEARTBEAT_INTERVAL_SECONDS",
        DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        minimum=0.0,
    )


def send_timeout_seconds() -> float:
    return _env_number(
        "WS_SEND_TIMEOUT_SECONDS",
        DEFAULT_SEND_TIMEOUT_SECONDS,
        minimum=0.0,
    )


def outgoing_queue_maxsize() -> int:
    value = _env_number(
        "WS_OUTGOING_QUEUE_MAXSIZE",
        float(DEFAULT_OUTGOING_QUEUE_MAXSIZE),
        minimum=1.0,
    )
    return int(value)


async def _send_with_timeout(
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    message: dict[str, Any],
    timeout: float,
) -> None:
    if timeout <= 0:
        await send_json(message)
        return
    try:
        await asyncio.wait_for(send_json(message), timeout=timeout)
    except asyncio.TimeoutError:
        raise WebSocketSendTimeout(
            f"send_json did not finish in {timeout:.1f}s"
        ) from None


def close_websocket_soon(
    loop: asyncio.AbstractEventLoop,
    websocket: Any,
    *,
    code: int = 1013,
) -> None:
    """异步关闭连接并吞掉异常。

    关闭可能因为对端已消失而失败；不吞掉会在事件循环里留下
    "Task exception was never retrieved" 噪声。
    """
    task = loop.create_task(websocket.close(code=code))
    task.add_done_callback(_consume_task_exception)


def _consume_task_exception(task: "asyncio.Task[Any]") -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        pass


async def relay_outgoing(
    queue: "asyncio.Queue[dict[str, Any] | None]",
    send_json: Callable[[dict[str, Any]], Awaitable[None]],
    *,
    heartbeat_seconds: float,
    send_timeout_seconds: float = 0.0,
) -> None:
    """把出站队列逐条写给客户端；空闲超过阈值就发一条心跳。

    ``None`` 是关闭哨兵：收到后直接返回，由调用方决定异常处理。
    发送超过 ``send_timeout_seconds`` 抛 ``WebSocketSendTimeout``。
    """
    while True:
        if heartbeat_seconds <= 0:
            message = await queue.get()
        else:
            try:
                message = await asyncio.wait_for(
                    queue.get(), timeout=heartbeat_seconds
                )
            except asyncio.TimeoutError:
                await _send_with_timeout(
                    send_json, heartbeat_message(), send_timeout_seconds
                )
                continue
        if message is None:
            return
        await _send_with_timeout(send_json, message, send_timeout_seconds)
