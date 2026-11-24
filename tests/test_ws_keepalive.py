"""WebSocket 空闲心跳与慢客户端保护的单元测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.web.keepalive import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    DEFAULT_OUTGOING_QUEUE_MAXSIZE,
    DEFAULT_SEND_TIMEOUT_SECONDS,
    WebSocketSendTimeout,
    heartbeat_interval_seconds,
    outgoing_queue_maxsize,
    relay_outgoing,
    send_timeout_seconds,
)
from multi_stream_asr.web.stream_handler import StreamHandler


class RecordingWebSocket:
    """只记录 send_json，不碰真实连接。"""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed_with: int | None = None

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


async def test_relay_sends_heartbeat_when_idle() -> None:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    task = asyncio.create_task(
        relay_outgoing(queue, send, heartbeat_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    queue.put_nowait(None)
    await asyncio.wait_for(task, timeout=1.0)

    assert sent, "空闲时必须发出心跳"
    assert all(item["type"] == "heartbeat" for item in sent)
    assert all(isinstance(item["ts"], int) for item in sent)


async def test_relay_forwards_messages_without_heartbeat() -> None:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    task = asyncio.create_task(
        relay_outgoing(queue, send, heartbeat_seconds=5.0)
    )
    queue.put_nowait({"type": "transcript", "text": "hi"})
    queue.put_nowait(None)
    await asyncio.wait_for(task, timeout=1.0)

    assert sent == [{"type": "transcript", "text": "hi"}]


async def test_relay_heartbeat_can_be_disabled() -> None:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    task = asyncio.create_task(
        relay_outgoing(queue, send, heartbeat_seconds=0)
    )
    await asyncio.sleep(0.02)
    assert sent == []
    queue.put_nowait(None)
    await asyncio.wait_for(task, timeout=1.0)
    assert sent == []


async def test_stream_handler_send_loop_emits_heartbeat() -> None:
    """接线验证：handler 的发送循环必须真的用心跳间隔。"""
    websocket = RecordingWebSocket()
    handler = StreamHandler(
        websocket=websocket,  # type: ignore[arg-type]
        session_manager=None,  # type: ignore[arg-type]
        summary_config=SummaryConfig(),
        output_dir=Path("."),
    )
    handler._heartbeat_seconds = 0.01

    task = asyncio.create_task(handler._send_loop())
    await asyncio.sleep(0.05)
    handler._outgoing_queue.put_nowait(None)
    await asyncio.wait_for(task, timeout=1.0)

    assert websocket.sent
    assert websocket.sent[0]["type"] == "heartbeat"


async def test_relay_raises_send_timeout_when_send_hangs() -> None:
    """半开连接：send_json 永不返回时必须抛超时，不能把发出任务挂死。"""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    queue.put_nowait({"type": "transcript", "text": "hi"})

    async def send(payload: dict[str, Any]) -> None:
        await asyncio.sleep(10)

    with pytest.raises(WebSocketSendTimeout):
        await asyncio.wait_for(
            relay_outgoing(
                queue,
                send,
                heartbeat_seconds=0,
                send_timeout_seconds=0.02,
            ),
            timeout=1.0,
        )


async def test_stream_handler_send_loop_drops_client_on_timeout() -> None:
    websocket = RecordingWebSocket()
    handler = StreamHandler(
        websocket=websocket,  # type: ignore[arg-type]
        session_manager=None,  # type: ignore[arg-type]
        summary_config=SummaryConfig(),
        output_dir=Path("."),
    )
    handler._loop = asyncio.get_running_loop()
    handler._heartbeat_seconds = 0
    handler._send_timeout = 0.02
    handler._enqueue({"type": "transcript", "text": "hi"})

    async def send_json(_payload: dict[str, Any]) -> None:
        await asyncio.sleep(10)

    websocket.send_json = send_json  # type: ignore[method-assign]
    await asyncio.wait_for(handler._send_loop(), timeout=1.0)
    await asyncio.sleep(0)

    assert handler._slow_client_dropped is True
    assert websocket.closed_with == 1013


async def test_stream_handler_drops_slow_client() -> None:
    """积压超限时主动断开，而不是让队列无限增长。"""
    websocket = RecordingWebSocket()
    handler = StreamHandler(
        websocket=websocket,  # type: ignore[arg-type]
        session_manager=None,  # type: ignore[arg-type]
        summary_config=SummaryConfig(),
        output_dir=Path("."),
    )
    handler._loop = asyncio.get_running_loop()
    handler._outgoing_maxsize = 3

    for index in range(3):
        handler._enqueue({"type": "transcript", "index": index})
    assert handler._outgoing_queue.qsize() == 3

    handler._enqueue({"type": "transcript", "index": 99})
    await asyncio.sleep(0)

    assert handler._slow_client_dropped is True
    assert handler._outgoing_queue.qsize() == 3, "超限消息不再入队"
    assert websocket.closed_with == 1013


def test_heartbeat_interval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WS_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    assert heartbeat_interval_seconds() == DEFAULT_HEARTBEAT_INTERVAL_SECONDS

    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "5")
    assert heartbeat_interval_seconds() == 5.0

    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "-1")
    assert heartbeat_interval_seconds() == 0.0

    monkeypatch.setenv("WS_HEARTBEAT_INTERVAL_SECONDS", "abc")
    assert heartbeat_interval_seconds() == DEFAULT_HEARTBEAT_INTERVAL_SECONDS


def test_outgoing_queue_maxsize_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WS_OUTGOING_QUEUE_MAXSIZE", raising=False)
    assert outgoing_queue_maxsize() == DEFAULT_OUTGOING_QUEUE_MAXSIZE

    monkeypatch.setenv("WS_OUTGOING_QUEUE_MAXSIZE", "7")
    assert outgoing_queue_maxsize() == 7

    monkeypatch.setenv("WS_OUTGOING_QUEUE_MAXSIZE", "0")
    assert outgoing_queue_maxsize() == 1

    monkeypatch.setenv("WS_OUTGOING_QUEUE_MAXSIZE", "x")
    assert outgoing_queue_maxsize() == DEFAULT_OUTGOING_QUEUE_MAXSIZE


def test_send_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WS_SEND_TIMEOUT_SECONDS", raising=False)
    assert send_timeout_seconds() == DEFAULT_SEND_TIMEOUT_SECONDS

    monkeypatch.setenv("WS_SEND_TIMEOUT_SECONDS", "3.5")
    assert send_timeout_seconds() == 3.5

    monkeypatch.setenv("WS_SEND_TIMEOUT_SECONDS", "-2")
    assert send_timeout_seconds() == 0.0

    monkeypatch.setenv("WS_SEND_TIMEOUT_SECONDS", "oops")
    assert send_timeout_seconds() == DEFAULT_SEND_TIMEOUT_SECONDS
