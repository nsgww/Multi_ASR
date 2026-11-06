from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from multi_stream_asr.session.process_pool import ProcessPoolSessionManager


class _Controller:
    def __init__(self) -> None:
        self.commands: list[dict] = []

    def send(self, command: dict) -> None:
        self.commands.append(command)


def _session(*, finished: bool = False) -> SimpleNamespace:
    event = asyncio.Event()
    if finished:
        event.set()
    return SimpleNamespace(
        lane_id=0,
        controller=_Controller(),
        finished_event=event,
    )


def _manager(*, quick: float = 0.05, grace: float = 0.2) -> ProcessPoolSessionManager:
    manager = ProcessPoolSessionManager.__new__(ProcessPoolSessionManager)
    manager._finish_quick_seconds = quick
    manager._finish_grace_seconds = grace
    manager._pushed: list = []
    return manager


@pytest.mark.asyncio
async def test_await_finish_returns_immediately_when_already_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    session = _session(finished=True)
    monkeypatch.setattr(manager, "push_audio", lambda *_a, **_k: None)

    assert await manager.await_finish(session) is True
    assert session.controller.commands == []


@pytest.mark.asyncio
async def test_await_finish_natural_path_skips_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(quick=0.2, grace=1.0)
    session = _session()
    pushed: list = []

    def push(_session, data) -> None:  # noqa: ANN001
        pushed.append(data)
        asyncio.get_running_loop().call_later(0.02, session.finished_event.set)

    monkeypatch.setattr(manager, "push_audio", push)

    assert await manager.await_finish(session) is True
    assert pushed == [None]
    assert session.controller.commands == []


@pytest.mark.asyncio
async def test_await_finish_sends_stop_after_quick_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(quick=0.05, grace=0.3)
    session = _session()
    monkeypatch.setattr(manager, "push_audio", lambda *_a, **_k: None)

    async def finish_after_stop() -> None:
        while not session.controller.commands:
            await asyncio.sleep(0.01)
        session.finished_event.set()

    watcher = asyncio.create_task(finish_after_stop())
    try:
        assert await manager.await_finish(session) is True
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    assert session.controller.commands == [{"type": "stop", "reason": "finish"}]


@pytest.mark.asyncio
async def test_await_finish_returns_false_on_full_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(quick=0.05, grace=0.12)
    session = _session()
    monkeypatch.setattr(manager, "push_audio", lambda *_a, **_k: None)

    assert await manager.await_finish(session) is False
    assert session.controller.commands == [{"type": "stop", "reason": "finish"}]
