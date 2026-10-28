from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from multi_stream_asr.web.multicast_runtime import MulticastJob, MulticastRuntimeRegistry

KEY = ("239.0.0.1", 9000)
REMAP = {"type": "speaker_remap", "lane_id": 0, "mapping": {"S3": "S1"}}


class _StubTask:
    """够用的 Task 替身：done()/cancel()/可 await。"""

    def __init__(self) -> None:
        self._done = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self._done = True

    def __await__(self):
        async def _noop() -> None:
            return None

        return _noop().__await__()


class _StubSource:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1


class _StubSessionManager:
    _lifecycle_timeout_seconds = 1.0

    def __init__(self, on_stop=None) -> None:
        self.pushed: list = []
        self.stopped: list = []
        self._on_stop = on_stop

    def push_audio(self, session, pcm) -> None:  # noqa: ANN001
        self.pushed.append(pcm)

    async def stop_session(self, session) -> None:  # noqa: ANN001
        self.stopped.append(session)
        if self._on_stop is not None:
            self._on_stop()


def make_job(*, listeners=None) -> MulticastJob:
    finished_event = asyncio.Event()
    finished_event.set()
    session = SimpleNamespace(lane_id=0, finished_event=finished_event)
    return MulticastJob(
        key=KEY,
        session=session,
        audio_source=_StubSource(),
        audio_task=_StubTask(),
        stats_task=_StubTask(),
        listeners=set(listeners or ()),
    )


def test_fanout_delivers_to_finished_but_registered_job() -> None:
    """收尾期间 job 已标记 finished 但仍可投递——这是尾句 FINAL 能落盘的前提。"""
    received: list[dict] = []
    registry = MulticastRuntimeRegistry(session_manager=_StubSessionManager())
    job = make_job(listeners=[received.append])
    job.finished = True
    registry._jobs[KEY] = job

    registry._fanout(KEY, REMAP)

    assert received == [REMAP]


def test_fanout_ignores_unknown_key() -> None:
    registry = MulticastRuntimeRegistry(session_manager=_StubSessionManager())
    registry._fanout(("1.2.3.4", 1), REMAP)  # 不应抛异常


@pytest.mark.asyncio
async def test_stop_job_keeps_job_attached_until_session_finishes() -> None:
    """回归：不能提前摘除 job / 清 listeners，否则收尾事件（尾句 FINAL、
    speaker_remap）会被丢掉，落盘就缺最后一句和说话人重标。"""
    received: list[dict] = []
    holder: dict = {}
    reachable_during_stop: list[bool] = []

    def publish_finish_event() -> None:
        registry = holder["registry"]
        job = holder["job"]
        reachable_during_stop.append(registry.get(*KEY) is job)
        registry._fanout(KEY, REMAP)

    manager = _StubSessionManager(on_stop=publish_finish_event)
    registry = MulticastRuntimeRegistry(session_manager=manager)
    job = make_job(listeners=[received.append])
    holder["registry"] = registry
    holder["job"] = job
    registry._jobs[KEY] = job

    stopped = await registry.stop_job(*KEY)

    assert stopped is job
    assert reachable_during_stop == [True], "收尾阶段 job 不应已被摘除"
    assert received == [REMAP], "收尾阶段的事件没有投递给监听者"
    assert registry.get(*KEY) is None, "收尾结束后必须摘除 job"


@pytest.mark.asyncio
async def test_stop_job_clears_listeners_after_finish() -> None:
    manager = _StubSessionManager()
    registry = MulticastRuntimeRegistry(session_manager=manager)
    job = make_job()
    registry._jobs[KEY] = job

    await registry.stop_job(*KEY)

    assert job.listeners == set()
    assert job.finished is True
    assert manager.pushed == [None]


@pytest.mark.asyncio
async def test_stop_job_returns_none_for_unknown_key() -> None:
    registry = MulticastRuntimeRegistry(session_manager=_StubSessionManager())
    assert await registry.stop_job(*KEY) is None
