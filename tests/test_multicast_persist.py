from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from multi_stream_asr.web.multicast_runtime import MulticastJob, MulticastRuntimeRegistry

KEY = ("239.0.0.1", 9000)


class _StubTask:
    def done(self) -> bool:
        return True

    def cancel(self) -> None:
        pass

    def __await__(self):
        async def _noop() -> None:
            return None

        return _noop().__await__()


class _StubSource:
    async def stop(self) -> None:
        pass


class _StubSessionManager:
    _lifecycle_timeout_seconds = 1.0

    def push_audio(self, session, pcm) -> None:  # noqa: ANN001
        pass

    async def stop_session(self, session) -> None:  # noqa: ANN001
        pass


def _make_job() -> MulticastJob:
    finished_event = asyncio.Event()
    finished_event.set()
    session = SimpleNamespace(lane_id=0, finished_event=finished_event)
    return MulticastJob(
        key=KEY,
        session=session,
        audio_source=_StubSource(),
        audio_task=_StubTask(),
        stats_task=_StubTask(),
    )


def test_save_persisted_writes_running_jobs(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "multicast_jobs.json"
    registry = MulticastRuntimeRegistry(
        session_manager=_StubSessionManager(), persist_path=path
    )
    registry._jobs[KEY] = _make_job()
    registry._save_persisted()

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"group": KEY[0], "port": KEY[1]}
    ]


def test_stop_job_updates_persist(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "multicast_jobs.json"
    registry = MulticastRuntimeRegistry(
        session_manager=_StubSessionManager(), persist_path=path
    )
    registry._jobs[KEY] = _make_job()
    registry._save_persisted()

    asyncio.run(registry.stop_job(*KEY))
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_stop_all_keeps_persist_for_restart(tmp_path) -> None:  # noqa: ANN001
    """服务关停走 stop_all：清单必须留下，重启后据此自动拉起。"""
    path = tmp_path / "multicast_jobs.json"
    registry = MulticastRuntimeRegistry(
        session_manager=_StubSessionManager(), persist_path=path
    )
    registry._jobs[KEY] = _make_job()
    registry._save_persisted()

    asyncio.run(registry.stop_all())
    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"group": KEY[0], "port": KEY[1]}
    ]


def test_modify_missing_job_returns_none(tmp_path) -> None:  # noqa: ANN001
    registry = MulticastRuntimeRegistry(
        session_manager=_StubSessionManager(),
        persist_path=tmp_path / "multicast_jobs.json",
    )
    job = asyncio.run(
        registry.modify_job(
            old_group="239.0.0.9",
            old_port=6020,
            new_group="239.0.0.10",
        )
    )
    assert job is None


def test_restore_skips_invalid_entries(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "multicast_jobs.json"
    path.write_text(
        json.dumps(
            [
                {"group": "not-an-ip", "port": 6020},
                {"group": "192.168.1.1", "port": 6020},  # 非组播地址
                {"group": "239.0.0.1", "port": 99999},
                "garbage",
            ]
        ),
        encoding="utf-8",
    )
    registry = MulticastRuntimeRegistry(
        session_manager=_StubSessionManager(), persist_path=path
    )
    restored = asyncio.run(
        registry.restore_jobs(
            local_ip="0.0.0.0",
            packet_bytes=1280,
            use_full_packet=False,
            heartbeat_interval=5.0,
            speexdsp_lib_path=None,
        )
    )
    assert restored == []
