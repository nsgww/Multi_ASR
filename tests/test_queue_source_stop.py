"""QueueAudioSource.stop 必须解阻 frames() 里阻塞的 get()。"""

from __future__ import annotations

import asyncio
import queue

import pytest

from multi_stream_asr.audio.queue_source import AsyncQueueAudioSource, QueueAudioSource


@pytest.mark.asyncio
async def test_queue_audio_source_stop_unblocks_frames() -> None:
    audio_queue: queue.Queue = queue.Queue()
    source = QueueAudioSource(stream_id=0, audio_queue=audio_queue)
    await source.prepare()

    async def consume() -> int:
        count = 0
        async for _frame in source.frames():
            count += 1
        return count

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    assert not task.done()

    await source.stop()
    count = await asyncio.wait_for(task, timeout=1)
    assert count >= 1  # 至少有一帧 finish


@pytest.mark.asyncio
async def test_async_queue_audio_source_stop_unblocks_frames() -> None:
    source = AsyncQueueAudioSource(stream_id=0)
    await source.prepare()

    async def consume() -> int:
        count = 0
        async for _frame in source.frames():
            count += 1
        return count

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    assert not task.done()

    await source.stop()
    count = await asyncio.wait_for(task, timeout=1)
    assert count >= 1


@pytest.mark.asyncio
async def test_prepare_drains_leftover_end_sentinels() -> None:
    audio_queue: queue.Queue = queue.Queue()
    source = QueueAudioSource(stream_id=0, audio_queue=audio_queue)
    audio_queue.put_nowait(None)
    audio_queue.put_nowait(None)
    audio_queue.put_nowait(b"\x00\x01")
    await source.prepare()
    assert audio_queue.get_nowait() == b"\x00\x01"
    with pytest.raises(queue.Empty):
        audio_queue.get_nowait()
