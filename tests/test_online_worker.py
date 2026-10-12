from __future__ import annotations

import asyncio
import threading

import pytest

from multi_stream_asr.pipeline.online_worker import OnlineConsumerWorker


def test_online_consumer_worker_loads_and_runs_on_the_same_thread() -> None:
    init_thread: list[int] = []
    consume_thread: list[int] = []

    class Recognizer:
        pass

    def factory() -> Recognizer:
        init_thread.append(threading.get_ident())
        return Recognizer()

    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-owner", factory=factory)
        recognizer = await worker.wait_until_loaded()

        def consume() -> None:
            assert recognizer is not None
            consume_thread.append(threading.get_ident())

        await asyncio.wrap_future(worker.run_consumer(consume))
        await worker.close()

    asyncio.run(exercise())

    assert init_thread == consume_thread


def test_online_consumer_worker_can_close_while_waiting_for_run() -> None:
    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-owner", factory=object)
        await worker.wait_until_loaded()
        await worker.close()
        assert not worker.is_alive()

    asyncio.run(exercise())


def test_online_worker_runs_actions_before_consumer_on_owner_thread() -> None:
    calls: list[tuple[str, int]] = []

    def factory() -> object:
        calls.append(("factory", threading.get_ident()))
        return object()

    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-actions", factory=factory)
        await worker.wait_until_loaded()
        try:
            await asyncio.wrap_future(
                worker.run_action(
                    lambda: calls.append(("warmup", threading.get_ident()))
                )
            )
            await asyncio.wrap_future(
                worker.run_consumer(
                    lambda: calls.append(("consumer", threading.get_ident()))
                )
            )
        finally:
            await worker.close()

    asyncio.run(exercise())

    assert [name for name, _ in calls] == ["factory", "warmup", "consumer"]
    assert len({thread_id for _, thread_id in calls}) == 1


def test_online_worker_reuses_the_owner_thread_for_two_consumers() -> None:
    calls: list[tuple[str, int]] = []

    def factory() -> object:
        calls.append(("factory", threading.get_ident()))
        return object()

    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-reuse", factory=factory)
        await worker.wait_until_loaded()
        try:
            for run_number in (1, 2):
                await asyncio.wrap_future(
                    worker.run_consumer(
                        lambda run_number=run_number: calls.append(
                            (f"consumer-{run_number}", threading.get_ident())
                        )
                    )
                )
        finally:
            await worker.close()

    asyncio.run(exercise())

    assert [name for name, _ in calls] == ["factory", "consumer-1", "consumer-2"]
    assert len({thread_id for _, thread_id in calls}) == 1


def test_online_worker_propagates_action_failure_and_can_close() -> None:
    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-action-error", factory=object)
        await worker.wait_until_loaded()

        def fail() -> None:
            raise RuntimeError("warmup failed")

        try:
            with pytest.raises(RuntimeError, match="warmup failed"):
                await asyncio.wrap_future(worker.run_action(fail))
        finally:
            await worker.close()
        assert not worker.is_alive()

    asyncio.run(exercise())
