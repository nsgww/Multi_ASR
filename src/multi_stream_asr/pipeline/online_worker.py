from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class _WorkerCommand:
    target: Callable[[], None]
    future: concurrent.futures.Future[None]
    terminal: bool


class OnlineConsumerWorker(Generic[T]):
    """Load and run an online recognizer on one persistent OS thread."""

    def __init__(self, *, name: str, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._loaded: concurrent.futures.Future[T] = concurrent.futures.Future()
        self._completed: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._commands: queue.Queue[_WorkerCommand | None] = queue.Queue(maxsize=1)
        self._command_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name)
        self._thread.start()

    async def wait_until_loaded(self) -> T:
        return await asyncio.wrap_future(self._loaded)

    def run_action(
        self,
        target: Callable[[], None],
    ) -> concurrent.futures.Future[None]:
        return self._submit(target, terminal=False)

    def run_consumer(
        self,
        target: Callable[[], None],
    ) -> concurrent.futures.Future[None]:
        return self._submit(target, terminal=False)

    def _submit(
        self,
        target: Callable[[], None],
        *,
        terminal: bool,
    ) -> concurrent.futures.Future[None]:
        with self._command_lock:
            if self._closed:
                raise RuntimeError("online consumer worker is closed")
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._commands.put(
                _WorkerCommand(target=target, future=future, terminal=terminal)
            )
            return future

    async def close(self) -> None:
        with self._command_lock:
            if not self._closed:
                self._closed = True
                self._commands.put(None)
        await asyncio.wrap_future(self._completed)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            try:
                recognizer = self._factory()
            except BaseException as exc:
                self._loaded.set_exception(exc)
                return
            self._loaded.set_result(recognizer)

            while True:
                command = self._commands.get()
                if command is None:
                    return
                if not command.future.set_running_or_notify_cancel():
                    if command.terminal:
                        return
                    continue
                try:
                    command.target()
                except BaseException as exc:
                    command.future.set_exception(exc)
                else:
                    command.future.set_result(None)
                if command.terminal:
                    return
        finally:
            self._completed.set_result(None)
