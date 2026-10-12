from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class SharedZipformerRuntime:
    """Serialize all SpacemiT/Zipformer calls onto one persistent OS thread.

    SpacemiT EP with ``SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1`` is not safe to
    call from arbitrary threads (load on thread A, decode on thread B → segfault).
    All ``from_transducer`` / ``create_stream`` / ``accept_waveform`` /
    ``decode_stream`` / ``reset`` work must happen on this thread.
    """

    def __init__(self, *, name: str = "shared-zipformer-runtime") -> None:
        self._commands: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def call(self, fn: Callable[[], T]) -> T:
        if self._closed:
            raise RuntimeError("shared zipformer runtime is closed")
        if threading.current_thread() is self._thread:
            return fn()

        box: dict[str, object] = {}
        done = threading.Event()

        def task() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:
                box["exc"] = exc
            finally:
                done.set()

        self._commands.put(task)
        done.wait()
        if "exc" in box:
            raise box["exc"]  # type: ignore[misc]
        return box["result"]  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._commands.put(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            _logger.warning("shared zipformer runtime thread did not stop in time")

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            if command is None:
                return
            command()
