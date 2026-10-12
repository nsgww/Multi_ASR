from __future__ import annotations

import asyncio
import contextlib
import json
import math
import multiprocessing as mp
import queue
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from multi_stream_asr.asr.offline.shared_server import SHARED_LOG_PATH, SharedQwenServerManager
from multi_stream_asr.audio.device import resolve_microphone_device
from multi_stream_asr.audio.local_file import select_input_files, validate_wav_16k_mono_pcm
from multi_stream_asr.config import (
    AppConfig,
    LaneConfig,
    LaneInputConfig,
    ZIPFORMER_WARMUP_AUDIO_PATH,
)
from multi_stream_asr.models import ASREvent, EventStatus, LaneFinalState, LaneMetrics
from multi_stream_asr.log.metrics import SharedTimingsTracker
from multi_stream_asr.log.qwen_errors import QwenErrorCoordinator
from multi_stream_asr.pipeline.ipc import (
    LaneIpcMessage,
    asr_event_from_payload,
    lane_metrics_from_payload,
)
from multi_stream_asr.pipeline.lane_runtime import LaneProcessConfig, run_lane_process
from multi_stream_asr.pipeline.state import PipelineState, assert_transition
from multi_stream_asr.web.event_bus import FrontendEventBus


_ZERO_LATENCY = {
    "count": 0,
    "total_ms": 0.0,
    "average_ms": 0.0,
    "max_ms": 0.0,
}
_ZERO_RTF = {"count": 0, "total": 0.0, "average": 0.0, "max": 0.0}
_ALLOWED_MESSAGE_TYPES = frozenset(
    {
        "asr_event",
        "lifecycle",
        "metrics_snapshot",
        "metrics_final",
        "fatal",
        "paths",
        "qwen_success",
        "qwen_error",
    }
)
_LIFECYCLE_STATES = frozenset(
    {
        "lane_loading",
        "lane_model_loaded",
        "lane_loaded",
        "lane_started",
        "lane_finished",
        "lane_stopped",
        "lane_failed",
    }
)
_ARTIFACT_PATH_KEYS = ("results_path", "model_outputs_path")
_QWEN_SUCCESS_PAYLOAD_KEYS = frozenset({"run_id", "sentence_id"})
_QWEN_ERROR_PAYLOAD_KEYS = frozenset(
    {"run_id", "sentence_id", "revision", "pending_text", "timestamp_ms", "request"}
)
_QWEN_REQUEST_FIELDS = frozenset(
    {
        "category",
        "error_type",
        "message",
        "method",
        "url",
        "timeout_ms",
        "audio_seconds",
        "sample_count",
        "status_code",
        "response_body",
        "response_body_truncated",
    }
)
_LOOP_RESULT_RETENTION_LIMIT = 100


class LaneProcessController:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        lane: LaneConfig,
        fake: bool,
        streaming: bool = False,
    ) -> None:
        self.app_config = app_config
        self.lane = lane
        self.fake = fake
        self.streaming = streaming
        self._ctx = mp.get_context("spawn")
        self.event_queue = self._ctx.Queue()
        self.command_queue = self._ctx.Queue()
        self.audio_queue: Any = self._ctx.Queue(maxsize=100) if streaming else None
        self.process: mp.Process | None = None
        self._queues_closed = False
        self._process_closed = False
        self._final_exitcode: int | None = None

    @property
    def exitcode(self) -> int | None:
        if self._process_closed:
            return self._final_exitcode
        return None if self.process is None else self.process.exitcode

    def start(self) -> None:
        process_config = LaneProcessConfig(
            app_config=self.app_config,
            lane=self.lane,
            fake=self.fake,
            streaming=self.streaming,
        )
        args: tuple[Any, ...] = (
            process_config,
            self.event_queue,
            self.command_queue,
        )
        if self.streaming:
            args = args + (self.audio_queue,)
        self.process = self._ctx.Process(
            target=run_lane_process,
            args=args,
            name=f"asr-lane-{self.lane.stream_id}",
        )
        self._process_closed = False
        self._final_exitcode = None
        self.process.start()

    def send(self, command: dict[str, Any]) -> None:
        json.dumps(command, allow_nan=False)
        self.command_queue.put(command)

    def is_alive(self) -> bool:
        if self._process_closed:
            return False
        return self.process is not None and self.process.is_alive()

    async def join(self, timeout: float) -> int | None:
        if self._process_closed:
            return self._final_exitcode
        if self.process is None:
            return None
        await asyncio.to_thread(self.process.join, timeout)
        return self.process.exitcode

    async def terminate_then_kill(
        self,
        *,
        terminate_timeout: float,
        kill_timeout: float,
    ) -> bool:
        if self.process is None or not self.process.is_alive():
            return False
        self.process.terminate()
        await asyncio.to_thread(self.process.join, terminate_timeout)
        if self.process.is_alive():
            self.process.kill()
            await asyncio.to_thread(self.process.join, kill_timeout)
        return True

    def close_queues(self) -> None:
        if self._queues_closed:
            return
        self._queues_closed = True
        queues = [self.command_queue, self.event_queue]
        audio_queue = getattr(self, "audio_queue", None)
        if audio_queue is not None:
            queues.append(audio_queue)
        for queue_obj in queues:
            cancel_join_thread = getattr(queue_obj, "cancel_join_thread", None)
            if cancel_join_thread is not None:
                cancel_join_thread()
            close = getattr(queue_obj, "close", None)
            if close is not None:
                close()

    def close_process(self) -> None:
        if self._process_closed:
            return
        if self.process is None:
            self._process_closed = True
            return
        if self.process.is_alive():
            raise RuntimeError("cannot close a running lane process handle")
        self._final_exitcode = getattr(self.process, "exitcode", None)
        close = getattr(self.process, "close", None)
        if close is not None:
            close()
        self._process_closed = True


class ProcessPipelineManager:
    def __init__(
        self,
        *,
        config: AppConfig,
        event_bus: FrontendEventBus,
        fake: bool = False,
        lifecycle_timeout_seconds: float | None = None,
        terminate_timeout_seconds: float = 5.0,
        kill_timeout_seconds: float = 2.0,
        queue_poll_seconds: float = 0.1,
        cleanup_margin_seconds: float = 5.0,
        shared_watch_interval_seconds: float = 0.1,
        qwen_error_coordinator: QwenErrorCoordinator | None = None,
    ) -> None:
        self._config = config
        self._lane_inputs = config.lane_inputs
        self._event_bus = event_bus
        self._fake = fake
        self._lifecycle_timeout_seconds = (
            config.shared_qwen.request_timeout_ms / 1000.0 + cleanup_margin_seconds
            if lifecycle_timeout_seconds is None
            else lifecycle_timeout_seconds
        )
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self._kill_timeout_seconds = kill_timeout_seconds
        self._queue_poll_seconds = queue_poll_seconds
        self._shared_watch_interval_seconds = shared_watch_interval_seconds
        self._state = PipelineState.UNLOADED
        self._lifecycle_lock = asyncio.Lock()
        self._process_shutdown_lock = asyncio.Lock()
        self._change_event = asyncio.Event()
        self._reader_stop = asyncio.Event()
        self._controllers: dict[int, LaneProcessController] = {}
        self._reader_tasks: dict[int, asyncio.Task[None]] = {}
        self._model_loaded: set[int] = set()
        self._loaded: set[int] = set()
        self._started: set[int] = set()
        self._starting_streams: set[int] = set()
        self._start_stop_requested = False
        self._finished: set[int] = set()
        self._terminal: set[int] = set()
        self._exited: set[int] = set()
        self._last_seq: dict[int, int] = {}
        self._frontend_fatal_streams: set[int] = set()
        self._fatal_streams: set[int] = set()
        self._final_metrics_streams: set[int] = set()
        self._lane_metrics: dict[int, LaneMetrics] = {
            lane.stream_id: LaneMetrics(stream_id=lane.stream_id)
            for lane in config.lane_configs
        }
        self._lane_summaries: dict[int, dict[str, Any]] = {}
        self._paths: dict[int, dict[str, str]] = {}
        self._failure: BaseException | None = None
        self._run_dir: Path | None = None
        self._completion_task: asyncio.Task[None] | None = None
        self._resources_closed = False
        self._force_used = False
        self._shared_manager: SharedQwenServerManager | None = None
        self._shared_watchdog_task: asyncio.Task[None] | None = None
        self._shared_stop_expected = False
        self._load_phase: str | None = None
        self._qwen_errors = qwen_error_coordinator or QwenErrorCoordinator(
            server_log_path=SHARED_LOG_PATH
        )
        self._qwen_errors_enabled = qwen_error_coordinator is not None or not fake
        self._qwen_run_active = False
        self._unload_task: asyncio.Task[None] | None = None
        self._unload_waiters = 0
        self._unload_result_latched = False

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def load_phase(self) -> str | None:
        return self._load_phase

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    @property
    def lane_metrics(self) -> list[LaneMetrics]:
        return [self._lane_metrics[lane.stream_id] for lane in self._config.lane_configs]

    @property
    def lane_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for lane in self._config.lane_configs:
            stream_id = lane.stream_id
            summary = self._lane_summaries.get(stream_id)
            if summary is None:
                metrics = self._lane_metrics[stream_id]
                summary = {
                    "lane_metrics": {
                        "stream_id": metrics.stream_id,
                        "processed_frames": metrics.processed_frames,
                        "dropped_frames": metrics.dropped_frames,
                        "backlogged_frames": metrics.backlogged_frames,
                        "offline_requests": metrics.offline_requests,
                        "final_state": metrics.final_state.value,
                    },
                    "online_frame_latency": dict(_ZERO_LATENCY),
                    "offline_rtf": dict(_ZERO_RTF),
                    "offline_shared_timings": SharedTimingsTracker().as_dict(),
                    "event_loop_lag": dict(_ZERO_LATENCY),
                }
            summaries.append(dict(summary))
        return summaries

    def set_lane_inputs(self, lane_inputs: tuple[LaneInputConfig, ...]) -> None:
        if self._state is not PipelineState.LOADED:
            raise ValueError(f"cannot update lane inputs from {self._state.value}")
        if len(lane_inputs) != self._config.lanes:
            raise ValueError("lane_inputs must match configured lanes")
        self._lane_inputs = lane_inputs

    async def load_models(self) -> None:
        warmup_path = ZIPFORMER_WARMUP_AUDIO_PATH.resolve()
        if not self._fake:
            validate_wav_16k_mono_pcm(warmup_path)

        async with self._lifecycle_lock:
            assert_transition(self._state, PipelineState.LOADING)
            self._state = PipelineState.LOADING
            self._load_phase = "loading_models"
            self._reset_for_load()
            try:
                if not self._fake:
                    self._shared_manager = SharedQwenServerManager(self._config)
                    await asyncio.to_thread(self._shared_manager.start)
                    self._shared_stop_expected = False
                    self._shared_watchdog_task = asyncio.create_task(
                        self._watch_shared_qwen(),
                        name="asr-shared-qwen-watchdog",
                    )
                if self._qwen_errors_enabled:
                    try:
                        self._qwen_errors.reset_for_model_load()
                    except Exception as exc:
                        raise RuntimeError(
                            "failed to initialize Qwen error coordinator: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                self._controllers = {
                    lane.stream_id: LaneProcessController(
                        app_config=self._config,
                        lane=lane,
                        fake=self._fake,
                    )
                    for lane in self._config.lane_configs
                }
                for stream_id, controller in self._controllers.items():
                    controller.start()
                    self._reader_tasks[stream_id] = asyncio.create_task(
                        self._read_lane_events(controller),
                        name=f"asr-lane-reader-{stream_id}",
                    )
                await self._wait_for(
                    lambda: len(self._model_loaded) == self._config.lanes,
                    timeout=self._lifecycle_timeout_seconds,
                )
                self._load_phase = "warming_zipformer"
                self._send_to_active(
                    {"type": "warmup", "input_path": str(warmup_path)}
                )
                await self._wait_for(
                    lambda: len(self._loaded) == self._config.lanes,
                    timeout=self._lifecycle_timeout_seconds,
                )
                assert_transition(self._state, PipelineState.LOADED)
                self._state = PipelineState.LOADED
            except Exception as exc:
                if self._failure is None:
                    await self._record_fatal(0, f"lane load failed: {exc}")
                self._send_to_active({"type": "shutdown"})
                await self._join_or_escalate()
                await self._finish_readers_and_close()
                await self._stop_shared_qwen()
                raise self._failure from exc
            finally:
                self._load_phase = None

    async def start_recognition(self) -> None:
        async with self._lifecycle_lock:
            microphone_device_names = self._resolve_microphone_inputs()
            input_commands = self._resolve_input_commands(microphone_device_names)
            assert_transition(self._state, PipelineState.STARTING)
            self._state = PipelineState.STARTING
            try:
                self._event_bus.set_result_retention_limit(
                    _LOOP_RESULT_RETENTION_LIMIT
                    if any(command.get("loop_audio", False) for command in input_commands.values())
                    else None
                )
                self._event_bus.clear_results()
                self._reset_for_run()
                self._run_dir = self._new_run_dir()
                if self._qwen_errors_enabled:
                    try:
                        self._qwen_errors.start_run(self._run_dir)
                    except Exception as exc:
                        raise RuntimeError(
                            "failed to start Qwen error coordinator run: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                self._qwen_run_active = True
                microphone_commands: list[tuple[int, dict[str, Any]]] = []
                for lane in self._config.lane_configs:
                    command: dict[str, Any] = {
                        "type": "start",
                        "run_dir": str(self._run_dir),
                    }
                    command.update(input_commands[lane.stream_id])
                    if command["input_kind"] == "microphone":
                        microphone_commands.append((lane.stream_id, command))
                    else:
                        self._controllers[lane.stream_id].send(command)
                        self._starting_streams.add(lane.stream_id)
                for stream_id, command in microphone_commands:
                    if self._start_stop_requested:
                        break
                    self._controllers[stream_id].send(command)
                    self._starting_streams.add(stream_id)
                    await self._wait_for(
                        lambda stream_id=stream_id: (
                            stream_id in self._started or self._start_stop_requested
                        ),
                        timeout=self._lifecycle_timeout_seconds,
                    )
                if self._start_stop_requested:
                    await self._wait_for(
                        lambda: self._starting_streams <= self._finished,
                        timeout=self._lifecycle_timeout_seconds,
                    )
                    await self._finish_qwen_run()
                    if self._failure is not None:
                        raise self._failure
                    assert_transition(self._state, PipelineState.STOPPING)
                    self._state = PipelineState.STOPPING
                    assert_transition(self._state, PipelineState.LOADED)
                    self._state = PipelineState.LOADED
                    return
                await self._wait_for(
                    lambda: len(self._started) == self._config.lanes,
                    timeout=self._lifecycle_timeout_seconds,
                )
                assert_transition(self._state, PipelineState.RUNNING)
                self._state = PipelineState.RUNNING
                self._completion_task = asyncio.create_task(
                    self._monitor_run_completion(),
                    name="asr-process-completion",
                )
            except BaseException as exc:
                cancelled = isinstance(exc, asyncio.CancelledError)
                if self._failure is None:
                    await self._record_fatal(0, f"recognition start failed: {exc}")
                self._send_to_active({"type": "shutdown"})
                await self._join_or_escalate()
                await self._finish_readers_and_close()
                await self._finish_qwen_run()
                if cancelled:
                    raise
                raise self._failure from exc

    async def wait_for_completion(self) -> None:
        if self._completion_task is None:
            raise RuntimeError("recognition has not been started")
        await self._completion_task
        if self._failure is not None:
            raise self._failure

    async def stop_recognition(self) -> None:
        stopping_start = self._state is PipelineState.STARTING
        if stopping_start:
            self._start_stop_requested = True
            for stream_id in tuple(self._starting_streams):
                controller = self._controllers.get(stream_id)
                if controller is not None and controller.is_alive():
                    with contextlib.suppress(Exception):
                        controller.send({"type": "stop", "reason": "user_requested"})
            self._signal_change()
        async with self._lifecycle_lock:
            if stopping_start and self._state is PipelineState.LOADED:
                return
            if self._state not in {PipelineState.STARTING, PipelineState.RUNNING}:
                raise ValueError(f"cannot stop recognition from {self._state.value}")
            assert_transition(self._state, PipelineState.STOPPING)
            self._state = PipelineState.STOPPING
            self._send_to_active({"type": "stop", "reason": "user_requested"})
        if self._completion_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._completion_task),
                    timeout=self._lifecycle_timeout_seconds,
                )
            except TimeoutError:
                await self._record_fatal(0, "lane stop timed out")
                self._send_to_active({"type": "shutdown"})
                await self._join_or_escalate()
                await self._finish_readers_and_close()
                self._completion_task.cancel()
                await asyncio.gather(self._completion_task, return_exceptions=True)
                await self._finish_qwen_run()

    async def unload_models(self) -> None:
        async with self._lifecycle_lock:
            unload_task = self._unload_task
            if unload_task is None:
                unload_task = asyncio.create_task(
                    self._unload_models_once(),
                    name="asr-process-unload",
                )
                self._unload_task = unload_task
                unload_task.add_done_callback(self._unload_task_done)
            self._unload_waiters += 1
        try:
            await asyncio.shield(unload_task)
        finally:
            async with self._lifecycle_lock:
                self._unload_waiters -= 1
                self._clear_finished_unload_task(unload_task)

    async def _unload_models_once(self) -> None:
        async with self._lifecycle_lock:
            if self._state is PipelineState.UNLOADED:
                return
            if self._state is PipelineState.FATAL:
                self._send_to_active({"type": "shutdown"})
            else:
                assert_transition(self._state, PipelineState.UNLOADING)
                self._state = PipelineState.UNLOADING
                self._send_to_active({"type": "shutdown"})
        if self._controllers and len(self._exited) < len(self._controllers):
            await self._join_or_escalate()
        alive_streams = [
            stream_id
            for stream_id, controller in self._controllers.items()
            if controller.is_alive()
        ]
        if alive_streams:
            if self._state is not PipelineState.FATAL:
                assert_transition(self._state, PipelineState.FATAL)
                self._state = PipelineState.FATAL
            joined_streams = ", ".join(str(stream_id) for stream_id in alive_streams)
            raise RuntimeError(
                f"lane teardown incomplete; processes still alive: {joined_streams}"
            )
        await self._finish_readers_and_close()
        await self._stop_shared_qwen()
        qwen_run_finished = await self._finish_qwen_run()
        if not qwen_run_finished:
            self._unload_result_latched = True
            assert self._failure is not None
            raise self._failure
        if self._qwen_errors_enabled:
            try:
                self._qwen_errors.reset_for_unload()
            except Exception as exc:
                self._unload_result_latched = True
                await self._record_fatal(
                    0,
                    "failed to reset Qwen error coordinator during unload: "
                    f"{type(exc).__name__}: {exc}",
                )
                raise self._failure from exc
        self._controllers.clear()
        self._reader_tasks.clear()
        self._model_loaded.clear()
        self._loaded.clear()
        self._terminal.clear()
        self._exited.clear()
        self._state = PipelineState.UNLOADED

    def _unload_task_done(self, unload_task: asyncio.Task[None]) -> None:
        if not unload_task.cancelled():
            unload_task.exception()
        asyncio.create_task(self._clear_finished_unload_task_when_idle(unload_task))

    async def _clear_finished_unload_task_when_idle(
        self,
        unload_task: asyncio.Task[None],
    ) -> None:
        async with self._lifecycle_lock:
            self._clear_finished_unload_task(unload_task)

    def _clear_finished_unload_task(
        self,
        unload_task: asyncio.Task[None],
    ) -> None:
        if (
            self._unload_task is unload_task
            and unload_task.done()
            and self._unload_waiters == 0
            and not self._unload_result_latched
        ):
            self._unload_task = None

    def _new_run_dir(self) -> Path:
        timestamp_ms = int(time.time() * 1000)
        while True:
            run_dir = Path(self._config.output_dir) / f"process-{timestamp_ms}"
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                timestamp_ms += 1
                continue
            return run_dir

    def _reset_for_load(self) -> None:
        self._change_event.clear()
        self._reader_stop.clear()
        self._model_loaded.clear()
        self._loaded.clear()
        self._started.clear()
        self._starting_streams.clear()
        self._start_stop_requested = False
        self._finished.clear()
        self._terminal.clear()
        self._exited.clear()
        self._last_seq.clear()
        self._frontend_fatal_streams.clear()
        self._fatal_streams.clear()
        self._final_metrics_streams.clear()
        self._lane_metrics = {
            lane.stream_id: LaneMetrics(stream_id=lane.stream_id)
            for lane in self._config.lane_configs
        }
        self._lane_summaries.clear()
        self._paths.clear()
        self._failure = None
        self._run_dir = None
        self._completion_task = None
        self._resources_closed = False
        self._force_used = False
        self._shared_stop_expected = False
        self._qwen_run_active = False

    def _reset_for_run(self) -> None:
        self._started.clear()
        self._starting_streams.clear()
        self._start_stop_requested = False
        self._finished.clear()
        self._final_metrics_streams.clear()
        self._lane_summaries.clear()
        self._paths.clear()
        self._lane_metrics = {
            lane.stream_id: LaneMetrics(stream_id=lane.stream_id)
            for lane in self._config.lane_configs
        }
        self._completion_task = None

    def _resolve_input_commands(
        self, microphone_device_names: Mapping[int, str]
    ) -> dict[int, dict[str, Any]]:
        if self._fake:
            return {
                lane.stream_id: {"input_kind": "fake"}
                for lane in self._config.lane_configs
            }
        default_paths: list[Path] | None = None
        commands: dict[int, dict[str, Any]] = {}
        for lane in self._config.lane_configs:
            source = (
                self._lane_inputs[lane.stream_id]
                if self._lane_inputs
                else LaneInputConfig()
            )
            if source.kind == "microphone":
                commands[lane.stream_id] = {
                    "input_kind": "microphone",
                    "microphone_device": source.microphone_device,
                    "microphone_device_name": microphone_device_names[lane.stream_id],
                }
                continue
            if source.kind == "local_wav":
                assert source.local_wav_path is not None
                path = source.local_wav_path.expanduser().resolve(strict=True)
            else:
                if default_paths is None:
                    default_paths = select_input_files(
                        self._config.input_dir,
                        self._config.lanes,
                    )
                path = default_paths[lane.stream_id].resolve(strict=True)
            validate_wav_16k_mono_pcm(path)
            commands[lane.stream_id] = {
                "input_kind": "local_wav",
                "input_path": str(path),
                "loop_audio": source.loop_audio,
            }
        return commands

    def _resolve_microphone_inputs(self) -> dict[int, str]:
        if self._fake:
            return {}
        resolved: dict[int, str] = {}
        for lane in self._config.lane_configs:
            source = (
                self._lane_inputs[lane.stream_id]
                if self._lane_inputs
                else LaneInputConfig()
            )
            if source.kind != "microphone":
                continue
            try:
                resolved[lane.stream_id] = resolve_microphone_device(
                    source.microphone_device_id
                    if source.microphone_device_id is not None
                    else source.microphone_device
                )
            except Exception as exc:
                device = (
                    "default"
                    if source.microphone_device_id is None
                    and source.microphone_device is None
                    else source.microphone_device_id or source.microphone_device
                )
                raise ValueError(
                    f"lane {lane.stream_id + 1} microphone device {device} is unavailable: {exc}"
                ) from exc
        return resolved

    async def _wait_for(
        self,
        predicate: Any,
        *,
        timeout: float | None = None,
        fail_on_failure: bool = True,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while not predicate():
            if fail_on_failure and self._failure is not None:
                raise self._failure
            self._change_event.clear()
            if predicate():
                continue
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("lane lifecycle timed out")
            await asyncio.wait_for(self._change_event.wait(), timeout=remaining)

    async def _monitor_run_completion(self) -> None:
        def run_finalized() -> bool:
            return (
                len(self._finished) == self._config.lanes
                and len(self._final_metrics_streams) == self._config.lanes
                and all(
                    len(self._paths.get(lane.stream_id, {})) == len(_ARTIFACT_PATH_KEYS)
                    for lane in self._config.lane_configs
                )
            )

        while not run_finalized() and self._failure is None:
            await self._wait_for(
                lambda: run_finalized() or self._failure is not None,
                fail_on_failure=False,
            )
        if self._failure is None:
            await self._finish_qwen_run()
        if self._failure is not None:
            self._send_to_active({"type": "shutdown"})
            await self._join_or_escalate()
            await self._finish_readers_and_close()
            await self._finish_qwen_run()
        if self._failure is None:
            if self._state is PipelineState.RUNNING:
                assert_transition(self._state, PipelineState.STOPPING)
                self._state = PipelineState.STOPPING
            if self._state is PipelineState.STOPPING:
                assert_transition(self._state, PipelineState.LOADED)
                self._state = PipelineState.LOADED

    async def _read_lane_events(self, controller: LaneProcessController) -> None:
        stream_id = controller.lane.stream_id
        try:
            while True:
                try:
                    raw = await asyncio.to_thread(
                        controller.event_queue.get,
                        timeout=self._queue_poll_seconds,
                    )
                except queue.Empty:
                    if controller.is_alive():
                        if self._reader_stop.is_set():
                            return
                        continue
                    await controller.join(0.0)
                    await self._drain_dead_lane_events(controller)
                    if stream_id not in self._terminal:
                        await self._record_fatal(
                            stream_id,
                            f"lane {stream_id} exited before lane_stopped/fatal "
                            f"(exit code {controller.exitcode})",
                        )
                    self._exited.add(stream_id)
                    self._signal_change()
                    return
                message = LaneIpcMessage.from_dict(raw)
                self._validate_controller_message(controller, message)
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._resources_closed:
                await self._record_fatal(
                    stream_id,
                    f"IPC reader failed for lane {stream_id}: "
                    f"{type(exc).__name__}: {exc}",
                )
            self._signal_change()

    async def _drain_dead_lane_events(self, controller: LaneProcessController) -> None:
        while True:
            try:
                raw = await asyncio.to_thread(
                    controller.event_queue.get,
                    timeout=self._queue_poll_seconds,
                )
            except queue.Empty:
                return
            message = LaneIpcMessage.from_dict(raw)
            self._validate_controller_message(controller, message)
            await self._handle_message(message)

    def _validate_controller_message(
        self,
        controller: LaneProcessController,
        message: LaneIpcMessage,
    ) -> None:
        expected_stream_id = controller.lane.stream_id
        if message.type not in _ALLOWED_MESSAGE_TYPES:
            raise ValueError(f"unsupported IPC message type: {message.type}")
        if message.stream_id != expected_stream_id:
            raise ValueError(
                f"lane {expected_stream_id} received IPC for stream {message.stream_id}"
            )
        payload = message.payload
        if message.type == "asr_event":
            payload_stream_id = payload.get("stream_id")
            if (
                not isinstance(payload_stream_id, int)
                or isinstance(payload_stream_id, bool)
                or payload_stream_id != expected_stream_id
            ):
                raise ValueError(
                    f"asr_event payload stream {payload_stream_id} "
                    f"does not match lane {expected_stream_id}"
                )
            return
        if message.type in {"metrics_snapshot", "metrics_final"}:
            lane_metrics = payload.get("lane_metrics")
            if not isinstance(lane_metrics, Mapping) or not lane_metrics:
                raise ValueError(f"{message.type} requires nonempty lane_metrics")
            metrics_stream_id = lane_metrics.get("stream_id")
            if (
                not isinstance(metrics_stream_id, int)
                or isinstance(metrics_stream_id, bool)
                or metrics_stream_id != expected_stream_id
            ):
                raise ValueError(
                    f"{message.type} lane_metrics stream {metrics_stream_id} "
                    f"does not match lane {expected_stream_id}"
                )
            if message.type == "metrics_snapshot":
                if any(key in payload for key in _ARTIFACT_PATH_KEYS):
                    raise ValueError("metrics_snapshot must not carry artifact paths")
            else:
                ProcessPipelineManager._validate_artifact_paths(
                    message_type=message.type,
                    payload=payload,
                    stream_id=expected_stream_id,
                    required=False,
                )
            return
        if message.type == "lifecycle":
            lifecycle_state = payload.get("state")
            if lifecycle_state not in _LIFECYCLE_STATES:
                raise ValueError(f"invalid lifecycle state: {lifecycle_state}")
            return
        if message.type == "paths":
            ProcessPipelineManager._validate_artifact_paths(
                message_type=message.type,
                payload=payload,
                stream_id=expected_stream_id,
                required=True,
            )
            return
        if message.type == "fatal":
            fatal_message = payload.get("message")
            if not isinstance(fatal_message, str) or not fatal_message.strip():
                raise ValueError("fatal requires a nonempty message")
            return
        if message.type in {"qwen_success", "qwen_error"}:
            self._validate_qwen_message(message.type, payload)

    def _validate_qwen_message(
        self,
        message_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if not self._qwen_run_active:
            raise ValueError(f"{message_type} requires an active run")
        expected_keys = (
            _QWEN_SUCCESS_PAYLOAD_KEYS
            if message_type == "qwen_success"
            else _QWEN_ERROR_PAYLOAD_KEYS
        )
        actual_keys = frozenset(payload)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"{message_type} requires exact payload fields; "
                f"missing={missing}, extra={extra}"
            )
        active_run_id = None if self._run_dir is None else self._run_dir.name
        if payload.get("run_id") != active_run_id or active_run_id is None:
            raise ValueError(
                f"{message_type} run_id {payload.get('run_id')!r} "
                f"does not match active run {active_run_id!r}"
            )
        self._validate_qwen_integer(message_type, "sentence_id", payload["sentence_id"])
        if message_type == "qwen_success":
            return
        self._validate_qwen_integer(message_type, "revision", payload["revision"])
        self._validate_qwen_integer(message_type, "timestamp_ms", payload["timestamp_ms"])
        if not isinstance(payload["pending_text"], str):
            raise ValueError("qwen_error pending_text must be a string")
        request = payload["request"]
        if not isinstance(request, Mapping):
            raise ValueError("qwen_error request must be a mapping")
        request_fields = frozenset(request)
        missing_request_fields = sorted(_QWEN_REQUEST_FIELDS - request_fields)
        if missing_request_fields:
            raise ValueError(
                "qwen_error request missing required fields: "
                + ", ".join(missing_request_fields)
            )
        extra_request_fields = sorted(request_fields - _QWEN_REQUEST_FIELDS)
        if extra_request_fields:
            raise ValueError(
                "qwen_error request has unsupported fields: "
                + ", ".join(extra_request_fields)
            )
        self._validate_qwen_request(request)

    @staticmethod
    def _validate_qwen_integer(message_type: str, field: str, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{message_type} {field} must be an integer")
        if value < 0:
            raise ValueError(f"{message_type} {field} must be nonnegative")

    @staticmethod
    def _validate_qwen_request(request: Mapping[str, Any]) -> None:
        for field in ("category", "error_type", "message", "method", "url"):
            value = request[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"qwen_error request {field} must be a nonempty string")
        for field in ("timeout_ms", "sample_count"):
            value = request[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"qwen_error request {field} must be a nonnegative integer"
                )
        audio_seconds = request["audio_seconds"]
        if (
            not isinstance(audio_seconds, (int, float))
            or isinstance(audio_seconds, bool)
            or not math.isfinite(audio_seconds)
            or audio_seconds < 0
        ):
            raise ValueError(
                "qwen_error request audio_seconds must be a nonnegative finite number"
            )
        status_code = request["status_code"]
        if status_code is not None and (
            not isinstance(status_code, int) or isinstance(status_code, bool)
        ):
            raise ValueError("qwen_error request status_code must be an integer or null")
        response_body = request["response_body"]
        if response_body is not None and not isinstance(response_body, str):
            raise ValueError("qwen_error request response_body must be a string or null")
        if not isinstance(request["response_body_truncated"], bool):
            raise ValueError(
                "qwen_error request response_body_truncated must be a boolean"
            )

    @staticmethod
    def _validate_artifact_paths(
        *,
        message_type: str,
        payload: Mapping[str, Any],
        stream_id: int,
        required: bool,
    ) -> None:
        present = tuple(key in payload for key in _ARTIFACT_PATH_KEYS)
        if not any(present) and not required:
            return
        expected = {
            "results_path": f"results/lane{stream_id}.txt",
            "model_outputs_path": f"model_outputs/lane{stream_id}.jsonl",
        }
        if not all(present) or any(
            payload.get(key) != expected[key]
            for key in _ARTIFACT_PATH_KEYS
        ):
            raise ValueError(
                f"{message_type} requires exact lane {stream_id} artifact paths"
            )

    async def _handle_message(self, message: LaneIpcMessage) -> None:
        last_seq = self._last_seq.get(message.stream_id, -1)
        if message.seq <= last_seq:
            return
        self._last_seq[message.stream_id] = message.seq
        stream_id = message.stream_id
        if message.type == "asr_event":
            event = asr_event_from_payload(message.payload)
            self._event_bus.publish(event)
            if event.status is EventStatus.FATAL_ERROR:
                self._frontend_fatal_streams.add(stream_id)
        elif message.type == "lifecycle":
            state = str(message.payload.get("state", ""))
            if state == "lane_model_loaded":
                self._model_loaded.add(stream_id)
            elif state == "lane_loaded":
                self._loaded.add(stream_id)
            elif state == "lane_started":
                self._started.add(stream_id)
            elif state == "lane_finished":
                self._finished.add(stream_id)
            elif state == "lane_stopped":
                self._terminal.add(stream_id)
        elif message.type in {"metrics_snapshot", "metrics_final"}:
            self._merge_metrics(stream_id, message.type, message.payload)
        elif message.type == "paths":
            self._merge_paths(stream_id, message.payload)
        elif message.type == "fatal":
            self._terminal.add(stream_id)
            await self._record_fatal(
                stream_id,
                str(message.payload.get("message", "lane failed")),
            )
        elif message.type == "qwen_success":
            if self._qwen_errors_enabled:
                try:
                    self._qwen_errors.record_success()
                except Exception as exc:
                    await self._record_fatal(
                        stream_id,
                        "failed to checkpoint successful Qwen request: "
                        f"{type(exc).__name__}: {exc}",
                    )
        elif message.type == "qwen_error":
            if self._qwen_errors_enabled:
                payload = message.payload
                try:
                    decision = self._qwen_errors.record_failure(
                        stream_id=stream_id,
                        time_ms=message.time_ms,
                        payload=payload,
                    )
                except Exception as exc:
                    await self._record_fatal(
                        stream_id,
                        "failed to persist Qwen request error: "
                        f"{type(exc).__name__}: {exc}",
                    )
                else:
                    request = dict(payload["request"])
                    status = request.get("status_code")
                    text = f"Qwen request failed: {request['error_type']}"
                    if status is not None:
                        text += f" (HTTP {status})"
                    self._event_bus.publish(
                        ASREvent.asr_result(
                            stream_id=stream_id,
                            sentence_id=int(payload["sentence_id"]),
                            revision=int(payload["revision"]),
                            status=EventStatus.QWEN_ERROR,
                            text=text,
                            timestamp_ms=int(payload["timestamp_ms"]),
                        )
                    )
                    if decision.should_fatal:
                        await self._record_fatal(
                            stream_id,
                            "Qwen request failure threshold reached "
                            f"({decision.session_sequence}/3)",
                        )
        self._signal_change()

    async def _finish_qwen_run(self) -> bool:
        if not self._qwen_run_active:
            return True
        self._qwen_run_active = False
        if not self._qwen_errors_enabled:
            return True
        try:
            self._qwen_errors.finish_run()
        except Exception as exc:
            await self._record_fatal(
                0,
                "failed to finish Qwen error coordinator run: "
                f"{type(exc).__name__}: {exc}",
            )
            return False
        return True

    def _merge_metrics(self, stream_id: int, message_type: str, payload: dict[str, Any]) -> None:
        if message_type == "metrics_snapshot" and stream_id in self._final_metrics_streams:
            return
        metrics_payload = dict(payload.get("lane_metrics", {}))
        if stream_id in self._fatal_streams:
            metrics_payload["final_state"] = LaneFinalState.FATAL.value
        if metrics_payload:
            self._lane_metrics[stream_id] = lane_metrics_from_payload(metrics_payload)
        if message_type != "metrics_final":
            return
        self._final_metrics_streams.add(stream_id)
        summary = dict(payload)
        if metrics_payload:
            summary["lane_metrics"] = metrics_payload
        summary.update(self._paths.get(stream_id, {}))
        self._lane_summaries[stream_id] = summary

    def _merge_paths(self, stream_id: int, payload: dict[str, Any]) -> None:
        paths = {
            key: payload[key]
            for key in ("results_path", "model_outputs_path")
            if key in payload
        }
        self._paths.setdefault(stream_id, {}).update(paths)
        if stream_id in self._lane_summaries:
            self._lane_summaries[stream_id].update(paths)

    async def _record_fatal(self, stream_id: int, message: str) -> None:
        self._terminal.add(stream_id)
        self._fatal_streams.add(stream_id)
        if self._failure is None:
            self._failure = RuntimeError(message)
        current = self._lane_metrics.get(stream_id, LaneMetrics(stream_id=stream_id))
        self._lane_metrics[stream_id] = replace(current, final_state=LaneFinalState.FATAL)
        summary = self._lane_summaries.get(stream_id)
        if summary is not None:
            summary_metrics = dict(summary.get("lane_metrics", {}))
            summary_metrics["final_state"] = LaneFinalState.FATAL.value
            summary["lane_metrics"] = summary_metrics
        if self._state is not PipelineState.FATAL and self._state is not PipelineState.UNLOADED:
            assert_transition(self._state, PipelineState.FATAL)
            self._state = PipelineState.FATAL
        if self._run_dir is not None and stream_id not in self._frontend_fatal_streams:
            self._event_bus.publish(
                ASREvent.asr_result(
                    stream_id=stream_id,
                    sentence_id=0,
                    revision=0,
                    status=EventStatus.FATAL_ERROR,
                    text=message,
                    timestamp_ms=int(time.time() * 1000),
                )
            )
            self._frontend_fatal_streams.add(stream_id)
        self._send_to_active(
            {"type": "stop", "reason": "sibling_fatal"},
        )
        self._signal_change()

    def _send_to_active(
        self,
        command: dict[str, Any],
        *,
        exclude: set[int] | None = None,
    ) -> None:
        excluded = exclude or set()
        for stream_id, controller in self._controllers.items():
            if stream_id in excluded or not controller.is_alive():
                continue
            with contextlib.suppress(Exception):
                controller.send(dict(command))

    async def _join_or_escalate(self) -> None:
        async with self._process_shutdown_lock:
            active = {
                stream_id: controller
                for stream_id, controller in self._controllers.items()
                if controller.is_alive()
            }
            if not active:
                return
            await asyncio.gather(
                *(
                    controller.join(self._lifecycle_timeout_seconds)
                    for controller in active.values()
                )
            )
            timed_out = {
                stream_id: controller
                for stream_id, controller in active.items()
                if controller.is_alive()
            }
            for stream_id in timed_out:
                await self._record_fatal(
                    stream_id,
                    f"lane {stream_id} did not stop within "
                    f"{self._lifecycle_timeout_seconds:.3f}s",
                )
            forced = False
            for controller in timed_out.values():
                forced = (
                    await controller.terminate_then_kill(
                        terminate_timeout=self._terminate_timeout_seconds,
                        kill_timeout=self._kill_timeout_seconds,
                    )
                    or forced
                )
            self._force_used = self._force_used or forced

    async def _watch_shared_qwen(self) -> None:
        assert self._shared_manager is not None
        while True:
            await asyncio.sleep(self._shared_watch_interval_seconds)
            if self._shared_stop_expected:
                return
            if not self._shared_manager.is_alive():
                await self._record_fatal(
                    0,
                    "shared qwen exited unexpectedly "
                    f"(exit code {self._shared_manager.exit_code})",
                )
                return

    async def _stop_shared_qwen(self) -> None:
        self._shared_stop_expected = True
        watchdog = self._shared_watchdog_task
        self._shared_watchdog_task = None
        if watchdog is not None and not watchdog.done():
            watchdog.cancel()
        if watchdog is not None:
            await asyncio.gather(watchdog, return_exceptions=True)
        manager = self._shared_manager
        self._shared_manager = None
        if manager is not None:
            await asyncio.to_thread(manager.stop)

    async def _finish_readers_and_close(self) -> None:
        if self._resources_closed or any(
            controller.is_alive() for controller in self._controllers.values()
        ):
            return
        self._reader_stop.set()
        if self._reader_tasks:
            await asyncio.gather(*self._reader_tasks.values(), return_exceptions=True)
        for controller in self._controllers.values():
            controller.close_queues()
        for controller in self._controllers.values():
            controller.close_process()
        self._resources_closed = True

    def _signal_change(self) -> None:
        self._change_event.set()
