from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Callable

from multi_stream_asr.asr.offline.base import OfflineRecognizer
from multi_stream_asr.asr.online.base import OnlineRecognizer
from multi_stream_asr.audio.base import AudioSource
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.log.metrics import EventLoopLagTracker, LatencyTracker, RtfTracker
from multi_stream_asr.log.model_output import ModelOutputRecorder
from multi_stream_asr.models import LaneMetrics
from multi_stream_asr.pipeline.pipeline import LanePipeline
from multi_stream_asr.pipeline.state import PipelineState, assert_transition
from multi_stream_asr.web.event_bus import FrontendEventBus


_LOOP_RESULT_RETENTION_LIMIT = 100


AudioSourceFactory = Callable[[LaneConfig], AudioSource]
OnlineRecognizerFactory = Callable[[LaneConfig], OnlineRecognizer]
OfflineRecognizerFactory = Callable[[LaneConfig], OfflineRecognizer]


@dataclass
class _LoadedLane:
    lane: LaneConfig
    online_recognizer: OnlineRecognizer
    offline_recognizer: OfflineRecognizer


class PipelineManager:
    def __init__(
        self,
        *,
        config: AppConfig,
        event_bus: FrontendEventBus,
        llama_server_manager: object,
        source_factory: AudioSourceFactory,
        online_recognizer_factory: OnlineRecognizerFactory,
        offline_recognizer_factory: OfflineRecognizerFactory,
        pipeline_factory: type[LanePipeline] = LanePipeline,
        model_output_recorder: ModelOutputRecorder | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._llama_server_manager = llama_server_manager
        self._source_factory = source_factory
        self._online_recognizer_factory = online_recognizer_factory
        self._offline_recognizer_factory = offline_recognizer_factory
        self._pipeline_factory = pipeline_factory
        self._model_output_recorder = model_output_recorder
        self._lifecycle_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._state = PipelineState.UNLOADED
        self._loaded_lanes: list[_LoadedLane] = []
        self._sources: list[AudioSource] = []
        self._pipelines: list[LanePipeline] = []
        self._run_task: asyncio.Task[None] | None = None
        self._completion_task: asyncio.Task[None] | None = None
        self._failure: Exception | None = None
        self._latency_tracker = LatencyTracker()
        self._rtf_tracker = RtfTracker()
        self._event_loop_lag_tracker = EventLoopLagTracker()
        self._event_loop_lag_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def lane_metrics(self) -> list[LaneMetrics]:
        return [pipeline.metrics for pipeline in self._pipelines]

    @property
    def latency_tracker(self) -> LatencyTracker:
        return self._latency_tracker

    @property
    def rtf_tracker(self) -> RtfTracker:
        return self._rtf_tracker

    @property
    def event_loop_lag_tracker(self) -> EventLoopLagTracker:
        return self._event_loop_lag_tracker

    async def load_models(self) -> None:
        async with self._lifecycle_lock:
            await self._load_models_locked()

    async def _load_models_locked(self) -> None:
        async with self._state_lock:
            assert_transition(self._state, PipelineState.LOADING)
            self._state = PipelineState.LOADING

        loaded_lanes: list[_LoadedLane] = []
        try:
            await asyncio.to_thread(self._llama_server_manager.start_all)
            for lane in self._config.lane_configs:
                loaded_lanes.append(
                    _LoadedLane(
                        lane=lane,
                        online_recognizer=self._online_recognizer_factory(lane),
                        offline_recognizer=self._offline_recognizer_factory(lane),
                    )
                )
        except Exception as exc:
            await self._close_loaded_lanes(loaded_lanes)
            await asyncio.to_thread(self._llama_server_manager.stop_all)
            async with self._state_lock:
                assert_transition(self._state, PipelineState.FATAL)
                self._state = PipelineState.FATAL
                self._failure = exc
            raise

        async with self._state_lock:
            self._loaded_lanes = loaded_lanes
            assert_transition(self._state, PipelineState.LOADED)
            self._state = PipelineState.LOADED

    async def start_recognition(self) -> None:
        async with self._lifecycle_lock:
            await self._start_recognition_locked()

    async def _start_recognition_locked(self) -> None:
        async with self._state_lock:
            previous_state = self._state
            previous_failure = self._failure
            previous_sources = list(self._sources)
            previous_pipelines = list(self._pipelines)
            previous_run_task = self._run_task
            previous_completion_task = self._completion_task
            previous_latency_tracker = self._latency_tracker
            previous_rtf_tracker = self._rtf_tracker
            assert_transition(self._state, PipelineState.STARTING)
            self._state = PipelineState.STARTING

        latency_tracker = LatencyTracker()
        rtf_tracker = RtfTracker()
        event_loop_lag_tracker = EventLoopLagTracker()
        sources: list[AudioSource] = []
        pipelines: list[LanePipeline] = []
        try:
            for loaded_lane in self._loaded_lanes:
                source = self._source_factory(loaded_lane.lane)
                pipeline = self._pipeline_factory(
                    stream_id=loaded_lane.lane.stream_id,
                    online_recognizer=loaded_lane.online_recognizer,
                    offline_recognizer=loaded_lane.offline_recognizer,
                    event_bus=self._event_bus,
                    segment_cache=SegmentCache(
                        sample_rate=16000,
                        capacity_seconds=self._config.segment_cache_seconds,
                    ),
                    endpoint=self._config.endpoint,
                    latency_tracker=latency_tracker,
                    rtf_tracker=rtf_tracker,
                    model_output_recorder=self._model_output_recorder,
                )
                sources.append(source)
                pipelines.append(pipeline)
        except Exception:
            await self._stop_sources(tuple(sources))
            async with self._state_lock:
                self._state = previous_state
                self._failure = previous_failure
                self._sources = previous_sources
                self._pipelines = previous_pipelines
                self._run_task = previous_run_task
                self._completion_task = previous_completion_task
                self._latency_tracker = previous_latency_tracker
                self._rtf_tracker = previous_rtf_tracker
            raise

        previous_results = self._event_bus.snapshot_results()
        async with self._state_lock:
            try:
                self._event_bus.clear_results()
                self._event_bus.set_result_retention_limit(
                    _LOOP_RESULT_RETENTION_LIMIT
                    if any(
                        self._config.input_for_lane(stream_id).loop_audio
                        for stream_id in range(self._config.lanes)
                    )
                    else None
                )
                self._failure = None
                self._latency_tracker = latency_tracker
                self._rtf_tracker = rtf_tracker
                self._event_loop_lag_tracker = event_loop_lag_tracker
                self._sources = sources
                self._pipelines = pipelines
                self._event_loop_lag_task = asyncio.create_task(
                    self._track_event_loop_lag(event_loop_lag_tracker)
                )
                self._run_task = asyncio.create_task(self._run_lanes())
                self._completion_task = asyncio.create_task(self._wait_for_run_completion(self._run_task))
                assert_transition(self._state, PipelineState.RUNNING)
                self._state = PipelineState.RUNNING
            except Exception:
                self._event_bus.restore_results(previous_results)
                self._state = previous_state
                self._failure = previous_failure
                self._sources = previous_sources
                self._pipelines = previous_pipelines
                self._run_task = previous_run_task
                self._completion_task = previous_completion_task
                self._latency_tracker = previous_latency_tracker
                self._rtf_tracker = previous_rtf_tracker
                raise

    async def wait_for_completion(self) -> None:
        async with self._state_lock:
            completion_task = self._completion_task
            failure = self._failure
        if completion_task is None:
            if failure is not None:
                raise failure
            return
        await completion_task

    async def stop_recognition(self) -> None:
        async with self._lifecycle_lock:
            await self._stop_recognition_locked()

    async def _stop_recognition_locked(self) -> None:
        async with self._state_lock:
            if self._state is PipelineState.STOPPED:
                return
            assert_transition(self._state, PipelineState.STOPPING)
            self._state = PipelineState.STOPPING
            sources = tuple(self._sources)
            pipelines = tuple(self._pipelines)
            completion_task = self._completion_task

        for pipeline in pipelines:
            pipeline.request_stop()
        await self._stop_sources(sources)
        if completion_task is not None:
            await completion_task

    async def unload_models(self) -> None:
        async with self._lifecycle_lock:
            await self._unload_models_locked()

    async def _unload_models_locked(self) -> None:
        async with self._state_lock:
            current = self._state
            if current is PipelineState.FATAL:
                self._state = PipelineState.UNLOADED
            else:
                assert_transition(current, PipelineState.UNLOADING)
                self._state = PipelineState.UNLOADING
            loaded_lanes = list(self._loaded_lanes)
            self._loaded_lanes = []
            self._sources = []
            self._run_task = None
            self._completion_task = None
            event_loop_lag_task = self._event_loop_lag_task
            self._event_loop_lag_task = None
            self._pipelines = []

        if event_loop_lag_task is not None:
            event_loop_lag_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await event_loop_lag_task
        await self._close_loaded_lanes(loaded_lanes)
        await asyncio.to_thread(self._llama_server_manager.stop_all)

        async with self._state_lock:
            if current is not PipelineState.FATAL:
                assert_transition(self._state, PipelineState.UNLOADED)
                self._state = PipelineState.UNLOADED

    async def _run_lanes(self) -> None:
        lane_tasks = [
            asyncio.create_task(self._run_lane(pipeline=pipeline, source=source))
            for pipeline, source in zip(self._pipelines, self._sources, strict=True)
        ]
        try:
            await asyncio.gather(*lane_tasks)
        except asyncio.CancelledError:
            for task in lane_tasks:
                task.cancel()
            await asyncio.gather(*lane_tasks, return_exceptions=True)
            raise
        except Exception as exc:
            await self._stop_sources(tuple(self._sources))
            results = await asyncio.gather(*lane_tasks, return_exceptions=True)
            failure = self._failure or exc
            for result in results:
                if isinstance(result, Exception):
                    failure = self._failure or result
                    break
            raise failure

    async def _run_lane(self, *, pipeline: LanePipeline, source: AudioSource) -> None:
        try:
            await pipeline.run(source)
        except Exception as exc:
            pipeline.mark_fatal(str(exc))
            await self._handle_lane_failure(exc)
            raise

    async def _handle_lane_failure(self, exc: Exception) -> None:
        async with self._state_lock:
            if self._state is PipelineState.FATAL:
                return
            assert_transition(self._state, PipelineState.FATAL)
            self._state = PipelineState.FATAL
            self._failure = exc
            sources = tuple(self._sources)
            pipelines = tuple(self._pipelines)

        for pipeline in pipelines:
            pipeline.request_stop()
        await self._stop_sources(sources)

    async def _wait_for_run_completion(self, run_task: asyncio.Task[None]) -> None:
        try:
            await run_task
        finally:
            await self._finalize_after_run(run_task)
        if self._failure is not None:
            raise self._failure

    async def _finalize_after_run(self, completed_run_task: asyncio.Task[None]) -> None:
        event_loop_lag_task: asyncio.Task[None] | None = None
        async with self._state_lock:
            if self._run_task is completed_run_task:
                self._run_task = None
            if self._completion_task is asyncio.current_task():
                self._completion_task = None
            if self._state is PipelineState.FATAL:
                return
            if self._state is PipelineState.RUNNING:
                assert_transition(self._state, PipelineState.STOPPING)
                self._state = PipelineState.STOPPING
            if self._state is PipelineState.STOPPING:
                assert_transition(self._state, PipelineState.LOADED)
                self._state = PipelineState.LOADED
            event_loop_lag_task = self._event_loop_lag_task
            self._event_loop_lag_task = None
        if event_loop_lag_task is not None:
            event_loop_lag_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await event_loop_lag_task

    async def _track_event_loop_lag(
        self,
        tracker: EventLoopLagTracker,
        *,
        interval_seconds: float = 0.1,
    ) -> None:
        expected = time.perf_counter() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = time.perf_counter()
            tracker.add(max(0.0, (now - expected) * 1000.0))
            expected = now + interval_seconds

    async def _stop_sources(self, sources: tuple[AudioSource, ...]) -> None:
        if not sources:
            return
        await asyncio.gather(*(source.stop() for source in sources), return_exceptions=True)

    async def _close_loaded_lanes(self, loaded_lanes: list[_LoadedLane]) -> None:
        if not loaded_lanes:
            return
        await asyncio.gather(
            *(loaded_lane.offline_recognizer.close() for loaded_lane in loaded_lanes),
            return_exceptions=True,
        )
