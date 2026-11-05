from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import time
import traceback
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from multi_stream_asr.asr.kws.sherpa_kws import SherpaKeywordSpotter
from multi_stream_asr.asr.kws.keywords_tool import find_invalid_keyword_tokens
from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
from multi_stream_asr.asr.offline.qwen3asr import Qwen3AsrClient
from multi_stream_asr.asr.hotwords import load_hotwords, resolve_sherpa_hotwords
from multi_stream_asr.asr.online.zipformer import (
    ZipformerRecognizer,
    write_zipformer_provider_config,
)
from multi_stream_asr.audio.local_file import LocalWavSource
from multi_stream_asr.audio.mic import MicrophoneSource
from multi_stream_asr.audio.queue_source import QueueAudioSource
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.log.lane_artifacts import (
    DetailJsonlWriter,
    LaneModelOutputJsonlRecorder,
    LaneResultWriter,
)
from multi_stream_asr.log.metrics import (
    EventLoopLagTracker,
    LatencyTracker,
    RtfTracker,
    SharedTimingsTracker,
)
from multi_stream_asr.models import (
    ASREvent,
    AudioFrame,
    LaneFinalState,
    LaneMetrics,
    OnlineRecognition,
    QwenSentenceFailure,
)
from multi_stream_asr.pipeline.ipc import asr_event_to_payload, lane_metrics_to_payload, make_message
from multi_stream_asr.pipeline.pipeline import LanePipeline
from multi_stream_asr.pipeline.online_worker import OnlineConsumerWorker
from multi_stream_asr.speaker import (
    SherpaSpeakerEmbedder,
    SpeakerTracker,
    VoiceprintRegistry,
    verify_speaker_model,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaneProcessConfig:
    app_config: AppConfig
    lane: LaneConfig
    fake: bool = False
    streaming: bool = False


class _FakeAudioSource:
    def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
        self.stream_id = stream_id
        self.frame_duration_ms = frame_duration_ms
        self._stopped = False

    async def prepare(self) -> None:
        self._stopped = False

    async def frames(self) -> AsyncIterator[AudioFrame]:
        frames = (
            AudioFrame(
                self.stream_id,
                0,
                np.asarray([0.1, 0.2], dtype=np.float32),
                self.frame_duration_ms,
            ),
            AudioFrame(
                self.stream_id,
                1,
                np.asarray([0.3, 0.4], dtype=np.float32),
                self.frame_duration_ms * 2,
            ),
        )
        for frame in frames:
            if self._stopped:
                break
            yield frame
            await asyncio.sleep(0.01)
        yield AudioFrame.finish(
            self.stream_id,
            len(frames),
            self.frame_duration_ms * (len(frames) + 1),
        )

    async def stop(self) -> None:
        self._stopped = True


class _FakeOnlineRecognizer:
    def __init__(self, lane: LaneConfig) -> None:
        self._outputs = [
            OnlineRecognition(
                lane.stream_id,
                0,
                0,
                f"fake partial lane {lane.stream_id}",
                100,
            ),
            OnlineRecognition(
                lane.stream_id,
                0,
                0,
                f"fake endpoint lane {lane.stream_id}",
                200,
                True,
            ),
        ]

    async def start_run(self) -> None:
        return None

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None:
        if frame.is_finish:
            return None
        return self._outputs.pop(0) if self._outputs else None

    async def reset(self) -> None:
        return None


class _FakeOfflineRecognizer:
    def __init__(self, lane: LaneConfig) -> None:
        self._lane = lane

    async def recognize(self, pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult:
        audio_seconds = max(float(np.asarray(pcm).size) / 16000.0, 1.0)
        return OfflineRecognitionResult(
            text=f"fake final lane {self._lane.stream_id}",
            elapsed_ms=400.0,
            audio_seconds=audio_seconds,
            rtf=0.4,
        )

    async def close(self) -> None:
        return None


async def warmup_zipformer(
    recognizer: Any,
    *,
    stream_id: int,
    input_path: Path,
    frame_duration_ms: int,
) -> None:
    source = LocalWavSource(
        stream_id=stream_id,
        path=input_path,
        frame_duration_ms=frame_duration_ms,
        realtime=False,
    )
    await source.prepare()
    try:
        async for frame in source.frames():
            if frame.is_finish:
                break
            await recognizer.accept(frame)
        await recognizer.reset()
    finally:
        await source.stop()


class LaneEventSink:
    def __init__(self, *, runtime: "LaneRuntime", result_writer: LaneResultWriter) -> None:
        self._runtime = runtime
        self._result_writer = result_writer

    def publish(self, event: ASREvent) -> None:
        self._result_writer.record_event(event)
        self._runtime.send("asr_event", asr_event_to_payload(event))


class LaneRuntime:
    def __init__(
        self,
        *,
        process_config: LaneProcessConfig,
        event_queue: Any,
        command_queue: Any,
        audio_queue: Any | None = None,
    ) -> None:
        self.process_config = process_config
        self.event_queue = event_queue
        self.command_queue = command_queue
        self.audio_queue = audio_queue
        self._seq = 0
        self._online_recognizer: Any | None = None
        self._online_worker: OnlineConsumerWorker[Any] | None = None
        self._offline_recognizer: Any | None = None
        self._pipeline: LanePipeline | None = None
        self._source: Any | None = None
        self._runtime_stop_requested = False
        self._shutdown_requested = False
        self._fatal_phase = "load"
        self._terminal_metrics: LaneMetrics | None = None
        self._artifacts_created = False
        self._offline_closed = False
        self._lane_stopped_sent = False
        self._kws_spotter: SherpaKeywordSpotter | None = None
        self._speaker_registry: VoiceprintRegistry | None = None
        self._speaker_embedder: SherpaSpeakerEmbedder | None = None
        self._latency_tracker = LatencyTracker()
        self._rtf_tracker = RtfTracker()
        self._shared_timings_tracker = SharedTimingsTracker()
        self._event_loop_lag_tracker = EventLoopLagTracker()
        self._frame_latency_writer: DetailJsonlWriter | None = None
        self._event_loop_lag_writer: DetailJsonlWriter | None = None
        self._rtf_writer: DetailJsonlWriter | None = None
        self._shared_timing_writer: DetailJsonlWriter | None = None

    @property
    def lane(self) -> LaneConfig:
        return self.process_config.lane

    @property
    def app_config(self) -> AppConfig:
        return self.process_config.app_config

    def send(self, message_type: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        self.event_queue.put(
            make_message(
                message_type=message_type,
                stream_id=self.lane.stream_id,
                seq=self._seq,
                payload=payload,
            ).to_dict()
        )

    async def run(self) -> int:
        self._fatal_phase = "load"
        try:
            command = await self._load()
            self._fatal_phase = "run"
            if command is not None and command["type"] != "warmup":
                self._fatal_phase = "stop"
                self._terminal_metrics = LaneMetrics(stream_id=self.lane.stream_id)
                self.send("metrics_final", self._metrics_payload(include_paths=True))
                await self._shutdown()
                return 0
            has_started = False
            while True:
                command = await self._wait_for_command({"start", "stop", "shutdown"})
                if command["type"] == "shutdown":
                    self._fatal_phase = "stop"
                    await self._shutdown()
                    return 0
                if command["type"] == "stop":
                    if not has_started:
                        self._fatal_phase = "stop"
                        self._terminal_metrics = LaneMetrics(stream_id=self.lane.stream_id)
                        self.send("metrics_final", self._metrics_payload(include_paths=True))
                        await self._shutdown()
                        return 0
                    continue
                await self._run_started(command)
                has_started = True
                if self._shutdown_requested:
                    self._fatal_phase = "stop"
                    await self._shutdown()
                    return 0
        except BaseException as exc:
            self._send_fatal_outputs(exc, phase=self._fatal_phase)
            await self._shutdown(suppress_errors=True)
            return 1

    async def _load(self) -> dict[str, Any] | None:
        self.send("lifecycle", {"state": "lane_loading", "message": ""})
        if self.process_config.fake:
            self._online_recognizer = _FakeOnlineRecognizer(self.lane)
            self._offline_recognizer = _FakeOfflineRecognizer(self.lane)
        else:
            provider_config = write_zipformer_provider_config(
                model_dir=self.app_config.models.online_model_dir,
                stream_id=self.lane.stream_id,
                lanes=self.app_config.lanes,
            )
            if self.app_config.hotwords_online:
                hotwords_bpe, hotwords = resolve_sherpa_hotwords(
                    self.app_config.hotwords_file,
                    self.app_config.models.online_model_dir / "tokens.txt",
                )
            else:
                # 在线热词关闭（SpacemiT sherpa mbs 不生效）：只读词表注入 Qwen 精修
                hotwords_bpe, hotwords = None, load_hotwords(
                    self.app_config.hotwords_file
                )
            self._online_worker = OnlineConsumerWorker(
                name=f"asr-lane-{self.lane.stream_id}-zipformer-consumer",
                factory=lambda: ZipformerRecognizer(
                    self.app_config.models.online_model_dir,
                    self.app_config.endpoint,
                    provider_config=provider_config,
                    hotwords_file=hotwords_bpe,
                    hotwords_score=self.app_config.hotwords_score,
                ),
            )
            self._online_recognizer = await self._online_worker.wait_until_loaded()
            self._offline_recognizer = Qwen3AsrClient(
                shared=self.app_config.shared_qwen,
                hotwords=tuple(hotwords),
            )
            await self._load_kws()
            self._load_speaker()

        self.send(
            "lifecycle",
            {
                "state": "lane_model_loaded",
                "message": "",
                "kws_loaded": self._kws_spotter is not None,
            },
        )
        command = await self._wait_for_command({"warmup", "stop", "shutdown"})
        if command["type"] != "warmup":
            return command

        if not self.process_config.fake:
            assert self._online_worker is not None
            assert self._online_recognizer is not None
            input_path = Path(str(command["input_path"]))
            future = self._online_worker.run_action(
                lambda: asyncio.run(
                    warmup_zipformer(
                        self._online_recognizer,
                        stream_id=self.lane.stream_id,
                        input_path=input_path,
                        frame_duration_ms=self.app_config.frame_duration_ms,
                    )
                )
            )
            await asyncio.wrap_future(future)

        self.send("lifecycle", {"state": "lane_loaded", "message": ""})
        return None

    async def _load_kws(self) -> None:
        """每 lane 子进程独立加载 KWS（模型 3.3M，开销可忽略）；失败仅告警。"""
        kws_cfg = self.app_config.kws
        if not kws_cfg.enabled:
            _logger.info("KWS disabled by config (lane=%d)", self.lane.stream_id)
            return
        spotter = SherpaKeywordSpotter(
            model_dir=kws_cfg.model_dir,
            keywords_file=kws_cfg.keywords_file,
            sample_rate=kws_cfg.sample_rate,
            num_threads=kws_cfg.num_threads,
            max_active_paths=kws_cfg.max_active_paths,
            keywords_score=kws_cfg.keywords_score,
            keywords_threshold=kws_cfg.keywords_threshold,
            num_trailing_blanks=kws_cfg.num_trailing_blanks,
            provider=kws_cfg.provider,
            repeat_interval=kws_cfg.repeat_interval,
        )
        try:
            await asyncio.to_thread(spotter.load)
        except Exception:
            _logger.exception(
                "KWS load failed (lane=%d model_dir=%s); continuing without KWS",
                self.lane.stream_id,
                kws_cfg.model_dir,
            )
            return
        self._kws_spotter = spotter

    def _load_speaker(self) -> None:
        """声纹库 + 子进程预检；embedder 本身在每次 run 的 tracker 里懒加载。"""
        cfg = self.app_config.speaker
        if not cfg.enabled:
            return
        model_path = Path(cfg.model_path)
        if not model_path.is_file():
            _logger.warning(
                "声纹模型不存在，本 lane 关闭声纹: %s (lane=%d)",
                model_path,
                self.lane.stream_id,
            )
            return
        registry_dir = cfg.registry_dir or (self.app_config.output_dir / "voiceprints")
        # 坏模型（常见是缺 model_type 元数据）会让 sherpa-onnx 段错误，
        # 必须在子进程外预检后再上线。
        verdict = verify_speaker_model(model_path)
        if not verdict.ok:
            _logger.error(
                "声纹模型预检失败，本 lane 关闭声纹: %s —— %s",
                model_path,
                verdict.detail,
            )
            return
        self._speaker_registry = VoiceprintRegistry(registry_dir)
        # embedder 每 lane 只建一份并在加载期预热（sherpa 加载模型约 6.5s，
        # 若等到首个句子才建会直接把 FINAL 拖慢 6 秒多）
        try:
            self._speaker_embedder = SherpaSpeakerEmbedder(
                model_path=model_path,
                num_threads=cfg.num_threads,
                provider=cfg.provider,
                sample_rate=cfg.sample_rate,
                window_seconds=cfg.window_seconds,
                hop_seconds=cfg.hop_seconds,
                trim_floor_ratio=cfg.trim_floor_ratio,
            )
            t0 = time.monotonic()
            self._speaker_embedder.embed(np.zeros(cfg.sample_rate // 2, dtype=np.float32))
            _logger.info(
                "声纹模型已预热 lane=%d (%.1fs)", self.lane.stream_id, time.monotonic() - t0
            )
        except Exception:
            _logger.exception("声纹模型预热失败，本 lane 关闭声纹 lane=%d", self.lane.stream_id)
            self._speaker_embedder = None
            self._speaker_registry = None

    def _create_speaker_tracker(self) -> SpeakerTracker | None:
        """每次 run 新建一个 tracker；embedder 复用 lane 级缓存。"""
        cfg = self.app_config.speaker
        if self.process_config.fake or not cfg.enabled or self._speaker_registry is None:
            return None
        if self._speaker_embedder is None:
            return None
        return SpeakerTracker(
            embedder=self._speaker_embedder,
            config=cfg,
            registry=self._speaker_registry,
            stream_id=self.lane.stream_id,
        )

    def _apply_set_hotwords(self, command: dict[str, Any]) -> None:
        words = tuple(str(w) for w in command.get("words", ()))
        if self._offline_recognizer is not None:
            self._offline_recognizer.set_hotwords(words)
        _logger.info(
            "lane %d hotwords updated at runtime: %d entries",
            self.lane.stream_id,
            len(words),
        )

    def _apply_reload_kws(self) -> None:
        if self._kws_spotter is None:
            _logger.warning("reload_kws ignored: KWS not loaded (lane=%d)", self.lane.stream_id)
            return
        # sherpa 遇到未知 token 会在 C++ 层直接终止进程（Encode keywords
        # failed → abort），reload 前先在 Python 侧校验，非法词表保持旧词不动。
        kws_cfg = self.app_config.kws
        try:
            lines = [
                ln.strip()
                for ln in Path(kws_cfg.keywords_file).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            invalid = find_invalid_keyword_tokens(lines, Path(kws_cfg.model_dir) / "tokens.txt")
        except Exception:
            _logger.exception("KWS keywords 校验失败，跳过重载 (lane=%d)", self.lane.stream_id)
            return
        if invalid:
            _logger.error(
                "KWS keywords 含未知 token，跳过重载 (lane=%d): %s",
                self.lane.stream_id,
                invalid,
            )
            return
        try:
            self._kws_spotter.reload()
        except Exception:
            _logger.exception("KWS reload failed (lane=%d)", self.lane.stream_id)

    def _apply_reload_speaker(self) -> None:
        """REST 注册/删除声纹后热重载注册表（从磁盘重读，当前会话即时生效）。"""
        if self._speaker_registry is None:
            return
        try:
            self._speaker_registry.reload()
            _logger.info(
                "lane %d 声纹库已热重载（%d 人）",
                self.lane.stream_id,
                len(self._speaker_registry),
            )
        except Exception:
            _logger.exception("声纹库热重载失败 (lane=%d)", self.lane.stream_id)

    async def _run_started(self, command: dict[str, Any]) -> None:
        self._reset_for_run()
        run_dir = Path(str(command["run_dir"]))

        def report_qwen_success(sentence_id: int) -> None:
            self.send(
                "qwen_success",
                {
                    "run_id": run_dir.name,
                    "sentence_id": sentence_id,
                },
            )

        def report_qwen_error(failure: QwenSentenceFailure) -> None:
            self.send(
                "qwen_error",
                {
                    "run_id": run_dir.name,
                    "sentence_id": failure.sentence_id,
                    "revision": failure.revision,
                    "pending_text": failure.pending_text,
                    "timestamp_ms": failure.timestamp_ms,
                    "request": failure.request,
                },
            )

        def report_kws(result: Any) -> None:
            self.send("kws_event", {"keyword": result.keyword})

        def report_speaker(payload: dict[str, Any]) -> None:
            mapping = payload.get("mapping") if isinstance(payload, dict) else None
            self.send("speaker_remap", {"mapping": mapping})

        result_writer: LaneResultWriter | None = None
        model_output_recorder: LaneModelOutputJsonlRecorder | None = None
        frame_latency_writer: DetailJsonlWriter | None = None
        event_loop_lag_writer: DetailJsonlWriter | None = None
        rtf_writer: DetailJsonlWriter | None = None
        shared_timing_writer: DetailJsonlWriter | None = None
        operation_error: BaseException | None = None
        operation_traceback = None
        close_error: BaseException | None = None
        try:
            loop_audio = bool(command.get("loop_audio", False))
            result_writer = LaneResultWriter(
                run_dir / "results" / f"lane{self.lane.stream_id}.txt",
                enabled=not loop_audio,
            )
            model_output_recorder = LaneModelOutputJsonlRecorder(
                run_dir / "model_outputs" / f"lane{self.lane.stream_id}.jsonl",
                enabled=not loop_audio,
            )
            if self.app_config.verbose and not loop_audio:
                detail_dir = run_dir / "metrics_detail"
                lane_filename = f"lane{self.lane.stream_id}.jsonl"
                frame_latency_writer = DetailJsonlWriter(detail_dir / "frame_latency" / lane_filename)
                event_loop_lag_writer = DetailJsonlWriter(detail_dir / "event_loop_lag" / lane_filename)
                rtf_writer = DetailJsonlWriter(detail_dir / "rtf" / lane_filename)
                shared_timing_writer = DetailJsonlWriter(detail_dir / "shared_timing" / lane_filename)
                self._event_loop_lag_writer = event_loop_lag_writer
            self._artifacts_created = True
            if self._online_recognizer is None or self._offline_recognizer is None:
                raise RuntimeError("lane recognizers are not loaded")
            self._source = self._build_source(command)
            self._pipeline = LanePipeline(
                stream_id=self.lane.stream_id,
                online_recognizer=self._online_recognizer,
                offline_recognizer=self._offline_recognizer,
                event_bus=LaneEventSink(runtime=self, result_writer=result_writer),
                segment_cache=SegmentCache(
                    sample_rate=16000,
                    capacity_seconds=self.app_config.segment_cache_seconds,
                ),
                endpoint=self.app_config.endpoint,
                latency_tracker=self._latency_tracker,
                rtf_tracker=self._rtf_tracker,
                shared_timings_tracker=self._shared_timings_tracker,
                model_output_recorder=model_output_recorder,
                frame_latency_writer=frame_latency_writer,
                rtf_writer=rtf_writer,
                shared_timing_writer=shared_timing_writer,
                consumer_runner=(
                    self._online_worker.run_consumer
                    if self._online_worker is not None
                    else None
                ),
                on_qwen_success=report_qwen_success,
                on_qwen_error=report_qwen_error,
                kws_session=(
                    self._kws_spotter.create_session()
                    if self._kws_spotter is not None
                    else None
                ),
                on_kws=report_kws,
                speaker_tracker=self._create_speaker_tracker(),
                on_speaker=report_speaker,
                offline_tail_timeout_seconds=(
                    self.app_config.shared_qwen.request_timeout_ms / 1000.0
                ),
            )
            self._runtime_stop_requested = False
            source_ready = asyncio.Event()
            lane_loop = asyncio.get_running_loop()
            pipeline_task = asyncio.create_task(
                self._pipeline.run(
                    self._source,
                    on_source_ready=lambda: lane_loop.call_soon_threadsafe(
                        source_ready.set
                    ),
                )
            )
            lag_task = asyncio.create_task(self._track_event_loop_lag())
            snapshot_task = asyncio.create_task(self._emit_metrics_snapshots())
            command_task = asyncio.create_task(self._watch_runtime_commands())
            source_ready_task = asyncio.create_task(source_ready.wait())
            ready_or_stopped, _ = await asyncio.wait(
                (
                    source_ready_task,
                    pipeline_task,
                    command_task,
                    lag_task,
                    snapshot_task,
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if (
                source_ready_task in ready_or_stopped
                and not self._runtime_stop_requested
            ):
                self.send("lifecycle", {"state": "lane_started", "message": ""})
            else:
                source_ready_task.cancel()
                await asyncio.gather(source_ready_task, return_exceptions=True)
            run_error = await self._supervise_started_tasks(
                pipeline_task=pipeline_task,
                command_task=command_task,
                lag_task=lag_task,
                snapshot_task=snapshot_task,
            )

            tail_task = asyncio.create_task(
                self._pipeline.wait_for_offline_tail(
                    timeout_seconds=self.app_config.shared_qwen.request_timeout_ms / 1000.0
                )
            )
            tail_result, tail_cancellation = await self._await_task_quiescent(tail_task)
            if run_error is None and tail_cancellation is not None:
                run_error = tail_cancellation
            tail_error = tail_result if isinstance(tail_result, BaseException) else None

            if run_error is not None:
                raise run_error
            if tail_error is not None:
                raise tail_error
            self.send("metrics_final", self._metrics_payload(include_paths=True))
            if self._artifacts_created:
                self.send("paths", self._paths_payload())
            self.send("lifecycle", {"state": "lane_finished", "message": ""})
        except BaseException as exc:
            operation_error = exc
            operation_traceback = exc.__traceback__
        finally:
            if result_writer is not None:
                try:
                    result_writer.close()
                except BaseException as exc:
                    close_error = exc
            if model_output_recorder is not None:
                try:
                    model_output_recorder.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
            for writer in (
                frame_latency_writer,
                event_loop_lag_writer,
                rtf_writer,
                shared_timing_writer,
            ):
                if writer is not None:
                    try:
                        writer.close()
                    except BaseException as exc:
                        if close_error is None:
                            close_error = exc

        if operation_error is not None:
            raise operation_error.with_traceback(operation_traceback)
        if close_error is not None:
            raise close_error

    def _reset_for_run(self) -> None:
        self._pipeline = None
        self._source = None
        self._runtime_stop_requested = False
        self._shutdown_requested = False
        self._terminal_metrics = None
        self._artifacts_created = False
        self._latency_tracker = LatencyTracker()
        self._rtf_tracker = RtfTracker()
        self._shared_timings_tracker = SharedTimingsTracker()
        self._event_loop_lag_tracker = EventLoopLagTracker()
        self._frame_latency_writer = None
        self._event_loop_lag_writer = None
        self._rtf_writer = None
        self._shared_timing_writer = None

    async def _supervise_started_tasks(
        self,
        *,
        pipeline_task: asyncio.Task[None],
        command_task: asyncio.Task[None],
        lag_task: asyncio.Task[None],
        snapshot_task: asyncio.Task[None],
    ) -> BaseException | None:
        caller_cancellation: asyncio.CancelledError | None = None
        background_error: BaseException | None = None
        source_stop_error: BaseException | None = None
        supervised = {pipeline_task, command_task, lag_task, snapshot_task}

        try:
            done, _ = await asyncio.wait(
                supervised,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pipeline_task not in done:
                background_error = self._background_task_error(
                    command_task=command_task,
                    lag_task=lag_task,
                    snapshot_task=snapshot_task,
                )
        except asyncio.CancelledError as exc:
            caller_cancellation = exc

        if not pipeline_task.done():
            self._pipeline.request_stop()
            stop_task = asyncio.create_task(self._source.stop())
            stop_result, stop_cancellation = await self._await_task_quiescent(stop_task)
            if caller_cancellation is None and stop_cancellation is not None:
                caller_cancellation = stop_cancellation
            if isinstance(stop_result, BaseException):
                source_stop_error = stop_result

        pipeline_result, pipeline_cancellation = await self._await_task_quiescent(
            pipeline_task
        )
        if caller_cancellation is None and pipeline_cancellation is not None:
            caller_cancellation = pipeline_cancellation
        pipeline_error = (
            pipeline_result if isinstance(pipeline_result, BaseException) else None
        )

        for task in (command_task, lag_task, snapshot_task):
            if not task.done():
                task.cancel()
        background_join = asyncio.ensure_future(
            asyncio.gather(
                command_task,
                lag_task,
                snapshot_task,
                return_exceptions=True,
            )
        )
        _, background_cancellation = await self._await_task_quiescent(background_join)
        if caller_cancellation is None and background_cancellation is not None:
            caller_cancellation = background_cancellation
        if background_error is None:
            background_error = self._background_task_error(
                command_task=command_task,
                lag_task=lag_task,
                snapshot_task=snapshot_task,
            )
        if background_error is None:
            background_error = source_stop_error

        return caller_cancellation or pipeline_error or background_error

    @staticmethod
    def _background_task_error(
        *,
        command_task: asyncio.Task[None],
        lag_task: asyncio.Task[None],
        snapshot_task: asyncio.Task[None],
    ) -> BaseException | None:
        for name, task, normal_return_allowed in (
            ("command watcher", command_task, True),
            ("event-loop lag tracker", lag_task, False),
            ("metrics snapshot task", snapshot_task, False),
        ):
            if not task.done() or task.cancelled():
                continue
            exception = task.exception()
            if exception is not None:
                return exception
            if not normal_return_allowed:
                return RuntimeError(f"{name} stopped unexpectedly")
        return None

    @staticmethod
    async def _await_task_quiescent(
        task: asyncio.Future,
    ) -> tuple[object, asyncio.CancelledError | None]:
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
        try:
            return task.result(), cancellation
        except BaseException as exc:
            return exc, cancellation

    def _build_source(self, command: dict[str, Any]) -> Any:
        if self.process_config.fake:
            return _FakeAudioSource(self.lane.stream_id, self.app_config.frame_duration_ms)
        if self.process_config.streaming and self.audio_queue is not None:
            return QueueAudioSource(
                stream_id=self.lane.stream_id,
                audio_queue=self.audio_queue,
                frame_duration_ms=self.app_config.frame_duration_ms,
                input_sample_rate=int(command.get("input_sample_rate", 16000)),
                speexdsp_lib_path=self.app_config.speexdsp_lib_path,
            )
        input_kind = command.get("input_kind", "local_wav")
        if input_kind == "microphone":
            device = command.get("microphone_device")
            if device is not None and (not isinstance(device, int) or isinstance(device, bool)):
                raise ValueError("microphone_device must be an integer or null")
            device_name = command.get("microphone_device_name")
            if device_name is not None and (
                not isinstance(device_name, str) or not device_name.strip()
            ):
                raise ValueError("microphone_device_name must be a non-empty string or null")
            return MicrophoneSource(
                stream_id=self.lane.stream_id,
                device=device_name or device,
                frame_duration_ms=self.app_config.frame_duration_ms,
            )
        if input_kind != "local_wav" or "input_path" not in command:
            raise ValueError(f"invalid lane input descriptor: {input_kind}")
        return LocalWavSource(
            stream_id=self.lane.stream_id,
            path=Path(str(command["input_path"])),
            frame_duration_ms=self.app_config.frame_duration_ms,
            realtime=True,
            loop_audio=bool(command.get("loop_audio", False)),
        )

    async def _wait_for_command(self, allowed: set[str]) -> dict[str, Any]:
        while True:
            try:
                command = await asyncio.to_thread(self.command_queue.get, timeout=0.1)
            except queue.Empty:
                continue
            command_type = str(command.get("type", ""))
            if command_type == "set_hotwords":
                self._apply_set_hotwords(command)
                continue
            if command_type == "reload_kws":
                self._apply_reload_kws()
                continue
            if command_type == "reload_speaker":
                self._apply_reload_speaker()
                continue
            if command_type in allowed:
                return dict(command)

    async def _watch_runtime_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            command_type = str(command.get("type", ""))
            if command_type == "set_hotwords":
                self._apply_set_hotwords(command)
                continue
            if command_type == "reload_kws":
                self._apply_reload_kws()
                continue
            if command_type == "reload_speaker":
                self._apply_reload_speaker()
                continue
            if command_type not in {"stop", "shutdown"}:
                continue
            self._runtime_stop_requested = True
            self._shutdown_requested = command_type == "shutdown"
            self._fatal_phase = "stop"
            if self._pipeline is not None:
                self._pipeline.request_stop()
                # 用户 stop 也要取消离线尾巴，否则长会话积压会拖死收尾；
                # 自然 finish（push None）仍走带超时的 _finish_after_offline。
                self._pipeline.cancel_offline_tasks()
            if self._source is not None:
                await self._source.stop()
            return

    async def _track_event_loop_lag(self, interval_seconds: float = 0.1) -> None:
        expected = time.perf_counter() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = time.perf_counter()
            lag_ms = max(0.0, (now - expected) * 1000.0)
            self._event_loop_lag_tracker.add(lag_ms)
            if self._event_loop_lag_writer is not None:
                self._event_loop_lag_writer.record({"lag_ms": lag_ms})
            expected = now + interval_seconds

    async def _emit_metrics_snapshots(self, interval_seconds: float = 0.1) -> None:
        while True:
            self.send("metrics_snapshot", self._metrics_payload(include_paths=False))
            await asyncio.sleep(interval_seconds)

    def _metrics_payload(self, *, include_paths: bool) -> dict[str, Any]:
        if self._terminal_metrics is not None:
            metrics = self._terminal_metrics
        elif self._pipeline is not None:
            metrics = self._pipeline.metrics
        else:
            metrics = LaneMetrics(self.lane.stream_id)
        payload: dict[str, Any] = {
            "lane_metrics": lane_metrics_to_payload(metrics),
            "online_frame_latency": self._latency_tracker.as_dict(),
            "offline_rtf": self._rtf_tracker.as_dict(),
            "offline_shared_timings": self._shared_timings_tracker.as_dict(),
            "event_loop_lag": self._event_loop_lag_tracker.as_dict(),
        }
        if include_paths and self._artifacts_created:
            payload.update(self._paths_payload())
        return payload

    def _paths_payload(self) -> dict[str, str]:
        return {
            "results_path": f"results/lane{self.lane.stream_id}.txt",
            "model_outputs_path": f"model_outputs/lane{self.lane.stream_id}.jsonl",
        }

    def _send_fatal_outputs(self, exc: BaseException, *, phase: str) -> None:
        if self._pipeline is not None:
            if phase == "run":
                with contextlib.suppress(BaseException):
                    self._pipeline.mark_fatal(str(exc))
            self._terminal_metrics = replace(
                self._pipeline.metrics,
                final_state=LaneFinalState.FATAL,
            )
        else:
            self._terminal_metrics = LaneMetrics(
                stream_id=self.lane.stream_id,
                final_state=LaneFinalState.FATAL,
            )

        if phase != "load":
            with contextlib.suppress(BaseException):
                self.send("metrics_final", self._metrics_payload(include_paths=True))
            if self._artifacts_created:
                with contextlib.suppress(BaseException):
                    self.send("paths", self._paths_payload())
        with contextlib.suppress(BaseException):
            self.send(
                "fatal",
                {
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                    "exit_code": 1,
                },
            )
        with contextlib.suppress(BaseException):
            self.send("lifecycle", {"state": "lane_failed", "message": str(exc)})

    async def _shutdown(self, *, suppress_errors: bool = False) -> None:
        errors: list[BaseException] = []
        if self._pipeline is not None:
            with contextlib.suppress(BaseException):
                self._pipeline.request_stop()
        if self._source is not None:
            try:
                await self._source.stop()
            except BaseException as exc:
                errors.append(exc)
        if self._offline_recognizer is not None:
            if not self._offline_closed:
                try:
                    await self._offline_recognizer.close()
                    self._offline_closed = True
                except BaseException as exc:
                    errors.append(exc)
        if self._online_worker is not None:
            try:
                await self._online_worker.close()
            except BaseException as exc:
                errors.append(exc)
        if (not errors or suppress_errors) and not self._lane_stopped_sent:
            try:
                self.send("lifecycle", {"state": "lane_stopped", "message": ""})
                self._lane_stopped_sent = True
            except BaseException as exc:
                errors.append(exc)
        if errors and not suppress_errors:
            raise errors[0]


def run_lane_process(
    process_config: LaneProcessConfig,
    event_queue: Any,
    command_queue: Any,
    audio_queue: Any | None = None,
) -> None:
    runtime = LaneRuntime(
        process_config=process_config,
        event_queue=event_queue,
        command_queue=command_queue,
        audio_queue=audio_queue,
    )
    raise SystemExit(asyncio.run(runtime.run()))
