from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path
from typing import Any, Callable

import numpy as np

from multi_stream_asr.asr.kws.sherpa_kws import (
    KWSResult,
    SherpaKeywordSpotter,
    SherpaKeywordSpotterSession,
)
from multi_stream_asr.asr.hotwords import load_hotwords, resolve_sherpa_hotwords
from multi_stream_asr.asr.offline.qwen3asr import Qwen3AsrClient
from multi_stream_asr.asr.online.zipformer import (
    ZipformerRecognizer,
    write_zipformer_provider_config,
)
from multi_stream_asr.audio.queue_source import QueueAudioSource
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.models import ASREvent, EventStatus
from multi_stream_asr.pipeline.pipeline import LanePipeline
from multi_stream_asr.session.zipformer_runtime import SharedZipformerRuntime
from multi_stream_asr.web import messages as msg
from multi_stream_asr.web.messages import asr_event_to_frontend

_logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]


class SessionEventBus:
    """Minimal event bus that forwards ASREvents to a frontend callback."""

    def __init__(self, callback: Callable[[ASREvent], None]) -> None:
        self._callback = callback

    def publish(self, event: ASREvent) -> None:
        self._callback(event)


class SharedModelManager:
    """Loads and holds the shared Zipformer and Qwen3 recognizer instances.

    Zipformer load/decode always runs on ``SharedZipformerRuntime``'s dedicated
    thread so SpacemiT EP does not segfault across threads.
    """

    def __init__(self, config: AppConfig, *, fake: bool = False) -> None:
        self._config = config
        self._fake = fake
        self._zipformer_recognizer: Any | None = None
        self._zipformer_runtime: SharedZipformerRuntime | None = None
        self._offline_recognizer: Any | None = None
        self._kws: SherpaKeywordSpotter | None = None
        self._hotwords: tuple[str, ...] = ()

    def set_hotwords(self, words: list[str]) -> None:
        """运行时热更新精修热词（在线 sherpa 热词需重启，默认关闭）。"""
        self._hotwords = tuple(words)
        if self._offline_recognizer is not None:
            self._offline_recognizer.set_hotwords(self._hotwords)
        _logger.info("hotwords updated at runtime: %d entries", len(self._hotwords))

    def reload_kws_keywords(self) -> None:
        """按 keywords_file 当前内容重建 KWS spotter（新会话生效）。"""
        if self._kws is None:
            raise RuntimeError("KWS 未加载（可能被配置禁用）")
        self._kws.reload()

    async def load(self) -> None:
        if self._fake:
            return
        if self._config.hotwords_online:
            hotwords_bpe, hotwords = resolve_sherpa_hotwords(
                self._config.hotwords_file,
                self._config.models.online_model_dir / "tokens.txt",
            )
        else:
            # 在线热词关闭（SpacemiT sherpa mbs 不生效）：只读词表注入 Qwen 精修
            hotwords_bpe, hotwords = None, load_hotwords(self._config.hotwords_file)
        self._hotwords = tuple(hotwords)
        provider_config = write_zipformer_provider_config(
            model_dir=self._config.models.online_model_dir,
            stream_id=0,
            lanes=self._config.lanes,
        )
        runtime = SharedZipformerRuntime()
        runtime.start()
        try:
            self._zipformer_recognizer = await asyncio.to_thread(
                runtime.call,
                lambda: self._load_zipformer(provider_config, hotwords_bpe),
            )
        except Exception:
            await asyncio.to_thread(runtime.close)
            raise
        self._zipformer_runtime = runtime
        self._offline_recognizer = Qwen3AsrClient(
            shared=self._config.shared_qwen,
            hotwords=self._hotwords,
        )
        _logger.info("shared Zipformer loaded on dedicated runtime thread")
        await self._load_kws()

    async def _load_kws(self) -> None:
        kws_cfg = self._config.kws
        if not kws_cfg.enabled:
            _logger.info("KWS disabled by config")
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
                "KWS load failed (model_dir=%s keywords=%s); continuing without KWS",
                kws_cfg.model_dir,
                kws_cfg.keywords_file,
            )
            return
        self._kws = spotter
        _logger.info(
            "shared KWS loaded model_dir=%s keywords=%s",
            kws_cfg.model_dir,
            kws_cfg.keywords_file,
        )

    def _load_zipformer(
        self,
        provider_config: Path,
        hotwords_bpe: Path | None = None,
    ) -> Any:
        import sherpa_onnx

        model_dir = self._config.models.online_model_dir

        # FunASR 格式 paraformer 包（am.mvn + config.yaml + model.onnx）：
        # 走 funasr_onnx 官方在线运行时（sherpa 转换版端点断句不可用）
        if (model_dir / "am.mvn").is_file() and (model_dir / "model.onnx").is_file():
            from multi_stream_asr.asr.online.funasr_paraformer import (
                FunAsrParaformerRecognizer,
            )
            _logger.info("在线模型为 FunASR paraformer 包，用 funasr_onnx 运行时: %s",
                         model_dir)
            return FunAsrParaformerRecognizer(model_dir)

        tokens = model_dir / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"required file not found: {tokens}")
        decoder = ZipformerRecognizer._pick_onnx(model_dir, "decoder")
        encoder = ZipformerRecognizer._pick_onnx(model_dir, "encoder", prefer_q=True)
        joiners = sorted(model_dir.glob("*joiner*.onnx"))
        is_paraformer = not joiners   # 无 joiner = paraformer 架构（非 transducer）
        joiner = joiners[0] if joiners else None

        decoding_method = "greedy_search"
        extra: dict[str, Any] = {}
        if hotwords_bpe is not None:
            # 热词偏置只在 modified_beam_search 下生效
            decoding_method = "modified_beam_search"
            extra["hotwords_file"] = str(hotwords_bpe)
            extra["hotwords_score"] = self._config.hotwords_score
            _logger.info(
                "zipformer hotwords enabled: file=%s score=%s",
                hotwords_bpe,
                self._config.hotwords_score,
            )

        if is_paraformer:
            # paraformer：EP 不支持其量化节点（实测编译报错），强制 CPU
            _logger.info("在线模型为 paraformer（无 joiner），使用 CPU provider")
            return sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                num_threads=1,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=self._config.endpoint.rule1_min_trailing_silence,
                rule2_min_trailing_silence=self._config.endpoint.rule2_min_trailing_silence,
                rule3_min_utterance_length=self._config.endpoint.rule3_min_utterance_length,
                decoding_method=decoding_method,
                provider="cpu",
                **extra,
            )

        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self._config.endpoint.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self._config.endpoint.rule2_min_trailing_silence,
            rule3_min_utterance_length=self._config.endpoint.rule3_min_utterance_length,
            decoding_method=decoding_method,
            provider=f"spacemit:{provider_config}",
            **extra,
        )

    def create_online_recognizer(self) -> Any:
        if self._fake:
            return _FakeOnlineRecognizer(LaneConfig(stream_id=0))
        if self._zipformer_recognizer is None or self._zipformer_runtime is None:
            raise RuntimeError("shared Zipformer model is not loaded")
        return ZipformerRecognizer(
            shared_recognizer=self._zipformer_recognizer,
            shared_runtime=self._zipformer_runtime,
            sample_rate=16000,
            feature_dim=80,
        )

    @property
    def kws_loaded(self) -> bool:
        return self._kws is not None

    def create_kws_session(self) -> SherpaKeywordSpotterSession | None:
        if self._fake or self._kws is None:
            return None
        return self._kws.create_session()

    @property
    def offline_recognizer(self) -> Any:
        if self._fake:
            return _FakeOfflineRecognizer(LaneConfig(stream_id=0))
        if self._offline_recognizer is None:
            raise RuntimeError("shared Qwen3 model is not loaded")
        return self._offline_recognizer

    async def close(self) -> None:
        if self._offline_recognizer is not None:
            await self._offline_recognizer.close()
            self._offline_recognizer = None
        if self._kws is not None:
            self._kws.close()
            self._kws = None
        self._zipformer_recognizer = None
        if self._zipformer_runtime is not None:
            await asyncio.to_thread(self._zipformer_runtime.close)
            self._zipformer_runtime = None


class _FakeOnlineRecognizer:
    def __init__(self, lane: LaneConfig) -> None:
        self._lane = lane
        self._outputs = [
            (f"fake partial lane {lane.stream_id}", 100, False),
            (f"fake endpoint lane {lane.stream_id}", 200, True),
        ]
        self._idx = 0

    async def start_run(self) -> None:
        return None

    async def accept(self, frame: Any) -> Any | None:
        from multi_stream_asr.models import AudioFrame, OnlineRecognition

        if getattr(frame, "is_finish", False):
            return None
        if self._idx >= len(self._outputs):
            return None
        text, timestamp_ms, is_endpoint = self._outputs[self._idx]
        self._idx += 1
        return OnlineRecognition(
            stream_id=self._lane.stream_id,
            sentence_id=0,
            revision=0,
            text=text,
            timestamp_ms=timestamp_ms,
            is_endpoint=is_endpoint,
        )

    async def reset(self) -> None:
        self._idx = 0


class _FakeOfflineRecognizer:
    def __init__(self, lane: LaneConfig) -> None:
        self._lane = lane

    async def recognize(self, pcm: np.ndarray, prompt: str) -> Any:
        from multi_stream_asr.asr.offline.base import OfflineRecognitionResult

        audio_seconds = max(float(np.asarray(pcm).size) / 16000.0, 1.0)
        return OfflineRecognitionResult(
            text=f"fake final lane {self._lane.stream_id}",
            elapsed_ms=400.0,
            audio_seconds=audio_seconds,
            rtf=0.4,
        )

    async def close(self) -> None:
        return None


class SingleProcessPipeline:
    """In-process lane runtime that uses shared model instances.

    Provides the same interface surface as ``LaneProcessController`` so that
    ``SessionManager`` can use it without knowing whether it is talking to a
    child process or an in-process pipeline.
    """

    def __init__(
        self,
        *,
        stream_id: int,
        app_config: AppConfig,
        shared_models: SharedModelManager,
        on_event: EventCallback,
        started_event: asyncio.Event,
        finished_event: asyncio.Event,
        input_sample_rate: int = 16000,
    ) -> None:
        self.stream_id = stream_id
        self.app_config = app_config
        self._on_event = on_event
        self._shared_models = shared_models
        self._started_event = started_event
        self._finished_event = finished_event
        self._input_sample_rate = input_sample_rate
        self._source = QueueAudioSource(
            stream_id=stream_id,
            audio_queue=queue.Queue(),
            frame_duration_ms=app_config.frame_duration_ms,
            input_sample_rate=input_sample_rate,
            speexdsp_lib_path=app_config.speexdsp_lib_path,
        )
        self.audio_queue = self._source.audio_queue
        self._pipeline = self._build_pipeline()
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _build_pipeline(self) -> LanePipeline:
        return LanePipeline(
            stream_id=self.stream_id,
            online_recognizer=self._shared_models.create_online_recognizer(),
            offline_recognizer=self._shared_models.offline_recognizer,
            event_bus=SessionEventBus(self._on_asr_event),
            segment_cache=SegmentCache(
                sample_rate=16000,
                capacity_seconds=self.app_config.segment_cache_seconds,
            ),
            endpoint=self.app_config.endpoint,
            kws_session=self._shared_models.create_kws_session(),
            on_kws=self._on_kws,
        )

    def _on_kws(self, result: KWSResult) -> None:
        _logger.info(
            "KWS event: stream=%d keyword=%r",
            self.stream_id,
            result.keyword,
        )
        payload = msg.kws_message(keyword=result.keyword)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._on_event, payload)
        else:
            self._on_event(payload)

    def _on_asr_event(self, event: ASREvent) -> None:
        _logger.info(
            "ASR event: stream=%d status=%s text=%r",
            event.stream_id,
            event.status.name if hasattr(event.status, "name") else event.status,
            getattr(event, "text", "")[:80],
        )
        if event.status is EventStatus.LANE_FINISHED:
            self._on_event(msg.finished_message(self.stream_id))
            return
        frontend_event = asr_event_to_frontend(event)
        if frontend_event is not None:
            self._on_event(frontend_event)

    def _on_source_ready(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._started_event.set)
        else:
            self._started_event.set()

    def is_alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, run_dir: Path) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(
            self._run_pipeline(),
            name=f"asr-inprocess-pipeline-{self.stream_id}",
        )

    async def restart(self, run_dir: Path) -> None:
        await self._cancel_task()
        self._started_event.clear()
        self._finished_event.clear()
        self._source = QueueAudioSource(
            stream_id=self.stream_id,
            audio_queue=queue.Queue(),
            frame_duration_ms=self.app_config.frame_duration_ms,
            input_sample_rate=self._input_sample_rate,
            speexdsp_lib_path=self.app_config.speexdsp_lib_path,
        )
        self.audio_queue = self._source.audio_queue
        self._pipeline = self._build_pipeline()
        self._task = None
        await self.start(run_dir)

    async def _run_pipeline(self) -> None:
        try:
            await self._pipeline.run(self._source, on_source_ready=self._on_source_ready)
        finally:
            self._finished_event.set()
            if not self._started_event.is_set():
                self._started_event.set()

    def send(self, command: dict[str, Any]) -> None:
        # In-process pipelines do not use the command queue; start/stop are
        # handled by ``start()`` / ``stop()`` directly.
        _logger.debug("ignoring command for in-process pipeline: %s", command)

    async def join(self, timeout: float) -> int | None:
        if self._task is None or self._task.done():
            return 0
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return 0

    async def terminate_then_kill(
        self, *, terminate_timeout: float, kill_timeout: float
    ) -> bool:
        return await self._cancel_task()

    def close_queues(self) -> None:
        pass

    def close_process(self) -> None:
        pass

    async def stop(self) -> None:
        await self._source.stop()
        await self._cancel_task()

    async def _cancel_task(self) -> bool:
        if self._task is None or self._task.done():
            return False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        return True
