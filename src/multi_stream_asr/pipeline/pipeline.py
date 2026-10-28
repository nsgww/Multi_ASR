from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import numpy as np
import os
from pathlib import Path
from typing import Any

from multi_stream_asr.asr.kws.sherpa_kws import KWSResult, SherpaKeywordSpotterSession
from multi_stream_asr.asr.offline.base import OfflineRecognizer
from multi_stream_asr.asr.offline.qwen3asr import QwenRequestError
from multi_stream_asr.asr.online.base import OnlineRecognizer
from multi_stream_asr.audio.base import AudioSource
from multi_stream_asr.buffer.audio_queue import OnlineFrameQueue
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import EndpointConfig, DEFAULT_VAD_MODEL_DIR
from multi_stream_asr.log.lane_artifacts import DetailJsonlWriter
from multi_stream_asr.log.metrics import (
    SHARED_TIMING_FIELDS,
    LatencyTracker,
    RtfTracker,
    SharedTimingsTracker,
)
from multi_stream_asr.log.model_output import ModelOutputRecord, ModelOutputRecorder
_logger = logging.getLogger(__name__)

# 语气词/无意义短文本：噪声被识别器误认的产物，不送校正不显示
_NOISE_TEXT_RE = re.compile(
    r"^[嗯啊哦呃额哼唉哎嘿嘻咳哼啧]+$")
_NOISE_WORDS = {"好", "好的", "是", "是的", "对", "谢谢", "再见", "再见啦",
                "拜拜", "嗯嗯", "啊啊", "哦哦"}


def _is_noise_text(text: str) -> bool:
    """识别结果是否视为噪声：清洗后不足 2 字，或纯语气词。"""
    clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)
    if len(clean) < 2:
        return True
    return clean in _NOISE_WORDS or bool(_NOISE_TEXT_RE.match(clean))


from multi_stream_asr.models import (
    ASREvent,
    AudioFrame,
    EventStatus,
    LaneFinalState,
    LaneMetrics,
    OnlineRecognition,
    QwenSentenceFailure,
    SpeakerInfo,
)
from multi_stream_asr.web.event_bus import FrontendEventBus


class LanePipeline:
    def __init__(
        self,
        *,
        stream_id: int,
        online_recognizer: OnlineRecognizer,
        offline_recognizer: OfflineRecognizer,
        event_bus: FrontendEventBus,
        segment_cache: SegmentCache,
        endpoint: EndpointConfig | None = None,
        latency_tracker: LatencyTracker | None = None,
        rtf_tracker: RtfTracker | None = None,
        shared_timings_tracker: SharedTimingsTracker | None = None,
        model_output_recorder: ModelOutputRecorder | None = None,
        frame_latency_writer: DetailJsonlWriter | None = None,
        rtf_writer: DetailJsonlWriter | None = None,
        shared_timing_writer: DetailJsonlWriter | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
        consumer_runner: (
            Callable[[Callable[[], None]], concurrent.futures.Future[None]] | None
        ) = None,
        on_qwen_success: Callable[[int], None] | None = None,
        on_qwen_error: Callable[[QwenSentenceFailure], None] | None = None,
        kws_session: SherpaKeywordSpotterSession | None = None,
        on_kws: Callable[[KWSResult], None] | None = None,
        speaker_tracker: Any | None = None,
        on_speaker: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.online_recognizer = online_recognizer
        self.offline_recognizer = offline_recognizer
        self.event_bus = event_bus
        self.segment_cache = segment_cache
        # 说话人跟踪器由 lane 进程（或会话管理器）按 lane 创建；None = 声纹关闭
        self.speaker_tracker = speaker_tracker
        self.on_speaker = on_speaker
        self._last_speaker: SpeakerInfo | None = None
        self.endpoint = endpoint or EndpointConfig()
        self.latency_tracker = latency_tracker or LatencyTracker()
        self.rtf_tracker = rtf_tracker or RtfTracker()
        self.shared_timings_tracker = shared_timings_tracker or SharedTimingsTracker()
        self.model_output_recorder = model_output_recorder
        self.frame_latency_writer = frame_latency_writer
        self.rtf_writer = rtf_writer
        self.shared_timing_writer = shared_timing_writer
        self._monotonic = monotonic
        self._consumer_runner = consumer_runner
        self.on_qwen_success = on_qwen_success
        self.on_qwen_error = on_qwen_error
        self.kws_session = kws_session
        self.on_kws = on_kws
        self._metrics_lock = threading.Lock()
        self._metrics = LaneMetrics(stream_id=stream_id)

        self._sentence_id = 0
        self._revision = 0
        self._current_online_text = ""
        self._offline_tasks: set[asyncio.Task[None]] = set()
        self._offline_tail: asyncio.Task[None] | None = None
        self._offline_scheduling_closed = False
        self._offline_error: BaseException | None = None
        self._last_event_sentence_id = 0
        self._last_event_revision = 0
        self._last_event_timestamp_ms = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event_loop_thread_id: int | None = None
        self._stop_requested: threading.Event | None = None
        self._frame_queue: OnlineFrameQueue | None = None

        # 断句模式：vad（VAD 主导，模型无关）/ recognizer（识别器 blank
        # 端点，仅 transducer 系模型可靠）。默认识别器自己声明：vad_driven=True
        # 的模型（如 funasr paraformer）自动走 VAD，zipformer 走端点。
        # ENDPOINT_MODE 环境变量可强制覆盖（vad / recognizer）。
        vad_driven = getattr(self.online_recognizer, "vad_driven", False)
        default_mode = "vad" if vad_driven else "recognizer"
        self._endpoint_mode = os.getenv("ENDPOINT_MODE", "").strip().lower() \
            or default_mode
        # vad_driven（funasr paraformer）demo 式 silero 门控状态：
        # silero 逐帧判定同时管断句和喂模型——非语音帧不进识别器
        # （噪声不会变成字、在线缓存不被静音喂到漂移），句尾 is_final
        # 冲刷后整句 reset。参数语义对齐 streaming_demo.py。
        self._rec_vad_driven = vad_driven
        self._pv_silero: Any | None = None          # (ort session, state, sr)
        self._pv_buf = np.zeros(0, dtype=np.float32)  # 待判定的残余采样
        self._pv_in_speech = False
        self._pv_loud = 0                            # 连续超阈值窗口数
        self._pv_quiet = 0                           # 连续低于阈值的采样数
        self._pv_pre: list[np.ndarray] = []          # 开口前预缓冲
        self._pv_seg: list[np.ndarray] = []          # 当前句音频
        self._pv_threshold = float(os.getenv("VAD_THRESHOLD", "0.2"))
        self._pv_loud_trigger = int(os.getenv("VAD_LOUD_TRIGGER", "3"))
        self._pv_min_silence_samples = int(
            float(os.getenv("VAD_MIN_SILENCE", "0.5")) * 16000)
        self._pv_min_speech_samples = int(
            float(os.getenv("VAD_MIN_SPEECH", "0.3")) * 16000)
        self._pv_pre_samples = int(
            float(os.getenv("VAD_PRE_BUFFER", "0.3")) * 16000)
        self._vad_since_cut = 0                      # 距上次断句的采样数（强制上限用）
        self._vad_max_utt_samples = int(
            float(os.getenv("VAD_MAX_UTT", "25")) * 16000)

    @property
    def metrics(self) -> LaneMetrics:
        with self._metrics_lock:
            return self._metrics

    def _increment_metrics(
        self,
        *,
        processed_frames: int = 0,
        dropped_frames: int = 0,
        backlogged_frames: int = 0,
        offline_requests: int = 0,
    ) -> None:
        with self._metrics_lock:
            self._metrics = replace(
                self._metrics,
                processed_frames=self._metrics.processed_frames + processed_frames,
                dropped_frames=self._metrics.dropped_frames + dropped_frames,
                backlogged_frames=self._metrics.backlogged_frames + backlogged_frames,
                offline_requests=self._metrics.offline_requests + offline_requests,
            )

    def _set_final_state(self, final_state: LaneFinalState) -> None:
        with self._metrics_lock:
            self._metrics = replace(self._metrics, final_state=final_state)

    async def run(
        self,
        source: AudioSource,
        *,
        on_source_ready: Callable[[], None] | None = None,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._event_loop_thread_id = threading.get_ident()
        self._offline_scheduling_closed = False
        # 缓冲 50 帧（100ms/帧 ≈ 5s）：纠错/Qwen 抢占 CPU 时消费会短暂落后，
        # 缓冲太小（原 10 帧 ≈ 1s）会触发丢最老帧，整句语音被静默丢弃
        frame_queue = OnlineFrameQueue(
            maxsize=int(os.getenv("FRAME_QUEUE_SIZE", "50"))
        )
        self._frame_queue = frame_queue
        producer_done = threading.Event()
        stop_requested = threading.Event()
        self._stop_requested = stop_requested
        source_error: list[BaseException] = []

        def consumer_target() -> None:
            self._consume_frames_thread(
                loop=self._require_loop(),
                frame_queue=frame_queue,
                producer_done=producer_done,
                source_error=source_error,
                stop_requested=stop_requested,
            )

        if self._consumer_runner is None:
            consumer_future = self._start_thread(
                name=f"asr-lane-{self.stream_id}-zipformer-consumer",
                target=consumer_target,
            )
        else:
            consumer_future = self._consumer_runner(consumer_target)
        producer_future = self._start_thread(
            name=f"asr-lane-{self.stream_id}-audio-producer",
            target=lambda: asyncio.run(
                self._produce_frames_thread(
                    source=source,
                    frame_queue=frame_queue,
                    producer_done=producer_done,
                    source_error=source_error,
                    stop_requested=stop_requested,
                    on_source_ready=on_source_ready,
                )
            ),
        )
        worker_futures = (producer_future, consumer_future)
        worker_wrappers = tuple(asyncio.wrap_future(future) for future in worker_futures)
        primary_error: BaseException | None = None
        primary_traceback = None
        source_stop_error: BaseException | None = None
        try:
            done, _ = await asyncio.wait(
                worker_wrappers,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            primary_error = next(
                (
                    future.exception()
                    for future in done
                    if not future.cancelled() and future.exception() is not None
                ),
                None,
            )
            if primary_error is not None:
                primary_traceback = primary_error.__traceback__
        finally:
            stop_requested.set()
            frame_queue.wake_all()
            stop_task = asyncio.create_task(source.stop())
            stop_result, stop_cancellation = await self._await_task_quiescent(stop_task)
            if primary_error is None and stop_cancellation is not None:
                primary_error = stop_cancellation
                primary_traceback = stop_cancellation.__traceback__
            if isinstance(stop_result, BaseException):
                source_stop_error = stop_result

            worker_join = asyncio.ensure_future(
                asyncio.gather(
                    *(asyncio.wrap_future(future) for future in worker_futures),
                    return_exceptions=True,
                )
            )
            worker_results, join_cancellation = await self._await_task_quiescent(worker_join)
            if primary_error is None and join_cancellation is not None:
                primary_error = join_cancellation
                primary_traceback = join_cancellation.__traceback__
            if primary_error is None and isinstance(worker_results, list):
                primary_error = next(
                    (
                        result
                        for result in worker_results
                        if isinstance(result, BaseException)
                    ),
                    None,
                )
                if primary_error is not None:
                    primary_traceback = primary_error.__traceback__

            for wrapper in worker_wrappers:
                if wrapper.done() and not wrapper.cancelled():
                    wrapper.exception()
                elif not wrapper.done():
                    wrapper.cancel()
            self._loop = None
            self._event_loop_thread_id = None
            self._stop_requested = None
            self._frame_queue = None

        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)
        if source_stop_error is not None:
            raise source_stop_error

    def request_stop(self) -> None:
        if self._stop_requested is not None:
            self._stop_requested.set()
        if self._frame_queue is not None:
            self._frame_queue.wake_all()

    def cancel_offline_tasks(self) -> None:
        self._offline_scheduling_closed = True
        for task in tuple(self._offline_tasks):
            if not task.done():
                task.cancel()

    async def wait_for_offline_tail(self, *, timeout_seconds: float) -> None:
        tasks = tuple(self._offline_tasks)
        if not tasks:
            if self._offline_error is not None:
                raise self._offline_error
            return

        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
        except asyncio.CancelledError:
            await self._cancel_and_wait_offline_tasks(tasks)
            raise
        if pending:
            cleanup_cancellation = await self._cancel_and_wait_offline_tasks(tasks)
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
            raise TimeoutError("offline recognition did not stop before the timeout")

        self._offline_tasks.difference_update(done)
        if self._offline_tail in done:
            self._offline_tail = None
        if self._offline_error is not None:
            raise self._offline_error

    async def _cancel_and_wait_offline_tasks(
        self,
        tasks: tuple[asyncio.Task[None], ...],
    ) -> asyncio.CancelledError | None:
        for task in tasks:
            if not task.done():
                task.cancel()
        cleanup = asyncio.ensure_future(asyncio.gather(*tasks, return_exceptions=True))
        _, cancellation = await self._await_task_quiescent(cleanup)
        self._offline_tasks.difference_update(tasks)
        if self._offline_tail in tasks:
            self._offline_tail = None
        return cancellation

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

    async def _produce_frames_thread(
        self,
        *,
        source: AudioSource,
        frame_queue: OnlineFrameQueue,
        producer_done: threading.Event,
        source_error: list[BaseException],
        stop_requested: threading.Event,
        on_source_ready: Callable[[], None] | None,
    ) -> None:
        observed_drops = 0
        observed_backlog = 0
        try:
            await source.prepare()
            if on_source_ready is not None:
                on_source_ready()
            if stop_requested.is_set():
                return
            async for frame in source.frames():
                if stop_requested.is_set():
                    break
                frame_queue.put(frame)
                new_drops = frame_queue.dropped_frames - observed_drops
                observed_drops = frame_queue.dropped_frames
                if new_drops:
                    self._increment_metrics(dropped_frames=new_drops)
                new_backlog = frame_queue.backlogged_frames - observed_backlog
                observed_backlog = frame_queue.backlogged_frames
                if new_backlog:
                    self._increment_metrics(backlogged_frames=new_backlog)
                if frame.is_finish:
                    break
                time.sleep(0)
        except BaseException as exc:
            source_error.append(exc)
            raise
        finally:
            producer_done.set()
            frame_queue.wake_all()

    def _consume_frames_thread(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        frame_queue: OnlineFrameQueue,
        producer_done: threading.Event,
        source_error: list[BaseException],
        stop_requested: threading.Event,
    ) -> None:
        start_run = getattr(self.online_recognizer, "start_run", None)
        if start_run is not None:
            self._run_async_in_thread(start_run())

        while not stop_requested.is_set():
            frame = frame_queue.wait_get(timeout=0.1)
            if frame is None:
                if producer_done.is_set():
                    if source_error:
                        raise source_error[0]
                    break
                continue
            if stop_requested.is_set():
                break

            if frame.is_finish:
                if self._endpoint_mode == "vad":
                    try:
                        if self._pv_in_speech:
                            self._close_paraformer_sentence(frame.timestamp_ms)
                    except Exception:
                        import traceback
                        _logger.error("finish 收尾异常: %s", traceback.format_exc())
                self._run_async_in_thread(self.online_recognizer.reset())
                asyncio.run_coroutine_threadsafe(
                    self._finish_after_offline(timestamp_ms=frame.timestamp_ms),
                    loop,
                ).result()
                break

            self.segment_cache.put(frame)
            self._increment_metrics(processed_frames=1)
            self._run_kws(frame.pcm)
            if self._endpoint_mode == "vad":
                # paraformer（vad_driven）：demo 式一体处理——silero 逐帧
                # 判定，语音段内的帧才喂识别器，句尾冲刷收尾。
                self._paraformer_frame(frame)
                continue

            started_at = self._monotonic()
            recognition = self._run_async_in_thread(self.online_recognizer.accept(frame))
            latency_ms = (self._monotonic() - started_at) * 1000.0
            self.latency_tracker.add(latency_ms)
            if self.frame_latency_writer is not None:
                self.frame_latency_writer.record({"latency_ms": latency_ms})
            if recognition is not None:
                self._handle_online_result(recognition)

    def _run_kws(self, pcm: np.ndarray) -> None:
        if self.kws_session is None:
            return
        try:
            hits = self.kws_session.detect(pcm)
        except Exception:
            # KWS must not break the ASR lane.
            return
        if not hits or self.on_kws is None:
            return
        for hit in hits:
            try:
                self.on_kws(hit)
            except Exception:
                pass

    @staticmethod
    def _run_async_in_thread(awaitable):
        return asyncio.run(awaitable)

    @staticmethod
    def _start_thread(*, name: str, target: Callable[[], None]) -> concurrent.futures.Future[None]:
        future: concurrent.futures.Future[None] = concurrent.futures.Future()

        def run_target() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                target()
            except BaseException as exc:
                if not future.cancelled():
                    future.set_exception(exc)
            else:
                if not future.cancelled():
                    future.set_result(None)

        thread = threading.Thread(target=run_target, name=name)
        thread.start()
        return future

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("LanePipeline event loop is not available")
        return self._loop

    def _handle_online_result(self, recognition: OnlineRecognition) -> None:
        text = recognition.text
        if text != self._current_online_text:
            self._current_online_text = text
            self._revision += 1
            self._publish_event(
                sentence_id=self._sentence_id,
                revision=self._revision,
                status=EventStatus.PARTIAL,
                text=text,
                timestamp_ms=recognition.timestamp_ms,
            )

        # VAD 主导断句：断句与收尾都在 _paraformer_frame 里完成，
        # 这里只负责刷新 PARTIAL，不看识别器的 is_endpoint
        #（非 transducer 模型的 blank 端点不可靠，如 paraformer）
        if self._endpoint_mode == "vad":
            return

        if not recognition.is_endpoint:
            return

        endpoint_text = self._current_online_text
        if endpoint_text:
            pending_revision = self._revision + 1
            self._revision = pending_revision
            pcm = self._take_endpoint_audio(recognition)
            self._publish_event(
                sentence_id=self._sentence_id,
                revision=pending_revision,
                status=EventStatus.PENDING,
                text=endpoint_text,
                timestamp_ms=recognition.timestamp_ms,
            )
            self._record_model_output(
                model="zipformer",
                sentence_id=self._sentence_id,
                status="endpoint",
                text=endpoint_text,
            )
            if pcm.size:
                self._schedule_offline_threadsafe(
                    sentence_id=self._sentence_id,
                    revision=pending_revision + 1,
                    prompt=endpoint_text,
                    pcm=pcm,
                    timestamp_ms=recognition.timestamp_ms,
                )
        else:
            self.segment_cache.cut(
                self.segment_cache.end_time_s
                - self.endpoint.rule2_min_trailing_silence
            )

        self._advance_sentence()

    def _take_endpoint_audio(self, recognition: OnlineRecognition) -> np.ndarray:
        full = self.segment_cache.audio()
        token_time = recognition.last_token_stream_s
        if token_time is None or not math.isfinite(token_time):
            self.segment_cache.reset()
            return full

        cut_at = token_time + self.endpoint.cut_margin_seconds
        if not math.isfinite(cut_at) or cut_at <= self.segment_cache.start_time_s:
            self.segment_cache.reset()
            return full

        pcm = self.segment_cache.cut(cut_at)
        if pcm.size == 0 and full.size != 0:
            self.segment_cache.reset()
            return full
        return pcm

    # ---------- VAD 主导断句（funasr paraformer 专用路径） ----------
    #
    # demo 式一体门控（对齐 streaming_demo.py）：silero 逐帧判定同时管
    # "断句"和"喂模型"——非语音帧不进识别器（噪声不会变成"嗯"之类的字，
    # 在线缓存也不会被静音喂到漂移）；句尾用 is_final=True 冲刷掉不足
    # 一个 stride 的尾巴再整句 reset，每句都是全新状态。
    # 可调环境变量：VAD_THRESHOLD / VAD_LOUD_TRIGGER / VAD_MIN_SILENCE /
    # VAD_MIN_SPEECH / VAD_PRE_BUFFER / VAD_MAX_UTT / VAD_MODEL_DIR。

    _SILERO_WINDOW = 512     # silero 输入窗口（16kHz 下 32ms）

    def _silero_conf(self, window: np.ndarray) -> float:
        """raw silero（onnxruntime）逐窗口语音置信度，内部状态跨调用保持。"""
        if self._pv_silero is None:
            import onnxruntime as ort

            model_dir = Path(os.getenv("VAD_MODEL_DIR", str(DEFAULT_VAD_MODEL_DIR)))
            model_path = None
            for name in ("silero_vad.onnx", "model.onnx", "vad.onnx"):
                p = model_dir / name
                if p.is_file():
                    model_path = p
                    break
            if model_path is None:
                files = list(model_dir.glob("*.onnx"))
                if len(files) == 1:
                    model_path = files[0]
            if model_path is None:
                raise FileNotFoundError(
                    f"no usable VAD model found in {model_dir}")
            sess = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
            state_input = next(i for i in sess.get_inputs()
                               if i.name == "state")
            shape = [d if isinstance(d, int) else 1
                     for d in state_input.shape]
            self._pv_silero = (
                sess,
                np.zeros(shape, dtype=np.float32),
                np.array(16000, dtype=np.int64),
            )
            _logger.info("silero VAD 加载完成: %s", model_path)
        sess, state, sr = self._pv_silero
        out, new_state = sess.run(None, {
            "input": window.reshape(1, -1).astype(np.float32),
            "state": state,
            "sr": sr,
        })
        self._pv_silero = (sess, new_state, sr)
        return float(out.reshape(-1)[0])

    def _paraformer_frame(self, frame: AudioFrame) -> None:
        """vad_driven 识别器（funasr paraformer）的逐帧入口。"""
        try:
            samples = frame.pcm if frame.pcm.dtype == np.float32 else \
                frame.pcm.astype(np.float32) / 32768.0

            # 1. 逐 silero 窗口更新语音状态（窗口 512，帧长不固定，
            #    残余采样留给下一帧拼）
            self._pv_buf = np.concatenate((self._pv_buf, samples))
            just_opened = False
            while self._pv_buf.size >= self._SILERO_WINDOW:
                window = self._pv_buf[: self._SILERO_WINDOW]
                self._pv_buf = self._pv_buf[self._SILERO_WINDOW:]
                conf = self._silero_conf(window)
                if not self._pv_in_speech:
                    if conf >= self._pv_threshold:
                        self._pv_loud += 1
                        if self._pv_loud > self._pv_loud_trigger:
                            self._pv_in_speech = True
                            self._pv_loud = 0
                            self._pv_quiet = 0
                            just_opened = True
                    else:
                        self._pv_loud = 0
                else:
                    if conf < self._pv_threshold:
                        self._pv_quiet += self._SILERO_WINDOW
                    else:
                        self._pv_quiet = 0

            if not self._pv_in_speech:
                # 非语音：只进预缓冲，不喂识别器
                self._pv_pre.append(samples)
                total = sum(x.size for x in self._pv_pre)
                while total > self._pv_pre_samples and self._pv_pre:
                    total -= self._pv_pre.pop(0).size
                return

            # 2. 语音段内：开口时先把预缓冲补喂（句头不丢字），再喂当前帧
            if just_opened and self._pv_pre:
                pre = np.concatenate(self._pv_pre)
                self._pv_pre.clear()
                self._pv_seg.append(pre)
                self._feed_paraformer(frame, pre)
            self._pv_seg.append(samples)
            self._feed_paraformer(frame, samples)
            self._vad_since_cut += samples.size

            # 3. 句尾判定：持续静音 / 超长强制切断（防无限积累）
            if self._pv_quiet >= self._pv_min_silence_samples or \
                    self._vad_since_cut >= self._vad_max_utt_samples:
                self._close_paraformer_sentence(frame.timestamp_ms)
        except Exception:
            import traceback
            _logger.error("_paraformer_frame 异常: %s", traceback.format_exc())

    def _feed_paraformer(self, frame: AudioFrame, samples: np.ndarray) -> None:
        """喂一段音频给 paraformer 并发布 PARTIAL。"""
        sub = replace(frame, pcm=samples)
        started_at = self._monotonic()
        recognition = self._run_async_in_thread(
            self.online_recognizer.accept(sub))
        latency_ms = (self._monotonic() - started_at) * 1000.0
        self.latency_tracker.add(latency_ms)
        if self.frame_latency_writer is not None:
            self.frame_latency_writer.record({"latency_ms": latency_ms})
        if recognition is not None:
            self._handle_online_result(recognition)

    def _close_paraformer_sentence(self, timestamp_ms: int) -> None:
        """VAD 判完一句：is_final 冲刷 → 收尾（PENDING + 送校正）→ 重置。"""
        seg = np.concatenate(self._pv_seg) if self._pv_seg else \
            np.zeros(0, dtype=np.float32)
        # 去掉句尾用于判静的那段静音（不进校正音频）
        keep = seg.size - self._pv_quiet
        seg = seg[:keep] if keep > 0 else np.zeros(0, dtype=np.float32)
        self._pv_seg.clear()
        self._pv_pre.clear()
        self._pv_in_speech = False
        self._pv_loud = 0
        self._pv_quiet = 0
        self._vad_since_cut = 0
        self._pv_buf = np.zeros(0, dtype=np.float32)

        if seg.size < self._pv_min_speech_samples:
            # 太短（咳嗽/噪声抖动）：整句丢弃，只重置识别器
            self._run_async_in_thread(self.online_recognizer.reset())
            self._advance_sentence()
            return

        # 句尾冲刷：不足一个 stride 的尾巴用 is_final=True 解完，
        # 文本补全后再收尾（PENDING 和校正 prompt 都是完整句）
        flush = getattr(self.online_recognizer, "flush", None)
        if callable(flush):
            flushed = self._run_async_in_thread(flush())
            if flushed:
                self._current_online_text = flushed
        self._finalize_vad_sentence(seg, timestamp_ms)

    def _finalize_vad_sentence(self, pcm: np.ndarray, timestamp_ms: int) -> None:
        """VAD 判完一句：出 PENDING → 送离线校正 → 重置识别器 → 推进句子号。

        噪声过滤：识别结果为空、过短或只有语气词时整句丢弃（不发 PENDING、
        不送校正）——否则校正的生成式模型会对着噪声幻觉出热词。"""
        text = self._current_online_text
        _logger.debug("finalize: text_len=%d pcm=%.1fs",
                      len(text), pcm.size / 16000)
        self.segment_cache.reset()      # VAD 段音频已在 pcm 里，缓存直接清
        if text and not _is_noise_text(text):
            pending_revision = self._revision + 1
            self._revision = pending_revision
            self._publish_event(
                sentence_id=self._sentence_id,
                revision=pending_revision,
                status=EventStatus.PENDING,
                text=text,
                timestamp_ms=timestamp_ms,
            )
            self._record_model_output(
                model="zipformer",
                sentence_id=self._sentence_id,
                status="vad_endpoint",
                text=text,
            )
            if pcm.size:
                _logger.debug("送离线校正: sentence=%d", self._sentence_id)
                self._schedule_offline_threadsafe(
                    sentence_id=self._sentence_id,
                    revision=pending_revision + 1,
                    prompt=text,
                    pcm=pcm,
                    timestamp_ms=timestamp_ms,
                )
        self._run_async_in_thread(self.online_recognizer.reset())
        self._advance_sentence()

    def _advance_sentence(self) -> None:
        self._sentence_id += 1
        self._revision = 0
        self._current_online_text = ""

    def _schedule_offline(
        self,
        *,
        sentence_id: int,
        revision: int,
        prompt: str,
        pcm: np.ndarray,
        timestamp_ms: int,
    ) -> None:
        if self._offline_scheduling_closed:
            return
        previous_task = self._offline_tail

        async def run_offline() -> None:
            if previous_task is not None:
                await previous_task

            # 声纹与离线 ASR 共用同一段 pcm，并行跑：
            # ASR 1.5~5s、声纹 0.1~0.3s，声纹开销被完全掩盖，FINAL 时刻不变。
            speaker_task = self._start_speaker_task(pcm)
            result = None
            error: QwenRequestError | None = None
            try:
                result = await self.offline_recognizer.recognize(pcm=pcm, prompt=prompt)
            except QwenRequestError as exc:
                error = exc

            # 无论 ASR 成功与否都要把声纹任务收干净，避免悬空 task
            speaker = await self._collect_speaker(speaker_task, sentence_id)

            if error is not None:
                if self.on_qwen_error is not None:
                    self.on_qwen_error(
                        QwenSentenceFailure(
                            stream_id=self.stream_id,
                            sentence_id=sentence_id,
                            revision=revision,
                            pending_text=prompt,
                            timestamp_ms=timestamp_ms,
                            request=error.failure.to_payload(),
                        )
                    )
                return
            if result is None:
                return

            if self.on_qwen_success is not None:
                self.on_qwen_success(sentence_id)
            self.rtf_tracker.add(result.audio_seconds, result.elapsed_ms)
            if self.rtf_writer is not None:
                self.rtf_writer.record(
                    {
                        "sentence_id": sentence_id,
                        "audio_seconds": result.audio_seconds,
                        "elapsed_ms": result.elapsed_ms,
                        "rtf": result.rtf,
                    }
                )
            if result.timings is not None:
                self.shared_timings_tracker.add(result.timings)
                if self.shared_timing_writer is not None:
                    filtered = _filter_shared_timings(result.timings)
                    if filtered:
                        self.shared_timing_writer.record(
                            {"sentence_id": sentence_id, **filtered}
                        )
            self._record_model_output(
                model="qwen3-asr",
                sentence_id=sentence_id,
                status="final",
                text=result.text,
                audio_seconds=result.audio_seconds,
                elapsed_ms=result.elapsed_ms,
                rtf=result.rtf,
                raw_response=result.raw_response,
            )
            self._publish_event(
                sentence_id=sentence_id,
                revision=revision,
                status=EventStatus.FINAL,
                text=result.text,
                timestamp_ms=timestamp_ms,
                speaker=speaker,
            )

        task = asyncio.create_task(run_offline())
        self._offline_tasks.add(task)
        task.add_done_callback(self._record_offline_task_done)
        self._offline_tail = task
        self._increment_metrics(offline_requests=1)

    def _record_offline_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            self._offline_tasks.discard(task)
            return
        exc = task.exception()
        if exc is not None:
            self._offline_error = exc
            return
        self._offline_tasks.discard(task)

    def _schedule_offline_threadsafe(
        self,
        *,
        sentence_id: int,
        revision: int,
        prompt: str,
        pcm: np.ndarray,
        timestamp_ms: int,
    ) -> None:
        if threading.get_ident() == self._event_loop_thread_id:
            self._schedule_offline(
                sentence_id=sentence_id,
                revision=revision,
                prompt=prompt,
                pcm=pcm,
                timestamp_ms=timestamp_ms,
            )
            return
        self._require_loop().call_soon_threadsafe(
            partial(
                self._schedule_offline,
                sentence_id=sentence_id,
                revision=revision,
                prompt=prompt,
                pcm=pcm,
                timestamp_ms=timestamp_ms,
            )
        )

    # ---------- 说话人分离 / 声纹 ----------

    def _start_speaker_task(self, pcm: np.ndarray) -> asyncio.Task | None:
        """把声纹分配丢进线程池并立刻返回 task。

        声纹是纯 CPU 的 ONNX 推理，放线程池才不会卡住事件循环；
        与离线 ASR 的 await 并行，所以不增加 FINAL 延迟。
        """
        tracker = self.speaker_tracker
        if tracker is None or not getattr(tracker, "enabled", False):
            return None
        if getattr(pcm, "size", 0) == 0:
            return None
        try:
            return asyncio.create_task(asyncio.to_thread(tracker.assign, pcm))
        except Exception:
            _logger.exception("创建声纹任务失败 lane=%d", self.stream_id)
            return None

    async def _collect_speaker(
        self, task: asyncio.Task | None, sentence_id: int
    ) -> SpeakerInfo | None:
        """等声纹任务结束。任何异常都降级成"无说话人"。"""
        if task is None:
            return None
        try:
            info = await task
        except Exception:
            _logger.exception(
                "声纹分配失败 lane=%d sentence=%d", self.stream_id, sentence_id
            )
            return None
        if info is not None:
            self._last_speaker = info
        return info

    async def _emit_speaker_remap(self) -> None:
        """会话末二遍重聚类，把开头几句的错标一次性纠正。

        通过 ``on_speaker`` 回调把 ``旧标签 -> 新标签`` 映射发给前端/落盘层；
        消费方必须**原子地**套用整张表（不能链式逐条替换）。
        """
        tracker = self.speaker_tracker
        if tracker is None or self.on_speaker is None:
            return
        try:
            mapping = await asyncio.to_thread(tracker.resegment)
        except Exception:
            _logger.exception("二遍重聚类失败 lane=%d", self.stream_id)
            return
        if not mapping:
            return
        try:
            self.on_speaker(
                {
                    "type": "speaker_remap",
                    "lane_id": self.stream_id,
                    "mapping": dict(mapping),
                }
            )
        except Exception:
            _logger.exception("speaker_remap 回调失败 lane=%d", self.stream_id)

    def _record_model_output(
        self,
        *,
        model: str,
        sentence_id: int,
        status: str,
        text: str,
        audio_seconds: float | None = None,
        elapsed_ms: float | None = None,
        rtf: float | None = None,
        raw_response: dict[str, object] | None = None,
    ) -> None:
        if self.model_output_recorder is None:
            return
        self.model_output_recorder.record(
            ModelOutputRecord(
                model=model,
                stream_id=self.stream_id,
                sentence_id=sentence_id,
                status=status,
                text=text,
                audio_seconds=audio_seconds,
                elapsed_ms=elapsed_ms,
                rtf=rtf,
                raw_response=raw_response,
            )
        )

    async def _finish_after_offline(self, *, timestamp_ms: int) -> None:
        endpoint_text = self._current_online_text
        if endpoint_text:
            pcm = self.segment_cache.audio()
            self.segment_cache.reset()
            pending_revision = self._revision + 1
            self._revision = pending_revision
            self._publish_event(
                sentence_id=self._sentence_id,
                revision=pending_revision,
                status=EventStatus.PENDING,
                text=endpoint_text,
                timestamp_ms=timestamp_ms,
            )
            self._record_model_output(
                model="zipformer",
                sentence_id=self._sentence_id,
                status="endpoint",
                text=endpoint_text,
            )
            if pcm.size:
                self._schedule_offline(
                    sentence_id=self._sentence_id,
                    revision=pending_revision + 1,
                    prompt=endpoint_text,
                    pcm=pcm,
                    timestamp_ms=timestamp_ms,
                )
            self._advance_sentence()
        else:
            self.segment_cache.reset()

        if self._offline_tasks:
            await asyncio.gather(*tuple(self._offline_tasks), return_exceptions=True)
            self._offline_tasks.clear()
        # 二遍重聚类必须放在所有离线任务之后、LANE_FINISHED 之前，
        # 保证修正映射在会话结束前送到前端/落盘层。
        await self._emit_speaker_remap()
        if self._offline_error is not None:
            raise self._offline_error
        self._set_final_state(LaneFinalState.FINISHED)
        self._publish_event(
            sentence_id=self._sentence_id,
            revision=0,
            status=EventStatus.LANE_FINISHED,
            text="",
            timestamp_ms=timestamp_ms,
        )

    def mark_fatal(self, message: str) -> None:
        self._set_final_state(LaneFinalState.FATAL)
        self._publish_event(
            sentence_id=self._last_event_sentence_id,
            revision=self._last_event_revision,
            status=EventStatus.FATAL_ERROR,
            text=message,
            timestamp_ms=self._last_event_timestamp_ms,
        )

    def _publish_event(
        self,
        *,
        sentence_id: int,
        revision: int,
        status: EventStatus,
        text: str,
        timestamp_ms: int,
        speaker: SpeakerInfo | None = None,
    ) -> None:
        self._last_event_sentence_id = sentence_id
        self._last_event_revision = revision
        self._last_event_timestamp_ms = timestamp_ms
        event = ASREvent.asr_result(
            stream_id=self.stream_id,
            sentence_id=sentence_id,
            revision=revision,
            status=status,
            text=text,
            timestamp_ms=timestamp_ms,
            speaker=speaker,
        )
        if self._loop is None:
            self.event_bus.publish(event)
            return
        if threading.get_ident() == self._event_loop_thread_id:
            self.event_bus.publish(event)
            return
        self._require_loop().call_soon_threadsafe(self.event_bus.publish, event)


def _filter_shared_timings(timings: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in timings.items()
        if key in SHARED_TIMING_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
