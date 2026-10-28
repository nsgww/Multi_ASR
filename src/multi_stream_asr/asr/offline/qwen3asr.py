from __future__ import annotations

import base64
import io
import re
import time
import wave
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import numpy as np

from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
from multi_stream_asr.config import SharedQwenConfig


QWEN_USER_PROMPT = "language Chinese<asr_text>"
QWEN_MAX_TOKENS = 256
MAX_ERROR_RESPONSE_BYTES = 64 * 1024

HOTWORDS_PROMPT_PREFIX = "热词："
# 模型偶尔会把热词提示语整句"回声"进转写结果，需要剥掉
_HOTWORDS_ECHO_RE = re.compile(r"热词：[^。\n]*。?")


def build_hotwords_user_prompt(hotwords: tuple[str, ...]) -> str:
    """构造带热词引导的 user 文本。

    注意：SMT ASR 服务（llama-server --smt-multi-asr）会丢弃 system 消息
    （2026-08-04 实测），只能走 user 文本前缀；且 1.7B 对"可能出现的专有名词："长格式会返回空/幻觉，
    必须用"热词："简式（2026-08-04 提示词矩阵实验结论）。
    """
    return f"{HOTWORDS_PROMPT_PREFIX}{'、'.join(hotwords)}。" + QWEN_USER_PROMPT


def strip_hotwords_echo(text: str, fallback: str = "") -> str:
    """剥掉转写结果中回声的热词提示语；剥完为空时回退到 fallback。"""
    if not text or HOTWORDS_PROMPT_PREFIX not in text:
        return text
    stripped = _HOTWORDS_ECHO_RE.sub("", text).strip()
    return stripped if stripped else fallback


SHARED_TIMING_FIELDS = {
    "queue_ms",
    "encode_ms",
    "prefill_ms",
    "decode_ms",
    "total_ms",
}


@dataclass(frozen=True)
class QwenRequestFailure:
    category: str
    error_type: str
    message: str
    method: str
    url: str
    timeout_ms: int
    audio_seconds: float
    sample_count: int
    status_code: int | None = None
    response_body: str | None = None
    response_body_truncated: bool = False

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


class QwenRequestError(RuntimeError):
    def __init__(self, failure: QwenRequestFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _bounded_response_body(response: httpx.Response) -> tuple[str, bool]:
    content = response.content
    truncated = len(content) > MAX_ERROR_RESPONSE_BYTES
    decoded = content[:MAX_ERROR_RESPONSE_BYTES].decode("utf-8", errors="replace")
    encoded = decoded.encode("utf-8")
    if len(encoded) > MAX_ERROR_RESPONSE_BYTES:
        truncated = True
        decoded = encoded[:MAX_ERROR_RESPONSE_BYTES].decode("utf-8", errors="ignore")
    return decoded, truncated


def encode_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    array = np.asarray(pcm, dtype=np.float32)
    clipped = np.clip(array, -1.0, 0.999969)
    pcm16 = (clipped * 32768.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def encode_wav_base64(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    return base64.b64encode(encode_wav_bytes(pcm, sample_rate=sample_rate)).decode("ascii")


def build_chat_request(
    pcm: np.ndarray,
    max_tokens: int = QWEN_MAX_TOKENS,
    hotwords: tuple[str, ...] = (),
) -> dict[str, Any]:
    user_text = (
        build_hotwords_user_prompt(hotwords) if hotwords else QWEN_USER_PROMPT
    )
    return {
        "model": "qwen3-asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encode_wav_base64(pcm),
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def parse_timings(payload: dict[str, Any]) -> dict[str, float] | None:
    raw = payload.get("multi_asr_timings")
    if not isinstance(raw, dict):
        return None
    timings = {
        key: float(value)
        for key, value in raw.items()
        if key in SHARED_TIMING_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    return timings or None


def clean_qwen_text(payload: dict) -> str:
    text: object | None = None

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = message.get("content")

    if text is None:
        text = payload.get("content")
    if text is None:
        text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError("missing recognized text in qwen3-asr response")

    cleaned = text
    start_tag = "<asr_text>"
    end_tag = "</asr_text>"

    start_index = cleaned.find(start_tag)
    if start_index != -1:
        cleaned = cleaned[start_index + len(start_tag) :]

    end_index = cleaned.find(end_tag)
    if end_index != -1:
        cleaned = cleaned[:end_index]

    cleaned = cleaned.strip()
    return cleaned


class Qwen3AsrClient:
    def __init__(
        self,
        *,
        shared: SharedQwenConfig,
        hotwords: tuple[str, ...] = (),
    ) -> None:
        self._base_url = f"http://{shared.host}:{shared.port}"
        self._request_timeout_ms = shared.request_timeout_ms
        self._hotwords = hotwords

    async def recognize(self, pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult:
        result = await self._recognize_once(pcm, prompt, self._hotwords)
        if not result.text and self._hotwords:
            # 1.7B 对"长热词表+短音频"会返回空（2026-08-04 实测），无热词重试一次
            result = await self._recognize_once(pcm, prompt, ())
        return result

    async def _recognize_once(
        self, pcm: np.ndarray, prompt: str, hotwords: tuple[str, ...]
    ) -> OfflineRecognitionResult:
        started = time.monotonic()
        sample_count = int(np.asarray(pcm).size)
        audio_seconds = float(sample_count) / 16000.0
        request_timeout_seconds = self._request_timeout_ms / 1000.0
        url = f"{self._base_url}/v1/chat/completions"
        request_metadata = {
            "method": "POST",
            "url": url,
            "timeout_ms": self._request_timeout_ms,
            "audio_seconds": audio_seconds,
            "sample_count": sample_count,
        }
        payload = build_chat_request(pcm, hotwords=hotwords)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=request_timeout_seconds,
                )
            except httpx.HTTPError as exc:
                raise QwenRequestError(
                    QwenRequestFailure(
                        category="transport",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        **request_metadata,
                    )
                ) from exc

            if not response.is_success:
                body, truncated = _bounded_response_body(response)
                raise QwenRequestError(
                    QwenRequestFailure(
                        category="http_status",
                        error_type="HTTPStatusError",
                        message=f"Qwen request returned HTTP {response.status_code}",
                        status_code=response.status_code,
                        response_body=body,
                        response_body_truncated=truncated,
                        **request_metadata,
                    )
                )

            try:
                raw_response = response.json()
            except ValueError as exc:
                body, truncated = _bounded_response_body(response)
                raise QwenRequestError(
                    QwenRequestFailure(
                        category="invalid_json",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        status_code=response.status_code,
                        response_body=body,
                        response_body_truncated=truncated,
                        **request_metadata,
                    )
                ) from exc

            try:
                text = clean_qwen_text(raw_response)
            except Exception as exc:
                body, truncated = _bounded_response_body(response)
                raise QwenRequestError(
                    QwenRequestFailure(
                        category="invalid_response",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        status_code=response.status_code,
                        response_body=body,
                        response_body_truncated=truncated,
                        **request_metadata,
                    )
                ) from exc

            if hotwords:
                # 剥掉热词提示语回声；整段都是回声时回退到在线识别文本
                text = strip_hotwords_echo(text, fallback=prompt)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        rtf = elapsed_ms / (audio_seconds * 1000.0) if audio_seconds > 0 else 0.0
        return OfflineRecognitionResult(
            text=text,
            elapsed_ms=elapsed_ms,
            audio_seconds=audio_seconds,
            rtf=rtf,
            raw_response=raw_response,
            timings=parse_timings(raw_response),
        )

    async def close(self) -> None:
        return None

    def set_hotwords(self, hotwords: tuple[str, ...]) -> None:
        """运行时热更新热词表，后续精修请求立即生效。"""
        self._hotwords = tuple(hotwords)
