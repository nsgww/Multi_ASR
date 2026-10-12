from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import dataclass

import httpx
import numpy as np
import pytest

from multi_stream_asr.asr.offline.qwen3asr import (
    MAX_ERROR_RESPONSE_BYTES,
    QWEN_USER_PROMPT,
    Qwen3AsrClient,
    QwenRequestError,
    build_chat_request,
    clean_qwen_text,
    encode_wav_base64,
    parse_timings,
)
from multi_stream_asr.config import SharedQwenConfig


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, object]

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict[str, object]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeAsyncClient:
    def __init__(self, *, post_responses: list[FakeResponse]):
        self.post_responses = post_responses
        self.calls: list[tuple[str, str, dict[str, object], float]] = []

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(("POST", url, json or {}, timeout))
        if not self.post_responses:
            raise AssertionError("unexpected post")
        return self.post_responses.pop(0)


def test_encode_wav_base64_writes_16k_mono_pcm16():
    pcm = np.asarray([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)

    encoded = encode_wav_base64(pcm)

    raw = base64.b64decode(encoded)
    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        frames = wav_file.readframes(wav_file.getnframes())
    decoded = np.frombuffer(frames, dtype=np.int16)
    assert decoded.tolist() == [-32768, -32768, -16384, 0, 16384, 32766, 32766]


def test_build_chat_request_matches_shared_service_format():
    pcm = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

    payload = build_chat_request(pcm)

    assert payload["model"] == "qwen3-asr"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 256
    assert "id_slot" not in payload
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0] == {
        "type": "input_audio",
        "input_audio": {"data": encode_wav_base64(pcm), "format": "wav"},
    }
    assert message["content"][1] == {
        "type": "text",
        "text": "language Chinese<asr_text>",
    }
    assert QWEN_USER_PROMPT == "language Chinese<asr_text>"


def test_parse_timings_accepts_only_known_numeric_fields():
    assert parse_timings(
        {
            "multi_asr_timings": {
                "queue_ms": 12,
                "encode_ms": 900.5,
                "unknown_ms": 1,
                "decode_ms": True,
            }
        }
    ) == {"queue_ms": 12.0, "encode_ms": 900.5}
    assert parse_timings({}) is None


@pytest.mark.asyncio
async def test_recognize_posts_once_and_parses_shared_timings(monkeypatch):
    client = FakeAsyncClient(
        post_responses=[
            FakeResponse(
                200,
                {
                    "choices": [
                        {"message": {"content": "<asr_text> hello world </asr_text>"}}
                    ],
                    "multi_asr_timings": {
                        "queue_ms": 12.5,
                        "encode_ms": 900.0,
                        "prefill_ms": 300.0,
                        "decode_ms": 800.0,
                        "total_ms": 2012.5,
                    },
                },
            )
        ]
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", lambda: client
    )
    recognizer = Qwen3AsrClient(
        shared=SharedQwenConfig(port=9090, request_timeout_ms=2000)
    )
    pcm = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

    result = await recognizer.recognize(pcm, prompt="ignored")

    assert result.text == "hello world"
    assert result.timings == {
        "queue_ms": 12.5,
        "encode_ms": 900.0,
        "prefill_ms": 300.0,
        "decode_ms": 800.0,
        "total_ms": 2012.5,
    }
    assert result.audio_seconds == pytest.approx(3 / 16000)
    assert len(client.calls) == 1
    assert client.calls[0][1] == "http://127.0.0.1:9090/v1/chat/completions"
    assert client.calls[0][2] == build_chat_request(pcm)
    assert client.calls[0][3] == 2.0


@pytest.mark.asyncio
async def test_recognize_tolerates_missing_timings(monkeypatch):
    client = FakeAsyncClient(
        post_responses=[FakeResponse(200, {"text": "<asr_text>ok</asr_text>"})]
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", lambda: client
    )
    recognizer = Qwen3AsrClient(shared=SharedQwenConfig())

    result = await recognizer.recognize(np.asarray([0.25], dtype=np.float32), prompt="")

    assert result.text == "ok"
    assert result.timings is None
    assert client.calls[0][3] == 120.0


@pytest.mark.asyncio
async def test_recognize_raises_on_server_busy_without_followup(monkeypatch):
    client = FakeAsyncClient(
        post_responses=[FakeResponse(503, {"error": "server busy: queue full"})]
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", lambda: client
    )
    recognizer = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError, match="Qwen request returned HTTP 503"):
        await recognizer.recognize(np.asarray([0.25], dtype=np.float32), prompt="")

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_recognize_http_500_preserves_bounded_response_body(monkeypatch):
    body = b"server detail:" + b"x" * (MAX_ERROR_RESPONSE_BYTES + 10)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient",
        lambda: real_client(transport=transport),
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig(request_timeout_ms=1234))

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(16000, dtype=np.float32), prompt="pending text")

    failure = raised.value.failure
    assert failure.category == "http_status"
    assert failure.status_code == 500
    assert len(failure.response_body.encode("utf-8")) <= 64 * 1024
    assert failure.response_body_truncated is True
    assert failure.method == "POST"
    assert failure.url.endswith("/v1/chat/completions")
    assert failure.timeout_ms == 1234
    assert failure.audio_seconds == 1.0
    assert failure.sample_count == 16000


@pytest.mark.asyncio
async def test_recognize_http_500_caps_encoded_body_with_invalid_utf8(monkeypatch):
    body = b"server detail:" + b"\xff" * (MAX_ERROR_RESPONSE_BYTES + 10)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient",
        lambda: real_client(transport=transport),
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(16000, dtype=np.float32), prompt="pending text")

    failure = raised.value.failure
    assert failure.response_body is not None
    assert failure.response_body.startswith("server detail:")
    assert len(failure.response_body.encode("utf-8")) <= MAX_ERROR_RESPONSE_BYTES
    assert failure.response_body_truncated is True


@pytest.mark.asyncio
async def test_recognize_http_500_marks_exact_raw_cap_invalid_utf8_as_truncated(monkeypatch):
    body = b"x" * (MAX_ERROR_RESPONSE_BYTES - 1) + b"\xff"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient",
        lambda: real_client(transport=transport),
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(16000, dtype=np.float32), prompt="pending text")

    failure = raised.value.failure
    assert failure.response_body is not None
    assert len(failure.response_body.encode("utf-8")) <= MAX_ERROR_RESPONSE_BYTES
    assert failure.response_body_truncated is True


@pytest.mark.asyncio
async def test_recognize_timeout_has_no_response_body(monkeypatch):
    class TimeoutClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *args, **kwargs):
            raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", args[0]))

    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", TimeoutClient
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(8000, dtype=np.float32), prompt="pending")

    assert raised.value.failure.category == "transport"
    assert raised.value.failure.status_code is None
    assert raised.value.failure.response_body is None
    assert raised.value.failure.sample_count == 8000


@pytest.mark.asyncio
async def test_recognize_invalid_json_is_structured(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient",
        lambda: real_client(transport=transport),
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(16000, dtype=np.float32), prompt="pending")

    assert raised.value.failure.category == "invalid_json"
    assert raised.value.failure.status_code == 200
    assert raised.value.failure.response_body == "not json"


@pytest.mark.asyncio
async def test_recognize_missing_text_is_structured(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {}}]}, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient",
        lambda: real_client(transport=transport),
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(16000, dtype=np.float32), prompt="pending")

    assert raised.value.failure.category == "invalid_response"
    assert raised.value.failure.status_code == 200
    assert raised.value.failure.response_body is not None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"choices": [{"message": {"content": "<asr_text> first </asr_text>"}}]}, "first"),
        ({"content": "<asr_text> second </asr_text>"}, "second"),
        ({"text": "<asr_text> third </asr_text>"}, "third"),
        (
            {"choices": [{"message": {"content": "prefix<asr_text>hello</asr_text>suffix"}}]},
            "hello",
        ),
        ({"choices": [{"message": {"content": "<asr_text>   </asr_text>"}}]}, ""),
    ],
)
def test_clean_qwen_text_accepts_supported_payload_shapes(payload, expected):
    assert clean_qwen_text(payload) == expected


def test_clean_qwen_text_rejects_missing_content():
    with pytest.raises(RuntimeError, match="missing recognized text"):
        clean_qwen_text({"choices": [{"message": {}}]})
