from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from multi_stream_asr.log.qwen_errors import (
    MAX_ERROR_RESPONSE_BYTES,
    QwenErrorCoordinator,
    QwenFailureDecision,
)


def failure_payload(status: int = 500, body: str = "decoder failed") -> dict[str, object]:
    return {
        "run_id": "process-1",
        "sentence_id": 34,
        "revision": 5,
        "pending_text": "pending text",
        "timestamp_ms": 232700,
        "request": {
            "category": "http_status",
            "error_type": "HTTPStatusError",
            "message": f"HTTP {status}",
            "method": "POST",
            "url": "http://127.0.0.1:8090/v1/chat/completions",
            "timeout_ms": 120000,
            "audio_seconds": 2.6,
            "sample_count": 41600,
            "status_code": status,
            "response_body": body,
            "response_body_truncated": False,
        },
    }


def start_coordinator(tmp_path: Path, initial_log: bytes = b"startup\n") -> tuple[QwenErrorCoordinator, Path, io.StringIO]:
    server_log = tmp_path / "llama.log"
    server_log.write_bytes(initial_log)
    stderr = io.StringIO()
    coordinator = QwenErrorCoordinator(server_log_path=server_log, stderr=stderr)
    coordinator.reset_for_model_load()
    return coordinator, server_log, stderr


def test_success_advances_checkpoint_and_failure_saves_only_new_bytes(tmp_path: Path):
    server_log = tmp_path / "llama.log"
    server_log.write_bytes(b"startup\n")
    stderr = io.StringIO()
    coordinator = QwenErrorCoordinator(server_log_path=server_log, stderr=stderr)
    coordinator.reset_for_model_load()
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)

    server_log.write_bytes(b"startup\nsuccess noise\n")
    coordinator.record_success()
    server_log.write_bytes(b"startup\nsuccess noise\nerror detail\n")
    decision = coordinator.record_failure(stream_id=2, time_ms=1000, payload=failure_payload())

    assert decision == QwenFailureDecision(1, 1, False)
    server_artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert "error detail" in server_artifact
    assert "success noise" not in server_artifact
    record = json.loads((run_dir / "errors/qwen_requests.jsonl").read_text(encoding="utf-8"))
    assert record["stream_id"] == 2
    assert record["pending_text"] == "pending text"
    assert record["run_failure_sequence"] == 1
    assert record["session_failure_sequence"] == 1


def test_failure_lazily_creates_artifacts_marks_empty_slice_and_prints_stderr(tmp_path: Path):
    coordinator, _server_log, stderr = start_coordinator(tmp_path)
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)

    assert not (run_dir / "errors").exists()
    coordinator.record_failure(stream_id=2, time_ms=1000, payload=failure_payload())

    assert (run_dir / "errors/qwen_requests.jsonl").is_file()
    assert "no new server log bytes" in (run_dir / "errors/qwen_server.log").read_text(
        encoding="utf-8"
    )
    printed = stderr.getvalue()
    assert "[qwen-error]" in printed
    assert "lane=2" in printed
    assert "sentence=34" in printed
    assert "session_failure=1" in printed
    assert "error_type=HTTPStatusError" in printed
    assert "status=500" in printed
    assert "response_body=decoder failed" in printed


def test_failure_persists_allowlisted_request_fields_and_bounds_response_body(tmp_path: Path):
    coordinator, _server_log, stderr = start_coordinator(tmp_path)
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    body = "x" * (MAX_ERROR_RESPONSE_BYTES - 1) + "\u754c"
    payload = failure_payload(body=body)
    request = payload["request"]
    assert isinstance(request, dict)
    request.update(
        {
            "audio": "sensitive audio payload",
            "base64": "sensitive base64 payload",
            "unexpected": "sensitive arbitrary payload",
        }
    )

    coordinator.record_failure(stream_id=2, time_ms=1000, payload=payload)

    record = json.loads((run_dir / "errors/qwen_requests.jsonl").read_text(encoding="utf-8"))
    assert set(record) == {
        "schema_version",
        "run_id",
        "time_ms",
        "stream_id",
        "sentence_id",
        "revision",
        "pending_text",
        "timestamp_ms",
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
        "run_failure_sequence",
        "session_failure_sequence",
    }
    assert len(record["response_body"].encode("utf-8")) <= MAX_ERROR_RESPONSE_BYTES
    assert record["response_body_truncated"] is True
    assert record["response_body"] == "x" * (MAX_ERROR_RESPONSE_BYTES - 1)
    serialized = json.dumps(record, ensure_ascii=False)
    printed = stderr.getvalue()
    for sensitive_value in (
        "sensitive audio payload",
        "sensitive base64 payload",
        "sensitive arbitrary payload",
    ):
        assert sensitive_value not in serialized
        assert sensitive_value not in printed


def test_consecutive_failures_copy_each_server_log_byte_once(tmp_path: Path):
    coordinator, server_log, _stderr = start_coordinator(tmp_path)
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)

    server_log.write_bytes(b"startup\nfirst failure\n")
    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())
    server_log.write_bytes(b"startup\nfirst failure\nsecond failure\n")
    coordinator.record_failure(stream_id=1, time_ms=2000, payload=failure_payload())

    artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert artifact.count("first failure") == 1
    assert artifact.count("second failure") == 1


def test_failure_records_partial_invalid_utf8_server_log_bytes_with_replacement(tmp_path: Path):
    coordinator, server_log, _stderr = start_coordinator(tmp_path)
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    server_log.write_bytes(b"startup\npartial record: \xff")

    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())

    artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert "partial record: \ufffd" in artifact


def test_failure_propagates_missing_server_log(tmp_path: Path):
    coordinator, server_log, _stderr = start_coordinator(tmp_path)
    coordinator.start_run(tmp_path / "process-1")
    server_log.unlink()

    with pytest.raises(FileNotFoundError):
        coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())


def test_failure_restarts_slice_when_server_log_is_truncated(tmp_path: Path):
    coordinator, server_log, _stderr = start_coordinator(tmp_path, b"long startup record\n")
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    server_log.write_bytes(b"short\n")

    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())

    artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert "short" in artifact
    assert "bytes=0:6" in artifact


def test_failure_restarts_slice_after_same_inode_copytruncate_and_regrowth(tmp_path: Path):
    initial_log = b"checkpoint before copytruncate\n"
    coordinator, server_log, _stderr = start_coordinator(tmp_path, initial_log)
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    original_inode = server_log.stat().st_ino
    replacement_log = b"replacement after copytruncate has regrown past the checkpoint\n"
    assert len(replacement_log) > len(initial_log)

    server_log.write_bytes(replacement_log)

    assert server_log.stat().st_ino == original_inode
    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())

    artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert "replacement after copytruncate" in artifact
    assert f"bytes=0:{len(replacement_log)}" in artifact


def test_failure_restarts_slice_when_server_log_is_rotated(tmp_path: Path):
    coordinator, server_log, _stderr = start_coordinator(tmp_path, b"old server log\n")
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    replacement = tmp_path / "llama.log.next"
    replacement.write_bytes(b"rotated server log has a longer replacement\n")
    replacement.replace(server_log)

    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())

    artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert "rotated server log has a longer replacement" in artifact
    assert "bytes=0:" in artifact


def test_failure_reads_log_snapshot_from_the_descriptor_it_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    coordinator, server_log, _stderr = start_coordinator(tmp_path, b"old log\n")
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    replacement_log = b"replacement installed immediately before the source open\n"
    original_open = Path.open
    rotated = False

    def open_after_rotation(path: Path, *args: object, **kwargs: object):
        nonlocal rotated
        if path == server_log and not rotated:
            rotated = True
            replacement = tmp_path / "llama.log.replacement"
            replacement.write_bytes(replacement_log)
            replacement.replace(server_log)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_after_rotation)

    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())

    artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert rotated
    assert "replacement installed immediately before the source open" in artifact
    assert f"bytes=0:{len(replacement_log)}" in artifact


def test_session_count_survives_runs_and_only_third_failure_is_fatal(tmp_path: Path):
    coordinator, _server_log, _stderr = start_coordinator(tmp_path)
    coordinator.start_run(tmp_path / "process-1")
    assert coordinator.record_failure(
        stream_id=0, time_ms=1000, payload=failure_payload()
    ) == QwenFailureDecision(1, 1, False)
    coordinator.finish_run()

    coordinator.start_run(tmp_path / "process-2")
    assert coordinator.record_failure(
        stream_id=0, time_ms=2000, payload=failure_payload()
    ) == QwenFailureDecision(1, 2, False)
    assert coordinator.record_failure(
        stream_id=0, time_ms=3000, payload=failure_payload()
    ) == QwenFailureDecision(2, 3, True)
    assert coordinator.record_failure(
        stream_id=0, time_ms=4000, payload=failure_payload()
    ) == QwenFailureDecision(3, 4, False)


def test_artifact_flush_failure_does_not_commit_counters_checkpoint_or_fatal_latch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    coordinator, server_log, _stderr = start_coordinator(tmp_path)
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)
    assert coordinator.record_failure(
        stream_id=0, time_ms=1000, payload=failure_payload()
    ) == QwenFailureDecision(1, 1, False)
    assert coordinator.record_failure(
        stream_id=0, time_ms=2000, payload=failure_payload()
    ) == QwenFailureDecision(2, 2, False)
    server_log.write_bytes(b"startup\nbytes from failed artifact write\n")
    original_open = Path.open

    class FailingFlushFile:
        def __init__(self, file: object) -> None:
            self._file = file

        def __enter__(self) -> "FailingFlushFile":
            self._file.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._file.__exit__(*args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._file, name)

        def flush(self) -> None:
            raise OSError("injected qwen server artifact flush failure")

    def open_with_failing_server_flush(path: Path, *args: object, **kwargs: object):
        file = original_open(path, *args, **kwargs)
        if path == run_dir / "errors/qwen_server.log":
            return FailingFlushFile(file)
        return file

    monkeypatch.setattr(Path, "open", open_with_failing_server_flush)

    with pytest.raises(OSError, match="injected qwen server artifact flush failure"):
        coordinator.record_failure(stream_id=0, time_ms=3000, payload=failure_payload())

    monkeypatch.undo()
    server_log.write_bytes(
        b"startup\nbytes from failed artifact write\nbytes after retry\n"
    )
    decision = coordinator.record_failure(stream_id=0, time_ms=4000, payload=failure_payload())

    assert decision == QwenFailureDecision(3, 3, True)
    server_artifact = (run_dir / "errors/qwen_server.log").read_text(encoding="utf-8")
    assert "bytes from failed artifact write" in server_artifact
    assert "bytes after retry" in server_artifact


def test_unload_resets_counts_checkpoint_and_run_association(tmp_path: Path):
    coordinator, server_log, _stderr = start_coordinator(tmp_path)
    coordinator.start_run(tmp_path / "process-1")
    coordinator.record_failure(stream_id=0, time_ms=1000, payload=failure_payload())
    coordinator.reset_for_unload()

    server_log.write_bytes(b"after unload\n")
    coordinator.start_run(tmp_path / "process-2")
    decision = coordinator.record_failure(stream_id=0, time_ms=2000, payload=failure_payload())

    assert decision == QwenFailureDecision(1, 1, False)
    artifact = (tmp_path / "process-2/errors/qwen_server.log").read_text(encoding="utf-8")
    assert "after unload" in artifact
    assert "bytes=0:13" in artifact
