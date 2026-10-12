from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO


CHECKPOINT_FINGERPRINT_BYTES = 128
MAX_ERROR_RESPONSE_BYTES = 64 * 1024
REQUEST_FIELD_ALLOWLIST = (
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
)


@dataclass(frozen=True)
class QwenFailureDecision:
    run_sequence: int
    session_sequence: int
    should_fatal: bool


@dataclass(frozen=True)
class _ServerLogSnapshot:
    end: int
    identity: tuple[int, int]
    fingerprint: bytes


class QwenErrorCoordinator:
    def __init__(self, *, server_log_path: Path, stderr: TextIO = sys.stderr) -> None:
        self.server_log_path = server_log_path
        self.stderr = stderr
        self._checkpoint = 0
        self._checkpoint_identity: tuple[int, int] | None = None
        self._checkpoint_fingerprint: bytes | None = None
        self._session_failures = 0
        self._run_failures = 0
        self._run_dir: Path | None = None
        self._fatal_latched = False

    def reset_for_model_load(self) -> None:
        self.finish_run()
        self._session_failures = 0
        self._run_failures = 0
        self._fatal_latched = False
        self._commit_checkpoint(self._server_log_snapshot())

    def start_run(self, run_dir: Path) -> None:
        self.finish_run()
        self._run_dir = run_dir
        self._run_failures = 0

    def record_success(self) -> None:
        self._commit_checkpoint(self._server_log_snapshot())

    def record_failure(
        self,
        *,
        stream_id: int,
        time_ms: int,
        payload: Mapping[str, object],
    ) -> QwenFailureDecision:
        if self._run_dir is None:
            raise RuntimeError("cannot record a Qwen failure without an active run")

        run_sequence = self._run_failures + 1
        session_sequence = self._session_failures + 1
        should_fatal = session_sequence >= 3 and not self._fatal_latched
        log_start, log_snapshot, log_bytes = self._read_server_log_since_checkpoint()
        request_record = self._request_record(
            stream_id=stream_id,
            time_ms=time_ms,
            payload=payload,
            run_sequence=run_sequence,
            session_sequence=session_sequence,
        )

        errors_dir = self._run_dir / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        with (errors_dir / "qwen_requests.jsonl").open("a", encoding="utf-8") as request_file:
            request_file.write(json.dumps(request_record, ensure_ascii=False, sort_keys=True))
            request_file.write("\n")
            request_file.flush()
        with (errors_dir / "qwen_server.log").open("a", encoding="utf-8") as server_file:
            server_file.write(self._server_log_record(
                stream_id=stream_id,
                time_ms=time_ms,
                sentence_id=payload["sentence_id"],
                start=log_start,
                end=log_snapshot.end,
                log_bytes=log_bytes,
            ))
            server_file.flush()

        self._run_failures = run_sequence
        self._session_failures = session_sequence
        self._commit_checkpoint(log_snapshot)
        if should_fatal:
            self._fatal_latched = True
        self._print_failure(stream_id, request_record)
        return QwenFailureDecision(run_sequence, session_sequence, should_fatal)

    def finish_run(self) -> None:
        self._run_dir = None

    def reset_for_unload(self) -> None:
        self.finish_run()
        self._checkpoint = 0
        self._checkpoint_identity = None
        self._checkpoint_fingerprint = None
        self._session_failures = 0
        self._run_failures = 0
        self._fatal_latched = False

    def _server_log_snapshot(self) -> _ServerLogSnapshot:
        with self.server_log_path.open("rb") as server_log:
            return self._snapshot_open_server_log(server_log)

    def _read_server_log_since_checkpoint(self) -> tuple[int, _ServerLogSnapshot, bytes]:
        with self.server_log_path.open("rb") as server_log:
            snapshot = self._snapshot_open_server_log(server_log)
            log_start = self._log_slice_start(server_log, snapshot)
            server_log.seek(log_start)
            return log_start, snapshot, server_log.read(snapshot.end - log_start)

    def _log_slice_start(self, server_log: BinaryIO, snapshot: _ServerLogSnapshot) -> int:
        if self._checkpoint_identity != snapshot.identity or snapshot.end < self._checkpoint:
            return 0
        if self._checkpoint_fingerprint != self._server_log_fingerprint(
            server_log, self._checkpoint
        ):
            return 0
        return self._checkpoint

    def _snapshot_open_server_log(self, server_log: BinaryIO) -> _ServerLogSnapshot:
        stat = os.fstat(server_log.fileno())
        return _ServerLogSnapshot(
            end=stat.st_size,
            identity=(stat.st_dev, stat.st_ino),
            fingerprint=self._server_log_fingerprint(server_log, stat.st_size),
        )

    def _commit_checkpoint(self, snapshot: _ServerLogSnapshot) -> None:
        self._checkpoint = snapshot.end
        self._checkpoint_identity = snapshot.identity
        self._checkpoint_fingerprint = snapshot.fingerprint

    @staticmethod
    def _server_log_fingerprint(server_log: BinaryIO, end: int) -> bytes:
        fingerprint = hashlib.blake2s(digest_size=16)
        prefix_size = min(end, CHECKPOINT_FINGERPRINT_BYTES)
        server_log.seek(0)
        fingerprint.update(server_log.read(prefix_size))
        suffix_start = max(prefix_size, end - CHECKPOINT_FINGERPRINT_BYTES)
        server_log.seek(suffix_start)
        fingerprint.update(server_log.read(end - suffix_start))
        return fingerprint.digest()

    @staticmethod
    def _request_record(
        *,
        stream_id: int,
        time_ms: int,
        payload: Mapping[str, object],
        run_sequence: int,
        session_sequence: int,
    ) -> dict[str, object]:
        request = payload["request"]
        if not isinstance(request, Mapping):
            raise TypeError("Qwen failure payload request must be a mapping")
        request_record = {field: request[field] for field in REQUEST_FIELD_ALLOWLIST}
        response_body, truncated_at_persistence = _truncate_response_body(
            request_record["response_body"]
        )
        request_record["response_body"] = response_body
        request_record["response_body_truncated"] = bool(
            request_record["response_body_truncated"]
        ) or truncated_at_persistence
        return {
            "schema_version": 1,
            "run_id": payload["run_id"],
            "time_ms": time_ms,
            "stream_id": stream_id,
            "sentence_id": payload["sentence_id"],
            "revision": payload["revision"],
            "pending_text": payload["pending_text"],
            "timestamp_ms": payload["timestamp_ms"],
            **request_record,
            "run_failure_sequence": run_sequence,
            "session_failure_sequence": session_sequence,
        }

    @staticmethod
    def _server_log_record(
        *,
        stream_id: int,
        time_ms: int,
        sentence_id: object,
        start: int,
        end: int,
        log_bytes: bytes,
    ) -> str:
        delimiter = (
            f"--- qwen failure time_ms={time_ms} lane={stream_id} "
            f"sentence={sentence_id} bytes={start}:{end} ---\n"
        )
        if not log_bytes:
            return f"{delimiter}no new server log bytes\n"
        copied = log_bytes.decode("utf-8", errors="replace")
        if not copied.endswith("\n"):
            copied += "\n"
        return f"{delimiter}{copied}"

    def _print_failure(self, stream_id: int, record: Mapping[str, object]) -> None:
        response_body = record["response_body"]
        print(
            "[qwen-error] "
            f"lane={stream_id} "
            f"sentence={record['sentence_id']} "
            f"session_failure={record['session_failure_sequence']} "
            f"error_type={record['error_type']} "
            f"status={record['status_code']} "
            f"response_body={response_body}",
            file=self.stderr,
            flush=True,
        )


def _truncate_response_body(response_body: object) -> tuple[str | None, bool]:
    if response_body is None:
        return None, False
    if not isinstance(response_body, str):
        raise TypeError("Qwen failure response_body must be a string or None")
    encoded = response_body.encode("utf-8")
    if len(encoded) <= MAX_ERROR_RESPONSE_BYTES:
        return response_body, False
    return encoded[:MAX_ERROR_RESPONSE_BYTES].decode("utf-8", errors="ignore"), True
