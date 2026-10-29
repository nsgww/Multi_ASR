from __future__ import annotations

# 历史遗留：本模块按性能线旧 WebApp（/api/recognition/* 批处理流 + 旧版
# index.html JS 结构）编写。主仓库已重写为 session 池架构（WebSocket 流式
# 会话/组播/声纹），对应实时覆盖见 test_frontend_api / test_lane_hotswap /
# test_speaker_api / test_buffers_metrics。恢复这些用例需按新契约重写。
import pytest

pytest.skip(
    "targets the retired perf-line webapp; superseded by test_frontend_api/"
    "test_lane_hotswap/test_speaker_api against the session-pool server",
    allow_module_level=True,
)

from dataclasses import replace

from fastapi.testclient import TestClient

from multi_stream_asr.config import AppConfig
from multi_stream_asr.web.server import create_app


def _index_html(tmp_path) -> str:
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        fake=True,
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    return response.text


def test_index_html_keeps_qwen_errors_separate_from_transcript_revision(tmp_path):
    html = _index_html(tmp_path)

    assert '["partial", "pending", "final"].includes(event.status)' in html
    assert "if (!isTranscriptEvent)" in html
    assert "`${streamId}:${sentenceId}:${event.status}`" in html
    assert ".result.qwen_error" in html
    assert ".result.fatal_error" in html
    assert 'const tracksRevision = ["partial", "pending", "final"].includes(event.status);' in html

    render_start = html.index("function renderLaneEvent(event)")
    render_end = html.index("function resetRecognitionView()", render_start)
    render_source = html[render_start:render_end]
    assert "lane.events.set(storageKey, event);" in render_source
    assert "row.className = `result ${event.status}`;" in render_source

    queue_start = html.index("function queueWebSocketEvent(event)")
    queue_end = html.index("function connectWebSocket()", queue_start)
    queue_source = html[queue_start:queue_end]
    assert "const key = laneKey(event.stream_id, event.sentence_id);" in queue_source
    flush_start = html.index("function flushPendingEvents()")
    flush_end = html.index("function queueWebSocketEvent(event)", flush_start)
    flush_source = html[flush_start:flush_end]
    assert 'if (event.status === "fatal_error") {' in flush_source
    assert 'appState.state = "fatal";' in flush_source


def test_index_html_batches_websocket_events_and_updates_rows_incrementally(tmp_path):
    html = _index_html(tmp_path)

    assert "pendingPartials: new Map()," in html
    assert "pendingNonPartials: []," in html
    assert "eventFlushScheduled: false," in html
    assert "window.requestAnimationFrame(flushPendingEvents);" in html

    ensure_start = html.index("function ensureLane(streamId)")
    ensure_end = html.index("function renderLaneGrid()", ensure_start)
    ensure_source = html[ensure_start:ensure_end]
    assert "rows: new Map()," in ensure_source

    render_start = html.index("function renderLaneEvent(event)")
    render_end = html.index("function resetRecognitionView()", render_start)
    render_source = html[render_start:render_end]
    assert "const row = lane.rows.get(storageKey)" in render_source
    assert "lane.rows.set(storageKey, row);" in render_source
    assert "lane.rows.delete(storageKey);" in render_source
    assert "lane.results.innerHTML = \"\";" not in render_source
