from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from multi_stream_asr.asr.hotwords import load_hotwords
from multi_stream_asr.asr.kws.keywords_tool import (
    keywords_file_to_words,
    words_to_keywords_lines,
    write_word_list,
)
from multi_stream_asr.asr.summary.config import SummaryConfig, load_summary_config
from multi_stream_asr.asr.summary.summary_client import SummaryClient
from multi_stream_asr.asr.summary.summary_server import SummaryServerManager
from multi_stream_asr.asr.text_filter import SensitiveWordFilter
from multi_stream_asr.config import AppConfig
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.session.manager import SessionManager
from multi_stream_asr.audio.multicast import (
    DEFAULT_MULTICAST_GROUP,
    DEFAULT_MULTICAST_PORT,
)
from multi_stream_asr.web.event_bus import FrontendEventBus
from multi_stream_asr.web.local_mic_handler import LocalMicHandler
from multi_stream_asr.web.multicast_handler import MulticastHandler
from multi_stream_asr.web.multicast_runtime import MulticastRuntimeRegistry
from multi_stream_asr.web.offline_handler import handle_offline_transcription
from multi_stream_asr.web.stream_handler import StreamHandler
from multi_stream_asr.web.wakeup_gate import WakeupSettings

_logger = logging.getLogger(__name__)

SessionManagerFactory = Callable[[AppConfig, FrontendEventBus], SessionManager]


class SummaryRequest(BaseModel):
    regenerate: bool = False
    style: str | None = None
    temperature: float | None = None
    random_temperature: bool = False
    min_temperature: float = 0.25
    max_temperature: float = 0.5


class WordListRequest(BaseModel):
    """动态词表更新请求：kind 为 hotwords/sensitive/keywords，words 为词列表。

    keywords 支持直接传中文（自动转拼音 token）或 "tokens@词" 原始格式。
    """

    kind: str
    words: list[str]


_WORD_KINDS = ("hotwords", "sensitive", "keywords")
_MAX_WORDS = 200
_MAX_WORD_LEN = 64
_WORD_UPLOAD_MAX_BYTES = 1024 * 1024
_WORD_UPLOAD_MAX_WORDS = 5000
_WORD_BACKUP_KEEP = 5
_WORD_FILE_HEADERS = {
    "hotwords": "# 热词表：一行一词，# 开头为注释（由 /v1/config/words 接口动态更新）",
    "sensitive": "# 敏感词表：一行一词，# 开头为注释（由 /v1/config/words 接口动态更新）",
    "keywords": "# KWS 唤醒词表（由 /v1/config/words 接口动态更新）",
}


def _default_session_manager_factory(
    config: AppConfig,
    event_bus: FrontendEventBus,
) -> SessionManager:
    return SessionManager(config=config, event_bus=event_bus)


def _status_payload(app: FastAPI) -> dict[str, Any]:
    manager: SessionManager = app.state.session_manager
    kws_cfg = app.state.config.kws
    return {
        "state": manager.state.value,
        "active_sessions": manager.active_sessions,
        "max_sessions": app.state.config.max_concurrent_sessions,
        "port": app.state.config.web_port,
        "kws": {
            "enabled": kws_cfg.enabled,
            "loaded": manager.kws_loaded,
            "model_dir": str(kws_cfg.model_dir),
            "keywords_file": str(kws_cfg.keywords_file),
        },
    }


async def _close_websockets(event_bus: FrontendEventBus) -> None:
    for websocket in tuple(event_bus._websockets):
        try:
            await websocket.close()
        except Exception:
            pass
        event_bus.unregister(websocket)


def create_app(
    config: AppConfig | None = None,
    *,
    session_manager: SessionManager | None = None,
    session_manager_factory: SessionManagerFactory | None = None,
    event_bus: FrontendEventBus | None = None,
    summary_config: SummaryConfig | None = None,
    summary_server_manager: SummaryServerManager | None = None,
) -> FastAPI:
    app_config = config or AppConfig()
    bus = event_bus or FrontendEventBus()
    factory = session_manager_factory or _default_session_manager_factory
    manager = session_manager or factory(app_config, bus)
    multicast_runtime = MulticastRuntimeRegistry(manager)
    summary_cfg = summary_config or load_summary_config(app_config.summary_config_path)
    sensitive_filter = SensitiveWordFilter.from_file(app_config.sensitive_words_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.broadcast_task = asyncio.create_task(bus.broadcast_loop())
        try:
            await manager.load_models()
            if summary_cfg.enabled and summary_cfg.provider == "llama" and summary_cfg.auto_start:
                if summary_server_manager is None:
                    app.state.summary_server_manager = SummaryServerManager(summary_cfg)
                else:
                    app.state.summary_server_manager = summary_server_manager
                await asyncio.to_thread(app.state.summary_server_manager.start)
            yield
        finally:
            with suppress(Exception):
                await multicast_runtime.stop_all()
            await manager.unload_models()
            if getattr(app.state, "summary_server_manager", None) is not None:
                with suppress(Exception):
                    await asyncio.to_thread(app.state.summary_server_manager.stop)
            await _close_websockets(bus)
            broadcast_task = app.state.broadcast_task
            if broadcast_task is not None:
                broadcast_task.cancel()
                with suppress(asyncio.CancelledError):
                    await broadcast_task

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.state.config = app_config
    app.state.event_bus = bus
    app.state.session_manager = manager
    app.state.multicast_runtime = multicast_runtime
    app.state.summary_config = summary_cfg
    app.state.sensitive_filter = sensitive_filter
    # 唤醒门控共享配置（运行时经 /v1/config/wakeup 调整，所有会话即时生效）
    wakeup_settings = WakeupSettings(
        window_seconds=app_config.wakeup_window_seconds
    )
    app.state.wakeup_settings = wakeup_settings
    app.state.broadcast_task = None
    app.state.summary_server_manager = None

    @app.get("/")
    async def index() -> dict[str, str]:
        return {
            "service": "multi-stream-asr",
            "api": "frontend-asr-api",
            "docs": "docs/frontend-asr-api.md",
            "test_page": "/test",
        }

    @app.get("/test")
    async def test_page() -> HTMLResponse:
        """Serve the local test HTML from the same origin to avoid file:// CORS issues."""
        candidates = [
            Path.cwd() / "test_stream.html",
            Path(__file__).resolve().parents[3] / "test_stream.html",
        ]
        for path in candidates:
            if path.is_file():
                return HTMLResponse(path.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="test_stream.html not found")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return _status_payload(app)

    def _words_file_for(kind: str) -> Path:
        if kind == "hotwords":
            return app_config.hotwords_file
        if kind == "sensitive":
            return app_config.sensitive_words_file
        if kind == "keywords":
            return app_config.kws.keywords_file
        raise HTTPException(
            status_code=400,
            detail=f"unknown kind: {kind!r}; expected one of {_WORD_KINDS}",
        )

    def _backup_word_file(path: Path) -> Path | None:
        """更新前备份旧词表文件（保留最近 5 份），无旧文件返回 None。"""
        if not path.is_file():
            return None
        backup = path.with_name(f"{path.name}.{time.strftime('%Y%m%d%H%M%S')}.bak")
        backup.write_bytes(path.read_bytes())
        old_backups = sorted(path.parent.glob(f"{path.name}.*.bak"))[:-_WORD_BACKUP_KEEP:]
        for old in old_backups:
            with suppress(OSError):
                old.unlink()
        return backup

    async def _apply_words(kind: str, words: list[str]) -> dict[str, Any]:
        """写词表文件并热更新（PUT/上传共用），失败回滚文件与内存状态。"""
        path = _words_file_for(kind)
        if kind == "keywords":
            return await _apply_keywords_lines(words_to_keywords_lines(words))
        old_words = load_hotwords(path)  # hotwords/sensitive 同为一行一词格式
        backup = _backup_word_file(path)
        write_word_list(path, words, header=_WORD_FILE_HEADERS[kind])
        try:
            if kind == "hotwords":
                manager.set_hotwords(words)
                note = "精修热词即时生效；在线 Zipformer 热词需重启（默认关闭）"
            else:
                sensitive_filter.update_words(words)
                note = "敏感词即时生效（含进行中的会话）"
        except Exception:
            # 热更新失败：回滚文件与内存
            write_word_list(path, old_words, header=_WORD_FILE_HEADERS[kind])
            if kind == "hotwords":
                manager.set_hotwords(old_words)
            else:
                sensitive_filter.update_words(old_words)
            raise
        response: dict[str, Any] = {
            "kind": kind,
            "count": len(words),
            "file": str(path),
            "note": note,
        }
        if backup is not None:
            response["backup"] = str(backup)
        return response

    async def _apply_keywords_lines(lines: list[str]) -> dict[str, Any]:
        """keywords 专用：写 token 化词表并重建 KWS，失败回滚。lines 已是 token 格式。"""
        path = _words_file_for("keywords")
        old_text = path.read_text(encoding="utf-8") if path.is_file() else None
        backup = _backup_word_file(path)
        # 注意：sherpa KWS 原生解析器不支持 # 注释行，keywords 文件不写注释头
        write_word_list(path, lines)
        try:
            await asyncio.to_thread(manager.reload_kws_keywords)
        except Exception:
            if old_text is not None:
                path.write_text(old_text, encoding="utf-8")
                with suppress(Exception):
                    await asyncio.to_thread(manager.reload_kws_keywords)
            raise
        response: dict[str, Any] = {
            "kind": "keywords",
            "count": len(lines),
            "file": str(path),
            "note": "唤醒词对新连接生效；已连接会话沿用旧词表",
            "lines": lines,  # 返回 token 行，便于核对多音字读音
        }
        if backup is not None:
            response["backup"] = str(backup)
        return response

    @app.get("/v1/config/words")
    async def get_words(kind: str) -> dict[str, Any]:
        path = _words_file_for(kind)
        if kind == "keywords":
            words = keywords_file_to_words(path)
        else:
            words = load_hotwords(path)  # hotwords/sensitive 同为一行一词格式
        return {"kind": kind, "words": words, "count": len(words), "file": str(path)}

    @app.put("/v1/config/words")
    async def put_words(req: WordListRequest) -> dict[str, Any]:
        _words_file_for(req.kind)  # 先校验 kind
        words: list[str] = []
        for raw in req.words:
            word = raw.strip()
            if not word:
                continue
            if len(word) > _MAX_WORD_LEN:
                raise HTTPException(status_code=400, detail=f"word too long: {word!r}")
            if word not in words:
                words.append(word)
        if len(words) > _MAX_WORDS:
            raise HTTPException(
                status_code=400, detail=f"too many words (max {_MAX_WORDS})"
            )
        try:
            response = await _apply_words(req.kind, words)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, **response}

    @app.post("/v1/config/words/upload")
    async def upload_words(kind: str, file: UploadFile, raw: bool = False) -> dict[str, Any]:
        """上传词表文件整体替换：一行一词，空行与 # 注释行忽略。

        - hotwords/sensitive：传中文词文本文件（UTF-8）
        - keywords：默认传中文词（服务端自动转拼音 token）；
          `raw=true` 时传 sherpa 原生 token 化文件（每行须含 @），原样落盘
        更新前自动备份旧文件（保留最近 5 份），热更新失败自动回滚。
        """
        _words_file_for(kind)  # 先校验 kind
        content = await file.read(_WORD_UPLOAD_MAX_BYTES + 1)
        if len(content) > _WORD_UPLOAD_MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"file too large (max {_WORD_UPLOAD_MAX_BYTES} bytes)",
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="file must be UTF-8 text") from exc

        entries: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if kind != "keywords" and len(line) > _MAX_WORD_LEN:
                raise HTTPException(status_code=400, detail=f"word too long: {line!r}")
            entries.append(line)
        if len(entries) > _WORD_UPLOAD_MAX_WORDS:
            raise HTTPException(
                status_code=400,
                detail=f"too many words (max {_WORD_UPLOAD_MAX_WORDS})",
            )

        try:
            if kind == "keywords" and raw:
                for line in entries:
                    if "@" not in line:
                        raise HTTPException(
                            status_code=400,
                            detail=f"raw keywords 行缺少 @ token 部分: {line!r}",
                        )
                response = await _apply_keywords_lines(entries)
            else:
                response = await _apply_words(kind, entries)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "filename": file.filename, **response}

    @app.get("/v1/config/wakeup")
    async def get_wakeup_config() -> dict[str, Any]:
        return {
            "window_seconds": wakeup_settings.window_seconds,
            "note": "唤醒后限时转写秒数；0 = 不限时",
        }

    @app.put("/v1/config/wakeup")
    async def put_wakeup_config(payload: dict[str, Any]) -> dict[str, Any]:
        value = payload.get("window_seconds")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(
                status_code=400, detail="window_seconds must be a number"
            )
        if not 0 <= float(value) <= 3600:
            raise HTTPException(
                status_code=400, detail="window_seconds must be in [0, 3600]"
            )
        wakeup_settings.window_seconds = float(value)
        return {
            "ok": True,
            "window_seconds": wakeup_settings.window_seconds,
            "note": "即时生效（含进行中的会话）；0 = 不限时",
        }

    @app.get("/v1/audio/devices")
    async def list_audio_devices() -> JSONResponse:
        """List pyaudio input devices available on the board."""
        try:
            import pyaudio
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"pyaudio not available: {exc}"
            ) from exc
        try:
            pa = pyaudio.PyAudio()
            devices: list[dict[str, Any]] = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": info.get("name", ""),
                            "max_input_channels": int(info.get("maxInputChannels", 0)),
                            "default_sample_rate": int(info.get("defaultSampleRate", 0)),
                        }
                    )
            pa.terminate()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to list devices: {exc}"
            ) from exc
        return JSONResponse(content={"devices": devices})

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket) -> None:
        params = websocket.query_params
        try:
            sample_rate = int(params.get("sample_rate", "16000"))
        except ValueError:
            sample_rate = 16000
        if sample_rate <= 0:
            sample_rate = 16000
        wakeup_enabled = params.get("wakeup", "true").lower() not in ("0", "false", "no")
        handler = StreamHandler(
            websocket=websocket,
            session_manager=manager,
            summary_config=summary_cfg,
            output_dir=app_config.output_dir,
            sample_rate=sample_rate,
            wakeup_enabled=wakeup_enabled,
            sensitive_filter=sensitive_filter,
            wakeup_settings=wakeup_settings,
        )
        await handler.run()

    @app.websocket("/ws/multicast")
    async def websocket_multicast(websocket: WebSocket) -> None:
        params = websocket.query_params
        handler = MulticastHandler(
            websocket=websocket,
            session_manager=manager,
            runtime=multicast_runtime,
            summary_config=summary_cfg,
            output_dir=app_config.output_dir,
            group=params.get("group", DEFAULT_MULTICAST_GROUP),
            port=int(params.get("port", DEFAULT_MULTICAST_PORT)),
            local_ip=params.get("local_ip", "0.0.0.0"),
            packet_bytes=int(params.get("packet_bytes", "1280")),
            use_full_packet=params.get("use_full_packet", "false").lower() == "true",
            # sm-sdk relay default 5s CMD_B wake-up for hardware multicast sources
            heartbeat_interval=float(params.get("heartbeat_interval", "5")),
            speexdsp_lib_path=app_config.speexdsp_lib_path,
            wakeup_enabled=params.get("wakeup", "true").lower() not in ("0", "false", "no"),
            sensitive_filter=sensitive_filter,
            wakeup_settings=wakeup_settings,
        )
        await handler.run()

    @app.websocket("/ws/local_mic")
    async def websocket_local_mic(websocket: WebSocket) -> None:
        params = websocket.query_params
        try:
            sample_rate = int(params.get("sample_rate", "16000"))
        except ValueError:
            sample_rate = 16000
        if sample_rate <= 0:
            sample_rate = 16000
        device_raw = params.get("device")
        if device_raw is None:
            device: int | str | None = None
        else:
            try:
                device = int(device_raw)
            except ValueError:
                device = device_raw
        try:
            chunk = int(params.get("chunk", "512"))
        except ValueError:
            chunk = 512
        if chunk <= 0:
            chunk = 512
        wakeup_enabled = params.get("wakeup", "true").lower() not in ("0", "false", "no")
        handler = LocalMicHandler(
            websocket=websocket,
            session_manager=manager,
            summary_config=summary_cfg,
            output_dir=app_config.output_dir,
            device=device,
            sample_rate=sample_rate,
            chunk=chunk,
            wakeup_enabled=wakeup_enabled,
            sensitive_filter=sensitive_filter,
            wakeup_settings=wakeup_settings,
        )
        await handler.run()

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile,
        response_format: str = Form("json"),
        enable_summary: bool = Form(False),
    ) -> Response:
        if manager.state is not PipelineState.LOADED:
            raise HTTPException(
                status_code=503,
                detail="service is not loaded",
            )
        return await handle_offline_transcription(
            config=app_config,
            summary_config=summary_cfg,
            file=file,
            response_format=response_format,
            enable_summary=enable_summary,
            lane_id=0,
            sensitive_filter=sensitive_filter,
        )

    @app.post("/v1/transcripts/{filename}/summary")
    async def regenerate_summary(filename: str, request: SummaryRequest) -> JSONResponse:
        transcript_dir = Path(app_config.output_dir) / "transcripts"
        transcript_path = transcript_dir / filename
        try:
            transcript_path = transcript_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="transcript not found") from exc
        if not transcript_path.is_relative_to(transcript_dir.resolve()):
            raise HTTPException(status_code=400, detail="invalid filename")
        if not transcript_path.is_file():
            raise HTTPException(status_code=404, detail="transcript file missing")

        if not summary_cfg.enabled:
            return JSONResponse(
                content={
                    "provider": "disabled",
                    "reason": "summary disabled",
                },
                status_code=503,
            )

        text = transcript_path.read_text(encoding="utf-8")
        client = SummaryClient(config=summary_cfg)
        try:
            result = await client.generate(
                text,
                style=request.style,
                temperature=request.temperature,
                random_temperature=request.random_temperature,
                min_temperature=request.min_temperature,
                max_temperature=request.max_temperature,
                regenerate=request.regenerate,
            )

            summary_dir = Path(app_config.output_dir) / "summaries"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / f"{filename}.summary.json"
            summary_path.write_text(
                json.dumps(result.to_payload(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return JSONResponse(content=result.to_payload())
        except Exception as exc:
            _logger.exception("summary regeneration failed")
            raise HTTPException(
                status_code=500,
                detail=f"summary generation failed: {exc}",
            ) from exc
        finally:
            await client.close()

    @app.get("/v1/transcripts/{filename}")
    async def download_transcript(filename: str) -> FileResponse:
        transcript_dir = Path(app_config.output_dir) / "transcripts"
        transcript_path = transcript_dir / filename
        try:
            transcript_path = transcript_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="transcript not found") from exc
        if not transcript_path.is_relative_to(transcript_dir.resolve()):
            raise HTTPException(status_code=400, detail="invalid filename")
        if not transcript_path.is_file():
            raise HTTPException(status_code=404, detail="transcript file missing")
        return FileResponse(transcript_path, filename=filename)

    return app
