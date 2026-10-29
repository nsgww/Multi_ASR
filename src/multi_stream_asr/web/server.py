from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, StrictInt

from multi_stream_asr.asr.hotwords import load_hotwords
from multi_stream_asr.asr.kws.keywords_tool import (
    find_invalid_keyword_tokens,
    keywords_file_to_words,
    words_to_keywords_lines,
    write_word_list,
)
from multi_stream_asr.asr.summary.config import SummaryConfig, load_summary_config
from multi_stream_asr.asr.summary.summary_client import SummaryClient
from multi_stream_asr.asr.summary.summary_server import SummaryServerManager
from multi_stream_asr.asr.text_filter import SensitiveWordFilter
from multi_stream_asr.config import (
    OFFLINE_MODELS,
    SUPPORTED_LANES,
    AppConfig,
    validate_lane_count,
    validate_offline_model_files,
)
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager
from multi_stream_asr.speaker import SherpaSpeakerEmbedder, VoiceprintRegistry
from multi_stream_asr.speaker.audio import wav_bytes_to_pcm16k
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

SessionManagerFactory = Callable[
    [AppConfig, FrontendEventBus], ProcessPoolSessionManager
]


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


class MulticastStartRequest(BaseModel):
    """新增一路组播转写。"""

    group: str
    port: int = 6020
    local_ip: str = "0.0.0.0"
    packet_bytes: int = 1280
    use_full_packet: bool = False
    # sm-sdk relay 默认 5s CMD_B 心跳唤醒硬件组播源
    heartbeat_interval: float = 5.0


class MulticastStopRequest(BaseModel):
    group: str
    port: int = 6020


class MulticastModifyRequest(BaseModel):
    """修改已有组播任务的地址。new_port 省略时沿用旧端口。"""

    old_group: str
    old_port: int = 6020
    new_group: str
    new_port: int | None = None


_WORD_KINDS = ("hotwords", "sensitive", "keywords")
_MAX_WORDS = 200
_MAX_WORD_LEN = 64
_WORD_UPLOAD_MAX_BYTES = 1024 * 1024
_WORD_UPLOAD_MAX_WORDS = 5000
_WORD_BACKUP_KEEP = 5
_MAX_SPEAKER_NAME_LEN = 64
_MAX_ENROLL_FILES = 8
_WORD_FILE_HEADERS = {
    "hotwords": "# 热词表：一行一词，# 开头为注释（由 /v1/config/words 接口动态更新）",
    "sensitive": "# 敏感词表：一行一词，# 开头为注释（由 /v1/config/words 接口动态更新）",
    "keywords": "# KWS 唤醒词表（由 /v1/config/words 接口动态更新）",
}


def _default_session_manager_factory(
    config: AppConfig,
    event_bus: FrontendEventBus,
) -> ProcessPoolSessionManager:
    return ProcessPoolSessionManager(config=config, event_bus=event_bus)


class LoadModelsRequest(BaseModel):
    """通道热改 / 模型重载请求（对齐性能调优线的 /api/models/load）。

    所有字段可选：不带的字段沿用当前配置。``lanes`` 变化需要先 unload
    再 load（服务端会整体重建 session 池）。
    """

    lanes: StrictInt | None = None
    offline_model_id: str | None = None
    offline_model_dir: Path | None = None


def _config_from_load_request(config: AppConfig, request: LoadModelsRequest) -> AppConfig:
    """把 load 请求折叠成新 AppConfig（只覆盖请求里出现的字段）。"""
    lanes = config.lanes if request.lanes is None else validate_lane_count(request.lanes)
    models = config.models
    kwargs: dict[str, Any] = {}
    if request.offline_model_id is not None:
        if request.offline_model_id not in OFFLINE_MODELS:
            raise KeyError(
                f"unsupported offline model: {request.offline_model_id!r}; "
                f"expected one of {sorted(OFFLINE_MODELS)}"
            )
        kwargs["offline_model_id"] = request.offline_model_id
    if request.offline_model_dir is not None:
        kwargs["offline_model_dir"] = request.offline_model_dir.expanduser()
    if kwargs:
        models = replace(models, **kwargs)
    return replace(config, lanes=lanes, models=models)


def _status_payload(app: FastAPI) -> dict[str, Any]:
    manager: ProcessPoolSessionManager = app.state.session_manager
    kws_cfg = app.state.config.kws
    return {
        "state": manager.state.value,
        "active_sessions": manager.active_sessions,
        "max_sessions": app.state.config.max_concurrent_sessions,
        "alive_lanes": manager.alive_lanes,
        "port": app.state.config.web_port,
        "lanes": app.state.config.lanes,
        "supported_lanes": list(SUPPORTED_LANES),
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
    session_manager: ProcessPoolSessionManager | None = None,
    session_manager_factory: SessionManagerFactory | None = None,
    event_bus: FrontendEventBus | None = None,
    summary_config: SummaryConfig | None = None,
    summary_server_manager: SummaryServerManager | None = None,
) -> FastAPI:
    app_config = config or AppConfig()
    bus = event_bus or FrontendEventBus()
    factory = session_manager_factory or _default_session_manager_factory
    manager = session_manager or factory(app_config, bus)

    def _live_manager() -> ProcessPoolSessionManager:
        # /api/models/load 通道热改会整体换掉 session 池；所有运行期引用
        # 必须走 app.state，闭包缓存初始实例会在热改后指向旧池
        return app.state.session_manager
    multicast_runtime = MulticastRuntimeRegistry(
        manager,
        max_jobs=app_config.max_concurrent_sessions,
        persist_path=app_config.output_dir / "multicast_jobs.json",
    )
    summary_cfg = summary_config or load_summary_config(app_config.summary_config_path)
    sensitive_filter = SensitiveWordFilter.from_file(app_config.sensitive_words_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.broadcast_task = asyncio.create_task(bus.broadcast_loop())
        try:
            await _live_manager().load_models()
            # 重启后自动拉起上次在跑的组播任务（清单由 registry 持久化维护）
            await multicast_runtime.restore_jobs(
                local_ip="0.0.0.0",
                packet_bytes=1280,
                use_full_packet=False,
                heartbeat_interval=5.0,
                speexdsp_lib_path=app_config.speexdsp_lib_path,
            )
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
            await _live_manager().unload_models()
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

    @app.post("/api/models/validate")
    async def validate_models(request: LoadModelsRequest | None = None) -> dict[str, Any]:
        request = request or LoadModelsRequest()
        try:
            next_config = _config_from_load_request(app.state.config, request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        spec = next_config.models.offline_model
        try:
            invalid = validate_offline_model_files(spec)
        except OSError:
            # 模型目录本身不存在/不可达：目录及全部必需文件记为缺失
            invalid = [spec.model_dir / name for name in spec.required_filenames]
        return {
            "valid": not invalid,
            "lanes": next_config.lanes,
            "offline_model_id": next_config.models.offline_model_id,
            "model_dir": str(next_config.models.offline_model_dir),
            "missing_files": [str(path) for path in invalid],
        }

    @app.post("/api/models/load")
    async def load_models(request: LoadModelsRequest | None = None) -> dict[str, Any]:
        """按请求重建 session 池（通道热改）。配置未变时仅重新加载模型。"""
        request = request or LoadModelsRequest()
        current: AppConfig = app.state.config
        try:
            next_config = _config_from_load_request(current, request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if next_config != current:
            manager = _live_manager()
            if manager.state is not PipelineState.UNLOADED:
                raise HTTPException(
                    status_code=409,
                    detail="unload models before changing lanes or offline model",
                )
            if multicast_runtime.list_jobs():
                raise HTTPException(
                    status_code=409,
                    detail="stop multicast jobs before changing lanes or offline model",
                )
            next_manager = factory(next_config, bus)
            try:
                await next_manager.load_models()
            except Exception:
                if next_manager.state is not PipelineState.UNLOADED:
                    with suppress(Exception):
                        await next_manager.unload_models()
                raise
            app.state.config = next_config
            app.state.session_manager = next_manager
            multicast_runtime.set_session_manager(next_manager)
        else:
            await _live_manager().load_models()
        return _status_payload(app)

    @app.post("/api/models/unload")
    async def unload_models() -> dict[str, Any]:
        manager = _live_manager()
        if manager.active_sessions:
            raise HTTPException(
                status_code=409,
                detail="stop active sessions before unloading models",
            )
        if multicast_runtime.list_jobs():
            raise HTTPException(
                status_code=409,
                detail="stop multicast jobs before unloading models",
            )
        await manager.unload_models()
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
                _live_manager().set_hotwords(words)
                note = "精修热词即时生效；在线 Zipformer 热词需重启（默认关闭）"
            else:
                sensitive_filter.update_words(words)
                note = "敏感词即时生效（含进行中的会话）"
        except Exception:
            # 热更新失败：回滚文件与内存
            write_word_list(path, old_words, header=_WORD_FILE_HEADERS[kind])
            if kind == "hotwords":
                _live_manager().set_hotwords(old_words)
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
        # sherpa 遇到 tokens.txt 里不存在的 token 会在 C++ 层直接 abort（杀掉
        # lane 进程），写文件前先校验，非法词表直接 400 拒绝。
        invalid = find_invalid_keyword_tokens(
            lines, Path(app_config.kws.model_dir) / "tokens.txt"
        )
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=(
                    "唤醒词含 KWS 词表外的 token（会崩溃识别进程，已拒绝）："
                    + "; ".join(f"{line!r} 缺 {missing}" for line, missing in invalid.items())
                ),
            )
        old_text = path.read_text(encoding="utf-8") if path.is_file() else None
        backup = _backup_word_file(path)
        # 注意：sherpa KWS 原生解析器不支持 # 注释行，keywords 文件不写注释头
        write_word_list(path, lines)
        try:
            await asyncio.to_thread(_live_manager().reload_kws_keywords)
        except Exception:
            if old_text is not None:
                path.write_text(old_text, encoding="utf-8")
                with suppress(Exception):
                    await asyncio.to_thread(_live_manager().reload_kws_keywords)
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
            session_manager=_live_manager(),
            summary_config=summary_cfg,
            output_dir=app_config.output_dir,
            sample_rate=sample_rate,
            wakeup_enabled=wakeup_enabled,
            sensitive_filter=sensitive_filter,
            wakeup_settings=wakeup_settings,
        )
        await handler.run()

    def _validate_multicast_target(group: str, port: int) -> None:
        try:
            addr = ipaddress.IPv4Address(group)
        except ipaddress.AddressValueError:
            raise HTTPException(status_code=400, detail=f"非法组播地址: {group!r}") from None
        if not addr.is_multicast:
            raise HTTPException(
                status_code=400, detail=f"{group} 不是组播地址（需在 224.0.0.0/4 内）"
            )
        if not (1 <= port <= 65535):
            raise HTTPException(status_code=400, detail=f"非法端口: {port}")

    @app.get("/v1/multicast")
    async def list_multicast() -> dict[str, Any]:
        return {
            "jobs": multicast_runtime.list_jobs(),
            "max_jobs": multicast_runtime.max_jobs,
        }

    @app.post("/v1/multicast/start")
    async def start_multicast(request: MulticastStartRequest) -> dict[str, Any]:
        """新增一路组播转写：自动从会话池分配一路 lane，无需挂 WS。"""
        _validate_multicast_target(request.group, request.port)
        try:
            job = await multicast_runtime.start_job(
                group=request.group,
                port=request.port,
                local_ip=request.local_ip,
                packet_bytes=request.packet_bytes,
                use_full_packet=request.use_full_packet,
                heartbeat_interval=request.heartbeat_interval,
                speexdsp_lib_path=app_config.speexdsp_lib_path,
                listener=None,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "ok": True,
            "group": request.group,
            "port": request.port,
            "lane_id": job.session.lane_id,
            "note": "收听转写结果：WS 连 /ws/multicast?group=...&port=...",
        }

    @app.post("/v1/multicast/stop")
    async def stop_multicast(request: MulticastStopRequest) -> dict[str, Any]:
        job = await multicast_runtime.stop_job(request.group, request.port)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"组播任务不存在: {request.group}:{request.port}",
            )
        return {"ok": True, "group": request.group, "port": request.port}

    @app.post("/v1/multicast/modify")
    async def modify_multicast(request: MulticastModifyRequest) -> dict[str, Any]:
        """修改已有任务的组播地址：沿用旧任务接收参数，停旧开新。"""
        new_port = request.new_port if request.new_port is not None else request.old_port
        _validate_multicast_target(request.new_group, new_port)
        try:
            job = await multicast_runtime.modify_job(
                old_group=request.old_group,
                old_port=request.old_port,
                new_group=request.new_group,
                new_port=new_port,
                speexdsp_lib_path=app_config.speexdsp_lib_path,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"组播任务不存在: {request.old_group}:{request.old_port}",
            )
        return {
            "ok": True,
            "old_group": request.old_group,
            "old_port": request.old_port,
            "group": request.new_group,
            "port": new_port,
            "lane_id": job.session.lane_id,
            "note": "原收听连接已断开，请用新地址重新收听",
        }

    # -- 声纹库管理（注册后热重载到各 lane，无需重启） ------------------------

    def _speaker_registry_dir() -> Path:
        return app_config.speaker.registry_dir or (
            app_config.output_dir / "voiceprints"
        )

    def _speaker_registry() -> VoiceprintRegistry:
        registry = getattr(app.state, "speaker_registry", None)
        if registry is None:
            registry = VoiceprintRegistry(_speaker_registry_dir())
            app.state.speaker_registry = registry
        return registry

    async def _speaker_embedder() -> SherpaSpeakerEmbedder:
        embedder = getattr(app.state, "speaker_embedder", None)
        if embedder is None:
            cfg = app_config.speaker
            embedder = await asyncio.to_thread(
                SherpaSpeakerEmbedder,
                model_path=cfg.model_path,
                num_threads=cfg.num_threads,
                provider=cfg.provider,
                sample_rate=cfg.sample_rate,
                window_seconds=cfg.window_seconds,
                hop_seconds=cfg.hop_seconds,
                trim_floor_ratio=cfg.trim_floor_ratio,
            )
            app.state.speaker_embedder = embedder
        return embedder

    def _broadcast_speaker_reload() -> None:
        reload_fn = getattr(_live_manager(), "reload_speaker_registry", None)
        if reload_fn is not None:
            reload_fn()

    def _require_speaker_enabled() -> None:
        if not app_config.speaker.enabled:
            raise HTTPException(status_code=503, detail="声纹识别未启用")

    @app.get("/v1/speakers")
    async def list_speakers() -> dict[str, Any]:
        registry = _speaker_registry()
        return {
            "enabled": app_config.speaker.enabled,
            "speakers": [
                {"name": n, "count": registry.count(n)} for n in registry.names()
            ],
        }

    @app.post("/v1/speakers")
    async def enroll_speaker(
        name: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        """注册声纹：multipart 上传 1~8 个 16bit PCM wav（任意采样率/声道）。"""
        _require_speaker_enabled()
        key = name.strip()
        if not key or len(key) > _MAX_SPEAKER_NAME_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"name 需为 1~{_MAX_SPEAKER_NAME_LEN} 个字符",
            )
        if not 1 <= len(files) <= _MAX_ENROLL_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"files 需为 1~{_MAX_ENROLL_FILES} 个 wav 文件",
            )
        embedder = await _speaker_embedder()
        if not getattr(embedder, "available", True):
            raise HTTPException(
                status_code=503,
                detail=f"声纹模型不可用: {getattr(embedder, 'last_error', None)}",
            )

        # 先全部提取完再入库，避免第 N 个文件失败导致半成品注册
        vectors: list[Any] = []
        for upload in files:
            data = await upload.read()
            try:
                pcm = wav_bytes_to_pcm16k(data)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"{upload.filename}: {exc}"
                ) from None
            vec = await asyncio.to_thread(embedder.embed, pcm)
            if vec is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"{upload.filename}: 音频过短或无声，无法提取声纹",
                )
            vectors.append(vec)

        registry = _speaker_registry()
        for vec in vectors:
            registry.enroll(key, vec)
        _broadcast_speaker_reload()
        return {
            "ok": True,
            "name": key,
            "enrolled": len(vectors),
            "total": registry.count(key),
        }

    @app.delete("/v1/speakers/{name}")
    async def delete_speaker(name: str) -> dict[str, Any]:
        _require_speaker_enabled()
        registry = _speaker_registry()
        if not registry.remove(name):
            raise HTTPException(status_code=404, detail=f"声纹不存在: {name}")
        _broadcast_speaker_reload()
        return {"ok": True, "name": name}

    @app.websocket("/ws/multicast")
    async def websocket_multicast(websocket: WebSocket) -> None:
        params = websocket.query_params
        handler = MulticastHandler(
            websocket=websocket,
            session_manager=_live_manager(),
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

    # 2026-10-28 起暂时停用本机麦克风入口（多设备抢板载 mic 会互相干扰）；
    # 恢复时取消下面这段的注释即可。
    # @app.websocket("/ws/local_mic")
    # async def websocket_local_mic(websocket: WebSocket) -> None:
    #     params = websocket.query_params
    #     try:
    #         sample_rate = int(params.get("sample_rate", "16000"))
    #     except ValueError:
    #         sample_rate = 16000
    #     if sample_rate <= 0:
    #         sample_rate = 16000
    #     device_raw = params.get("device")
    #     if device_raw is None:
    #         device: int | str | None = None
    #     else:
    #         try:
    #             device = int(device_raw)
    #         except ValueError:
    #             device = device_raw
    #     try:
    #         chunk = int(params.get("chunk", "512"))
    #     except ValueError:
    #         chunk = 512
    #     if chunk <= 0:
    #         chunk = 512
    #     wakeup_enabled = params.get("wakeup", "true").lower() not in ("0", "false", "no")
    #     handler = LocalMicHandler(
    #         websocket=websocket,
    #         session_manager=manager,
    #         summary_config=summary_cfg,
    #         output_dir=app_config.output_dir,
    #         device=device,
    #         sample_rate=sample_rate,
    #         chunk=chunk,
    #         wakeup_enabled=wakeup_enabled,
    #         sensitive_filter=sensitive_filter,
    #         wakeup_settings=wakeup_settings,
    #     )
    #     await handler.run()
    _ = LocalMicHandler  # 保留 import，路由恢复时直接用

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        file: UploadFile,
        response_format: str = Form("json"),
        enable_summary: bool = Form(False),
    ) -> Response:
        if _live_manager().state is not PipelineState.LOADED:
            raise HTTPException(
                status_code=503,
                detail="service is not loaded",
            )
        return await handle_offline_transcription(
            config=app.state.config,
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
