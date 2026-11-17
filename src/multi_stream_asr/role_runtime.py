"""OpenStore 换模对接：current-roles.json 读取与运行时模型路径解析。

交接链路：OpenStore 写 /app/lg-store/database/models/current-roles.json
→ 应用启动时按其中的 runtimePath 加载模型；用户切换后 OpenStore 前端
调 POST /reload → 服务重启，启动逻辑读本文件选出的新模型生效。

本应用消费的 role：
  streaming-asr → ModelConfig.online_model_dir（zipformer 流式）
  qwen3-asr     → ModelConfig.offline_model_dir（Qwen3-ASR 校正）
  vad           → AppConfig.vad_model_dir
  kws           → KWSConfig.model_dir
  offline-asr   → AppConfig.offline_asr_model_dir（sensevoice 非流式）
  speaker       → SpeakerConfig.model_path（声纹 embedding onnx 文件）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

from multi_stream_asr.config import AppConfig

_logger = logging.getLogger(__name__)

DEFAULT_ROLES_PATH = Path("/app/lg-store/database/models/current-roles.json")

ROLE_STREAMING_ASR = "streaming-asr"
ROLE_QWEN3_ASR = "qwen3-asr"
ROLE_VAD = "vad"
ROLE_KWS = "kws"
ROLE_OFFLINE_ASR = "offline-asr"
ROLE_SPEAKER = "speaker"
ROLE_LLM_SUMMARY = "llm-summary"

_MODEL_SUFFIXES = (".gguf", ".onnx")


def load_current_roles(path: Path | None = None) -> dict[str, str]:
    """返回 role -> runtimePath；文件缺失/损坏返回 {}（不阻断启动）。"""
    path = path or DEFAULT_ROLES_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        roles = raw.get("roles", {})
        if not isinstance(roles, dict):
            return {}
        result: dict[str, str] = {}
        for role, info in roles.items():
            if isinstance(info, dict) and info.get("runtimePath"):
                result[role] = str(info["runtimePath"])
        return result
    except (OSError, ValueError, AttributeError) as exc:
        _logger.warning("current-roles.json 读取失败，沿用内置路径: %s", exc)
        return {}


def resolve_runtime_model_dir(runtime_path: str | Path) -> Path | None:
    """定位 runtimePath 下的模型目录：优先本身，否则第一个含模型文件的子目录。

    lg-store 的 runtime 布局是 ``runtime/<role>/<version>/<模型目录>``，
    多一层版本目录，这里把这一层剥掉。
    """
    base = Path(runtime_path)
    if not base.is_dir():
        return None
    candidates = [base, *sorted(p for p in base.iterdir() if p.is_dir())]
    for candidate in candidates:
        try:
            if any(
                p.is_file()
                and (
                    p.suffix.lower() in _MODEL_SUFFIXES or p.name == "bpe.model"
                )
                for p in candidate.iterdir()
            ):
                return candidate
        except OSError:
            continue
    return None


def resolve_role_gguf(role: str) -> str | None:
    """解析 role 指向的 gguf 文件路径（llm 类 role 用），解析失败返回 None。"""
    runtime_path = load_current_roles().get(role)
    if not runtime_path:
        return None
    model_dir = resolve_runtime_model_dir(runtime_path)
    if model_dir is None:
        _logger.warning("role %s runtimePath 无模型文件: %s", role, runtime_path)
        return None
    gguf = next(iter(sorted(model_dir.glob("*.gguf"))), None)
    return str(gguf) if gguf is not None else None


def apply_role_overrides(config: AppConfig) -> AppConfig:
    """启动时按 current-roles.json 覆盖模型目录；解析失败的角色保持原路径。"""
    if os.getenv("MODEL_ROLES_DISABLE", "").strip():
        return config
    roles = load_current_roles()
    if not roles:
        return config

    models = config.models
    vad_model_dir = config.vad_model_dir
    offline_asr_model_dir = config.offline_asr_model_dir
    kws = config.kws
    speaker = config.speaker

    def pick(role: str) -> Path | None:
        runtime_path = roles.get(role)
        if not runtime_path:
            return None
        model_dir = resolve_runtime_model_dir(runtime_path)
        if model_dir is None:
            _logger.warning("role %s runtimePath 无模型文件: %s", role, runtime_path)
            return None
        _logger.info("role %s -> %s", role, model_dir)
        return model_dir

    updates = 0
    model_dir = pick(ROLE_STREAMING_ASR)
    if model_dir is not None and model_dir != models.online_model_dir:
        models = replace(models, online_model_dir=model_dir)
        updates += 1
    model_dir = pick(ROLE_QWEN3_ASR)
    if model_dir is not None and model_dir != models.offline_model_dir:
        models = replace(models, offline_model_dir=model_dir)
        updates += 1
    model_dir = pick(ROLE_VAD)
    if model_dir is not None and model_dir != vad_model_dir:
        vad_model_dir = model_dir
        updates += 1
    model_dir = pick(ROLE_OFFLINE_ASR)
    if model_dir is not None and model_dir != offline_asr_model_dir:
        offline_asr_model_dir = model_dir
        updates += 1
    model_dir = pick(ROLE_KWS)
    if model_dir is not None and model_dir != kws.model_dir:
        kws = replace(kws, model_dir=model_dir)
        updates += 1
    # speaker 是模型文件路径（onnx），不是目录：取解析目录里的 onnx 文件
    model_dir = pick(ROLE_SPEAKER)
    if model_dir is not None:
        onnx = next(iter(sorted(model_dir.glob("*.onnx"))), None)
        if onnx is not None and onnx != speaker.model_path:
            speaker = replace(speaker, model_path=onnx)
            updates += 1

    if updates == 0:
        return config
    return replace(
        config,
        models=models,
        vad_model_dir=vad_model_dir,
        offline_asr_model_dir=offline_asr_model_dir,
        kws=kws,
        speaker=speaker,
    )
