#!/usr/bin/env python3
"""声纹库管理：注册、列出、删除、试识别。

用法（在项目根目录执行，需 PYTHONPATH=src）::

    PYTHONPATH=src python scripts/enroll_speaker.py --list
    PYTHONPATH=src python scripts/enroll_speaker.py --name 张三 --wav zhang1.wav zhang2.wav
    PYTHONPATH=src python scripts/enroll_speaker.py --check probe.wav
    PYTHONPATH=src python scripts/enroll_speaker.py --remove 张三

默认声纹库是 ``output/voiceprints``，与服务端的 ``<output>/voiceprints`` 一致。
每次注册建议给 3 段以上、每段 2~5 秒的干净语音；远场/混响重的声音不要拿来
注册，会把质心带偏。
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

from multi_stream_asr.config import DEFAULT_SPEAKER_MODEL_PATH, SpeakerConfig
from multi_stream_asr.speaker import SherpaSpeakerEmbedder, VoiceprintRegistry

SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray:
    """读 wav 为 16k 单声道 float32；非 16k 用线性插值重采样。"""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        axis = np.arange(audio.size)
        target = np.linspace(0, audio.size - 1, int(audio.size * SAMPLE_RATE / rate))
        audio = np.interp(target, axis, audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def build_embedder(model_path: Path) -> SherpaSpeakerEmbedder:
    config = SpeakerConfig(enabled=True, model_path=model_path)
    embedder = SherpaSpeakerEmbedder(
        model_path=model_path,
        num_threads=config.num_threads,
        provider=config.provider,
        sample_rate=config.sample_rate,
        window_seconds=config.window_seconds,
        hop_seconds=config.hop_seconds,
        trim_floor_ratio=config.trim_floor_ratio,
    )
    if not embedder.available:
        raise SystemExit(
            f"声纹模型不可用: {model_path}\n"
            "请先下载 CAM++ onnx（见 README「说话人分离与声纹识别」一节）"
        )
    return embedder


def embed_file(embedder: SherpaSpeakerEmbedder, path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"音频文件不存在: {path}")
    vector = embedder.embed(load_wav(path))
    if vector is None:
        raise SystemExit(
            f"无法从 {path} 提取声纹（时长不足或全是静音，建议 2~5 秒有效语音）"
        )
    return vector


def cmd_list(registry: VoiceprintRegistry) -> int:
    names = registry.names()
    if not names:
        print(f"声纹库为空: {registry.directory}")
        return 0
    print(f"声纹库: {registry.directory}")
    for name in names:
        print(f"  {name}  ({registry.count(name)} 条)")
    return 0


def cmd_enroll(
    registry: VoiceprintRegistry,
    embedder: SherpaSpeakerEmbedder,
    name: str,
    wavs: list[Path],
) -> int:
    for path in wavs:
        vector = embed_file(embedder, path)
        count = registry.enroll(name, vector)
        print(f"  已注册 {path.name} -> {name}（共 {count} 条）")
    print(f"完成: {name} 现有 {registry.count(name)} 条声纹")
    return 0


def cmd_remove(registry: VoiceprintRegistry, name: str) -> int:
    if registry.remove(name):
        print(f"已删除: {name}")
        return 0
    print(f"声纹库里没有: {name}")
    return 1


def cmd_check(
    registry: VoiceprintRegistry,
    embedder: SherpaSpeakerEmbedder,
    path: Path,
) -> int:
    vector = embed_file(embedder, path)
    config = SpeakerConfig()
    scored = registry.score_all(vector)
    match = registry.identify(
        vector,
        threshold=config.identify_threshold,
        margin=config.identify_margin,
    )

    if not scored:
        print(f"声纹库为空: {registry.directory}")
        return 1

    print(f"门限: threshold={config.identify_threshold:.2f} margin={config.identify_margin:.2f}")
    for name, score in scored[:3]:
        print(f"  {score:.4f}  {name}")
    if match is None:
        best_name, best_score = scored[0]
        reason = (
            "得分低于门限"
            if best_score < config.identify_threshold
            else "与第二名差距不足（可能是音色相近的两个人）"
        )
        print(f"未命中：最接近的是 {best_name}（{best_score:.4f}），{reason}")
        print("若同一人经常被判未命中，用 scripts/speaker_calibrate.py 重新标定后降低 threshold。")
        return 1
    print(f"命中: {match.name}  score={match.score:.4f}  margin={match.margin:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="声纹库管理（注册 / 列出 / 删除 / 试识别）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("output/voiceprints"),
        help="声纹库目录",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_SPEAKER_MODEL_PATH,
        help="声纹 embedding 模型（CAM++ onnx）",
    )
    parser.add_argument("--name", help="要注册的姓名")
    parser.add_argument(
        "--wav",
        type=Path,
        nargs="+",
        action="extend",
        default=None,
        help="该姓名的注册音频（可多个，建议 3 段以上）",
    )
    parser.add_argument("--check", type=Path, help="试识别一段音频")
    parser.add_argument("--remove", help="删除某个姓名")
    parser.add_argument("--list", action="store_true", help="列出声纹库")
    args = parser.parse_args(argv)

    registry = VoiceprintRegistry(args.registry)
    if args.list:
        return cmd_list(registry)
    if args.remove:
        return cmd_remove(registry, args.remove)
    if args.check is not None:
        return cmd_check(registry, build_embedder(args.model), args.check)
    if args.name and args.wav:
        return cmd_enroll(registry, build_embedder(args.model), args.name, args.wav)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
