#!/usr/bin/env python3
"""声纹阈值标定：算同一人 / 不同人的余弦分布，给出可用阈值。

这是上线前**不可跳过**的一步。默认阈值（match 0.55 / new 0.40）是按干净近场
给的；全向麦 + 会议室混响下同人余弦会掉到 0.4 附近、异人会升到 0.3+，区间一
重叠，任何单阈值都会错分。

用法（在项目根目录、``PYTHONPATH=src``）::

    # 显式分组：每个说话人给若干段
    python scripts/speaker_calibrate.py --model M.onnx \
        --speaker 张三=z1.wav,z2.wav,z3.wav \
        --speaker 李四=l1.wav,l2.wav

    # 快速体检：每个 wav 切成多段，段间算类内、文件间算类间
    python scripts/speaker_calibrate.py --model M.onnx \
        --auto-split-dir src/multi_stream_asr/assets/wav16k --split-seconds 3

只看结论：
* 间隔 > 0.20 → 单阈值可分开
* 间隔 0~0.20 → 必须双门限 + 多窗，且二遍重聚类不可关
* 间隔 < 0    → 重叠，纯软件做不好：换麦阵，或产品降级为人工标人

**注意**：``--auto-split-dir`` 只是快速体检，它假设"一个文件 = 一个人"。真实
录音（新闻/直播/会议）经常一个文件里就换了人，那样标出来的类内会被严重低估。
正式标定请用 ``--speaker 姓名=a.wav,b.wav`` 分组，每组确认是同一个人。
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

from multi_stream_asr.config import DEFAULT_SPEAKER_MODEL_PATH, SpeakerConfig
from multi_stream_asr.speaker import SherpaSpeakerEmbedder, verify_speaker_model

SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray:
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


def parse_speaker_arg(raw: str) -> tuple[str, list[Path]]:
    name, _, files = raw.partition("=")
    name = name.strip()
    paths = [Path(item.strip()) for item in files.split(",") if item.strip()]
    if not name or not paths:
        raise argparse.ArgumentTypeError(f"格式应为 姓名=a.wav,b.wav，收到: {raw}")
    return name, paths


def split_file(path: Path, split_seconds: float, per_file: int) -> list[np.ndarray]:
    """从整个文件里**均匀**取样若干段。

    不要取相邻片段：同一个人连着说同一话题时余弦会明显偏高，那样标出来的
    类内分布过于乐观，上线后换个话题就判成两个人。
    """
    audio = load_wav(path)
    window = int(split_seconds * SAMPLE_RATE)
    if audio.size < int(1.0 * SAMPLE_RATE):
        return []
    if audio.size <= window:
        return [audio]
    if per_file <= 1:
        return [audio[:window]]
    span = audio.size - window
    starts = sorted({int(round(span * index / (per_file - 1))) for index in range(per_file)})
    return [audio[start : start + window] for start in starts]


def build_embeddings(
    embedder: SherpaSpeakerEmbedder,
    groups: dict[str, list[np.ndarray]],
) -> tuple[dict[str, list[np.ndarray]], list[str]]:
    vectors: dict[str, list[np.ndarray]] = {}
    notes: list[str] = []
    for name, chunks in groups.items():
        kept: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            vector = embedder.embed(chunk)
            if vector is None:
                notes.append(f"{name} 第 {index + 1} 段提不出声纹（{len(chunk) / SAMPLE_RATE:.2f}s），已跳过")
                continue
            kept.append(vector)
        if kept:
            vectors[name] = kept
        else:
            notes.append(f"{name} 全部音频都提不出声纹，已整体跳过")
    return vectors, notes


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def describe(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="声纹余弦分布标定",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_SPEAKER_MODEL_PATH)
    parser.add_argument(
        "--speaker",
        action="append",
        type=parse_speaker_arg,
        default=None,
        metavar="NAME=WAV[,WAV...]",
        help="显式分组，可重复",
    )
    parser.add_argument("--auto-split-dir", type=Path, default=None)
    parser.add_argument("--split-seconds", type=float, default=3.0)
    parser.add_argument("--per-file", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None, help="把原始余弦写到文件")
    args = parser.parse_args(argv)

    verdict = verify_speaker_model(args.model)
    if not verdict.ok:
        print(f"模型预检失败: {verdict.detail}", file=sys.stderr)
        print(
            "常见原因：onnx 缺少 model_type 元数据（原生 3D-Speaker 导出包都没有）。"
            "\n先跑：python scripts/patch_speaker_model.py --input <原始> --output <修好的>",
            file=sys.stderr,
        )
        return 2

    groups: dict[str, list[np.ndarray]] = {}
    if args.speaker:
        for name, paths in args.speaker:
            chunks = groups.setdefault(name, [])
            for path in paths:
                if not path.is_file():
                    print(f"跳过不存在的文件: {path}", file=sys.stderr)
                    continue
                chunks.append(load_wav(path))
    if args.auto_split_dir is not None:
        wavs = sorted(args.auto_split_dir.glob("*.wav"))
        if not wavs:
            print(f"{args.auto_split_dir} 下没有 wav", file=sys.stderr)
            return 2
        print(
            "提示: --auto-split-dir 假设「一个文件 = 一个人」。真实录音（新闻/直播/"
            "会议）经常中途换人，会让类内被低估。正式标定请用 --speaker 分组。",
            file=sys.stderr,
        )
        for path in wavs:
            chunks = split_file(path, args.split_seconds, args.per_file)
            if not chunks:
                print(f"跳过过短的音频: {path.name}", file=sys.stderr)
                continue
            # 同一个文件的多段算"同一个人"，跨文件才算"不同人"
            groups.setdefault(path.stem, []).extend(chunks)
    if not groups:
        parser.print_help()
        return 1

    config = SpeakerConfig(enabled=True, model_path=args.model)
    embedder = SherpaSpeakerEmbedder(
        model_path=args.model,
        num_threads=config.num_threads,
        provider=config.provider,
        sample_rate=config.sample_rate,
        window_seconds=config.window_seconds,
        hop_seconds=config.hop_seconds,
        trim_floor_ratio=config.trim_floor_ratio,
    )
    vectors, notes = build_embeddings(embedder, groups)
    for note in notes:
        print(f"  ! {note}", file=sys.stderr)
    if len(vectors) < 2:
        print("至少需要两组能提出声纹的音频", file=sys.stderr)
        return 2

    intra: list[float] = []
    intra_pairs: list[tuple[str, float]] = []
    for name, items in vectors.items():
        for a, b in itertools.combinations(items, 2):
            score = cosine(a, b)
            intra.append(score)
            intra_pairs.append((name, score))
    inter: list[float] = []
    inter_pairs: list[tuple[str, str, float]] = []
    for name_a, name_b in itertools.combinations(vectors, 2):
        for a in vectors[name_a]:
            for b in vectors[name_b]:
                score = cosine(a, b)
                inter.append(score)
                inter_pairs.append((name_a, name_b, score))

    dim = len(next(iter(vectors.values()))[0])
    total = sum(len(items) for items in vectors.values())
    print(f"\n模型: {args.model}")
    print(f"预检: OK   embedding 维度={dim}")
    print(f"分组: {len(vectors)} 组 / {total} 条")
    print()
    if intra:
        stats = describe(intra)
        print(
            f"类内（同一人） n={stats['n']:3d}  "
            f"min={stats['min']:.3f}  mean={stats['mean']:.3f}  max={stats['max']:.3f}"
        )
    stats_inter = describe(inter)
    print(
        f"类间（不同人） n={stats_inter['n']:3d}  "
        f"min={stats_inter['min']:.3f}  mean={stats_inter['mean']:.3f}  "
        f"max={stats_inter['max']:.3f}"
    )

    suggestion: dict[str, object] = {
        "dim": dim,
        "intra": intra,
        "inter": inter,
    }

    if not intra:
        print("\n只测到 1 组，无法评估间隔。给同一人 2 段以上再跑一次。")
        if args.json:
            args.json.write_text(json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    gap = min(intra) - max(inter)
    print(f"\n间隔 = min_intra - max_inter = {gap:+.3f}")
    print("\n最容易被混的「不同人」组合（Top 5）:")
    for name_a, name_b, score in sorted(inter_pairs, key=lambda item: -item[2])[:5]:
        print(f"  {score:.3f}  {name_a}  <->  {name_b}")
    print("\n最不像的「同一人」组合（Bottom 5）:")
    for name, score in sorted(intra_pairs, key=lambda item: item[1])[:5]:
        print(f"  {score:.3f}  {name}")
    if gap > 0.20:
        print("结论: 单阈值可分开，可以直接上。")
    elif gap >= 0.0:
        print("结论: 区间勉强分开，但余量小。必须双门限 + 多窗，二遍重聚类不能关。")
    else:
        print("结论: 区间重叠，任何单阈值都会错分。")
        print("      纯软件做不好 —— 换麦克风阵列，或把产品降级成人工标人。")

    match_threshold = (max(inter) + min(intra)) / 2
    new_threshold = max(0.0, max(inter) + (match_threshold - max(inter)) * 0.5)
    identify = min(0.95, max(match_threshold, min(intra) - 0.05))
    print()
    print("建议（写进 SpeakerConfig 或按需改默认值）:")
    print(f"  match_threshold       = {match_threshold:.3f}")
    print(f"  new_speaker_threshold = {new_threshold:.3f}")
    print(f"  identify_threshold    = {identify:.3f}")
    print(f"  merge_threshold       = {max(match_threshold, 0.6):.3f}")

    if args.json:
        suggestion.update(
            {
                "gap": gap,
                "match_threshold": match_threshold,
                "new_speaker_threshold": new_threshold,
                "identify_threshold": identify,
            }
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n原始数据已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
