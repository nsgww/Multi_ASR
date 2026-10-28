from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import uvicorn

from multi_stream_asr.config import AppConfig, SpeakerConfig
from multi_stream_asr.headless.runner import HeadlessRunner
from multi_stream_asr.web.server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multi-stream-asr")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--input-dir", type=Path, default=AppConfig().input_dir)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # 说话人分离 / 声纹识别（默认关闭，关闭时行为与原版逐字节一致）
    parser.add_argument(
        "--speaker",
        action="store_true",
        help="启用说话人分离与声纹识别",
    )
    parser.add_argument(
        "--speaker-model",
        type=Path,
        default=None,
        help="声纹 embedding 模型（CAM++ onnx）；给了路径即自动启用",
    )
    parser.add_argument(
        "--speaker-registry",
        type=Path,
        default=None,
        help="声纹库目录；默认 <output>/voiceprints；给了路径即自动启用",
    )
    return parser


def build_speaker_config(args: argparse.Namespace) -> SpeakerConfig:
    """把 CLI 参数折叠成 SpeakerConfig。

    ``--speaker-model`` / ``--speaker-registry`` 单独出现时也自动启用，
    避免"我设了路径但没生效"这种坑。
    """
    enabled = bool(
        args.speaker
        or args.speaker_model is not None
        or args.speaker_registry is not None
    )
    config = SpeakerConfig(enabled=enabled)
    if args.speaker_model is not None:
        config = replace(config, model_path=Path(args.speaker_model).expanduser())
    if args.speaker_registry is not None:
        config = replace(
            config, registry_dir=Path(args.speaker_registry).expanduser()
        )
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fake and not args.headless:
        parser.error("--fake requires --headless")
    args.input_dir = Path(args.input_dir).expanduser().resolve()
    return args


async def _run_async(args: argparse.Namespace) -> int:
    config = AppConfig(
        lanes=args.lanes,
        input_dir=args.input_dir,
        verbose=args.verbose,
        speaker=build_speaker_config(args),
    )
    runner = HeadlessRunner(config=config, fake=args.fake)
    return await runner.run()


def _run_web(args: argparse.Namespace) -> int:
    config = AppConfig(
        lanes=args.lanes,
        input_dir=args.input_dir,
        verbose=args.verbose,
        web_port=args.port,
        max_concurrent_sessions=args.lanes,
        speaker=build_speaker_config(args),
    )
    app = create_app(config=config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.headless:
        return asyncio.run(_run_async(args))
    return _run_web(args)


if __name__ == "__main__":
    raise SystemExit(main())
