from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

import uvicorn

from multi_stream_asr.config import AppConfig
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
    return parser


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
