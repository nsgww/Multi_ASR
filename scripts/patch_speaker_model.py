#!/usr/bin/env python3
"""给 3D-Speaker / WeSpeaker 的 onnx 补 sherpa-onnx 要求的元数据。

背景：从 ModelScope / 3D-Speaker 仓库直接导出的 onnx **没有** ``model_type``
元数据，sherpa-onnx 2.x 的 ``SpeakerEmbeddingExtractor`` 读不到会直接报
``No model_type in the metadata``，而且其 C++ 实现随后会**段错误把整个进程崩掉**
（不是抛异常）。所以模型必须先补元数据再上线。

用法（需要 ``pip install onnx``，一次性工具，不是运行时依赖）::

    python scripts/patch_speaker_model.py \
        --input E:/model/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx \
        --output E:/model/patched/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_METADATA = {
    "model_type": "3d-speaker",
    "framework": "3d-speaker",
    "version": "1",
    "sample_rate": "16000",
    "language": "Chinese",
    "feat_dim": "80",
    "n_mels": "80",
    "num_mels": "80",
    "normalize_samples": "1",
    "dither": "0",
    "window_type": "hamming",
}


def _static_last_dim(value_info) -> int | None:  # noqa: ANN001
    try:
        dims = value_info.type.tensor_type.shape.dim
    except Exception:  # noqa: BLE001
        return None
    if not dims:
        return None
    last = dims[-1]
    return int(last.dim_value) if last.HasField("dim_value") else None


def patch(
    source: Path,
    target: Path,
    *,
    model_type: str,
    overwrite_existing: bool,
    output_dim: int | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    import onnx

    model = onnx.load(str(source), load_external_data=False)
    existing = {prop.key: prop.value for prop in model.metadata_props}

    wanted = dict(DEFAULT_METADATA)
    wanted["model_type"] = model_type
    if extra:
        wanted.update(extra)

    # 维度从图里读，别手写：不同导出版本可能不是 80/192
    feat_dim = _static_last_dim(model.graph.input[0]) if model.graph.input else None
    out_dim = _static_last_dim(model.graph.output[0]) if model.graph.output else None
    if output_dim is not None:
        out_dim = int(output_dim)
    if feat_dim:
        wanted["feat_dim"] = str(feat_dim)
    if out_dim:
        wanted["output_dim"] = str(out_dim)
    elif "output_dim" not in wanted:
        raise SystemExit(
            "无法从图里推断 output_dim（输出维度是符号维度），"
            "请用 --output-dim 显式指定"
        )

    for key, value in wanted.items():
        if key in existing and not overwrite_existing:
            continue
        for prop in model.metadata_props:
            if prop.key == key:
                prop.value = value
                break
        else:
            model.metadata_props.append(
                onnx.StringStringEntryProto(key=key, value=value)
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))

    result = {prop.key: prop.value for prop in model.metadata_props}
    return {"before": existing, "after": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="给声纹 onnx 补 sherpa-onnx 需要的元数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="原始 onnx")
    parser.add_argument("--output", type=Path, required=True, help="输出 onnx")
    parser.add_argument(
        "--model-type",
        default="3d-speaker",
        help="sherpa-onnx 的 model_type；3D-Speaker 系列用 3d-speaker，"
        "WeSpeaker 系列用 wespeaker",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=None,
        help="embedding 维度；默认从图的输出形状推断",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="已存在的元数据键也覆盖",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="额外覆盖任意元数据键，可重复（如 --set normalize_samples=0）",
    )
    args = parser.parse_args(argv)

    extra: dict[str, str] = {}
    for item in args.set or ():
        key, _, value = item.partition("=")
        if not key.strip():
            raise SystemExit(f"--set 格式应为 KEY=VALUE，收到: {item}")
        extra[key.strip()] = value.strip()

    if not args.input.is_file():
        raise SystemExit(f"输入不存在: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("输出不能与输入相同，请另存一份")

    try:
        info = patch(
            args.input,
            args.output,
            model_type=args.model_type,
            overwrite_existing=args.overwrite_existing,
            output_dim=args.output_dim,
            extra=extra,
        )
    except ImportError:
        raise SystemExit("需要 onnx：pip install onnx") from None

    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"原有元数据: {info['before'] or '(空)'}")
    print("写入后:")
    for key, value in sorted(info["after"].items()):
        print(f"  {key} = {value}")
    print("\n下一步用标定脚本确认能加载且区分度正常：")
    print(
        "  python scripts/speaker_calibrate.py "
        f"--model {args.output} --auto-split-dir src/multi_stream_asr/assets/wav16k"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
