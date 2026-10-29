from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pytest
import subprocess
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = REPO_ROOT / "scripts" / "setup.sh"


def test_setup_script_installs_models_from_verified_archives_and_recovers_downloads(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("setup.sh 是板端 Linux 脚本，Windows 无法执行")
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    _write_archives(archive_dir)
    models_dir = tmp_path / "models" / "asr"
    environment = {
        **os.environ,
        "ASR_MODEL_BASE_URL": archive_dir.as_uri(),
    }
    command = [
        "bash",
        str(SETUP_SCRIPT),
        "--skip-apt",
        "--skip-python",
        "--models-dir",
        str(models_dir),
    ]

    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)

    assert (models_dir / "qwen3asr" / "Qwen3-ASR-0.6B-text-q40.gguf").is_file()
    assert (
        models_dir / "qwen3asr-1.7b" / "Qwen3-ASR-1.7B-text-q4km.gguf"
    ).is_file()
    assert (
        models_dir
        / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        / "encoder-epoch-99-avg-1.q.onnx"
    ).is_file()
    assert '"model_path": "Qwen3-ASR-0.6B-text-q40.gguf"' in (
        models_dir / "qwen3asr" / "config.json"
    ).read_text(encoding="utf-8")
    assert "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1" in (
        models_dir
        / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        / "zipformer.config"
    ).read_text(encoding="utf-8")

    download = models_dir.parent / ".downloads" / "zipformer-streaming.tar.gz"
    initial_mtime = download.stat().st_mtime_ns
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)
    assert download.stat().st_mtime_ns == initial_mtime

    download.write_bytes(b"corrupted")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)
    assert hashlib.md5(download.read_bytes()).hexdigest() == _archive_md5(
        archive_dir / "zipformer-streaming.tar.gz"
    )


def _write_archives(archive_dir: Path) -> None:
    _write_archive(
        archive_dir / "qwen3-asr-0.6B-dynq-q40.tar.gz",
        "upstream-0.6b",
        (
            "Qwen3-ASR-0.6B-text-q40.gguf",
            "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
            "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
        ),
    )
    _write_archive(
        archive_dir / "qwen3-asr-1.7B-dynq-q4km.tar.gz",
        "upstream-1.7b",
        (
            "Qwen3-ASR-1.7B-text-q4km.gguf",
            "Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx",
            "Qwen3-ASR-1.7B-encoder-backend.dynq.onnx",
        ),
    )
    _write_archive(
        archive_dir / "zipformer-streaming.tar.gz",
        "upstream-zipformer",
        (
            "tokens.txt",
            "encoder-epoch-99-avg-1.q.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.onnx",
        ),
    )


def _write_archive(archive: Path, directory: str, filenames: tuple[str, ...]) -> None:
    staging = archive.parent / directory
    staging.mkdir()
    for filename in filenames:
        (staging / filename).write_text(filename, encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname=directory)
    archive.with_name(f"{archive.name}.md5").write_text(
        f"{_archive_md5(archive)}  {archive.name}\n",
        encoding="utf-8",
    )


def _archive_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()
