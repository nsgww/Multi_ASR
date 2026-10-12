#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="/var/lib/sm-sdk/models/asr"
MODEL_BASE_URL="${ASR_MODEL_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/asr}"
SKIP_APT=0
SKIP_PYTHON=0

usage() {
  cat <<'EOF'
Usage: scripts/setup.sh [--skip-apt] [--skip-python] [--models-dir PATH]

Install the Python 3.13 environment and ASR models. Set ASR_MODEL_BASE_URL to
use a mirror or a local file:// directory containing the archives and .md5 files.
EOF
}

while (($#)); do
  case "$1" in
    --skip-apt) SKIP_APT=1 ;;
    --skip-python) SKIP_PYTHON=1 ;;
    --models-dir)
      MODELS_DIR="$2"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if (( ! SKIP_APT )); then
  sudo apt-get update
  sudo apt-get install -y python3.13 python3.13-venv libportaudio2 ca-certificates tar
fi

if ! command -v python3.13 >/dev/null; then
  echo "python3.13 is required; rerun without --skip-apt or install it first" >&2
  exit 1
fi

PYTHON_BIN=python3.13
if (( ! SKIP_PYTHON )); then
  VENV_DIR="$PROJECT_ROOT/.venv313"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3.13 -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  export UV_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple"
  export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
  export PIP_INDEX_URL="https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple"
  export PIP_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple https://mirrors.aliyun.com/pypi/simple"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e ".[dev]" sounddevice
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: --no-deps sherpa-onnx \
    --index-url "https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple"
fi

MODELS_DIR="$("$PYTHON_BIN" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$MODELS_DIR")"
DOWNLOAD_DIR="$(dirname "$MODELS_DIR")/.downloads"
mkdir -p "$MODELS_DIR" "$DOWNLOAD_DIR"

md5_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.md5()
with pathlib.Path(sys.argv[1]).open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
}

fetch_text() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1]) as response:
    print(response.read().decode("utf-8"), end="")
PY
}

download_file() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import pathlib
import sys
import urllib.request

url, destination = sys.argv[1:]
path = pathlib.Path(destination)
with urllib.request.urlopen(url) as response, path.open("wb") as output:
    while True:
        block = response.read(1024 * 1024)
        if not block:
            break
        output.write(block)
PY
}

ensure_archive() {
  local archive_name="$1"
  local archive_url="$MODEL_BASE_URL/$archive_name"
  local archive_path="$DOWNLOAD_DIR/$archive_name"
  local expected_md5
  expected_md5="$(fetch_text "$archive_url.md5" | awk 'NR == 1 { print $1 }')"
  if [[ ! "$expected_md5" =~ ^[0-9a-fA-F]{32}$ ]]; then
    echo "invalid MD5 for $archive_url" >&2
    exit 1
  fi
  if [[ -f "$archive_path" ]] && [[ "$(md5_file "$archive_path")" == "$expected_md5" ]]; then
    echo "verified download exists: $archive_name"
    return
  fi
  rm -f "$archive_path" "$archive_path.part"
  echo "downloading: $archive_name"
  download_file "$archive_url" "$archive_path.part"
  if [[ "$(md5_file "$archive_path.part")" != "$expected_md5" ]]; then
    rm -f "$archive_path.part"
    echo "MD5 mismatch after downloading $archive_name" >&2
    exit 1
  fi
  mv "$archive_path.part" "$archive_path"
}

install_model() {
  local archive_name="$1"
  local target_name="$2"
  shift 2
  local target="$MODELS_DIR/$target_name"
  local filename
  if [[ -d "$target" ]]; then
    local complete=1
    for filename in "$@"; do
      [[ -f "$target/$filename" ]] || complete=0
    done
    if (( complete )); then
      echo "model already installed: $target_name"
      return
    fi
  fi
  local staging
  staging="$(mktemp -d "$MODELS_DIR/.extract.XXXXXX")"
  trap 'rm -rf "$staging"' RETURN
  tar -xzf "$DOWNLOAD_DIR/$archive_name" -C "$staging"
  local marker
  marker="$(find "$staging" -type f -name "$1" -print -quit)"
  if [[ -z "$marker" ]]; then
    echo "archive $archive_name does not contain $1" >&2
    exit 1
  fi
  local extracted_root
  extracted_root="$(dirname "$marker")"
  for filename in "$@"; do
    if [[ ! -f "$extracted_root/$filename" ]]; then
      echo "archive $archive_name does not contain $filename" >&2
      exit 1
    fi
  done
  rm -rf "$target"
  mv "$extracted_root" "$target"
  rm -rf "$staging"
  trap - RETURN
  echo "installed model: $target_name"
}

ensure_archive "qwen3-asr-0.6B-dynq-q40.tar.gz"
ensure_archive "qwen3-asr-1.7B-dynq-q4km.tar.gz"
ensure_archive "zipformer-streaming.tar.gz"

install_model "qwen3-asr-0.6B-dynq-q40.tar.gz" "qwen3asr" \
  "Qwen3-ASR-0.6B-text-q40.gguf" \
  "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx" \
  "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"
install_model "qwen3-asr-1.7B-dynq-q4km.tar.gz" "qwen3asr-1.7b" \
  "Qwen3-ASR-1.7B-text-q4km.gguf" \
  "Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx" \
  "Qwen3-ASR-1.7B-encoder-backend.dynq.onnx"
install_model "zipformer-streaming.tar.gz" "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20" \
  "tokens.txt" \
  "encoder-epoch-99-avg-1.q.onnx" \
  "decoder-epoch-99-avg-1.onnx" \
  "joiner-epoch-99-avg-1.onnx"

install -m 0644 "$PROJECT_ROOT/scripts/model-configs/qwen3asr-0.6b-config.json" \
  "$MODELS_DIR/qwen3asr/config.json"
install -m 0644 "$PROJECT_ROOT/scripts/model-configs/qwen3asr-1.7b-config.json" \
  "$MODELS_DIR/qwen3asr-1.7b/config.json"
install -m 0644 "$PROJECT_ROOT/scripts/model-configs/zipformer.config" \
  "$MODELS_DIR/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/zipformer.config"

echo "setup complete"
