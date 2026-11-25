#!/usr/bin/env bash
# multi-stream-asr 虚拟环境一键部署（板端 Bianbu / RISC-V）
# 用法: sudo bash deploy/setup_env.sh
# 幂等，可重复执行。只装环境，不下载模型（模型走 OpenStore）。
set -euo pipefail

ENV_DIR="${ENV_DIR:-/opt/envs/multi-stream-asr}"

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "==> 安装系统依赖（python3.13 / portaudio / libsndfile / speexdsp）"
  apt-get update -qq
  apt-get install -y -qq python3.13 python3.13-venv \
    libportaudio2 portaudio19-dev libasound2-dev \
    libsndfile1 libspeexdsp1 ca-certificates
else
  echo "==> python3.13 已存在，跳过 apt（如需补系统包请手动 apt install）"
fi

# 定制 wheel 索引：onnxruntime(33) / sherpa-onnx(81) / 阿里镜像兜底
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
export PIP_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple"

if [[ -x "$ENV_DIR/bin/python" ]]; then
  echo "==> venv 已存在: $ENV_DIR（跳过创建；重建请先 rm -rf）"
else
  echo "==> 创建 venv: $ENV_DIR"
  python3.13 -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

echo "==> 安装项目依赖"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PIP" install --upgrade pip -q
# 项目本体 + hardware 额外依赖（sounddevice / sherpa-onnx）
"$PIP" install -q -e "$PROJECT_ROOT[hardware]"
# 板端实测需手动补的包（pyproject 未声明）
"$PIP" install -q \
  "onnxruntime==1.24.2+spacemit.a1" \
  "PyAudio==0.2.14" \
  "soundfile==0.13.1"
# funasr_onnx 0.4.3 pin 了 numpy<=1.26.4，与本栈 numpy 2.x 直接冲突——旧板是分批装
# 绕过的。实测 paraformer 备用路径在 numpy 2.x 下 import/运行正常（线上 zipformer
# 不依赖它），故 --no-deps 跳过该 pin。
"$PIP" install -q --no-deps "funasr_onnx==0.4.3"
# librosa 在 riscv64 上装不了（依赖 numba/llvmlite 无 riscv64 wheel），而 funasr_onnx
# 只在文件加载路径用 librosa.load——写同名桩模块（wave+numpy 实现 load/resample），
# 与旧板一致；paraformer 为备用识别器，走 PCM 流式路径时不触发。
SITE_PKGS="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if ! "$PY" -c 'import librosa' 2>/dev/null; then
  cat > "$SITE_PKGS/librosa.py" <<'EOF'
"""librosa 桩：RISC-V 上 librosa(numba/llvmlite) 装不了。
funasr_onnx 只在文件加载路径用 librosa.load，这里用 wave+numpy 实现。"""
import wave
import numpy as np

__version__ = "0.0-stub"


def load(path, sr=16000, mono=True):
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        ch = f.getnchannels()
        raw = f.readframes(f.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if mono and ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr and rate != sr:
        x = np.arange(len(audio))
        xi = np.linspace(0, len(audio) - 1, int(len(audio) * sr / rate))
        audio = np.interp(xi, x, audio).astype(np.float32)
    return audio, sr


def resample(y, orig_sr, target_sr):
    y = np.asarray(y, dtype=np.float32)
    x = np.arange(len(y))
    xi = np.linspace(0, len(y) - 1, int(len(y) * target_sr / orig_sr))
    return np.interp(xi, x, y).astype(np.float32)
EOF
  echo "==> 已写入 librosa 桩模块（riscv64 无 numba，funasr 文件加载路径专用）"
fi

echo "==> 验证"
"$PY" -c "import sherpa_onnx, onnxruntime, funasr_onnx, sounddevice; print('env OK:', onnxruntime.__version__)"

echo ""
echo "完成。启动服务: sudo systemctl restart multi-stream-asr（首次加载 1-2 分钟）"
