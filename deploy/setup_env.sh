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
  echo "==> python3.13 已存在，跳过 python 安装"
  # 只跳过 python，系统库仍要按需补：缺 libspeexdsp1 会让组播直接
  # SpeexResamplerError，缺 libportaudio2 会让麦克风打不开。
  missing_libs=()
  ldconfig -p 2>/dev/null | grep -q 'libspeexdsp\.so\.1' || missing_libs+=(libspeexdsp1)
  ldconfig -p 2>/dev/null | grep -q 'libportaudio\.so\.2' || missing_libs+=(libportaudio2)
  ldconfig -p 2>/dev/null | grep -q 'libsndfile\.so\.1' || missing_libs+=(libsndfile1)
  if (( ${#missing_libs[@]} )); then
    echo "==> 补装系统库: ${missing_libs[*]}"
    if ! { apt-get update -qq && apt-get install -y -qq "${missing_libs[@]}"; }; then
      echo "!! 系统库安装失败，组播/麦克风可能不可用：apt install ${missing_libs[*]}"
    fi
  else
    echo "==> 系统库齐全（speexdsp / portaudio / sndfile）"
  fi
  # kaldi-native-fbank 在 riscv64 上没有预编译 wheel 时要源码构建，提前提示
  for tool in cmake g++; do
    command -v "$tool" >/dev/null 2>&1 \
      || echo "   提示: 未找到 $tool，kaldi-native-fbank 源码构建会失败（只影响 paraformer 备用路径）"
  done
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
# --no-deps 会把它自己的运行时依赖一起跳过（新板首次装就暴露：No module named 'jieba'）。
# 按 import 链实测，paraformer-online 只需要这几个：
#   jieba(纯 Python)            <- funasr_onnx.utils.utils
#   kaldi-native-fbank(C++ 扩展) <- funasr_onnx.utils.frontend
#   PyYAML                      <- funasr_onnx.utils.utils（项目已装，显式补一次防漏）
# sentencepiece / scipy / pydub 只在其它入口用到，本项目 paraformer-online 路径不碰，
# 不装（sentencepiece 还要源码编译，riscv64 上没必要）。
"$PIP" install -q jieba "PyYAML>=5.1.2"
if ! "$PIP" install -q kaldi-native-fbank; then
  echo "!! kaldi-native-fbank 安装失败（riscv64 通常没有预编译 wheel）"
  echo "   → paraformer 备用识别器不可用，zipformer 主链路不受影响"
  echo "   → 需要它时先装工具链再重跑："
  echo "     apt install -y build-essential cmake python3.13-dev"
fi
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
# 分组验证：必需依赖缺失才失败；funasr 只影响 paraformer 备用路径，缺了也要
# 把 zipformer 主链路装完（旧版一条 import 全串在一起，缺 jieba 就整脚本报错）。
"$PY" - <<'PYEOF'
import importlib
import sys

required_ok = True
for name in ("sherpa_onnx", "onnxruntime", "sounddevice"):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        print(f"  [OK]   {name} {version}".rstrip())
    except Exception as exc:  # noqa: BLE001 - 装环境脚本要把原因打全
        required_ok = False
        print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")

for name in ("funasr_onnx.paraformer_online_bin",):
    try:
        importlib.import_module(name)
        print(f"  [OK]   {name}（paraformer 备用路径可用）")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] {name}: {type(exc).__name__}: {exc}")
        print("         zipformer 主链路不受影响；需要 paraformer 时补依赖后重跑")

sys.exit(0 if required_ok else 1)
PYEOF

echo ""
echo "完成。启动服务: sudo systemctl restart multi-stream-asr（首次加载 1-2 分钟）"
