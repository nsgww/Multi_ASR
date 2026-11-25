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

echo "==> 验证"
"$PY" -c "import sherpa_onnx, onnxruntime, funasr_onnx, sounddevice; print('env OK:', onnxruntime.__version__)"

echo ""
echo "完成。启动服务: sudo systemctl restart multi-stream-asr（首次加载 1-2 分钟）"
