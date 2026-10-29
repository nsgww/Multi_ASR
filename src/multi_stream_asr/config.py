from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
from pathlib import Path


SUPPORTED_LANES = (2, 4, 6, 8)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONLINE_MODEL_DIR = Path(
    "/var/lib/sm-sdk/models/asr/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
)
DEFAULT_OFFLINE_MODEL_DIR = Path("/var/lib/sm-sdk/models/asr/qwen3asr")
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "assets" / "wav16k"
DEFAULT_VAD_MODEL_DIR = Path("/var/lib/sm-sdk/models/vad/silero-vad/v5")
DEFAULT_OFFLINE_ASR_MODEL_DIR = Path("/var/lib/sm-sdk/models/asr/sensevoice")
DEFAULT_SUMMARY_CONFIG_PATH = PROJECT_ROOT / "summary.yaml"
DEFAULT_KWS_MODEL_DIR = Path(
    "/var/lib/sm-sdk/models/kws/"
    "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
)
DEFAULT_KWS_KEYWORDS_FILE = PROJECT_ROOT / "config" / "keywords.txt"
DEFAULT_SPEAKER_MODEL_PATH = Path(
    "/var/lib/sm-sdk/models/speaker/"
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
)
ZIPFORMER_WARMUP_AUDIO_PATH = (
    Path(__file__).resolve().parent / "assets" / "zipformer_warmup.wav"
)

ENV_QWEN_ENCODER_CORES = "MULTI_STREAM_ASR_QWEN_ENCODER_CORES"
ENV_QWEN_DECODER_CORES = "MULTI_STREAM_ASR_QWEN_DECODER_CORES"
# Encoder 和 decoder 默认各占 A100 集群的一半（不相交）：encoder intra-op
# 线程绑 8-9，decoder（ggml 设备核心集，由 SPACEMIT_PERFER_CORE_ID 决定）
# 占 10-15。两两者重叠会让 encoder 慢约 1.6x，并在持续负载下出现整跑停顿。
# 2+6 是 8 lane 长音频（Zipformer 分段）下实测最优的拆分：encoder 串行化后
# 2 核足够，多出的 6 个 decoder 核正是连续批处理解码的瓶颈（per-request
# offline_rtf 1.034/1.052 (4+4) -> 0.815/0.791 (2+6)）。-np 跟随 decoder 核数。
DEFAULT_QWEN_ENCODER_CORES = (8, 9)
DEFAULT_QWEN_DECODER_CORES = (10, 11, 12, 13, 14, 15)


def _env_core_ids(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Return CPU core ids from ``name`` (``"8,9"`` or ``"8;9"``), else default.

    Lets a deployment re-split the 16-core SoC between the ASR encoder and the
    shared Qwen decoder without rebuilding the package.
    """
    value = os.environ.get(name)
    if not value or not value.strip():
        return default
    cores: list[int] = []
    for part in value.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            core = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a comma-separated list of core ids, got {value!r}"
            ) from exc
        if core < 0:
            raise ValueError(f"{name} core ids must be non-negative, got {core!r}")
        cores.append(core)
    if not cores:
        raise ValueError(f"{name} must list at least one core id, got {value!r}")
    return tuple(cores)


def qwen_encoder_cores() -> tuple[int, ...]:
    return _env_core_ids(ENV_QWEN_ENCODER_CORES, DEFAULT_QWEN_ENCODER_CORES)


def qwen_decoder_cores() -> tuple[int, ...]:
    return _env_core_ids(ENV_QWEN_DECODER_CORES, DEFAULT_QWEN_DECODER_CORES)


@dataclass(frozen=True)
class KWSConfig:
    """Keyword spotting (sherpa-onnx zipformer KWS)."""

    enabled: bool = True
    model_dir: Path = DEFAULT_KWS_MODEL_DIR
    keywords_file: Path = DEFAULT_KWS_KEYWORDS_FILE
    num_threads: int = 1
    max_active_paths: int = 4
    num_trailing_blanks: int = 1
    keywords_score: float = 1.0
    keywords_threshold: float = 0.25
    provider: str = "cpu"
    sample_rate: int = 16000
    repeat_interval: float = 1.0


@dataclass(frozen=True)
class OfflineModelSpec:
    model_id: str
    model_dir: Path
    text_model_filename: str
    frontend_filename: str
    backend_filename: str
    d_model: int
    hidden_size: int
    output_dim: int

    @property
    def required_filenames(self) -> tuple[str, str, str]:
        return (
            self.text_model_filename,
            self.frontend_filename,
            self.backend_filename,
        )


OFFLINE_MODELS = {
    "qwen3-asr-0.6b-q40": OfflineModelSpec(
        model_id="qwen3-asr-0.6b-q40",
        model_dir=DEFAULT_OFFLINE_MODEL_DIR,
        text_model_filename="Qwen3-ASR-0.6B-text-q40.gguf",
        frontend_filename="Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        backend_filename="Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
        d_model=896,
        hidden_size=1024,
        output_dim=1024,
    ),
    "qwen3-asr-1.7b-q4km": OfflineModelSpec(
        model_id="qwen3-asr-1.7b-q4km",
        model_dir=Path("/var/lib/sm-sdk/models/asr/qwen3asr-1.7b"),
        text_model_filename="Qwen3-ASR-1.7B-text-q4km.gguf",
        frontend_filename="Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx",
        backend_filename="Qwen3-ASR-1.7B-encoder-backend.dynq.onnx",
        d_model=1024,
        hidden_size=2048,
        output_dim=2048,
    ),
    # SpacemiT 官方转换的 1.7B 生产包（2026-08-04 启用，替换 0.6B 以提升准确率）
    "qwen3-asr-1.7b-q40": OfflineModelSpec(
        model_id="qwen3-asr-1.7b-q40",
        model_dir=Path("/var/lib/sm-sdk/models/asr/qwen3asr-1.7b"),
        text_model_filename="Qwen3-ASR-1.7B-text-q4km.gguf",
        frontend_filename="Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx",
        backend_filename="Qwen3-ASR-1.7B-encoder-backend.dynq.onnx",
        d_model=1024,
        hidden_size=2048,
        output_dim=2048,
    ),
}


@dataclass(frozen=True)
class LaneInputConfig:
    kind: str = "default_files"
    local_wav_path: Path | None = None
    microphone_device: int | None = None
    microphone_device_id: str | None = None
    loop_audio: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"default_files", "local_wav", "microphone"}:
            raise ValueError(f"unsupported lane input kind: {self.kind}")
        if self.kind == "local_wav" and self.local_wav_path is None:
            raise ValueError("local_wav requires local_wav_path")
        if self.kind != "local_wav" and self.local_wav_path is not None:
            raise ValueError("only local_wav accepts local_wav_path")
        if self.kind != "microphone" and self.microphone_device is not None:
            raise ValueError("only microphone accepts microphone_device")
        if self.kind != "microphone" and self.microphone_device_id is not None:
            raise ValueError("only microphone accepts microphone_device_id")
        if self.kind == "microphone" and self.loop_audio:
            raise ValueError("only file inputs accept loop_audio")
        if not isinstance(self.loop_audio, bool):
            raise ValueError("loop_audio must be a boolean")
        if self.microphone_device is not None and (
            not isinstance(self.microphone_device, int)
            or isinstance(self.microphone_device, bool)
        ):
            raise ValueError("microphone_device must be an integer or null")
        if (
            self.microphone_device_id is not None
            and not self.microphone_device_id.strip()
        ):
            raise ValueError("microphone_device_id must be non-empty")


def validate_offline_model_files(spec: OfflineModelSpec) -> list[Path]:
    model_directory = spec.model_dir.expanduser().resolve(strict=True)
    invalid_files: list[Path] = []
    for filename in spec.required_filenames:
        required_path = model_directory / filename
        try:
            resolved_path = required_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            invalid_files.append(required_path)
            continue
        if (
            not resolved_path.is_relative_to(model_directory)
            or not resolved_path.is_file()
            or not os.access(resolved_path, os.R_OK)
        ):
            invalid_files.append(required_path)
    return invalid_files


@dataclass(frozen=True)
class EndpointConfig:
    # rule1（纯静音空端点，must_contain_nonsilence=False）必须保持远大于句间停顿：
    # 它每 ~rule1 秒周期性触发空文本端点，端点换流时会吃掉下一句的开头
    # （1.0s 时间隔的连续句子实测丢开头/整句）。设为 60s 等价于禁用，
    # 真正的句切由 rule2（含语音、0.7s 尾部静音）承担。
    rule1_min_trailing_silence: float = 60.0
    rule2_min_trailing_silence: float = 0.7
    rule3_min_utterance_length: float = 10.0
    cut_margin_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.cut_margin_seconds) or not (
            0.0 <= self.cut_margin_seconds < self.rule2_min_trailing_silence
        ):
            raise ValueError(
                "cut_margin_seconds must be finite, non-negative, and less than "
                "rule2_min_trailing_silence"
            )


@dataclass(frozen=True)
class ModelConfig:
    online_model_dir: Path = DEFAULT_ONLINE_MODEL_DIR
    # 2026-08-04 起默认使用官方 1.7B（热词能力+准确率更好）；
    # 回退 0.6B：环境变量 OFFLINE_MODEL_DIR=/var/lib/sm-sdk/models/asr/qwen3asr
    # （规格按目录内容自动匹配，offline_model_id 只是兜底）
    offline_model_id: str = os.getenv("OFFLINE_MODEL_ID", "qwen3-asr-1.7b-q4km")
    offline_model_dir: Path = Path(
        os.getenv("OFFLINE_MODEL_DIR", "/var/lib/sm-sdk/models/asr/qwen3asr-1.7b")
    )
    online_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.offline_model_id not in OFFLINE_MODELS:
            raise ValueError(f"unsupported offline model: {self.offline_model_id}")

    @property
    def offline_model(self) -> OfflineModelSpec:
        # 按目录实际内容自动匹配规格：切换不同家族版本（1.7B↔0.6B）时
        # 文件名和维度参数不同，以目录里真实存在的文件为准选 spec
        d = self.offline_model_dir
        try:
            names = {p.name for p in d.iterdir() if p.is_file()}
        except OSError:
            names = set()
        for spec in OFFLINE_MODELS.values():
            if all(fn in names for fn in spec.required_filenames):
                return replace(spec, model_dir=d)
        return replace(
            OFFLINE_MODELS[self.offline_model_id],
            model_dir=self.offline_model_dir,
        )


@dataclass(frozen=True)
class SharedQwenConfig:
    """共享 Qwen3-ASR llama-server 设置（资源分配对齐性能调优线）。

    核心拆分默认 2+6（encoder 绑 8-9，decoder 占 10-15），两者不相交：
    重叠会让串行 encoder 慢约 1.6x 并在持续负载下整跑停顿。
    ``decoder_threads`` 默认 1：SpacemiT 后端按 ``decoder_cores`` 列出的
    每核自行调度 worker 线程（``QWEN_DECODER_THREADS`` 可覆盖）。
    ``n_parallel`` 跟随 decoder 核数而非 lane 数：llama-server 给每个 slot
    均分 ``n_ctx``，decoder 同时能处理的请求数受核数限制，多余 slot 只会在
    忙解码器后排队。两个核心列表均可通过
    ``MULTI_STREAM_ASR_QWEN_ENCODER_CORES`` / ``..._DECODER_CORES`` 按机重分。
    """

    host: str = "127.0.0.1"
    port: int = 8090
    encoder_thread_num: int = field(
        default_factory=lambda: len(qwen_encoder_cores())
    )
    encoder_affinity: str = field(
        default_factory=lambda: ";".join(str(c) for c in qwen_encoder_cores())
    )
    decoder_cores: str = field(
        default_factory=lambda: ",".join(str(c) for c in qwen_decoder_cores())
    )
    decoder_threads: int = field(
        default_factory=lambda: int(os.getenv("QWEN_DECODER_THREADS", "1"))
    )
    decoder_batch_threads: int = 1
    batch_size: int = 2048
    ubatch_size: int = 512
    http_threads: int = 24
    n_parallel: int = field(default_factory=lambda: len(qwen_decoder_cores()))
    n_ctx: int = 16384
    request_timeout_ms: int = 120000
    device: str = "SPACEMIT0"

    def __post_init__(self) -> None:
        # llama.cpp 0.2.x 要求每次启动都带 `--device SPACEMIT0`，否则模型
        # 不会落到 ggml-spacemit 后端、SpacemiT 核心闲置。空字符串曾表示
        # "省略该参数"，会静默退回 CPU 启动，因此直接拒绝。
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(
                "device must be a non-empty device name (llama.cpp 0.2.x "
                "requires --device SPACEMIT0)"
            )


@dataclass(frozen=True)
class LaneConfig:
    stream_id: int


@dataclass(frozen=True)
class SpeakerConfig:
    """说话人分离与声纹识别。

    默认 **关闭**：``enabled=False`` 时 pipeline 完全不加载模型、不生成
    ``speaker`` 字段，行为与改动前逐字节一致。

    阈值默认值是按 CAM++ 在**干净近场**下的典型分布给的；混响远场必须重新
    标定（同人余弦会从 0.6+ 掉到 0.4 附近），不要直接照搬。
    """

    enabled: bool = False
    model_path: Path = DEFAULT_SPEAKER_MODEL_PATH
    # 声纹库目录；None 表示用 <output_dir>/voiceprints。目录不存在=只做分离不做识别
    registry_dir: Path | None = None
    provider: str = "cpu"
    # 声纹提取线程数：1 线程时多窗（1.5s/0.5s hop）提 4s 音频约 2.3s，
    # 会拖慢 FINAL（FINAL 等声纹结果）；K3 上 2 线程约 1.2s、4 线程约 0.9s
    num_threads: int = int(os.getenv("SPEAKER_NUM_THREADS", "1"))
    sample_rate: int = 16000

    # 段长下限：短于这个长度提不出可靠声纹，直接沿用上一句
    min_duration_seconds: float = 1.2
    # 多窗平均：混响下单窗不稳，切窗取均值
    window_seconds: float = 1.5
    hop_seconds: float = 0.5
    # 首尾按能量裁剪的峰值比例（去混响拖尾）
    trim_floor_ratio: float = 0.25

    # 在线聚类
    # 实测（CAM++ + 干净近场单人语音，见 README 6.2）：
    #   同一人 3s 段之间 0.69~0.84；不同人 0.04~0.21 → 间隔约 0.48，很好分。
    #   但带 BGM / 远场混响时同人会掉到 0.42、异人升到 0.53，分布直接重叠。
    # 所以这组默认值只适用于干净近场；远场必须用 scripts/speaker_calibrate.py
    # 按真实房间重新标定，别照搬。
    match_threshold: float = 0.55
    new_speaker_threshold: float = 0.45
    merge_threshold: float = 0.70
    max_speakers: int = 8
    sticky_bonus: float = 0.05

    # 1:N 声纹识别（双门限）。干净近场同人 0.69+，留足余量取 0.55。
    identify_threshold: float = 0.55
    identify_margin: float = 0.05

    # 会话末全量重聚类（第二遍），用 revision/重标映射纠正早期错分
    enable_second_pass: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.new_speaker_threshold <= self.match_threshold <= 1.0:
            raise ValueError(
                "speaker thresholds must satisfy "
                "0 <= new_speaker_threshold <= match_threshold <= 1"
            )
        if not 0.0 <= self.identify_margin <= 1.0:
            raise ValueError("identify_margin must be within [0, 1]")
        if not 0.0 <= self.trim_floor_ratio < 1.0:
            raise ValueError("trim_floor_ratio must be within [0, 1)")
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        if self.max_speakers < 1:
            raise ValueError("max_speakers must be >= 1")


@dataclass(frozen=True)
class AppConfig:
    lanes: int = 4
    frame_duration_ms: int = 100
    input_dir: Path = DEFAULT_INPUT_DIR
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    shared_qwen: SharedQwenConfig = field(default_factory=SharedQwenConfig)
    lane_inputs: tuple[LaneInputConfig, ...] = ()
    output_dir: Path = Path("output").resolve()
    verbose: bool = False
    # Web API settings
    web_port: int = 8060
    max_concurrent_sessions: int = 8
    vad_model_dir: Path = DEFAULT_VAD_MODEL_DIR
    offline_asr_model_dir: Path = DEFAULT_OFFLINE_ASR_MODEL_DIR
    summary_config_path: Path = DEFAULT_SUMMARY_CONFIG_PATH
    speexdsp_lib_path: Path | None = None
    kws: KWSConfig = field(default_factory=KWSConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    sensitive_words_file: Path = Path("config/sensitive_words.txt")
    hotwords_file: Path = Path("config/hotwords.txt")
    hotwords_score: float = 1.5
    # 唤醒后限时转写秒数（0=不限时），可经 /v1/config/wakeup 运行时调整
    wakeup_window_seconds: float = 30.0
    # 在线 Zipformer 热词偏置开关。当前板子的 SpacemiT sherpa-onnx 的
    # modified_beam_search 不生效（2026-08-04 实测）且慢约 8%，默认关闭，
    # 热词只注入离线 Qwen 精修 prompt；将来推理库修复后可打开。
    hotwords_online: bool = False

    def __post_init__(self) -> None:
        if self.lanes not in SUPPORTED_LANES:
            raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {self.lanes}")
        if self.lane_inputs and len(self.lane_inputs) != self.lanes:
            raise ValueError("lane_inputs must be empty or match lanes")

    @property
    def segment_cache_seconds(self) -> float:
        return 1.5 * self.endpoint.rule3_min_utterance_length

    @property
    def lane_configs(self) -> list[LaneConfig]:
        return build_lane_configs(self.lanes)

    def input_for_lane(self, stream_id: int) -> LaneInputConfig:
        if stream_id < 0 or stream_id >= self.lanes:
            raise ValueError(f"stream_id out of range: {stream_id}")
        if self.lane_inputs:
            return self.lane_inputs[stream_id]
        return LaneInputConfig()


def build_lane_configs(lanes: int) -> list[LaneConfig]:
    if lanes not in SUPPORTED_LANES:
        raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {lanes}")
    return [LaneConfig(stream_id=stream_id) for stream_id in range(lanes)]
