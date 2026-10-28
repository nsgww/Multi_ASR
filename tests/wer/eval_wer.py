"""WER/CER 评测：回放标注数据集 → 实时转写服务 → 计算字/词错误率。

一次回放同时出两份成绩：

* **流式裸识别**：每句定稿（FINAL）前的最后一版 PARTIAL，即 Zipformer 的输出
* **最终校正**：每句最高 revision 的 FINAL，即 Qwen3-ASR 纠错后的输出

两者差值 = 纠错模块的净收益。

用法::

    python tests/wer/eval_wer.py \
        --dataset E:/ASR/a/sm-sdk-0.4.3.-1-1/dataset \
        --server ws://192.168.1.179:8000 \
        --limit 300

数据集格式（SenseVoice 风格）::

    dataset/
      data.jsonl            # 每行 {"key": "...", "target": "参考文本", ...}
      audio/<key>.wav       # 16kHz 单声道 16bit

依赖：websockets、numpy（jieba 可选，装了才算 WER）。
注意：每评测一条，服务端会按正常流程落盘一份 transcript txt，
大批量评测后记得清理板端的 transcripts 目录。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import websockets

SAMPLE_RATE = 16000
# 回放块大小：20ms @16k 16bit mono = 640 字节
CHUNK_BYTES = 640


# ---------------------------------------------------------------- 文本归一化


def normalize(text: str) -> str:
    """CER 比对前的归一化：NFKC + 去标点/空白 + 小写，只留汉字/字母/数字。"""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text.lower() if ch.isalnum() or "一" <= ch <= "鿿")


# ---------------------------------------------------------------- 编辑距离


def edit_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """字符级编辑距离，返回 (替换 S, 删除 D, 插入 I)。"""
    n, m = len(ref), len(hyp)
    # dp[i][j] = ref[:i] -> hyp[:j] 的最小编辑数
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)

    # 回溯统计 S/D/I（优先级：替换 > 删除 > 插入，结果不因并列值漂移）
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                s += 1 if cost else 0
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
            continue
        j -= 1
        ins += 1
    return s, d, ins


@dataclass
class ErrorStats:
    """累计错误统计；cer = (S+D+I)/N（micro，按总字数加权）。"""

    s: int = 0
    d: int = 0
    i: int = 0
    n: int = 0  # 参考总字数
    per_clip_cer: list[float] = field(default_factory=list)

    def add(self, ref: str, hyp: str) -> float:
        s, d, i = edit_counts(ref, hyp)
        self.s += s
        self.d += d
        self.i += i
        self.n += len(ref)
        cer = (s + d + i) / len(ref) if ref else (1.0 if hyp else 0.0)
        self.per_clip_cer.append(cer)
        return cer

    @property
    def edits(self) -> int:
        return self.s + self.d + self.i

    @property
    def micro(self) -> float:
        return self.edits / self.n if self.n else 0.0

    @property
    def macro(self) -> float:
        return sum(self.per_clip_cer) / len(self.per_clip_cer) if self.per_clip_cer else 0.0


def word_error_rate(pairs: list[tuple[str, str]]) -> float | None:
    """jieba 分词后的 WER；没装 jieba 返回 None。"""
    try:
        import jieba  # type: ignore
    except ImportError:
        return None
    stats = ErrorStats()
    for ref, hyp in pairs:
        stats.add(" ".join(jieba.lcut(ref)), " ".join(jieba.lcut(hyp)))
    return stats.micro


# ---------------------------------------------------------------- 数据集


def load_manifest(dataset: Path, limit: int) -> list[tuple[str, str, Path]]:
    """返回 [(key, 参考文本, wav 路径)]，按 key 排序保证可复现。"""
    manifest = dataset / "data.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"找不到标注文件: {manifest}")
    items: list[tuple[str, str, Path]] = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = str(row.get("key") or "").strip()
            target = str(row.get("target") or "").strip()
            wav = dataset / "audio" / f"{key}.wav"
            if key and target and wav.is_file():
                items.append((key, target, wav))
    items.sort(key=lambda item: item[0])
    return items[:limit] if limit > 0 else items


def read_pcm16(path: Path) -> bytes:
    """读 wav 为 int16 mono 16k 原始 PCM 字节；格式不符直接报错（不重采样）。"""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(
                f"{path.name}: 需要 16kHz/mono/16bit，实际 "
                f"{w.getframerate()}Hz/{w.getnchannels()}ch/{w.getsampwidth() * 8}bit"
            )
        return w.readframes(w.getnframes())


# ---------------------------------------------------------------- 回放客户端


class ClipResult:
    def __init__(self) -> None:
        # sentence_id -> 最后一版 partial（Zipformer 裸识别）
        self.partials: dict[int, str] = {}
        # sentence_id -> (最高 revision, 文本)（纠错后）
        self.finals: dict[int, tuple[int, str]] = {}

    def add(self, msg: dict) -> None:
        if msg.get("type") != "transcript":
            return
        sid = msg.get("sentence_id")
        seg = msg.get("segment") or {}
        text = str(seg.get("text") or "")
        if not isinstance(sid, int):
            return
        if msg.get("is_partial"):
            if text:
                self.partials[sid] = text
        elif msg.get("is_final"):
            rev = int(msg.get("revision") or 0)
            cur = self.finals.get(sid)
            if cur is None or rev >= cur[0]:
                self.finals[sid] = (rev, text)

    def raw_text(self) -> str:
        """流式裸识别全文：有 partial 用 partial，没有则退回 final。"""
        ids = sorted(set(self.partials) | set(self.finals))
        return "".join(self.partials.get(i) or self.finals.get(i, (0, ""))[1] for i in ids)

    def final_text(self) -> str:
        """最终校正全文。"""
        return "".join(self.finals[i][1] for i in sorted(self.finals))


async def _wait_status(ws, want: str, timeout: float) -> None:
    """等到指定 status；期间的 transcript 一律丢弃（回合间隙不该有）。"""
    deadline = time.perf_counter() + timeout
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), max(deadline - time.perf_counter(), 1)))
        if msg.get("type") == "status" and msg.get("status") == want:
            return
        if msg.get("type") == "error":
            raise RuntimeError(f"服务端报错: {msg.get('message')}")


async def evaluate_clip(
    ws,
    pcm: bytes,
    *,
    speed: float,
    finish_timeout: float,
) -> ClipResult:
    """回放一条音频并收集两版文本。调用前必须处于 started 状态。"""
    result = ClipResult()
    finished = asyncio.Event()

    async def collector() -> None:
        while not finished.is_set():
            try:
                msg = json.loads(await ws.recv())
            except websockets.exceptions.ConnectionClosed:
                finished.set()
                return
            mtype = msg.get("type")
            if mtype == "finished":
                finished.set()
                return
            if mtype == "error":
                print(f"    [服务端错误] {msg.get('message')}")
                if "finish" in str(msg.get("message") or ""):
                    finished.set()
                    return
            result.add(msg)

    task = asyncio.create_task(collector())
    try:
        if speed > 0:
            # 按累计时间表调度，避免固定 sleep 累积漂移
            sent = 0
            t0 = time.perf_counter()
            for off in range(0, len(pcm), CHUNK_BYTES):
                await ws.send(pcm[off : off + CHUNK_BYTES])
                sent += 1
                target = sent * (CHUNK_BYTES / 2) / SAMPLE_RATE / speed
                delay = target - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
        else:
            for off in range(0, len(pcm), CHUNK_BYTES):
                await ws.send(pcm[off : off + CHUNK_BYTES])
        await ws.send(json.dumps({"type": "finish"}))
        try:
            await asyncio.wait_for(finished.wait(), finish_timeout)
        except asyncio.TimeoutError:
            print(f"    [警告] finish 超时（{finish_timeout:.0f}s），已收集的结果照常评分")
    finally:
        finished.set()
        # collector 可能正阻塞在 ws.recv() 上，finished 事件唤不醒它，必须取消
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return result


# ---------------------------------------------------------------- 报告


def write_report(
    out_dir: Path,
    *,
    server: str,
    total: int,
    raw: ErrorStats,
    corr: ErrorStats,
    wer: float | None,
    details: list[dict],
    elapsed: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    lines = [
        f"# WER/CER 评测报告 {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"服务: {server}  样本: {total}  耗时: {elapsed / 60:.1f} 分钟",
        "",
        "## 总体（micro = 总编辑数/总字数，macro = 逐条平均）",
        f"流式裸识别 CER: {raw.micro:.2%} (S={raw.s} D={raw.d} I={raw.i} / N={raw.n})  macro={raw.macro:.2%}",
        f"最终校正   CER: {corr.micro:.2%} (S={corr.s} D={corr.d} I={corr.i} / N={corr.n})  macro={corr.macro:.2%}",
        f"纠错收益: {raw.micro - corr.micro:+.2%}",
    ]
    if wer is not None:
        lines.append(f"WER(jieba, 校正后): {wer:.2%}")
    else:
        lines.append("WER: 跳过（未安装 jieba）")
    lines += ["", "## 逐条明细（按校正后 CER 降序）"]
    for row in sorted(details, key=lambda r: r["cer_corr"], reverse=True):
        lines.append(
            f"{row['key']}  raw={row['cer_raw']:.1%} corr={row['cer_corr']:.1%}\n"
            f"  ref : {row['ref']}\n"
            f"  raw : {row['hyp_raw']}\n"
            f"  corr: {row['hyp_corr']}"
        )

    report = out_dir / f"wer-report-{stamp}.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"wer-report-{stamp}.json").write_text(
        json.dumps(
            {
                "server": server,
                "total": total,
                "elapsed_s": round(elapsed, 1),
                "cer_raw": {"micro": raw.micro, "macro": raw.macro, "s": raw.s, "d": raw.d, "i": raw.i, "n": raw.n},
                "cer_corr": {"micro": corr.micro, "macro": corr.macro, "s": corr.s, "d": corr.d, "i": corr.i, "n": corr.n},
                "wer_corr_jieba": wer,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入: {report}")


# ---------------------------------------------------------------- main


async def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    items = load_manifest(dataset, args.limit)
    if not items:
        raise SystemExit("数据集为空或没有有效条目")
    print(f"样本 {len(items)} 条，服务器 {args.server}")

    url = f"{args.server}/ws/stream?sample_rate={SAMPLE_RATE}&wakeup={'true' if args.wakeup else 'false'}"
    raw_stats, corr_stats = ErrorStats(), ErrorStats()
    pairs: list[tuple[str, str]] = []
    details: list[dict] = []
    t0 = time.perf_counter()

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "start"}))
        await _wait_status(ws, "started", timeout=args.session_timeout)

        for idx, (key, ref_raw, wav) in enumerate(items, 1):
            ref = normalize(ref_raw)
            if not ref:
                print(f"[{idx}/{len(items)}] {key}: 参考文本归一化后为空，跳过")
                continue
            pcm = read_pcm16(wav)

            result = await evaluate_clip(
                ws, pcm, speed=args.speed, finish_timeout=args.finish_timeout
            )
            hyp_raw = normalize(result.raw_text())
            hyp_corr = normalize(result.final_text())
            cer_raw = raw_stats.add(ref, hyp_raw)
            cer_corr = corr_stats.add(ref, hyp_corr)
            pairs.append((ref, hyp_corr))
            details.append(
                {
                    "key": key,
                    "ref": ref_raw,
                    "hyp_raw": hyp_raw,
                    "hyp_corr": hyp_corr,
                    "cer_raw": round(cer_raw, 4),
                    "cer_corr": round(cer_corr, 4),
                }
            )
            print(
                f"[{idx}/{len(items)}] {key}  raw={cer_raw:.1%} corr={cer_corr:.1%}  "
                f"(累计 raw={raw_stats.micro:.2%} corr={corr_stats.micro:.2%})"
            )

            if idx < len(items):
                await ws.send(json.dumps({"type": "new_transcription"}))
                await _wait_status(ws, "ready", timeout=args.session_timeout)
                # new_transcription 后服务端 _started=False，音频会被丢弃，
                # 必须重新 start 才会继续转写
                await ws.send(json.dumps({"type": "start"}))
                await _wait_status(ws, "started", timeout=args.session_timeout)

    elapsed = time.perf_counter() - t0
    wer = word_error_rate(pairs)
    write_report(
        Path(args.out),
        server=args.server,
        total=len(details),
        raw=raw_stats,
        corr=corr_stats,
        wer=wer,
        details=details,
        elapsed=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="实时转写 WER/CER 回放评测")
    parser.add_argument("--dataset", required=True, help="数据集目录（含 data.jsonl 和 audio/）")
    parser.add_argument("--server", default="ws://192.168.1.179:8000", help="服务地址（不含路径）")
    parser.add_argument("--limit", type=int, default=300, help="评测条数；0 = 全部")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="回放速度倍率；默认 1.0 实时。注意：服务端按帧率消费，"
        "全速灌入会溢出丢帧导致零输出，不要盲目加速",
    )
    parser.add_argument("--finish-timeout", type=float, default=90, help="finish 后等最终结果的超时（秒）")
    parser.add_argument("--session-timeout", type=float, default=60, help="建/换会话的超时（秒）")
    parser.add_argument(
        "--wakeup",
        action="store_true",
        help="保留唤醒门控（默认关闭：评测音频没有唤醒词，必须关门控才有转写输出）",
    )
    parser.add_argument("--out", default="tests/wer/reports", help="报告输出目录")
    args = parser.parse_args(sys.argv[1:])
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
