"""一次性探针：回放一条音频，打印 finish 前后所有服务端消息和到达时间。"""
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
CHUNK_BYTES = 640


def read_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        return w.readframes(w.getnframes())


async def main() -> None:
    wav = Path(sys.argv[1])
    server = sys.argv[2] if len(sys.argv) > 2 else "ws://192.168.1.179:8000"
    pcm = read_pcm16(wav)
    print(f"audio {len(pcm) / 2 / SAMPLE_RATE:.2f}s -> {server}")
    t0 = time.perf_counter()

    def ts() -> str:
        return f"{time.perf_counter() - t0:7.2f}s"

    url = f"{server}/ws/stream?sample_rate=16000&wakeup=false"
    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        async def listener() -> None:
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "transcript":
                        seg = msg.get("segment") or {}
                        spk = seg.get('speaker')
                        spk_s = f" spk={spk.get('id')}/{spk.get('label')} conf={spk.get('confidence')}" if isinstance(spk, dict) else ""
                        print(
                            f"[{ts()}] transcript sid={msg.get('sentence_id')} "
                            f"rev={msg.get('revision')} partial={msg.get('is_partial')} "
                            f"final={msg.get('is_final')}{spk_s} text={str(seg.get('text'))[:30]!r}"
                        )
                    else:
                        print(f"[{ts()}] {json.dumps(msg, ensure_ascii=False)[:200]}")
            except Exception as exc:
                print(f"[{ts()}] listener ended: {exc!r}")

        task = asyncio.create_task(listener())
        await ws.send(json.dumps({"type": "start"}))
        await asyncio.sleep(1.0)  # 等 started

        sent = 0
        ta = time.perf_counter()
        for off in range(0, len(pcm), CHUNK_BYTES):
            await ws.send(pcm[off : off + CHUNK_BYTES])
            sent += 1
            target = sent * (CHUNK_BYTES / 2) / SAMPLE_RATE
            delay = target - (time.perf_counter() - ta)
            if delay > 0:
                await asyncio.sleep(delay)
        print(f"[{ts()}] audio sent, -> finish")
        await ws.send(json.dumps({"type": "finish"}))
        await asyncio.sleep(30)  # 观察 finish 后 30s 内的所有消息
        print(f"[{ts()}] probe done")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


asyncio.run(main())
