import json

path = "/opt/multi-stream-asr/output/session-1794822358976-0/model_outputs/lane0.jsonl"
recs = [json.loads(l) for l in open(path)]
by_sid = {}
for d in recs:
    by_sid.setdefault(d.get("sentence_id"), {})[d.get("model")] = d.get("text") or ""

for sid in (129, 189, 209):
    pair = by_sid.get(sid, {})
    online = pair.get("zipformer", "")
    qwen = pair.get("qwen3-asr", "")
    info = next(r for r in recs if r.get("sentence_id") == sid and r.get("model") == "qwen3-asr")
    print("=== sid %s  audio=%ss rtf=%s ===" % (sid, info.get("audio_seconds"), info.get("rtf")))
    print("[在线 zipformer] len=%d: %s" % (len(online), online[:150]))
    print("[校正 qwen3-asr] len=%d: %s" % (len(qwen), qwen[:150]))
    print()
