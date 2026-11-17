import json, glob, re

paths = sorted(glob.glob("/opt/multi-stream-asr/output/session-179483*/model_outputs/lane0.jsonl"))
recs = [json.loads(l) for l in open(paths[-1])] if paths else []
print("records:", len(recs))
for d in recs:
    t = d.get("text") or ""
    m = re.search(r"(.{2,8}?)\1{4,}", t)
    if m:
        frag = t[max(0, m.start() - 25):m.start() + 25]
        model = d.get("model")
        sid = d.get("sentence_id")
        print("[%s] sid=%s len=%s 片段: ...%s..." % (model, sid, len(t), frag))
