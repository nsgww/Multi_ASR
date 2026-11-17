import json, urllib.request, urllib.error

sp = open("/var/lib/models/llm/summarization/gemma-4-E2B/system_prompt.txt", encoding="utf-8").read()
tx = open("/opt/multi-stream-asr/output/transcripts/spacemit-asr-lane0-1794655555913.txt", encoding="utf-8").read()

def call(payload, tag):
    req = urllib.request.Request(
        "http://127.0.0.1:8065/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        r = json.load(urllib.request.urlopen(req, timeout=240))
        c = r["choices"][0]
        print("=== " + tag + " finish=" + str(c.get("finish_reason")) + " ===")
        print(c["message"]["content"][:600])
    except urllib.error.HTTPError as e:
        print("=== " + tag + " HTTP " + str(e.code) + " === " + e.read().decode()[:200])

msgs = [{"role": "system", "content": sp}, {"role": "user", "content": tx[:1500]}]

call({"model": "gemma", "messages": msgs, "temperature": 0.0, "max_tokens": 1024}, "t0")
call({"model": "gemma", "messages": msgs, "temperature": 0.2, "max_tokens": 1024,
      "repeat_penalty": 1.1, "presence_penalty": 0.3}, "rp1.1-pp0.3")
