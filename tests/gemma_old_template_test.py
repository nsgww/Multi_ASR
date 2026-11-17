import json, urllib.request

OLD_TEMPLATE = """/no_think
你是语音内容摘要助手。请根据转写文本生成中文核心内容总结。

目标：
用少量要点让读者快速理解音频主要内容与关键信息。

要求：
1. 只基于原文，不得补充或推测
2. 必须归纳总结，不逐句复述
3. 抓取核心信息：在说什么、发生了什么、讨论了什么
4. 保留关键事实：人名、事项、结论、决定、问题、时间
5. 去除口语、重复、无意义填充
6. 最多3-5条要点（视内容而定）

输出格式：
- 要点1
- 要点2
- 要点3
- 要点4（如有必要）
- 要点5（如有必要）

转写内容：
{text}
"""

tx = open("/opt/multi-stream-asr/output/transcripts/spacemit-asr-lane0-1794655555913.txt", encoding="utf-8").read()[:1500]

def call(msgs, tag):
    payload = {
        "model": "gemma",
        "messages": msgs,
        "temperature": 0.2,
        "max_tokens": 1024,
        "repeat_penalty": 1.1,
        "presence_penalty": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8065/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=240))
    c = r["choices"][0]
    print("=== %s finish=%s ===" % (tag, c.get("finish_reason")))
    print(c["message"]["content"][:600])
    print()

call([{"role": "user", "content": OLD_TEMPLATE.format(text=tx)}], "旧 prompt_template（单 user 消息）")
