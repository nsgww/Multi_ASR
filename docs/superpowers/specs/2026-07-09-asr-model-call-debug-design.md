# ASR Model Call Debug Design Spec

## Purpose

Fix the current real-run recognition correctness issue by first eliminating model invocation mismatches. The application must call sherpa-onnx Zipformer and Qwen3-ASR in the same request style as the known-good reference project, print model recognition output during real runs, and validate that 4-lane headless recognition produces non-empty, content-correct final text.

## Scope

In scope:

- Align `Qwen3AsrClient` request construction, WAV encoding, response parsing, and slot cleanup with `/home/liangff/Workspace/projects/4-asr-python/asr4/qwen_client.py`.
- Align `ZipformerRecognizer` model file selection and result extraction with `/home/liangff/Workspace/projects/4-asr-python/asr4/zipformer_stream.py`.
- Force Zipformer to use the corresponding int8 model files where available: `encoder-epoch-99-avg-1.int8.onnx`, `decoder-epoch-99-avg-1.onnx`, `joiner-epoch-99-avg-1.int8.onnx`, and `tokens.txt`.
- Add model-output diagnostics so real runs print enough Qwen and Zipformer output to prove whether each model is returning correct text.
- Persist headless model-output diagnostics into the run directory so correctness can be inspected after a run.
- Update tests to lock the corrected request payload, int8 model selection, parsing, and diagnostic output behavior.
- Run local tests and remote K3 Python 3.13 validation, including real 4-lane headless recognition.

Out of scope:

- Changing frontend `pending` or `final` replacement semantics.
- Changing endpointing rules or segment cache policy.
- Adding Python-side explicit CPU binding. Sherpa-onnx and `llama-server` remain responsible for binding; this project only passes configuration.
- Rewriting model internals or `llama-server`.

## Runtime Constraints

- Remote development, validation, and inference must use Python 3.13 and the project-local `.venv313`.
- Remote dependency installation must happen only after activating `.venv313`.
- Remote dependency installation must export the Spacemit and Aliyun package index variables first.
- `sherpa-onnx` is available from `https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple`.
- Real validation target remains 4 lanes with headless mode.
- Target metrics remain online frame latency about 50 ms and whole-segment offline RTF about 0.4; correctness is evaluated before performance tuning.

## Model Invocation Design

### Qwen3-ASR

`Qwen3AsrClient` will build the chat completions payload using the known-good OpenAI-style audio content block:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a speech recognition model."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-wav>",
            "format": "wav"
          }
        },
        {
          "type": "text",
          "text": ""
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "id_slot": 0
}
```

The online prompt text remains available to the `recognize(pcm, prompt)` interface but is not inserted into the Qwen text block for this fix, matching the reference implementation. The response parser continues to accept `choices[0].message.content`, top-level `text`, and top-level `content`, strips `<asr_text>` tags, and treats an empty cleaned string as a valid final response.

The client will print one concise diagnostic line for each completed Qwen request, including lane or port, audio seconds, elapsed milliseconds, RTF, cleaned text length, and a short cleaned-text preview. When debug persistence is enabled by headless mode, it will also write per-request records containing raw response JSON, cleaned text, timing, and audio duration.

### Zipformer

`ZipformerRecognizer` will prefer exact reference model files:

- `encoder-epoch-99-avg-1.int8.onnx`
- `decoder-epoch-99-avg-1.onnx`
- `joiner-epoch-99-avg-1.int8.onnx`
- `tokens.txt`

If those files are missing, construction fails with a clear `FileNotFoundError` rather than silently selecting a different model via a broad glob. Result extraction will match the reference behavior: if `get_result(stream)` returns an object with a `text` attribute, use that text; otherwise convert the result to `str`.

The recognizer will print endpoint diagnostics when a non-empty endpoint is detected, including lane, timestamp, and endpoint text preview. Partial diagnostics remain limited to avoid flooding stdout.

## Diagnostic Persistence

Headless real runs save final transcripts and model-output diagnostics under the generated run directory using per-lane files. The old aggregate `results.json` and `model_outputs.jsonl` files are not written.

Required artifact:

- `results/lane<N>.txt`: final complete transcript for lane `N`, one final sentence per line.
- `model_outputs/lane<N>.jsonl`: one compact JSON object per model output diagnostic for lane `N`.

Required `model_outputs/lane<N>.jsonl` fields:

- `model`: `qwen3-asr` or `zipformer`.
- `stream_id`: integer lane ID.
- `sentence_id`: integer sentence ID.
- `audio_seconds`: number for Qwen requests, null for Zipformer diagnostics.
- `rtf`: number for Qwen requests, null for Zipformer diagnostics.
- `text`: cleaned model output text.

## Validation Design

Unit tests must prove:

- Qwen WAV encoding matches 16 kHz mono PCM16 WAV.
- Qwen request body uses `messages[].content[].input_audio`, `system`, `temperature=0`, `max_tokens=256`, and `id_slot=0`.
- Qwen slot cleanup posts to `/slots/0?action=erase`.
- Qwen parser accepts supported response shapes and valid empty cleaned text.
- Zipformer selects the exact int8 encoder and joiner filenames and the expected decoder filename.
- Zipformer extracts `result.text` when `get_result()` returns an object.
- Diagnostic records are printed or written without altering ASR event semantics.

Remote validation must run:

```bash
. .venv313/bin/activate
python -m pytest -v
python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k
```

Expected real-run result:

- All 4 lanes reach `finished`.
- No frame drops in the normal WAV validation run.
- `results/lane<N>.txt` has the final complete transcript for each lane.
- `model_outputs/lane<N>.jsonl` contains compact model output records for each lane.
- Final text is compared against `/home/liangff/Workspace/projects/4-asr-python/run_logs/asr-results-20260709-142934/*_final.txt` for sanity, not byte-identical equality.
- `online_frame_latency.average_ms` and `offline_rtf.average` are reported.
- No app-started `llama-server` process remains after unload.

## Risks

- The Qwen payload mismatch is the likely root cause, but a segmenting or audio-cache issue could still affect correctness after the request body is fixed.
- Persisting full raw Qwen responses may create large run logs; the diagnostic file is acceptable for this debugging phase and can be reduced later.
- Failing fast on missing exact int8 Zipformer files is intentional because silent fallback can mask model mismatch during validation.
