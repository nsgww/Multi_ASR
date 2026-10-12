# Restore Zipformer Intra Thread Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore one Zipformer EP intra thread per lane, then benchmark Qwen3-ASR 0.6B at 2/4 lanes and 1.7B at 2 lanes with three real K3 runs each.

**Architecture:** Keep the per-lane provider config, X100 architecture preference, and global intra pool. Replace only the lane-count-dependent value with `1`, validate the 4-lane drop regression, then collect the fixed matrix under one controlled K3 environment.

**Tech Stack:** Python 3.13, pytest, sherpa-onnx, SpaceMIT EP, K3 `.venv313`, Qwen3-ASR shared server.

## Global Constraints

- All supported lane counts write `SPACEMIT_EP_INTRA_THREAD_NUM = 1`.
- Preserve `SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064` and `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1`.
- Use only K3 project `.venv313`; do not install dependencies.
- Before each real run, remove prior test lane/Qwen processes; restore the Web service after validation.
- Use `~/Music/wav16k`, `n_parallel=4`, `n_ctx=8192`, three sequential runs per matrix group.
- Report each artifact path, online avg/max, offline RTF avg/max, total/per-lane drops, and three-run arithmetic averages.

---

## File Structure

- `src/multi_stream_asr/asr/online/zipformer.py`: write the provider config.
- `tests/test_zipformer.py`: assert the generated intra thread value for every lane count.
- `tests/test_lane_runtime.py`: assert runtime-created config content.
- `docs/reports/2026-07-13-qwen3-post-intra-restore-performance.md`: raw K3 samples and averages.

### Task 1: Fix the provider config

**Files:**
- Modify: `tests/test_zipformer.py:70-90`
- Modify: `tests/test_lane_runtime.py:220-245`
- Modify: `src/multi_stream_asr/asr/online/zipformer.py:116-135`

**Produces:** `write_zipformer_provider_config(..., lanes: int) -> Path` always writes `SPACEMIT_EP_INTRA_THREAD_NUM = 1` while retaining its existing API and config path.

- [ ] **Step 1: Write a failing test**

Replace the lane/thread parametrization in `tests/test_zipformer.py` with:

```python
@pytest.mark.parametrize("lanes", [2, 4, 6, 8])
def test_write_zipformer_provider_config_uses_one_intra_thread_for_all_lane_counts(
    tmp_path: Path, lanes: int
):
    path = write_zipformer_provider_config(
        runtime_dir=tmp_path / "runtime", stream_id=3, lanes=lanes
    )
    assert "SPACEMIT_EP_INTRA_THREAD_NUM = 1\\n" in path.read_text(encoding="utf-8")
```

Change the runtime integration expectation in `tests/test_lane_runtime.py` from `2` to `1`.

- [ ] **Step 2: Verify RED**

```bash
PATH="$PWD/.venv313/bin:$PATH" PYTHONPATH=. .venv313/bin/python -m pytest tests/test_zipformer.py tests/test_lane_runtime.py -q
```

Expected: the 2-lane and 4-lane expectations fail because the generated text contains `2`.

- [ ] **Step 3: Minimal implementation**

Replace the implementation assignment with:

```python
intra_threads = 1
```

Keep `lanes` in the function signature for compatibility.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused command, then `pytest -q`; commit the three changed files with message `fix: use one zipformer intra thread per lane`.

### Task 2: Validate the restored 4-lane path on K3

**Files:**
- Create: `docs/reports/2026-07-13-qwen3-post-intra-restore-performance.md`

**Produces:** one real 4-lane 0.6B q40 artifact whose four configs contain intra=1 and whose total drops are below 10.

- [ ] **Step 1: Sync and confirm config**

Sync Task 1 files to K3. Start one real 4-lane 0.6B q40 run with `~/Music/wav16k`, `n_parallel=4`, and `n_ctx=8192`; inspect all four generated provider configs and require `SPACEMIT_EP_INTRA_THREAD_NUM = 1`.

- [ ] **Step 2: Verify run artifact**

After natural completion, read `metrics.json`; require all lanes are `finished`, Zipformer/Qwen sentence-ID sets match per lane, and total drops are below 10. Record online avg/max and offline RTF avg/max in the report.

### Task 3: Run the three-by-three real matrix

**Files:**
- Modify: `docs/reports/2026-07-13-qwen3-post-intra-restore-performance.md`

**Produces:** three raw runs plus one arithmetic-average row for each group: 0.6B/2 lanes, 0.6B/4 lanes, 1.7B/2 lanes.

- [ ] **Step 1: Execute the matrix**

For each group, perform three sequential runs. Before every run assert no `multiprocessing.spawn` or `llama-multi-asr` test process remains. Use the current code, local WAV input, `n_parallel=4`, and `n_ctx=8192`.

- [ ] **Step 2: Validate and summarize each artifact**

For every run require final lane states, aligned sentence IDs, and existing results/model outputs. Save its absolute artifact path, wall time, online avg/max, offline RTF avg/max, and drops. Compute unweighted three-run means for the report.

- [ ] **Step 3: Final verification and commit**

Run local full `pytest -q`, verify K3 `/health` after restoring the Web service, cross-check reported means against the nine `metrics.json` files, then commit the report as `docs: report post-restore qwen3 performance`.
