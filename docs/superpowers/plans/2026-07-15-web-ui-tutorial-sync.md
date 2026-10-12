# Web UI Tutorial Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the standalone Web UI tutorial match current controls and teach looped file recognition through a working local simulation.

**Architecture:** Keep `docs/web-ui-tutorial.html` self-contained. Rework its lane input rows and simulation state around the current Web UI’s input/loop visibility rules and incremental sentence-group result model, then expand the in-page guided tutorial to teach the new flow.

**Tech Stack:** Standalone HTML, CSS, browser JavaScript, Chromium, Node.js syntax checking.

## Global Constraints

- The tutorial remains a browser-only simulation and must not call `/api/*`, connect a WebSocket, load models, or capture microphone audio.
- `Loop audio` is visible for **Default files** and **Local WAV**, and hidden and reset to off for **Microphone**.
- While recognition is active, lane input controls are read-only.
- Keep at most 100 sentence groups per lane; delayed Qwen errors for removed sentences must not reappear.
- A loop-enabled file lane repeats until Stop; Stop cancels scheduled simulation work and returns to loaded.
- Do not introduce a new test suite.

---

## Task 1: Synchronize the simulated UI and recognition model

**Files:**

- Modify: `docs/web-ui-tutorial.html`

**Interfaces:**

- Produces: lane input objects with `kind`, `local_wav_path`, `microphone_device`, and `loop_audio` properties.
- Produces: `updateLaneInputVisibility(row, input)`, `renderLaneEvent(event)`, `resetRecognitionView()`, and simulation scheduling that later tutorial steps invoke through existing controls.

- [ ] **Step 1: Extend a lane input with the loop state and add the control.**

  Change the default input and row creation to use the current UI’s loop shape:

  ```js
  function defaultLaneInput() {
    return { kind: "default_files", local_wav_path: null, microphone_device: null, loop_audio: false };
  }

  const loop = document.createElement("label");
  loop.className = "loop-audio hidden";
  loop.innerHTML = '<input type="checkbox"> Loop audio';
  row.appendChild(loop);
  loop.querySelector("input").addEventListener("change", () => {
    input.loop_audio = loop.querySelector("input").checked;
  });
  ```

- [ ] **Step 2: Implement visibility and read-only rules.**

  Replace the row-local visibility closure with a shared helper and call it after input-kind changes and row creation:

  ```js
  function updateLaneInputVisibility(row, input) {
    row.querySelector(".local-wav-path").classList.toggle("hidden", input.kind !== "local_wav");
    row.querySelector(".local-wav-picker").classList.toggle("hidden", input.kind !== "local_wav");
    row.querySelector(".microphone-device").classList.toggle("hidden", input.kind !== "microphone");
    row.querySelector(".loop-audio").classList.toggle("hidden", input.kind === "microphone");
    row.querySelector(".loop-audio input").checked = input.loop_audio === true;
  }
  ```

  When changing to microphone, set `input.loop_audio = false`. In `updateControlAvailability`, compute `recognitionActive` from `starting`, `running`, and `stopping`, then disable lane row selects and buttons whenever it is true.

- [ ] **Step 3: Replace full result rebuilding with sentence-group incremental updates.**

  Give each lane `rows: new Map()` and `prunedSentenceId: -1`. Store transcript rows by `stream_id:sentence_id` and non-transcript rows by `stream_id:sentence_id:status`. Update or append one row per event. On more than 100 distinct sentence keys, remove every event and row belonging to the oldest sentence, delete its revision, and record its sentence id. Return `false` without rendering a `qwen_error` for an already pruned sentence.

  Use the following row update shape:

  ```js
  const row = lane.rows.get(storageKey) || document.createElement("div");
  if (!lane.rows.has(storageKey)) {
    lane.rows.set(storageKey, row);
    lane.results.appendChild(row);
  }
  row.className = `result ${event.status}`;
  row.textContent = isTranscriptEvent ? event.text : (event.text || event.status);
  ```

- [ ] **Step 4: Make simulation repeat only loop-enabled file lanes.**

  Schedule each lane’s sample sentences with an integer `cycle` and calculate the emitted sentence identifier as `cycle * sentences.length + sentenceIndex`. At the end of a loop-enabled non-microphone lane, schedule the next cycle only if the tutorial state is still running. For non-loop lanes, emit `lane_finished` once. Keep all timer ids in `simulation.timers`; Stop clears them, sets state to `loaded`, and renders status.

- [ ] **Step 5: Manually exercise the simulation in Chromium.**

  Open:

  ```bash
  chromium docs/web-ui-tutorial.html
  ```

  Confirm Default files and Local WAV display Loop audio; Microphone hides it and clears it; a loop-enabled file lane continues to emit repeated sentence cycles until Stop; and an ordinary lane completes back to loaded.

- [ ] **Step 6: Commit the synchronized simulation.**

  ```bash
  git add docs/web-ui-tutorial.html
  git commit -m "docs: sync web UI tutorial simulation"
  ```

## Task 2: Refresh the guided tutorial and verify standalone delivery

**Files:**

- Modify: `docs/web-ui-tutorial.html`

**Interfaces:**

- Consumes: loop-enabled lane controls and simulation behavior from Task 1.
- Produces: updated `STEPS` content for the in-page previous/next navigation.

- [ ] **Step 1: Update guided content to current user flow.**

  Keep the existing model, lane, path-picker, microphone, start/stop, and results guidance. Add a separate loop exercise with this content:

  ```text
  循环文件识别：在 Default files 或 Local WAV 输入上勾选 Loop audio，开始识别后样例会持续重复，适合连续观察或压测。点击 Stop Recognition 结束循环。麦克风是实时输入，因此不会显示此选项。
  ```

  State in the start/stop step that active recognition locks lane input controls, and state in the result step that the page updates sentence rows incrementally and retains the newest 100 sentences per lane.

- [ ] **Step 2: Run a standalone JavaScript syntax check.**

  Extract the inline script and check it with Node without executing browser APIs:

  ```bash
  SCRIPT=$(mktemp --suffix=.js)
  sed -n '/<script>/,/<\/script>/p' docs/web-ui-tutorial.html | sed '1d;$d' > "$SCRIPT"
  node --check "$SCRIPT"
  rm -f "$SCRIPT"
  ```

  Expected: exit code 0 with no syntax output.

- [ ] **Step 3: Check markup and diff hygiene.**

  ```bash
  rg -n 'Loop audio|循环文件识别|updateLaneInputVisibility|prunedSentenceId|loop_audio' docs/web-ui-tutorial.html
  git diff --check
  ```

  Expected: all required content is present and no whitespace errors are reported.

- [ ] **Step 4: Commit the tutorial guidance.**

  ```bash
  git add docs/web-ui-tutorial.html
  git commit -m "docs: explain looped file recognition"
  ```
