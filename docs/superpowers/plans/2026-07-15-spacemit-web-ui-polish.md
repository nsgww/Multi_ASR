# Spacemit Web UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply a light Spacemit-branded visual system and local official logo to the Multi-Stream ASR web console without changing its workflow.

**Architecture:** Retain the single-page inline CSS/JavaScript architecture. Add a narrowly scoped endpoint for the package-local SVG, add a compact brand block before the existing toolbar controls, and replace repeated neutral color values with a small Spacemit token set while preserving all existing element IDs and state logic.

**Tech Stack:** Python 3.13, FastAPI, inline CSS/HTML/JavaScript, pytest.

## Global Constraints

- Use the official SVG colors `#B0E237`, `#C0E767`, and `#D0EE90` as the accent family.
- Do not introduce external frontend dependencies, network asset requests, fonts, or animations.
- Preserve every existing DOM ID, JavaScript hook, server route, dialog behavior, and responsive lane-count layout.
- Keep visible keyboard focus and readable semantic transcript/error states.
- Package the SVG with the Python distribution and serve it from the application.

## File Map

- Create: `src/multi_stream_asr/web/static/spacemit-logo.svg` — official local mark.
- Modify: `src/multi_stream_asr/web/server.py` — serve the SVG alongside the index.
- Modify: `src/multi_stream_asr/web/static/index.html` — brand block and compact visual-system CSS.
- Modify: `pyproject.toml` — include SVG package data.
- Modify: `tests/test_web_server.py` — verify the served HTML references the local logo and its SVG response is available.

---

### Task 1: Package and Serve the Local Brand Asset

**Files:**
- Create: `src/multi_stream_asr/web/static/spacemit-logo.svg`
- Modify: `src/multi_stream_asr/web/server.py:440-443`
- Modify: `pyproject.toml:25`
- Test: `tests/test_web_server.py:939-956`

**Interfaces:**
- Produces: `GET /spacemit-logo.svg` returning `image/svg+xml` from `web/static/spacemit-logo.svg`.
- Consumes: the existing `FileResponse` import and static directory adjacent to `server.py`.

- [ ] **Step 1: Write a failing static-asset test**

  Add this test immediately before `test_index_html_contains_lane_selector_and_controls`:

  ```python
  def test_index_html_references_and_serves_local_spacemit_logo(tmp_path: Path):
      with TestClient(create_app(output_dir=tmp_path)) as client:
          index = client.get("/")
          logo = client.get("/spacemit-logo.svg")

      assert index.status_code == 200
      assert 'src="/spacemit-logo.svg"' in index.text
      assert logo.status_code == 200
      assert logo.headers["content-type"].startswith("image/svg+xml")
      assert '#B0E237' in logo.text
  ```

- [ ] **Step 2: Run the focused test and verify RED**

  Run:

  ```bash
  source .venv313/bin/activate
  pytest tests/test_web_server.py::test_index_html_references_and_serves_local_spacemit_logo -v
  ```

  Expected: FAIL because the page has no local logo reference or route.

- [ ] **Step 3: Add the official SVG, package pattern, and response route**

  Copy the official three-path SVG mark into `src/multi_stream_asr/web/static/spacemit-logo.svg`. Change the package-data declaration to:

  ```toml
  "multi_stream_asr.web" = ["static/*.html", "static/*.svg"]
  ```

  Define the shared static directory and route adjacent to the existing root route:

  ```python
  static_directory = Path(__file__).with_name("static")

  @app.get("/")
  async def index() -> FileResponse:
      return FileResponse(static_directory / "index.html")

  @app.get("/spacemit-logo.svg", include_in_schema=False)
  async def spacemit_logo() -> FileResponse:
      return FileResponse(static_directory / "spacemit-logo.svg", media_type="image/svg+xml")
  ```

- [ ] **Step 4: Run the focused test and verify GREEN**

  Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit the asset delivery change**

  ```bash
  git add pyproject.toml src/multi_stream_asr/web/server.py src/multi_stream_asr/web/static/spacemit-logo.svg tests/test_web_server.py
  git commit -m "feat: serve local spacemit logo"
  ```

### Task 2: Apply the Compact Spacemit Visual System

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html:7-235`
- Test: `tests/test_web_server.py:939-956`

**Interfaces:**
- Consumes: `/spacemit-logo.svg` from Task 1.
- Preserves: `#load-models`, `#unload-models`, `#start-recognition`, `#stop-recognition`, all configuration controls, dialog IDs, and JavaScript functions unchanged.
- Produces: a `.brand` toolbar child and CSS custom properties (`--accent`, `--accent-soft`, `--surface`, `--ink`, `--muted`, `--border`) used by controls and panels.

- [ ] **Step 1: Extend the static HTML assertion before styling**

  In `test_index_html_references_and_serves_local_spacemit_logo`, add:

  ```python
      assert 'class="brand"' in index.text
      assert 'class="brand-copy"' in index.text
      assert '--accent: #b0e237' in index.text.lower()
      assert '.lane::before' in index.text
  ```

- [ ] **Step 2: Run the focused test and verify RED**

  Run the Step 2 command from Task 1. Expected: FAIL because the brand block and design tokens are absent.

- [ ] **Step 3: Add minimal token-based CSS and brand markup**

  Replace the root token section with the following base, then update the existing rules to consume the tokens rather than duplicate neutral hex values:

  ```css
  :root {
    color-scheme: light;
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    --accent: #b0e237;
    --accent-soft: #edffdc;
    --accent-deep: #5d7d00;
    --canvas: #f5f6f2;
    --surface: #ffffff;
    --ink: #171b16;
    --muted: #697065;
    --border: #dce1d7;
  }
  ```

  Insert before the existing `.control-group` in `.toolbar`:

  ```html
  <div class="brand">
    <img src="/spacemit-logo.svg" alt="Spacemit" width="27" height="34" />
    <span class="brand-copy"><strong>Spacemit</strong><small>Multi-Stream ASR</small></span>
  </div>
  ```

  Add `.brand`, `.brand-copy`, `.toolbar`, `button`, `button:focus-visible`, `.lane`, `.lane::before`, `.result`, and small-screen rules so the toolbar becomes graphite, primary controls use the lime accent, cards gain a 1px token border/8px radius/shadow, lanes receive a 3px accent edge, and the brand wraps cleanly below 800px. Keep `.fatal_error` red and all existing `hidden`, grid, and dialog behavior unchanged.

- [ ] **Step 4: Run regression tests and inspect the rendered structure**

  Run:

  ```bash
  source .venv313/bin/activate
  pytest tests/test_web_server.py -v
  python -m compileall -q src
  ```

  Expected: all web-server tests pass and compilation exits zero. Start the app with `python -m multi_stream_asr.main --host 127.0.0.1 --port 8000 --fake` and verify at desktop and 800px mobile widths that buttons, controls, four lane cards, the dialog, and the green focus ring remain visible without horizontal overflow.

- [ ] **Step 5: Commit the visual polish**

  ```bash
  git add src/multi_stream_asr/web/static/index.html tests/test_web_server.py
  git commit -m "feat: polish web ui with spacemit styling"
  ```
