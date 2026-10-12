# K3 User Onboarding Lark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a plain-language SpaceMIT K3 setup and usage guide to the specified Lark wiki page.

**Architecture:** Preserve the target page's existing resources, then replace only its instructional body with XML blocks. The guide uses one K3/Bianbu path, a short launch flow, an explicit HTML tutorial entry point, and user-facing troubleshooting.

**Tech Stack:** Lark CLI user identity, XML document blocks, Bash, Python 3.13, SpaceMIT K3/Bianbu.

## Global Constraints

- Cover only SpaceMIT K3 / Bianbu (RISC-V).
- Write for a user without a technical background; do not split by role.
- Do not include project test steps or implementation details.
- Use `zipformer-streaming.tar.gz` through `scripts/setup.sh` rather than the deprecated `zipformer.tar.gz` resource.
- Preserve existing page resources unless the page is intentionally empty.

---

### Task 1: Inspect the target page and publish the guide

**Files:**
- Modify: Lark wiki page `MPCaw7KxmiBKxck8Rauc5Xr6nee`

**Interfaces:**
- Consumes: `scripts/setup.sh`, `README.md`, `docs/web-ui-tutorial.html`.
- Produces: a step-by-step K3 onboarding page with deployment, browser use, tutorial entry, troubleshooting, and a completion checklist.

- [ ] **Step 1: Fetch the target page in full XML form**

Run: `lark-cli docs +fetch --doc "https://spacemit.feishu.cn/wiki/MPCaw7KxmiBKxck8Rauc5Xr6nee?from=from_copylink" --detail full --as user`

Expected: the current title, blocks, and any embedded resources are visible before editing.

- [ ] **Step 2: Write the guide using XML blocks**

Use `lark-cli docs +update` with block-level updates when the existing page has content to preserve; otherwise use a single XML overwrite. Include the exact commands for transferring source, installing prerequisites, running `scripts/setup.sh`, starting `python -m multi_stream_asr.main --host 0.0.0.0 --port 8000`, and opening the HTML tutorial.

- [ ] **Step 3: Fetch the page and verify its key instructions**

Run: `lark-cli docs +fetch --doc "https://spacemit.feishu.cn/wiki/MPCaw7KxmiBKxck8Rauc5Xr6nee?from=from_copylink" --detail simple --as user`

Expected: it contains K3/Bianbu, `scripts/setup.sh`, port `8000`, `docs/web-ui-tutorial.html`, the tutorial order, and troubleshooting; it contains no test command.

- [ ] **Step 4: Commit the plan record**

Run: `git add docs/superpowers/plans/2026-07-14-k3-user-onboarding-lark.md && git commit -m "docs: plan K3 user onboarding wiki guide"`

Expected: the local plan is committed without staging unrelated files.
