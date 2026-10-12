# Web UI Tutorial Style Synchronization Design

## Goal

Synchronize `docs/web-ui-tutorial.html` with the visual language of
`src/multi_stream_asr/web/static/index.html` while preserving the tutorial as a
standalone, offline learning page.

## Scope

The tutorial’s simulated application area adopts the current Web UI’s color
tokens, typography, toolbar treatment, control hierarchy, configuration-card
appearance, lane cards, result rows, and path-picker styling. Its existing
simulation and guided learning behavior do not change.

The tutorial panel remains a separate teaching surface because the production
page has no equivalent. It uses the same colors, borders, button treatment,
and typography as the application area while retaining its step navigation and
readable content structure.

## Style Model

The tutorial defines and uses the production palette:

- `--accent: #b0e237`, `--accent-soft: #edffdc`, and
  `--accent-deep: #5d7d00` for primary actions and selected teaching states.
- `--canvas: #f5f6f2`, `--surface: #ffffff`, `--ink: #171b16`,
  `--muted: #697065`, and `--border: #dce1d7` for page surfaces and content.
- Inter/Segoe UI/Arial typography, the production button focus ring and
  disabled-state treatment, compact controls, lane accent rails, and subtle
  card shadows.

The simulated app receives the dark toolbar and bright primary Load/Start
actions used by the current UI. Configuration rows, path picker, lane cards,
and result states use the current UI’s spacing, radius, borders, and status
colors. The tutorial panel must remain visually subordinate to the simulation
and avoid creating a second competing dark toolbar.

## Responsiveness and Accessibility

Keep the existing narrow-screen layout: the simulated app and tutorial panel
stack vertically. Reuse the current UI’s visible focus treatment and preserve
all existing labels, button text, and semantic HTML.

## Verification

- Compare the tutorial CSS selectors with the corresponding production CSS
  rules to confirm the shared visual tokens and component treatments.
- Run a Node syntax check on the extracted tutorial script; styling changes
  must not break its standalone simulation.
- Open the file in Chromium when available to inspect both desktop and narrow
  layouts. If Chromium is unavailable, document that boundary and verify the
  HTML/CSS source plus script syntax locally.
