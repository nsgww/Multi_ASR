# Spacemit Web UI Polish Design

## Goal

Give the existing multi-stream ASR control page a compact Spacemit visual
identity without changing its controls, data flow, or responsive lane layout.

## Visual system

- Use the Spacemit logo greens as the only accent family: `#B0E237` as the
  primary action color, with `#C0E767` and `#D0EE90` for restrained highlights.
- Keep the working surface light and warm, with graphite text and borders for
  readable transcripts and a professional control-console feel.
- Define a small set of CSS custom properties for surface, ink, muted text,
  border, accent, accent-soft, and focus colors. Existing rules consume these
  properties rather than duplicating color declarations.

## Layout and components

- Add a compact brand block to the left of the existing toolbar controls. It
  contains a local copy of the official SVG mark, the product name, and a small
  `MULTI-STREAM ASR` descriptor. It wraps naturally with the existing toolbar
  on narrow screens.
- Turn the toolbar into the page's dark graphite header. Primary workflow
  buttons use the green accent; secondary actions remain outlined.
- Give the status row, configuration area, input cards, lane cards, dialog, and
  result rows consistent light surfaces, hairline borders, modest corner radii,
  and subtle shadows. Lane headers receive an accent indicator; transcript
  state colors remain semantically distinct.
- Add keyboard-visible focus styles and intentional hover/disabled treatment;
  do not use animation or external font dependencies.

## Asset handling

Store the official SVG mark as `src/multi_stream_asr/web/static/spacemit-logo.svg`
and reference it with a relative `<img>` path. The page remains usable when it
is served without Internet access.

## Compatibility and validation

- Preserve all current DOM IDs, JavaScript hooks, labels, dialog behavior, and
  lane-grid breakpoints.
- Keep all styling in the existing inline `<style>` element and add no frontend
  packages.
- Verify the existing web-server tests and inspect the served page at desktop
  and mobile widths for overflow, contrast, and focus visibility.
