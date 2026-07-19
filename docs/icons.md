# Icon registry

The single place that tracks every icon the UI wants. The editor is
CapCut-shaped and icon-heavy, but a lot of controls still ship a **text or
single-character placeholder** (`"Split"`, `"B"`, `"✕"`, `"^"`, …) where a
real glyph belongs. This file is the to-do list that turns those into art.

## Workflow

When you build UI and reach for an icon that doesn't exist yet, **don't
block on it**:

1. Ship the control now with a short text/char placeholder, matching the
   existing pattern for that widget (`ToolButton { label: "Split" }`,
   `HeadToggle { label: "L" }`, a `CutlassText { text: "✕" }`, …).
2. **Register it here** under the right section, newest first, in the
   registry format below.
3. Later, someone fetches the SVG, drops it in the icon folder, and swaps
   the placeholder for an `Image` — then flips the entry to `[x]`.

This keeps features moving while leaving a precise, fetchable shopping list
behind. The same loop is codified for the agent in
`.cursor/rules/icons.mdc`.

### Registry format

```
- [ ] `lucide-name` — placeholder `"X"` — `path/to/file.slint` — what it does.
```

- `[ ]` = needed (placeholder live in the UI) · `[x]` = fetched + wired in.
- `lucide-name` is the intended icon (see *Source* below). If unsure, give
  the closest name and a note.
- Always include the **placeholder string** and the **file** so it's
  trivial to find and replace.

## Where icons live

All UI icons live under the **single** tracked root
`assets/icon/` (transport in `icon/`, library glyphs in
`icon/library/`, and Lucide/Tabler fetches under `shell/`, `launch/`,
`timeline/`, `text/`, `inspector/`, `ui/`). Reference them from Slint with
`@image-url` paths relative to each `.slint` file (see `app.slint` and
panel files). Media scratch files for local dev and tests live in
gitignored `local-assets/assets/` (`frames/`, `proxy/` stay ignored too).
The dock icon is also loaded from `assets/icon/` via `include_bytes!` in
`src/main.rs`.

Loaded via `@image-url(...)` relative to the `.slint` file, then tinted with
`colorize:` so one SVG works across themes.

## Source

Primary: **[Lucide](https://lucide.dev)** (MIT, single-stroke, matches the
existing line look — keep the 2px default stroke). Fallback for the few it
lacks cleanly (`letter-spacing`, `line-height`): **[Tabler](https://tabler.io/icons)** (MIT).

## Already shipped

`play`, `pause`, `fullscreen` (preview transport) · library tabs/sections
`media`, `audio`, `text`, `stickers`, `effects`, `transitions`, `stock`,
`ai`, `sfx`, `filters`, `adjustment` · logo `cutlass.png` /
`cutlass-in-app.png` · chat `send` / `circle-stop`.

---

## Registry

### Window controls — `shell/title-bar.slint`

- [x] `minus` — `assets/icon/shell/minus.svg` — minimize.
- [x] `square` — `assets/icon/shell/square.svg` — maximize.
- [x] `copy` — `assets/icon/shell/copy.svg` — restore (when maximized).
- [x] `x` — `assets/icon/shell/x.svg` — close.
- [x] `sparkles` — `assets/icon/shell/sparkles.svg` — AI assistant dock toggle.
- [x] `settings` — `assets/icon/shell/settings.svg` — settings gear.
- [ ] `upload` — placeholder `"Export"` — `shell/title-bar.slint` — export action stays worded (`AccentButton`).
- [x] (logo) — `assets/icon/cutlass-in-app.png` — brand mark.

### Start screen — `launch.slint`

- [x] `plus` — `assets/icon/launch/plus.svg` — New project tile mark.
- [x] `folder-open` — `assets/icon/launch/folder-open.svg` — Open project tile mark.
- [x] `clapperboard` — `assets/icon/launch/clapperboard.svg` — recent-project thumb chip.
- [x] window controls — reuse `assets/icon/shell/{minus,square,copy,x}.svg`.

### Timeline toolbar — `panels/timeline/toolbar.slint`

- [x] `undo-2` — `assets/icon/timeline/undo-2.svg`.
- [x] `redo-2` — `assets/icon/timeline/redo-2.svg`.
- [x] `scissors` — `assets/icon/timeline/scissors.svg` — split at playhead.
- [x] `flag` — `assets/icon/timeline/flag.svg` — add marker.
- [x] `trash-2` — `assets/icon/timeline/trash-2.svg`.
- [x] `audio-lines` — `assets/icon/timeline/audio-lines.svg` — extract audio.
- [x] `repeat` — `assets/icon/timeline/repeat.svg` — loop.
- [x] `magnet` — `assets/icon/timeline/magnet.svg` — main-track gapless magnet.
- [x] `grid-3x3` — `assets/icon/timeline/grid-3x3.svg` — auto-snap (distinct from Magnet).
- [x] `link` — `assets/icon/timeline/link.svg`.
- [x] `unlink` — `assets/icon/timeline/unlink.svg`.
- [x] `scan` — `assets/icon/timeline/scan.svg` — zoom to fit.
- [x] `zoom-out` — `assets/icon/timeline/zoom-out.svg`.
- [x] `zoom-in` — `assets/icon/timeline/zoom-in.svg`.

### Track headers — `panels/timeline/track-head.slint`

- [x] `eye` / `eye-off` — `assets/icon/timeline/eye.svg` · `eye-off.svg`.
- [x] `volume-2` / `volume-x` — `assets/icon/timeline/volume-2.svg` · `volume-x.svg`.
- [x] `mic` — `assets/icon/timeline/mic.svg` — voice / duck source.
- [x] `lock` / `lock-open` — `assets/icon/timeline/lock.svg` · `lock-open.svg`.

### Text inspector — `panels/inspector/text-inspector.slint`

- [x] `bold` — `assets/icon/text/bold.svg`.
- [x] `underline` — `assets/icon/text/underline.svg`.
- [x] `italic` — `assets/icon/text/italic.svg`.
- [x] `case-upper` — `assets/icon/text/case-upper.svg`.
- [x] `case-lower` — `assets/icon/text/case-lower.svg`.
- [x] `case-sensitive` — `assets/icon/text/case-sensitive.svg`.
- [x] `align-left` / `align-center` / `align-right` — `assets/icon/text/`.
- [x] `align-start-vertical` / `align-center-vertical` / `align-end-vertical` — `assets/icon/text/`.
- [x] `wrap-text` — `assets/icon/text/wrap-text.svg`.
- [x] `letter-spacing` (Tabler) — `assets/icon/text/letter-spacing.svg`.
- [x] `line-height` (Tabler) — `assets/icon/text/line-height.svg`.
- [ ] keyframe in/out icons — placeholder `"|<" "+" ">|" "T" "B"` — disabled animation row (lower priority).

### Inspector (general)

- [ ] `chevron-up` / `chevron-down` — placeholder `"^"` — section collapse caret (`inspector/inspector-widgets.slint`, `inspector/transform-inspector.slint`) — assets ready at `assets/icon/inspector/`.
- [x] `spline` — `assets/icon/inspector/spline.svg` — keyframe easing.
- [ ] `scan` + `expand` — placeholder `"Fit"` / `"Fill"` — transform fit/fill stays worded (`SubtleButton`); assets ready at `assets/icon/inspector/`.
- [ ] `trash-2` — placeholder `"Remove"` — remove effect stays worded; asset ready at `assets/icon/inspector/trash-2.svg`.
- [x] `flip-horizontal` / `flip-vertical` — `assets/icon/inspector/`.

### Dropdowns & pickers

- [x] `chevron-down` — `assets/icon/ui/chevron-down.svg` — dropdown / color-swatch / look / animation pickers.

### Library & tiles

- [x] `folder-plus` — `assets/icon/library/folder-plus.svg` — import button.
- [x] `wand-sparkles` — `assets/icon/library/wand-sparkles.svg` — effect/transition tile glyph.
- [x] `image` — `assets/icon/library/image.svg` — still-image badge.
- [x] `triangle-alert` — `assets/icon/library/triangle-alert.svg` — missing-media badge.

### Misc

- [x] `x` — `assets/icon/ui/x.svg` — transition remove.
- [x] `check` — `assets/icon/ui/check.svg` — agent dry-run checkbox; gallery dropdown selected row (`crates/cutlass-ui-gallery/ui/components/dropdown.slint`).
- [x] `chevron-down` — `assets/icon/ui/chevron-down.svg` — gallery dropdown trigger.
- [x] `send` — `assets/icon/chat/send.svg` — agent submit.
- [x] `circle-stop` — `assets/icon/chat/stop.svg` — agent cancel.

### Fine as text (no icon needed)

Timecode `/` separators, the zoom `%` readout, and word buttons in dialogs
(Browse… / Cancel / Export / Done / OK / Locate… / New project / etc.).
Preview / track-height cycle labels on the timeline toolbar stay worded.
