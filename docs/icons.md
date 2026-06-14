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
`crates/cutlass-ui/ui/assets/icon/` (transport in `icon/`, library glyphs in
`icon/library/`). `.gitignore` ignores `assets/` everywhere **except** this
dir (`!crates/cutlass-ui/ui/assets/`), so anything dropped here commits
normally — drop new icons in and reference them with a relative `@image-url`.
Only the media scratch dirs (repo-root `assets/`, `frames/`, `proxy/`) stay
ignored. The dock icon is also loaded from here via `include_bytes!` in
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
`cutlass-in-app.png`.

---

## Registry

### Window controls — `shell/title-bar.slint`

- [ ] `minus` — placeholder `"─"` — `shell/title-bar.slint` — minimize.
- [ ] `square` — placeholder `"□"` — `shell/title-bar.slint` — maximize.
- [ ] `copy` — placeholder `"❐"` — `shell/title-bar.slint` — restore (when maximized).
- [ ] `x` — placeholder `"✕"` — `shell/title-bar.slint` — close.
- [ ] `sparkles` — placeholder `"Assistant"` — `shell/title-bar.slint` — AI assistant dock toggle.
- [ ] `upload` — placeholder `"Export"` — `shell/title-bar.slint` — export action (AccentButton).
- [ ] (logo) — placeholder `"C"` — `shell/title-bar.slint` — brand mark; use the existing logo, not a letter.

### Timeline toolbar — `panels/timeline/toolbar.slint`

- [ ] `undo-2` — placeholder `"Undo"` — `panels/timeline/toolbar.slint`.
- [ ] `redo-2` — placeholder `"Redo"` — `panels/timeline/toolbar.slint`.
- [ ] `scissors` — placeholder `"Split"` — `panels/timeline/toolbar.slint` — split at playhead.
- [ ] `flag` — placeholder `"Marker"` — `panels/timeline/toolbar.slint` — add marker.
- [ ] `trash-2` — placeholder `"Delete"` — `panels/timeline/toolbar.slint`.
- [ ] `repeat` — placeholder `"Loop"` — `panels/timeline/toolbar.slint`.
- [ ] `magnet` — placeholder `"Magnet"` — `panels/timeline/toolbar.slint` — main-track gapless magnet.
- [ ] `magnet` (variant — must read different from Magnet) — placeholder `"Snap"` — `panels/timeline/toolbar.slint` — auto-snap toggle.
- [ ] `link` — placeholder `"Link"` — `panels/timeline/toolbar.slint`.
- [ ] `unlink` — placeholder `"Unlink"` — `panels/timeline/toolbar.slint`.
- [ ] `scan` — placeholder `"Fit"` — `panels/timeline/toolbar.slint` — zoom to fit.
- [ ] `zoom-out` — placeholder `"−"` — `panels/timeline/toolbar.slint`.
- [ ] `zoom-in` — placeholder `"+"` — `panels/timeline/toolbar.slint`.

### Track headers — `panels/timeline/track-head.slint`

- [ ] `eye` / `eye-off` — placeholder `"V"` — `panels/timeline/track-head.slint` — visibility (visual lanes).
- [ ] `volume-2` / `volume-x` — placeholder `"M"` — `panels/timeline/track-head.slint` — mute (audio lanes).
- [ ] `mic` — placeholder `"V"` — `panels/timeline/track-head.slint` — voice / duck source tag (audio lanes).
- [ ] `lock` / `lock-open` — placeholder `"L"` — `panels/timeline/track-head.slint` — lock lane.

### Text inspector — `panels/inspector/text-inspector.slint`

- [ ] `bold` — placeholder `"B"` — text bold.
- [ ] `underline` — placeholder `"U"` — text underline.
- [ ] `italic` — placeholder `"I"` — text italic.
- [ ] `case-upper` — placeholder `"TT"` — uppercase.
- [ ] `case-lower` — placeholder `"tt"` — lowercase.
- [ ] `case-sensitive` — placeholder `"Tt"` — title case.
- [ ] `align-left` — placeholder `"|<"` — horizontal align left.
- [ ] `align-center` — placeholder `"-"` — horizontal align center.
- [ ] `align-right` — placeholder `">|"` — horizontal align right.
- [ ] `vertical-align-top` — placeholder `"T"` — vertical align top.
- [ ] `vertical-align-middle` — placeholder `"M"` — vertical align middle.
- [ ] `vertical-align-bottom` — placeholder `"B"` — vertical align bottom.
- [ ] `wrap-text` — placeholder `"On"/"Off"` — wrap toggle.
- [ ] `letter-spacing` (Tabler) — placeholder `"C"` — letter spacing prefix.
- [ ] `line-height` (Tabler) — placeholder `"L"` — line spacing prefix.
- [ ] keyframe in/out icons — placeholder `"|<" "+" ">|" "T" "B"` — disabled animation row (lower priority).

### Inspector (general)

- [ ] `chevron-up` / `chevron-down` — placeholder `"^"` — section collapse caret (`inspector/inspector-widgets.slint`, `inspector/transform-inspector.slint`).
- [ ] `spline` — placeholder `"~"` — keyframe easing trigger (`inspector/keyframe-control.slint`).
- [ ] `scan` + `expand` — placeholder `"Fit"` / `"Fill"` — transform fit/fill (`inspector/transform-inspector.slint`).
- [ ] `trash-2` — placeholder `"Remove"` — remove effect (`inspector/effects-inspector.slint`).
- [ ] `flip-horizontal` — placeholder `"Flip H"` — crop mirror (`inspector/crop-inspector.slint`).
- [ ] `flip-vertical` — placeholder `"Flip V"` — crop mirror (`inspector/crop-inspector.slint`).

### Dropdowns & pickers

- [ ] `chevron-down` — placeholder `"v"` — dropdown chevron (`components/dropdown.slint`).
- [ ] `chevron-down` — placeholder `"v"` — color-swatch chevron (`components/color-swatch.slint`).

### Library & tiles

- [ ] `plus` / `folder-plus` — placeholder `"+  Import"` — import button (`panels/library/library.slint`).
- [ ] `wand-2` / `sparkles` — placeholder `"fx"` — effect/transition tile glyph (`panels/library/tiles.slint`).
- [ ] `image` — placeholder `"IMG"` — still-image badge (`panels/library/tiles.slint`).
- [ ] `alert-triangle` / `unlink` — placeholder `"Missing"` — missing-media badge (`panels/library/tiles.slint`).

### Misc

- [ ] `x` — placeholder `"×"` — transition remove (`panels/timeline/transition-pill.slint`).
- [ ] `check` — placeholder `"✓"` — agent dry-run checkbox (`panels/agent/agent.slint`).
- [ ] `send` — placeholder `"Send"` — agent submit (optional) (`panels/agent/agent.slint`).
- [ ] `circle-stop` — placeholder `"Stop"` — agent cancel (optional) (`panels/agent/agent.slint`).

### Fine as text (no icon needed)

Timecode `/` separators, the zoom `%` readout, and word buttons in dialogs
(Browse… / Cancel / Export / Done / OK / Locate… / New project / etc.).
