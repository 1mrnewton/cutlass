# Changelog

Notes for the latest release. For previous releases, see the
[GitHub releases page](https://github.com/1Mr-Newton/cutlass/releases).

## [alpha-0.6.1] — 2026-07-19

### Added

- **Text stroke, background, and shadow.** Title and caption clips render
  outline, fill plate, and drop shadow in preview and export, with live
  inspector sliders for the effect metrics.

- **Text layout controls.** Wrap, letter tracking, font weight, and canvas
  alignment are honored when resolving and compositing text layers.

- **Custom `.cube` LUTs.** Browse and apply your own LUT files from the look
  inspector, alongside the bundled starter pack.

- **Desktop icon set.** Lucide/Tabler icons wired across the editor chrome,
  plus a generated SVG set for timeline and toolbar actions (add, select,
  undo/redo, split, delete, marker, crop, reverse, transcript, AI tools,
  audio enhance, and related placeholders).

### Changed

- **Timeline playhead auto-scroll.** The timeline keeps the playhead in view
  while playing.

- **Editor shell polish.** Pane clamping, lighter panel headers, inspector
  tab type, and scroll views that no longer steal drag gestures from
  controls.

- **Preview text transforms.** Text stays visible while transforming on the
  canvas.

- **Slint 1.17 and wgpu 29.** Desktop and compositor dependency bump.

[alpha-0.6.1]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.6.1
