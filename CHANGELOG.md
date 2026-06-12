# Changelog

## [Unreleased]

### Keyframes & animatable parameters (M2 foundation)

- New `Param<T>` system in the model: any animatable property is a
  constant or an eased keyframe curve (linear / ease-in / ease-out /
  ease-in-out / cubic-bezier).
- Clip transforms (position, scale, rotation, opacity) are now
  animatable; preview and export sample curves per frame with no
  measurable hot-path cost.
- New undoable commands: `SetParamKeyframe`, `RemoveParamKeyframe`,
  `SetParamConstant`; transform gestures committed at the playhead write
  keyframes on already-animated properties (CapCut compose semantics).
- The AI agent can animate: `set_param_keyframe` / `remove_param_keyframe`
  / `set_param_constant` joined the tool vocabulary (schema v2) — e.g.
  "fade the clip in over the first second".
- Project format: schema v2. Old (v1) projects open unchanged; projects
  saved by this build require this build or newer. Never-animated
  projects keep the v1 field shapes.
- Inspector keyframe UI (CapCut diamond UX): every transform/blend row
  in the video and text inspectors grows a keyframe cluster — diamond
  toggles a keyframe at the playhead, ◀ ▶ jump between keyframes, and an
  easing flyout re-eases the keyframe under the playhead.
- Inspector value rows, the preview selection box, and preview gestures
  now track the playhead-sampled value on animated clips, so what you
  grab is what's rendered; a transient "Keyframe added" chip appears
  when a gesture writes a keyframe.
- Timeline keyframe markers: selected clips show a diamond per keyframe
  tick (all animated properties merged). Drag a diamond to retime the
  keyframes under it, right-click to delete them — either way one undo
  restores everything.

## [alpha-0.1.0] — 2026-06-11

First public alpha of the Cutlass desktop editor. Expect rough edges, missing
features, and no project compatibility guarantees yet.

### Editor (`cutlass-ui`)

- Import video and audio, drag clips onto a multi-lane timeline with filmstrip
  thumbnails and waveforms.
- CapCut-style editing: snap, main-track magnet, linked video+audio drops,
  trim, split, delete, ripple-delete, multi-select, group drag, undo/redo.
- Live GPU preview with scrubbing and real-time playback.
- Audio playback with device-clock A/V sync; mute toggles honored live.
- Transport: Space play/pause, JKL shuttle, loop toggle, in/out range marks.
- Frameless window with custom title bar; fullscreen preview mode.
- Export dialog: timeline → H.264/AAC MP4 with resolution, frame rate, and
  quality presets.

### Engine (under the hood)

- Deterministic edit commands with full undo/redo history.
- FFmpeg decode with hardware acceleration where available; GOP-aware
  sequential decode and on-disk frame cache for smooth playback.
- WGPU compositor for preview and export.

### Downloads

| Platform | Artifact |
| --- | --- |
| macOS (Apple Silicon) | `Cutlass-*-macos-arm64.zip` — unzip, drag `Cutlass.app` to Applications. **First launch:** right-click → Open (not notarized). See `INSTALL-macos.txt`. |
| Linux (x86_64) | `Cutlass-*-linux-x86_64.tar.gz` — extract and run `./cutlass-ui`; requires FFmpeg |

macOS builds bundle FFmpeg. Linux builds expect FFmpeg shared libraries on the
system (see `README-INSTALL.txt` in the archive).

### Known limitations

- **No AI agent yet** — the natural-language editing layer is not built; all
  edits are manual or via the headless command API.
- **Alpha stability** — crashes, perf cliffs on pathological media, and UI
  polish gaps are expected; please file issues.
- **macOS Intel** — not built in CI for this alpha; build from source or use
  Rosetta with the arm64 build.
- **MP3 seek accuracy** — mid-stream seeks on MP3 can be tens of ms off;
  MP4/AAC is sample-accurate.

### Build from source

```bash
brew install ffmpeg pkg-config   # macOS
cargo build --release -p cutlass-ui
cargo run --release -p cutlass-ui
```

See [README.md](README.md) for prerequisites and the `cutlass-app` CLI smoke test.

[alpha-0.1.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.1.0
