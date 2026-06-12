# Changelog

## [Unreleased]

### Fixed

- **Animated clips no longer shake.** Moving a keyframed layer (e.g. a
  title gliding across the canvas) shimmered in preview and export: the
  layer was translated by sub-pixel amounts every frame through the
  bilinear sampler, so glyph edges pulsed between sharp and blurred.
  Unrotated layers now pixel-snap their placement — text stays bit-crisp
  while it moves.
- **Export frame rates above the timeline rate now animate at the output
  rate.** Keyframed transforms sample at the exact output frame time
  (sub-frame), so a 60 fps export of a 24 fps timeline renders 60 Hz
  motion instead of repeating 24 fps positions in an uneven 3-2 cadence.
- **The agent's generated-clip tools work with small local models.** Tool
  schemas are fully inlined (no `$ref` indirection, schema v7), the
  `generator` argument documents its exact JSON shapes, and rejections
  carry a corrective example — adding a text clip via gemma-class models
  no longer dead-ends.

### Crop & flip (M1)

- Clips can be **cropped** (trim a fraction off each edge; the kept region
  re-fits the canvas centered, CapCut-style) and **mirrored** horizontally
  / vertically. Works on any visual clip — video, image, text, shapes.
- Inspector grows a Crop section: per-edge inset rows (slider + numeric
  entry, double-click to reset) and Flip H / Flip V chips; the preview's
  hit-test and selection box follow the cropped region.
- One undoable edit per change (`SetClipCrop`); splitting a cropped clip
  keeps the framing on both halves.
- The AI agent can crop too: `set_clip_crop` tool (tool schema v6) —
  "crop 25% off both sides and mirror it" works, with omitted edges left
  unchanged.
- Rendering is free: the compositor samples a per-layer UV sub-rect, and
  reversed UVs encode the flips — no extra passes, preview and export
  share the path.
- Deliberate gap: no draggable crop-handles mode in the preview yet —
  numeric insets only.

## [alpha-0.2.0] — 2026-06-12

The first **AI alpha**: prompt-to-edit ships. This release also lands the
keyframe/animation system, clip speed and reverse, clip volume and fades,
image import, timeline markers, and the project lifecycle (save/open/
autosave/crash recovery) that alpha-0.1.0 lacked.

### AI agent: prompt-to-edit (M3 foundation)

- New `cutlass-ai` crate: an LLM-facing wire vocabulary generated from the
  edit-command layer (tool schemas versioned and snapshot-tested), with
  validation that lowers model output to real commands against the live
  project — phantom generators and project/file commands are rejected by
  construction.
- Agent chat panel in the editor: prompts stream plan/status, each applied
  edit renders as a human-readable action list, and every prompt is exactly
  **one undo entry** — rehearsed in a sandbox first, then replayed
  atomically, with rollback on failure.
- Dry-run mode previews the action list without touching the timeline;
  read-only Q&A ("how long is the timeline?") answers from a compact
  project description without mutating anything.
- Provider-abstracted: any OpenAI-compatible endpoint works — local
  (Ollama, llama.cpp-server) or cloud — configured in
  `~/.cutlass/config.toml` (`[ai]` table; keys never live in project
  files).
- Eval harness: scripted prompt → expected-timeline tests against a stub
  provider catch agent regressions in CI without a live model.

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

### Clip speed & reverse (M1)

- Media clips can play at any constant speed (0.05×–100×) and in
  reverse: `speed`/`reversed` on the clip retime preview, export, trim,
  and split alike; the timeline length re-derives from the speed.
- Inspector "Speed" section on video and audio clips: preset dropdown
  (0.25×–4×) plus a Reverse toggle; retimed clips wear a `2x` / `0.5x R`
  badge on the timeline and their filmstrips stretch to match.
- The AI agent can retime: `set_clip_speed` joined the tool vocabulary
  (schema v3) — e.g. "play the middle clip backwards at double speed".
- Audio of retimed clips is muted (playback and export) until varispeed
  lands in M8, so what you hear is what you ship.

### Clip audio: volume & fades (M1)

- Media clips carry `volume` (0–10×, 1.0 = as recorded) and fade in/out
  lengths; both mixers (playback and export) apply sample-accurate linear
  ramps from the same shared gain curve.
- Inspector "Audio" section on audio-lane clips: volume slider (0–200%)
  plus fade-in/fade-out sliders bounded by the clip length.
- Splitting a clip keeps its volume on both halves and partitions the
  fades CapCut-style.
- Timeline audio badge: clips with non-default audio wear a compact chip
  next to the retime badge — a struck-out speaker when muted, a "57%"
  label on a non-default volume, a fade ramp when only fades are set.
- The AI agent can mix: `set_clip_audio` joined the tool vocabulary
  (schema v4) — video-lane targets steer to the linked audio companion.
- Constant volume for now; envelopes/keyframes ride M8.

### Image import (M1)

- PNG / JPEG / WebP stills import as media: probed, decoded, and placed
  as 5-second default clips that transform and composite like video.
  Library tiles show the rendered thumbnail and badge the kind. A
  still's duration is a placement default, not a bound — image clips
  trim out past it freely.

### Timeline markers (M1)

- Named, colored markers on the timeline ruler: the toolbar flag (or
  `M`) drops one at the playhead, right-click removes it — all undoable.
- The AI agent can anchor: `add_marker` / `remove_marker` / `set_marker`
  joined the tool vocabulary (schema v5) — moving and renaming markers
  is agent-reachable even though the UI gesture for it comes later.

### Project lifecycle & M0 stabilization

- Project lifecycle in the editor: New / Open / Save / Save As / Recent,
  dirty-state dot in the title bar, save prompt on close.
- Autosave + crash recovery: periodic snapshots under
  `~/.cutlass/autosave/`, restore offered on next launch.
- Missing-media relink: opening a project whose source files moved now
  surfaces a relink dialog (re-pick per file or point at a folder);
  library tiles badge missing media until it's repaired.
- Ripple trim on the magnet track: trimming a main-lane clip with the
  magnet on shifts everything downstream to follow — one undo entry,
  linked A/V kept in sync.
- Format versioning policy: project schema v2 tolerates unknown optional
  fields, so older builds' projects keep opening as fields are added;
  migration scaffold + tests in place.
- Styled titles: text clips grew a full `TextStyle` (font, size, color,
  stroke, shadow, background, spacing, case, alignment) with matching
  inspector controls.
- Library panel with media thumbnails; interactive preview transforms
  (move / scale / rotate on the canvas) round-trip with the inspector.
- Selection now survives undo/redo and agent edits: every projection
  republish prunes vanished clip ids and re-anchors the primary.
- Phantom features hidden: Effects / Transitions / Filters / Adjustment
  library tabs removed and effect/filter/adjustment lanes skipped by the
  projection until their milestones land (the Stickers tab stays — shape/
  solid generators are real; model enums round-trip untouched).
- Group copy/duplicate paste the whole selection as one block (lanes and
  relative placement preserved, link groups re-linked, one undo); a
  toolbar Unlink button dissolves the selection's link groups undoably.
- README/CHANGELOG honesty pass: feature claims now state exactly what
  ships (the unwired proxy claim is gone, the crate table covers all
  eleven crates).

### Downloads

| Platform | Artifact |
| --- | --- |
| macOS (Apple Silicon) | `Cutlass-*-macos-arm64.zip` — unzip, drag `Cutlass.app` to Applications. **First launch:** right-click → Open (not notarized). See `INSTALL-macos.txt`. |
| Linux (x86_64) | `Cutlass-*-linux-x86_64.tar.gz` — extract and run `./cutlass-ui`; requires FFmpeg |

### Using the AI agent

The agent needs an LLM endpoint — none is bundled. Point
`~/.cutlass/config.toml` at any OpenAI-compatible server, local or cloud:

```toml
[ai]
base_url = "http://localhost:11434/v1"   # e.g. Ollama
model = "qwen2.5:14b"
# api_key = "sk-..."                     # for cloud endpoints
```

### Known limitations

- **Retimed clips are silent** — audio on speed ≠ 1× clips mutes until
  varispeed lands (M8).
- **No crop or canvas/aspect presets yet** — both are next on the
  roadmap (M1 close-out).
- **Agent quality tracks the model you give it** — small local models
  may tool-call poorly; dry-run mode previews every plan before it
  touches the timeline.
- **Alpha stability** — crashes and UI polish gaps are expected; please
  file issues.
- **macOS Intel** — not built in CI; build from source or use Rosetta.
- **MP3 seek accuracy** — mid-stream seeks on MP3 can be tens of ms off;
  MP4/AAC is sample-accurate.

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

[alpha-0.2.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.2.0
[alpha-0.1.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.1.0
