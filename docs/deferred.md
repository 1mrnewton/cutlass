# Deferred work

Parked intentionally (Jul 2026). Not the next thing to pick up unless
priorities change. When something here ships, move it to
[ROADMAP.md](../ROADMAP.md) as done (or delete the entry) and fix any
README / overview claims that still oversell it.

## Audio

### Pitch-preserving time-stretch

- **Wanted:** CapCut-style pitch lock — retimed clips (flat speed or speed
  curve) keep original pitch instead of chipmunk / slow-mo pitch.
- **Today:** `clip.preserve_pitch` and `SetClipPitch` exist in the model, UI,
  and commands. Preview/export mix via `ExportAudioMixer` / `audio_dsp`
  **varispeed** only — pitch always follows rate. The flag is ignored at mix
  time.
- **Where:** `crates/cutlass-render/src/export_audio.rs`,
  `crates/cutlass-render/src/audio_dsp.rs`, `apps/cutlass-desktop/src/audio.rs`.
- **Also:** README states pitch follows rate for now; keep it that way until
  this lands.

### Reversed-clip audio

- **Wanted:** Reversed media clips export and preview with audio played
  backward.
- **Today:** Reversed clips are skipped in the export mixer (forward-only
  readers); they export silent. Video reverse still works.
- **Where:** `ExportAudioMixer::for_project` skips `clip.reversed`.

### Auto-duck (sidechain under voice)

- **Wanted:** Mark an audio lane as `duck_source` (voice), select music, run
  “duck under voice” — engine writes volume keyframes on the music wherever
  voice overlaps.
- **Today:** Desktop UI + `EditCommand::DuckLanes` exist;
  `dispatch` returns `Unsupported` (“needs the decoder's audio reader”).
  AI tool `duck` was removed from the schema for the same reason.
- **Where:** `crates/cutlass-engine/src/action/dispatch.rs`,
  `apps/cutlass-desktop` inspector / `preview_worker` duck path,
  track-head `duck-source` toggle.

### Beat detection

- **Wanted:** Analyze a music clip, store beat ticks, draw markers, snap
  timeline edges to the beat grid; clear beats.
- **Today:** UI + `DetectBeats` / `ClearBeats` exist; engine returns
  `Unsupported`. AI tool `detect_beats` removed from the schema.
- **Where:** same dispatch arms; inspector Detect / Clear beats;
  `Clip.beats` in the projection model.

## Platform / media backends

### Non-Apple (and incomplete non-macOS) export / decode

- **Wanted:** Real H.264/AAC export and media decode on Linux (and solid
  Windows packaging if that matters).
- **Today:** Apple (VideoToolbox) and Windows (Media Foundation) decode and
  export work. Linux encoder/decoder return `Unsupported`; Linux desktop
  builds are preview-only in the README (UI runs; media won't play) and Linux
  packaging is marked dormant.
- **Where:** `crates/cutlass-encoder`, `crates/cutlass-decoder`,
  [ROADMAP.md](../ROADMAP.md) item 9, `packaging/README.md`.

### Zero-copy GPU surface import off Apple

- **Wanted:** Import decoder GPU surfaces (AHardwareBuffer / DXGI / etc.)
  into the compositor without a CPU round-trip.
- **Today:** Zero-copy win is Apple-oriented; other platforms error or fall
  back. Android MediaCodec GPU/AHardwareBuffer output explicitly not
  implemented (`OutputMode::Cpu` only).
- **Where:** `crates/cutlass-compositor` GPU import paths,
  `crates/cutlass-decoder` Android / REVIEW notes.

## Effects / compositor coverage

### Catalog effects and transitions without WGSL

- **Wanted:** Every catalog effect/transition id has a real GPU pass.
- **Today:** Starter set has shaders (`crossfade`, `wipe_left`,
  `gaussian_blur`, `vignette`, `pixelate`, plus others in the effect
  pack). Remaining catalog ids are **safe passthroughs**; unknown
  transitions fall back to crossfade.
- **Where:** `crates/cutlass-compositor/src/effect_render.rs`,
  `passes.rs` coverage tables.

### Richer text style rendering

- **Wanted:** Bold, italic, underline, letter-spacing, vertical alignment,
  wrap — as stored on `TextStyle`.
- **Today:** Font, size, fill, horizontal alignment, line spacing, case,
  stroke, background, and shadow render. Remaining fields persist in the
  model but are not mapped through `map_text_style` / `cutlass-text` yet.
- **Where:** `crates/cutlass-render/src/resolve.rs` (`map_text_style`),
  `crates/cutlass-text`; `crates/cutlass-py/api-design.md` “v1 render coverage”.

## AI assistant

### Guarded `Import` (Phase 5 stretch)

- **Wanted:** Agent can import media behind an explicit per-prompt
  confirmation card. `Open` / `Save` / `Export` stay human-only.
- **Today:** Edit-only vocabulary; stretch deliberately deferred until the
  trust model is proven in alpha.
- **Where:** [docs/ai-agent-roadmap.md](ai-agent-roadmap.md) Phase 5.

### Agent polish (post-M3, not blocking)

- Smarter `describe_project` windowing for huge timelines (playhead /
  named ranges) instead of id-sorted truncation only.
- Conversation / transcript persistence across sessions.
- Local-model tool-calling quality remains a product risk (mitigated by
  closed schema + dry-run), not a code milestone.

## Docs / marketing honesty (when revisiting)

The README no longer claims pitch preservation, auto-duck, or beat detection
(fixed Jul 2026). When one of those audio items ships, add it back to the
README's feature list.
