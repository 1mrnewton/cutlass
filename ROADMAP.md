# Cutlass Roadmap

Ordered list of upcoming work. Each item should be small enough to land as one
reviewable change (or a short series). Update this file as items complete.

## Already landed (context, not work)

- Timeline → MP4 export, end to end: per-frame GPU composite + H.264/AAC mux
  (`crates/cutlass-render/src/export.rs`), wired into the engine
  (`Command::Project(Export)`), the desktop export job
  (`apps/cutlass-desktop/src/preview_worker.rs`), and mobile
  (`crates/cutlass-mobile/src/export_job.rs`). Apple and Windows encoders ship;
  Linux export still returns `Unsupported` (see item 9).
- `cutlass-py`: v2 track-first API (`Project`, `Track`, `Clip`, still import,
  `Sticker` / `stickers()`, look `animations()` / `set_animation`, `get_frame()
  -> numpy`, `export()`), PyPI wheels via maturin, passing integration suite.
- Param/keyframe system (`crates/cutlass-models/src/param.rs`) including speed
  curves; look data model (`crates/cutlass-models/src/look.rs`) drives clip
  grades, masks/chroma, effects/transitions, lane passes, stickers, and
  entrance/exit/combo animations at resolve time.
- Export audio: shared `ExportAudioMixer` varispeed-resamples retimed clips and
  RNNoise-denoises flagged clips in preview and export (pitch-preserving stretch
  and reversed-clip audio still deferred).
- Compositor pipeline benchmark over real media
  (`crates/cutlass-render/examples/composite_bench.rs`) — use it to guard GPU
  cost regressions for render work.

## 1. Render clip color adjustments and filter presets — DONE

Per-clip `clip.adjust` (`ColorAdjustments`) and `clip.filter` (preset id ×
intensity) now grade pixels in preview and export.

- `ColorGrade` on `CompositeLayer` (`crates/cutlass-compositor/src/layer.rs`),
  applied in all four fragment shaders via the shared
  `crates/cutlass-compositor/shaders/grade.wgsl`.
- Preset id → recipe table and `effective_grade()` in
  `crates/cutlass-render/src/grade.rs`; threaded through
  `resolve_clip` → `SceneLayer.grade` → `Realized` → `with_grade()`.
- Lane-level `Generator::Filter`/`Adjustment` bars now apply as canvas passes
  (item 4).
- Benchmarked with `composite_bench`: compositor timings within run noise.

## 2. Render masks and chroma key — DONE

Per-clip masks and chroma key now affect preview and export.

- `LayerEffects` on `CompositeLayer` carries mask/chroma state into the
  compositor, with `rgba_fx`/`yuv_fx` shader paths for media and RGBA layers.
- Mask shapes (linear/mirror/circle/rectangle/heart/star) and chroma key
  strength/shadow are sampled from persisted clip look data.

## 3. Clip effects and transitions (M4 GPU passes) — DONE

Clip effects and transitions now flow from the model into preview/export.

- `ResolvedPass` samples `clip.effects`; the renderer packs them into
  compositor `PassInstance`s and runs effect chains through offscreen passes.
- `crossfade`, `wipe_left`, `gaussian_blur`, `vignette`, and `pixelate` have
  WGSL coverage. Other catalog effects are safe passthroughs for now; remaining
  transition ids fall back to crossfade.
- Drift tests keep model and compositor catalogs aligned.

## 4. Lane-level generators: Effect / Filter / Adjustment — DONE

Effect/filter/adjustment generator bars now apply to everything already
composited beneath their track.

- `LayerSource::CanvasPass` represents a geometry-free scene layer carrying a
  sampled effect chain and/or `ColorGrade`.
- `CompositorLayer::CanvasPass` snapshots the current canvas, runs the existing
  offscreen effect chain plus a full-canvas grade pass, and replaces the canvas.
- Resolver, compositor, and render smoke tests cover lane pass ordering,
  no-op elision, gesture fallback, grade, and effect execution.

## 5. Stickers — DONE

Stickers are first-class generated content end to end.

- `Generator::Sticker { asset }` references a bundled catalog
  (`cutlass-models/src/sticker.rs`, bytes embedded from `assets/stickers/`);
  legacy payload-less `"Sticker"` documents still deserialize.
- `cutlass-decoder` decodes animations from bytes (GIF/APNG portable via
  `gif`/`png`, plus animated WebP through ImageIO on Apple).
- Resolve emits `LayerSource::Sticker` (intrinsic pixels as reference pixels,
  the shape convention); the renderer caches decoded frame sequences and
  composites the looping frame as an `Rgba` layer in preview and export.
- Desktop Library tiles (with real thumbnails), mobile `AddSticker { asset }`,
  and Python `Sticker(asset)` / `cutlass.stickers()` are wired up.
- The starter pack is placeholder art; swapping in real artwork only touches
  `assets/stickers/` and the catalog table (a drift test pins them together).

## 6. Look animations (entrance / exit / combo) — DONE

Drive the persisted animation catalogs from `look.rs` through the param system
at resolve time (transform/opacity over the clip's local timeline).

- `cutlass-render/src/animation.rs` maps every catalog id to transform/opacity
  deltas; sampled in `resolve_clip` after the clip's keyframed transform.
- Combo presets loop over a fixed period; in/out windows default to ~0.5 s
  (clamped to half the clip length). A catalog drift test pins coverage.
- Desktop inspector Animation tab wires In/Out/Combo pickers to
  `SetClipAnimation`; cutlass-py exposes `animations()` and
  `clip.set_animation(slot, id)`.

## 7. Export audio: retimed and denoised clips — DONE

Retimed clips (speed / speed-curve ramps) and denoise-flagged clips now mix in
preview and export via the shared [`ExportAudioMixer`](crates/cutlass-render/src/export_audio.rs).

- [`audio_dsp.rs`](crates/cutlass-render/src/audio_dsp.rs): `DenoiseReader` wraps
  each channel through RNNoise (`nnnoiseless`); varispeed source-frame mapping
  reuses `speed_curve_integral` / `speed_curve_source_fraction` from the model.
- Warped spans linearly interpolate decoded source PCM; unity-speed spans keep
  the fast seek-and-stream path. Pitch-preserving time-stretch and reversed-
  clip audio are deferred (reversed clips still export silent).

## 8. cutlass-py polish — DONE

Still-image import, docs, and PyPI packaging are aligned with the engine.

- `import_media` already probed stills (`MediaSource::image` → `LayerSource::Still`);
  this pass added positive PNG import/placement/`get_frame` tests and renamed the
  negative test to cover missing/corrupt files only.
- README and `api-design.md` now document still images (`kind == "image"`, 5 s
  default duration, any placement length on `video` tracks).
- PyPI wheels ship via maturin (`pyproject.toml`, `pywheels.yml` CI).

## 9. Non-Apple export backends — DEFERRED

Parked with the rest of the intentionally deferred work in
[docs/deferred.md](docs/deferred.md) (pitch-preserving audio, reverse audio,
duck, beats, Linux/Windows media, effect/text coverage gaps, AI Import
stretch). Revisit when cross-platform export matters.

## 11. AI assistant (cutlass-ai + desktop wiring) — DONE

Restored the `cutlass-ai` crate from pre-mobile-pivot history and wired the
desktop assistant panel end to end.

- `crates/cutlass-ai`: wire format, validation, OpenAI-compatible provider,
  agent loop, eval harness, tool schema v20 (look commands; `duck` /
  `detect_beats` removed — engine returns `Unsupported` on this line).
- `apps/cutlass-desktop/src/agent.rs`: sandbox rehearsal + plan replay via
  `preview_worker` (`SnapshotProject`, `AgentApplyPlan`, `agent_replay`).
- `AgentStore` callbacks, provider settings, and connection test in `main.rs`.
- `docs/ai-agent-roadmap.md` restored as the phase-by-phase reference.

## 10. Docs debt — DONE

- `.cursor/rules/overview.mdc` and this file now reflect stickers, look
  animations, export audio, and cutlass-py (still import, PyPI wheels).
- [CONTRIBUTING.md](CONTRIBUTING.md) points at this roadmap (replacing stale
  `docs/v1-roadmap.md` references).
- `crates/cutlass-py/api-design.md` intro updated from a v2 proposal to the
  shipped API reference.
