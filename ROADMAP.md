# Cutlass Roadmap

Ordered list of upcoming work. Each item should be small enough to land as one
reviewable change (or a short series). Update this file as items complete.

## Already landed (context, not work)

- Timeline → MP4 export, end to end: per-frame GPU composite + H.264/AAC mux
  (`crates/cutlass-render/src/export.rs`), wired into the engine
  (`Command::Project(Export)`), the desktop export job
  (`apps/cutlass-desktop/src/preview_worker.rs`), and mobile
  (`crates/cutlass-mobile/src/export_job.rs`).
- `cutlass-py`: functional v2 track-first API (`Project`, `Track`, `Clip`,
  `get_frame() -> numpy`, `export()`), with a passing integration suite.
- Param/keyframe system (`crates/cutlass-models/src/param.rs`) including speed
  curves; Phase I "look" data model (`crates/cutlass-models/src/look.rs`)
  persisted + validated but render-neutral.
- Compositor pipeline benchmark over real media
  (`crates/cutlass-render/examples/composite_bench.rs`) — use it to guard GPU
  cost regressions for the render work below.

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

## 6. Look animations (entrance / exit / combo)

Drive the persisted animation catalogs from `look.rs` through the param system
at resolve time (transform/opacity over the clip's local timeline).

## 7. Export audio: retimed and denoised clips

Video retimes via speed curves, but affected clips currently export **silent**
(`crates/cutlass-render/src/export_audio.rs`). Implement varispeed resampling
for speed-ramped audio and run RNNoise in the export mix.

## 8. cutlass-py polish

- Allow still-image `import_media` (renderer already supports
  `LayerSource::Still`; the Python side may still reject PNGs).
- Sync README with current capabilities; optional PyPI packaging via maturin.

## 9. Non-Apple export backends

`crates/cutlass-encoder` returns `Unsupported` on Linux/Windows. Add a backend
(e.g. FFmpeg libx264 or platform encoders) if cross-platform export matters.

## 10. Docs debt

- Keep `.cursor/rules/overview.mdc` and this roadmap in sync as stickers,
  look animations, export audio, and Python packaging land.
