# Changelog

Notes for the latest release. For previous releases, see the
[GitHub releases page](https://github.com/1Mr-Newton/cutlass/releases).

## [alpha-0.7.0] — 2026-07-22

### Added

- **Blend modes.** Every visual clip has a blend mode dropdown (multiply,
  screen, overlay, darken, lighten, and friends) composited on the GPU.

- **Layer styles.** Drop shadow, glow, outline, and background plate on any
  visual clip — video, text, shapes, stickers. Every style property is
  animatable, with live preview while dragging sliders.

- **Keyframe graph editor.** A drawer under the timeline plots parameter
  curves; drag keyframes, insert and delete them, and shape segment easing
  with bezier handles.

- **Motion paths.** Position keyframes take spatial bezier tangents, so
  clips travel curves instead of straight lines. The path draws on the
  preview canvas with draggable points and tangent handles.

- **Motion blur.** Per-clip motion blur supersamples animated transforms
  in the compositor.

- **Per-character text animation.** Character-cascade presets rendered
  through a new GPU glyph atlas with an instanced text pipeline, plus
  speed / intensity / stagger knobs on all animation presets.

- **Mask and chroma key inspectors.** Clip masks gained animatable geometry
  (position, size, rotation, roundness); mask and chroma sections are now
  editable in the clip inspector.

- **Six new color adjust sliders.** Tint, hue, highlights, shadows,
  sharpness, and vignette, alongside the existing grade controls.

- **Animatable crop.** Crop rectangles interpolate between keyframes.

- **Per-axis scale.** Scale X and Y independently from the transform
  inspector (old uniform-scale projects load unchanged).

- **Audio pan.** Animatable constant-power pan in the shared preview /
  export mixer.

- **More easing.** Hold (step) easing, plus named presets: snappy,
  overshoot, anticipate, bounce, elastic, and back.

- **Typed effect parameters.** Effects can take color and vec2 parameters
  (not just scalars), with new duotone and color-overlay effect passes.

### Changed

- **Almost everything is keyframable.** Text style metrics, filter / LUT /
  adjust intensities, mask and chroma settings, and layer styles all route
  through the same keyframe system, with keyframe controls in every
  inspector row and full coverage on the AI wire.

- **AI setup is provider-based.** Configure any OpenAI-compatible provider
  (key, base URL, model) in settings; the old cloud account flow is gone.

- **Text backgrounds animate.** The text background card (color, padding,
  radius, opacity) is animatable and rendered in preview and export.

[alpha-0.7.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.7.0
