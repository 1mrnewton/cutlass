# Keyframes roadmap — `Param<T>`, the M2 keystone

One animation system, built once, used by everything after: transforms
today; effect parameters (M4), color (M5), masks (M6), volume envelopes
and speed ramps (M8 / M2-late) all ride the same type. This document is
the feature-area plan for `v1-roadmap.md` § M2.

## Status legend

- [x] shipped
- [ ] not started / in progress

---

## Design (locked in Phase 0)

- **`Param<T>` is an enum: `Constant(T)` or `Keyframed(Vec<Keyframe<T>>)`**
  (`cutlass-models/src/param.rs`). A keyframe is `(tick, value, easing)`;
  keyframed vecs are non-empty and strictly sorted by tick (mutators
  preserve it, `validate_shape` checks it on load).
- **Ticks are clip-relative**: a keyframe's `tick` is the offset from the
  owning clip's timeline start, at the timeline rate. Moving a clip moves
  its animation for free — no fix-ups in `MoveClip`/`ShiftClips`.
  Commands take *absolute* timeline positions and the engine converts
  (`Clip::animation_tick`), so the agent and UI never see clip-relative
  math.
- **Easing**: `linear`, `ease_in` (quadratic), `ease_out`, `ease_in_out`
  (smoothstep), and CSS-style `bezier { points: [x1, y1, x2, y2] }`
  (Newton + bisection solve, x clamped to `0..=1` by validation). Easing
  describes the segment *leaving* a keyframe.
- **Serialization is compact and backward-compatible**: a constant param
  serializes as the bare value (`"scale": 1.5`) — byte-identical to the
  pre-M2 format — and a keyframed one as `{"kf":[{"t":..,"v":..,"e":..}]}`
  (linear easing elided). Old projects load unchanged; never-animated
  projects keep the old shape.
- **Schema v2**: files that may carry keyframes are stamped `version: 2`
  on save (the writer defines the format, even for projects loaded from
  v1 files). v2 readers accept v1 files; v1 builds refuse v2 files with a
  clear unsupported-schema error instead of half-parsing curves.
- **`AnimatedTransform` replaces the stored `ClipTransform`** on clips:
  `position: Param<[f32;2]>`, `scale`/`rotation`/`opacity: Param<f32>`.
  `ClipTransform` (plain floats) survives as the *sampled value* type the
  compositor, gestures, overrides, and inspector exchange —
  `AnimatedTransform::sample(tick) -> ClipTransform` is the bridge.
- **Sampling is hot-path**: pure, allocation-free, O(log k) binary search
  + eased lerp per property per layer per frame. Measured: warm 1080p
  `get_frame` with six animated segments is indistinguishable from the
  constant-transform case (~1.77 ms vs ~1.82 ms; the cost is composite +
  readback, not sampling). Guarded by the
  `preview/get_frame/solid_1080p_animated_warm` criterion bench.
- **Gesture compose semantics (CapCut)**: `SetClipTransform` carries
  `at: Option<RationalTime>`. With `Some(playhead)` — what the preview
  gesture commit sends — properties that already have keyframes get a
  keyframe written at the playhead and constants stay constant; with
  `None` everything flattens to constants. Never-animated clips behave
  exactly as pre-M2 either way.

## Phase 0 — Model, engine, commands, agent (foundation) ✅

- [x] **`Param<T>`** in `cutlass-models` with easing, sampling, mutation
      helpers, shape validation, serde round-trips (constant ↔ bare
      value; keyframed ↔ `{"kf":[...]}`).
- [x] **`AnimatedTransform`** migration of `Clip.transform`; per-property
      validation rules (finite position/rotation, scale > 0, opacity
      `0..=1`) enforced for constants and every keyframe.
- [x] **Schema v2** with read-forward of v1 files and re-stamp on save.
- [x] **Commands**: `SetParamKeyframe` / `RemoveParamKeyframe` /
      `SetParamConstant` (+ `SetClipTransform.at`), each an undoable
      engine action with full-clip-restore inverses, covered by
      inverse round-trip tests.
- [x] **Engine evaluation**: `resolve_layers` samples the transform at
      the clip-relative frame tick; export inherits for free; the live
      gesture override still substitutes the whole sampled value.
- [x] **Agent vocabulary**: `set_param_keyframe`, `remove_param_keyframe`,
      `set_param_constant` wire DTOs + validation (clip-extent rejection
      messages), action-log lines, schema snapshot v2, and eval cases
      ("fade the clip in over the first second" works end-to-end, one
      undo per prompt).
- [x] **Bench guard**: animated-vs-constant warm preview case in
      `cutlass-engine/benches/preview.rs` (also fixed the bench's
      solid-on-video-lane fixture, broken since lane typing).

## Phase 1 — Inspector keyframe UI ✅

- [x] **Diamond per property row** (CapCut UX): toggle adds/removes a
      keyframe at the playhead; filled when the playhead sits on a
      keyframe, hollow-active when the property is animated. One
      `KeyframeControl` cluster (◀ ◆ ▶ + easing flyout) shared by the
      transform inspector and the text inspector's transform/blend rows;
      diamonds gate on the playhead being inside the clip (engine rule).
- [x] **Prev/next keyframe navigation** arrows per row; playhead jumps.
- [x] **Playhead-accurate value rows**: the projection publishes each
      clip's keyframe curves once (`kf-*` lists on the Slint `Clip`,
      absolute ticks), and a pure `InspectorBackend.sample-transform`
      callback re-samples in Rust with the engine's `Param` math per
      playhead move — no projection republish per tick. The same sample
      drives preview hit-testing, the selection box, and gesture
      resolvers, so geometry follows the rendered frame on animated
      clips.
- [x] **Easing picker** per keyframe (linear / ease presets; bezier
      round-trips through the projection but the editor waits).
- [x] **Gesture + keyframe interplay polish**: preview drag on an
      animated clip writes through `SetClipTransform { at: playhead }`
      (already wired) — surfaced: the worker bumps a commit epoch when a
      gesture writes keyframes and the inspector shows a transient
      "Keyframe added" chip.

## Phase 2 — Timeline keyframe markers ✅

- [x] **Diamonds on the selected clip's body** at keyframe positions
      (all animated properties merged, CapCut-style). The projection's
      `kf-*` curves feed a pure `KeyframeBackend.ticks` callback (merged +
      deduped in Rust); diamonds only render on selected, unlocked clips.
- [x] **Drag a diamond to retime** the keyframe — decided for the
      remove+set composition over a new command: the worker collects every
      property keyframed at the grabbed tick and replays remove + set
      (same value and easing) per property inside one history group, so
      one undo puts the merged diamond back and a keyframe already at the
      target is replaced (diamonds merge). The gesture drags a root-level
      ghost snapped to ticks and clamped inside the clip; the grabbed
      diamond itself never moves (a moving TouchArea would feed back into
      its own `mouse-x`).
- [x] **Right-click delete** on a diamond: removes every property's
      keyframe at that tick, also one history group.

## Phase 3 — Speed curves (blocked on M1)

- [ ] `speed` lands in M1 as a constant; retiming it as a
      `Param<f32>` gives velocity ramps. Needs source-time integration
      (speed is a *rate*: source position is the integral of the curve),
      so it is deliberately after the basic system proves out.
- [ ] Presets (montage, hero moment) as data over the same curves.

## Phase 4 — Tick model audit

- [ ] Slint's tick properties are `i32` against the engine's `i64`
      (documented debt in `timeline-roadmap.md`). Keyframes make dense,
      long timelines likelier; audit the clamps before they bite.

## Later milestones that reuse `Param`

- M4 effects: `EffectInstance { effect_id, params: Map<String, Param> }`.
- M5 color: every slider in the fused color pass.
- M6 masks: mask transform params (animated reveals).
- M8 audio: `volume` envelopes into both mixers; ducking writes ordinary
  volume keyframes.
