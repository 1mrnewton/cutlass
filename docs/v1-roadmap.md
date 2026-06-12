# Cutlass v1 Roadmap — from credible alpha to a real CapCut alternative

Policy: **we follow CapCut.** When a feature or UX question comes up, the
answer is "what does CapCut desktop do?" — then we build the local-first
version of it. This document is the master plan; the feature-area roadmaps
(`timeline-roadmap.md`, `preview-roadmap.md`, `playback-roadmap.md`, and the
new ones each milestone will spawn) hang off it.

Scope policy, stated once:

- **Local-first.** No cloud storage, no cloud sync, no collaboration, no
  account system, no template marketplace. Projects are files on disk;
  assets are files on disk.
- **AI is a first-class feature, provider-abstracted.** Local inference is
  the default where it's feasible (captions, silence detection, TTS,
  matting); LLM-driven editing ships behind a provider trait so cloud
  providers (OpenAI/Anthropic/Gemini/etc.) plug in later without touching
  the agent. *"Local-first" never means "local-only" for AI.*
- **Not every CapCut feature ships.** The explicit non-goals list is below;
  everything else in CapCut's desktop editor is fair game for v1 or the
  post-v1 backlog.

## Status legend

- [x] shipped
- [ ] not started / in progress

---

## 1. What "v1" means

v1 is the release where a creator can pick Cutlass instead of CapCut for a
real short-form/medium-form project and not hit a wall. Concretely:

1. **Project lifecycle is safe.** New / Open / Save / Save As / Recent /
   autosave / crash recovery — you cannot lose work.
2. **Editing core is at parity** with daily-driver CapCut: cut, trim
   (ripple included), speed, volume, fades, crop, multi-track, linked A/V,
   keyframes on the core properties.
3. **The look toolkit exists**: effects, filters, adjustment layers, color
   correction + LUTs + curves + HSL, chroma key, masks, transitions.
4. **Text and graphics are real**: styled text (font/color/stroke/shadow/
   background/spacing), text animations, text presets, stickers.
5. **Audio is real**: clip volume/fades/envelopes, ducking, noise
   reduction, varispeed.
6. **The AI layer exists and is the differentiator**: a prompt box that
   edits the timeline through the command layer, plus the AI media tools
   people actually use daily (auto captions, transcript editing, silence
   removal, TTS, background removal).
7. **It ships properly**: notarized macOS build, Linux build, Windows
   build, stable project format with a versioning story.

Everything in this doc is sequenced so each milestone ships something
usable on its own (the house rule from the existing roadmaps).

---

## 2. Where we are today — honest review

State as of `alpha-0.1.0` (2026-06-11). The full audit lives in this
section so we stop re-discovering it.

### What is genuinely strong

- **Headless engine with a closed, undoable command vocabulary**
  (`cutlass-commands` → `cutlass-engine/src/action/`): add/split/trim/move/
  remove/ripple/link/track-flags/transform/generator, all with inverse
  actions, compound history groups, rollback-on-failure. Extensively
  integration-tested. This is the foundation the AI agent was designed for,
  and it's real.
- **Playback pipeline is fast.** GOP-aware sequential decode + exact-tick
  cache keys hit 1080p24 at ~3ms/frame and 4K60 at ~9.6ms/frame cache-cold
  (`playback-roadmap.md` Phase 2). Audio-clock-mastered A/V sync, JKL,
  loop, in/out ranges.
- **Timeline UX is deep**: snap, main-track magnet ripple, linked A/V,
  multi-select, group drag, marquee, filmstrips, waveforms, track flags,
  compound undo. Phases 0–10 of `timeline-roadmap.md` shipped.
- **Preview canvas is interactive**: hit-testing, move/scale/rotate
  gestures, center guides, inspector round-trip — all through one shared
  `layer_placement` geometry.
- **Export works**: frame-by-frame composite → H.264/AAC MP4 with
  resolution/fps/CRF presets, cancellable, audio mixed fail-loud.

### What is missing or broken (the critical list)

1. **No AI anywhere.** `cutlass-commands` is a schema with a doc comment
   anticipating an agent. No provider code, no prompt UI, no tool schema
   generation. The product's stated identity is 0% built.
2. **No project Save/Open in the UI.** The engine has
   `ProjectCommand::Save/Open/Load` and the CLI uses them; `cutlass-ui`
   exposes only Import and Export. Every UI session is ephemeral —
   **data-loss by design**. This is the single worst gap for a real user.
3. **Phantom features in the model.** `Generator::{Sticker, Effect,
   Filter, Adjustment}` and their track kinds exist as enum placeholders;
   `composite.rs` silently skips them. They pollute the model and UI
   surface without rendering. Either wire them (the plan below) or they
   must stay hidden from users until then.
4. **No keyframes, no transitions, no speed, no volume, no crop, no
   color, no masks, no keying.** The entire "look and motion" layer of a
   CapCut-class editor is absent — and several of them require a model
   change (parameter/keyframe system) that gets more expensive the longer
   we wait.
5. **Proxy pipeline unwired.** `cutlass-encoder::build_proxy` exists and
   is tested, but preview decode never uses it; the README claims a
   "proxy/transcode cache" that is actually a decoded-YUV frame cache.
   Fix the wiring or fix the README — currently it's misleading.
6. **Text is a string.** No font, size, color, stroke, shadow, alignment,
   spacing — `cosmic-text` rasterizes a default style only.
7. **Audio model has no volume/fade/speed fields**; mute is the only
   audio control. MP3 mid-stream seek is approximate.
8. **Known timeline debt** (from `timeline-roadmap.md`): no ripple trim on
   the magnet track, no group copy/duplicate, no unlink gesture, selection
   goes stale after undo, Slint tick model is `i32` vs engine `i64`.
9. **Readback-bound preview**: composite → RGBA readback → Slint copy is
   ~half the 4K frame budget; the shared-wgpu-texture path is designed but
   unbuilt.
10. **Packaging**: macOS arm64 unsigned/un-notarized (right-click-to-open
    alpha), Linux tarball needs system FFmpeg, **no Windows build at all**.
11. **No images.** The library imports video and audio only — stills
    (PNG/JPEG) are table stakes for a CapCut-class editor.
12. **Schema rigidity**: project format is strict v1-only with no
    forward-compat policy; shipping updates will break saved projects
    unless we define one now.
13. **Test gaps**: engine/model coverage is excellent, but there are no UI
    integration tests, no golden-frame render tests, and `cutlass-commands`
    and `cutlass-encoder` have no dedicated test crates.

### Crate map (for orientation)

| Crate | State |
| --- | --- |
| `cutlass-models` | Solid. Needs: params/keyframes, speed, volume, effects data, format versioning. |
| `cutlass-commands` | Solid schema. Needs: new commands per milestone + JSON tool-schema export for the agent. |
| `cutlass-engine` | Solid. Needs: keyframe interpolation, effect graph hookup, proxy wiring, new actions. |
| `cutlass-compositor` | Works, minimal. Needs: effect/filter graph, blend modes, masks, keying, LUTs. |
| `cutlass-decoder` | Strong. Needs: image decode, varispeed audio, proxy-aware open. |
| `cutlass-encoder` | Works. Needs: HEVC, GIF, audio-only export, proxy integration consumer. |
| `cutlass-probe` | Fine. Needs: image probing. |
| `cutlass-cache` | Fine. Needs: proxy + raster artifacts alongside YUV frames. |
| `cutlass-ui` | Deep but narrow. Needs: project lifecycle, panels (effects/filters/color/text/stickers/AI), keyframe UI. |
| `cutlass-app` | Fine as smoke test. |
| *(new)* `cutlass-ai` | Agent runtime + provider abstraction. Does not exist. |
| *(new)* `cutlass-ml` | Local inference (whisper, matting, TTS, beat/silence detection). Does not exist. |

---

## 3. CapCut parity matrix

What CapCut desktop has, and what we do about each. "M#" = milestone below.
Research base: the CapCut desktop 2025–2026 feature set (editor + AI toolkit).

### Editing core

| CapCut feature | Cutlass today | v1 plan |
| --- | --- | --- |
| Cut / split / trim / ripple | ✅ except ripple-trim | M0 finishes ripple trim |
| Multi-track, linked A/V, magnet | ✅ | — |
| Keyframes (position/scale/rotation/opacity/volume/effects) | ❌ | **M2** — the keystone |
| Speed: constant, reverse | ❌ | M1 |
| Speed: curves / velocity ramps | ❌ | M2 (rides keyframes) |
| Crop / flip / non-uniform scale | ❌ | M1 |
| Image (stills) import | ❌ | M1 |
| Compound clips / nested timelines | ❌ | post-v1 |
| Multi-cam | ❌ | non-goal for v1 |
| Markers | ❌ | M1 (cheap, agent-useful) |
| Canvas/aspect presets (9:16, 1:1, …) + background | partial (fixed canvas) | M1 |

### Look

| CapCut feature | Cutlass today | v1 plan |
| --- | --- | --- |
| Effects library (drag-drop visual effects) | model placeholder only | **M4** effect engine + starter pack |
| Filters (preset looks) | placeholder | M5 (presets over the color stack) |
| Adjustment layers | placeholder | M4 |
| Transitions (crossfade, wipes, motion) | ❌ | M4 |
| Color: basic correction (exposure/contrast/temp/tint/sat) | ❌ | **M5** |
| Color: curves (Luma/R/G/B), HSL, color wheels | ❌ | M5 |
| LUT import (.cube) | ❌ | M5 |
| Auto color correction | ❌ | M9 (AI-assisted, optional) |
| Chroma key (green screen) | ❌ | **M6** |
| Masks (linear/circle/rect; pen later) | ❌ | M6 |
| Blend modes | ❌ (src-over only) | M6 |
| Stabilization, flicker removal, relight | ❌ | post-v1 |

### Text, stickers, templates

| CapCut feature | Cutlass today | v1 plan |
| --- | --- | --- |
| Text styling (font/size/color/stroke/shadow/bg/spacing/align) | ❌ (plain string) | **M7** |
| Text animations (in/loop/out) | ❌ | M7 |
| Text templates (preset animated titles) | ❌ | M7 (local preset packs) |
| Stickers (animated overlays) | placeholder | M7 (local packs: Lottie/APNG/WebP) |
| Auto captions (styled subtitle track) | ❌ | **M9** |
| Project templates / community templates | ❌ | non-goal (cloud ecosystem) |

### Audio

| CapCut feature | Cutlass today | v1 plan |
| --- | --- | --- |
| Clip volume + fades | ❌ | **M8** (model fields land in M1) |
| Volume keyframes / envelopes | ❌ | M8 (rides M2) |
| Audio ducking (auto-lower music under speech) | ❌ | M8 |
| Noise reduction | ❌ | M8 (rnnoise-class, local) |
| Varispeed audio (pitch-corrected speed) | ❌ (speed ≠1 mutes) | M8 |
| Beat detection / beat sync markers | ❌ | M8 |
| Voice changer / voice FX | ❌ | post-v1 |
| Audio separation (vocals/music stems) | ❌ | post-v1 |
| Extract audio from video | ✅ (linked companion clip) | — |
| Scrub audio bursts | ❌ | M8 |

### AI

| CapCut feature | Cutlass today | v1 plan |
| --- | --- | --- |
| **Prompt-to-edit agent** (our identity; CapCut has no real equivalent) | ✅ (M3 foundation: chat panel, sandbox + atomic replay, dry-run preview, one undo per prompt) | vocabulary grows every milestone |
| Auto captions + translation | ❌ | M9 (whisper local; cloud later) |
| Transcript-based editing (edit video by editing text) | ❌ | M9 — flagship |
| AutoCut / silence removal | ❌ | M9 (energy-based, local) |
| Text-to-speech | ❌ | M9 (local TTS; cloud voices later) |
| Background removal (no green screen) | ❌ | M9 (video matting, local ONNX) |
| AI upscaling / enhance | ❌ | post-v1 |
| Script-to-video, text-to-video generation | ❌ | non-goal for v1 (cloud-gen later via providers) |
| AI avatars, voice cloning | ❌ | non-goal |
| AI stickers / text-to-image | ❌ | post-v1 (provider-gated) |

### Export & platform

| CapCut feature | Cutlass today | v1 plan |
| --- | --- | --- |
| MP4 H.264 + AAC, res/fps/quality | ✅ | — |
| HEVC/H.265, MOV | ❌ | M10 |
| GIF export | ❌ | M10 |
| Audio-only export (MP3/WAV) | ❌ | M10 |
| Cover frame, bitrate control | ❌ | M10 |
| Background/queued export | ❌ (modal job) | post-v1 |
| Windows / macOS / Linux | mac+linux, unsigned | M10: + Windows, notarized macOS |
| Direct social upload | ❌ | non-goal |

### Explicit non-goals for v1

Cloud storage/sync/collab/teams, template marketplace, mobile/web ports,
multi-cam, AI avatars, voice cloning, generative text-to-video, stock
asset marketplaces (we ship small curated local packs + user folders
instead), direct social publishing. These are either cloud-ecosystem
features (out of scope by principle) or generative features that arrive
post-v1 through the cloud-provider seam.

---

## 4. Architecture invariants (apply to every milestone)

Carried over from the shipped roadmaps — these are why the codebase is
good, and every new system follows them:

- **The engine is the single source of truth.** Every user-visible
  mutation is a `cutlass_commands` command applied on the worker thread;
  the UI re-renders from the republished projection. No Slint-side state
  mutation, ever. *The AI agent is just another command source* — it gets
  correctness, undo, and validation for free, which is the entire point
  of the command-layer design.
- **Every command is undoable.** New commands ship with inverse actions
  and compound-group support. One gesture (or one AI prompt) = one
  history entry.
- **One resolver, shared by preview and commit.** Pure Rust callbacks
  behind `ui/lib/*-backend.slint`; the ghost never lies.
- **Decode, composite, and inference stay off the UI thread.**
- **Hot paths are measured.** Per-frame work is allocation-light and
  benchmarked (`playback_bench`, criterion benches) before and after.

New invariants this roadmap introduces:

- **Effects are data.** An effect instance is `{effect_id, parameters}`
  in the model; the compositor owns a registry mapping ids to GPU passes.
  Project files never serialize shader code. This keeps the agent able to
  *add* effects ("add a glitch effect to clip 2") by emitting plain data.
- **Anything animatable is a `Param`.** One parameter type (constant or
  keyframed curve) shared by transforms, effect parameters, volume, and
  speed — built once in M2, reused everywhere after.
- **AI is provider-abstracted.** `cutlass-ai` defines traits
  (`ChatProvider`, `TranscribeProvider`, `TtsProvider`, …); local
  implementations land first, cloud implementations are additive. No
  feature may hard-code a provider.
- **AI proposes, the engine disposes.** Agent output is validated
  commands, applied through normal dispatch, grouped per prompt, and
  undoable like any gesture. The agent never gets a side channel into
  project state.
- **Format stability from M0 on.** Schema bumps come with migration code
  and a read-forward policy (unknown optional fields tolerated). A v1
  user's project must open in every later v1.x.

---

## 5. Milestones

Ordering rationale, in one paragraph: M0 stops the bleeding (data loss,
phantom features, honesty). M1 fills the cheap-but-mandatory editing gaps
that need small model changes (speed/volume fields, images, crop). M2 is
the keystone — the parameter/keyframe system that effects, ramps, color,
masks, and audio envelopes all sit on; doing it before the look stack
exists is 10× cheaper than retrofitting. M3 ships the agent early, because
it only depends on the command layer (which exists today) and its
vocabulary then grows for free with every later milestone — shipping our
differentiator at month N instead of month N+10. M4–M8 build the look and
sound stack in dependency order (effect engine → color → keying/masks →
text/stickers → audio). M9 ships the AI media tools on top of `cutlass-ml`.
M10 is performance, platform, and release hardening.

### M0 — Stop the bleeding (stabilize the alpha)

Goal: nothing in the shipped app lies to users or loses their work.

- [x] **Project lifecycle in the UI**: New / Open / Save / Save As /
      Recent menu, riding the existing `ProjectCommand::{Save, Open}`;
      dirty-state tracking (title-bar dot), save prompt on close.
      Detailed plan: `project-lifecycle-roadmap.md` (Phases 1–3 — done).
- [x] **Autosave + crash recovery**: periodic snapshot to
      `~/.cutlass/autosave/`, offer restore on next launch
      (`project-lifecycle-roadmap.md` Phase 4 — done).
- [ ] **Missing-media relink** flow on open (engine `Load` already keeps
      placeholder entries; UI needs the relink dialog).
- [ ] **Hide phantom kinds**: sticker/effect/filter/adjustment generators
      and lanes removed from the UI surface until their milestones land
      (model keeps them; users stop seeing dead features).
- [ ] **Selection survives undo/redo** (clear or remap on history steps —
      tracked debt in `timeline-roadmap.md`).
- [ ] **Ripple trim on the magnet track** (`TrimClip` + `ShiftClips`
      composition; the deliberate gap from timeline Phase 7).
- [ ] **Group copy/duplicate + unlink gesture** (timeline Phase 10 gaps).
- [ ] **README/CHANGELOG honesty pass**: fix the proxy claim, fix the
      crate-responsibility table, state exactly what ships.
- [ ] **Format versioning policy**: schema v2 = v1 + tolerated unknown
      optional fields; write the migration scaffold + tests now, before
      M1 starts adding fields.

Exit: a user can edit for a week in Cutlass without losing a project or
clicking a button that does nothing.

### M1 — Editing core parity

Goal: the everyday CapCut edit vocabulary, minus animation.

- [ ] **Image import**: PNG/JPEG/WebP stills as media (probe + decode +
      default 5s clips, transform/crop like video). Library thumbnails.
- [ ] **Clip speed (constant + reverse)**: `speed: Rational` +
      `reversed: bool` on media clips; retime in `resolve_layers` and
      export; timeline duration math, badges (`2x`), inspector control.
      Audio of retimed clips mutes until M8 varispeed.
- [ ] **Clip volume + fade in/out fields** on the model + inspector +
      export mixer + playback mixer (constant volume now; envelopes in
      M8). Fade = first-class fields like CapCut, not keyframe sugar.
- [ ] **Crop** (normalized rect on `ClipTransform`) + **flip H/V** —
      compositor samples the sub-rect; preview gets crop handles mode.
- [ ] **Canvas settings**: project resolution/aspect presets (16:9, 9:16,
      1:1, 4:5, 21:9), background color per project; "fit/fill" clip
      helpers.
- [ ] **Markers** on the timeline ruler (named, colored) — cheap, and the
      agent + beat-sync (M8) both want them as anchors.
- [ ] **Timeline UX**: speed/volume badges on clips (the reserved slot
      from timeline Phase 8), drag-content preview polish.

Exit: a talking-head + b-roll edit needs nothing Cutlass doesn't have,
except looks and animation.

### M2 — Parameters & keyframes (the keystone)

Goal: one animation system, built once, used by everything after.
Detailed plan: `keyframes-roadmap.md`.

- [x] **`Param<T>` in `cutlass-models`**: constant or keyframed; keyframe
      = `(tick, value, easing)` with linear / ease-in / ease-out /
      ease-in-out / bezier easing. Serialized compactly (constants stay
      bare values — old files load unchanged, never-animated saves keep
      the old shape); schema v2 with v1 read-forward.
- [x] **Migrate `ClipTransform` + opacity to `Param`s** (constant params
      behave identically — zero visual change until a keyframe is added).
      *Volume joins when M1 lands the field, same as speed below.*
- [x] **Engine evaluation**: `resolve_layers` samples params at the frame
      tick (pure, allocation-free — benched: animated ≈ constant cost);
      export inherits for free. Interactive transform override composes
      with keyframed values CapCut-style (gesture commit keyframes at
      the playhead via `SetClipTransform.at`, or sets the constant).
- [x] **Commands**: `SetParamKeyframe`, `RemoveParamKeyframe`,
      `SetParamConstant` — undoable, group-friendly, agent-ready (in the
      agent vocabulary with evals: "fade the clip in over the first
      second" works end-to-end, one undo per prompt).
- [x] **Inspector keyframe UI**: the CapCut diamond per property row
      (add/remove/navigate keyframes at the playhead, easing picker per
      keyframe), value rows show the playhead-sampled value via UI-side
      `Param` sampling over projected curves — no republish per tick.
      Preview hit-test / selection box / gestures follow the sampled
      frame; a "Keyframe added" chip surfaces gesture-written keyframes.
- [ ] **Timeline keyframe markers** on selected clips (diamonds on the
      clip body; drag to retime — CapCut behavior).
- [ ] **Speed curves**: retime `speed` as a keyframable param →
      velocity-edit ramps; presets (montage, hero moment) as data.
      *Blocked on M1's speed field.*
- [ ] **Tick model audit**: keyframes make long/dense timelines likelier —
      resolve the Slint `i32` vs engine `i64` clamp now.

Exit: position/scale/rotation/opacity/volume/speed animate with eased
keyframes in preview and export, with CapCut's diamond UX.

### M3 — The AI agent (prompt-to-edit foundation)

Goal: the reason Cutlass exists. Ships early because the command layer is
ready today, then grows its vocabulary with each later milestone for free.
Detailed plan: `ai-agent-roadmap.md`.

- [x] **`cutlass-ai` crate**: `ChatProvider` trait (chat + tool-calling,
      streaming); first providers: **local** (Ollama / llama.cpp-server
      HTTP, OpenAI-compatible) and a **generic OpenAI-compatible remote**
      (covers OpenAI/compatible gateways day one — "cloud providers
      later" is then config, not code). Keys/config in
      `~/.cutlass/config.toml`; never in project files.
- [x] **Tool schema from the command layer**: JSON Schema for every
      command in the vocabulary, versioned with the crate — landed as a
      dedicated wire layer (LLM-shaped DTOs; edit commands only) rather
      than serde derives on `cutlass-commands`; see `ai-agent-roadmap.md`
      Phase 1 for the rationale. A `describe_project()` tool gives the
      model a compact timeline summary (tracks, clips, ids, times,
      media metadata).
- [x] **Agent loop**: prompt → provider tool-calls → validate → dispatch
      inside **one history group per prompt** with `rollback_group` on
      failure — an AI edit is exactly as safe and as undoable as a drag.
      (Landed as sandbox-rehearse-then-atomic-replay; see
      `ai-agent-roadmap.md` Phase 4.)
- [x] **Chat panel in `cutlass-ui`**: prompt box + transcript; streamed
      plan/status; each applied edit rendered as a human-readable action
      list ("split clip at 00:12, deleted 3 clips, added text 'INTRO'");
      one-click undo per prompt.
- [x] **Read-only Q&A**: "how long is the timeline?", "which clips have
      no audio?" — `describe_project` answers without mutating.
- [x] **Guardrails**: command whitelist (no `Open`/`Save`/`Export` in
      the vocabulary at all for M3), max-commands-per-prompt cap,
      dry-run mode that previews the action list before applying.
- [x] **Eval harness**: scripted prompt → expected-timeline tests against
      a stub provider, so agent regressions are caught in CI without a
      live model.

Exit: "cut the first 3 seconds, add a title that says INTRO, speed up the
middle clip 2x" works against a local model — undoable, auditable.
*(Shipped minus the speed change: clip speed isn't an engine command yet —
it joins the vocabulary via the Phase 5 checklist when M2 lands it.)*

### M4 — Effect engine & transitions

Goal: the rendering substrate for everything visual that isn't a plain
clip; the placeholder kinds finally become real.

- [ ] **Compositor effect graph**: per-layer post-pass chain
      (`texture → effect pass(es) → blend`). Effect registry: id →
      WGSL pass + parameter layout. Ping-pong intermediate targets;
      passes batched per frame.
- [ ] **Model**: `EffectInstance { effect_id, params: Map<String, Param> }`
      attached to clips; `Effect`/`Adjustment` generators become real —
      an adjustment clip applies its chain to everything composited below
      it (CapCut semantics).
- [ ] **Commands**: `AddEffect` / `RemoveEffect` / `SetEffectParam` —
      undoable, in the agent vocabulary ("add a blur to the background
      clip").
- [ ] **Starter effect pack** (~10, all parametric, all keyframable via
      M2): gaussian blur, sharpen, pixelate/mosaic, glitch (RGB split +
      displacement), chromatic aberration, vignette, grain, glow/bloom,
      zoom-blur, mirror.
- [ ] **Transitions**: model = junction object between abutting clips on
      one lane (duration + transition_id + params). Compositor renders
      the overlap window with both frames. Starter set: crossfade, dip to
      black/white, wipe L/R/U/D, slide, zoom/whip, blur-through.
      Timeline UI: drop targets at junctions (timeline Phase 11), drag
      duration handles.
- [ ] **Effects panel** in the UI: browsable/searchable grid with hover
      preview, drag onto clips or lanes; transitions tab.
- [ ] **Golden-frame tests**: every effect + transition renders a fixture
      frame compared against a stored reference (the render-correctness
      backstop for all later look work).

Exit: drag a glitch onto a clip, crossfade between two clips, drop an
adjustment layer over a stack — preview and export agree.

### M5 — Color: correction, grading, LUTs, filters

Goal: the CapCut "Adjust" panel, local-first.

- [ ] **Color pipeline pass** in the effect graph (one fused WGSL pass,
      ordered): white balance (temp/tint) → exposure/contrast/highlights/
      shadows/whites/blacks → saturation/vibrance → curves → HSL → LUT.
      All parameters `Param`-keyframable.
- [ ] **Basic correction UI**: the slider stack on clip + adjustment-layer
      inspectors.
- [ ] **Curves**: Luma/R/G/B spline editor (catmull-rom → 1D LUT texture).
- [ ] **HSL secondary**: 8 color bands × hue/sat/luma shift.
- [ ] **Color wheels**: lift/gamma/gain shadows/midtones/highlights.
- [ ] **LUT import**: `.cube` (3D LUT) parser → 3D texture sampling;
      intensity slider; LUT browser fed by a user folder
      (`~/.cutlass/luts/`) + a small bundled pack.
- [ ] **Filters = color presets**: a filter is a saved parameter set for
      the color pass (+ optional grain/vignette) — shippable as data,
      user-saveable, agent-applicable ("make it look cinematic teal").
      This is what the `Filter` placeholder becomes.
- [ ] **Scopes (stretch)**: histogram + waveform monitor in the preview
      panel — compute on the GPU from the already-composited frame.

Exit: log-ish footage can be corrected, graded with curves/HSL/wheels,
finished with an imported `.cube` LUT — keyframable, exportable.

### M6 — Keying, masks, blend modes

Goal: compositing power tools.

- [ ] **Chroma key** effect: key color picker (eyedropper on the preview),
      intensity/range, shadow retention, spill suppression, matte
      view toggle. GPU pass producing alpha; quality target = CapCut's
      well-lit-green-screen results.
- [ ] **Masks** on any visual clip: linear, circle/ellipse, rectangle
      (rounded), with feather, opacity, invert; mask transform is
      `Param`-keyframable (animated reveals). Preview gets on-canvas mask
      handles (the Phase-3/4 gesture pattern). Pen/bezier mask = post-v1.
- [ ] **Blend modes** per clip: normal, multiply, screen, overlay,
      darken, lighten, add, soft light — in the blit shaders' blend
      stage.
- [ ] **Background removal seam**: the mask/matte plumbing lands here so
      M9's AI matting just supplies an alpha stream into an existing
      pipeline.

Exit: green-screen comps, masked split-screens, and blend-mode looks work
in preview and export.

### M7 — Text & graphics, for real

Goal: titles people actually publish.

- [ ] **Rich text model**: font family (system enumeration via
      `fontdb`/cosmic-text), size, weight/italic, fill color, stroke
      (color/width), shadow (color/offset/blur/alpha), background card
      (color/radius/padding), letter/line spacing, alignment, multi-line.
- [ ] **Raster pipeline upgrade**: text rasters at effective transformed
      size (kills the scale-blur noted in `preview-roadmap.md`), cache
      keyed on full style + size.
- [ ] **Text inspector**: CapCut's style panel (font picker with preview,
      color swatches, stroke/shadow/background sections).
- [ ] **Text animations**: in / loop / out slots (typewriter, fade,
      slide, pop, wave, …) implemented as parameter presets over M2
      keyframes + per-glyph offsets where needed; duration handles in the
      inspector.
- [ ] **Text templates**: preset bundles (style + animation + layout)
      shipped as local data packs; user-saveable presets.
- [ ] **Stickers become real**: animated sticker rendering — Lottie
      (`rlottie`-class rasterizer or pre-rendered), APNG/animated-WebP,
      and static PNG packs; sticker panel with a small bundled pack + a
      user folder (`~/.cutlass/stickers/`). The `Sticker` generator and
      lane finally render.
- [ ] **Caption-track groundwork**: a styled subtitle lane (uniform style,
      per-cue text/timing) — the rendering target M9's auto-captions fill.

Exit: a CapCut-style animated title and a sticker can be styled, animated,
previewed, and exported.

### M8 — Audio suite

Goal: sound that doesn't need a DAW round-trip.

- [ ] **Volume envelopes**: `volume` as keyframed `Param` flowing into
      both mixers; envelope handles drawn on audio clips (CapCut line +
      points UX).
- [ ] **Fades**: fade-in/out handles on clip corners (sugar over the
      envelope evaluation, stored as the M1 fields).
- [ ] **Varispeed audio**: time-stretch with pitch preservation
      (`signalsmith-stretch`-class or rubberband) so M1/M2 speed clips
      finally play sound; pitch-shift toggle (chipmunk mode optional, as
      CapCut offers).
- [ ] **Audio ducking**: sidechain — detect speech-band energy on chosen
      "voice" lanes, auto-keyframe music lanes down (attack/release/
      threshold/amount controls, written as ordinary volume keyframes so
      ducking is inspectable and editable after the fact).
- [ ] **Noise reduction**: rnnoise-class denoise as an audio effect on
      clips (offline render into the mixer path, cached).
- [ ] **Beat detection**: onset analysis (local DSP) → beat markers on
      audio clips; "snap to beats" in the timeline magnet; the substrate
      for auto beat-sync edits (agent + M9).
- [ ] **Audio scrub bursts** while dragging the playhead (the reserved
      `AudioReader` seam from `playback-roadmap.md` Phase 4).
- [ ] **MP3 frame-exact seek index** (lazily built) — kills the known
      tens-of-ms offset debt.

Exit: music ducks under narration, denoised voice, beat-snapped cuts,
audible speed ramps.

### M9 — AI media tools

Goal: the CapCut AI features people use daily — local-first, provider-
abstracted. Depends on: M3 (agent + providers), M6 (matte plumbing),
M7 (caption track), M8 (beat markers).

- [ ] **`cutlass-ml` crate**: local inference runtimes behind traits —
      whisper.cpp (transcribe), ONNX Runtime (matting), a local TTS
      (Piper/Kokoro-class). Models downloaded on demand to
      `~/.cutlass/models/` with checksums; every capability also has a
      provider seam for cloud later.
- [ ] **Auto captions**: transcribe audio lanes → styled cues on the M7
      caption track; word-level timestamps; edit text in the inspector;
      caption style presets. Translation rides cloud providers later.
- [ ] **Transcript-based editing (flagship)**: transcript panel where
      selecting/deleting words ripple-cuts the underlying clips (text ↔
      time mapping from whisper word stamps, edits emitted as ordinary
      ripple commands → undoable). This + the M3 agent is the "AI-first"
      identity, shipped.
- [ ] **Silence removal / AutoCut**: energy-based silence detection →
      proposed cut list rendered as a preview (dry-run UI from M3) →
      one-click apply as a single history group.
- [ ] **Text-to-speech**: text clip / script → voiceover audio clip,
      local voices; provider seam for premium cloud voices.
- [ ] **Background removal**: video matting (RVM/MODNet-class ONNX) →
      alpha matte stream feeding the M6 matte input; cached per clip
      like proxies; quality toggle (fast/quality models).
- [ ] **Agent superpowers**: the agent gains tools over all of the above
      ("caption this video and cut the silences", "duck the music under
      my voice", "cut on the beats") — each is just commands + analysis
      tools, no new safety surface.
- [ ] **Cloud provider expansion**: Anthropic/Gemini-native adapters,
      provider picker UI, per-feature provider routing (e.g. local
      whisper + cloud LLM). Config-only for users.

Exit: import an interview → auto captions → delete filler words in the
transcript → TTS an intro line → "make the music duck under speech" — all
local, all undoable.

### M10 — Performance, platform, release

Goal: ship v1 like a real product.

- [ ] **Wire proxies into preview**: `build_proxy` output registered in
      `cutlass-cache`, decoder opens proxy for preview when present,
      background proxy generation on import (toggleable), original media
      always used for export. (Or, if 4K-cold benchmarks say the frame
      cache already wins on target hardware: delete the proxy claim and
      the dead code — decide with measurements, not vibes.)
- [ ] **Shared-texture preview**: composite into a wgpu texture Slint
      renders directly (same wgpu 28 instance already shared), killing
      the readback+copy that dominates 4K frame cost. Design pass flagged
      in `playback-roadmap.md` Phase 2.
- [ ] **Long-timeline hardening**: stress tests (1h timeline, 500 clips,
      dense keyframes); fix what profiling finds; bench guardrails in CI.
- [ ] **Windows**: build, package (FFmpeg bundling), CI lane, installer.
- [ ] **macOS**: Developer ID signing + notarization; universal or Intel
      lane decision.
- [ ] **Linux**: AppImage or Flatpak with bundled FFmpeg (drop the
      "install libs yourself" tarball as the only option).
- [ ] **Export expansion**: HEVC, MOV container, GIF, audio-only
      MP3/WAV, bitrate control, cover-frame picker, social presets
      (YouTube 16:9, TikTok/Reels 9:16).
- [ ] **Project format freeze**: v1 schema documented, migration tests
      from every alpha schema, forward-compat policy published.
- [ ] **Docs & onboarding**: user guide, shortcut sheet, sample project,
      model-download UX for AI features, honest feature matrix on the
      README.
- [ ] **QA pass**: crash triage, pathological-media corpus (VFR, odd
      pixel formats, broken files), memory/GPU leak soak tests.

Exit: v1.0 tagged on three platforms, notarized/signed, with a project
format we promise to keep opening.

---

## 6. Continuous tracks (every milestone)

- **Tests**: every new command ships with inverse round-trip tests;
  every renderer feature ships with a golden-frame test (from M4 the
  fixture corpus is mandatory); the agent ships with stub-provider eval
  tests. UI gets smoke automation when Slint's testing story allows;
  until then, pure-Rust resolver tests stay mandatory (the existing
  pattern).
- **Benchmarks**: `playback_bench` + criterion benches run before/after
  any hot-path change; per-frame param evaluation, effect passes, and
  matte sampling are hot paths by definition (see `perf.mdc`).
- **Docs**: each milestone spawns/updates its feature-area roadmap doc in
  this folder, same format as `timeline-roadmap.md`.
- **CHANGELOG + alpha releases**: keep tagging alphas per milestone —
  M0 → `alpha-0.2`, M3 → first "AI" alpha, etc. Real users on every
  milestone is how we keep "we follow CapCut" honest.

## 7. Dependency graph (why this order)

```
M0 (lifecycle/honesty)
 └─ M1 (model fields: speed/volume/images/crop)
     └─ M2 (Param/keyframes)  ← keystone
         ├─ M3 (AI agent foundation) ──────────────┐
         ├─ M4 (effect engine, transitions)        │ agent vocabulary
         │    ├─ M5 (color/LUT/filters)            │ grows with every
         │    └─ M6 (keying/masks/blends)          │ milestone
         ├─ M7 (text/stickers/caption track)       │
         └─ M8 (audio suite)                       │
              └─ M9 (AI media tools) ←─────────────┘  (needs M3+M6+M7+M8)
                   └─ M10 (perf/platform/release)
```

M3 can start the moment M0 lands (it depends only on the command layer);
it's sequenced after M2 in the numbering because the keyframe commands
should be in its vocabulary from day one, but the crates can be built in
parallel by separate efforts.

## 8. Risks & open questions

- **Param/keyframe system (M2) is the riskiest design.** It touches
  models, engine hot path, inspector, timeline, and export at once.
  Mitigation: constant-param migration first (zero behavior change),
  keyframes second; benchmark `resolve_layers` before/after.
- **Effect graph perf at 4K60.** Ping-pong passes + readback could eat
  the headroom Phase 2 won. Mitigation: golden benches per effect, fuse
  the color pipeline into one pass, land shared-texture preview (M10's
  item) earlier if measurements demand it.
- **Local LLM quality for the agent.** Small local models may tool-call
  poorly. Mitigation: the OpenAI-compatible provider works day one for
  users with a key; design prompts around a compact command vocabulary
  (it's a closed schema — much easier than open-ended codegen); dry-run
  preview means bad plans are inspectable, not destructive.
- **Lottie/sticker rendering** is a rabbit hole. Mitigation: ship APNG/
  WebP packs first; Lottie only if a maintained Rust rasterizer holds up.
- **FFmpeg licensing** for bundled Windows/Linux builds (GPL components,
  x264). Decide the build flags + license posture in M10, not at tag time.
- **Scope creep is the existential risk.** CapCut is a decade of features
  with a billion-dollar org behind it. The matrix above is the line:
  things not in it don't enter v1 without going through this doc first.

## 9. v1 release checklist (the bar)

- [ ] All milestone exit criteria M0–M10 met.
- [ ] A non-developer can: install on macOS/Windows/Linux, import 4K
      footage + music + images, edit (cuts, ramps, keyframes, transitions,
      effects, color + LUT, chroma key, masks, styled animated text,
      stickers), caption it with AI, prompt the agent through real edits,
      duck the music, and export 4K HEVC — without touching a terminal.
- [ ] No data loss across crash/kill during edit, save, or export.
- [ ] Projects from the first v1 beta open in v1.0.
- [ ] Playback realtime at 4K60 on the M-series reference box, 1080p60 on
      a mid-range Windows laptop, with the look stack applied.
- [ ] Zero phantom UI: every visible control does something.
