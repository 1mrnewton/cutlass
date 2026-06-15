# Changelog

## [Unreleased]

The **audio alpha** (M8): sound that doesn't need a DAW round-trip. Clip
volume becomes a keyframable envelope, fades are corner handles, retimed
clips finally play (pitch-corrected) — including speed ramps — music ducks
itself under narration, clips can be denoised and beat-detected, the
playhead chirps while you scrub, and MP3 seeks land sample-close. Preview
and export agree on every one, and each edit is a single undo.

### Volume envelopes (M8 Phase 1)

- **Clip volume is now a keyframable envelope.** `Clip.volume` is a
  `Param<f32>` (the M2 keyframe type), so the gain can ride eased keyframes
  across a clip, not just sit at a constant. Backward-compatible on disk: a
  constant still serializes as the bare value (`"volume": 0.8`,
  byte-identical to pre-M8 saves) and only an animated clip writes the
  `{"kf":[...]}` form — old projects load unchanged.
- **Both mixers sample the envelope per sample-frame.** A shared
  `audio_gain_at` evaluates the envelope and multiplies the fades on top;
  each mixer rebases the clip-relative tick keyframes into sample frames
  once per span so the hot per-sample lookup stays an O(log k) compare. The
  unity fast path (constant 1.0, no fades) still skips the gain loop.
- **Inspector + on-clip UI.** A keyframe diamond on the Volume row reads the
  gain at the playhead and adds/removes a point; the slider sculpts the
  keyframe on an animated clip or sets the flat level on a constant one. The
  gain curve is drawn over the waveform with a dot per keyframe, and a
  timeline envelope chip marks an animated clip.
- **Agent vocabulary.** `volume` joins the agent's keyframe tools so a
  prompt like "fade the music down under the voice" writes a real envelope.

### Fades as corner handles (M8 Phase 2)

- **Fades preserve automation.** `set_clip_audio`'s volume is now optional:
  setting a flat level flattens the envelope (the basic slider), while a
  fade-only edit keeps the gain (constant or keyframed) and just moves the
  ramps — so "fade the music out" past a keyframed clip no longer wipes it.
- **Drag the corners.** Grab the top corners of a selected audio clip to set
  fade-in (left) / fade-out (right) — a darkening triangle with a grab dot
  per ramp, committed as one envelope-preserving edit on release.

### Varispeed audio (M8 Phase 3)

- **Retimed clips play again.** Both mixers drop the "audio mutes while
  retimed" rule and render speed ≠ 1× and reversed clips through an offline
  per-span time-stretch (`signalsmith-stretch`, MIT) that is computed once,
  cached, and served 1:1 — so preview and export use identical samples and
  reverse is a buffered flip.
- **Pitch lock.** `Clip.preserve_pitch` (serde-default true) drives the
  transpose: pitch-corrected stretch by default, pitch-follows-speed
  ("chipmunk") when off, via a `set_clip_pitch` command, a "Keep pitch"
  switch in the Speed inspector (flips the whole A/V link group), and an
  agent tool.
- **Speed ramps too.** A time-varying render warps the audio along the same
  normalized curve the picture uses, so velocity ramps stay in sync and
  preview matches export.

### Audio ducking (M8 Phase 4)

- **Auto-lower music under speech.** A sidechain analysis band-passes the
  voice (300–3400 Hz), follows its RMS, and turns it into a
  threshold/attack/release gain-reduction curve, thinned to the few points a
  volume envelope needs. The `duck` flow writes the result as **ordinary
  volume keyframes** on each music clip — dipped onto the clip's own level,
  not overwritten — so the ducking is inspectable and editable after the
  fact, and both mixers duck identically with no extra plumbing. One undo.
- **Two ways in.** A `duck` agent tool ("duck the music under the
  narration"), and a CapCut-style UI: tag a lane as the voice with a **"V"**
  toggle in the track header, then hit **"Duck under voice"** in a music
  clip's audio inspector.

### Noise reduction (M8 Phase 5)

- **One-click denoise.** A "Reduce noise" toggle runs a clip's audio through
  RNNoise (`nnnoiseless`, a pure-Rust port — no C binding, no model file to
  ship) to suppress steady background noise (hiss, hum, room tone) while
  keeping speech. Like varispeed it's an offline per-span render, computed
  once and cached, so preview and export are bit-identical; denoise also
  stacks on top of a retimed clip's stretched audio.
- **Model + agent.** `Clip.denoise` (media clips only, absent from saves
  when off) with an undoable `SetClipDenoise` command and a `set_denoise`
  agent tool that steers a video-lane target to its linked audio companion.

### Beat detection & snap (M8 Phase 6)

- **Find the beats.** A local spectral-flux onset detector marks the beats
  on an audio clip; **"Detect beats" / "Re-detect" / "Clear"** live in the
  audio inspector and the markers draw as ticks along the clip's bottom edge.
  Beats are stored in source ticks, so they ride the content through trims
  and splits.
- **Snap to them.** The timeline magnet snaps clip edges onto a clip's beat
  ticks, alongside the existing edge / playhead candidates — the substrate
  for agent and M9 beat-sync. A `detect_beats` agent tool drives it from a
  prompt.

### Smaller audio wins

- **Audio scrub bursts (M8 Phase 7).** Dragging the playhead while paused
  now chirps a short (~85 ms) audio burst from the scrubbed position, so you
  can hear where you are. The burst never advances the transport — the drag
  drives the playhead, not the clock — and the newest position always wins.
- **MP3 frame-exact seek (M8 Phase 8).** A lazily-built, byte-exact MP3 seek
  index re-anchors each mid-stream seek from the true frame offset instead of
  FFmpeg's estimated PTS, killing the tens-of-ms MP3 seek error noted in the
  previous alpha. MP4/AAC was already sample-accurate.

### Silence removal / AutoCut (M9 Phase 1)

- **Cut the silences in one move.** A new energy-based silence detector
  (`detect_silences`, pure DSP next to beat detection / ducking) finds the
  pauses in a clip's audio; the engine maps them to timeline ticks and
  ripple-deletes each span so the remaining speech closes up. The whole pass
  is a single undoable history entry — one undo restores every clip and
  shift. Rejected on generated, silent, and retimed clips.
- **In the inspector.** A **"Remove silences"** button in the audio clip
  inspector runs it on the selected clip (broadcast-sane defaults: -40 dB
  gate, 0.5 s minimum pause, 80 ms padding). Hidden on retimed clips.
- **From a prompt.** A `remove_silences` agent tool (optional `min_pause` /
  `padding` / `threshold`) lowers to the same command, so "cut the silences
  out of the interview" just works. Tool schema → v19.

### `cutlass-ml` foundation (M9 Phase 2)

- **New inference crate.** `cutlass-ml` opens the local-first, provider-
  abstracted home for model-backed media tools. The `Transcribe` trait is the
  seam (blocking, sample-domain `&[f32]` in, plain data out — the same shape
  as the audio DSP), with `Transcript` / `Segment` / `Word` result types
  carrying word-level timing, and a deterministic `StubTranscriber` so the
  features that consume transcripts can be built and tested without a model on
  disk. The crate is off the default build (like the planned `cutlass-py`).
- **Models are data, fetched on demand.** A model cache resolves and downloads
  weights under `~/.cutlass/models/`, streaming to a `.part` sidecar and
  verifying a SHA-256 before installing atomically — a present, valid file
  costs no network. An `[ml]` config table (mirroring `[ai]`) picks the local
  whisper model or routes to a cloud provider.
- **Local whisper.cpp backend.** `WhisperTranscriber` runs whisper.cpp (via
  `whisper-rs`) on 16 kHz mono and returns word-timed segments — the substrate
  for captions and transcript editing. It's behind an opt-in `whisper` feature,
  so the C/C++ + cmake toolchain never touches the default build; a model
  registry (tiny/base/small `.en`, official Hugging Face builds verified by
  SHA-256) feeds the cache. (Engine/worker wiring lands next.)

### Auto captions (M9 Phase 4)

- **Captions are just text clips.** Auto-captioning a clip decodes its speech,
  transcribes it through the injected `Transcribe` backend, and adds the words
  as subtitle-styled text clips on a fresh **"Captions"** lane — ordinary
  `Generator::Text` clips, so they restyle, move, retime, and delete like any
  title afterward (matching how CapCut treats captions). The whole pass is a
  single undoable history entry: one undo removes the lane and every cue.
- **Readable cue lines, not raw words.** A pure cue planner (`cutlass-ml`)
  packs word-timed speech into lines that respect a character budget, a max
  duration, inter-word pauses, and sentence boundaries, falling back to one cue
  per segment when word timing is absent — so cues read like subtitles instead
  of a word-per-clip stutter.
- **In the inspector.** A **"Speech to text"** tab in the audio clip inspector
  carries a **"Generate captions"** button; the UI worker lazily builds the
  whisper backend on first use (downloading the configured model if needed)
  behind the opt-in `whisper` feature, runs off the UI thread, and publishes
  the result. Rejected on generated, retimed, and audio-less clips.
- **From a prompt.** A `caption_clip` agent tool lowers to a new `CaptionClip`
  command that `Engine::apply` routes to the same caption path, so "add
  captions to the interview" just works. Tool schema → v20.
- **Deferred.** Captions run synchronously on the worker for now (no progress
  bar / off-thread cancellation surfaced in the UI yet), and a soft
  caption-group identity for batch restyle/export is left for M7's caption
  track work — captions land as independent text clips until then.

### Transcript-based editing (M9 Phase 3)

- **Editing the text edits the video.** A new **Transcript** panel transcribes
  the selected clip and lays out its words; select a run of words, hit delete,
  and that span is ripple-cut out of the clip so the remaining speech closes
  up — one undoable history entry. Clicking a word seeks the playhead to it.
  This + the M3 agent is the "AI-first" identity.
- **A shared ripple-cut primitive.** The split + cut + ripple-close edit that
  silence removal grew was generalized into `action::edit::ripple_cut` over
  arbitrary timeline ranges (the snapshot inverse that survives redo without
  re-minting clip ids came along), and the mono audio decode both paths share
  moved to a `clip_audio` seam. Silence removal now rides the same primitive.
- **Read-only transcription, durable delete.** `Engine::transcribe_clip` runs
  the backend off a `&Engine` (no mutation; rejected on generated, retimed, and
  audio-less clips), while `Engine::ripple_delete_ranges` applies the word
  delete as one undoable cut. The panel reflows in place — struck words park at
  the cut start and later words shift left by the removed span — so transcript
  and timeline stay in lockstep without a re-transcribe.
- **Deferred.** The transcript is transient (re-derived per session, not saved
  in the project), there's no transcript agent tool yet ("delete the filler
  words"), no linked A/V ripple-together, and you can only delete spans — not
  yet rename/replace a word.

## [alpha-0.4.0] — 2026-06-15

The **Windows & performance alpha**: Windows joins macOS and Linux with
real double-click installers (both x64 and arm64), preview gets
dramatically faster on high-resolution footage, and the library learns
to delete media — referenced sources cascade their clips away in a
single undo.

### Windows support

- **Real installers, not just portable archives.** A new Inno Setup
  build packages `cutlass-ui.exe` + the bundled FFmpeg DLLs + licenses
  into a single `Setup.exe` (Program Files install, Start-menu shortcut,
  uninstaller, optional desktop icon); the portable `.zip` still ships
  alongside it. Both Windows and macOS now build **native artifacts for
  each architecture** in CI — Windows x86_64 + arm64, macOS Apple Silicon
  + Intel.
- **Native Windows window frame.** The macOS custom-title-bar approach is
  generalized to Windows: keep the OS-drawn frame (native resize, Aero
  snap, drop shadow, rounded corners) and only suppress the caption so
  the custom Slint title bar shows through (`WM_NCCALCSIZE` reclaims the
  caption strip, `WM_NCHITTEST` re-adds the top resize band). Linux/BSD,
  which have no "frame minus titlebar" mode, stay fully frameless.
- **Export fixed on stock Windows FFmpeg.** LGPL FFmpeg builds ship no
  libx264, so the old fallback could pick a hardware-surface-only encoder
  (e.g. `h264_d3d12va`) that rejects the pipeline's software frames and
  surfaced as "failed to open media". Encoder selection is now
  format-aware: prefer software libx264/libopenh264, otherwise fall back
  to a CPU-frame-capable hardware encoder (Media Foundation, then
  NVENC/AMF/QSV) and feed it NV12 — a surface-only encoder is never handed
  to the software pipeline.

### Faster preview

- **Preview runs at a 720p cap, end-to-end.** A decode miss now
  downscales the native frame to the preview height *before* it enters
  the cache, so the frame cache, GPU upload, composite, canvas, and UI
  readback all shrink with it (~9× fewer bytes versus 4K). Decode still
  runs at native resolution and **export is untouched** (full source, no
  cache); `import_media` registers each source's cache spec at the scaled
  dims, so flipping the cap auto-drops the stale on-disk index.
- **Playback stutter fixed.** Frames are now cached under the requested
  `target_ticks`, not the decoded frame's PTS — which rarely matched on
  off-grid rates (e.g. 60.03 fps), so every revisit missed and the
  read-ahead prefetch turned each miss into a backward seek (~3 fps and an
  `mmco: unref short failure` flood). Same key on both sides means
  prefetch warms exactly what the render reads: the render+prefetch path
  drops from **~325 ms to ~2.7 ms per frame (3 → 365 fps)**, guarded by a
  new `playback_prefetch` bench.
- **No more per-frame GPU texture churn.** The preview hot path allocated
  three ~12 MB upload textures per 4K frame and issued two queue submits.
  Upload textures now live in a pool (bucketed by format/size, reused
  across frames) and the canvas→buffer copy folds into the render encoder
  for a single submit. Warm 4K `get_frame` on an M5 Pro: 4K24 17.4 → 8.3 ms
  mean, 4K60 p95 28.1 → 9.8 ms — well under the 20 ms / 50 fps budget.
- New `scale_yuv420p` swscale helper in `cutlass-decoder` (AREA
  resampling, PTS preserved) backs the downscale; identity and invalid
  target sizes short-circuit.

### Library: media management

- **Remove media from a project.** Right-click a library tile →
  *Remove from project*. Unused sources delete immediately; a source still
  used by clips raises a confirm dialog that removes the referencing clips
  **and** the source as one undoable cascade — emptied lanes are pruned and
  the source's cached thumbnail is evicted.
- New undoable `RemoveMedia` command (the inverse of media insert). The
  model rejects removing a referenced source unless its clips are removed
  in the same history group, so a stray delete can never orphan a clip.
- Library tiles show a per-source **clip usage count**, computed in one
  pass over the timeline — the delete flow reads it to decide whether to
  drop a source straight away or confirm the cascade first.

### UI

- Library panel restyle: a darker surface palette, line-duotone tab icons
  (media / audio / text / stickers / effects / transitions), and larger
  tabs with a scroll view for overflow.

### Downloads

| Platform | Artifact |
| --- | --- |
| Windows (x64 / arm64) | `Cutlass-*-windows-*-Setup.exe` — run the installer; or the portable `Cutlass-*-windows-*.zip` |
| macOS (Apple Silicon / Intel) | `Cutlass-*-macos-arm64.zip` / `Cutlass-*-macos-x86_64.zip` — unzip, drag `Cutlass.app` to Applications. **First launch:** right-click → Open (not notarized). See `INSTALL-macos.txt`. |
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
- **Crop is numeric-only** — no draggable crop-handles mode in the
  preview yet.
- **Agent quality tracks the model you give it** — small local models
  may tool-call poorly; dry-run mode previews every plan before it
  touches the timeline.
- **Alpha stability** — crashes and UI polish gaps are expected; please
  file issues.
- **MP3 seek accuracy** — mid-stream seeks on MP3 can be tens of ms off;
  MP4/AAC is sample-accurate.

## [alpha-0.3.0] — 2026-06-14

The **effects alpha**: the GPU effect engine, transitions, adjustment
layers, and the M1 close-out (canvas settings, crop & flip) all ship,
the editor gets a restyled dark-blue/gold theme with a cold-start launch
screen and a settings dialog, and a batch of animation-smoothness fixes
land in preview and export.

### Fixed

- **Preview no longer freezes keyframed motion after a drag.** Releasing a
  transform gesture (or an inspector slider) could race the preview worker's
  message coalescing: the commit was processed mid-drain and the stale
  gesture override re-applied after it, pinning the clip at the release
  transform on every later frame — animation only showed up in the exported
  file. The coalescing loops now preserve queue order, so a commit or clear
  is never followed by a stale override.
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

### Effects & transitions (M4)

- **GPU effect engine.** Clips carry an ordered **effect chain**; the
  compositor renders an affected layer placed-but-opaque into a canvas-sized
  scratch texture, ping-pongs it through each effect's passes (two scratch
  textures reused across the whole frame), then blends the result back with
  the layer's opacity. Layers without effects keep the original single-pass
  path untouched — no regression on the common case (guarded by the
  `composite` benches).
- **Starter pack of 10 effects**: gaussian blur, vignette, sharpen, pixelate,
  glitch, chromatic aberration, grain, glow, zoom-blur, mirror. Each ships a
  golden-frame test and a criterion bench. Effects are **data** — an effect
  catalog (id, label, param specs with default/min/max) lives in
  `cutlass-models`; the compositor owns the WGSL.
- **Effect parameters are keyframable**: they ride the same `Param` system as
  transforms (`ClipParam::Effect`), so the constant-value quick edit and the
  animated path share one engine.
- **Adjustment layers are real.** An adjustment clip applies its effect chain
  to the **accumulated canvas below it** (CapCut semantics) — the compositor
  closes the current pass, ping-pongs the canvas itself, then keeps stacking
  layers above. Adjustment lanes are no longer hidden.
- **Transitions at clip junctions.** A transition stored on the track blends
  the outgoing and incoming clips across a window centered on the cut; the
  engine resolves **both** clips' frames (source times clamped at media
  bounds, no repositioning) and emits a dual-input layer that a transition
  registry blends with a progress uniform. Starter set: crossfade,
  dip-to-black, dip-to-white, wipe left/right/up/down, slide, zoom, blur.
- **Structural edits prune dead junctions.** Trim / move / split / remove /
  ripple drop a transition whose clips no longer abut, inside the edit's own
  history group — so a single undo restores both the structural change and
  the junction.
- **Effects & Transitions are back in the Library** as browsable catalog
  tabs; clicking a tile applies the effect to the selected clip or the
  transition at its right junction. The inspector grows an **Effects**
  section (per-effect param sliders + remove), and the timeline shows a
  **transition pill** at each junction with edge-drag resize and a remove
  control.
- **The AI agent can do all of it**: `add_effect` / `remove_effect` /
  `set_effect_param` and `add_transition` / `remove_transition` /
  `set_transition` tools (tool schema **v10**), with catalog validation,
  action-log lines, and `describe_project` listing each clip's effects and
  transitions.
- This advances **M4 (effect engine & transitions)** on the v1 roadmap.

### Canvas settings (M1)

- Projects now own their **canvas shape and background**: aspect presets
  (16:9, 9:16, 1:1, 4:5, 21:9 — or auto, which follows the footage like
  before) and a per-project background color shown wherever no clip
  covers the canvas. Switching presets keeps the footage's quality tier
  (shortest edge) and reshapes the long edge, so a 1080p landscape
  project becomes a true 1080×1920 vertical, not a crop.
- File ▸ Canvas Settings… dialog: ratio dropdown + background color
  swatch with a live canvas-size readout; every change is one undoable
  edit (`SetCanvas`).
- Inspector grows **Fit / Fill** buttons: one click letterboxes a clip
  inside the canvas or cover-fills it (centered, crop-aware) — the
  CapCut reframe moves for "make this landscape clip fill my vertical
  canvas".
- The AI agent can do it too: `set_canvas` tool (tool schema v8) —
  "make this a vertical short with a dark grey background" works, with
  omitted fields keeping their current values; `describe_project` now
  reports non-default canvas state.
- Preview, export, and empty timelines all render the background color —
  it's the compositor's clear color, not an extra layer.
- This closes **M1 (editing core parity)** on the v1 roadmap.

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
- **Crop is numeric-only** — no draggable crop-handles mode in the
  preview yet.
- **Agent quality tracks the model you give it** — small local models
  may tool-call poorly; dry-run mode previews every plan before it
  touches the timeline.
- **Alpha stability** — crashes and UI polish gaps are expected; please
  file issues.
- **macOS Intel** — not built in CI; build from source or use Rosetta.
- **MP3 seek accuracy** — mid-stream seeks on MP3 can be tens of ms off;
  MP4/AAC is sample-accurate.

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

[alpha-0.4.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.4.0
[alpha-0.3.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.3.0
[alpha-0.2.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.2.0
[alpha-0.1.0]: https://github.com/1Mr-Newton/cutlass/releases/tag/alpha-0.1.0
