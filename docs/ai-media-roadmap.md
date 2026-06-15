# AI media tools roadmap — M9

The CapCut AI features people use daily — captions, transcript editing,
silence removal, text-to-speech, background removal — local-first and
provider-abstracted. This is the feature-area plan for `v1-roadmap.md` § M9.

Policy reminder: **AI is a first-class, provider-abstracted feature.** Local
inference is the default where it's feasible; every capability also has a
provider seam so cloud models plug in later (M9's last phase) without touching
the feature code. **AI proposes, the engine disposes**: every media-tool edit
lowers to ordinary `cutlass_commands`, applied through normal dispatch, grouped
per action, and undoable like any gesture.

## Dependency reality (why the order below)

M9's v1-roadmap entry depends on M3 (agent + providers), M6 (matte plumbing),
M7 (caption track), and M8 (beat markers). As of `alpha-0.4.0`:

- **M3 ✅** — the agent, providers, wire/validate/eval harness all shipped.
- **M8 ✅** — the audio suite shipped; its pure-DSP-in-`cutlass-decoder`
  pattern (beat detection, ducking, denoise) is what the energy-based tools
  here reuse directly.
- **M6 ❌ not started** — no mask/matte input in the compositor yet, so
  **background removal** (which feeds an alpha matte into the M6 pipeline) is
  *gated* and lands after M6.
- **M7 ❌ not started** — no styled caption track to render cues into, so
  **auto captions** are *gated* and land after M7.

So the order is **unblocked-first**: silence removal (no model, no gated dep)
ships first, then the model-backed transcribe foundation and the features that
ride it, and the two gated features land when their substrate (M6/M7) exists.

## Architecture invariants (apply to every phase)

- **Provider-abstracted, local-first, never local-only.** `cutlass-ml`
  defines the inference traits (`Transcribe`, `Matte`, `Tts`, …); local
  runtimes (whisper.cpp, ONNX Runtime, a Piper/Kokoro-class TTS) land first,
  cloud adapters are additive. No feature hard-codes a runtime.
- **Models are data, downloaded on demand.** Model files live in
  `~/.cutlass/models/` (the config-dir convention `recent.json` / `autosave/`
  / `config.toml` established), fetched on first use with a checksum, never
  bundled into the binary or a project file.
- **AI proposes, the engine disposes.** Analysis produces a *proposed* edit
  (a cut list, a set of cues, a generated audio clip); applying it is ordinary
  commands in one history group with `rollback_group` on failure. The dry-run
  preview from M3 (the action list before applying) is reused as the review
  surface for destructive tools like AutoCut.
- **Pure DSP / inference stays model-free and off the UI thread.** The
  sample-domain analysis (silence, energy) takes `&[f32]` and returns plain
  `Vec`s with no media/model/timeline types — the M8 seam (`detect_beats`,
  `speech_band_energy`) — so the tricky parts unit-test on synthetic input.
  The engine owns decode and timeline mapping; inference runs on a worker.
- **The vocabulary grows by the M3 checklist.** Every agent-exposed tool is a
  wire DTO + validation + action-log phrasing + a versioned schema-snapshot
  bump + a stub-provider eval. No tool joins the vocabulary by accident.

## Status legend

- [x] shipped
- [ ] not started / in progress

---

## Phase 1 — Silence removal / AutoCut (unblocked, model-free)

Energy-based silence detection → a proposed cut list → reviewed in the M3
dry-run preview → applied as one undoable history group of ripple deletes.
Needs no model and no gated dependency: it's pure DSP in `cutlass-decoder`
(next to beat detection / ducking, the M8 precedent — **not** `cutlass-ml`,
which is for model-backed inference) plus the existing ripple commands.

- [x] **Silence DSP** (`cutlass-decoder/src/audio/silence.rs`):
      `detect_silences(mono, sample_rate, &SilenceSettings) -> Vec<(f64, f64)>`
      returns silent spans in seconds. A control-rate (≈100 Hz, the ducking
      hop) broadband RMS envelope marks hops below `threshold`; contiguous
      below-threshold runs longer than `min_silence` become spans, each shrunk
      inward by `keep_padding` so word onsets/offsets aren't clipped. Pure,
      unit-tested on synthetic tone-burst / silence signals.
- [x] **Cut planner** (engine, pure): silence spans (decoded per media clip,
      mapped seconds → timeline ticks through the clip's live window) → a list
      of timeline ranges to ripple-delete on the clip's track, merging spans
      that abut after frame-rounding and clamping each to the clip's own
      `[start, end)`. Unit-tested as a pure planner (no decode).
- [x] **`RemoveSilences` command + engine action**: decode the target clip,
      run the planner, and apply the cut ranges as split + ripple-delete on the
      clip's track in one undoable entry. The inverse is a single track-clips
      snapshot (`SetTrackClipsAction`) rather than a composition of the
      primitives' inverses — composing those re-mints clip ids on redo and
      strands the chained ripple-delete; the snapshot restores the exact clips
      (ids included) and oscillates cleanly. Rejected on generated clips, media
      without audio, and retimed clips (the seconds → tick map is linear only
      at 1×). Mirrors the `DuckLanes` "analysis writes ordinary edits" shape.
      *Deferred to a follow-up: linked A/V ripple-together and a whole-timeline
      magnet ripple (today the cut ripples the target clip's own track).*
- [x] **Agent tool**: `remove_silences` (clip id + optional `min_pause` /
      `padding` / `threshold`) lowers to `RemoveSilences` — "cut the silences
      out of the interview" from a prompt. Wire DTO + validation (rejects
      generated / silent / retimed with named reasons) + action-log phrasing +
      schema snapshot bump (v19) + dry-run & rejection evals.
- [x] **UI**: a **"Remove silences"** button in the audio clip inspector runs
      it on the selected clip as one undoable history entry, with broadcast-sane
      defaults (-40 dB gate, 0.5 s minimum pause, 80 ms padding) and hidden on
      retimed clips. *Deferred to a follow-up: the M3 dry-run review surface
      for the proposed cuts and threshold / min-pause / padding controls.*

## Phase 2 — `cutlass-ml` crate + transcribe foundation

- [x] **`cutlass-ml` crate scaffold**: a workspace member kept off
      `default-members` (like the planned `cutlass-py`), so the editor build
      stays lean. The `Transcribe` trait is the provider seam (blocking,
      sample-domain `&[f32]` in, plain data out — the M8 DSP convention), with
      `Transcript` / `Segment` / `Word` result types (word timing,
      serde-serializable), `TranscribeOptions`, distinct `TranscribeError`
      kinds, and a deterministic `StubTranscriber` so downstream consumers test
      without a model on disk.
- [x] **Model download/cache helper** (`cutlass-ml/src/models.rs`):
      `ModelCache::ensure` resolves a `ModelSpec` under `~/.cutlass/models/`,
      streaming the download to a `.part` sidecar, hashing as it goes, and
      renaming into place only after the SHA-256 matches; a present, valid file
      short-circuits with no network. Pure verify/resolve unit-test offline.
- [x] **`[ml]` config table** (`cutlass-ml/src/config.rs`): mirrors `[ai]` in
      `~/.cutlass/config.toml` — pick the local whisper model or route to a
      cloud provider. Local-first: an absent/empty table yields a usable local
      configuration; cloud credentials resolve from the environment.
- [x] **Local whisper.cpp backend** (feature-gated): `WhisperTranscriber`
      implements `Transcribe` over `whisper-rs` 0.16 — load a ggml model, run
      the full pipeline on 16 kHz mono, return word-timed segments (per-token
      `t0`/`t1` → seconds, special tokens dropped), responsive cancel via the
      abort callback. Behind an opt-in `whisper` feature so the C/C++ + cmake
      toolchain never touches the default build or CI. The real model registry
      (tiny.en / base.en / small.en, official HF URLs + SHA-256s) lives in
      `models.rs` and feeds the cache.
- [ ] **Word-level transcription** (engine + worker): audio lane → decode at
      16 kHz mono → `Transcribe` → segments + word timestamps mapped to source
      ticks, the substrate both captions (Phase 4) and transcript editing
      (Phase 3) consume. Off-thread, progress-reported through the established
      handle pattern.

## Phase 3 — Transcript-based editing (flagship) — needs Phase 2

- [ ] **Transcript panel**: words mapped to source time from whisper stamps;
      selecting/deleting words emits ordinary ripple-cut commands on the
      underlying clips (one undoable history group), so editing the text edits
      the video. This + the M3 agent is the "AI-first" identity shipped.

## Phase 4 — Auto captions — **gated on M7 (caption track)**

- [ ] Transcribe audio lanes → styled cues on the M7 caption track; word-level
      timing; edit text in the inspector; caption style presets. Translation
      rides cloud providers later. *Blocked until M7 lands the styled subtitle
      lane to render cues into.*

## Phase 5 — Text-to-speech — needs a `cutlass-ml` TTS runtime

- [ ] A text/script → voiceover audio clip path: a local TTS runtime behind
      the `Tts` trait, the generated audio added as an ordinary audio clip;
      provider seam for premium cloud voices.

## Phase 6 — Background removal — **gated on M6 (matte plumbing)**

- [ ] Video matting (RVM/MODNet-class ONNX, via `cutlass-ml`) → an alpha matte
      stream feeding the M6 matte input; cached per clip like proxies;
      fast/quality model toggle. *Blocked until M6 lands the matte/mask input
      in the compositor.*

## Phase 7 — Agent superpowers

- [ ] The agent gains tools over all of the above — "caption this and cut the
      silences", "cut on the beats" — each just commands + the analysis tools,
      no new safety surface beyond the M3 checklist per tool.

## Phase 8 — Cloud provider expansion

- [ ] Anthropic/Gemini-native adapters, a provider-picker UI, and per-feature
      provider routing (e.g. local whisper + cloud LLM). Config-only for users.

---

## Exit

Import an interview → auto captions → delete filler words in the transcript →
TTS an intro line → "make the music duck under speech" — all local, all
undoable. (Captions and background removal land with M7 and M6 respectively;
silence removal, transcript editing, and TTS do not wait on them.)
