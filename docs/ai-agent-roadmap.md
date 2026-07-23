# AI Agent Roadmap — the Everything Agent (v2)

**Status (macos-dev, Jul 2026):** v2. v1 (M3) shipped the closed-vocabulary
timeline agent — wire format + validation, OpenAI-compatible provider,
sandbox-rehearse-then-replay loop, chat panel, eval harness — end to end in
`crates/cutlass-ai` and `apps/cutlass-desktop/src/agent.rs`; see
[v1 foundation (shipped)](#v1-foundation-shipped) for the compressed record.
v2 grows that agent into the app-wide agent that is the product's selling
point: an assistant that can see the project, drive the app, understand the
footage, and compute — not just edit the timeline.

Policy: **the agent is the reason Cutlass exists**, and v1 proved the trust
machinery on the highest-stakes surface (timeline mutation). v2 extends the
same discipline — validated, observable, undoable-or-confirmed — to
everything else the user can do, instead of inventing a second, looser
mechanism per feature. Local-first stays law: every capability below works
against a local model and local analysis; cloud raises quality, never gates.

## v1 foundation (shipped)

Compressed from the v1 doc (phases 0–5, all landed on this line):

- [x] **Closed command vocabulary** over the engine's command layer: wire
      DTOs shaped for LLMs (`#[serde(tag = "command")]`, seconds not ticks,
      raw-int ids), validated against a project snapshot with model-readable
      rejections; schema v21 includes M2 keyframes and look commands.
- [x] **Tool schema export** via `schemars`, one tool per command,
      `TOOL_SCHEMA_VERSION` + snapshot test — schema drift is a reviewed
      diff, never an accident.
- [x] **`describe_project` + `EditorContext`** pushed fresh each prompt
      (summary, selection, playhead, in/out) — context is pushed, not
      retrieved.
- [x] **`ChatProvider` trait + OpenAI-compatible SSE provider** (Ollama,
      llama.cpp-server, LM Studio, OpenAI, gateways), config in
      `~/.cutlass/config.toml` `[ai]`; `ScriptedProvider` test double; error
      taxonomy the UI can speak.
- [x] **The loop**: per-call validate → apply → outcome-or-rejection fed
      back; caps (tool calls, turns); dry-run; human-readable action log.
- [x] **Sandbox-rehearse-then-replay**: prompts rehearse against a
      snapshot-seeded sandbox engine, then the validated plan replays
      atomically on the live engine as **one undo group** with sandbox→live
      id remapping. One prompt = one Cmd+Z.
- [x] **Chat panel**: streamed transcript, action lines, dry-run preview
      card (Apply/Discard), undo chip, not-configured setup state, Stop.
- [x] **Eval harness**: scripted-provider prompts against fixture projects
      asserting final timeline + action log, in CI, zero network.
- [x] **Q&A without mutation** + the **vocabulary growth checklist** (wire
      DTO, schema snapshot, action line, eval case — per new command).
- v1's deferred stretch (guarded `Import` with per-prompt confirmation) is
  superseded by the tier model below.

## Vision

The agent can do (almost) everything the user can — and things they can't:

- **Senses.** Screenshot the timeline and preview, grab frames from any
  asset at any time, read analysis results — the model stops editing blind.
- **App control.** Window, panels, playback transport, zoom, selection,
  theme, caches, storage locations — "make the preview bigger and loop the
  selection" is a prompt, not a hunt through menus.
- **Project operations.** Import, export, save, open, relink, templates —
  the full project lifecycle, with confirmations where the stakes demand.
- **Semantic media understanding.** "Cut all my kills from this gameplay",
  "make C. Ronaldo edits from this match — dribbles, goals, respect
  moments", beat-synced montages, silence removal, auto-captions. The
  agent's edge over every prompt-box gimmick is that it can *watch the
  footage* and then execute frame-exact edits through the validated
  vocabulary.
- **Open-ended computation.** A Python runtime (cutlass-py + analysis
  libraries) for the long tail no fixed toolset covers.

Everything runs local-first (local models, local analysis, local Python) and
cloud-optional (frontier models and hosted vision raise quality when
configured — the cloud-roadmap BYOK rules apply unchanged).

## Trust model — validated edits plus three host tiers

Validated edits remain their own closed execution plane: every edit is
validated against a snapshot, rehearsed in the sandbox, and replayed in one
or more explicit undo phases pinned by the schema snapshot test. Host tools
cannot bypass that plane or mutate the timeline directly.

Every host tool then declares exactly one `ToolTier`:

1. **ReadOnly** — screenshots, frame grabs, state reads, and analysis
   queries. These cannot mutate app, project, or system state and never
   confirm.
2. **Workspace** — safe, reversible app/project state: panel, playback,
   window, theme, zoom, and selection control. These run immediately; the
   next call or a user click can reverse them.
3. **System** — destructive or external effects: cache clear, storage
   relocation, writes outside managed project storage, opening external
   apps, and Python execution. The loop fails closed before execution unless
   the embedding host authorizes the call. Desktop authorization follows the
   `[ai] autonomy` setting: `ask` (default) parks the call behind a per-call
   confirmation card; `full` runs without prompts. The card names the effect
   precisely ("runs now; not undoable from Cutlass" — the MCP doc's phrasing,
   same posture).

**Namespaced host tools.** Everything beyond the edit vocabulary is a host
tool named `{namespace}_{tool}` — namespaces `app`, `project`, `system`,
`media`, `analysis`, `python`, `job`. The built-in edit vocabulary stays
unprefixed. Dispatch requires exact membership in the host registry; the
namespace is validated at registration time and used for grouping, not as a
wildcard router. Unknown names fall through to the closed wire-command parser
and return a model-readable rejection. The edit schema snapshot continues to
cover only the closed vocabulary; host tools carry their own registry and
their own tests. Structural containment, not textual: no host tool can mutate
the timeline except by the model reading its result and issuing validated
edit commands.

## Status legend

- [x] shipped
- [ ] not started / in progress ("(in flight)" = landing in the current
  dev cycle)

---

## Phase 0 — Runtime foundations

What the loop, panel, and provider layer need before any new capability can
land cleanly.

- [x] **Multimodal messages**: `ImagePart` on the provider wire, a
      per-request image budget, text-only history (images are referenced,
      not resent — image tokens are the expensive kind).
- [x] **`[ai] autonomy` setting** (`cutlass-settings`): `ask` default /
      `full`, the System-tier gate above.
- [x] **ToolHost registry + loop dispatch + phase commits**:
      host tools register exact `{namespace}_{tool}` specs; the loop dispatches
      only registry members; `commit_progress` lets the model commit rehearsed edits in
      named phases — per-phase undo groups instead of one monolithic group
      for long multi-step prompts.
- [x] **Provider retries**: transient network/5xx failures retry
      with backoff instead of aborting the prompt and rolling back a
      half-built plan.
- [x] **`cutlass-jobs` background-job registry**: std-only
      `JobManager` — named worker threads, progress + detail snapshots,
      cooperative cancel, subscriber bridge for the UI, bounded finished
      history. Exports, analysis, transcription, and Python runs migrate to
      it as those phases land; `job_list` / `job_status` / `job_cancel`
      then expose the shared registry.
- [x] **Per-draft session persistence**: conversation history + transcript
      survive draft close/reopen (sidecar next to the draft, following the
      recents/autosave conventions — never inside the project file).
- [ ] **Desktop System authorization broker** (in flight): fail-closed loop
      authorization has landed; the `ask` confirmation card and `full`
      bypass are being wired to the desktop host.

Exit: a host tool registered under a namespace is dispatchable from the
loop; job lifecycle tests prove ordered progress, cancellation, and bounded
history; reopening a draft restores its conversation; a provider blip
mid-prompt is a retry, not a rollback; System calls cannot execute without
desktop authorization.

## Phase 1 — Vision: the agent gets eyes

The single highest-leverage v2 capability: every later phase assumes the
model can check its own work.

- [x] **Frame-capture primitives**: bounded project/asset renders use a
      private lazy renderer, the real compositor, safe media-cache reuse,
      exact frame snapping, path-redacted labels, and in-memory PNG output.
- [ ] **`media_*` senses**: timeline screenshot, preview frame at playhead
      or given time, frame grabs from any pool asset at any source time,
      asset thumbnail strips. All Workspace tier; all return images through
      the multimodal wire under the image budget.
- [x] **Schematic timeline map renderer**: render the timeline as a labeled schematic
      image server-side (tracks, clips with names/ids, markers, playhead,
      explicit time window) —
      cheaper, crisper, and more model-legible at small sizes than a raw UI
      screenshot, and it works headless (no Slint in the loop). The UI
      screenshot stays available for "what does the user actually see".
- [ ] **Sandbox self-verify**: after rehearsing edits, the loop offers the
      model a render of the *sandbox* state (schematic map + composited
      frame grabs) so it verifies placement/timing before the plan is ever
      presented — catch the wrong-lane title in rehearsal, not in review.
- [x] **Inline image transcript rows**: the agent panel has bounded,
      aspect-aware image cards and persists text-only placeholders. Emitting
      sensed host-tool images into those rows remains part of `media_*` wiring.

Exit: "add a title over the intro, then check it" produces a plan the model
already verified against a sandbox frame grab; the transcript shows the
grab; an eval case locks the verify loop in with a scripted vision turn.

## Phase 2a — App control

- [ ] **`app_*` tools** via winit + the existing Slint globals: window
      position/size, playback transport (play/pause/seek/loop selection),
      panel toggles (library/inspector/agent/timeline zoom), active
      selection, theme. Workspace tier — every one reversible by the next
      call or a click.
- [ ] **State echo**: each `app_*` call returns the resulting app state
      block, so the model never operates on a stale picture of the shell.

Exit: "play the last 5 seconds looped, hide the library, dark theme" works
as one prompt; evals cover the tool results; nothing here can touch project
data.

## Phase 2b — System: caches, storage, external

- [ ] **Cache registry**: one enumerable registry of app caches (proxies,
      thumbnails, waveforms, transcript/models when they land) with size,
      clear, and relocate operations — the substrate for both the Settings
      UI and the `system_*` tools.
- [ ] **`[storage]` settings + Settings UI section**: cache/storage
      locations become visible, configurable, and relocatable (move-then-
      swap, never delete-then-hope).
- [ ] **`system_*` tools**: cache sizes/clear/relocate, reveal-in-Finder,
      open-external. All System tier: `ask` autonomy shows the confirmation
      card per call.

Exit: "free up disk space" enumerates caches with sizes, proposes clears,
and executes only through confirmations (or autonomy `full`); Settings shows
the same registry the agent uses.

## Phase 2c — Project operations

- [ ] **`project_*` tools**: import media, export (spawned as a
      `cutlass-jobs` job with progress + cancel), save, open, new,
      list-drafts, relink missing media, apply/save templates.
- [ ] **Tier mapping with teeth**: reads (list-drafts, relink candidates)
      are Workspace; mutations that can lose unsaved work (open, new) or
      write outside the app dirs (export, import-by-path) are System tier
      with card copy naming the file paths involved.
- [ ] **Export-as-job**: the agent starts an export and keeps working;
      `job_status` reports progress; completion lands in the transcript.

Exit: "export this draft for YouTube and start a new project from the vlog
template" runs end-to-end with exactly the confirmations the tier table
demands, and the export is cancellable mid-flight.

## Phase 2d — Command gaps: close the vocabulary delta

The agent wire still covers a subset of the engine. Every gap follows the
v1 growth checklist (wire DTO + validation, schema snapshot, action line,
eval case).

- [ ] **New/wired engine commands**: `ExtractAudio` wired into the agent
      vocabulary, `UnlinkClips`, duplicate-with-properties, freeze frame,
      effect reorder.
- [ ] **Full `EditCommand` surface**: expand the wire until the vocabulary
      delta is zero (masks, chroma, retiming, markers — everything the UI
      can do that the wire can't yet), bumping the schema version once per
      landed batch.

Exit: the wire-vs-`EditCommand` coverage table in `cutlass-ai` docs reads
complete; schema snapshot bumped; every added command has its eval.

## Phase 3a — `cutlass-analysis`: deterministic senses

A new crate for local, model-free media analysis — the cheap-and-always-on
layer under the VLM.

- [ ] **Shot detection** (frame-difference/histogram cuts) over decoded
      frames via the existing decode path.
- [ ] **Audio DSP**: silence spans, loudness contours, beat grid — the
      substrate for silence removal and beat-synced montages.
- [ ] **Moments index (SQLite)**: a content-hash-keyed per-media cache of
      analysis results with time range + kind + confidence, written by
      analysis jobs (`cutlass-jobs`), queried by `analysis_*` tools.

Exit: importing gameplay footage and running analysis yields queryable
shots/silences/beats with timestamps; `analysis_query` answers from the
index with zero model involvement.

## Phase 3b — Transcription

- [ ] **whisper.cpp via `whisper-rs`**: local transcription as a job, word
      + segment timestamps into the moments index. (A proven prototype
      exists in the separate cutlass-playground repo — port, don't
      research.)
- [ ] **Model download manager**: on-demand whisper model fetch with
      size/license surfaced, stored under the cache registry (2b), never
      bundled into the app.
- [ ] **`analysis_transcribe` / transcript queries**: transcript search by
      word with exact time ranges — the substrate for filler-word cuts,
      quote-finding, and captions.

Exit: "cut the part where I say the sponsor name" resolves through the
transcript index to a frame-exact validated edit.

## Phase 3c — VLM moments: semantic understanding

- [ ] **Vision provider seam**: Gemini-style native video input where the
      provider supports it; OpenAI-compatible frame batches (sampled stills
      + timestamps) everywhere else — one `analysis_find_moments` surface
      over both.
- [ ] **Propose → verify pipeline**: pass 1 proposes candidate moments from
      sparse frames; pass 2 re-samples densely around candidates and
      confirms/refines boundaries before anything reaches the index —
      VLM recall is cheap, precision is what edits need.
- [ ] **Skill packs**: prompt-level moment taxonomies for gameplay (kills,
      deaths, clutches) and sports (dribbles, goals, celebrations,
      "respect" moments) shipped as data, not code — the C. Ronaldo edit is
      a skill pack plus the pipeline.

Exit: "find every kill in this match" yields indexed moments with
timestamps + confidence from footage the eval fixtures pin; "make a montage
of them" is then pure Phase-1-verified vocabulary work.

## Phase 4 — Python runtime

- [ ] **uv-bootstrapped venv, optional**: first `python_run` offers to
      provision an isolated environment (uv + pinned cutlass-py + analysis
      libs) under the app data dir; no system Python touched; fully
      removable via the cache registry.
- [ ] **`python_run` tool**: System tier, always. Scripts get the project
      handle through cutlass-py, run as a cancellable job, and return
      stdout + declared artifacts (files, images) with size caps — artifact
      images flow back through the multimodal wire.
- [ ] **Import boundary**: artifacts never auto-enter the project; media an
      agent script produces goes through the same import consent as any
      other file.

Exit: "plot the loudness of the master track" provisions (once, with
consent), runs, and renders the plot inline; a runaway script dies by
`job_cancel`.

## Phase 5a — Superpowers

The features the phases above exist for — each mostly composition:

- [ ] **Embeddings search** over transcripts + moments ("find where I talk
      about the patch notes").
- [ ] **Auto-captions** with karaoke styling (3b timestamps → text
      vocabulary; styles from the text-preset system).
- [ ] **Beat-synced montage** (3a beats + 3c moments + validated edits).
- [ ] **Silence removal** (3a silence spans → ripple deletes, preview
      card as always).
- [ ] **Vertical reframe** (subject-tracked crop keyframes for 9:16
      exports).
- [ ] **Agent memory**: per-project durable notes (naming conventions,
      user preferences) folded into the system prompt; user-visible and
      user-editable.
- [ ] **Session history UI**: browse/reopen past conversations per draft
      (on the Phase 0 persistence).

Exit: each superpower ships with an eval scenario and a demo script; the
gameplay and football prompts from the Vision section run end-to-end on a
stock machine with local models.

## Phase 5b — MCP + voice

- [ ] **MCP client** (external tools into our loop) per docs/mcp-design.md
      — that doc's open questions (approval UX under streaming, Windows
      stdio, secrets, server lifetime) must be resolved and its status line
      flipped before implementation starts; the fenced `mcp__` namespace
      slots into the Phase 0 dispatch rule unchanged.
- [x] **MCP server** (Cutlass tools for external agents): expose Workspace-
      tier reads and validated-edit submission over MCP so external agents
      script Cutlass with the same trust model — never a raw engine door
      (headless stdio v1: crates/cutlass-mcp — reads, validated edit batches,
      frame grabs, export; desktop-session attach and streamable HTTP still
      open).
- [ ] **Push-to-talk voice input**: hold-to-talk in the agent panel,
      transcribed locally by the 3b whisper stack into the prompt box —
      dictation, not a wake word.

Exit: an external MCP client lists and calls Cutlass tools under the tier
rules; a held key turns speech into a prompt with no cloud round-trip.

---

## Known gaps / open questions

- **Local-model vision quality.** Small local VLMs misread timelines and
  frames; the schematic map (drawn labels, not pixels-of-UI) is the
  mitigation, but the floor for "self-verify" on local-only setups is
  unproven. May need a "verify only when vision-capable" capability flag
  per provider.
- **Image token budgets and cost.** Screenshots + frame grabs + self-verify
  multiply per-prompt image counts fast. The per-request budget and
  referenced-not-resent history are structural; the right defaults
  (resolution, count, when to drop to schematic-only) need live tuning.
- **VLM latency on long footage.** Frame-batch analysis of an hour of
  gameplay is minutes-to-hours depending on provider; the propose→verify
  split and the persistent moments index amortize it, but first-run
  expectations need UI honesty (jobs with progress, resumable).
- **Python sandboxing story.** "Isolated venv" is not a security sandbox.
  What OS-level containment means per platform (macOS sandbox-exec is
  deprecated; Windows has no cheap equivalent) is unresolved — until then
  the System tier confirmation card *is* the sandbox, and the docs must say
  so plainly.
- **Whisper model sizes/licensing.** Multi-GB model downloads, per-model
  licenses, and non-English quality tiers need surfacing in the download
  manager UI, not buried in docs.
- **Screenshot fidelity vs. schematic renders.** Two renderers can drift
  from the truth differently: the schematic map must be generated from the
  same projection the UI renders, snapshot-tested, or the model will
  "verify" against a lie.
- **Confirmation fatigue vs. autonomy.** If `ask` fires too often, users
  flip to `full` and the tier model loses its teeth. Tier boundaries (what
  is genuinely Workspace-safe) deserve periodic re-audit as tools land —
  the mitigation is fewer, better-scoped System tools, not more cards.
