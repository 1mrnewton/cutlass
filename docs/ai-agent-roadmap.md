# AI Agent Roadmap — prompt-to-edit foundation (v1 M3)

Policy: **the agent is the reason Cutlass exists.** CapCut has no real
equivalent of a prompt box that edits the timeline; this is the
differentiator, and it ships early because it depends only on the command
layer — which has existed, undoable and integration-tested, since the
headless days. Every later milestone (keyframes, effects, color, audio)
grows the agent's vocabulary for free; nothing here waits on them.

This doc tracks v1-roadmap **M3 — "The AI agent"**. The dependency is M0's
command layer plus one M0 honesty item pulled forward as a hard
prerequisite: the agent must never emit the phantom generator kinds
(`Sticker`/`Effect`/`Filter`/`Adjustment`) that `composite.rs` silently
drops — an agent that "adds a sticker" which renders nothing is the
phantom-feature problem with a megaphone.

## Architecture invariants (apply to every phase)

- **The agent is just another command source.** Agent output is validated
  `cutlass_commands` commands applied on the worker thread through the
  same `Engine::apply` as every gesture; the UI learns the result from
  the republished projection. The agent gets correctness, undo, and
  validation for free — that is the entire point of the command-layer
  design, and the agent never gets a side channel into project state.
- **One prompt = one history group.** The loop wraps each prompt in
  `begin_group`/`end_group` (the compound-gesture pattern all over
  `preview_worker.rs`); an aborted prompt is `rollback_group`ed. An AI
  edit is exactly as undoable as a drag — one Cmd+Z per prompt.
- **AI proposes, the engine disposes.** Validation rejects, it never
  guesses: unknown commands, dead ids, phantom generators, and
  out-of-vocabulary requests fail loudly back to the model (which may
  retry) and visibly to the user. No silent fixups.
- **Provider-abstracted, local-first, never local-only.** `cutlass-ai`
  defines the `ChatProvider` trait; the first implementation is generic
  OpenAI-compatible HTTP, which covers Ollama / llama.cpp-server /
  LM Studio locally *and* OpenAI-class clouds day one. Adding providers
  is config or one adapter, never an agent change.
- **Keys and provider config live in `~/.cutlass/config.toml`** (the
  config-dir convention the settings crate established) — never in
  project files, never serialized into `.cutlass`.
- **Network and inference stay off the UI thread.** The agent runs on
  its own thread, talks to the engine via the worker's ordered mutation
  lane, and streams status to the UI through the established
  handle/callback pattern.
- **The vocabulary is closed and versioned.** The tool schema is
  generated from one wire layer, snapshot-tested, and versioned with the
  crate. New engine commands join the vocabulary by a checklist (Phase 5),
  not by accident.

## Status legend

- [x] shipped
- [ ] not started / in progress

---

## Phase 0 — Foundation (done — what this builds on)

All engine-side, all tested:

- [x] Closed command vocabulary: 15 `EditCommand`s + 5 `ProjectCommand`s
      (`crates/cutlass-commands/src/command.rs`), every edit returning an
      inverse action through `dispatch` (`action/dispatch.rs`), with
      `EditOutcome` telling callers what happened.
- [x] History groups with rollback: `Engine::{begin_group, end_group,
      rollback_group}` (`engine.rs`), used by every compound gesture in
      `preview_worker.rs` — the exact transaction shape a prompt needs.
- [x] Worker thread owns the engine; every mutation republishes the
      projection (`publish_projection`) — agent edits will repaint the
      timeline, inspector, and preview with zero new plumbing.
- [x] Models are fully serde (`cutlass-models`, project persistence) —
      ids serialize, `RationalTime`/`TimeRange` serialize, and the UI
      already round-trips raw u64 ids (`parse_raw_id`).
- [x] Config-dir convention: `~/.cutlass/config.toml` for settings; the
      per-user data dir's `projects/` for app-owned project drafts.

What does **not** exist (the work): `cutlass-commands` has no serde and
no JSON schema; there is no provider code, no agent loop, no chat UI, no
eval harness. The product's stated identity is 0% built — this roadmap
is that 0% → shipped.

## Phase 1 — `cutlass-ai`: wire format, validation, tool schema ✅

The smallest slice that makes the command layer machine-callable, with
no network anywhere yet: JSON in → validated `Command`s out → applied to
a real engine in tests. Everything later stands on this.

- [x] **Crate scaffold**: `crates/cutlass-ai`, workspace member, deps
      `cutlass-commands`/`cutlass-models` + `serde`/`serde_json`/
      `schemars`. Deliberately **no** `cutlass-engine` dependency: the
      agent emits commands and reads summaries through a narrow bridge
      trait (Phase 3), so the crate stays light and the eval harness
      stays honest about the boundary.
- [x] **Agent wire format**: dedicated serde DTOs (`wire.rs`) mirroring
      `EditCommand`, *not* serde derives on `cutlass-commands` itself.
      Deviation from the v1-roadmap sketch, deliberately: the wire layer
      is shaped for LLM ergonomics — `#[serde(tag = "command")]` flat
      objects, **times as fractional seconds** (models reason in
      seconds; converted to frame-snapped `RationalTime` at the project
      timeline rate during validation), **ids as plain integers** (the
      raw u64s the UI already uses), enums as lowercase strings. Keeps
      internal refactors of `cutlass-commands` from silently changing
      the prompt-visible schema.
- [x] **Validation + lowering** (`validate.rs`): wire command →
      `cutlass_commands::Command` against a project snapshot — id
      existence, range sanity, track-kind rules, second→tick conversion.
      **Whitelist enforced here**: edit commands only (no
      `ProjectCommand` in the vocabulary at all in M3 — open/save/
      export/import stay human), and `AddGenerated`/`SetGenerator`
      accept only `Text`/`SolidColor`/`Shape`. Every rejection carries a
      model-readable reason string ("clip 12 does not exist; current
      clips are …").
- [x] **Tool schema export**: JSON Schema per wire command via
      `schemars`, one tool per command (flat single-purpose tools
      tool-call better than one mega-tool, and 17 is well inside every
      provider's comfort zone), plus `describe_project`. Versioned
      constant (`TOOL_SCHEMA_VERSION`) + snapshot test
      (`tests/snapshots/tools.json`, re-bless with `BLESS_TOOL_SCHEMA=1`)
      so schema drift is a reviewed diff, never an accident.
- [x] **`describe_project()`** (`describe.rs`): compact, token-bounded
      timeline summary from `&Project` — project rate, tracks in
      stack order (id, kind, name, flags), clips per track (id, media or
      generator, timeline start/duration in seconds *and* exact frames,
      source range, link groups), media pool (id, file name, duration,
      dimensions, has-audio). Deterministic output order (stack order /
      start order / id order) so eval tests can assert on it verbatim.
      `EditorContext` (selection, playhead, in/out) lives here too,
      ready for Phase 3's prompt assembly. Phantom generator clips
      surface as `content: other` — visible, not editable.
- [x] **Round-trip tests**: every wire command deserializes from
      realistic JSON, validates against a fixture project, lowers to the
      expected `EditCommand`, and applies through a real `Engine`
      (dev-dependency, `tests/engine_roundtrip.rs` — fake media pool
      entries, no decode) — plus rejection tests for the guardrails
      (dead id, bad range, lane mismatches, sub-frame deltas; phantom
      generators and project commands are unrepresentable by
      construction). A 10-command prompt-sized scenario applies and
      fully unwinds with 10 undos.

Exit: `serde_json::from_str` → `validate` → `Engine::apply` edits a
fixture project correctly in CI, and the generated tool schema is a
checked-in, reviewed artifact. ✔ shipped.

## Phase 2 — Providers ✅

Decision: **no agent framework** (evaluated AutoAgents, the strongest
Rust candidate). The loop is the product here — one-group-per-prompt
transactions, per-call rejection feedback, dry-run, rollback — and
frameworks own the loop; they'd also drag a tokio runtime into a
std-thread app and take ownership of the prompt-visible schema. What a
framework actually replaces (provider HTTP + SSE + the turn loop) is a
few hundred well-tested lines. The `ChatProvider` trait is the seam: a
framework's provider layer can become *one implementation* later (M9)
if the provider matrix demands it, without touching the agent.

- [x] **`ChatProvider` trait** (`provider.rs`): chat completion over a
      message history with tool definitions, streamed text (an
      `on_text` delta callback; tool calls arrive whole in the returned
      `ChatTurn` with its finish reason), and cooperative cancellation
      via an `AtomicBool`. Blocking trait on a dedicated agent
      thread — no tokio runtime enters the app for v1 (matches the
      std-thread architecture everywhere else).
- [x] **OpenAI-compatible provider** (`providers/openai_compat.rs`):
      `POST {base_url}/chat/completions` with `tools` + SSE streaming
      over `ureq`. Tool-call argument fragments accumulate per index
      across chunks (parallel calls supported). One implementation,
      many backends: Ollama (`localhost:11434/v1`), llama.cpp-server,
      LM Studio, OpenAI, OpenRouter-style gateways — "cloud providers
      later" becomes config, not code, exactly as the v1 roadmap wants.
- [x] **Config**: `~/.cutlass/config.toml` — `[ai]` table with
      `base_url`, `model`, `api_key` (or `api_key_env` to read an
      environment variable instead of storing the key). Parsed in
      `cutlass-ai` (`config.rs`), absent file = agent not configured (a
      state the UI surfaces in Phase 4, never a crash).
- [x] **`ScriptedProvider` test double**: deterministic canned
      tool-call sequences, no network, records every request for
      assertions — the substrate for every loop test and the Phase 3
      eval harness.
- [x] **Error taxonomy**: `ProviderError::{NotConfigured, Network,
      Provider{status}, Protocol, Cancelled}` carried distinctly, so
      the UI can say "Ollama isn't running at localhost:11434" instead
      of "something failed".

Exit: a unit test streams a tool call out of a recorded SSE fixture, and
a hand-run binary gets a real completion from a local Ollama. ✔ shipped —
`examples/chat_probe.rs` against local Ollama (gemma4) streams text and
emits a real assembled `trim_clip` tool call.

## Phase 3 — Agent loop, guardrails, eval harness ✅

The brain: prompt in, validated-and-applied command group out, every
step observable.

- [x] **`EngineBridge` trait** (`agent.rs`): the loop's whole world —
      `summary() -> ProjectSummary`, `apply(&WireCommand) ->
      Result<EditOutcome, String>` (validate + dispatch host-side,
      model-readable errors), `check` for dry-run, and the group
      markers (`begin/end/rollback`). The UI implements it over a
      sandbox engine whose plan replays onto the live one (Phase 4);
      tests implement it over a plain `Engine`. The loop cannot name a
      file path, a socket, or a Slint type.
- [x] **The loop** (`agent::run_prompt`): system prompt (vocabulary +
      house rules + current `describe_project`) → provider turn → for
      each tool call: validate → apply → feed the outcome (or the
      rejection reason) back as the tool result → repeat until the
      model finishes or a cap trips. Failed commands don't abort the
      prompt — the model sees the error and may correct course (that's
      the point of per-call feedback); the group rolls back only when
      the prompt aborts (cancellation, provider error, cap exceeded).
      The house rules teach trim-head semantics explicitly (live gemma4
      got it wrong without the rule, right with it).
- [x] **Prompt context assembly — pushed, not retrieved**: every prompt
      carries fresh state — the `describe_project` summary plus an
      **`EditorContext`** snapshot (selected clip ids, playhead position,
      in/out range) captured when the user hits send. "Trim the selected
      clip", "split at the playhead", "delete everything before the in
      point" resolve against this block; tool results carry updated
      state back so later calls in the same prompt see the world they
      changed. Stale references (selection changed mid-prompt, undone
      clips) fail validation loudly instead of editing the wrong thing.
- [x] **Guardrails**: max tool calls per prompt (default 32) and max
      provider turns (default 16); unknown tool names rejected with the
      valid list (defense in depth with Phase 1's closed vocabulary);
      **dry-run mode** — validate and collect the plan without
      applying, for the Phase 4 preview card.
- [x] **Action log**: every applied command renders a human-readable
      line from the wire command + `EditOutcome` ("split clip 7 at
      12.40s (new clip 21)", "added text 'INTRO' at 0.00s for 3.00s on
      track 3") — the transcript entry, the undo tooltip, and the
      eval-test assertion format, all one renderer
      (`agent::describe_action`).
- [x] **Eval harness** (`tests/agent_eval.rs`): scripted-provider
      prompts against fixture projects asserting on the final timeline
      and the action log — "cut the first 3 seconds", "add a title that
      says INTRO" (a model-simulator provider that reads the new track
      id out of the tool result), "delete every clip on the music
      track", a multi-step correction case (first tool call rejected,
      model retries), a cap-trip rollback case, dry-run, Q&A-without-
      editing, and provider-failure rollback. Runs in CI with zero
      network; this is how agent regressions get caught without a live
      model.

Exit: `cargo test -p cutlass-ai` proves prompt → correct timeline → one
undo entry, including the failure paths, against the stub provider.
✔ shipped — and verified live end-to-end: `examples/agent_probe.rs`
ran "cut the first 3 seconds of the selected clip" through local
Ollama against a real engine and produced the frame-exact trim.

## Phase 4 — Chat panel in `cutlass-ui` ✅

- [x] **Worker integration** (`src/agent.rs`) — *landed as
      sandbox-rehearse-then-replay rather than per-call worker
      round-trips*: the engine's history groups don't nest, so holding a
      group open on the live engine across network waits would swallow
      any user edit made while the model thinks. Instead the agent
      worker thread rehearses the prompt against a throwaway sandbox
      `Engine` seeded with a project snapshot (one `SnapshotProject`
      round-trip on the ordered mutation lane) — tool calls really
      apply, so the model sees created clip/track ids and the world it
      changed — and on completion the validated plan replays atomically
      on the live engine (`WorkerMsg::AgentApplyPlan`): one history
      group, re-validated step by step against the live project,
      sandbox-allocated ids remapped onto live ones
      (`WireCommand::remap_ids`), `publish_projection` after every step
      so the user watches the plan land. Any replay failure (the
      project changed mid-prompt) rolls the whole group back and says
      so. `EditorContext` snapshots at send time in `main.rs`
      (selection + playhead + in/out from `TimelineStore`, ticks →
      seconds at the sequence rate). Session-epoch rules hold: the
      `app.slint` epoch watcher fires `AgentStore.session-changed`,
      which cancels the running prompt and discards any parked plan.
      Covered by `agent::tests` (replay with id remapping + single
      undo; stale-plan rollback) — ids are process-global atomics, so
      the live engine always allocates different ids than the sandbox
      and the tests genuinely exercise the remap.
- [x] **Chat panel** (`ui/panels/agent/agent.slint`): docked at the far
      right behind a title-bar "Assistant" toggle, resizable via the
      house split grip. Prompt box (Enter or Send), transcript with
      streamed assistant text, kind-styled entries (user bubble /
      assistant text / accent-ticked action lines / status / applied /
      error), "Thinking…" while running, auto-pinned scroll. Slint
      stays a dumb renderer of `AgentStore.transcript`; the agent
      worker mutates the model from Rust via the event loop.
- [x] **Action list per prompt**: Phase 3 action-log lines stream in as
      entries; the "Applied N edits as one undo step" entry carries a
      one-click **Undo** chip (plain history undo — one group per
      prompt makes it free), offered while it's the newest entry.
- [x] **Dry-run preview card**: panel toggle ("Preview before
      applying"), default **on**. The rehearsed plan parks in the agent
      worker and the panel shows the planned-edit list with Apply /
      Discard; Apply replays the validated plan in one group. A new
      prompt or a session change discards a stale plan.
- [x] **Not-configured state**: no `[ai]` table → the panel renders
      setup instructions (config path + a config.toml example +
      provider notes) instead of a dead prompt box; config re-reads on
      every send, so fixing it needs no restart.
- [x] **Error surfaces**: config errors, provider failures, cap trips,
      and replay failures land in the transcript as error entries that
      say what (didn't) happen — aborts always report "nothing was
      applied".
- [x] **Cancellation**: Stop button sets the shared cancel flag; the
      provider aborts between stream chunks, the sandbox group rolls
      back, and the transcript records the stop. Nothing ever reached
      the live engine.

Exit: "cut the first 3 seconds and add a title that says INTRO" works
end-to-end against local Ollama — watched live, listed as actions,
undone with one click. The M3 exit criterion, shipped.

## Phase 5 — Read-only Q&A + vocabulary growth policy ✅ (stretch deferred)

- [x] **Q&A without mutation**: "how long is the timeline?", "which
      clips have no audio?" — the model answers from `describe_project`
      and finishes without tool calls. Shipped as prompt-tuning plus
      eval coverage: the system prompt now teaches that the pushed
      state is a fresh send-time snapshot — answer questions directly
      from it (no redundant `describe_project` round-trip, which
      matters on slow local models), name clips/tracks by id so
      answers are checkable, and admit when the state can't answer
      rather than guess. Eval cases lock it in: "which clips have no
      audio?" answered in **one turn** from the pushed summary
      (`has_audio` reaches the prompt), and an answer-only turn under
      dry-run yields zero actions — so the preview card and the
      "Applied N edits" line never render for a question (the Phase 4
      empty-plan guard, now contract-tested).
- [x] **Vocabulary growth checklist**, documented in `cutlass-ai`:
      every new `EditCommand` lands with (1) wire DTO + validation,
      (2) schema snapshot update, (3) action-log line, (4) one eval
      case. M2's keyframe commands and M4's effect commands join this
      way — the "grows for free" promise made enforceable. Lives as
      the "Growing the vocabulary" section of the crate docs
      (`cutlass-ai/src/lib.rs`), next to the code it governs.
      *First exercise of the checklist: M2's `set_param_keyframe` /
      `remove_param_keyframe` / `set_param_constant` joined (tool
      schema v2) — "fade the clip in over the first second" is an
      eval case.*
- [ ] **Guarded project commands (stretch)**: `Import` behind an
      explicit per-prompt user confirmation card — the first crack in
      the edit-only wall, deliberately after the trust model is proven.
      `Open`/`Save`/`Export` stay human-only through v1. *Deferred
      past M3 on purpose: let the alpha prove the trust model first.*

---

## Known gaps / open questions

- **Local model tool-calling quality is the headline risk.** Small
  local models may emit malformed or boneheaded tool calls. Mitigations
  are structural: a closed 15-command schema (far easier than open
  codegen), per-call rejection feedback, dry-run default-on in the
  first alpha, caps + rollback. The OpenAI-compatible provider means
  anyone with a key gets frontier-quality calls day one.
- **Seconds rounding**: the wire format's fractional seconds snap to
  frame ticks at validation; a model asking for `1.0001s` gets the
  nearest frame. Fine for editing semantics; revisit only if eval
  cases surface drift.
- **`describe_project` token growth**: a 500-clip timeline blows the
  summary budget. v1 ships id-sorted truncation with counts ("track 2:
  41 more clips…") and a follow-up tool query if the model asks;
  smarter windowing (around the playhead / named ranges) is post-M3.
- **No conversation persistence**: the transcript dies with the
  session. Acceptable for M3; revisit alongside project metadata
  (`ProjectMetadata` already exists for notes) if users want history.
- **Undo of *parts* of a prompt** isn't a thing — one group is
  all-or-nothing by design. If users want surgical reverts, that's a
  history-UI feature, not an agent feature.
- **Selection staleness after undo** (the tracked M0/timeline debt)
  gets more visible when an agent edits under the user's selection;
  the M0 fix should land before or with Phase 4.
- **Tick model**: Slint's `i32` ticks vs the engine's `i64` is tracked
  for M2; the agent operates engine-side in `i64` and is unaffected,
  but `describe_project` inherits whatever the resolution audit
  decides.
