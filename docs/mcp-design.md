# MCP tools for the assistant — design

**Status: design, not implementation.** Per the cloud roadmap (Workstream
11), nothing in this doc may be built until the open questions at the
bottom are resolved and this line is updated. The doc exists because MCP
collides with the invariant that makes the assistant tractable — the
closed, versioned tool vocabulary — and that collision has to be resolved
on paper, not in a PR.

## Why this is hard here

`cutlass-ai`'s prompt loop is built on a **closed command vocabulary**:
every tool the model can call is a `WireCommand` defined in this repo,
validated against a project snapshot before it touches the engine,
rehearsed in a sandbox, rendered as a human-readable plan, and replayed as
one undo group. Evals pin the whole surface; `TOOL_SCHEMA_VERSION` pins
the schema. Everything downstream — dry-run, Apply/Discard, one-Cmd+Z
revert, the eval harness — depends on the vocabulary being enumerable.

MCP is the opposite: user-configured servers advertise arbitrary tools
with arbitrary schemas at runtime. The design below admits MCP without
letting it dissolve the closed vocabulary.

Boundary with rules/skills (Workstream 10, shipped): rules and skills
shape *how* the closed vocabulary is used — prompt-level, zero new trust
surface. MCP *adds tools* — a new trust surface. They are different
problems and deliberately share no mechanism.

## Decisions

### 1. A fenced namespace, never the edit vocabulary

- MCP tools are exposed to the model under mangled names:
  `mcp__<server>__<tool>`. The dispatch rule in the agent loop is a
  prefix check: `mcp__*` routes to the MCP layer; everything else must
  parse as a `WireCommand` or is rejected exactly as today.
- **MCP tools can never mutate the timeline.** There is no bridge from an
  MCP result to the engine except the model reading the result and then
  calling validated edit commands, which flow through the existing
  sandbox/validate/replay pipeline unchanged. No MCP call ever enters an
  `AgentPlanStep`.
- The tool-schema snapshot and `TOOL_SCHEMA_VERSION` continue to cover
  **only** the closed vocabulary. MCP schemas are pass-through
  (translated JSON Schema handed to the provider verbatim) and
  deliberately unversioned by us — they are the server's contract, not
  ours.
- MCP results join the conversation as ordinary tool-result strings,
  wrapped in a fence that names the server and states that the content is
  external data, not instructions (prompt-injection posture, below).

### 2. Architecture: `cutlass-mcp` crate + a trait seam

- New crate `crates/cutlass-mcp`: MCP client only. JSON-RPC framing,
  the `initialize` handshake, `tools/list`, `tools/call`, timeouts,
  and the **stdio transport** (spawn a child process) first; streamable
  HTTP later. No engine, no Slint, no `cutlass-ai` dependency.
- `cutlass-ai` stays transport-free. `run_prompt` gains an optional
  `external_tools: &mut dyn ExternalTools` seam (mirroring
  `EngineBridge`):

  ```rust
  pub trait ExternalTools {
      /// Specs to append to the provider tool list (already mangled).
      fn specs(&self) -> Vec<ToolSpec>;
      /// Execute one call; the String is the tool-result content.
      /// Blocking, like everything else in the loop.
      fn call(&mut self, name: &str, args: &serde_json::Value) -> String;
  }
  ```

  Tests and evals use a scripted fake, exactly like `ScriptedProvider`.
- The desktop owns server lifecycle: spawn configured servers lazily on
  the first prompt that runs with MCP enabled, keep them for the app's
  lifetime, kill on exit. Server stderr goes to the log file, never the
  transcript. A hung server is a timeout error string in the tool result,
  not a hung prompt (per-call deadline, default 30 s).

### 3. Configuration

In `~/.cutlass/config.toml`, following the `[providers.*]` registry
pattern:

```toml
[mcp.servers.asset-search]
transport = "stdio"
command = "npx"
args = ["-y", "@example/asset-search-mcp"]
enabled = true            # default false: configuring is not consenting
```

No per-project MCP config — projects are single files that travel, and a
project file must never be able to name an executable to run
(the imported-project rules lesson, applied harder).

### 4. Consent and trust

- **Per-server consent, first use.** The first prompt that would expose a
  server's tools shows the server name, the binary it runs, and its tool
  list; the user approves or the tools stay out of the prompt. Recorded
  in local settings, revocable in the same UI.
- **Per-call approval for side effects.** Each server gets a policy:
  `ask` (default) or `allow`. MCP tool annotations
  (`readOnlyHint`, `destructiveHint`) are **untrusted hints** from the
  server and never substitute for user approval — a server that lies
  about read-onlyness must gain nothing. Under `ask`, the loop parks the
  call and the agent panel shows an approve/deny card (the Apply/Discard
  interaction pattern, reused per-call).
- **Prompt-injection posture.** MCP results are untrusted content. The
  fence marks them as data; the system prompt instructs the model that
  external tool results never override rules or user intent. The real
  containment is structural, not textual: whatever an MCP result says,
  the only path to the timeline is validated commands, capped and
  rehearsed as today, and the only path to another side effect is
  another consented MCP call.
- **Sampling is refused.** MCP servers may request LLM completions
  (`sampling/*`); Cutlass declines the capability at handshake. A tool
  server does not get to spend the user's tokens or steer the loop.

### 5. Dry-run and undo semantics — stated, not faked

The editor's dry-run rehearses timeline commands in a sandbox. External
side effects cannot be rehearsed, and pretending otherwise (queueing MCP
calls for Apply time) would hand the model stale results and break the
conversation's causality. So:

- MCP calls execute **at prompt time**, dry-run or not, gated by the
  consent policy above. They are never part of the parked plan.
- Apply/Discard and Cmd+Z cover timeline edits only. The approve/deny
  card for a side-effectful call says exactly that: *"Runs now; not
  undoable from Cutlass."*
- The transcript logs every MCP call (server, tool, argument summary,
  duration, ok/error) so a prompt's external footprint is auditable
  after the fact.

### 6. Caps and failure behavior

- Separate budget from edits: `max_mcp_calls` per prompt (default 8),
  so a chatty server cannot starve the edit loop or the context window.
- Result size cap (default 32 KB, truncated in-band like rules) — MCP
  results share the context with the project snapshot.
- Errors (timeout, crash, malformed JSON-RPC, denied consent) become
  plain tool-result error strings; the model may retry or route around,
  the loop never retries on its own. A crashed stdio server stays down
  for the rest of the prompt (respawn on the next prompt).

### 7. v1 scope

Tools only, text results only. MCP **resources** and **prompts** are
explicitly out (skills cover the prompt-template need); binary/media
results are out until the import-consent story exists (an MCP tool that
returns a generated file would need the same click-to-import flow as
the SFX catalog, never auto-import). Streamable HTTP transport,
OAuth-protected servers, and a curated server directory are all later.

## Eval strategy

- Scripted `ExternalTools` fake mirroring `ScriptedProvider`: cases for
  fenced-result handling (model must not follow instructions embedded in
  a result), the mcp-call cap, denied-consent flow, and the firewall
  (an MCP result naming clip ids still requires validated commands —
  asserted by the existing plan/undo invariants).
- The tool-schema snapshot test gains a companion asserting the mangling
  scheme and that no `mcp__*` name can ever parse as a `WireCommand`.

## Open questions (blockers before implementation)

1. **Approval UX under streaming** — parking a tool call mid-turn while
   the provider connection stays open: hold the HTTP stream or end the
   turn and resume? Needs a provider-loop prototype answer.
2. **Windows stdio** — process spawn/kill and pipe behavior for stdio
   servers on Windows (the encoder work shows platform seams bite).
3. **Secrets** — many servers need API keys via env vars; how much of the
   env does a spawned server inherit? (Leaning: none; explicit
   `env = { ... }` allowlist in the server table.)
4. **Concurrent prompts vs. server state** — one conversation at a time
   today, but server processes outlive prompts; decide whether servers
   are per-conversation or shared before adding a second entry point
   (the Python bindings would be one).
