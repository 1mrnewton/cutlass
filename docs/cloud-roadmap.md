# Cloud Roadmap — backend, asset catalog, and BYOK architecture

**Status (macos-dev, Jul 2026):** desktop v1 ships **anonymous cloud + BYOK
only**. Cutlass account / device sign-in / managed credits are **not in the
desktop client** for this release (backend/website may still exist out of
tree). This doc is the client-side architecture for everything cloud-shaped
that remains: stock media, templates, text presets, SFX/LUT packs, fal BYOK
generation, and the update check. The backend's own technical notes live in
`cutlass-backend/docs/ARCHITECTURE.md`.

Policy: **Cutlass is free, and the cloud is optional.** The backend exists so
anonymous catalogs and stock search work without users juggling provider
accounts for those surfaces — never to gate the editor.

## Governing principles (apply to every phase)

- **BYOK always works.** Inference and paid generation use the user's own
  key(s); the backend is uninvolved. There is no managed / credits path in
  the desktop client for v1.
- **Free assets are free and anonymous.** Stock, SFX, LUTs, templates, and
  text presets cost no credits and need no account — anonymous,
  rate-limited, cacheable access.
- **The backend never touches projects/timelines/encoding.** It is an
  I/O gateway: read-mostly asset catalog, stock search, latest-version.
  Heavy work stays in the editor or at upstream providers.
- **No media bytes through the backend.** Stock *search* goes through the
  backend (the provider API keys must stay server-side — an embedded key in
  an open-source binary is public, and rate limits are per key). The actual
  media files download **directly from the provider CDNs** (keyless URLs),
  so stock bandwidth never hits our egress. Cutlass-owned assets (template
  bundles, SFX, LUTs, Lottie files) serve from object storage/CDN, not the
  Axum process.
- **The editor never blocks on the backend.** Fully offline editing is
  normal. Catalog fetches are background work (ETag-cached in the data
  dir); cloud Library sections degrade to their placeholders when
  unreachable. Network stays off the UI thread (the AI-agent invariant).
- **Old clients keep working.** Shipped desktop builds live in the wild for
  months and cannot be force-updated. API evolution is additive-only within
  `/v1` (new optional fields, new endpoints); breaking changes mean `/v2`
  with `/v1` kept alive on a stated deprecation window. The shared DTO
  crate (`cutlass-cloud`) encodes this: unknown-field-tolerant serde
  everywhere. The app gets a lightweight update-check nudge
  (`/v1/app/latest-version` behind a non-blocking launch-screen chip); a
  real auto-updater is explicitly later.
- **Privacy is explicit.** BYOK keys never transit our servers. No telemetry
  without opt-in.

## The client seam: `crates/cutlass-cloud`

One crate owns all backend/provider HTTP for the editor, shaped like
`cutlass-ai`: engine-free, blocking HTTP on worker threads, trait-based so
tests use scripted fakes.

- **DTOs are the contract source of truth.** Request/response types for
  every backend route live here; `cutlass-backend` consumes this crate as a
  git dependency and its contract tests fail CI on drift. The editor repo
  stays self-contained (no path deps across sibling repos).
- **Routing rule**, applied per capability: BYOK key configured → call the
  provider directly; else anonymous access only (assets/stock yes,
  inference no).
- **`StockProvider` trait** with two impls: backend-routed (anonymous) and
  direct-Pexels/Pixabay (user-supplied stock keys — then even search skips
  the backend).
- **Downloads** (stock files, template bundles, packs) follow the
  `proxy.rs` worker pattern: progress callbacks, cancellation, atomic
  tmp-then-rename writes.
- **Cache management:** downloads land in a quota-managed cache dir (LRU
  eviction above a configurable cap; files imported into a project are
  exempt — they're pool media). Settings gets a "clear download cache"
  action.
- **Account / auth modules are out of the client** for v1 (no keychain
  session, no device flow, no managed generation provider).

## Credentials and config

- `~/.cutlass/config.toml` (single owner: `cutlass-settings`) holds a
  `[providers.<name>]` key registry (literal key or `_env` indirection,
  the `AiSettings` pattern) and a `[cloud]` table (`base_url` for the API
  host — override only). Legacy `[account].base_url` still loads into
  `[cloud]`.
- The assistant uses the `[ai]` endpoint / model / key fields only
  (OpenAI-compatible or local). Library AI media generation uses
  `[providers.fal]`.
- Desktop surfaces: Settings AI provider fields, fal route label in the
  Library, and the launch-screen update chip (`UpdateBackend`).

## Asset kinds

| Kind | Source | Serving | Engine work needed |
|------|--------|---------|--------------------|
| Stock video/photo/music | Pexels, Pixabay (more later) | backend search → direct CDN download | none (existing import path) |
| Templates | first-party catalog | CDN bundle | apply-template flow |
| Text presets | first-party catalog | CDN JSON | existing text + look-animation pipeline |
| SFX | first-party catalog | CDN audio | none (import path) |
| LUTs | first-party catalog | CDN `.cube` | compositor LUT pass |
| Lottie | first-party catalog | CDN | Lottie decoder |
| AI image/video/TTS | fal.ai (BYOK) | direct fal queue → CDN result | none (import path) |

## Scope statements

- **Desktop-first.** `cutlass-cloud` is engine-free Rust, so
  `cutlass-mobile` can expose it over FFI later — but mobile parity for
  stock/templates/AI generation is out of scope for this roadmap.
- **`cutlass-py` gets none of this.** It stays a local scripting wrapper.
- **Launch rail "Learn" tab**: out of scope. If it ever ships, it rides
  the asset catalog as a links/articles feed.
- **Community submissions** (templates, skills): schema fields reserved
  now, pipeline later.
- **Managed accounts / credits**: deferred; not part of desktop v1.

## Workstreams

Ordered; each lands independently. Details per workstream live with the
code and in `cutlass-backend/docs/ARCHITECTURE.md`.

1. **Architecture docs** — this file plus the backend architecture update
   (stock search, asset catalog).
2. **`cutlass-cloud`, anonymous half** — DTOs, stock/catalog client,
   `StockProvider` trait, download cache. No auth anywhere.
3. **Backend foundation & ops** — Postgres + migrations, config,
   rate-limit middleware, staging/prod deploy, observability, CI with
   DTO contract tests.
4. **Stock media slice** — `/v1/stock/search` (metadata only) + Library
   Stock sections browsing → direct-CDN download → existing import path.
5. **Templates** — bundle format (a raw `.cutlasst` references sample
   media by local path and is not distributable), minimal authoring flow
   (slot-marking UI or `cutlass-py` script), backend catalog, launch-rail
   gallery, `ApplyTemplate` pick flow. Text presets ride along
   (bundled-OFL-fonts-only).
6. **Update check** — anonymous `latest-version` + launch-screen chip
   (`UpdateBackend` / `src/updates.rs`).
7. **AI generation surfaces** — Library AI sections (prompt → fal job →
   poll → download → import), TTS/voiceover; assistant via `[ai]` BYOK /
   local endpoint only.
8. **Lottie** — decoder backend (dotlottie-rs vs velato vs rlottie),
   capped-fps/on-demand frame strategy (never pre-render-all like
   stickers), file-backed animated asset model.
9. **SFX + LUT packs** — catalog browsing/import; LUT browsing gates on
   the `.cube` compositor pass landing (no phantom features).
10. **Agent rules & skills** — `~/.cutlass/agent/` (rules, skills, slash
    commands), read-only `read_skill` tool via the vocabulary growth
    checklist, project rules in `ProjectMetadata` (shown before first use
    on imported projects), bundled first-party skills; skills packs join
    the asset catalog later. Prompt-level only — the closed command
    vocabulary is untouched.
11. **MCP tools for the assistant** — design doc first
    ([docs/mcp-design.md](mcp-design.md)); no implementation until it
    exists. Rules/skills shape *how* the closed vocabulary is used; MCP
    *adds tools* (a new trust surface) — different problems, never one
    mechanism.
