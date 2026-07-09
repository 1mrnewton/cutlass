# Cloud Roadmap — backend, asset catalog, and BYOK architecture

**Status (macos-dev, Jul 2026):** in progress. The editor is fully local by
default; `../cutlass-backend` serves stock/catalog/generation with JWKS-
verified auth, and `../cutlass-website` owns identity (better-auth) and
billing (Polar). This doc is the client-side architecture for everything
cloud-shaped: stock media, templates, text presets, SFX/LUT packs, AI
generation, accounts, and credits. The backend's own technical notes live in
`cutlass-backend/docs/ARCHITECTURE.md`; this doc owns the editor's seams and
the policies all sides share.

Policy: **Cutlass is free, and the cloud is optional.** The backend exists so
users don't have to juggle provider accounts — never to gate features. Every
principle below defends that.

## Governing principles (apply to every phase)

- **BYOK always works.** Every cloud capability is reachable with the user's
  own key(s), backend uninvolved. The managed path (Cutlass account +
  credits) is convenience only — one login instead of managing
  ElevenLabs / OpenAI / Google / fal / stock-site keys separately.
- **Credits only — no subscriptions.** Users buy prepaid credit packs via
  [Polar.sh](https://polar.sh) (one-time products with meter-credit
  benefits) and top up when they run out. No recurring billing, no expiring
  grants, no tiers. Credits meter **only model inference** on the managed
  path. Polar is configured credits-only (meter with no metered overage
  price), so an empty balance can never auto-charge — the app refuses the
  call and prompts a top-up. Polar is merchant of record: VAT/sales tax and
  chargebacks are theirs.
- **Free assets are free and anonymous.** Stock, SFX, LUTs, templates, and
  text presets cost no credits and need no account — anonymous,
  rate-limited, cacheable access.
- **The backend never touches projects/timelines/encoding.** It is an
  I/O gateway: JWT verification, credit metering, AI proxy, read-mostly
  asset catalog, stock search. Heavy work stays in the editor or at
  upstream providers.
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
- **Privacy is explicit.** BYOK keys never transit our servers. Managed-path
  prompts and generation jobs do — the docs and UI say so. No telemetry
  without opt-in. Account deletion and GDPR data export ride on
  better-auth's user-deletion hooks and must be wired before accounts
  launch, not after.

## The client seam: `crates/cutlass-cloud`

One crate owns all backend/provider HTTP for the editor, shaped like
`cutlass-ai`: engine-free, blocking HTTP on worker threads, trait-based so
tests use scripted fakes.

- **DTOs are the contract source of truth.** Request/response types for
  every backend route live here; `cutlass-backend` consumes this crate as a
  git dependency and its contract tests fail CI on drift. The editor repo
  stays self-contained (no path deps across sibling repos).
- **Routing rule**, applied per capability: BYOK key configured → call the
  provider directly; else signed in → managed path through the backend;
  else anonymous access only (assets/stock yes, inference no).
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

## Credentials and accounts

- **Identity and billing live on the website** (`cutlass-website`:
  better-auth + the Polar plugin), not in the Rust backend. The backend
  only verifies the website's EdDSA JWTs against its JWKS endpoint and
  meters credits (debits/refunds/balance); sign-in, sessions, checkout,
  the `order.paid` webhook, and credit grants are all website concerns.
  Both services share one Postgres.
- `~/.cutlass/config.toml` (single owner: `cutlass-settings`) grows a
  `[providers.<name>]` key registry (literal key or `_env` indirection,
  the `AiSettings` pattern) and an `[account]` table (`base_url` for the
  API, `auth_base_url` for the website — overrides only).
- **Desktop sign-in is the device-authorization flow** (RFC 8628, a
  first-class better-auth plugin): one "Sign in with browser" button, the
  app shows a short code and opens the website's approval page, then
  polls for the session token and exchanges it for a short-lived JWT.
  Provider choice (GitHub / Google — OAuth-only, no password storage, no
  transactional-email dependency) happens on the website.
- Tokens live in the **OS keychain** (`keyring` crate: macOS Keychain /
  Windows Credential Manager / Linux secret-service, degrading with a
  warning where absent) — never in config.toml, never in projects. The
  long-lived better-auth session token sits in the `refresh_token` seat
  of `StoredSession`; refresh means re-fetching a JWT from the website's
  `/api/auth/token`.
- Desktop surfaces: sign-in in settings/launch, balance display (from
  the shared credit ledger via the backend), "Buy credits" opening the
  website's `/account` page in the system browser (the device-flow
  approval already left a session there), and a visible "your key vs
  Cutlass credits" indicator wherever a paid call can originate.

## Asset kinds

| Kind | Source | Serving | Engine work needed |
|------|--------|---------|--------------------|
| Stock video/photo/music | Pexels, Pixabay (more later) | backend search → direct CDN download | none (existing import path) |
| Templates | first-party authored | catalog + bundle from CDN | bundle format + authoring UI (apply exists) |
| Text presets | first-party JSON | catalog from CDN | none (existing text + look animations) |
| SFX packs | curated CC0 | catalog + files from CDN | none (audio import) |
| LUTs | first-party (baked from grade recipes) | catalog + files from CDN | `.cube` 3D-LUT compositor pass |
| Lottie stickers | first-party / CC0 | catalog + files from CDN | decoder backend + file-backed asset model |
| Agent skills/rules packs | first-party, later community | catalog from CDN | none (plain files in `~/.cutlass/agent/`) |

## Content licensing policy: "CC0 in, CC0 out" (+ OFL for fonts)

Everything in the Cutlass asset catalog is first-party authored or ingested
from strictly CC0 sources, and published as CC0.

- **LUTs**: first-party — the starter pack is generated from the
  `cutlass-render` grade preset recipes (bake a `.cube` per recipe) plus
  hand-tuned additions; pinned by a drift test like the sticker catalog.
- **Text presets**: first-party JSON over our own `TextStyle` + animation
  catalogs. **Fonts are the exception to CC0**: fonts live in the OFL
  world. Presets/templates may reference **only fonts bundled with Cutlass
  (OFL-licensed)**; a missing-font reference falls back to the generic
  family visibly documented, never as a silent surprise (the renderer
  already falls back silently — presets must not depend on machine-local
  fonts). Any future "font packs" catalog kind is OFL-only.
- **SFX**: curated strictly CC0 (Freesound's CC0 subset, Kenney audio,
  etc.). Never Pexels/Pixabay audio — their licenses prohibit
  redistribution, and serving from our CDN *is* redistribution.
  Provider-licensed stock flows only through the direct-CDN stock path;
  the catalog carries only what we may host.
- **Templates**: structure is first-party. Bundled sample media must be
  self-produced, AI-generated (with provider terms permitting
  redistribution, recorded per asset), or CC0 — never Pexels/Pixabay
  clips. Music slots ship with a CC0 track or empty (the music slot is
  swappable; "pick a soundtrack" is part of the fill flow). Sample media
  stays short and ~720p for bundle size.
- **Provenance manifest**: every ingested asset records source URL,
  original author, license evidence, ingest date, and checksum in the
  catalog DB; a documented takedown path handles fraudulent-CC0 claims.
- **Community submissions (later)**: the catalog schema carries `author` +
  `license` fields from day one; submission terms grant CC0 (or an
  irrevocable redistribution license) when submissions open.
- Consequence for the app: **no attribution machinery for catalog
  assets**; attribution UI exists only in the stock browsing flow (Pexels/
  Pixabay attribution comes back in search responses as a goodwill
  gesture, not a legal apparatus).

## Legal checklist (before accounts open)

- Terms of Service + privacy policy published.
- Refund policy for credit packs stated at checkout (Polar as merchant of
  record handles tax and chargebacks; the policy text is ours).
- GDPR export + deletion endpoints in the auth spec.

## Scope statements

- **Desktop-first.** `cutlass-cloud` is engine-free Rust, so
  `cutlass-mobile` can expose it over FFI later — but mobile parity for
  stock/templates/AI generation is out of scope for this roadmap.
- **`cutlass-py` gets none of this.** It stays a local scripting wrapper.
- **Launch rail "Learn" tab**: out of scope. If it ever ships, it rides
  the asset catalog as a links/articles feed.
- **Community submissions** (templates, skills): schema fields reserved
  now, pipeline later.

## Workstreams

Ordered; each lands independently. Details per workstream live with the
code and in `cutlass-backend/docs/ARCHITECTURE.md`.

1. **Architecture docs** — this file plus the backend architecture update
   (Polar billing, stock search, asset catalog, auth decision).
2. **`cutlass-cloud`, anonymous half** — DTOs, stock/catalog client,
   `StockProvider` trait, download cache. No auth anywhere.
3. **Backend foundation & ops** — Postgres + migrations, config,
   rate-limit middleware, staging/prod deploy, observability, CI with
   DTO contract tests, Polar sandbox on staging. Prerequisite for any
   served feature.
4. **Stock media slice** — `/v1/stock/search` (metadata only) + Library
   Stock sections browsing → direct-CDN download → existing import path.
5. **Templates** — bundle format (a raw `.cutlasst` references sample
   media by local path and is not distributable), minimal authoring flow
   (slot-marking UI or `cutlass-py` script), backend catalog, launch-rail
   gallery, `ApplyTemplate` pick flow. Text presets ride along
   (bundled-OFL-fonts-only).
6. **Accounts & managed routing** — `[providers.*]` registry, keychain
   tokens, device-flow sign-in against the website's better-auth,
   balance display, "Buy credits" → website account page, update-check
   nudge. (Website side: better-auth + Polar plugin, shared Postgres.)
7. **AI generation surfaces** — Library AI sections (prompt → job → poll
   → download → import), TTS/voiceover; third provider mode in
   `cutlass-ai` ("Cutlass account") with out-of-credits handling;
   per-user rate limits **and** a user-configurable per-day spend cap on
   the managed path.
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
