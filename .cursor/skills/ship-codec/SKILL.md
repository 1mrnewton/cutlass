---
name: ship-codec
description: Implement and ship native Rust audio, video, or image codecs in Cutlass with comprehensive, evidence-based validation. Use when asked to build, implement, finish, harden, or ship a codec decoder, encoder, or both, including AAC, Opus, FLAC, MP3, H.264, HEVC, AV1, VP8, VP9, ProRes, JPEG, PNG, and similar formats. Also use for codec completion and readiness work. Honor decoder-only or encoder-only scope; exclude networking codecs, text encodings, and unrelated serialization.
---

# Ship Codec

Carry a codec request through native implementation, integration, tests, hardening, measurements, and an honest readiness report. Treat “ship” as a request to finish the requested implementation and verification; it does not independently authorize a release, merge, publication, or messages to people.

## Load the right references

Read [implementation.md](references/implementation.md), [verification.md](references/verification.md), and [deliverables.md](references/deliverables.md) for every implementation task.

- For audio, also read [audio.md](references/audio.md).
- For video or still images, also read [video-image.md](references/video-image.md).
- For an explicitly limited task such as audit-only, honor that limit and read only the relevant sections.
- Resolve these paths relative to this skill directory, wherever the host installed it. Do not assume a global skill path or require network access to read the package.

## 1. Resolve the request and audit the checkout

1. Identify codec, direction (decode, encode, or both), domain, intended playback/export use, existing work, and any explicit profile/feature constraints.
2. Respect direction exactly. A decoder request does not authorize building an encoder; “both” requires two real implementations. Use independent references to test the absent direction. If direction is omitted, infer from the active task and repository; ask one concise question only when no sound inference exists and the choice changes the work materially.
3. Read all rule files under the repository's `.cursor/rules`, their scope/front matter and references, applicable `AGENTS.md`, manifests, dependency policy, toolchain, and relevant CI. Apply each rule to the paths it governs. Report unavailable instructions accurately, without inventing their contents.
4. Inspect the branch, commit, working changes, actual code, tests, capability registration, and engine paths. Preserve unrelated changes. Reuse existing implementations and shared contracts.
5. Distinguish functioning code from scaffolding. Do not invent APIs because an earlier prompt mentioned them.

For Cutlass, expect one crate per codec under `crates/codecs/<codec>`, shared codec/decode/encode boundaries, separate container crates, and playback/export consumers. Verify current names and dependencies. Extend the existing codec crate. Add only the minimal shared contracts a real implementation needs; preserve an acyclic dependency graph.

## 2. Establish a testable scope before coding

Create or update the capability matrix and implementation checklist described in [deliverables.md](references/deliverables.md). Make it small enough to maintain and precise enough to disprove an overclaim.

- Separate decoder acceptance from encoder emission.
- Enumerate profiles/levels, coding tools, rates or bit depths, layouts/chroma, framing/container adapters, metadata, state transitions, and explicit resource limits.
- Determine the required release target from the user's request, repository requirements, and real import/playback/export paths. If unspecified, state a practical target with justification and proceed. Do not silently choose an intra-only, lossless-only, mono-only, or own-encoder-only subset as general support.
- Distinguish required functionality, optional extensions, and unsupported configurations. A codec family name does not automatically require every historical/adjacent format. Once marked required, a feature remains required unless the user or applicable project requirements change it.
- Cite the normative specification edition/amendments and the sources for critical syntax/tables. Use primary documentation; verify unfamiliar or evolving details. Do not substitute another codec's behavior for the standard.
- Declare numerical comparison, rate/quality, performance, memory, and fuzz budgets before tuning. Use actual project budgets; otherwise state justified measurable defaults tied to named hardware and corpora.
- Break a large codec into milestones without lowering the final target. If an official reference is unavailable, record the exact validation gap and continue independent work.

Do not stop at a plan when implementation was requested. Complete necessary work autonomously; ask only for a genuine blocker or scope decision that cannot be inferred.

## 3. Preserve the native implementation boundary

Implement software algorithms in safe Rust. In Cutlass, preserve `#![forbid(unsafe_code)]`, inherited workspace lints, dependency bans, license policy, and toolchain constraints. Verify the active rules before adding utilities.

Do not replace the requested codec with another complete codec crate, C/C++ library, bundled foreign binary/Wasm, subprocess, OS codec call, or hidden fallback. Reuse approved lower-level Rust utilities where useful; audit new dependencies and transitive build tools.

Keep foreign implementations isolated as permitted test oracles/fixture generators. They must not enter the production dependency graph or become requirements for a default build. Obey rules that restrict test tools too.

Keep platform codecs separate from the software implementation. Use existing platform adapter crates; confine any permitted unsafe/FFI to reviewed adapters. Query capabilities. On Apple platforms investigate VideoToolbox for video and AudioToolbox/AudioConverter for AAC/audio. Never infer hardware acceleration merely from OS API use. Platform availability cannot satisfy native-software acceptance gates.

## 4. Implement and integrate

Follow [implementation.md](references/implementation.md) and the domain reference. Build a correct scalar implementation before optimization.

For every required matrix cell:

1. Implement actual syntax, algorithms, state, and typed failure behavior.
2. Add focused positive/negative tests and independent expected outputs.
3. Integrate packet/frame/configuration flow at existing engine boundaries.
4. Run the relevant checks and resolve failures.
5. Update the capability matrix with evidence.

Keep byte framing/demux/mux, codec algorithms, resampling/color conversion, rendering/mixing, and device I/O at their architectural owners. A shared parser must have one owner; do not introduce cycles or duplicate incompatible types.

Handle incomplete streaming input, malformed complete units, backpressure, drain, discard/reset, seek/preroll, configuration changes, cancellation, and bounded queues as distinct contracts. Do not silently conceal corruption or return success with empty/silent output for unsupported content.

Exercise actual playback/export consumers where implemented. If they are scaffolds, create executable headless integration through real available contracts and name the missing application path.

## 5. Prove the requested direction independently

Follow the full applicable matrix in [verification.md](references/verification.md).

- Decoder: independently produced streams -> this decoder -> normative/reference comparison.
- Encoder: source media -> this encoder -> independent decoders plus bitstream validation and quality/rate checks.
- Both: run both paths above; add internal round trips as supplementary coverage.
- Lossless: require exact reconstruction in the specified sample domain; compare metadata separately.
- Lossy: use the prescribed conformance tolerances and separately declared perceptual/rate gates. Do not demand encoder byte equality or declare quality from one signal/metric.
- Do not count two frontends using the same underlying codec as two independent implementations.

Run required workspace CI checks, conformance/interoperability suites, lifecycle and integration tests, fuzz campaigns, quality tests where applicable, and release benchmarks. A skipped suite, unavailable reference, compiled fuzz target, or unexecuted command is not a pass.

## 6. Optimize without breaking the contract

Profile representative and worst-case content in release builds. Keep correctness gates active. Verify optimized paths against the scalar/reference behavior under the applicable tolerance. Respect unsafe policy; do not add unsafe intrinsics just to meet a speed target.

Measure throughput, latency distribution, algorithmic/startup delay, allocations, peak/steady memory, worst-case work, and cancellation responsiveness on named hardware. Preserve bounded memory on long streams and useful operation on the supported fallback path.

## 7. Finish with evidence

Update capability reporting, usage/configuration documentation, regression corpus/manifest, reproducible validation commands, CI, and benchmark results. Use the report structure in [deliverables.md](references/deliverables.md).

Call the task complete only when every required capability and gate has evidence. Leave required failures visible; use “partial,” “unsupported,” “not run,” or “blocked” when accurate. State exact exclusions and their consequences for playback/export. Do not equate passing unit tests with general codec support or production readiness.

No TODO-only implementations, no fixture recognition, no default silence/blank frames, no hidden foreign fallback, no lowered thresholds after failures, no suppressed lints, and no replacing required features with permanent Unsupported errors.
