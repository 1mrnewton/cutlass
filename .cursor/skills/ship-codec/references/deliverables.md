# Scope, evidence, and completion records

Maintain these records next to the codec or under the project's existing documentation/test conventions. Reuse existing equivalents instead of adding duplicate files. Keep the records proportional; do not build a documentation framework.

## Request and scope record

Record:

- Codec, requested direction(s), domain, use case, checkout revision, and relevant rule paths.
- Normative specification editions/amendments, accessible sources, and missing references.
- Existing code versus scaffolds and minimal shared-boundary changes.
- Required target, justified assumptions/defaults, optional extensions, exclusions, and resource limits.
- Milestones, numeric quality/rate/performance/memory gates, conformance tolerances and sources, fuzz budgets, and release acceptance checks.

For a decoder-only request, mark encoder work outside scope. For an encoder-only request, use independent decoders for validation. For both, retain independent tests in both directions.

## Capability matrix template

| Capability/configuration | Required? | Decode behavior | Encode behavior | Positive/negative fixtures | Executable evidence | Status/limits |
| --- | --- | --- | --- | --- | --- | --- |
| Exact profile/tool combination | Yes/optional/no | Accept/error/outside scope | Emit/error/outside scope | Manifest IDs | Test/job/report paths | Planned/implemented/verified/unsupported/blocked |

Expand rows for codec-specific profiles/tools, rates/depths/chroma/layouts, framing/container bindings, metadata, state/lifecycle behavior, and limits. Give combinations their own rows when support is conditional.

“Implemented” means code exists. “Verified” requires checks actually executed and expected results met. A profile flag or rejected stream does not establish positive support. Do not downgrade a required row to optional just to finish.

## Fixture manifest template

Use the project's preferred machine-readable format. Each external/generated fixture should identify:

- Stable ID and path/acquisition method.
- SHA-256, original source URL or generator version and exact command.
- License and redistribution status; required local acquisition for restricted vectors.
- Producer and decoder backend identities, configuration, coding tools exercised, source media domain, and expected outcome.
- Expected samples/frames, channel/component order, timing, reference output, and comparison method/tolerance provenance.
- Purpose (positive, negative, regression, conformance, quality, stress) and deterministic seed where applicable.

Never check in restricted media merely because it is downloadable. Keep test oracles outside production dependencies. Fail empty test selection; distinguish unavailable corpora from tests that passed.

## Validation report template

1. **Implementation:** requested direction(s), changed modules, APIs, and actual engine/container integrations.
2. **Rules:** exact applicable instruction paths and any verified missing files or unresolved conflicts.
3. **Capabilities:** final matrix, exclusions, typed errors, known input/output limits, and readiness implications.
4. **Correctness:** conformance vectors/methods, independent producer/decoder identities, numerical comparison results, and every missing reference.
5. **Encoding quality:** lossless exactness or lossy quality/rate ladder results, matched reference settings, worst cases, threshold definitions, and pending human checks.
6. **Lifecycle/integration:** chunking, backpressure, EOF/drain, reset/seek/preroll, config changes, cancellation, timing/trim, playback/export checks, and absent app paths.
7. **Robustness:** property tests, malformed corpus, fuzz targets and actual campaign durations/corpus/limits/findings, minimized regressions, unresolved issues.
8. **Build/dependencies:** exact commands and exit statuses, feature/target execution, lints and production dependency/unsafe compliance, explicit missing-tool skips.
9. **Performance:** hardware/compiler/build/settings/corpus/threads, throughput, latency, delay, allocations, memory, cancellation, and pass/fail against budgets.
10. **Conclusion:** complete/partial/blocked, remaining required items, exact reason and effect, reproducible commands/report locations.

Separate “passed,” “failed,” “not run,” “not available,” “unsupported,” and “outside requested scope.” Include exit status and outcome; never fabricate a test result or hide failures under an aggregate green summary.

## Completion gate

Finish only when all required matrix rows and applicable acceptance checks pass with evidence. Otherwise deliver completed independent work, name every unmet gate, and call the result partial/blocked.

Passing unit tests alone does not establish conformance, useful encoder quality, secure input handling, complete stream lifecycle, application integration, or production readiness. A narrow decoder may be useful scaffolding; expose that scope in API capability reporting and documentation.

Keep a checkpoint for long work so a fresh agent can resume from completed changes, evidence, current failures, and next required milestone. Do not rerun already passing expensive checks without a relevant code/config change, prior failure, or unresolved concern.

Treat release/publish/merge actions according to the user's explicit authorization and project workflow. The word “ship” requests completed engineering work, not automatic permission for every external action.
