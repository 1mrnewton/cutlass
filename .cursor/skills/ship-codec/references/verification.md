# Codec verification and acceptance

## Contents

- Coverage and independent evidence
- Conformance and interoperability
- Quality, rate, and lossless correctness
- State, streaming, and integration
- Hostile input, properties, and fuzzing
- Builds, performance, and CI

## Coverage and independent evidence

Build a traceable matrix: required capability -> positive input -> negative/boundary case -> expected property/reference -> executable check -> result. Include tools and combinations, not only profile labels. Favor fixtures known to exercise a feature; a file tagged with a profile need not exercise every tool.

Pin corpus provenance, hashes, seeds, licensing, generation commands, exact encoder/decoder versions, backend identities, expected metadata, and tolerance sources. Separate distributable small regressions from external large/restricted corpora. Keep ordinary tests deterministic and offline.

Do not generate all expected results from the implementation under test. Use analytic invariants, a deliberately independent mathematical model, official reference outputs, and independently produced media.

## Conformance and interoperability

Use official conformance suites/methods for the exact required profiles/tools where available. Record standard editions and prescribed precision/tolerance. Do not call an inaccessible suite passed or a differential test an official conformance result.

For decoding, process independent streams with this decoder and compare output, metadata, frame/sample counts, ordering, timing, and prescribed reconstruction. Cover multiple producer implementations, hardware-produced content where available, and valid edge configurations.

For encoding, validate emitted syntax/semantics and decode with independent implementations. Prefer two independently implemented decoders where feasible; document when only one exists or is accessible. Two programs sharing libavcodec, FDK, or another backend are one implementation.

For both directions, run both independent paths and supplementary self round trips. Never accept a shared bug because internal encode/decode agrees.

Compare within the intended reconstruction stage. Separate required codec output from optional film grain, concealment, resampling, color conversion, dithering, and display/device effects. Do not erase discrepancies through undocumented normalization.

Compare exact pixels/samples where the specification is exact. Use prescribed tolerances for permitted implementation variation. Handle stochastic tools using their defined tests/metrics; do not exempt unrelated errors.

Track unsupported and malformed fixtures separately from valid conformance input. A decoder that rejects all difficult input has not passed the positive suite.

## Quality, rate, and lossless correctness

For lossy encoders, define measurable quality and rate gates before tuning. Match actual rates, source domain, latency/settings, and reference encoder conditions. Report per-item distributions and worst cases, not only an average.

Use appropriate objective perceptual metrics and aligned, level-controlled/blinded human comparison where needed. State metric versions, preprocessing, limitations, and pending human evaluation. Do not invent human results or infer perceptual transparency from PSNR/SNR alone.

Use representative, difficult, and synthetic corpus items. Check bitrate accuracy, peaks, buffer/reservoir limits, latency, and repeatability across quality/rate ladders and all promised settings. A syntactically valid but unusably poor encoder fails a declared quality gate.

For lossless claims, require exact reconstruction of the specified original domain, covering extrema, unusual dimensions/lengths, channel/component patterns, alpha, high precision, and randomized signals. Distinguish a codec being lossless from a pipeline containing lossy input conversion. Compare metadata independently.

## State, streaming, and integration

Exercise arbitrary chunking, one-byte chunks, many units per input, zero output and delayed output, backpressure, empty input, minimum sizes, partial tails, long streams, repeated drain/reset, invalid API order, cancellation, and reuse after failures.

Assert chunk-invariant output/timing under equivalent conditions. Account for every consumed input and valid output exactly once. Prevent duplicate/dropped tail data and stale state after seek/source change.

Test actual engine paths: import/playback, seek/preroll, trim/loop, timeline boundaries, and export/reopen for the requested directions. Verify rational timing, audio/video sync, configuration changes, and missing/corrupt packets. If app-level APIs are scaffolds, run headless tests through real available contracts and report that limit.

## Hostile input, properties, and fuzzing

Test truncation at syntax boundaries; invalid/reserved fields; overlarge counts/lengths/dimensions; broken entropy codes and escapes; illegal tool combinations; corrupt state/config changes; false sync; decompression bombs; and worst-case expensive valid streams.

Require no panic, overflow, out-of-bounds access, unbounded allocation/queue growth, infinite loop, uncontrolled recursion, or stale-state reuse. Enforce explicit size/time/work limits. Preserve useful errors and documented recovery behavior.

Use property-based tests for bit operations, valid syntax serialize/parse, frame/sample accounting, chunking invariance, and codec-specific mathematical invariants. Do not rely only on randomly mutating a tiny happy-path fixture.

Fuzz independent entry points for configuration, raw units, transports, entropy/syntax paths, and stateful feed/drain/reset sequences. Fuzz encoder config/source validation and bounded encode/drain if encoding is requested. Instrument permitted platform adapters appropriately; safe core code still needs logic/resource testing.

Set a recorded release campaign budget before implementation. If project policy is absent, start with at least 30 minutes per primary untrusted decoder/transport/config target, plus bounded encoder fuzzing and longer scheduled campaigns according to risk. State target, runner, duration, seeds/corpus, limits, coverage/finding summary, and minimized regressions. Compilation alone is not a fuzz run; zero crashes is evidence for the recorded campaign, not proof of safety.

Convert every finding into a minimized targeted regression, rerun affected campaigns, and keep unresolved findings as acceptance blockers.

## Builds, performance, and CI

Run active project-required checks. For the observed Cutlass workspace, verify these remain appropriate:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo deny check
```

Also run meaningful targeted codec/container/integration tests, doc tests, release-mode tests, and supported feature combinations. Validate default builds without external codecs/platform backends. Audit transitive dependencies and ensure inherited lints remain active. Do not fabricate successful results when tooling is missing.

Use native runners for Linux, Windows, macOS Intel/Apple Silicon where supported and available; label cross-compilation separately from executed tests. Exercise default/optional features as applicable. Do not demand unsupported target combinations blindly.

Benchmark release builds on named hardware with recorded OS/compiler/options, content, profile, settings, threads, and repetitions. Measure throughput/real-time factor, first-output and algorithmic delay, p50/p95/p99 processing latency, allocations, peak and steady memory, long-run boundedness, cancellation latency, and worst-case valid work.

Use project budgets. If absent, declare concrete realistic budgets tied to named hardware and use cases before tuning; for common media playback/audio tasks target at least real-time at the explicitly chosen reference configuration. Report misses honestly. Avoid a universal FPS or MB/s promise across codecs, dimensions, channels, and CPUs.

Keep fast deterministic checks in normal CI. Separate external conformance/oracle, quality, scheduled fuzz, and benchmark jobs where appropriate. Make local missing-tool skips explicit, fail jobs that promise those tools, and fail empty suites. Do not silently turn release gates into optional jobs.
