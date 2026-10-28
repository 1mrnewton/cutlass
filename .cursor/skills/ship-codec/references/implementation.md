# Native codec implementation contracts

## Architecture and configuration

Inspect the actual project's dependency direction before adding public types. Keep the codec usable independently of UI, file paths, devices, platform frameworks, and specific containers. Reuse shared bitstream/DSP/picture/PCM utilities where they exist; introduce minimal typed contracts where they are still scaffolds.

Document codec configuration separately from transport headers and container metadata. Validate profile/tool/rate/layout combinations together; checking each enum independently can accept impossible combinations. Return a typed unsupported configuration rather than claiming capabilities from a registration flag.

Keep parsing and reconstruction/analysis boundaries testable. Give tables, constants, generated artifacts, and algorithms provenance. Verify lengths and indices before indexing, checked arithmetic before allocating, and iteration limits before processing hostile input.

## Public API behavior

Specify and test all of these:

| Contract | Required decision |
| --- | --- |
| Input | Complete access unit versus arbitrary byte chunks; ownership; maximum sizes; consumed bytes; format/config side data. |
| Output | Zero/one/many outputs per input; explicit buffer layout, units, metadata, lifetime, and valid content region. |
| Flow control | Need-input, output-ready, backpressure, end-of-stream, and errors; bounded pending input/output. |
| Timing | Rational time base, PTS/duration, decode/display order where relevant, exact sample/frame counts, missing timestamps. |
| End of stream | Drain delayed output; define repeat drain and input after end-of-stream. |
| Reset/discard | Clear histories, references, queues, cached config, and counters as defined; distinguish from drain. |
| Seek | Required preroll/keyframe behavior, history reconstruction, discontinuity, stale output suppression. |
| Changes | Surface rate/layout/resolution/config changes before consumers interpret output with old configuration. |
| Failure | Typed malformed/unsupported/resource/API-state errors; location/context; input consumption and reusable state after failure. |
| Recovery | Strict default or explicit engine policy; bounded resync/concealment; mark affected output. |
| Concurrency | Ownership/thread-safety, deterministic work partition, cancellation, bounded workers/queues. |

Avoid heavy work and unpredictable waits/allocations inside real-time device callbacks. Allocate/warm up at setup where practical. Avoid global mutable codec state, hidden I/O, logging full input, and control paths that panic.

Keep implementations usable when optional platform backends are absent. Test explicit software selection and actual capability negotiation.

## Decoder implementation

Implement the advertised syntax and reconstruction semantics in specification order. Do not skip a coding tool because a parser can skip its bytes if the tool changes reconstructed output.

Cover applicable stream headers/configuration, unit boundaries, entropy coding and adaptation/reset, inverse quantization/transforms, prediction, reconstruction, filters, channel/component mapping, postprocessing required by the format, metadata, and persistent histories/reference buffers.

Enforce codec-specific normative maxima plus configurable lower resource limits. Cap expanded size, recursion/partition depth, reference count, channels/components, per-unit elements, entropy escape loops, and resync work. Safe Rust alone does not prevent CPU/memory exhaustion.

For unsupported tools, determine whether a precise error can be returned before state changes. Either stage/commit new state on success or document a reset requirement after failure. Never reuse partially corrupted reference or overlap state silently.

Distinguish normal end-of-input, a partial streaming unit, a truncated complete unit, and a structurally invalid unit. Make truncated-stream handling consistent and test every syntax boundary.

## Encoder implementation

Build a standards-valid output path and a useful quality/rate implementation.

Implement applicable analysis, prediction/mode decisions, transforms, quantization, entropy coding, legal header/tool signaling, reference/history updates, and output/configuration metadata. Ensure encoder decisions match what the signaled profile permits.

Bound search, lookahead, reservoir/buffer operations, iterations, and output size. Check worst-case legal inputs, not only typical footage/music. Emit deterministic output within a documented compiler/platform/threading scope where practical.

For lossy formats, implement justified rate-distortion or psychoacoustic decisions; a fixed quantizer demo is not a shipping encoder. Specify actual quality/rate modes, legal bitrate constraints, buffering, latency tradeoffs, keyframe/transient decisions, and impossible-request behavior.

For lossless formats, preserve the promised sample/pixel domain exactly, including precision and component mapping. Perform no unrequested resampling, chroma conversion, normalization, clipping, alpha discard, or irreversible color transform.

Do not make the paired decoder the only bitstream validator. Test interoperability before tuning optimizations around your own implementation.

## Containers and application boundaries

Keep raw codec payload, transport framing, and mux/demux responsibilities distinct. Verify length-prefixed versus start-code/raw framing, extradata/config records, codec identifiers, edit/trim metadata, timestamps, and packetization. Place adapters with their current project owner; avoid cycles to reuse one helper.

Exercise import -> demux -> decode -> engine frame and engine frame -> encode -> mux -> reopen with an independent implementation, for the requested direction(s). Require this through executable available APIs; label absent application paths accurately.

Validate cancellation and cleanup on success, error, and user abort. Do not publish output metadata before initialization is complete or leave a successful-looking truncated export after failure.

Treat native platform acceleration as an independent backend: capability query, negotiation, error mapping, reset/drain/cancellation, ownership, and backend-specific tests. Preserve native-software validation and software availability.
