# AV1 decoder architecture

Crate: `cutlass-codec-av1`. Decode only. It depends on `cutlass-core` for
`VideoFrame`, `PixelFormat`, `ColorSpace`, and `RationalTime`. It does not
depend on the editor, compositor, timeline, or another codec.

Normative behavior follows AOMedia AV1 Bitstream & Decoding Process
Specification v1.0.0 with Errata 1 (git `4f2ff41559ab4ac241d1957a17e1f1c2814e26d2`).
libaom v3.13.1 (`d772e334cc724105040382a977ebb10dfd393293`) is the decoding
oracle. dav1d 1.5.4 (`54706fc6bc0cdecab7e9593974a4039cc038fca7`) is an
independent cross-check when built. See `REFERENCES.md`.

## Input

`InputFraming::LowOverhead` (default) and `InputFraming::AnnexB` are explicit.
The decoder does not sniff framing.

A `submit` packet is accepted all-or-nothing. It must contain one or more
complete OBUs. A truncated OBU rejects the packet and does not change committed
sequence or reference state. A valid frame header whose tile groups have not
all arrived is an incomplete frame: the packet is consumed and `receive`
returns `NeedInput`. `drain` then reports that unfinished frame as an error
and does not invent tiles.

`obu_has_size_field == 0` is legal only when the framing contract supplies an
enclosing length (Annex B `obu_length`, or an explicitly bounded low-overhead
unit). The size field and the enclosing length must agree when both exist.

Tile-list OBUs are parsed only far enough to bound their payload. The
specialized large-scale tile mode is **unsupported** and returns
`DecodeError::Unsupported`. It is never decoded as a tile group.

## Output

`receive` returns `Frame`, `NeedInput`, or `Drained`. Hidden frames update
references and do not produce a presentation. Show-existing produces its own
presentation record and does not reuse the stored frame's original timestamp.

Default film-grain mode applies the reference-process synthesis to a copy of
the restored image. `FilmGrainMode::Reconstruction` returns the grain-free
restored image and still keeps the parsed grain parameters on the frame.
Grain never writes into reference pixels.

Samples are native precision, low-aligned, in `PixelFormat` planar layouts
(`Yuv420p` / `Yuv422p` / `Yuv444p` and 10/12-bit `*p10`/`*p12`, or `Gray*`).
10/12-bit storage is native-endian `u16` in the low bits. This is not P010.
Monochrome has one plane. Identity-matrix streams stay planar with
`MatrixCoefficients::Identity` and are not converted to packed RGB.

Coded size, upscaled size, render size, and the visible rectangle are separate
fields on `FrameInfo`. `VideoFrame.coded_size` is the upscaled reconstructed
size. `VideoFrame.visible` is the render rectangle clipped to that buffer
(render size is display metadata and does not allocate a second scaled surface).

## State lifetimes

| State | Lifetime |
| --- | --- |
| Immutable tables (CDF defaults, scans, quantizers, filters) | Process, read-only |
| `SequenceHeader` | Until a new coded video sequence replaces it |
| Partial OBU / tile-group assembly | Until the frame completes, fails, or `reset` |
| Eight reference slots | Committed at frame wrapup; cleared by `reset` or keyframe rules |
| Tile CDF working copy | Tile only; the context-update tile is the one saved |
| Block scratch, predictions, residuals | Reused per block; not published |
| Output `VideoFrame` | Caller-owned, immutable, not aliased to decoder scratch |

Reference slots store restored pixels (no grain) plus the header state later
frames load: CDFs, segmentation, loop-filter deltas, global motion, order
hint, frame id, motion field, and grain parameters. Pixel buffers are
reference-counted. Show-existing clones the `Arc` and builds a new
presentation.

`reset` drops assembly, outputs pending only according to `reset` (queued
presentations are discarded), references, and sequence header. `drain` does
not. A decode-time failure after a completed earlier frame in the same packet
leaves that frame queued; the decoder enters `needs_reset` and rejects further
input until `reset`.

## Scheduling

The scalar decoder runs on the caller thread. `DecoderConfig::workers` must be
1. There is no process-global stream lock; instances are independent.

## Limits

Defaults live in `DecoderLimits`. A sequence header's max dimensions are not
allocation authority: the configured caps are. Allocation uses checked
arithmetic. Public plane buffers are fully initialized.
