# AV1 decoder plan

Ordered work. Each step is a compiling checkpoint. Status is updated in
`COVERAGE.md`.

1. Shared planar formats for 4:2:2, 4:4:4, 10/12-bit low-aligned samples, and
   monochrome. Done in `cutlass-core`.
2. Crate, error model, limits, low-overhead and Annex B framing, bit reader,
   sequence header, uncompressed header.
3. Multisymbol range decoder and default CDFs extracted from pinned libaom
   (spec representation, not libaom inverse CDFs).
4. Intra keyframe reconstruction: partitions, modes, coefficients, inverse
   transforms, dequant, loop filter / CDEF / restoration / superres when the
   frame signals them, film grain.
5. Inter frames: references, motion vectors, compound and warp, skip mode,
   show-existing, frame-end CDF commit.
6. Profiles 1 and 2, 12-bit, still picture, operating points, metadata side
   data.
7. Differential harness against pinned `aomdec` (grain off and grain on),
   fuzz targets, benchmarks.
8. Engine registration is the crate itself plus `capabilities()`. File demux
   stays outside the decoder. The example `av1dec` accepts IVF or raw
   low-overhead / Annex B and dumps planar frames.

Routine choices already made:

- Public API is `Av1Decoder::{new, submit, receive, drain, reset, capabilities}`
  in this crate. `cutlass_core::DecodeError` stays the file-decoder error;
  this crate has its own structured `DecodeError` and `From` into the core type.
- Workers greater than 1 are rejected until a schedule-independent tile
  implementation exists.
- Specialized tile-list mode is unsupported on purpose.
- Strict decoding. No concealment.
