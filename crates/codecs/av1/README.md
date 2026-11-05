# cutlass-av1

An AV1 decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`, no
dependencies outside this workspace, no C libraries). FFmpeg with libdav1d,
libaom, SVT-AV1 and rav1e is used only as a test oracle and as the
benchmark baseline. `STATUS.md` is the detailed support and test record.

## Decoder

`dec::decoder::Av1Core` implements the decoding process of the AV1
specification (v1.0.0 with errata 1) for all three profiles:

- 8, 10 and 12 bits; 4:2:0, 4:2:2, 4:4:4 and monochrome; odd sizes.
- Every intra tool: directional modes with angle deltas and edge filtering,
  smooth/Paeth, recursive filter intra, chroma from luma, palette, intra
  block copy.
- Every inter tool: single and compound references, NEWMV/NEAR/NEAREST/
  GLOBAL and the compound modes, reference MV projection
  (`use_ref_frame_mvs`), OBMC, local and global warped motion, wedge,
  difference-weighted and distance-weighted compound, inter-intra, skip
  mode, switchable and dual interpolation filters, scaled references
  (reference scaling and frame size changes).
- All transforms (DCT/ADST/flip/identity up to 64 points, WHT for
  lossless), quantizer matrices, segmentation, delta q / delta lf.
- Deblocking, CDEF, loop restoration (Wiener and self-guided), super
  resolution, film grain synthesis.
- 64x64 and 128x128 superblocks, uniform and non-uniform tiles, multiple
  tile groups, hidden frames, `show_existing_frame`, error resilient and
  intra-only frames, frame ids, order hints.
- Low-overhead OBU streams (as in IVF, MP4 and Matroska) and Annex B
  length-delimited temporal units (`Av1Core::decode_annexb`).

Not implemented: large-scale tile decoding (`OBU_TILE_LIST`, the light-field
extension). libdav1d, and therefore FFmpeg, does not implement it either.
Only operating point 0 is output.

Correctness: the output of 46 streams produced by libaom, SVT-AV1 and rav1e
through FFmpeg, covering the tools above, matches libdav1d byte for byte
(`tools/conf.py`, `tools/checkall.sh`). A mutation fuzzer
(`examples/av1fuzz.rs`) finds no panics; corrupt input returns a
`CodecError` and leaves the reference frames unchanged.

With several threads (`Av1Core::threads`) parsing, reconstruction and the
post filters run in a pipeline. It takes about 1.9x the time of libdav1d
single threaded and 1.6x with 4 threads; libdav1d uses hand-written
assembly (numbers in `STATUS.md`).

## Encoder

`enc::encoder::Av1Enc` codes 8/10/12-bit 4:2:0, 4:2:2, 4:4:4 and
monochrome (and RGB via the public wrapper). It runs the decoder's own tile
syntax code with an arithmetic *writer*, so every bitstream it produces is
decoded by construction; an internal `Av1Core` decodes each packet to make
the reference frames. Tools used:

- Hierarchical mini-GOPs (8 frames by default) behind a hidden alt-ref,
  `show_existing_frame`, up to seven references, per-layer q offsets.
- Partition search from transform-coded RD estimates (4x4 to 64x64).
- Intra: all 13 modes, angle deltas, filter intra, chroma from luma,
  palette (with per-frame screen content detection).
- Inter: motion search over every distinct reference (pruned by speed),
  NEAREST/NEAR/GLOBAL/NEW and compound NEAREST/NEW, temporal MV prediction,
  switchable interpolation filters, local warped motion, OBMC, skip mode.
  Global motion estimation and alt-ref temporal filtering are optional.
- Transform type and size by RD; RDOQ with exact coefficient rates from the
  live CDFs.
- Per-frame search of the deblocking level and CDEF strengths.
- Constant q or a bitrate target (`RateControl::Bitrate`).
- Tile-parallel coding on several threads (`set_threads`); the bitstream
  does not depend on the thread count.

Quality and speed against FFmpeg's encoders are in `STATUS.md`. In short:
on natural video it needs fewer bits than rav1e `-speed 8` for the same
PSNR and about as many as libaom `-cpu-used 6` (within a few percent
either way); key frames and synthetic content are further behind. It runs at about half the speed of libaom at that
preset (single thread).

## API

```rust
use cutlass_av1::{Av1Decoder, Av1Encoder};
use cutlass_decode::{CodecId, CodecParams, Decoder};
use cutlass_encode::Encoder;

let params = CodecParams::video(CodecId::Av1, 352, 288);
let mut enc = Av1Encoder::open(&params)?;
enc.set_quality(120); // base q index; or set a bitrate in CodecParams
enc.set_speed(6);     // 0 slowest/best .. 10 fastest
enc.set_gop(8);       // 1 = no reordering, one packet per picture
let mut packets = Vec::new();
// for each picture: packets.extend(enc.push_frame(&frame)?);
packets.extend(enc.drain()?);

let mut dec = Av1Decoder::open(&params)?;
for p in &packets {
    for out in dec.push(p)? { /* out.as_video() */ }
}
```

Lower level: `dec::decoder::Av1Core::decode(&[u8])` takes one temporal
unit and returns the shown frames; `enc::encoder::Av1Enc::encode` takes
plane slices and returns finished temporal units (`flush` at the end).

## Examples and tools

- `examples/av1dec.rs` IVF to raw YUV (`--features stats` prints tool use).
- `examples/av1enc.rs` Y4M to IVF (`--q`, `--kbps`, `--speed`, `--gop`,
  `--keyint`, `--tiles`, `--threads`, `--global-motion`,
  `--temporal-filter`, `--recon`).
- `examples/av1bench.rs` decode timing with a thread count.
- `examples/av1fuzz.rs` mutation fuzzer.
- `tools/` the scripts behind the numbers in `STATUS.md`: stream generation
  and libdav1d comparison (`conf.py`, `cases.py`, `checkall.sh`), the
  encoder format matrix (`encmatrix.sh`), and rate-distortion comparisons
  (`rdcmp.sh`, `kbd.sh`, `bd.py`). Encoder constants can be overridden from
  the environment for sweeps when built with `--features tune`.
