# cutlass-vp9

A VP9 decoder and encoder in Rust. The decoder takes every profile (0–3,
8/10/12-bit, 4:2:0/4:2:2/4:4:0/4:4:4, sRGB) and matches FFmpeg's `vp9`
decoder sample for sample. The encoder writes every profile too (0–3,
8/10/12-bit, 4:2:0/4:2:2/4:4:0/4:4:4) with
rate-distortion searched partitions and modes, golden frames, tile columns on
threads, and one-pass rate control. The crate is safe Rust
(`#![forbid(unsafe_code)]`); its vector kernels live in `cutlass-simd`
(`cutlass_simd::vp9`), where each `unsafe` block is one call into a
`#[target_feature]` kernel with a scalar twin and a test.

> Status: work in progress on the `vp9` branch. Numbers below are from this
> branch and are refreshed as the work goes on.

## What it supports

Decoder (`Vp9Decoder`, over `Decoder`):

| Profile | Pictures |
| --- | --- |
| 0 | 8-bit 4:2:0 |
| 1 | 8-bit 4:2:2, 4:4:0, 4:4:4, and sRGB (published as GBR) |
| 2 | 10-bit and 12-bit 4:2:0 |
| 3 | 10-bit and 12-bit 4:2:2, 4:4:0, 4:4:4, and sRGB |

- Key, inter and intra-only frames, hidden frames, `show_existing_frame`,
  superframes, tiles, eight reference slots, scaled references (and
  mid-stream size changes), segmentation, the loop filter with deltas,
  compound prediction, switchable filters, probability adaptation, error
  resilient and frame parallel modes, lossless frames.
- Threads: frame threads (one frame's reconstruction overlaps the next one's
  parse) and tile-column threads for the parse. `Vp9Decoder::open` uses one
  thread per core; the frames still being made when `push` returns come out
  of later pushes or `drain`, as with FFmpeg's frame threads.

Encoder (`Vp9Encoder`, `EngineEncoder`):

- Profiles 0 to 3, any size up to 65536×65536 (odd sizes too), quantizer
  index 0 (lossless, Walsh-Hadamard) to 255.
- Partitions from 64×64 to 8×8 (rectangles below speed 4), 4×4 intra
  blocks on key frames (each 4×4 with its own mode), and inter blocks
  under 8×8 (4×4, 8×4, 4×8, each sub-block with its own mode and vector,
  speeds 0–1); all ten intra modes, NEAREST/NEAR/ZERO/NEW inter modes on
  LAST, GOLDEN and ALTREF, compound prediction (a past picture averaged
  with the alt-ref ahead, chosen per block, speeds 0–6), sub-sample motion
  (to eighths where the vector may be), per-block transform size
  (`TX_MODE_SELECT`, 4×4 to 32×32 with the intra ADST types), skip, the loop
  filter (libvpx's level from the quantizer), probability adaptation.
- Segmentation: with `Vp9Options::aq` (libvpx's `aq-mode=1`) each 32×32
  region goes in one of five segments by its luma variance, each with
  its own quantizer (flat regions finer); the map is coded per block.
- A golden frame every 12 frames, key frames every `keyint` (240) frames;
  key and golden frames take finer quantizers. With `Vp9Options::lag`
  (15 and up; 16 by default below speed 9) an interval whose end the current picture reaches starts
  with a hidden alt-ref frame, the temporally filtered (`enc::arf`, after
  libvpx) picture at its end, sent in a superframe with the next shown
  frame; the overlay at its position refreshes GOLDEN.
- Probability updates in the compressed header, chosen on the frame's own
  counts: the coefficients' and every other updatable one (skip, tx size,
  inter modes, filters, intra/inter, references and compound, intra modes,
  partitions, motion vectors). The final pass's symbols name their
  probabilities (`enc::probs`) and are written once the updates are
  known.
- Speeds 0–10 (`Vp9Options::speed`, default 5). Up to 6 every superblock is
  searched by trial coding; 7 and up code inter frames in one pass.
- Threads: tile columns, each on its own thread (`Vp9Options::threads`, 0 =
  one per core), up to the tile columns the width allows; at speed 7 and
  up the motion search runs first over the whole frame, its superblock
  rows shared evenly among the threads, and luma and chroma are filtered
  on separate threads.
- Every chroma format of 8-bit VP9 (`Vp9Options::chroma`: 4:2:0 in
  profile 0, and 4:2:2, 4:4:0 and 4:4:4 in profile 1, FFmpeg's `-pix_fmt
  yuv420p` to `yuv444p`; `CodecParams` with those formats opens them).
  The encoder takes the subsampling from the frame header it parses
  back, and every tool works with it (alt-refs, segments, tile rows,
  lossless, two passes); in 4:2:2 and 4:4:0 the halves whose chroma has
  no size of its own, which libvpx refuses (8×16, 16×32 and 32×64 in
  4:2:2, the 2:1 ones in 4:4:0), are never chosen, and the loop filter
  takes the frame's grid of block sizes as the decoder's does. FFmpeg's
  `vp9`, libvpx and this decoder decode 28 streams across the four
  formats, speeds 2 to 10, odd sizes, two passes and AQ to the same
  pictures, and lossless comes back exact. On the grainy 720p pan in
  4:4:4 (30 frames), against libvpx-vp9 at good, cpu-used 1 in 4:4:4:
  speed 7 is −20% / −23% at the same PSNR-Y / averaged PSNR in 1.76 s
  against 2.30 s, and speed 6 −32% / −35% in 3.1 s.
- 10 and 12 bits (`Vp9Options::bit_depth`: profile 2 for 4:2:0, 3 for the
  other formats; `Vp9Encoder::push_planes16` and `encode_planes16` take
  `u16` samples in the low bits, and `CodecParams` with FFmpeg's
  `yuv420p10le` to `yuv444p12le` formats opens them; the `vp9enc` example
  reads them with `bits=10` or `12`). The encoder is generic over its
  samples as the decoder is: one path for `u8` and `u16`, each with its
  own AVX2 kernels for interpolation, the loop filter and the SADs
  (`cutlass_simd::vp9::{mc16, lf16, sad16}` for 16 bits). Distortions
  stay at the samples' scale, so what weighs against them scales with
  it: the RD lambda comes
  from the bit depth's own step (its square grows as the SSE does), the
  SAD lambda is libvpx's `sad_per_bit16` on the step brought to 8 bits
  and scaled back up, the loop filter level is libvpx's `LPF_PICK_FROM_Q`
  for 10 and 12 bits, the temporal filter's thresholds and strength and
  the intra trial's threshold scale with the samples, the largest token
  category takes 16 or 18 bits, and rate control models 8-bit-scaled
  steps. On 8-bit content shifted up two or four bits (a grainy 720p pan,
  60 frames, speed 6), every quantizer from 100 to 180 spends within 1.3%
  of the 8-bit stream's bits at 0.02 to 0.06 dB more PSNR; at 60, where
  the 10- and 12-bit steps are finer than 8-bit's, 10% and 13% more bits
  for 0.75 and 0.97 dB more. FFmpeg's `vp9`, libvpx and this decoder
  decode 210 streams (10- and 12-bit, all four chroma formats, 640×360
  and 333×211, speeds 2 to 10, lag 16 and 0, quantizers 0, 60 and 180)
  to the same pictures, and the 70 lossless ones come back exact.
  Against libvpx-vp9 at 10 bits (good, cpu-used 1, 4 threads each) the
  gap is the 8-bit one: on the grainy pan speed 6 is −38% / −34% at the
  same PSNR-Y / averaged PSNR (−38.5% / −34.8% on the pan made at 8
  bits), speed 8
  −19% / −7.9%; on a clean synthetic pan (`testsrc2`), where we are
  behind, +19% / +28% (+20% / +29% at 8 bits). With its own kernels the
  16-bit path costs little more than the 8-bit one: on the grainy pan at
  q 110, speed 6, 4 threads (whole-process medians of seven runs
  interleaved with libvpx's, libvpx at good, cpu-used 1, 1.4 Mb/s, 4
  threads, `-row-mt 1 -tile-columns 2`):

  | | 8-bit | 10-bit | 10 / 8 |
  | --- | ---: | ---: | ---: |
  | Cutlass, speed 6 | 2.44 s | 2.71 s | 1.11 |
  | Cutlass before the 16-bit kernels | 2.43 s | 6.46 s | 2.66 |
  | libvpx-vp9, cpu-used 1 | 1.94 s | 3.02 s | 1.56 |

  The streams are byte for byte those of the scalar 16-bit path (speeds
  2, 6 and 10, 4:2:0 and 4:4:4, lossy and lossless, 10 and 12 bits).
  Decoding libvpx's 1080p 4:2:0 at 10 bits (200 frames of `mandelbrot`,
  4 Mb/s, four tile columns), the decoder is now slightly ahead of
  FFmpeg's `vp9` on one core (medians 2 to 4% lower) and ahead on four
  threads, with the same pictures. Medians of interleaved runs on a
  shared VM with AVX-VNNI (two sessions, 17 and 19 runs on one core, 15
  and 15 on four threads; single runs vary by ±10%):

  | 10-bit, 1080p | Cutlass | Before | FFmpeg | Cutlass faster |
  | --- | ---: | ---: | ---: | --- |
  | 1 core | 1.23 / 1.30 s | 1.50 / 1.68 s | 1.28 / 1.32 s | 15 of 17, 12 of 19 (1 tie) |
  | 4 threads | 0.47 / 0.56 s | 0.60 / 0.66 s | 0.56 / 0.61 s | 13 of 15, 13 of 15 (1 tie) |

  The 8-bit decoder is unchanged by this, within the noise (1.03 s
  before and 0.99 s after on the same stream at 8 bits, 15 interleaved
  runs; FFmpeg 1.02 s). What moved the 10-bit decode, from 5.78 s on one core before
  the 16-bit kernels:
  - The 16-bit kernels for interpolation, the loop filter and the SADs
    (1.44 to 1.58 s).
  - Pictures go out as views of the decoder's own planes, as 8-bit ones
    do (the little-endian bytes of the `u16` samples, on a little-endian
    target), instead of a copy into tight rows: 6% of a one-core decode.
  - The inverse transforms run on the 8-bit path's 8-wide `i32` lanes,
    not libvpx's 64-bit `tran_high_t`, for every block whose
    coefficients' magnitudes sum to at most a bound per size (39 000 to
    77 000). A test derives the bound from the transforms themselves, run
    on bounds of each input's weight in every value they make, so below
    it no sum, product or rounding leaves `i32` and the two lane widths
    give the same values; above it the 64-bit lanes run as before.
  - The reference rows of the next inter block 32 wide or more are hinted
    into the cache while the current one is predicted
    (`cutlass_simd::vp9::prefetch`): a 64×64 block reads 71 rows a stride
    apart, more than the hardware prefetchers follow, and at 10 bits
    twice the bytes; about 5%. Smaller blocks and 8-bit pictures lost
    more than they gained and take no hints.
  - The interpolation kernel holds sums to the range in 32 bits against a
    constant top (the compiler otherwise added a clamp of its own around
    each pack), interleaves each pair of source rows once, puts the odd
    outputs across into the high halves instead of interleaving and
    packing, and has an AVX-VNNI build (`vpdpwssd`, chosen at run time):
    a 64×64 2-D prediction in cache from 0.46 to 0.37 ns a sample.
  - The 16-bit loop filter skips the filters no line of an edge takes,
    and stores only what can have changed.

  What is left is spread out: on the first 60 frames Cutlass runs 2.83 G
  instructions (on AVX2 alone) against FFmpeg's 2.39 G, nearly half of
  them in interpolation and the loop filter, and the interpolation's time is
  still partly waiting on the reference rows. A lossless frame no longer
  lets an inter block drop its residual (both its costs are zero there,
  so the skip bit's probability decided): lossless with alt-refs came
  back inexact before, at 8 bits too.
- Tile rows (`Vp9Options::tile_rows`, log2 of the count, 0 to 2, as
  libvpx's `tile-rows`): each starts its own arithmetic coder, so a
  decoder can start every row's at once; nothing else changes, and the
  pictures are the same (four tile rows cost 0.2% more bits at 720p).
- Rate: a fixed quantizer (`with_quantizer`, or bitrate 0) or a bitrate,
  per frame type, from a model of bits against quantizer step. FFmpeg's
  `libvpx-vp9` modes map onto it: `-b:v` alone is the bitrate on average
  (`end-usage=vbr`); `-crf` with `-b:v` is that with a floor on the
  quantizer (`min_q`), and `-crf` with `-b:v 0` a fixed quantizer, where
  `Vp9Encoder::qindex_of_level` turns libvpx's 0–63 levels into indices.
  A constant bitrate (`-minrate = -maxrate`, `end-usage=cbr`) is
  `Vp9Options::cbr_buffer_ms`: smaller shares for golden and key frames,
  golden and key frames at most 16 indices finer than the inter frames,
  the first key frame coarser, and a frame over its budget paid back
  within half the buffer. On a grainy 720p pan at 2 Mb/s and speed 9 its
  half-second windows stay within −24% and +13% of the rate (−49% and
  +32% on average; libvpx's realtime 6 CBR −45% and +5%) at 39.9 dB
  averaged PSNR, against libvpx's 38.2 dB.
- Two passes (FFmpeg's `-pass 1` and `-pass 2`): a first pass over the
  pictures at a fixed quantizer, fast (the `vp9enc` example runs it at
  speed 10 with `twopass=1`, at `Vp9Encoder::first_pass_qindex`, about
  the quantizer the bitrate will take), gives each picture's bits to
  `Vp9Encoder::set_second_pass`. The second pass codes at one quantizer
  for every frame (as a fixed quantizer, which puts the bits where the
  pictures need them), the one that spends the bitrate: frame i takes
  about k·b_i·(s/s₁)^-1.25 bits, with b_i its first-pass bits and k the
  passes' scale, measured on the frames coded so far, apart for inter
  frames, for key frames and cuts, and for golden frames, whose bits an
  alt-ref takes from its target when it is coded (the first pass codes
  none). k starts at 0.5 (a speed-10 first pass spends about twice speed
  6's), each class's leans on the overall one until a few frames' worth
  are in, and over the last ten frames a miss is spread as if the stream
  went on, not made up on them alone. A cut takes a key frame's
  quantizer, and blocks that code over a bit a sample also try the next
  smaller transform, as speeds 0–5 do. Against libvpx's two passes (good,
  cpu-used 2), at the same PSNR-Y / averaged PSNR, speed 6:

  | 90 frames of 720p | Before | Now |
  |---|---|---|
  | `mandelbrot`, the grainy pan, `testsrc2` (1–4 Mb/s) | +5.3% / +16.9% | −10.7% / +4.5% |
  | still picture, grainy pan, clean clip (0.5–3 Mb/s) | +2.9% / +11.8% | −19.7% / −5.8% |

  in the same time as before (6.6 s, against libvpx's 10.6 s with four
  threads each, at 2 Mb/s), within 2.4% of the bitrate (4.3% on the
  60-frame pan below at 1 Mb/s). What is left is
  the chroma of the grainy pan (+19% at the same averaged PSNR on that
  scene alone, −12% at the same PSNR-Y). Of the gain, the first pass at
  the bitrate's quantizer (a fixed 120 reached too far: grain loses its
  bits faster with the quantizer than clean pictures do, and the second
  pass ran out of bits on the last scene) is most on the second clip,
  the smaller transforms (3 to 4% more time) most on the first, the
  classes and the cut's quantizer some on both. On the 60 frames of the
  grainy pan alone (no cut), the second pass takes 30% / 18% off the bits
  it took before, over the narrow range its PSNR-Y tops out in (43.3 to
  44.3 dB; +2.4% / +23% against libvpx's, from +34% / +49%); its one pass
  is unchanged, bit for bit, as every one-pass stream. Golden frames every six
  frames in the second pass took 1 to 2% more off, at 11% more time;
  coding a cut as a key frame, libvpx's way, did better on the first clip
  and worse on the second; the chroma's RD weight brought luma down with
  it on both.
- Cuts: a picture whose sampled 16×16 blocks match the last picture
  worse than twice the error of their own means (libvpx compares inter
  and intra errors the same way) starts a golden frame: LAST and GOLDEN
  take the new scene, an alt-ref of the old one is dropped, every block
  still chooses intra or inter, and the probabilities the stream has
  learnt are kept. (A key frame, as libvpx codes a cut, did worse here on
  a cut to another picture of the same subject: 13% more bits at one
  pass.) Within one scene the ratio stays under 0.95, grain and all, so
  no single-scene stream changes. On 90 frames of 720p with two cuts to
  other content (FFmpeg's `mandelbrot`, the grainy pan, `testsrc2`), at
  speed 6: −13% / −12% of the bits at the same PSNR-Y / averaged PSNR at
  one pass, −6.5% / −5.3% at two; on the clip with two cuts between
  related pictures, −4% / −3.6% and −7.1% / −3.6% (before the second
  pass's model above).

## How it works

### Decoder (`src/dec`, `src/decode.rs`)

- `dec::bool`: the bool decoder, a 64-bit window refilled a word at a time,
  normalized with `leading_zeros`.
- `dec::parse`: one superblock row of one tile: modes, motion vectors
  (libvpx's candidate search with its early exits), coefficient tokens. It
  writes a compact syntax record (`dec::syntax`) and the loop filter masks,
  and counts symbols for adaptation.
- `dec::recon`: prediction and residual from that record: intra edges,
  8-tap motion compensation (scaled references too), inverse transforms.
- `dec::lf`: the loop filter of a superblock row, in libvpx's mask order,
  one row behind reconstruction.
- `dec::mod`: a frame's parse, all tile columns at once on threads or a row
  at a time; its reconstruction on a worker that publishes finished rows
  (`cutlass_simd::vp9::frame::SharedFrame`), so the next frame's motion
  compensation waits only for the rows it reads.
- `dsp`: transforms, intra predictors, motion compensation and the loop
  filter, generic over 8 and 16-bit samples; the hot paths call the
  AVX2 kernels in `cutlass_simd::vp9` (`mc` and `lf` for 8 bits, `mc16`
  and `lf16` for 10 and 12), and the transforms run in lanes compiled for
  AVX2 (`cutlass_simd::vp9::wide`).

### Encoder (`src/enc`)

- `enc::mod`: a frame. It writes the headers, parses them back with the
  decoder's own parser, codes the tile columns (on threads), merges them,
  filters, and commits the frame through `Decoder::commit` — so the encoder
  predicts from exactly the references, motion and probabilities a decoder
  holds, and its reconstruction is the decoder's (`dec::recon`, `dec::lf`).
- `enc::tile`: a tile column. For each superblock, `rd_node` searches the
  partition: each square is trial-coded whole (`best_block`: its candidate
  modes coded and costed, the cheapest kept) and split, with snapshots of
  every context and sample the coding touches, and SSE + λ·bits deciding.
  The winner is then coded for real. Coefficients are quantized by
  rate-distortion (`rd_levels`: each level the cheaper of the two nearest
  in distortion plus λ times its token bits, then the end of block placed by
  cost); inter blocks drop their residual when it does not pay.
- `enc::tx`: forward transforms, the exact inverses of the decoder's,
  measured from it.
- `enc::api`: the codec traits, rate control, frame types.
- `dsp::sad`: SAD, SSE and bilinear sub-sample SAD, written over fixed-width
  rows so they compile to `vpsadbw` under AVX2; at 10 and 12 bits the SADs
  and SSE are the kernels of `cutlass_simd::vp9::sad16`.

## Using it

```rust
use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet};
use cutlass_vp9::Vp9Decoder;

let params = CodecParams::video(CodecId::Vp9, 1920, 1080);
let mut decoder = Vp9Decoder::open(&params)?;
for output in decoder.push(&Packet::video(frame_bytes, 0, 1, 30, true))? {
    // output.frame: planar picture
}
for output in decoder.drain()? {
    // the frames the threads still held
}
```

```rust
use cutlass_vp9::{Vp9Encoder, Vp9Options};

let mut encoder = Vp9Encoder::with_options(1280, 720, Vp9Options { qindex: 120, speed: 5, ..Default::default() })?;
let (bytes, key) = encoder.encode_planes([(y, 1280), (u, 640), (v, 640)], 1.0 / 30.0)?;
```

Examples: `examples/roundtrip.rs` (decode a committed key frame) and
`examples/vp9enc.rs` (raw 4:2:0 to IVF:
`vp9enc in.yuv 1280 720 out.ivf q=120 speed=5 threads=4`).

## How it is tested

```
cargo test -p cutlass-vp9
```

- Decoder: committed streams against stored FFmpeg output, header and bool
  coder tests, a fuzzer (`decoder_fuzz_does_not_panic`), and `src/prove.rs`,
  which encodes streams with FFmpeg's libvpx and compares against
  `ffmpeg -c:v vp9` when those tools are present.
- `tools/corpus.sh` writes 29 libvpx streams (profiles 0–3, tiles,
  lossless, odd sizes, AQ, error resilient, frame parallel, realtime, GBR,
  1080p noise, a mid-stream resize) and `tools/check.sh` compares every
  frame with FFmpeg: all 29 match, threaded and on one core.
- Encoder: every test decodes the encoder's output with this decoder and
  requires the picture to equal the encoder's own reconstruction (key,
  inter and golden frames, odd sizes, lossless, four tile columns on four
  threads, two columns past 4096 samples); FFmpeg decodes the same streams
  to the same pictures.

## Speed and size against FFmpeg

Machine: the 4-core x86-64 VM this branch is developed on (AVX2),
FFmpeg 7 with libvpx. Timings are medians of whole-process runs.

**Decode** (`tools/bench.py`): 1920×1080 streams written by libvpx, 300
frames of 4:4:4 (profile 1, `mandel.ivf`) and 200 of 4:2:0 (`m420.ivf`,
libvpx at 4 Mb/s, four tile columns). Medians of nine runs on four
threads; one core is the fastest of five, pinned with `taskset`.

| Stream | Threads | Cutlass | FFmpeg `vp9` | Ratio | Memory (Cutlass / FFmpeg) |
| --- | --- | ---: | ---: | ---: | --- |
| 4:2:0 | 1 core | 0.98 s | 0.97 s | 1.01 | |
| 4:2:0 | 4 | 0.41 s | 0.47 s | 0.86 | 58 MB / 57 MB |
| 4:4:4 | 1 core | 1.88 s | 1.88 s | 1.00 | |
| 4:4:4 | 4 | 0.70 s | 0.79 s | 0.88 | 80 MB / 78 MB |

On four threads Cutlass is ahead of FFmpeg on both streams (it won all 7
runs on each). On one core, in 31 runs alternating with FFmpeg's
(ratio of the medians, with a 95% bootstrap interval):

| Stream, 1 core | Cutlass / FFmpeg | 95% interval | Cutlass faster |
| --- | ---: | --- | --- |
| 4:2:0 (`m420.ivf`) | 0.997 | 0.972–1.030 | 14 of 31 |
| 4:4:4 (`mandel.ivf`) | 1.011 | 0.972–1.032 | 12 of 31 |
| 4:2:0, 60 frames at 3.7 Mb/s (`q60.ivf`) | 0.948 | 0.929–0.973 | 25 of 31 |

On the two main streams the two are level: neither is measurably the
faster. On the third Cutlass is ahead. (The
4:4:4 stream started at 38.8 s on one core.) What moved the decoder most
recently:

- Each superblock row is loop filtered as soon as it is made, while it
  is in the caches, instead of after the next row: the row's last line
  of each plane is kept as reconstructed for the intra prediction below
  it (FFmpeg's `intra_pred_data`). 4% on one core for 4:4:4, and 8 to 10%
  on four threads.
- The loop filter's drivers pass over strips, rows and columns with
  nothing to filter before building or looking anything up (2% fewer
  instructions).
- The parse fills runs of structures in a loop, and the interpolation
  block takes its filters by reference, not as two 26-byte copies.
- Horizontal loop filter edges go through the filter two at a time, side
  by side in 256-bit vectors (the filter's arithmetic written once over
  the vector width); a strip's edges first, then its inner ones, which is
  libvpx's order in every column. Vertical edges paired 16 rows apart
  were slower (one 15-tap group sends all four down the long path) and
  are left as they were.
- The coefficient loop tops the bool decoder's window up once a token,
  so its end-of-block, zero, one and sign reads skip their refill check
  (2% fewer instructions), and runs as a function of its own.
- The common motion compensation call no longer carries the frame of
  the rare paths (scaled references, windows crossing the edge): a
  quarter fewer instructions a call.
- The kernels' entry points took their block by value and rebound it,
  which moved all 144 bytes through a `memcpy` call on every prediction
  and every edge (6% of a 4:4:4 decode).
- The loop filter's edge, built a byte at a time on the caller's stack
  and read back wider, stalled each call on a failed store forward; it
  now comes in registers.
- Vertical edges are filtered down a superblock before across it, so an
  inner edge no longer reads back what the edge 4 to its left has just
  stored (the order along each line is libvpx's).
- 4:4:4 walks the loop filter's masks once for its three planes, and
  predicts the three planes of a block in one pass.
- Pictures still being read wait in the spare list instead of being
  freed, so threaded decoding no longer allocates and clears most of
  its pictures; and fewer large pictures are in flight (4:4:4 decodes
  8% faster in a quarter less memory).
- Earlier: the interpolation kernel with vector loads and stores and
  row pairs kept between output rows, the loop filter loading its outer
  taps only when the 15-tap filter can run, and the calling thread
  parsing a tile column itself.

The machine is a shared VM and single runs vary by ±10% (FFmpeg's own
times moved from 0.80 s to 0.98 s on the 4:4:4 stream within a day).
What is left on one core is small and spread out: the coefficient loop
runs 426 M instructions on the first 60 frames of the 4:2:0 stream
against FFmpeg's 394 M, the loop filter's kernel fewer than FFmpeg's but
with more time on its transposed loads, and the block parse around them.
Neither of the release build's other levers moves it. Built as one
codegen unit (as the H.264 crates are), the decoder is the same within
the noise. Built from a profile (as the H.264 and AV1 codecs are) trained
on synthetic video through our encoder and on libvpx streams of FFmpeg's
`mandelbrot` and `testsrc2` sources, 4:2:0 and 4:4:4, it is the same too
(1.012 against 1.012 of FFmpeg's time on 4:2:0, medians of fifteen
interleaved runs). Trained on the measured streams themselves, it was 5%
faster on 4:2:0, which a release profile cannot be trained on.
Nor does reconstructing each superblock as soon as it is parsed, while
its coefficients are in the caches, instead of a row at a time (FFmpeg's
order on one thread): every stream the same, and 1.033 against 1.025 of
FFmpeg's time on 4:2:0 (medians of fifteen interleaved runs).

**Encode** (`tools/encbench.sh`, BD-rate with `tools/bdrate.py`):
1280×720, 4 threads on both sides (libvpx with `-row-mt 1
-tile-columns 2`), bitrates from 0.3 to 8 Mb/s. BD-rate is the bitrate
change at equal PSNR-Y: negative is smaller files than libvpx-vp9. Times
are the mean per run.

A detailed picture panning 7 pixels a frame, with film grain, 60 frames
(times are the mean of the four runs; Cutlass at q 80 to 140, libvpx at
1 to 4 Mb/s). BD-rate on PSNR-Y and on the PSNR averaged over the planes
(FFmpeg's `psnr` "average"):

| Cutlass | libvpx-vp9 | BD-rate Y | BD-rate avg | Time (Cutlass / libvpx) |
| --- | --- | ---: | ---: | --- |
| speed 4 | good, cpu-used 0 | −24% | −1.6% | 4.33 s / 7.50 s |
| speed 6 | good, cpu-used 0 | −18% | −0.9% | 2.61 s / 7.50 s |
| speed 6 | good, cpu-used 1 | −24% | −16% | 2.61 s / 2.43 s |
| speed 6, lag 0 | good, cpu-used 1 | −16% | −3.6% | 2.15 s / 2.43 s |
| speed 6, lag 0 | good, cpu-used 2 | −19% | −1.2% | 2.15 s / 2.29 s |
| speed 7, lag 0 | good, cpu-used 3 | −27% | −4.8% | 1.51 s / 1.68 s |
| speed 7, lag 0 | good, cpu-used 5 | −32% | −13% | 1.51 s / 1.61 s |
| speed 9 | realtime, cpu-used 6 | −49% | −41% | 0.45 s / 0.51 s |
| speed 10 | realtime, cpu-used 8 | −68% | −62% | 0.38 s / 0.43 s |

All in one sitting, Cutlass at its defaults unless `lag 0` (16 pictures
of lookahead below speed 9, none at 9 and 10); speed 10's time is the
median of five runs interleaved with libvpx's (all five faster; the
sitting's single run was 0.46 s against 0.43 s). FFmpeg's `libvpx-vp9`
encodes at good, cpu-used 1 with 25 pictures of lookahead unless told
otherwise: speed 6 without lookahead is ahead of it on both measures in
less time, and every libvpx preset, from good cpu-used 0 to realtime 8,
has a Cutlass speed ahead of it on both measures in less time. Speed 6
with its lookahead is past libvpx's slowest, cpu-used 0, in a third of
its time. Without alt-refs, cpu-used 0 keeps more chroma (0.8 dB more U
and V PSNR at 13% fewer bits than speed 5's 42.96 dB Y at 1.3 Mb/s):
from frame 16 on, its temporally filtered alt-refs predict the
grain-free picture, which counts most in chroma. The lookahead costs
about a fifth more time at speed 6 (alt-ref frames are coded at a finer
quantizer, and at speed 6 at the fastest) and takes 10 to 11% off the
bits on the pan, 6 to 10% on the clean clip below. (The speed 9 and 10
BD-rates are on narrower ranges of PSNR, where the curves overlap.)

On a clean 720p clip (no grain, 60 frames) speed 7 is ahead of good,
cpu-used 1 on both measures (−15% on PSNR-Y, −3.2% averaged) in three
quarters of its time (1.27 s against 1.74 s), and −33% / −19% against
cpu-used 5 in its time; speed 6 is −9% / −11% against cpu-used 1 in its
time.

What moved the encoder most recently, at speed 6 unless said:

- Speeds 9 and 10 cap the transform at 8×8 and speed 8 at 16×16: chroma's
  transform follows luma's and nothing at these speeds tries a smaller
  one, so large blocks dropped chroma detail (23 to 25% fewer bits at
  the same averaged PSNR from speed 9, clean and grainy, and 0.7 to 7%
  at the same PSNR-Y, in the same time; −11 to −13% averaged at speed
  8). The lookahead is on by default below speed 9.

- The alt-ref's test of whether the interval holds together took a
  match only within 16 samples of where it was, so a steady pan never
  made one; it takes one anywhere in the search range now. Alt-refs are
  coded at speed 6 at the fastest (the one-pass path's short search
  found nothing a golden interval away and coded them as intra, larger
  than a key frame), and the temporal filter runs on threads. With lag
  16: −24% / −1.6% against cpu-used 0, from −15% / +8.4% without.
- Speed 10: each plane's columns go into the picture and through the
  loop filter on its own thread while the tiles' symbols are written
  (they were one after the other), and the diamond's four points are one
  SAD call (−7% of the time, the same streams).

- With the refinement pruned, speed 6 can afford speed 5's sub-sample
  scoring on the real filters and its check of every whole sample within
  two of the diamond's choice (4% fewer bits at the same averaged PSNR,
  in the same time); speed 5 takes speed 6's prunes and keeps its own
  transform and chroma searches (it was 1.75 times libvpx cpu-used 1's
  time, now its time, with 20% / 4.6% fewer bits).

- Candidates are kept for a trial only within a fifth of the best one's
  score (SAD and vector rate for inter, SATD for intra): −17% and −9% of
  the time for about 1% of the bits; intra is tried in inter frames past
  eight levels of error a sample (−5%).
- The sub-sample refinement tries one diagonal of four, as libvpx's
  pruned tree does (−11 to −13%, the same BD-rate); RD quantization walks
  only to the last coefficient of half a step and keeps smaller ones at
  zero (−9% of the instructions at high quality); a coded whole block
  under half the quantizer's noise ends the partition search (−6%).
- Chroma coefficients' bits weigh 1/0.7 of luma's: 3 to 4% fewer bits at
  the same averaged PSNR, 19 to 22% at the same PSNR-Y.

- Rate weighs twice what libvpx's `rdmult` gives it in the RD decisions:
  8 to 17% fewer bits at the same PSNR (luma, and averaged over the
  planes), in the same time.
- The search's trials take each coefficient's dead-zone level, costed
  from the frame's token costs (a table made once a frame, not once a
  transform block), and stop where the dead zone stops; only the chosen
  blocks weigh levels against each other. With the transposed and
  inverse scan tables and a loop of their own: 40% fewer instructions,
  the same BD-rate.
- RD quantization's 20 KB of scratch persists instead of being cleared
  per transform block, and the encoder's snapshots are saved in place
  instead of allocating three cleared buffers each (−16%).
- The forward DCT folds its rows (half the multiplies) and uses fused
  multiply-adds (FMA, checked with AVX2): −14% of the instructions.
- GOLDEN and ALTREF are not searched while they predicted under 1% of
  the last inter frame that searched them (−14% of the time, +0.5% of
  the bits).
- Speed 7 also tries its best inter candidate with the next smaller
  transform: 2.4% fewer bits in about the same time.
- Earlier, at speed 10: the dead-zone quantizer on AVX2
  (`cutlass_simd::vp9::quant`, −19% of the time on one thread) and a
  tile's superblock rows lending each other their scratch (−9%).

In bitrate mode (1–4 Mb/s, the same clip) speed 5 is −1.8% against good,
cpu-used 5 on PSNR-Y and higher on chroma.

The same picture nearly still, with stronger grain:

| Cutlass | libvpx-vp9 | BD-rate | Time (Cutlass / libvpx) |
| --- | --- | ---: | --- |
| speed 5 | good, cpu-used 5 | −35% | 1.78 s / 1.58 s |
| speed 5, lag 16 (alt-refs) | good, cpu-used 5 | −39% | 2.30 s / 1.58 s |

What moved the encoder (BD-rate on the panning clip, each step against
the one before): key and golden frames at 0.35 and 0.55 of the inter
quantizer step (from 0.45 and 0.7) −17% over 60 frames; sub-sample search
on the real 8-tap filters and an exhaustive ±2 whole-sample check −2.3%; RD partition and mode search and RD quantization
−28% (and −50% on the key frame's own size), key and golden frames
finer than inter frames with GOLDEN as a second reference −45% (from
+131% to +28% against libvpx good), the partition breakout −7% (and 2–3×
faster), coefficient probability updates −2.6%, rectangular partitions
−0.9% (speed < 4 only: they double the time), compound prediction with
the alt-ref −2.5% at speed 3 and −1.5% at speed 5 (on the nearly still
clip with `lag 16`), the non-coefficient probability updates −2% on the
first inter frame (within ±0.5% over 30 frames), inter blocks under 8×8
−0.3% at 352×288 (speed 1). Alt-refs gain on still
content (−1.8% overall, −16% at the highest rates) and are left out where
the picture moves too far for one picture to stand for the interval.

## Limits and gaps

- Not registered in the desktop app (the app has no VP9 opener yet). The
  crate registers on the video factories and offers `register_engine_encoders`
  for the engine's `EncodeFactory`.
- Interpolation filters switch per block (EIGHTTAP, SMOOTH, SHARP, by
  trial) at speeds 0–2; faster speeds keep EIGHTTAP.
- The encoder writes BT.601 studio range only, and no sRGB (GBR).
- Encoder tools, by speed: compound prediction (a past picture with the
  alt-ref ahead, chosen per block) at 0–6 with `lag`; inter blocks under
  8×8 (4×4, 8×4, 4×8, each sub-block NEAREST, NEAR, ZERO or NEW) at 0–1;
  rectangular partitions below 4; probability updates of the
  coefficients and of every other updatable probability (modes, filter,
  references, partition, tx size, skip, vectors) at all speeds;
  segmentation with adaptive quantization (`aq`, off by default).
  Segment features other than the quantizer, temporal segment maps and
  frame-parallel or error-resilient streams are not written.
- At equal time the encoder is within about 8% of libvpx's good presets
  on moving content (ahead with more time, and far ahead on still content
  and against the realtime presets), and its fastest speed takes libvpx
  realtime's time at equal bitrate with much better quality; tile columns
  limit how well 4 threads share a 720p frame. The decoder is level with FFmpeg on 4 threads on 4:2:0 and
  1.1–1.25× its time on one core. Its pictures go out without a copy, as
  shared views of the decoder's own planes at their padded stride
  (`Plane::samples` is a `Samples`, copied only on a first write;
  `Plane::packed` gives the visible rows); 10- and 12-bit samples are
  their little-endian bytes (copied into new buffers only on a
  big-endian target).

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7.
- libvpx was read where the specification and libvpx's streams disagree;
  no libvpx code is compiled into the crate.
