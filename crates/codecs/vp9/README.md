# cutlass-vp9

A VP9 decoder and encoder in Rust. The decoder takes every profile (0–3,
8/10/12-bit, 4:2:0/4:2:2/4:4:0/4:4:4, sRGB) and matches FFmpeg's `vp9`
decoder sample for sample. The encoder writes profile 0 (8-bit 4:2:0) with
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

- Profile 0, any size up to 65536×65536 (odd sizes too), quantizer index 0
  (lossless, Walsh-Hadamard) to 255.
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
- Rate: a fixed quantizer (`with_quantizer`, or bitrate 0) or a bitrate,
  per frame type, from a model of bits against quantizer step.

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
  filter, generic over 8 and 16-bit samples; the 8-bit hot paths call the
  AVX2 kernels in `cutlass_simd::vp9` (`mc`, `lf`), and the transforms run
  in lanes compiled for AVX2 (`cutlass_simd::vp9::wide`).

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
  rows so they compile to `vpsadbw` under AVX2.

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
- The encoder writes profile 0 (8-bit 4:2:0) only, BT.601 studio range.
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
  1.1–1.25× its time on one core. Its 8-bit pictures go out without a
  copy, as shared views of the decoder's own planes at their padded
  stride (`Plane::samples` is a `Samples`, copied only on a first write;
  `Plane::packed` gives the visible rows); high-bit-depth pictures are
  still packed into new buffers.

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7.
- libvpx was read where the specification and libvpx's streams disagree;
  no libvpx code is compiled into the crate.
