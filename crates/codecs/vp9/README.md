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
  (15 and up) an interval whose end the current picture reaches starts
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
| 4:2:0 | 1 core | 1.03 s | 0.97 s | 1.06 | |
| 4:2:0 | 4 | 0.42 s | 0.44 s | 0.96 | 54 MB / 57 MB |
| 4:4:4 | 1 core | 1.93 s | 1.80 s | 1.07 | |
| 4:4:4 | 4 | 0.81 s | 0.80 s | 1.01 | 78 MB / 78 MB |

On four threads Cutlass is level with FFmpeg on the 4:4:4 stream (it
won 6 of the 9 runs) and ahead on the 4:2:0 one (6 of 9). (The 4:4:4
stream started at 38.8 s on one core.) What moved the decoder most
recently:

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
What is left on one core is the coefficient parse (the bool decoder is
libvpx's and FFmpeg's, but FFmpeg's whole token loop takes about 0.43 s
of the 4:4:4 stream against Cutlass's 0.55 s) and the loop filter,
whose kernel retires fewer instructions than FFmpeg's but waits longer
on its transposed loads.

**Encode** (`tools/encbench.sh`, BD-rate with `tools/bdrate.py`): 30
frames of 1280×720, 4 threads on both sides (libvpx with `-row-mt 1
-tile-columns 2`), bitrates from 0.3 to 8 Mb/s. BD-rate is the bitrate
change at equal PSNR-Y: negative is smaller files than libvpx-vp9. Times
are the mean per run.

A detailed picture panning 7 pixels a frame, with film grain, 60 frames
(times are the mean of the four runs):

| Cutlass | libvpx-vp9 | BD-rate | Time (Cutlass / libvpx) |
| --- | --- | ---: | --- |
| speed 5 | good, cpu-used 1 | −4% | 5.4 s / 3.9 s |
| speed 5 | good, cpu-used 5 | −17% | 5.4 s / 2.4 s |
| speed 6 | good, cpu-used 5 | −7% | 4.3 s / 2.4 s |
| speed 7 | good, cpu-used 5 | +8% | 2.5 s / 2.4 s |
| speed 9 | realtime, cpu-used 6 | −33% | 0.70 s / 0.61 s |
| speed 9 | realtime, cpu-used 8 | −57% | 0.70 s / 0.52 s |
| speed 10 | realtime, cpu-used 8 | −62% to −69% | 0.46–0.50 s / 0.49–0.52 s |

At equal bitrate speed 10 now takes libvpx realtime's time (0.50 s against
0.49 s at 4.2 Mb/s, 0.49 s against 0.52 s at 2.2–2.5 Mb/s) at 2 to 4.7 dB
more PSNR-Y. What moved it: the dead-zone quantizer on AVX2
(`cutlass_simd::vp9::quant`, −19% of the time on one thread) and a tile's
superblock rows lending each other their scratch instead of making and
zeroing 240 KB each (−9%).

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
