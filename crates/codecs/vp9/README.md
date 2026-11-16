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
  one per core; `Vp9Options::tile_columns` sets them apart), up to the
  tile columns the width allows, and each tile column's superblock rows
  on two or more threads at once in a wavefront (libvpx's `row-mt`,
  `enc::tile::Lanes`): a row codes a superblock once the row above has
  finished the one above it and the one to the right of that, taking
  from them the contexts, motion, search seeds and last line of samples
  it reads; the rows are put together after, and the frames are byte for
  byte those one thread makes. At speed 7 and up the motion search runs
  first over the whole frame, its superblock rows shared evenly among
  the threads, and luma and chroma are filtered on separate threads.
  From speed 9 each tile column's rows are on one thread (a superblock
  is too little work for the wavefront), the loop filter follows the
  tiles row by row on threads of its own, and at speed 9 each tile
  column's thread makes its own part of the motion search as it goes,
  and its symbols are written on a thread of their own as its rows are
  coded (the probability updates decided once every tile column has
  coded three quarters of its rows); on one thread, each row's search
  comes just before the row.
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
  (`cutlass_simd::vp9::{mc16, lf16, sad16}` for 16 bits, `sad8` for 8).
  Distortions
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
  quantizer (`Vp9Options::cq_level`, libvpx's constrained quality), and
  `-crf` with `-b:v 0` a fixed quantizer, where
  `Vp9Encoder::qindex_of_level` turns libvpx's 0–63 levels into indices
  (up to speed 8 a fixed quantizer codes golden frames and alt-refs at
  half the inter step, and filters alt-refs at strength 1 over 5
  pictures).
  At a variable bitrate in one pass, golden frames, alt-refs and key
  frames take the inter step scaled by their type (as a fixed quantizer
  and two passes do), and the inter step is the one at which the frames
  to come spend the bitrate; the inter frames' model moves by a
  twentieth of each frame once settled; with a lookahead, its pictures
  are first coded fast at a fixed quantizer (`Core::probe`, a first
  pass over the lookahead), and their bits plan the first key frame
  (over 30 frames, at most 20 frames' worth) and start the model; what a
  key frame takes past its share falls due over the 30 frames after it.
  See "Against FFmpeg's libvpx-vp9" below.
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
  smaller transform, as speeds 0–5 do. Golden frames come sooner as the
  content moves faster: 64 samples of motion apart (the median vector of
  the sampled blocks of detail that moved, from each golden picture to
  the next one), 6 to 12 pictures; the first pass's bits are reclassed
  as each interval is chosen (a golden place's as its neighbours' times
  what the nearest first-pass golden frame took over its own). The
  alt-ref filter's search starts from the block to the left's match too,
  and a match still poor (64 of error a sample) is searched again on sums
  of 4×4 samples. The best four inter candidates are ranked again with
  their chroma's SAD added. Every shown picture also refreshes one of
  slots 3 to 7, so that they hold the last five, and each frame (alt-refs
  too) reads GOLDEN from whichever of them, or the golden frame, predicts
  it with far less chroma error at a whole-sample match: under 0.7 of the
  golden frame's at no more than 1.25 of its luma error
  (`arf::phase_error`, on a sample of 16×16 blocks, each searched from
  where the content has gone since that picture: `arf::shift`, the median
  vector between pictures, summed). The least luma error a sample it finds
  is the picture's noise, and a whole block that needs no residual and
  keeps under 1.2 times it ends the partition search. The alt-ref filter
  runs at strength 3. Against libvpx's two passes (good, cpu-used 2), at
  the same PSNR-Y / averaged PSNR, speed 6:

  | 720p | First pass at q 120 | Chroma ranked, golden by motion | Now |
  |---|---|---|---|
  | `mandelbrot`, the grainy pan, `testsrc2` (90 frames, 1–4 Mb/s) | +5.3% / +16.9% | −12.9% / +1.6% | −14.4% / −2.7% |
  | still picture, grainy pan, clean clip (90 frames, 0.5–3 Mb/s) | +2.9% / +11.8% | −20.0% / −7.8% | −26.5% / −21.2% |
  | the grainy pan alone (60 frames, 1–3 Mb/s) | +34% / +49% (1–4 Mb/s) | −10.6% / +7.9% | −27.1% / −25.2% |

  still under libvpx's time (four threads each, interleaved, the mean of
  three: 9.8 s against 9.1 s before and libvpx's 10.2 s on the first clip
  at 2 Mb/s; 8.4 s, 10.0 s and 14.0 s on the second; on the pan 4.3 s,
  4.2 s and 7.7 s at 1 Mb/s, 6.9 s, 5.6 s and 8.1 s at 2, 8.8 s, 9.6 s and
  9.6 s at 3), within 3.2% of the bitrate (3.8% under it at 0.5 Mb/s). On
  a pan by a sample and a half a picture (a fractal panned at 2560×1440
  and scaled down, with grain), 15.1% / 18.1% fewer bits than before at
  the same PSNR-Y / averaged PSNR, in a third less time, most of it the noise's
  end of the search. One pass was unchanged by that round, bit for bit;
  since, one pass keeps the reference pool too, up to speed 7 (see
  "Chroma" in "Against FFmpeg's libvpx-vp9"), and a cheaper one at speeds
  9 and 10 (see "The gaps" there). Since then too, the second pass codes
  a scene that predicts better than the key frame's at a finer step
  (`api::SCENE_BETA`), searches without the one-pass breakouts
  (`FrameParams::full_search`) and weighs chroma's bits 0.75 of luma's.

  Where the chroma gap came from: the grainy pan is of a fractal, flat
  colour gradients and an edge of fine colour detail, and its luma moves
  whole samples, 3 or 4 down and 7 or 8 across a picture, so the chroma
  is sampled afresh at every odd displacement and no sub-sample filter
  predicts its fine detail there. Blocks predicted at chroma half samples
  carried 90% of this encoder's chroma bits (libvpx's: 66%), and an
  alt-ref took 6 kB where its golden frame was an even displacement away
  against 25 kB where it was not. libvpx predicts 45% of the picture from
  vectors whole in chroma (from its alt-refs and golden frames, every
  four pictures), this encoder 28% before the pool. At 2 Mb/s and the
  same bits and PSNR-Y, its PSNR-U was 0.9 dB under libvpx's; with the
  pool, 0.6 dB over it, and at the quantizer the bitrate then takes (about
  40 for inter frames where it was 85) the partition search went to 8×8
  three times as often, grain being over the quantizer's noise
  everywhere, which the noise's end of the search takes back. Before the
  pool, the search decided little of it: a finer chroma quantizer (uv
  delta-q −4 to −12: at most 0.2 dB more PSNR-U at 1 Mb/s), a lower
  chroma RD weight (bits as luma's, or 0.6 of them: PSNR-U lower still,
  as the rate control coarsens everything), chroma kept out of the
  dead-zone trials, no early end of the partition search, every inter
  candidate tried, a 16×16 transform cap, smaller transforms tried
  everywhere, SHARP for the whole frame or per block (speed 2's
  switchable filters), 8 candidates ranked instead of 4, speed 4: none
  more than a point or two, or worse. Nor did candidates at the searched
  vector rounded to whole chroma samples (ranked with their chroma: the
  luma then misses by a sample), the alt-ref's chroma at its half samples
  (the mean of the two around, not the truncated whole sample), a weaker
  or stronger alt-ref filter or one over fewer pictures (2.6 points at
  best), finer alt-refs (0.47 of the inter step: 1.8 points), or a coarser
  key frame (0.6 of the inter step: +15% at the same PSNR-Y, 0.7: +10
  points averaged). Fixed golden intervals swing the pan by where its
  vectors fall against chroma's half samples (averaged, with the
  candidates ranked but the filter unseeded: +13% at 4 and 6, +0.4% at 7,
  +8.7% at 8, +9 to +10% at 9 and 10, +15% at 12; libvpx's 4 with its
  step ratios, +9 to +10%), and 8 for every scene lost 9 points of PSNR-Y
  on the second clip. With the pool, a coarser alt-ref (0.65 of the
  inter step) gains 0.8 points on the first clip and loses 2 on the pan;
  inter frames 0.85 of the step when GOLDEN reads the pool, a first k of
  0.35 or 0.7, the luma error relative to the golden frame's in the
  choice (chroma under 0.7 of its share): each lost on one clip or more.
  The motion is measured on blocks of detail only, with the coarse
  search: the filter's diamond, from zero, settles near its first step of
  16 on fine detail and anywhere on flat grain, and measured the pan at
  twice its motion. What is left on the first clip is its allocation:
  libvpx gives the pan in it 196 kB of the 770 at 2 Mb/s (inter frames at
  q 100 to 124, the fractal and `testsrc2` at 127 to 184), this encoder
  119 kB at one quantizer (about 130), PSNR-U 39.2 dB against 41.8 there
  and 0.7 to 1.5 dB over libvpx's PSNR-Y on the other two.
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
- Jumps: a picture that looks a cut but is the last one moved far (a
  camera that jumps 200 samples, past the searches' reach of a few dozen)
  is found on sums of 4×4 samples over a sample of 32×32 blocks of
  detail, refined to the sample, and taken when most of them agree
  (`arf::jump`): the frame is an inter frame whose blocks also start
  their search on LAST at that vector (`FrameEncoder::set_global`). On
  the clip of a fractal that jumps twice, the frame after each jump
  went from 40 kB to 0.9 kB, and the clip takes 30% fewer bits at a
  fixed quantizer (libvpx found the same vectors).
- libvpx-vp9's other options (see the table in "Against FFmpeg's
  libvpx-vp9"): colour space and range (`color_space`, `full_range`,
  sRGB in 4:4:4), render size, loop filter sharpness, error resilient
  and frame parallel frames, a static threshold, alt-refs off, the
  alt-ref filter's strength and pictures, undershoot and overshoot
  bands, `keyint_min`, constrained quality, a key frame size limit and
  the `vpcC` level, each tested.

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
  cost); inter blocks drop their residual when it does not pay. From speed
  5 a trial takes each coefficient's dead-zone level, counts its token bits
  as it walks the scan (the bits the trial is charged), places the end of
  block as it goes and stops where no end past it can cost less; its inter
  blocks are measured on their coefficients and not reconstructed.
- `enc::tx`: forward transforms, the exact inverses of the decoder's,
  measured from it.
- `enc::api`: the codec traits, rate control, frame types.
- `dsp::sad`: SAD, SSE and bilinear sub-sample SAD, written over fixed-width
  rows so they compile to `vpsadbw` under AVX2; the SADs and SSE are the
  kernels of `cutlass_simd::vp9::sad8` (8 bits) and `sad16` (10 and 12).

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
`vp9enc in.yuv 1280 720 out.ivf q=120 speed=5 threads=4`; its doc
comment lists the options, libvpx-vp9's among them).

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
threads; on one core (pinned with `taskset`), the medians of the 31
runs below.

| Stream | Threads | Cutlass | FFmpeg `vp9` | Ratio | Memory (Cutlass / FFmpeg) |
| --- | --- | ---: | ---: | ---: | --- |
| 4:2:0 | 1 core | 0.98 s | 1.03 s | 0.96 | |
| 4:2:0 | 4 | 0.41 s | 0.47 s | 0.86 | 58 MB / 57 MB |
| 4:4:4 | 1 core | 1.87 s | 2.04 s | 0.91 | |
| 4:4:4 | 4 | 0.70 s | 0.79 s | 0.88 | 80 MB / 78 MB |

On four threads Cutlass is ahead of FFmpeg on both streams (it won all 7
runs on each). On one core, in 31 runs alternating with FFmpeg's
(ratio of the medians, with a 95% bootstrap interval), on 15 November
2026 at commit `10ec7f50`:

| Stream, 1 core | Cutlass / FFmpeg | 95% interval | Cutlass faster |
| --- | ---: | --- | --- |
| 4:2:0 (`m420.ivf`) | 0.957 | 0.920–0.993 | 26 of 31 |
| 4:4:4 (`mandel.ivf`) | 0.914 | 0.869–0.952 | 27 of 31 |
| 4:2:0, 60 frames at 3.7 Mb/s (`q60.ivf`) | 0.900 | 0.832–0.983 | 24 of 31 |

Cutlass is faster than FFmpeg on all three streams on one core. A
second set of 31 runs gave 0.937 (0.906–0.982), 0.896 (0.876–0.983)
and 0.833 (0.781–0.903). The build before these changes, in the same
runs, was 1.011, 1.012 and 0.980 of FFmpeg's time. (The 4:4:4 stream
started at 38.8 s on one core.) What moved the one-core decoder last:

- Every plane started 16 bytes into a cache line: each was an
  allocation of its own, which the allocator starts 16 bytes into a
  page, and with strides of 64 samples every row of a superblock did
  the same. Half of the 32-byte loads and stores of motion
  compensation, the loop filter and the residual split across two
  lines, and the whole-sample copies of the 4:4:4 stream (60% of each
  of its first 40 pictures) took 1.6 times FFmpeg's time. The planes,
  the interpolation's rows between its passes and the transform's
  buffers start on a line now: 7 to 10% on 4:4:4, 3% on 4:2:0.
- The intrinsics `_mm256_packus_epi16` and `_mm_packus_epi16` are
  written over generic vector operations in this Rust (1.97), and LLVM
  keeps their clamp in front of the `vpackuswb` it emits: two `pmaxsw`
  and two `pminsw` (or two `pminuw`) more for every 32 bytes packed. The
  8-bit interpolation and loop filter pack with the one instruction
  (inline assembly on registers, `cutlass_simd::vp9::pack`): 10% fewer
  instructions in the interpolation, 2% in the filter.
- References repeat their last row into 16 rows below the picture, so
  the bottom superblock row of a 1080-line stream predicts from the
  reference in place instead of from a copy of its window with the edge
  repeated (1.3% fewer instructions on the 4:4:4 stream).
- Tile columns parsed one after another write the frame's motion map
  in place, instead of their own maps copied together at the end of the
  frame (0.9% of a 4:2:0 decode with four tile columns); the previous
  frame's map is reused rather than made and cleared again (a clone of
  it outlived the frame); without segmentation the segment maps are
  neither copied in nor out; the parse state's 13 KB of counts is boxed
  so that the state moves without them. 2% fewer instructions in all.
- The coefficient loop runs compiled with BMI2 and LZCNT
  (`cutlass_simd::vp9::wide`), for the bool decoder's shifts and
  leading-zero count; 1 to 2% in the shortest runs, within the noise of
  the medians.

Nothing got slower on four threads or at 10 bits: medians of 21 runs
on four cores alternating with the build before, 0.80 s against 0.82 s
for 4:4:4 and 0.47 s against 0.48 s for 4:2:0; and of 15 on one core
for a 10-bit 4:2:0 stream (1920×1080, 200 frames), 1.30 s against
1.34 s (FFmpeg 1.38 s).

Tried in this round and left out: software prefetch of the next
block's reference rows at 8 bits (the same, now that rows are aligned;
the hardware prefetchers keep up), transparent huge pages for the
pictures (glibc's `hugetlb` tunable: the same, so TLB misses are not
what the copies wait on), and the whole block parse compiled with BMI2
like the coefficient loop (the same within the noise; only what it
inlines gets the features).

What moved the decoder before that:

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
Against FFmpeg's profile on one core (its symbols from Ubuntu's
debuginfod), Cutlass now spends less time than FFmpeg in the
coefficient loop (24% of the 4:2:0 decode against 26%), the loop filter
(21% against 23%) and about the same in motion compensation (27%). It
spends more in the inverse transforms (3.4% against 1.5%; 5.6% against
2.1% on 4:4:4), which run in 32-bit lanes where FFmpeg's run in 16, and
in the block parse and its bookkeeping around them (about 25% against
21%). Neither of the release build's other levers moves it. Built as one
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
| speed 6 | good, cpu-used 0 | −25% | −2.7% | 1.37 s / 7.12 s |
| speed 6 | good, cpu-used 1 | −29% | −17% | 1.37 s / 2.61 s |
| speed 6, lag 0 | good, cpu-used 1 | −30% | −4.1% | 1.15 s / 2.61 s |
| speed 6, lag 0 | good, cpu-used 2 | −33% | −1.7% | 1.15 s / 2.08 s |
| speed 7, lag 0 | good, cpu-used 3 | −26% | −3.8% | 1.08 s / 1.61 s |
| speed 7, lag 0 | good, cpu-used 5 | −31% | −12% | 1.08 s / 1.74 s |
| speed 9 | realtime, cpu-used 6 | −49% | −41% | 0.45 s / 0.51 s |
| speed 10 | realtime, cpu-used 8 | −68% | −62% | 0.38 s / 0.43 s |

The speed 6 and 7 rows (and libvpx's cpu-used 0 to 5) are from 16
November 2026, at commit `e97ec739`; the others are from the sitting
before. With the build before that commit, speed 6 was −27% / +6.3%
against cpu-used 0 and −31.5% / −0.1% without lookahead against
cpu-used 2, in 1.58 and 1.27 s; speed 7 −26% / −3.6% and −31% / −12%, in
1.11 s.

The rest in one sitting, Cutlass at its defaults unless `lag 0` (16
pictures of lookahead below speed 9, none at 9 and 10); speed 10's time is the
median of five runs interleaved with libvpx's (all five faster; the
sitting's single run was 0.46 s against 0.43 s). FFmpeg's `libvpx-vp9`
encodes at good, cpu-used 1 with 25 pictures of lookahead unless told
otherwise: speed 6 without lookahead is ahead of it on both measures in
less time, and every libvpx preset, from good cpu-used 0 to realtime 8,
has a Cutlass speed ahead of it on both measures in less time. Speed 6
with its lookahead is past libvpx's slowest, cpu-used 0, in a fifth of
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

## Against FFmpeg's libvpx-vp9

FFmpeg's `libvpx-vp9` encoder as FFmpeg users run it, against this
encoder through `examples/vp9enc.rs`, on three 1280×720 4:2:0 clips at
30 fps:

- three scenes: FFmpeg's `mandelbrot`, a grainy pan of a fractal,
  `testsrc2`, with two cuts (90 frames, 1–4 Mb/s);
- a still scene, then a fractal panning, which jumps over 200 samples twice
  (90 frames, 0.5–3 Mb/s);
- the grainy pan alone (60 frames, 1–4 Mb/s).

FFmpeg 6.1.1 (Ubuntu's, libvpx 1.14): `ffmpeg -s 1280x720 -pix_fmt yuv420p
-r 30 -i clip.yuv -c:v libvpx-vp9 -b:v R -threads 4 [options] out.ivf`,
which is good deadline, cpu-used 1, 25 pictures of lookahead, one pass,
no alt-refs (libvpx makes them only in two passes), four tile columns.
Cutlass: `vp9enc clip.yuv 1280 720 out.ivf rate=R threads=4
[options]`, speed 5 and 16 pictures of lookahead by default. Every
stream is decoded by `ffmpeg -c:v vp9` and measured with FFmpeg's `psnr`
filter against the source; the averaged PSNR is (6 Y + U + V) / 8 of its
per-plane PSNRs. BD-rate is the bitrate change at the same PSNR,
piecewise linear in log rate over the range both curves cover
(`tools/bdrate.py`): negative is fewer bits than FFmpeg. Times are
whole-process wall clock, the mean over the four rates of the median of
three runs interleaved with FFmpeg's, on the shared 4-core VM (other
sessions loaded it throughout: single runs moved by ±10%).

At its defaults, on four threads (`-threads 4`, `threads=4`) and on one
(`-threads 1`, `threads=1`: libvpx keeps its four tile columns there,
this encoder codes one, which is why its BD-rate moves). BD-rate at the
same PSNR-Y, PSNR-U, PSNR-V, FFmpeg's `psnr` "average" (pixel-weighted,
(4 Y + U + V) / 6 in samples) and the (6 Y + U + V) / 8 average of the
per-plane PSNRs; time a stream, FFmpeg / Cutlass, and their ratio:

| Speed 5, 4 threads | Y | U | V | FFmpeg avg | 6:1:1 | Time | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| three scenes | −3.2% | −1.1% | −3.3% | −2.9% | −3.0% | 6.91 / 3.81 s | 0.55 |
| still scene, two jumps | −20.2% | −33.7% | −33.9% | −28.1% | −25.1% | 4.70 / 2.89 s | 0.62 |
| grainy pan | −31.5% | −50.5% | −50.6% | −43.5% | −39.3% | 2.92 / 2.37 s | 0.81 |

| Speed 5, 1 thread | Y | U | V | FFmpeg avg | 6:1:1 | Time | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| three scenes | −3.8% | −2.6% | −4.6% | −3.8% | −3.8% | 12.70 / 7.49 s | 0.59 |
| still scene, two jumps | −22.0% | −36.3% | −36.6% | −30.6% | −27.3% | 11.55 / 6.71 s | 0.58 |
| grainy pan | −32.9% | −54.9% | −55.2% | −46.9% | −42.1% | 7.27 / 5.93 s | 0.82 |

On 16 November 2026, at commit `cc2ed538` (times: medians of three runs
on four threads, two on one, interleaved with FFmpeg's and with the
build before). Before it, in the same runs and measured the same way:

| Speed 5, before | Y | U | V | FFmpeg avg | 6:1:1 | Ratio, 4 / 1 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| three scenes, 4 threads | −9.2% | +21.6% | +17.2% | +2.1% | −3.0% | 0.51 / 0.57 |
| still scene, 4 threads | −13.7% | −3.3% | −3.8% | −7.1% | −10.3% | 0.58 / 0.60 |
| grainy pan, 4 threads | −20.1% | −10.4% | −11.1% | −12.5% | −15.9% | 0.73 / 0.80 |
| three scenes, 1 thread | −10.1% | +19.9% | +15.0% | +0.9% | −4.0% | |
| still scene, 1 thread | −16.8% | −4.8% | −5.2% | −9.0% | −12.8% | |
| grainy pan, 1 thread | −19.6% | −21.0% | −21.6% | −20.5% | −20.5% | |

See "Chroma" below for what changed. The tables after it are from the
round before unless they say otherwise.

The round before, on 16 November 2026, at commit `e97ec739`: speed 5 at
−9.2% / −3.0%, −13.7% / −10.3% and −20.1% / −15.9% at the same PSNR-Y /
6:1:1 average on four threads in 0.51, 0.59 and 0.69 of FFmpeg's time,
and −10.1% / −4.0%, −16.8% / −12.8% and −19.6% / −20.5% on one in 0.56,
0.59 and 0.74 (6.10 / 3.13 s, 4.18 / 2.45 s, 2.78 / 1.93 s and 12.30 /
6.87 s, 10.94 / 6.43 s, 6.93 / 5.15 s a stream). Every rate was under FFmpeg's
time: the closest are the pan at 4 Mb/s on four threads (0.77) and at
2 Mb/s on one (0.78). The pan's rates, a stream (FFmpeg / Cutlass):

| Pan | 1 Mb/s | 2 Mb/s | 3 Mb/s | 4 Mb/s |
| --- | ---: | ---: | ---: | ---: |
| 4 threads | 2.35 / 1.25 s | 2.46 / 1.78 s | 3.15 / 2.23 s | 3.18 / 2.46 s |
| 1 thread | 5.21 / 3.34 s | 5.86 / 4.58 s | 7.60 / 5.81 s | 9.05 / 6.86 s |

Before the changes in "What moved" below: −8.6% / −1.8%, −12.0% / −5.6%
and −19.9% / −16.1% on four threads at 0.64, 0.75 and 0.98 of FFmpeg's
time, and −9.2% / −2.6%, −16.4% / −10.4% and −18.9% / −18.4% on one at
0.73, 0.81 and 1.20 (the pan at 3 Mb/s in 10.3 s against 7.8 s). In the
round before that: −14.0% / −6.6%, −6.3% / −4.5% and −16.2% / −15.5% at
1.04, 1.39 and 1.71 times FFmpeg's time on four threads.

**Chroma.** On the three scenes, at the same bits as FFmpeg's default,
this encoder's PSNR-U and V were 1.2 dB under libvpx's with PSNR-Y level
or over it: bits went to luma at chroma's expense. Taken apart by scene
(the three 30-picture scenes coded alone, BD-rate against FFmpeg's
default at the same PSNR-Y / U): `mandelbrot` −1.7% / +9.8%, the grainy
pan −14.3% / +3.5%, `testsrc2` +9.1% / +50.6%. In the clip, the pan got
half libvpx's bits (155 kB against 230 kB at 2 Mb/s) and its inter
frames' chroma swung between 35.6 and 40.9 dB picture to picture: its
content moves whole luma samples, so every other picture's chroma is
at a half sample from LAST and GOLDEN, and blurs (see "Where the chroma
gap came from" above). What moved (commit `cc2ed538`; speeds 9 and 10
and the first pass of two are the same streams):

- One pass keeps the reference pool of two passes up to speed 7
  (`api::POOL_SPEED`): GOLDEN reads whichever of the last five pictures,
  or the golden frame, predicts the chroma at whole samples. On its own
  at speed 5: −16 / −32 points of BD-rate at the same PSNR-Y / U on the
  pan, −16 / −19 on the still scene's clip, −1 / −5 on the three scenes.
  At a fixed quantizer it costs no time; at a bitrate the pictures it
  predicts better leave the bitrate finer quantizers, which on the pan
  at 3 and 4 Mb/s took 1.3 to 1.6 times the time on one thread.
- Up to speed 6 (`tile::CHROMA_SPEED`) chroma's coefficient bits weigh
  0.8 of luma's in RD quantization (`CHROMA_RATE`; 1/0.7 before, libvpx
  0.6 and 0.625 in its coefficient optimization), and chroma's error
  weighs twice luma's in the trials' decisions of mode, partition,
  transform size and skip (`CHROMA_DIST`). Before the pool, 0.8 traded 8
  points of PSNR-Y for 12 of PSNR-U on the three scenes, with FFmpeg's
  average where it was; with it, the trade on the three scenes is 7 for
  15, FFmpeg's average gains 1.5, and the other two clips give 5 to 10
  points of the pool's PSNR-Y margin for 1 to 9 of PSNR-U. The
  error weight adds 3 points at PSNR-U and V and 1 to 2.5 at FFmpeg's
  average for 0.6 of PSNR-Y on the three scenes.
- The partition search's stop at a whole block that codes a little
  (`CHEAP_RATE`, `CHEAP_DIST`) takes errors up to 0.64 of the step
  squared a sample (0.16) and 12/256 bits a sample (8): at the finer
  quantizers the pool brings, grain is over the step's noise and the
  stop never came. The pan's time on one thread went back from 1.04 to
  0.78–0.82 of FFmpeg's for under half a point of BD-rate (a point of
  PSNR-U on the three scenes from the rate's 12).

Speed 6 (four threads; before → now): three scenes −8.2 / +22.4 / +17.5
/ +3.0 / −2.0% → −1.5 / +0.2 / −2.1 / −1.3 / −1.4% (Y / U / V / FFmpeg
average / 6:1:1); still scene −12.6 / +0.4 / −0.1 / −4.2 / −8.3% →
−21.2 / −34.9 / −35.1 / −29.3 / −26.2%; pan −17.7 / −11.8 / −12.4 /
−13.3 / −15.4% → −30.7 / −49.1 / −49.2 / −42.3 / −38.3%; in 0.54, 0.59
and 0.78 of FFmpeg's time (0.51, 0.55 and 0.75 before). Speed 7 against
the same FFmpeg default: the still scene −5.2 / −6.1% at PSNR-Y / U
(+17.9 / +27.9% before), the pan −13.1 / −23.1% (−1.4 / +9.5%), the
three scenes still +23% / +79% (+25% / +94%). Speed 8 is changed only in
its alt-refs, coded at speed 6 (within 3 points, a little worse).

Two passes (`twopass=1` against `-pass 1/2`, four threads; before →
now): three scenes −11.2 / +28.6 / +21.0 / +3.7 / −2.4% → −4.7 / +9.6 /
+4.5 / −0.1 / −1.7%; still scene −43.1 / −26.7 / −26.8 / −35.1 /
−38.4% → −38.3 / −32.8 / −32.7 / −35.2 / −36.4%; pan −24.1 / −15.7 /
−15.4 / −19.7 / −21.1% → −15.9 / −25.5 / −24.9 / −21.0 / −19.8%, in
1.06, 1.07 and 0.97 of the time before (still about a third of
libvpx's). Two passes already had the pool; the weights alone trade
luma for chroma there, and on the pan they lose 8 points of PSNR-Y for
10 of PSNR-U. Other weights for the second pass (0.65 to 1.43 and 1 to
3) were no better at FFmpeg's average.

Realtime (speeds 9 and 10 against `-deadline realtime -cpu-used 8`, the
same streams as before): speed 10 −6.0 / +4.9 / +1.7 / −2.5 / −4.1% on
the three scenes, −66.7 / −52.8 / −53.4 / −61.3 / −64.8% on the still
scene, n/a / −56.6 / −57.5 / −67.8 / −74.3% on the pan; speed 9 −14.7 /
−2.9 / −5.3 / −10.6 / −12.5%, −68.9 / −55.0 / −55.7 / −63.5 / −67.0%,
n/a / −58.8 / −59.4 / −69.6 / −75.7%. Speed 10's chroma on the three
scenes is still behind; a lower chroma lambda in its dead-zone costs
(0.7 or 0.5 of luma's) moved 2.5 points of PSNR-U for 3 of PSNR-Y and
lost at FFmpeg's average; the pool changes nothing there (no lookahead)
and the error weight lost 4 points of PSNR-Y for 2 of PSNR-U, so they
are left as they were.

Tried and left out: a finer chroma quantizer in the frame header
(`uv_dc_delta_q`, `uv_ac_delta_q` −8 and −15 on `testsrc2`: PSNR-U 3 to
7 points worse at the same bits, as RD quantization then drops more),
every chroma intra mode tried (or the best by chroma SATD; in every
frame or key frames only: within 1.5 points), the chroma SAD weighing
two or three times luma's in the candidates' ranking (within a point,
3 worse on the pan at three), the loop filter level ±3 and ±6 (within 2
points, but the pan 4 points worse at PSNR-U at −3), golden frames and
alt-refs at 0.45 or 0.35 of the inter step (0.55; worse on two clips),
the probe's inter model scaled for the pool (rates 2 to 4% nearer on the
pan, 0.5 to 3.5 points worse on the three scenes), a key frame's spread of its
excess bits also for a cut (1 point better at FFmpeg's average on the
three scenes, but 3% over the rate), a floor on the inter quantizer at
the picture's noise (time, but 10 to 25% under the rate), and a stop
of the partition search at a coded block under the picture's noise (40
points worse).

**The gaps.** After "Chroma", speed 5 at a bitrate was ahead of
FFmpeg's default on every plane and average of every clip, but two
passes, speed 10, constant quality and speed 6 were still behind on the
three scenes in one column or two. In the round after (16 November
2026, commits `75836650`, `8e2e64ed` and `c0063234`), each against its
FFmpeg counterpart, four threads, BD-rate at the same PSNR-Y / PSNR-U /
PSNR-V / FFmpeg's average / 6:1:1 average, before → now. Speed 5 at a
bitrate writes the same streams as before (0.53, 0.61 and 0.72 of
FFmpeg's time, from 0.53, 0.62 and 0.73 in the same runs), and so does
speed 10 at a fixed quantizer, the first pass of two.

| Two passes, `-pass 1/2` | Y | U | V | FFmpeg avg | 6:1:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| three scenes, before | −4.7% | +9.6% | +4.5% | −0.1% | −1.7% |
| three scenes, now | −6.4% | −0.9% | −5.3% | −5.1% | −5.6% |
| still scene, before | −38.3% | −32.8% | −32.7% | −35.2% | −36.4% |
| still scene, now | −37.9% | −34.2% | −34.1% | −35.7% | −36.5% |
| grainy pan, before | −15.9% | −25.5% | −24.9% | −21.0% | −19.8% |
| grainy pan, now | −14.6% | −28.8% | −28.0% | −22.0% | −20.2% |

in 0.30, 0.32 and 0.38 of libvpx's two passes' time (0.27, 0.29 and
0.35 before; 6.5, 6.4 and 5.8 s a stream against 22.1, 19.7 and 15.5 s,
the mean over the rates of medians of three runs interleaved). What
moved it:

- A scene that starts at a cut and predicts better than the last key
  frame's scene is coded at a finer step (`api::SCENE_BETA`): the
  ratio of its first picture's first-pass bits to its inter frames'
  (over the 12 pictures after it) is taken against the key frame
  scene's, and the step scaled by that to the power −0.45, down to half
  (never coarser). The three scenes' grainy pan predicts nearly whole
  from the reference pool (a ratio of 15 against the fractal's 5.6) and
  its inter frames code under 1 kB at 2 Mb/s, so its quality is what its
  golden frames and alt-refs carry on: at one step for the clip it took
  120 kB of the 770 at 2 Mb/s with PSNR-U at 39.9 dB, where libvpx gives
  it 193 kB for 41.8 dB (its alt-refs every four pictures at q 55 to 84);
  at 0.65 of the step (with the rest of this round), 153 kB for 41.5
  dB, and `testsrc2` 275 kB where it took 309. The rate model weighs the
  scene's first-pass bits by the step's scale up to where the first
  pass shows the scene end (`Pass2::scene_end`). A clip without a cut
  is coded as before.
- The second pass leaves out the partition search's fast breakouts and
  measures inter trials by reconstructing them, not on their
  coefficients (`FrameParams::full_search`): on the three scenes (on
  two threads, with the step above) 1.7 points fewer bits at the same
  PSNR-Y and 3.6 to 5.2 at PSNR-U and V, in 1.1 to 1.25 times the time.
- Golden intervals by motion count a picture as moving only when one
  block in ten of its sample moved (`arf::MOVED_SHARE`): `testsrc2`'s
  few moving things (a twentieth of its blocks, 12 samples a picture)
  gave it golden frames every 6 pictures, alt-refs of 6 to 11 kB at
  1 Mb/s, where libvpx's are 16 apart and take 23 and 8 kB (on the three
  scenes, two threads, 1.3 points fewer bits at the same PSNR-Y, 0.7 and
  0.8 at PSNR-U and V).
- The second pass weighs chroma's bits 0.75 of luma's (0.8): PSNR-U on
  the three scenes from +0.04% to −0.9% for 0.4 points of PSNR-Y, and
  0.4 and 1.3 points of PSNR-Y for 1 to 3 at PSNR-U on the other two.

| Speed 10, realtime 8 | Y | U | V | FFmpeg avg | 6:1:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| three scenes, before | −6.0% | +4.9% | +1.7% | −2.5% | −4.1% |
| three scenes, now | −8.3% | −0.7% | −2.5% | −5.8% | −7.0% |
| still scene, before | −66.7% | −52.8% | −53.4% | −61.3% | −64.8% |
| still scene, now | −69.3% | −58.0% | −58.4% | −65.0% | −67.8% |
| grainy pan, before | n/a | −56.6% | −57.5% | −67.8% | −74.3% |
| grainy pan, now | n/a | −60.7% | −61.3% | −70.3% | n/a |

| Speed 9, realtime 8 | Y | U | V | FFmpeg avg | 6:1:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| three scenes, before | −14.7% | −2.9% | −5.3% | −10.6% | −12.5% |
| three scenes, now | −17.0% | −9.4% | −10.7% | −14.4% | −15.6% |
| still scene, before | −68.9% | −55.0% | −55.7% | −63.5% | −67.0% |
| still scene, now | −72.0% | −59.6% | −60.0% | −67.2% | −70.4% |
| grainy pan, before | n/a | −58.8% | −59.4% | −69.6% | −75.7% |
| grainy pan, now | n/a | −61.2% | −61.7% | −71.2% | −76.9% |

(n/a: the curves do not meet.) Speeds 9 and 10 at a bitrate keep a
reference pool too (`api::RT_POOL_SPEED`), a cheap one: every shown
picture refreshes one of slots 3 to 5, and GOLDEN reads the last of
them (or the golden frame) that stands an even number of luma samples
from where the content is now, so that chroma predicts from it at whole
samples; its blocks try the vector of that displacement (the vector
from neither NEAREST nor NEAR: at these speeds GOLDEN is not searched).
Where the content stands is the sum of each picture's motion, found on
luma profiles (`arf::Profile`: the differences from the sample above
summed along each row, and from the one to the left down each column,
the offsets of least difference within 32 samples): with the extra
GOLDEN candidate, 2.3% more instructions at speed 10 on the still
scene's clip. On the three scenes at 2 Mb/s the pan's PSNR-U went from
34.9 to 35.4 dB at the same bits.
Time against FFmpeg's realtime 8: on one thread, paired runs pinned to
their own cores (medians of five), 0.89, 0.90 and 0.88 (0.89, 0.84 and
0.90 before); on four threads, interleaved (eleven runs), 1.00, 1.02
and 0.90 (0.97, 1.00 and 0.86 before in the same runs, where the
previous round measured 0.91, 0.91 and 0.82: the machine's load moves
these by several points, and in this sitting FFmpeg ran relatively
faster). So speed 10 is over 0.95 of FFmpeg's time on four threads on
two clips here, 3 to 4% over the build before. Speed 9: 0.95, 1.02 and 0.86 on four threads
(0.94, 1.01 and 0.82 before).

| Constant quality, `-crf` | Y | U | V | FFmpeg avg | 6:1:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| three scenes, before | +0.2% | −10.2% | −10.1% | −4.1% | −2.5% |
| three scenes, now | −4.3% | −15.6% | −16.6% | −9.4% | −7.5% |
| still scene, before | −0.9% | −33.3% | −33.3% | −23.1% | −15.2% |
| still scene, now | −5.5% | −41.3% | −41.2% | −29.9% | −21.3% |
| grainy pan, before | −21.1% | −39.7% | −39.7% | −32.5% | −27.9% |
| grainy pan, now | −25.9% | −45.9% | −45.9% | −38.4% | −33.4% |

A fixed quantizer at speed 5 searches as the second pass does
(`api::FULL_SEARCH_SPEED`), in 1.22, 1.39 and 1.38 times the time:
0.55, 0.82 and 0.76 of libvpx's (3.96, 3.21 and 2.21 s a stream against
7.23, 3.93 and 2.90 s; 0.45, 0.59 and 0.55 before; the mean over each
curve's points of medians of three runs). With every one-pass prune
left out too (the second pass's whole search), the same within a point
in 1.2 times that time.

| Speed 6, FFmpeg's default | Y | U | V | FFmpeg avg | 6:1:1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| three scenes, before | −1.5% | +0.2% | −2.1% | −1.3% | −1.4% |
| three scenes, now | −1.0% | −1.4% | −3.3% | −1.5% | −1.3% |
| still scene, before | −21.2% | −34.9% | −35.1% | −29.3% | −26.2% |
| still scene, now | −20.2% | −36.1% | −36.3% | −29.7% | −26.1% |
| grainy pan, before | −30.7% | −49.1% | −49.2% | −42.3% | −38.3% |
| grainy pan, now | −29.8% | −50.3% | −50.4% | −42.7% | −38.2% |

Speed 6 in one pass weighs chroma's bits 0.75 of luma's
(`tile::CHROMA_RATE_SIX`; 0.7: −0.3 / −1.4% at PSNR-Y / U on the three
scenes), in 1.00, 0.93 and 0.99 of speed 5's time (0.53, 0.57 and 0.71
of FFmpeg's): on the three scenes speed 6 is no faster than speed 5 in
these runs, nor was it before (0.99). Speed 7's alt-refs, coded at
speed 6, take it too: on the three clips 0.2 to 2.6 points fewer bits at
every column. Speed 8's alt-refs weigh chroma as speed 8 does and stop
the partition search at a cheap block at the thresholds of before the
pool (`FrameParams::fast_arf`): speed 8 writes the streams it wrote
before "Chroma", which had cost it 1 to 3 points at PSNR-Y.

Tried and left out in this round. Two passes: finer golden frames and
alt-refs only in groups that predict well (libvpx's boost: to half the
step, PSNR-U on the three scenes +9.6% to +8.5% but the still scene's
from −32.8% to −25.7%); golden intervals of 32 samples of motion (4
pictures on the pan, as libvpx: worse on all three clips); the scene's
step by a window of 12 pictures around each picture instead of a scene's
(clips without cuts lost 3 to 7 points of PSNR-Y: the window's ratio
wanders); chroma's error weighed 3 and its bits 0.6 in the second pass
(the three scenes' PSNR-U −0.6%, but PSNR-Y −0.5% there and 2 to 3
points elsewhere); the inter candidates' margin left out too (PSNR-U
−0.6%, the still scene's PSNR-Y 1.2 points worse); six candidates
ranked with their chroma (−0.3%); speed 4's search in the second pass
(the three scenes at −6.4 / +5.7 / −0.1%, every clip better, but 1.45
to 1.75 times the time: the pan at 0.53 to 0.69 of libvpx's). Speed 10:
the median of the coded vectors on LAST as the motion (they run short,
6 samples where the content moves 7 or 8, and the parity is lost),
nine vectors around the last motion voted on by blocks of detail
(content moving 7.5 samples a picture splits the vote), and
`arf::shift` itself on blocks 64 apart (the same streams as the
profiles, 5 to 9% more time); the two best candidates ranked with their
chroma (the fractal 1 dB under at PSNR-Y, the pan no better). Speed 6:
chroma's error weighed 2.5 (PSNR-U −0.3% only). Speed 8: the old stop
with speed 6's chroma weights (0.3 to 1.4 points better at PSNR-U, 0.8
worse at PSNR-Y on the three scenes).

Still behind: `testsrc2` alone, where libvpx is ahead at every speed
here on PSNR-Y too (at fixed quantizers against `-crf`: +9% / +33% at
PSNR-Y / U at speed 5, −2.8% / +18% at speed 1); its key frame takes 9%
more bits than libvpx's at the same PSNR-Y with 1.2 dB less PSNR-U.
Rate: the grainy pan predicts nearly whole from the pool (inter frames
of 1 to 2 kB at 2 Mb/s, PSNR-Y level at 44.3 dB from 2.7 to 3.9 Mb/s),
and one pass comes in 12%, 16%, 9% and 3% under 1, 2, 3 and 4 Mb/s
there (6%, 2% under and 0.3%, 1.5% over before; FFmpeg 14% and 1% over
and 12% and 18% under); the other two clips are where they were (the
three scenes 6% over at 1 Mb/s, the still scene within 5%). At a
constant bitrate (2 Mb/s, `cbr=1000`, speed 5) the pan comes in 6.4%
over (2.6% before), the other two within 1.6% and 3.8% as before. The pan on
four threads is at 0.81 of FFmpeg's time (0.73 before), the rest at
0.55 to 0.62.

With `-row-mt 1` (in an earlier sitting `-row-mt 1 -tile-columns 2
-frame-parallel 1` wrote the same streams), and speed 6 against it and
against the defaults (BD-rate; FFmpeg's time a stream with `-row-mt 1`,
and Cutlass's at speeds 5 and 6):

| Clip | Speed 5 / `-row-mt 1` | Speed 6 / `-row-mt 1` | Speed 6 / defaults | Time (FFmpeg / 5 / 6) | Speed 6 before / defaults |
| --- | ---: | ---: | ---: | ---: | ---: |
| three scenes | −8.9% / −2.8% | −7.9% / −1.9% | −8.2% / −2.0% | 6.15 / 3.07 / 2.96 s | +2.6% / +8.6% |
| still scene, two jumps | −13.0% / −10.9% | −11.8% / −8.9% | −12.6% / −8.3% | 4.08 / 2.42 / 2.29 s | −8.3% / −6.3% |
| grainy pan | −19.4% / −14.9% | −17.0% / −14.5% | −17.7% / −15.4% | 2.82 / 2.08 / 1.85 s | −14.2% / −12.6% |

Speed 6 is now ahead of FFmpeg's defaults on both measures on all three
clips, in 0.96, 0.95 and 0.89 of speed 5's time (FFmpeg's time here is
at its defaults; on 16 November 2026, at commit `53c8a338`, medians of
three runs of each, in turn, four threads). Before, it took speed 5's
time on the three scenes (1.01 of it in the same runs; 0.98 and 0.94 on
the other two) and was behind libvpx's cpu-used 1 there: its chroma was
under libvpx's at coarse quantizers. Most of that was the next smaller
transform, which speed 5 tries and speed 6 did not; speed 6 now
searches as speed 5 does but with one inter candidate at every size and
the smaller transform only after a best mode that codes over 32 bits an
8×8 (`SIX_TX_BITS`). Speeds 7 and 8 change only in their alt-refs,
coded at speed 6 (on the three clips 0.2 to 8 points fewer bits at the same
PSNR-Y, and as many or fewer at the averaged PSNR but for speed 8 on the
pan, 1.1 points more).

Two passes (`-pass 1` then `-pass 2`, cpu-used 1 both) against
`twopass=1` (a speed-10 first pass, then speed 5), one run each. The
second pass leaves out the one-pass prunes (its prune of the smaller
transform cost 4 points on the three scenes there, the cheap-block stop
and one candidate below a point or two): its streams are the same as
before the changes below, and the times of the round before are in the
last column.

| Clip | `-pass 1/2` | Time (FFmpeg / Cutlass) | Time before |
| --- | ---: | ---: | ---: |
| three scenes | −11.2% / −2.4% | 18.38 / 4.94 s | 21.65 / 6.09 s |
| still scene, two jumps | −43.1% / −38.4% | 17.15 / 4.70 s | 20.90 / 5.59 s |
| grainy pan | −24.1% / −21.1% | 12.97 / 4.51 s | 14.57 / 5.19 s |

Against the recipe of Google's VOD guide (`-row-mt 1 -tile-columns 2`,
`-speed 4` then `-speed 2`): −16.7% / −7.9%, −42.0% / −38.5% and
−25.3% / −22.5%, in 4.94, 4.70 and 4.51 s against libvpx's 10.22, 11.66
and 8.48 s.

Realtime (`-deadline realtime -cpu-used 8`, `-threads 4` and
`-threads 1`) against speeds 9 and 10 (no lookahead), BD-rate and time a
stream (FFmpeg / speed 10 / speed 9, the sum over the four rates of
medians of five runs, each encoder in turn, over four), and the ratios
of speed 10's and speed 9's times to FFmpeg's:

| Clip | Speed 10 | Speed 9 | 4 threads | Ratio | 1 thread | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| three scenes | −6.0% / −4.1% | −14.7% / −12.5% | 0.57 / 0.52 / 0.51 s | 0.91 / 0.90 | 0.96 / 0.82 / 0.90 s | 0.86 / 0.94 |
| still scene, two jumps | −66.7% / −64.8% | −68.9% / −67.0% | 0.53 / 0.48 / 0.48 s | 0.91 / 0.91 | 0.96 / 0.85 / 0.85 s | 0.89 / 0.88 |
| grainy pan | n/a / −74.3% | n/a / −75.7% | 0.44 / 0.36 / 0.34 s | 0.82 / 0.77 | 0.79 / 0.66 / 0.66 s | 0.83 / 0.83 |

On 16 November 2026, at commit `53c8a338`; the BD-rates are on four
threads (on one: −6.1% / −4.1% and −15.1% / −13.0% on the three scenes,
−67.2% / −65.4% and −69.4% / −67.6% on the still scene, speed 9 n/a /
−76.0% on the pan). Speed 9 at commit `28ea1fda`, in the same runs:
1.10, 1.00 and 0.90 of FFmpeg's time on four threads, 1.05, 0.96 and
0.88 on one, at −14.3% / −12.2%, −69.0% / −67.1% and n/a on four
threads. Speed 9 now comes in within 4% of the rate (on the three
scenes 1.04, 2.09, 3.11 and 4.12 Mb/s; it took 3.24 and 4.27 at 3 and 4
Mb/s), FFmpeg 1.23, 2.21, 2.96 and 3.72 Mb/s. In the round before that
(the same BD-rates, but speed 9's −15.2% / −13.1% on the three scenes),
speed 10 took 1.10, 1.13 and 0.91 of FFmpeg's time on four threads and
1.09, 1.09 and 0.99 on one; speed 9 1.31, 1.21 and 0.98, and 1.33, 1.25
and 1.18. On the pan the curves of PSNR-Y do not meet: speed 9 at 0.98
Mb/s (41.1 dB PSNR-Y) is past libvpx's realtime 8 at 4.3 Mb/s (40.6 dB).
On four threads FFmpeg's time hardly grows with the rate (0.51 to 0.57 s
a stream on the three scenes from 1 to 4 Mb/s) and speed 9's does (0.44
to 0.58 s): it is furthest from FFmpeg's time at 1 Mb/s and closest at
4.

What moved speed 9 in this round (commit `a771f1d4`; speed 10, speeds 0
to 8 and two passes are the same streams):

- One pass at a variable bitrate budgeted the golden frames to come at
  an alt-ref's step, but without lookahead a golden frame is no alt-ref
  and takes a key frame's: one every 12 frames took 1.8 times its share,
  and the debt it left stood (the inter frames' payback balanced it), so
  that every clip came in 6 to 9% over the rate. Budgeted at the key
  frame's step, within 4%, and 1 point fewer bits at the same PSNR on
  the three scenes (`GOLDEN_KEY_SPEED`). 2.5% of the time on one thread.
- The trials of intra against inter (14% of the blocks, 2% of the area
  on the three scenes, but the detailed blocks: the token walk of their
  trials was a tenth of the time) cost each tx block's tokens by the
  dead zone's bit model, at 0.7 of a bit a model bit (`DZ_TRIAL_RATE`):
  7% of the time for 0.3 points. The winner is written as its trial
  coded it (the trial reconstructs as coding does from speed 9) instead
  of being coded again.
- A 16×16 block whose best starting vector is under 4 of SAD a sample
  skips its diamond (`STILL_SAD`): 3% for 0.3 points.
- On threads, the probability updates are decided on the counts of the
  first three quarters of the superblock rows of every tile column
  (`EARLY_SPEED`), and each tile column's symbols are written on a
  thread of its own as its rows are coded, instead of after the frame
  (an eighth of the time at 4 Mb/s on four threads; on 4:2:0 720p each
  tile column's coder writes about 8 ns a symbol, 17 million symbols at
  4 Mb/s on the three scenes). The same BD-rate within 0.1 point (on
  half the rows, 1.5 points worse).
- Tried and left out: the symbol writer without its branch on the bit
  (slower: the branch predicts well, the select does not break the
  chain), with `leading_zeros` for its normalization table (the same);
  the motion searches of each tile column on threads of their own ahead
  of its coding (the same streams, no faster: four threads are busy
  already); two lanes a tile column (5% slower); four tile rows for the
  writers (the same); the updates decided on the frame before's counts
  (8 points worse: its counts are already in the probabilities by
  adaptation) or none on inter frames (7 points); a shorter payback of
  the rate's debt (12 frames: 7 points worse); intra not tried where its
  SATD is over 1.5 times the inter SAD (0.9 points); the token model in
  every trial at speed 6 (9 points).

Constant quality (`-crf 24, 32, 40, 48 -b:v 0`) against a fixed
quantizer (`q=60` to `210` in steps of 30, speed 5; times are the mean
over each curve's points, medians of three runs), at commit `ca9c61bf`,
and as measured before it and three rounds before:

| Clip | BD-rate | Time (FFmpeg / Cutlass) | Before | Three rounds before |
| --- | ---: | ---: | ---: | ---: |
| three scenes | −5.3% / −1.4% | 6.45 / 2.53 s | −4.5% / −0.3% | −7.3% / −2.9%, 6.73 / 6.11 s |
| still scene, two jumps | −4.4% / −4.1% | 3.57 / 1.75 s | +1.3% / +1.3% | +10.0% / +4.9%, 3.93 / 4.76 s |
| grainy pan | −13.9% / −10.8% | 2.44 / 1.22 s | −8.4% / −4.9% | −1.3% / −0.3%, 2.64 / 3.09 s |

Sampled at `q=50` to `225` instead (seven points), the still scene is at
0.0% / −0.1% (+3.8% / +4.1% before). There the gap was at the low rates
(0.3 to 0.6 Mb/s), and is still (see "What is left").

After "Chroma" (commit `cc2ed538`; Y / U / V / FFmpeg average / 6:1:1,
four threads): three scenes +0.2 / −10.2 / −10.1 / −4.1 / −2.5% (−5.3 /
+11.2 / +9.9 / +1.7 / −1.4% before), still scene −0.9 / −33.3 / −33.3 /
−23.1 / −15.2% (−4.4 / −3.5 / −3.7 / −3.4 / −4.1%), pan −21.1 / −39.7 /
−39.7 / −32.5 / −27.9% (−13.9 / −5.0 / −5.6 / −7.5 / −10.8%). At a fixed
quantizer the chroma weights move bits from luma to chroma with no rate
control to give them back: PSNR-Y on the three scenes is now level with
libvpx's (+0.2%), behind by a hair.

What moved in this round (16 November 2026, commits `84916b5c` and
`ca9c61bf`): the realtime speeds under FFmpeg's realtime 8 time, and
constant quality ahead on every clip. Speed 10 is bit for bit the same
streams, and so are speeds 0 to 8 at a bitrate, at a constant bitrate
and in two passes; speed 9 and the fixed quantizer at speeds 0 to 8
change.

- A DCT block whose residual bounds every coefficient inside the dead
  zone is not transformed (speeds 9 and 10, where every block is
  quantized by the dead zone): the DC by the residual's sum through the
  DC's gain, any AC coefficient by the residual's variance through the
  transform's gain, with margins for the transform's roundings
  (`dz_bounded`, tested against the transform and the dead zone over the
  quantizer's range). On the grainy pan at 2 Mb/s that is two thirds of
  the tx blocks. A tx block's residual, its squares and its sum are one
  AVX2 pass.
- From speed 9 a tile column's rows are coded on one thread: a
  superblock is too little work for the wavefront's waits (on four
  threads at speed 10 the tiles took 0.7 of the time against two lanes).
  With one lane a tile column, the loop filter follows the tiles row by
  row on threads of its own (`lf_rows`), each superblock row as soon as
  every tile column has sent it, instead of after the frame: on four
  threads it was a tenth of speed 10's time.
- On one thread, each superblock row's fast search is made just before
  the row is coded (the same search, in the same order, its samples
  still in the cache), and the searches' buffers are kept from frame to
  frame (allocated every frame, their page faults were a sixth of speed
  9's tiles on four threads). Its candidates take each vector's SAD once,
  and a candidate at whole samples a plain SAD.
- The tiles' symbols are written with the coder's state in registers and
  their probabilities looked up by id from one table (still a twelfth of
  speed 9's time at 4 Mb/s: one coder's symbols go one after another).
- The jump search (`arf::jump`) measures a row of vectors at once, sum
  by sum across the row's offsets in vector lanes, and its blocks on the
  encoder's threads: 50 to 17 ms on the still scene on one thread.
- Speed 9's 8×8 blocks check only their candidate vectors, as speed 10's
  smaller blocks do (their diamonds were a tenth of its time, for 0.6
  points at the same PSNR), and on threads each tile column's thread
  makes its own columns' part of the fast search as it codes them, with
  its own copy of the seeds (those across its left edge are the last
  frame's). Speed 10 keeps the frame's search on threads of its own: it
  is also the first pass of two passes and the probe of one, whose bits
  the other speeds' rate control reads.
- A fixed quantizer up to speed 8 codes its golden frames and alt-refs
  at half the inter step (0.55 before), and filters its alt-refs at
  strength 1 over 5 pictures (5, 4 or 3 by the quantizer, over 7): the
  golden step alone took the still scene to +0.1% / −0.3%, the weaker
  filter alone to −0.2% / +0.4%. Speeds 6 and 8 at a fixed quantizer take
  fewer bits on all three clips too (speed 6 on the still scene from
  +10.6% / +7.8% to +1.9% / +0.2% against FFmpeg's cpu-used 1).
- Tried and left out. Realtime: intra without its trial at speed 9 when
  its SATD is a tenth of the inter SAD (it won 89% of those trials; 1.7
  points on the three scenes, no faster in its runs), no diamond at speed
  9's 16×16 blocks either (2.4 points, no faster in its runs), the
  scene-cut test on threads (slower: it is a fifth of a millisecond a
  frame), a branchless bool coder (as fast: the coder's chain of range,
  split and shift is the time). Constant
  quality: a finer key frame (a quarter or a fifth of the inter step: 3
  to 20 points worse on the still scene), coarser alt-refs (worse on all
  three), the loop filter weaker on inter frames (better on the pan,
  worse on the still scene), no alt-refs (the still scene at −2.6% /
  −0.4%, but the three scenes +4.5% averaged), and libvpx's quantizers
  by position (192, 184, 173, 154 at `-crf 48`: 10 points worse).

The round before that: speed 5 in 0.67 to 0.74 of its time on four
threads (3.13, 2.45 and 1.93 s a stream, from 4.24, 3.32 and 2.89 s) and
0.58 to 0.74 on one (6.87, 6.43 and 5.15 s, from 9.24, 9.26 and 8.85 s),
with as many bits at the same PSNR or fewer:

- Bit for bit the same streams, 7% of the time on the pan at 3 Mb/s: the
  RD walk's first pass over a transform block (each coefficient's size
  in steps, their squares' sum, the scan's last that could take a level)
  was written for the compiler to vectorize and ran scalar, as did a
  trial's second pass in scan order for its dead-zone levels, context
  energies and gains (together a tenth of the encoder's time). They are
  one AVX2 pass in natural order now (`cutlass_simd::vp9::quant::scan`),
  the levels and energies kept by natural position for the walk.
- A whole block that codes a residual in under 8/256 bits a sample, mode
  info and all, with an error under 0.16 of the step squared a sample,
  ends the partition search (speeds 5 and 6, one pass). On the pan at
  3 Mb/s the 64×64 block stayed whole in 72% of the superblocks, three
  quarters of those whose split was tried came out within 2% of not
  splitting either way, and the split's trials were three quarters of
  the search: a block coding a few levels with the largest transform
  splits into quarters that code the same again, each with its own mode
  info. 0.81 of the time on the pan at 4 Mb/s, one thread.
- One pass ranks its best inter candidates again with their chroma's SAD
  added (speeds 5 and 6, as the second pass did), and at speed 5 blocks
  of 32×32 and up then try only the first: the second, within a tenth of
  the first's luma score, won 40% of the 64×64 trials, mostly for its
  chroma (without the ranking, one candidate cost 8 to 12 points at the
  averaged PSNR; with it, the same within the noise). −9% of the time.
- At speed 5, an inter frame's 64×64 block does not try the next smaller
  transform (−8%), and inter frames try one intra mode (two; −4%).
- Speed 6 shares the second and third (0.8 to 0.94 of its time): at
  fixed quantizers on the pan it is ahead of libvpx's cpu-used 0 again
  with its lookahead (−24.8% / −2.7%; the round before's prunes had left
  it at −26.7% / +6.3%), and of cpu-used 2 without (−32.6% / −1.7%, from
  −31.5% / −0.1%); at a bitrate it trades 4 points at the same PSNR-Y on
  the pan for 1.5 at the averaged PSNR, and is within 1.3 points on the
  other two clips. Speeds 7 and 8 change only in their alt-refs (coded
  at speed 6): within 0.3 points.
- Left out: one inter candidate at speed 6 too (2 points of chroma
  against cpu-used 2); the speed-5 prunes in the second pass (0.6 to 1.7
  points there, which is in a quarter of libvpx's time already); and,
  from the trials' costs, a split given up when its first quarters,
  extrapolated, cost more than the whole block (on the three scenes it
  lost several percent of the trials' cost for any real saving), and
  stops by rate alone (they also caught blocks that skip, whose splits
  win by 20% and more).

Two rounds before, speed 5 in 0.53 to 0.63 of its time on four threads
(4.24, 3.32 and 2.89 s a stream, from 6.76, 6.27 and 5.02 s), each step
measured on the three clips at speed 5:

- Bit for bit the same streams, about a fifth of the time: 8-bit SADs and
  SSE on AVX2 (`cutlass_simd::vp9::sad8`: 32 samples a `vpsadbw`, the sums
  kept in lanes to the end of the block; they were 15% of the time on the
  pan, 6% after); a trial's token bits taken from the RD walk that chose
  its levels rather than walked again in `write_coefs`; that walk stopped
  where no end of block past it can cost less (on the still picture's
  32×32 blocks the dead zone's last level is at 884 of 1024 coefficients
  on average, the end of block at 44, and the walk now ends at 146), and
  compiled for AVX2; a block's candidates not putting back the samples the
  one before left; the 5×5 whole-sample refinement four SADs at a time.
- Inter trials measured on their coefficients and not reconstructed
  (libvpx's transform-domain distortion): as many bits, a few percent less
  time.
- The partition search ends at a 16×16 or 32×32 block that needs no
  residual (its quarters won 4% of the time on the pan at 3 Mb/s and 1 to
  3% on the still picture, and were half the pan's trials); at a block
  that needs none and keeps LAST where it is under an eighth of the
  step squared a sample; at one that needs none under a sixteenth (a 24th
  before). On the still scene at 2 Mb/s, 40% fewer trials.
- No smaller transform after an inter frame's best that codes no
  residual, at one pass (three fifths of those trials on the pan, a tenth
  of the time; with the stops above, 3 to 4 points more bits at the same
  PSNR-Y and 2 to 3 averaged on the first two clips, under 1 on the pan:
  what is left of their margin is the price of the pan's time); the
  second inter candidate only within a tenth of the first's score (a
  fifth); in inter frames two intra modes (three), tried where the best
  inter error is over seven a sample (five).
- Tried and left out: bilinear sub-sample refinement instead of the 8-tap
  filters (1 to 2 points on all three), the search skipped on a reference
  whose best start is twice another's (libvpx's adaptive motion search: 2.6
  points on the three scenes), a 3×3 refinement on GOLDEN and ALTREF (4 on
  the pan), no smaller transform for inter 8×8 blocks (7 on the three
  scenes), a stop after the first candidate when it skips (2.5 on the
  still scene), a looser stop for blocks that code a residual (1 to 4 on
  all three); and, bit for bit, splitting the trial walk's levels into a
  vector pass (no faster) and bounding the final RD walk too (no faster).

Before that, at speed 5 against FFmpeg's defaults (three scenes, still
scene, grainy pan), from −2.0% / +2.7%, +6.5% / +14.6% and +32% / +22%:

- Rows on threads (the wavefront above): speed 5 on 720p in four fifths
  of its time, the same streams.
- The one-pass inter model moving by a twentieth of each frame (it moved
  by 0.4): −11% / −9% on three scenes, −5% on the still scene.
- One pass ties golden frames, alt-refs and key frames to the inter step,
  plans the first key frame from the lookahead's probe, and spreads a key
  frame's excess over 30 frames. Before, frame types were budgeted
  apart: on the pan the alt-refs came out at q 160 while the inter frames
  predicting from them were at q 125, and the first key frame was planned
  from a guess (q 112 against libvpx's 85); 2 dB of PSNR at 1 Mb/s.
  −33% / −27% on the pan, and a golden frame without an alt-ref at a key
  frame's step: −19% at speed 10 on the pan.
- Jumps: the still-scene clip's two jumps were coded as cuts (40 kB each,
  intra) where libvpx's search found them (0.8 kB): −30% at a fixed
  quantizer, −16% at one pass.

What is left:

- Time on the pan at its top rates: 0.76 to 0.78 of libvpx's on one
  thread from 2 to 4 Mb/s, where the quantizer is under the grain and
  nearly every block codes a residual. The partition search is still
  most of the time there: of the whole encode, the RD walk of the trials'
  levels is a quarter (one token after another, each one's context from
  the ones before it), the forward transforms a tenth and the motion
  search about a sixth. libvpx's learnt partition pruning
  (`ml_var_partition_pruning`, `rd_ml_partition`) has no counterpart
  here: a block's rate and error tell its tied splits from the ones that
  win (above), but not the ones that win by a little from the ones that
  lose by a little. The margin on the three scenes at the averaged PSNR
  is −3.0% on four threads and −4.0% on one. Tried and left out
  before: a still block taking LAST at zero without a search (7%
  faster on the still clip, but at any threshold loose enough to matter
  it lost the pan), the above-right seed of the search dropped so that a
  row needs only the superblock above it (8% faster, 5% more bits on the
  pan), three or four rows on threads per tile column (no faster).
- Time on the pan with the reference pool in one pass (see "Chroma"):
  0.81 of FFmpeg's on four threads and 0.78 to 0.82 on one, at its
  finer quantizers; and its rate, 3 to 16% under (see there).
- Chroma, and the three scenes' margin. Speed 5 is ahead of libvpx's
  cpu-used 1 there on every plane and average, but by 1.1 to 4.6
  points (on four threads: −3.2% PSNR-Y, −1.1% PSNR-U); `testsrc2` is
  where it is lost (see "Chroma"). After "The gaps", every mode is
  ahead of its FFmpeg counterpart on every column there, but narrowly
  at PSNR-U: speed 6 −1.4%, speed 10 −0.7%, two passes −0.9% (at a
  chroma rate of 0.8 in the second pass, +0.04%); a run on another
  number of threads (other tile columns) moves these by a point. Speed 6
  takes speed 5's time on the three scenes (it did before too), and
  speed 10 0.90 to 1.02 of FFmpeg's realtime 8 on four threads in the
  last runs (0.86 to 1.00 before, in the same runs: 3 to 4% more). Before "Chroma": speed 6 was ahead of libvpx's
  cpu-used 1 on the three scenes by 2.0 points at the averaged PSNR (8.2
  at PSNR-Y), speed 5 by 3.0. At speed 6, a coarser gate on the smaller transform (128 bits an
  8×8) lost 5 points, intra left out of 64×64 blocks 1.1, speed 5's
  threshold for trying intra at eight levels 1.9, and no compound
  prediction or no whole-sample check around the diamond's choice 2.7
  and 4 points. Weighing chroma's bits more in RD quantization (2.2 times
  luma's, from 1/0.7) gains 6–17% at the same PSNR-Y and 1–7% at the
  6:1:1 average but loses 2–5% at FFmpeg's 4:1:1 average and 9–18% at
  PSNR-U, so it was left as it was (and is now 0.8, the other way).
- Constant quality on the still scene: −4.4% / −4.1% at `q=60` to
  `210`, but 0.0% / −0.1% sampled at `q=50` to `225`: at `-crf 48` (0.3
  Mb/s) libvpx codes the pan after the still picture (frames 31 to 90) in
  43 kB at 41.7 dB PSNR-Y, where `q=165` takes 103 kB for 41.4 dB, and
  that end of the curve decides the margin. libvpx's one pass at a fixed
  level modulates its quantizer by position (finer every second, fourth
  and eighth frame: 192, 184, 173, 154 at `-crf 48`); that pattern on a
  fixed quantizer was 10 points worse here.
- Realtime: speed 9 takes 0.90 and 0.94 of libvpx's realtime 8 time on
  the three scenes on four threads and one (0.77 to 0.91 on the other
  clips), closest at 4 Mb/s, where its time grows with the bits and
  libvpx's hardly does (libvpx also comes in 7% under the rate there,
  this encoder 3% over it). What is left of its time over speed 10's is
  its trials of intra against inter (speed 10's rule in their place
  costs 6 points) and its diamonds at 16×16 and 32×32. On four threads
  the four tile columns still carry uneven work (the busiest takes 1.5
  to 3 times the least busy's time), but taking their motion searches
  off them made no difference: every thread is busy. The constant bitrate path is within 2% of the
  rate at 1 to 4 Mb/s at both speeds, but at 500 kb/s on the still
  scene speed 10 takes 655 kb/s at the coarsest quantizer (speed 9 is
  on the rate there).

Matching libvpx's speeds by these runs and the ones above: good
cpu-used 0 → speed 4; cpu-used 1 (FFmpeg's default) → 5; cpu-used
2–3 → 6; cpu-used 4–5 → 7; realtime 5–6 → 9; realtime 7–8 → 10.

### Options: FFmpeg's `libvpx-vp9` against `Vp9Options`

Every option of `ffmpeg -h encoder=libvpx-vp9` (FFmpeg 6.1), the generic
options it reads, and what the encoder has for each (`vp9enc` names in
brackets). "Added" is new in this round; each has a test in
`enc::api::tests` or `enc::tests`.

| FFmpeg | libvpx | Cutlass |
| --- | --- | --- |
| `-b:v` | `rc_target_bitrate`, VBR | `bitrate` (`rate=`) |
| `-crf N -b:v 0` | `end-usage=q`, `cq-level` | `qindex` = `Vp9Encoder::qindex_of_level(N)` (`crf=`, `q=`) |
| `-crf N -b:v R` | `end-usage=cq` | `cq_level` (`crf=` with `rate=`, `cq=`): added |
| `-minrate` = `-maxrate` = `-b:v`, `-bufsize` | `end-usage=cbr`, `rc_buf_sz` | `cbr_buffer_ms` (`cbr=`) |
| `-qmin`, `-qmax` | `min-q`, `max-q` | `min_q`, `max_q` as indices (`min_q=`, `max_q=`: added to `vp9enc`) |
| `-undershoot-pct`, `-overshoot-pct` | `undershoot-pct`, `overshoot-pct` | `undershoot_pct`, `overshoot_pct`: dead bands on what is paid back (added) |
| `-max-intra-rate` | `max-intra-rate` | `max_intra_pct` (added) |
| `-drop-threshold` | `drop-frame` | missing: no frame is dropped |
| `-g` | `kf-max-dist` | `keyint` |
| `-keyint_min` | `kf-min-dist` | `keyint_min`: a cut that far from the last key frame is one (added); intervals are fixed as with `-keyint_min` = `-g` |
| `-force_key_frames` | `VPX_EFLAG_FORCE_KF` | `Vp9Encoder::force_keyframe` (the engine's `force_keyframe`) |
| `-pass 1` / `-pass 2` | `g_pass`, stats | `Vp9Encoder::first_pass_qindex`, `set_second_pass` (`twopass=1`) |
| `-deadline` best / good / realtime, `-cpu-used` (`-speed`, `-quality`) | | `speed` 0–10 (`speed=`); see the speeds below |
| `-lag-in-frames` (`-rc_lookahead`) | `g_lag_in_frames` | `lag` (`lag=`) |
| `-auto-alt-ref` 0 / 1 | `enable-auto-alt-ref` | `alt_ref` (`alt_ref=0`, added); libvpx's levels 2–6 (layered alt-refs) are not |
| `-arnr-maxframes` | `arnr-maxframes` | `arnr_max_frames` (`arnr_frames=`, added) |
| `-arnr-strength` | `arnr-strength` | `arnr_strength` (`arnr_strength=`, added) |
| `-arnr-type` | `arnr-type` | n/a: libvpx's VP9 encoder ignores it too (VP8 only) |
| `-min-gf-interval` | `min-gf-interval` | internal (golden every 12; 6 to 12 by motion in two passes), not an option |
| `-tune psnr` | `tune` | the default |
| `-tune ssim` | `tune=ssim` | missing |
| `-tune-content screen` / `film` | `tune-content` | missing |
| `-aq-mode 0` / `1` | `aq-mode` | `aq` (`aq=1`: variance AQ) |
| `-aq-mode 2` / `3` / `4` | complexity, cyclic refresh, equator360 | missing |
| `-noise-sensitivity` | `noise-sensitivity` | missing (no temporal denoiser) |
| `-static-thresh` | `static-thresh` | `static_thresh` (`static_thresh=`, added) |
| `-sharpness` | `sharpness` | `sharpness` (`sharpness=`, added) |
| `-error-resilient default` | `error-resilient` | `error_resilient` (`error_resilient=1`, added); `partitions` is VP8's |
| `-frame-parallel` | `frame-parallel` | `frame_parallel` (`frame_parallel=1`, added) |
| `-lossless 1` | `lossless` | `qindex` 0 (`lossless=1`) |
| `-tile-columns` | `tile-columns` | `tile_columns` (`tile_columns=`, added; by default one per thread) |
| `-tile-rows` | `tile-rows` | `tile_rows` (`tile_rows=`) |
| `-row-mt 1` | `row-mt` | always below speed 9: rows on threads (`enc::tile::Lanes`, added); from speed 9 one thread a tile column, as many as the threads allow beyond one |
| `-threads` | `g_threads` | `threads` (`threads=`) |
| `-level` | `target-level` | `level`: named in `vpcC` (added), not enforced |
| `-ts-parameters` | temporal layers | missing (no SVC / temporal layers) |
| `-corpus-complexity` | corpus VBR | missing |
| `-enable-tpl` | TPL model | n/a: two passes have their own allocation |
| `-pix_fmt yuv420p` … `yuv444p12le` | profiles 0–3 | `chroma`, `bit_depth` (`chroma=`, `bits=`) |
| `-pix_fmt gbrp`, `gbrp10le`, `gbrp12le` | `VPX_CS_SRGB` | `ColorSpace::Srgb` with 4:4:4: planes G, B, R (added) |
| `-pix_fmt yuva420p` | alpha as a second stream in WebM `BlockAdditional` | missing: a container feature |
| `-colorspace`, `-color_range` | `color-space`, `color-range` | `color_space`, `full_range` (`colorspace=`, `range=`), and from `CodecParams` (added) |
| `-aspect` / sample aspect ratio | `VP9E_SET_RENDER_SIZE` | `render_size` (`render=WxH`, added) |
| regions of interest (side data) | ROI map | missing |

Decoders: FFmpeg's `vp9` and `libvpx-vp9` decoders have no options of
their own; of the generic ones, frame threads and slice (tile) threads
are `Vp9Decoder`'s frame threads and tile-column parse threads, and
hardware decoding is out of scope. Every frame the decoder publishes
carries the colour space, range and render size.

### Bitstream features

| Feature | Decoder | Encoder |
| --- | --- | --- |
| Profiles 0–3, 8/10/12 bits, 4:2:0/4:2:2/4:4:0/4:4:4 | yes | yes |
| Colour space, range; sRGB | yes | yes (added) |
| Render size | yes | yes (added) |
| Key, inter, hidden (alt-ref) frames, superframes | yes | yes |
| Intra-only frames | yes | no (libvpx writes them for SVC and resilience) |
| `show_existing_frame` | yes | no |
| Error resilient, frame parallel | yes | yes (added) |
| Scaled references (a size change mid-stream) | yes | no |
| Lossless (WHT) | yes | yes |
| Tile columns and rows | yes | yes |
| Segmentation: quantizer | yes | yes (AQ) |
| Segmentation: loop filter, reference, skip; temporal map | yes | no |
| Loop filter sharpness | yes | yes (added) |
| Loop filter deltas (mode, reference) | yes | defaults only |
| Compound prediction, switchable filters, 8 slots | yes | yes |

## Limits and gaps

- Not registered in the desktop app (the app has no VP9 opener yet). The
  crate registers on the video factories and offers `register_engine_encoders`
  for the engine's `EncodeFactory`.
- Interpolation filters switch per block (EIGHTTAP, SMOOTH, SHARP, by
  trial) at speeds 0–2; faster speeds keep EIGHTTAP.
- The encoder signals any colour space and range, and sRGB in 4:4:4,
  but does not convert: the planes go in as they are coded.
- Encoder tools, by speed: compound prediction (a past picture with the
  alt-ref ahead, chosen per block) at 0–6 with `lag`; inter blocks under
  8×8 (4×4, 8×4, 4×8, each sub-block NEAREST, NEAR, ZERO or NEW) at 0–1;
  rectangular partitions below 4; probability updates of the
  coefficients and of every other updatable probability (modes, filter,
  references, partition, tx size, skip, vectors) at all speeds;
  segmentation with adaptive quantization (`aq`, off by default).
  Segment features other than the quantizer, temporal segment maps,
  intra-only frames, scaled references (a size change within the
  stream) and `show_existing_frame` are not written. Error resilient frames take no
  compound prediction (with it their compressed headers did not parse
  back: to look into).
- Against FFmpeg's `libvpx-vp9` as FFmpeg runs it ("Against FFmpeg's
  libvpx-vp9" above), the default speed takes fewer bits at the same
  PSNR-Y, PSNR-U, PSNR-V and averaged PSNR on every clip (1.1% to 55%),
  in 0.53 to 0.82 of libvpx's time on one thread and on four; two passes
  fewer bits at every PSNR on every clip (0.9% to 38%) in 0.30 to 0.38 of
  its time; constant quality with 4.3% to 46% fewer bits at every PSNR;
  speed 6 fewer bits than libvpx's default at every PSNR on every clip
  (1.0% to 50%), but in speed 5's time on the three scenes (0.93 to
  1.00 of it). Speed 10 has 0.7% to 70% fewer bits than libvpx realtime
  8 at every PSNR, in 0.88 to 0.90 of its time on one thread and 0.90 to
  1.02 on four (see "The gaps": over 0.95 on two clips in the last
  runs, where the build before took 0.97 and 1.00); speed 9 9% to 77%
  fewer, in 0.86 to 1.02 of it on four threads.
- At equal time the encoder is within about 8% of libvpx's good presets
  on moving content (ahead with more time, and far ahead on still content
  and against the realtime presets), and its fastest speed takes libvpx
  realtime's time at equal bitrate with much better quality; tile columns
  limit how well 4 threads share a 720p frame. The decoder takes 0.86–0.94
  of FFmpeg's time on 4 threads and on one core ("Speed and size against
  FFmpeg"). Its pictures go out without a copy, as
  shared views of the decoder's own planes at their padded stride
  (`Plane::samples` is a `Samples`, copied only on a first write;
  `Plane::packed` gives the visible rows); 10- and 12-bit samples are
  their little-endian bytes (copied into new buffers only on a
  big-endian target).

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7.
- libvpx was read where the specification and libvpx's streams disagree;
  no libvpx code is compiled into the crate.
