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
- Alpha (WebM's BlockAdditional 1, FFmpeg's `libvpx-vp9` `yuva420p`): a
  packet's alpha packet (`SideData::BlockAdditional` id 1) is decoded by a
  second decoder into the frame's alpha plane (`Vp9AlphaDecoder` alone).

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
  filter (its level searched on part of each frame up to speed 5, as
  libvpx's `LPF_PICK_FROM_SUBIMAGE`, and libvpx's level from the quantizer
  past it), probability adaptation.
- Alpha (`Vp9Options::alpha`, FFmpeg's `-pix_fmt yuva420p`): the alpha
  plane coded by a second encoder as its own VP9 stream, one alpha packet
  beside each packet ("Options measured").
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
  for 10 and 12 bits (and the search's start up to speed 5), the temporal filter's thresholds and strength and
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
  searches the loop filter level (`pick_level`, up to speed 5), filters,
  and commits the frame through `Decoder::commit` — so the encoder
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

Examples: `examples/roundtrip.rs` (decode a committed key frame),
`examples/vp9enc.rs` (raw 4:2:0 to IVF, or to WebM or Matroska by the
output's extension:
`vp9enc in.yuv 1280 720 out.ivf q=120 speed=5 threads=4`; its doc
comment lists the options, libvpx-vp9's among them; `alpha=FILE` codes
alpha) and `examples/vp9dec.rs` (the VP9 track of a WebM or Matroska file
to raw `yuv420p`, or `yuva420p` with alpha: `vp9dec in.webm out.yuv`).

With alpha (`Vp9Options::alpha`), each picture goes in with its alpha
plane and comes out with its alpha packet, which WebM keeps as the
block's BlockAdditional 1:

```rust
let mut encoder = Vp9Encoder::with_options(1280, 720, Vp9Options { alpha: true, ..Default::default() })?;
for coded in encoder.push_planes_alpha([(y, 1280), (u, 640), (v, 640), (a, 1280)], 1.0 / 30.0, (pts, 1, 30))? {
    // coded.data, coded.alpha, coded.keyframe, coded.clock
}
let plane = Vp9AlphaDecoder::new().decode(&alpha_packet)?; // or Vp9Decoder with SideData::BlockAdditional
```

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
- Alpha: `tests/alpha_interop.rs` writes a WebM with alpha through
  `cutlass-matroska` that FFmpeg's `libvpx-vp9` decodes to the pictures
  and alpha this decoder gives, and decodes FFmpeg's `yuva420p` WebM to
  FFmpeg's YUVA (skipped without `ffmpeg`).

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
faster). So speed 10 was over 0.95 of FFmpeg's time on four threads on
two clips here, 3 to 4% over the build before. Speed 9: 0.95, 1.02 and 0.86 on four threads
(0.94, 1.01 and 0.82 before).

The pool costs no CPU time (the same task-clock, ±5%, as the build
before it); the four threads were idle more. Speed 10 kept 2.2 of the
four cores busy, and its main thread did about 30% of the wall time on
its own: half of that starting and waking threads (each frame started
four tile threads, three loop-filter threads and four symbol writers),
the rest copies and reading the input. Two changes, both byte-identical
(41 of 41 configurations of the identity check):

- **One loop-filter thread for the three planes** (`lf_rows`), fed by
  one channel of each plane's rows: two threads fewer to start each
  frame, and fewer waking on the four cores the tile threads already
  fill. 6 to 11% less time at speed 10 on four threads.
- **The example reads the input on a thread of its own** (`vp9enc`, a
  few pictures ahead, as FFmpeg's demuxer runs on its own thread): 1 to
  4% less.

Tried and dropped, both byte-identical and no faster: building the
motion profile in bands on threads, and the cut check on threads.

Now, against realtime 8 on four threads (interleaved, medians of seven
or nine rounds of all four rates): speed 10 0.90, 0.95 and 0.82 (the
build with the pool, before these two changes, 1.03, 1.02 and 0.90 in
the same runs; the build before the pool 0.95–0.97, 0.93–0.97 and
0.85–0.87); speed 9 0.86, 0.88 and 0.76 (0.95, 0.96 and 0.88; before
the pool 0.91, 0.96 and 0.83). On one thread (paired, pinned, medians
of five) speed 10 is 0.90, 0.88 and 0.85.

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
chroma's error weighed 2.5 (PSNR-U −0.3% only); no compound prediction
(`COMPOUND_SPEED` 5, to take speed 6 under speed 5's time on the three
scenes): 3.5 to 9.8 points more bits at the same PSNR-Y than speed 6
with it, and behind FFmpeg's default on the three scenes (+3.1 / +3.8 /
+1.2 / +2.9 / +3.0), so speed 6 keeps it. Leaving out its extra chroma intra modes
(as speed 7 does) cost less but still too much: −0.2 / −0.0 / −1.7 /
−0.5 / −0.4 on the three scenes, level with FFmpeg at PSNR-U, so speed
6 keeps those too and takes speed 5's time there; everything it does
past speed 5 is what keeps it ahead. Speed 8: the old stop
with speed 6's chroma weights (0.3 to 1.4 points better at PSNR-U, 0.8
worse at PSNR-Y on the three scenes).

The two thinnest margins, then widened. Speed 10's PSNR-U on the three
scenes (−0.7%) and two passes' (−0.9%) were the closest any measure came
to libvpx's. Speed 10 at a bitrate now codes chroma 3 quantizer steps
finer than luma (`api::RT_UV_DELTA`, the header's chroma deltas, which
the encoder had always written as 0), and the second pass weighs chroma's
bits 0.7 of luma's (`tile::CHROMA_RATE_TWO`, from 0.75). BD-rate at the
same PSNR-Y / U / V / FFmpeg's average / 6:1:1, four threads:

| Three scenes | Before | Now |
| --- | --- | --- |
| Speed 10 vs realtime 8 | −8.3 / −0.7 / −2.5 / −5.8 / −7.0 | −6.9 / −1.6 / −4.3 / −5.5 / −6.1 |
| Two passes vs `-pass 1/2` | −6.4 / −0.9 / −5.3 / −5.1 / −5.6 | −6.0 / −1.6 / −5.9 / −5.2 / −5.4 |

The closest measure is now 1.6 points ahead in both, where it was 0.7
and 0.9; each gives back a little PSNR-Y for it. On the other two clips speed 10 is −69.6 /
−57.2 / −57.7 / −65.0 / −68.0 (still scene) and n/a / −59.3 / −60.0 /
−70.0 / n/a (pan), and two passes −37.6 / −34.3 / −34.2 / −35.5 / −36.3
and −14.3 / −29.0 / −28.3 / −21.9 / −20.0. Larger chroma deltas trade
on: −6 gives −5.1 / −2.3 / −5.3 / −4.7 / −4.9 on the three scenes, −10
−2.6 / −3.5 / −6.4 / −3.6 / −3.3. At speed 9, whose closest measure is
9 points ahead, −3 bought nothing (−15.4 / −9.7 / −11.5 / −13.7 / −14.5
from −17.0 / −9.4 / −10.7 / −14.4 / −15.6); a chroma bit weight of 1.15
instead of 1/0.7 at speeds 9 and 10 changed no stream (their realtime
path does not weigh chroma's bits). Speed 10 on four threads: 0.92,
0.96 and 0.82 of realtime 8's time (0.91, 0.92 and 0.83 for the build
before, in the same runs). The streams were checked in FFmpeg's `vp9`
(threaded and on one thread), libvpx and this decoder: 20 of 20 decodes
the same; every other configuration of the identity check is
byte-identical (36 of 41; the five that changed are speed 10 at a
bitrate and two passes).

**Synthetic pans.** The alpha plane of "Alpha" (in "Options measured"),
a 640×360 plane of `128 + 127 sin((X + 6N)/50) cos((Y − 3N)/40)`
(chroma 128, 60 pictures), moves by whole samples, (−6, +3) a picture,
so that nearly every block predicts exactly from one vector; yet at
`crf=15` to `55` (speed 5, four threads) this encoder took +57% bits
at the same PSNR-Y against FFmpeg's `-crf` (speeds 0 and 2: +25%),
with inter frames of twice libvpx's size, 2.5 dB under. Decoded by this
crate's decoder with the modes, vectors and residuals of each block
printed, libvpx's frames and ours, two causes:

- The motion search never read past the reference's edges. Its SADs
  (`sad_ref`, the four at once, `sad_sub`, `sad_sub_exact`) were `None`
  for a window over an edge (and the 8-tap one wanted three samples
  inside), so content that moves in from an edge (here the top three
  rows and the right six columns, each picture) could not take the
  motion, nor its neighbours' NEAREST and NEAR where those cross: the
  top superblock row coded NEWMV of (0, 4.5) to (0, 7.5) samples with
  a residual each picture, which made the row below code NEAR (12 to 24
  blocks a picture), and the right column took intra or vertical
  vectors. libvpx reads its 160-sample borders. Now a window reaches
  `tile::EDGE_REACH` (16) samples past each edge, the edge samples
  repeated as the decoder predicts there (a window over the top or the
  bottom only reads the edge row again, at a stride of 0; one over the
  sides is copied).
- The alt-ref's temporal filter blurred the pan. Its search in one pass
  started from zero alone, and its diamond stops in the valleys of a
  smooth picture: one block in five matched two samples off its exact
  match, at an error its weights still take (the filtered alt-ref was
  0.39 a sample in squared error from its picture; now 0.03). It now
  starts from the match of the block to its left too (as two passes'
  did), reaches past the edges as the search does, and filters chroma
  at half a sample where the luma vector is odd (the mean of the two
  chroma samples it falls between; it took the one before, half a
  sample off: on the grainy pan, 7 to 8 samples a picture, every other
  picture's chroma).

The gradient (bytes of the IVF, PSNR-Y), FFmpeg's `-crf` against
`crf=` here:

| crf | FFmpeg | Before | Now |
| --- | --- | --- | --- |
| 15 | 17767, 55.10 dB | 17570, 51.86 dB | 13491, 52.43 dB |
| 25 | 14431, 53.64 dB | 12580, 50.63 dB | 9941, 50.94 dB |
| 35 | 8704, 51.31 dB | 10130, 48.66 dB | 7760, 48.73 dB |
| 45 | 6428, 48.40 dB | 8660, 45.93 dB | 6505, 46.04 dB |
| 55 | 5395, 45.09 dB | 7654, 42.80 dB | 5969, 42.68 dB |

+57.0% → +17.9% at the same PSNR-Y (the edges alone: +22.8%; speed 2
+5.5%). `testsrc2` (640×360, 60 pictures, the colour stream of
"Alpha"): +18.2% → +18.1%. On the README's three clips, against the
build before (BD-rate at the same PSNR-Y / U / V / FFmpeg's average /
6:1:1, four threads; the realtime speeds with `cbr=1000`, constant
quality at `q=60` to `210`):

| Before → now | Three scenes | Still scene, two jumps | Grainy pan |
| --- | --- | --- | --- |
| Speed 5 | −0.6 / −0.4 / −0.3 / −0.5 / −0.6 | −0.8 / −0.9 / −0.9 / −0.9 / −0.9 | −1.7 / −3.1 / −3.1 / −2.3 / −2.2 |
| Speed 6 | −0.6 / +0.4 / +0.2 / −0.3 / −0.4 | −2.3 / −0.4 / −0.3 / −1.2 / −1.6 | −2.3 / −0.9 / −0.8 / −1.0 / −1.7 |
| Two passes | −0.6 / −0.5 / −0.6 / −0.6 / −0.6 | +0.6 / −1.1 / −1.1 / −0.4 / −0.1 | −3.7 / −2.8 / −2.7 / −3.2 / −3.2 |
| Constant quality | −0.4 / +0.1 / −0.1 / −0.2 / −0.3 | −1.6 / +0.1 / +0.2 / −0.7 / −1.1 | −3.3 / −0.7 / −0.9 / −2.1 / −2.6 |
| Speed 9 | +0.1 / −0.2 / −0.6 / −0.1 / −0.0 | −2.7 / −0.7 / −0.1 / −1.5 / −2.1 | −3.0 / −1.6 / −1.7 / −2.2 / −2.6 |
| Speed 10 | −0.2 / −0.8 / −0.6 / −0.4 / −0.3 | −3.0 / −2.8 / −2.8 / −2.9 / −3.0 | −1.6 / −2.9 / −2.7 / −2.1 / −1.8 |

and against FFmpeg (speeds 5 and 6 against its default, two passes
against `-pass 1/2`, constant quality against `-crf`, speeds 9 and 10
against realtime 8): speed 5 −4.0 / −1.4 / −3.6 / −3.4 / −3.7 on the
three scenes, −20.8 / −34.6 / −34.8 / −28.9 / −25.8 on the still scene,
−31.9 / −50.4 / −50.5 / −43.5 / −39.4 on the pan; speed 6 −1.7 / −0.9 /
−3.0 / −1.8 / −1.8, −21.5 / −35.6 / −35.7 / −29.8 / −26.6, −30.9 /
−47.2 / −47.5 / −41.2 / −37.7; two passes −6.5 / −2.1 / −6.4 / −5.7 /
−5.9, −37.4 / −35.1 / −35.0 / −36.0 / −36.4, −16.2 / −30.8 / −30.1 /
−23.7 / −21.8; constant quality −4.6 / −15.4 / −16.5 / −9.5 / −7.7,
−5.6 / −40.9 / −40.7 / −29.7 / −21.1, −26.2 / −45.1 / −45.2 / −38.0 /
−33.1; speed 9 −11.4 / −7.5 / −9.7 / −10.4 / −10.8, −61.1 / −54.3 /
−55.0 / −58.2 / −59.7, −64.0 / −57.0 / −57.8 / −60.8 / −62.4; speed 10
−0.1 / −1.2 / −3.8 / −1.1 / −0.7, −41.2 / −40.6 / −40.9 / −40.8 /
−41.1, −48.8 / −50.6 / −51.5 / −49.7 / −49.4. Every column on every
clip is ahead of FFmpeg, as before. CPU time on one thread (1 and 3
Mb/s on the three scenes and the pan, 0.5 and 2 on the still scene,
the least of two or three runs each, against the build before): speed
5 1.01, 1.01 and 1.03; speed 6 1.05, 1.04 and 1.07; speed 10 1.02,
1.03 and 1.02 (the copies of windows over the edges, the filter's
seeded search). Every configuration of the identity check changed (41
of 41: the search reads past the edges at every speed); 56 decodes of
14 new streams (the gradient at speeds 2, 5, 6 and 10, `testsrc2` in
two passes and at `crf=30`, the pan at 10 bits, the three clips at
speeds 5, 6, 9 and 10, two passes and constant quality) are the same in
FFmpeg's `vp9` (threaded and on one thread), libvpx and this decoder.

Tried and left out: the edges alone without the filter's changes, on
speed 6 the still scene lost 1.8 points at every column; the filter's
seed and half samples without the edges, constant quality lost 1.4 to
2.2 points on the still scene (with both, every mode gains). A reach of
8 or 32 samples (before the half samples, at speed 6: the pan's PSNR-U
+0.2 and +2.7 points against the build before, where 16 gave +1.3). What
is left on the gradient: its key frame, at the same quantizer, is 1.7
dB under libvpx's (at `q=100` 1552 bytes and 48.5 dB, libvpx's at 99
1715 bytes and 50.2 dB, coded in 32×32 blocks and transforms with many
more coefficients than ours; a finer key frame or a lower lambda on it
moved along the same curve); speed 6 is at +80%
(its partition search stops at whole 64×64 blocks on this smooth
picture, so the new content at the edges costs a 64×64 block a
picture); the loop filter level from the quantizer (libvpx searches
it: on `testsrc2` its levels go from 1 to 14 at `-crf 35`, where ours
are 21; half our level there gave 14.8% where the level from the
quantizer gave 17.8%), and per-block filters (speed 5 with SMOOTH and
SHARP tried, before the filter's changes: +18.1% on the gradient from
+22.8%, in more time).

**Smooth and synthetic pictures, again** (16 November 2026, after
`e11866b6`). What "Synthetic pans" left: the gradient at +17.9% against
FFmpeg's `-crf` (its key frame 1.7 dB under libvpx's at the same
quantizer) and `testsrc2` at +18.1%. Coded alone at `q=68` and decoded
by this crate's decoder with each block's mode, transform and end of
block printed, libvpx's key frame of the gradient is 199 blocks of 32×32
in DC, H, V and TM whose ends of block sum to 15 000, at 0.45 a sample
of squared error; ours had 24 blocks of 64×64 and 60 of 8×8, mostly TM
and the directional modes, ends of block summing to 4 000, at 0.67. Speeds 0 to 3 code that key frame alike, 0.2 dB under libvpx's
curve; speed 5 was 0.6 dB under speed 3 for 6% more bits. Turning speed
5's key frame features off one at a time, and the inter frames' (the
gradient at speed 4: +7.5%, at 2: +5.0%), gave three causes, and the
loop filter a fourth:

- The intra modes a key frame's block tries: speed 5 tries the three
  best by SATD, and only those within a fifth of the best's, and on a
  smooth picture SATD ranks them poorly. With a full search (a fixed
  quantizer up to speed 5, or a second pass) a key frame's blocks now
  try the best four, however far behind (`KEY_FULL_SPEED`): its key
  frame is 0.06 dB under speed 3's for 2% more bits, in 1.18 times the
  instructions of a key frame (720p, a fine quantizer).
- The inter candidate of large blocks: from 32×32 speed 5 tries one, the
  best by SAD and vector rate, and on the gradient's pan the search
  finds vectors a sample to four off the motion, down, whose SAD is a few
  units under the exact vector's (0.1 to 1% of a 64×64 block's SAD: the
  reference's quantization noise), where NEAREST has the exact one and
  wins its trial (a sixth of the 64×64 blocks, a quarter of the 32×32
  ones, with both tried). With a full search a NEW vector now also tries
  the free one (NEAREST, NEAR or ZERO of its reference) ranked after it
  within 3% of its score (`CLOSE_FREE`).
- The loop filter level came from the quantizer. libvpx searches it
  (`vp9_pick_filter_level`): on `testsrc2` its levels go from 1 to 14 at
  `-crf 35`, where ours were 21, and halving ours gave 14.5%. Up to speed
  5 the level is now searched (`pick_level`, libvpx's
  `search_filter_level` on part of the picture): one superblock row in
  four, spread over the frame, filtered in luma at the quantizer's level
  and a quarter of it either side, moving to the side of less error
  (libvpx's bias for the lower level kept), the step halved while the
  level stands, down to 2; levels 2 to 61, whose masks the encoder
  rebases from the quantizer's (`Mask::rebase`) rather than build them
  again. The level found is kept only where the three planes' errors,
  each against its own at the quantizer's level, fall by 1% on average.
  A frame whose level is searched is not filtered row by row as its tiles
  come (the frames are the same whatever the threads).
- Not a cause: the key frame's quantizer (libvpx's `-crf` codes it at a
  quarter of the inter step; ours at 0.35 of it): at 0.25 no better, at
  0.18 1 to 1.6 points worse.

The gradient and `testsrc2` (bytes of the IVF, PSNR-Y; speed 5, four
threads), FFmpeg's `-crf` against `crf=` here:

| crf | Gradient, FFmpeg | Before | Now | `testsrc2`, FFmpeg | Before | Now |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 17767, 55.10 dB | 13491, 52.43 dB | 13433, 52.83 dB | 357003, 54.36 dB | 339547, 52.39 dB | 339601, 52.41 dB |
| 25 | 14431, 53.64 dB | 9941, 50.94 dB | 9991, 51.37 dB | 272682, 50.37 dB | 255025, 48.53 dB | 254403, 48.54 dB |
| 35 | 8704, 51.31 dB | 7760, 48.73 dB | 7491, 49.01 dB | 175644, 45.59 dB | 154368, 43.25 dB | 153152, 43.34 dB |
| 45 | 6428, 48.40 dB | 6505, 46.04 dB | 6269, 46.38 dB | 87402, 40.61 dB | 68119, 38.26 dB | 65830, 38.47 dB |
| 55 | 5395, 45.09 dB | 5969, 42.68 dB | 5555, 43.10 dB | 33071, 37.41 dB | 22194, 34.55 dB | 21597, 34.56 dB |

At the same PSNR-Y, the gradient +17.9% → +10.9%, `testsrc2` +18.1% →
+15.2%. By part: the level's search alone, +17.9% and +15.5%; with the
key frames, +13.2% and +15.1%; the key frames and free vectors without
the search, +10.9% and +17.7%. On the README's three clips, against the
build before (BD-rate at the same PSNR-Y / U / V / FFmpeg's average /
6:1:1, four threads; constant quality at `q=60` to `210`):

| Before → now | Three scenes | Still scene, two jumps | Grainy pan |
| --- | --- | --- | --- |
| Speed 5 | +0.0 / −0.1 / −0.6 / −0.1 / −0.1 | −2.6 / −2.2 / −2.4 / −2.4 / −2.5 | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |
| Two passes | −0.8 / −1.5 / −1.1 / −1.0 / −1.0 | −2.1 / −0.2 / −0.3 / −1.1 / −1.4 | −1.7 / +0.1 / +0.2 / −0.6 / −0.9 |
| Constant quality | −0.6 / −1.3 / −1.2 / −0.8 / −0.7 | −3.8 / −2.5 / −2.4 / −3.1 / −3.4 | −1.4 / −2.1 / −2.0 / −1.7 / −1.6 |

Speeds 6, 9 and 10 write the same streams as before (the pan at speed 5
too: its searched levels never took off the 1%). Against FFmpeg now:
speed 5 −3.9 / −1.6 / −4.4 / −3.6 / −3.7 on the three scenes, −22.7 /
−36.2 / −36.5 / −30.7 / −27.6 on the still scene, −31.9 / −50.4 / −50.5
/ −43.5 / −39.4 on the pan; two passes −7.4 / −3.6 / −7.5 / −6.7 / −6.9,
−38.4 / −35.2 / −35.2 / −36.6 / −37.2, −17.0 / −30.7 / −30.0 / −24.0 /
−22.3; constant quality −4.9 / −16.3 / −17.3 / −10.0 / −8.1, −8.2 /
−42.2 / −41.9 / −31.4 / −23.1, −27.6 / −46.6 / −46.7 / −39.4 / −34.6.
CPU time on one thread against the build before (1 and 3 Mb/s on the
three scenes and the pan, 0.5 and 2 on the still scene, `q=80` and
`160` at constant quality; medians of three rounds, in turn): speed 5
0.995, 1.025 and 0.997; speed 6 1.008, 1.013 and 1.017 (the same
streams: the noise of the measure); speed 10 0.991, 1.015 and 0.963;
two passes 1.008, 1.018 and 1.035; constant quality 1.026, 1.059 and
1.030 (the key frames' fuller search and the free vectors' trials, 2%
each on the still scene). Speed 10 on four threads against FFmpeg's
realtime 8: 0.97, 1.01 and 0.84 (the build before in the same runs:
0.94, 1.01 and 0.86). 11 of the identity check's 41 configurations
changed (speeds 2 and 5 at a fixed quantizer, and at a bitrate where a
searched level took off the 1%; two passes; constant quality), the
other 30 (every one at speeds 6 to 10) did not; 56 decodes of 14 new streams (the gradient
and `testsrc2` at `crf=35`, `testsrc2` in two passes and at speed 2, 10
bits at three quantizers, 4:4:4, a noisy picture at a bitrate, speed 2
on one thread, the three clips' constant quality, two passes and speed
5) are the same in FFmpeg's `vp9` (threaded and on one thread), libvpx
and this decoder.

Tried and left out: the level searched on luma alone (libvpx's measure):
the grainy pan at speed 5 lost 4 points of PSNR-U, as its luma's error
wants less filter and its chroma's more; the three planes' errors
summed, 2.3 points; each plane's against its own, averaged, 1.6; with a
margin of 1%, 0.4; with level 0 (no filter) left out of the levels, as
now, none (the 0.4 was the pan's key frame turning its filter off). The
whole frame searched (`testsrc2` 0.4 points better than a row in four)
took 10% more time at speed 5, a row in three 5%, libvpx's single band
in the middle (an eighth of the rows) lost 2 points on `testsrc2`. The
key frames' fuller search at a bitrate in one pass: the pan lost 2.2
points of PSNR-U for 2.5 of PSNR-Y (the key frame's size moves the rate
control there), so only with a full search; with the key frames' trials
quantized by RD too (not by the dead zone), the gradient a point better,
for 1.69 times a key frame's instructions. The free vector preferred
without its trial (ranked first within 3%, at every speed and mode): 1.8
points lost on the still scene at speed 6, 0.6 on the pan at speed 5. Two inter candidates at every
32×32 and larger block with a full search: the gradient at +8.9%, for
5% more time at constant quality on the pan; the second candidate tried
whenever within 3% of the first (not only a free one after a NEW), 3
to 7% more time at a bitrate. Rectangular partitions (speed 4's halves, tried before
the split): `testsrc2` 5 points better, in 2.5 times the time; after a
split that won and only where its quarters pair up across or down, 0.6
points; after every split that won, 3.6 points for 1.66 times the time.
`testsrc2` without alt-refs: 2.2 points better there, 8 worse on the
gradient. What is left: `testsrc2` at +15.2% (rectangular partitions
are 5 points of it; with these changes speed 4 is at +13.0%, speed 2 at
+6.1%), the gradient at +10.9% (speed 2: +5.0%).

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

**Seeded reference searches and halves after a split** (17 November
2026, after `b564507e`). What was left: the gradient at +10.9% against
FFmpeg's `-crf`, `testsrc2` at +15.2%.

- The search on ALTREF started from zero and LAST's neighbours' vectors,
  and an alt-ref's search on LAST from its own; on a pan both stopped in
  the valleys of smooth pictures and grain, far from where the content
  stood. With the reference pool (up to speed 6), they now also start
  from the whole-picture offset the pool tracks between the two pictures
  (`FrameEncoder::set_ref_seed`), and reach that far past the usual 64
  samples. Speeds 7 and up write the same streams.
- At speeds 4 and 5 with a full search (a fixed quantizer, a second
  pass), an inter frame's 16×16 block whose split won also tries its
  halves, across or down, where two of its quarters side by side share a
  reference and vector: libvpx's rectangular partitions, pruned by the
  split's own choices. Each half takes the best of the whole block's and
  its quarters' predictions by SAD and vector rate, with no search of its
  own; only the direction with more matching pairs; intra modes only
  where the block or a quarter chose intra; 4×4 transforms only where a
  quarter took them.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y, 15
to 55, four threads):

| | Before | Seeded searches | And halves |
| --- | --- | --- | --- |
| Gradient | +10.9% | +6.0% | +5.8% |
| `testsrc2` | +15.2% | +12.8% | +11.4% |

On the README's three clips (three scenes / still scene / grainy pan),
BD-rate Y / U / V / FFmpeg's average / 6:1:1 against the build before:

| Mode | Three scenes | Still scene | Grainy pan |
| --- | --- | --- | --- |
| Speed 5, 1–4 Mb/s | −1.6 / −2.8 / −3.0 / −2.1 / −1.9 | −5.1 / −9.7 / −9.5 / −7.5 / −6.7 | −7.3 / −14.5 / −14.2 / −11.5 / −10.2 |
| Speed 6 | −1.4 / −2.9 / −2.8 / −2.0 / −1.8 | −4.2 / −9.8 / −10.0 / −7.4 / −6.2 | −6.8 / −16.2 / −15.9 / −12.3 / −10.6 |
| Two passes | −2.8 / −2.4 / −2.7 / −2.7 / −2.8 | −3.5 / −3.5 / −3.3 / −3.4 / −3.5 | −6.2 / −5.2 / −5.5 / −5.8 / −5.8 |
| Fixed q 60–210 | −3.8 / −4.0 / −3.9 / −3.9 / −3.8 | −9.8 / −9.7 / −9.9 / −9.8 / −9.8 | −25.0 / −24.7 / −24.7 / −25.0 / −24.9 |

At the same quantizer the PSNR stays within 0.1 dB and the pan's streams
take 13–30% fewer bytes. Speed 5 against FFmpeg's default is now −5.7 /
−5.0 / −8.2 / −6.0 / −5.9 on the three scenes, −26.3 / −41.9 / −42.1 /
−35.4 / −32.0 on the still scene and −36.2 / −59.7 / −59.7 / −51.1 /
−46.1 on the pan; speed 6 −3.3 / −4.4 / −6.2 / −4.0 / −3.7, −24.6 /
−42.3 / −42.7 / −35.0 / −31.1 and −35.2 / −59.9 / −59.7 / −50.8 /
−45.6.

CPU time on one thread against the build before (30 pictures, median of
three interleaved runs pinned to one core; three scenes, still scene,
pan): `q=120` 1.02, 1.12, 0.97; speed 5 at 2 Mb/s 0.97, 0.92, 0.93;
speed 6 0.94, 0.97, 1.11; two passes 1.01, 1.08, 0.99. The halves cost
time on the still scene at a fixed quantizer and in two passes, where
they are tried; the seeded searches end sooner on the pan at speed 5.

Checks: 27 of the 41 configurations of the identity check change (speeds
2, 5 and 6, two passes and fixed quantizers), none at speeds 7 to 10. FFmpeg's
`vp9` (threaded and on one thread), libvpx and this decoder agree on 7
new streams (28 of 28 decodes: the pan at `q=120`, two passes, speed 6,
4:4:4 at `q=60`, 10-bit at speed 6, `testsrc2` and the gradient at
`crf=35`, the last on one thread).

Tried and dropped: each half taking the best two candidates in both
directions (`testsrc2` +12.8% → +12.0%); the single best in the
direction with more pairs gives +11.9% with fewer trials. What is left:
`testsrc2` at +11.4%, the gradient at +5.8%.

**Small objects, the loop filter and key frames at a fixed quantizer**
(17 November 2026, after `9c3ecf54`). What was left: `testsrc2` at
+11.4% against FFmpeg's `-crf` (+9.9% / +6.9% at the same PSNR-U / V),
the gradient at +5.8%. Coded at `-crf 35`, libvpx's `testsrc2` has no
alt-ref: LAST alone, a golden frame every 10 pictures, and quantizers 140
/ 131 / 120 / 98 by the picture's place in 8. Measured by region at the
same size (FFmpeg's `-crf 35` against `crf=32` here), the squares of
noise (3% of the picture, two thirds of the error) are as good in both;
the 8×8 blocks that hold a row of that noise (it starts a row below an
8×8 edge) were 1.5 dB under libvpx's, and the rest of the picture (94%)
2 dB under. Three causes:

- The partition search's stop for a whole block that needs no residual
  (under a full search: its error under the quantizer's step squared over
  24 a sample) summed the block's error, which hid a small object (a
  moving square, a figure of the clock, a row of noise) in a block that
  predicted the rest exactly; the object went uncoded. At a fixed
  quantizer the stop now also needs every 8×8's luma error under the same
  bound (`Tile::peak_sse`). Not in a second pass, where the still scene
  lost 0.4 points of PSNR-Y; second passes write the same streams.
- The loop filter's search (`pick_level`) measures the frame it filters
  alone. On pictures that stand still, whose blocks keep their reference
  whole and have their edges filtered again each picture, the levels it
  found blurred the pictures after it (libvpx's on `testsrc2` are 1 to
  14, ours were 6 to 21). At a fixed quantizer in one pass an inter
  frame now takes 0.6 of the level found (`LF_FIXED_SCALE`).
- A key frame of a smooth picture codes a gradient in 64×32 and 32×64
  halves for less than in four 32×32 blocks; speeds 0 to 3 try halves at
  every size, speed 5 at none. The first picture of the gradient alone
  was +6.5% against libvpx's at the same PSNR-Y (speed 3: +2.7%). From
  speed 4 at a fixed quantizer in one pass, a key frame's 64×64 block
  whose whole cost came within 1.1 times the best of whole and split now
  tries its halves after the split (`KEY_HALVES`, `Tile::try_halves`):
  +5.2%.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y, and
for `testsrc2` U / V, 15 to 55, four threads):

| | Before | 8×8 peak | And the level | And key halves |
| --- | --- | --- | --- | --- |
| `testsrc2` | +11.4 / +9.9 / +6.9% | +10.0 / +9.1 / +3.2% | +8.8 / +8.0 / +2.9% | +8.7 / +8.0 / +2.7% |
| Gradient | +5.8% | +6.2% | +4.3% | +3.6% |

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, FFmpeg | Before | Now | Gradient, FFmpeg | Before | Now |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 357003, 54.36 dB | 333587, 52.49 dB | 333472, 52.72 dB | 17767, 55.10 dB | 12926, 53.01 dB | 12916, 53.06 dB |
| 25 | 272682, 50.37 dB | 248888, 48.58 dB | 250075, 48.80 dB | 14431, 53.64 dB | 9636, 51.56 dB | 9557, 51.57 dB |
| 35 | 175644, 45.59 dB | 150636, 43.46 dB | 150857, 43.60 dB | 8704, 51.31 dB | 7311, 49.25 dB | 7238, 49.42 dB |
| 45 | 87402, 40.61 dB | 64038, 38.56 dB | 64750, 38.83 dB | 6428, 48.40 dB | 6161, 46.58 dB | 5996, 46.71 dB |
| 55 | 33071, 37.41 dB | 20929, 34.67 dB | 22915, 35.27 dB | 5395, 45.09 dB | 5534, 43.58 dB | 5620, 43.54 dB |

Speed 2 at `crf=` (the peak and the level; key halves are from speed 4):
`testsrc2` +3.1 / −1.8 / −7.4% → +2.0 / −2.6 / −7.7%, the gradient
−0.7% → −1.4%. On the README's three clips at `q=60` to `210` against the
build before (BD-rate Y / U / V / FFmpeg's average / 6:1:1):

| Fixed q | Three scenes | Still scene | Grainy pan |
| --- | --- | --- | --- |
| 8×8 peak | −1.3 / −0.3 / −0.7 / −1.0 / −1.1 | −0.6 / −2.3 / −2.3 / −1.5 / −1.1 | −0.0 / −0.1 / −0.0 / −0.1 / −0.0 |
| All three | −1.4 / −0.9 / −1.0 / −1.3 / −1.3 | −0.4 / −2.2 / −2.2 / −1.3 / −0.9 | −0.3 / −0.8 / −0.5 / −0.5 / −0.4 |

Against FFmpeg's `-crf` there: −9.3 / −20.4 / −21.4 / −14.2 / −12.4 on the
three scenes, −17.8 / −49.3 / −49.2 / −39.4 / −31.7 on the still scene and
−45.9 / −60.5 / −60.5 / −54.9 / −51.2 on the pan (−8.1, −16.6 and −45.7
at the same PSNR-Y before). Every other mode (speeds 5 to 10 at a bitrate,
two passes) writes the same streams as before.

CPU time on one thread against the build before (30 pictures, medians of
five interleaved runs pinned to one core; three scenes, still scene,
pan): `q=120` 0.96, 1.11 and 0.98, `q=80` 1.06, 1.08 and 0.97 (1.015 and
1.034 on average). The still scene, with its two jumps, pays most: its
key frames' halves and the peak's splits.

Checks: of the identity check's 41 configurations, those at a fixed
quantizer at speeds 2 and 5 (and the still scene at `q=120`) change, the
other 36 (every bitrate, two passes, speeds 6 to 10) do not. FFmpeg's `vp9`
(threaded and on one thread), libvpx and this decoder agree on 10 new
streams (40 of 40 decodes: `testsrc2` at `crf=35` at speeds 5 and 4, at
`crf=30` at speed 2 and at `crf=40` on one thread, the gradient at
`crf=35`, a 4:2:0 picture at `crf=25` on two threads, the pan at `q=120`,
the still scene at `q=80`, the pan at 10 bits at `q=100`, 4:4:4 at
`q=60`). New tests: a 2×2 square in a flat 64×64 block that predicts whole
is coded; inter frames take `LF_FIXED_SCALE` of the searched level in one
pass only; a smooth key frame codes 64×64 blocks in halves in one pass
and none in a second (each fails without its change).

Tried and dropped (BD-rate on `testsrc2` / the gradient at speed 5 against
FFmpeg's `-crf`, from the build of each step):

- Halves after a winning split at 32×32, and at 32×32 and 64×64 too (the
  quarters' pairing prune): +11.9% and +12.1% from +11.4%. Halves after
  a split that lost, or in both directions without the pairing: +9.1%,
  +8.2% from +8.8%.
- Halves before the split at every size, with their own search, as speeds
  0 to 3 try them: +6.8% from +11.4%, in twice the time; at 16×16 only,
  +6.7% from +8.8%, in 1.34 to 1.67 times the time on the README's clips.
  Pruned to whole blocks that code a residual, to the direction their
  error is the more uneven, or to the halves' seeds without a search (the
  whole block's choice and each reference's NEAREST and NEAR), the gain
  went with the time (+8.1% to +9.7%, at 1.14 to 1.22 times).
- libvpx's structure: layered alt-refs (`arf_layers=2`, `3`: +14.3%,
  +15.5%); its quantizers by place in 8 (0.5 / 0.71 / 0.86 of the step):
  +15.2% (+12.6% without alt-refs); key frames at 0.25 and 0.2 of the step
  (+12.2%, +13.0%), golden frames at 0.4 and 0.6 (+13.0%, +11.5%).
- Lambda: the trials' × 0.7 and × 1.4 (+12.3%, +15.0%), RD quantization's
  (+11.9%, +14.4%); chroma's error weighed 1 rather than 2: +10.9% at the
  same PSNR-Y for +14.5% / +11.9% at PSNR-U / V.
- Each of speed 5's prunes set back to speed 4's alone: neutral or worse
  on `testsrc2` (one candidate at 32×32 and up +13.2%, the subpixel
  diagonals +11.9%, the dead zone's trials +12.4%); a transform cap of
  16×16 or 8×8 for inter blocks, sub-8×8 inter blocks: +11.2% to +11.5%.
- The 8×8 peak with chroma's error in it as the trials weigh it:
  `testsrc2` +10.1 / +5.6 / −0.7%, but 1.6 to 2.5 points of PSNR-Y lost on
  the still scene and the pan at fixed quantizers. With the bound doubled:
  +10.6% on `testsrc2` for +5.8% on the gradient. No stop at all: +9.8%
  for 1.46 times the time.
- The level's scale at 0.45 (the still scene a point worse) and 0.7 (half
  the gain); the search's level kept without its 1% margin (`testsrc2`
  +9.1%, the gradient +4.6%, the still scene's chroma worse); the scale on
  key frames (+9.1% / +4.8%), in a second pass (the pan 1.7 points worse),
  at a bitrate (0.1 to 1.4 points of PSNR-Y lost).
- Key frames' halves at every size with RD-quantized trials: the gradient
  +3.6% with them alone, but 1.2 to 1.8 times the time on 30 pictures;
  RD-quantized trials alone 1.0 to 1.2 times. Inter frames' 64×64 halves
  where whole came close: nothing.
- Candidates past the first without the margin at a fixed quantizer: the
  gradient +2.8%, `testsrc2` +8.5%, for 1.03 to 1.08 times the time; three
  intra candidates in inter frames: `testsrc2` +8.0%, 1.05 to 1.19 times.

What is left: `testsrc2` at +8.7% (speed 4: +7.1%, speed 2: +2.0%), the
gradient at +3.6% (speed 4: +2.3%). Most of `testsrc2`'s is halves before
the split, which cost more time than speed 5 has.

**A mesh after the diamond, and a second smaller transform** (17
November 2026, after `944bc05a`). What was left: `testsrc2` at +8.7% /
+8.0% / +2.7% against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V, 15 to 55, speed 5, four threads), the gradient at +3.6%. Coded at
FFmpeg's `-crf 35` and at `crf=32` here (175 644 and 181 194 bytes), the
error of `testsrc2` by region, ours against libvpx's: blocks whose source
did not change, 1.19 against 0.82 million (squared error, luma, the 59
inter frames); blocks that changed, 20.5 against 18.8 million. Mapped by
8×8, the excess sits on the small objects that move over the flat bars:
the grey bar that grows 6 samples a picture at each end, the square that
moves down by the green bar, the ring of small squares, the clock's
figures. Decoded with each block's mode and vector printed, libvpx codes
the grey bar's ends in the first inter frame as NEWMV (0, 6) and (0, 4)
samples and NEAREST, all without a residual; ours took vector 0 with a
residual, intra, and ALTREF at (0, −3.75). One cause, and a smaller one:

- The whole-sample search is a diamond from the neighbours' vectors and
  the pool's seeds, then the square of two around its choice. Over a flat
  background a small object leaves the same error at every vector short
  of its own (the grey bar's 8×8 block: 2 samples wrong a row from 0 to 4
  samples across, none at 6), and the diamond stops at its start; on
  GOLDEN and ALTREF, many pictures away, the motion is many times a
  picture's and the diamond stops in the valleys between. A square of
  radius 4, 6 or 8 at the end took `testsrc2` from +8.7% to +7.7%, +6.4%
  and +5.7%; a mesh of every second sample within 16 at every block size,
  to +4.0%, in 1.3 to 1.4 times the time on the README's clips (with a
  floor on its SAD). At a
  fixed quantizer in one pass a 16×16 block's search now ends with a mesh
  (`tile::MESH_RANGE`, `mesh_in`): every second sample within 32 of the
  diamond's choice, measured on every other row, where the choice leaves
  more than 1.5 a sample of SAD (`MESH_ERROR`) in few samples (its squared
  error over 8 times its SAD, `MESH_SPARSE`: grain spreads its error, and
  a mesh seldom finds it a better vector). A `psadbw` gives both halves
  of a 16-wide row (`cutlass_simd::vp9::sad8::halves4`), so one pass
  keeps the best two of the block and of each 8×8 quarter; both are
  measured whole, the quarters' at the quarters' own searches, which take
  no mesh of their own. GOLDEN and ALTREF take one only where LAST's did.
  libvpx's slower speeds search meshes too (`full_pixel_exhaustive`).
- From speed 4, after the best mode, one transform size smaller is tried;
  speeds 0 to 3 try two. At a fixed quantizer in one pass, the second is
  now tried where the first won.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V):

| | Before | Mesh, every reference | And the transform | Mesh on GOLDEN and ALTREF after LAST's (now) |
| --- | --- | --- | --- | --- |
| `testsrc2` | +8.7 / +8.0 / +2.7% | +2.9 / +1.7 / −2.5% | +2.4 / +1.8 / −3.4% | +3.1 / +2.2 / −2.9% |
| Gradient | +3.6% | +3.6% | +3.6% | +3.6% |

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, FFmpeg | Before | Now | Gradient, FFmpeg | Before | Now |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 357003, 54.36 dB | 333472, 52.72 dB | 324395, 52.76 dB | 17767, 55.10 dB | 12916, 53.06 dB | 13000, 53.05 dB |
| 25 | 272682, 50.37 dB | 250075, 48.80 dB | 242857, 48.89 dB | 14431, 53.64 dB | 9557, 51.57 dB | 9559, 51.57 dB |
| 35 | 175644, 45.59 dB | 150857, 43.60 dB | 145479, 43.67 dB | 8704, 51.31 dB | 7238, 49.42 dB | 7196, 49.39 dB |
| 45 | 87402, 40.61 dB | 64750, 38.83 dB | 61447, 39.00 dB | 6428, 48.40 dB | 5996, 46.71 dB | 5996, 46.71 dB |
| 55 | 33071, 37.41 dB | 22915, 35.27 dB | 21703, 35.53 dB | 5395, 45.09 dB | 5620, 43.54 dB | 5620, 43.54 dB |

Speed 2 on `testsrc2`: +2.0 / −2.6 / −7.7% → −1.9 / −6.4 / −11.2%. The
gradient's search never meets the mesh's bounds (its error is small and
spread), and its streams barely change. Five points of BD-rate on
`testsrc2` move by up to a point between builds that differ in little;
with nine (`crf=15` to `55` by 5, against the same five of FFmpeg's) it is
+10.7% before and +5.1% now.

On the README's three clips at `q=60` to `210` against the build before
(BD-rate Y / U / V / FFmpeg's average / 6:1:1):

| Fixed q | Three scenes | Still scene | Grainy pan |
| --- | --- | --- | --- |
| Now | −3.1 / −2.1 / −2.0 / −2.7 / −2.8 | −0.8 / −0.1 / +0.0 / −0.4 / −0.6 | −0.2 / +0.1 / +0.0 / −0.1 / −0.1 |

Against FFmpeg's `-crf` there: −11.9 / −22.0 / −23.0 / −16.4 / −14.7 on
the three scenes, −17.0 / −49.1 / −48.9 / −39.0 / −31.2 on the still
scene and −46.0 / −60.4 / −60.5 / −55.0 / −51.3 on the pan (−9.3, −17.8
and −45.9 at the same PSNR-Y before, measured against the same FFmpeg
curves). Every other mode (speeds 5 to 10 at a bitrate, two passes)
writes the same streams as before.

Time, four threads, against FFmpeg's `-crf 24` to `48` (`-threads 4`) and
the build before, three rounds interleaved, the mean over each one's
points of the median of the rounds (Cutlass at `q=60` to `210`):

| | FFmpeg | Before | Now |
| --- | --- | --- | --- |
| Three scenes | 6.74 s | 4.12 s (0.61) | 4.63 s (0.69) |
| Still scene | 3.87 s | 3.30 s (0.85) | 3.24 s (0.84) |
| Grainy pan | 2.71 s | 2.25 s (0.83) | 2.24 s (0.83) |

At the same bitrate (each one's time interpolated in log rate over the
range both cover; FFmpeg's time hardly falls below 2.3 s at its lowest
rates), 0.70, 1.06 and 1.04 now, 0.62, 1.07 and 1.06 before. In pairs on
the pan alone (`q=120`, six runs each) the two changes cost 2%; the three
scenes, whose busy blocks take the mesh most, 12% (above).

Checks: of the identity check's 41 configurations, the seven at a fixed
quantizer (speeds 2 and 5 on three pictures, the still scene at `q=120`)
change, the other 34 (every bitrate, two passes, speeds 6 to 10) do not.
FFmpeg's `vp9` (threaded and on one thread), libvpx and this decoder agree
on 10 new streams (40 of 40 decodes: `testsrc2` at `crf=35`, at `crf=25`
at speed 4, at `crf=30` at speed 2 and at `crf=40` on one thread, the
gradient at `crf=35`, 4:4:4 at `q=60`, the pan at 10 bits at `q=100` and
at `q=120`, the still scene at `q=80`, the three scenes at `q=150` on two
threads). New tests: a textured 6×6 square moved 12 and 14 samples over a
flat picture codes in 24 bytes at a fixed quantizer (70 without the mesh,
as a second pass); a picture of specks takes inter transforms two sizes
under the block's in one pass and none in a second (each fails without its
change); `halves4` matches its scalar twin.

Tried and dropped (BD-rate on `testsrc2` at speed 5 against FFmpeg's
`-crf`, from +8.7%, unless said):

- The mesh at every second sample within 48 (+2.3%) or 64: the stop that
  `a_reference_seed_reaches_past_the_search_range` reads (a seeded search
  takes half the bytes of an unseeded one) no longer holds: within 48 the
  mesh finds most of a 100-sample jump without the seed. Within 36 and
  dense to 16, every fourth beyond: +2.8% for half the points. Within 32
  at every fourth sample: +3.7%.
- The mesh on every reference (+2.4% with the transform's change): on the
  grainy pan 7 to 10% of the time. Its gate at SSE over 12 or 16 times
  the SAD: as many meshes on the pan, `testsrc2` 0.6 points worse; at 0.5
  or 0.25 a sample of SAD: no better.
- A mesh at every block size, on every row (+4.0%, 1.3 to 1.4 times the
  time); one only where the diamond's last four neighbours come within 0
  to 5% of its choice (the plateau the mesh is for), with the vectors
  below as starts: +5.2% to +7.0% against +4.0% without that gate.
- The previous frame's most common vectors on each reference as starts of
  the search: the gradient +3.6% → +1.0% (from ALTREF's alone +1.5%),
  `testsrc2` +7.8%, but the still scene at a fixed quantizer +0.7 to +2.5
  points of PSNR-Y, with or without a rate for them and a floor on their
  count. On the gradient they find the periodic plane's aliases for the
  content that enters at the right edge, from an alt-ref far behind.
- No alt-ref: the gradient −10.5%, `testsrc2` +3.3%, but the three clips
  at a fixed quantizer +5.2, +5.3 and +17.6 points. No alt-ref where the
  filtered picture is within 0.06 a sample (squared) of its target
  (nothing to filter, as on both synthetic sources): the gradient −10.5%,
  the three scenes +0.9 and the still scene +0.6 points; with the pool's
  noise under 1 too, the still scene +2.4. Most of the gradient's gain is
  again the old reference's periodic aliases.
- Key frames at 0.25 of the step: `testsrc2` +3.0% (from +2.9%), the
  gradient +4.0%. The loop filter's fixed-quantizer scale at 0.45 and
  0.3: the gradient +3.8% and +5.7%. The 8×8 peak on whole blocks that
  code a residual: a full search never stops there.

What is left: `testsrc2` at +3.1% (+2.2% / −2.9% at PSNR-U / V), most of
it at the low rates (at the same PSNR-Y, +8% to +15% from `crf=35` up),
where libvpx's squares of noise and its key frame are better (on its
first picture alone speed 5 is at +8.1%, speed 0 at +5.6%: a vertical
edge of a bar left in a V_PRED 32×32 block without a residual at `q=55`,
which libvpx codes at 47, and copied down 200 rows and through every
picture after); the gradient at +3.6%.

**A key frame's intra search, and the golden frame an overlay kept**
(17 November 2026, after `cf5a1a10`). What was left: `testsrc2` at +3.1%
/ +2.2% / −2.9% against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V, `crf=15` to `55`, speed 5, four threads), the gradient at +3.6%. The
first picture of each coded alone (`crf=10` to `60`) was at +8.1% / +4.9%
/ +2.2% and +4.3% against libvpx's (speed 0: +6.0% and +2.9%); at the
same quantizer (68 and 70) libvpx's key frame of `testsrc2` took 9 138
bytes for 51.84 dB, this encoder's 9 358 for 50.77 dB, with the excess
error spread over the squares of noise and the rest alike. Decoded with
each block's modes printed, and the key frame's features turned on one
at a time, three causes, and in the inter frames a fourth:

- A split 8×8 of a key frame took each 4×4's mode by SATD alone (with a
  little for leaving DC). libvpx's `rd_pick_intra_sub_8x8_y_mode` codes
  each and keeps the least RD cost. Now, at a fixed quantizer in one pass
  up to speed 5 (`KEY_RD_SPEED`), each 4×4 takes the best by RD of the
  three best by SATD (`KEY_RD_4X4`), its mode's bits counted in the
  context of the 4×4s above and left of it: the first picture of
  `testsrc2` +8.1% → +6.5% (all ten modes: +6.1%).
- A block's luma modes were the four best by SATD; on the smooth
  gradient SATD ranks the mode that wins fifth or sixth. Six now
  (`KEY_RD_CANDS`): the gradient's first picture +4.3% → −1.3% (ten: the
  same), `testsrc2`'s +8.1% → +7.9%.
- The best intra mode then tried one more chroma mode, the luma's (or TM
  where that was DC): the chroma mode was mostly DC. A key frame's best
  now tries the best chroma mode by the SATD of both planes besides DC
  (`Tile::uv_ranked`): with the two above, `testsrc2`'s first picture
  +6.4% / +3.9% / +1.9% → +4.6% / +2.5% / −0.4% (every chroma mode: +4.6%
  / +1.9% / −0.6%, in twice the key frame's time).
- After an alt-ref's overlay GOLDEN read the overlay and ALTREF the
  alt-ref, the same picture, and the golden frame before was gone: the
  overlay was coded into slot 1, over it. libvpx keeps that golden frame
  as the new ALTREF (`vp9_preserve_existing_gf`). At a fixed quantizer in
  one pass the overlay is now coded into the alt-ref's slot and becomes
  GOLDEN, and the golden frame before it, in the other slot, ALTREF; the
  next alt-ref takes ALTREF's slot (`OVERLAY_KEEPS_GOLDEN`). The pool's
  pick of GOLDEN (`Core::pick_golden`) started from slot 1 whatever held
  the golden frame; it starts from its slot.

SATD ranks every intra trial and was a scalar loop: 9.8% of a key frame's
time on the grainy pan, 3.7% of its first 30 pictures. It runs on AVX2
now for 8-bit squares (`cutlass_simd::vp9::sad8::satd`: a 16-wide row a
vector, the last butterfly as `2 max(|p|, |q|)`), the same sums. A 720p
key frame of the pan (one thread, `q=120`) takes 0.95 s, 0.74 s before.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V; in brackets with nine points, `crf=15` to `55` by 5, against the same
five of FFmpeg's):

| | Before | Key frames | And the overlay (now) |
| --- | --- | --- | --- |
| `testsrc2` | +3.1 / +2.2 / −2.9% (+5.1 / +3.1 / −2.9%) | +2.8 / +3.4 / −2.8% (+4.7 / +3.5 / −3.0%) | +2.5 / +2.3 / −3.4% (+4.4 / +2.7 / −3.6%) |
| Gradient | +3.6% (+3.2%) | +2.4% (+1.6%) | +3.0% (+2.6%) |

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, before | Now | Gradient, before | Now |
| --- | --- | --- | --- | --- |
| 15 | 324395, 52.76 dB | 323861, 52.78 dB | 13000, 53.05 dB | 13042, 53.24 dB |
| 25 | 242857, 48.89 dB | 241204, 48.88 dB | 9559, 51.57 dB | 9530, 51.64 dB |
| 35 | 145479, 43.67 dB | 144495, 43.67 dB | 7196, 49.39 dB | 7128, 49.25 dB |
| 45 | 61447, 39.00 dB | 61280, 39.03 dB | 5996, 46.71 dB | 5939, 46.51 dB |
| 55 | 21703, 35.53 dB | 21811, 35.44 dB | 5620, 43.54 dB | 5641, 43.35 dB |

The overlay's change gives back 0.6 points of the gradient's gain (within
what five points move between builds that differ in little) for a point
on `testsrc2` and the clips' chroma below. On the README's three clips
at `q=60` to `210` against the build before (BD-rate Y / U / V / FFmpeg's
average / 6:1:1):

| Fixed q | Three scenes | Still scene | Grainy pan |
| --- | --- | --- | --- |
| Key frames | −0.4 / −0.1 / −0.2 / −0.3 / −0.4 | −2.8 / −1.6 / −1.8 / −2.3 / −2.5 | −2.0 / −0.5 / −0.6 / −1.3 / −1.6 |
| And the overlay (now) | −0.2 / −0.8 / −0.7 / −0.4 / −0.3 | −2.7 / −4.3 / −4.5 / −3.5 / −3.2 | −2.2 / −2.2 / −2.2 / −2.2 / −2.2 |

Every other mode (speeds 5 to 10 at a bitrate, two passes) writes the
same streams as before.

Time, four threads, against FFmpeg's `-crf 24` to `48` (`-threads 4`) and
the build before, two rounds interleaved, the mean over each one's points
of the median of the rounds (Cutlass at `q=60` to `210`):

| | FFmpeg | Before | Now |
| --- | --- | --- | --- |
| Three scenes | 7.40 s | 4.97 s (0.67) | 5.02 s (0.68) |
| Still scene | 4.12 s | 3.50 s (0.85) | 3.45 s (0.84) |
| Grainy pan | 2.84 s | 2.35 s (0.83) | 2.51 s (0.89) |

In three more rounds on the pan alone, 1.02 of the build before (the key
frame: one in 60 pictures, coded on fewer threads than the rest).

Checks: of the identity check's 41 configurations, the seven at a fixed
quantizer change, the other 34 (every bitrate, two passes, speeds 6 to 10)
do not. FFmpeg's `vp9` (threaded and on one thread), libvpx and this
decoder agree on 10 new streams (40 of 40 decodes: `testsrc2` at
`crf=35`, at `crf=30` at speed 2 and at `crf=40` on one thread, the
gradient at `crf=35`, the pan at `q=120` and at 10 bits at `q=100`, the
still scene at `q=80`, the three scenes at `q=150` on two threads, 4:4:4
at `q=60`, 4:2:0 at speed 4). New tests: a key frame of horizontal luma
stripes over vertical chroma ones takes V (or TM) for chroma in one pass
and never in a second; after an overlay ALTREF reads slot 1 in one pass
and slot 2 in a second (each fails without its change); `satd` matches its
scalar twin.

Tried and dropped (BD-rate against FFmpeg's `-crf`, from the build of
each step; for the first picture alone, as above):

- The key frame's lambda × 0.5 / 0.7 / 1.4: its first picture of
  `testsrc2` +6.8% / +7.1% / +11.0%, the gradient's +3.6% / +3.3% /
  +7.2%; × 0.7 with the rest, the still scene's chroma +3.1 points at a
  fixed quantizer.
- The key frame's loop filter level × 0.3 / 0.6 / 1.5: first pictures
  +7.9% / +8.0% / +8.7% (`testsrc2`), +10.2% / +7.1% / +4.0% (gradient).
  The inter frames' level forced to 4, 8, 12 or 16 (libvpx's are 3 to 10
  on the gradient, ours 23): the gradient +5.8%, +3.6%, +3.2%, +6.1%, its
  pictures' decay unchanged.
- An 8×8 of a key frame in two halves, 8×4 or 4×8, their modes by RD:
  `testsrc2`'s first picture +4.6% → +4.2%, for 0.3 s of a 0.74 s key
  frame. Every transform size for every candidate: worse (+6.4%).
- The overlay into the alt-ref's slot with GOLDEN left on the golden
  frame before and ALTREF the overlay (the slots not traded): the gradient
  −3.0%, `testsrc2` +1.6% / +1.4% / −3.9%, but the still scene +1.3 / +2.3
  / +2.4 points at a fixed quantizer. The gradient's gain is the old
  picture searched near zero, where it finds the periodic plane's aliases
  of the content entering at the edges (4% to 26% of the area of the
  pictures that use it); ALTREF's search starts where the old picture's
  content stands. Gates on GOLDEN's share before the overlay (10%, 5% and
  35% on the gradient, `testsrc2` and the still scene), on the picture
  having moved since, or not jumped since, kept the loss. GOLDEN seeded
  from where the overlay stands: the gradient +2.4%, `testsrc2` +2.3% /
  +2.0% / −3.2%.

Not pursued this time: the previous frame's popular vectors as starts
(the still scene's loss in "A mesh after the diamond").

What is left: `testsrc2` at +2.5% / +2.3% / −3.4%, the gradient at +3.0%.
Without alt-refs the gradient is at −11.4%, for 5 to 18 points lost on
the clips; most of that gain is again an old reference's aliases.

**The alt-ref's reference prune, the previous frame's vectors, and
drawn pictures** (17 November 2026, after `b2982b7d`). What was left:
`testsrc2` at +2.5% / +2.3% / −3.4% against FFmpeg's `-crf` (BD-rate at
the same PSNR-Y / U / V, `crf=15` to `55`, speed 5, four threads), the
gradient at +3.0%. Three findings, and three changes:

- The previous frame's most common vectors as starts of the search had
  taken the gradient to +1.0% (in "A mesh after the diamond"), but cost
  the still scene 0.7 to 2.5 points of PSNR-Y at a fixed quantizer, for
  no reason found then. At `q=120` its stream grew by 6.5 kB, and all of
  it was one hidden alt-ref (after the golden frame of the second jump):
  19 818 bytes before, 26 277 with the starts. Decoded with each block's
  reference and vector printed, the alt-ref before coded its 64×64 blocks
  from ALTREF at (−84, −76) samples without a residual (the scene had
  come back to an earlier place); with the starts it never searched
  ALTREF. ALTREF had predicted under 1% of every inter frame before it in
  both builds, and from speed 5 such a reference is left out of the
  search but every fourth frame (`REF_PRUNE_SPEED`); the starts' own
  ALTREF wins (128 samples of the frame before) moved that count by a
  frame, and the alt-ref fell on a frame that left ALTREF out. Not a
  worse RD choice, nor the vectors' context or the probabilities: the
  phase of the prune. An alt-ref (one frame in a dozen, coded finer, and
  read by every frame up to its overlay) at a fixed quantizer in one
  pass now searches every reference: the still scene −2.1 / −1.9 / −1.7
  points (BD-rate at the same PSNR-Y / U / V), the other clips and the
  synthetic sources as before. Then the starts (`tile::POP_STARTS`, the
  two most common whole-sample vectors other than zero, each of two 8×8
  blocks or more, on each reference) take the gradient to +0.4% with no
  loss on the clips (four starts: +0.5%, and on the clips' PSNR-U and V
  up to 0.7 points less, on the pan 0.3 points less of PSNR-Y).
- libvpx's `-crf` (one pass, a constant quantizer) on both sources codes
  no alt-ref: LAST only, a key frame at 0.25 of the step (`testsrc2` at
  `-crf 45`: 99, ours 121; at `55`: 145, ours 164), and the inter frames
  at 0.5 / 1 / 0.85 / 1 / 0.7 / 1 / 0.85 / 1 of it by the frame's index in
  eights (142, 180, 172, 180, 161, …). At `crf=45` ours spent a fifth of
  `testsrc2`'s bits on its four alt-refs (3.0 to 3.4 kB each, five times
  an inter frame). libvpx's cycle here (from +2.5% / +3.0%): `testsrc2`
  +5.4%, the gradient +4.0%; with no alt-ref too, +5.3% and −10.3%; no
  alt-ref alone, +3.5% and −11.4% (the gradient's gain is the key frame's
  periodic aliases, kept as ALTREF). At the same bits (FFmpeg's `-crf 55`,
  33 071 bytes, ours `crf=51`, 33 167) the squared error of `testsrc2`'s
  luma, libvpx's and ours: blocks whose source did not change 9.1 and 14.6
  million, blocks that changed 152 and 164; the key frame alone
  (quantizers 145 and 147) 4 788 bytes at 43.07 dB and 4 197 at 40.98,
  libvpx's with more coefficients (14 016 against 11 117) and halves of
  8×8 to 32×32 blocks.
- Drawn pictures (graphics, animation, screen content) repeat the
  picture before exactly where they stand still or move as a whole; a
  camera's never do. At a fixed quantizer in one pass a picture of which
  half the 16×16 blocks or more repeat the one before, at the vector the
  whole picture moved by (`arf::exact_share`: `testsrc2` 0.66, the
  gradient 0.92, the grainy pan 0; the three scenes' `testsrc2` and a
  drawn pan at the end of the still scene, 0.93 and 0.97), or whose next
  picture repeats as much, is drawn (`DRAWN_SHARE`). Its key frame,
  whose blocks the frames after keep for as long as the picture stands,
  takes 0.3 of the inter step (not 0.35) and searches as speeds 0 to 3:
  halves at every size, trials quantized by RD (`tile::Shared::drawn_key`);
  its golden frames and alt-refs, which have no noise to average, take
  0.6 (not 0.5). libvpx's two passes size a key frame's boost from its
  share of still blocks, and search more in pictures they find drawn
  (`FC_GRAPHICS_ANIMATION`).

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V; in brackets with nine points, `crf=15` to `55` by 5, against the same
five of FFmpeg's):

| | Before | The alt-ref's prune and the starts | And drawn pictures (now) |
| --- | --- | --- | --- |
| `testsrc2` | +2.5 / +2.3 / −3.4% (+4.4 / +2.7 / −3.6%) | +2.1 / +2.2 / −3.2% (+4.1 / +2.7 / −3.8%) | +1.0 / +0.9 / −3.9% (+3.1 / +1.2 / −4.4%) |
| Gradient | +3.0% (+2.6%) | +0.4% (−0.1%) | −0.3% (−0.7%) |

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, before | Now | Gradient, before | Now |
| --- | --- | --- | --- | --- |
| 15 | 323861, 52.78 dB | 321071, 52.65 dB | 13042, 53.24 dB | 12696, 53.38 dB |
| 25 | 241204, 48.88 dB | 240212, 48.84 dB | 9530, 51.64 dB | 9267, 51.89 dB |
| 35 | 144495, 43.67 dB | 142966, 43.62 dB | 7128, 49.25 dB | 7141, 49.57 dB |
| 45 | 61280, 39.03 dB | 58143, 39.03 dB | 5939, 46.51 dB | 5984, 46.62 dB |
| 55 | 21811, 35.44 dB | 21057, 35.48 dB | 5641, 43.35 dB | 5360, 43.72 dB |

On the README's three clips at `q=60` to `210` against the build before
(BD-rate Y / U / V / FFmpeg's average / 6:1:1):

| Fixed q | Three scenes | Still scene | Grainy pan |
| --- | --- | --- | --- |
| The alt-ref's prune | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 | −2.1 / −1.9 / −1.7 / −1.9 / −2.0 | the same streams |
| And the starts | −0.1 / 0.0 / 0.0 / 0.0 / −0.1 | −2.1 / −2.4 / −2.2 / −2.2 / −2.2 | −0.7 / −0.9 / −1.0 / −0.9 / −0.8 |
| And drawn pictures (now) | −1.1 / −0.1 / −0.6 / −0.8 / −0.9 | −3.0 / −2.5 / −2.3 / −2.7 / −2.8 | −0.7 / −0.9 / −1.0 / −0.9 / −0.8 |

Every other mode (speeds 5 to 10 at a bitrate, two passes, speeds 6 to
8 at a fixed quantizer) writes the same streams as before.

Time, four threads, against FFmpeg's `-crf 24` to `48` (`-threads 4`) and
the build before, three rounds interleaved, the mean over each one's
points of the median of the rounds (Cutlass at `q=60` to `210`):

| | FFmpeg | Before | Now |
| --- | --- | --- | --- |
| Three scenes | 7.34 s | 5.13 s (0.70) | 5.19 s (0.71) |
| Still scene | 4.53 s | 3.68 s (0.81) | 3.63 s (0.80) |
| Grainy pan | 2.95 s | 2.64 s (0.90) | 2.58 s (0.87) |

Runs a few minutes apart moved these ratios by up to 0.04 (the still
scene before: 0.81 to 0.86, with FFmpeg's own time 3.9 to 4.7 s); in
pairs at `q=120` the starts cost 0.5% on the pan and 2% on the other
two clips.

Checks: of the identity check's 41 configurations, the seven at a fixed
quantizer change, the other 34 do not. FFmpeg's `vp9` (threaded and on
one thread), libvpx and this decoder agree on 10 new streams (40 of 40
decodes: `testsrc2` at `crf=35` and `55`, at `crf=30` at speed 2 and at
`crf=40` on one thread, the gradient at `crf=35`, the still scene at
`q=80`, the three scenes at `q=150` on two threads, the pan at `q=120`
and at 10 bits at `q=100`, 4:4:4 at `q=60`). New tests: after frames
that never predict from ALTREF, a hidden alt-ref of the picture ALTREF
holds codes from it wherever the prune's count stood; a band that starts
to move by 48 samples (past the diamond and the mesh) takes the vector
the rest of the picture took the frame before; a drawn picture's key
frame takes 0.3 of the step and its alt-refs 0.6, a noisy one's key
frame 0.35; a drawn key frame takes halves under 64×64 (each fails
without its change).

Tried and dropped (BD-rate on `testsrc2` / the gradient against FFmpeg's
`-crf`, from +2.1% / +0.4% after the starts, unless said):

- RD quantization's lambda on key frames × 0.5 and × 0.25, on every
  frame × 0.7 and × 0.5, the RD lambda of key frames × 0.5 and of every
  frame × 0.8: `testsrc2` +2.0% at best (+3.7% at × 0.5 on every
  frame). libvpx's speed 5 does not optimize its coefficients at all;
  ours keeping fewer of them is not where the loss is.
- Key frames at 0.2, 0.25 or 0.3 of the step on every picture (not only
  drawn ones): `testsrc2` +2.0%, +2.2% and +1.6%, the gradient +0.6%,
  +1.0% and +0.6%. Golden frames and alt-refs at 0.4, 0.6 or 0.7 (before
  the starts, from +2.5% / +3.0%): +3.1%, +2.1% and +1.9%, the gradient
  +3.9%, +2.5% and +2.9%. libvpx's cycle of inter steps with ours: above.
  Only `testsrc2`'s background stays from the key frame: 15% of its 16×16
  blocks over the 60 pictures (87% from one picture to the next).
- Speed 4's trials at speed 5 (on every picture): the candidates' margin
  `testsrc2` +1.7%, the smaller transform after a block without a
  residual +1.5% (12% of the time on the pan), trials quantized by RD
  +1.5% (the gradient −0.5%), no reference prune (the gradient −0.2%);
  all four +0.7% / −1.0% in 1.21 to 1.24 times the time. Halves at
  every size (speeds 0 to 3): +0.4% alone in twice the time; with the
  four, −0.6% / −1.6%. Speed 4 itself: +0.1% / 0.0%; speed 3: −2.1% /
  +4.3%; speed 2: −2.6% / −4.8%.
- On drawn pictures' inter frames (with the quantizers above unless
  said): halves at every block that changed, with the margin and the
  dead-zone trials left out there, the quantizers as before: +0.4% /
  −0.7%, in 1.35 times the time on the three scenes and 1.11 on the
  still scene at `q=120` (a third of each is drawn). Halves only where a
  block covers both blocks that changed and blocks that did not, with a
  drawn key frame's search on the alt-refs too: −0.1% / −1.5%, the
  three scenes −1.1 points more, but 0.90 and 0.91 of FFmpeg's time on
  the still scene in two measures (0.85 and 0.86 before). Halves only in
  the direction that parts what changed from what did not: −0.1% /
  −1.4%, 0.97; the margin and the dead-zone trials alone where blocks
  changed: +0.6% / −0.3%, 0.95 (the trials quantized by RD at `q=60`
  most of it).
  Halves after a split at 32×32 and 64×64 (as `RECT_SPEED` at 16×16, the
  quarters' vectors, no search): +0.8%. The alt-refs' halves and
  trials as a drawn key frame's: +0.7% / −0.7% in one measure, 4% more
  time on the still scene in it.

What is left: `testsrc2` at +1.0% / +0.9% / −3.9% (nine points: +3.1%),
most of it at the low rates (at FFmpeg's `-crf 55`, 0.36 dB of PSNR-Y
under at the same bits; within 0.11 dB at the other points), the gradient
at −0.3% (nine points −0.7%). The halves and the trials that would take
`testsrc2` past libvpx cost more time than the still scene's drawn pan
allows.

**The coarse end, and drawn pictures that stand still** (17 November
2026, after `2999cff7`). What was left: `testsrc2` at +1.0% / +0.9% /
−3.9% against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U / V,
`crf=15` to `55`, speed 5, four threads; nine points +3.1%), the
gradient at −0.3%. Against FFmpeg at more points (`-crf 15` to `61`, by
one or two at the coarse end), the bits this encoder took at the same
PSNR-Y were within −0.7% and +2.9% from `crf=15` to `45`, but 10% over
at `crf=50` (37.5 dB) and 20% over at `crf=55`. Decoded with each block's
mode and residual bits printed (the cost of each symbol at its
probability), four findings, and four changes:

- From quantizer 208 or so the loop filter's level from the quantizer is
  63, past what the masks can be rebased from (2 to 61), and the level
  was not searched at all: `testsrc2`'s inter frames at `crf=55` were
  filtered at 63, libvpx's at 3 to 16. At a fixed quantizer in one pass
  the search now starts from 61 at most (`LF_SEARCH_CAP`): `testsrc2` at
  `crf=55` 21 057 bytes at 35.48 dB → 19 572 at 35.65 dB.
- libvpx codes vectors in quarters of a sample from quantizer 200
  (`HIGH_PRECISION_MV_QTHRESH`); this encoder allowed eighths at every
  quantizer. At a fixed quantizer in one pass it now does as libvpx
  (`HP_MAX_Q`): at `crf=50`, 1.5 points fewer bits at the same PSNR-Y
  (from 150 on, the alt-refs at 172 lost as much).
- At `crf=50` against FFmpeg's `-crf 55`, at the same PSNR-Y, this
  encoder's inter frames coded the picture at least as well as libvpx's
  by region (squared error and bits of the blocks that did not change,
  that changed, and of the squares of noise), but a fifth of its bits
  went on its four hidden alt-refs; libvpx's `-crf` codes none. With no
  alt-ref `testsrc2` took fewer bits from `crf=40` down and more above,
  and the gradient 15% fewer at every point; but without them on every
  drawn picture from `q=160` the still scene, whose drawn pan moves, lost
  1.6 points of PSNR-Y (BD-rate at `q=60` to `210`). A drawn
  picture (`DRAWN_SHARE`) is now also measured where it stands: the
  share of its 16×16 blocks the picture before holds exactly at no
  vector (`testsrc2` 0.81, where `arf::shift` finds the vector of what
  moves, (6, −4); the gradient 0; the still scene's drawn pan 0.12 to
  0.28). One that stands still (half its blocks or more, at the start of
  a golden interval and at its end) codes no alt-ref from quantizer 160
  (`STILL_ARF_Q`; from 140, `testsrc2` +0.7%; from 180, +3.0% at nine
  points).
- At `q=192` against FFmpeg's `-crf 52` (as many bits), this encoder's
  squared error outside the squares of noise was 38 against libvpx's 28
  million, and half the difference was the counter's digits, which
  change every picture: 12.0 against 6.8 million. libvpx codes the rows of a glyph that moved
  by a sample as 4×4 blocks with vectors of their own, without a
  residual; this encoder kept the 8×8 block's vector and its error (39 000
  where libvpx's 1 000), as blocks under 8×8 are tried only up to speed 1
  (`SUB8_SPEED`). In an inter frame of a drawn picture that stands still,
  at a fixed quantizer in one pass, the 16×16 blocks that changed from
  the picture before (`arf::still_blocks`, `tile::Shared::still`) now
  try them: each 8×8 block there its 4×4, 8×4 and 4×8 sub-blocks with the
  whole block's reference, as speeds 0 and 1; and each 32×32 or 64×64
  block that covers both blocks that changed and blocks that did not
  tries its halves after the whole block, as speeds 0 to 3 and a drawn
  key frame. Where nothing changed nothing is tried, and a drawn picture
  that moves (a pan) tries none.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V; in brackets with nine points, `crf=15` to `55` by 5, against the same
five of FFmpeg's):

| | `testsrc2` | Gradient |
| --- | --- | --- |
| Before | +1.0 / +0.9 / −3.9% (+3.1 / +1.2 / −4.4%) | −0.3% (−0.7%) |
| The filter's level and quarters | +0.7 / +0.5 / −4.6% (+3.0 / +1.0 / −4.4%) | −0.2% (−0.8%) |
| And no alt-ref on still pictures | +0.3 / −0.1 / −4.7% (+2.7 / +0.7 / −4.9%) | −0.2% (−0.8%) |
| And blocks under 8×8 and halves (now) | −1.7 / −4.2 / −8.7% (+1.3 / −2.1 / −7.4%) | −0.2% (−0.8%) |

With nine points of this encoder's at `crf=17` to `57` (the grid moved
by two) against the same five of FFmpeg's, PSNR-Y: +3.9% before, +3.1%
without the alt-refs, +1.7% now. Blocks under 8×8 alone: −0.9 / −0.5 /
−5.3% (+1.9%); halves alone: −0.5 / −3.6 / −8.0% (+2.4%).

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, before | Now | Gradient, before | Now |
| --- | --- | --- | --- | --- |
| 15 | 321071, 52.65 dB | 319877, 52.70 dB | 12696, 53.38 dB | 12696, 53.38 dB |
| 25 | 240212, 48.84 dB | 239847, 48.89 dB | 9267, 51.89 dB | 9267, 51.89 dB |
| 35 | 142966, 43.62 dB | 142253, 43.70 dB | 7141, 49.57 dB | 7141, 49.57 dB |
| 45 | 58143, 39.03 dB | 56002, 39.11 dB | 5984, 46.62 dB | 5984, 46.62 dB |
| 55 | 21057, 35.48 dB | 18831, 35.85 dB | 5360, 43.72 dB | 5426, 43.71 dB |

On the README's three clips at `q=60` to `210` against the build before
(BD-rate Y / U / V / FFmpeg's average / 6:1:1):

| Fixed q | Three scenes | Still scene | Grainy pan |
| --- | --- | --- | --- |
| The filter's level and quarters | −0.6 / −1.2 / −0.9 / −0.8 / −0.7 | −0.7 / −0.5 / −0.5 / −0.6 / −0.7 | −0.6 / −0.7 / −0.7 / −0.7 / −0.6 |
| And no alt-ref on still pictures | −0.2 / −0.6 / −0.4 / −0.3 / −0.3 | −0.7 / −0.5 / −0.5 / −0.6 / −0.7 | −0.6 / −0.7 / −0.7 / −0.7 / −0.6 |
| And blocks under 8×8 and halves (now) | −0.2 / −1.2 / −0.8 / −0.5 / −0.4 | −0.7 / −0.5 / −0.5 / −0.6 / −0.7 | −0.6 / −0.7 / −0.7 / −0.7 / −0.6 |

(The first two changes act at `q=210` only; the three scenes' `testsrc2`
loses 0.4 points of PSNR-Y without its alt-refs at 180 and 210.) Every
other mode (speeds 5 to 10 at a bitrate, two passes, speeds 6 to 8 at a
fixed quantizer) writes the same streams as before.

Time, four threads, against FFmpeg's `-crf 24` to `48` (`-threads 4`) and
the build before, three rounds interleaved, the mean over each one's
points of the median of the rounds (Cutlass at `q=60` to `210`; the still
scene and the pan from a second run, the first having put both builds
there at 0.92 to 0.94 of FFmpeg's time, FFmpeg's own 4.1 s):

| | FFmpeg | Before | Now |
| --- | --- | --- | --- |
| Three scenes | 7.21 s | 5.09 s (0.71) | 5.64 s (0.78) |
| Still scene | 4.44 s | 3.60 s (0.81) | 3.61 s (0.81) |
| Grainy pan | 3.08 s | 2.68 s (0.87) | 2.65 s (0.86) |

The three scenes' 11% is their `testsrc2`, a third of the clip; the still
scene's drawn pan moves and tries nothing more.

Checks: of the identity check's 41 configurations, four at a fixed
quantizer change (the synthetic 4:2:0 and 4:4:4 sources at `q=60`,
speeds 2 and 5), the other 37 do not. FFmpeg's `vp9` (threaded and on
one thread), libvpx and this decoder agree on 10 new streams (40 of 40
decodes: `testsrc2` at `crf=35`, `50` and `55`, at `crf=45` on one
thread and at `crf=30` at speed 2, the gradient at `crf=55`, the three
scenes at `q=180` on two threads, the still scene and the pan at
`q=210`, 4:4:4 at `q=200`). New tests: an inter frame at quantizer 230
searches its filter level; from quantizer 200 a fixed quantizer's
inter frames code no eighths, under it and in a second pass they do; a
still drawn picture codes no alt-ref from `STILL_ARF_Q`, under it it
does, and a drawn pan does above it; a still drawn picture's changed
blocks try sub-blocks under 8×8 with vectors of their own, and the
halves of blocks over what changed and what did not (each fails without
its change).

Tried and dropped (BD-rate on `testsrc2` against FFmpeg's `-crf`, five
points [nine], from the step before unless said):

- The first picture alone (of three, so that it is found drawn) is 3.2%
  over libvpx's at every rate, and as much at speeds 0 and 3: not the
  search. At the same quantizer (146 and 145) its squared error is 1.5
  times libvpx's for 13% fewer bits, most of it in the squares of noise.
  Dead-zone quantization on key frames, as libvpx's speed 5 (no
  trellis): +4.9% on the picture (rounding 0.375 the same, 0.3 +5.9%).
- From `q=180` on drawn pictures: alt-refs at 0.75 and 0.9 of the step
  (not 0.6), +0.4% [+3.1%] and +0.6% [+3.3%]; key frames at 0.25 and
  0.35 (not 0.3), +1.3% and +1.1% [+3.5%]; the inter frames' loop filter
  at 0.4 or 0.8 of the level found (not 0.6), and the level searched on
  every superblock row (not one in four; steadier, but +1.4% [+1.4%]
  after the last change against −1.7% [+1.3%]): within the noise. No
  compound prediction from 180: +1.1% [+3.5%]; never: +2.0% [+4.3%].
  `arnr_strength=0`: +0.4% [+3.0%]; one picture filtered: +0.8% [+3.0%].
- RD lambda × 1.3 and × 0.8 on inter frames from 180: within the noise.
  × 1.5 and × 2 in 8×8 blocks of variance over 1000 (the squares of
  noise): +3.0% and +5.4%; × 0.7 and × 0.85, within the noise.
- With no alt-ref on still pictures: libvpx's cycle of inter steps (0.5
  / 1 / 0.85 / 1 / 0.7 / 1 / 0.85 / 1): +1.8% (its square root +0.7%);
  golden frames at 0.5, 0.65, 0.75, 0.82 or 0.9 of the step (not 0.6):
  +0.8%, +0.1%, −0.2%, +0.3%, −0.1% (+2.8% to +3.2% on the moved grid):
  noise.
- In still pictures' changed blocks, after the halves: halves only in
  the direction that parts what changed from what did not, −0.1% [+2.6%]
  against −0.5% [+2.4%]; with the candidates' margin left out there too,
  −1.5% [+1.2%] (+1.1% on the moved grid, −3.1 / −8.6% chroma) against
  −1.7% [+1.3%] for more trials; with the dead-zone trials left out too,
  as the round before: 1.4 times the time on the three scenes.

What is left: `testsrc2` at −1.7% / −4.2% / −8.7%, with nine points
+1.3% (the coarse end: from `crf=40`, 37.4 to 40.4 dB, five to eight
points over FFmpeg's curve at the same PSNR-Y, the first picture 3% over
libvpx's at every rate), the gradient at −0.2% (nine points −0.8%).

**Squares of noise, and the smooth filter** (17 November 2026, after
`8b877186`). What was left: `testsrc2` at −1.7% / −4.2% / −8.7%, +1.3%
with nine points. Two findings about the measure first. The nine points
are this encoder's at `crf=15` to `55` by 5 against FFmpeg's five (15 to
55 by 10), and FFmpeg's own nine against its five come to +2.1% / +2.2%
/ +2.2%: its points at `-crf 40` and `50` lie 7.3% and 7.7% over the
chords between its five, so the denser curve loses about two points to
the sparser one whoever draws it. Against FFmpeg's own nine, this
encoder was at −1.0% / −4.4% / −9.6%. And a 5% cut in bits at one point
moves the nine-point figure by 0.3 (at `crf=15` and `50`) to 1.0 points
(`crf=35`): the middle of the range carries it.

The first picture alone (of three, so that it is found drawn) was 3.2%
over libvpx's, and FFmpeg's `libvpx-vp9` runs at its speed 1 (FFmpeg's
default `-cpu-used`), not 5: trellis quantization, and rectangular
partitions on key frames. libvpx's lambda on a key frame is the DC step
squared times 4.35 plus a thousandth of the quantizer
(`vp9_compute_rd_mult`), 0.0022 times the step squared for a bit at the
scale of the squared error (its trellis 1.25 times that on luma, 0.75 on
chroma); this encoder's is 0.00358, 1.6 times libvpx's. At 0.6 times,
the first picture is 1.4% over libvpx's (0.4 times, 1.6%; 0.3, 1.9%;
1.5, 6.0%), and with 8×4 and 4×8 intra blocks under 8×8 too (which
libvpx codes for 107 of its 734 blocks, this encoder for none) 0.6%.
But the whole stream did not move with either: nine points +1.1% to
+1.5% with the lambda at 0.4 to 0.75 times, +1.3% with the blocks, both
+1.1%.

Most of `testsrc2`'s bits go on its squares of noise (160×80 samples of
noise in 16×16 squares, redrawn every picture): in the inter frames at
`crf=35` against FFmpeg's `-crf 38`, 96 of about 120 kB, and 30 of 36
million of the squared error. Cut out alone (192×96, nine points each
side), this encoder was at +0.9% / +0.8% / −7.3%, and 5.0% and 6.7%
over FFmpeg's curve at `crf=35` and `40`. At the same PSNR-Y there
(33.4 dB), libvpx's inter frames take 16.5 kB of modes and 94.3 kB of
coefficients, this encoder's 11.9 and 105.2 kB; libvpx's filter is
switchable, its blocks 43% smooth, 45% regular, while from speed 3 this
encoder's frames keep the regular filter (`SWITCHABLE_SPEED`, 2). None
of the speed-5 prunes lifted alone helped the cut-out (the candidates'
margin, the breakouts, one candidate, the subpel prune, the skipped
transform trials, the dead-zone trials, the reference prune, sub-blocks,
halves: between −0.1 and +0.8 points); speed 3 was at −0.8%.

In a drawn picture at a fixed quantizer in one pass, the inter blocks
now pick their filter up to speed 5 (`DRAWN_FILTER_SPEED`), as speeds 0
to 2 do: the best candidate tries the smooth and the sharp filter, where
its vectors leave a whole sample in some plane (on whole samples every
filter predicts alike). The choice weighs chroma's error as it is, not
twice as the trials weigh it (`tile::Shared::filter_by_luma`): weighed
twice, the smooth filter, which predicts chroma's detail better, cost
luma's, and the still scene's drawn pan (its pictures 60 to 89) lost
0.7 points of PSNR-Y for 1.1 and 1.2 of PSNR-U and V; now 0.1 for 0.9
and 1.0. The cut-out, taken as drawn: −1.0% / −1.3% / −8.5%.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V; in brackets nine points against FFmpeg's five):

| | `testsrc2` | Gradient |
| --- | --- | --- |
| Before | −1.7 / −4.2 / −8.7% (+1.3 / −2.1 / −7.4%) | −0.2% (−0.8%) |
| Each block's filter, chroma weighed twice | −2.0 / −4.8 / −9.2% (+0.9 / −3.1 / −8.2%) | −2.4% (−3.2%) |
| By luma's error and chroma's (now) | −2.2 / −5.5 / −9.1% (+0.3 / −3.6 / −8.4%) | −2.4% (−3.2%) |

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, before | Now | Gradient, before | Now |
| --- | --- | --- | --- | --- |
| 15 | 319877, 52.70 dB | 319068, 52.71 dB | 12696, 53.38 dB | 12638, 53.72 dB |
| 25 | 239847, 48.89 dB | 238127, 48.91 dB | 9267, 51.89 dB | 9381, 52.17 dB |
| 35 | 142253, 43.70 dB | 140854, 43.68 dB | 7141, 49.57 dB | 7219, 49.74 dB |
| 45 | 56002, 39.11 dB | 55956, 39.06 dB | 5984, 46.62 dB | 5933, 46.66 dB |
| 55 | 18831, 35.85 dB | 18402, 35.95 dB | 5426, 43.71 dB | 5358, 43.69 dB |

On the README's three clips at `q=60` to `210` against the build before
(BD-rate Y / U / V / FFmpeg's average / 6:1:1): the three scenes −0.5 /
−0.4 / −0.4 / −0.5 / −0.5, the still scene +0.1 / −0.2 / −0.2 / −0.0 /
+0.0, the pan unchanged (it is not drawn). On every picture, not
only drawn ones, the three clips gained −1.0 / −0.4 / +0.1 points of
PSNR-Y and −1.2 / −2.1 / −1.0 of PSNR-U, but in 6 to 11% more time,
which puts the still scene and the pan at 0.94 of FFmpeg's.

Time, four threads, against FFmpeg's `-crf 24` to `48` and the build
before, three rounds interleaved, the mean over each one's points of the
median of the rounds (Cutlass at `q=60` to `210`):

| | FFmpeg | Before | Now | Every picture |
| --- | --- | --- | --- | --- |
| Three scenes | 7.32 s | 5.76 s (0.79) | 5.88 s (0.80) | 6.51 s (0.89) |
| Still scene | 4.24 s | 3.70 s (0.87) | 3.76 s (0.89) | 3.98 s (0.94) |
| Grainy pan | 2.89 s | 2.50 s (0.86) | 2.53 s (0.88) | 2.72 s (0.94) |

Checks: of the identity check's 41 configurations, five at a fixed
quantizer change (the synthetic 4:2:0 and 4:4:4 sources at `q=60`,
speeds 2 and 5, and the noisy 4:2:0 one at speed 2), the other 36 do
not. FFmpeg's `vp9`
(threaded and on one thread), libvpx and this decoder agree on 10 new
streams (40 of 40 decodes). New test: noise and the same noise through
the smooth filter half a sample across, drawn, codes the filter per
block and takes the smooth one; not drawn, the frame keeps one filter
(fails without the change).

**The squares of noise in still pictures** (17 November 2026, after the
filter). The squares of noise are the 16×16 blocks of a still drawn
picture that changed from the picture before (`tile::Shared::still`),
where blocks under 8×8 and the halves of the larger blocks over what
changed and what did not are tried already. They sit a row off the grid
of 16×16 blocks (at `y = 225` and every 16 rows on), and a 16×16 block
that changed covers one cell of the map of changes, so its halves were
never tried; and an inter frame's block tries one intra mode at speed 5.
Gated on the blocks that changed (BD-rate on `testsrc2` against FFmpeg's
`-crf`, five points [nine]; time on the three scenes at `q=120` on one
core):

| Tried where a 16×16 block changed | PSNR-Y | Time |
| --- | --- | --- |
| Nothing more (the filter, before) | −2.2% [+0.3%] | 1 |
| Two intra modes | −2.6% [−0.0%] | |
| Four intra modes | −3.0% [−0.3%] | 1.04 |
| Halves at every size over what changed | −3.4% [−0.8%] | 1.17 |
| Halves of 16×16 blocks (and 32×32, 64×64 over both, as before) | −2.9% [−0.5%] | 1.12 |
| Those halves, two intra modes (now) | −3.4% [−1.0%] | 1.14 |
| Those halves, four intra modes | −4.1% [−1.4%] | 1.19 |
| Halves only after a 16×16 split won, four intra modes | −3.5% [−0.6%] | 1.07 |
| Halves from the whole block's and quarters' choices, no search, four intra modes | −2.9% [−0.4%] | |

Across only (16×8) or down only (8×16) the halves of 16×16 blocks did
+0.1% and −0.2% [nine points]. With four threads over `q=60` to `210`,
FFmpeg's three scenes at 7.29 s: the build before at 0.79 of it, two
intra modes with the halves at 0.86 (as the halves after a split with
four), four at 0.93, over the 0.9 this encoder keeps to. A 16×16 block
that changed now tries its halves before its split, with a search of
their own (`STILL_HALVES_LEVEL`), and a block over one tries two intra
modes (`STILL_INTRA_CANDS`).

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V; in brackets nine points against FFmpeg's five):

| | `testsrc2` | Gradient |
| --- | --- | --- |
| Before | −2.2 / −5.5 / −9.1% (+0.3 / −3.6 / −8.4%) | −2.4% (−3.2%) |
| Now | −3.4 / −7.2 / −11.6% (−1.0 / −6.0 / −11.0%) | −2.4% (−3.2%) |

Against FFmpeg's own nine points, −3.2 / −8.1 / −13.0%; every one of the
nine is under FFmpeg's curve (by 0.6% at `crf=50` and `55` to 5.5% at
`45`). The gradient is not still (it pans) and writes the same streams.

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, before | Now |
| --- | --- | --- |
| 15 | 319068, 52.71 dB | 316954, 52.78 dB |
| 25 | 238127, 48.91 dB | 237311, 48.99 dB |
| 35 | 140854, 43.68 dB | 140443, 43.75 dB |
| 45 | 55956, 39.06 dB | 56028, 39.13 dB |
| 55 | 18402, 35.95 dB | 18371, 35.96 dB |

On the README's three clips at `q=60` to `210` against the build before
(BD-rate Y / U / V / FFmpeg's average / 6:1:1): the three scenes −0.4 /
−0.7 / −0.8 / −0.5 / −0.5; the still scene and the pan write the same
streams (nothing in them is still and drawn at once). Against the build
before the filter: the three scenes −0.9 / −1.1 / −1.2 / −1.0 / −0.9,
the still scene +0.1 / −0.2 / −0.2 / −0.0 / +0.0, the pan the same.

Time, four threads, against FFmpeg's `-crf 24` to `48` and the build
before the filter, three rounds interleaved, the mean over each one's
points of the median of the rounds (Cutlass at `q=60` to `210`; the pan
writes the same streams in both, its 4% the noise of the measure):

| | FFmpeg | Before the filter | Now |
| --- | --- | --- | --- |
| Three scenes | 7.04 s | 5.47 s (0.78) | 6.09 s (0.87) |
| Still scene | 4.11 s | 3.55 s (0.86) | 3.64 s (0.89) |
| Grainy pan | 2.91 s | 2.54 s (0.87) | 2.45 s (0.84) |

Checks: of the identity check's 41 configurations, five at a fixed
quantizer change from the build before the filter (the synthetic 4:2:0
and 4:4:4 sources at `q=60`, speeds 2 and 5, the noisy 4:2:0 one at
speed 2), the other 36 do not (every one at a bitrate, in two passes, or
at speeds 6 to 10). FFmpeg's `vp9` (threaded and on one thread), libvpx
and this decoder agree on 10 new streams (40 of 40 decodes: `testsrc2`
at `crf=25`, `35`, `50`, at `crf=45` on one thread and at `crf=30` at
speed 4, the gradient at `crf=35`, the three scenes at `q=120` and
`180`, the still scene at `q=150`, the synthetic 4:2:0 source at
`q=60`). New tests: a still drawn picture's 16×16 blocks that changed
try their halves before their split, and a block over them two intra
modes; elsewhere none and one (each fails without its change).

What is left: `testsrc2` at −3.4% / −7.2% / −11.6% against FFmpeg's
`-crf`, −1.0% / −6.0% / −11.0% with nine points; the gradient at −2.4%
(nine points −3.2%). The first picture alone is still 3.2% over
libvpx's: libvpx's lower lambda and its 8×4 and 4×8 intra blocks take
it to 0.6%, but not the stream, and the lambda costs the README's clips
1.5 to 3 points of PSNR-U and V.

**A drawn picture's key frame** (17 November 2026, after the squares of
noise). The first picture's 3.2% over libvpx's, again: a key frame of a
drawn picture (`tile::Shared::drawn_key`: the encoder finds it drawn from
the picture queued after it) at a fixed quantizer in one pass, up to
speed 5 (`KEY_RD_SPEED`), now codes with 0.6 times the lambda
(`DRAWN_KEY_LAMBDA`), and its 8×8 blocks try 8×4 and 4×8 halves besides
their four 4×4s, each half with one mode (an 8×4's by RD over both its
4×4s; `Tile::drawn_key_rd`). On every key frame, as tried before, the
lambda cost the README's clips 1.5 to 3 points of PSNR-U and V; their
key frames are not drawn, and they now write the same streams.

The first picture of three of `testsrc2`, `crf=10` to `60`, against
libvpx's (FFmpeg's `-crf`; BD-rate at the same PSNR-Y / U / V): +3.2 /
−0.2 / −1.6% before, +0.6 / −1.1 / −1.2% now.

Speed 5 `crf=` against FFmpeg's `-crf` (BD-rate at the same PSNR-Y / U /
V; in brackets nine points against FFmpeg's five):

| | `testsrc2` | Gradient |
| --- | --- | --- |
| Before | −3.4 / −7.2 / −11.6% (−1.0 / −6.0 / −11.0%) | −2.4% (−3.2%) |
| Now | −3.9 / −8.1 / −12.2% (−1.1 / −6.6 / −11.0%) | −3.0% (−3.3%) |

(bytes of the IVF, PSNR-Y):

| crf | `testsrc2`, before | Now | Gradient, before | Now |
| --- | --- | --- | --- | --- |
| 15 | 316954, 52.78 dB | 317090, 52.83 dB | 12638, 53.72 dB | 13165, 53.86 dB |
| 25 | 237311, 48.99 dB | 237356, 49.03 dB | 9381, 52.17 dB | 9654, 52.44 dB |
| 35 | 140443, 43.75 dB | 140793, 43.81 dB | 7219, 49.74 dB | 7380, 50.00 dB |
| 45 | 56028, 39.13 dB | 56499, 39.17 dB | 5933, 46.66 dB | 6000, 47.04 dB |
| 55 | 18371, 35.96 dB | 18634, 36.15 dB | 5358, 43.69 dB | 5344, 43.96 dB |

On the README's three clips at `q=60` to `210`: the same streams as the
build before. Yet with the 8×4/4×8 trial inlined into the partition
search the still scene took 2.7% longer with the same streams: 0.905 of
FFmpeg's time against 0.882, where the build before timed against a copy
of itself gave 0.882 and 0.881. The trial now sits out of line
(`Tile::drawn_key_halves`, `#[cold]`, `#[inline(never)]`), and a split
that wins returns at once when there are no halves to try. Time, four
threads, against FFmpeg's `-crf 24` to `48` and the build before, rounds
interleaved, the mean over each one's points of the median of the rounds
(Cutlass at `q=60` to `210`; the still scene five rounds, the others
three):

| | FFmpeg | Before | Now |
| --- | --- | --- | --- |
| Three scenes | 7.96 s | 6.61 s (0.83) | 6.75 s (0.85) |
| Still scene | 4.85 s | 4.23 s (0.87) | 4.20 s (0.87) |
| Grainy pan | 3.45 s | 2.95 s (0.86) | 2.92 s (0.85) |

The three scenes' third of `testsrc2` is a drawn picture, and its key
frame now searches more.

Checks: of the identity check's 41 configurations, four at a fixed
quantizer change (the synthetic 4:2:0 and 4:4:4 sources at `q=60`,
speeds 2 and 5), the other 37 do not (every one at a bitrate, in two
passes, or at speeds 6 to 10). FFmpeg's `vp9` (threaded and on one
thread), libvpx and this decoder agree on 10 new streams (40 of 40
decodes: `testsrc2` at `crf=25`, `35`, `50`, at `crf=45` on one thread
and at `crf=15` at speed 2, the gradient at `crf=35`, the synthetic
4:2:0 source at `q=60` at speeds 2 and 5, the 4:4:4 one at `q=60`, the
three scenes at `q=120`). New test: a drawn key frame codes 8×4 and 4×8
intra blocks, its reconstruction the decoder's; not drawn, none (fails
without the change).

What is left: `testsrc2` at −3.9% / −8.1% / −12.2% against FFmpeg's
`-crf`, −1.1% / −6.6% / −11.0% with nine points; the gradient at −3.0%
(nine points −3.3%). The first picture alone is 0.6% over libvpx's.

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
  at PSNR-U: speed 6 −1.4%, speed 10 −1.6%, two passes −1.6% (−0.7%
  and −0.9% before the chroma deltas and weights above); a run on
  another number of threads (other tile columns) moves these by a
  point. Speed 6
  takes speed 5's time on the three scenes (it did before too), and
  speed 10 0.82 to 0.95 of FFmpeg's realtime 8 on four threads, after
  the loop filter's planes went to one thread (the reference pool had
  taken it to 0.90 to 1.02; see "The gaps"). Before "Chroma": speed 6 was ahead of libvpx's
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
| `-drop-threshold` | `drop-frame` | `drop_frame_pct` (`drop=`, added): one pass at a bitrate; see "Options measured" |
| `-g` | `kf-max-dist` | `keyint` |
| `-keyint_min` | `kf-min-dist` | `keyint_min`: a cut that far from the last key frame is one (added); intervals are fixed as with `-keyint_min` = `-g` |
| `-force_key_frames` | `VPX_EFLAG_FORCE_KF` | `Vp9Encoder::force_keyframe` (the engine's `force_keyframe`) |
| `-pass 1` / `-pass 2` | `g_pass`, stats | `Vp9Encoder::first_pass_qindex`, `set_second_pass` (`twopass=1`) |
| `-deadline` best / good / realtime, `-cpu-used` (`-speed`, `-quality`) | | `speed` 0–10 (`speed=`); see the speeds below |
| `-lag-in-frames` (`-rc_lookahead`) | `g_lag_in_frames` | `lag` (`lag=`) |
| `-auto-alt-ref` 0 / 1 | `enable-auto-alt-ref` | `alt_ref` (`alt_ref=0`, added) |
| `-auto-alt-ref` 2–6 | multi-layer alt-refs | `alt_ref_layers` 2 or 3 (`arf_layers=`, added): hidden alt-refs within the golden group, shown with `show_existing_frame`; see "Options measured" (they lose here) |
| `-arnr-maxframes` | `arnr-maxframes` | `arnr_max_frames` (`arnr_frames=`, added) |
| `-arnr-strength` | `arnr-strength` | `arnr_strength` (`arnr_strength=`, added) |
| `-arnr-type` | `arnr-type` | n/a: libvpx's VP9 encoder ignores it too (VP8 only) |
| `-min-gf-interval` | `min-gf-interval` | internal (golden every 12; 6 to 12 by motion in two passes), not an option |
| `-tune psnr` | `tune` | the default |
| `-tune ssim` | `tune=ssim` | `tune: Tune::Ssim` (`ssim=1`, added); see "Options measured" |
| `-tune-content screen` | `tune-content` | `tune_content: TuneContent::Screen` (`screen=1`, added); see "Options measured" |
| `-tune-content film` | `tune-content` | `tune_content: TuneContent::Film` (`film=1`, added): libvpx's RD penalties on grain lost; see "Options measured" |
| `-aq-mode 0` / `1` | `aq-mode` | `aq` (`aq=1`: variance AQ), or `aq_mode: AqMode::Variance` |
| `-aq-mode 3` / `4` | cyclic refresh, equator360 | `aq_mode`: `AqMode::CyclicRefresh` (`aq=3`), `AqMode::Equator360` (`aq=4`), added |
| `-aq-mode 2` | complexity AQ | `aq_mode: AqMode::Complexity` (`aq=2`, added): each superblock's segment by the rate its source's variance projects, before it is coded (libvpx's is the rate its search spent); see "Options measured" |
| `-noise-sensitivity` | `noise-sensitivity` | `noise_sensitivity` (`noise=`, added): a temporal denoiser of luma; see "Options measured" |
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
| `-ts-parameters` | temporal layers | `temporal_layers` 2 or 3 (`layers=`, added): libvpx's L1T2 and L1T3 patterns; spatial layers are not |
| `-corpus-complexity` | corpus VBR | `corpus_complexity` with `Vp9Encoder::first_pass_complexity` (`corpus=` with `twopass=1`, added), in this encoder's own unit; see "Options measured" |
| `-enable-tpl` | TPL model | n/a: two passes have their own allocation |
| `-pix_fmt yuv420p` … `yuv444p12le` | profiles 0–3 | `chroma`, `bit_depth` (`chroma=`, `bits=`) |
| `-pix_fmt gbrp`, `gbrp10le`, `gbrp12le` | `VPX_CS_SRGB` | `ColorSpace::Srgb` with 4:4:4: planes G, B, R (added) |
| `-pix_fmt yuva420p` | alpha as a second stream in WebM `BlockAdditional` | `alpha` (`alpha=FILE`, added): a second encoder codes the alpha plane as the luma of a 4:2:0 picture with chroma at 128, one alpha packet a packet (`Vp9Encoder::push_planes_alpha`, `flush_alpha`; on the codec contract a frame's alpha plane in, `SideData::BlockAdditional` id 1 out); `cutlass-matroska` writes it as each block's BlockAdditional (BlockAddID 1) with AlphaMode 1 and reads it back; `Vp9Decoder` decodes it into the frame's alpha (`Vp9AlphaDecoder` on its own), as FFmpeg's `libvpx-vp9` decoder does; see "Options measured" |
| `-colorspace`, `-color_range` | `color-space`, `color-range` | `color_space`, `full_range` (`colorspace=`, `range=`), and from `CodecParams` (added) |
| `-aspect` / sample aspect ratio | `VP9E_SET_RENDER_SIZE` | `render_size` (`render=WxH`, added) |
| regions of interest (side data) | ROI map | `Vp9Encoder::set_roi_map` with a `RoiMap` (`roi_q=`, `roi_ref=`, `roi_skip=` in `vp9enc`, added): a segment a block, each with a quantizer and loop filter delta, a reference and skip |

### Options measured

**Frame dropping** (`drop_frame_pct`, FFmpeg's `-drop-threshold`). One
pass at a bitrate keeps libvpx's buffer model beside its own: the buffer
starts at 4/6 of its size (`cbr_buffer_ms`, or libvpx's 6 s at a
variable bitrate), takes each frame's share of the rate in and its bits
out, and holds at most its size. While it is under the threshold (in
percent of its optimal level, 5/6 of its size) every other inter frame is
dropped, and one frame in 1 + the decimation factor is coded while it
stays under (libvpx's `vp9_test_drop`); under empty every inter frame is.
A dropped frame gives no packet (`push_planes` returns nothing for it,
`encode_planes` empty data), its share of the rate goes to the frames
after it, and they predict from the frame coded before it. Key frames,
golden frames, cuts and the frame an alt-ref comes with are never
dropped, nor a frame of two passes. libvpx drops in one-pass CBR only.
Constant bitrate with a 1 s buffer, `drop=30` against FFmpeg's
`-deadline realtime -cpu-used 8 -b:v R -minrate R -maxrate R -bufsize R
-drop-threshold 30`, four threads, the README's three clips (packets,
rate, the most a 1 s buffer held past the rate, PSNR-Y with each dropped
frame shown as the one before):

| Clip, 250 kb/s | Speed 10 | Speed 10, `drop=30` | Speed 9, `drop=30` | libvpx rt 8 | libvpx rt 8, 30 |
| --- | --- | --- | --- | --- | --- |
| Three scenes | 90, 413 kb/s, 2.25 s, 30.7 dB | 57, 281, 0.90, 27.1 | 88, 251, 0.49, 31.5 | 90, 335, 1.13, 29.7 | 74, 290, 0.78, 28.3 |
| Still and jumps | 90, 668, 5.02, 28.5 | 37, 305, 0.73, 23.7 | 90, 251, 0.33, 31.9 | 90, 435, 2.22, 28.4 | 57, 304, 0.86, 26.1 |
| Grainy pan | 60, 695, 3.56, 28.4 | 28, 337, 0.75, 24.2 | 60, 256, 0.20, 31.1 | 60, 382, 1.05, 27.9 | 45, 319, 0.58, 26.0 |

Speed 10 runs out of quantizer at this rate (it codes at the coarsest),
so it is the one that drops: it comes within 12 to 35% of the rate (from
65 to 178% over), as libvpx's realtime 8 comes within 16 to 28% (from 34
to 74% over). Speed 9 holds the rate on its own quantizers and drops at
most two frames. At 500 kb/s speed 10 drops 1, 13 and 8 frames (to 508,
572 and 603 kb/s from 508, 657 and 692), libvpx 0, 8 and 0.

**Cyclic refresh** (`AqMode::CyclicRefresh`, `-aq-mode 3`). Every inter
frame puts a tenth of the superblocks, the next ones in raster order, in
segment 1, at the quantizer that takes three times the bits (bits as the
step to the power −1.25; at most half the index finer); key frames
refresh everything and code none. libvpx's version also leaves out
blocks coded finely of late and puts a refreshed block that coded
nothing back in segment 0; this one refreshes whole superblocks in turn.
libvpx's ratio is 2: here that lost 4.0 points of PSNR-Y BD-rate summed
over the three clips at speed 10, 3 won 2.2, 4 won 6.8 but lost 6.9 on
the three scenes. BD-rate at a constant bitrate (1 s buffer, 0.5/1 to 3/4
Mb/s, four threads) against the same speed without it, PSNR-Y / FFmpeg's
average:

| | Three scenes | Still and jumps | Grainy pan |
| --- | --- | --- | --- |
| Speed 10, `aq=3` | +4.8 / +4.8% | −3.8 / −2.9% | −3.2 / −1.4% |
| Speed 9, `aq=3` | +2.9 / +2.8% | +1.7 / +2.5% | +0.8 / +3.2% |
| libvpx realtime 8, `-aq-mode 3` against 0 | +4.8 / +4.2% | −11.2 / −9.2% | −14.9 / −8.2% |
| Speed 10 `aq=3` against libvpx rt 8 `-aq-mode 3` | −9.1 / −7.2% | −16.5 / −16.7% | −15.0 / −16.4% |
| Speed 9 `aq=3` against libvpx rt 8 `-aq-mode 3` | −18.8 / −15.4% | −44.4 / −41.6% | −28.8 / −27.4% |

As in libvpx, it is for how a call looks (no block stays coarse for
long) rather than for PSNR; libvpx gains PSNR by it on two clips because
its constant bitrate without it is far behind (this encoder's speed 10
without it is 12 to 14% ahead of libvpx's realtime 8 with `-aq-mode 3`
at PSNR-Y). Time: 1.01 to 1.06 of the
time without. Weighing the refreshed superblocks' rate against their
error at their own quantizer's lambda, as region of interest maps do
(see below), gained 6.8 points summed at speed 10 but lost 9.8 at speed
9, and was left out.

**Complexity AQ** (`AqMode::Complexity`, `-aq-mode 2`). libvpx's
`vp9_aq_complexity.c`: five segments, 3 at the frame's quantizer and the
others at the quantizers of 1.75 to 2.5, 1.25 to 1.75, 1.05 to 1.25 and
0.8 to 0.9 times its rate (by strength, from the quantizer's step), and
each superblock in the first segment both of whose thresholds it is
under, of its rate over the frame's mean (0.15 to 0.75, then 2) and of
its log variance (`ln(256 var + 1)`, 6 to 9): flat, cheap superblocks
finer, the costliest coarser. libvpx's rate is the one its RD search spent on
the block at the frame's quantizer; here it is projected before the frame
is coded, from the variance of each 8×8 block of the source (the bits of
a Gaussian of that variance at the step: half the log of 1 + 12 var over
the pixel step squared, a sample), on every frame, hidden alt-refs too.
Each segment's blocks weigh rate against error at their own quantizer
(at the frame's lambda it lost 27.9 points summed over the clips, 6.1
with it). With a bitrate, as in libvpx, there is no segmentation while a
superblock's share of a frame is under 256 bits (at 720p and 30 pictures
a second, under about 1.7 Mb/s). One pass at speed 5, four threads,
BD-rate at the same PSNR-Y / FFmpeg's average:

| | Three scenes | Still and jumps | Grainy pan |
| --- | --- | --- | --- |
| `aq=2` against no AQ | +1.9 / +1.4% | +1.0 / +1.0% | +3.2 / +4.2% |
| libvpx `-aq-mode 2` against 0 | +8.5 / +9.3% | +46.1 / +38.1% | +42.8 / +31.9% |
| `aq=2` against libvpx `-aq-mode 2` | −8.0 / −8.6% | −42.1 / −45.0% | −50.7 / −55.3% |

As every AQ mode, it gives up PSNR to spend bits where coding errors
show; libvpx's (1.14) gives up far more in one pass here, in 1.8 to 2.0
times its time. Ours takes 1.04 (the pan) to 1.3 (still and jumps at 3
Mb/s) times the time without it.

**360° pictures** (`AqMode::Equator360`, `-aq-mode 4`): libvpx's bands,
the outer eighths of the rows in segment 2 at the bits of 0.6 of the
rate, the eighths next to them in segment 1 at 0.75, on every frame. On
the three scenes (not a 360° picture) at `q=120`, 0.9% fewer bits for
0.25 dB less PSNR-Y over the whole picture; libvpx's `-crf 30 -aq-mode 4`
against `-aq-mode 0` 2.7% fewer for 0.29 dB.

**Screen content** (`TuneContent::Screen`, `-tune-content screen`).
Three changes: no alt-refs (LAST predicts still text exactly, and an
alt-ref costs its bits for nothing: on the clip below, alt-refs off alone
take 20% off the bits at speed 6); inter frames below speed 10 try intra
modes when the best inter candidate's error is over 1 a sample (text and
windows that appear predict from nothing before them; speed 10 picks
intra by a rule, not a trial, and keeps its own); and a whole-sample
vector that matches exactly is not refined. Measured on a synthetic
desktop (1280×720, 90 frames: an editor whose text scrolls 3 and 6
samples a frame, a terminal that types, a moving cursor, a window
dragged across, drawn with DejaVu Sans Mono), one pass at
150 kb/s to 1.2 Mb/s, four threads, BD-rate at the same PSNR-Y / FFmpeg's
average:

| | Against the same speed | Against libvpx with `-tune-content screen` |
| --- | --- | --- |
| Speed 6, `screen=1` | −23.6 / −24.2%, same time | −3.0 / −4.0% against good, `-cpu-used 2`, in 0.40 of its time |
| Speed 9, `screen=1` | −0.8 / −0.8% | −62.0 / −62.0% against realtime 8 |
| Speed 10, `screen=1` | 0.0 / 0.0% | −57.2 / −57.2% against realtime 8 |

Speed 6 without it was 27% behind libvpx's good deadline on this clip
(libvpx's alt-refs cost it little here; ours took 14 to 28 kB a golden
interval). libvpx's own `-tune-content screen` changes nothing at the
good deadline (the same bytes) and costs 40% at realtime 8 here, where it
also overshoots the rate by 1.5 to 2.6 times.

**Film** (`TuneContent::Film`, `-tune-content film`). What libvpx's
film tuning does, in `vp9_rdopt.c`, applied in this encoder's RD search
of inter frames (speeds 0 to 6; libvpx's realtime search has none of
it): each candidate's RD cost, once it is coded, is raised by libvpx's
`rd_variance_adjustment`, where the lower of the source's and the
reconstruction's variance is at most 250 an 8×8 block (in a second pass
500 for intra and compound candidates, 2500 for DC): by 2.5 times the
share by which the reconstruction's variance falls short of the
source's (a quarter of the share it is over), at most 2.5 times, twice
that for intra and compound candidates (libvpx without film: at most
16%, from 125). Before that, on blocks from 16×16 whose 8×8 parts differ
in energy (libvpx's `vp9_get_sub_block_energy` range), an intra DC
candidate's cost is raised by the range, other intra ones by a quarter
of it, compound ones by a quarter. Grain is noise a smooth prediction
drops; the penalties keep it. Not done: libvpx's film also searches
more intra modes in its speed features, weakens its temporal filter by 2
on 32×32 blocks of variance at most 2 a sample, and scales the 250 by
the first pass's noise estimate of the group (this encoder has none).
On the grainy pan, one pass and two at speed 5 (four threads) against
FFmpeg's default and `-tune-content film`, BD-rate at the same PSNR-Y /
FFmpeg's average, and the reconstruction's share of the source's
variance in its flat 8×8 blocks (variance under 8 a sample: 77% of the
blocks) at 1 to 4 Mb/s:

| | BD-rate against the same without film | Grain kept, without → with film |
| --- | --- | --- |
| Speed 5, `film=1` | +2.6 / +5.5% | 25.5, 24.6, 25.2, 27.9% → 25.6, 24.8, 25.7, 29.0% |
| Two passes, `film=1` | +4.9 / +7.9% | 25.0, 24.4, 24.4, 25.1% → 25.3, 24.7, 25.0, 26.6% |
| libvpx `-tune-content film` | +1.7 / −0.1% | 25.2, 24.7, 24.8, 25.2% → 26.9, 25.5, 25.5, 26.3% |
| Speed 5 `film=1` against libvpx `-tune-content film` | −31.3 / −41.2% | |

Both keep a little more grain for a few percent of PSNR, ours more of it
at the higher rates and libvpx's at the lowest; ours in 0.95 to 1.09 of
the time without it.

**SSIM tuning** (`Tune::Ssim`, `-tune ssim`). libvpx's scale of the RD
lambda by 16×16 block (from the mean variance of its 8×8 blocks, by
libvpx's fit 67.035434 (1 − e^(−0.0021489 var)) + 17.492222, over the
scales' geometric mean in the frame), but a superblock's geometric mean
of them, to the power 0.5, for every block in it: with a lambda per 16×16
block the partition search weighed a block's rate against its parts' at
different lambdas, and took 33 to 46% more bits at the
same SSIM. One pass at 0.5/1 to 3/4 Mb/s, speed 5, BD-rate at the same
SSIM-Y in dB (FFmpeg's `ssim` filter) / PSNR-Y, against no tuning:

| | Three scenes | Still and jumps |
| --- | --- | --- |
| `ssim=1`, power 0.5 | +2.2 / +2.6% | −0.8 / +1.9% |
| `ssim=1`, power 1 (libvpx's) | +8.0% / — | +9.0% / — |
| libvpx `-tune ssim` against its default | +10.2% / — | −4.8% / — |
| `ssim=1` against libvpx `-tune ssim` | +1.6% | −1.6% |

Neither encoder gains SSIM by it on these clips; the grainy pan's SSIM
stops rising at 15.3 dB from 1.7 Mb/s for both, so a BD-rate there means
nothing. It is there for the option's sake, at what it costs.

**Temporal denoiser** (`noise_sensitivity`, `-noise-sensitivity`). Before
a picture is coded, each 16×16 block of its luma is pulled toward the last
denoised picture along the block's motion (whole samples: the best of the
zero vector, the block's last vector and its neighbours', refined by two
and one samples, kept over zero only when it is an eighth better), by
libvpx's `vp9_denoiser_filter` steps: a sample within 3 of the reference
(4 from level 3) takes its value, the others move by 3, 4 or 6 toward it
(one more at motion of up to 3 samples, two from level 3), and a block
that would move by more than 2 (3) a sample on average is taken back by
up to 3 a sample or left as it is; a block whose motion-compensated mean
squared difference is over 144 is left as it is. The result is coded, and
is the next picture's reference; chroma goes through as it is, as in
libvpx. On the pan with noise of 34.7 dB added (60 frames, 720p; the
clean pan is the truth), speed 9 at a constant bitrate, PSNR-Y against
the clean picture at the same rate:

| Rate | 1 Mb/s | 2 | 3 | 4 | 6 | 9 | 13 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Off | 36.26 dB | 37.84 | 38.77 | 39.14 | 38.36 | 37.58 | 37.36 |
| `noise=1` | 36.24 | 37.77 | 38.90 | 39.26 | 39.30 | 38.73 | 38.03 |
| `noise=3` | | | | 39.17 | 39.26 | 38.59 | 38.09 |

Coded without it, the picture gets further from the clean one past 4
Mb/s (the bits go to the noise); with it, up to 1.2 dB nearer. Against
the noisy source it loses, as it should (1.2 to 1.9% BD-rate at 1 to 4
Mb/s). It takes 3.7 ms a 720p picture on one thread (speed 9: from 0.89
to 1.11 s for the 60 pictures). FFmpeg 6.1's `libvpx-vp9` here was built
without libvpx's denoiser (`-noise-sensitivity 1` writes the same bytes),
so there is no comparison.

**Temporal layers** (`temporal_layers`, `-ts-parameters`). libvpx's
patterns (`vpx_temporal_svc_encoder` modes 1 and 3): with two layers,
layer 0 every second picture, reading and refreshing LAST, and layer 1
the others, reading LAST and GOLDEN and refreshing GOLDEN; with three,
pictures 0, 1, 2, 3 of each four in layers 0, 2, 1, 2, layer 0 reading
and refreshing LAST, layer 1 reading LAST and refreshing GOLDEN, layer 2
reading LAST (and GOLDEN on the fourth) and refreshing nothing. Every
frame is error resilient: a decoder of fewer layers misses frames whose
probabilities and motion the next frame would otherwise read (WebRTC
sets libvpx's error resilience for the same reason). No lookahead, no
alt-refs or golden frames of their own, cuts coded as inter frames, no
reference pool. The rate control gives each layer's frames libvpx's
usual split (60/40%; 60/20/20%) with a model of their own.
`Vp9Encoder::temporal_layer` names the layer of the picture coded last
(`vp9enc keep_layers=N` writes the packets of the lowest N). The packets
of layers 0, and 0 and 1, decode alone to exactly the full stream's
pictures in our decoder, FFmpeg's and libvpx's. Speed 9, three layers, a
constant bitrate (1 s buffer), BD-rate at the same PSNR-Y / FFmpeg's
average against libvpx's realtime 8 with `-ts-parameters` (the same
pattern and split) and `-error-resilient default` (without it, its base
layer does not decode alone):

| | Three scenes | Still and jumps | Grainy pan |
| --- | --- | --- | --- |
| All layers | +11.0 / +7.2% | −15.2 / −15.5% | −52.7 / −50.5% |
| Layer 0 alone (7.5 pictures a second) | +7.7 / +10.9% | +27.4 / +27.1% | −24.4 / −22.6% |
| All layers, against one layer at the same speed | +57.0 / +46.1% | +98.2 / +89.9% | +0.2 / −1.0% |

One layer at speed 9 is far ahead of libvpx on the first two clips by
its reference pool, golden frames and cuts, which layers give up, and by
the probabilities it carries from frame to frame; the base layer, whose
pictures are four apart, is where libvpx stays ahead on two clips.

**Region of interest map** (`Vp9Encoder::set_roi_map`, libvpx's
`VP9E_SET_ROI_MAP`, FFmpeg's regions of interest). The caller's segment
for every 8×8 block, and for each of the eight segments a quantizer
delta, a loop filter level delta, one reference (inter frames) and skip
(inter frames: no residual and the zero vector, as the bitstream implies
them; not with the intra reference). The encoder keeps every block to one
segment (a square that spans two is split), weighs each segment's rate
against its error at the segment's own quantizer (with the frame's
lambda a finer segment took the bits of its quantizer and came back no
nearer), keeps each block of a reference segment to its reference, and
codes a skip segment's blocks as the zero vector with no residual, never
under 8×8; a map's lossless frames and hidden alt-refs are coded without
it. Every feature decodes alike in FFmpeg's `vp9`, libvpx and ours
(speeds 2, 6, 9 and 10, with and without alt-refs). The middle half of the
three scenes (640×360 of 1280×720) 40 indices finer, 60 frames, PSNR-Y of
the middle and of the whole picture:

| | Rate | Middle | Whole |
| --- | --- | --- | --- |
| Speed 6, `q=120` | 2232 → 3212 kb/s | 40.35 → 43.30 dB | 43.06 → 44.19 |
| Speed 9, 2 Mb/s constant bitrate | 1989 → 1975 kb/s | 34.87 → 36.03 | 38.32 → 37.41 |
| libvpx realtime 8, 2 Mb/s CBR, `addroi=…:qoffset=-0.3` | 1959 → 2018 kb/s | 34.21 → 35.12 | 37.16 → 33.58 |

At a bitrate the region's gain is paid for by the rest of the picture;
here at a quarter of libvpx's cost to the whole picture for 1.3 times
its gain in the region. libvpx (1.14 here) applies the map only at the
realtime deadline: at the good deadline FFmpeg's `addroi` changes no
byte.

**Corpus VBR** (`corpus_complexity`, `-corpus-complexity`). In two
passes, a clip spends the bitrate times its complexity over the
corpus's (a quarter to four times), so that a corpus of clips comes to
the bitrate as a whole and its easy clips leave bits to its hard ones.
A clip's complexity is `Vp9Encoder::first_pass_complexity`: ten times the
mean bits a 16×16 block of luma takes in its first pass, brought to
quantizer index 100 (bits as the step to the power −1.25); the corpus's
is the mean over its clips, each weighed by its length. libvpx's number
is its first pass's mean modified error, which this encoder has no
equivalent of, so the values do not carry over. The three clips as a
corpus at 2 Mb/s (complexities 588, 293 and 311; the corpus 408):

| | Three scenes | Still and jumps | Grainy pan | All |
| --- | --- | --- | --- | --- |
| Each at the bitrate | 2052 kb/s, 42.78 dB | 2028, 43.96 | 1967, 44.20 | 2022 kB |
| `corpus=408` | 2975, 45.15 | 1486, 43.77 | 1501, 43.89 | 2048 kB |
| libvpx `-cpu-used 4`, then `-corpus-complexity 400` | 2365 → 2312 | 2012 → 1686 | 1844 → 1717 | |

**Alpha** (`alpha`, `-pix_fmt yuva420p`). As in FFmpeg's `libvpxenc.c`,
a second encoder takes the stream's options (quantizer, bitrate, speed,
lookahead) and codes each alpha plane as the luma of an 8-bit 4:2:0
picture whose chroma is 128. It codes a picture's alpha when the stream
has coded the picture, from its own lookahead of the same pictures (so it
has alt-refs of its own), a key frame wherever the stream has one (forced,
so that a seek to any key frame finds alpha), and it drops none: the
alpha of a picture the stream drops is passed over. The stream's packets
are the ones the same options code without alpha. WebM stores each alpha
packet as its block's BlockAdditional, BlockAddID 1, and the track's
AlphaMode as 1 (`cutlass-matroska`; `vp9enc out.webm alpha=FILE`). The
decoder decodes a packet's alpha with a second decoder and hands the
plane out as the frame's alpha, as FFmpeg's `libvpx-vp9` decoder does
(its native `vp9` decoder leaves alpha out); an alpha packet that fails
leaves its picture without alpha. Checked both ways against FFmpeg 6.1
(`tests/alpha_interop.rs` at 160×96, and by hand at 640×360 with the
commands below): a WebM written by `vp9enc` and `cutlass-matroska`
decodes in `ffmpeg -c:v libvpx-vp9` to the same Y, U, V and alpha as in
`vp9dec`, every sample of all 60 pictures, and FFmpeg's own `libvpx-vp9`
`yuva420p` WebM decodes in `vp9dec` (one thread and four) to the YUVA
FFmpeg's `libvpx-vp9` decoder gives, every sample.

```
ffmpeg -f lavfi -i testsrc2=s=640x360:d=2:r=30 \
  -f lavfi -i "nullsrc=s=640x360:d=2:r=30,format=gray,geq=lum='128+127*sin((X+6*N)/50)*cos((Y-3*N)/40)'" \
  -filter_complex "[0][1]alphamerge,format=yuva420p" -f rawvideo src.yuva
ffmpeg -f rawvideo -pix_fmt yuva420p -s 640x360 -i src.yuva -vf format=yuv420p -f rawvideo src.yuv
ffmpeg -f rawvideo -pix_fmt yuva420p -s 640x360 -i src.yuva -vf extractplanes=a -f rawvideo src_alpha.y
# Ours to FFmpeg: the two decodes are the same bytes.
vp9enc src.yuv 640 360 ours.webm alpha=src_alpha.y crf=30 threads=4
ffmpeg -c:v libvpx-vp9 -i ours.webm -pix_fmt yuva420p -f rawvideo ours_by_ffmpeg.yuva
vp9dec ours.webm ours_by_us.yuva
cmp ours_by_ffmpeg.yuva ours_by_us.yuva
# FFmpeg's to ours: the same.
ffmpeg -f rawvideo -pix_fmt yuva420p -s 640x360 -r 30 -i src.yuva \
  -c:v libvpx-vp9 -pix_fmt yuva420p -crf 30 -b:v 0 -threads 4 ffmpeg.webm
ffmpeg -c:v libvpx-vp9 -i ffmpeg.webm -pix_fmt yuva420p -f rawvideo ffmpeg_by_ffmpeg.yuva
vp9dec ffmpeg.webm ffmpeg_by_us.yuva threads=4
cmp ffmpeg_by_ffmpeg.yuva ffmpeg_by_us.yuva
```

Size and quality against FFmpeg at the same `-crf` (15 to 55; `crf=` here,
speed 5 against FFmpeg's default `-cpu-used 1`, four threads), on that
source (a moving sinusoidal gradient) and on the same picture with a
moving soft-edged disc as its alpha, BD-rate of the alpha stream's bytes
(the BlockAdditionals) at the same alpha PSNR:

| crf | FFmpeg alpha bytes, PSNR | Ours | Ours, before "Synthetic pans" |
| --- | --- | --- | --- |
| 15 | 17015, 55.10 dB | 12739, 52.43 dB | 16818, 51.86 dB |
| 25 | 13679, 53.64 dB | 9189, 50.94 dB | 11828, 50.63 dB |
| 35 | 7952, 51.31 dB | 7008, 48.73 dB | 9378, 48.66 dB |
| 45 | 5676, 48.40 dB | 5753, 46.04 dB | 7908, 45.93 dB |
| 55 | 4643, 45.09 dB | 5217, 42.68 dB | 6902, 42.80 dB |

The alpha stream takes +20% bits at the same alpha PSNR against
libvpx's on the gradient (+65% before "Synthetic pans" in "Against
FFmpeg's libvpx-vp9", which found why; the disc, +59% then, was not
measured again); the colour stream of the same files +18% at the same
PSNR-Y, as before (this synthetic picture is unlike the README's clips,
where the encoder is ahead). Coding each alpha plane as a plain gray
stream gives the same bytes on both sides (ours 9189 at `crf=25`,
11828 before, libvpx's 13679 in an IVF less its headers), so the gap is
the encoder's on these smooth planes, not the alpha path's. The alpha stream's own
lookahead took it from +78% to +65% (coding each alpha picture as it
came, with no lookahead, was the first version). Time at `crf=30`: 1.65 s
against 1.15 s without alpha (FFmpeg's 3.2 s against 2.7 s).

**Layered alt-refs** (`alt_ref_layers`, `-auto-alt-ref` 2 to 6). With
the lookahead, once the golden group's alt-ref is coded, the span from
the picture coded last to the nearest alt-ref ahead is halved by a
further hidden alt-ref one layer down, while the layers allow and the
span is 8 pictures or more (6 for the third layer): filtered from the
pictures around its own target (at most half the span away), coded at
0.84 (third layer 0.95) of the inter step, and reading the alt-ref ahead
of it as ALTREF, as the pictures before it do. In its target's place it is
shown as it is (`show_existing_frame`, one byte), as libvpx shows its
mid-group alt-refs, and the next picture reads it as LAST. It takes its
slot from 3 to 7 out of the reference pool while it waits, and joins the
pool once shown. The streams decode alike in FFmpeg's `vp9` (frame
threads and one thread), libvpx and ours. Two passes at speed 5 (lag 16),
BD-rate at the same PSNR-Y / FFmpeg's average; libvpx two passes,
`-lag-in-frames 25`:

| | Three scenes | Still and jumps | Grainy pan |
| --- | --- | --- | --- |
| `arf_layers=2` against one layer | +1.5 / +1.2% | +3.5 / +5.4% | +0.8 / +1.6% |
| `arf_layers=3` against one layer | +2.4 / +2.0% | +2.9 / +5.1% | +0.8 / +1.6% |
| libvpx `-auto-alt-ref 2` against 1 | −8.0 / −6.2% | −6.7 / −2.4% | −5.3 / −3.0% |
| libvpx `-auto-alt-ref 6` against 1 | −6.4 / −2.8% | −9.8 / +0.1% | −10.4 / −1.2% |
| One layer against libvpx `-auto-alt-ref 1` | −6.3 / −5.5% | −37.6 / −35.9% | −14.6 / −22.6% |
| `arf_layers=2` against libvpx `-auto-alt-ref 2` | +3.4 / +2.1% | −30.8 / −33.9% | −10.0 / −22.1% |
| `arf_layers=3` against libvpx `-auto-alt-ref 6` | +2.8 / −0.2% | −29.0 / −37.3% | −9.9 / −23.6% |

They lose here, in 1.0 to 1.1 times the time: this encoder's golden
groups are short (12 pictures, 6 to 12 by motion in two passes), its
reference pool already gives each picture near ones, and a mid-group
alt-ref is a filtered picture shown in place of its source. Longer
groups, where
libvpx gains, cost more than layers win back: golden groups of up to 16
with lag 25 took one layer 27.8 points (PSNR-Y, summed over the clips)
over the default, and two layers won 6.5 of them back. Halving spans
from 4 pictures lost 7.5 points summed (11.9 with the second layer at
0.7 of the step, 19.9 at 0.56); an overlay coded in the target's place
instead of `show_existing_frame` (with the slots taken from the pool for
good) lost 15.6.

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
| `show_existing_frame` | yes | yes (layered alt-refs, added) |
| Error resilient, frame parallel | yes | yes (added) |
| Scaled references (a size change mid-stream) | yes | no |
| Lossless (WHT) | yes | yes |
| Tile columns and rows | yes | yes |
| Segmentation: quantizer | yes | yes (AQ, complexity AQ, cyclic refresh, 360°) |
| Segmentation: loop filter, reference, skip | yes | yes (ROI map, added) |
| Segmentation: temporal map | yes | no |
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
  segmentation with adaptive quantization (`aq`, `aq_mode`, off by
  default), and a region of interest map's quantizer, loop filter,
  reference and skip by segment (`set_roi_map`). Temporal segment maps,
  intra-only frames and scaled references (a size change within the
  stream) are not written; `show_existing_frame` only with layered
  alt-refs. Error resilient frames take no
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
  8 at every PSNR, in 0.85 to 0.90 of its time on one thread and 0.82 to
  0.95 on four (see "The gaps"); speed 9 9% to 77% fewer, in 0.76 to
  0.88 of it on four threads.
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
