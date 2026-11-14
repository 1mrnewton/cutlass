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
- Every partition from 64×64 to 8×8 (square), all ten intra modes,
  NEAREST/NEAR/ZERO/NEW inter modes on LAST and GOLDEN, sub-sample motion
  (to eighths where the vector may be), per-block transform size
  (`TX_MODE_SELECT`, 4×4 to 32×32 with the intra ADST types), skip, the loop
  filter (libvpx's level from the quantizer), probability adaptation.
- A golden frame every 12 frames, key frames every `keyint` (240) frames;
  key and golden frames take finer quantizers.
- Speeds 0–9 (`Vp9Options::speed`, default 5). Up to 6 every superblock is
  searched by trial coding; 7 and up code inter frames in one pass.
- Threads: tile columns, each on its own thread (`Vp9Options::threads`, 0 =
  one per core), up to the tile columns the width allows.
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

**Decode** (`tools/bench.py`): 300 frames of 1920×1080 written by libvpx.

| Threads | Cutlass | FFmpeg `vp9` | Ratio |
| --- | ---: | ---: | ---: |
| 1 core | 3.44 s | 2.16 s | 1.59 |
| 4 | 1.26 s | 0.98 s | 1.29 |

(Started at 38.8 s on one core: 18× faster so far.)

**Encode** (`tools/encbench.sh`, BD-rate with `tools/bdrate.py`): 30 frames
of 1280×720 (a pan over a detailed picture with film grain), 4 threads on
both sides (libvpx with `-row-mt 1 -tile-columns 2`). BD-rate is the bitrate
change at equal PSNR-Y; negative is smaller files.

| Cutlass | libvpx-vp9 | BD-rate | Time (Cutlass / libvpx) |
| --- | --- | ---: | --- |
| speed 9 | realtime, cpu-used 8 | −65% | 0.46–0.74 s / 0.27–0.37 s |
| speed 5 | realtime, cpu-used 8 | −72% | 1.4–4.0 s / 0.27–0.37 s |
| speed 5 | good, cpu-used 1 | +26% | 1.4–4.0 s / 1.2–1.7 s |

## Limits and gaps

- Not registered in the desktop app (the app has no VP9 opener yet). The
  crate registers on the video factories and offers `register_engine_encoders`
  for the engine's `EncodeFactory`.
- The encoder writes profile 0 only, no alt-ref (hidden) frames, no compound
  prediction, no 4×4 or rectangular partitions, no segmentation, no
  probability updates in the compressed header, EIGHTTAP filters only.
- The encoder always writes BT.601, studio range.

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7.
- libvpx was read where the specification and libvpx's streams disagree;
  no libvpx code is compiled into the crate.
