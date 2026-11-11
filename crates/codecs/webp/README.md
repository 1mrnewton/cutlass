# cutlass-webp

A WebP decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`): lossy (VP8 key frames), lossless (VP8L), alpha, and animation. It depends only on `cutlass-decode` and `cutlass-encode` and links no other WebP library. The desktop app registers both sides for `CodecId::Webp` ahead of the platform codecs, so WebP files are decoded and encoded here on every platform, and the app's animated WebP export uses this encoder.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Lossless (`VP8L`) | Every transform, the color cache, grouped prefix codes | Efforts 0 to 9; palette, subtract-green, predictor and cross-color transforms chosen per picture |
| Lossy (`VP8 `) | Key frames: token partitions, segments, filter deltas, the simple and the normal loop filter | Quality 0 to 100, methods 0 to 6; segments, per-block mode choice by rate and distortion, fitted token probabilities |
| Alpha with a lossy picture (`ALPH`) | Raw and lossless, every filter, the level-reduction flag | Coded losslessly |
| `VP8X` with `ICCP` | Profile passed through on `icc_profile`, not applied | Written for a still that carries a profile |
| Animation (`ANIM`, `ANMF`) | Frame offsets, blending, disposal, lossy and lossless frames mixed | Each frame after the first is the smallest rectangle that changed |
| `EXIF`, `XMP ` and unknown chunks | Skipped | Not written |

Decoded pictures are 8-bit: `ColorType::Rgb`, or `ColorType::Rgba` with straight alpha when the file carries alpha. Animation frames are always RGBA, and each one is the whole canvas after that frame is composed. Composition follows libwebp's animation decoder: the canvas starts transparent, the `ANIM` background color is ignored, and disposing a frame clears its rectangle to transparent.

The encoder accepts a still of any color type and bit depth and converts it to 8-bit RGBA first (a palette is expanded, a color key becomes zero alpha).

What is refused:

| Input | Error |
| --- | --- |
| A VP8 inter frame, a hidden VP8 frame | `Unsupported` (`vp8 inter frame in webp`, `vp8 hidden frame`) |
| Encoder `extra` that is not empty or four valid bytes | `Unsupported` (`webp encoder options`) |
| Encoder picture with a zero edge or an edge over 16 383 | `Unsupported` (`webp size`) |
| A platform buffer, or a frame that is not RGBA 8-bit, on `push_frame` | `Unsupported` |
| A canvas that differs from its bitstream, a frame outside the canvas, broken chunks or bitstreams | `Malformed` |
| A file or bitstream that ends early | `Truncated` |
| Anything past the limits below | `Limit` |
| An empty packet, an audio `push` on the encoder, an animation packet before the one that starts the file, a frame whose size differs from the first | `InvalidApi` |

## Using it

`WebpDecoder::open` and `WebpEncoder::open` take a `CodecParams` whose codec is `CodecId::Webp`. `register_decoders` and `register_encoders` add them to a `DecodeFactory` and an `EncodeFactory`.

Decoder:

- A still file is one packet and one picture.
- An animation arrives as the `cutlass-still` container splits it: the first packet is the file start through the first `ANMF` chunk, and each later packet is one `ANMF` chunk. The decoder keeps the canvas between packets until `reset`. A failed packet leaves the canvas as it was.
- A whole animated file in one packet returns every canvas, timed in milliseconds from the stored frame durations (a stored 0 counts as 100 ms).
- `decode_picture` returns `StillOutput` values. `push` returns the same pictures as RGBA `Output::Picture` frames, opaque where the file has no alpha.

Encoder:

- `push_picture` returns one complete `.webp` file as one packet.
- `push_frame` takes RGBA 8-bit CPU frames and returns nothing until `drain`, which returns the animation as one packet per frame in the same split the decoder reads. Joining the packets is the file. The loop count is written as 0 for the muxer to set.
- `WebpOptions` goes in `CodecParams.extra` through `to_extra()`: `lossless`, `quality` (0 to 100, as `cwebp -q`, lossy only), `effort` (0 to 6 lossy as `cwebp -m`, 0 to 9 lossless as `cwebp -z`), and `exact` (keep the color under fully transparent pixels). An empty `extra` is lossy, quality 75, effort 4. `WebpOptions::lossless()` is lossless at effort 6.
- A raised cancel flag makes `push_picture`, `push_frame` and `drain` return `Cancelled`. `reset` drops the frames that are waiting.

```rust
use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{BitDepth, ColorType, Encoder, PictureInput, StillImage};
use cutlass_webp::{WebpDecoder, WebpEncoder, WebpOptions};

let mut params = CodecParams::video(CodecId::Webp, 2, 2);
params.extra = WebpOptions::lossless().to_extra();
let image = StillImage::from_samples(2, 2, ColorType::Rgb, BitDepth::Eight, vec![200; 12]);
let mut encoder = WebpEncoder::open(&params)?;
let packets = encoder.push_picture(&PictureInput { image })?;

let mut decoder = WebpDecoder::open(&CodecParams::video(CodecId::Webp, 2, 2))?;
let file = Packet::video(packets[0].data.clone(), 0, 0, 0, true);
let pictures = decoder.decode_picture(&file)?;
assert_eq!(pictures[0].image.width, 2);
```

## How it is tested

`cargo test -p cutlass-webp` runs three kinds of test.

Files kept in `tests/fixtures`, with the expected RGBA beside each one (`tests/oracle.rs`). The files were written by `cwebp`, `img2webp`, `webpmux` and FFmpeg's libvpx encoder; the expected pixels are `dwebp` output for stills and Pillow's libwebp frames for animations. These tests open each file through `cutlass-still`, need no installed tool, and require exact pixels. They cover lossy pictures with each loop filter and four token partitions, every alpha coding and filter, lossless pictures with palettes of 2 to 200 colors, and two animations.

Comparisons against installed tools (`src/tests/`), on pictures made during the test:

- Decoder: files from `cwebp`, `img2webp`, `gif2webp` and `webpmux` must decode to exactly the pixels of `dwebp`; lossy files are also compared with FFmpeg's decoder, lossless files with macOS ImageIO through `sips`, and animations with Pillow frame by frame.
- Encoder: lossless files must come back exact from `dwebp` and pass `webpinfo`; lossy files must decode the same here and in `dwebp`, and are compared with `cwebp` output for size and quality.
- A test that needs a tool that is not on `PATH` passes without comparing anything. Set `WEBP_REQUIRE_ORACLES` to make that a failure.

Tests that need nothing outside the crate:

- `src/tests/malformed.rs`: hand-built bad bitstreams (a backward reference before the first pixel, a copy past the last, an over-subscribed code, a repeated transform, a bad color cache size, an inter frame, a partition past the end), every truncation of a lossless, a lossy and an alpha bitstream, the size limits, and a frame outside its canvas.
- `src/tests/fuzz.rs`: 4000 mutations of six files this crate writes, each decoded whole and packet by packet; a panic fails the test. A second test encodes random pictures with random settings and decodes them back. `WEBP_FUZZ_ROUNDS` and `WEBP_FUZZ_SEED` change the run. There is no coverage-guided fuzz target.
- `src/tests/lifecycle.rs`: reset, drain, cancel, and refused calls.

Ignored by default, because they need files that are not in the repository: `corpus_stills_match_dwebp_exactly` and `corpus_animations_match_libwebp` (`WEBP_CORPUS_DIR`), `measure_lossless_against_cwebp` and `measure_lossy_against_cwebp` (`WEBP_SOURCE_DIR`), and `oracle_directory` (`CUTLASS_WEBP_ORACLE`).

`cargo bench -p cutlass-webp` (`benches/throughput.rs`) times decode and encode of one 1920x1080 frame beside `cwebp` and `dwebp`, after checking the decode equals `dwebp`. It prints a skip line when those tools are missing.

## Speed against FFmpeg

Measured on 11 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, webp.** Input: one picture of 1920×1080 (24 kB), written by cwebp with `-q 80`, opened and decoded 200 times.

|  | Cutlass | FFmpeg (`webp`) |
| --- | ---: | ---: |
| Wall time | 2.09 s | 1.31 s |
| Speed | 96 pictures/s | 153 pictures/s |
| CPU time | 2.08 s | 1.31 s |
| Peak memory | 28 MB | 22 MB |
| Threads allowed | 1 | 1 |

Cutlass is **1.6× slower** than FFmpeg here. The two outputs are not bit-identical: 50.53 dB PSNR between them, largest single difference 25. The Rust decoder returns `rgb24` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale.

**Decode, webp-lossless.** Input: one picture of 1920×1080 (0.4 MB), written by cwebp with `-lossless`, opened and decoded 200 times.

|  | Cutlass | FFmpeg (`webp`) |
| --- | ---: | ---: |
| Wall time | 4.79 s | 10.90 s |
| Speed | 42 pictures/s | 18 pictures/s |
| CPU time | 4.78 s | 10.84 s |
| Peak memory | 34 MB | 28 MB |
| Threads allowed | 1 | 1 |

Cutlass is **2.3× faster** than FFmpeg here. Both decoders produced exactly the same samples. The Rust decoder returns `rgb24` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale.

Encoding and animated files were not measured: the tool encodes moving pictures and sound only, and the FFmpeg here cannot write WebP.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Longest edge of one bitstream | 16 383 pixels |
| Pixels in one picture or canvas | 67 108 864 (8192 x 8192) |
| Canvas pixels returned by one call, all frames together | 67 108 864 |
| Frames in one packet, and frames one encoder holds | 4096 |
| Chunks in one packet | 16 384 |
| ICC profile | 4 MiB |
| VP8L decoding-table entries for one picture | 33 554 432 |

A size past these is refused before a buffer of that size is allocated.

- One thread, no SIMD.
- No incremental decoding: a packet is a whole still or a whole frame.
- The encoder holds every coded frame of an animation in memory until `drain`, and refuses frame 4097.
- The encoder checks its cancel flag between pictures, not inside one.
- No trellis quantization, and no encoder settings beyond the four in `WebpOptions`: no target size, and none of `cwebp`'s `-sns`, `-f` or `-sharpness`.
- `EXIF` and `XMP ` are dropped on decode and never written. An ICC profile is carried, not applied, and is written for stills only.
- Encoded animation frames never blend and never dispose; each replaces its rectangle.
- The VP8 key-frame decoder and encoder here are this crate's own code. They do not share code with `cutlass-vp8`.

## Specifications

- RFC 9649, WebP Image Format: the RIFF container, `VP8X`, `ICCP`, `ANIM`, `ANMF`, `ALPH`, and the VP8L lossless bitstream
- RFC 6386, VP8 Data Format and Decoding Guide: the key frame a lossy WebP carries
