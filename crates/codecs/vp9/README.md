# cutlass-vp9

A VP9 decoder and a small profile 0 VP9 encoder. The crate is safe Rust
(`#![forbid(unsafe_code)]`), depends only on `cutlass-decode` and
`cutlass-encode`, and links no C library.

**The desktop app does not use this crate.** The app lists it as a dependency
but never registers it, so Cutlass does not decode or encode VP9 with it today.
The crate registers on `VideoDecodeFactory` and `VideoEncodeFactory`, and the
app builds a `DecodeFactory` and an `EncodeFactory`; there is no opener for
those two yet.

## What it supports

Decoder (`Vp9Decoder`, over `Decoder`):

| Profile | Pictures |
| --- | --- |
| 0 | 8-bit 4:2:0 |
| 1 | 8-bit 4:2:2, 4:4:0, 4:4:4, and sRGB (published as GBR) |
| 2 | 10-bit and 12-bit 4:2:0 |
| 3 | 10-bit and 12-bit 4:2:2, 4:4:0, 4:4:4, and sRGB |

- Key, inter and intra-only frames, hidden frames (no picture comes out),
  `show_existing_frame`, and a superframe as one packet of several frames.
- Tiles, eight reference slots, scaled references, segmentation, the loop
  filter, compound prediction, probability adaptation, error resilient and
  frame parallel modes, lossless frames.
- Each `VideoOutput` carries a planar `VideoFrame` at the coded width and
  height with the packet's `pts`, `duration` and `timescale`. Samples above 8
  bits are little-endian `u16`. The frame carries the range, the render size, and
  named color for BT.601, BT.709, BT.2020 and sRGB; other color spaces are
  published as unspecified.

Refused by the decoder:

- A side above 16832 or more than 35,651,584 luma samples: `CodecError::Limit`,
  at `open` and again for each frame.
- A reference whose size ratio to the frame is not legal:
  `CodecError::Unsupported`.
- RGB in profile 0 or 2, a set reserved bit, a bad frame marker or sync code,
  a superframe index that does not add up: `CodecError::Malformed`.
- An empty frame: `CodecError::InvalidApi`.

A frame that fails leaves the references and probabilities as they were. When
one frame of a superframe fails, the whole packet is undone.

Encoder (`Vp9Encoder`):

- Input is `Yuv420p` only; another format is `Unsupported`. Both sides must be
  multiples of 8 (`Unsupported` otherwise) and at most 4096 (`Limit`).
- One packet per picture, no delay. A key frame every 30 pictures, inter
  frames between them that predict from the previous picture.
- `CodecParams::bitrate` of 0 keeps quantizer index 32. A positive bitrate
  moves the quantizer between 8 and 208 to stay near the target, and then each
  input needs a positive `duration` and a `timescale` (`InvalidApi` if not).
  `Vp9Encoder::with_quantizer` fixes the index; 0 is lossless.

## Using it

Public items: `Vp9Decoder`, `Vp9Encoder`, `Decoder`, `register_decoders`,
`register_encoders`. `Vp9Decoder` implements `VideoDecoder` and `Vp9Encoder`
implements `VideoEncoder`; both have an empty `drain` and a `cancel_flag` that
`push` checks. `Decoder::decode` takes the bytes of one frame and returns the
planes of a shown frame, or `None` for a hidden one.

```rust
use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet};
use cutlass_vp9::Vp9Decoder;

let params = CodecParams::video(CodecId::Vp9, 32, 32);
let mut decoder = Vp9Decoder::open(&params)?;
let packet = Packet::video(frame_bytes, 0, 1, 30, true);
for output in decoder.push(&packet)? {
    let (width, height) = (output.frame.width, output.frame.height);
}
```

`examples/roundtrip.rs` decodes the committed 32x32 key frame this way:

```
cargo run --release -p cutlass-vp9 --example roundtrip
```

## How it is tested

```
cargo test -p cutlass-vp9
```

Run everywhere, with no outside tool:

- Committed profile 0 streams in `fixtures/` (pictures of 32x32 to 64x64),
  each compared with stored YUV from FFmpeg's `vp9` decoder or libvpx: key
  frames, error resilient inter frames, inter frames that adapt probabilities,
  an alt-ref sequence, references scaled by 1/2 and 4/5, segmentation, and a
  52x52 pair with the loop filter.
- Encoder round trips through this decoder: lossless output equals the source,
  lossy output stays above PSNR bounds, and 30 frames at 500 kbps stay within
  0.70 to 1.25 of the target.
- Header parsing, the bool coder, the transforms, the scaled motion
  compensation steps, and the decoder and encoder lifecycle (reset, cancel,
  a failed frame keeps the references).
- `decoder_fuzz_does_not_panic` mutates encoded frames for 2 seconds.
  `VP9_FUZZ_SECS` and `VP9_FUZZ_SEED` change the time and the seed.

Skipped unless tools or files are present (the test prints `skipped` and
passes):

- `independent_decoders_match_our_yuv` gives the encoder's output to FFmpeg's
  `vp9` and `libvpx-vp9` decoders, and to GStreamer when it is installed.
  With `VP9_REQUIRE_ORACLES` set, a missing tool fails the test instead.
- The 24 tests in `src/prove.rs` encode real videos with FFmpeg's libvpx and
  compare this decoder with `ffmpeg -c:v vp9`. They cover profiles 1 to 3, 10
  and 12 bits, several tile columns and rows, the interpolation filters,
  frame parallel mode, loop filter and segment settings, a width above 4096,
  3840x2160, and the official level 1 streams. They need `ffmpeg` and a local
  folder of sample videos that is not in the repository, so they do not run in
  CI. Three of them also compile `tests/roi_map.c` against the system libvpx
  headers with `cc`.
- None of these skipped tests were run for this description.

`release_320x240_meets_realtime_decode` only measures in a release build and
fails below 30 frames per second.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, vp9.** Input: 600 frames of 1920×1080 at 30 frames a second (9.7 MB), written by FFmpeg with `-c:v libvpx-vp9 -b:v 3M -deadline good -cpu-used 4 -row-mt 1`.

|  | Cutlass | FFmpeg (`vp9`) |
| --- | ---: | ---: |
| Wall time | 17.91 s | 1.80 s |
| Speed | 34 frames/s | 333 frames/s |
| CPU time | 17.87 s | 1.79 s |
| Peak memory | 57 MB | 27 MB |
| Threads allowed | 1 | 1 |

Cutlass is **9.9× slower** than FFmpeg here. Both decoders produced exactly the same samples.

The encoder was not measured. The decoder is timed from an IVF file, because the desktop app does not register this crate.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- Not registered in the desktop app, and it has no opener for the app's
  `DecodeFactory` and `EncodeFactory`.
- The workspace has no WebM or Matroska reader yet (that crate is a
  placeholder). The tests read IVF.
- The decoder uses one thread; tiles are decoded in order.
- The decoder copies each picture several times: reconstruction to packed
  bytes, packed bytes back to a reference, and again into the output frame.
- Everything beyond 8-bit 4:2:0 at small sizes is compared with another
  decoder only by the skipped tests above. Whether any committed fixture uses
  compound prediction was not verified.
- A packet whose last byte looks like a superframe marker but whose index does
  not match is `Malformed`; it is not retried as a single frame.
- SMPTE 240 and the reserved color spaces are published as unspecified.
- The encoder emits profile 0 only: every frame shown and error resilient,
  one tile, DC intra prediction, integer motion from the previous picture,
  transforms no larger than 8x8, no loop filter, no segmentation, no compound
  prediction, no hidden or show-existing frames, no superframes. Quantizer 0
  uses the 4x4 Walsh-Hadamard transform.
- The encoder always writes color space 1 (BT.601) and studio range, whatever
  the input frame says.

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7.
- libvpx was read where the specification text disagrees with the streams
  libvpx writes: intra frames use the key frame partition probabilities, the
  partition context uses the libvpx lookup values, and loop filter deltas are
  ignored when deltas are disabled. No libvpx code is compiled into the crate.
