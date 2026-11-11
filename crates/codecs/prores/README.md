# cutlass-prores

An Apple ProRes decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`), following SMPTE RDD 36. It decodes all six profiles, progressive and interlaced, with alpha, and encodes all six as progressive frames at Apple's published data rates. Slices are decoded and encoded in parallel on `cutlass-pool`.

The desktop app registers it for `CodecId::Prores` after the platform codecs. On macOS, with hardware decode or encode switched on in Settings, VideoToolbox is asked first and this crate runs when it declines. With the switch off, and on Windows and Linux, this crate is the ProRes codec. `cutlass-ffi` also registers it after the platform codecs.

## What it supports

One packet is one frame, and every frame is a key frame.

| Profile | Code | Chroma | Decoded depth | Encoder target at 1920 by 1080, 29.97 fps |
| --- | --- | --- | --- | --- |
| ProRes 422 Proxy | `apco` | 4:2:2 | 10 | 45 Mb/s |
| ProRes 422 LT | `apcs` | 4:2:2 | 10 | 102 Mb/s |
| ProRes 422 | `apcn` | 4:2:2 | 10 | 147 Mb/s |
| ProRes 422 HQ | `apch` | 4:2:2 | 10 | 220 Mb/s |
| ProRes 4444 | `ap4h` | 4:4:4, optional alpha | 12 | 330 Mb/s |
| ProRes 4444 XQ | `ap4x` | 4:4:4, optional alpha | 12 | 495 Mb/s |

Decoder:

- Frame and picture headers, slices of 1, 2, 4 or 8 macroblocks, custom and default quantization matrices, 8-bit and 16-bit alpha, bitstream versions 0 and 1.
- Interlaced frames, either field first. The two fields are woven into one picture.
- Output is `Yuv422p10`, `Yuv422p12`, `Yuv444p10` or `Yuv444p12`, 16-bit little-endian samples, limited range, clamped to `4..=2^depth - 5`. Alpha comes back on `VideoFrame::alpha` at its coded 8 or 16 bits. Primaries, transfer and matrix come from the frame header.
- The depth follows the code in `CodecParams::extra`: 12 for `ap4h` and `ap4x`, 10 for the others. With no code, a 4:4:4 or alpha frame is 12-bit and a 4:2:2 frame is 10-bit. `with_depth` forces 10 or 12.
- `ProresPictureDecoder`, the one the app uses, converts each frame to 8-bit 4:2:0 and drops alpha. An odd width or height grows by one repeated sample.

Encoder:

- The code in `extra` picks the profile; empty means `apcn`.
- `encode` takes 8-, 10- or 12-bit planar 4:2:0, 4:2:2 or 4:4:4. It converts to the profile's chroma: 4:2:0 rows are repeated, 4:4:4 pairs are averaged for a 4:2:2 profile, and narrower chroma is repeated for a 4:4:4 profile. A 4444 profile keeps the frame's alpha and codes it losslessly; a 4:2:2 profile drops it.
- `encode_rgba` takes 8-bit RGBA and converts it to BT.709 limited range. A 4444 profile keeps the alpha channel.
- Rate control: one quantization index for the frame, the finest whose slices fit the budget, searched from the last frame's index. Then single slices move one index finer while the budget allows. Alpha is outside the budget. The budget is the profile's target for the frame size divided over 29.97 frames, so a frame costs the same at any frame rate. `with_budget_scale` scales it from 0.05 to 4. `CodecParams::bitrate` is not read.
- Output has encoder identifier `ctls`, both quantization matrices in every frame, 8-macroblock slices, and the frame's color tags (or the parameters' when the frame names none).

Refused:

| Input | Error |
| --- | --- |
| Another codec id | `Unsupported` `codec is not ProRes` |
| Bitstream version above 1 | `Unsupported` `bitstream version` |
| Encoder `extra` that is not one of the six codes | `Unsupported` `ProRes profile` |
| A frame over the size limits | `Limit` `frame dimensions` |
| Cut-off input | `Truncated` |
| Bad header, slice table, quantization index, coefficient run or alpha data | `Malformed`, with the field named in `what` |
| An empty packet; a depth other than 10 or 12 | `InvalidApi` |
| Encoder: other pixel formats, platform frames, audio | `Unsupported` |
| Encoder: a frame whose size differs from the parameters, or a wrong RGBA length | `InvalidApi` |

A raised cancel flag gives `Cancelled` from either side, and `reset` clears it. A refused frame leaves the decoder able to decode the next one.

## Using it

- `ProresDecoder` implements `VideoDecoder`; `decode(&[u8])` returns `DecodedFrame { header, frame }`. `decode_frame` is the same step as a free function with an explicit pool, and `parse_frame_header` reads only the header.
- `ProresEncoder` implements `VideoEncoder` (planar frames) and `Encoder` (CPU frames in RGBA or 8-bit 4:2:0). `extradata()` is the profile code a movie muxer writes as the sample entry.
- `with_threads(n)` on either gives it its own pool; the default is the shared pool.
- `target_mbps` and `target_frame_bits` report the encoder's budget for a profile and size.
- `register_decoders` and `register_encoders` add the 8-bit picture decoder and the encoder to the app's factories. `register_video_decoders` and `register_video_encoders` add the full-depth ones to the video factories; the app does not call those two.

```rust
use cutlass_decode::{CodecError, CodecId, CodecParams, PixelFormat};
use cutlass_prores::{Profile, ProresDecoder, ProresEncoder};

/// Decode one frame at full depth and code it again as ProRes 422 HQ.
fn to_hq(frame: &[u8]) -> Result<Vec<u8>, CodecError> {
    let mut params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
    let mut decoder = ProresDecoder::open(&params)?;
    let decoded = decoder.decode(frame)?;

    params.extra = Profile::Hq.fourcc().to_vec();
    let mut encoder = ProresEncoder::open(&params)?;
    encoder.encode(&decoded.frame)
}
```

`examples/decode_raw.rs` decodes a file of back-to-back frames to raw planes.

## How it is tested

`cargo test -p cutlass-prores` needs no tool:

- The entropy codes and the quantizer scale against hand-coded values, both scans, and the inverse DCT against the RDD 36 Annex A accuracy procedure.
- Every profile through this encoder and back through this decoder, with a PSNR floor per profile. Odd sizes down to 1 by 1, exact 16-bit alpha, RGBA colors, and identical bytes from one worker and from many.
- A hand-built flat frame, every truncation of a valid frame, each bad header field by name, and 4 000 seeded bit-flip mutations that must not panic.

`cargo test -p cutlass-prores --release --test external` compares against other implementations. FFmpeg, ffprobe and `avconvert` are started as separate programs; the crate links none of them.

- Decode: streams made by FFmpeg's `prores_ks` and `prores_aw` encoders and by VideoToolbox, including interlaced, alpha, narrow slices, odd sizes and 2160p. Every Y, Cb and Cr sample must be within 1 of FFmpeg's decoder, with at most 5 % of samples differing at 10 bits and 25 % at 12 bits, and alpha must be identical. On Apple platforms the output must also be no further from VideoToolbox's decoder than FFmpeg's is.
- Encode: for each profile the rate must be between 0.80 and 1.05 times the target, luma PSNR must be within 0.3 dB of `prores_ks` or better at no more than 1.05 times its rate, and on Apple platforms VideoToolbox must decode the stream.
- End to end: an FFmpeg movie is demuxed, decoded, encoded and muxed again with `cutlass-isobmff`. ffprobe's stream description and packet times must be identical, and on a Mac with `avconvert` AVFoundation must read all frames and keep the color tags.

These tests need `ffmpeg` and `ffprobe` on `PATH` and named clips in a folder outside the repository. Without them each test prints what is missing and passes; setting `PRORES_REQUIRE_ORACLES` makes that a failure. The CI workflow installs no FFmpeg and has no clips, and the crate holds no stored ProRes frames, so on CI the decoder is only checked against this crate's own encoder and the hand-built frame.

Ignored by default: `fuzz_campaign` in `src/hardening.rs`, a seeded mutation run over four valid frames for `PRORES_FUZZ_SECONDS` seconds (60 when unset). It is not coverage-guided.

`benches/throughput.rs` needs FFmpeg and the same clips, and prints why and exits 0 without them.

Not run: official ProRes conformance streams.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, prores: not measured.** The run stopped with: `media-bench: picture dumps differ in size: 933120000 and 2488320000 samples`

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Limits, in `src/header.rs`: longest edge 16 384 pixels and at most 67 108 864 pixels in a frame. A larger header is `Limit` before any plane is allocated. Below the limit the planes are allocated in full once the headers and slice tables have been read, up to 2 bytes per sample for each of Y, Cb, Cr and alpha, and that allocation is not fallible. The decoder holds nothing between packets.

A coded slice is at most 65 535 bytes. The encoder coarsens a slice that would be larger.

Gaps:

- The encoder writes progressive frames only. Interlaced frames are decoded, not encoded.
- The encoder writes 0 (unspecified) for the aspect-ratio and frame-rate codes in the frame header.
- The app's picture path gets 8-bit 4:2:0 without alpha. The full-depth output with alpha is on the crate API only.
- The export panel offers Proxy, LT, 422 and HQ. 4444 and 4444 XQ are on the crate API only.
- The data rate cannot be set in bits per second; only `with_budget_scale` changes it.
- `encode` does not check the plane sizes of the frame it is given; samples outside a plane read as 0.
- ProRes RAW is a different codec and is not handled.
- No coverage-guided fuzzing.

## Specifications

- SMPTE RDD 36:2022, Apple ProRes Bitstream Syntax and Decoding Process.
- Apple ProRes white paper (April 2022): the target data rates per profile and frame size.
- ITU-T H.273: the primaries, transfer and matrix code points in the frame header.
