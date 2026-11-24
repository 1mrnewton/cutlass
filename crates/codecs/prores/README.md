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
- `ProresPictureDecoder`, the one the app uses, gives each frame as 8-bit 4:2:0 and drops alpha. Each slice job writes its own rows of the 8-bit picture from the rows it has just decoded, so no wide frame is built and nothing is converted afterwards. An odd width or height grows by one repeated sample. The bytes are the same as `to_yuv420p` of the wide frame.
- A job decodes into working rows that the decoder keeps between frames. `ProresDecoder::recycle` takes a wide frame's planes back, and the next frame of that size is decoded into them without an allocation.

Encoder:

- The code in `extra` picks the profile; empty means `apcn`.
- `encode` takes 8-, 10- or 12-bit planar 4:2:0, 4:2:2 or 4:4:4. It converts to the profile's chroma: 4:2:0 rows are repeated, 4:4:4 pairs are averaged for a 4:2:2 profile, and narrower chroma is repeated for a 4:4:4 profile. A 4444 profile keeps the frame's alpha and codes it losslessly; a 4:2:2 profile drops it.
- `encode_rgba` takes 8-bit RGBA and converts it to BT.709 limited range. A 4444 profile keeps the alpha channel.
- Rate control: one quantization index for the frame, the finest whose slices fit the budget, searched from the last frame's index. Then single slices move one index finer while the budget allows. Each worker reads a slice from the caller's planes, transforms it once, and codes it at the last frame's index and the one finer; when the index has not moved that one pass is the whole frame, and each index the search asks about beyond those costs one more pass. Alpha is outside the budget. The budget is the profile's target for the frame size divided over 29.97 frames, so a frame costs the same at any frame rate. `with_budget_scale` scales it from 0.05 to 4. `CodecParams::bitrate` is not read.
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
| Encoder: a plane whose stride or byte count does not hold its rows, or whose size does not match the frame's format | `InvalidApi` `plane size` |

A raised cancel flag gives `Cancelled` from either side, and `reset` clears it. A refused frame leaves the decoder able to decode the next one.

## Using it

- `ProresDecoder` implements `VideoDecoder`; `decode(&[u8])` returns `DecodedFrame { header, frame }`. `decode_frame` is the same step as a free function with an explicit pool, and `parse_frame_header` reads only the header.
- `ProresEncoder` implements `VideoEncoder` (planar frames) and `Encoder` (CPU frames in RGBA or 8-bit 4:2:0). `extradata()` is the profile code a movie muxer writes as the sample entry.
- `with_threads(n)` on either gives it its own pool; the default is the shared pool.
- `ProresDecoder::recycle(frame)` hands a decoded frame's planes back for the next frame.
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

`examples/decode_raw.rs` decodes a file of back-to-back frames to raw planes. `examples/decode_speed.rs` and `examples/encode_speed.rs` time the decoder and the encoder alone, on frames or raw pictures already in memory.

## How it is tested

`cargo test -p cutlass-prores` needs no tool:

- The entropy codes and the quantizer scale against hand-coded values, both scans, and the inverse DCT against the RDD 36 Annex A accuracy procedure.
- Every profile through this encoder and back through this decoder, with a PSNR floor per profile. Odd sizes down to 1 by 1, exact 16-bit alpha, RGBA colors, and identical bytes from one worker and from many.
- The 8-bit picture path against `to_yuv420p` of the wide frame, on both chroma layouts, odd sizes and alpha, with 1, 5 and 16 workers; the same errors from both paths on cut and damaged frames; a recycled frame leaving the next one unchanged.
- The sample store and the quantizer's rounding against `floor` and `round` over every half step and the floats around it; the bit reader and writer against one bit at a time; the codeword tables against the worked-out codewords.
- Planes that are too small refused by name; the app's 8-bit 4:2:0 input coding to the same bytes as the planar frame; frames whose quantization index jumps in both directions coding to the same bytes as a fresh encoder.
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

Measured on 24 November 2026 at commit `068aa00c`, on a 4-core Intel Xeon at 2.10 GHz (family 6 model 207, a KVM guest, 15 GB) running Ubuntu 24.04, against FFmpeg 6.1.1. Built with `cargo build --release` and rustc 1.97.0. Both sides had 4 threads.

**Decode, prores.** Input: 300 frames of 1920×1080 at 30 frames a second (195 MB), written by FFmpeg with `-c:v prores_ks -profile:v 2 -pix_fmt yuv422p10le`.

|  | Cutlass | FFmpeg (`prores`) | Cutlass before this work (`bcab8681`) |
| --- | ---: | ---: | ---: |
| Wall time | 1.48 s | 2.27 s | 7.81 s |
| Speed | 203 frames/s | 132 frames/s | 38 frames/s |
| CPU time | 5.36 s | 7.11 s | 18.37 s |
| Peak memory | 13 MB | 71 MB | 17 MB |
| Instructions, 30 frames | 5.79 billion | 4.04 billion | 18.23 billion |
| Threads allowed | 4 | 4 | 4 |

Cutlass is **1.5× faster** than FFmpeg here, and runs more instructions to do it: its inner loops choose with selects where FFmpeg's branch. The two outputs are not bit-identical: 53.85 dB PSNR between them, largest single difference 4. The Rust decoder returns `yuv420p` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale; left at 10-bit 4:2:2, FFmpeg takes 1.71 s. With `-profile:v 3` (HQ, 292 MB) the same run is 1.87 s against 2.90 s. On one thread, 30 frames four times over: 1.89 s against 2.80 s.

The stream is 10-bit 4:2:2. The Rust decoder hands the app 8-bit 4:2:0 and writes it inside the decoder, so FFmpeg is asked for 8-bit 4:2:0 too, and the two conversions round differently.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process, start-up included, run five times in turn; the table shows the median run. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. The FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly. This is the method of `scripts/bench_media.py`, run by hand: the script times with `/usr/bin/time -l`, which is the macOS form. The guest has no hardware counters, so `perf stat` reports no instructions; the instruction row is `valgrind --tool=cachegrind --cache-sim=no` over the first 30 frames, whole process, where both sides run their AVX2 paths. The pictures are synthetic (a Mandelbrot zoom mixed with a test pattern, blurred, with noise), because the street scene the earlier tables used was not on this machine.

**Every profile, `benches/throughput.rs`.** Frames a second inside one process, 12 pictures a measurement from 6 frames of the same synthetic clips, on all 4 cores and on one. The decode columns are the full-depth frame, handed back to the decoder after each one. Runs of 12 pictures vary by about a tenth.

| Size | Profile | Encode, 4 cores | Encode, 1 core | Decode, 4 cores | Decode, 1 core | Mb/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1080p | 422 Proxy | 96.6 | 33.1 | 385.0 | 121.1 | 45 |
| 1080p | 422 LT | 92.7 | 25.3 | 207.4 | 80.8 | 102 |
| 1080p | 422 | 56.2 | 22.1 | 168.0 | 63.8 | 147 |
| 1080p | 422 HQ | 56.4 | 19.0 | 126.5 | 41.9 | 220 |
| 1080p | 4444 | 57.0 | 16.0 | 101.0 | 31.0 | 330 |
| 1080p | 4444 XQ | 36.9 | 9.1 | 91.7 | 25.4 | 495 |
| 2160p | 422 Proxy | 19.6 | 6.3 | 98.5 | 29.5 | 181 |
| 2160p | 422 LT | 26.1 | 7.1 | 64.8 | 18.9 | 410 |
| 2160p | 422 | 14.0 | 4.3 | 47.5 | 13.8 | 591 |
| 2160p | 422 HQ | 18.4 | 4.9 | 35.1 | 10.1 | 884 |
| 2160p | 4444 | 16.2 | 4.4 | 26.4 | 7.9 | 1326 |
| 2160p | 4444 XQ | 14.6 | 4.2 | 21.2 | 6.6 | 1989 |

The same bench at `bcab8681` on this machine, for the rows this work set out to move:

| Size | Profile | Encode, 4 cores | Encode, 1 core | Decode, 4 cores | Decode, 1 core |
| --- | --- | ---: | ---: | ---: | ---: |
| 1080p | 422 HQ | 14.8 | 5.8 | 57.7 | 18.0 |
| 2160p | 422 HQ | 3.9 | 1.5 | 14.9 | 4.5 |
| 2160p | 4444 | 2.8 | 1.1 | 11.7 | 3.0 |
| 2160p | 4444 XQ | 2.7 | 1.0 | 9.2 | 2.7 |

1080p HQ encodes 3.8 times as fast on all cores and the 2160p 4444 profiles 5.4 to 5.8 times. The encoder writes the same bytes as before for every input tried. On these four cores 2160p 4444 and XQ are still under 30 frames a second. FFmpeg's own decoder on the 2160p HQ stream, one `ffmpeg` process with start-up: 17.9 frames a second.

Not measured here: the 16-thread figures on the Apple M5 Pro. The last ones, from 11 November 2026 at commit `07266d8f`, are from before this work: 1080p 4:2:2 decode through the app path at 316 frames a second against FFmpeg's 652.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>` on macOS, or the two programs above by hand; and `cargo bench -p cutlass-prores --bench throughput`.

## Limits and gaps

Limits, in `src/header.rs`: longest edge 16 384 pixels and at most 67 108 864 pixels in a frame. A larger header is `Limit` before any plane is allocated. Below the limit the planes are allocated in full once the headers and slice tables have been read, up to 2 bytes per sample for each of Y, Cb, Cr and alpha, and that allocation is not fallible. The decoder holds nothing between packets.

A coded slice is at most 65 535 bytes. The encoder coarsens a slice that would be larger.

Gaps:

- The encoder writes progressive frames only. Interlaced frames are decoded, not encoded.
- The encoder writes 0 (unspecified) for the aspect-ratio and frame-rate codes in the frame header.
- The app's picture path gets 8-bit 4:2:0 without alpha. The full-depth output with alpha is on the crate API only.
- The export panel offers Proxy, LT, 422 and HQ. 4444 and 4444 XQ are on the crate API only.
- The data rate cannot be set in bits per second; only `with_budget_scale` changes it.
- ProRes RAW is a different codec and is not handled.
- No coverage-guided fuzzing.

## Specifications

- SMPTE RDD 36:2022, Apple ProRes Bitstream Syntax and Decoding Process.
- Apple ProRes white paper (April 2022): the target data rates per profile and frame size.
- ITU-T H.273: the primaries, transfer and matrix code points in the frame header.
