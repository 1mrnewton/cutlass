# cutlass-mjpeg

A Motion JPEG decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`), built on the JPEG core in `cutlass-jpeg`. This crate adds what Motion JPEG needs around a JPEG image: default Huffman tables, tables carried between packets, two-field packets woven into one frame, and rate control for the encoder.

The desktop app registers it for `CodecId::Mjpeg` ahead of the platform codecs, for both decode and encode, so Motion JPEG always runs in this crate. `cutlass-ffi` does the same. The reason is in the app's registration code: VideoToolbox opens a Motion JPEG stream and then fails 4:2:2 frames, and a factory only moves to the next codec when one refuses at open.

## What it supports

One packet is one frame, and every frame is a key frame. A packet holds one JPEG image, or the two field images of one interlaced frame.

Decoded:

- Everything `cutlass-jpeg` decodes: baseline SOF0, extended sequential SOF1 and progressive SOF2 Huffman images, 8-bit, gray or three components, restart intervals, 8-bit and 16-bit quantization tables.
- Images with no DHT segment. Huffman ids 0 and 1 start as the ITU-T T.81 Annex K.3 tables.
- Images with no DQT or DHT of their own. Tables defined by an earlier packet, or by a table stream in `CodecParams::extra`, stay in force. `reset` goes back to the tables the decoder opened with. A packet that fails changes nothing.
- Two fields in one packet, woven row by row. The first packet after open or reset decides: it is a field pair when its first image has the container width (or none is known), a height under three quarters of the container height, and a second image follows. Field order comes from the `fiel` bytes in `extra`, then from an `AVID` comment, and is top field first otherwise.
- QuickTime format A (`mjpa`): the APP1 `mjpg` header is accepted and the second field is found by its start-of-image marker.
- A `CS=ITU601` comment marks studio-range YCbCr.
- A size or sampling change from one packet to the next.

Decoder output:

| Decoder | Trait | Output |
| --- | --- | --- |
| `MjpegPictureDecoder` | `Decoder` (what the app uses) | Straight RGBA, 8 bits, alpha 255 |
| `MjpegDecoder` | `VideoDecoder` | Planar 8-bit: `Gray8`, `Yuv420p`, `Yuv422p`, `Yuv440p`, `Yuv444p`, or `Gbrp` for RGB images. Other samplings are upsampled to `Yuv444p`. The matrix is BT.601 and the range is full unless the frame is marked `CS=ITU601` |

A woven pair of 4:2:0 or 4:4:0 fields comes out as `Yuv422p` or `Yuv444p`: each field's chroma is stretched to the field height before weaving, so every frame row carries chroma from its own field. EXIF orientation is not applied to video frames.

Encoded: baseline SOF0 with a JFIF APP0 segment and its own DQT and DHT in every frame, so any packet decodes alone.

| Encoder input | Result |
| --- | --- |
| RGBA (`Encoder::push_frame`, `encode_rgba`) | 4:2:0 by default; `with_sampling` picks 4:2:2, 4:4:4 or gray. Alpha is dropped |
| Planar `Gray8`, `Yuv420p`, `Yuv422p`, `Yuv444p` (`VideoEncoder::push`, `encode_frame`) | Keeps its own sampling. Must be full range, BT.601 or untagged |

`extra` empty or one byte `1..=100` sets a fixed quality (75 when empty). With no quality byte, a nonzero `CodecParams::bitrate` selects `RateMode::Bitrate`: each frame takes the highest quality whose size fits its share of the rate, and unused bits carry to the next frame. `with_restart_interval` writes restart markers. An encoder opened at 0 by 0 accepts any frame size; one opened at a size refuses another.

Refused:

| Input | Error |
| --- | --- |
| Format B: `mjpb` in `extra`, or a packet that starts with a format B header | `Unsupported` `motion jpeg format b` |
| `extra` that is not empty, a `jpeg`/`mjpa` code with optional two `fiel` bytes, or a JPEG table stream | `Unsupported` `motion jpeg extradata` |
| A lone field when `fiel` names two fields, an `AVI1` marker has polarity 1 or 2, or the stream already paired fields | `Unsupported` `one field per packet` |
| A second field whose size, color or sampling differs from the first | `Malformed` `second field` |
| A packet over 64 MiB | `Limit` `motion jpeg packet` |
| An empty packet | `InvalidApi` `empty packet` |
| Anything `cutlass-jpeg` refuses (arithmetic, lossless, 12-bit, CMYK, oversize) | That crate's error |
| Encoder: studio-range planar input, another matrix, another pixel format | `Unsupported` `studio range input`, `color matrix`, `pixel format` |
| Encoder: bitrate mode and a frame with no duration or timescale | `InvalidApi` `frame duration` |

## Using it

- `MjpegDecoder::open` takes `CodecParams` with the container's frame size in `width` and `height` (zero for unknown, in which case every packet is one frame). `decode(&[u8])` returns a `Picture`; `Picture::to_rgba` and `Picture::into_frame` give pixels. `last_field_order` reports the stored order of the last frame's fields.
- `MjpegEncoder::open`, then `encode_rgba` or `encode_frame`. `extradata()` is `jpeg`, the sample-entry code a movie muxer writes.
- `register_decoders` and `register_encoders` add the RGBA picture decoder and the encoder to the app's factories. `register_video_decoders` and `register_video_encoders` add the planar ones to the video factories; the app does not call those two.

```rust
use cutlass_decode::{CodecError, CodecId, CodecParams, PixelFormat};
use cutlass_mjpeg::{MjpegDecoder, MjpegEncoder, Sampling};

/// Decode one packet and code it again as 4:2:2 at quality 75.
fn recode(packet: &[u8]) -> Result<Vec<u8>, CodecError> {
    let params = CodecParams::with_format(CodecId::Mjpeg, 0, 0, PixelFormat::None);
    let mut decoder = MjpegDecoder::open(&params)?;
    let picture = decoder.decode(packet)?;
    let (width, height) = (picture.width, picture.height);
    let mut rgba = Vec::new();
    picture.to_rgba(&mut rgba)?;

    let mut encoder = MjpegEncoder::open(&params)?.with_sampling(Sampling::Yuv422);
    encoder.encode_rgba(width, height, &rgba, 1, 30) // duration 1 at timescale 30
}
```

## How it is tested

`cargo test -p cutlass-mjpeg` runs everything below; all tests are inside the crate.

Tests that need no tool (`src/tests.rs`, `src/fuzz.rs`, one in `src/oracle.rs`):

- Table rules, field pairing and weaving, field order, format A, format B refusal, studio range, size changes, the encoder round trip, bitrate mode within 10 % of its target over a 30-frame clip, cancel and reset.
- Truncation at every byte of a frame must be refused, 2 000 single-bit flips must not panic, and after each one the good packet must decode unchanged.
- `mutated_packets_fail_closed_and_leave_the_decoder_as_it_was`: 3 000 seeded mutations of an eight-packet corpus, with encoder cases mixed in.
- `committed_reference_frames_match_their_djpeg_digests`: five frames stored in `fixtures/` (cjpeg 4:2:0, 4:2:2 progressive, gray with restart, an FFmpeg 4:2:2 frame with its DHT stripped, and a field pair) must decode to samples whose digest equals the recorded digest of `djpeg -dct int` output.

Tests that start `ffmpeg`, `cjpeg` and `djpeg` as separate programs (the rest of `src/oracle.rs`). FFmpeg is only a comparison tool; the crate does not link it.

- RGBA output must equal `djpeg -dct int` byte for byte on FFmpeg and cjpeg streams: 4:2:0, 4:2:2, 4:4:4, 4:4:0, gray, RGB, progressive, SOF1, restart markers, DHT stripped, and field pairs in both orders.
- Planar output must be within 1 of FFmpeg's planes for every sample.
- Encoder output must open in `ffmpeg -xerror` with no message. At quality 30, 75 and 95 and each sampling its size must be within 2 % of `cjpeg -dct int`, and `djpeg` must decode it to the same samples as cjpeg's file.

These tests return early and pass when a tool is missing. Setting `MJPEG_REQUIRE_ORACLES` makes a missing tool a failure. The CI workflow installs none of the tools, so there only the committed-fixture test checks against an independent decoder. The source pictures are footage from a folder outside the repository when it is present, and FFmpeg's test pattern otherwise.

Ignored by default: `fuzz_campaign`, the same mutation fuzzer on a time budget (`MJPEG_FUZZ_SECONDS`, 60 when unset, and `MJPEG_FUZZ_SEED`). It is not coverage-guided.

`benches/throughput.rs` times encode and decode of 1080p frames. It needs FFmpeg and one named clip outside the repository, and prints why and exits 0 without them.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, mjpeg.** Input: 600 frames of 1920×1080 at 30 frames a second (237 MB), written by FFmpeg with `-c:v mjpeg -q:v 4`.

|  | Cutlass | FFmpeg (`mjpeg`) |
| --- | ---: | ---: |
| Wall time | 4.61 s | 2.31 s |
| Speed | 130 frames/s | 260 frames/s |
| CPU time | 4.58 s | 2.29 s |
| Peak memory | 16 MB | 25 MB |
| Threads allowed | 1 | 1 |

Cutlass is **2.0× slower** than FFmpeg here. The two outputs are not bit-identical: 55.85 dB PSNR between them, largest single difference 34. The Rust decoder returns `rgba` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Packet | 64 MiB (`MAX_PACKET`), checked before the packet is read |
| Frame edge | 16 384 pixels |
| Pixels per frame | 16 777 216 |
| Marker segments per image | 4 096 |
| Queued input or output | None: a packet is decoded when pushed, and the encoder has no delay |

The frame limits are `cutlass-jpeg`'s and apply to the encoder too.

Gaps:

- Format B (`mjpb`) is refused.
- Streams that store one field per packet are refused, not paired across packets.
- Bytes after the last image of a packet are ignored, so a third image is dropped without an error.
- The encoder writes whole progressive frames only. It writes no field pairs.
- The interlace decision is made once, on the first packet after open or reset.
- The woven frame is handed on as progressive. The stored field order is on `Picture::fields` and `last_field_order` only; `VideoFrame` and the RGBA picture do not carry it.
- The app's export path sends RGBA, so it codes 4:2:0. 4:2:2 and 4:4:4 are reachable through this crate's API only.
- Bitrate mode counts the whole scan at each quality it tries, so it costs more than a fixed quality.
- No coverage-guided fuzzing.

## Specifications

- ITU-T T.81 | ISO/IEC 10918-1: the JPEG image, abbreviated formats (B.5), Annex K.3 Huffman tables.
- ITU-T T.871 (JFIF): full-range BT.601 YCbCr.
- Apple QuickTime File Format: Motion JPEG format A and the `fiel` atom.
- OpenDML AVI: the APP0 `AVI1` field marker.
- Conventions taken from FFmpeg's Motion JPEG decoder and checked against it: the three-quarter-height field test, the `AVID` comment, and the `CS=ITU601` comment.
