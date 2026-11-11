# cutlass-mpeg2

An MPEG-2 video (ITU-T H.262, ISO/IEC 13818-2) and MPEG-1 video decoder, and an MPEG-2 Main profile encoder. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app registers both: the decoder plays MPEG-2 tracks from MP4 and MOV files, and the encoder is what export uses when MPEG-2 is picked. Both are registered after the platform codecs, so on macOS the VideoToolbox opener is asked first when hardware decode is on, and this crate runs when that opener declines. On the test Mac VideoToolbox had no MPEG-2 decoder (`ffmpeg -hwaccel videotoolbox` reports `Failed setup for format videotoolbox_vld`), so this crate decoded every stream there.

## What it supports

Everything is 8-bit and runs on one thread.

| | Decoder | Encoder |
| --- | --- | --- |
| Streams | MPEG-2 and MPEG-1 video | MPEG-2 Main profile |
| Chroma | 4:2:0, 4:2:2, 4:4:4 | 4:2:0 |
| Picture types | I, P, B, and MPEG-1 D pictures | I, P, B, with two B pictures between anchors |
| Picture structure | Frame pictures and field pictures, both fields in one packet or one per packet | Progressive frame pictures |
| Prediction | Frame, field, 16×8, and dual prime | Frame, half-sample, forward, backward, and both |
| Transform and scan | Frame and field DCT, zigzag and alternate scan, both intra VLC tables, both quantizer scales, loaded quantizer matrices | Frame DCT, zigzag scan, default matrices |
| GOPs | Open and closed | Closed. A new GOP starts every half second of pictures, at a scene cut, and on a forced keyframe |
| Rate | n/a | Test Model 5 bit allocation with a VBV buffer model; constant-rate mode on request |
| Levels | The profile and level byte is not enforced | Main, High-1440, or High level, the lowest that holds the stream |

The decoder returns frames in display order. An I or P frame is held until the next one arrives, and `drain` releases it. Each `VideoOutput` carries a 4:2:0 picture; a 4:2:2 or 4:4:4 stream also carries its full-chroma planes in `yuv422` or `yuv444`, and its 4:2:0 picture is made by averaging the chroma. Color tags, the sample aspect ratio, and the frame rate come from the sequence headers.

What is refused:

- A scalable extension returns `CodecError::Unsupported` ("MPEG-2 scalable layers").
- A coded picture above the pixel cap returns `CodecError::Limit`. The default cap is `DEFAULT_MAX_PIXELS`, 8192×8192 luma samples.
- A damaged unit returns `CodecError::Malformed` or `CodecError::Truncated`, and the decoder stays as it was. There is no error concealment.
- An empty packet returns `CodecError::InvalidApi`.
- The encoder returns `CodecError::Unsupported` for an odd width or height, a picture larger than 1920×1152, a bitrate above 80 Mb/s, a frame rate other than 23.976, 24, 25, 29.97, 30, 50, 59.94, or 60, and more luma samples a second than High level holds. That last rule rules out 1080p at 50 or 60 pictures a second. `main_profile_level` answers the same question before an export starts.

## Using it

- `Mpeg2Decoder`: `new(max_pixels)` or `default()`, `open(&CodecParams)`, `configure` for a sequence header stored by a container, then `push`, `drain`, and `reset`.
- `es::Splitter` and `es::split` cut a raw elementary stream into access units. Both fields of a field-coded frame stay in one unit.
- `Mpeg2Encoder::open(&CodecParams)` implements `VideoEncoder`. Packets leave in decode order, and every I picture carries the sequence headers. A zero bitrate picks about 0.3 bits per sample.
- `register_decoders` and `register_encoders` put them on the engine's `DecodeFactory` and `EncodeFactory`.
- `examples/m2v_yuv.rs` decodes an elementary stream to raw YUV, `examples/yuv_m2v.rs` encodes raw 4:2:0 to an elementary stream, and `examples/es_dump.rs` lists access units.

```rust
use cutlass_decode::Packet;
use cutlass_mpeg2::Mpeg2Decoder;
use cutlass_mpeg2::es::split;

let mut decoder = Mpeg2Decoder::default();
let mut frames = Vec::new();
for unit in split(&stream)? {
    let packet = Packet {
        stream: 0,
        data: unit.data,
        pts: i64::try_from(unit.display_index)?,
        duration: 1,
        timescale: 25,
        keyframe: unit.coding_type == 1,
        side: Vec::new(),
    };
    frames.extend(decoder.push(&packet)?);
}
frames.extend(decoder.drain()?);
```

## How it is tested

Run with `cargo test -p cutlass-mpeg2`.

The standard does not fix the inverse transform bit for bit, so pictures are compared with FFmpeg within a bound, not for equality. On a 600-frame 1080p stream written by FFmpeg, this decoder's pictures differ from FFmpeg's by at most 2 in any sample (68.4 dB PSNR).

- **Checked-in streams** (`tests/oracle.rs`). Small streams sit in `tests/fixtures` with FFmpeg's decode beside each. FFmpeg wrote them, except `chroma444.m2v`, written by hand from H.262 because FFmpeg cannot encode 4:4:4, and `inter_fields.m2v`, written by `tests/inter_fields.rs`. They cover: progressive Main profile, interlaced frame pictures with field motion, field DCT, alternate scan and the non-linear quantizer scale, 4:2:2 with a loaded matrix, 4:4:4, and MPEG-1. No sample may differ by more than 2, at most 3% of samples may differ, and every plane must reach 55 dB.
- **Field pictures** (`tests/fields.rs`, `tests/inter_fields.rs`). The streams are written bit by bit by the tests. `inter_fields.m2v` covers P and B field pictures, 16×8 and dual-prime prediction, and skipped macroblocks, and has no residual, so it must match the stored FFmpeg decode exactly.
- **Inverse transform** (`src/dct.rs`): the IEEE 1180 accuracy test on three input ranges.
- **Encoder** (`tests/encode.rs`). This crate's decoder reads back every picture in display order. The tests also check the level, the rate (0.9 to 1.1 of the target), that the VBV buffer never underflows, scene cuts, forced keyframes, drain, reset, cancel, and refusals.
- **Lifecycle** (`tests/lifecycle.rs`): display order, reset, truncated and malformed units, the pixel cap, a size change, and chunked input to the splitter.
- **Fuzzing** (`tests/fuzz.rs`). `cargo test` runs a few hundred seeded rounds against the decoder and the encoder. `MPEG2_FUZZ_SECS` and `MPEG2_ENCODE_FUZZ_SECS` run a timed campaign, and `MPEG2_FUZZ_SEED` sets the seed.
- **MP4** (`tests/isobmff.rs`). The encoder's packets go through the workspace MP4 muxer and demuxer and back through the decoder. With FFmpeg installed, FFmpeg must decode the file to within 50 dB of this decoder.
- **Real recordings** (`tests/local.rs`). Private sample recordings are encoded and decoded both ways against FFmpeg. They are not in the repository.

The FFmpeg legs of the MP4 and field tests, and the whole recordings test, print `skipped:` and pass when FFmpeg or the recordings are missing. `MPEG2_REQUIRE_ORACLES=1` turns each skip into a failure.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, mpeg2.** Input: 600 frames of 1920×1080 at 30 frames a second (137 MB), written by FFmpeg with `-c:v mpeg2video -q:v 4`.

|  | Cutlass | FFmpeg (`mpeg2video`) |
| --- | ---: | ---: |
| Wall time | 2.58 s | 1.46 s |
| Speed | 233 frames/s | 411 frames/s |
| CPU time | 2.56 s | 1.45 s |
| Peak memory | 26 MB | 23 MB |
| Threads allowed | 1 | 1 |

Cutlass is **1.8× slower** than FFmpeg here. The two outputs are not bit-identical: 68.39 dB PSNR between them, largest single difference 2.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- **Containers.** The app reads MPEG-2 from MP4 and MOV only. Transport streams, program streams, and raw `.m2v` files do not open in Cutlass.
- **Playback is 4:2:0.** The engine decoder hands playback the 4:2:0 picture, so a 4:2:2 or 4:4:4 stream is shown with averaged chroma. The full planes are available only through the `VideoDecoder` path.
- **Sample aspect ratio is not passed to playback.** `VideoOutput::sample_aspect` carries it, and the engine decoder leaves it out of the picture description, so a stream with non-square samples is shown with square samples.
- **Interlace.** Both fields are woven into one frame. The crate does not deinterlace.
- **Odd sizes** are rounded up to the next even width and height on output.
- **No error concealment.** A damaged picture is an error, not a patched picture.
- **No scalable layers**, in either direction.
- **The encoder writes one kind of stream:** progressive 4:2:0 Main profile with `q_scale_type` 0 and zero frame-rate extension fields. It does not write interlaced, 4:2:2, or 4:4:4 video.
- **The GOP length is not a setting.** It is half the frame rate in pictures, with two B pictures between anchors.
- **Size and rate limits when encoding:** at most 1920×1152, at most 80 Mb/s, and no 1080p at 50 or 60 pictures a second. A picture taller than 1152, such as portrait 1080×1920, is refused.
- **Splitter:** one access unit may hold at most 32 MiB (`es::MAX_UNIT_BYTES`).
- **Fuzz seed:** `MPEG2_FUZZ_SEED` is read as a decimal number. A hex value is ignored without a message and the default seed is used.
- The crate builds at `opt-level = 3` in dev builds (workspace `Cargo.toml`), with overflow checks and debug assertions still on.

## Specifications

- ITU-T H.262, also published as ISO/IEC 13818-2 (MPEG-2 video).
- ISO/IEC 11172-2 (MPEG-1 video).
- IEEE 1180-1990, the accuracy test for the inverse transform.
- MPEG-2 Test Model 5, for rate control.
