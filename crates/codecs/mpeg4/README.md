# cutlass-mpeg4

An MPEG-4 Visual (MPEG-4 Part 2, ISO/IEC 14496-2) decoder and encoder for rectangular 8-bit 4:2:0 video. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app registers both after the platform codecs: on macOS the VideoToolbox opener is asked first when the hardware switches are on, and this crate runs when that opener is absent or declines. The decoder and the encoder are separate. A tool the decoder accepts is not always a tool the encoder writes.

## What it supports

### Decoder

| Tool | Decoder |
| --- | --- |
| I, P, and B pictures | Accepted. Pictures leave in display order. |
| S-VOPs with global motion (GMC) | Up to three warping points. |
| Quarter-sample motion | Accepted, with the workarounds old DivX and Xvid streams need. |
| H.263 and MPEG quantizers | Both. A stream may carry its own matrices. |
| Interlace | Accepted, including field motion and field DCT. |
| Data partitioning and RVLC | Accepted, except together with interlace. |
| Resync markers (video packets) | Accepted. |
| Short header | H.263 baseline. |
| Packed bitstreams | Accepted: a P-VOP and its B-VOP in one packet, up to three VOPs a packet. |

Each output is one packed 4:2:0 picture (Y, then U, then V) with the stream's color primaries, transfer, and matrix. A not-coded VOP gives no picture. Headers may come in `CodecParams::extra` (the `esds` configuration), as `SideData::Config` on a packet, or in the stream.

What is refused, each with `CodecError::Unsupported`:

- Arbitrary shape, static sprites, sprite brightness change, more than three GMC warping points, a global motion warp out of range, N-bit video, NEWPRED, reduced-resolution VOPs, the studio profile, and chroma formats other than 4:2:0.
- A complexity estimation header with method 2 or 3.
- A real enhancement layer. A scalability flag with no layer fields is ignored.
- Data partitioning combined with interlace.
- In short-header streams: four motion vectors and the optional H.263 modes.

A layer above the pixel cap, or more than three VOPs in one packet, returns `CodecError::Limit`. After a failed packet, the pictures that packet completed before the error come out with the next `push` or `drain`.

### Encoder

`EngineEncoder` is the export encoder. It writes one VOP per packet and returns the stream headers from `extradata`.

| | Simple (default) | Advanced Simple (`ADVANCED_SIMPLE` in `CodecParams::extra`) |
| --- | --- | --- |
| Pictures | An I-VOP, then P-VOPs | Every other later picture is a B-VOP between two P-VOPs |
| Motion | Half-sample, one or four vectors | Quarter-sample, one or four vectors |
| Search range | ±8 pixels, so every vector fits `fcode` 1 | Same |
| Quantizer | H.263, 1 to 31 | H.263, 1 to 31 |

A bitrate above zero moves the quantizer from picture to picture. Zero keeps the starting quantizer. A forced keyframe writes an I-VOP. A width or height that is not a multiple of 16 is padded by repeating the edge, and the layer keeps the visible size.

The encoder returns `CodecError::InvalidApi` for a zero or odd width or height, a packet timescale of zero or above 65535, a timescale that changes mid-stream, and a picture of the wrong size. An edge above 8191 returns `CodecError::Limit`. `CodecParams::extra` that is neither empty nor `ADVANCED_SIMPLE` returns `CodecError::Unsupported`.

`encode_intra` and `encode_advanced` are simpler whole-stream functions with a fixed quantizer and a size that must be a whole number of macroblocks. `encode_advanced` can write B-VOPs, quarter-sample motion, and the MPEG quantizer with the default matrices.

## Using it

- `register_engine_decoders` and `register_engine_encoders` put the codec on the engine's `DecodeFactory` and `EncodeFactory`.
- `EngineDecoder::open(&CodecParams)` implements the engine's `Decoder` trait: `push`, `drain`, `reset`.
- `EngineEncoder::open(&CodecParams)` implements `VideoEncoder`. It accepts `Yuv420p` and `Rgba8` parameters, because export converts each frame to 4:2:0 first.
- `encode_intra` and `encode_advanced` return a whole elementary stream from raw planes.

```rust
use cutlass_decode::{CodecId, CodecParams, Decoder, Output, Packet, PictureSamples};
use cutlass_mpeg4::EngineDecoder;

let mut decoder = EngineDecoder::open(&CodecParams::video(CodecId::Mpeg4, 0, 0))?;
let mut outputs = Vec::new();
// `packets` holds one VOP each. The headers ride in front of the first VOP.
for (n, data) in packets.into_iter().enumerate() {
    outputs.extend(decoder.push(&Packet::video(data, n as i64, 1, 25, n == 0))?);
}
outputs.extend(decoder.drain()?);
for output in outputs {
    if let Output::Picture(picture) = output {
        if let PictureSamples::Cpu(bytes) = picture.samples() {
            // Packed 4:2:0 at `picture.info().width` × `picture.info().height`.
            let _ = bytes;
        }
    }
}
```

## How it is tested

Run with `cargo test -p cutlass-mpeg4`.

The standard does not fix the inverse transform bit for bit. The checked-in comparisons use FFmpeg decodes stored in the repository and must match byte for byte. Against FFmpeg on a 600-frame 1080p stream that FFmpeg wrote, the decoder's pictures differ by at most 1 in any sample (80.9 dB PSNR).

- **Checked-in streams** (`tests/decode/lifecycle.rs`). Three FFmpeg-written streams, Advanced Simple with B-VOPs and quarter samples, data partitioned, and interlaced, must decode to the FFmpeg output stored beside them. The same file covers display order and times, reset, damage that lasts until the next I-VOP, hostile bytes, and headers out of band.
- **Packed bitstreams** (`tests/decode/packed.rs`): pictures leave in display order with the right times, also after a failed B-VOP.
- **MP4** (`tests/mp4.rs`). FFmpeg-written MP4 files go through the workspace ISOBMFF demuxer. An `mp4v` entry is MPEG-4 Visual only when its `esds` object type is 0x20.
- **Encoder** (unit tests in `src/enc.rs` and `src/engine.rs`). Streams are decoded back by this crate's decoder: intra, predicted, four-vector, bidirectional, quarter-sample, MPEG-quantizer, and padded sizes. With FFmpeg installed, FFmpeg must reconstruct the same samples. Stored digests keep part of that check running where FFmpeg is absent. A one-second 64×48 stream must land within 25% of 200, 400, and 800 kbit/s.
- **Inverse transform** (`src/idct.rs`): the IEEE 1180 accuracy test.
- **Fuzzing** (`decoder_fuzz_does_not_panic`, `encoder_fuzz_does_not_panic` in `src/prove.rs`). Each runs 2 seconds under `cargo test`. `MPEG4_FUZZ_SECS` and `MPEG4_ENC_FUZZ_SECS` set a longer run, and `MPEG4_FUZZ_SEED` and `MPEG4_ENC_FUZZ_SEED` set the seed. The decoder fuzzer caps pictures at 1920×1088.
- **Corpus** (`tests/decode/corpus.rs`). A folder of elementary streams with FFmpeg decodes, kept outside the repository and named by `MPEG4_CORPUS`, is compared bit for bit. The test prints `skipped` and passes when the folder is missing.
- **Release gate** (`release_decode_and_encode_meet_their_gates`). Ignored by default. It checks a time and memory budget: `cargo test -p cutlass-mpeg4 --release --lib release_decode_and_encode_meet_their_gates -- --ignored --nocapture`.

Tests that need FFmpeg print a message and pass without it. `MPEG4_REQUIRE_ORACLES=1` turns a missing tool into a failure.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, mpeg4.** Input: 600 frames of 1920×1080 at 30 frames a second (117 MB), written by FFmpeg with `-c:v mpeg4 -q:v 4`.

|  | Cutlass | FFmpeg (`mpeg4`) |
| --- | ---: | ---: |
| Wall time | 3.83 s | 1.38 s |
| Speed | 157 frames/s | 435 frames/s |
| CPU time | 3.81 s | 1.36 s |
| Peak memory | 24 MB | 28 MB |
| Threads allowed | 1 | 1 |

Cutlass is **2.8× slower** than FFmpeg here. The two outputs are not bit-identical: 80.91 dB PSNR between them, largest single difference 1.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- **Picture size.** A decoded picture is at most 8192×8192 pixels, and each edge of a layer is at most 8191. An odd edge is shown even: the last column or row is dropped.
- **Sample aspect ratio is dropped.** The decoder reads past the layer's aspect ratio fields, so a stream with non-square samples is shown with square samples.
- **Full range is not passed on.** The picture description has no field for it.
- **One thread** for decode and for encode.
- **The encoder does not write** GMC, sprites, interlace, data partitioning, resync markers, RVLC, short headers, custom matrices, shape, N-bit video, NEWPRED, reduced resolution, or scalability.
- **Simple rate control.** The starting quantizer is chosen as if the stream had 30 pictures a second (`initial_quant`), whatever the real rate.
- **Advanced Simple is opt-in.** The desktop app never passes `ADVANCED_SIMPLE`, so its exports are Simple. The code comment gives the reason: Apple's MPEG-4 decoder refuses Advanced Simple streams. That behavior is not checked by a test in this crate.
- **On a Mac this decoder always runs.** The app asks VideoToolbox first when hardware decode is on, but the `mp4v` format description `cutlass-platform-apple` builds has no `esds`, and `VTDecompressionSessionCreate` refuses it with `-12909`. `videotoolbox_decodes_the_rust_mpeg4_export` in that crate skips for that reason. On the test Mac, `VTCompressionSessionCreate` for MPEG-4 also returned `-12908`, so export uses this encoder too.
- **`encode_intra` and `encode_advanced`** use a fixed 25 Hz clock and a fixed quantizer, and refuse sizes that are not a multiple of 16.
- The crate builds at `opt-level = 3` in dev builds (workspace `Cargo.toml`), with overflow checks and debug assertions still on.

## Specifications

- ISO/IEC 14496-2, *Coding of audio-visual objects, Part 2: Visual*.
- ITU-T H.263, for short-header streams.
- ISO/IEC 14496-1, for the `esds` decoder configuration.
- IEEE 1180-1990, the accuracy test for the inverse transform.
