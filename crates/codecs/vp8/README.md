# cutlass-vp8

A VP8 decoder and encoder that follow RFC 6386. The crate is safe Rust (`#![forbid(unsafe_code)]`) and depends only on the workspace's `cutlass-decode` and `cutlass-encode` crates. The desktop app registers both directions: VP8 video in an MP4 plays through this decoder, and export can write VP8 into an MP4. The platform codec crates have no VP8 codec, so this crate does the work on every desktop.

## What it supports

VP8 is always 8-bit 4:2:0. Both directions run on one thread.

| | Decoder | Encoder |
| --- | --- | --- |
| Bitstream versions 0 to 3 | Accepted | Written (`Vp8Options::version`) |
| Key and inter frames | Accepted | Key frames on an interval, on request, and at scene cuts |
| Intra 16×16 and `B_PRED` | Accepted | Chosen by rate and distortion |
| Inter modes, including split motion vectors | Accepted | Written, on the last, golden, and altref pictures |
| Golden and altref references, copy and sign-bias flags | Accepted | Golden refreshes; optional reference copies (`copy_references`) |
| Hidden frames (`show_frame = 0`) | Accepted, no picture comes out | Optional hidden altref with lookahead (`hidden_altref`, `hidden_golden`) |
| Segmentation | Accepted | Optional four segments by luma activity (`segments`, `absolute_segments`) |
| Loop filter, normal and simple | Accepted | Level searched per frame; sharpness 0 to 7 |
| Token partitions | 1, 2, 4, or 8 | 1, 2, 4, or 8 |
| Display scale, color-space, and clamping bits | Read and reported | Written on request |
| Rate | n/a | Fixed quantizer (0 to 127) or a target bitrate |
| Size | Up to 16383×16383, odd sizes included, a new size on any key frame | Up to 16383×16383 |

What is refused:

- A frame tag with a version above 3 returns `CodecError::Unsupported`.
- A short or damaged frame returns `CodecError::Truncated` or `CodecError::Malformed`. An inter frame before any key frame is `Malformed`. A failed frame leaves the decoder as it was.
- A key frame larger than the decoder's pixel cap returns `CodecError::Limit`.
- The encoder returns `CodecError::InvalidApi` for an empty picture, a partition count other than 1, 2, 4, or 8, a quantizer above 127, a version above 3, sharpness above 7, a scale code above 3, `hidden_altref` with a golden interval outside 2 to 64, `hidden_golden` without `hidden_altref`, and `absolute_segments` without `segments`.
- `Vp8Decoder::open` and `Vp8Encoder::open` accept only `PixelFormat::Yuv420p`. `EngineEncoder::open` also accepts `Rgba8`, because export converts each frame to 4:2:0 before it reaches the encoder.

## Using it

- `Decoder` is the core decoder: `decode` takes one frame's bytes and returns `Ok(Some(Picture))` for a shown frame and `Ok(None)` for a hidden one. `with_max_pixels` sets the largest key frame it accepts.
- `Vp8Decoder` and `Vp8Encoder` implement the `VideoDecoder` and `VideoEncoder` traits. `register_decoders` and `register_encoders` put them on a `VideoDecodeFactory` and a `VideoEncodeFactory`.
- `Vp8Options` holds the encoder settings. `Vp8Encoder::with_options` opens with them, `force_keyframe` asks for a key frame, and `vpcc_record` returns the `vpcC` payload an MP4 track carries.
- `EngineDecoder` and `EngineEncoder` are what the app uses. `register_engine_decoders` and `register_engine_encoders` put them on the engine's `DecodeFactory` and `EncodeFactory`.

```rust
use cutlass_vp8::Decoder;

// Refuse key frames above about one megapixel.
let mut decoder = Decoder::with_max_pixels(1 << 20);
// `frame_bytes` is one VP8 frame, as a container hands it over.
if let Some(picture) = decoder.decode(&frame_bytes)? {
    // Planes cover whole macroblocks. Crop to `picture.width` × `picture.height`.
    let stride = picture.frame.mbw * 16;
    let first_row = &picture.frame.y[..picture.width as usize];
    let _ = (stride, first_row);
}
```

## How it is tested

Run with `cargo test -p cutlass-vp8`.

- **Checked-in key frames** (`tests/oracle.rs`). Three key frames, one from libvpx and two from libwebp, are stored with a reference decode beside each. The decoder must give the same bytes. This runs everywhere.
- **Encoder against decoders** (`src/prove_encode.rs`). The encoder runs on pictures generated in code, once per option: quantizers, odd sizes, partitions, bitrate, segments, versions, altref copies, hidden altrefs, hidden golden frames, error resilience, header bits, and scene cuts. This crate's decoder must reproduce the encoder's reconstruction exactly. When FFmpeg is installed, its `vp8` decoder and its `libvpx` decoder must give the same pictures too. Without FFmpeg that part is skipped with a message.
- **libvpx streams** (`src/prove.rs`). FFmpeg's `libvpx` encodes private sample recordings with many option sets, and the decoder must match FFmpeg's `vp8` and `libvpx` decoders exactly. One independent 1080p WebM is compared with FFmpeg as well. These recordings are not in the repository. Without them, or without FFmpeg, each test prints `skipped` and passes without checking anything.
- **Lifecycle** (`src/prove_lifecycle.rs`): reset, a failed frame leaving the state alone, malformed headers, a size change at a key frame, hidden frames, and the trait contracts for drain and cancel.
- **Fuzzing** (`decoder_fuzz_does_not_panic`, `encoder_fuzz_does_not_panic`). Both run for 2 seconds under `cargo test`. `VP8_FUZZ_SECS` and `VP8_ENC_FUZZ_SECS` set a longer run, and `VP8_FUZZ_SEED` and `VP8_ENC_FUZZ_SEED` set the seed, in decimal or `0x` hex. The decoder fuzzer feeds mutated frames to the core decoder and to `EngineDecoder`. The encoder fuzzer draws random options, sizes, and call orders, and every packet must decode to the encoder's reconstruction.
- **MP4** (`tests/isobmff.rs`). The encoder's packets go through the workspace MP4 muxer and demuxer and back through the decoder. With FFmpeg installed, FFmpeg must read the same file to the same pictures.
- **Quality gate** (`gate_quality_and_rate_against_libvpx`). Ignored by default. It compares quality and rate with libvpx on the private recordings: `cargo test --release -p cutlass-vp8 -- --ignored gate --nocapture`.

`cargo bench -p cutlass-vp8` times decode and encode beside FFmpeg and prints a skip line when FFmpeg is not installed.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, vp8.** Input: 600 frames of 1920×1080 at 30 frames a second (11 MB), written by FFmpeg with `-c:v libvpx -b:v 4M -deadline good -cpu-used 4`.

|  | Cutlass | FFmpeg (`vp8`) |
| --- | ---: | ---: |
| Wall time | 2.91 s | 1.24 s |
| Speed | 206 frames/s | 484 frames/s |
| CPU time | 2.89 s | 1.23 s |
| Peak memory | 32 MB | 25 MB |
| Threads allowed | 1 | 1 |

Cutlass is **2.3× slower** than FFmpeg here. Both decoders produced exactly the same samples.

**Encode, vp8.** Input: 300 raw frames of 1920×1080 at 30 frames a second, asked for 6 Mbit/s.

|  | Cutlass | FFmpeg (`libvpx realtime`) | FFmpeg (`libvpx good`) |
| --- | ---: | ---: | ---: |
| Wall time | 51.29 s | 5.28 s | 30.44 s |
| Speed | 5.8 frames/s | 57 frames/s | 9.9 frames/s |
| CPU time | 51.24 s | 5.32 s | 30.46 s |
| Peak memory | 48 MB | 142 MB | 142 MB |
| Output size | 7.5 MB | 7.8 MB | 7.8 MB |
| Quality kept (PSNR) | 35.53 dB | 35.11 dB | 35.85 dB |
| Threads allowed | 1 | 1 | 1 |

Cutlass is **9.7× slower** than `libvpx realtime` here. Cutlass is **1.7× slower** than `libvpx good` here. Quality is the PSNR between the source pictures and what each encoder's output decodes to; higher is closer to the source. The encoders do not make the same choices, so read speed, size and quality together.

The decoder is timed from an IVF file, read by the tool itself.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- **No WebM.** The Matroska crate is a placeholder, so `.webm` files do not open in Cutlass. VP8 is read from and written to MP4 (`vp08` with a `vpcC` box) only. The desktop app refuses a VP8 export to MOV.
- **One thread** for decode and for encode.
- **Pixel caps differ by entry point.** `EngineDecoder` refuses key frames above 8192×8192 pixels. `Decoder::new` and `Vp8Decoder::open` set no area cap, only the 16383 edge limit. A caller that reads untrusted input through them should use `Decoder::with_max_pixels`.
- **Odd sizes in playback.** `EngineDecoder` drops the last column or row of an odd-sized stream, because the engine's 4:2:0 pictures have even edges. `Decoder` and `Vp8Decoder` keep the odd size.
- **Display scale is not applied.** The scale bits are reported (`Picture::hscale`, `Picture::vscale`, `Vp8Decoder::scaling`), and no picture is resized.
- **Color.** `EngineDecoder` takes primaries and transfer from the `vpcC` bytes and falls back to BT.601. `EngineEncoder` always marks its stream BT.709, which is what export's conversion produces.
- **The clamping promise is not checked.** With `Vp8Options::clamping_type` set, the stream tells decoders that no clamping is needed. The encoder does not verify that its reconstruction stays in range.
- **Export uses default options.** `EngineEncoder` passes only the bitrate. The other `Vp8Options` fields are reachable through `Vp8Encoder::with_options` only.
- **Hidden altrefs add delay.** With `hidden_altref`, packets come up to `golden_interval` frames late (at most 64), and `drain` returns the rest.
- **Frame size limits when writing.** The first partition must stay under 512 KiB and each token partition but the last under 16 MiB, or the encoder returns `CodecError::Limit`.
- **Where the sources disagree.** RFC 6386 Section 20 (the reference decoder), libvpx and FFmpeg read a few corner cases differently. This crate's choices:
  - A segment's loop-filter level is clamped to 0..63 before the reference and mode deltas are added (`mb_filter`), as Section 20 and libvpx do. FFmpeg does not clamp.
  - A segment's quantizer index is not clamped before the per-plane deltas, as Section 20 and FFmpeg do. libvpx clamps.
  - `copy_buffer_to_golden = 2` in the same frame as an altref copy gives golden the new altref, as Section 20 and libvpx do. FFmpeg gives it the old altref.
  - A loop-filter delta whose update flag is clear keeps its value, as libvpx and FFmpeg do. Section 20 sets it to 0.
  - `mb_no_coeff_skip = 0` means skip is 0, as libvpx and FFmpeg do. Section 20 reuses a stale flag.

  The encoder never writes a stream where these readings differ.
- The crate builds at `opt-level = 3` in dev builds (workspace `Cargo.toml`), with overflow checks and debug assertions still on.

## Specifications

- RFC 6386, *VP8 Data Format and Decoding Guide*. The constant tables are transcribed from Section 20, the reference decoder source.
- *VP Codec ISO Media File Format Binding*, for the `vpcC` payload in MP4.
