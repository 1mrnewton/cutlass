# cutlass-hevc

An HEVC (H.265) video decoder in safe Rust. There is no encoder. The crate has `#![forbid(unsafe_code)]`. The desktop app registers it after the platform decoders: on macOS and Windows the platform decoder is asked first when hardware decode is on in Settings, and this decoder is used when that opener declines, when hardware decode is off, and always on Linux.

## What it supports

It takes what an MP4 or QuickTime track holds (the `hvcC` record as extradata and length-prefixed samples) or an Annex B byte stream, and returns pictures in output order, cropped by the conformance window.

| | |
| --- | --- |
| Streams | 4:2:0 at 8 to 12 bits, luma and chroma depth equal. This covers Main, Main 10, and Main Still Picture. The profile number in the stream is not checked. |
| Coding tools | Asymmetric partitions, temporal motion vector prediction, weighted prediction, PCM, 4×4 transform skip, scaling lists, transquant bypass, sign data hiding, constrained intra prediction |
| Loop filters | Deblocking with slice and PPS overrides, sample adaptive offset |
| Picture partitions | Slices, dependent slice segments, tiles, wavefronts, entry points |
| References | Reference picture sets with inter-set prediction, long-term pictures, list modification, mid-gray stand-ins for references the stream names and does not send, skipping of RASL pictures after a random access point, output by the bumping process of clause C.5.2 |
| Framing | Annex B, length prefixes of 1 to 4 bytes, `hvcC` or Annex B parameter sets as extradata |
| Tags | Color range, primaries, transfer, matrix, sample aspect, and timing from the VUI are returned with each picture |

Every picture comes back as 8-bit 4:2:0 in `VideoOutput::picture`. For a deeper stream each sample there is shifted down to 8 bits, and the full samples are also returned in `yuv420p10` (10 bits) or `yuv420_high` (9, 11, and 12 bits). The opener the engine uses (`Decoder`) returns only the packed 8-bit picture.

Refused, all as `CodecError::Unsupported`:

- 4:0:0, 4:2:2, and 4:4:4 (`"chroma format other than 4:2:0"`), and separate color planes.
- A depth above 12 bits, or luma and chroma depths that differ (`"bit depth"`).
- Range extension tools (`"range extension coding tools"`, `"range extension picture tools"`, `"transform skip larger than 4x4"`, `"sao offset scale"`).
- Multilayer, 3D, and screen content extensions (`"sps extension"`, `"pps extension"`).

NAL units of a layer other than 0 are skipped. VPS, SEI, and access unit delimiters are ignored.

## Using it

`HevcDecoder::open` takes a `CodecParams` and decodes on the process-wide worker pool. `HevcDecoder::open_with_threads` gives the decoder its own pool; 0 or 1 decodes on the caller's thread. `register_decoders` adds the decoder to a `DecodeFactory`. The decoder implements `VideoDecoder`.

One `push` is one access unit. It returns the pictures that are ready, which may be none. `drain` returns the pictures still waiting for output. A `push` that fails leaves the decoder as it was. `nal_length_size` 0 means Annex B; an `hvcC` record in `CodecParams::extra` sets the length size itself. The picture size comes from the stream, so the size in `CodecParams` can be 0.

```rust
use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};
use cutlass_hevc::HevcDecoder;

/// `hvcc` is the track's `hvcC` record. `samples` are its length-prefixed samples.
fn decode(hvcc: &[u8], samples: &[Vec<u8>]) -> Result<Vec<VideoOutput>, CodecError> {
    let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
    params.extra = hvcc.to_vec();
    let mut decoder = HevcDecoder::open(&params)?;
    let mut pictures = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        let packet = Packet::video(sample.clone(), index as i64, 1, 30, index == 0);
        pictures.extend(decoder.push(&packet)?);
    }
    pictures.extend(decoder.drain()?);
    Ok(pictures)
}
```

## How it is tested

FFmpeg is a separate program the tests call to compare against; the crate does not link it.

```text
cargo test -p cutlass-hevc
```

- **`tests/oracle.rs`.** Small streams are written by libx265 through the `ffmpeg` program, decoded here and by `ffmpeg`, and every sample of every picture must be equal, in output order. Each stream is decoded twice, as Annex B and with 4-byte length prefixes. The cases are intra only, P and B pictures, asymmetric partitions, temporal prediction off, weighted prediction, open and closed groups of pictures, many references, SAO and deblocking on and off, wavefronts, several slices, transform skip, scaling lists, lossless, constrained intra with sign hiding, 16×16 coding tree blocks, and 10-bit and 12-bit streams. One case is written by the `x265` program instead.
- **Threads.** Three cases are decoded with 1, 2, 5, and 16 workers and every picture must be the same.
- **`tests/malformed.rs`.** Hand-built parameter sets with bad or huge values must be refused, a failed `push` must leave earlier parameter sets in place, and random bytes must not panic. libx265 streams are cut short and have bits flipped, 200 rounds a stream by default (`HEVC_FUZZ_ROUNDS` changes that), and must not panic.
- **`tests/local_media.rs`.** Phone and screen recordings are read through `cutlass-isobmff` and the first 30 pictures are compared with `ffmpeg`. These files are not in the repository.
- **Unit tests in `src/`.** Bit reading, CABAC setup, the transform matrix, intra modes, motion vector scaling, sample interpolation, and NAL framing.
- A 600-frame 1080p Main-profile stream written by libx265 decodes to the same bytes as FFmpeg's decoder (see the speed section).

What does not run by default:

- A test in `tests/oracle.rs` prints `skipped` and passes when `ffmpeg` is not on the `PATH` or has no libx265. The corrupted-stream test does the same. Without that program, no test decodes a whole picture.
- The tests in `tests/local_media.rs` pass without checking anything when the media folder is missing.
- Two tests are `#[ignore]` and run by hand: `from_environment` checks one case named by environment variables, and `bench_from_environment` times a stream named by `HEVC_BENCH`.
- No test builds a stream with tiles, PCM blocks, long-term reference pictures, or dependent slice segments. Those paths are not verified against another decoder.
- The ITU-T conformance bitstreams are not in the repository and have not been run.
- There is no bench target and no example.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, hevc.** Input: 600 frames of 1920×1080 at 30 frames a second (14 MB), written by FFmpeg with `-c:v libx265 -preset medium -crf 26 -tag:v hvc1 -x265-params log-level=error`.

|  | Cutlass | FFmpeg (`hevc`) |
| --- | ---: | ---: |
| Wall time | 2.23 s | 0.86 s |
| Speed | 269 frames/s | 698 frames/s |
| CPU time | 10.69 s | 2.56 s |
| Peak memory | 61 MB | 165 MB |
| Threads allowed | 16 | 16 |

Cutlass is **2.6× slower** than FFmpeg here. Both decoders produced exactly the same samples.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value | Error past it |
| --- | --- | --- |
| Picture edge | 16,888 samples | `CodecError::Limit { what: "picture size" }` |
| Picture area | 35,651,584 luma samples (level 6.2) | the same |
| Pictures in the buffer | 16 | `CodecError::Malformed` |
| NAL units in one access unit | 4096 | `Limit { what: "nal units in one access unit" }` |
| Parameter set ids | 16 SPS, 64 PPS | `Malformed` |
| Workers | 16 | |

Threads: the coding tree block rows of a wavefront slice segment decode on separate workers, each row two blocks behind the row above. Deblocking and SAO run in bands of rows for every stream. A stream without wavefronts, or one that uses tiles, decodes its blocks on one thread; only the loop filters use the pool.

Gaps:

- No error concealment. A damaged access unit is an error and the decoder keeps its earlier state.
- SEI is not read, so HDR metadata carried there (mastering display, content light level) is not returned.
- `field_seq_flag` is read and ignored. No code pairs fields into frames.
- All sample planes are held as 16-bit values, also for an 8-bit stream.

## Specifications

- Recommendation ITU-T H.265, also published as ISO/IEC 23008-2: clauses 6 to 9, Annex B (byte stream), Annex C.5.2 (output order), and Annex E (VUI).
- ISO/IEC 14496-15 clause 8.3.3, for the `hvcC` record and length-prefixed samples.
