# cutlass-hevc

An HEVC (H.265) video decoder in safe Rust. There is no encoder. The crate has `#![forbid(unsafe_code)]`; its vector kernels (interpolation, inverse transforms, deblocking, SAO) are in [`cutlass-simd`](../../simd), which checks at run time that the CPU has AVX2 and otherwise leaves the work to this crate's scalar code. The desktop app registers it after the platform decoders: on macOS and Windows the platform decoder is asked first when hardware decode is on in Settings, and this decoder is used when that opener declines, when hardware decode is off, and always on Linux.

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

`HevcDecoder::open` takes a `CodecParams` and may use as many threads as the machine runs at once, at most 16. `HevcDecoder::open_with_threads` sets that number; 0 or 1 decodes every picture on the caller's thread. `set_threads(frames, rows)` fixes how many pictures decode at once and how many threads take the CTB rows of a picture with wavefronts; without it the decoder chooses per picture (see Threads below). `register_decoders` adds the decoder to a `DecodeFactory`. The decoder implements `VideoDecoder`.

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
- **Unit tests in `src/`.** Bit reading, the CABAC engine against a byte-at-a-time engine written from clause 9.3.4.3, the transform matrix, intra modes, motion vector scaling, sample interpolation, emulation prevention, and NAL framing.
- **Kernels.** Each vector kernel in `cutlass-simd` (`hevc::mc`, `hevc::idct`, `hevc::deblock`, `hevc::sao`) is compared with a scalar version written from the specification on random blocks, sizes and edges: `cargo test -p cutlass-simd`.
- **`examples/`.** `hevcdec <in> <out.yuv> [threads]` writes the pictures of an MP4, QuickTime or Annex B stream as raw planes (the bytes `ffmpeg -f rawvideo` writes); `hevcbench <in> [runs] [threads] [frames rows]` times whole decodes.
- **`benches/decode.rs`.** `cargo bench -p cutlass-hevc` writes 240 pictures of 1080p with libx265 (or takes `HEVC_BENCH=<file>`), checks that the samples equal FFmpeg's, then times both with the same number of threads. Without `ffmpeg` it prints `skip:`.
- A 600-frame 1080p Main-profile stream written by libx265 decodes to the same bytes as FFmpeg's decoder (see the speed section).

What does not run by default:

- A test in `tests/oracle.rs` prints `skipped` and passes when `ffmpeg` is not on the `PATH` or has no libx265. The corrupted-stream test does the same. Without that program, no test decodes a whole picture.
- The tests in `tests/local_media.rs` pass without checking anything when the media folder is missing.
- Two tests are `#[ignore]` and run by hand: `from_environment` checks one case named by environment variables, and `bench_from_environment` times a stream named by `HEVC_BENCH`.
- No test builds a stream with tiles, PCM blocks, long-term reference pictures, or dependent slice segments. Those paths are not verified against another decoder.
- The ITU-T conformance bitstreams are not in the repository and have not been run.

## Speed against FFmpeg

**Decode, hevc.** Measured on 13 November 2026 on branch `hevc`, on a cloud VM with 4 cores of an Intel Xeon at 2.1 GHz (other VMs share the host; the same run varies by about 10% from one minute to the next), against FFmpeg 6.1.1 from Ubuntu 24.04. Built with `cargo build --release -p media-bench` (no profile) and rustc 1.97.0. Input: 600 frames of 1920×1080 at 30 frames a second (21 MB), written by libx265 at its medium preset with wavefronts. The Rust side is [`tools/media-bench`](../../../tools/media-bench); the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly. Each is a whole process, run in turn (seven times with four threads, five with one); the table shows the median run.

|  | Cutlass | FFmpeg (`hevc`) |
| --- | ---: | ---: |
| Wall time, 4 threads | 3.11 s | 2.74 s |
| Speed, 4 threads | 193 frames/s | 219 frames/s |
| CPU time, 4 threads | 9.89 s | 8.05 s |
| Wall time, 1 thread | 8.49 s | 7.78 s |
| Peak memory, 4 threads | 77 MB | 99 MB |
| Threads allowed | 4 | 4 |

Cutlass is **1.13× slower** than FFmpeg here with four threads and **1.09× slower** on one. Both decoders produced exactly the same samples. Runs of the same comparison on the same day measured between 1.03× and 1.19× with four threads (median of seven, later the same day: 3.01 s against 2.74 s, 1.10×). `media-bench` also copies every picture into the packed buffer the engine's `Decoder` returns, which the FFmpeg program does not do; `cargo bench -p cutlass-hevc`, which times the decode alone, measured 2.95 s against FFmpeg's 2.88 s (one process, startup included) on this file.

Where the time goes, on one thread and against FFmpeg's own profile on the same file: entropy decoding (CABAC and residual coding) takes about 30% less time than FFmpeg's; the boundary strengths of deblocking about 15% less; interpolation and the motion-vector derivation around it, the deblocking filters, and the writing of final rows and output pictures take more.

What made it faster, in order of the gain measured on this machine: decoding into a ring of four CTB-row buffers that stay in cache instead of a picture-sized buffer per picture, and the same for the per-4×4 maps of motion, modes and QP; interpolation kernels that read each row of the reference window as one fixed-size array (1.34G instructions to 0.85G on 40 frames, against FFmpeg's 0.91G); significance and greater-than-one flags handled without a branch on the decoded value; splitting four threads into two pictures of two wavefront rows; writing the final rows of pictures no other picture references straight to the output; the inverse transform adding its residual to the samples itself; the CABAC engine kept in registers through residual coding.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`; or `HEVC_BENCH=<clip> cargo bench -p cutlass-hevc`.

## Limits and gaps

| Limit | Value | Error past it |
| --- | --- | --- |
| Picture edge | 16,888 samples | `CodecError::Limit { what: "picture size" }` |
| Picture area | 35,651,584 luma samples (level 6.2) | the same |
| Pictures in the buffer | 16 | `CodecError::Malformed` |
| NAL units in one access unit | 4096 | `Limit { what: "nal units in one access unit" }` |
| Parameter set ids | 16 SPS, 64 PPS | `Malformed` |
| Workers | 16 | |

Threads: several pictures decode at once, each on a thread of its own, a picture waiting only for the rows of its references that it reads; the CTB rows of a wavefront slice segment decode on threads of their own, each row two CTBs behind the one above. With four threads or more, a picture with wavefronts takes two threads and half as many pictures decode at once; otherwise every thread takes a picture. Each CTB row is deblocked on whichever thread finishes it, as soon as it is complete and its motion is kept, and gets SAO once the row below is deblocked. A stream with tiles decodes its CTBs in tile order on one thread per picture. The output does not depend on the number of threads.

Gaps:

- No error concealment. A damaged access unit is an error and the decoder keeps its earlier state.
- SEI is not read, so HDR metadata carried there (mastering display, content light level) is not returned.
- `field_seq_flag` is read and ignored. No code pairs fields into frames.
- The vector kernels take 8-bit samples on x86-64 with AVX2. Deeper streams, and other CPUs, use the scalar code, which is several times slower.
- A picture that a later picture references, and every picture of a cropped stream, is written to its frame and then copied to the output.

## Specifications

- Recommendation ITU-T H.265, also published as ISO/IEC 23008-2: clauses 6 to 9, Annex B (byte stream), Annex C.5.2 (output order), and Annex E (VUI).
- ISO/IEC 14496-15 clause 8.3.3, for the `hvcC` record and length-prefixed samples.
