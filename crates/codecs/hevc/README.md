# cutlass-hevc

An HEVC (H.265) video decoder in safe Rust. There is no encoder. The crate has `#![forbid(unsafe_code)]`; its vector kernels (interpolation, inverse transforms, deblocking, SAO) are in [`cutlass-simd`](../../simd), which checks at run time that the CPU has AVX2 and otherwise leaves the work to this crate's scalar code. The desktop app registers it after the platform decoders: on macOS and Windows the platform decoder is asked first when hardware decode is on in Settings, and this decoder is used when that opener declines, when hardware decode is off, and always on Linux.

It is built the way [`cutlass-av1`](../av1)'s decoder is: safe Rust over the kernels of `cutlass-simd`; references shared between pictures decoding at once through `cutlass_simd::shared_rows`, whose rows readers see as they become final; `register_decoders` for the engine's `DecodeFactory`; a `decode` bench (`harness = false`) that times it against FFmpeg; `dec`, `bench`, and `fuzz` examples; and this README's sections.

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
- **Threads.** Three cases are decoded with 1, 2, 5, and 16 workers, and with the rows of each picture on several threads (one picture on three, two on two, two on four), and every picture must be the same.
- **`tests/malformed.rs`.** Hand-built parameter sets with bad or huge values must be refused, a failed `push` must leave earlier parameter sets in place, and random bytes must not panic. libx265 streams are cut short and have bits flipped, 200 rounds a stream by default (`HEVC_FUZZ_ROUNDS` changes that), and must not panic.
- **`tests/local_media.rs`.** Phone and screen recordings are read through `cutlass-isobmff` and the first 30 pictures are compared with `ffmpeg`. These files are not in the repository.
- **Unit tests in `src/`.** Bit reading, the CABAC engine against a byte-at-a-time engine written from clause 9.3.4.3, the transform matrix, intra modes, motion vector scaling, sample interpolation, emulation prevention, and NAL framing.
- **Kernels.** Each vector kernel in `cutlass-simd` (`hevc::mc` and `hevc::mc16` for 8 and for 9 to 12 bits, `hevc::idct`, `hevc::deblock`, `hevc::sao` at 8, 9, 10 and 12 bits) is compared with a scalar version written from the specification on random blocks, sizes and edges, and `shared_rows` with the frames it lends out as output: `cargo test -p cutlass-simd`.
- **`examples/`.** As for `cutlass-av1` (`av1dec`, `av1bench`, `av1fuzz`): `hevcdec <in> <out.yuv> [threads]` writes the pictures of an MP4, QuickTime or Annex B stream as raw planes (the bytes `ffmpeg -f rawvideo` writes); `hevcbench <in> [runs] [threads] [frames rows]` times whole decodes; `hevcfuzz <iterations> <in>...` flips bits, overwrites bytes, cuts access units short and scrambles their headers, decodes each result with 1, 2 and 4 threads in turn, and fails on any panic or on a decode not finished within 60 seconds. On 13 November 2026, 3,000 iterations on each of four small libx265 streams (8-bit with B pictures and weighted prediction, 10-bit with slices and transform skip, lossless with 16×16 blocks, 12-bit) and 150 on a 1080p clip ran with no panic and no hang.
- **`benches/decode.rs`.** `cargo bench -p cutlass-hevc` writes 240 pictures of 1080p with libx265 (or takes `HEVC_BENCH=<file>`), checks that the samples equal FFmpeg's, then times both with the same number of threads. Without `ffmpeg` it prints `skip:`.
- A 600-frame 1080p Main-profile stream written by libx265 decodes to the same bytes as FFmpeg's decoder (see the speed section).

What does not run by default:

- A test in `tests/oracle.rs` prints `skipped` and passes when `ffmpeg` is not on the `PATH` or has no libx265. The corrupted-stream test does the same. Without that program, no test decodes a whole picture.
- The tests in `tests/local_media.rs` pass without checking anything when the media folder is missing.
- Two tests are `#[ignore]` and run by hand: `from_environment` checks one case named by environment variables, and `bench_from_environment` times a stream named by `HEVC_BENCH`.
- No test builds a stream with tiles, PCM blocks, long-term reference pictures, or dependent slice segments. Those paths are not verified against another decoder.
- The ITU-T conformance bitstreams are not in the repository and have not been run.

## Speed against FFmpeg

**Decode, hevc.** Measured on 14 November 2026 on branch `hevc`, on a cloud VM with 4 cores of an Intel Xeon at 2.3 GHz, one thread a core (other VMs share the host; the same run varies by 5% to 10% from one minute to the next), against FFmpeg 6.1.1 from Ubuntu 24.04. Built with `cargo build --release -p media-bench` (no profile) and rustc 1.97.0. The Rust side is [`tools/media-bench`](../../../tools/media-bench); the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly. Each is a whole process, start-up included, run in turn nine times; the tables show the median run. "4 threads" gives each decoder four threads on the four cores; "1 core" pins each to one core with one thread. Every 8-bit input decodes to exactly FFmpeg's samples. The 8-bit rows are from commit `671fb7d`, the 10-bit rows from `6999185` (fifteen runs), which changed only 10-bit decoding.

The inputs were all written by libx265 3.5 with 64×64 coding tree blocks, SAO, weighted prediction, and three reference pictures (two for 2160p):

| Input | Size | Frames | Written with |
| --- | --- | ---: | --- |
| `mandel` | 1920×1080 | 600 | preset medium, CRF 26, 4 B pictures, wavefronts (21 MB) |
| `m120`, `mid40` | 1920×1080 | 120, 40 | the same settings, other scenes |
| `small` | 480×272 | 600 | the same settings |
| `nowpp` | 1920×1080 | 300 | CRF 28, 4 B pictures, no wavefronts |
| `pframes` | 1920×1080 | 300 | CRF 28, P pictures only, each referencing the last |
| `uhd` | 3840×2160 | 90 | CRF 28, 4 B pictures, wavefronts |
| `ten` | 1920×1080, 10-bit | 240 | CRF 28, 4 B pictures, wavefronts |

Wall time, Cutlass against FFmpeg (`hevc`), and Cutlass's time as a fraction of FFmpeg's:

| Input | 4 threads | | | 1 core | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | Cutlass | FFmpeg | ratio | Cutlass | FFmpeg | ratio |
| `mandel` | 1.63 s | 1.99 s | **0.82** | 5.14 s | 5.76 s | **0.89** |
| `m120` | 0.30 s | 0.37 s | **0.81** | 0.86 s | 0.96 s | **0.90** |
| `mid40` | 0.11 s | 0.18 s | **0.61** | 0.31 s | 0.38 s | **0.82** |
| `small` | 0.19 s | 0.30 s | **0.63** | 0.40 s | 0.42 s | **0.95** |
| `nowpp` | 0.68 s | 0.94 s | **0.72** | 2.01 s | 2.17 s | **0.93** |
| `pframes` | 0.91 s | 0.92 s | **0.99** | 2.61 s | 2.83 s | **0.92** |
| `uhd` | 0.77 s | 0.83 s | **0.93** | 2.25 s | 2.58 s | **0.87** |
| `ten`, FFmpeg made 8-bit | 0.85 s | 1.15 s | **0.74** | 2.75 s | 3.47 s | **0.79** |
| `ten`, FFmpeg left 10-bit | 0.85 s | 0.96 s | **0.89** | 2.75 s | 2.81 s | **0.98** |

Peak memory (resident set), Cutlass against FFmpeg:

| Input | 4 threads | 1 core |
| --- | ---: | ---: |
| `mandel` | 104 MB / 99 MB | 28 MB / 68 MB |
| `m120` | 90 MB / 96 MB | 28 MB / 67 MB |
| `mid40` | 73 MB / 99 MB | 28 MB / 69 MB |
| `small` | 15 MB / 46 MB | 9 MB / 42 MB |
| `nowpp` | 92 MB / 96 MB | 28 MB / 67 MB |
| `pframes` | 83 MB / 83 MB | 25 MB / 63 MB |
| `uhd` | 276 MB / 252 MB | 87 MB / 156 MB |
| `ten` | 185 MB / 124 MB | 56 MB / 86 MB |

Cutlass is faster than FFmpeg on every input and both settings. The closest is `pframes` with four threads, a chain of P pictures where every picture waits on the rows of the one before: 0.99 in this run, and in longer runs in turn on their own 0.95 (21 runs: 0.90 s against 0.95 s) and 0.95 again (31 runs: 0.99 s against 1.04 s); runs of 7 and 15 earlier the same day measured 1.00 and 1.02. Cutlass's engine decoder returns 8-bit pictures, so the first `ten` row has the FFmpeg program convert each 10-bit frame to `yuv420p` (its `to:yuv420p` mode) for the same output; the second leaves FFmpeg's frames as it decodes them, 10-bit, which is less work than Cutlass does. In that second comparison on one core the two are within noise: 15 runs in turn gave 0.98 here, and 0.97 (median) in a run of its own, where FFmpeg's fastest run was 7% faster than Cutlass's fastest. Processor time is lower than FFmpeg's on every input on one core. With four threads it is lower on `mandel`, `m120`, `mid40`, `nowpp`, and `ten` against FFmpeg made 8-bit; within 2% above it on `small`, `pframes`, and `uhd`; and 12% above it on `ten` against FFmpeg left in 10-bit (2.91 s against 2.60 s).

**Decode alone.** `HEVC_BENCH=<mandel> cargo bench -p cutlass-hevc`, which pushes the access units straight into `HevcDecoder` with four threads and times FFmpeg's `ffmpeg -f null` on the same file, measured **1.78 s against FFmpeg's 2.19 s** (1.23× faster) on the same machine as the tables, with the samples checked equal to FFmpeg's first.

Where the time goes, on one core against FFmpeg's own profile of `nowpp`: entropy decoding (CABAC and residual coding) takes about 10% less time than FFmpeg's, interpolation 25% less, and deblocking with its boundary strengths 15% less; motion vector derivation and the setup of each prediction block take about 40% more, and the copies of final rows into frames twice as long.

What made it faster, in order of the gain measured on this machine: decoding into a ring of CTB-row buffers that stay in cache instead of a picture-sized buffer per picture, and the same for the per-4×4 maps of motion, modes and QP; a picture waiting for a reference lending its place to another (1.15× from 1.00× to 1.05× with four threads); interpolation kernels that read each row of the reference window as one fixed-size array (1.34G instructions to 0.85G on 40 frames, against FFmpeg's 0.91G); asking for every line of a block's reference window before filtering it (5% to 9% on one core, 5% with four threads); the 16-bit interpolation, SAO and weighting kernels for deeper samples, then interpolation that writes their samples itself, weighted, as the 8-bit kernels do (13%), and deep pictures no picture references shifted straight into the output instead of kept in a 16-bit frame (4% to 5%, 25 MB less): the 10-bit stream from 2.07× the time of FFmpeg left in 10-bit frames to 0.89×; significance and greater-than-one flags handled without a branch on the decoded value; deblocking each CTB row as soon as it is complete and making all but its last eight lines final right then, on the thread that built it (6% with four threads); output pictures that are their frame's own samples, one buffer shared with the caller, instead of a copy (30 to 40 MB less memory with four threads on 1080p, 137 MB on 2160p, and 1% to 3% faster); clearing each band of a frame or an output picture before SAO writes it, so its lines come into cache without being read (3%); one picture per thread instead of wavefront rows split between threads (2% wall time, 5% processor time); intra prediction computed a row at a time over whole slices of the references, and the availability of neighbours found once per CTB (1778M instructions to 1663M on 24 frames, with the merge and AMVP lists that stop at the candidate chosen); the inverse transform adding its residual to the samples itself; the CABAC engine kept in registers through residual coding; wavefront row progress reported without a lock or a system call. Tried and dropped: an AVX-512 interpolation kernel (slower on this CPU), waiting further behind a reference (10% to 30% slower), fewer pictures in flight, spinning before sleeping on a reference's rows, giving a free slot to the earliest picture first (2% to 4% slower), asking for the chroma windows before luma is filtered (2% to 4% slower), and SAO edge offsets over a whole CTB in one call (no gain: the rows' lookup cost what the calls saved).

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`; or `HEVC_BENCH=<clip> cargo bench -p cutlass-hevc`. For a 10-bit clip, time the FFmpeg program with `to:yuv420p` for the same output.

## Limits and gaps

| Limit | Value | Error past it |
| --- | --- | --- |
| Picture edge | 16,888 samples | `CodecError::Limit { what: "picture size" }` |
| Picture area | 35,651,584 luma samples (level 6.2) | the same |
| Pictures in the buffer | 16 | `CodecError::Malformed` |
| NAL units in one access unit | 4096 | `Limit { what: "nal units in one access unit" }` |
| Parameter set ids | 16 SPS, 64 PPS | `Malformed` |
| Workers | 16 | |

Threads: several pictures decode at once, each on a thread of its own, a picture waiting only for the rows of its references that it reads. Up to eight threads every thread takes a picture, and half as many pictures again are started: no more pictures decode at once than there are threads, and a picture that waits for rows or motion of a reference gives its place to one of the others until the reference catches up. Past eight threads, the CTB rows of a wavefront slice segment share the threads, each row two CTBs behind the one above (`HevcDecoder::set_threads` chooses either split). Each CTB row is deblocked as soon as it is complete and its motion is kept, and SAO then makes final all but its last eight lines (the rest wait for the row below), on the thread that finished the row; those lines reach the pictures that reference it at once. A stream with tiles decodes its CTBs in tile order on one thread per picture. The output does not depend on the number of threads.

Gaps:

- No error concealment. A damaged access unit is an error and the decoder keeps its earlier state.
- SEI is not read, so HDR metadata carried there (mastering display, content light level) is not returned.
- `field_seq_flag` is read and ignored. No code pairs fields into frames.
- The vector kernels need x86-64 with AVX2; other CPUs use the scalar code, which is several times slower. For samples above 8 bits, interpolation, default weighting, SAO, and the inverse transforms have kernels; deblocking and explicit weighting of those samples are scalar code compiled for AVX2.
- A picture of a cropped stream is written to its frame and then copied to the output; so is a picture above 8 bits that a later picture references (shifted down to 8 bits). Other 8-bit pictures are their frame's own samples, shared with the caller; other deeper ones go straight to the output.
- With four threads the decoder holds up to six pictures at once. On 8-bit 1080p its peak memory is between 26 MB below and 5 MB above FFmpeg's, on 2160p 24 MB above; on the 10-bit stream it is 61 MB above (185 MB against 124 MB): its reference pictures are 16-bit frames, and every picture it outputs also has its own 8-bit copy until it is taken. On one core it takes less memory than FFmpeg on every input.

## Specifications

- Recommendation ITU-T H.265, also published as ISO/IEC 23008-2: clauses 6 to 9, Annex B (byte stream), Annex C.5.2 (output order), and Annex E (VUI).
- ISO/IEC 14496-15 clause 8.3.3, for the `hvcC` record and length-prefixed samples.
