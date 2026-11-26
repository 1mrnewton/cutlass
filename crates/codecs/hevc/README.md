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
| Tags | Color range, primaries, transfer, matrix, sample aspect, and timing from the VUI are returned with each picture; the engine's pictures also carry the three code points as coded, the chroma sample position, and the SEI messages below |
| SEI | Mastering display colour volume (137), content light level (144), alternative transfer characteristics (147), user data registered by ITU-T T.35 (4, which carries HDR10+), and `pic_struct` of picture timing (1), in prefix SEI units of a sample or of the `hvcC` record; T.35 also in suffix units |

There are two ways to take pictures, and both return the samples of the stream as they are, with no conversion.

**The engine's pictures** (`HevcDecoder::push_pictures`, and the `Decoder` that `register_decoders` gives the engine's `DecodeFactory`) are one buffer each, luma then Cb then Cr with no gap, at the depth of the stream:

| Stream | `PictureInfo::pixel` | A sample |
| --- | --- | --- |
| 8 bits | `Yuv420p` | one byte |
| 10 bits | `Yuv420p10` | two bytes, low byte first, the bytes `ffmpeg -pix_fmt yuv420p10le` writes |
| 12 bits | `Yuv420p12` | two bytes, low byte first (`yuv420p12le`) |
| 9 and 11 bits | `Yuv420p10` and `Yuv420p12` | doubled into the next depth: there is no format for 9 or 11 bits. Studio swing keeps its levels exactly (black 32 of 9 bits is 64 of 10); full swing white comes out one code low (1022 for 511) |

Until 24 November 2026 these pictures were always `Yuv420p`, each deeper sample cut down to 8 bits, with primaries and transfer as their only description. The buffer is shared with the decoder (`PictureOutput::cpu_shared`), which fills it again for a later picture once the caller has let it go, at every depth.

`PictureInfo` holds what the stream says: `range` (studio swing when the VUI names none, as clause E.3.1 infers), `primaries`, `transfer`, `matrix`, and the code points behind them in `cicp_primaries`, `cicp_transfer` and `cicp_matrix` (2 when the VUI names none); `chroma_position` (`Vertical` for sample location 0, the default, and `Colocated` for 2); `num_units_in_display_tick` and `time_scale`; `mastering_display` and `content_light` from the SEI messages. A mastering display or light level lasts from the access unit that carries it to the end of its coded video sequence, so the pictures between two IDR pictures are described too; one in the `hvcC` record holds for every sequence that sends none of its own. An alternative transfer characteristics message replaces the VUI's transfer in both `transfer` and `cicp_transfer`: a broadcast HLG stream whose VUI says BT.2020 (14) is returned as HLG (18), as FFmpeg returns it. The mastering display is converted from the message's steps (0.00002 for chromaticity, 0.0001 cd/m² for luminance, green first) to `MasteringDisplay`'s fixed point, red first; the chromaticity and the minimum luminance convert back exactly, the maximum luminance to the nearest 1/256 cd/m². T.35 messages are in `PictureOutput::t35`, in stream order, for the picture of their access unit, however the pictures are reordered; at most 32 are kept for one picture. `HevcPicture::timing` is `pic_struct`, `source_scan_type` and `duplicate_flag` of the picture's timing message, when the VUI says the message has them.

An SEI message that is cut short, or that says it is longer than its NAL unit, is dropped with the messages after it in that unit; it is never an error, and the picture decodes.

**`VideoDecoder`** (`VideoOutput`) returns each plane on its own: the 8-bit planes in `picture` and, for a deeper stream, the full samples in `yuv420p10` (10 bits) or `yuv420_high` (9, 11, and 12 bits), with `picture` each sample shifted down to 8 bits, which is how `cutlass-decode` defines that field (a shift, not a rounding). Its `transfer` follows the alternative transfer characteristics message too. `VideoOutput` has no place for the HDR messages.

What has no place in the engine's types and is not returned:

- A Dolby Vision RPU (NAL unit type 62) or enhancement layer (63). `PictureOutput` carries T.35 messages and nothing else beside the samples, so these units are skipped, as every unit type the decoder does not read is.
- `pic_struct` through the `Decoder` the engine opens: `PictureOutput` has no field for it. `push_pictures` returns it.
- The sample aspect ratio in `PictureInfo`: `render_width` and `render_height` are the cropped size. `VideoOutput::sample_aspect` has it.

In the engine a picture's `matrix` now reaches `cutlass-color`, which has weights for BT.709, BT.601 and BT.2020. A stream that names another matrix (FCC, SMPTE 240M, YCgCo, ICtCp and the like) was converted as BT.709 before, because the matrix was not passed on, and is now a color miss, as it is for AV1.

Refused, all as `CodecError::Unsupported`:

- 4:0:0, 4:2:2, and 4:4:4 (`"chroma format other than 4:2:0"`), and separate color planes.
- A depth above 12 bits, or luma and chroma depths that differ (`"bit depth"`).
- Range extension tools (`"range extension coding tools"`, `"range extension picture tools"`, `"transform skip larger than 4x4"`, `"sao offset scale"`).
- Multilayer, 3D, and screen content extensions (`"sps extension"`, `"pps extension"`).

NAL units of a layer other than 0 are skipped, in a sample and in the `hvcC` record (until 24 November 2026 a parameter set of another layer in the record was read as one of layer 0, and `open` failed on it or it replaced the real one). VPS, access unit delimiters, and the SEI messages not named above are ignored.

## Using it

`HevcDecoder::open` takes a `CodecParams` and may use as many threads as the machine runs at once, at most 16. `HevcDecoder::open_with_threads` sets that number; 0 or 1 decodes every picture on the caller's thread. `set_threads(frames, rows)` fixes how many pictures decode at once and how many threads take the CTB rows of a picture with wavefronts; without it the decoder chooses per picture (see Threads below). `register_decoders` adds the decoder to a `DecodeFactory`. The decoder implements `VideoDecoder`.

One `push` is one access unit. It returns the pictures that are ready, which may be none. `push_pictures` and `drain_pictures` follow the same rules and return the engine's pictures (`HevcPicture`); a decoder is used through one pair or the other, not both. `drain` returns the pictures still waiting for output. A `push` that fails leaves the decoder as it was. `nal_length_size` 0 means Annex B; an `hvcC` record in `CodecParams::extra` sets the length size itself. The picture size comes from the stream, so the size in `CodecParams` can be 0.

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
- **`tests/packed.rs`.** Six libx265 streams of six pictures each are in `tests/fixtures` (8-bit studio and full swing, 10-bit PQ with a mastering display, light levels and an HDR10+ message on every picture, 10-bit HLG named only by the alternative transfer characteristics message and cropped from 72×48 to 70×46, 12-bit, and fields with `pic_struct`), each with an `.expect` file: a hash of the samples `ffmpeg -f rawvideo` wrote for every picture and the lines `ffprobe -show_frames` printed for it. The engine's pictures must have those hashes and those lines with 1, 2 and 4 threads; this runs with no FFmpeg. When `ffmpeg` and `ffprobe` are installed the streams are decoded by them again and every byte and line is compared. Other tests: HDR10+ stays with its picture through reordering, T.35 in a suffix unit, the description lasting between IDR pictures, the `VideoDecoder` planes equal to the packed bytes, the registered opener, buffers filled again at every depth, 40 rounds of broken SEI units that must not change a sample, an `hvcC` record with parameter sets of layer 1 and a light level, and `reset`. `HEVC_FIXTURES=write` writes the fixtures again.
- **Threads.** Three cases are decoded with 1, 2, 5, and 16 workers, and with the rows of each picture on several threads (one picture on three, two on two, two on four), and every picture must be the same.
- **`tests/malformed.rs`.** Hand-built parameter sets with bad or huge values must be refused, a failed `push` must leave earlier parameter sets in place, and random bytes must not panic. libx265 streams are cut short and have bits flipped, 200 rounds a stream by default (`HEVC_FUZZ_ROUNDS` changes that), and must not panic.
- **`tests/local_media.rs`.** Phone and screen recordings are read through `cutlass-isobmff` and the first 30 pictures are compared with `ffmpeg`. These files are not in the repository.
- **Unit tests in `src/`.** SEI messages (each kind, long types and sizes, short and cut messages, random bytes), the packing of deep samples, bit reading, the CABAC engine against a byte-at-a-time engine written from clause 9.3.4.3, the transform matrix, intra modes, motion vector scaling, sample interpolation, emulation prevention, and NAL framing.
- **Kernels.** Each vector kernel in `cutlass-simd` (`hevc::mc` and `hevc::mc16` for 8 and for 9 to 12 bits, `hevc::idct`, `hevc::deblock`, `hevc::sao` at 8, 9, 10 and 12 bits) is compared with a scalar version written from the specification on random blocks, sizes and edges, and `shared_rows` with the frames it lends out as output: `cargo test -p cutlass-simd`.
- **`examples/`.** As for `cutlass-av1` (`av1dec`, `av1bench`, `av1fuzz`): `hevcdec <in> <out.yuv> [threads]` writes the pictures of an MP4, QuickTime or Annex B stream as raw planes (the bytes `ffmpeg -f rawvideo` writes); `hevcbench <in> [runs] [threads] [frames rows]` times whole decodes; `hevcfuzz <iterations> <in>...` flips bits, overwrites bytes, cuts access units short and scrambles their headers, decodes each result with 1, 2 and 4 threads in turn, and fails on any panic or on a decode not finished within 60 seconds. On 13 November 2026, 3,000 iterations on each of four small libx265 streams (8-bit with B pictures and weighted prediction, 10-bit with slices and transform skip, lossless with 16×16 blocks, 12-bit) and 150 on a 1080p clip ran with no panic and no hang.
- **`benches/decode.rs`.** `cargo bench -p cutlass-hevc` writes 240 pictures of 1080p with libx265 (or takes `HEVC_BENCH=<file>`), checks that the engine's pictures equal the samples FFmpeg writes at the depth of the stream, then times both with the same number of threads. Without `ffmpeg` it prints `skip:`. (Until 24 November 2026 it compared the 8-bit planes and so failed on any stream above 8 bits.)
- A 600-frame 1080p Main-profile stream written by libx265 decodes to the same bytes as FFmpeg's decoder (see the speed section).

What does not run by default:

- A test in `tests/oracle.rs` prints `skipped` and passes when `ffmpeg` is not on the `PATH` or has no libx265. The corrupted-stream test does the same. Without that program, the fixtures of `tests/packed.rs` are the only whole pictures decoded.
- No test has a 9-bit or 11-bit stream (libx265 writes neither); only the packing of their samples is tested. No test has a Dolby Vision stream.
- The tests in `tests/local_media.rs` pass without checking anything when the media folder is missing.
- Two tests are `#[ignore]` and run by hand: `from_environment` checks one case named by environment variables, and `bench_from_environment` times a stream named by `HEVC_BENCH`.
- No test builds a stream with tiles, PCM blocks, long-term reference pictures, or dependent slice segments. Those paths are not verified against another decoder.
- The ITU-T conformance bitstreams are not in the repository and have not been run.

## Speed against FFmpeg

**Decode, hevc.** Measured on 14 November 2026 on branch `hevc` at commit `a52c5d3`, on a cloud VM with 4 cores of an Intel Xeon at 2.3 GHz, one thread a core (other VMs share the host; the same run varies by 5% to 10% from one minute to the next), against FFmpeg 6.1.1 from Ubuntu 24.04. Built with `cargo build --release -p media-bench` (no profile) and rustc 1.97.0. The Rust side is [`tools/media-bench`](../../../tools/media-bench); the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly. Each is a whole process, start-up included, run in turn fifteen times, Cutlass then FFmpeg; the tables show the median run, and "wins" the runs of the fifteen pairs in which Cutlass finished first. "4 threads" gives each decoder four threads on the four cores; "1 core" pins each to one core with one thread. Every 8-bit input decodes to exactly FFmpeg's samples.

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

Wall time, Cutlass against FFmpeg (`hevc`), Cutlass's time as a fraction of FFmpeg's, and the pairs Cutlass won:

| Input | 4 threads | | | | 1 core | | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | Cutlass | FFmpeg | ratio | wins | Cutlass | FFmpeg | ratio | wins |
| `mandel` | 1.55 s | 1.96 s | **0.79** | 15/15 | 4.82 s | 5.64 s | **0.86** | 15/15 |
| `m120` | 0.30 s | 0.40 s | **0.75** | 15/15 | 0.88 s | 0.98 s | **0.90** | 15/15 |
| `mid40` | 0.11 s | 0.19 s | **0.58** | 15/15 | 0.31 s | 0.39 s | **0.80** | 15/15 |
| `small` | 0.19 s | 0.31 s | **0.61** | 15/15 | 0.40 s | 0.43 s | **0.93** | 13/15 |
| `nowpp` | 0.68 s | 0.97 s | **0.70** | 15/15 | 2.13 s | 2.31 s | **0.92** | 15/15 |
| `pframes` | 0.90 s | 0.96 s | **0.94** | 14/15 | 2.60 s | 3.07 s | **0.85** | 15/15 |
| `uhd` | 0.76 s | 0.85 s | **0.89** | 15/15 | 2.47 s | 2.77 s | **0.89** | 15/15 |
| `ten`, FFmpeg made 8-bit | 0.88 s | 1.27 s | **0.69** | 15/15 | 2.72 s | 3.95 s | **0.69** | 15/15 |
| `ten`, FFmpeg left 10-bit | 0.88 s | 1.00 s | **0.88** | 15/15 | 2.72 s | 3.03 s | **0.90** | 12/15 |

Peak memory (resident set), Cutlass against FFmpeg:

| Input | 4 threads | 1 core |
| --- | ---: | ---: |
| `mandel` | 79 MB / 99 MB | 28 MB / 68 MB |
| `m120` | 73 MB / 96 MB | 28 MB / 68 MB |
| `mid40` | 67 MB / 99 MB | 28 MB / 69 MB |
| `small` | 14 MB / 46 MB | 9 MB / 42 MB |
| `nowpp` | 72 MB / 96 MB | 28 MB / 68 MB |
| `pframes` | 64 MB / 83 MB | 25 MB / 63 MB |
| `uhd` | 234 MB / 252 MB | 87 MB / 156 MB |
| `ten` | 118 MB / 127 MB (123 MB left 10-bit) | 53 MB / 89 MB (86 MB left 10-bit) |

Cutlass is faster than FFmpeg on every input, with four threads and on one core, takes less processor time on every one, and less memory on every one. Of the 240 pairs of runs it finished first in 237, and in 27 of the 30 against FFmpeg left in 10-bit; in every cell its fastest run was faster than FFmpeg's fastest (the ratio of the fastest runs is between 0.59 and 0.97). Cutlass's engine decoder returns 8-bit pictures, so the first `ten` row has the FFmpeg program convert each 10-bit frame to `yuv420p` (its `to:yuv420p` mode) for the same output; the second leaves FFmpeg's frames as it decodes them, 10-bit, which is less work than Cutlass does, and Cutlass is faster there too.

The closest is `pframes` with four threads, a chain of P pictures each waiting on the rows of the one before. Until commit `4effa15` it was level with FFmpeg (between 0.95 and 1.02 in measurements of 7 to 41 runs): its blocks took the explicit weighted prediction the stream turns on even where the weights are those of 1 with no offset, and now take the default weighting's cheaper kernels for those. With four threads both decoders keep about 3.2 of the 4 cores busy on this stream, waiting for the rows of the reference; with no waits at all (which makes wrong pictures, as a measurement only) Cutlass would be 15% faster, and its processor time with four threads is still about 10% above that on one core, from reading the motion and samples of a reference another core has just written: four separate one-core decodes at once take no more time each than one alone, and a stream of intra pictures only takes 3% more.

Peak memory with four threads was above FFmpeg's until commit `a52c5d3` (by up to 60 MB on the 10-bit stream, and from one run to the next by 20 to 30 MB): a finished picture went back to the spare pictures still holding its slices, and with them the reference frames they named, so up to 18 spare pictures kept frames alive that the buffer of pictures had dropped. They now let them go as they are handed back, and peak memory fell by 20 to 45 MB on every stream.

**Decode alone.** `HEVC_BENCH=<mandel> cargo bench -p cutlass-hevc`, which pushes the access units straight into `HevcDecoder` with four threads and times FFmpeg's `ffmpeg -f null` on the same file, measured **1.69 s against FFmpeg's 2.20 s** (1.30× faster) on the same machine as the tables, at the same commit, with the samples checked equal to FFmpeg's first.

Where the time goes, on one core against FFmpeg's own profile of `nowpp`: entropy decoding (CABAC and residual coding) takes about 10% less time than FFmpeg's, interpolation 25% less, and deblocking with its boundary strengths 15% less; motion vector derivation and the setup of each prediction block take about 40% more, and the copies of final rows into frames twice as long.

What made it faster, in order of the gain measured on this machine: decoding into a ring of CTB-row buffers that stay in cache instead of a picture-sized buffer per picture, and the same for the per-4×4 maps of motion, modes and QP; a picture waiting for a reference lending its place to another (1.15× from 1.00× to 1.05× with four threads); interpolation kernels that read each row of the reference window as one fixed-size array (1.34G instructions to 0.85G on 40 frames, against FFmpeg's 0.91G); asking for every line of a block's reference window before filtering it (5% to 9% on one core, 5% with four threads); the 16-bit interpolation, SAO and weighting kernels for deeper samples, then interpolation that writes their samples itself, weighted, as the 8-bit kernels do (13%), deep pictures no picture references shifted straight into the output instead of kept in a 16-bit frame (4% to 5%, 25 MB less), deblocking of 9 to 11 bits on the 8-bit kernel's filter, blocks at the top and bottom read in place instead of copied (5%), and deep prediction called outside the AVX2 wrapper that had inlined all of it into one function (7%): the 10-bit stream from 2.07× the time of FFmpeg left in 10-bit frames to 0.88×; one more thread decoding than pictures to a thread, so that no core waits while a thread wakes (0% to 6% with four threads); the default weighting where explicit weights are those of 1 (6% on P pictures with weighted prediction on, which made `pframes` faster than FFmpeg with four threads); deep reference pictures shifted to 8 bits when taken for output instead of beside their frame (18 MB less); spare pictures that let their references go (20 to 45 MB less on every stream); significance and greater-than-one flags handled without a branch on the decoded value; deblocking each CTB row as soon as it is complete and making all but its last eight lines final right then, on the thread that built it (6% with four threads); output pictures that are their frame's own samples, one buffer shared with the caller, instead of a copy (30 to 40 MB less memory with four threads on 1080p, 137 MB on 2160p, and 1% to 3% faster); clearing each band of a frame or an output picture before SAO writes it, so its lines come into cache without being read (3%); one picture per thread instead of wavefront rows split between threads (2% wall time, 5% processor time); intra prediction computed a row at a time over whole slices of the references, and the availability of neighbours found once per CTB (1778M instructions to 1663M on 24 frames, with the merge and AMVP lists that stop at the candidate chosen); the inverse transform adding its residual to the samples itself; the CABAC engine kept in registers through residual coding; wavefront row progress reported without a lock or a system call. Tried and dropped: an AVX-512 interpolation kernel (slower on this CPU), waiting further behind a reference (10% to 30% slower), fewer pictures in flight, spinning before sleeping on a reference's rows, giving a free slot to the earliest picture first (2% to 4% slower), asking for the chroma windows before luma is filtered (2% to 4% slower), SAO edge offsets over a whole CTB through closures (no gain: the rows' lookup cost what the calls saved; walking the CTB row's buffer by stride instead saves 18%), reading references by a count of their rows kept per slice instead of loading the count for each block, pinning each thread to a core (slower), fewer spare CTB rows, keeping each thread's row buffers on that thread, and the rows of pictures split between threads (all no gain or slower).

**Engine pictures at the stream's depth.** Measured on 24 November 2026 on branch `codec/hevc-10bit-output` at commit `3ebd6936`, against `development` at `bcab8681`, on a cloud VM with 4 cores of an Intel Xeon at 2.1 GHz with AVX2 (other VMs share the host; the same run varied by up to 10% within the session), FFmpeg 6.1.1 from Ubuntu 24.04, rustc 1.97.0. The change makes the engine's pictures of a 10-bit stream 10-bit (twice the bytes) where they were cut to 8 bits, so the question was whether decode got slower. Inputs: 240 pictures of 1920×1080 `testsrc2`, libx265 preset medium, 8-bit at CRF 26 (`eight`) and 10-bit at CRF 28 (`ten`), in MP4. Each side is a whole process run in turn, `media-bench decode picture <clip>` built from each commit and `ffmpeg -threads N -i <clip> -f null -` (the `ffmpeg` program, not `scripts/ffmpeg_decode.c`, so these are not the numbers of the tables above); 15 rounds with four threads, 11 pinned to one core with one thread; median wall time. The new build's 240 pictures are byte for byte what `ffmpeg -f rawvideo` writes for both clips (746,496,000 and 1,492,992,000 bytes); the old build's `ten` output was the 8-bit one, so old and new do different work there by design.

| Input | 4 threads | | | 1 core | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | before | after | FFmpeg | before | after | FFmpeg |
| `eight` | 0.80 s | 0.79 s | 1.01 s | 2.09 s | 2.16 s | 2.27 s |
| `ten` | 1.23 s | 1.23 s | 1.09 s | 3.59 s | 3.56 s | 2.91 s |

Before and after are within the run-to-run spread on both inputs (the new build finished first in 12 of 15 and 8 of 15 pairs with four threads, 7 of 11 and 7 of 11 on one core): no regression measured. This VM gives `perf stat` no hardware counters (`instructions` is `<not supported>`), so retired instructions could not be read; in their place, instructions executed under `valgrind --tool=cachegrind --cache-sim=no` for 24 pictures of the 10-bit input on one thread: 2.260 G before, 2.217 G after, 1.499 G for FFmpeg. `HEVC_BENCH=<clip> cargo bench -p cutlass-hevc` (best of five, four threads, decode alone against one `ffmpeg -f null` process) gave 0.68 s against FFmpeg's 0.86 s on `eight` and 1.11 s against 0.98 s on `ten`.

On this machine the 10-bit stream decodes slower than FFmpeg does, before this change and after it (1.13× the time with four threads, 1.22× on one core, 1.48× the instructions), where the table above, from another machine and another FFmpeg harness, has it faster. That was not looked into here.

Two things were tried for the deeper output. The first build copied each band into an output band it had cleared first, as the 8-bit path does to bring the lines into cache: `ten` took 3.57 s on one core against 3.38 s before the change in that round (1.21 s against 1.13 s with four threads). Not clearing a band that is about to be overwritten whole brought that to 3.47 s and 1.17 s, and copying 10-bit and 12-bit samples with no shift, in the code compiled for AVX2, to the times in the table. Not tried: lending a reference picture's frame to the caller instead of copying it, as 8-bit pictures are lent; the frame is 16-bit words and the output bytes, which needs a view of one as the other in `cutlass-simd`.

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
- The SEI messages named under "What it supports" are read; every other one is skipped, and the Dolby Vision RPU is not returned.
- Fields are returned one picture each, with `pic_struct` from `push_pictures`. No code pairs fields into frames.
- The vector kernels need x86-64 with AVX2; other CPUs use the scalar code, which is several times slower. For samples above 8 bits, interpolation, default weighting, SAO, and the inverse transforms have kernels; deblocking and explicit weighting of those samples are scalar code compiled for AVX2.
- A picture of a cropped 8-bit stream is written to its frame and then copied to the output; a picture above 8 bits that a later picture references, or that is cropped, is copied from its frame when it is taken for output, on the caller's thread: the frame holds 16-bit words and the output is bytes, and without `unsafe` one cannot be lent as the other. Other 8-bit pictures are their frame's own samples, shared with the caller; other deeper ones are copied to the output a band at a time as they are filtered.
- With four threads the decoder holds up to six pictures at once, against FFmpeg's four; its peak memory is lower than FFmpeg's all the same on every test stream (see the speed section).

## Specifications

- Recommendation ITU-T H.265, also published as ISO/IEC 23008-2: clauses 6 to 9, Annex B (byte stream), Annex C.5.2 (output order), and Annex E (VUI).
- ISO/IEC 14496-15 clause 8.3.3, for the `hvcC` record and length-prefixed samples.
