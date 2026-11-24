# cutlass-h264

An H.264 (AVC) decoder and encoder in safe Rust. The crate has `#![forbid(unsafe_code)]` and depends only on `cutlass-decode`, `cutlass-encode` and `cutlass-h264-kernels`, which holds its sample loops and the `unsafe` their NEON versions need. `H264Decoder::open` decodes on the caller's thread; `with_threads` gives it threads, and the decoders `register_decoders` adds, which are the ones the app opens, take up to eight of the machine's cores. The desktop app registers both: on macOS and Windows the platform codec is asked first when hardware decode or encode is on in Settings, and this crate is used when that opener declines, when hardware is off, and always on Linux.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Profiles | `profile_idc` 66, 77, 88, 100, 110, 122, 244 (Baseline, Main, Extended, High, High 10, High 4:2:2, High 4:4:4 Predictive). The constraint flags are not checked. | High (`profile_idc` 100), `level_idc` 31 |
| Chroma and depth | 4:2:0, 4:2:2, 4:4:4 at 8 to 14 bits, luma and chroma depth equal. Separate color planes. | 8-bit 4:2:0, even width and height |
| Entropy coding | CAVLC and CABAC | CABAC |
| Slices | I, P, B, SP, SI. Several slices per picture, slice groups with map types 0 to 6, data partitions. SP and SI only at 8-bit 4:2:0. | One slice per picture. I, P, and one B between two references. |
| Pictures | Frames and macroblock-adaptive frames (MBAFF). Field pictures (PAFF): intra pairs are exact, but a field picture's reference lists are built from frames, not fields, so prediction between fields is not to the recommendation. | Frames |
| Prediction | Skip, direct, partitions down to 4×4, explicit and implicit weighted prediction | Intra 4×4 DC. P: skip, 16×16, 16×8, 8×16, four 8×8. B: 16×16 on list 0, list 1, or both. |
| Transform | 4×4 and 8×8, scaling lists, transform bypass, deblocking, frame crop | 4×4, and 8×8 for inter luma when its residual is closer |
| Ordering | Picture order count types 0, 1, 2. Reference marking, list modification, `frame_num` gaps. | Type 0, sliding window, two reference frames |
| SEI and VUI | Picture timing changes the output time. Film grain is applied to the shown picture, not to the reference. Other SEI is skipped. Sample aspect, timing, and color tags are returned with the picture; the engine's picture (`PictureInfo`) names the primaries, transfer, matrix (`matrix` and `cicp_matrix`) and the sample range, so a full-range stream is shown as full range. | No SEI. A VUI with bitstream restrictions. |
| Framing | Annex B, length prefixes of 1, 2, or 4 bytes, `avcC` extradata | Annex B when `nal_length_size` is 0, else 4-byte lengths. `extradata()` is the `avcC` record. |

The decoder returns every picture as 8-bit 4:2:0 in `VideoOutput::picture`. A deeper or wider stream also fills `yuv420p10`, `yuv420_high`, `yuv422`, `yuv422_high`, `yuv444`, or `yuv444_high` with the full samples. The opener the engine uses (`Decoder`) returns only the packed 8-bit 4:2:0 picture.

Scalable (Annex G), multiview (Annex H), and 3D (Annex I) streams are partly decoded. A subset sequence parameter set is read for `profile_idc` 83, 86, 118, 128, 135, 138, and 139. What reconstructs is what the tests in `src/annex_g.rs` build; the module docs in `src/lib.rs` list those cases one by one.

Refused:

- Monochrome (`chroma_format_idc` 0): `CodecError::Unsupported { what: "chroma format" }`.
- A depth above 14 bits, or luma and chroma depths that differ: `Unsupported { what: "bit depth" }`.
- Any other `profile_idc`: `Unsupported { what: "profile" }`.
- Scalable, multiview, and 3D tools outside the tested cases: `Unsupported` with `"inter-layer prediction"`, `"3d"`, `"scan idx"`, `"reference base picture"`, `"svc vui"`, `"mvc vui"`, or `"mvcd vui"`.
- A length prefix of 3 bytes: `Unsupported { what: "nal length size" }`.
- A picture with missing macroblocks: `CodecError::Malformed`. There is no error concealment.
- Encoder: a `nal_length_size` other than 0 or 4, an odd width or height, or a size past the limits below.

## Using it

`H264Decoder::open` and `H264Encoder::open` take a `CodecParams`. `register_decoders` and `register_encoders` add both to a `DecodeFactory` or `EncodeFactory`. The decoder implements `VideoDecoder`, the encoder `VideoEncoder`.

One `push` to the decoder is one access unit. It returns the pictures that are ready in display order, which may be none. `drain` returns the pictures still held for reordering. A `push` that fails leaves the decoder as it was.

The decoders `register_decoders` adds decode on other threads, on as many cores as the machine has up to eight (`with_threads_from_the_second_picture`). The pictures and their order are the same as on one thread, byte for byte, on every thread count. What differs: the first picture after opening or a `reset` is decoded on the calling thread, so a seek that wants one picture gets it from the same call as before and starts no thread; the pictures after it may come out some calls later than on one thread (up to the pictures in flight, about one and a half per core), so a caller that wants a certain picture keeps pushing or drains, as the engine's reader does; and an error in one of those pictures is returned by the call that would have returned the picture. No thread is started until a second picture needs one, at most four per core are ever started (eight were seen on four cores), and a thread with nothing to decode for half a second ends, so a clip that is open but not playing holds none. Interlaced, 4:2:2, 4:4:4, multiview and scalable pictures stay on the calling thread. The engine's picture is packed into a buffer the decoder shares with it (`PictureOutput::cpu_shared`) and fills again once the engine has let it go. `nal_length_size` 0 means Annex B; an `avcC` record in `CodecParams::extra` sets the length size itself.

The encoder holds one picture back so it can code a B picture, so a `push` may return no packet and `drain` returns the last one. A picture is intra only when it is the first one or `force_keyframe` is set. With `bitrate` set, the frame QP follows the bit budget; `constant_rate` resets that budget each second. With `bitrate` 0 the QP stays at 26.

```rust
use cutlass_decode::{CodecId, CodecParams, VideoDecoder, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_h264::{H264Decoder, H264Encoder};

fn encode_one() -> Result<(), Box<dyn std::error::Error>> {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let mut encoder = H264Encoder::open(&params)?;
    let packets = encoder.push(&VideoInput {
        picture: Yuv420p::solid(16, 16, 40, 80, 160)?,
        pts: 0,
        duration: 1,
        timescale: 30,
        force_keyframe: true,
    })?;
    let mut decoder = H264Decoder::open(&params)?;
    for packet in &packets {
        decoder.push(packet)?;
    }
    decoder.drain()?;
    Ok(())
}
```

`examples/roundtrip.rs` asks FFmpeg for a 64×64 High-profile file, decodes it, and prints the frame count and size:

```text
cargo run -p cutlass-h264 --example roundtrip
```

## How it is tested

All tests are unit tests inside `src/`. FFmpeg is a separate program the tests call to compare against; the crate does not link it.

```text
cargo test -p cutlass-h264
cargo test -p cutlass-h264 --release -- --ignored
```

- **Decoder against FFmpeg.** Streams written by libx264 are decoded here and by `ffmpeg`, and every sample inside the cropped rectangle must be equal. This covers Baseline, Main, and High with CAVLC and CABAC, B pictures and temporal direct, weighted prediction, several slices, crop, scaling lists (`src/mb.rs`, `src/dec.rs`), High 10 (`src/high10.rs`), 4:2:2 and 4:4:4 at 8 and 10 bits, clean and noisy, on one thread and on four (`src/chroma.rs`), an intra field pair and macroblock-adaptive frames with I, P, and B under CABAC and CAVLC at 4:2:0 and at 8-bit and 10-bit 4:2:2 (`src/fields.rs`, `src/chroma.rs`; 4:4:4 interlaced is not checked), lossless, including noisy pictures libx264 writes partly as I_PCM inside CABAC slices (`src/lossless.rs`), and picture sizes up to the level limits (`src/annex_a.rs`).
- **Decoder against the specification.** Tools FFmpeg does not decode are checked against sample values worked out from the specification on small hand-built streams: slice groups (`src/fmo.rs`), data partitions, SP and SI (`src/part.rs`, `src/sp.rs`), separate color planes (`src/planes.rs`), depths of 11 and 13 bits (`src/depth.rs`, which compares 9, 12, and 14 bits with FFmpeg), picture timing and film grain (`src/sei_pic.rs`), and the scalable, multiview, and 3D cases (`src/annex_g.rs`).
- **Encoder.** The encoder's output is decoded by this decoder and by FFmpeg, and both must give the same samples (the `*_matches_ffmpeg` tests in `src/enc.rs`). `testsrc2_meets_the_luma_and_rate_gates` requires a luma PSNR of at least 30 dB and a size within 0.75 to 1.25 times `bitrate * duration`. `natural_clip_meets_the_luma_gate` requires 28 dB on a camera clip. `bottom_rows_survive_a_partial_macroblock_row` encodes scrolling pictures at 96×72, 96×80 and 88×72 and requires the bottom 24 rows to reach 35 dB luma and 28 dB chroma in every picture, with FFmpeg decoding the same samples: the encoder codes whole macroblocks, with the picture's last row and column repeated into the padding, and predicts from the uncropped reconstruction, as every decoder does.
- **The app's opener.** The decoders `register_decoders` adds must return the same bytes as one thread on 1 to 8 and 16 cores, give the picture asked for when scrubbed back and forth (`reset`, then pushes from a key picture), return the first picture from the same call as one thread does without starting a thread, end their threads when left alone, keep interlaced, 4:2:2 and 4:4:4 streams on the calling thread, and name the range and matrix of the VUI (`src/dec.rs`).
- **Bad input.** Truncated units, bad lengths, bad parameter sets, and 64 random packets must return an error, and a known picture must still decode afterward (`hostile_framing_is_rejected_and_the_decoder_stays_usable`, `hostile_parameter_sets_are_rejected`, `random_packets_fail_closed`).
- Five films from 576×1024 to 3840×2160 decode to the same bytes as FFmpeg's decoder, on 1 thread and on 18 (see the speed section). The films are not in the repository, so no test repeats this.

What does not run by default:

- A test that needs `ffmpeg` returns early and passes when `ffmpeg` is not found. The example and the bench print `skip: ffmpeg is not installed` and exit 0.
- `natural_clip_meets_the_luma_gate` and `uhd_3840x2160_high_matches_ffmpeg` need media files that are not in the repository and pass without checking anything when the files are missing.
- Three tests are `#[ignore]`: `thirty_minute_decoder_campaign` and `bounded_encoder_campaign` (random input for a fixed time) and `high_720p_decode_meets_the_release_gates` (a speed and memory gate).
- The ITU-T conformance bitstreams are not in the repository and have not been run.
- `cargo bench -p cutlass-h264` (`benches/against_ffmpeg.rs`) times 1280×720 decode and encode against the `ffmpeg` program and refuses to print a time unless the samples match first.

## Speed against FFmpeg

**Decode, h264.** Measured on 14 November 2026 at commit `3c8dc585`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1 as Homebrew builds it. Built with rustc 1.100.0 through `scripts/pgo.py`, as a release is, with the AV1 decoder's profile merged in. The input is the video stream of five films that are not in the repository, each as an Annex B file, all High profile, 8-bit 4:2:0.

One thread, processor cycles of the whole process (median of 7 runs taken in turn):

| Film | Pictures | Cutlass | FFmpeg (`h264`) | Cutlass ÷ FFmpeg |
| --- | ---: | ---: | ---: | ---: |
| 576×1024 | 453 | 0.812G | 0.901G | 0.90 |
| 720×1280 | 391 | 1.473G | 1.591G | 0.93 |
| 1920×1080 | 600 | 7.286G | 7.563G | 0.96 |
| 3840×2160, 24 a second | 351 | 22.127G | 22.486G | 0.98 |
| 3840×2160, 60 a second | 571 | 21.113G | 21.968G | 0.96 |

On one thread Cutlass takes 2 to 10% fewer cycles than this FFmpeg on all five. The 1920×1080 film took 1.65 s against 1.70 s, 364 against 353 pictures a second. Cutlass retires as many instructions as FFmpeg on the smallest film and more on the other four (29.0G against 24.2G on the 1920×1080 one), and gets through them in fewer cycles.

Which FFmpeg matters. A static FFmpeg built from its source of 29 October 2026 with only the H.264 decoder in it took 3 to 11% fewer cycles than Homebrew's `ffmpeg` command on three of these films. Against that build, on three runs of each the same day, Cutlass took 0.96 of its cycles on the 576×1024 film, 1.02 on the 1920×1080 one and 1.01 on the 3840×2160 one at 24 a second: level, not ahead.

18 threads (`with_threads(18)`, FFmpeg's `thread_count` 18), wall time and processor time of the whole process (median of 9 runs taken in turn):

| Film | Cutlass wall | FFmpeg wall | Cutlass CPU | FFmpeg CPU |
| --- | ---: | ---: | ---: | ---: |
| 576×1024 | 0.03 s | 0.04 s | 0.39 s | 0.31 s |
| 720×1280 | 0.07 s | 0.10 s | 0.73 s | 0.60 s |
| 1920×1080 | 0.16 s | 0.19 s | 2.41 s | 2.24 s |
| 3840×2160, 24 a second | 0.46 s | 0.43 s | 6.65 s | 6.00 s |
| 3840×2160, 60 a second | 0.55 s | 0.68 s | 7.73 s | 6.91 s |

Read the threaded table with care. `/usr/bin/time` reports hundredths of a second, and the FFmpeg process takes about 16 ms to start where the Cutlass one takes about 2 ms, so the two short films say little. Other work started on the machine during this run (load average 3 at its start, 7 at its end). Cutlass was behind on the 3840×2160 film at 24 a second, and used 8 to 26% more processor time than FFmpeg on every film.

How the decode was measured: each side is a whole process timed with `/usr/bin/time -l`, start-up and file reading included, the two run one after the other. The Cutlass side is a small program that reads the file, splits it into access units, pushes each to `H264Decoder` and drops the pictures; it is not in the repository. The FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which reads the same file with libavformat and decodes it with libavcodec, with no filter graph and no output stage. Before the timing, every sample of every picture of all five films was the same from both decoders, with Cutlass on 1 thread and on 18.

**Decode through the opener the app uses, on x86.** Measured on 24 November 2026 at commit `c29c63bc`, on a 4-core Intel Xeon virtual machine (2.1 GHz, AVX-512, 15 GB, Linux 6.18), against FFmpeg 6.1.1 as Ubuntu 24.04 builds it. Built with `cargo build --release` and rustc 1.97.0, without the profile a release is built with. The streams were written by libx264 (`-preset medium`) from `testsrc2` with noise: 300 pictures of 1920×1080 at 8 Mbit/s and 150 pictures of 3840×2160 at 35 Mbit/s, High profile, 8-bit 4:2:0. They are synthetic; no camera footage was at hand on this machine.

Wall time and processor time (user plus system) of the whole process, median of 5 runs taken in turn. Both sides are pinned to the same cores with `taskset`; Cutlass takes its thread count from them and FFmpeg is given the same `thread_count`:

| Stream | Threads | Cutlass wall | FFmpeg wall | Cutlass CPU | FFmpeg CPU | Cutlass ÷ FFmpeg, wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1920×1080 | 1 | 5.52 s | 1.97 s | 5.46 s | 1.95 s | 2.80 |
| 1920×1080 | 2 | 3.30 s | 1.20 s | 5.70 s | 2.13 s | 2.75 |
| 1920×1080 | 4 | 1.83 s | 0.81 s | 5.80 s | 2.34 s | 2.26 |
| 3840×2160 | 1 | 12.71 s | 6.41 s | 12.47 s | 6.28 s | 1.98 |
| 3840×2160 | 2 | 7.14 s | 3.61 s | 12.99 s | 6.27 s | 1.98 |
| 3840×2160 | 4 | 4.12 s | 2.21 s | 13.33 s | 6.37 s | 1.86 |

Four is every core this machine has, so there is no row above it. What the table says:

- Cutlass is behind FFmpeg here on every row, 1.9 to 2.8 times the wall time. On this processor the sample loops run their portable versions, where FFmpeg has x86 assembly; the tables above, where Cutlass is level, are from an Arm machine, whose loops have NEON versions in `cutlass-h264-kernels`. Threads do not change that ratio much.
- Threads are what the app gains. Before this change the app's opener decoded on one thread whatever the machine had: 5.52 s and 12.53 s for these two streams. On four cores it now takes 1.83 s and 4.12 s, 3.0 times faster on both, for 6 to 7% more processor time.
- Retired instructions were not measured: this virtual machine gives `perf stat` no hardware counters (`instructions` and `cycles` are `<not supported>`).

Before the timing, the packed pictures of both streams were written out with Cutlass on 1, 2, 3 and 4 cores and by `H264Decoder::open` alone, and each file had the same MD5 as `ffmpeg -f rawvideo -pix_fmt yuv420p` wrote with 1 thread and with 4. The Cutlass side is [`examples/app_decode.rs`](examples/app_decode.rs), which opens the decoder through `register_decoders` and a `DecodeFactory` as the engine does and drops each packed picture; the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c) in its `picture` mode.

Three smaller measurements from the same day and machine:

- Starting threads only when a picture needs one, instead of four per core when the decoder is opened, cost nothing that could be measured: `with_threads(4)` took 1.62 s before and 1.57 to 1.65 s after on the 1920×1080 stream, and 3.52 to 3.57 s before and 3.64 to 3.65 s after on the 3840×2160 one (2 to 3 runs each). The process had 8 decoding threads at its most where it had 16 before.
- Packing the engine's picture into a buffer that is filled again, instead of a new one for each picture, halved the system time on one thread (0.23 s to 0.09 s at 1920×1080, 0.39 s to 0.18 s at 3840×2160) and took 3% off the wall time at 1920×1080 (5.58 s to 5.40 s, median of 3); at 3840×2160 the wall time did not change beyond the spread of the runs. Peak memory rose by 4 to 9 MB, the buffers kept.
- The packed picture is still a copy of the three planes, made on the calling thread. On four cores the same decoder without that copy (`examples/profile_decode.rs`, `THREADS=4`) took 1.62 s and 3.6 s where the app's opener took 1.83 s and 4.12 s, so the copy and the first picture decoded on the calling thread cost about a tenth. Packing on the decoding threads would win most of that back and is not done.

On 10 November 2026, at commit `ed0743d2`, the decoder took 9.47 s on one thread for 600 pictures of 1920×1080 written by libx264, where FFmpeg took 2.48 s.

**Encode, h264.** Measured on 10 November 2026 at commit `ed0743d2`, same machine, FFmpeg 8.1.1, `cargo build --release` with rustc 1.99.1, and not measured again since. Input: 300 raw frames of 1920×1080 at 30 frames a second, asked for 8 Mbit/s.

|  | Cutlass | FFmpeg (`libx264 ultrafast`) | FFmpeg (`libx264 medium`) |
| --- | ---: | ---: | ---: |
| Wall time | 482 s | 1.39 s | 12.75 s |
| Speed | 0.6 frames/s | 216 frames/s | 24 frames/s |
| CPU time | 482 s | 1.44 s | 12.78 s |
| Peak memory | 70 MB | 95 MB | 414 MB |
| Output size | 10 MB | 10 MB | 10 MB |
| Quality kept (PSNR) | 32.02 dB | 29.89 dB | 38.82 dB |
| Threads allowed | 1 | 1 | 1 |

Cutlass is **346.9× slower** than `libx264 ultrafast` here. Cutlass is **37.8× slower** than `libx264 medium` here. Quality is the PSNR between the source pictures and what each encoder's output decodes to; higher is closer to the source. The encoders do not make the same choices, so read speed, size and quality together.

How the encode was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Decoder | Encoder |
| --- | --- | --- |
| Picture size | Any size a level in Table A-1 allows: at most 139,264 macroblocks and 16,880 samples on an edge. Larger is `CodecError::Limit { what: "picture size" }`. | At most 4096 on an edge and 8,388,608 pixels: `Limit { what: "coded size" }` |
| Reference frames | 16 | 2 |
| One NAL unit | 8 MiB | |
| Parameter set ids | 32 SPS, 256 PPS | One of each |
| Slice groups | 8 | 1 |
| Threads | As the app opens it, the machine's cores up to eight, from the second picture on; `H264Decoder::open` alone stays on the calling thread and `with_threads` takes any number. Interlaced, 4:2:2, 4:4:4, multiview and scalable pictures stay on the calling thread. At most four threads per core, started as pictures need them and ended after half a second idle. | One |

Decoder gaps:

- The refusals listed under "What it supports".
- An auxiliary picture (NAL type 19, with its sequence parameter set extension, type 13) is skipped, as FFmpeg skips it, so an alpha plane carried that way is not returned. Reserved and unspecified NAL types (0, 17, 18, 22 to 31) are skipped too.
- A damaged picture is not repaired. The bad-input tests only check that a later IDR picture still decodes.
- Film grain follows H.274, whose random source is not fixed, so grained pictures need not match another decoder.
- A damaged picture decoded on another thread fails every later picture that predicts from it, up to the next IDR picture, where on one thread the `push` of the damaged access unit fails and leaves the decoder as it was. Neither is a repair.

Encoder gaps:

- Intra macroblocks use only 4×4 DC prediction, and P and B pictures hold no intra macroblocks. A scene change inside a group of pictures is coded with inter prediction unless the caller sets `force_keyframe`.
- B pictures are 16×16 only, with no skip or direct macroblocks, and are never references.
- The motion search is a four-point diamond of at most 16 steps and ±16 samples, then half and quarter sample steps.
- No weighted prediction, 4×4 sub-macroblock partitions, CAVLC, interlace, several slices, or depths above 8 bits.
- The stream always says level 3.1, also for a picture larger than that level allows.
- The first QP is chosen as if the clip ran at 30 pictures a second.
- No presets and no multithreading.

## Specifications

- Recommendation ITU-T H.264 (08/2021), also published as ISO/IEC 14496-10: clauses 7, 8, and 9, and Annexes A, B, D, E, G, H, and I.
- Recommendation ITU-T H.274, for film grain synthesis.
- ISO/IEC 14496-15 (`avcC` and length-prefixed samples) was not obtained. The record parsing and the 1, 2 and 4-byte length prefixes were checked against FFmpeg's `libavformat/avc.c` instead.
