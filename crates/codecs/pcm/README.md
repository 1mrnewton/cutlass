# cutlass-pcm

`cutlass-pcm` decodes and encodes raw PCM sound in safe Rust (`#![forbid(unsafe_code)]`). A packet is a block of samples a container has already framed; the crate does not read WAV, AIFF or CAF headers.

The desktop app registers the decoder and the encoder after the platform codecs. On macOS the platform crate also registers PCM and is asked first, so this crate opens when the platform one declines, for example for a 64-bit float layout. On Windows and Linux it is the only PCM codec the app registers.

## What it supports

Every coding below is accepted little-endian or big-endian, interleaved or planar. Planar bytes are one plane per channel, in channel order. Decoded sound is always interleaved.

| Coding | Decodes to | Encodes from |
| --- | --- | --- |
| Signed 16-bit | `SampleFormat::I16`, the same bits | `I16`, the same bits |
| Signed 24-bit, three bytes | `SampleFormat::F32`, divided by 8388607 | `F32`, times 8388607, rounded to nearest with ties to even, held to -8388608..=8388607 |
| Signed 32-bit | `SampleFormat::F32`, divided by 2147483647 | `F32`, times 2147483647, rounded the same way, held to the `i32` range |
| 32-bit float | `SampleFormat::F32`, the same bits | `F32`, the same bits |
| 64-bit float | `SampleFormat::F32`, narrowed with `as f32` | `F32`, widened with `f64::from` |
| Signed 64-bit | `SampleFormat::F32`, rounded once to `f32` and scaled by 2^-63 | refused |
| Signed 8-bit (AIFF, CAF) | `SampleFormat::I16`, shifted up 8 bits | refused |
| Unsigned 8-bit, 128 is silence (WAV, AVI, Matroska, QuickTime `raw `) | `SampleFormat::I16`, less 128, shifted up 8 bits | refused |
| G.711 A-law | `SampleFormat::I16`, the G.711 expansion, 8 to 32256 | refused |
| G.711 mu-law | `SampleFormat::I16`, the G.711 expansion, 0 to 32124 | refused |

Integer full scale is the largest positive value of the width, so the most negative 24-bit code decodes slightly below -1. For 64-bit integers that value is 2^63, since `i64::MAX` rounds to 2^63 in a double, so `i64::MIN` decodes to exactly -1. Byte order is ignored for the one-byte codings. Finite floats outside ±1 are kept.

The layout is three bytes in `CodecParams.extra`, which the container writes:

| Byte | Values |
| --- | --- |
| 0, coding | 1 s16, 2 s24, 3 s32, 4 f32, 5 f64, 6 s8, 7 A-law, 8 mu-law, 9 s64, 11 u8 |
| 1, byte order | 0 little, 1 big |
| 2, arrangement | 0 interleaved, 1 planar |

An empty `extra` follows `CodecParams.sample_format`: `I16` is interleaved little-endian s16 and `F32` is interleaved little-endian f32.

What is refused:

| Input | Error |
| --- | --- |
| Coding 10 (24-bit in a 32-bit word; no container writes it) | `Unsupported { what: "s24 in 32" }` |
| Any other coding byte | `Unsupported { what: "pcm coding" }` |
| Encoding s8, u8, A-law, mu-law or s64 | `Unsupported { what: "s8 encode" }`, `"u8 encode"`, `"a-law encode"`, `"mu-law encode"`, `"s64 encode"` |
| `extra` that is not empty and not three bytes; a bad byte-order or arrangement byte | `Malformed` |
| Empty `extra` and no `sample_format` | `Unsupported { what: "pcm layout" }` |
| A `sample_format` that is not what the coding decodes to | `InvalidApi { what: "pcm sample format" }` |
| A float sample that is not finite, on decode or encode | `NonFinite` |
| A packet that is not a whole number of frames | `Truncated { what: "pcm frame" }` |
| An empty packet | `InvalidApi { what: "empty packet" }` |
| A `SideData::Config` that differs from the open layout | `Unsupported { what: "configuration change" }` |
| Encoder input with another rate, channel layout or sample format | `InvalidApi` |
| A picture pushed to the encoder | `Unsupported` |

A partial trailing frame is an error, not silence and not a dropped tail. A failed packet leaves the decoder usable for the next one.

## Using it

- `PcmDecoder::open(&CodecParams)` and `PcmEncoder::open(&CodecParams)` take `CodecId::Pcm`, a sample rate, a `ChannelLayout` and the layout bytes. `layout()` returns the `Layout` that was opened.
- `Layout::new(Coding, Endian, Arrangement)` describes a layout and `Layout::to_extra` gives its three bytes.
- `register_decoders` and `register_encoders` add the crate to a `DecodeFactory` and an `EncodeFactory`.

`Decoder::push` returns one `Output::Audio` with the packet's time. `Encoder::push` returns one packet for a buffer with samples and none for an empty buffer; packet times count samples from zero. PCM has no delay, so `drain` returns nothing on both sides. `Encoder::reset` sets the sample count back to zero.

```rust
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Packet, SampleFormat};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_pcm::{Arrangement, Coding, Endian, Layout, PcmDecoder, PcmEncoder};

let spec = Layout::new(Coding::S16, Endian::Little, Arrangement::Interleaved);
let mut params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::stereo());
params.extra = spec.to_extra().to_vec();
params.sample_format = Some(SampleFormat::I16);

let mut decoder = PcmDecoder::open(&params)?;
let outputs = decoder.push(&Packet::audio(bytes, 0, frames, 48_000))?;

let mut encoder = PcmEncoder::open(&params)?;
let packets = encoder.push(&AudioInput { pcm })?;
```

`CodecParams::audio` sets `sample_format` to `F32`, so set `I16` for 16-bit PCM as above. `examples/roundtrip.rs` is a complete program: `cargo run --release -p cutlass-pcm --example roundtrip`.

## How it is tested

All tests are unit tests inside `src/`; run them with `cargo test -p cutlass-pcm`.

- Known byte patterns for each coding, both byte orders and planar input, including full scale, the most negative code, negative zero and loud finite floats; the G.711 zeros and full-scale codes; `i64::MAX` and `i64::MIN`.
- Round trips: s16, f32 and f64 encode and decode back to the same samples, and random s16 buffers keep every sample. All 2^24 codes of s24 decode and encode back to themselves (`every_s24_code_survives_a_decode_and_an_encode`); s32 comes back within 128 codes, and exactly for codes a float holds.
- The integer encoders: the bytes for 0, ±1, a tie, and samples past full scale are written out in the test for both byte orders and planar; the rounding equals `f64::round_ties_even` on 60,000 random samples; a non-finite sample, an `I16` buffer and a wrong layout are refused and leave the clock alone.
- Errors: partial frames, short buffers, overflowing lengths, non-finite floats, refused codings, and that a failed push changes nothing for the next push.
- Against FFmpeg (`ffmpeg_reads_an_independent_bitstream`): FFmpeg reads encoded s16 and f32 packets back unchanged, and FFmpeg's widening of 24-bit samples matches the decoder's scale. Skipped with a line on stderr when `ffmpeg` is not installed; set `PCM_REQUIRE_ORACLES` to make that a failure.
- Against FFmpeg, bit-exact (`ffmpeg_decodes_every_byte_code_the_same`, `ffmpeg_decodes_s64_to_the_same_floats`): all 256 codes of A-law, mu-law, s8 and u8 decode to the same s16 values as FFmpeg's `alaw`, `mulaw`, `s8` and `u8` decoders widened by libswresample, and 4,006 random 64-bit integers (a 64-bit WAV, which FFmpeg reads as `pcm_s64le`) give the same `f32` bits as FFmpeg's float output.
- Against FFmpeg (`ffmpeg_reads_the_integer_codes_this_encoder_writes`): FFmpeg reads the s24 and s32 bytes this encoder writes as the codes that were meant. FFmpeg's own encode of the same floats is within two codes and not always equal, because FFmpeg scales a float by 2^31 and, for s24, drops the low byte, where this crate takes positive full scale as the largest code and rounds. On the 4,102 samples of the test the largest difference is one code, for both widths (FFmpeg 6.1.1).
- A separate measurement found that a 16-bit WAV file decodes to the same bytes as FFmpeg's output.
- `local_files_match_an_independent_pcm_dump` decodes FFmpeg's 16-bit dump of two named files from a local media folder that is not in the repository. It is skipped when the folder is missing.

Not run by default: `thirty_minute_decoder_campaign` and `bounded_encoder_campaign` (seeded random input; `PCM_FUZZ_SECS` and `PCM_ENCODER_FUZZ_SECS` set the run lengths, `PCM_FUZZ_SEED` the decoder seed in hex) and `release_decode_and_encode_throughput` are `#[ignore]`. The decoder campaign, with the s8, u8, A-law, mu-law and s64 layouts added, ran 1800 s from seed `0x711d3c0` on 12 November 2026 (release build at `f2d59632`): 15,627,983,193 inputs, 10,755,512,978 refused with an error, no panic and no failed check, at most 2.9 MB resident. The encoder campaign and the throughput test have no recorded result.

## Speed against FFmpeg

Measured on 24 November 2026 at commit `4439627d`, on a 4-vCPU KVM guest (Intel Xeon at 2.10 GHz with AVX-512, 15 GB) running Ubuntu 24.04 on Linux 6.18, against FFmpeg 6.1.1 from Ubuntu's packages. Built with `cargo build --release` and rustc 1.97.0.

**Read and decode a file.** 5 minutes of 48 kHz stereo, read by the container crate and decoded here. For the 16-bit files and the float file the decoded samples have the same SHA-256 on the old build, the new build and FFmpeg (`-f s16le`, `-f f32le`). For 24-bit the old and new builds agree with each other; FFmpeg divides by 2^23 where this crate divides by 2^23 - 1, so its floats differ in the last bits by design.

| File | | Cutlass before | Cutlass | libavformat and libavcodec | `ffmpeg` command |
| --- | --- | ---: | ---: | ---: | ---: |
| WAV s16 (57.6 MB) | Wall time | 9,768 ms | 36.0 ms | 56.7 ms | 978 ms |
| | Instructions | 39,716 M | 176.5 M | 322.3 M | 814.8 M |
| WAV s24 (86.4 MB) | Wall time | 9,868 ms | 73.6 ms | 84.1 ms | 1,473 ms |
| | Instructions | 40,494 M | 465.9 M | 715.0 M | 1,615 M |
| WAV f32 (115.2 MB) | Wall time | 9,641 ms | 72.9 ms | 103.7 ms | 1,972 ms |
| | Instructions | 40,134 M | 413.4 M | 452.1 M | 1,593 M |
| AIFF s16 (57.6 MB) | Wall time | 8,867 ms | 38.3 ms | 58.5 ms | 914 ms |
| | Instructions | 35,929 M | 145.8 M | 402.4 M | 889.6 M |
| CAF s16 (57.6 MB) | Wall time | 57.3 ms | 38.1 ms | 77.6 ms | 996 ms |
| | Instructions | 535.3 M | 145.9 M | 404.0 M | 892.3 M |

Cutlass decodes the 16-bit WAV in **0.63× the time** of libavformat and libavcodec and 0.55× the instructions, where it was 172× slower on this machine (245× on the Apple M5 Pro of the 10 November measurement). Most of that was the WAV and AIFF demuxers handing over one frame a packet and the engine reading the file twice; the CAF row, whose demuxer already grouped frames, shows this crate's own share: 535 M instructions down to 146 M.

**Conversion alone.** One 4,096-frame stereo packet decoded, and the result encoded, 2,000 times in a loop; median of five runs, nanoseconds a sample. The output bytes of the old and new code are the same in every row.

| Layout | Decode before | Decode | | Encode before | Encode | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s16 little-endian | 0.648 | 0.028 | 23× | 2.759 | 0.048 | 57× |
| s16 big-endian | 0.644 | 0.059 | 11× | 2.875 | 0.065 | 44× |
| s16 planar | 1.429 | 0.510 | 2.8× | 1.512 | 0.526 | 2.9× |
| s24 little-endian | 1.770 | 0.748 | 2.4× | refused | 1.826 | |
| s32 little-endian | 1.294 | 0.637 | 2.0× | refused | 1.653 | |
| f32 little-endian | 0.703 | 0.248 | 2.8× | 3.098 | 0.277 | 11× |
| f32 big-endian | 0.756 | 0.420 | 1.8× | 3.047 | 0.461 | 6.6× |
| f64 little-endian | 1.296 | 0.385 | 3.4× | 3.444 | 0.471 | 7.3× |
| u8 | 0.668 | 0.049 | 14× | | | |

For the s16 little-endian round trip the loop went from 66.8 to 4.7 user-space instructions a sample, and for f32 from 82.3 to 13.4.

What changed: the byte order and arrangement are matched once for a packet, and the samples are converted by one loop over fixed-size chunks, which the compiler can turn into a vector loop (s16 now converts about 35 samples a nanosecond). Before, every sample went through a closure that returned a `Result`. The check for a non-finite float is one pass over the converted samples with no early exit. Planar sound is one pass a channel; it gained least, and its loop, which steps through the output a channel at a time, was not taken further. The 24-bit and 32-bit conversions still divide in double precision so that the result stays bit-identical to the old code; multiplying by a reciprocal was not tried, because it would change the last bit of some samples.

How it was measured: every column is a whole process, start-up included, reading the same file from the page cache and throwing the result away. Each was run five times in turn and the tables show the median wall and CPU time. Instructions are user-space instructions from one run under `valgrind --tool=cachegrind` (`I refs`); the kernel's share of a read is not in them. `perf stat` was not used because this virtual machine exposes no hardware counters (`instructions: <not supported>`). The Cutlass columns are [`tools/media-bench`](../../../tools/media-bench) (``media-bench decode sound <file>``), which registers only the Rust codecs; "before" is the same tool built at `bcab8681`, the commit this work started from. The libavformat column is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c) built against the same FFmpeg, one thread, which calls libavformat and libavcodec with no filter graph and no output stage. The `ffmpeg` column is the command line (``ffmpeg -threads 1 -i <file> -f null -`, which also re-encodes to 16-bit for the null output`); it starts more threads than it is asked to and its CPU time is above its wall time for that reason. `scripts/bench_media.py` was not used: it reads `/usr/bin/time -l`, which is macOS's. The conversion table is from a separate small program, not in the repository, that links this crate at the two commits and calls `PcmDecoder::push` and `PcmEncoder::push` in a loop.

## Limits and gaps

| Limit | Value | Error |
| --- | --- | --- |
| Packet | 16 MiB | `Limit { what: "packet bytes" }` |
| Decoded or encoded buffer | 32 MiB | `Limit { what: "allocation" }` |
| Sample rate | 1 to 384000 Hz | `InvalidApi` at 0, `Limit` above |
| Channels | 1 to 8, the bound of `ChannelLayout` | `InvalidApi` when the layout is missing |

- Coding 10, 24-bit samples in 32-bit words, is not decoded: no container in the workspace writes it. CAF's unpacked integer samples are high-aligned and AudioToolbox reads every bit of the word, so `cutlass-caf` maps them to the coding of the word width instead.
- s16, f32 and f64 are encoded exactly. s24 and s32 are encoded from the float mix sample by rounding, with no dither; a sample past full scale becomes the end code. The 8-bit, G.711 and s64 codings are not encoded. Export does not yet offer a 24-bit or 32-bit choice: the encoder is here, and nothing in `crates/export` or the app selects it.
- 32-bit integers decode to `f32`, which has a 24-bit mantissa, so their low bits are rounded. The most negative 32-bit code decodes to exactly -1.
- Resampling, dither and channel remixing are outside the crate.
- `Decoder::push` returns a `Vec<Output>` and `Pcm` owns its `ChannelLayout`, so each packet still costs one small vector and one clone of the layout besides the sample buffer. Both come from the `cutlass-decode` interface, which this crate does not own; at 64 KiB a packet that is two small allocations for each 64 KiB of samples.

## Specifications

Raw PCM has no bitstream specification of its own. Integers are two's complement. Floats are IEEE 754 binary32 and binary64.
