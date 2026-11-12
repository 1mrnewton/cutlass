# cutlass-pcm

`cutlass-pcm` decodes and encodes raw PCM sound in safe Rust (`#![forbid(unsafe_code)]`). A packet is a block of samples a container has already framed; the crate does not read WAV, AIFF or CAF headers.

The desktop app registers the decoder and the encoder after the platform codecs. On macOS the platform crate also registers PCM and is asked first, so this crate opens when the platform one declines, for example for a 64-bit float layout. On Windows and Linux it is the only PCM codec the app registers.

## What it supports

Every coding below is accepted little-endian or big-endian, interleaved or planar. Planar bytes are one plane per channel, in channel order. Decoded sound is always interleaved.

| Coding | Decodes to | Encodes from |
| --- | --- | --- |
| Signed 16-bit | `SampleFormat::I16`, the same bits | `I16`, the same bits |
| Signed 24-bit, three bytes | `SampleFormat::F32`, divided by 8388607 | refused |
| Signed 32-bit | `SampleFormat::F32`, divided by 2147483647 | refused |
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
| Encoding s24, s32, s8, u8, A-law, mu-law or s64 | `Unsupported { what: "s24 encode" }`, `"s32 encode"`, `"s8 encode"`, `"u8 encode"`, `"a-law encode"`, `"mu-law encode"`, `"s64 encode"` |
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
- Round trips: s16, f32 and f64 encode and decode back to the same samples, and random s16 buffers keep every sample.
- Errors: partial frames, short buffers, overflowing lengths, non-finite floats, refused codings, and that a failed push changes nothing for the next push.
- Against FFmpeg (`ffmpeg_reads_an_independent_bitstream`): FFmpeg reads encoded s16 and f32 packets back unchanged, and FFmpeg's widening of 24-bit samples matches the decoder's scale. Skipped with a line on stderr when `ffmpeg` is not installed; set `PCM_REQUIRE_ORACLES` to make that a failure.
- Against FFmpeg, bit-exact (`ffmpeg_decodes_every_byte_code_the_same`, `ffmpeg_decodes_s64_to_the_same_floats`): all 256 codes of A-law, mu-law, s8 and u8 decode to the same s16 values as FFmpeg's `alaw`, `mulaw`, `s8` and `u8` decoders widened by libswresample, and 4,006 random 64-bit integers (a 64-bit WAV, which FFmpeg reads as `pcm_s64le`) give the same `f32` bits as FFmpeg's float output.
- A separate measurement found that a 16-bit WAV file decodes to the same bytes as FFmpeg's output.
- `local_files_match_an_independent_pcm_dump` decodes FFmpeg's 16-bit dump of two named files from a local media folder that is not in the repository. It is skipped when the folder is missing.

Not run by default: `thirty_minute_decoder_campaign` and `bounded_encoder_campaign` (seeded random input; `PCM_FUZZ_SECS` and `PCM_ENCODER_FUZZ_SECS` set the run lengths, `PCM_FUZZ_SEED` the decoder seed in hex) and `release_decode_and_encode_throughput` are `#[ignore]`. The decoder campaign, with the s8, u8, A-law, mu-law and s64 layouts added, ran 1800 s from seed `0x711d3c0` on 12 November 2026 (release build at `f2d59632`): 15,627,983,193 inputs, 10,755,512,978 refused with an error, no panic and no failed check, at most 2.9 MB resident. The encoder campaign and the throughput test have no recorded result.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, pcm.** Input: 5 minutes of stereo sound at 44,100 Hz (53 MB), written by FFmpeg with `-c:a pcm_s16le`.

|  | Cutlass | FFmpeg (`pcm_s16le`) |
| --- | ---: | ---: |
| Wall time | 9.81 s | 0.04 s |
| Speed | 31× real time | 7,500× real time |
| CPU time | 9.79 s | 0.03 s |
| Peak memory | 2.9 MB | 13 MB |
| Threads allowed | 1 | 1 |

Cutlass is **245.2× slower** than FFmpeg here. Both decoders produced exactly the same samples.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value | Error |
| --- | --- | --- |
| Packet | 16 MiB | `Limit { what: "packet bytes" }` |
| Decoded or encoded buffer | 32 MiB | `Limit { what: "allocation" }` |
| Sample rate | 1 to 384000 Hz | `InvalidApi` at 0, `Limit` above |
| Channels | 1 to 8, the bound of `ChannelLayout` | `InvalidApi` when the layout is missing |

- Coding 10, 24-bit samples in 32-bit words, is not decoded: no container in the workspace writes it. CAF's unpacked integer samples are high-aligned and AudioToolbox reads every bit of the word, so `cutlass-caf` maps them to the coding of the word width instead.
- Only s16, f32 and f64 can be encoded. Every other coding decodes to a wider mix sample, and writing it back would have to quantize or compand.
- 32-bit integers decode to `f32`, which has a 24-bit mantissa, so their low bits are rounded. The most negative 32-bit code decodes to exactly -1.
- Resampling, dither and channel remixing are outside the crate.
- Samples are converted one at a time through a checked closure; there is no bulk copy path for layouts that already match the machine's byte order.

## Specifications

Raw PCM has no bitstream specification of its own. Integers are two's complement. Floats are IEEE 754 binary32 and binary64.
