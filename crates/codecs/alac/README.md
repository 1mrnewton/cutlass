# cutlass-alac

`cutlass-alac` decodes and encodes Apple Lossless (ALAC). It is safe Rust: the crate has `#![forbid(unsafe_code)]` and depends only on `cutlass-decode` and `cutlass-encode`. One packet is one ALAC frame. Containers frame the packets: `cutlass-isobmff` carries ALAC in MP4 and M4A, and this crate reads no file format itself.

The desktop app registers the decoder and the encoder, ahead of the platform codecs, so an ALAC file decodes and encodes the same on every platform.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Bit depth | 16, 20, 24, 32 | 16, 20, 24, 32 |
| Channels | 1 to 8 | 1 to 8 |
| Frame length | 1 to 262144 samples per channel | 1 to 262144, default 4096 |
| Elements | SCE, CPE, LFE; DSE and FIL are skipped | SCE and CPE, in the reference element map |
| Coding | Compressed and escape (raw) elements, partial frames, shifted low bytes at 24 and 32 bits | The same; an element is stored raw when compressing it would not make it smaller |
| Modes | n/a | `EncodeMode::Default` and `EncodeMode::Fast` |

The magic cookie is the 24-byte `ALACSpecificConfig`, then an optional 24-byte `chan` atom. `Cookie::parse` reads it bare, as an MP4 `alac` box payload, as a full `alac` atom, and wrapped in `frma`. `Cookie::to_bytes` writes the bare form.

Channels come out in the order the frame stores them: C; L R; C L R; C L R Cs; C L R Ls Rs; C L R Ls Rs LFE; C L R Ls Rs Cs LFE; C Lc Rc L R Ls Rs LFE. The encoder takes three or more channels with those labels in any order and reorders them. Mono and stereo take any labels.

Through the `Decoder` and `Encoder` traits the sound crosses as mix PCM:

- Decode: 16-bit is `SampleFormat::I16`, exact. 20 and 24-bit are `SampleFormat::F32` over `2^(depth - 1)`, exact. 32-bit is `F32`, rounded to what an `f32` holds.
- Encode: with an empty `extra`, `I16` input writes 16-bit and `F32` input writes 24-bit, in 4096-sample frames. A cookie in `extra` sets the depth and the frame length.

What is refused, and how:

| Input | Error |
| --- | --- |
| Packet over 16 MiB (`MAX_PACKET_BYTES`) | `Limit` |
| Frame length 0 | `Malformed` |
| Frame length over 262144 | `Limit` |
| Bit depth other than 16, 20, 24, 32 | `Unsupported` |
| 0 channels | `Malformed` |
| More than 8 channels | `Unsupported` |
| Cookie version other than 0 | `Unsupported` |
| Golomb limit `kb` over 31 | `Malformed` |
| `chan` tag whose count differs from the channel count | `Malformed` |
| CCE or PCE element | `Unsupported` |
| Empty packet, or no cookie at open | `InvalidApi` |
| Encoder float input outside ±1 | `OutOfRange` |
| Encoder float input that is not finite | `NonFinite` |

A frame that fails leaves the decoder as it was.

## Using it

`AlacDecoder` and `AlacEncoder` work on interleaved `i32` samples at the stream's depth. `register_decoders` and `register_encoders` add them to a `DecodeFactory` and an `EncodeFactory` under `CodecId::Alac`.

```rust
use cutlass_alac::{AlacDecoder, AlacEncoder, EncoderConfig};
use cutlass_decode::ChannelLayout;
use cutlass_encode::Encoder;

// Encode: 16, 20, 24, or 32 bits, 1 through 8 channels.
let config = EncoderConfig::new(44_100, 2, 16);
let mut encoder = AlacEncoder::new(config, &ChannelLayout::stereo())?;
let mut packets = encoder.push_samples(&samples)?;
packets.extend(encoder.drain()?);
let cookie = encoder.cookie().to_bytes();

// Decode: the cookie in any form `Cookie::parse` reads.
let mut decoder = AlacDecoder::new(&cookie)?;
for packet in &packets {
    let frame = decoder.decode_frame(&packet.data)?; // frame.samples: Vec<i32>
}
```

`examples/roundtrip.rs` is the same round trip as a program: `cargo run --release -p cutlass-alac --example roundtrip`.

Other public items:

- `EncoderConfig` holds the rate, channel count, depth, frame length, and mode. `EncodeMode::Fast` is the reference encoder's fast stereo path; mono and multichannel streams encode as in `Default`.
- `AlacConfig` and `Cookie` are the parsed cookie. `default_layout` gives the frame channel order for a count, and `LAYOUT_TAGS` the layout tag written for each count.
- `AlacDecoder::open` and `AlacEncoder::open` take `CodecParams`, with the cookie in `extra`.
- Through the `Encoder` trait, the first packet carries the cookie as `SideData::Config`. The packet `drain` writes carries it again with the largest frame size and the average bit rate filled in.
- Through `Decoder::push`, a packet whose duration is shorter than its frame gives only that many samples. AudioToolbox pads the last frame to full length and states the real length in the container. `decode_frame` returns the whole frame.

## How it is tested

The tests are in `src/tests` and in each module.

- `decode_oracle.rs`: ALAC files written by FFmpeg and by `afconvert` are demuxed with `cutlass-isobmff` and decoded here. The integers must equal FFmpeg's decode and `afconvert`'s, for stereo, mono, 5.1, every count from one to eight channels, and 20 and 32-bit streams.
- `encode_oracle.rs`: frames written here must decode back to the input here, in FFmpeg, and in `afconvert`. `compression_matches_the_reference_encoders` requires the stereo frames to equal `afconvert -d alac` in default mode and `afconvert -d alac -q 0` in fast mode.
- `end_to_end.rs`: an `.m4a` from `afconvert` goes through demux, decode, encode, and mux, and FFmpeg, `afconvert`, and `afinfo` must read the result as they read the source.
- `hostile.rs`: hand-built frames with each named fault, truncations and bit flips of real frames, random packets, and malformed cookies. Nothing may panic.
- `lifecycle.rs`: the trait behavior: cookie side data, reset, cancel, padded last frames, float and integer input checks.
- `campaign.rs`: three seeded campaigns (decoder, cookie, round trip). They are `#[ignore]` and run only when asked. They are not libFuzzer runs.

```text
cargo test -p cutlass-alac
ALAC_REQUIRE_ORACLES=1 cargo test -p cutlass-alac
ALAC_FUZZ_SECS=1800 cargo test -p cutlass-alac --release -- --ignored campaign
cargo bench -p cutlass-alac --bench against_ffmpeg
```

A test that needs `ffmpeg` prints a message and passes without checking anything when the tool is missing. `afconvert` and `afinfo` are macOS system tools; without them the checks against AudioToolbox are left out and the end-to-end test skips. Most oracle tests cut their sound from real recordings in a folder outside the repository and skip when it is missing. `ALAC_REQUIRE_ORACLES` turns every such skip into a failure.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, alac.** Input: 30 minutes of stereo sound at 44,100 Hz (227 MB), written by FFmpeg with `-c:a alac`.

|  | Cutlass | FFmpeg (`alac`) |
| --- | ---: | ---: |
| Wall time | 2.20 s | 2.17 s |
| Speed | 822× real time | 833× real time |
| CPU time | 2.17 s | 2.16 s |
| Peak memory | 4.3 MB | 13 MB |
| Threads allowed | 1 | 1 |

Cutlass and FFmpeg take **about the same time** here. Both decoders produced exactly the same samples.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- CAF is not read or written.
- Through the traits, 32-bit sound is rounded to `f32`. `decode_frame` and `push_samples` are exact at every depth.
- The final cookie, with the largest frame size and the average bit rate, goes out only on the packet `drain` writes. When the input is an exact multiple of the frame length, `drain` writes no packet and those two fields stay zero in the only cookie sent. `AlacEncoder::cookie` still returns the full one.
- The encoder follows Apple's reference search, not a harder one. Identical bytes to `afconvert` are checked for stereo only; mono and multichannel frames are checked for losslessness.
- A `chan` tag is checked for its channel count only. The channel labels are always the reference layout for the count.
- After open, the `Encoder` trait checks the rate and the channel count of each input, not its channel labels.
- The encoder does not clip. A caller must bring floats inside ±1 first; the app's export does.
- The code is single-threaded and has no SIMD.

## Specifications

There is no ALAC standard. The code follows Apple's open-source ALAC reference:

- `ALACDecoder.cpp` and `ALACEncoder.cpp`, for the frame and element syntax and the encoder search.
- `ag_dec.c` and `ag_enc.c`, for the adaptive Golomb coding.
- `dp_dec.c` and `dp_enc.c`, for the adaptive predictor.
- `ALACAudioTypes.h`, for the channel layout tags.
- `ALACMagicCookieDescription.txt`, for the cookie.
