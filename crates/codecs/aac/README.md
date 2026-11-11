# cutlass-aac

`cutlass-aac` is an AAC-LC decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). It reads and writes raw AAC access units. ADTS and LOAS/LATM framing is in `cutlass-adts`, and MP4 boxes are in `cutlass-isobmff`.

The desktop app registers both the decoder and the encoder. On macOS and Windows the platform AAC codec is registered ahead of this crate, so this crate opens when the platform one declines. On Linux it is the only AAC codec the app registers.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Object type | AAC-LC (type 2) | AAC-LC |
| Frame length | 1024 or 960 samples | 1024 from `open`; 960 through `EncoderConfig` |
| Sample rate | 96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350 Hz | the same 13 rates |
| Channels | configurations 1 to 7, 11 and 12, or a program config inside the `AudioSpecificConfig`, up to 8 channels | a layout that matches configuration 1 to 7, 11 or 12 |
| PCM | interleaved `f32`, full scale 1.0 | interleaved `f32`, or `i16` divided by 32768 |

Channel order is the AAC element order, center first, not WAV order. Configuration 6 is 5.1, 7 is the wide 7.1 layout and 12 is 7.1 with side surrounds.

The decoder reads single, pair and LFE elements with Huffman coding and escapes, pulse data, temporal noise shaping, perceptual noise substitution, mid/side and intensity stereo, sine and KBD windows, and long, start, short and stop blocks. Data stream and fill elements are skipped, so dynamic range data in a fill element is not applied.

The encoder uses KBD windows, one frame of lookahead for block switching (one decision shared by the non-LFE channels), mid/side in bands where the two channels of a pair are strongly correlated, and a lowpass on the LFE channel. It never writes temporal noise shaping, noise substitution, intensity stereo, pulse data, prediction or coupling. The same samples and settings give the same bytes.

Rate control is `RateControl::Average { bitrate }` or `RateControl::Variable { quality, cap }`, with quality 1 to 10. Average mode aims at a mean rate and does not pad access units to one size. A bitrate must be at least 8000 bps and at most 6144 bits per channel per frame; anything else is `CodecError::InvalidApi { what: "bitrate" }`.

Every encoded packet carries `SideData::Gapless` with an encoder delay of one frame. The first packet also carries `SideData::Config` with the `AudioSpecificConfig`. The packets `drain` returns carry the trailing padding.

What is refused:

| Input | Error |
| --- | --- |
| HE-AAC: spectral band replication or parametric stereo, in the config or in a fill element | `Unsupported { what: "sbr" }` |
| Main, SSR, LTP, error-resilient, USAC and every other object type | `Unsupported` |
| An explicit 24-bit sample rate | `Unsupported { what: "explicit sample rate" }` |
| Coupling elements, a program config inside an access unit, gain control, a core coder, the third extension flag | `Unsupported` |
| Prediction in an LC stream | `Malformed { what: "prediction" }` |
| Channel configuration 13, or any other the table does not list | `Unsupported { what: "channel configuration" }` |
| A `SideData::Config` that differs from the open config | `Unsupported { what: "configuration change" }` |
| An empty packet | `InvalidApi { what: "empty packet" }` |
| A picture pushed to the encoder | `Unsupported { what: "picture" }` |
| Encoder input that is not finite, or beyond full scale | `NonFinite`, `OutOfRange` |

An HE-AAC stream is refused, not played as its low band alone. The crate has no ADIF reader.

## Using it

`register_decoders` and `register_encoders` add the crate to a `DecodeFactory` and an `EncodeFactory` under `CodecId::Aac`. The types can also be used directly:

- `AacDecoder::open(&CodecParams)` reads `CodecParams.extra` as an `AudioSpecificConfig`. With an empty `extra` it builds an AAC-LC config from the sample rate and layout. `push_packet` returns one interleaved `f32` frame, `drain_pcm` returns the overlap tail, and `config` returns the `AudioConfig`. The `Decoder` trait methods wrap these and stamp each frame with a time.
- `AacEncoder::open(&CodecParams)` encodes 1024-sample frames at `params.bitrate`; a bitrate of 0 means 64 kbps per channel. `AacEncoder::from_config(EncoderConfig)` also selects a 960-sample frame or variable quality.

```rust
use cutlass_aac::{AacDecoder, AacEncoder, EncoderConfig, Profile, RateControl};
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Pcm};
use cutlass_encode::{AudioInput, Encoder};

let mut params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
params.bitrate = 96_000;

let mut encoder = AacEncoder::open(&params)?;
let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), samples)?;
let mut packets = encoder.push(&AudioInput { pcm })?;
packets.extend(encoder.drain()?);

let mut decoder = AacDecoder::open(&params)?;
for packet in &packets {
    let _frame = decoder.push_packet(packet)?;
}
let _tail = decoder.drain_pcm()?;

let _variable = AacEncoder::from_config(EncoderConfig {
    profile: Profile::AacLc,
    mpeg2: false,
    sample_rate: 48_000,
    layout: ChannelLayout::mono(),
    frame_length: 1024,
    rate: RateControl::Variable { quality: 6, cap: None },
})?;
```

The codec calls return `CodecError`. `Pcm::from_f32` returns `PcmError`, so map it before using `?` beside them.

`examples/roundtrip.rs` asks an installed `ffmpeg` for a short ADTS file, strips the headers and decodes the access units: `cargo run --release -p cutlass-aac --example roundtrip`. Without `ffmpeg` it prints a skip line and exits 0.

## How it is tested

All tests are unit tests inside `src/`; run them with `cargo test -p cutlass-aac`.

- Round trips through the crate's own encoder and decoder, with a floor on the signal-to-noise ratio after the delay is trimmed: mono 1 kHz sine at 96 kbps with 1024 and 960 sample frames (40 dB), stereo at 128 kbps (35 dB per channel), a tone near Nyquist (20 dB), 5.1 at 384 kbps (30 dB, LFE 25 dB), 7.1 at 512 kbps (25 dB per channel), and 16-bit input (40 dB). Silence stays below `1e-4` and a DC level keeps its mean within 0.02.
- Against FFmpeg: `ffmpeg_decodes_our_access_units` and `we_decode_an_ffmpeg_stream` need 40 dB and a lag of at most 1024 samples. They are skipped with a line on stderr when `ffmpeg` is not installed; set `AAC_REQUIRE_ORACLES` to make a missing `ffmpeg` fail.
- Rate: `noise_over_two_seconds_stays_inside_the_rate_window` keeps average mode between 0.70 and 1.10 times the target.
- Decoder tools the encoder never writes (intensity, pulse, noise substitution) are tested on hand-built access units.
- Bad input: `random_access_units_fail_closed` pushes 256 random packets and checks that an error leaves the frame count unchanged. A wrong ADTS CRC and a truncated frame are checked the same way. A decoded frame with a sample that is not finite or is past 4.0 comes out as silence marked `concealed`.
- The filterbank, Huffman tables, quantizer, bit reader and config parser have their own tests.

Not run or missing:

- `thirty_minute_decoder_campaign` is `#[ignore]`; no result is recorded here.
- No published conformance streams (ISO/IEC 14496-26) are used.
- The program config path of the `AudioSpecificConfig` parser has no test.
- No test decodes a real-world file; the FFmpeg tests use a generated sine.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, aac.** Input: 30 minutes of stereo sound at 44,100 Hz (29 MB), written by FFmpeg with `-c:a aac -b:a 128k`.

|  | Cutlass | FFmpeg (`aac`) |
| --- | ---: | ---: |
| Wall time | 5.68 s | 0.56 s |
| Speed | 318× real time | 3,229× real time |
| CPU time | 5.65 s | 0.55 s |
| Peak memory | 12 MB | 21 MB |
| Threads allowed | 1 | 1 |

Cutlass is **10.1× slower** than FFmpeg here. The two outputs are not bit-identical: 51.62 dB PSNR between them, largest single difference 0.957 (lengths 79,744,000 and 79,742,976 sample frames, compared after sliding one by 1024).

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- An access unit larger than 8192 bytes is `Limit { what: "access unit bytes" }`, on decode and on encode.
- An `AudioSpecificConfig` longer than 64 bytes, more than 32 elements in one access unit, a data stream element over 512 bytes and a fill element over 270 bytes are `Limit` errors.
- At most 8 channels.
- HE-AAC v1 and v2, Main, SSR, LTP, error-resilient profiles, LD, ELD, USAC and ADIF are not implemented.
- A stream with an explicit sample rate does not open.
- Dynamic range control is skipped, not applied.
- The encoder writes none of TNS, noise substitution, intensity stereo or pulse data.
- `EncoderConfig::mpeg2` only applies the MPEG-2 limits (no 960-sample frame, no 7350 Hz). The decoder never opens in MPEG-2 mode.
- The decoder copies every channel's overlap buffer on each packet so that a failed frame changes nothing.
- Memory use has not been measured.

## Specifications

- ISO/IEC 14496-3:2009, subpart 4: AAC-LC syntax, the filterbank and the decoder tools.
- ISO/IEC 13818-7: MPEG-2 AAC and the bits the ADTS CRC covers.
- ISO/IEC 11172-3: the CRC-16 polynomial `x^16 + x^15 + x^2 + 1`, starting at `0xFFFF`.
