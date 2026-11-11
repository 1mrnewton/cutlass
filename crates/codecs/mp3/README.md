# cutlass-mp3

`cutlass-mp3` is an MPEG audio Layer III decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). It covers MPEG-1, MPEG-2 LSF and MPEG-2.5, from 8 kHz to 48 kHz, mono and stereo. A packet is one whole frame, header included, as `cutlass-mp3-container` reads and writes it.

The desktop app registers the decoder and the encoder ahead of the platform codecs, so an MP3 file decodes and encodes the same on every platform.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Versions | MPEG-1, MPEG-2 LSF, MPEG-2.5 | the same |
| Sample rates | 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000 Hz | the same |
| Channels | mono, stereo, joint stereo, dual channel (decoded as two channels) | mono, or joint stereo with mid/side chosen per frame |
| Bitrate | every Layer III rate, variable rate, free format | `Mp3Rate::Constant(kbps)` at any Layer III rate of the version, or `Mp3Rate::Variable(0..=9)`, 0 best |
| PCM | interleaved `f32`, full scale 1.0; 1152 samples per channel for MPEG-1, 576 otherwise | interleaved `f32` or `i16` |

The decoder reads long, start, short, stop and mixed blocks, mid/side and intensity stereo (MPEG-1 and LSF positions), `scfsi`, `preflag`, `scalefac_scale`, subblock gain, every Huffman table and both count1 tables, the CRC, and the bit reservoir up to 511 bytes back.

The encoder has a polyphase analysis filter and MDCT with block switching on attacks, a psychoacoustic model, scalefactor and subblock-gain noise shaping, Huffman table and region choice, a bit reservoir, and a lowpass set by the bitrate or quality. It writes `preflag` on MPEG-1 long blocks. It does not write intensity stereo, mixed blocks, `scfsi`, a CRC or free format.

Gapless playback:

- The decoder adds `DECODER_DELAY` (529) samples. With `SideData::Gapless` on a packet, output times count from the first valid sample: the encoder delay plus 529 samples are marked `skip_leading`, and the last `padding - 529` samples are held back and returned by `drain` marked `skip_trailing`. Without that side data every sample is returned and `drain` is empty.
- The encoder puts `SideData::Gapless` on every packet with `ENCODER_DELAY` (528). The last packet carries the trailing padding. Pass these to `Mp3Mux::set_trim` so the file records them.

After `open` or `reset`, a frame whose main data starts in bytes the decoder never saw decodes its missing granules as silence and marks them in `skip_leading`.

What is refused:

| Input | Error |
| --- | --- |
| Layer I or Layer II | `Unsupported { what: "layer" }` |
| Reserved header fields | `Malformed` |
| A CRC that does not match | `Malformed { what: "crc" }` |
| A sample rate or channel count that changes mid-stream, or differs from the open parameters | `Malformed` |
| A packet shorter than its frame; longer than its frame | `Truncated`; `Malformed { what: "frame length" }` |
| An empty packet | `InvalidApi { what: "empty packet" }` |
| Decoder opened with more than two channels, or a rate outside the nine | `Unsupported` |
| Encoder: a rate, channel count, bitrate or quality outside the table | `Unsupported` |
| Encoder input that is not finite, beyond full scale, or of the wrong rate or layout | `NonFinite`, `OutOfRange`, `InvalidApi` |
| A picture pushed to the encoder | `Unsupported { what: "picture input" }` |

A failed push leaves the decoder as it was.

## Using it

- `Mp3Decoder::open(&CodecParams)`: `codec` must be `CodecId::Mp3`. A sample rate of 0 and no layout lock onto the first frame. `extra` is ignored. `stream()` returns the rate and channel count once a frame has decoded.
- `Mp3Encoder::new(Mp3EncoderConfig { sample_rate, channels, rate })`.
- `Mp3Encoder::open(&CodecParams)`: with `constant_rate` set, the nearest Layer III rate to `bitrate`; otherwise the variable quality whose typical rate is nearest. A bitrate of 0 is variable quality 2.
- `register_decoders` and `register_encoders` add the crate to a `DecodeFactory` and an `EncodeFactory`.
- The `header` module is public: `FrameHeader`, `Version`, `Mode`, `classify_rate`, `bitrates`, `bitrate_index` and `MAX_FRAME_BYTES`.

```rust
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Pcm};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_mp3::{Mp3Decoder, Mp3Encoder, Mp3EncoderConfig, Mp3Rate};

let config = Mp3EncoderConfig { sample_rate: 44_100, channels: 2, rate: Mp3Rate::Variable(2) };
let mut encoder = Mp3Encoder::new(config)?;
let pcm = Pcm::from_f32(44_100, ChannelLayout::stereo(), samples)?;
let mut packets = encoder.push(&AudioInput { pcm })?;
packets.extend(encoder.drain()?);

let params = CodecParams::audio(CodecId::Mp3, 44_100, ChannelLayout::stereo());
let mut decoder = Mp3Decoder::open(&params)?;
for packet in &packets {
    let _outputs = decoder.push(packet)?;
}
let _tail = decoder.drain()?;
```

`Pcm::from_f32` returns `PcmError`; map it before using `?` beside the codec calls.

Two examples work on files through `cutlass-mp3-container`:

```text
cargo run --release -p cutlass-mp3 --example decode -- in.mp3 out.f32 --gapless
cargo run --release -p cutlass-mp3 --example encode -- in.f32 44100 2 out.mp3 vbr 2
```

## How it is tested

`cargo test -p cutlass-mp3` runs unit tests in `src/` and the integration tests in `tests/mp3/`.

Decoder, compared with other decoders. Each comparison asks for the same length and the ISO/IEC 11172-4 bounds at full scale 1.0: RMS difference below `2^-15 / sqrt(12)` and no sample further than `2^-14`.

- Six short fixtures in `tests/fixtures/`, against FFmpeg and mpg123. `tests/fixtures/README.md` records how they were made: LAME 3.100 on synthetic signals. They cover MPEG-1, LSF and MPEG-2.5, joint stereo with a CRC, variable rate, forced mid/side and free format. The free-format file is checked against mpg123 only, because FFmpeg refuses free format above 320 kbps.
- Random valid streams built by `src/synthetic.rs` for every sample rate, with random block shapes, tables, scalefactors and intensity bounds, against FFmpeg and mpg123. Four corners where the two disagree are each checked against the one this crate follows: LSF intensity positions from 16 (mpg123), and MPEG-1 intensity positions above 7, `scfsi` on an intensity right channel and MPEG-2.5 mixed blocks (FFmpeg).
- MP3 files from a local media folder that is not in the repository, against FFmpeg. These tests are skipped when the folder is missing. The one that decodes a thirty-minute file is `#[ignore]`.

Encoder:

- Every sample rate and layout round-trips to the input length.
- mpg123 and FFmpeg decode the encoder's files within the bounds above of this crate's own decode. `lame --decode` returns exactly the input length, within 1.5 steps of 16-bit.
- On a synthetic 44.1 kHz stereo signal: constant 64, 128, 192 and 320 kbps stay within 0.2% of the rate with at least 20, 26, 30 and 32 dB signal-to-noise; variable quality 0, 2, 4, 6 and 9 give at least 30, 28, 25, 22 and 15 dB with falling rates.
- Silence codes to silence, chunking the input does not change the bytes, short inputs keep their length, and full-scale square waves and clicks stay in range.

Bad input and lifecycle: every truncation of every fixture frame, out-of-range side information, reserved header fields, random and mutated frames, seek and reset, drain twice, a refused push, and the cancel flag. `tests/mp3/fuzz.rs` is a seeded in-process campaign over the decoder and encoder (it is not coverage-guided); two seconds of it run with every test run. The long run is `MP3_FUZZ_SECS=600 cargo test --release -p cutlass-mp3 --test mp3 -- --ignored --nocapture campaign`.

Skipped or not run:

- A missing `ffmpeg` or `mpg123` skips its comparisons with a line on stderr. `MP3_REQUIRE_ORACLES=1` turns that into a failure for the integration tests only; the random-stream tests in `src/synthetic.rs` and the `lame` and `afinfo` checks always skip quietly.
- No published conformance streams (ISO/IEC 11172-4) are used.
- No listening test has been run on the encoder.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, mp3.** Input: 30 minutes of stereo sound at 44,100 Hz (29 MB), written by FFmpeg with `-c:a libmp3lame -b:a 128k`.

|  | Cutlass | FFmpeg (`mp3float`) |
| --- | ---: | ---: |
| Wall time | 2.43 s | 0.70 s |
| Speed | 744× real time | 2,583× real time |
| CPU time | 2.41 s | 0.70 s |
| Peak memory | 3.3 MB | 13 MB |
| Threads allowed | 1 | 1 |

Cutlass is **3.5× slower** than FFmpeg here. The two outputs are not bit-identical: 148.60 dB PSNR between them, largest single difference 1.48e-06.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- A frame larger than 4096 bytes is `Limit { what: "frame bytes" }`. This is also the cap on free format.
- Layer I and Layer II are not decoded.
- More than two channels are not supported.
- The encoder does not write intensity stereo, mixed blocks, `scfsi`, a CRC or free format. Those decoder paths are tested with the LAME fixtures and the random streams only.
- Both encoder channels share one block type.
- The decoder allocates its main-data and PCM buffers for every frame.
- Memory use has not been measured.

## Specifications

- ISO/IEC 11172-3:1993, Layer III: header, side information, Huffman tables, requantization, stereo, the hybrid filterbank and the CRC-16.
- ISO/IEC 13818-3:1998 for the lower sampling frequencies.
- ISO/IEC 11172-4 for the decoder accuracy bounds the tests use.
- MPEG-2.5 (8, 11.025 and 12 kHz) has no ISO text; the MPEG-2 rules apply.
- The LAME tag convention for encoder delay and padding: `ENCODER_DELAY` 528, plus 529 samples of decoder delay.
