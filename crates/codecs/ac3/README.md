# cutlass-ac3

An AC-3 and E-AC-3 audio decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). One crate covers both syntaxes of ATSC A/52. The desktop app registers the decoder and the encoder for `CodecId::Ac3` and `CodecId::Eac3`, ahead of the platform codecs.

## What it supports

Decoder:

| Area | Accepted |
| --- | --- |
| Syntax | `bsid` 0 to 8 (AC-3) and 11 to 16 (E-AC-3). Either codec id decodes both |
| Sample rate | 32, 44.1 and 48 kHz. A rate of 0 at open lets the first frame decide |
| Channels | every `acmod`, with or without LFE; at most 8 output channels |
| AC-3 tools | coupling, rematrixing, block switching, delta bit allocation, phase flags |
| E-AC-3 frames | 1, 2, 3 or 6 blocks, several syncframes in one access unit |
| E-AC-3 tools | spectral extension, the adaptive hybrid transform, gain-adaptive quantization, enhanced coupling in long blocks, transient pre-noise processing |
| E-AC-3 substreams | dependent substreams (7.1), and a choice of independent program 0 to 7 |
| Dynamic range | `DynamicRange::Off` (default), `Line { cut, boost }`, `Heavy` |
| Downmix | `Downmix::None` (default), `LoRo`, `LtRt`, `Mono` |
| Gapless | `SideData::Gapless` trims the encoder delay and the trailing padding |

One push is one access unit: one AC-3 syncframe, or an E-AC-3 independent frame with its dependent frames and the frames of other programs. Output is interleaved `f32` at full scale 1.0 in WAVEFORMATEXTENSIBLE speaker order. A/52 surrounds come out as side channels, a single surround as back center, and the rear pair of a 7.1 program as the back pair. The layout locks on the first decoded unit. Every syncframe's CRC is checked.

`Metadata` exposes the dialogue level, compression words, mix levels, surround flags and production information of the last unit. `Tools` counts the coding tools seen since open or `reset`.

Encoder:

| Area | AC-3 | E-AC-3 |
| --- | --- | --- |
| Layouts | mono to 5.1: `acmod` 1 to 7, with or without LFE; the surround pair may be the side or the back pair | the same, plus 7.1 (side pair and back pair) as a 5.1 frame and one dependent frame |
| Sample rate | 32, 44.1, 48 kHz | 32, 44.1, 48 kHz |
| Bit rate | a Table 5.18 rate; `Ac3Encoder::open` snaps a request to the nearest | any rate that fits a syncframe |
| Blocks per syncframe | 6 | 1, 2, 3 or 6 |
| Tools written | coupling, rematrixing, block switching on transients | those, plus spectral extension, the adaptive hybrid transform with gain-adaptive quantization, and transient pre-noise words |
| Metadata | `dialnorm`, `bsmod`, `dsurmod`, downmix codes | those, plus `lfemixlevcod` and `pgmscl` |

A bit rate of 0 picks the default. In kbit/s for 1, 2, 3, 4, 5 and 5.1 channels: AC-3 96, 192, 256, 320, 384, 448; E-AC-3 64, 128, 192, 224, 256, 384. A 7.1 program defaults to 384 plus 128 for the back pair.

The decoded stream lags the input by `ENCODER_DELAY`, 256 samples. Every packet carries `SideData::Gapless` with that delay. The first packet also carries the ISOBMFF `dac3` or `dec3` payload as `SideData::Config`, and the last packet from `drain` carries the trailing padding. `dynrng` and `compr` words are written only when `EncoderMetadata::compression` is on.

What is refused:

| Case | Error |
| --- | --- |
| `bsid` 9, 10 or above 16; E-AC-3 reduced sample rates | `Unsupported` |
| Enhanced coupling in a short block | `Unsupported` |
| A dependent channel location outside the 11 speakers (unless `Extension::Core`), more than 8 channels | `Unsupported` |
| Bad syncword, reserved rate or frame size code, failed CRC, bytes after the last syncframe, a dependent frame before any independent one, a sample rate change | `Malformed` |
| A syncframe cut short | `Truncated` |
| An empty packet, a program above 7 | `InvalidApi` |
| More than 432 syncframes in one access unit | `Limit` |
| Encoder: other sample rates, other layouts, an AC-3 rate not in Table 5.18 (`Ac3Encoder::new`), a rate too low for the layout, LFE mix or program scale in AC-3 | `Unsupported` |
| Encoder: an E-AC-3 frame that cannot be packed at the chosen rate | `Limit` |
| Encoder input: wrong rate or layout, non-finite, beyond full scale | `InvalidApi`, `NonFinite`, `OutOfRange` |

A failed push leaves the decoder as it was. A bad syncframe is an error, not concealed audio.

## Using it

- `Ac3Decoder::open(&CodecParams)` and `Ac3Decoder::with_options(&CodecParams, Options)`. No extradata is needed: every syncframe carries its own configuration. `metadata()`, `tools()`, `stream()` and `skipped_dependents()` report on what was decoded.
- `Ac3Encoder::open(&CodecParams)` and `Ac3Encoder::new(EncoderConfig)`. `config_record()` returns the `dac3` or `dec3` payload and `frame_samples()` the samples per syncframe.
- `probe(&[u8])` reads the first `PROBE_BYTES` of a syncframe into a `SyncInfo` (syntax, substream, frame size, rate, blocks). A reader uses it to split a raw stream.
- `Dac3` and `Dec3` parse and write the ISOBMFF configuration records.
- `register_decoders` and `register_encoders` add both codec ids to a `DecodeFactory` and an `EncodeFactory`.

```rust
use cutlass_ac3::{Ac3Decoder, Ac3Encoder, Downmix, EncoderConfig, Format, Options};
use cutlass_decode::{CodecId, CodecParams, Decoder, Output};
use cutlass_encode::{AudioInput, ChannelLayout, Encoder, Pcm};

let layout = ChannelLayout::stereo();
let mut config = EncoderConfig::new(Format::Eac3, 48_000, layout.clone());
config.bit_rate = 192_000; // 0 picks the default
let mut encoder = Ac3Encoder::new(config)?;
let pcm = Pcm::from_f32(48_000, layout.clone(), samples)?;
let mut packets = encoder.push(&AudioInput { pcm })?;
packets.extend(encoder.drain()?);

let options = Options { downmix: Downmix::LoRo, ..Options::default() };
let params = CodecParams::audio(CodecId::Eac3, 48_000, layout);
let mut decoder = Ac3Decoder::with_options(&params, options)?;
for packet in &packets {
    for output in decoder.push(packet)? {
        if let Output::Audio(audio) = output {
            let _interleaved = audio.pcm.as_f32();
        }
    }
}
// With gapless side data the decoder holds back the padding; drain returns the rest.
let _tail = decoder.drain()?;
```

`examples/decode.rs` decodes a raw `.ac3` or `.eac3` file to raw `f32`:

```text
cargo run --release -p cutlass-ac3 --example decode -- in.ac3 out.f32 [--drc line|heavy] [--downmix loro|ltrt|mono] [--program N] [--core]
```

## How it is tested

Run `cargo test -p cutlass-ac3`. The integration tests are one binary, `tests/ac3`.

- Fixtures: eleven short files made by FFmpeg's `ac3` and `eac3` encoders from one tone per speaker (`tests/fixtures/README.md`). FFmpeg's decode of each is checked in under `tests/fixtures/reference`, so `every_fixture_matches_the_committed_ffmpeg_pcm` runs with no tool installed.
- Live references: when `ffmpeg` is on the path the decoder is compared with it, and FFmpeg must decode what the encoder writes at every layout, rate and block count. On macOS `afconvert` is a second reference. Apple's decoder does not render every mode as discrete channels, so its test compares only the speakers it renders.
- FATE samples: FFmpeg's encoders do not write spectral extension, the hybrid transform, dependent substreams, short frames or block switching. `tests/ac3/fate.rs` checks those on FFmpeg's FATE samples when `AC3_FATE_SAMPLES` names the folder. The samples are not in the repository.
- Program material: `tests/ac3/real.rs` encodes a few seconds of speech, music and thunder from a folder of local recordings and compares this decoder, FFmpeg and Apple. It skips when the folder is missing.
- Encoder checks: frame sizes against Table 5.18, any E-AC-3 rate at any block count, decoded length equal to the input, metadata round trips, 7.1 speaker placement, and noise and impulse results within 1 dB of FFmpeg's encoder.
- Lifecycle: empty packets, failed pushes, reset, drain, seek preroll, cancel, and every truncation of a frame.
- Mutation: `bounded_mutations_do_not_panic` runs 2048 mutated frames and 128 encoder rounds on every test run. cargo-fuzz is not used. The long campaign is ignored by default:

```text
AC3_FUZZ_SECS=1800 cargo test -p cutlass-ac3 --release --test ac3 -- --ignored campaign --nocapture
```

A missing reference tool prints `skipped` and the test passes. `AC3_REQUIRE_ORACLES=1` turns a missing tool or folder into a failure.

Enhanced coupling is checked for self-consistency only: synthetic frames, and the angle and chaos formulas against Annex E. FFmpeg does not decode the tool and no outside sample was available, so nothing independent confirms that decode.

`benches/throughput.rs` times both codecs and, when it is installed, an `ffmpeg` process.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, ac3.** Input: 30 minutes of stereo sound at 44,100 Hz (44 MB), written by FFmpeg with `-c:a ac3 -b:a 192k`.

|  | Cutlass | FFmpeg (`ac3`) |
| --- | ---: | ---: |
| Wall time | 1.65 s | 0.51 s |
| Speed | 1,096× real time | 3,546× real time |
| CPU time | 1.65 s | 0.51 s |
| Peak memory | 7.8 MB | 18 MB |
| Threads allowed | 1 | 1 |

Cutlass is **3.2× slower** than FFmpeg here. The two outputs are not bit-identical: 81.05 dB PSNR between them, largest single difference 0.00347 (lengths 79,742,976 and 79,742,720 sample frames, compared after sliding one by 256).

**Decode, eac3.** Input: 30 minutes of stereo sound at 44,100 Hz (44 MB), written by FFmpeg with `-c:a eac3 -b:a 192k`.

|  | Cutlass | FFmpeg (`eac3`) |
| --- | ---: | ---: |
| Wall time | 1.61 s | 0.50 s |
| Speed | 1,123× real time | 3,616× real time |
| CPU time | 1.60 s | 0.50 s |
| Peak memory | 7.8 MB | 18 MB |
| Threads allowed | 1 | 1 |

Cutlass is **3.2× slower** than FFmpeg here. The two outputs are not bit-identical: 81.86 dB PSNR between them, largest single difference 0.00379 (lengths 79,742,976 and 79,742,720 sample frames, compared after sliding one by 256).

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- A syncframe is at most 4096 bytes (`MAX_FRAME_BYTES`). An access unit holds at most 432 syncframes.
- Output is at most 8 channels. Dependent channel locations beyond the 11 mapped speakers are refused, or dropped with `Extension::Core`.
- No loss concealment. `concealed` is always false and a CRC failure is an error for the caller to handle.
- E-AC-3 reduced sample rates are not decoded or written.
- Enhanced coupling is never written. On decode, the last block of each syncframe is rebuilt with the following block taken as zero, because a syncframe is rendered before the next one arrives. Annex E 3.5.5.1 uses the first block of the next syncframe there, so those coupled bins can differ from the specification when the angles are not zero.
- A short block on a channel in enhanced coupling is refused.
- The encoder's rate control never chooses delta bit allocation segments.
- The encoder writes one program, `substreamid` 0. It does not write 1+1 dual mono.
- Program scale (`pgmscl`) is parsed and reported, not applied. `program_gain` turns it into a gain for a mixer.
- FFmpeg skips transient pre-noise processing, so a stream that carries those words does not decode sample for sample the same there.
- There is no reader for raw `.ac3` or `.eac3` files in Cutlass. The codec is reached through ISOBMFF files (MP4, MOV, M4A); the WAV reader also maps the AC-3 and E-AC-3 format tags.
- The live reference tests have not been verified on Linux or Windows.

## Specifications

- ATSC A/52:2018, Digital Audio Compression (AC-3, E-AC-3): the main document for AC-3.
- A/52 Annex D, alternate bit stream syntax (extended mix levels).
- A/52 Annex E, Enhanced AC-3.
