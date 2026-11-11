# cutlass-vorbis

A Vorbis I audio decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). Both work on raw Vorbis packets; Ogg pages and granule positions belong to `cutlass-ogg`. The desktop app registers both for `CodecId::Vorbis`, ahead of the platform codecs, so Vorbis sound decodes and encodes the same on every desktop.

## What it supports

| Area | Decoder accepts | Encoder writes |
| --- | --- | --- |
| Channels | 1 to 8, in the Vorbis channel order | 1 to 8, in the Vorbis channel order |
| Sample rate | any nonzero rate | 8 kHz to 192 kHz |
| Block sizes | 64 to 8192, short at most long | 256 and 2048 up to 50 kHz, 512 and 4096 above |
| Codebooks | ordered and sparse lengths, lookup types 0, 1 and 2, single-entry books | lookup type 1 and books with no lookup |
| Floors | floor 0 and floor 1 | floor 1, multiplier 2 |
| Residues | types 0, 1 and 2 | type 1 for uncoupled channels, type 2 for coupled pairs |
| Mappings | several submaps, coupling steps | square-polar coupling of fixed channel pairs; one submap per pair and one for the rest |
| Modes | as many as the setup header lists | two: short and long, chosen by a transient detector |
| Comment header | parsed, not used for decoding | vendor `Cutlass`, no fields |
| Input and output | interleaved `f32`, full scale 1.0, not clipped | interleaved `f32` or `i16` |

The codec configuration is the three header packets. `Headers::split` takes them in Xiph lacing, as three 16-bit big-endian lengths each before its packet, or back to back. `Headers::join` writes Xiph lacing.

Timing:

- A decoded packet returns a quarter of the previous block plus a quarter of its own block per channel. The first packet after open or `reset` returns nothing. `drain` returns nothing.
- A packet stamped with a duration shorter than it decodes is an end trim, reported as `skip_trailing`. A duration of 0 counts as a length only after a packet has arrived with a nonzero one.
- Header packets inside the stream (a chained link), or a `SideData::Config` that differs from the current one, replace the configuration and restart as after a reset.
- The encoder has no delay. Its first packet has pts 0 and duration 0 and carries the headers as `SideData::Config`. The last packet from `drain` is shortened to the input length and carries `SideData::Gapless` with the trailing padding.

Encoder rate control: a bitrate of 0 uses quality 0.4. Any other bitrate is an average-rate target, and `constant_rate` tightens that loop. `VorbisEncoder::open_with` takes an explicit quality from 0.0 to 1.0.

What is refused:

| Case | Error |
| --- | --- |
| Another codec id, more than 8 channels | `Unsupported` |
| Bad header type, signature, version, block sizes or framing bit | `Malformed` |
| A header that ends early, or an audio packet that ends before its mode and window flags | `Truncated` |
| An empty packet | `InvalidApi` |
| An audio packet naming a mode the setup lacks | `Malformed` |
| Encoder rate outside 8 kHz to 192 kHz, or a layout that is not the Vorbis order | `Unsupported` |
| Encoder opened with no layout, a non-finite quality, or pushed after `drain` | `InvalidApi` |
| Encoder input that is not finite, or beyond full scale | `NonFinite`, `OutOfRange` |
| Picture input | `Unsupported` |

A failed decode or push leaves the decoder or encoder as it was.

## Using it

- `VorbisDecoder::open(&CodecParams)` reads the headers from `params.extra`. `VorbisDecoder::from_config(&[u8])` takes the headers directly. `identification()` and `layout()` describe the stream.
- `VorbisEncoder::open(&CodecParams)` and `VorbisEncoder::open_with(&CodecParams, Options)`.
- `Headers`, `Identification` and `Comments` parse and write the header packets.
- `register_decoders` and `register_encoders` add the codec to a `DecodeFactory` and an `EncodeFactory`.

```rust
use cutlass_decode::{Decoder, Output};
use cutlass_encode::{AudioInput, ChannelLayout, CodecId, CodecParams, Encoder, Pcm, SideData};
use cutlass_vorbis::{VorbisDecoder, VorbisEncoder};

let layout = ChannelLayout::stereo();
let mut params = CodecParams::audio(CodecId::Vorbis, 44_100, layout.clone());
params.bitrate = 128_000; // 0 picks quality 0.4
let mut encoder = VorbisEncoder::open(&params)?;
let pcm = Pcm::from_f32(44_100, layout, samples)?;
let mut packets = encoder.push(&AudioInput { pcm })?;
packets.extend(encoder.drain()?);

// The first packet carries the three headers.
let SideData::Config(headers) = &packets[0].side[0] else {
    return Err("no headers".into());
};
let mut decoder = VorbisDecoder::from_config(headers)?;
for packet in &packets {
    for output in decoder.push(packet)? {
        if let Output::Audio(audio) = output {
            let _interleaved = audio.pcm.as_f32();
            let _end_trim = audio.skip_trailing;
        }
    }
}
```

## How it is tested

Run `cargo test -p cutlass-vorbis`.

- Unit tests in `src` cover bit packing, codebooks, floors, the MDCT, windows, header forms, the setup round trip, and the encoder's books, floor fit, coupling and residue coding.
- `tests/decoder.rs` decodes six checked-in Ogg files (four from libvorbis, one from FFmpeg's own Vorbis encoder, and a chain of two of them; `tests/fixtures/README.md` lists how each was made). It checks the sample count against the end granule, the per-packet output length, end trims, a chained file, configuration changes, and that damaged or truncated packets fail without changing the decoder.
- `tests/decoder_oracle.rs` compares decoded samples with libvorbis (through GStreamer `vorbisdec`) and FFmpeg's own Vorbis decoder, on the fixtures and on files encoded at test time by libvorbis and by FFmpeg. The gate is 90 dB SNR per reference.
- `tests/synthetic.rs` writes streams the encoder never produces (floor 0, residue 0 and 1, a lookup type 2 book, four modes) and requires libvorbis and FFmpeg to decode them to this decoder's samples at 90 dB. FFmpeg refuses the lookup type 2 book, so that case is libvorbis only.
- `tests/encoder_oracle.rs` muxes encoder output with `cutlass-ogg` and requires libvorbis and FFmpeg to decode exactly the input's sample count and to agree with this decoder at 90 dB. It also checks that a higher quality spends more bits, that silence stays under 2 kb/s, that a bitrate target is met within 10%, and that the log-spectral distance is at most 1.5 dB above libvorbis at the same rate.
- `tests/fuzz.rs` runs in-process mutation, not cargo-fuzz. `fuzz_seeds_do_not_panic` runs 96 decode mutants and 32 encoder opens on every test run. Two timed campaigns are ignored by default:

```text
VORBIS_DECODE_FUZZ_SECS=1800 VORBIS_ENCODE_FUZZ_SECS=600 cargo test --release -p cutlass-vorbis --test fuzz -- --ignored --nocapture --test-threads=2
```

The three oracle test files need `ffmpeg` and GStreamer's `vorbisenc` and `vorbisdec` on the path. Without them they print `skipped` and pass. In that case no test compares decoded sample values with another decoder: only counts, finiteness, and the round trip through this crate's own encoder are checked. One oracle case also needs a local WebM file and skips without it.

`tests/measure.rs` is an ignored release test that times encode, decode and encoder cancellation.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, vorbis.** Input: 30 minutes of stereo sound at 44,100 Hz (19 MB), written by FFmpeg with `-c:a vorbis -strict -2 -q:a 4`.

|  | Cutlass | FFmpeg (`vorbis`) |
| --- | ---: | ---: |
| Wall time | 1.24 s | 0.53 s |
| Speed | 1,458× real time | 3,412× real time |
| CPU time | 1.23 s | 0.52 s |
| Peak memory | 4.4 MB | 13 MB |
| Threads allowed | 1 | 1 |

Cutlass is **2.3× slower** than FFmpeg here. The two outputs are not bit-identical: 87.28 dB PSNR between them, largest single difference 0.0222.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Limits:

- More than 8 channels is `Unsupported` (`MAX_CHANNELS`).
- A configuration over 16 MiB is `Limit` (`MAX_CONFIG_BYTES`).
- The codebooks of one setup may hold at most 2^22 entries in total and 2^22 vector values in total; more is `Limit`. The Vorbis format itself allows larger books.
- Lookup table sizes are checked against the bits left in the header before anything is allocated.

Gaps:

- The encoder does not write floor 0, residue 0, lookup type 2 books, more than two modes, or comment fields. The decoder reads all of them.
- There is no loss concealment. A bad packet is an error and `concealed` is always false.
- The decoder has no cancel flag; one packet is the unit of work. The encoder has one.
- Vorbis inside Matroska or WebM does not open in Cutlass, because there is no Matroska reader yet. Only Ogg Vorbis files reach this decoder.
- The oracle tests have not been verified on Linux or Windows.
- No reference PCM is checked in, so a machine without FFmpeg and GStreamer does not check decoded sample values.

## Specifications

- Vorbis I specification (Xiph.Org): bit packing, codebooks, floors 0 and 1, residues 0 to 2, mappings, modes, windows, the audio packet, and the channel order of section 4.3.9.
- Vorbis I Appendix A, for how Ogg granule positions give the start and the end trim.
