# cutlass-opus

`cutlass-opus` decodes and encodes Opus (RFC 6716 as updated by RFC 8251). It is safe Rust: the crate has `#![forbid(unsafe_code)]` and depends only on `cutlass-decode` and `cutlass-encode`. One packet is one Opus packet. Ogg framing is in `cutlass-ogg` and MP4 framing in `cutlass-isobmff`.

The desktop app registers the decoder and the encoder, ahead of the platform codecs, so Opus decodes and encodes the same on every platform.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Modes | SILK-only, hybrid, CELT-only | CELT-only |
| Frame sizes | 2.5 to 60 ms, frame-count codes 0 to 3 with padding, up to 120 ms per packet | 20 ms, one frame per packet |
| Channels | Mono and stereo streams; multistream up to 8 output channels | Mono or stereo |
| Channel mapping family | 0, 1, and 255 | 0 |
| Sample rate | Output at 8, 12, 16, 24, or 48 kHz | 48 kHz input only |
| Bitrate | n/a | 6 to 510 kb/s, VBR or CBR |
| Loss | Concealment (`conceal`) and in-band FEC (`decode_fec`) | Writes no FEC data |

PCM is interleaved `f32` at full scale 1.0. Packet times count at 48 kHz.

Decoder details:

- `CodecParams.extra` is an `OpusHead` packet or a `dOps` box body. With no configuration, a mono or stereo layout makes a one-stream header.
- `params.sample_rate` picks the output rate when it is one of the five above. Any other rate decodes at 48 kHz.
- Multistream output channels follow the mapping table in the header, including silent channels, in the Vorbis channel order. The header's output gain is applied.
- The pre-skip is not removed. The container's edit says how many samples to drop from the start.
- A packet whose `duration` is shorter than its decoded length carries the difference as `skip_trailing`.

Encoder details:

- A `bitrate` of 0 picks 64 kb/s mono or 96 kb/s stereo. Other values are clamped to 6 to 510 kb/s. `constant_rate` selects CBR; otherwise the rate is VBR.
- The bitrate picks the coded bandwidth: narrowband at the lowest rates, fullband from about 18 kb/s per channel.
- Input is `f32` or `i16` PCM.
- The first packet carries the `OpusHead` as `SideData::Config`, with a pre-skip of 120 samples. The last packet is shortened so the durations add up to the pre-skip plus the input, and carries `SideData::Gapless`.

What is refused, and how:

- A damaged packet is `CodecError::Malformed` or `Truncated`. A failed decode leaves the decoder as it was.
- An empty packet is `InvalidApi`.
- Mapping families 2 and 3 (ambisonics) and the reserved families are `Unsupported`. More than 8 output channels is `Limit`.
- An `OpusHead` major version above 0, or a `dOps` version other than 0, is `Unsupported`.
- The encoder refuses a rate other than 48 kHz and a channel count other than 1 or 2 with `Unsupported`.
- A non-finite sample is `NonFinite` and a sample over full scale is `OutOfRange`. Nothing is consumed.
- A push after `drain` is `InvalidApi` until `reset`.

## Using it

`OpusDecoder` and `OpusEncoder` implement the `Decoder` and `Encoder` traits. `register_decoders` and `register_encoders` add them to a `DecodeFactory` and an `EncodeFactory` under `CodecId::Opus`.

```rust
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Output};
use cutlass_encode::{AudioInput, Encoder, Pcm};
use cutlass_opus::{OpusDecoder, OpusEncoder};

let mut params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
params.bitrate = 96_000;

let mut encoder = OpusEncoder::open(&params)?;
let pcm = Pcm::from_f32(48_000, ChannelLayout::stereo(), samples)?;
let mut packets = encoder.push(&AudioInput { pcm })?;
packets.extend(encoder.drain()?);

let mut decoder = OpusDecoder::open(&params)?;
for packet in &packets {
    for output in decoder.push(packet)? {
        if let Output::Audio(audio) = output {
            let _ = audio.pcm;
        }
    }
}
```

Other public items:

- `OpusDecoder::with_head` takes a parsed `OpusHead` and an output rate. `decode`, `conceal`, and `decode_fec` return the samples directly. `conceal_output` returns an output marked `concealed`.
- `OpusDecoder::final_range` is the range coder state after the last packet, the value the RFC test vectors record.
- `OpusHead` parses and writes both the Ogg form (`to_ogg`) and the MP4 form (`to_dops`). `vorbis_mapping` gives the default family 1 mapping for a channel count.
- `parse_packet` splits a packet into frames (`Frames`), and `Toc` reads the first byte into its `Mode`, `Bandwidth`, and frame size.

## How it is tested

- Unit tests sit beside each module: the range coder, packet framing, the header, SILK, CELT, and the decoder and encoder lifecycles.
- `the_rfc_8251_vectors_match_every_final_range_and_pass_opus_compare` decodes the twelve RFC 8251 test vectors in stereo and in mono and compares every final range. With `CUTLASS_OPUS_COMPARE` naming the RFC `opus_compare` tool, each output must pass it. The vectors are not in the repository: `CUTLASS_OPUS_VECTORS` names their folder, and the test skips without it.
- `tests/interop.rs`: FFmpeg, with libopus and with its own Opus encoder, writes `.opus` and `.mp4` files covering SILK, hybrid, CELT, 2.5 to 60 ms frames, and 5.1. They are read by `cutlass-ogg` or `cutlass-isobmff` and decoded here. The sample count must equal libopus's and the samples must match to at least 60 dB. Files written here must open in FFmpeg with the input's sample count.
- `tests/encoder_oracle.rs`: every file written here must decode in FFmpeg. The VBR mean must be within 10% of the request on every item that is not sparse, every CBR packet must have the exact size, and the mean log-spectral distance must be at most 1.5 dB above libopus's at 64, 96, and 128 kb/s stereo.
- `tests/campaign.rs`: seeded in-process campaigns on every test run, 2 seconds each by default. Damaged and random packets go into the decoder, and random settings and hostile samples into the encoder. Nothing may panic and every sample must be finite. These are not libFuzzer runs.

```text
cargo test -p cutlass-opus
CUTLASS_OPUS_VECTORS=<folder> CUTLASS_OPUS_COMPARE=<opus_compare> cargo test --release -p cutlass-opus --lib decode::
CUTLASS_OPUS_CAMPAIGN_SECONDS=300 cargo test -p cutlass-opus --test campaign -- --nocapture
cargo bench -p cutlass-opus --bench against_ffmpeg
```

The interop and encoder tests skip when `ffmpeg` built with libopus is not installed. `CUTLASS_LOCAL_FILES` names a folder of real recordings for them; without it they use synthetic sound. Some SILK and CELT unit tests also read the vectors, and one FEC test reads reference streams named by `CUTLASS_OPUS_SILK_FEC`; they skip without them. The quality gate is a spectral measure; no listening test has been done.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, opus.** Input: 30 minutes of stereo sound at 48,000 Hz (19 MB), written by FFmpeg with `-c:a libopus -b:a 96k`.

|  | Cutlass | FFmpeg (`opus`) |
| --- | ---: | ---: |
| Wall time | 2.97 s | 1.74 s |
| Speed | 609× real time | 1,039× real time |
| CPU time | 2.94 s | 1.72 s |
| Peak memory | 4.4 MB | 13 MB |
| Threads allowed | 1 | 1 |

Cutlass is **1.7× slower** than FFmpeg here. The two outputs are not bit-identical: 80.64 dB PSNR between them, largest single difference 0.0134 (lengths 86,795,388 and 86,795,076 sample frames, compared after sliding one by 312).

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- The encoder is CELT-only. It does not write SILK or hybrid frames, so low-bitrate speech is coded as CELT.
- The encoder takes 48 kHz mono or stereo only and writes 20 ms frames. There is no multistream encoding, no DTX, and no in-band FEC on the encode side.
- Output stops at 8 channels. Ambisonic mapping families 2 and 3 are refused.
- Family 255 has no speaker positions. Its channels are labeled by count only so that a buffer can be built.
- Nothing outside this crate calls `conceal`, `conceal_output`, or `decode_fec`.
- The decoder copies its whole stream state before every packet, so that a failed packet can be undone. The encoder does the same on every push.
- Opus in WebM does not open in Cutlass: the Matroska container is not built.
- The app's export writes Opus alone, in an `.opus` file. A movie with Opus sound is refused.

## Specifications

- RFC 6716, Definition of the Opus Audio Codec.
- RFC 8251, Updates to the Opus Audio Codec.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec, for `OpusHead`, the channel mapping, the output gain, and the pre-skip.
- Encapsulation of Opus in ISO Base Media File Format, for the `dOps` box.
