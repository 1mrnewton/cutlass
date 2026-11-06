# cutlass-opus

## What this crate is

`cutlass-opus` is a native safe Rust Opus decoder and encoder. The crate forbids `unsafe` and links no libopus, FFmpeg, or other C. ffmpeg, the RFC `opus_compare` tool, and the RFC 8251 test vectors are external oracles used by the tests and by `benches/against_ffmpeg.rs`.

The decoder reads every Opus packet RFC 6716 and RFC 8251 define: SILK, hybrid, and CELT frames, every frame size and bandwidth, mono and stereo streams, and multistream packets for channel mapping families 0, 1, and 255 (up to 8 output channels). The encoder writes CELT-only packets. Ogg framing lives in `cutlass-ogg` and MP4 framing in `cutlass-isobmff`. The longer support record, with every check and its numbers, is `STATUS.md`.

## Specification

- RFC 6716, Definition of the Opus Audio Codec.
- RFC 8251, Updates to the Opus Audio Codec, and its test vectors.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec, for `OpusHead`, the channel mapping, output gain, and pre-skip.
- Encapsulation of Opus in ISO Base Media File Format 0.6.8, for the `dOps` box this crate also reads.

## API

PCM is interleaved `f32` at full scale 1.0. Timestamps count at 48 kHz.

`OpusDecoder::open` reads `CodecParams.extra` as an `OpusHead` packet or a `dOps` box body; with no configuration, a mono or stereo layout makes a single-stream header. `params.sample_rate` picks the output rate when it is 8, 12, 16, 24, or 48 kHz, else 48 kHz. `OpusDecoder::with_head` takes a parsed `OpusHead`. Through `Decoder::push` each packet gives one `AudioOutput`; a packet whose `duration` is shorter than its decoded length carries the difference as `skip_trailing`. The pre-skip is left to the container's edit. `decode`, `conceal`, and `decode_fec` give the samples directly; `final_range` is the range coder state the RFC vectors record.

`OpusEncoder::open` takes 48 kHz mono or stereo. A `bitrate` of 0 picks 64 kb/s mono or 96 kb/s stereo; others are clamped to 6 to 510 kb/s. `constant_rate` selects CBR. Packets are 20 ms. The first packet carries the `OpusHead` as `SideData::Config` (pre-skip 120); the last packet is shortened to the end of the input and carries `SideData::Gapless`.

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

`register_decoders` and `register_encoders` install this crate for `CodecId::Opus` on a `DecodeFactory` and an `EncodeFactory`. The desktop app and the phone register them ahead of the platform codecs.

## Tests

- `cargo test -p cutlass-opus` runs the unit tests, the decoder and encoder campaigns, and, when ffmpeg with libopus is installed, the encoder oracle and the interop tests.
- `CUTLASS_OPUS_VECTORS=<opus_newvectors folder> CUTLASS_OPUS_COMPARE=<opus_compare> cargo test --release -p cutlass-opus --lib decode::` checks every final range of the twelve RFC 8251 vectors and runs `opus_compare` on each output.
- `CUTLASS_LOCAL_FILES` names the folder of real recordings the oracle and interop tests use; without it they use synthetic sound.
- `cargo bench -p cutlass-opus --bench against_ffmpeg` times decode and encode beside ffmpeg.

## Not built

SILK and hybrid encoding, encoding at rates other than 48 kHz, DTX, in-band FEC encoding, multistream encoding, and the ambisonic mapping families 2 and 3.
