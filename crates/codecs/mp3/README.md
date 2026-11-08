# cutlass-mp3

## What this crate is

`cutlass-mp3` is a native safe Rust MPEG audio Layer III decoder and encoder: MPEG-1, MPEG-2 LSF and MPEG-2.5, 8 kHz to 48 kHz, mono and stereo. The crate forbids `unsafe` and links no MP3 library. ffmpeg, mpg123, LAME and afconvert are test oracles only.

A packet is one whole frame, header included, as `cutlass-mp3-container` reads and writes it. The container crate handles ID3, the Xing/Info and LAME tag, and the `iTunSMPB` comment. The support record, with every measurement, is `STATUS.md`.

## Specification

- ISO/IEC 11172-3:1993 for Layer III and the filterbanks.
- ISO/IEC 13818-3:1998 for the lower sampling frequencies.
- ISO/IEC 11172-4 for the decoder accuracy bounds the tests use.
- MPEG-2.5 is the Fraunhofer extension and follows the MPEG-2 rules.

## API

Float PCM full scale is 1.0. 16-bit PCM is accepted by the encoder.

- `Mp3Decoder::open(&CodecParams)`: `codec` must be `CodecId::Mp3`. A sample rate of 0 and no layout lock onto the first frame. `push` decodes one frame and returns interleaved `f32`. With `SideData::Gapless` on a packet, the encoder delay plus `DECODER_DELAY` (529) is skipped at the start and the padding is held back from the end; `drain` returns the held tail. A frame whose reservoir bytes are gone (after a seek or reset) is skipped, not invented.
- `Mp3Encoder::new(Mp3EncoderConfig { sample_rate, channels, rate })` with `Mp3Rate::Constant(kbps)` or `Mp3Rate::Variable(0..=9)` (V0 best). `Mp3Encoder::open(&CodecParams)` maps `constant_rate` and `bitrate`; bitrate 0 is V2. Every packet carries `SideData::Gapless`, delay `ENCODER_DELAY` (528) and, on the last packet, the padding. Hand that to `Mp3Mux::set_trim` (or the shared `Mux::set_trim`) so the file's LAME tag and `iTunSMPB` comment carry it.
- `register_decoders` and `register_encoders` install the crate on a `DecodeFactory` and `EncodeFactory`. The desktop app and the phone register them ahead of the platform codecs, so an MP3 file decodes the same everywhere.

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

## Support

Decode: every Layer III tool, including intensity stereo (MPEG-1 and LSF positions), mid/side, long, start, short, stop and mixed blocks, `scfsi`, `preflag`, subblock gain, all 32 Huffman table selections, count1 tables A and B, the CRC, the bit reservoir and free format. Layer I and II are `Unsupported`.

Encode: polyphase analysis and MDCT with block switching on attacks, a psychoacoustic model, scalefactor and subblock-gain noise shaping, Huffman table and region choice, mid/side by frame, the bit reservoir, constant bitrate at every legal rate, and V0–V9 variable bitrate. It does not write intensity stereo, mixed blocks, `scfsi`, `preflag` or a CRC.

Export writes sound-alone MP3 from the desktop app at a constant bitrate. MP3 beside a picture is refused.

## Accuracy

The decoder matches ffmpeg 8.1.1 and mpg123 1.33.5 within the ISO/IEC 11172-4 bounds on the LAME fixtures, the local media (including a 30-minute file), and random valid streams that exercise every tool. Where those two decoders disagree with each other, `STATUS.md` names the corner and which one this crate follows.

The encoder's files decode in mpg123 (gapless, within the ISO bounds of this crate's decode), ffmpeg (full length), `lame --decode` (exact input length) and afconvert (sample-aligned through `iTunSMPB`). Against LAME 3.100 on ten 20-second items, at constant 128 kbps this encoder has the lower crude noise-to-mask ratio on six items, is within 0.6 dB on two, and is worse on a tone-plus-noise stereo item (8.6 against 1.6 dB) and a sweep (−9.7 against −11.6 dB); V2 spends 7.7% more bits. Perceptual quality is pending human listening.

## Performance

`cargo bench -p cutlass-mp3 --bench speed` on an Apple M5 Pro, one thread, a 30-minute 44.1 kHz stereo file: decode 597× real time (ffmpeg 1453×), encode 58× at 128 kbps (LAME 139×) and 81× at V2 (LAME 159×).

## Examples

```text
cargo run --release -p cutlass-mp3 --example decode -- in.mp3 out.f32 --gapless
cargo run --release -p cutlass-mp3 --example encode -- in.f32 44100 2 out.mp3 vbr 2
```

On `tests/fixtures/sine_44100_j128_crc.mp3` and then on its output, those print:

```text
rate 44100 channels 2 frames 21 trim Some(Trim { encoder_delay: 576, trailing_padding: 1566 })
frames 21 kbps 43.0 delay 528 padding 1614 speed 99.2x
```

## Tests

`cargo test -p cutlass-mp3` runs the decoder, encoder and random-syntax oracle tests (skipped with a note when ffmpeg, mpg123 or lame is missing; `MP3_REQUIRE_ORACLES=1` makes a missing one fail) and a 2-second in-process fuzz campaign. `--ignored` adds the 30-minute file and the long campaign (`MP3_FUZZ_SECS`).

## Gaps

- ISO/IEC 11172-4 conformance bitstreams were not obtained.
- The encoder does not write intensity stereo or mixed blocks, and its quality on tonal-plus-noise stereo and sweeps is below LAME's.
- ffmpeg plays this encoder's files with the priming, because it reads the delay only from a LAME-named tag.
- Memory was not measured. Linux was cross-linted, not run; Windows was not built.
