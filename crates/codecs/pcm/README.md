# cutlass-pcm

## What this crate is

`cutlass-pcm` converts raw PCM packets to and from the mix buffer. The packet is samples a container has already framed. The crate is native safe Rust, with `unsafe` forbidden. The crate builds without FFmpeg. The benchmark shells out to an installed `ffmpeg` and uses it as an oracle.

`PcmDecoder` and `PcmEncoder` are the software PCM codec. `register_decoders` and `register_encoders` add them to the decode and encode factories.

## Specification

Integer and float layouts follow WAVE PCM and IEEE 754. `Layout` names the coding, byte order, and plane order. The container owns the RIFF header.

- WAVE format tag `0x0001`: signed 16-bit, packed signed 24-bit, and signed 32-bit two's complement. Positive full scale is the largest positive integer of that width: 32767, 8388607, and 2147483647. Packed 24-bit is three bytes.
- WAVE format tag `0x0003`: IEEE 754 binary32 and binary64. Float full scale is 1.0. Finite values outside ±1 are kept. A non-finite value fails the frame.
- `Endian` is little or big. `Arrangement` is interleaved or planar. Planar bytes are one plane per channel, concatenated in channel order. Decode returns interleaved frames.

The rules come from the IBM and Microsoft *Multimedia Programming Interface and Data Specifications 1.0* (August 1991) for WAVE PCM, Microsoft `WAVEFORMATEX` tags `0x0001` and `0x0003`, and IEEE Std 754-2019.

A length that is not a whole number of frames fails as a truncated PCM frame. `drain` returns no packets and no audio. The decoder has no delay.

## API

Open with `CodecParams` for `CodecId::Pcm`, a sample rate, and a `ChannelLayout`. `Layout::to_extra` stores the coding, endian, and arrangement in `CodecParams.extra`.

```rust
let spec = Layout::new(Coding::S16, Endian::Little, Arrangement::Interleaved);
let mut params = CodecParams::audio(CodecId::Pcm, 48_000, ChannelLayout::stereo());
params.extra = spec.to_extra().to_vec();
params.sample_format = Some(SampleFormat::I16);

let mut decoder = PcmDecoder::open(&params)?;
let outputs = decoder.push(&packet)?;

let mut encoder = PcmEncoder::open(&params)?;
let packets = encoder.push(&AudioInput { pcm })?;
```

`CodecParams::audio` starts `sample_format` at `SampleFormat::F32`. Set `SampleFormat::I16` for s16. Empty `extra` with `I16` means interleaved little-endian s16. Empty `extra` with `F32` means interleaved little-endian f32. When both `extra` and `sample_format` are set, the format must be the coding's output format.

`extra` is empty or exactly three bytes:

| Byte | Values |
| --- | --- |
| 0 coding | 1 s16, 2 s24, 3 s32, 4 f32, 5 f64 |
| 1 endian | 0 little, 1 big |
| 2 arrangement | 0 interleaved, 1 planar |

`Decoder::push` takes one `Packet` and returns one `Output::Audio`. `Encoder::push` of a non-empty buffer returns one `Packet`. An empty buffer returns no packet. `Pcm.samples` is the frame count. The buffer is interleaved. `Decoder::reset` keeps the open layout. `Encoder::reset` zeroes the sample clock. A `SideData::Config` whose bytes differ from the open extra fails that push.

## Support

Each coding below accepts either endian and either arrangement.

| Coding | Decode | Encode |
| --- | --- | --- |
| s16 | `SampleFormat::I16`, the same bits | the same bits |
| s24, three bytes | `SampleFormat::F32`, divide by 8388607 | refused |
| s32 | `SampleFormat::F32`, divide by 2147483647 | refused |
| f32 | `SampleFormat::F32`, the same bits | the same bits |
| f64 | `SampleFormat::F32` with `as f32` | `f64::from` of each f32 sample |

The most negative s24 code stays below -1. The most negative s32 code becomes -1 in f32.

| Limit | Value |
| --- | --- |
| Packet | 16 MiB |
| Decoded or encoded buffer | 32 MiB |
| Sample rate | 1 through 384000 Hz |
| Channels | 1 through 8, from `ChannelLayout` |

8-bit PCM, A-law, mu-law, s64, and 24-bit samples stored in a 32-bit word are refused. An empty packet is refused. A partial trailing frame is refused.

## Accuracy

The benchmark in Performance decoded ffmpeg's s16le stereo payload and compared every i16 sample. The stdout line is `max abs sample delta 0`, so that buffer is bit-exact. The encoder wrote the same bytes. ffmpeg accepted those bytes and read the same s16le payload back. That check ran before the timed passes.

s16 copies the integer bits. f32 copies finite sample bits. f64 narrows with `as f32` on decode and widens with `f64::from` on encode. s24 and s32 decode by the divisions in Support, and the stored value is f32. This run measured the s16le buffer only.

## Performance

One s16le stereo payload. The command was `cargo bench -p cutlass-pcm`.

`date` at the end of that run:

```text
Thu Oct 29 21:06:06 GMT 2026
```

`sysctl -n machdep.cpu.brand_string`:

```text
Apple M5 Pro
```

`ffmpeg -version`, first line, binary `/opt/homebrew/bin/ffmpeg`:

```text
ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
```

Bench stdout:

```text
s16le stereo, 48000 Hz, 48000 frames, 96000 samples, 192000 bytes
max abs sample delta 0
ffmpeg accepted the encoded bytes
median of 3 after 1 warmup
cutlass times bytes in memory
ffmpeg time includes process startup
ffmpeg rows convert pcm_s16le to pcm_s16le
cutlass decode median 0.000036 s min 0.000035 s max 0.000036 s
ffmpeg decode median 0.020225 s min 0.020133 s max 0.020420 s
cutlass encode median 0.000114 s min 0.000114 s max 0.000116 s
ffmpeg encode median 0.018037 s min 0.017674 s max 0.018272 s
```

The four medians are copied from that stdout. Each row is the median of 3 timed runs after one warmup. Min and max are those three runs. Cutlass times `push` on bytes already in memory. Both ffmpeg rows are `pcm_s16le` to `pcm_s16le`, and each includes process startup.

## Example

`examples/roundtrip.rs` decodes four stereo s16le frames, encodes those samples, and decodes the packet. The frames are `0, 0`, `32767, -32768`, `1000, -1000`, and `42, -42`. It prints the decoded rate, channel count, and `Pcm.samples`.

```text
cargo run --release -p cutlass-pcm --example roundtrip
```

```text
rate 48000
channels 2
samples 4
```

`samples 4` is the frame count.

## Gaps

8-bit PCM, A-law, mu-law, s64, and 24-bit samples packed into 32-bit words are refused. s24 and s32 encode are refused. Picture input is refused.

WAV, AIFF, and CAF headers stay in the container crates. Resampling, dither, and channel remix stay outside this crate.

The timed result is one interleaved little-endian s16 stereo buffer on the machine named above.
