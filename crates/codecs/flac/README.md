# cutlass-flac

## What this crate is

`cutlass-flac` decodes and encodes FLAC frames (RFC 9639). It is native safe Rust, with `unsafe` forbidden. There is no C, no FFmpeg, and no FLAC library underneath. The `flac`, `metaflac`, and `ffmpeg` binaries are used by the tests and the benchmark as oracles only.

`FlacDecoder` and `FlacEncoder` are the codec. `register_decoders` and `register_encoders` add them to the decode and encode factories under `CodecId::Flac`. The file around the frames (`fLaC`, metadata blocks, frame splitting, seeking) is `cutlass-flac-container` in `crates/containers/flac`.

## Specification

RFC 9639, *Free Lossless Audio Codec (FLAC)*, December 2024. Every part of the frame format is read:

- Frame header: fixed and variable blocking, every block-size and sample-rate code, the coded frame or sample number up to 7 bytes, CRC-8.
- Subframes: constant, verbatim, fixed orders 0 to 4, LPC orders 1 to 32, wasted bits.
- Residual: Rice with 4-bit and 5-bit parameters, escaped partitions, partition orders 0 to 15.
- Channels: 1 to 8 independent, left/side, side/right, mid/side.
- Depths 4 to 32 bits, sample rates 1 Hz to 1,048,575 Hz.
- Frame CRC-16 and the STREAMINFO MD5 of the samples.

The encoder writes the same format. Its choices follow the libFLAC presets for levels 0 to 8.

## API

The native calls work on interleaved `i32` samples at the stream's depth.

```rust
let settings = Settings::level(5)?;
let mut encoder = FlacEncoder::new(44_100, 2, 16, settings)?;
let mut frames = encoder.encode(&interleaved)?;
frames.extend(encoder.finish()?);
let info = encoder.stream_info(); // sample count, sizes, and MD5 now known

let mut decoder = FlacDecoder::new(Some(info));
for frame in &frames {
    let decoded = decoder.decode(&frame.data)?; // decoded.samples: Vec<i32>
}
assert_eq!(decoder.md5_check(), Md5Check::Match);
```

Through the factories, `CodecParams.extra` is the 34-byte STREAMINFO body.

- Decode: `SampleFormat::I16` is exact for depths up to 16. `F32` divides by `2^(bits-1) - 1`, so a 24-bit sample survives the trip to `f32` and back.
- Encode: the depth is taken from `extra` when it holds a STREAMINFO, else 16 for `I16` and 24 for `F32`. The first packet carries the opening STREAMINFO as `SideData::Config`. The last packet from `drain` carries the final one with the sample count and MD5, for the muxer to write back.
- A channel layout is reordered into the RFC order. A layout with no RFC order is refused.
- `Settings` exposes the block size, LPC order, partition orders, stereo search, and apodization windows. `Settings::level(n)` gives the preset.

## Accuracy

Lossless both ways. The decoder matches `flac -d` and ffmpeg sample for sample on every stream in the test suite, and every stream this encoder writes passes `flac -t` and decodes identically with `flac -d` and ffmpeg. Details and counts are in `STATUS.md`.

## Size and speed

Encoded size against `flac` 1.5.0 at the same level is between 0.70% smaller and 0.03% larger on every source measured. Level 5 is within 0.01% at worst.

On an Apple M5 Pro, 60 s of 44.1 kHz stereo music, one thread:

| | Ours | `flac` 1.5.0 |
| --- | --- | --- |
| Encode level 0 | 814× real time | 1329× |
| Encode level 5 | 449× | 990× |
| Encode level 8 | 77× | 450× |
| Decode | 978× | 1546× |

The command is `cargo bench -p cutlass-flac --bench against_flac`. The `flac` times include process launch and file I/O.

## Gaps

- No SIMD and no threads: encode takes 1.6 times libFLAC's time at level 0, 2.2 times at level 5, and 5.8 times at level 8; decode takes 1.6 times.
- Levels 6 to 8 use partial and punch-out Tukey windows in place of libFLAC's `subdivide_tukey`. The sizes measured are never more than 0.03% larger than libFLAC's.
- The app exports 16-bit FLAC: the mix is clipped to 16 bits before it is encoded.
