# cutlass-alac

## What this crate is

`cutlass-alac` decodes and encodes Apple Lossless (ALAC). It is native safe Rust, with `unsafe` forbidden. It builds without C, FFmpeg, AudioToolbox, or an ALAC library. The tests and the benchmark shell out to an installed `ffmpeg` and, on macOS, `afconvert` and `afinfo`, and use them as oracles.

`AlacDecoder` and `AlacEncoder` are the software ALAC codec. `register_decoders` and `register_encoders` add them to the decode and encode factories. The desktop app registers them after the platform codecs.

One packet is one ALAC frame. The container frames the packets: `cutlass-isobmff` reads and writes ALAC in MP4 and M4A. This crate does not read MP4 or CAF. The support record is `STATUS.md`.

## Specification

The bitstream is Apple's open-source ALAC reference (`ALACDecoder.cpp`, `ALACEncoder.cpp`, `ag_dec.c`, `ag_enc.c`, `dp_dec.c`, `dp_enc.c`, `matrix_dec.c`, `matrix_enc.c`, `ALACAudioTypes.h`, `ALACMagicCookieDescription.txt`), Apache 2.0, read online and not copied.

- A frame is a series of elements: SCE (one channel), CPE (a channel pair), LFE, DSE and FIL (skipped), and END. CCE and PCE are refused, as in the reference.
- Each element is stored raw (an escape) or compressed. A compressed channel is an adaptive linear predictor of order 0 through 31, then adaptive Golomb coding of the residual. A pair may be mixed into a weighted mid and a side channel. 24 and 32-bit audio keeps its low one or two bytes raw beside the compressed part.
- The magic cookie is the 24-byte `ALACSpecificConfig`, then an optional 24-byte `chan` atom with a channel-layout tag.
- Channels 3 through 8 follow the reference element map and layout tags: C L R; C L R Cs; C L R Ls Rs; C L R Ls Rs LFE; C L R Ls Rs Cs LFE; C Lc Rc L R Ls Rs LFE.

## API

```rust
// Decode: the cookie in any form Cookie::parse reads.
let mut decoder = AlacDecoder::new(&cookie)?;
let frame = decoder.decode_frame(&packet)?; // i32 samples at the stream's depth

// Encode: 16, 20, 24, or 32 bits, 1 through 8 channels.
let config = EncoderConfig::new(44_100, 2, 16);
let mut encoder = AlacEncoder::new(config, &ChannelLayout::stereo())?;
let mut packets = encoder.push_samples(&samples)?;
packets.extend(encoder.drain()?);
let cookie = encoder.cookie().to_bytes();
```

Through the traits, open with `CodecParams` for `CodecId::Alac`:

- Decode: `extra` is the cookie. `Decoder::push` takes one frame and returns one `Output::Audio` in frame channel order (`AlacDecoder::layout`). 16-bit is `SampleFormat::I16`. 20, 24, and 32-bit are `SampleFormat::F32` over `2^(depth - 1)`. A frame that fails leaves the decoder as it was. `drain` is empty: frames are independent.
- Encode: an empty `extra` writes 16-bit from `I16` input and 24-bit from `F32` input, in 4096-sample frames. A cookie in `extra` sets the depth and frame length. The first packet carries the cookie as `SideData::Config`, and the packet `drain` writes carries it again with the largest frame size and average rate filled in. Float input outside ±1 or not finite is refused.

`EncodeMode::Default` is the reference encoder's search. `EncodeMode::Fast` is its fast stereo path; mono and multichannel encode as default.

## Accuracy

Lossless both ways, checked on real audio against two independent implementations, ffmpeg 8.1.1 (libavcodec) and AudioToolbox (`afconvert`):

- This decoder gives the same integers as both, on streams from both encoders, at 16, 20, 24, and 32 bits, for mono, stereo, 5.1, and every count from one to eight channels.
- This encoder's frames decode back to the input in this decoder, ffmpeg, and `afconvert`, at every depth, for one to eight channels, in both modes, and with odd frame lengths.
- On the real stereo (16 and 24-bit) and 5.1 files checked, the default mode writes the same frames as `afconvert -d alac`, byte for byte, and on stereo the fast mode writes the same frames as `afconvert -d alac -q 0`. Mono and other counts are checked for losslessness, not for identical bytes.

## Performance

`cargo bench -p cutlass-alac` on an Apple M5 Pro, 60 s of real stereo audio at 44.1 kHz, median of 3 after one warmup, real time divided by wall time:

| | 16-bit | 24-bit |
| --- | --- | --- |
| Encode, default | 590× | 558× |
| Encode, fast | 611× | 578× |
| Decode | 1000× | 877× |
| ffmpeg encode (process) | 594× | 557× |
| ffmpeg decode (process) | 1936× | 1895× |

The ffmpeg rows include process startup and reading the file. `STATUS.md` has the full output.

## Example

```text
cargo run --release -p cutlass-alac --example roundtrip
```

```text
frames 11
cookie 24 bytes
coded 39035 bytes of 176400
lossless true
```

## Gaps

CAF is not written or read. Through the traits, 32-bit audio comes out as `f32`, which keeps 24 bits of each sample; `AlacDecoder::decode_frame` is exact. The app's media reader and export take mono and stereo only. See `STATUS.md`.
