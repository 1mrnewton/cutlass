# cutlass-flac

`cutlass-flac` decodes and encodes FLAC frames (RFC 9639). It is safe Rust: the crate has `#![forbid(unsafe_code)]` and depends only on `cutlass-decode` and `cutlass-encode`. One packet is one FLAC frame. The `fLaC` marker, the metadata blocks, frame splitting, and seeking belong to `cutlass-flac-container`.

The desktop app registers the decoder and the encoder, ahead of the platform codecs, so a FLAC file decodes and encodes the same on every platform.

## What it supports

| | Decoder | Encoder |
| --- | --- | --- |
| Subframes | Constant, verbatim, fixed orders 0 to 4, LPC orders 1 to 32, wasted bits | The same kinds, the cheapest chosen per subframe |
| Residual | Rice with 4-bit and 5-bit parameters, escaped partitions, partition orders 0 to 15 | The same, searched inside the range `Settings` gives |
| Channels | 1 to 8: independent, left/side, side/right, mid/side | 1 to 8; stereo is independent, estimated, or exhaustive (`Stereo`) |
| Bit depth | 4 to 32 | 4 to 32 |
| Sample rate | Every frame header code, or the STREAMINFO rate | 1 Hz to 1,048,575 Hz |
| Blocking | Fixed and variable | Fixed only, block size 16 to 65535 |
| Integrity | CRC-8, CRC-16, and the STREAMINFO MD5 (`md5_check`) | Writes all three |
| Levels | n/a | 0 to 8 (`Settings::level`), default 5 |

Through the `Decoder` and `Encoder` traits the sound crosses as mix PCM:

- Decode to `SampleFormat::F32` (the default) divides by `2^(bits-1) - 1`. Depths up to 24 bits survive the trip to `f32` and back exactly; deeper samples are rounded.
- Decode to `SampleFormat::I16` is exact for depths up to 16, shifted up to 16 bits. A deeper stream is refused with `Unsupported`.
- Encode takes `I16` for a 16-bit stream, or `F32` for any depth. The depth is the STREAMINFO's when `extra` holds one, else 16 for `I16` and 24 for `F32`.
- A channel layout is reordered into the RFC 9639 order on the way in. A layout that lacks one of the RFC's channels for its count is refused.

What is refused, and how:

- A damaged frame is `CodecError::Malformed` or `Truncated`: a bad CRC, reserved codes, a set padding bit, wasted bits that leave no sample bits, partitions that do not divide the block, a predictor longer than the block, LPC precision 15, a negative LPC shift, a residual outside 32 bits, a sample outside the depth, nonzero frame padding, trailing bytes, or a header that disagrees with STREAMINFO.
- A frame whose rate, channel count, or depth differs from the stream's is `Unsupported`.
- An empty packet is `InvalidApi`. A packet over 16 MiB (`MAX_FRAME_BYTES`) is `Limit`.
- Encoder input outside the depth is `OutOfRange`, a non-finite float is `NonFinite`, and a sample count that does not divide by the channel count is `UnevenChannels`. Float input is not clipped.
- A failed decode leaves the decoder as it was. Refused encoder input adds nothing to the stream.

## Using it

`FlacDecoder` and `FlacEncoder` work on interleaved `i32` samples at the stream's depth. `register_decoders` and `register_encoders` add them to a `DecodeFactory` and an `EncodeFactory` under `CodecId::Flac`.

```rust
use cutlass_flac::{FlacDecoder, FlacEncoder, Md5Check, Settings};

let settings = Settings::level(5)?;
let mut encoder = FlacEncoder::new(44_100, 2, 16, settings)?;
let mut frames = encoder.encode(&interleaved)?;
frames.extend(encoder.finish()?);
let info = encoder.stream_info(); // sample count, frame sizes, and MD5

let mut decoder = FlacDecoder::new(Some(info));
for frame in &frames {
    let decoded = decoder.decode(&frame.data)?; // decoded.samples: Vec<i32>
}
assert_eq!(decoder.md5_check(), Md5Check::Match);
```

Other public items:

- `Settings` holds the block size, the highest LPC order, the partition order range, the stereo search, and the apodization windows (`Apodization`). `FlacEncoder::open_with` takes engine parameters and explicit settings.
- `StreamInfo` parses and writes the 34-byte STREAMINFO body. `CodecParams.extra` may be empty, that body, or `fLaC` followed by a STREAMINFO block.
- `FrameHeader` parses and writes a frame header. `crc` and `md5` are the checksums. `layout` gives the RFC channel order for a channel count.
- Through the `Encoder` trait, the first packet carries the opening STREAMINFO as `SideData::Config`. The last packet from `drain` carries the final one, with the sample count, frame sizes, and MD5, for a muxer to write back.
- `md5_check` reports `Match` or `Mismatch` only after every sample was decoded in order from the first. Otherwise it is `Incomplete`, or `Unknown` when there is no signature.

## How it is tested

Unit tests in `src/decode/tests.rs` build frames by hand and damage them: every truncation of a frame is refused, every single flipped bit is refused or decodes without a panic, and the faults in the list above each have a test. `src/encode/tests.rs` round-trips every level, depths 4 to 32, 1 to 8 channels, noise, silence, extremes, wasted bits, odd and short blocks, every stereo mode, layout reordering, reset, and cancel. `f32_round_trips_every_24_bit_sample` checks all 2^24 values.

`tests/external` compares against the `flac` and `ffmpeg` command-line tools:

- `libflac_streams_decode_bit_exactly`: 29 streams written by `flac`, across depths, channel counts, rate codes, block sizes, and LPC order 32. The decoded samples must equal the source, `flac -d`, and FFmpeg, and the MD5 must match.
- `a_variable_block_size_stream_decodes_bit_exactly` and `every_channel_assignment_is_exercised` cover variable blocking and the stereo modes.
- `our_streams_pass_flac_and_ffmpeg_bit_exactly` and `extreme_settings_pass_flac_and_ffmpeg`: streams written here must pass `flac -t` and decode to the source in `flac -d` and FFmpeg.
- `compression_against_libflac_is_recorded` prints the encoded size beside `flac` at levels 0, 5, and 8 and fails if level 5 is more than 2% larger than `flac -5`.
- `tests/external/fuzz.rs` runs seeded in-process campaigns: damaged frames into the decoder (only a panic fails), and random settings and signals through the encoder (the round trip must be exact). These are not libFuzzer runs.

```text
cargo test -p cutlass-flac
FLAC_REQUIRE_ORACLES=1 cargo test --release -p cutlass-flac
FLAC_FUZZ_SECS=1800 cargo test --release -p cutlass-flac --test external -- --ignored --nocapture
cargo bench -p cutlass-flac --bench against_flac
```

A test that needs `flac` or `ffmpeg` prints a message and passes without checking anything when the tool is missing. `FLAC_REQUIRE_ORACLES` turns that skip into a failure. The size comparison also reads real recordings from a folder outside the repository and uses only synthetic signals when it is missing. The long campaigns are `#[ignore]` and run only when asked.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, flac.** Input: 30 minutes of stereo sound at 44,100 Hz (140 MB), written by FFmpeg with `-c:a flac`.

|  | Cutlass | FFmpeg (`flac`) |
| --- | ---: | ---: |
| Wall time | 2.96 s | 0.75 s |
| Speed | 611× real time | 2,411× real time |
| CPU time | 2.94 s | 0.74 s |
| Peak memory | 3.9 MB | 12 MB |
| Threads allowed | 1 | 1 |

Cutlass is **3.9× slower** than FFmpeg here. The two outputs are not bit-identical: 158.37 dB PSNR between them, largest single difference 1.19e-07.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- A packet larger than 16 MiB is refused.
- The encoder writes fixed blocking only.
- Levels 6 to 8 use a fixed set of partial and punch-out Tukey windows where the reference encoder uses `subdivide_tukey`.
- The code is single-threaded and has no SIMD.
- Through the traits, 32-bit sound is rounded to `f32`. `FlacDecoder::decode` and `FlacEncoder::encode` are exact at every depth.
- `md5_check` is `Incomplete`, not `Unknown`, for a STREAMINFO that has a signature but no sample count.
- FLAC in MP4 does not open: `StreamInfo::from_extra` does not accept a `dfLa` box body, which is what `cutlass-isobmff` hands over for a `fLaC` sample entry.
- Ogg FLAC does not open: `cutlass-ogg` reports it as an unknown codec.
- The app's export writes 16-bit FLAC: the mix is clipped to 16 bits before it reaches the encoder.

## Specifications

- RFC 9639, Free Lossless Audio Codec (FLAC).
- RFC 1321, the MD5 Message-Digest Algorithm, for the STREAMINFO signature.
