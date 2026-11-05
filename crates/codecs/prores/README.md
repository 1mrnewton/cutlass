# cutlass-prores

Native safe Rust Apple ProRes decoder and encoder in `cutlass-prores`, per SMPTE RDD 36:2022. No `unsafe`. No C, no FFmpeg, no VideoToolbox. ffmpeg, ffprobe, VideoToolbox (through ffmpeg) and AVFoundation (through `avconvert`) are oracles in the tests and the bench only, started as separate processes.

## What this crate is

`ProresDecoder` takes one ProRes frame and returns its picture at 10 or 12 bits with any alpha plane. `ProresEncoder` takes a planar picture or an RGBA picture and returns one ProRes frame at Apple's target rate for the profile. One packet is one frame; every ProRes frame is a key frame.

`register_decoders`, `register_video_decoders`, `register_encoders` and `register_video_encoders` register `CodecId::Prores`. The desktop app and `cutlass-ffi` register `register_decoders` and `register_encoders` after the platform codecs; neither registers the video variants yet, so the 4444 alpha path is not reached from the app. Registered after the platform, so on a Mac VideoToolbox is asked first and this crate is the fallback, and elsewhere this crate is the only ProRes codec.

## Specification

The bitstream is SMPTE RDD 36:2022: the frame header, the picture header, the slice table, slices of 1, 2, 4 or 8 macroblocks, the entropy codes for DC and AC coefficients, the scans, the quantization matrices, and the alpha run-length codes. The inverse transform is a separable floating-point DCT that passes the RDD 36 Annex A (IEEE 1180) accuracy test. Output samples are rounded and clamped to `4..=2^b - 5`, as ffmpeg's decoder does.

The six profiles, by sample entry:

| Profile | Code | Chroma | Decoded depth | Apple's 1080p29.97 target |
| --- | --- | --- | --- | --- |
| ProRes 422 Proxy | `apco` | 4:2:2 | 10 | 45 Mb/s |
| ProRes 422 LT | `apcs` | 4:2:2 | 10 | 102 Mb/s |
| ProRes 422 | `apcn` | 4:2:2 | 10 | 147 Mb/s |
| ProRes 422 HQ | `apch` | 4:2:2 | 10 | 220 Mb/s |
| ProRes 4444 | `ap4h` | 4:4:4 + alpha | 12 | 330 Mb/s |
| ProRes 4444 XQ | `ap4x` | 4:4:4 + alpha | 12 | 495 Mb/s |

The targets are the table in Apple's ProRes white paper (April 2022). Other sizes scale Apple's HQ table by pixel count and the profile's ratio to HQ.

## API

`ProresDecoder::open` and `ProresEncoder::open` take a `CodecParams` whose `codec` is `CodecId::Prores`. Any other codec is `Unsupported` (`codec is not ProRes`). `params.extra` is the sample-entry code, the convention this repository's movie demuxer and muxer use for ProRes.

Decoder:

- The code in `extra` sets the output depth: 12 bits for `ap4h` and `ap4x`, 10 for the 4:2:2 codes. Without a code, a 4:4:4 or alpha frame is 12-bit and a 4:2:2 frame is 10-bit. `with_depth(10 | 12)` forces one.
- `decode(&[u8])` returns `DecodedFrame { header, frame }`. The frame is `Yuv422p10`, `Yuv422p12`, `Yuv444p10` or `Yuv444p12` with 16-bit little-endian samples. `frame.alpha` holds the alpha plane at 16 bits when the frame has one. `frame.primaries`, `transfer` and `matrix` come from the frame header.
- Interlaced frames are decoded field by field and woven into one picture.
- `with_threads(n)` gives the decoder its own pool; the default shares the workspace pool. Slices are decoded in bands of 16 rows (32 for interlaced). One worker and many give the same bytes.
- `ProresDecoder` implements `VideoDecoder`. `ProresPictureDecoder` implements `Decoder` and returns 8-bit 4:2:0 for the app's picture path; it drops alpha.

Encoder:

- The code in `extra` picks the profile. Empty is `apcn`. Any other code is `Unsupported` (`ProRes profile`).
- `encode(&VideoFrame)` takes 8-, 10- or 12-bit 4:2:0, 4:2:2 or 4:4:4 and converts it to the profile's chroma: 4:2:0 chroma rows are repeated, 4:4:4 chroma pairs are averaged for a 4:2:2 profile, and narrower chroma is repeated for a 4:4:4 profile. A 4:4:4 profile keeps the frame's alpha; a 4:2:2 profile drops it. The frame's color tags are written; untagged frames use the parameters'.
- `encode_rgba(width, height, &[u8])` converts RGBA to BT.709 limited range.
- The rate control searches for the finest frame quantizer whose frame fits the profile's budget, starting from the last frame's index, then moves slices one index finer in a spread order while the budget allows. Alpha is coded losslessly and is outside the budget. `with_budget_scale(0.05..=4.0)` scales the budget. `params.bitrate` is not read.
- The encoder writes `ctls` as its encoder identifier, version 0 for 4:2:2 and 1 for 4:4:4, Apple's own quantization matrices, 8-macroblock slices, and progressive frames.
- `ProresEncoder` implements `VideoEncoder` and `Encoder`. `Encoder::push_frame` takes RGBA or 8-bit 4:2:0. Audio is `InvalidApi` and platform frames are `Unsupported`.

The decoder checks the cancel flag before each band and the encoder between rate-control passes; both return `Cancelled`, and `reset` clears the flag.

## Accuracy

Decoder, against ffmpeg 8.1.1's decoder and VideoToolbox's (through `ffmpeg -hwaccel videotoolbox`), on 1080p and 2160p streams from three encoders (ffmpeg's `prores_ks`, ffmpeg's `prores_aw`, and VideoToolbox):

- Every Y, Cb and Cr sample is within 1 of ffmpeg's. Alpha is identical.
- Against VideoToolbox, this decoder is off by at most 1 in 0.5–1.5% of samples at 10 bits and 5–7% at 12 bits. ffmpeg's decoder is off by up to 4 in 1.6–3% at 10 bits and by up to 2 in 13–21% at 12 bits. This crate's float transform is closer to Apple's decoder than ffmpeg's integer one.

Encoder, six frames of a 1080p29.97 BT.709 clip, PSNR of Y against the 10-bit source (12-bit for 4444):

| Profile | Target | This crate | prores_ks | VideoToolbox |
| --- | --- | --- | --- | --- |
| Proxy | 45 | 45.0 Mb/s, 40.74 dB | 46.0 Mb/s, 38.34 dB | 44.2 Mb/s, 40.36 dB |
| LT | 102 | 102.0 Mb/s, 49.83 dB | 108.0 Mb/s, 44.98 dB | 98.2 Mb/s, 49.49 dB |
| 422 | 147 | 147.0 Mb/s, 54.22 dB | 153.1 Mb/s, 48.79 dB | 145.9 Mb/s, 54.19 dB |
| HQ | 220 | 220.0 Mb/s, 59.50 dB | 226.1 Mb/s, 54.35 dB | 218.7 Mb/s, 59.51 dB |
| 4444 | 330 | 330.0 Mb/s, 62.88 dB | 313.8 Mb/s, 58.85 dB | 333.9 Mb/s, 63.22 dB |
| 4444 XQ | 495 | 495.0 Mb/s, 68.39 dB | 391.2 Mb/s, 63.40 dB | 530.5 Mb/s, 69.02 dB |

VideoToolbox decodes every stream this encoder writes. Full numbers, chroma and SSIM are in `STATUS.md`.

## Performance

`cargo bench -p cutlass-prores --bench throughput`, Apple M5 Pro (18 cores), rustc 1.99.1, 30 pictures from 6 real frames, frames per second:

| Size | Profile | Encode, all cores | Encode, one | Decode, all cores | Decode, one |
| --- | --- | --- | --- | --- | --- |
| 1080p | HQ | 122.5 | 24.8 | 761.6 | 73.6 |
| 2160p | Proxy | 55.5 | 12.3 | 465.9 | 51.2 |
| 2160p | HQ | 42.3 | 6.8 | 226.6 | 18.3 |
| 2160p | 4444 XQ | 26.1 | 4.0 | 124.6 | 10.2 |

2160p HQ decodes at 7.6 times real time at 29.97 fps on all cores. ffmpeg 8.1.1's decoder ran the same stream at 121.9 fps, startup included. Every row, and the machine load during the run, is in `STATUS.md`.

## Tests

```text
cargo test -p cutlass-prores
cargo test -p cutlass-prores --release --test external
cargo bench -p cutlass-prores --bench throughput
PRORES_FUZZ_SECONDS=1800 cargo test --release -p cutlass-prores --lib fuzz_campaign -- --ignored
```

The external tests and the bench need ffmpeg and ffprobe on `PATH` and the clips in `../local-files` beside the checkout. Without them they print why and pass; `PRORES_REQUIRE_ORACLES=1` turns a missing tool into a failure. The VideoToolbox and AVFoundation checks run only on a Mac.

## Gaps

- The encoder writes progressive frames only. Interlaced frames are decoded, not encoded.
- The export panel offers Proxy, LT, 422 and HQ. 4444 and 4444 XQ are reachable through the API, not the panel.
- The app's import path asks this crate for 8-bit 4:2:0 and drops alpha; the full-depth `VideoDecoder` output is not wired into playback.
- `params.bitrate` is not read; the profile sets the rate.
- Fuzzing is an in-process, time-budgeted mutation campaign, not cargo-fuzz.
