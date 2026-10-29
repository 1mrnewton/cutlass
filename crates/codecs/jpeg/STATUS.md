# JPEG support record

Checkout: branch `jpeg`. Native safe Rust in `cutlass-jpeg`. No `unsafe`. No libjpeg, no `image` crate, no ImageIO, and no subprocess in the library. The still container and Motion JPEG are other crates and are out of scope.

This file is the scope record. A row stays planned until a test named here has been executed. Timing and fuzz numbers below are from runs on this machine, not estimates.

## Specifications and sources read

Normative:

- ITU-T T.81 (1992) | ISO/IEC 10918-1. Markers, Huffman coding (Figures C.1, C.2, F.12, F.16), sequential and progressive scans (Annex F and Annex G), and the DCT (Annex A).
- ITU-T T.871. JFIF YCbCr reconstruction. The integer factors are `round(factor * 65536)`.

Also read before the Cutlass code, for the algorithm and the edge cases. The source was not copied into this repo and is not linked.

- FFmpeg 8.1.1 `libavcodec/mjpegdec.c`, `mjpegenc.c`, and `jpegtables.c`. The installed ffmpeg is 8.1.1. Progressive Huffman and the SOF marker switch live in `mjpegdec.c`. Arithmetic SOF cases are logged and skipped there. This crate refuses them by name before a scan.
- libjpeg-turbo 3.1.0 sources `src/jdhuff.c`, `jdphuff.c`, `jdmarker.c`, `jidctint.c`, `jfdctint.c`, `jdcolor.c`, `jcparam.c`, and `jchuff.c`. The installed `cjpeg` and `djpeg` are libjpeg-turbo 3.2.0. GStreamer 1.28.3 `gstjpegdec.c` and `gstjpegenc.c` call libjpeg. They do not implement Huffman or the DCT. `gstjpegparser.h` was used for marker values.
- Annex K Huffman and quantization tables are the specification's tables. The same numbers appear in those projects.

Where those projects disagree with the specification, this crate follows the specification. The consequences are in the matrix.

## What this crate is

`JpegDecoder` and `JpegEncoder` speak one JPEG image per packet. A still is one packet and one picture. The `still` container is a separate crate, so a file test hands the file bytes to the codec as that packet.

Baseline SOF0, extended sequential SOF1, and progressive SOF2 decode to 8-bit grey or 8-bit RGB. One component is grey. Three components are RGB when the Adobe APP14 transform byte is 0 or the component identifiers are `R`, `G`, `B`. Otherwise they are JFIF YCbCr and are converted here, because that conversion is part of reconstructing the picture. Samples are `StillImage` values, not a `PixelFormat`.

The encoder writes baseline SOF0 only. Grey is 4:4:4. Color is 4:2:0. Quality 1..=100 scales the Annex K quantizers the way the Independent JPEG Group does. `CodecParams.extra` empty is quality 75. One byte in `1..=100` is that quality. Any other extra is `Unsupported` (`jpeg quality`). Bitrate is not a JPEG mode. A nonzero bitrate does not change the quantizer.

`register_decoders` and `register_encoders` register `CodecId::Jpeg` the same way PNG registers itself. The desktop registers the software opener after the platform opener. On macOS, ImageIO accepts JPEG first. Tests in this crate open the software factory directly. ImageIO is not the proof for these gates.

EXIF orientation, when the APP1 TIFF block is readable, is stored on `JpegInfo` and is not applied. The coded width and height stay the SOF values. A broken EXIF block is ignored. The picture still decodes.

## Resource limits

Declared before implementation:

| Limit | Value |
| --- | --- |
| Edge | 16 384 pixels |
| Pixels | 16 777 216 |
| Bitstream | 64 MiB |
| Markers | 4 096 |
| Successive-approximation low bit | 13 |
| Blocks in one MCU | 10 for baseline SOF0, 64 otherwise |
| Sampling factors | 1..=4 |
| Huffman table ids | 0..=3, and 0 or 1 in a baseline scan |

A hostile size, including a 65535×65535 frame, returns `Limit` before the coefficient planes exist. Coefficient storage uses `try_reserve_exact`.

## Gates

Lossy comparison checks size and format. Analytic blocks also check samples. There is no encoder byte-identity requirement.

| Gate | Budget |
| --- | --- |
| Independent decode | A cjpeg file and an ffmpeg mjpeg file match width, height, color, and sample count. Not produced by this encoder |
| Independent encode | ffmpeg decodes this encoder. Size and format, plus mean absolute error at most 20 per channel on a smooth quality-90 gradient |
| Analytic fixtures | Hand-built Huffman blocks, including progressive and restarts, exact samples |
| Self round trip | Extra coverage, not the proof. Flat quality-100 images stay exact |
| Decode throughput | 1920×1080 4:2:0, one thread, release, under 250 ms on this machine |
| Encode throughput | The same frame, under 500 ms |
| Memory | One coefficient plane per component, one reconstructed plane per component, and one output buffer. No growth across images. A failed picture does not poison the next one |
| Fuzz | At least 30 minutes on the decoder entry, random bytes plus bit flips of a valid image, recorded seed and findings |
| Workspace | `cargo fmt --all -- --check`, `cargo lint`, `cargo test --workspace` |

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Baseline SOF0, 8-bit, 1 or 3 components | Grey or RGB | Grey 4:4:4, color 4:2:0 | Hand-built zero, DC, and AC blocks. cjpeg and ffmpeg files. This encoder opened by ffmpeg | Verified |
| Extended sequential SOF1, 8-bit Huffman | Same scan as baseline, including more than two Huffman ids | Not emitted | Marker `0xC1` on the zero block | Verified |
| Progressive SOF2 Huffman, including successive approximation | DC and AC first scans and refinement | Not emitted | Hand-built AC scan, DC point transform, and AC correction. cjpeg `-progressive` | Verified |
| Restart intervals | Reset DC predictors and the EOB run | Not emitted | Hand-built DRI of 1. cjpeg `-restart 1` | Verified on decode |
| Frame size | Width and height from SOF. `read_jpeg_info` stops at the first SOS | Taken from the still | Header-only file returns the size and does not decode. Orientation 6 does not swap the axes | Verified |
| EXIF orientation | Exposed on `JpegInfo`, not applied. Bad EXIF ignored | Not emitted | APP1 orientation 6, and a broken APP1 | Verified |
| DQT 8-bit and 16-bit, DHT, DRI, APP0–15, COM | Accepted. A zero quantizer is malformed | 8-bit Annex K tables and standard Huffman tables. APP0 JFIF | Hand-built tables. Encoder output opens in ffmpeg | Verified |
| Arithmetic SOF9–11, SOF13–15, and DAC | Refused, `what` is `arithmetic coding` | Not emitted | Each marker, including through `read_jpeg_info` | Verified, refused |
| Lossless, differential, JPEG-LS, hierarchical, DNL, reserved `0xC8` | Refused by those names | Not emitted | One marker each | Verified, refused |
| 12-bit and other precision, CMYK, other component counts | Refused: `jpeg precision`, `cmyk`, `component count` | 8-bit grey and RGB only | SOF bodies | Verified, refused |
| Alpha, indexed, 16-bit stills | Not a JPEG mode | Refused: `jpeg alpha`, `jpeg indexed`, `jpeg precision` | Encoder call | Verified, refused |
| Hostile and truncated input | Error, no panic. The next picture on the same decoder still decodes | Cancel returns `Cancelled` and `reset` clears it | Empty, short, length 0, huge DQT, SOS before SOF, sampling 0, 65535², 4097 markers, truncated entropy | Verified |
| Quality 1..=100 | n/a | IJG scale. Quality 10 is smaller than quality 90 on a noisy picture | Encoder tests | Verified |
| Factory and export selection | `DecodeFactory` for `CodecId::Jpeg` | `EncodeFactory` for `CodecId::Jpeg`. Desktop registers it after the platform opener | Crate factory test and `apps/desktop` | Verified |
| Motion JPEG, the still container, ImageIO | Not this crate | Not this crate | Outside the issues | Unsupported |
| Fancy upsampling, ICC application, thumbnails, bitrate control | Box upsampling. ICC and thumbnails are not applied. Bitrate does not select a quantizer | 4:2:0 box average | Recorded so a caller does not expect them | Unsupported |

## Interoperability consequences

- GStreamer `jpegdec` expands through libjpeg. Its buffers are not an independent coded-sample oracle. `djpeg` is the same libjpeg-turbo family as `cjpeg`. ffmpeg's mjpeg decoder is the other implementation used here.
- ImageIO accepts JPEG on macOS and converts to RGBA, including orientation. That decoder is not this crate and does not satisfy these gates.
- Trailing bytes after a complete EOI are ignored. A second image in the same buffer is not a second frame.
- An AC scan before any DC coefficient for that component is malformed. A repeated band is malformed. A progressive image that ends after the DC scan is legal: the missing AC coefficients stay zero.
- The encoder pads the last entropy byte with ones and stuffs `0xFF` as `FF 00`. Restart markers are not written.
- Color 4:4:4 and 4:2:2 files decode. The encoder does not emit them. It emits 4:2:0 for RGB.

## Lifecycle

One push is one complete image. A short buffer is `Truncated`, not `NeedMoreData`. The container owns partial reads. A failed decode drops that picture and leaves the decoder usable. `drain` is empty, including a second drain. `reset` clears the encoder cancel flag. `push` of an empty packet is `InvalidApi`. Audio `push` is `InvalidApi`. There is no inter-frame state.

## Readiness

Decode and encode of one JPEG still are in `cutlass-jpeg`. Arithmetic coding is refused by name. This is not a release, and it is not Motion JPEG.

### Implementation

`JpegDecoder::decode_picture` and `JpegEncoder::push_picture` take one JPEG image per packet. Huffman, the DCT, and JFIF color live in this crate. `register_decoders` and `register_encoders` register `CodecId::Jpeg`. `apps/desktop` registers them after the platform openers.

### Rules

Followed `.cursor/skills/ship-codec/SKILL.md` and `references/video-image.md`, plus the codec rules under `.cursor/rules`. No missing skill file.

### Correctness

- Analytic streams, not produced by this encoder: mid-grey, white DC, one horizontal coefficient, 4:2:0 MCU padding, SOF1, progressive AC, successive DC, successive AC correction, and a restart.
- cjpeg (libjpeg-turbo 3.2.0) baseline, `-progressive`, and `-restart 1` match width, height, color, and sample count.
- ffmpeg 8.1.1 (`/opt/homebrew/bin/ffmpeg`) mjpeg matches size and sample count. The same ffmpeg decodes this encoder's quality-90 gradient with mean absolute error at most 20.
- `../local-files` has no `.jpg`. When the folder and ffmpeg are present, the test extracts one frame from the smallest video and decodes it. The test returns without failing when the folder is absent.

### Performance

Release, one thread, rustc 1.99.1, Apple M5 Pro. Workload: 1920×1080 RGB encoded at quality 75 (4:2:0) and decoded back. Two runs: encode 19 ms then 12 ms, decode 22 ms then 17 ms, 97 959 bytes. Both are inside the 500 ms and 250 ms budgets. The slower run is the number to keep. This is not a memory-profile number. The process peak was not sampled separately from the test harness.

### Fuzz

Release, one thread, rustc 1.99.1, Apple M5 Pro. `decoder_fuzz_campaign` ran for 1800 seconds and exited 0. Seed `0x4a504500ffd8ffd9`. 1 251 795 929 inputs: random buffers up to 4096 bytes, and bit flips of a valid 8×8 baseline image. No crash and no hang. The campaign is not coverage-guided. Findings: none. No regression was added because nothing failed.

### Not run

- Official JPEG conformance bitstreams beyond the analytic, cjpeg, and ffmpeg files above.
- Linux and Windows.
- `cargo deny` in this session. CI runs it; this session ran `cargo fmt --all -- --check`, `cargo lint`, and `cargo test --workspace`.
- A separate peak-RSS sample. The timing numbers are the test's own elapsed time, not a profiler.
