# cutlass-jpeg

A JPEG still decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). It decodes baseline, extended sequential and progressive Huffman JPEG to 8-bit gray or RGB, and encodes baseline JPEG. It also exposes a planar core (component planes in, component planes out) that `cutlass-mjpeg` builds Motion JPEG on.

The desktop app registers it for `CodecId::Jpeg` after the platform codecs. On macOS (ImageIO) and Windows (Windows Imaging Component) the platform opener is asked first, whatever the hardware switches in Settings say, and this crate runs when that opener declines. On Linux it is the JPEG codec. On every platform `cutlass-probe` reads EXIF orientation with `read_jpeg_info`, and `cutlass-effects` decodes its bundled preview pictures with `JpegDecoder` directly.

## What it supports

| | Decode | Encode |
| --- | --- | --- |
| Coding | Baseline SOF0, extended sequential SOF1, progressive SOF2 with successive approximation; Huffman only | Baseline SOF0 |
| Precision | 8 bits | 8 bits |
| Components | 1 (gray) or 3 | Gray, or YCbCr from RGB |
| Sampling | Factors 1 to 4 per component, so 4:4:4, 4:2:2, 4:2:0, 4:4:0 and others | `JpegEncoder`: gray 4:4:4, RGB 4:2:0. `Coefficients`: also 4:4:4 and 4:2:2 |
| Restart intervals | Yes; a marker other than the next `RSTn` is malformed | Only through `WriteSettings::restart_interval` |
| Tables | 8-bit and 16-bit DQT, DHT ids 0 to 3 (0 and 1 in a baseline scan) | Annex K quantizers scaled by quality, Annex K.3 Huffman tables, written in every image |
| Markers | APP0 `JFIF` and `AVI1`, APP1 EXIF orientation, APP14 Adobe, COM; other APPn are skipped | APP0 `JFIF` |

Three components are RGB when the Adobe transform byte is 0 or the component ids are `R`, `G`, `B`. Otherwise they are JFIF YCbCr and are converted with the ITU-T T.871 factors in 16.16 fixed point. The transforms are the libjpeg `islow` integer DCT pair. Chroma at half width, half height or both is upsampled with libjpeg's triangle filter; other ratios repeat the sample.

`JpegDecoder::decode_picture` returns one `StillImage`, `ColorType::Gray` or `ColorType::Rgb`, 8-bit samples stored as `u16`, turned upright by the EXIF orientation when APP1 carries one. `JpegInfo` and `Image` keep the stored width and height, and `Image` does not apply the orientation. A broken EXIF block is ignored. Bytes after the end-of-image marker are ignored.

Encoder quality is 1 to 100 on the Independent JPEG Group scale. `CodecParams::extra` empty means 75; one byte in `1..=100` is that quality. `CodecParams::bitrate` is not read.

Refused, with the `what` text of the error:

| Input | Error |
| --- | --- |
| Arithmetic coding (SOF9 to SOF11, SOF13 to SOF15, DAC) | `Unsupported` `arithmetic coding` |
| Lossless SOF3 | `Unsupported` `lossless jpeg` |
| Differential SOF5 to SOF7 | `Unsupported` `differential jpeg` |
| JPEG-LS, hierarchical (DHP), marker `0xC8` | `Unsupported` `jpeg-ls`, `hierarchical jpeg`, `reserved jpeg` |
| DNL, or a frame with zero width or height | `Unsupported` `dnl` |
| Precision other than 8 | `Unsupported` `jpeg precision` |
| Four components; any count other than 1 or 3 | `Unsupported` `cmyk`; `component count` |
| A scan whose Huffman or quantization table was never defined | `Malformed` |
| Short input | `Truncated` or `Malformed`; never a request for more data |
| Encoder input with alpha, indexed color, or a depth other than 8 | `Unsupported` `jpeg alpha`, `jpeg indexed`, `jpeg precision` |
| Encoder `extra` that is not empty or one byte in `1..=100` | `Unsupported` `jpeg quality` |

A failed picture leaves the decoder usable for the next one.

## Using it

- `JpegDecoder` implements `cutlass_decode::Decoder`. Call `decode_picture`; `push` is `InvalidApi`.
- `JpegEncoder` implements `cutlass_encode::Encoder`. Call `push_picture`; it returns one key-frame packet. `push_frame` is `Unsupported` and audio `push` is `InvalidApi`.
- `read_jpeg_info` reads the frame header and stops at the first scan.
- `register_decoders` and `register_encoders` add the two to a factory.
- Planar core: `decode_image` decodes one image to an `Image` of component planes and carries a `Tables` value from one image to the next. `read_tables` reads a table-only stream. `Image::full_planes`, `to_rgb` and `to_rgba` upsample. `Planes::from_rgb` makes YCbCr planes, and `Coefficients` transforms them once (`transform`), counts scan bits at any quality (`scan_bits`), and writes an image (`write`).

```rust
use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Packet, PixelFormat, StillImage};
use cutlass_jpeg::{JpegDecoder, read_jpeg_info};

fn decode(file: Vec<u8>) -> Result<StillImage, CodecError> {
    let info = read_jpeg_info(&file)?; // width, height, components, progressive, orientation
    assert!(info.width > 0);
    let params = CodecParams::with_format(CodecId::Jpeg, 0, 0, PixelFormat::None);
    let mut decoder = JpegDecoder::open(&params)?;
    let mut pictures = decoder.decode_picture(&Packet::audio(file, 0, 0, 0))?;
    Ok(pictures.remove(0).image)
}
```

## How it is tested

`cargo test -p cutlass-jpeg` runs the tests in `src/tests.rs`:

- Hand-built streams with exact expected samples: a zero block, DC and AC coefficients, SOF1, 4:2:0 MCU padding, progressive AC, successive-approximation DC and AC, and a restart interval.
- All eight EXIF orientations, a broken EXIF block, every refusal in the table above, hostile headers (65535 by 65535, 4097 markers, a zero-length segment, a scan before the frame), and the factory and encoder calls.
- Encoder round trips through this decoder, and the reciprocal quantizer against plain division for every step from 1 to 255.
- With `cjpeg` and `djpeg` on `PATH`: a 67 by 45 picture in ten `cjpeg` layouts (four samplings, gray, restart, progressive, optimized tables, quality 100 and 5) must decode to exactly the bytes of `djpeg -dct int`.
- With `ffmpeg` on `PATH`: FFmpeg must decode this encoder's quality-90 gradient with a mean absolute error of at most 20, and an FFmpeg-made JPEG must decode to the right size. FFmpeg is a separate program started by the tests; the crate does not link it.

A test whose tool is missing returns early and passes. Setting `JPEG_REQUIRE_ORACLES` makes those three tests fail instead. The CI workflow installs none of the tools. Two more tests read footage from a folder outside the repository and return silently when it is absent, even with `JPEG_REQUIRE_ORACLES` set.

Two tests are ignored by default: `release_full_hd_budget` (a timing budget for one 1920 by 1080 frame) and `decoder_fuzz_campaign` (`JPEG_FUZZ_SECS` seconds, 30 minutes when unset). The campaign feeds random buffers and one-byte changes of an 8 by 8 gray baseline image. It is not coverage-guided and has no progressive or color seed.

`benches/against_ffmpeg.rs` and `examples/roundtrip.rs` compare pixels with FFmpeg on one 16 by 16 flat gray frame.

Not run: official JPEG conformance streams.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, jpeg.** Input: one picture of 1920×1080 (0.4 MB), written by FFmpeg with `-c:v mjpeg -q:v 3 -pix_fmt yuvj420p`, opened and decoded 200 times.

|  | Cutlass | FFmpeg (`mjpeg`) |
| --- | ---: | ---: |
| Wall time | 2.23 s | 0.88 s |
| Speed | 90 pictures/s | 227 pictures/s |
| CPU time | 2.22 s | 0.88 s |
| Peak memory | 27 MB | 23 MB |
| Threads allowed | 1 | 1 |

Cutlass is **2.5× slower** than FFmpeg here. The two outputs are not bit-identical: 53.87 dB PSNR between them, largest single difference 21. The Rust decoder returns `rgb24` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale.

Encoding was not measured: the tool encodes moving pictures and sound only.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Limits, in `src/limits.rs` and `src/image.rs`:

| Limit | Value | Error past it |
| --- | --- | --- |
| Longest edge | 16 384 pixels | `Limit` `jpeg size` |
| Pixels in one picture | 16 777 216 | `Limit` `jpeg size` |
| Input bytes | 64 MiB | `Limit` `jpeg bytes` |
| Marker segments in one image | 4 096 | `Limit` `jpeg markers` |
| Blocks in one MCU, baseline | 10 | `Malformed` `mcu` |
| Successive-approximation low bit | 13 | `Malformed` `progressive scan` |

The size check runs on the frame header, before any plane is allocated, and planes are reserved with `try_reserve_exact`. A progressive picture also keeps one `i32` per coefficient until the end of the image.

Gaps:

- The pixel limit refuses pictures larger than 4096 by 4096 in area, which includes common camera sizes such as 6000 by 4000.
- No arithmetic, lossless, differential, hierarchical, JPEG-LS, 12-bit, or CMYK decoding.
- The encoder writes no progressive scans, no SOF1, no optimized Huffman tables, and no EXIF.
- ICC profiles and embedded thumbnails are not read.
- A JPEG image with no Huffman tables fails here. `cutlass-mjpeg` supplies the default tables for Motion JPEG through `Tables::standard_huffman`.
- Decoded stills are stored one `u16` per 8-bit sample.

## Specifications

- ITU-T T.81 | ISO/IEC 10918-1: markers, Huffman coding, sequential and progressive scans, restart intervals, abbreviated table streams (B.5), component sizes (A.1.1).
- ITU-T T.871 (JFIF): YCbCr conversion.
- The Independent JPEG Group's quality scale and `islow` integer DCT, which `djpeg -dct int` uses.
- EXIF orientation tag `0x0112` in the APP1 TIFF block.
