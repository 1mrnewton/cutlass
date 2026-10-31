# cutlass-jpeg

Native safe Rust JPEG decoder and encoder in `cutlass-jpeg`. No `unsafe`. FFmpeg is not linked. The benchmark and the example look for `/opt/homebrew/bin/ffmpeg`, then for `ffmpeg` on `PATH`, and start that program as a separate process when they compare pixels.

## What this crate is

`JpegDecoder` takes one JPEG packet and returns one 8-bit still. `JpegEncoder` takes one 8-bit gray or RGB still and returns one baseline JPEG packet. `read_jpeg_info` reads the frame header and stops at the first scan. `register_decoders` and `register_encoders` register `CodecId::Jpeg`.

One call is one image. Bytes after the end-of-image marker are ignored, and a second image in the same buffer is not a second picture. EXIF orientation, when the APP1 TIFF block is readable, is stored on `JpegInfo` and is not applied. The width and height stay the frame-header values. A broken EXIF block is ignored and the picture still decodes.

## Specification

The decoder accepts baseline SOF0, extended sequential SOF1, and progressive SOF2 Huffman scans from ITU-T T.81 (ISO/IEC 10918-1), including restart intervals. Three-component YCbCr is converted to RGB with the ITU-T T.871 factors in `color.rs`, in the integer form `round(factor * 65536)`.

The encoder writes baseline SOF0 only. Gray uses sampling 4:4:4. RGB uses 4:2:0, and each chroma sample is the average of a 2×2 block. Quality `1..=100` scales the Annex K quantizers the way the Independent JPEG Group does: below 50 the scale is `5000 / quality`, and otherwise `200 - 2 * quality`. An empty `CodecParams.extra` selects quality 75. A nonzero bitrate does not change the quantizer.

## API

`JpegDecoder::open` and `JpegEncoder::open` take a `CodecParams` whose `codec` is `CodecId::Jpeg`. Any other id is `Unsupported` (`jpeg`).

`extra` chooses the encoder quality. Empty is 75. One byte in `1..=100` is that quality. Any other `extra` is `Unsupported` (`jpeg quality`). The decoder does not read `extra`.

`JpegDecoder` implements `Decoder`. `decode_picture` returns one `StillOutput`. An empty packet is `InvalidApi` (`empty packet`). `push` is `InvalidApi` (`jpeg produces a picture`). `drain` is empty.

`JpegEncoder` implements `Encoder`. `push_picture` returns one keyframe `Packet`. `push` of audio is `InvalidApi` (`jpeg encodes a picture`). `push_frame` is `Unsupported` (`frame`). `drain` is empty. `reset` clears the cancel flag.

`read_jpeg_info` returns `JpegInfo` with `width`, `height`, `components`, `progressive`, and `orientation`.

Decoded and encoded stills are `StillImage` values. One component is `ColorType::Gray`. Three components are `ColorType::Rgb`. Samples are 8-bit values stored as `u16`. Alpha and indexed color are refused by the encoder.

## Support

Decoded:

- Baseline SOF0, extended sequential SOF1, and progressive SOF2, Huffman coded, precision 8.
- One component (gray) or three. Three components are treated as RGB when the Adobe APP14 transform byte is 0 or the component ids are `R`, `G`, `B`. Otherwise they are JFIF YCbCr and converted in this crate.
- Sampling factors 1 through 4, which covers 4:4:4, 4:2:2, and 4:2:0. Reconstruction repeats the coded sample across the upsampled block.
- Restart intervals. A baseline scan may use Huffman table ids 0 and 1. Other scans may use ids 0 through 3.
- 8-bit and 16-bit quantization tables. A zero quantizer is malformed.
- APP and comment markers. APP0 JFIF is recognized. APP1 orientation is recorded and not applied.

Encoded:

- Baseline SOF0 only. Gray is 4:4:4. RGB is 4:2:0. Quality 1 through 100. An APP0 JFIF marker and the standard Huffman tables are written. Restart markers are not written.

Refused, by name:

- Lossless SOF3 (`lossless jpeg`). The encoder does not emit it.
- Arithmetic SOF9 through SOF11, SOF13 through SOF15, and DAC (`arithmetic coding`).
- Differential SOF5 through SOF7 (`differential jpeg`).
- JPEG-LS (`jpeg-ls`).
- Hierarchical (`hierarchical jpeg`).
- DNL (`dnl`), including a frame whose width or height is zero.
- Precision other than 8 (`jpeg precision`).
- Four components (`cmyk`) and any other component count (`component count`).
- Encoder input with alpha (`jpeg alpha`), indexed color (`jpeg indexed`), or a depth other than 8 (`jpeg precision`).

Progressive Huffman is decoded. It is not encoded. Subsampled frames are decoded. The encoder's RGB output is 4:2:0. Its gray output is 4:4:4.

Limits in `limits.rs`: longest edge 16384 pixels, at most 16777216 pixels, bitstream at most 64 MiB, at most 4096 markers, successive-approximation low bit at most 13. A baseline frame may hold at most 10 blocks in one MCU. Any other frame may hold at most 64. A size past those limits returns `Limit` before the coefficient planes are allocated. Coefficient storage uses `try_reserve_exact`.

## Accuracy

Pixels were compared to ffmpeg before the timings in Performance. The sample is one 16×16 gray frame, every sample 128. The benchmark fails if the samples disagree.

ffmpeg wrote a 202-byte baseline JPEG with 4:2:0 sampling. `JpegDecoder` returned 16×16 `Rgb`, 768 samples, equal to ffmpeg's `rgb24` bytes. `JpegEncoder` at quality 75 wrote a 333-byte baseline gray JPEG. `JpegDecoder` returned 16×16 `Gray`, 256 samples, equal to ffmpeg's `gray` bytes and to the source frame.

Those lines from the benchmark stdout:

```text
ffmpeg jpeg: 202 bytes baseline
cutlass jpeg: 333 bytes baseline
decode pixels: match (16x16 Rgb, 768 samples)
encode pixels: match (16x16 Gray, 256 samples)
```

The frame is flat gray, so the 4:2:0 chroma planes are constant. This run does not measure a color gradient, a progressive file, or a photograph.

## Performance

Command:

```text
cargo bench -p cutlass-jpeg
```

Date of that run: `Thu Oct 29 21:10:31 GMT 2026`.

`sysctl -n machdep.cpu.brand_string` printed `Apple M5 Pro`.

`/opt/homebrew/bin/ffmpeg -version` began:

```text
ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
```

Stdout of `benches/against_ffmpeg.rs` from that command:

```text
sample: 16x16 gray 128
ffmpeg: /opt/homebrew/bin/ffmpeg
ffmpeg jpeg: 202 bytes baseline
cutlass jpeg: 333 bytes baseline
decode pixels: match (16x16 Rgb, 768 samples)
encode pixels: match (16x16 Gray, 256 samples)
median of 3 after 1 warmup
cutlass decode: median 2.250 us min 2.167 us max 2.834 us
ffmpeg decode: median 19948.917 us min 19669.375 us max 19988.750 us
cutlass encode: median 4.416 us min 3.542 us max 5.500 us
ffmpeg decode of cutlass jpeg: median 19171.125 us min 18861.375 us max 19198.375 us
```

Each figure is copied from that stdout. The median is the middle of three timed passes after one warmup. A cutlass row stays in this process. An ffmpeg row starts a new process, so its time includes process startup. The frame is 16×16, so these times are not a full-frame video rate.

## Example

```text
cargo run --release -p cutlass-jpeg --example roundtrip
```

The example uses the same 16×16 gray frame. It asks ffmpeg for a baseline JPEG, decodes that with `JpegDecoder`, encodes the frame with `JpegEncoder` at quality 75, and asks ffmpeg to decode the encoded packet. It prints the decoded width, height, and whether both comparisons matched. This run printed:

```text
width=16 height=16 matched=true
```

## Gaps

- Lossless, arithmetic, differential, hierarchical, and JPEG-LS decoding are refused. The encoder writes none of them.
- The encoder does not write progressive scans, extended sequential SOF1, restart markers, 4:2:2, or RGB 4:4:4.
- EXIF orientation is not applied. ICC profiles and thumbnails are not applied.
- Upsampling repeats the coded sample. It does not filter.
- One packet is one still. This crate does not walk a motion JPEG sequence.
- The pixel comparison and the timings are one flat 16×16 gray frame on this machine. Color gradients, progressive files, photographs, Linux, and Windows were not part of this run.
