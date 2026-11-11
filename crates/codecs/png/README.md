# cutlass-png

A PNG decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). Inflate, deflate, the filters and CRC-32 are all in this crate; it depends only on `cutlass-decode` and `cutlass-encode`. The desktop app registers both sides for `CodecId::Png`, after the platform codecs, so on macOS and Windows the system PNG codec is asked first and this crate is the one used on Linux. The window icon, the `cutlass-stickers` crate, and saving the frame at the playhead as a PNG call this crate directly.

## What it supports

One packet is one whole PNG file and gives one picture. The picture keeps what the file declared: samples are not scaled, palette indices stay indices, and alpha is straight.

| Color type | Bit depths |
| --- | --- |
| Gray | 1, 2, 4, 8, 16 |
| RGB | 8, 16 |
| Indexed, with `PLTE` | 1, 2, 4, 8 |
| Gray with alpha | 8, 16 |
| RGBA | 8, 16 |

- Filter method 0 with filter types 0 to 4, non-interlaced and Adam7, on both sides.
- `IDAT` split over any number of chunks, empty ones included. The decoder reads stored, fixed and dynamic deflate blocks and checks the Adler-32 and every chunk CRC.
- `tRNS` is kept as a color key or a palette alpha table and is not applied to the samples.
- `gAMA`, `cHRM`, `sRGB`, `iCCP`, `sBIT`, `bKGD`, `pHYs`, `tEXt`, `zTXt` and `iTXt` are read into the picture and written back when set. None of them changes a sample.
- Any other ancillary chunk is skipped. That includes `cICP`, `mDCV`, `cLLI`, `eXIf`, `hIST`, `sPLT`, `tIME` and the animation chunks `acTL`, `fcTL` and `fdAT`: an animated PNG decodes to its default still picture only.

The encoder picks, for each row, the filter with the smallest sum of absolute byte values, writes the zlib stream as one deflate block (fixed or dynamic Huffman, whichever is smaller), and splits `IDAT` every 8192 bytes. It writes Adam7 when `CodecParams.extra` is `[1]`; an empty `extra` or `[0]` is non-interlaced. It adds no chunk the picture does not carry.

What is refused:

| Input | Error |
| --- | --- |
| Compression method, filter method or interlace method the format does not define (filter method 64 included) | `Unsupported` |
| `extra` other than empty, `[0]` or `[1]` on the encoder | `Unsupported` (`png interlace`) |
| Bad signature, bad CRC, an unknown critical chunk, chunks out of order, a missing or misplaced `PLTE`, `tRNS` on a picture with alpha, a bad filter byte, bytes after `IEND` | `Malformed` |
| A file that ends early | `Truncated` |
| A size past the limits below | `Limit` |
| An empty packet, an audio `push`, a picture whose fields disagree | `InvalidApi` |

## Using it

`PngDecoder::open` and `PngEncoder::open` take a `CodecParams` whose codec is `CodecId::Png`; only `extra` is read. `register_decoders` and `register_encoders` add the same two to a `DecodeFactory` and an `EncodeFactory`.

- `Decoder::decode_picture` takes the file as one packet and returns one `StillOutput`. Its `image` is a `StillImage`: `width`, `height`, `color`, `depth`, and `samples` as `u16`, row by row from the top, plus the palette, transparency and metadata fields.
- `Encoder::push_picture` takes a `StillImage` and returns the file as one packet.
- `Encoder::push_frame` takes one tightly packed RGBA 8-bit CPU frame and returns the file stamped with that frame's time. A platform buffer or another pixel format is `Unsupported`.
- `drain` returns nothing on both sides. Neither side keeps state between pictures.

```rust
use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{Encoder, PictureInput};
use cutlass_png::{PngDecoder, PngEncoder};

let params = CodecParams::video(CodecId::Png, 0, 0);
let mut decoder = PngDecoder::open(&params)?;
let pictures = decoder.decode_picture(&Packet::video(file_bytes, 0, 0, 0, true))?;
let image = &pictures[0].image;

let mut encoder = PngEncoder::open(&params)?;
let packets = encoder.push_picture(&PictureInput { image: image.clone() })?;
let png = &packets[0].data;
```

`examples/roundtrip.rs` decodes a small PNG that FFmpeg writes and prints `match` when the samples are the bytes FFmpeg was given.

## How it is tested

`cargo test -p cutlass-png` runs:

- Hand-built files with stored zlib blocks and hand-computed filters: gray at 1, 8 and 16 bits, RGBA, 8-bit indexed color with `tRNS`, the Sub, Up, Average and Paeth filters, Adam7 placement at 1, 2, 4 and 8 bits, metadata, one-byte `IDAT` chunks, and the animation chunks.
- Malformed files: bad CRC, truncation, bad signature, filter type 5, filter method 64, a zero edge, an edge of 20 000, trailing bytes, a chunk between two `IDAT` chunks, an unknown critical chunk, and palette and `tRNS` placement.
- Round trips through the encoder: gray, RGB, gray with alpha and RGBA at every depth they allow, with and without Adam7; 2-bit indexed color; and 24 small random pictures. 48 random buffers must all fail.
- FFmpeg, both ways, exact samples: it decodes this encoder's RGB, RGBA, gray with alpha, 16-bit gray, 16-bit RGB and 8-bit indexed files, and this decoder reads its RGB (default and `-pred mixed`), RGBA, gray with alpha, 8-bit indexed and 1-bit gray files and its 16-bit gray, gray with alpha, RGB and RGBA files.
- ImageMagick: this decoder reads its 2-bit and 4-bit gray files and its Adam7 gray file, and it reads this encoder's Adam7 gray file, exact samples.
- Python's `zlib` against the inflater and deflater in `zlib.rs`.

The pictures in the FFmpeg and ImageMagick tests are a few pixels across.

The FFmpeg and ImageMagick tests look the tools up on `PATH` and pass without comparing anything when a tool is missing; set `PNG_REQUIRE_ORACLES` to make that a failure. The Python test fails when `python3` is missing. One more test compares a larger picture with FFmpeg and is skipped when that sample file is not on the machine.

Two tests are ignored by default: a release timing check on a 1920x1080 frame, and `decoder_fuzz_campaign`, which feeds random buffers and one-byte mutations of a small file to the decoder for `PNG_FUZZ_SECS` seconds (30 minutes when unset). It is not coverage-guided.

In a separate measurement, a 1920x1080 RGB PNG written by FFmpeg decoded to the same bytes as FFmpeg's own decoder.

Not covered: the ISO/IEC 15948 conformance suite has not been run. No file from another writer tests indexed color at 1, 2 or 4 bits; 2 and 4 bits have this crate's own round trip only, and 1-bit indexed color has no test.

`cargo bench -p cutlass-png` (`benches/against_ffmpeg.rs`) times encode and decode beside the `ffmpeg` program after checking the pixels agree, and prints a skip line when FFmpeg is not installed.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, png.** Input: one picture of 1920×1080 (2.6 MB), written by FFmpeg with `-c:v png -pix_fmt rgb24`, opened and decoded 200 times.

|  | Cutlass | FFmpeg (`png`) |
| --- | ---: | ---: |
| Wall time | 10.31 s | 6.33 s |
| Speed | 19 pictures/s | 32 pictures/s |
| CPU time | 10.29 s | 6.31 s |
| Peak memory | 31 MB | 27 MB |
| Threads allowed | 1 | 1 |

Cutlass is **1.6× slower** than FFmpeg here. Both decoders produced exactly the same samples. The Rust decoder returns `rgb24` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale.

Encoding was not measured: the tool encodes moving pictures and sound only.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Longest edge | 16 384 pixels |
| Pixels in one picture | 16 777 216 |
| One chunk | 16 MiB |
| All `IDAT` data together | 128 MiB |
| One text or ICC profile after inflating | 1 MiB |
| Palette | 256 entries |

- The whole file must be in one packet. A short packet is `Truncated`; there is no incremental decoding.
- Every sample is stored as a `u16`, so an 8-bit picture takes two bytes per sample in memory.
- Animated PNG is not played and not written.
- Color is not converted: gamma, chromaticity, the sRGB intent and the ICC profile are carried, never applied. Alpha is not premultiplied and low bit depths are not widened.
- The encoder has no compression level setting.
- An `iTXt` chunk with no language tag and no translated keyword is written back as `tEXt` or `zTXt`.
- The encoder checks its cancel flag while filtering rows, not while compressing.

## Specifications

- W3C Portable Network Graphics (PNG) Specification, Third Edition (the static-image rules match ISO/IEC 15948:2004)
- RFC 1950, ZLIB Compressed Data Format
- RFC 1951, DEFLATE Compressed Data Format
