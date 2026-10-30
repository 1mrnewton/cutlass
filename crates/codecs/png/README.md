# cutlass-png

Native safe Rust PNG decoder and encoder. The library has no `unsafe` and does not link libpng, ImageIO, or FFmpeg. FFmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg` is an oracle for tests and for `benches/against_ffmpeg.rs`.

## What this crate is

This is the software PNG codec. `PngDecoder` turns one datastream into one `StillImage`. `PngEncoder` turns one `StillImage` into one datastream. The picture keeps the channels and depth the file declared: grey, RGB, grey with alpha, RGBA, or palette indices, at 1, 2, 4, 8, or 16 bits where that type allows. A 1-bit, 2-bit, or 4-bit sample stays inside `0 .. 2^depth - 1`. A 16-bit sample stays a 16-bit integer. An index stays an index. A `tRNS` chunk stays a color key or a palette alpha table. Gamma, chromaticity, an sRGB intent, and an ICC profile sit on the image; the sample values stay as coded. Alpha is straight.

An edge may be at most 16 384 pixels, and one image may hold at most 16 777 216 pixels. The scope record is [STATUS.md](STATUS.md).

## Specification

The syntax this crate implements is the W3C Portable Network Graphics (PNG) Specification (Third Edition), Recommendation, 24 June 2025: <https://www.w3.org/TR/png-3/>. The static-image rules agree with ISO/IEC 15948:2004. Each IDAT sequence is one zlib stream (RFC 1950) of deflate data (RFC 1951), with a window of at most 32 768 bytes and a checked Adler-32. Chunk CRC-32 is the ISO-3309 polynomial.

`CodecParams.extra` of `[1]` asks the encoder for Adam7. An empty extra, or `[0]`, writes a non-interlaced image. Where another implementation disagrees with the specification, this crate follows the specification. Those consequences are in [STATUS.md](STATUS.md).

## API

Open either side with `CodecId::Png`. `decode_picture` returns `Vec<StillOutput>`. `push_picture` takes a `PictureInput` and returns `Vec<Packet>`.

```rust
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{Encoder, PictureInput};
use cutlass_png::{PngDecoder, PngEncoder};

let params = CodecParams::audio(CodecId::Png, 1, ChannelLayout::mono());
let mut decoder = PngDecoder::open(&params)?;
let pictures = decoder.decode_picture(&Packet::audio(png, 0, 0, 0))?;
let image = &pictures[0].image;

let mut encoder = PngEncoder::open(&params)?;
let packets = encoder.push_picture(&PictureInput {
    image: image.clone(),
})?;
let encoded = &packets[0].data;
```

`StillImage` stores `width`, `height`, `color` (`ColorType`), `depth` (`BitDepth`), and row-major `samples` as `u16` values, top row first. `register_decoders` and `register_encoders` register `CodecId::Png` on the shared factories. An audio `push` and an empty packet are `CodecError::InvalidApi`. A short buffer is `CodecError::Truncated`.

## Support

Verified for a static datastream: the signature, IHDR, IDAT, IEND, and CRC; greyscale at 1, 2, 4, 8, and 16 bits; truecolor at 8 and 16 bits; greyscale and truecolor with alpha at 8 and 16 bits; indexed color at 8 bits with PLTE; filter types 0 through 4; no interlace and Adam7; IDAT split across chunks, including an empty chunk; and the ancillary chunks `gAMA`, `cHRM`, `sRGB`, `iCCP`, `sBIT`, `pHYs`, `bKGD`, `tEXt`, `zTXt`, and `iTXt`, recorded and left unapplied. An unknown ancillary chunk is ignored. An unknown critical chunk is an error.

Unsupported: APNG as animation, filter methods other than 0, color conversion, premultiplied alpha, scaling samples to 8 bits, and the chunks `cICP`, `mDCV`, `cLLI`, `eXIf`, `hIST`, `sPLT`, and `tIME`.

The full matrix, including rows that are only partly evidenced, is [STATUS.md](STATUS.md).

## Accuracy

A lossless match is byte-identical samples in the declared domain. Metadata is checked on its own. The oracles are hand-built stored-zlib images, ffmpeg 8.1.1, and ImageMagick. A round trip through this encoder is extra coverage. On the run below, the bench compared pixels with ffmpeg before it started timing, and that comparison passed. The example prints `match` when the decoded samples are the RGBA bytes ffmpeg was given.

Indexed files at 1, 2, and 4 bits have no independent oracle in the record. The ISO/IEC 15948 conformance suite was not run. The evidence list is in [STATUS.md](STATUS.md).

## Performance

One run, Thu Oct 29 21:03:29 GMT 2026. CPU from `sysctl -n machdep.cpu.brand_string`: Apple M5 Pro. FFmpeg from `ffmpeg -version`: ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers. Command: `cargo bench -p cutlass-png`. An ffmpeg row includes process startup. Cutlass stays in this process. The stdout below is that run.

```text
PNG wall time, release, one thread
reps: 5 after 1 warmup, median of the timed passes
encode out is the PNG size. decode out is the raw sample count, the same number as ffmpeg's rgba bytes.
cutlass decode stores u16 samples. An ffmpeg row includes process start. That start is the version row.
pixels were compared before the timed passes
ffmpeg: /opt/homebrew/bin/ffmpeg

picture                  side     op         median      min      max          out  note
process start            ffmpeg   version   17.9 ms  17.7 ms  19.5 ms            0  ffmpeg version 8.1.1
1920x1080 gradient       cutlass  encode    35.8 ms  35.3 ms  36.0 ms         9652  filter search, dynamic or fixed Huffman
1920x1080 gradient       cutlass  decode    18.6 ms  18.5 ms  19.2 ms      8294400  png written by cutlass
1920x1080 gradient       ffmpeg   encode    49.6 ms  44.0 ms  64.1 ms        59164  default pred paeth, zlib default
1920x1080 gradient       ffmpeg   encode    52.0 ms  51.2 ms  52.3 ms        10548  -pred mixed, zlib default
1920x1080 gradient       ffmpeg   decode    27.8 ms  25.4 ms  29.5 ms      8294400  png written by cutlass
1920x1080 noise          cutlass  encode   228.3 ms 224.9 ms 254.6 ms      8311144  filter search, dynamic or fixed Huffman
1920x1080 noise          cutlass  decode    55.9 ms  54.5 ms  58.0 ms      8294400  png written by cutlass
1920x1080 noise          ffmpeg   encode   128.5 ms 125.1 ms 129.7 ms      8322399  default pred paeth, zlib default
1920x1080 noise          ffmpeg   encode   138.8 ms 137.8 ms 144.0 ms      8322399  -pred mixed, zlib default
1920x1080 noise          ffmpeg   decode    55.1 ms  54.1 ms  56.2 ms      8294400  png written by cutlass
motion-graphics.png      cutlass  decode    33.0 ms  32.0 ms  34.2 ms     18889752  the file on disk
motion-graphics.png      ffmpeg   decode    58.9 ms  56.9 ms  63.0 ms     18889752  the file on disk
motion-graphics.png      cutlass  encode   145.2 ms 136.4 ms 149.4 ms       720638  filter search, dynamic or fixed Huffman
motion-graphics.png      ffmpeg   encode   108.7 ms 106.8 ms 118.2 ms       707538  default pred paeth, zlib default
motion-graphics.png      ffmpeg   encode   133.8 ms 132.8 ms 134.1 ms       702900  -pred mixed, zlib default
```

Earlier timings stay in [STATUS.md](STATUS.md), labeled as that earlier run.

## Example

`examples/roundtrip.rs` has ffmpeg write an 8×8 RGBA PNG, decodes it with `PngDecoder`, and prints the width, height, and `match`.

```text
cargo run --release -p cutlass-png --example roundtrip
```

```text
width 8
height 8
match
```

When ffmpeg is absent, the example prints `skip: ffmpeg is not installed` and exits 0. The bench does the same.

## Gaps

- Independent indexed files at 1, 2, and 4 bits have not been run. Eight-bit indexed color is verified.
- APNG is one static IDAT picture. Animation frames are not decoded.
- `cICP`, `mDCV`, `cLLI`, `eXIf`, `hIST`, `sPLT`, and `tIME` are ignored on decode and omitted on encode.
- Color conversion, premultiplied alpha, and scaling samples into 8 bits belong to the color crate.
- The ISO/IEC 15948 conformance suite was not run.
- Linux and Windows were not run.
