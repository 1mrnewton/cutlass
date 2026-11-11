# cutlass-gif

A GIF decoder and encoder in safe Rust (`#![forbid(unsafe_code)]`). The decoder paints each image of a GIF onto the screen and returns the whole screen as RGBA; the encoder turns RGBA frames into an animated GIF. The desktop app registers the decoder for `CodecId::Gif` on every platform, and its GIF export opens this encoder.

## What it supports

Decoder:

- GIF87a and GIF89a.
- Global and local color tables.
- LZW image data with a minimum code size of 1 to 11, including the table reset.
- Interlaced images.
- The graphic control extension: the transparent index, and disposal (keep, clear the image's rectangle to transparent, or put back what was under it).
- Images smaller than the screen and offset inside it. Pixels that fall outside the screen are dropped.
- Comment, application and plain text extensions are skipped.

Output is one `Output::Picture` per image: the full screen as RGBA 8-bit, with BT.709 primaries and the sRGB transfer, zero alpha where nothing has been painted, stamped with the packet's `pts` and `timescale`.

Encoder:

- Input is tightly packed RGBA 8-bit CPU frames, all the same size, each edge at most 65 535 pixels.
- Every frame is written as one full-screen image with its own local color table of 2 to 256 colors. The palette is a median cut over the frame's colors, grouped at five bits per channel. There is no global color table.
- The frame's duration becomes the delay in hundredths of a second, never less than 2.
- Alpha is ignored; every frame is opaque.

What is refused:

| Input | Error |
| --- | --- |
| `GifEncoder::open` with another codec id | `Unsupported` (`gif`) |
| `GifDecoder::open` with another codec id | `InvalidApi` (`gif params`) |
| A packet before the one that carries the header | `InvalidApi` (`gif header`) |
| A bad signature, an unknown block, a bad LZW code or code size | `Malformed` |
| A header, color table, extension or image that ends early | `Truncated` |
| A screen or image past the pixel limit, or a screen with a zero edge | `Limit` |
| Encoder frame that is a platform buffer, not RGBA 8-bit, or wider or taller than 65 535 | `Unsupported` |
| Encoder frame with a zero edge, a size that differs from the first frame, or too few bytes; an audio `push` | `InvalidApi` |
| `decode_picture` and `push_picture` | `Unsupported` (the trait defaults) |

Two things are accepted without an error: LZW data that ends before the image is full leaves the remaining pixels as they were, and a color index past the end of the table paints nothing.

## Using it

The public items are `GifDecoder`, `GifEncoder`, `register_decoders`, `register_encoders`, and `GIF_TICKS_PER_SECOND` (100, the unit of a GIF frame delay).

Packets follow the split the `cutlass-still` container makes of a GIF file. The first packet starts at the first byte of the file, so it holds the header and the logical screen. Every packet holds one graphic control extension and one image. The last packet also holds the trailer. Joining the packets in order gives the file.

- `Decoder::push` decodes every image in the packet. The decoder keeps the screen between packets, so packets must arrive in order from the first. After a seek, call `reset` and start again from the first packet. A packet that begins with a GIF header starts a new screen.
- `Encoder::push_frame` holds one frame back, so the first call returns nothing and each later call returns the packet of the frame before. `drain` returns the last packet, with the trailer.
- `reset` on the encoder starts a new file and clears the cancel flag.

```rust
use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{CpuFrame, Encoder, FrameInput, FrameTime, PixelFormat};
use cutlass_gif::{GifDecoder, GifEncoder};

let params = CodecParams::video(CodecId::Gif, 6, 4);
let mut encoder = GifEncoder::open(&params)?;
let mut packets = Vec::new();
for (n, rgba) in frames.into_iter().enumerate() {
    // One frame lasts 1/10 s.
    let time = FrameTime::new(n as i64, 1, 10)?;
    let frame = CpuFrame::new(6, 4, PixelFormat::Rgba8, rgba, time)?;
    packets.extend(encoder.push_frame(&FrameInput::Cpu(frame))?);
}
packets.extend(encoder.drain()?);

let mut decoder = GifDecoder::open(&params)?;
for packet in &packets {
    let screens = decoder.push(&Packet::video(packet.data.clone(), packet.pts, 1, 10, true))?;
}
```

## How it is tested

`cargo test -p cutlass-gif` runs unit tests inside the source files:

- `decode.rs`: frames from this encoder come back in order; hand-written 4x2 files check the transparent index and each disposal pixel by pixel; interlaced rows land in the right place; every cut and every one-byte change of a small file returns without a panic; an oversized screen is `Limit`.
- `encode.rs`: the packets join into one file with the trailer on the last, a short delay is raised to 2, and a frame of another size is refused.
- `lzw.rs`: 20 000 pseudo-random indexes round trip through a table reset, read back by a second LZW reader written for the test.
- `quant.rs`: a few colors are kept exactly, and a gradient stays within a fixed error per channel.

No test in this crate compares against FFmpeg or any other GIF program, and the hand-written files are built with this crate's own LZW writer. There are no sample files, no fuzz campaign and no benchmark here. Tests in `cutlass-export` and in the desktop app use this crate for GIF export and import.

In a separate measurement, a 250-frame 960x540 GIF written by FFmpeg decoded to the same RGBA bytes as FFmpeg's own decoder. The encoder's output has not been checked against another decoder: not verified.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Decode, gif.** Input: 250 pictures in one file of 960×540 (33 MB), written by FFmpeg with `-vf fps=25,scale=960:540`, opened and decoded 10 times.

|  | Cutlass | FFmpeg (`gif`) |
| --- | ---: | ---: |
| Wall time | 5.46 s | 5.20 s |
| Speed | 458 pictures/s | 481 pictures/s |
| CPU time | 5.45 s | 5.18 s |
| Peak memory | 79 MB | 25 MB |
| Threads allowed | 1 | 1 |

Cutlass and FFmpeg take **about the same time** here. Both decoders produced exactly the same samples. The Rust decoder returns `rgba` pictures, so FFmpeg's time includes converting each picture to that layout with libswscale.

Encoding was not measured: the tool encodes moving pictures and sound only.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- The decoder accepts a screen, and each image, of at most 67 108 864 pixels (a quarter of 16 384 x 16 384). The encoder has no pixel limit beyond the 65 535 edge.
- The decoder allocates the whole RGBA screen when it reads the header, before any image data, and each output is a copy of that screen.
- The decoder does not read frame delays or the loop count. Timing comes from the packet, so every image inside one packet gets the same `pts`. The container splits a file into one image per packet.
- The background color index and the pixel aspect ratio are ignored. The screen starts transparent.
- The encoder writes no loop extension; the `cutlass-still` muxer adds it.
- The encoder does no dithering, writes no transparency, and stores every frame whole. It does not store only the part that changed, so files are larger than they need to be.
- Colors are grouped at five bits per channel before the palette is cut, so two colors closer than that share one palette entry.
- The encoder checks its cancel flag at the start of `push_frame` only.
- No single-picture entry points: a still GIF goes through `push` and `push_frame` like an animation.

## Specifications

- GIF89a: Graphics Interchange Format, Version 89a (CompuServe, 1990)
- GIF87a, for the signature older files carry
