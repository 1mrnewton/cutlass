# cutlass-still

Native safe Rust demux and mux for a JPEG, PNG, WebP, or GIF file that is not inside a movie. `#![forbid(unsafe_code)]`. FFmpeg is a comparison tool. It is not a dependency, and this crate does not link it.

## What this crate is

The crate reads the signature, walks the container just far enough to name the pictures, and returns those pictures as packets. A single JPEG, PNG, or still WebP is one packet: the file bytes. A GIF is one packet per image. An animated WebP is one packet per `ANMF` chunk. Writing the packets back is concatenation, with an optional loop count stored as metadata.

Pixel decoding stays in the codec crates. Huffman, inflate, and LZW are not implemented here. Timestamps stay in the file's own clock. A still picture has no clock.

The longer support record is `SUPPORT.md`, next to this file.

## Specification

The walks follow these documents:

- ITU-T T.81. A JPEG starts at SOI (`FF D8`). A segment length includes its own two bytes. The entropy-coded scan after SOS is scanned for the next marker, and `FF 00` stays a stuffed byte. EOI (`FF D9`) ends the scan. The scan is not Huffman-decoded. Dimensions come from the first SOF segment.
- The W3C PNG specification. A chunk is a 32-bit length, a type, that many data bytes, and a CRC-32 over the type and the data. The CRC uses the ISO HDLC polynomial `0xEDB88320`. The image bytes stay compressed.
- The WebP container specification. The file is a RIFF form whose kind is `WEBP`. A still file is a `VP8 ` or `VP8L` bitstream, or a `VP8X` file whose animation flag is clear. An animated file sets that flag, carries an `ANIM` chunk, and stores one frame per `ANMF` chunk. An `ANMF` duration is milliseconds.
- GIF87a and GIF89a. A graphic-control delay is a count of hundredths of a second. The logical screen size is the picture size this crate reports.

## API

Open a readable, seekable file with `StillDemux::open` and an `Input`:

```rust
let demux = StillDemux::open(Input::random(file, length, still_limits()))?;
```

`still_limits` caps one file, and therefore one packet, at 32 MiB, and caps a GIF or animated WebP at 4096 frames. `demux.kind()` is `StillKind::Jpeg`, `Png`, `Webp`, or `Gif`. `Demux::read_packet` returns the next packet. `Demux::seek` uses the index built at open.

`probe` and `StillProbe` look at the first 12 bytes (`PROBE_BYTES`). A finished JPEG, PNG, GIF87a, GIF89a, or RIFF `WEBP` signature is accepted. A short prefix of one of those asks for more bytes. Any other header is refused.

`StillMux::new` writes one stream. `Mux::add_stream`, `Mux::write_packet`, and `Mux::finish` produce the file. `set_loop_count` stores a Netscape loop or an `ANIM` loop. `bytes` is the finished file, and it is absent when finish fails.

A JPEG, PNG, or still WebP packet has a missing timestamp and timescale 0. GIF uses timescale `GIF_TIMESCALE` (100). Animated WebP uses `WEBP_TIMESCALE` (1000). A missing or zero GIF delay becomes `GIF_DEFAULT_DELAY` (10 ticks). A zero `ANMF` duration becomes `WEBP_DEFAULT_DELAY_MS` (100 ticks). A stored delay above zero is kept, including WebP durations 1 through 10. `loop_count` is metadata: `None` means the file stored none, and `Some(0)` means repeat forever. The packet list is one pass either way. The picture size is the `FrameSize` on the stream returned by `streams()`.

## Support

| Format | Open | Packets |
| --- | --- | --- |
| JPEG | Yes | One packet, the file bytes, no clock |
| PNG | Yes | One packet, the file bytes, no clock. An `acTL` sequence is not split |
| Still WebP | Yes | One packet, the file bytes, no clock. Lossy `VP8 ` and lossless `VP8L` |
| Animated WebP | Yes | One packet per `ANMF`, in file order, millisecond delays |
| GIF87a and GIF89a | Yes | One packet per image, in file order, timescale 100 |

Concatenating GIF packets, or animated WebP packets, is the file. The first packet starts at byte 0. The trailer, or the tail of the RIFF form, stays on the last packet. Only the first frame of a sequence is a keyframe. A still picture is a keyframe at tick 0.

`MAX_FILE_BYTES` is 32 MiB. `MAX_FRAMES` is 4096. `MAX_CHUNK_DEPTH` is 2. A declared size past the remaining bytes is `OpenError::HostileSize`, and no demuxer is returned.

## Accuracy

This crate returns packet bytes. It does not return pixels, so the comparison is the packet payload against ffmpeg.

On Thu Oct 29 21:06:25 GMT 2026, `cargo bench -p cutlass-still` compared each sample to `ffmpeg -c copy` before timing. The packets matched:

| Format | Picture | Packet bytes |
| --- | --- | --- |
| JPEG | 32×32 | 230 |
| PNG | 32×32 | 120 |
| WebP | 32×32 | 84 |
| GIF | 32×32 | 911 |

Each of those files is one picture and one packet. The WebP bytes were copied by ffmpeg from a file `cwebp` 1.6.0 wrote. This ffmpeg build has no WebP encoder.

## Performance

The numbers below are the stdout of that same run. Nothing in the table was typed by hand.

Command: `cargo bench -p cutlass-still`

ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers` at `/opt/homebrew/bin/ffmpeg`

CPU: `Apple M5 Pro` (`sysctl -n machdep.cpu.brand_string`)

Date: Thu Oct 29 21:06:25 GMT 2026

One 32×32 picture per format. Cutlass time is `StillDemux::open` plus the packet read in this process. ffmpeg time is `ffmpeg -c copy -f null -` and includes process startup. The figure is the median of 3 after one warmup.

```text
cutlass-still open versus ffmpeg
ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
ffmpeg: /opt/homebrew/bin/ffmpeg
sample: one 32x32 picture
reps: median of 3 after 1 warmup
cutlass times StillDemux::open and the packet read in this process
ffmpeg times -c copy -f null and includes process startup
packet bytes were compared to ffmpeg -c copy before timing
jpeg 32x32 230 bytes match ffmpeg, written by ffmpeg
png 32x32 120 bytes match ffmpeg, written by ffmpeg
webp 32x32 84 bytes match ffmpeg, written by cwebp 1.6.0
gif 32x32 911 bytes match ffmpeg, written by ffmpeg

format   side           median          min          max    bytes
jpeg     cutlass      0.014 ms     0.012 ms     0.020 ms      230
jpeg     ffmpeg      19.543 ms    19.508 ms    19.674 ms      230
png      cutlass      0.011 ms     0.011 ms     0.014 ms      120
png      ffmpeg      17.097 ms    16.819 ms    17.985 ms      120
webp     cutlass      0.013 ms     0.011 ms     0.014 ms       84
webp     ffmpeg      17.123 ms    16.603 ms    17.653 ms       84
gif      cutlass      0.011 ms     0.010 ms     0.014 ms      911
gif      ffmpeg      20.067 ms    18.404 ms    23.261 ms      911
```

The timed GIF is one frame, and the timed WebP is one still picture. Animated WebP and a multi-frame GIF are opened by the code above. This run did not time them, so they have no speed row.

## Example

`examples/open.rs` asks ffmpeg for one 32×32 PNG, opens it with `StillDemux::open`, and prints the picture size.

```text
cargo run --release -p cutlass-still --example open
```

That command printed:

```text
32x32
```

## Gaps

These are refused or left out. None of them has a speed row.

- HEIC and AVIF. The signature is not JPEG, PNG, WebP, or GIF.
- A movie container. The probe stops on an `ftyp`, EBML, AVI, or WAVE header.
- APNG frames. The file opens as one PNG packet. The frames are not split.
- JPEG 2000, and any other bitstream that does not start with SOI, the PNG signature, `GIF87a`, `GIF89a`, or RIFF `WEBP`.
- Pixel decode, Huffman, inflate, and LZW.
- A frame rate invented for a still picture. Timescale 0 and a missing timestamp stay missing.
- GIF plain text. It is skipped, and it is not a packet.
- Repeating packets for a loop. The loop count is metadata.
- Fragmented mux output. Finish writes one file.
