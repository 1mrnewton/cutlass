# cutlass-still

`cutlass-still` reads and writes a JPEG, PNG, WebP, or GIF file that is not inside a movie. It demuxes the file into packets, one per picture, and muxes packets back into a file. It does not decode pixels: Huffman, inflate, and LZW are left to the codec crates. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens these files with `StillDemux`, and the export crate writes GIF and animated WebP files with `StillMux`.

## What it supports

| Format | Packets | Clock |
| --- | --- | --- |
| JPEG | One packet: the file bytes | None. Timescale 0, no timestamp |
| PNG | One packet: the file bytes. An APNG (`acTL`) is not split | None |
| Still WebP (`VP8 `, `VP8L`, or `VP8X` without the animation flag) | One packet: the file bytes | None |
| Animated WebP | One packet per `ANMF` chunk, in file order | `WEBP_TIMESCALE`, 1000 ticks per second |
| GIF87a and GIF89a | One packet per image, in file order | `GIF_TIMESCALE`, 100 ticks per second, also for a one-image GIF |

For GIF and animated WebP the first packet starts at byte 0, so it carries the header. The trailer, or the tail of the RIFF form, stays on the last packet. Joining the packets gives the file back. Only the first frame is a keyframe. Bytes after a JPEG's EOI or a PNG's `IEND` stay in the one packet.

Delays are kept as stored, with two defaults. A GIF frame with no delay or a delay of 0 gets `GIF_DEFAULT_DELAY` (10 ticks, 100 ms). An `ANMF` duration of 0 gets `WEBP_DEFAULT_DELAY_MS` (100 ticks, 100 ms). Any stored value above 0 is kept. The packet bytes are never rewritten.

The stream carries the picture size (`FrameSize`): the first SOF segment of a JPEG, the logical screen of a GIF. The extradata is empty. `loop_count()` returns the Netscape or `ANIM` loop count: `None` when the file stores none, `Some(0)` for repeat forever. The packets are one pass either way. `frame(index)` returns the disposal method and, for WebP, the blend bit of a frame.

`StillMux` takes one stream of codec JPEG, PNG, WebP, or GIF and joins its packets. `set_loop_count` writes the loop count into the Netscape block or the `ANIM` chunk. At `finish` the result is parsed again by the demuxer; if it does not open, the mux fails and `bytes()` is `None`.

What is refused:

- Any other signature: `OpenError::Unrecognized`. `probe` refuses a movie header (`ftyp`, EBML, RIFF `AVI ` or `WAVE`) within the first 12 bytes.
- A file above 32 MiB, a JPEG segment, PNG chunk, RIFF chunk, or GIF sub-block whose size runs past the remaining bytes, more than 4096 frames, or WebP chunks nested deeper than the children of `ANMF`: `OpenError::HostileSize`.
- A cut-off signature or body: `OpenError::Truncated`.
- A PNG chunk with a wrong CRC-32: `OpenError::Source`.
- A failed open returns only the error. No stream list and no packets are published.
- In the muxer: a second stream (`MuxError::TooManyTracks`), a codec that is not one of the four (`MuxError::UnknownCodec`), a total above 32 MiB or 4096 packets (`MuxError::HostileSize`), and a track duration that does not match the packets (`MuxError::DurationMismatch`).

## Using it

- `probe` and `StillProbe` decide from the first `PROBE_BYTES` (12). A short prefix of a known signature asks for more bytes.
- `StillDemux::open` takes a `cutlass_container::Input`. `still_limits()` gives the limits to build it with. `StillDemux` implements `cutlass_container::Demux`.
- `kind()` is `StillKind::Jpeg`, `Png`, `Webp`, or `Gif`. `animated()`, `loop_count()`, `media_end()`, `frame_count()`, and `frame()` describe the sequence.
- `StillMux::new()` implements `cutlass_container::Mux`. It writes into memory; `bytes()` is the finished file.

```rust
use cutlass_container::{Demux, Input};
use cutlass_still::{StillDemux, still_limits};

let file = std::fs::File::open(path)?;
let length = file.metadata()?.len();
let mut demux = StillDemux::open(Input::random(file, length, still_limits()))?;
let size = demux.streams()[0].params().size();
while let Some(packet) = demux.read_packet()? {
    let _bytes = packet.bytes();
}
```

`seek` answers from an index built at open and reads nothing. On a forward-only input it returns `SeekError::ForwardOnly`.

`cargo run --release -p cutlass-still --example open` asks FFmpeg for one 32×32 PNG, opens it, and prints `32x32`. It fails with `ffmpeg is not installed` when FFmpeg is missing.

## How it is tested

```sh
cargo test -p cutlass-still
```

- Unit tests in `src/tests.rs` build files by hand. They cover one packet per still picture, trailing bytes, APNG kept whole, GIF and animated WebP splitting and delays, the two default delays, loop counts, the probe, truncation, hostile sizes, a bad PNG CRC, the frame limit, seek, forward input, cancel, and the muxer's round trips and refusals.
- `tests/external/oracle.rs` makes a JPEG, PNG, WebP, and a two-frame GIF with ImageMagick and compares packet positions, sizes, and GIF delays with `ffprobe`. It prints `skipped` and passes when `ffprobe` or `magick` is not on the `PATH`. FFmpeg does not split an animated WebP, so that case is compared with libwebp's `webpinfo` on a file written by Python's PIL; it is skipped when `python3` or `webpinfo` is missing.
- `open_scans_a_large_jpeg_and_seek_does_not_read_it_again` opens an 8 MiB JPEG and checks that seek reads zero bytes. In a release build it also fails if open takes 100 ms or seek takes 5 ms.
- `demux_and_mux_campaign` is a fuzz loop over probe, open, read, seek, and mux. It is `#[ignore]`d and runs for `STILL_FUZZ_SECS` seconds (default 1800) with `cargo test --release -p cutlass-still --test external -- --ignored`.
- One test opens a PNG and refuses movies from a folder of real files kept outside the repository. It prints `skipped` and passes when the folder is absent.
- `cargo bench -p cutlass-still` compares each sample's packet bytes with `ffmpeg -c copy` before it times anything. The samples are one-picture files only.

## Speed against FFmpeg

Not measured. A still file is one packet, or one packet per picture for a GIF or an animated WebP, so there is no packet stream to time against libavformat. The codec crates time the decoding.

## Limits and gaps

| Limit | Value |
| --- | --- |
| File, and so one packet (`MAX_FILE_BYTES`) | 32 MiB |
| Frames in a GIF or animated WebP (`MAX_FRAMES`) | 4096 |
| WebP chunk nesting (`MAX_CHUNK_DEPTH`) | 2 |
| Probe window | 12 bytes |

The demuxer reads the whole file into memory at open and keeps it. Each packet read copies its bytes out of that buffer. The muxer also holds every packet in memory. A picture file above 32 MiB does not open at all.

Gaps, also listed in the `UNSUPPORTED` constant:

- APNG frames are not split. The file is one PNG packet.
- HEIC and AVIF are not this crate's signatures. They are ISO base media files.
- No pixel decode, Huffman, inflate, or LZW.
- A still picture gets no invented frame rate. Its timestamp stays missing.
- A loop count never repeats packets.
- A GIF plain-text extension is skipped. It is not a packet.
- The muxer writes one whole file. It has no fragmented output and does not write to a stream as packets arrive.
- No fixed-seed fuzz run is part of the normal test run; the fuzz loop only runs when asked.

## Specifications

- ITU-T T.81, for the JPEG marker layout.
- W3C PNG specification, for the signature, chunks, and CRC-32.
- WebP container specification, for `VP8X`, `ANIM`, and `ANMF`.
- GIF87a and GIF89a.
