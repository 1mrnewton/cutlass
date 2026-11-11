# cutlass-flac-container

Reader and writer for native FLAC files: the `fLaC` marker, the metadata blocks, and the frames. It splits a file into frames and writes frames back into a file; decoding and encoding the sound is the job of `cutlass-flac`. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens `.flac` files through `FlacDemux`, and sound-only export writes them through `FlacMux`.

## What it supports

One packet is one FLAC frame, from its header to its CRC-16. The stream timescale is the sample rate, and the stream's extradata is the 34-byte STREAMINFO body.

| Part of the file | Reading | Writing |
| --- | --- | --- |
| STREAMINFO | Read; must be the first block and appear once | Written first, then filled in at `finish` with the sample count, block and frame sizes, and the MD5 an encoder handed over |
| SEEKTABLE | Read, at most once; placeholder points dropped | Reserved up front (128 points by default) and filled at `finish`, one point about every 10 seconds |
| Other metadata blocks (comments, pictures, padding, cue sheet, application) | Skipped by their length, not read | Not written |
| Frames, fixed or variable block size | Split where the next valid header starts and the CRC-16 closes the frame | Each packet is checked: header, CRC-16, frame or sample number, duration |
| Timestamps | Frame number times the block size, or the coded sample number | Must continue from the previous frame |
| ID3v1 tag at the end | Left out of the sound data | Not written |

Seeking uses the SEEKTABLE when the file has one. Without one, it bisects the file down to a 256 KiB window. Either way it then walks frames to the one that holds the target sample.

What is refused when reading:

- A file that starts with an ID3v2 tag: `OpenError::UnsupportedBrand`. `probe` refuses it too.
- No `fLaC` marker: `OpenError::Unrecognized`.
- STREAMINFO missing, repeated, not first, or with a wrong length; a second SEEKTABLE; seek points out of order or after a placeholder; metadata block type 127; more than `MAX_METADATA_BLOCKS` blocks; no frame after the metadata: `OpenError::Source`.
- A frame longer than `Limits::max_packet_bytes`: `ReadError::HostileSize`.
- Bytes that are not a frame header where a frame should start: `ReadError::Source`.

What the writer refuses, as `MuxError`: a codec other than FLAC, a second stream, a channel count outside one to eight, a sample rate of zero or 1,048,576 and above, extradata that is not this stream's STREAMINFO, a frame whose CRC-16 fails, a frame that does not match the stream or is not the next one, a change of blocking strategy or bit depth, and a finish whose duration is not the number of samples written.

## Using it

- `FlacDemux::open(reader, length)` opens a `Read + Seek` source; `open_limited` takes caller `Limits`. It implements `cutlass_container::Demux`.
- `FlacDemux::stream_info`, `seek_points` and `first_frame` expose what open read. `damaged_tail` is `true` after a read reached a cut last frame, which is not returned.
- `FlacMux::new(writer)` and `FlacMux::with_seek_points(writer, slots)` take a `Write + Seek`. `FlacMux` implements `cutlass_container::Mux`. `set_config` takes the STREAMINFO an encoder produces at the end, which carries the MD5.
- `probe(header)` accepts the four-byte `fLaC` marker. `FlacFamily` is the same check as a `Probe` type.
- `parse_header`, `FrameHeader`, `StreamInfo`, `SeekPoint`, `crc8` and `crc16` are public for callers that frame FLAC themselves.

```rust
use std::fs::File;
use std::io::BufReader;

use cutlass_container::Demux;
use cutlass_flac_container::FlacDemux;

fn count_samples(path: &std::path::Path) -> Result<i64, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut demux = FlacDemux::open(BufReader::new(file), length)?;
    let mut samples = 0;
    while let Some(packet) = demux.read_packet()? {
        // packet.bytes() is one whole frame; its duration is the block size.
        samples += packet.timestamp().duration().unwrap_or(0);
    }
    Ok(samples)
}
```

## How it is tested

```text
cargo test -p cutlass-flac-container
```

- `src/tests.rs` muxes and demuxes hand-built frames: round trips with fixed and variable block sizes, seeking with and without a SEEKTABLE, every metadata refusal, a cut last frame, cancel, and each writer refusal. One test flips single bits at every byte of a small file and reads and seeks it.
- `tests/external/oracle.rs` demuxes files written by the `flac` tool and by FFmpeg and compares every frame's bytes with `flac --analyze` and every packet's time and size with `ffprobe`, then decodes the packets and checks the MD5. It also checks random seeks, and remuxes each file and has `flac -t`, `metaflac` and `ffprobe` read the result. These tests return without checking anything when `flac`, `ffmpeg`, `ffprobe` or `metaflac` is not installed.
- `tests/external/e2e.rs` runs a recording through demux, decode, encode and mux and has `flac -t` and FFmpeg read the result. It needs private sample files and is skipped when their folder or the tools are absent.
- `tests/external/fuzz.rs` is an in-process campaign over `probe`, `FlacDemux` and `FlacMux`. A short run (2000 seeds or 20 seconds, whichever comes first) is part of the normal suite. The long run is `#[ignore]`d:

```text
FLAC_FUZZ_SECS=1800 cargo test --release -p cutlass-flac-container --test external -- --ignored --nocapture
```

There is no benchmark in this crate.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, flac.** Input: one 140 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 0.29 s | 0.15 s |
| CPU time | 0.28 s | 0.14 s |
| Peak memory | 2.9 MB | 12 MB |
| Packets returned | 17,306 | 17,306 |

Cutlass is **1.9× slower** than FFmpeg here.

Writing files was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- At most 4096 metadata blocks (`MAX_METADATA_BLOCKS`). A SEEKTABLE may hold at most `Limits::max_index_entries` points. One frame may not exceed `Limits::max_packet_bytes`.
- The writer reserves at most 65,535 seek points and remembers at most 16,777,216 frame starts for the table.
- STREAMINFO stores the sample count in 36 bits. A longer stream is written with the count zero, which means unknown.
- A file that starts with an ID3v2 tag does not open.
- Tags, cover pictures and cue sheets are not exposed when reading, and the writer cannot store them.
- Ogg FLAC and FLAC in MP4 are not handled here.
- There is no forward-only open; the reader must seek.
- A frame in the middle of a file whose CRC-16 fails is still returned, cut at the next header that carries the expected number. Finding that cut can read ahead as far as `Limits::max_packet_bytes`. Only a cut last frame is reported, through `damaged_tail`.
- Sound data is taken to end at an ID3v1 tag when the last 128 bytes of the file start with `TAG`.
- A cancelled or failed write leaves the sample count at zero. The frames already written still read back.

## Specifications

- RFC 9639, Free Lossless Audio Codec (FLAC): the file layout, the STREAMINFO and SEEKTABLE blocks, the frame header, and the two CRCs.
