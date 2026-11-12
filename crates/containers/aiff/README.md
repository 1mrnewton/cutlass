# cutlass-aiff

Reader and writer for AIFF and AIFF-C sound files. It finds the `COMM` and `SSND` chunks, hands out the sound as packets with sample-accurate times, and writes linear PCM back into a file; it does not decode samples. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens `.aiff` files through `AiffDemux`. Export does not write AIFF; `AiffMux` is used only by tests.

## What it supports

The stream timescale is the sample rate. `FEATURES` lists the same rows in code.

| Sound data | Reading | Writing |
| --- | --- | --- |
| AIFF, big-endian integer PCM, 16, 24 or 32 bits | One sample frame per packet, codec `CodecId::Pcm` | Written as AIFF |
| AIFF-C `NONE`, `twos`, `in24`, `in32` (big-endian integer) | Same | Not written; big-endian integer goes out as AIFF |
| AIFF-C `sowt` (little-endian integer) | Same | Written as AIFF-C |
| AIFF-C `fl32`, `fl64` (big-endian float) | Same | Written as AIFF-C |
| 8-bit PCM, including `raw ` | Packets are returned with coding byte 6 | Written |
| `alaw`, `ulaw` (also `ALAW`, `ULAW`, which FFmpeg reads and AudioToolbox does not) | One sample frame per packet, `CodecId::Pcm` with coding byte 7 or 8 | No |
| `ima4` | One 34-byte block per channel per packet, 64 sample frames | No |
| `MAC3`, `MAC6` | One block per packet, 6 sample frames | No |
| `GSM ` | One 33-byte block per packet, 160 sample frames | No |

PCM streams carry three bytes of extradata, the layout `cutlass-pcm` reads: coding, endian (1 is big-endian), interleaved (always 0). The compressed kinds are published as `StreamCodec::Unknown` with the fourcc as the tag; this crate does not map them to a `CodecId`.

Chunks other than `COMM` and `SSND` are skipped and counted (`skipped_chunks`). The `SSND` offset field is honored. `COMM` may come after `SSND` when the reader can seek.

What is refused, all as `OpenError`:

- Any other AIFF-C compression (`QDM2`, `QDMC`, `Qclp`, `G722` and so on): `Source`, with the fourcc in the message.
- A sample size that is not 8, 16, 24 or 32 bits, or one that does not match the compression type: `Source`.
- A sample rate that is zero, negative, fractional, not finite, or above `u32::MAX`: `Source`.
- A missing or duplicate `COMM` or `SSND`, zero channels, a sample count that is not a whole number of blocks: `Source`.
- A chunk or form that runs past its parent, more than 64 channels, more than 4096 chunks, a `COMM` over 4096 bytes: `HostileSize`.
- Sound data shorter than `COMM` declares: `Truncated`.

The writer refuses anything but `CodecId::Pcm`, little-endian float, planar layouts, a second stream, a packet that is not exactly one sample frame, and a finish whose duration does not equal the packet count. On a refused finish no bytes are written.

## Using it

- `AiffDemux::open(reader, length)` opens a `Read + Seek` source; `open_limited` takes caller `Limits`.
- `AiffDemux::open_forward(reader)` reads in order without seeking. Seek is then refused with `SeekError::ForwardOnly`, and `SSND` before `COMM` is an error.
- `AiffDemux` implements `cutlass_container::Demux`. It also has `sample_rate`, `channels`, `duration_samples`, `is_aifc` and `skipped_chunks`.
- `AiffMux::new(writer)` implements `cutlass_container::Mux`. `into_inner` returns the writer.
- `probe(header)` accepts `FORM....AIFF` and `FORM....AIFC` from the first 12 bytes. `AiffFamily` is the same check as a `Probe` type.

```rust
use std::fs::File;
use std::io::BufReader;

use cutlass_aiff::AiffDemux;
use cutlass_container::Demux;

fn sample_frames(path: &std::path::Path) -> Result<u64, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut demux = AiffDemux::open(BufReader::new(file), length)?;
    let mut frames = 0;
    while let Some(packet) = demux.read_packet()? {
        // For PCM, packet.bytes() is one sample frame.
        let _ = packet.bytes();
        frames += 1;
    }
    Ok(frames)
}
```

## How it is tested

```text
cargo test -p cutlass-aiff
```

- `src/behavior.rs` builds files by hand and checks packets, timestamps, chunk skipping, the `SSND` offset, seeking, forward reading, each compression's block size, every refusal above, and the writer's round trips. `mutated_bytes_fail_closed` flips every byte of a small file and opens every prefix of it.
- `tests/external/oracle.rs` has FFmpeg write 16-bit big-endian, 16-bit little-endian and 32-bit float AIFF files, reads them here and compares the bytes with the source, then has FFmpeg decode files this crate wrote. `afinfo` is asked for the sample rate when it is installed. The test is skipped when `ffmpeg` or `ffprobe` is not on `PATH`.
- `tests/external/local.rs` opens private sample files and is skipped when their folder is absent.

There is no long fuzz campaign and no benchmark. The compressed kinds (`alaw`, `ulaw`, `ima4`, MACE, GSM) are tested only with hand-built files, not against another reader.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, aiff.** Input: one 53 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 4.16 s | 0.02 s |
| CPU time | 4.15 s | 0.01 s |
| Peak memory | 2.6 MB | 11 MB |
| Packets returned | 13,230,000 | 12,920 |

Cutlass is **208.0× slower** than FFmpeg here.

Writing files was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

- Header limits: 4096 chunks in a form, 64 channels, 4096 bytes of `COMM`. One packet may not exceed `Limits::max_packet_bytes`. There is no limit on the length of the sound.
- `COMM` stores the sample count in 32 bits and the form size in 32 bits, so the writer refuses a file over 4 GiB.
- A PCM packet is one sample frame, and each read allocates one packet. `AiffDemux` does not buffer its reader, so wrap a `File` in a `BufReader`.
- The writer holds every packet in memory and writes the whole file at `finish`.
- Sample sizes that are not a whole number of bytes (12-bit, 20-bit) are refused.
- IMA 4:1, MACE and GSM are split into blocks but have no `CodecId`, and the writer does not write them. The writer does not write A-law or μ-law either.
- Markers, instrument data, names, comments and ID3 chunks are skipped, not exposed, and not written.
- A forward reader stops at the sound and never reads chunks after it. A skipped chunk larger than `Limits::max_header_bytes` is refused on a forward reader.

## Specifications

- Audio Interchange File Format: "AIFF", version 1.3 (Apple Computer, 1989).
- Audio Interchange File Format AIFF-C (Apple Computer, 1991).
