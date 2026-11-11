# cutlass-mp3-container

Reader and writer for MP3 files (MPEG audio Layer III). It splits a file into frames and writes frames back into a file; it does not decode or encode sound, which is the job of `cutlass-mp3`. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens `.mp3` files through `Mp3Demux`, and sound-only export writes them through `Mp3Mux`.

## What it supports

One packet is one Layer III frame, including its four-byte header. Times are sample counts at the frame's sample rate, which is also the stream timescale.

| File feature | Reading | Writing |
| --- | --- | --- |
| MPEG-1, MPEG-2 and MPEG-2.5 Layer III, mono or stereo | One frame per packet | One `CodecId::Mp3` stream |
| Constant and variable bitrate | Read in order; seek uses an index when the file has one | `Info` frame when every packet has the same bitrate, otherwise `Xing` |
| Xing, Info and VBRI header frames | Used as duration and seek index, never returned as a packet | A leading Xing or Info frame with frame count, byte count and a 100-entry seek table |
| Free format (bitrate index 0) | Frame length learned from the distance to the next free-format header | Packets written as given, file marked `Xing` |
| Encoder delay and trailing padding | Reported as `Trim` on the stream; no samples are dropped | Stored in the LAME field of the leading frame, each at most 4095 |
| ID3v2 at the start (up to four tags), ID3v1 and APEv2 at the end | Skipped | One ID3v2.3 tag with an `iTunSMPB` comment, only when a trim is set before the first packet |

Bytes between frames are skipped while looking for the next frame, up to `MAX_RESYNC_BYTES`.

What is refused:

- Layer I and Layer II: `ContainerError::Unsupported { what: "layer" }`, once a second header confirms the layer.
- A single free-format frame with no second header to fix its length: `Unsupported { what: "free format" }`.
- An ID3v2 tag that declares more than `MAX_ID3_BYTES`: `Limit { what: "id3" }`. The tag is never loaded.
- A file with no frame: `NoFrames`. A frame cut off by the end of the file: `Truncated { what: "frame" }`.
- A positive seek in a file with no Xing or VBRI index and no constant bitrate: `SeekUnavailable`. The read position does not move.
- In the writer: a codec other than MP3, a second stream, a channel count other than one or two, a sample rate MPEG audio does not have, extradata, a packet whose length does not match its header, a packet that is itself a Xing, Info or VBRI frame, and a timestamp that is not the next sample.

## Using it

- `Mp3Demux::open(reader)` takes any `Read + Seek`. It implements this crate's `Demux` trait (`streams`, `duration`, `duration_exact`, `next_packet`, `seek`, `cancel`) and the workspace trait `cutlass_container::Demux`.
- `Mp3Demux::index_kind` says how a seek finds its byte offset: `IndexKind::XingToc`, `VbriToc`, `ConstantBitrate`, `AverageBytes` or `None`.
- `seek` returns a `SeekLanding` with the landing time and `preroll_frames`, the number of earlier frames a decoder needs for the bit reservoir (9 for MPEG-1, 12 for MPEG-2 and MPEG-2.5).
- `Mp3Mux::new(writer)` takes any `Write + Seek`. It implements this crate's `Mux` trait (`add_stream`, `write_packet`, `finish`, `cancel`) and `cutlass_container::Mux`. `set_trim` sets the encoder delay and padding and may be called until `finish`. `into_inner` returns the writer.
- `probe(prefix)` returns `Probe::Matched`, `Probe::NeedMore` or `Probe::Reject`. A prefix that holds a whole ID3v2 tag matches; a bare frame matches only when a second compatible header follows it.

```rust
use std::io::Cursor;
use cutlass_mp3_container::{Demux, Mp3Demux};

fn count_frames(bytes: Vec<u8>) -> Result<u64, cutlass_mp3_container::ContainerError> {
    let mut demux = Mp3Demux::open(Cursor::new(bytes))?;
    let mut frames = 0;
    while let Some(packet) = demux.next_packet()? {
        // packet.data is one whole frame; packet.pts is in samples.
        let _ = packet.pts;
        frames += 1;
    }
    Ok(frames)
}
```

`examples/demux.rs` writes eight frames with `Mp3Mux` and reads them back, or reads a file given as an argument:

```text
cargo run --release -p cutlass-mp3-container --example demux -- file.mp3
```

## How it is tested

```text
cargo test -p cutlass-mp3-container
```

- Unit tests in `src/` build files by hand and check frame sizes against the ISO formula, ID3 and APE skipping, Xing, Info and VBRI parsing, free format, seeking, and every refusal listed above.
- `src/property.rs` feeds random bytes, truncated files and oversized ID3 headers and checks they fail without a short packet.
- `tests/external/oracle.rs` has FFmpeg encode a file (through `libmp3lame`), then compares every packet size and timestamp with `ffprobe`. It does the same for a `lame -V 4` file when `lame` is installed, checks the sample rate and frame count against `mpg123` when it is installed, and has `ffprobe` read a file this crate wrote. The whole test is skipped when `ffmpeg` or `ffprobe` is not on `PATH`.
- `tests/external/measure.rs` checks that an indexed seek in a 4000-frame file reads less than 128 KiB, and in release builds asserts wall-clock budgets for open, a full read and a seek.
- `tests/external/local.rs` reads private sample files and is skipped when their folder is absent.
- `tests/external/fuzz.rs` is an in-process campaign over `probe`, `Mp3Demux` and `Mp3Mux` with random, truncated and bit-flipped input. It is `#[ignore]`d; run it with:

```text
MP3_FUZZ_SECS=60 cargo test --release -p cutlass-mp3-container --test external -- --ignored --nocapture
```

`benches/against_ffmpeg.rs` compares packet sizes and timestamps with `ffprobe` before it times anything, and exits with a skip message when FFmpeg is missing. GStreamer has not been used as a comparison.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, mp3.** Input: one 29 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 0.07 s | 0.03 s |
| CPU time | 0.06 s | 0.02 s |
| Peak memory | 2.8 MB | 13 MB |
| Packets returned | 69,223 | 69,223 |

Cutlass is **2.3× slower** than FFmpeg here.

Writing files was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value |
| --- | --- |
| ID3v2 tag skipped (`MAX_ID3_BYTES`) | 16 MiB |
| APEv2 tag skipped | 16 MiB |
| Search for the next frame (`MAX_RESYNC_BYTES`) | 64 KiB |
| One frame, including free format (`MAX_FRAME_BYTES`) | 4096 bytes |
| Search either side of a seek position (`SEEK_WINDOW_BYTES`) | 4096 bytes |
| VBRI seek entries kept | 4096 |
| Leading ID3v2 tags | 4 |
| Read buffer | 32 KiB |

- Layer I and Layer II files do not open.
- ID3 text and pictures are not exposed, and the writer stores no title, artist or cover.
- Encoder delay and padding are metadata only. They are not subtracted from `pts`; the decoder or caller applies them.
- Without a frame count from Xing, Info or VBRI, `duration` is an estimate from the first frame's bitrate and `duration_exact` is `false`. For a free-format file with no index it is `None`.
- A seek through the Xing table or an average-bytes estimate lands on the frame nearest the estimated byte and numbers it as the requested frame. In a variable-bitrate file the timestamps after such a seek can differ from the true position.
- The writer signs the LAME tag's encoder name as `Cutlass`. FFmpeg (`libavformat/mp3dec.c`) reads the delay and padding there only when the name starts with `LAME`, `Lavf` or `Lavc`, so it plays our priming and padding unless the `iTunSMPB` comment is present, which is written only when a trim is set before the first packet.
- The writer keeps trim values to 12 bits each. A longer delay or padding is refused with `Limit { what: "encoder delay" }`.
- Through `cutlass_container::Mux`, each packet is also kept in memory until `finish`, in addition to being written.

## Specifications

- ISO/IEC 11172-3:1993 (MPEG-1 audio), clause 2.4.1.3, and ISO/IEC 13818-3:1998 (MPEG-2 audio), clause 2.4.2.3, for the frame header and frame length.
- The MPEG-2.5 extension for 8, 11.025 and 12 kHz, which uses the MPEG-2 Layer III frame rules.
- The Xing/Info header with the LAME extension, as LAME writes it, and the Fraunhofer VBRI header at byte 36 of the first frame.
- ID3v2 header (id3.org, version 2.4.0) for the synchsafe size, the 128-byte ID3v1 `TAG`, and the APEv2 footer.
