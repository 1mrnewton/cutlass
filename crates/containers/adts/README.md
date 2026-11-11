# cutlass-adts

`cutlass-adts` frames one AAC-LC stream as raw ADTS, the layout of an `.aac` file. It demuxes a file into raw access units with the ADTS header removed, and muxes access units back into ADTS frames. It does not decode AAC. The crate is safe Rust (`#![forbid(unsafe_code)]`). The same crate also holds a LOAS/LATM reader and writer and a small one-track M4A reader and writer.

The desktop app does not open ADTS files. The probe crate uses `AdtsFile` only to recognize a raw AAC file and refuse it with a message, because the stream cannot seek.

## What it supports

ADTS, reading:

| Field | Accepted |
| --- | --- |
| Header kind | MPEG-4 (ID 0) and MPEG-2 (ID 1) |
| Profile | AAC-LC only (2-bit profile value 1) |
| Sample rate | 96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000 Hz; 7350 Hz in an MPEG-4 header only |
| Channel configuration | 1 (mono), 2 (stereo), 3, 4, 5, 6 (5.1), 7 (wide 7.1) |
| Protection | No CRC (7-byte header), or a 16-bit CRC with one raw block |
| Raw blocks per frame | One. Two to four when the header carries each block's position and the header CRC matches |
| Frame length | Up to 8191 bytes, header included |

ADTS stores no timestamps. Each raw block gets a presentation time on the sample-rate clock: the first is 0 and each lasts 1024 samples. When a frame has a CRC, the CRC (and for a single block, the 7 header bytes) is kept on the packet as `SideData::Protection`.

Bytes that are not a frame header are skipped until a header is found, up to 1 MiB.

ADTS, writing: one AAC stream, profile 1, a 7-byte header, buffer fullness `0x7FF`, one raw block per `write_packet`. A packet that carries `SideData::Protection` is written with its CRC. `write_blocks` writes two to four raw blocks in one frame with positions and a header CRC. `set_mpeg2(true)` sets the MPEG-2 ID bit.

What is refused (all as `cutlass_codec::access::CodecError`):

- A profile other than AAC-LC: `Unsupported` (`main`, `ssr`, or `ltp`).
- Channel configuration 0: `Unsupported { what: "program config" }`.
- Sample-rate index 13, 14, or 15: `Malformed`. 7350 Hz in an MPEG-2 header: `Unsupported`.
- Several raw blocks without a CRC, which means without positions: `Unsupported`.
- A header CRC that does not match on a multi-block frame: `Malformed { what: "crc" }`.
- A change of sample rate, channel layout, or MPEG ID inside the stream: `Unsupported { what: "configuration change" }`.
- A partial frame at the end: `Truncated`.
- In the muxer: a second stream, a codec other than AAC, an empty packet, a frame over 8191 bytes, a sample rate outside the table, or a layout that matches no channel configuration from 1 to 7.

LOAS/LATM (`LatmDemuxer`, `LatmMuxer`, `LatmFile`): one program, one layer, `audioMuxVersion` 0, one subframe, `frameLengthType` 0, no other-data, no LATM CRC. The `AudioSpecificConfig` must name AAC-LC, alone or under an SBR or PS object type. Anything else is `Unsupported`. The muxer needs the `AudioSpecificConfig` in `CodecParams::extra`.

M4A (`Mp4Demuxer`, `Mp4Muxer`): one AAC track in an `M4A ` file with an edit list for encoder delay and trailing padding. Video tracks, fragmented files, and a second track are refused. The whole file is held in memory. The app does not use this pair; it reads and writes M4A with `cutlass-isobmff`.

## Using it

`AdtsFile<R: Read>` owns a reader and implements `cutlass_container::Demux`. `LatmFile` is the same for LOAS. `seek` always returns `ReadError::Seek(SeekError::ForwardOnly)`.

```rust
use cutlass_adts::AdtsFile;
use cutlass_container::Demux;

let mut file = AdtsFile::open(reader)?;
let streams = file.streams();
while let Some(packet) = file.read_packet()? {
    let pts = packet.timestamp().presentation();
    let timescale = packet.timescale().ticks_per_second();
    let unit = packet.bytes();
    let _ = (pts, timescale, unit);
}
```

`AdtsDemuxer` is the byte feeder underneath: `push_bytes`, `finish_bytes`, then `next_packet` from the crate's own `Demuxer` trait. `AdtsMuxer` implements the crate's `Muxer` trait and returns the finished bytes.

```rust
use cutlass_adts::{AdtsMuxer, Muxer};
use cutlass_codec::CodecId;
use cutlass_codec::access::{ChannelLayout, CodecParams, Packet};

let mut mux = AdtsMuxer::new();
mux.add_stream(CodecParams::audio(
    CodecId::Aac,
    48_000,
    ChannelLayout::mono(),
))?;
mux.write_packet(&Packet::audio(unit, 0, 1024, 48_000))?;
let adts = mux.finish()?;
```

`cargo run --release -p cutlass-adts --example demux` writes two access units with `AdtsMuxer`, reads them with `AdtsFile`, and prints the frame count and the first timestamp.

## How it is tested

```sh
cargo test -p cutlass-adts
```

- Round trips of raw access units through each of the three framings, including a CRC-protected block, two raw blocks with positions, and a priming edit in M4A.
- Feeding one byte at a time gives the same packets as feeding the whole buffer.
- Refusals: a false sync before the frame is skipped; a truncated tail, unpositioned blocks, 7350 Hz under MPEG-2, a second LATM program, a video track, and an ADTS frame handed to `Mp4Muxer` as a sample are errors.
- Random bytes into each of the three demuxers must not panic.
- Three FFmpeg tests, one per framing. FFmpeg encodes a short AAC file, the crate demuxes and muxes it again, and FFmpeg must decode both files to the same PCM. When `ffmpeg` is not found they print `skipped` and pass, unless `AAC_REQUIRE_ORACLES` is set.
- `thirty_minute_adts_campaign` feeds random bytes for 30 minutes. It is `#[ignore]`d and runs only with `cargo test -p cutlass-adts -- --ignored`.
- `cargo bench -p cutlass-adts` checks frame count, frame sizes, and presentation times of one FFmpeg-written ADTS file against `ffprobe` before it times anything, then does the same for the file `AdtsMuxer` writes back.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, adts.** Input: one 29 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 0.01 s | 0.04 s |
| CPU time | 0.00 s | 0.03 s |
| Peak memory | 3.0 MB | 14 MB |
| Packets returned | 77,875 | 77,875 |

Cutlass is **4.0× faster** than FFmpeg here.

Writing files was not measured. LOAS/LATM was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Limits:

| Limit | Value |
| --- | --- |
| ADTS frame | 8191 bytes |
| Bytes buffered by `AdtsDemuxer` or `LatmDemuxer` | 1 MiB |
| Bytes skipped while looking for a header | 1 MiB |
| File held by `Mp4Demuxer` | 32 MiB |
| Samples in an `Mp4Demuxer` file | 100,000 |

`AdtsFile` reads its input 4096 bytes at a time. `AdtsMuxer`, `LatmMuxer`, and `Mp4Muxer` build the whole output in memory and return it from `finish`.

Gaps:

- No seek. This is why the app refuses raw AAC files.
- There is no `cutlass_container::Mux` implementation for ADTS, LOAS, or the M4A pair. The muxers only implement this crate's own `Muxer` trait.
- A header is accepted on its own. The demuxer does not check that another header follows the frame, so bytes that look like a header inside leading junk can be taken as the first frame.
- ID3 and APE tags are not parsed. They are skipped as junk, within the 1 MiB limit.
- Every ADTS raw block is stamped as 1024 samples, also when the stream uses 960-sample frames.
- The CRC of a single-block frame and the per-block CRCs are stored and not checked. Only the multi-block header CRC is checked.
- Spectral band replication and parametric stereo are not signaled in an ADTS header. They stay inside the access unit for the decoder.
- `ChannelLayout::seven_one()` matches no channel configuration, so the muxer refuses it. Configuration 7 is `ChannelLayout::seven_one_wide()`.
- One stream only.

## Specifications

- ISO/IEC 13818-7, for the ADTS frame and `adts_header_error_check`.
- ISO/IEC 11172-3, for the CRC.
- ISO/IEC 14496-3 subclause 1.7, for LOAS and LATM.
