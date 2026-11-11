# cutlass-ogg

An Ogg demuxer and muxer in safe Rust (`#![forbid(unsafe_code)]`), with the Opus and Vorbis mappings. It reads pages, checks their checksums, joins packets across pages, and times each packet from the page granule positions. Packets are returned as stored; nothing is decoded here. The desktop app opens Ogg files with `OggDemux`, and sound-only Opus and Vorbis exports are written with `OggMux`.

The crate depends on `cutlass-codec` and `cutlass-container` only. It walks the Vorbis setup header itself to learn each mode's block size.

## What it supports

| Kind | Demux | Mux |
| --- | --- | --- |
| Opus, channel mapping family 0 and 1 | opens, times, trims, seeks | writes |
| Opus, channel mapping family 255 | parsed; not checked against a file from another tool | accepted |
| Vorbis | opens, times, trims, seeks, including a stream that starts before zero | writes |
| Several logical streams in one file | each Opus or Vorbis stream is a track | each stream is a track |
| FLAC, Theora, Speex and other mappings | listed as `Unknown("ogg:flac")` and so on; pages skipped | refused |
| Chained files | first link only; `chained()` is true | one link |
| Damaged pages | a bad checksum resyncs; lost packets become `gaps()` | n/a |

Streams and times:

- An Opus stream has timescale 48000, and its extradata is the `OpusHead`. A Vorbis stream has its sample rate as timescale, and its extradata is the three headers in Xiph lacing.
- A packet's time is the granule position of its first sample. For Opus this is before the pre-skip, so it equals FFmpeg's `pts` plus the pre-skip. Every packet is a keyframe.
- An Opus packet's duration comes from its TOC byte. A Vorbis packet lasts a quarter of its block plus a quarter of the previous packet's block, and the first lasts 0.
- Each Opus stream carries `Trim::new(Some(pre_skip), trailing)` and one `Edit` whose `media_time` is `start + pre_skip`. Each Vorbis stream carries a `Trim` whose leading part is the samples before zero and an `Edit` whose `media_time` is `max(start, 0)`. `trailing` is what the last page trims.
- A page granule that differs from the packet timeline by less than the shortest packet is treated as rounding and counted in `granule_drift()`. A larger step forward is a gap. A larger step back is `OggError::GranuleBackwards`.
- Seek bisects on page granules and lands on the packet that holds the target less the pre-roll: 3840 samples for Opus, half the long block for Vorbis.

What is refused or reduced:

| Case | Result |
| --- | --- |
| Missing or broken `OpusHead` or `OpusTags` | open fails: `BadOpusHead`, `UnsupportedOpusVersion`, `BadOpusTags`, `MissingOpusTags`, `TagsPageNotFinished` |
| Missing or broken Vorbis headers | open fails: `BadVorbisHeader`, `UnsupportedVorbisVersion`, `BadVorbisSetup`, `MissingVorbisHeaders`, `SetupPageNotFinished` |
| An Opus stream shorter than its pre-skip | open fails: `PreSkipPastEnd` |
| More streams than `Limits::max_tracks` | open fails: `TooManyStreams` |
| A packet over the cap | one `ReadError::HostileSize`, then reading continues |
| More than 256 KiB without a valid page | `NoSync` |
| Seek on a forward input | `SeekError::ForwardOnly` |
| Mux: an unknown codec | `MuxError::UnknownCodec` |
| Mux: a known codec other than Opus or Vorbis, packets that overlap or leave a gap, a trimmed packet that is not the last | `MuxError::Rejected` |
| Mux: a timescale that is not 48000 (Opus) or the sample rate (Vorbis) | `MuxError::TimescaleMismatch` |

`OpenError` and `ReadError` have no Ogg variants, so the typed `OggError` is kept in `last_error()`. After a refused call the mux is failed and `finish` returns `FailedMux`.

## Using it

- `OggDemux::open(reader, length)` opens a random-access input. `open_limited` takes explicit `Limits`. `open_forward` reads once from the start: it cannot seek, and the edit duration is 0 because the end is unknown. `OggDemux` implements `cutlass_container::Demux`.
- `opus(track)` and `vorbis(track)` return what open learned: the parsed headers, the first and last sample positions, and the end trim. `tags(track)` returns the `OpusTags`.
- `crc_failures()`, `skipped_bytes()`, `lost_pages()`, `dropped_pieces()`, `ignored_pages()` and `granule_drift()` count what a damaged or unusual file cost. `last_seek()` reports the bytes, probes and pages of the last seek.
- `OggMux::new(writer)` implements `cutlass_container::Mux` and writes pages as packets arrive. `with_first_serial` changes the first serial number from `FIRST_SERIAL`.
- `probe` and `OggFamily` recognize the five bytes `OggS\0`.
- `page` and `crc` are public modules for page parsing, lacing and the checksum.

```rust
use std::fs::File;

use cutlass_container::{Demux, SeekTo};
use cutlass_ogg::OggDemux;

let file = File::open(path)?;
let length = file.metadata()?.len();
let mut demux = OggDemux::open(file, length)?;
let stream = &demux.streams()[0];
let pre_skip = stream.trim().encoder_delay();
while let Some(packet) = demux.read_packet()? {
    let _granule_time = packet.timestamp().presentation();
    let _opus = packet.bytes();
}
demux.seek(SeekTo { track: 0, time: 48_000 * 60, end: None })?;
```

The mux takes the `OpusHead` from the stream's extradata or from `set_config` before the first packet; an MP4 `dOps` body is rewritten as an `OpusHead`. The Vorbis headers come the same way, in Xiph lacing, as three 16-bit lengths each before its packet, or back to back. Packets of a stream must be contiguous in time. Only the last packet may be shorter than its natural duration, and that becomes the end trim on the last page.

```rust
use std::io::Cursor;

use cutlass_codec::{CodecId, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_ogg::OggMux;

let mut mux = OggMux::new(Cursor::new(Vec::new()));
let track = mux.add_stream(NewStream {
    params: StreamParams::try_audio(StreamCodec::Known(CodecId::Opus), 48_000, 2, opus_head)?,
    timescale: Timescale::new(48_000),
    language: None,
})?;
for packet in &packets {
    mux.write_packet(packet)?;
}
mux.finish(&[TrackDuration { track, timescale: Timescale::new(48_000), ticks }])?;
let bytes = mux.into_inner().into_inner();
```

Mux page layout: every stream's first header page comes first (`OpusHead` or the Vorbis identification header, alone, granule 0), then the other headers, then audio. `OpusTags` is written with vendor `cutlass`; the Vorbis comment is kept as given. An audio page is closed once its body holds 4096 bytes (`MUX_PAGE_BYTES`) or its packets span one second, and whenever its 255 lacing values are used.

## How it is tested

Run `cargo test -p cutlass-ogg`.

- Unit tests cover the checksum (against a bitwise CRC), page parsing and lacing, `OpusHead` and `OpusTags`, the Vorbis headers and setup walk, and the packet timeline.
- Behavior tests in `src/behavior.rs` (Opus) and `src/vorbis_behavior.rs` (Vorbis) mux files and read them back: packets across three pages, several streams, chained files, forward input, seeks with pre-roll, damaged and truncated pages, unknown mappings, a Vorbis stream that starts before zero, and a remux of packets read from a checked-in libvorbis file after a seek.
- `tests/fuzz.rs` runs in-process mutation, not cargo-fuzz: 32 demux seeds and 64 mux seeds on every test run. The demux corpus is files this crate muxes plus five checked-in Vorbis files made by libvorbis and FFmpeg (`tests/fixtures/README.md`). Mutants are probed, opened both ways, read to the end and seeked; a finished mux must read back to exactly the accepted packets. The long campaigns are ignored by default, and `OGG_FUZZ_SECS` sets their length:

```text
OGG_FUZZ_SECS=300 cargo test --release -p cutlass-ogg --test fuzz -- --ignored --nocapture
```

- `tests/external` compares against other tools. Files are encoded at test time by FFmpeg and, for Vorbis, by libvorbis through GStreamer. The demuxer is compared with `ffprobe` packet lists and FFmpeg's payload bytes, and the edit length with the samples FFmpeg decodes. Files written by `OggMux` must give FFmpeg the same packets, times and payload as their source, and decode to the same sample count in FFmpeg and GStreamer. These tests need `ffmpeg`, `ffprobe` and a folder of local recordings named by `CUTLASS_LOCAL_FILES`. They print `skipped` and pass when one is missing. GStreamer checks are skipped when `gst-launch-1.0` is not installed.

Without those tools and that folder, no test compares this crate's packet times with another demuxer. Linux and Windows runs of the external tests are not verified.

`tests/external/measure.rs` holds two ignored release measurements of open, read and seek on a thirty-minute file.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, ogg-opus.** Input: one 19 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 0.04 s | 0.04 s |
| CPU time | 0.03 s | 0.04 s |
| Peak memory | 3.4 MB | 13 MB |
| Packets returned | 90,412 | 90,412 |

Cutlass and FFmpeg take **about the same time** here.

**Read every packet, ogg-vorbis.** Input: one 19 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 0.04 s | 0.04 s |
| CPU time | 0.03 s | 0.04 s |
| Peak memory | 3.4 MB | 13 MB |
| Packets returned | 77,875 | 77,875 |

Cutlass and FFmpeg take **about the same time** here.

Writing files was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Limits:

| Limit | Value |
| --- | --- |
| Page | 65,307 bytes (`MAX_PAGE_BYTES`) |
| Resync scan | 256 KiB (`MAX_RESYNC_BYTES`) |
| Opus packet | 61,440 bytes per Opus stream in the mapping, and `Limits::max_packet_bytes` |
| Vorbis packet | `Limits::max_packet_bytes` |
| Logical streams | `Limits::max_tracks` |
| Header packets | `Limits::max_header_bytes`; `OpusTags` or a Vorbis comment above it is cut, a Vorbis setup above it fails open |
| Extradata | `Limits::max_extradata_bytes`; above it a Vorbis comment is replaced by one with only its vendor |
| Comments kept | 65,536 (`MAX_COMMENTS`) |
| Scan for the first and last audio pages at open | 8 MiB each |
| Seek and link-end bisection | 64 probes each |

Gaps:

- A chained file returns its first link only. Later links are not read.
- FLAC, Theora, Speex and other mappings are listed and skipped, never demuxed or written. The app's probe refuses an Ogg file with no Opus or Vorbis stream.
- Opus channel mapping family 255 has no test against a file from another tool.
- The checked-in Vorbis files are mono, stereo and 5.1. Other channel counts are not tested here.
- A packet with a malformed Opus TOC, or a Vorbis packet naming a mode the setup lacks, is returned with duration 0.
- Seek landings report `ordinal` 0 at the start and `u32::MAX` elsewhere, because Ogg has no packet count.
- Opus or Vorbis inside WebM or Matroska is not this crate's job, and no Matroska reader exists yet.

## Specifications

- RFC 3533, The Ogg Encapsulation Format: pages, lacing, the checksum, grouping, chaining.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec: `OpusHead`, `OpusTags`, granule positions, pre-skip, end trimming, seeking with pre-roll.
- RFC 6716 section 3.1: the TOC byte that gives an Opus packet's duration.
- Vorbis I specification, sections 4.2 (the three headers), 4.3.1 and 4.3.8 (the mode number and a packet's duration), and A.2 (Ogg encapsulation, starts before zero, end trimming).
