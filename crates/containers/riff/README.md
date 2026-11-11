# cutlass-riff

`cutlass-riff` reads and writes WAV files, both classic RIFF WAVE and RF64. It demuxes a file into packets of stored sample bytes and muxes packets back into a file. It does not decode or convert the samples. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens WAV files with `WavDemux`, and the export crate writes PCM WAV files with `WavMux`.

## What it supports

A file has one sound track. Timestamps are ticks of the sample rate.

| Kind | Demux | Mux |
| --- | --- | --- |
| RIFF WAVE | Opens | `WavMux::new` |
| RF64 with `ds64` | Sizes of `0xFFFFFFFF` are taken from `ds64` | `WavMux::rf64`, which writes `ds64` for a small file too |
| Integer PCM (tag `0x0001`), 8, 16, 24, 32, or 64 bits | One sample frame per packet | `PCMWAVEFORMAT`, no `fact` |
| Float PCM (tag `0x0003`), 32 or 64 bits | One sample frame per packet | Written with a `fact` chunk |
| `WAVEFORMATEXTENSIBLE` | The channel mask is in the stream extradata at byte 8 | Written when the stream has a channel mask, more than two channels, a valid-bit width that differs from the stored width, or came from an extensible file |
| A-law (`0x0006`), μ-law (`0x0007`) | One sample frame per packet, codec tag `wav:0006` or `wav:0007` | Refused |
| MPEG sound (`0x0050`, `0x0055`) | One MPEG frame per packet. `0x0055` is `CodecId::Mp3` | One frame per packet |
| ADPCM and other block codecs | One `nBlockAlign` block per packet | The packet duration must match the block |
| AAC (`0x00FF`, `0x1600`, `0x706D`, `0x4143`, `0xA106`), AC-3 (`0x2000`), FLAC (`0xF1AC`), E-AC-3 (by GUID) | Mapped to that codec id | Stored |
| Any other format tag | Stream kept, codec tag `wav:XXXX` | Refused |
| Unknown extensible GUID | Stream kept, codec tag `wav-guid:…` | Refused |
| Unknown chunk up to 64 KiB | Kept (`WavDemux::kept`) | Written before `data` (`WavMux::keep_chunk`) |
| Unknown chunk above 64 KiB | Skipped and named (`WavDemux::dropped`) | Not written |
| Odd-length chunk | The pad byte is skipped | The pad byte is written |

When the block align is 1 and the tag is not one of the sample or MPEG tags, the whole `data` chunk is one packet if it fits the packet limit.

For PCM the length is the `data` chunk size divided by the block align, also when `fact` disagrees. For compressed sound, `fact` and the `ds64` sample count must agree.

`PCMWAVEFORMAT` is kept for 16-bit stereo at any sample rate. The sample rate alone does not make the file extensible.

What is refused:

- `RIFX` and `BW64`: `probe` and `open` report `UnsupportedBrand`.
- A chunk size that runs past its parent or the file: `OpenError::HostileSize`. The size is not clipped to the bytes that are present.
- `ds64` values that disagree with the `data` chunk: `OpenError::Source`.
- A truncated header, or `LIST` nesting deeper than 8: an open error and no stream list.
- In the muxer: A-law, μ-law, and unknown codecs; a second track; a packet that is not a whole number of frames or whose duration does not match its bytes; a classic RIFF file above 4 GiB (`MuxError::HostileSize`); a track duration that does not equal the sum of the packets (`MuxError::DurationMismatch`), in which case nothing is written.

## Using it

- `probe` and `WavFamily` decide from the first 12 bytes.
- `WavDemux::open(reader, length)` takes a `Read + Seek` reader and the file length. `open_forward` takes a reader it will not seek; it stops at `data`, and `seek` returns `SeekError::ForwardOnly`.
- `WavDemux` implements `cutlass_container::Demux`. `format()` returns the parsed `fmt ` chunk as a `WavFormat`. `duration_samples()` is the sample count, or `None` for MPEG sound until the frames are read.
- `WavMux` implements `cutlass_container::Mux`.
- `FEATURES` lists what the build reads and writes as data.

```rust
use std::fs::File;

use cutlass_container::Demux;
use cutlass_riff::WavDemux;

let file = File::open(path)?;
let length = file.metadata()?.len();
let mut demux = WavDemux::open(file, length)?;
let stream = &demux.streams()[0];
let rate = stream.params().sample_rate();
let channels = stream.params().channels();
while let Some(packet) = demux.read_packet()? {
    let _ticks = packet.timestamp().presentation();
    let _bytes = packet.bytes();
}
```

```rust
use std::io::Cursor;

use cutlass_codec::{CodecId, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Mux, NewStream, TrackDuration, TrackId};
use cutlass_riff::WavMux;

let params = StreamParams::try_audio(
    StreamCodec::Known(CodecId::Pcm),
    44_100,
    2,
    Vec::new(),
)?;
let mut mux = WavMux::new(Cursor::new(Vec::new()));
mux.add_stream(NewStream {
    params,
    timescale: Timescale::new(44_100),
    language: None,
})?;
mux.write_packet(&packet)?;
mux.finish(&[TrackDuration {
    track: TrackId::new(0),
    timescale: Timescale::new(44_100),
    ticks,
}])?;
let bytes = mux.into_inner().into_inner();
```

A PCM packet handed to the muxer may hold any whole number of sample frames.

Seek on PCM and other constant-block sound is arithmetic: `data` start plus index times block size. It reads no payload. Seek on MPEG sound walks the frame headers from the start of `data`.

`cargo run --release -p cutlass-riff --example demux` writes eight silent stereo frames with `WavMux`, reads them with `WavDemux`, and prints the sample rate, channels, packet count, and sample count.

## How it is tested

```sh
cargo test -p cutlass-riff
```

- Unit tests in `src/behavior.rs` build files in memory. They cover each row of the table above, RF64 and `ds64` disagreement, `fact` handling, kept and dropped chunks, odd padding, forward readers, `data` before `fmt `, seek without reading payload, cancel, hostile sizes, and mux round trips.
- `tests/external/oracle.rs` asks FFmpeg for a 16-bit PCM file and a float file, and writes a WAV and an RF64 file with `WavMux`. `ffprobe` must report the same sample rate, channel count, codec name, and length. Python's `wave` module must read the same frame bytes from the FFmpeg file and from the muxed file. The test prints `skipped` and passes when `ffmpeg` or `ffprobe` is not on the `PATH`. `afinfo` and GStreamer's `wavparse` are also asked when they are installed.
- `fuzz_seeds_do_not_panic` runs 64 fixed seeds on every test run. `fuzz_campaign` is `#[ignore]`d; it runs for `WAV_FUZZ_SECS` seconds (default 1800) with `cargo test -p cutlass-riff -- --ignored`.
- `pcm_open_and_seek_stay_in_budget` checks that a PCM seek reads no payload. In a release build it also fails if open takes 50 ms or seek takes 20 ms.
- Two tests also look for real WAV files in a folder kept outside the repository. They check nothing extra when the folder is absent.
- `cargo bench -p cutlass-riff` compares sample count, time span, and payload bytes of one FFmpeg-written PCM file before it times anything.

FFmpeg groups PCM into larger packets than this crate, so the tests compare sample counts and the joined payload, not packet counts.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, wav.** Input: one 53 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 4.23 s | 0.03 s |
| CPU time | 4.21 s | 0.03 s |
| Peak memory | 2.6 MB | 13 MB |
| Packets returned | 13,230,000 | 3,230 |

Cutlass is **141.0× slower** than FFmpeg here.

Writing files was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Tracks | 1 |
| `LIST` and `RIFF` nesting | 8 |
| Chunk headers in one file | 4,096 |
| Unknown chunk kept | 64 KiB |
| `ds64` table rows | 64 |
| MPEG resync window | 64 KiB |
| Header bytes | 8 MiB |
| One packet | 16 MiB |
| Extradata | 1 MiB |

Gaps:

- PCM, A-law, and μ-law are returned one sample frame per packet. Every packet is its own read call and its own allocation, so one second of 44.1 kHz sound is 44,100 reads. `WavDemux` does not buffer, and the desktop app hands it an unbuffered `File`. Reading a long WAV file is slow because of this. No test times a full read of a long file; the budget test times only open and seek.
- `WavMux` keeps every packet in memory and builds the whole file in memory at `finish`. Nothing reaches the writer before that.
- `WavMux::new` does not switch to RF64 when the file passes 4 GiB. It fails at `finish`. The caller has to pick `WavMux::rf64` up front.
- `RIFX` (big-endian) and `BW64` files do not open.
- A-law and μ-law streams are labeled with an unknown codec tag, not a named codec id.
- A forward reader does not see chunks after `data`. A non-`data` chunk larger than 64 KiB on a forward reader is an error.
- The language on `NewStream` is accepted and not written.
- `wavl` and SMV files are not handled.
- Of the block codecs, only IMA ADPCM has a test.

## Specifications

- Multimedia Programming Interface and Data Specifications 1.0 (IBM and Microsoft, 1991), for RIFF and WAVE.
- Microsoft `WAVEFORMATEX` and `WAVEFORMATEXTENSIBLE`.
- EBU Tech 3306, for RF64 and `ds64`.
