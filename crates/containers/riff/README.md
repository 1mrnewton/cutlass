# cutlass-riff

`cutlass-riff` reads and writes WAV files: classic RIFF WAVE and RF64, and it reads BW64. It demuxes a file into packets of stored sample bytes and muxes packets back into a file. It does not decode or convert the samples. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens WAV files with `WavDemux`, and the export crate writes PCM WAV files with `WavMux`.

## What it supports

A file has one sound track. Timestamps are ticks of the sample rate.

| Kind | Demux | Mux |
| --- | --- | --- |
| RIFF WAVE | Opens | `WavMux::new` |
| RF64 with `ds64` | Sizes of `0xFFFFFFFF` are taken from `ds64` | `WavMux::rf64`, which writes `ds64` for a small file too |
| BW64 (ITU-R BS.2088) | Read as RF64, which it is laid out as; `brand()` is `Brand::Bw64` | Not written |
| Integer PCM (tag `0x0001`), 8, 16, 24, 32, or 64 bits | Whole sample frames, up to 64 KiB a packet | `PCMWAVEFORMAT`, no `fact` |
| Float PCM (tag `0x0003`), 32 or 64 bits | Whole sample frames, up to 64 KiB a packet | Written with a `fact` chunk |
| `WAVEFORMATEXTENSIBLE` | The channel mask is in the stream extradata at byte 8 | Written when the stream has a channel mask, more than two channels, a valid-bit width that differs from the stored width, or came from an extensible file |
| A-law (`0x0006`), μ-law (`0x0007`), also as an extensible subformat | Whole sample frames, up to 64 KiB a packet, `CodecId::Pcm`; the extradata keeps the tag and marks the bytes compressed, and the engine hands `cutlass-pcm` codings 7 and 8 | Written back with the same tag and a `fact` chunk |
| MPEG sound (`0x0050`, `0x0055`) | One MPEG frame per packet. `0x0055` is `CodecId::Mp3` | One frame per packet |
| ADPCM and other block codecs | One `nBlockAlign` block per packet | The packet duration must match the block |
| AAC (`0x00FF`, `0x1600`, `0x706D`, `0x4143`, `0xA106`), AC-3 (`0x2000`), FLAC (`0xF1AC`), E-AC-3 (by GUID) | Mapped to that codec id | Stored |
| Any other format tag | Stream kept, codec tag `wav:XXXX` | Refused |
| Unknown extensible GUID | Stream kept, codec tag `wav-guid:…` | Refused |
| Unknown chunk up to 64 KiB | Kept (`WavDemux::kept`) | Written before `data` (`WavMux::keep_chunk`) |
| Unknown chunk above 64 KiB | Skipped and named (`WavDemux::dropped`) | Not written |
| Odd-length chunk | The pad byte is skipped | The pad byte is written |

When the block align is 1 and the tag is not one of the sample or MPEG tags, the whole `data` chunk is one packet if it fits the packet limit.

A PCM, A-law, or μ-law packet is a run of whole sample frames of at most `PCM_PACKET_BYTES` (64 KiB: 16,384 frames of 16-bit stereo), read with one call. Its timestamp is the first frame and its duration the number of frames.

For PCM the length is the `data` bytes that are in the file divided by the block align. A `fact` or `ds64` sample count is not read for PCM, so one that is zero or wrong does no harm. For compressed sound, `fact` and the `ds64` sample count must agree.

### Unfinished and cut files

The `data` size is treated as a claim, and the sound is what the file holds of it. These open and play to the last whole frame, as FFmpeg plays them:

| File | How it is read | Flag |
| --- | --- | --- |
| `data` size `0xFFFFFFFF` with no `ds64`, as `ffmpeg -f wav pipe:` writes | The sound runs to the end of the file | `is_open_ended()` |
| `data` size 0 under a RIFF size of 0 or `0xFFFFFFFF`, or under a RIFF size that ends before the sound starts | The sound runs to the end of the file | `is_open_ended()` |
| `data` size larger than the bytes left in the file (a recording or download that stopped) | Clamped to the file | `is_cut()` |
| `data` that ends inside a sample frame or a codec block | The partial frame is dropped | `is_cut()` |
| A chunk after `data` that runs past the end of the file | The sound plays; the chunk is named in `dropped()` | |
| RIFF size larger than the file | Clamped to the file | |

A forward reader cannot see the end of the file at open: it reads until the input ends, returns the whole frames that arrived, and `is_cut()` turns true then. With no stated size it reports no duration (`duration_samples()` is `None`).

A `data` size of 0 in a file whose RIFF size is filled in is an empty file, and a chunk after it is not played as sound.

`PCMWAVEFORMAT` is kept for 16-bit stereo at any sample rate. The sample rate alone does not make the file extensible.

What is refused:

- `RIFX`: `probe` and `open` report `UnsupportedBrand`.
- A header chunk (`fmt `, `fact`, `ds64`, `LIST`, or an unknown chunk before the sound) whose size runs past its parent or the file: `OpenError::HostileSize`. Only the sound is clipped to the bytes that are present.
- A `ds64` data size that disagrees with a `data` chunk size that is not `0xFFFFFFFF`: `OpenError::Source`.
- A truncated header, or `LIST` nesting deeper than 8: an open error and no stream list.
- In the muxer: unknown codecs; a second track; a packet that is not a whole number of frames or whose duration does not match its bytes; a classic RIFF file above 4 GiB (`MuxError::HostileSize`); a track duration that does not equal the sum of the packets (`MuxError::DurationMismatch`), in which case nothing is written.

## Using it

- `probe` and `WavFamily` decide from the first 12 bytes.
- `WavDemux::open(reader, length)` takes a `Read + Seek` reader and the file length. `open_forward` takes a reader it will not seek; it stops at `data`, and `seek` returns `SeekError::ForwardOnly`.
- `WavDemux` implements `cutlass_container::Demux`, including `indexed_end`, so a caller that wants the length does not read the packets. `format()` returns the parsed `fmt ` chunk as a `WavFormat`. `duration_samples()` is the sample count, or `None` for MPEG sound until the frames are read. `is_cut()` and `is_open_ended()` report a file that was not finished.
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

Seek on PCM and other constant-block sound is arithmetic: `data` start plus index times block size. It reads no payload. On PCM the landing is the asked frame itself, not the start of a run: the next packet begins on that frame, and `Landing::Packet::time` is its tick. The landing's `ordinal` is the frame index and stops at `u32::MAX` in a file of more than 2^32 frames; the time and byte position stay exact. Seek on MPEG sound walks the frame headers from the start of `data`.

`cargo run --release -p cutlass-riff --example demux` writes eight silent stereo frames with `WavMux`, reads them with `WavDemux`, and prints the sample rate, channels, packet count, and sample count.

## How it is tested

```sh
cargo test -p cutlass-riff
```

- Unit tests in `src/behavior.rs` build files in memory. They cover each row of the table above, the packet runs and the packet limit, RF64 and `ds64` disagreement, `fact` handling, kept and dropped chunks, odd padding, forward readers, `data` before `fmt `, seek without reading payload, cancel, hostile sizes, and mux round trips.
- `tests/external/damaged.rs` opens the seven files in `tests/fixtures` (see the README there), on a seekable reader and on a forward one, and checks the frames, the flags, a seek to the last frame, and that a cut header is still refused. With FFmpeg on the `PATH` it compares the samples with `ffmpeg -f s16le`. It also seeks past frame 2^32 in a 4 GiB file that exists only as a reader of zeros.
- `tests/external/oracle.rs` asks FFmpeg for a 16-bit PCM file and a float file, and writes a WAV and an RF64 file with `WavMux`. `ffprobe` must report the same sample rate, channel count, codec name, and length. Python's `wave` module must read the same frame bytes from the FFmpeg file and from the muxed file. The test prints `skipped` and passes when `ffmpeg` or `ffprobe` is not on the `PATH`. `afinfo` and GStreamer's `wavparse` are also asked when they are installed.
- A desktop reader test (`g711_wav_and_aiff_files_play_like_afconvert_and_ffmpeg` in `apps/desktop/src/readers.rs`) plays A-law and μ-law WAV files from afconvert and FFmpeg (classic, extensible with three channels, and RF64) and the files `WavMux` writes back from them, and gets the same s16 samples as `afconvert -d LEI16` and FFmpeg's decode. It skips without the tools.
- `fuzz_seeds_do_not_panic` runs 64 fixed seeds on every test run. `fuzz_campaign` is `#[ignore]`d; it runs for `WAV_FUZZ_SECS` seconds (default 1800) with `cargo test -p cutlass-riff -- --ignored`.
- `pcm_open_and_seek_stay_in_budget` checks that a PCM seek reads no payload. In a release build it also fails if open takes 50 ms or seek takes 20 ms.
- Two tests also look for real WAV files in a folder kept outside the repository. They check nothing extra when the folder is absent.
- `cargo bench -p cutlass-riff` compares sample count, time span, and payload bytes of one FFmpeg-written PCM file before it times anything.

FFmpeg groups PCM into packets of 4,096 frames and this crate into 64 KiB, so the tests compare sample counts and the joined payload, not packet counts.

## Speed against FFmpeg

Measured on 24 November 2026 at commit `4439627d`, on a 4-vCPU KVM guest (Intel Xeon at 2.10 GHz with AVX-512, 15 GB) running Ubuntu 24.04 on Linux 6.18, against FFmpeg 6.1.1 from Ubuntu's packages. Built with `cargo build --release` and rustc 1.97.0.

**Read every packet, wav.** Input: 5 minutes of 48 kHz 16-bit stereo, 57.6 MB, written by FFmpeg. Nothing is decoded.

|  | Cutlass before | Cutlass | libavformat | `ffmpeg` command |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 3,636 ms | 12.9 ms | 59.8 ms | 320 ms |
| CPU time | 3,614 ms | 12.3 ms | 58.7 ms | 453 ms |
| Instructions | 7,964 M | 59.0 M | 186.6 M | 425.8 M |
| Packets returned | 14,400,000 | 879 | 14,063 | |

Cutlass reads the file in **0.22× the time** of libavformat and 0.32× the instructions, and is 282× faster than it was. Before this change it was 61× slower than libavformat on this machine (the 141× of the 10 November measurement, taken on an Apple M5 Pro, was the same loop). The joined payload is the same 57,600,000 bytes on every side.

What changed: a packet is now a run of frames read with one call, where it was one frame, one seek check, one read and one allocation. Nothing else was needed; no read-ahead buffer was added, because a 64 KiB read is already the buffer.

Decoding the same file through `cutlass-pcm` is in that crate's README. Writing files was not measured.

How it was measured: every column is a whole process, start-up included, reading the same file from the page cache and throwing the result away. Each was run five times in turn and the tables show the median wall and CPU time. Instructions are user-space instructions from one run under `valgrind --tool=cachegrind` (`I refs`); the kernel's share of a read is not in them. `perf stat` was not used because this virtual machine exposes no hardware counters (`instructions: <not supported>`). The Cutlass columns are [`tools/media-bench`](../../../tools/media-bench) (``media-bench demux <file>``), which registers only the Rust codecs; "before" is the same tool built at `bcab8681`, the commit this work started from. The libavformat column is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c) built against the same FFmpeg, one thread, which calls libavformat and libavcodec with no filter graph and no output stage. The `ffmpeg` column is the command line (``ffmpeg -i <file> -c copy -f null -``); it starts more threads than it is asked to and its CPU time is above its wall time for that reason. `scripts/bench_media.py` was not used: it reads `/usr/bin/time -l`, which is macOS's.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Tracks | 1 |
| `LIST` and `RIFF` nesting | 8 |
| Chunk headers in one file | 4,096 |
| Unknown chunk kept | 64 KiB |
| `ds64` table rows | 64 |
| MPEG resync window | 64 KiB |
| One PCM packet | 64 KiB of whole frames |
| Header bytes | 8 MiB |
| One packet | 16 MiB |
| Extradata | 1 MiB |

Gaps:

- Each packet is its own allocation, zero-filled before the read: `Input` reads into an initialised slice. At 64 KiB a packet that is one `memset` for each read.
- No test times a full read of a long file; the budget test times only open and seek. The numbers above are from a manual run.
- `cargo bench -p cutlass-riff` asks `sysctl` for the machine and so runs on macOS only.
- `WavMux` keeps every packet in memory and builds the whole file in memory at `finish`. Nothing reaches the writer before that.
- `WavMux::new` does not switch to RF64 when the file passes 4 GiB. It fails at `finish`. The caller has to pick `WavMux::rf64` up front.
- `RIFX` (big-endian) files do not open. `BW64` files are read, not written, and their `axml` and `chna` chunks are kept or dropped like any unknown chunk.
- A forward reader does not see chunks after `data`. A non-`data` chunk larger than 64 KiB on a forward reader is an error.
- The language on `NewStream` is accepted and not written.
- `wavl` and SMV files are not handled.
- Of the block codecs, only IMA ADPCM has a test.

## Specifications

- Multimedia Programming Interface and Data Specifications 1.0 (IBM and Microsoft, 1991), for RIFF and WAVE.
- Microsoft `WAVEFORMATEX` and `WAVEFORMATEXTENSIBLE`.
- EBU Tech 3306, for RF64 and `ds64`.
- ITU-R BS.2088, for the BW64 brand.
