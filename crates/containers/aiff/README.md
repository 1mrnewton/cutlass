# cutlass-aiff

Reader and writer for AIFF and AIFF-C sound files. It finds the `COMM` and `SSND` chunks, hands out the sound as packets with sample-accurate times, and writes linear PCM back into a file; it does not decode samples. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens `.aiff` files through `AiffDemux`. Export does not write AIFF; `AiffMux` is used only by tests.

## What it supports

The stream timescale is the sample rate. `FEATURES` lists the same rows in code.

| Sound data | Reading | Writing |
| --- | --- | --- |
| AIFF, big-endian integer PCM, 1 to 32 bits | Whole sample frames, up to 64 KiB a packet, codec `CodecId::Pcm`. A depth that is not a whole number of bytes (12-bit, 20-bit) is stored left-justified in the next whole byte width and is read at that width | 8, 16, 24 and 32 bits are written as AIFF |
| AIFF-C `NONE`, `twos`, `in24`, `in32` (big-endian integer) | Same | Not written; big-endian integer goes out as AIFF |
| AIFF-C `sowt` (little-endian integer) | Same | Written as AIFF-C |
| AIFF-C `fl32`, `fl64` (big-endian float) | Same | Written as AIFF-C |
| AIFF-C `FL32`, `FL64` (the same floats under the upper-case tags QuickTime wrote) | Same | Not written |
| Signed 8-bit PCM (AIFF, or AIFF-C `NONE`) | Same, with coding byte 6 | Written |
| AIFF-C `raw ` (unsigned 8-bit, 128 for silence) | Same, with coding byte 11 | Not written |
| `alaw`, `ulaw` (also `ALAW`, `ULAW`, which FFmpeg reads and AudioToolbox does not) | Whole sample frames, `CodecId::Pcm` with coding byte 7 or 8 | No |
| `ima4` | One 34-byte block per channel per packet, 64 sample frames | No |
| `MAC3`, `MAC6` | One block per packet, 6 sample frames | No |
| `GSM ` | One 33-byte block per packet, 160 sample frames | No |

PCM streams carry three bytes of extradata, the layout `cutlass-pcm` reads: coding, endian (1 is big-endian), interleaved (always 0). The compressed kinds are published as `StreamCodec::Unknown` with the fourcc as the tag; this crate does not map them to a `CodecId`.

Chunks other than `COMM` and `SSND` are skipped and counted (`skipped_chunks`). The `SSND` offset field is honored. `COMM` may come after `SSND` when the reader can seek.

A PCM packet is a run of whole sample frames of at most 64 KiB (16,384 frames of 16-bit stereo), read with one call. Its timestamp is the first frame and its duration the number of frames. A seek on PCM lands on the asked frame itself, and the next packet starts there.

The sample rate is an 80-bit float. A fractional rate, such as the 22254.5454 Hz of early Macintosh sound, is rounded to the nearest hertz with a half going up (22255), which is what FFmpeg does.

`raw ` was read as signed 8-bit before 24 November 2026. It is offset binary (QuickTime File Format, sound sample descriptions; FFmpeg maps it to `pcm_u8`), so such a file played with every sample 128 steps off.

### Unfinished and cut files

The sound is what the file holds. These open and play to the last whole frame, as FFmpeg plays them:

| File | How it is read |
| --- | --- |
| `SSND` or the FORM states more bytes than the file has (a recording or download that stopped) | Clamped to the file; `is_cut()` is true when frames `COMM` counts are missing |
| `COMM` counts more sample frames than `SSND` holds | The frames that are there; `is_cut()` |
| `numSampleFrames` is 0 | The whole frames in `SSND` |
| The last chunk has an odd size and no pad byte | Read as it is |
| The FORM size is smaller than the file but `SSND` is all there | `SSND` is read from the file |
| A chunk after the sound runs past the end | The sound plays; the chunk is counted in `skipped_chunks` |

`COMM` is still the length when it counts fewer frames than `SSND` holds. FFmpeg reads every byte of `SSND` in that case; this crate follows the count. A forward reader learns that a file is short only when the input ends: it returns the whole frames that arrived and `is_cut()` turns true then.

What is refused, all as `OpenError`:

- Any other AIFF-C compression (`QDM2`, `QDMC`, `Qclp`, `G722` and so on): `Source`, with the fourcc in the message.
- A sample size of 0 or above 32 bits, or one that does not match the compression type: `Source`.
- A sample rate that rounds to zero, or is negative, not finite, or above `u32::MAX`: `Source`.
- A missing or duplicate `COMM` or `SSND`, zero channels, a sample count that is not a whole number of blocks: `Source`.
- A chunk before the sound that runs past the form or the file, more than 64 channels, more than 4096 chunks, a `COMM` over 4096 bytes: `HostileSize`.
- A file that ends inside the `SSND` header: `Truncated`.

The writer refuses anything but `CodecId::Pcm`, little-endian float, planar layouts, a second stream, a packet that is not exactly one sample frame, and a finish whose duration does not equal the packet count. On a refused finish no bytes are written.

## Using it

- `AiffDemux::open(reader, length)` opens a `Read + Seek` source; `open_limited` takes caller `Limits`.
- `AiffDemux::open_forward(reader)` reads in order without seeking. Seek is then refused with `SeekError::ForwardOnly`, and `SSND` before `COMM` is an error.
- `AiffDemux` implements `cutlass_container::Demux`, including `indexed_end`, so a caller that wants the length does not read the packets. It also has `sample_rate`, `channels`, `duration_samples`, `is_aifc`, `is_cut` and `skipped_chunks`.
- `AiffMux::new(writer)` implements `cutlass_container::Mux`. `into_inner` returns the writer.
- `probe(header)` accepts `FORM....AIFF` and `FORM....AIFC` from the first 12 bytes. `AiffFamily` is the same check as a `Probe` type.

```rust
use std::fs::File;

use cutlass_aiff::AiffDemux;
use cutlass_container::Demux;

fn sample_frames(path: &std::path::Path) -> Result<u64, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut demux = AiffDemux::open(file, length)?;
    let mut frames = 0;
    while let Some(packet) = demux.read_packet()? {
        // For PCM, packet.bytes() is a run of whole sample frames, and
        // the duration is how many.
        let _ = packet.bytes();
        frames += u64::try_from(packet.timestamp().duration().unwrap_or(0))?;
    }
    Ok(frames)
}
```

## How it is tested

```text
cargo test -p cutlass-aiff
```

- `src/behavior.rs` builds files by hand and checks packets, timestamps, the packet runs, chunk skipping, the `SSND` offset, seeking, forward reading, each compression's block size, the bit depths, the cut and unfinished files above, every refusal above, and the writer's round trips. `mutated_bytes_fail_closed` flips every byte of a small file and opens every prefix of it.
- `tests/external/oracle.rs` has FFmpeg write 16-bit big-endian, 16-bit little-endian and 32-bit float AIFF files, reads them here and compares the bytes with the source, then has FFmpeg decode files this crate wrote. `afinfo` is asked for the sample rate when it is installed. The test is skipped when `ffmpeg` or `ffprobe` is not on `PATH`.
- `tests/external/damaged.rs` opens the ten files in `tests/fixtures` (see the README there), on a seekable reader and on a forward one, and checks the layout bytes, the rate, the frames as 16-bit samples, and a seek to the last frame. With FFmpeg on the `PATH` it compares those samples with `ffmpeg -f s16le` and the rate with `ffprobe`.
- `tests/external/local.rs` opens private sample files and is skipped when their folder is absent.

A desktop reader test (`g711_wav_and_aiff_files_play_like_afconvert_and_ffmpeg` in `apps/desktop/src/readers.rs`) plays `alaw` and `ulaw` files from afconvert and FFmpeg, and upper-case copies of them, and gets the same s16 samples as `afconvert -d LEI16` and FFmpeg's decode; it skips without the tools.

There is no long fuzz campaign and no committed benchmark; the numbers below are from a manual run. The compressed kinds (`ima4`, MACE, GSM) are tested only with hand-built files, not against another reader.

## Speed against FFmpeg

Measured on 24 November 2026 at commit `4439627d`, on a 4-vCPU KVM guest (Intel Xeon at 2.10 GHz with AVX-512, 15 GB) running Ubuntu 24.04 on Linux 6.18, against FFmpeg 6.1.1 from Ubuntu's packages. Built with `cargo build --release` and rustc 1.97.0.

**Read every packet, aiff.** Input: 5 minutes of 48 kHz 16-bit stereo, 57.6 MB, written by FFmpeg. Nothing is decoded.

|  | Cutlass before | Cutlass | libavformat | `ffmpeg` command |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 3,260 ms | 11.0 ms | 38.4 ms | 331 ms |
| CPU time | 3,252 ms | 10.6 ms | 38.0 ms | 475 ms |
| Instructions | 6,063 M | 58.9 M | 105.0 M | 335.3 M |
| Packets returned | 14,400,000 | 879 | 14,063 | |

Cutlass reads the file in **0.29× the time** of libavformat and 0.56× the instructions, and is 296× faster than it was. Before this change it was 85× slower than libavformat on this machine (the 208× of the 10 November measurement was taken on an Apple M5 Pro). The joined payload is the same 57,600,000 bytes on every side.

**Read and decode, aiff.** The same file through the engine and `cutlass-pcm` to 16-bit samples; the decoded samples have the same SHA-256 on the old build, the new build and `ffmpeg -f s16le`.

|  | Cutlass before | Cutlass | libavformat and libavcodec | `ffmpeg` command |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 8,867 ms | 38.3 ms | 58.5 ms | 914 ms |
| CPU time | 8,834 ms | 37.6 ms | 56.9 ms | 1,042 ms |
| Instructions | 35,929 M | 145.8 M | 402.4 M | 889.6 M |

Part of the old decode time was the engine reading every packet once at open to learn the length; `indexed_end` now answers that from the header.

Writing files was not measured.

How it was measured: every column is a whole process, start-up included, reading the same file from the page cache and throwing the result away. Each was run five times in turn and the tables show the median wall and CPU time. Instructions are user-space instructions from one run under `valgrind --tool=cachegrind` (`I refs`); the kernel's share of a read is not in them. `perf stat` was not used because this virtual machine exposes no hardware counters (`instructions: <not supported>`). The Cutlass columns are [`tools/media-bench`](../../../tools/media-bench) (``media-bench demux <file>` and `media-bench decode sound <file>``), which registers only the Rust codecs; "before" is the same tool built at `bcab8681`, the commit this work started from. The libavformat column is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c) built against the same FFmpeg, one thread, which calls libavformat and libavcodec with no filter graph and no output stage. The `ffmpeg` column is the command line (``ffmpeg -i <file> -c copy -f null -` to read, `ffmpeg -threads 1 -i <file> -f null -` to decode, which also re-encodes to 16-bit for the null output`); it starts more threads than it is asked to and its CPU time is above its wall time for that reason. `scripts/bench_media.py` was not used: it reads `/usr/bin/time -l`, which is macOS's.

## Limits and gaps

- Header limits: 4096 chunks in a form, 64 channels, 4096 bytes of `COMM`. One packet may not exceed `Limits::max_packet_bytes`. There is no limit on the length of the sound.
- `COMM` stores the sample count in 32 bits and the form size in 32 bits, so the writer refuses a file over 4 GiB.
- Each packet is its own allocation, zero-filled before the read. `AiffDemux` does not buffer its reader and does not need to: a PCM read is up to 64 KiB.
- `AiffMux` still takes exactly one sample frame per packet, so a PCM packet from `AiffDemux` has to be split into frames before it is written back.
- The writer holds every packet in memory and writes the whole file at `finish`.
- A forward reader needs the `SSND` size: a streamed AIFF that states none is not read to the end of the input.
- IMA 4:1, MACE and GSM are split into blocks but have no `CodecId`, and the writer does not write them. The writer does not write A-law or μ-law either.
- Markers, instrument data, names, comments and ID3 chunks are skipped, not exposed, and not written.
- A forward reader stops at the sound and never reads chunks after it. A skipped chunk larger than `Limits::max_header_bytes` is refused on a forward reader.

## Specifications

- Audio Interchange File Format: "AIFF", version 1.3 (Apple Computer, 1989).
- Audio Interchange File Format AIFF-C (Apple Computer, 1991).
