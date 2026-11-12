# cutlass-mpegts

An MPEG transport stream demuxer and muxer in safe Rust (`#![forbid(unsafe_code)]`), after ITU-T H.222.0 (ISO/IEC 13818-1). It reads `.ts`, `.m2ts` and `.mts` files: it finds the packet grid, reads the PAT and PMT, reassembles PES packets, splits them into access units with their times and keyframes, and seeks without an index. It writes one program of TS or M2TS that FFmpeg, ffprobe and GStreamer read back. Nothing is decoded here.

The crate depends on `cutlass-codec`, `cutlass-container` and `cutlass-adts` (for LATM). It runs no other library, tool or OS parser.

The probe crate reports a transport stream's picture size and length, and the desktop app, the FFI layer and `media-bench` open transport streams with `TsDemux`. Export writes them: `cutlass_export::write_movie` with `ContainerId::MpegTs` writes TS, or M2TS when the destination ends in `.m2ts` or `.mts`, with the muxer deriving decode times (below), since this workspace's encoders report presentation times only. The desktop export panel offers no container choice; that is a product decision outside this crate.

## What it supports

| | Demux | Mux |
| --- | --- | --- |
| Packet size | 188-byte TS, 192-byte M2TS (BDAV), 204-byte TS with a FEC trailer | 188 or 192 |
| Sync | five sync bytes at one stride; lost sync is searched for again and the skipped bytes counted | n/a |
| Tables | PAT and PMT with CRC32, sections across packets, several sections in one packet; a later PMT version of the open program is counted (`pmt_changes`) | PAT and PMT before the first data, every 100 ms of decode time, and before each video keyframe |
| Service names | DVB SDT (PID 0x11): provider and service name of each program (`Program::service`), UTF-8 behind 0x15, other text read as Latin-1 | an SDT every 500 ms, as FFmpeg writes one: provider `Cutlass` and service `Service01` unless `with_service` names them |
| Programs | every program the PAT lists; one is read (policy below) | one program, number 1, PMT on PID 0x1000 |
| PES | reassembled across packets, header split across packets, known or zero `PES_packet_length` | one PES packet per video access unit, with length 0; audio frames that follow each other share a PES packet of up to 2930 bytes and 0.2 s, as FFmpeg packs them |
| Timestamps | PTS and DTS on the 90 kHz clock, 33-bit wrap unwrapped | moved to 90 kHz, plus a 1.4 s delay, as FFmpeg does; with `derive_decode_times`, DTS derived from the PTS and the stream's reorder depth |
| PCR | parsed; not used to time packets | on the first video stream (else the first stream), at least every 20 ms; PCR-only packets bridge gaps |
| Continuity | counter checked per PID: a lost packet drops its PES packet, duplicates are skipped, `discontinuity_indicator` excuses a jump | kept per PID |
| Errors | `transport_error_indicator` and scrambled packets dropped and counted | n/a |
| Seek | bisection on PES timestamps, no index; lands on the keyframe before, whether or not the random access indicator marks it | n/a |
| Duration | from the last timestamps in the file's tail | n/a |

Codecs, by PMT `stream_type`:

| Codec | Demux | Mux |
| --- | --- | --- |
| H.264 (0x1B) | access units, IDR and recovery-point keyframes, picture size, parameter sets as Annex B extradata | Annex B with an AUD; `avcC` input converted; parameter sets put before keyframes that lack them |
| HEVC (0x24) | access units, IRAP keyframes, picture size | Annex B with an AUD; `hvcC` input converted; `HEVC` registration |
| MPEG-1/2 video (0x01, 0x02) | pictures, both fields of a frame in one unit, I-picture keyframes | 0x02; sequence header before keyframes that lack one |
| MPEG-4 Part 2 (0x10) | VOPs, I-VOP keyframes, picture size and the headers up to the first VOP as extradata | 0x10; the headers put before keyframes that lack them |
| AAC in ADTS (0x0F) | frames, ADTS header taken off, AudioSpecificConfig from the header | raw AAC gets an ADTS header; ADTS input passed through |
| AAC in LATM/LOAS (0x11) | LOAS frames; payload and config come from `cutlass-adts` | not written: AAC is always ADTS |
| AC-3 (0x81, or 0x06 with an AC-3 descriptor or registration) | frames, rate, channels | 0x81, `AC-3` registration |
| E-AC-3 (0x87; 0x84 and 0xA1 under an HDMV registration; 0x06 with a descriptor) | frames, dependent substreams joined to their unit | 0x87, `EAC3` registration |
| MPEG audio layer III (0x03, 0x04) | frames, codec `Mp3` | 0x03, or 0x04 below 32 kHz |
| MPEG audio layers I and II | framed and timed, but an unknown codec: nothing in the workspace decodes them | refused |
| Opus (0x06 with `Opus` registration) | the TS control header taken off; `OpusHead` built from the DVB extension descriptor | control header, registration and extension descriptor, as FFmpeg writes them |
| Private data (0x06) with nothing naming it | ADTS AAC, AC-3 or E-AC-3 when its first bytes hold two frame headers a frame apart (FFmpeg's M2TS mode writes AAC this way); else one PES packet per packet, codec `Unknown("mpegts:0x06")` | n/a |
| Anything else | one PES packet per packet, codec `Unknown("mpegts:0xNN")` | refused: `UnknownCodec` or `Rejected` |

**Which program is read.** `TsDemux::open` reads the first program in PAT order that has a stream this crate frames, else the first program with any stream. `open_program(reader, length, number)` reads the one asked for, and fails if its PMT is not in the opening bytes. `programs()` lists them all.

**Times.** Every stream has timescale 90000 (`TIMESCALE`). A packet's PTS and DTS are the PES ones, unwrapped onto one line: each timestamp is placed on the copy of the 33-bit circle nearest the last timestamp of the program, so a jump of less than about 13.25 hours either way comes out right, and a file that runs across the wrap keeps rising. An audio frame that starts in a PES packet without a timestamp is timed from the frame before and its sample count; a video access unit in that place has no timestamp. Every stream carries one edit: from the program start (the lowest timestamp in the opening bytes) for the program's duration, so the streams stay in step.

**Duration.** On a random-access input the last 1 MiB is read, then 2, 4, up to 16 MiB, until every stream that had data shows a timestamp there. The end is the latest timestamp plus that unit's length (audio) or the shortest gap between timestamps seen there (video). A forward input has no end.

**Seek.** There is no index. A seek:

1. bisects the file on the PES timestamps of the asked-for stream, at most 64 probes of at most 4 MiB each, to a byte position whose first timestamp is not past the target;
2. for a picture stream (H.264, HEVC, MPEG-2, MPEG-4 Part 2), reads transport headers only, back from there in doubling windows from 1 MiB up to 64 MiB, for the last PES packet of that stream marked with the random access indicator and whose PTS is not past the target;
3. frames that one access unit and, if it is a keyframe, walks the stream's later PES packets up to the target, reading each header and at most 64 KiB of payload, enough to classify its first picture. A keyframe the muxer left unmarked moves the landing to it: an MP4 sync-sample table can list fewer keyframes than the bitstream holds (2 of the 40 local movies do), and a remux marks only the listed ones. The preroll is the PES packets from the landing whose PTS is before the target;
4. otherwise (an audio stream, no mark, or the marked unit is not a keyframe: the indicator is only a hint) frames forward from the bisection's position, starting further back while no keyframe comes before the target, doubling from 1 MiB at most 24 times, and lands on the last keyframe not past the target. Audio takes this path because FFmpeg packs several frames in one PES packet, so a marked PES start can sit up to a PES packet before the frame that holds the target.

Reading resumes at the transport packet that started the landing's PES packet. Packets of other streams framed before the landing are held (at most 4096) and returned right after it. A failed seek leaves the read position where it was. `last_seek()` reports the bytes, probes, runs, units and headers the last seek cost. A forward input cannot seek.

## Limits

Every buffer is bounded. The `Limits` given to `open_limited` and in `MuxOptions` cap packet size (`max_packet_bytes`), tracks (`max_tracks`, and at most 255 demuxed and 3840 muxed) and extradata. The crate's own caps are public constants:

| Constant | Value | What it bounds |
| --- | --- | --- |
| `MAX_RESYNC_BYTES` | 1 MiB | bytes searched for the grid in one read call |
| `MAX_SECTION_BYTES` | 4096 | one PSI section |
| `MAX_PROGRAMS`, `MAX_DESCRIPTORS` | 256, 256 | programs read from the PAT, descriptors in one loop |
| `MAX_OPEN_SCAN_BYTES` | 16 MiB | bytes open reads for tables, parameters and the start |
| `MAX_TAIL_SCAN_BYTES` | 16 MiB | bytes read back from the end for the duration |
| `MAX_SEEK_PROBES`, `MAX_PROBE_WINDOW_BYTES` | 64, 4 MiB | the bisection |
| `MAX_RANDOM_ACCESS_SCAN_BYTES` | 64 MiB | the header-only scan back for a marked keyframe |
| `MAX_SEEK_WIDENINGS` | 24 | the fallback's doubling steps |
| `MAX_HELD_PACKETS` | 4096 | other streams' packets held after a seek |
| `MAX_BUFFERED_BYTES` | 96 MiB | PES buffers and pending units of all tracks together; past it the largest is dropped and `shed()` counts it |
| `MAX_PARAMS_BYTES`, `MAX_PARAMS_PES` | 1 MiB, 64 | what open keeps to learn a stream's parameters |

Damage is counted, not fatal: `counters()` (continuity errors, duplicates, transport errors, bad packets, scrambled, dropped PES packets, dropped units, short PES packets, PMT changes), `resyncs()`, `skipped_bytes()`, `held_dropped()` and `shed()`.

**Private streams.** A PMT entry of type 0x06 with no descriptor or registration naming its codec is read as ADTS AAC, AC-3 or E-AC-3 when its first payload bytes hold two frame headers of that syntax one frame apart. FFmpeg's M2TS mode writes AAC this way. Open reads at most 16 KiB of each such stream for this.

**Derived decode times.** With `MuxOptions::derive_decode_times`, the muxer ignores the packets' DTS. A stream with reorder depth R gives its i-th unit the (i − R)-th smallest PTS so far as its DTS, and the first R units sit one frame step apart before the smallest PTS, as x264 times its output. R is the H.264 VUI `max_num_reorder_frames`, else 0 for `pic_order_cnt_type` 2 or an intra profile, else what the level's DPB holds (at most 16); the HEVC `sps_max_num_reorder_pics`; 16 when no parameter set is found; 1 for MPEG-2 and MPEG-4 Part 2; 0 for audio. The tests check this against FFmpeg's `trace_headers` reading of x264 and x265 parameter sets. Export turns it on.

The muxer holds units per stream and writes them in decode-time order across streams. A unit leaves once every stream holds one, or once the held units span 10 s of decode time or 32 MiB. A stretch of decode time with no data gets PCR-only packets (at most 3000, a minute) and tables every 100 ms.

What the muxer refuses, leaving itself failed:

| Case | Error |
| --- | --- |
| A codec it does not carry | `UnknownCodec` or `Rejected` |
| `add_stream` after the first packet | `Rejected` |
| A timescale that is not a clock, or a packet in another timescale than its stream | `Clockless`, `TimescaleMismatch` |
| A packet with neither PTS nor DTS | `MissingTimestamp` |
| PTS before DTS, or a DTS that goes back | `Rejected` |
| An audio unit over 65522 bytes, or any unit over `max_packet_bytes` | `HostileSize` |
| An `avcC`, `hvcC` or AudioSpecificConfig it cannot read; an Opus channel mapping with no MPEG-TS code | `Rejected` |

## Using it

- `TsDemux::open(reader, length)` opens a random-access input; `open_limited` takes `Limits`; `open_program` picks a program; `open_forward(reader)` reads once from the start and cannot seek. `TsDemux` implements `cutlass_container::Demux`; track ids are PIDs.
- `programs()`, `program()`, `elementary_streams()` (stream types and descriptors), `packet_size()`, `start()` and `end()` report what open found.
- `TsMux::new(writer)` writes TS, `TsMux::m2ts(writer)` writes M2TS, and `with_options` sets the packet size, the delay, `Limits` and `derive_decode_times`. `with_service(provider, name)` names the SDT's service; each name is cut to 126 bytes at a character boundary. `TsMux` implements `cutlass_container::Mux`. Streams get PIDs 0x100, 0x101 and so on (`pids()`).
- `probe` and `TsFamily` recognize five sync bytes at one stride in the first kilobyte.
- `crc`, `packet`, `pes`, `psi`, `section`, `time`, `es` and `pipeline` are public modules for the lower layers.

Examples:

```text
cargo run -p cutlass-mpegts --example demux -- FILE [--packets] [--seek SECONDS]
cargo run -p cutlass-mpegts --example remux -- IN OUT [--m2ts]
cargo run -p cutlass-mpegts --release --example bench -- open FILE REPEAT | read FILE | seek FILE N FIRST LAST
```

`remux` copies an MP4, MOV or transport stream's carried streams into a new transport stream and names the ones it leaves out. It reads the input twice: the first pass finds how far each stream's decode times must move earlier so that no sample shows before it decodes. MP4s with negative composition offsets, such as macOS screen recordings, need this; FFmpeg's MOV demuxer moves its DTS the same way.

## How it is tested

Run `cargo test -p cutlass-mpegts` (add `--release` for the external tests).

- 69 unit tests: CRC32 against a bitwise reference, packet and adaptation parsing, sections across packets, PAT, PMT and SDT (an SDT FFmpeg wrote, with a UTF-8 name), DVB text, the stream-type table, PES headers, the 33-bit unwrap, each elementary-stream framer (MPEG-4 Part 2 against a video object layer FFmpeg wrote), the H.264 and HEVC reorder depth against x264 and x265 parameter sets, ADTS detection in the muxer, and naming private streams.
- 20 behavior tests (`tests/behavior.rs`) mux synthetic H.264 and AAC and read them back: every unit from TS and M2TS, forward input, seeks to the keyframe before the target with the random access indicator present, cleared, set on every picture, and left off every other keyframe, a failed seek keeping the read position, a cut file, a lost packet, a duplicate, garbage between packets, a 204-byte stream read and seeked as its 188-byte packets, a PMT version change, the SDT names (default, UTF-8, and cut), the 33-bit wrap, two programs, input that is not a transport stream, hostile packets, refused mux input, other timescales, derived decode times, and audio frames sharing PES packets.
- 9 fixture tests (`tests/fixtures.rs`) read eight small files written by FFmpeg 8.1.1 and GStreamer 1.28.3; `tests/fixtures/README.md` has the commands and checksums. They cover H.264, HEVC and MPEG-2 video; ADTS, LATM, AC-3, E-AC-3, MP2, MP3 and Opus audio; M2TS with an HDMV registration; two programs; and GStreamer's `mpegtsmux` with PTS starting at 3600 s. Their packet counts, keyframes and first and last PTS were compared packet by packet with `ffprobe -show_packets`, and all eight matched. Each file is also seeked.
- `tests/external` compares with other tools at test time. It needs `ffmpeg` and `ffprobe`, and `gst-launch-1.0` for the GStreamer checks; without them each test prints `skipped` and passes.
  - Ten FFmpeg-written variants (H.264 with B-frames and AAC, HEVC and AC-3, MPEG-2 and MP2, interlaced MPEG-2 and MP3, MPEG-4 Part 2 with B-VOPs and MP3, LATM, Opus, E-AC-3 M2TS, AAC M2TS written as private data, AC-3 M2TS) are read packet by packet against ffprobe (PTS, DTS, duration, keyframe, and size less ADTS headers; AAC at 44.1 kHz within one tick) and their SDT names against ffprobe's `service_provider` and `service_name`. Every stream is seeked to nine targets, from before the start to past the end; each landing must be the stream's last keyframe at or before the target, and the next packet that keyframe.
  - Two GStreamer `mpegtsmux` files (H.264 with AAC, H.264 with MP3) are read against ffprobe, and a two-program FFmpeg file is opened by program number.
  - Seven MP4 and MOV sources (H.264 with AAC at 48 and 44.1 kHz, HEVC with Opus, MPEG-2 with AC-3, H.264 with MP3, H.264 with E-AC-3, MPEG-4 Part 2 with B-VOPs and AAC) are remuxed by this crate to TS and M2TS. ffprobe must see the same packets and the same service names (the M2TS ones not ASCII), `ffmpeg -f null` must decode them without a message, and GStreamer's `tsdemux` must read them to the end. The structure is checked: continuity counters, PAT and PMT repetition, the PCR interval, no timestamp before its PCR, and rising M2TS arrival times.
  - `local.rs` remuxes up to `TS_LOCAL_FILES` (default 5) movies of up to 150 MB from `CUTLASS_LOCAL_FILES` (default `../local-files` beside the checkout), reads them back, checks the structure and seeks. On 12 November 2026 all 40 local movies (phone, screen and downloaded recordings; H.264 and HEVC up to 4096x2160; AAC) passed with 0 problems.
- `tests/fuzz.rs` runs in-process mutation campaigns, not cargo-fuzz (libfuzzer-sys builds C++). The demux corpus is the eight fixtures, each remuxed by this crate as TS and as M2TS, and a file whose clock wraps past 2^33: 25 files. Mutants flip, insert, delete, splice and duplicate bytes; drop, repeat and swap packets; and hit header, adaptation, PES and table fields. Half get their CRCs repaired so the tables are reached. Each mutant is probed, opened three ways, read to the end and seeked, and every call must return within bounds. Mux sessions build random stream setups and packet sequences over every carried codec, valid and not, a quarter of them deriving decode times and a third naming the service with any text; a refused call must leave the mux failed, clean H.264 and AAC sessions must read back as exactly the accepted packets (with derived decode times in order and not after their PTS), and their service names must read back cut at a character boundary. `fuzz_seeds_demux` and `fuzz_seeds_mux` run 32 seeds each on every test run. The long campaigns are ignored; `TS_FUZZ_SECS` sets their length and `TS_FUZZ_SEED` their first seed:

```text
TS_FUZZ_SECS=600 TS_FUZZ_SEED=100000 cargo test --release -p cutlass-mpegts --test fuzz -- --ignored --nocapture
```

Results on 12 November 2026, release build, after the last code change (seed origins new for this round; earlier rounds started at 1000):

| Campaign | Build | Seeds | Length | Work | Result |
| --- | --- | --- | --- | --- | --- |
| demux | `eb074aeb` | 100000 to 149517 | 600 s | 2,475,900 mutants | passed; slowest input 109.5 ms |
| mux | `eb074aeb` | 200000 on | 253 s | stopped at seed 216528 | failed: the test's own bound on a cut SDT name was a byte too tight (fixed in `6cee5861`; the muxer was right) |
| mux | `6cee5861` | 200000 to 246699 | 600 s | 934,000 sessions, 373,717 clean ones read back, 22,231,989 packets | passed |

Earlier rounds on older builds: demux 7,643,950 mutants in 1800 s, and 4,128,600 in 900 s at `278b6407` (before the reorder-depth, export, audio-packing and SDT commits); mux 959,740 sessions in 600 s. All passed. Since `eb074aeb` only a doc comment and the fuzz test have changed.

Outside this crate: the export crate's `a_transport_stream_movie_reads_back_with_decode_times` writes H.264 and MPEG-4 Part 2 movies (with B-frames) as `.ts` and `.m2ts` and checks with ffprobe that decode times rise and are never after their picture's time, and that FFmpeg decodes them without a message. The desktop app's `transport_streams_play_their_picture_and_sound` test opens a muxed transport stream through the app's readers and checks its probed length, and the probe crate's `a_transport_stream_reports_its_picture_and_its_length` checks size and length.

These tests ran on macOS only. The crate is linted for Linux (`x86_64-unknown-linux-gnu`) and Windows (`x86_64-pc-windows-msvc`) with `cargo clippy -D warnings`; test runs on those systems are not verified here.

## Speed against FFmpeg

Measured on 12 November 2026 at commit `eb074aeb` on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1 from Homebrew, with rustc 1.99.1 and `cargo build --release`. The FFmpeg side is `scripts/ffmpeg_decode.c` (modes `open`, `demux` and `seek:N:FIRST:LAST`) and the Rust side is `examples/bench.rs`; both read the file through their usual buffered I/O. Each number is the median of five runs: retired instructions and peak memory from `/usr/bin/time -l`. Wall time is left out: the machine was running other builds and two fuzz campaigns.

Inputs, made by FFmpeg from local recordings and not checked in:

- `h264_aac_1080p.ts`: 74.7 MB, 530.6 s, H.264 1920x1080 at 30 fps and AAC.
- `hevc_4k.ts` and `hevc_4k.m2ts`: 144.5 and 147.6 MB, 10.9 s, HEVC 3840x2160 at 120 fps and AAC. In the M2TS, FFmpeg writes the AAC as private data (0x06) with no descriptor naming it.

| File | Job | Cutlass | FFmpeg | Ratio |
| --- | --- | --- | --- | --- |
| any | start the process, open nothing | 15.4 M, 2.0 MiB | 287 M, 7.5 MiB | 0.05 |
| `h264_aac_1080p.ts` | open 20 times | 208 M, 4.0 MiB | 1376 M, 27.4 MiB | 0.15 |
| `h264_aac_1080p.ts` | read all 38765 packets | 618 M, 4.0 MiB | 937 M, 20.1 MiB | 0.66 |
| `h264_aac_1080p.ts` | open, then 100 seeks of the video, each read to its next packet | 362 M, 4.8 MiB | 606 M, 20.5 MiB | 0.60 |
| `hevc_4k.ts` | open 20 times | 1133 M, 9.3 MiB | 10158 M, 71.4 MiB | 0.11 |
| `hevc_4k.ts` | read all 1811 packets | 1109 M, 7.7 MiB | 2944 M, 59.1 MiB | 0.38 |
| `hevc_4k.ts` | 100 seeks | 2044 M, 8.1 MiB | 3002 M, 59.4 MiB | 0.68 |
| `hevc_4k.m2ts` | open 20 times | 1135 M, 8.8 MiB | 10172 M, 71.3 MiB | 0.11 |
| `hevc_4k.m2ts` | read all 1811 packets | 1110 M, 7.5 MiB | 3003 M, 59.2 MiB | 0.37 |
| `hevc_4k.m2ts` | 100 seeks | 2036 M, 7.8 MiB | 3304 M, 60.2 MiB | 0.62 |

Instructions are for the whole process, start-up included. What makes the comparison less than even:

- FFmpeg's open runs `avformat_find_stream_info`, which decodes frames to fill in the stream parameters; this crate reads only headers and parameter sets. The open rows mostly measure that difference. FFmpeg's start-up, with its dynamic libraries, is about 270 M instructions more than this crate's.
- The read rows give the same packet counts. FFmpeg's bytes include the 7-byte ADTS header of every AAC frame, which this crate takes off (3577 bytes over 511 frames in the 4K files).
- Both sides seek to the same 100 targets, drawn by one 64-bit xorshift over the video's span. FFmpeg's TS demuxer has no seek function of its own, so `av_seek_frame` with `AVSEEK_FLAG_BACKWARD` runs libavformat's timestamp bisection through `mpegts_get_dts`, which adds every packet it reads that has a DTS to the stream's index as a keyframe (from reading the FFmpeg source; not measured). So FFmpeg's landing need not be a keyframe, and later seeks reuse the index it built. This crate always lands on the last keyframe not after the target, keeps no index, and so does more work per seek. The landings differ: the sum of landing PTS is 58,800,514 here and 63,245,648 in FFmpeg on the 4K files, and 2,197,866,000 and 2,220,894,000 on the 1080p one, so FFmpeg lands later on average.
- Each seek here read about 18.7 MB of the 4K TS, 18.9 MB of the 4K M2TS and 3.9 MB of the 1080p file (`last_seek().bytes`). FFmpeg's bytes were not measured.
- At `5f51cba2`, before the seek walked to unmarked keyframes, the seek rows were 313 M, 1823 M and 1830 M. The walk costs 16% more on the 1080p file and 11 to 12% on the 4K ones; on the 4K files every keyframe is marked and the landings did not change, so this is its cost where it finds nothing. Before the random-access-indicator path, 100 seeks of `hevc_4k.ts` took 12710 M instructions, 4.2 times FFmpeg's.

## Known gaps

Each of these was looked at and left on purpose; the reason is given.

- **MPEG audio layers I and II** are framed and timed but reported as unknown codecs `mpegts:mp2` and `mpegts:mp1`, and the muxer refuses them. `CodecId` has no MP1 or MP2 entry and nothing in the workspace decodes them; adding one is a workspace decision, not a container one.
- **Subtitles and teletext** (DVB subtitles, teletext, HDMV PGS) are carried as opaque streams, one PES packet per packet. The codec and container model has no subtitle stream kind to give them.
- **LATM output.** The muxer writes AAC only as ADTS (stream type 0x0F), the usual carriage that FFmpeg and GStreamer write by default; LATM is read.
- **Program changes.** A new PMT version after open is counted, not followed: the `Demux` trait fixes the stream list at open, so a PID the new version adds is not read.
- **Time-base discontinuities.** Times are unwrapped against the last timestamp, so a jump of more than about 13.25 hours, or a splice that steps far back, is placed on the nearest turn of the 33-bit clock rather than starting a new time line. Bisection seeking needs times that rise through the file, so a spliced file cannot be seeked reliably by any index-free method. FFmpeg's demuxer also passes such jumps on (it uses `discontinuity_indicator` only for its continuity check) and flags the format `AVFMT_TS_DISCONT` for the caller to handle.
- **Interleaving bound.** The muxer holds at most 10 s or 32 MiB of units to write streams in decode order. A source whose streams arrive further apart is written partly out of order, so some PES packets get less than the 0.7 s PCR lead. The bound keeps memory fixed; in all 40 local movies no timestamp came before its PCR.
- **Preroll on the fast seek path** counts PES packets, not access units, between the landing and the target. They are the same when each PES packet holds one picture, as FFmpeg, GStreamer and this muxer write video; counting pictures would mean framing every payload, which the fast path exists to avoid.
- **Negative composition offsets.** `IsoBmffFile` gives samples with PTS before DTS for MP4s with negative `ctts`; a caller remuxing those must either set `derive_decode_times` or move DTS earlier per stream, as `examples/remux.rs` does.
- **The SDT is read from the opening bytes only**: the PAT and PMT search and the parameter pass. FFmpeg writes it first; a broadcast capture that sends it later than those bytes reaches gets no service names. A longer scan would cost every open of a file without an SDT, such as a camera or Blu-ray M2TS.
