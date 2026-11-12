# cutlass-avi

`cutlass-avi` reads and writes AVI files: classic AVI 1.0 and OpenDML AVI 2.0 (`AVIX` segments, `indx` super indexes, `ix##` standard indexes, files over 1 GiB). It demuxes a file into packets and muxes packets back into a file. It does not decode or convert payloads. The crate is safe Rust (`#![forbid(unsafe_code)]`) and runs no foreign library, subprocess, or OS parser.

The desktop app and the FFI readers open AVI files with `AviDemux`, and `cutlass-probe` reports an AVI file's picture size and length. The export path does not write AVI; `AviMux` is there for a caller that asks for it.

## What it supports

Timestamps stay in each stream's own clock: the timescale is `dwRate / g` and one AVI unit (a frame, a sample block or an audio frame) is `dwScale / g` ticks, `g` being their greatest common divisor. An ffprobe tick is one AVI unit, so ffprobe's numbers times `dwScale / g` are this crate's.

| Capability | Demux | Mux | Evidence | Status |
| --- | --- | --- | --- | --- |
| AVI 1.0, one `RIFF AVI ` with `idx1` | Opens | Written as the first segment, `idx1` included | ffmpeg cases, GStreamer `avimux`, `relative_and_absolute_idx1_give_the_same_packets` | Verified |
| OpenDML 2.0: `indx`, `ix##`, `AVIX`, `dmlh` | `indx` is read first; a stream without one, or a broken one, falls back to `idx1` and a scan of the `AVIX` segments | Every stream has an `indx`, every segment an `ix##` per stream; a new `AVIX` starts at `riff_bytes` (1 GiB by default) | `written_opendml_files_read_back_in_ffprobe_and_gstreamer`, `an_ffmpeg_opendml_file_over_one_gib_matches_ffprobe` (1.6 GB, two segments, `--ignored`) | Verified; no muxed file over 1 GiB was written, only many small segments |
| `idx1` offsets from `movi` or from the file start | Both; the base is picked by checking the first entries against the chunks they name | From `movi` | `relative_and_absolute_idx1_give_the_same_packets` | Verified (in-memory files) |
| Missing or broken index | A scan of `movi` rebuilds it: it reads chunk headers, descends `LIST rec `, resyncs over garbage up to 1 MiB, and reads up to 4 KiB of a video chunk to tell a keyframe (H.264, HEVC, MPEG-4, MPEG-2, VP8, VP9, AV1; MJPEG and raw are all keyframes). `fallbacks()` names the index that was dropped and why | Not applicable | `a_scan_reads_keyframes_from_the_bitstream_and_skips_rec_lists`, `an_idx1_pointing_outside_movi_falls_back_to_a_scan`, `garbage_between_chunks_is_skipped_by_a_resync`, `a_cut_recording_reads_up_to_the_cut` | Verified (in-memory files) |
| `LIST rec ` | Read | Not written | as above | Verified (in-memory files) |
| Keyframes | `AVIIF_KEYFRAME` in `idx1`, the delta bit of an `ix##` size, or the scan | Written in both indexes | ffmpeg and GStreamer cases | Verified |
| Dropped frames (empty video chunks) | Lengthen the packet before them | A gap in decode time, or a packet longer than one frame, writes empty chunks (60,000 per gap at most) | `an_empty_chunk_is_a_dropped_frame`, `durations_of_several_frames_write_dropped_frames`, local clips copied by FFmpeg at 600 Hz | Verified |
| Interleaving | An interleaved file is read in byte order; one whose streams drift more than 2 s apart is read in time order | Packets are written in the order they arrive | `a_non_interleaved_file_reads_in_time_order` | Verified (demux); the caller interleaves for the muxer |
| Seek | From the index, without reading a payload (`consumed()` does not change) | Not applicable | `seeks_land_from_the_index_without_reading_payloads`, local clips | Verified |
| H.264 (`H264`, `avc1`, `X264`, `DAVC`, ...) | Annex B or `avcC` extradata as stored; decode time only | `H264`, or `avc1` when the config is an `avcC` record | ffmpeg (B-frames), local clips, remux | Verified |
| HEVC (`HEVC`, `H265`, `hvc1`, ...) | Decode time only | `HEVC` | ffmpeg (`-tag:v HEVC`), local clips | Verified |
| MPEG-4 Part 2 (`FMP4`, `XVID`, `DIVX`, `DX50`, ...) | Decode time only | `FMP4` | ffmpeg `mpeg4` | Verified |
| MPEG-1/2 (`MPG2`, `MPG1`, ...) | Decode time only | `MPG2` | ffmpeg `mpeg2video` | Verified |
| MJPEG (`MJPG`, `AVI1`, ...) | Decode and presentation time; all keyframes | `MJPG` | ffmpeg, GStreamer, a muxed file read by both | Verified |
| VP8 (`VP80`) | Decode and presentation time | `VP80` | ffmpeg `libvpx` | Verified |
| VP9 (`VP90`), AV1 (`AV01`) | Mapped; decode and presentation time | `VP90`, `AV01` | Unit tests only | Implemented, not checked against another tool |
| Raw video (`BI_RGB`, `YUY2`, `I420`, ...) | Unknown codec tag (`avi:BI_RGB`, `avi:YUY2`, ...), packets read | Refused | ffmpeg `rawvideo` | Demux verified; mux unsupported |
| Integer PCM, 8 to 32 bits | One chunk per packet | Constant bit rate, whole sample frames | ffmpeg s16, s24 (extensible), GStreamer, a muxed file | Verified |
| Float PCM | One chunk per packet | As integer PCM | ffmpeg `pcm_f32le` | Verified (demux); mux unit-tested |
| `WAVE_FORMAT_EXTENSIBLE` | Read; the `strf` bytes after `WAVEFORMATEX` are the extradata | Written by `cutlass_riff`'s rules (more than two channels, a valid-bit width that differs from the stored width, ...) | ffmpeg s24 | Verified (demux) |
| MP3 (`0x0055`, `0x0050`) | One chunk per packet; VBR counted in `nBlockAlign` frames | `0x0055` with `MPEGLAYER3WAVEFORMAT`, one frame per chunk | ffmpeg `libmp3lame`, a muxed file | Verified |
| AAC (`0x00FF`, ...) | One chunk per packet; the `strf` extra is the AudioSpecificConfig | `0x00FF`, one frame per chunk; needs the AudioSpecificConfig | ffmpeg, local clips, a muxed file | Verified |
| AC-3 (`0x2000`) | One chunk per packet | `0x2000`, one frame per chunk | ffmpeg `ac3` | Verified (demux); mux unit-tested |
| Other audio tags | Unknown codec tag `wav:XXXX`, packets read | Refused | `unknown_codecs_are_published_and_read` | Demux only |
| `txts` and other stream types | Unknown codec tag, packets read | Not written | `unknown_codecs_are_published_and_read` | Demux only |
| Forward-only input | `AviForwardDemux::open(reader)` takes any `Read` and reads chunks in file order, `AVIX` segments included, with no index: keyframes come from the bitstream, a video packet is returned once its stream's next chunk arrives (so dropped frames lengthen it), and seeking is `SeekError::ForwardOnly` | The muxer needs `Write + Seek` to patch headers | `forward_reading_*`; every ffmpeg case read both ways gives the same packets | Demux verified; mux unsupported |

`FEATURES` in the crate lists the same rows as data.

Errors, not panics:

- A size that runs past its parent or the file, or a stream count, super index, chunk count, header or extradata over the limits below: `OpenError::HostileSize` or another open error. A broken index is not an open error while a scan can rebuild it.
- A chunk whose header does not match the index, or a packet over the packet limit: that read returns an error and the next read goes on with the following chunk.
- A truncated file reads up to the cut.
- In the muxer: raw video; audio other than PCM, MP3, AAC and AC-3; a PCM packet that is not whole sample frames or not contiguous; VBR audio frames of differing length or over 65,535 bytes; a stream with no packet when the header has to be written; a stream that starts more than an hour in; and track durations that do not equal the packets written. Each is `MuxError::Rejected` or `MuxError::DurationMismatch`.

## Using it

- `probe` and `AviFamily` decide from the first 12 bytes (`RIFF`, size, `AVI `).
- `AviDemux::open(reader, length)` takes a `Read + Seek` reader and the file length and implements `cutlass_container::Demux`. `index_source()`, `fallbacks()`, `resyncs()`, `truncated()`, `interleaved()`, `riff_count()`, `main_header()` and `stream_header(track)` say what the file held and how it was read.
- `AviMux::new(writer)` or `AviMux::with_options(writer, AviMuxOptions { .. })` takes a `Write + Seek` writer and implements `cutlass_container::Mux`. The header is written once every video stream has 8 packets and every audio stream one (64 MiB is held at most), so the frame rate comes from real times.

Examples:

- `cargo run --release -p cutlass-avi --example packets -- file.avi` prints the streams and every packet (track, decode, presentation, duration, size, keyframe), in the shape the integration tests compare with ffprobe.
- `cargo run --release -p cutlass-avi --example remux -- in.avi out.avi [riff-mib]` copies an AVI file through `AviDemux` and `AviMux`; a small `riff-mib` makes many `AVIX` segments.
- `cargo run --release -p cutlass-avi --example bench -- <open|read|seek> file.avi <count>` is the Rust side of the benchmark below.

## How it is tested

```sh
cargo test -p cutlass-avi
cargo test -p cutlass-avi --test external -- --ignored   # 1.6 GB OpenDML file and the fuzz campaign
```

- `src/behavior.rs` builds files in memory: relative and absolute `idx1`, a scan with `rec ` lists and bitstream keyframes, an index outside `movi`, a cut file, garbage between chunks, hostile headers, unknown codecs and a text stream, dropped frames, VBR MP3, a non-interleaved file, seek without payload reads, cancel, a misnamed chunk, an oversized packet, and the muxer: round trips, many `RIFF` segments whose `indx`, `ix##` and `idx1` agree, dropped frames, constant-rate PCM, refusals, a config handed over with the first packet, and times off the frame grid.
- `tests/external/oracle.rs`, with FFmpeg 8.1.1 and GStreamer 1.28.3 as independent tools:
  - FFmpeg writes seven files (MJPEG + PCM s16, H.264 with B-frames + AAC, MPEG-4 + MP3, MPEG-2 + AC-3, VP8 + float PCM, HEVC + s24 extensible PCM, raw video + PCM). Every packet must match ffprobe and `framecrc` for size, Adler-32, keyframe flag, decode time and duration. PCM is compared as one byte stream, because FFmpeg splits a PCM chunk into 1024-block packets.
  - A GStreamer `avimux` file matches ffprobe the same way.
  - Files `AviMux` writes, in one segment and in many, and an MJPEG + PCM + MP3 file, are read back by ffprobe and by GStreamer's `avidemux` byte for byte.
  - `an_ffmpeg_opendml_file_over_one_gib_matches_ffprobe` (`--ignored`) writes a 1.6 GB, two-segment FFmpeg file and checks every packet; it passed on 12 November 2026 at commit `fcecd917`, in 12.8 s.
- `tests/external/local.rs` copies up to two clips per codec from `../local-files` (outside the repository) into AVI with `ffmpeg -c copy -t 10`, checks them against ffprobe, seeks each stream to before the start, the start, a third, half, just before the end and past the end without reading a payload, and remuxes them through `AviMux`. 14 clips (H.264, HEVC, AAC, MJPEG cover art, PCM) passed on 12 November 2026. The test checks nothing when the folder is absent.
- `tests/external/fuzz.rs`: `fuzz_seeds_do_not_panic` runs 199 fixed seeds on every test run. `fuzz_campaign` (`--ignored`; `AVI_FUZZ_SECS`, default 1800; `AVI_FUZZ_SEED`) generates random bytes, random chunk layouts and real muxed movies, mutates them, and drives probe, open, every read and seeks under `catch_unwind`; it also hands the muxer random packet lists and checks that a finished file reads back with every packet. The campaign that was run is in the readiness section.

## Speed against FFmpeg

Measured on 12 November 2026 at commit `dd8eaa7a`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1 (libavformat). Built with `cargo build --release` and rustc 1.99.1. Each row is the median of five runs after one warm-up; each run is a whole process under `/usr/bin/time -l`, start-up included on both sides (`/usr/bin/true` costs 8.7 M instructions). Instructions retired are the comparison; wall times on this shared machine are only a guide.

Inputs, made with FFmpeg:

- `h264aac.avi`: a 530 s 1080p H.264 + AAC clip copied into AVI, 77 MB, one segment, `idx1` only. FFmpeg's 600 Hz video clock gives 341,132 `idx1` entries, 302,367 of them empty chunks; 38,765 packets.
- `mjpeg_pcm.avi`: 300 s of 320×240 MJPEG at 30 fps with 48 kHz stereo s16 PCM, 149 MB, 23,063 packets.
- `odml.avi`: `h264aac.avi` remuxed by `AviMux` with 16 MiB segments: five `RIFF`s, read through `indx` and `ix##`.

The Rust side is `examples/bench.rs` (and `tools/media-bench demux`, the shared harness); the FFmpeg side is `benches/ffmpeg_avi.c` (and `scripts/ffmpeg_decode.c demux`). Both read every packet and touch its bytes. Packet counts, byte totals and, for seeks, the bytes of every landing packet match between the sides.

| Case | Cutlass (M instructions) | FFmpeg (M instructions) | Cutlass / FFmpeg | Peak memory, Cutlass / FFmpeg |
| --- | ---: | ---: | ---: | --- |
| h264aac, open ×100 | 5,828 | 22,151 | 0.26 | 5.2 / 15.1 MB |
| h264aac, read every packet | 151 | 969 | 0.16 | 6.0 / 13.2 MB |
| h264aac, read (media-bench vs `ffmpeg_decode.c`) | 155 | 1,028 | 0.15 | 6.4 / 21.0 MB |
| h264aac, 2,000 seeks, one packet read after each | 168 | 803 | 0.21 | 5.3 / 12.9 MB |
| mjpeg_pcm, open ×100 | 741 | 3,116 | 0.24 | 4.7 / 12.5 MB |
| mjpeg_pcm, read every packet | 128 | 511 | 0.25 | 4.0 / 11.1 MB |
| mjpeg_pcm, read (media-bench vs `ffmpeg_decode.c`) | 131 | 515 | 0.25 | 4.5 / 11.9 MB |
| mjpeg_pcm, 2,000 seeks | 69 | 355 | 0.20 | 4.0 / 11.1 MB |
| odml, open ×100 | 1,171 | 4,068 | 0.29 | 6.3 / 13.0 MB |
| odml, read every packet | 103 | 598 | 0.17 | 4.8 / 12.5 MB |
| odml, read (media-bench vs `ffmpeg_decode.c`) | 107 | 657 | 0.16 | 5.2 / 20.5 MB |
| odml, 2,000 seeks | 121 | 624 | 0.19 | 4.6 / 13.1 MB |
| h264aac, remux to AVI | 247 | 1,362 | 0.18 | 7.3 / 103.7 MB |
| mjpeg_pcm, remux to AVI | 279 | 754 | 0.37 | 5.4 / 187.1 MB |

Cutlass retires 2.7 to 6.7 times fewer instructions than libavformat in these cases. Wall times were 0.01 to 0.21 s for Cutlass and 0.02 to 1.15 s for FFmpeg.

What differs between the sides:

- `benches/ffmpeg_avi.c` does not call `avformat_find_stream_info` for open, read and seek, because it decodes frames; `scripts/ffmpeg_decode.c` does call it, which is the extra in its rows. The remux calls it too: FFmpeg's AVI muxer needs the audio frame size it finds.
- FFmpeg returns 1024-block PCM packets, runs the parsers its AVI demuxer asks for inside `av_read_frame`, and builds its own index tables; Cutlass returns one packet per chunk. The packet counts above are equal because no PCM chunk in `mjpeg_pcm` is longer than 1024 blocks.
- FFmpeg's remux writes the 302,367 empty video chunks it read back out (77 MB); Cutlass folds them into frame durations, writes none for this clip, and its output is 70 MB. ffprobe reads 38,765 packets from both outputs.
- Opening read 136 to 333 KB of the local clips (64 KiB read-ahead blocks over the header and the index); a seek reads nothing.

To run it again: `cargo build --release -p cutlass-avi --examples` and `cargo build --release -p media-bench`, build the C harness as its header says, make the three inputs as above, and time each pair with `/usr/bin/time -l`.

## Limits

| Limit | Value |
| --- | --- |
| Streams | 64 (`Limits::standard`); 100 by the chunk-id format |
| Header bytes (`hdrl`) | 8 MiB |
| One packet | 16 MiB |
| Extradata | 1 MiB |
| Index entries | 16 M |
| Chunks at one level of `hdrl` or at the top of one segment | 4,096 |
| `RIFF` segments | 65,536 |
| Super index entries | 65,536 |
| Resync window in a scan | 1 MiB |
| Bytes read to sniff a keyframe | 4 KiB |
| Empty chunks for one muxed gap | 60,000 |

## Readiness

Fuzzing, 12 November 2026, `cargo test --release` builds of `fuzz_campaign`, seed origin `0xa51000` both times:

| Code | Build | Time | Seeds | Files opened | Packets read | Mux runs read back | Slowest seed | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dd8eaa7a` (the demux and mux code as shipped) | release with overflow checks and debug assertions | 2,100 s | 222,156,780 | 42,549,697 | 856,754,265 | 5,060,562 | 267 ms | No panic, no failure |
| `d26b5ba4` (before the dropped-frame and decode-time changes) | release | 2,100 s | 240,161,337 | 45,999,757 | 926,205,237 | 4,565,121 | 157 ms | No panic, no failure |

The two runs start from the same seed origin, so the second walked the same generated inputs over newer code rather than new inputs. Every mux run in them is bounded mux fuzzing: a random packet list either is refused or finishes, and a finished file reads back with every packet. This is an in-process generator and mutator, not a coverage-guided fuzzer; `cargo fuzz` was not used.

Demux is ready for the formats marked verified above: each was checked packet for packet against ffprobe or GStreamer, the malformed-input tests and the fuzz campaign found no panic, and opening, reading and seeking retired 0.15 to 0.29 times libavformat's instructions in the benchmark. Mux writes OpenDML files that ffprobe and GStreamer read back byte for byte for H.264, MJPEG, PCM, MP3 and AAC.

Gaps:

- Raw video is demuxed as an unknown codec and is not muxed. Audio other than PCM, MP3, AAC and AC-3 (E-AC-3, FLAC, Opus, ADPCM, ...) is not muxed.
- A CBR MP3 or AC-3 chunk that holds several frames is returned as one packet; it is not split into frames.
- VP9 and AV1 in AVI, and float PCM and AC-3 from the muxer, are covered by unit tests only.
- The muxer writes packets in arrival order; it does not interleave. It holds packets until every video stream has 8, so the first header write may hold up to 64 MiB.
- No muxed file over 1 GiB was written in a test; segment switching was checked with small segments.
- An index that points outside `movi` is replaced by a scan and recorded in `fallbacks()`, not reported as an open error.
- The engine does not play 8-bit PCM, and the export path does not write AVI.

## Specifications

- Microsoft, AVI RIFF File Reference (`AVIMAINHEADER`, `AVISTREAMHEADER`, `AVIOLDINDEX`, `AVISUPERINDEX`, `AVISTDINDEX`).
- OpenDML AVI File Format Extensions 1.02.
- Microsoft `BITMAPINFOHEADER`, `WAVEFORMATEX`, `WAVEFORMATEXTENSIBLE`, `MPEGLAYER3WAVEFORMAT`.
