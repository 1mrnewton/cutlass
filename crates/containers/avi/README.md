# cutlass-avi

`cutlass-avi` reads and writes AVI files: classic AVI 1.0 and OpenDML AVI 2.0 (`AVIX` segments, `indx` super indexes, `ix##` standard indexes, files over 1 GiB). It demuxes a file into packets and muxes packets back into a file. It does not decode or convert payloads. The crate is safe Rust (`#![forbid(unsafe_code)]`) and runs no foreign library, subprocess, or OS parser.

The desktop app and the FFI readers open AVI files with `AviDemux`, and `cutlass-probe` reports an AVI file's picture size and length. `cutlass_export::write_movie` writes an AVI with `AviMux` when the request's container is `ContainerId::Avi`; the desktop export panel does not offer that choice yet.

## What it supports

Timestamps stay in each stream's own clock: the timescale is `dwRate / g` and one AVI unit (a frame, a sample block or an audio frame) is `dwScale / g` ticks, `g` being their greatest common divisor. An ffprobe tick is one AVI unit, so ffprobe's numbers times `dwScale / g` are this crate's.

| Capability | Demux | Mux | Evidence | Status |
| --- | --- | --- | --- | --- |
| AVI 1.0, one `RIFF AVI ` with `idx1` | Opens | Written as the first segment, `idx1` included | ffmpeg cases, GStreamer `avimux`, `relative_and_absolute_idx1_give_the_same_packets` | Verified |
| OpenDML 2.0: `indx`, `ix##`, `AVIX`, `dmlh` | `indx` is read first; a stream without one, or a broken one, falls back to `idx1` and a scan of the `AVIX` segments | Every stream has an `indx`, every segment an `ix##` per stream; a new `AVIX` starts at `riff_bytes` (1 GiB by default) | `written_opendml_files_read_back_in_ffprobe_and_gstreamer`, `an_ffmpeg_opendml_file_over_one_gib_matches_ffprobe` (1.6 GB from FFmpeg, two segments, `--ignored`) | Verified, `a_muxed_file_over_one_gib_reads_back_in_ffprobe_and_gstreamer` (1.2 GB from `AviMux`, two segments, `--ignored`) included |
| `idx1` offsets from `movi` or from the file start | Both; the base is picked by checking the first entries against the chunks they name | From `movi` | `relative_and_absolute_idx1_give_the_same_packets` | Verified (in-memory files) |
| Missing or broken index | A scan of `movi` rebuilds it: it reads chunk headers, descends `LIST rec `, resyncs over garbage up to 1 MiB, and reads up to 4 KiB of a video chunk to tell a keyframe (H.264, HEVC, MPEG-4, MPEG-2, VP8, VP9, AV1; MJPEG and raw are all keyframes). `fallbacks()` names the index that was dropped and why, and `repair()` returns a `Repair` (dropped indexes, resyncs, a cut) whose `Display` reads as a warning; `open_strict` refuses such a file | Not applicable | `a_scan_reads_keyframes_from_the_bitstream_and_skips_rec_lists`, `an_idx1_pointing_outside_movi_falls_back_to_a_scan`, `garbage_between_chunks_is_skipped_by_a_resync`, `a_cut_recording_reads_up_to_the_cut` | Verified (in-memory files) |
| `LIST rec ` | Read | Not written | as above | Verified (in-memory files) |
| Keyframes | `AVIIF_KEYFRAME` in `idx1`, the delta bit of an `ix##` size, or the scan | Written in both indexes | ffmpeg and GStreamer cases | Verified |
| Dropped frames (empty video chunks) | Lengthen the packet before them | A gap in decode time, or a packet longer than one frame, writes empty chunks (60,000 per gap at most) | `an_empty_chunk_is_a_dropped_frame`, `durations_of_several_frames_write_dropped_frames`, local clips copied by FFmpeg at 600 Hz | Verified |
| Interleaving | An interleaved file is read in byte order; one whose streams drift more than 2 s apart is read in time order | The muxer puts packets in time order across streams itself, holding up to `interleave_bytes` (16 MiB by default); past that the earliest held packet is written and `FinishedMux::interleaved()` is false | `a_non_interleaved_file_reads_in_time_order`, `the_muxer_puts_streams_sent_in_runs_in_time_order`, `past_the_interleave_limit_packets_go_out_and_the_file_says_so` | Verified |
| Seek | From the index, without reading a payload (`consumed()` does not change) | Not applicable | `seeks_land_from_the_index_without_reading_payloads`, local clips | Verified |
| H.264 (`H264`, `avc1`, `X264`, `DAVC`, ...) | Annex B or `avcC` extradata as stored; decode time only | `H264`, or `avc1` when the config is an `avcC` record | ffmpeg (B-frames), local clips, remux | Verified |
| HEVC (`HEVC`, `H265`, `hvc1`, ...) | Decode time only | `HEVC` | ffmpeg (`-tag:v HEVC`), local clips | Verified |
| MPEG-4 Part 2 (`FMP4`, `XVID`, `DIVX`, `DX50`, ...) | Decode time only | `FMP4` | ffmpeg `mpeg4` | Verified |
| MPEG-1/2 (`MPG2`, `MPG1`, ...) | Decode time only | `MPG2` | ffmpeg `mpeg2video` | Verified |
| MJPEG (`MJPG`, `AVI1`, ...) | Decode and presentation time; all keyframes | `MJPG` | ffmpeg, GStreamer, a muxed file read by both | Verified |
| VP8 (`VP80`) | Decode and presentation time | `VP80` | ffmpeg `libvpx` | Verified |
| VP9 (`VP90`), AV1 (`AV01`) | Decode and presentation time | `VP90`, `AV01` | ffmpeg `libvpx-vp9` + AC-3 and `libsvtav1` + PCM, packet for packet; both copied through `AviMux` and read back by ffprobe and GStreamer | Verified |
| Raw video (`BI_RGB`, `BI_BITFIELDS`, `I420`, `IYUV`, `YV12`, `NV12`, `NV21`, `YUY2`, `UYVY`, `YVYU`, `Y800`, `GREY`, ...) | `CodecId::RawVideo`; the extradata is the whole `BITMAPINFOHEADER` (palette included), and `raw_layout` gives the FourCC, bit depth, size, row order, frame bytes and, for `I420`/`IYUV` and `Y800`/`GREY`, a `PixelFormat` | `CodecId::RawVideo` with that header as the config, `db` chunks; a frame size that does not match the header is refused | ffmpeg `rawvideo` bgr24 and yuv420p, packet for packet, and copied through `AviMux` into ffprobe and GStreamer | Verified |
| Integer PCM, 8 to 32 bits | One chunk per packet | Constant bit rate, whole sample frames | ffmpeg s16, s24 (extensible), GStreamer, a muxed file | Verified |
| Float PCM | One chunk per packet | As integer PCM | ffmpeg `pcm_f32le`, and copied through `AviMux` into ffprobe and GStreamer | Verified |
| `WAVE_FORMAT_EXTENSIBLE` | Read; the `strf` bytes after `WAVEFORMATEX` are the extradata | Written by `cutlass_riff`'s rules (more than two channels, a valid-bit width that differs from the stored width, ...) | ffmpeg s24 | Verified |
| MP3 (`0x0055`), MPEG-1 Layer I/II (`0x0050`, `wav:0050`) | One packet per frame: a chunk holding several frames, or ending inside one, is cut at the frame headers as FFmpeg's parser cuts it; constant-bitrate frames take the time of their first byte, variable-bitrate ones the chunk's time and then each frame's samples | `0x0055` with `MPEGLAYER3WAVEFORMAT`, `0x0050` with `MPEG1WAVEFORMAT`; one frame per chunk | ffmpeg `libmp3lame` and `mp2`, `chunks_of_several_frames_match_ffmpeg_parsers` (fixed 1000- and 4608-byte chunks against ffprobe's frames), muxed files | Verified |
| AAC (`0x00FF`, ...) | One chunk per packet; the `strf` extra is the AudioSpecificConfig | `0x00FF`, one frame per chunk; needs the AudioSpecificConfig | ffmpeg, local clips, a muxed file | Verified |
| AC-3 (`0x2000`) | One packet per frame, cut as for MP3 | `0x2000`, one frame per chunk | ffmpeg `ac3`, `chunks_of_several_frames_match_ffmpeg_parsers` (2000-byte chunks), and muxed with VP9 into ffprobe and GStreamer | Verified |
| E-AC-3 (`WAVE_FORMAT_EXTENSIBLE`, subformat `A7FB87AF-2D02-42FB-A4D4-05CD93843BDD`) | One packet per frame. FFmpeg writes E-AC-3 with `nBlockAlign` 2 and `dwScale` 128, which makes ffprobe's own times wrong; this crate counts one unit per chunk for AC-3 and E-AC-3 when `nBlockAlign` is 4 or less | The same extensible header FFmpeg writes, one frame per chunk | ffmpeg `eac3` + AV1 copied through `AviMux` into ffprobe and GStreamer | Verified |
| FLAC (`0xF1AC`) | The `strf` extra (`STREAMINFO`) is the extradata | `0xF1AC` with `STREAMINFO`, one frame per chunk, as FFmpeg writes it | ffmpeg `flac` + MJPEG copied through `AviMux` into ffprobe and GStreamer | Verified. Not in the packet-for-packet FFmpeg demux cases: for FLAC in AVI, `framecrc` reports one packet more than ffprobe |
| A-law (`0x0006`), mu-law (`0x0007`) | Unknown codec tags `wav:0006`, `wav:0007`, one chunk per packet | Constant bit rate, as PCM | ffmpeg `pcm_alaw`, `pcm_mulaw` copied through `AviMux` into ffprobe and GStreamer | Verified |
| Other audio tags (ADPCM, Vorbis, ...) | Unknown codec tag `wav:XXXX`, packets read | Refused (see below) | `unknown_codecs_are_published_and_read` | Demux only |
| `txts` and other stream types | Unknown codec tag, packets read | Not written | `unknown_codecs_are_published_and_read` | Demux only |
| Forward-only input | `AviForwardDemux::open(reader)` takes any `Read` and reads chunks in file order, `AVIX` segments included, with no index: keyframes come from the bitstream, a video packet is returned once its stream's next chunk arrives (so dropped frames lengthen it), and seeking is `SeekError::ForwardOnly` | The muxer needs `Write + Seek` to patch headers | `forward_reading_*`; every ffmpeg case read both ways gives the same packets | Demux verified; mux unsupported |

`FEATURES` in the crate lists the same rows as data.

Errors, not panics:

- A size that runs past its parent or the file, or a stream count, super index, chunk count, header or extradata over the limits below: `OpenError::HostileSize` or another open error. A broken index is not an open error while a scan can rebuild it.
- A chunk whose header does not match the index, or a packet over the packet limit: that read returns an error and the next read goes on with the following chunk.
- A truncated file reads up to the cut.
- In the muxer: audio other than PCM, A-law, mu-law, MP3, MPEG-1 Layer II, AAC, AC-3, E-AC-3 and FLAC; raw video whose frame size does not match its header; a PCM packet that is not whole sample frames or not contiguous; VBR audio frames of differing length or over 65,535 bytes; a stream with no packet when the header has to be written; a stream that starts more than an hour in; and track durations that do not equal the packets written. Each is `MuxError::Rejected` or `MuxError::DurationMismatch`.

## Using it

- `probe` and `AviFamily` decide from the first 12 bytes (`RIFF`, size, `AVI `).
- `AviDemux::open(reader, length)` takes a `Read + Seek` reader and the file length and implements `cutlass_container::Demux`. `index_source()`, `fallbacks()`, `resyncs()`, `truncated()`, `interleaved()`, `riff_count()`, `main_header()` and `stream_header(track)` say what the file held and how it was read; `repair()` sums up a damaged file in one value a caller can show as a warning, and `open_strict(reader, length)` refuses such a file instead.
- `AviForwardDemux::open(reader)` takes any `Read` (a pipe, a download) and reads once, front to back, with no index; it cannot seek.
- `raw_layout(extradata)` reads a raw video stream's `BITMAPINFOHEADER`.
- `AviMux::new(writer)` or `AviMux::with_options(writer, AviMuxOptions { .. })` takes a `Write + Seek` writer and implements `cutlass_container::Mux`. The header is written once every video stream has 8 packets and every audio stream one (`header_buffer_bytes`, 64 MiB, is held at most), so the frame rate comes from real times. After that the muxer holds up to `interleave_bytes` (16 MiB) to put the streams in time order, so a caller may send each stream in runs.

Examples:

- `cargo run --release -p cutlass-avi --example packets -- file.avi` prints the streams and every packet (track, decode, presentation, duration, size, keyframe), in the shape the integration tests compare with ffprobe.
- `cargo run --release -p cutlass-avi --example remux -- in.avi out.avi [riff-mib]` copies an AVI file through `AviDemux` and `AviMux`; a small `riff-mib` makes many `AVIX` segments.
- `cargo run --release -p cutlass-avi --example bench -- <open|read|seek> file.avi <count>` is the Rust side of the benchmark below.

## How it is tested

```sh
cargo test -p cutlass-avi
cargo test -p cutlass-avi --test external -- --ignored   # the two files over 1 GiB and the fuzz campaign
```

- `src/behavior.rs` builds files in memory: relative and absolute `idx1`, a scan with `rec ` lists and bitstream keyframes, an index outside `movi`, a cut file, garbage between chunks, hostile headers, unknown codecs and a text stream, dropped frames, VBR MP3, a non-interleaved file, seek without payload reads, cancel, a misnamed chunk, an oversized packet, MP3 frames cut out of chunks, forward-only reading (the same packets as the indexed reader on every hand-built and many-`RIFF` file, every cut and every flipped byte of a muxed file without a panic), and the muxer: round trips, many `RIFF` segments whose `indx`, `ix##` and `idx1` agree, dropped frames, constant-rate PCM, refusals, a config handed over with the first packet, times off the frame grid, and streams sent in long runs put in time order.
- `tests/external/oracle.rs`, with FFmpeg 8.1.1 and GStreamer 1.28.3 as independent tools:
  - FFmpeg writes nine files (MJPEG + PCM s16, H.264 with B-frames + AAC, MPEG-4 + MP3, MPEG-2 + AC-3, VP8 + float PCM, HEVC + s24 extensible PCM, raw video + PCM, VP9 + AC-3, AV1 + PCM). Every packet must match ffprobe and `framecrc` for size, Adler-32, keyframe flag, decode time and duration. PCM is compared as one byte stream, because FFmpeg splits a PCM chunk into 1024-block packets. Each file is also read by `AviForwardDemux` and must give the same packets, keyframes included.
  - `tests/external/framed.rs`: FFmpeg writes MP3, MP2 and AC-3 elementary streams; the test stores them in fixed-size chunks that hold several frames and end inside one (as old capture tools did), and every frame must match ffprobe's parsed frames for size, Adler-32 and time (155, 167 and 125 frames; times within 2, 0 and 0 ticks of a 16,000 or 24,000 Hz byte clock, because FFmpeg adds each frame's rounded duration while this crate takes each frame's first byte).
  - FFmpeg files with raw bgr24 + float PCM, raw yuv420p + s24, VP9 + AC-3, AV1 + E-AC-3 and MJPEG + FLAC, and with MJPEG + MP2 + A-law + mu-law, are copied through `AviMux` and read back by ffprobe and GStreamer.
  - A GStreamer `avimux` file matches ffprobe the same way.
  - Files `AviMux` writes, in one segment and in many, and an MJPEG + PCM + MP3 file, are read back by ffprobe and by GStreamer's `avidemux` byte for byte.
  - `an_ffmpeg_opendml_file_over_one_gib_matches_ffprobe` (`--ignored`) writes a 1.6 GB, two-segment FFmpeg file and checks every packet; it passed on 12 November 2026 at commit `fcecd917`, in 12.8 s.
  - `a_muxed_file_over_one_gib_reads_back_in_ffprobe_and_gstreamer` (`--ignored`) has `AviMux` write 1,246,492,384 bytes (6,000 H.264-shaped pictures of about 200 KB and PCM, two segments) straight to disk; the indexed and the forward reader, ffprobe (size, Adler-32, keyframe and time of every picture, the time of every PCM piece) and GStreamer (every byte of both streams) read it back. It passed on 12 November 2026 at commit `f3acddcb`.
- `crates/export` (`a_movie_exports_as_avi_that_ffprobe_and_gstreamer_read_back`): H.264 + AAC and MPEG-4 Visual + MP3 exported through `write_movie` as AVI; this crate, ffprobe (codec names and packet counts) and GStreamer (every picture and sound buffer, byte for byte) read them back, and FFmpeg decodes them with `-xerror` and no message.
- `tests/external/local.rs` copies up to two clips per codec from `../local-files` (outside the repository) into AVI with `ffmpeg -c copy -t 10`, checks them against ffprobe, seeks each stream to before the start, the start, a third, half, just before the end and past the end without reading a payload, and remuxes them through `AviMux`. 14 clips (H.264, HEVC, AAC, MJPEG cover art, PCM) passed on 12 November 2026. The test checks nothing when the folder is absent.
- `tests/external/fuzz.rs`: `fuzz_seeds_do_not_panic` runs 199 fixed seeds on every test run. `fuzz_campaign` (`--ignored`; `AVI_FUZZ_SECS`, default 1800; `AVI_FUZZ_SEED`) generates random bytes, random chunk layouts and real muxed movies, mutates them, and drives probe, open, every read and seeks under `catch_unwind`; every input also goes through `AviForwardDemux`; it also hands the muxer random packet lists (with a random interleave limit) and checks that a finished file reads back with every packet. The campaign that was run is in the readiness section.

## Speed against FFmpeg

Measured on 12 November 2026 at commit `a48845bf`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1 (libavformat). Built with `cargo build --release` and rustc 1.99.1. Each row is the median of five runs after one warm-up; each run is a whole process under `/usr/bin/time -l`, start-up included on both sides (`/usr/bin/true` costs 8.5 M instructions). Instructions retired are the comparison; wall times and peak memory on this shared machine are only a guide (the same binary's peak memory for 100 opens varied from 8.0 to 10.7 MB between runs).

Inputs, made with FFmpeg:

- `h264aac.avi`: a 530 s 1080p H.264 + AAC clip copied into AVI, 77 MB, one segment, `idx1` only. FFmpeg's 600 Hz video clock gives 341,132 `idx1` entries, 302,367 of them empty chunks; 38,765 packets.
- `mjpeg_pcm.avi`: 300 s of 320×240 MJPEG at 30 fps with 48 kHz stereo s16 PCM, 149 MB, 23,063 packets.
- `odml.avi`: `h264aac.avi` remuxed by `AviMux` with 16 MiB segments: five `RIFF`s, read through `indx` and `ix##`.

The Rust side is `examples/bench.rs` and `examples/remux.rs` (and `tools/media-bench demux`, the shared harness); the FFmpeg side is `benches/ffmpeg_avi.c` (and `scripts/ffmpeg_decode.c demux`). Both read every packet and touch its bytes. Packet counts, byte totals and, for seeks, the bytes of every landing packet match between the sides.

| Case | Cutlass (M instructions) | FFmpeg (M instructions) | Cutlass / FFmpeg | Peak memory, Cutlass / FFmpeg |
| --- | ---: | ---: | ---: | --- |
| h264aac, open ×100 | 5,947 | 22,142 | 0.27 | 10.0 / 14.1 MB |
| h264aac, read every packet | 155 | 970 | 0.16 | 5.8 / 12.8 MB |
| h264aac, read (media-bench vs `ffmpeg_decode.c`) | 158 | 1,028 | 0.15 | 6.0 / 20.8 MB |
| h264aac, 2,000 seeks, one packet read after each | 169 | 803 | 0.21 | 5.3 / 12.6 MB |
| mjpeg_pcm, open ×100 | 757 | 3,115 | 0.24 | 7.8 / 12.4 MB |
| mjpeg_pcm, read every packet | 129 | 511 | 0.25 | 4.0 / 11.1 MB |
| mjpeg_pcm, read (media-bench vs `ffmpeg_decode.c`) | 133 | 515 | 0.26 | 4.4 / 11.9 MB |
| mjpeg_pcm, 2,000 seeks | 69 | 355 | 0.20 | 4.0 / 11.0 MB |
| odml, open ×100 | 1,186 | 4,070 | 0.29 | 6.2 / 12.2 MB |
| odml, read every packet | 106 | 598 | 0.18 | 4.7 / 12.7 MB |
| odml, read (media-bench vs `ffmpeg_decode.c`) | 109 | 657 | 0.17 | 5.2 / 20.7 MB |
| odml, 2,000 seeks | 122 | 623 | 0.20 | 4.4 / 12.6 MB |
| h264aac, remux to AVI | 276 | 1,357 | 0.20 | 8.0 / 106.0 MB |
| mjpeg_pcm, remux to AVI | 322 | 757 | 0.43 | 5.4 / 187.1 MB |

Cutlass retires 2.3 to 6.7 times fewer instructions than libavformat in these cases.

Against the first measurement (commit `dd8eaa7a`): opening `h264aac` costs 2% more (5,947 against 5,828 M; the frame-cutting bookkeeping on every index row), and remuxing costs 11 to 15% more (276 against 247 M, 322 against 279 M) because the muxer now interleaves: `Mux::write_packet` lends a packet, so one that has to wait is copied. Measured with the copy switched off, the same remux costs 249 and 284 M. A caller that already sends packets in time order can set `interleave_bytes` to 0 and skip the copy.

`scripts/bench_media.py`'s `avi` case (`--modes demux --only demux:avi`), run at commit `f3acddcb`: FFmpeg copies the shared H.264 + AAC test clip into a 300 MB `long.avi` (57,064 packets), and `media-bench demux` reads it in 326 M instructions against 1,445 M for `ffmpeg_decode.c` (0.23), 0.03 s against 0.10 s, 8.4 MB against 22 MB at peak.

What differs between the sides:

- `benches/ffmpeg_avi.c` does not call `avformat_find_stream_info` for open, read and seek, because it decodes frames; `scripts/ffmpeg_decode.c` does call it, which is the extra in its rows. The remux calls it too: FFmpeg's AVI muxer needs the audio frame size it finds.
- FFmpeg returns 1024-block PCM packets, runs the parsers its AVI demuxer asks for inside `av_read_frame`, and builds its own index tables; Cutlass returns one packet per chunk. The packet counts above are equal because no PCM chunk in `mjpeg_pcm` is longer than 1024 blocks.
- FFmpeg's remux writes the 302,367 empty video chunks it read back out (77 MB); Cutlass folds them into frame durations, writes none for this clip, and its output is 70 MB. ffprobe reads 38,765 packets from both outputs.
- Opening read 136 to 333 KB of the local clips (64 KiB read-ahead blocks over the header and the index); a seek reads nothing.

To run it again: `cargo build --release -p cutlass-avi --examples` and `cargo build --release -p media-bench`, build the C harness as its header says, make the three inputs as above, and time each pair with `/usr/bin/time -l`. For the shared harness: `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder> --modes decode,demux --only decode:h264,decode:aac,demux:avi` (the decode cases make the clip the `avi` case copies).

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

Gaps and choices:

- Audio the muxer refuses, and why. IMA ADPCM (`0x0011`) and MS ADPCM (`0x0002`) need their `nBlockAlign` and samples per block in the header, which `StreamParams` cannot carry yet. Opus and ALAC: FFmpeg's AVI muxer refuses both and its `riff.c` has no WAVE tag for Opus, so there is no mapping that other readers know. Vorbis has no registered tag, and the two tools disagree: FFmpeg writes `0x566F` (`'Vo'`, which its `riff.c` calls a hack), and GStreamer 1.28.3 reports such a file's sound as `audio/x-avi-unknown` (checked here); GStreamer reads the old Xiph tags `0x674F` to `0x6771`, which FFmpeg does not write. So Vorbis is not written. Each is `MuxError::Rejected`.
- A damaged file (an index outside `movi`, garbage, a cut) opens, as it does in FFmpeg and GStreamer, and `repair()` says so. The crate does not log; the desktop app does not show an open-time warning yet, because it has no place for one.
- E-AC-3 from FFmpeg's own AVI muxer: times here follow the chunks, where ffprobe's follow FFmpeg's wrong `dwScale` and differ.
- The muxer needs `Write + Seek`: it patches the headers and writes the indexes at the end.
- The engine does not play 8-bit PCM (the PCM codec is being fixed separately), and the desktop export panel does not offer AVI.

## Specifications

- Microsoft, AVI RIFF File Reference (`AVIMAINHEADER`, `AVISTREAMHEADER`, `AVIOLDINDEX`, `AVISUPERINDEX`, `AVISTDINDEX`).
- OpenDML AVI File Format Extensions 1.02.
- Microsoft `BITMAPINFOHEADER`, `WAVEFORMATEX`, `WAVEFORMATEXTENSIBLE`, `MPEGLAYER3WAVEFORMAT`.
