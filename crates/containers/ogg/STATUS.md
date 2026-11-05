# Ogg support record

Branch `opus-ogg`. Native safe Rust in `cutlass-ogg`. `#![forbid(unsafe_code)]`, workspace lints. No bundled C, FFmpeg, or GStreamer: both are test oracles only. The crate depends on `cutlass-codec` and `cutlass-container`.

Demux and mux of Ogg (RFC 3533) with the Opus mapping (RFC 7845) are in scope. Other Ogg mappings are listed and skipped. Opus in WebM or Matroska is out of scope.

## Specifications and sources

Followed:

- RFC 3533, The Ogg Encapsulation Format: page header, lacing, header type flags, page sequence numbers, grouping of beginning-of-stream pages, chaining.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec: `OpusHead` and `OpusTags`, channel mapping families, granule position and pre-skip (sections 4.1 to 4.5), end trimming (4.4), the initial granule rules (4.5), seeking with 80 ms pre-roll (4.6), the 61,440-byte packet limit (6).
- RFC 6716 section 3.1: the TOC byte and frame-count byte that give a packet's duration.

Read before the implementation, and not copied:

- `../pub-packages/ffmpeg/libavformat/oggdec.c`, `oggdec.h`, `oggparseopus.c`, `oggenc.c`. FFmpeg stores the checksum of `"OggS"` byte-swapped as `0x4fa9b05f`; the RFC value computed here is `0x5FB0A94F`, checked against a real ffmpeg page. FFmpeg's packet `pts` is the granule minus pre-skip, and it gives a packet that end trimming removes whole a duration of 1.
- `../pub-packages/gstreamer/subprojects/gst-plugins-base/ext/ogg/gstoggstream.c`, `gstoggmux.c`, `gstoggdemux.c`. GStreamer reports each buffer's `offset_end` as the granule position after the packet.

## Timestamp contract

All times use timescale 48,000 in the granule domain.

- A packet's presentation and decode time are equal: the granule position of its first sample, before pre-skip. ffprobe `pts` plus pre-skip equals this value.
- The duration comes from the TOC. On the end-of-stream page the last packets are clamped so the stream ends at that page's granule; a packet removed whole has duration 0.
- Every packet is a keyframe.
- A granule that goes backwards is `OggError::GranuleBackwards`, returned as a read error.
- Each Opus stream has `Trim::new(Some(pre_skip), trailing)` and one `Edit` with `media_time = start + pre_skip` and `duration = end - pre_skip - start`, where `start` is the first packet's time and `end` the last page's granule. `trailing` is the samples the last page trims.
- RFC 7845 section 4.5 is applied to the first audio page with a completed packet: start is its granule minus its samples; a granule below the samples fails open unless the page is also the last, in which case start is 0 and the granule must be at least the pre-skip.
- A stream whose playable length (`end - start - pre_skip`) is negative fails open with `PreSkipPastEnd`. This is stricter than the RFC, which only names the single-page case.

## Decisions

| Case | Behavior |
| --- | --- |
| Chained file | Only the first link is returned. On a random-access input open finds where the link ends (by bisection when the last page is not the link's own) and `chained()` is true; reads stop there. A forward input stops at the first new beginning-of-stream page. |
| Bad page checksum | The page is skipped and counted (`crc_failures()`); the reader resyncs one byte at a time, at most 256 KiB, then fails with `NoSync`. |
| Page sequence gap | The packet that spans the gap is dropped, never glued. The time jump is recorded in `gaps()`; `lost_pages()` and `dropped_pieces()` count it. |
| Truncated final page | The complete packets before it are returned, then one `ReadError::Truncated`, then `None`. |
| Packet with a malformed TOC | Returned with duration 0. |
| Packet over the stream cap (61,440 bytes times the stream count, and the `Limits` packet cap) | One `ReadError::HostileSize`, then reading continues. |
| Missing or garbled `OpusHead` or `OpusTags` | Typed open error (`BadOpusHead`, `UnsupportedOpusVersion`, `BadOpusTags`, `MissingOpusTags`, `TagsPageNotFinished`). |
| `OpusTags` longer than the header limit | Cut at the limit; comments past the cut are ignored (`OpusTags.cut`). |
| Other mappings (Vorbis, FLAC, Theora, Speex, others) | Listed as `StreamCodec::Unknown("ogg:vorbis")` and so on, timescale 0, first packet as extradata. Their pages are skipped and counted in `ignored_pages()`. |
| Forward open | Same packets. The end is unknown: the edit duration is 0 and `trailing` is `None`. Seek is `SeekError::ForwardOnly`. |
| Seek | Bisects on page granules (interpolated for the first two probes, then midpoints) down to a 64 KiB window, then scans pages. Lands on the packet that holds `max(target - 3840, start)`. `Landing::Packet { time, preroll, byte_position, .. }` with `index_used: false`; `byte_position` is the page the read resumes from; `ordinal` is 0 at the start and `u32::MAX` elsewhere because Ogg has no packet count. Past the end is `Landing::End`. On failure the read position and queued packets are restored. At most 64 probes and 8 MiB. Other tracks resume at their first complete packet after the landing page. |
| Seek in a damaged file | When the pages before the target are lost the seek lands on the first packet after the hole, at its true time, which can be after the target. |
| `consumed()` | Bytes read by open and packet reads. Seek reads are in `last_seek()` and `total_read()`. |
| `cancel()` | Persistent. Later reads and seeks return `Cancelled`. |
| Typed reasons | `ReadError` and `OpenError` have no Ogg variants, so Ogg reasons are mapped to the nearest variant (`Source { message }` mostly) and the typed `OggError` is kept in `last_error()`. |

Mux:

| Case | Behavior |
| --- | --- |
| Codec | Opus only. `Unknown` is `MuxError::UnknownCodec`; any other known codec is `Rejected`. |
| Configuration | `OpusHead` from extradata, or `set_config` before the first packet. An MP4 `dOps` body is rewritten as `OpusHead`. The channel count must match the stream. No configuration by the first packet is an error. |
| Page order | Every stream's beginning-of-stream page (`OpusHead` alone, granule 0), then every stream's `OpusTags` page (vendor `cutlass`, granule 0), then audio. Streams cannot be added after the first packet. |
| Page policy | An audio page is closed before the next packet once a packet has completed on it and its body holds 4,096 bytes or its completed packets span 48,000 samples, and whenever its 255 lacing values are used. Packets continue across pages. Granule is the end of the last packet completed on the page, or -1. |
| Last page | Written by `finish` with the end-of-stream flag; an empty stream gets an empty end page with granule 0. A final packet shorter than its TOC becomes end trimming. |
| Cropped start with end trimming | RFC 7845 4.5 reads a first-and-last page as starting at granule minus samples, so the packets before the trimmed one are written on their own page. One trimmed packet alone starting after 0 is refused at finish. |
| Stream shorter than its pre-skip | Refused at finish. |
| Timing | Timescale 48,000. Each stream's packets must be contiguous; an overlap, gap, a duration longer than the TOC, a trimmed packet that is not the last, or a negative first time is `Rejected`. |
| Serials and sequences | Serial `0x6375746C` ("cutl") for the first stream, plus one per stream (`with_first_serial` changes the base). Page sequence starts at 0 per stream. |
| Failure | Any refused call fails the mux; `finish` then returns `FailedMux` and `status()` is `Failed`. A write error is `Rejected("ogg write failed: ...")`. |
| `finish` | Checks each track's duration against the packet sum, writes the end pages, flushes, and returns `FinishedMux::complete` with `IndexPlace::Absent` and `MuxMode::Progressive`. |
| `interleave` | Pages are written as packets arrive; it only checks the mux is building. |

## Limits

| Limit | Value |
| --- | --- |
| Page | 65,307 bytes (27 + 255 + 255 × 255) |
| Resync scan | 256 KiB, then `NoSync` |
| Opus packet | 61,440 bytes per Opus stream in the mapping, and `Limits::max_packet_bytes` |
| Logical streams | `Limits::max_tracks` (64), `TooManyStreams` above |
| Header packets | `Limits::max_header_bytes`; `OpusTags` above it is cut |
| Comments | 65,536 |
| Open start scan / tail scan | 8 MiB each |
| Seek and link-end bisection | 64 probes each |
| Seek pre-roll | 3,840 samples |

Every length read from the file is checked against these before a buffer is reserved.

## Capability matrix

| Capability | Demux | Mux | Evidence | Status |
| --- | --- | --- | --- | --- |
| Page checksum (poly `0x04C11DB7`, init 0, no reflection, field zeroed) | Checked | Written | `crc` unit tests against a bitwise CRC and `"OggS"`; ffmpeg and GStreamer read every muxed page | Verified |
| Lacing: 255 multiples, empty packets, packets over three pages | Joined | Written | `a_packet_across_three_pages_and_255_multiples_round_trip`, page unit tests | Verified |
| Opus family 0, mono and stereo | Read | Written | `ffmpeg_files_demux_to_the_same_packets_times_and_trim`, `ffmpeg_and_gstreamer_read_remuxed_files_like_the_source` | Verified against libavformat and GStreamer |
| Opus family 1 (5.1) | Read | Written | same tests, `libopus-5.1-family-1` | Verified against libavformat and GStreamer |
| Opus family 255 | Parsed | Accepted | `OpusHead` unit tests only | Partial: no independent file |
| Pre-skip, start, end trimming, edit | Applied | Written | ffmpeg and GStreamer decoded sample counts equal the edit duration for 8 encodes; remuxed files decode to the same counts | Verified |
| Frame sizes 2.5 to 120 ms, code 3 packets | Durations from TOC | Checked against TOC | 2.5, 10, 20, 60, 120 ms variants | Verified |
| Several streams in one file | Each Opus stream a track; FLAC listed as `ogg:flac` | Two streams interleaved | `a_multiplexed_file_keeps_each_opus_stream_and_lists_flac_as_unknown`, `ffmpeg_reads_two_opus_streams_from_one_mux` | Verified |
| Chained file | First link, `chained()` | One link | `a_chained_file_returns_its_first_link` (two ffmpeg files concatenated), `a_second_link_is_not_returned` | Verified. Later links are not read |
| Seek | Bisection, pre-roll | n/a | `every_seek_lands_before_the_preroll_and_returns_no_stale_packet`, oracle seeks on 8 encodes, the 30-minute measurement | Verified |
| Damaged input | Resync, gaps, typed errors | n/a | behavior tests and the mutation campaign | Verified for the cases listed |
| Forward input | Read, no seek | n/a | `a_forward_input_reads_the_same_packets_and_refuses_seek` | Verified |
| Vorbis, FLAC, Theora, Speex packets | Not returned | Refused | `an_unknown_mapping_is_listed_and_its_pages_skipped` | Unsupported |
| Opus in WebM or Matroska | n/a | n/a | n/a | Out of scope |
| Engine, ffi, and probe registration | n/a | n/a | Outside this crate | Not done here |
| Linux and Windows | Not run | Not run | This file was written on macOS | Not run |

## Commands and results

All on this machine: `Apple M5 Pro` (`sysctl -n machdep.cpu.brand_string`), macOS, ffmpeg 8.1.1 at `/opt/homebrew/bin`, GStreamer 1.28.3, 5 November 2026.

`cargo fmt -p cutlass-ogg`, `cargo clippy -p cutlass-ogg --all-targets -- -D warnings`, and `cargo test -p cutlass-ogg` pass: 64 unit and behavior tests, 2 mutation seed tests. The external tests skip without `local-files`.

### Independent producers

`CUTLASS_LOCAL_FILES=/Users/mrnewton/dev/cutlass/local-files cargo test -p cutlass-ogg --test external -- --nocapture`: 5 passed, 1 ignored (the measurement).

ffmpeg encodes each variant from a local file. For every packet this crate's demuxer matches ffprobe (`-show_packets -show_data_hash sha256`) on count, `pts + pre_skip`, size, and duration (except ffmpeg's 1 for a packet trimmed whole, where this crate says 0), and the concatenated bytes match `ffmpeg -c copy -f data`; the first and last packets' sha256 match. GStreamer `oggdemux` matches on count, size, and packet end granule. The edit duration equals the samples ffmpeg and GStreamer `opusdec` decode.

| Variant | Packets | Pre-skip | End granule | Trailing | Decoded (ffmpeg = GStreamer) |
| --- | --- | --- | --- | --- | --- |
| libopus stereo 96k | 1001 | 312 | 960312 | 648 | 960000 |
| libopus mono 32k, 60 ms | 334 | 312 | 960312 | 1608 | 960000 |
| libopus stereo 128k, 2.5 ms | 4001 | 120 | 480120 | 0 | 480000 |
| libopus 24k, 120 ms | 101 | 312 | 576312 | 5448 | 576000 |
| libopus voip 16k, 10 ms, 16 kHz input | 801 | 312 | 384312 | 168 | 384000 |
| ffmpeg native opus stereo 128k | 1001 | 120 | 960120 | 840 | 960000 |
| ffmpeg native opus mono 64k | 751 | 120 | 720120 | 840 | 720000 |
| libopus 5.1 family 1, 256k | 501 | 312 | 480312 | 648 | 480000 |

Each oracle run also seeks to 8 targets and past the end.

### Mux interop

Six of those files are demuxed by this crate and muxed again by `OggMux`, three with the `OpusHead` in extradata and three through `set_config`. For each, ffprobe reports the same stream (codec, rate, channels, `initial_padding`, extradata size) and the same packet list (pts, duration, size, sha256) for the remux as for the source; the `-f data` payloads are equal; ffmpeg and GStreamer decode the same sample count; and this crate reads back the same packets, trim, edit, and `OpusHead`, with vendor `cutlass`.

| Case | Packets | Decoded | Source bytes | Remux bytes |
| --- | --- | --- | --- | --- |
| libopus stereo 96k | 1001 | 960000 | 231452 | 232146 |
| libopus mono 60 ms | 334 | 960000 | 78625 | 78586 |
| libopus 120 ms | 101 | 576000 | 26462 | 26423 |
| native mono 64k | 751 | 720000 | 122413 | 122543 |
| native stereo 128k | 1001 | 960000 | 324276 | 325500 |
| libopus 5.1 | 501 | 480000 | 390057 | 391885 |

Two streams (stereo and mono, 1752 packets) muxed into one file give ffprobe the same per-stream packet lists as the two sources.

### Measurement

`CUTLASS_LOCAL_FILES=/Users/mrnewton/dev/cutlass/local-files cargo test --release -p cutlass-ogg --test external measure -- --ignored --nocapture`

`rain_bedroom_night_30min.mp3` encoded by libopus at 96 kb/s: 20,670,700 bytes, 89,927 packets, 1798.5 s. The machine was shared with other test runs (load average about 20), so times are upper bounds.

| Measure | Result |
| --- | --- |
| Open (best of 5) | 0.369 ms, 162,575 bytes read |
| Open to first packet | 0.398 ms |
| Full read | 52.8 ms, 392 MB/s, about 34,000× real time |
| 10 seeks, seek to first packet | mean 0.163 ms, worst 0.238 ms |
| Seek cost | at most 2 probes, 4 pages, and 131,072 bytes; 1 to 51 payload scans |

Every landing is the packet holding `target - 3840` in the full packet list; every seek after the start reports 5 pre-roll packets. libavformat's own seek (`ffprobe -read_intervals T%+#1`) lands at its page boundaries, 8,640 to 43,200 samples earlier than this crate. For a target of 0 it returned a first packet at 48,000; that was not investigated.

### Fuzzing

cargo-fuzz was not run. It is not installed, libfuzzer-sys builds C++ through `cc`, which this crate's rules do not allow, and a fuzz crate would need a workspace change outside this crate. `tests/fuzz.rs` runs in-process deterministic campaigns instead (xorshift per seed). Demux mutants come from muxed files with bit flips, byte writes, inserts, deletes, cuts, splices, duplicated ranges, and header-field hits, half with page checksums repaired; each is probed, opened random-access and forward, read to the end, and seeked, with bounded calls, packet limits, per-track time order, and the landing packet checked. Mux sessions use random streams, configurations, sizes up to 130,000 bytes, durations, and time jumps; a refused call must leave the mux failed and a finished mux must read back to exactly the accepted packets.

| Campaign | Command | Result |
| --- | --- | --- |
| Seeds (every test run) | `cargo test -p cutlass-ogg --test fuzz` | 256 demux mutants and 512 mux sessions, pass |
| Demux, 300 s | `OGG_FUZZ_SECS=300 cargo test --release -p cutlass-ogg --test fuzz fuzz_campaign_demux -- --ignored --nocapture` | 722,000 mutants over 14,440 seeds, no failure, slowest input 96.3 ms |
| Mux, 300 s | `OGG_FUZZ_SECS=300 cargo test --release -p cutlass-ogg --test fuzz fuzz_campaign_mux -- --ignored --nocapture` | 5,204,560 sessions, 275,824 finished and read back, no failure |

The campaigns found three muxer bugs, each fixed with a behavior test: a stream shorter than its pre-skip was written and then refused by the demuxer; a cropped start with end trimming on one page read back with the wrong start; and the same when the first packet spans pages. Both 300 s runs are on commit `46915201` (the demuxer is unchanged since `94c059fe`).

## Partial or not run

- Opus family 255 is parsed and accepted but not checked against an independent file.
- Vorbis, FLAC, Theora, and Speex mappings are listed, not demuxed.
- Chained files return the first link only.
- cargo-fuzz not run (reason above).
- Linux and Windows not run.
- Registration in the engine, probe list, and ffi is outside this crate.
