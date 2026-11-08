# Ogg support record

Branch `vorbis` (the Opus mapping was built on `opus`). Native safe Rust in `cutlass-ogg`. `#![forbid(unsafe_code)]`, workspace lints. No bundled C, FFmpeg, GStreamer, or libvorbis: they are test oracles only. The crate depends on `cutlass-codec` and `cutlass-container`, not on `cutlass-vorbis`: it walks the Vorbis setup header itself to learn each mode's block size.

Demux and mux of Ogg (RFC 3533) with the Opus mapping (RFC 7845) and the Vorbis mapping (Vorbis I specification, section A and section 4.2) are in scope. Other Ogg mappings are listed and skipped. Opus and Vorbis in WebM or Matroska are out of scope here.

## Specifications and sources

Followed:

- RFC 3533, The Ogg Encapsulation Format: page header, lacing, header type flags, page sequence numbers, grouping of beginning-of-stream pages, chaining.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec: `OpusHead` and `OpusTags`, channel mapping families, granule position and pre-skip (sections 4.1 to 4.5), end trimming (4.4), the initial granule rules (4.5), seeking with 80 ms pre-roll (4.6), the 61,440-byte packet limit (6).
- RFC 6716 section 3.1: the TOC byte and frame-count byte that give a packet's duration.
- Vorbis I specification: section 4.2 (identification, comment, and setup headers; the setup header is walked to its mode list, section 4.2.4), section 4.3.1 (the mode number in an audio packet), section 1.3.2 and 4.3.8 (a packet returns a quarter of each of its block and the block before; the first returns nothing), section A.2 (header pages, granule positions, a first page whose granule is below its samples starts before zero, end trimming on the last page), and section 5 (comment fields).

Read before the implementation, and not copied:

- `../pub-packages/ffmpeg/libavformat/oggdec.c`, `oggdec.h`, `oggparseopus.c`, `oggenc.c`. FFmpeg stores the checksum of `"OggS"` byte-swapped as `0x4fa9b05f`; the RFC value computed here is `0x5FB0A94F`, checked against a real ffmpeg page. FFmpeg's packet `pts` is the granule minus pre-skip, and it gives a packet that end trimming removes whole a duration of 1.
- `../pub-packages/gstreamer/subprojects/gst-plugins-base/ext/ogg/gstoggstream.c`, `gstoggmux.c`, `gstoggdemux.c`. GStreamer reports each buffer's `offset_end` as the granule position after the packet.
- Vorbis: `libavformat/oggparsevorbis.c` and `libavcodec/vorbis_parser.c` (packet durations from the mode list), the Vorbis part of `gstoggstream.c`, and `libavformat/oggenc.c` (the id header alone on the first page, then comment and setup, header granule 0). libavformat rebuilds the comment header from its metadata; this crate keeps the comment packet as given.

## Timestamp contract

Opus times use timescale 48,000 in the granule domain. Vorbis times use the stream's sample rate, and are sample positions (section A.2: the granule is the position of the last sample a packet completes).

- A packet's presentation and decode time are equal: the granule position of its first sample, before pre-skip. ffprobe `pts` plus pre-skip equals this value.
- The duration comes from the TOC. On the end-of-stream page the last packets are clamped so the stream ends at that page's granule; a packet removed whole has duration 0.
- Every packet is a keyframe.
- A granule that goes backwards is `OggError::GranuleBackwards`, returned as a read error.
- Each Opus stream has `Trim::new(Some(pre_skip), trailing)` and one `Edit` with `media_time = start + pre_skip` and `duration = end - pre_skip - start`, where `start` is the first packet's time and `end` the last page's granule. `trailing` is the samples the last page trims.
- RFC 7845 section 4.5 is applied to the first audio page with a completed packet: start is its granule minus its samples; a granule below the samples fails open unless the page is also the last, in which case start is 0 and the granule must be at least the pre-skip.
- A stream whose playable length (`end - start - pre_skip`) is negative fails open with `PreSkipPastEnd`. This is stricter than the RFC, which only names the single-page case.

Vorbis:

- A packet lasts a quarter of its block plus a quarter of the previous packet's block. The first audio packet lasts 0, as a decoder returns nothing for it. The first packet after a seek keeps the duration it has in the whole file; the decoder still returns nothing for it. Header packets and empty packets last 0.
- The first audio page with a completed packet sets the start: its granule minus its packets' samples. When that is negative the stream starts before zero (a stream copy cut mid-file does this); when the page is also the last page, the start is 0 and the end is trimmed.
- Each Vorbis stream has `Trim::new(Some(max(-start, 0)), trailing)` and one `Edit` with `media_time = max(start, 0)` and `duration = end - media_time`, so the samples before zero that section A.2 discards are skipped. FFmpeg and GStreamer both decode those samples; the engine skips them through the edit.
- Extradata is the three headers in Xiph lacing, as libavformat and Matroska carry them. When they exceed `Limits::max_extradata_bytes` the comment is replaced by one with only the vendor string; when even that is too large, open fails with `HostileSize`.
- Seek pre-roll is half the long block (section 1.3.2: the overlap a decoder needs).

Both mappings:

- A page whose granule disagrees with the running packet timeline by less than the shortest packet (Opus 120 samples, Vorbis half the short block) is rounding: the timeline stays contiguous and `granule_drift()` counts it. libavformat's WebM-to-Ogg stream copy rebuilds granules from millisecond timestamps and gives such pages. A larger forward difference is a gap; a larger backward one is `GranuleBackwards`.

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
| Missing or garbled Vorbis headers | Typed open error (`BadVorbisHeader`, `UnsupportedVorbisVersion`, `BadVorbisSetup`, `MissingVorbisHeaders`, `SetupPageNotFinished`). The identification header must be alone and complete on its page, and the setup header must end its page. |
| Vorbis setup longer than the header limit | `HostileSize`: the setup cannot be cut, as packet durations need its modes. A long comment header is cut like `OpusTags`. |
| Vorbis audio packet naming a mode the setup lacks | Returned with duration 0, as a decoder skips it. |
| Other mappings (FLAC, Theora, Speex, others) | Listed as `StreamCodec::Unknown("ogg:flac")` and so on, timescale 0, first packet as extradata. Their pages are skipped and counted in `ignored_pages()`. |
| Forward open | Same packets. The end is unknown: the edit duration is 0 and `trailing` is `None`. Seek is `SeekError::ForwardOnly`. |
| Seek | Bisects on page granules down to a 64 KiB window, then scans pages. Each probe interpolates between the bounds, a quarter window toward the bound that has not moved yet; a further probe on the same side doubles that step, and three in a row fall back to a midpoint. Lands on the packet that holds `max(target - preroll, start)` (3,840 for Opus, half the long block for Vorbis). `Landing::Packet { time, preroll, byte_position, .. }` with `index_used: false`; `byte_position` is the page the read resumes from; `ordinal` is 0 at the start and `u32::MAX` elsewhere because Ogg has no packet count. Past the end is `Landing::End`. On failure the read position and queued packets are restored. At most 64 probes and 8 MiB. Other tracks resume at their first complete packet after the landing page. |
| Seek in a damaged file | When the pages before the target are lost the seek lands on the first packet after the hole, at its true time, which can be after the target. |
| `consumed()` | Bytes read by open and packet reads. Seek reads are in `last_seek()` and `total_read()`. |
| `cancel()` | Persistent. Later reads and seeks return `Cancelled`. |
| Typed reasons | `ReadError` and `OpenError` have no Ogg variants, so Ogg reasons are mapped to the nearest variant (`Source { message }` mostly) and the typed `OggError` is kept in `last_error()`. |

Mux:

| Case | Behavior |
| --- | --- |
| Codec | Opus and Vorbis. `Unknown` is `MuxError::UnknownCodec`; any other known codec is `Rejected`. |
| Configuration | Opus: `OpusHead` from extradata, or `set_config` before the first packet; an MP4 `dOps` body is rewritten as `OpusHead`. Vorbis: the three headers, in Xiph lacing (libavformat, Matroska), as three 16-bit big-endian lengths each before its packet, or back to back; the identification header's channels and rate must match the stream, and the setup is walked to its modes. No configuration by the first packet is an error. |
| Page order | Every stream's beginning-of-stream page (`OpusHead` or the Vorbis identification header, alone, granule 0), then every stream's other headers: `OpusTags` (vendor `cutlass`) on one page, or the Vorbis comment (kept as given) and setup, which may span pages, with the setup ending its page. A header page that completes a packet has granule 0, one that completes none -1. Then audio. Streams cannot be added after the first packet. |
| Page policy | An audio page is closed before the next packet once a packet has completed on it, its granule is 0 or more, and its body holds 4,096 bytes or its completed packets span one second at the stream's rate, and whenever its 255 lacing values are used. Packets continue across pages. Granule is the end of the last packet completed on the page, or -1. |
| Vorbis packet times | Timescale is the sample rate. Each packet's duration must be at most its natural duration (from its mode and the previous packet's), and only the last may be shorter. Header packets and packets naming an unknown mode are `Rejected`. |
| Vorbis start before zero | Written: the first audio page stays open until a packet ends at 0 or later, so its granule says how far before zero the stream starts. If the stream ends on that page it is not marked last: a trimmed final packet goes alone on the last page, and an untrimmed stream ends with an empty last page. A page that would close with a negative granule (255 lacing values before 0, or a stream that reaches 0 only on its trimmed last packet) is `Rejected`. |
| Last page | Written by `finish` with the end-of-stream flag; an empty stream gets an empty end page with granule 0. A final packet shorter than its TOC becomes end trimming. |
| Cropped start with end trimming | RFC 7845 4.5 reads a first-and-last page as starting at granule minus samples, so the packets before the trimmed one are written on their own page. One trimmed packet alone starting after 0 is refused at finish. |
| Stream shorter than its pre-skip | Refused at finish. |
| Timing | Timescale 48,000 for Opus, the sample rate for Vorbis. Each stream's packets must be contiguous; an overlap, gap, a duration longer than the natural one (a first Vorbis packet may carry what it returns after a short or a long block, as one read after a seek does), a trimmed packet that is not the last, or a negative first Opus time is `Rejected`. |
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
| Vorbis packet | `Limits::max_packet_bytes` |
| Logical streams | `Limits::max_tracks` (64), `TooManyStreams` above |
| Header packets | `Limits::max_header_bytes`; `OpusTags` and the Vorbis comment above it are cut, a Vorbis setup above it is `HostileSize` |
| Vorbis setup walk | Every count read (codebooks, entries, dimensions, floors, residues, mappings, modes) is checked against the bytes left before it is used; the `lookup1` root is found without overflow |
| Extradata | `Limits::max_extradata_bytes`; a Vorbis stream falls back to a vendor-only comment above it |
| Comments | 65,536 |
| Open start scan / tail scan | 8 MiB each |
| Seek and link-end bisection | 64 probes each |
| Seek pre-roll | 3,840 samples (Opus); half the long block (Vorbis) |

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
| Vorbis headers, mono to 5.1, 22.05 to 48 kHz, one to two modes, blocks 256/2048 and 1024/1024 and 2048/2048 | Read | Written | `vorbis_files_demux_to_the_same_packets_times_and_trim`, `remuxed_vorbis_files_read_like_their_source` on 7 files from libvorbis, FFmpeg's encoder, a stream copy cut, and a WebM copy; the 26 in-crate Vorbis behavior tests and 10 header unit tests | Verified against libavformat and GStreamer |
| Vorbis start before zero | Applied | Written | the stream copy cut (start -3,648), its remux, and `a_vorbis_stream_that_starts_before_zero_muxes_and_reads_back` | Verified against libavformat and GStreamer |
| Vorbis stream copy from a seek | n/a | Written: a first packet stamped with what it returns after a short or a long block is taken, and the stream starts where it ends | `packets_demuxed_after_a_seek_remux_and_read_back`: the libvorbis stereo fixture, seeked to 10,000 and remuxed, reads back with the same packets and times after the first, and the source's end | Verified by this crate's demuxer; no independent reader run on that remux |
| Vorbis end trimming, edit | Applied | Written | ffmpeg decodes `end - start` samples for every exact-granule file and its remux; GStreamer the same or one more | Verified |
| Rounded granules (WebM-to-Ogg copy) | Contiguous, `granule_drift()` | n/a | `webm-copy` and `a_granule_off_by_less_than_a_packet_keeps_the_timeline` | Verified |
| Opus and Vorbis in one file | Each a track with its own timescale | Written | `a_file_with_opus_and_vorbis_reads_both`, `opus_and_vorbis_mux_into_one_file`, the fuzz corpus | Verified in this crate only |
| FLAC, Theora, Speex packets | Not returned | Refused | `an_unknown_mapping_is_listed_and_its_pages_skipped` | Unsupported |
| Opus or Vorbis in WebM or Matroska | n/a | n/a | n/a | Out of scope |
| App registration (Vorbis) | Desktop and phone readers, `cutlass-probe` | Sound export to `.ogg` | `an_ogg_vorbis_file_opens_seeks_and_decodes` (desktop reader), `an_ogg_vorbis_file_is_audio_as_long_as_its_samples` (probe, libvorbis fixtures against ffprobe), `an_ogg_vorbis_file_imports_as_a_sound` (phone import), `sound_alone_writes_a_vorbis_file` (export), the desktop export test (export, probe, import) | Verified on macOS |
| App registration (Opus) | Desktop and phone readers, `cutlass-probe` | Sound export to `.opus` | `an_opus_file_opens_seeks_and_decodes_after_the_pre_skip` (desktop), `an_ogg_opus_file_is_audio_as_long_as_its_samples_after_the_pre_skip` (probe), `sound_alone_writes_an_opus_file` (export), the desktop export test | Verified. Ogg files of other mappings than Opus and Vorbis are refused by the probe ("only Opus and Vorbis in Ogg are available") |
| Linux and Windows | Not run | Not run | This file was written on macOS | Not run |

## Commands and results

All on this machine: `Apple M5 Pro` (`sysctl -n machdep.cpu.brand_string`), macOS, ffmpeg 8.1.1 at `/opt/homebrew/bin`, GStreamer 1.28.3 with libvorbis 1.3.7. The Opus results are from 5 November 2026, the Vorbis results from 8 November 2026.

`cargo fmt -p cutlass-ogg`, `cargo clippy -p cutlass-ogg --all-targets -- -D warnings`, and `cargo test -p cutlass-ogg` pass: 106 unit and behavior tests, 2 mutation seed tests. The external tests skip without `local-files`.

### Independent producers

`CUTLASS_LOCAL_FILES=/Users/mrnewton/dev/cutlass/local-files cargo test -p cutlass-ogg --test external -- --nocapture`: 8 passed, 2 ignored (the measurements).

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

### Vorbis producers

PCM from the local `.m4a` (12 s from 30 s in; 8 s for 5.1, panned by ffmpeg) is encoded by libvorbis through GStreamer `vorbisenc ! oggmux`, and by FFmpeg's native encoder. `ffmpeg -ss 1 -c copy` cuts the libvorbis q0.4 file, and `ffmpeg -c:a copy` copies the Vorbis track of the local `.webm` into Ogg. For every packet this crate's demuxer matches ffprobe on count, size, and sha256 (first and last), and the bytes match `ffmpeg -c copy -f data`. Every packet ends where ffprobe's ends. Starts match from the second packet, except where libavformat gives the first short packet after a long one half a short block where section 4.3.8 gives a quarter of each block (its end still matches). libavformat starts each stream as if a short block came before the first packet, so its first packet starts that much earlier. The extradata matches libavformat's size once the comment is rebuilt with only its vendor, as libavformat does. GStreamer `oggdemux` matches on count, size, and each packet's end granule; on a first page that starts before zero it counts that page's packets from zero. ffmpeg decodes `end - start` samples (neither decoder drops the samples before zero); GStreamer `vorbisdec` decodes the same or one more.

| File | Packets | Rate | Blocks | Start | End | Trailing | Decoded (ffmpeg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| libvorbis q0.4 stereo | 959 | 48000 | 256, 2048 | 0 | 576000 | 0 | 576000 |
| libvorbis q0 mono 22050 | 518 | 22050 | 1024, 1024 | 0 | 264600 | 104 | 264600 |
| libvorbis managed 96k, 44100 | 825 | 44100 | 256, 2048 | 0 | 529200 | 80 | 529200 |
| libvorbis q0.3 5.1 | 649 | 48000 | 256, 2048 | 0 | 384000 | 0 | 384000 |
| stream copy cut from 1 s | 903 | 48000 | 256, 2048 | -3648 | 528000 | 0 | 531648 |
| FFmpeg native q4 | 564 | 48000 | 2048, 2048 | 0 | 576000 | 512 | 576000 |
| WebM track copied to Ogg | 1433 | 48000 | 256, 2048 | 112 | 1466064 | 0 | 1465920 |

The WebM copy's granules are rebuilt from millisecond times, so its pages drift from the packet timeline by -32 to +16 samples. This crate keeps the timeline, counts the drift in `granule_drift()`, and its ends stay within half a short block of libavformat's and GStreamer's. Each file is also seeked to 8 targets and past the end, and `a_chained_vorbis_file_returns_its_first_link` reads the first of two FFmpeg files (48 kHz, then 22.05 kHz) concatenated.

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

The seven Vorbis files are demuxed and muxed again, alternately with the headers in extradata and through `set_config` as 16-bit lengths. Each remux passes the whole Vorbis producer check above with exact granules, and seeks. ffprobe reports the same stream and payload bytes as the source. For the six exact-granule sources, ffprobe also gives the same sizes, hashes, and packet ends, and ffmpeg decodes the same sample count. ffprobe's start for a first short packet after a long one depends on where pages break, and differs on 6 packets of the cut file. This crate reads back the same packets, times, comment, start, trim, and edit; for the WebM copy the end is the packet timeline's, 32 samples before the source's rounded granule.

| Case | Packets | Start | Source bytes | Remux bytes |
| --- | --- | --- | --- | --- |
| libvorbis q0.4 stereo | 959 | 0 | 183483 | 183483 |
| libvorbis q0 mono 22050 | 518 | 0 | 52563 | 52239 |
| libvorbis managed 96k | 825 | 0 | 150409 | 150409 |
| libvorbis q0.3 5.1 | 649 | 0 | 208436 | 208436 |
| stream copy cut | 903 | -3648 | 169807 | 170563 |
| FFmpeg native q4 | 564 | 0 | 191193 | 192057 |
| WebM copy | 1433 | 112 | 7848 | 7848 |

### Measurement

`CUTLASS_LOCAL_FILES=/Users/mrnewton/dev/cutlass/local-files cargo test --release -p cutlass-ogg --test external measure -- --ignored --nocapture`

`rain_bedroom_night_30min.mp3` encoded by libopus at 96 kb/s (20,670,700 bytes, 89,927 packets, 1798.5 s) and by libvorbis at quality 0.4 through GStreamer (28,689,105 bytes, 122,937 packets, 44.1 kHz, 1798.6 s). Run on 8 November 2026 with a load average of about 5 from other work, so times are upper bounds.

| Measure | Opus | Vorbis |
| --- | --- | --- |
| Open (best of 5) | 0.292 ms, 162,575 bytes read | 0.294 ms, 155,714 bytes read |
| Open to first packet | 0.316 ms | 0.303 ms |
| Full read | 43.7 ms, 473 MB/s, about 41,000× real time | 62.4 ms, 460 MB/s, about 28,800× real time |
| 10 seeks, seek to first packet | mean 0.134 ms | mean 0.073 ms |
| Seek cost | at most 2 probes, 4 pages, and 131,072 bytes | at most 2 probes, 7 pages, and 81,920 bytes |

Before the seek's interpolation was made to bracket the target, the Vorbis file took 11 probes and up to 196,608 bytes per seek (mean 0.127 ms): both interpolated probes fell short of the target, and midpoints did the rest. `a_seek_in_a_long_variable_rate_file_takes_few_probes` locks the bound on a 7 MB file whose bitrate swings ninefold: at most 6 probes (the old rule took 9, halving alone 7) and at most 4 on average.

Every landing is the packet holding `target` less the pre-roll in the full packet list; every Opus seek after the start reports 5 pre-roll packets, and Vorbis seeks 2 to 9. libavformat's own seek (`ffprobe -read_intervals T%+#1`) lands at its page boundaries, 8,640 to 43,200 samples earlier than this crate. For a target of 0 it returned a first packet at 48,000; that was not investigated.

### Fuzzing

cargo-fuzz was not run. It is not installed, libfuzzer-sys builds C++ through `cc`, which this crate's rules do not allow, and a fuzz crate would need a workspace change outside this crate. `tests/fuzz.rs` runs in-process deterministic campaigns instead (xorshift per seed). Demux mutants come from muxed Opus and Vorbis files (one Vorbis stream starting before zero, one beside an Opus stream) and from the libvorbis and FFmpeg files in `tests/fixtures`, including a chain, with bit flips, byte writes, inserts, deletes, cuts, splices, duplicated ranges, and header-field hits, half with page checksums repaired; each is probed, opened random-access and forward, read to the end, and seeked, with bounded calls, packet limits, per-track time order, and the landing packet checked. Mux sessions use random Opus and Vorbis streams (Vorbis with the fixtures' real setup headers, cut or random configurations, random modes including unknown ones and header-like packets, and starts before zero), configurations, sizes up to 130,000 bytes, durations, and time jumps; a refused call must leave the mux failed and a finished mux must read back to exactly the accepted packets.

| Campaign | Command | Result |
| --- | --- | --- |
| Seeds (every test run) | `cargo test -p cutlass-ogg --test fuzz` | 256 demux mutants and 512 mux sessions, pass; at least one Vorbis packet read back is asserted |
| Demux, 300 s (Opus only) | `OGG_FUZZ_SECS=300 cargo test --release -p cutlass-ogg --test fuzz fuzz_campaign_demux -- --ignored --nocapture` | 722,000 mutants over 14,440 seeds, no failure, slowest input 96.3 ms |
| Mux, 300 s (Opus only) | `OGG_FUZZ_SECS=300 cargo test --release -p cutlass-ogg --test fuzz fuzz_campaign_mux -- --ignored --nocapture` | 5,204,560 sessions, 275,824 finished and read back, no failure |
| Demux, 120 s, with Vorbis | `OGG_FUZZ_SECS=120 cargo test --release -p cutlass-ogg --test fuzz fuzz_campaign -- --ignored --nocapture --test-threads 2` | 503,300 mutants over 10,066 seeds, no failure, slowest input 82.4 ms |
| Mux, 120 s, with Vorbis | same command | 2,060,020 sessions, 106,032 finished and read back with 231,822 Vorbis packets, no failure |

The campaigns found three muxer bugs, each fixed with a behavior test: a stream shorter than its pre-skip was written and then refused by the demuxer; a cropped start with end trimming on one page read back with the wrong start; and the same when the first packet spans pages. Both 300 s runs are on commit `46915201` (the demuxer is unchanged since `94c059fe`).

## Partial or not run

- Opus family 255 is parsed and accepted but not checked against an independent file.
- FLAC, Theora, and Speex mappings are listed, not demuxed.
- A Vorbis file with three or more modes, or blocks other than those above, is covered by unit tests only.
- Chained files return the first link only.
- cargo-fuzz not run (reason above).
- Linux and Windows not run.
- The engine's probe duration is the last packet's end, so it counts the pre-skip; `cutlass-probe` reports the edit's length.
