# cutlass-matroska

`cutlass-matroska` reads and writes Matroska and WebM files (`.mkv`, `.mka`, `.webm`). It demuxes a file into timed packets and muxes packets back into a file. It does not decode or convert the frames. The crate is safe Rust (`#![forbid(unsafe_code)]`) and depends only on `cutlass-codec` and `cutlass-container`. No foreign library, subprocess or OS parser runs; FFmpeg and GStreamer are used only by the tests and the benchmark, as oracles and to make files.

The desktop app opens Matroska and WebM files with `MatroskaDemux` (`apps/desktop/src/readers.rs`), the import probe reads their size and duration (`crates/probe`), and `tools/media-bench` reads them. `cutlass_export::write_movie` writes Matroska for `ContainerId::Matroska`, and WebM when the destination ends in `.webm`; the desktop export panel does not offer either yet.

Specifications: RFC 8794 (EBML), RFC 9559 (Matroska), the Matroska codec mappings (matroska.org, "Codec Specifications"), the WebM container guidelines (webmproject.org), and RFC 7845 and RFC 6716 for the Opus pre-skip and packet length. FFmpeg's `libavformat/matroskadec.c` and `matroska.h`, and GStreamer's `gst-plugins-good/gst/matroska/matroska-demux.c` and `matroska-mux.c`, were read from the local checkouts for the order of operations and the edge cases; no code was taken from them. mkvtoolnix is not installed here, so it was neither read nor used as an oracle.

## What it supports

Timestamps stay in the file's clock. A stream's timescale is 10⁹ / TimestampScale ticks per second when that divides exactly (1000 for the usual 1 ms scale); otherwise the stream counts nanoseconds. A block's time is its Cluster's Timestamp plus the block's 16-bit offset, and it is the presentation time.

| Capability | Demux | Mux | Evidence |
| --- | --- | --- | --- |
| DocType `matroska` and `webm`, EBML header | Opens. Another DocType is not Matroska; an EBMLReadVersion other than 1 or a DocTypeReadVersion above 4 is `UnsupportedBrand` | `MatroskaMux::new` writes `matroska`, `MatroskaMux::webm` writes `webm`, both DocTypeVersion 4 and DocTypeReadVersion 2 | behavior tests; ffprobe and GStreamer read our files |
| WebM codec subset | Any codec is read | `webm` refuses anything but VP8, VP9, AV1, Opus and Vorbis | `webm_keeps_to_its_codecs_and_writes_opus_delay` |
| SeekHead | Followed, including one secondary SeekHead, to find Info, Tracks and Cues | Written at the front in a reserved 96-byte area, filled in at the end | behavior and oracle tests |
| Info: TimestampScale, Duration | Read; Duration is the track length when present | Written; Duration filled in at the end | ffprobe durations match |
| Tracks: number, type, CodecID, CodecPrivate, language, DefaultDuration, CodecDelay, SeekPreRoll, picture size, sample rate and channels | Read; a repeated track number is `OpenError::DuplicateTrack` | Written | ffprobe comparisons |
| CodecIDs | H.264, HEVC, AV1, VP8, VP9, MPEG-1/2, MPEG-4 Part 2, Motion JPEG, ProRes, AAC (with the legacy `A_AAC/MPEG4/...` ids), Opus, Vorbis, FLAC, MP3, AC-3, E-AC-3, ALAC, PCM (integer little and big endian, float), and `V_MS/VFW/FOURCC` and `A_MS/ACM` for those codecs. Anything else, subtitles included, opens as a stream with an unknown codec tag and its packets | The same codecs under their native CodecIDs; an unknown codec is `MuxError::UnknownCodec` | `codec.rs` tests, 17 ffmpeg cases |
| ProRes | The 8-byte frame size and `icpf` atom go back in front of each frame | Taken off each frame, as the mapping asks | `prores_frames_lose_their_first_atom_on_disk_and_get_it_back`; ffmpeg payloads match |
| Clusters of known size | Read | Written, the size filled in when the Cluster closes. A new Cluster starts at a video keyframe once the Cluster holds 4 KiB, after 5 s or 5 MiB, or when a block offset would not fit 16 bits | behavior tests |
| Clusters of unknown size (live recordings) | Read; one ends at the next top-level element | Not written | `vp9_live.webm` oracle case, behavior tests |
| SimpleBlock, BlockGroup (BlockDuration, ReferenceBlock) | Read; a BlockGroup without ReferenceBlock is a keyframe | SimpleBlock when the packet lasts the track's DefaultDuration, else a BlockGroup with BlockDuration; ReferenceBlock on non-keyframes | behavior tests, ffprobe keyframe flags |
| Lacing: Xiph, EBML, fixed | One packet per frame; laced frames step by DefaultDuration, and a BlockDuration is shared out over the lace | Not written (one frame per block), by decision: see "Lacing on mux" below | `block.rs` and behavior tests |
| Cues | Read at open when the file has an audio track (to find its end padding), else on the first seek. A seek starts from the track's own cue points, else from any track's | Written at the end: every video keyframe, or each track's first block in a Cluster for a file without video | `seeks_in_ffprobe`, measure test |
| Seek without Cues | Builds a Cluster list from Cluster headers once, then walks block headers | n/a: the muxer always writes Cues | behavior and measure tests |
| Seek walks kept | Up to 16,384 block headers that seeks walked over are kept by the Cluster each walk started from, so seeking back over the same stretch reads nothing; past that they are dropped and kept afresh | n/a | `seeks_over_a_stretch_walked_before_land_as_a_fresh_reader_does`, `seeks_land_the_same_once_the_kept_walks_are_full`, measure test |
| Forward-only input (`open_forward`) | Reads packets; `seek` is `SeekError::ForwardOnly` | The muxer needs `Write + Seek` | behavior tests |
| Header stripping (ContentCompAlgo 3) | The stripped bytes go back in front of each frame, and of CodecPrivate when that is encoded too | Not written | `header_stripping_puts_the_bytes_back_on_every_frame` |
| zlib, bzlib, lzo; encryption | The track opens with an unknown codec tag (`A_AAC+zlib`, `...+encrypted`) and the stored bytes; nothing is decompressed or decrypted | Not written | `encrypted_and_zlib_tracks_open_as_unknown_codecs_with_their_bytes_as_stored` |
| CodecDelay | An edit whose media time is the delay in ticks, and the encoder delay in samples (`Trim`); for Opus that is the OpusHead pre-skip | Opus: the pre-skip, with SeekPreRoll 80 ms. Other audio: the encoder delay given with `set_trim` before the track's first packet | behavior tests; remuxed copies keep every oracle track's delay |
| DiscardPadding | Each audio track's last block's padding is its stream's trailing trim, in samples, read at open from the last cue point's Cluster to the end (none without Cues). An Opus packet's duration is what of it plays, so a block with only a DefaultDuration loses its padding; another codec's packet keeps its whole frame (a BlockDuration gets its padding back) and the trim states the padding | Written for trailing padding given with `set_trim`, which also shortens the BlockDuration, and for an Opus packet shorter than its TOC byte says by more than one track tick; a stated Opus padding within that tick is written exactly | `discard_padding_shortens_an_opus_packet_and_is_the_tracks_end_trim`; FFmpeg's and GStreamer's files give the padding ffprobe reports, and remuxed copies keep it |
| FLAC STREAMINFO sent again at the end | n/a | Written over the first when it is the same length; a different length is refused | behavior test |
| TrackTimestampScale other than 1 | Refused at open (`OpenError::Source`): the two mature readers disagree on what it scales (see below) | Never written | `a_track_timestamp_scale_other_than_one_is_refused` |
| Chained Segments | Only the first Segment is read, forward and random; the reader ends cleanly where it ends (see below) | One Segment | `a_second_segment_after_the_first_is_not_read` |
| Chapters, Tags, Attachments, BlockAdditions | Skipped | Not written | - |

`FEATURES` in `src/lib.rs` lists the same rows as data.

Bounds (`src/limits.rs`, plus the shared `cutlass_container::Limits`): an EBML header above 4 KiB, more than 4096 top-level elements before the first Cluster, a Cues element above 64 MiB, header-stripping bytes above 4 KiB, and any element larger than the bytes left in the file are refused before anything is allocated, as are packets, extradata, track counts and seek indexes above the shared caps. A seek walks at most 1,000,000 block headers, and a Cluster scan at most 4,000,000 Clusters. A hostile file ends in a typed error, not a panic.

## Using it

- `probe` and `MatroskaFamily` decide from the 4-byte EBML magic; a longer window also lets the DocType decide.
- `MatroskaDemux::open(reader, length)` takes a `Read + Seek` reader and the file length. It reads ahead itself, 64 KiB at a time, so a plain `File` is enough. `open_forward` takes a reader it will not seek.
- `MatroskaDemux` implements `cutlass_container::Demux`. `ebml_header()`, `info()`, `track_entries()`, `clock()` and `duration_ticks()` give the parsed header.
- `MatroskaMux` implements `cutlass_container::Mux`. Give it a buffered writer: it writes frame by frame, and a 1 MiB `BufWriter` took an H.264 remux from 397 M to 218 M instructions against the default 8 KiB one.

```rust
use std::fs::File;

use cutlass_container::{Demux, SeekTo};
use cutlass_matroska::MatroskaDemux;

let file = File::open(path)?;
let length = file.metadata()?.len();
let mut demux = MatroskaDemux::open(file, length)?;
let track = demux.streams()[0].id().get();
demux.seek(SeekTo { track, time: 1500, end: None })?;
while let Some(packet) = demux.read_packet()? {
    let _ticks = packet.timestamp().presentation();
}
```

Examples:

- `cargo run -p cutlass-matroska --example round_trip` writes a short WebM in memory, reads it back and seeks in it. It needs no file.
- `cargo run -p cutlass-matroska --example packets -- movie.mkv [seconds]` prints the streams and every packet, after an optional seek.
- `cargo run -p cutlass-matroska --example remux -- in.mkv out.webm` copies every track into a new file.
- `cargo run --release -p cutlass-matroska --example bench -- <open|read|seek|remux> ...` is the Rust side of the benchmark below.

## How it is tested

```sh
cargo test -p cutlass-matroska
MKV_FUZZ_SECS=600 MKV_FUZZ_ORIGIN=5000000 cargo test -p cutlass-matroska --release --test external fuzz_campaign -- --ignored --nocapture
MKV_FUZZ_REPLAY=1234:56 cargo test -p cutlass-matroska --release --test external fuzz_replay -- --ignored --nocapture
```

- 29 unit tests: EBML numbers and sizes, lacing, the CodecID mapping, header stripping, Opus packet lengths, the read-ahead buffer, the probe, and the support matrix.
- 40 behavior tests (`tests/behavior`) build files in memory: each lacing kind, BlockGroups, unknown-size Clusters, Info and Tracks found through the SeekHead, Cues and their absence, seek landings and preroll, kept seek walks against a fresh reader, DiscardPadding, a chained second Segment, cancellation, other DocTypes and newer versions, every truncation of a file, hostile sizes, and the muxer's round trips and refusals.
- `tests/external` (9 tests, and 3 ignored ones for the long fuzz campaigns and a replay; each says it skipped and passes when `ffmpeg` or `ffprobe` is not on the `PATH`):
  - 17 files written by FFmpeg: H.264/AAC, HEVC/Opus, VP9/Opus, AV1/Opus, VP8/Vorbis, MPEG-2/MP3, MJPEG/16-bit PCM, ProRes/24-bit PCM, MPEG-4/AC-3, H.264/E-AC-3 with 100 ms Clusters, Cues at the front, a live (unknown-size) VP9 file, and FLAC, ALAC, 32-bit float, 32-bit big-endian and 8-bit PCM on their own. Every packet matches `ffprobe` (presentation time after the CodecDelay edit, size, keyframe flag, duration), each audio track's trailing trim is its last packet's DiscardPadding as ffprobe reports it, and the joined payload matches `ffmpeg -c copy -f data`.
  - 5 files written by GStreamer: VP8/Vorbis and streamable (live) VP8/Opus from `webmmux`, and FLAC, MP3 and 16-bit big-endian PCM from `matroskamux`. Each is compared with libavformat as above and, pad by pad, with GStreamer's own `matroskademux`.
  - Each FFmpeg file copied by this crate's muxer: the copy holds the same bytes, keyframe flags and times per track and keeps every track's CodecDelay and end padding; libavformat reads it packet for packet as this crate does, `matroskademux` returns the same packet sizes, and `ffprobe -read_intervals` seeks it to a keyframe. GStreamer's demuxer shares no code with libavformat, so the muxer is checked by two independent readers. By hand, at the last change to the muxer, `ffmpeg -f null` decoded all 17 copies without an error, and `gst-discoverer-1.0` read 16 of them; for the ALAC copy it reports no decoder for `A_ALAC`, which it also does for FFmpeg's own ALAC file.
  - Seeks land on the keyframes `ffprobe` lists.
  - Every `.mkv`, `.mka` and `.webm` in the local media folder (`CUTLASS_LOCAL_MEDIA`, by default `local-files` beside the checkout) is read to the end against libavformat and seeked through. Here that was one 1920x1080 WebM of 2334 packets.
  - The measure test, below.
- On every run: 12 seeds × 250 mutants through the demuxer, and 400 random packet sequences through the muxer, each read back.
- The desktop app's tests open a WebM with Opus written by `MatroskaMux`, at a 0.5 ms and a 1 ms tick, and compare the decoded samples from the start and after a seek. The probe crate checks a WebM's size and duration. The export crate writes an `.mkv` and a `.webm` movie and reads them back here and in FFmpeg and GStreamer.

### Fuzz campaigns

The demux campaign mutates valid seed files (bit flips, interesting bytes, inserted and deleted runs, truncation, spliced copies, eight-octet and one-octet sizes set huge or unknown) and runs open, a read to the end and seeks. The mux campaign sends random packet sequences, a tenth of them hostile, through the muxer and reads each finished file back. Both ran 600 seconds, in parallel, in a release build on the M5 Pro:

| Code | Demux | Mux | Failures |
| --- | --- | --- | --- |
| `15f7b79a` | 23,154,000 mutants from 231,540 seeds; 13,029,218 opened; 811,490,457 packets read; 118,697,135 seeks landed; slowest input 147.8 ms | 7,765,272 sequences; 5,431,225 finished and read back | none |
| `8664e0a3` | 22,934,400 mutants from 229,344 seeds; 12,907,358 opened; 804,228,878 packets read; 117,583,400 seeks landed; slowest input 145.0 ms | 8,025,045 sequences; 5,612,525 finished and read back | none |
| `15addba6`, seeds from 20,000,000 | 21,478,900 mutants from 214,789 seeds; 12,085,300 opened; 753,124,076 packets read; 110,116,728 seeks landed; slowest input 52.2 ms | 7,072,472 sequences; 4,947,429 finished and read back | none |

"None" means no panic and no failed comparison. The second run covers the muxer's Opus padding, the trim in samples and the cached seek starts. The third starts from new seeds (`MKV_FUZZ_ORIGIN=20000000`; the first two both started at 1,000 and 100,000, so they ran the same mutants) and covers DiscardPadding on demux, the end-padding walk at open, the kept seek walks and full-buffer reads.

The slowest inputs are stalls of the machine, not slow paths. A campaign names its eight slowest inputs and `fuzz_replay` times each again, ten times: the third run's eight (52.2 down to 13.7 ms, five of them from seeds within 73 of each other, so from the same few seconds) each take 0.01 to 0.05 ms when replayed. Rerunning the old seeds 1,000 onward on this code for 600 s (40,922,300 mutants, nothing found) gave a slowest input of 80.5 ms that replays in 0.02 ms. Every mutant is at most a few kilobytes and the campaign averages 28 µs one; other campaigns and a benchmark were running on the machine at the same time.

## Speed against FFmpeg

Measured on 12 November 2026 on an Apple M5 Pro running macOS 27.0.1, against FFmpeg 8.1.1, with rustc 1.99.1 and a release build. Each side is a whole process timed with `/usr/bin/time -l`, start-up included: one warm-up, then five runs each in turn, the median shown, one thread each. The Rust side is `examples/bench.rs`, the FFmpeg side `bench/ffmpeg_matroska.c`, which calls libavformat directly. Instructions retired is the measure; wall time on this machine is too noisy at these lengths. To run it again: `crates/containers/matroska/bench/against_ffmpeg.sh <work folder>`.

The files, made by FFmpeg: H.264 1920x1080 at 6 Mb/s with B-frames and a keyframe every 2 s plus AAC, 120 s (92 MB, 9226 packets); VP9 640x360 at 1 Mb/s plus Opus, 60 s (7.6 MB, 4801 packets); FLAC, 600 s (8.1 MB, 6250 packets). The script stops unless both sides read the same packet count and bytes.

| File | Job | Cutlass | libavformat | Past open: Cutlass / libavformat |
| --- | --- | ---: | ---: | ---: |
| H.264 + AAC `.mkv` | open | 16.3 M | 285.3 M | |
| | read every packet | 95.3 M | 415.3 M | 79.0 M / 130.0 M (0.61×) |
| | 200 seeks | 52.3 M | 321.4 M | 36.0 M / 36.1 M (1.00×) |
| | remux | 220.9 M | 621.1 M | 204.6 M / 335.8 M (0.61×) |
| VP9 + Opus `.webm` | open | 16.1 M | 285.3 M | |
| | read every packet | 34.4 M | 322.7 M | 18.3 M / 37.4 M (0.49×) |
| | 200 seeks | 29.2 M | 297.6 M | 13.0 M / 12.3 M (1.06×) |
| | remux | 59.4 M | 358.0 M | 43.3 M / 72.7 M (0.60×) |
| FLAC `.mka` | open | 15.7 M | 285.3 M | |
| | read every packet | 38.2 M | 334.8 M | 22.5 M / 49.5 M (0.45×) |
| | 200 seeks | 25.8 M | 291.6 M | 10.0 M / 6.4 M (1.58×) |
| | remux | 68.6 M | 363.7 M | 52.9 M / 78.4 M (0.68×) |

Instructions retired. Peak memory was 2.2 to 4.6 MB for Cutlass and 9.6 to 22.5 MB for libavformat.

How to read it:

- Most of libavformat's total is loading and starting the FFmpeg libraries, which are linked dynamically; the Rust side is one static binary. The last column takes each side's `open` run off its other runs, which leaves the container work, and is the fairer comparison. Opening now includes reading the Cues and walking the last GOP for the audio end padding: 0.6 M instructions more on the H.264 file than before.
- Reading every packet and remuxing take 0.45 to 0.68 times libavformat's instructions past open.
- Seeking takes 1.0 to 1.6 times as many, down from 1.9 to 3.9. The two do different work: `av_seek_frame` goes to the cue point and stops, while this crate walks the block headers from the cue's Cluster to the target, to land on the last keyframe at or before it and to count the preroll packets the `Demux` contract reports. In the FLAC file this lands on the frame at or before each target, where libavformat lands on the Cluster start, up to a few seconds earlier. The landed packets are the same as before the changes (the sums of their times are unchanged). Two changes did it, measured one at a time on the whole 200-seek run:
  - Every refill reads the whole 64 KiB buffer, after a seek too, instead of starting at 4 KiB. A read or seek call costs about 5,000 instructions against 0.2 a byte, so fewer, larger reads win: 83 to 76 M (H.264), 62 to 42 M (VP9), 34 to 25 M (FLAC). Reading on through a gap instead of seeking over it was tried as well and did not help (82 M on H.264 with up to 256 KiB read through).
  - The block headers a walk passed are kept (up to 16,384), so a seek into a stretch walked before reads nothing: 76 to 52 M (H.264), 42 to 29 M (VP9). The 200 targets fall in 60 and 30 Clusters there, so most seeks revisit one; FLAC's 121 Clusters are mostly visited once and gain nothing. A seek into a stretch not walked before costs what the first change leaves.
  - CueRelativePosition would not help these files: there is one cue point per Cluster, and it points 9 to 472 bytes into the Cluster, past the Timestamp and at most a few small audio blocks that the walk's first 64 KiB read brings anyway.
- For remux, the libavformat side also calls `avformat_find_stream_info`, which its muxer needs to rebuild decode times for H.264 B-frames, and that decodes a few frames. Its other jobs skip that call.

What a seek reads (`tests/external/measure.rs`: 20 s of 640x360 H.264 at 4 Mb/s with a keyframe every 2 s, plus AAC, 10.2 MB): opening reads 918 KB, the first buffer and then the Cues and the last GOP, walked for the AAC track's end padding. With Cues, six seeks read 131 KB to 852 KB each, whole 64 KiB buffers across the stretch they walk, and inspect no frame payload. The same six seeks again on the same reader read nothing. Without Cues, the first seek reads 1.2 MB to build the Cluster list, and the others cost what they do with Cues.

## Limits and gaps

| Item | State |
| --- | --- |
| Export | `write_movie` writes MKV and WebM (`crates/export`, `a_movie_is_written_as_matroska_or_webm`: MPEG-4 + AAC `.mkv` and VP8 + Opus `.webm` read back here, in ffprobe and FFmpeg's decoder, and in `gst-discoverer-1.0`). The desktop export panel does not offer a container choice; that is a product decision |
| Precision at the usual 1 ms scale | Times are rounded to the millisecond, as the file stores them. The engine takes an encoder delay its edit can only round (a 2.5 ms Opus pre-skip becomes a 3 ms edit) from the trim in samples instead, so the desktop app starts on the exact sample (`a_millisecond_webm_starts_after_the_exact_opus_pre_skip`). An edit that states the delay exactly, as Ogg's and MP4's do, is used as before |
| End padding without Cues | A file without Cues states no trailing trim: finding its last Cluster would mean walking all of them at open. Its Opus packets are still shortened by their DiscardPadding |
| Open cost | With an audio track, open reads the Cues and walks the block headers from the last cue point's Cluster to the end, about one GOP: 918 KB of the 10.2 MB measure file |
| Lacing on mux | Not written; see below |
| Compressed or encrypted tracks | Opened with an unknown codec tag; not decompressed or decrypted |
| TrackTimestampScale other than 1 | Refused; see below |
| Chained Segments | The first Segment only; see below |
| mkvtoolnix | Not installed; not used as an oracle |

### Lacing on mux

Lacing would save a block header (about 6 octets) per frame, less the lace sizes. Counted from the audio packet sizes ffprobe lists, laced 4 or 8 frames to a block:

| File | Audio frames (mean size) | Xiph, 8 a block | EBML, 8 a block |
| --- | ---: | ---: | ---: |
| VP9 + Opus `.webm`, 60 s | 3001 (305 B) | 12.8 KB: 0.17% of the file | 15.0 KB: 0.20% |
| H.264 + AAC `.mkv`, 120 s | 5626 (341 B) | 23.9 KB: 0.03% | 28.1 KB: 0.03% |
| FLAC `.mka`, 600 s | 6250 (1293 B) | 4.7 KB: 0.06% | 31.2 KB: 0.38% |
| Opus 32 kb/s alone, 60 s | 3001 (122 B) | 15.1 KB: 3.9% | 15.1 KB: 3.9% |

With video it is under 0.4%. Only low-rate audio on its own gains a few percent, and lacing costs there: laced frames have no time of their own (they step by DefaultDuration), a seek lands on the block, and DiscardPadding covers the whole block. FFmpeg's muxer writes `FlagLacing 0` ("no lacing (yet)", `matroskaenc.c`) and GStreamer's `matroskamux` does not lace either. So the muxer writes one frame per block; the demuxer reads all three lacings.

### TrackTimestampScale

FFmpeg and GStreamer both read it, but scale different things: FFmpeg's track time base is TimestampScale × TrackTimestampScale and it divides only the Cluster time by the scale, so the block offset is scaled and the Cluster time is not (`matroskadec.c`, `matroska_parse_block`); GStreamer multiplies the sum of the Cluster time and the offset (`matroska-demux.c`, `gst_matroska_demux_parse_blockgroup_or_simpleblock`). The same file gives two different time lines, so there is no reading to match, and a file with a scale other than 1 is refused with that scale and track named. A scale of exactly 1 opens (tested).

### Chained Segments

Two whole files one after the other, as a live server chains them, read in FFmpeg and GStreamer as the first file's packets followed by the second's with times starting over (checked by hand: 51 and 102 Opus packets for one and two copies of a one-second WebM, the 52nd at -7 ms). A `Demux` has one track list and one time line, so this reader keeps to the first Segment and ends cleanly at its end, forward and random (`a_second_segment_after_the_first_is_not_read`).

## Readiness

Demux: ready for import. Files from FFmpeg and GStreamer read packet for packet as libavformat reads them, with the end padding ffprobe reports; seeking uses Cues, works without them, and costs about what libavformat's does on video files; the desktop app plays WebM with Opus from the exact sample after the pre-skip; and the fuzz campaigns found nothing.

Mux: writes Matroska and WebM that libavformat and GStreamer read back, with Cues and a SeekHead, keeping codec delay and end padding for Opus and other audio. `write_movie` exports to it; the desktop export panel does not offer it yet.
