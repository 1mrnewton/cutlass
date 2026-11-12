# cutlass-matroska

`cutlass-matroska` reads and writes Matroska and WebM files (`.mkv`, `.mka`, `.webm`). It demuxes a file into timed packets and muxes packets back into a file. It does not decode or convert the frames. The crate is safe Rust (`#![forbid(unsafe_code)]`) and depends only on `cutlass-codec` and `cutlass-container`. No foreign library, subprocess or OS parser runs; FFmpeg and GStreamer are used only by the tests and the benchmark, as oracles and to make files.

The desktop app opens Matroska and WebM files with `MatroskaDemux` (`apps/desktop/src/readers.rs`), the import probe reads their size and duration (`crates/probe`), and `tools/media-bench` reads them. Export does not write Matroska yet: `MatroskaMux` implements `cutlass_container::Mux`, but `cutlass_export::write_movie` still takes ISOBMFF only.

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
| Lacing: Xiph, EBML, fixed | One packet per frame; laced frames step by DefaultDuration, and a BlockDuration is shared out over the lace | Not written (one frame per block) | `block.rs` and behavior tests |
| Cues | Read on the first seek. A seek starts from the track's own cue points, else from any track's | Written at the end: every video keyframe, or each track's first block in a Cluster for a file without video | `seeks_in_ffprobe`, measure test |
| Seek without Cues | Builds a Cluster list from Cluster headers once, then walks block headers | n/a: the muxer always writes Cues | behavior and measure tests |
| Forward-only input (`open_forward`) | Reads packets; `seek` is `SeekError::ForwardOnly` | The muxer needs `Write + Seek` | behavior tests |
| Header stripping (ContentCompAlgo 3) | The stripped bytes go back in front of each frame, and of CodecPrivate when that is encoded too | Not written | `header_stripping_puts_the_bytes_back_on_every_frame` |
| zlib, bzlib, lzo; encryption | The track opens with an unknown codec tag (`A_AAC+zlib`, `...+encrypted`) and the stored bytes; nothing is decompressed or decrypted | Not written | `encrypted_and_zlib_tracks_open_as_unknown_codecs_with_their_bytes_as_stored` |
| CodecDelay | An edit whose media time is the delay in ticks, and the encoder delay in samples (`Trim`); for Opus that is the OpusHead pre-skip | Opus: the pre-skip, with SeekPreRoll 80 ms. Other audio: the encoder delay given with `set_trim` before the track's first packet | behavior tests; remuxed copies keep every oracle track's delay |
| DiscardPadding | Ignored; the last packet's duration is its BlockDuration | Written for trailing padding given with `set_trim`, which also shortens the BlockDuration, and for an Opus packet shorter than its TOC byte says by more than one track tick | `an_opus_packet_shorter_than_its_toc_has_the_rest_discarded`, `trim_in_samples_fits_a_millisecond_track` |
| FLAC STREAMINFO sent again at the end | n/a | Written over the first when it is the same length; a different length is refused | behavior test |
| TrackTimestampScale other than 1 | Refused at open (`OpenError::Source`) | Never written | `a_track_timestamp_scale_other_than_one_is_refused` |
| Chained Segments | Reading stops after the first Segment | One Segment | not tested |
| Chapters, Tags, Attachments, BlockAdditions | Skipped | Not written | - |

`FEATURES` in `src/lib.rs` lists the same rows as data.

Bounds (`src/limits.rs`, plus the shared `cutlass_container::Limits`): an EBML header above 4 KiB, more than 4096 top-level elements before the first Cluster, a Cues element above 64 MiB, header-stripping bytes above 4 KiB, and any element larger than the bytes left in the file are refused before anything is allocated, as are packets, extradata, track counts and seek indexes above the shared caps. A seek walks at most 1,000,000 block headers, and a Cluster scan at most 4,000,000 Clusters. A hostile file ends in a typed error, not a panic.

## Using it

- `probe` and `MatroskaFamily` decide from the 4-byte EBML magic; a longer window also lets the DocType decide.
- `MatroskaDemux::open(reader, length)` takes a `Read + Seek` reader and the file length. It reads ahead itself (64 KiB, starting at 4 KiB after a seek), so a plain `File` is enough. `open_forward` takes a reader it will not seek.
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
MKV_FUZZ_SECS=600 cargo test -p cutlass-matroska --release --test external fuzz_campaign -- --ignored --nocapture
```

- 29 unit tests: EBML numbers and sizes, lacing, the CodecID mapping, header stripping, Opus packet lengths, the read-ahead buffer, the probe, and the support matrix.
- 35 behavior tests (`tests/behavior`) build files in memory: each lacing kind, BlockGroups, unknown-size Clusters, Info and Tracks found through the SeekHead, Cues and their absence, seek landings and preroll, cancellation, other DocTypes and newer versions, every truncation of a file, hostile sizes, and the muxer's round trips and refusals.
- `tests/external` (9 tests; each says it skipped and passes when `ffmpeg` or `ffprobe` is not on the `PATH`):
  - 17 files written by FFmpeg: H.264/AAC, HEVC/Opus, VP9/Opus, AV1/Opus, VP8/Vorbis, MPEG-2/MP3, MJPEG/16-bit PCM, ProRes/24-bit PCM, MPEG-4/AC-3, H.264/E-AC-3 with 100 ms Clusters, Cues at the front, a live (unknown-size) VP9 file, and FLAC, ALAC, 32-bit float, 32-bit big-endian and 8-bit PCM on their own. Every packet matches `ffprobe` (presentation time after the CodecDelay edit, size, keyframe flag, duration), and the joined payload matches `ffmpeg -c copy -f data`.
  - 5 files written by GStreamer: VP8/Vorbis and streamable (live) VP8/Opus from `webmmux`, and FLAC, MP3 and 16-bit big-endian PCM from `matroskamux`. Each is compared with libavformat as above and, pad by pad, with GStreamer's own `matroskademux`.
  - Each FFmpeg file copied by this crate's muxer: the copy holds the same bytes, keyframe flags and times per track and keeps every track's CodecDelay; libavformat reads it packet for packet as this crate does, `matroskademux` returns the same packet sizes, and `ffprobe -read_intervals` seeks it to a keyframe. GStreamer's demuxer shares no code with libavformat, so the muxer is checked by two independent readers. By hand, at the last change to the muxer, `ffmpeg -f null` decoded all 17 copies without an error, and `gst-discoverer-1.0` read 16 of them; for the ALAC copy it reports no decoder for `A_ALAC`, which it also does for FFmpeg's own ALAC file.
  - Seeks land on the keyframes `ffprobe` lists.
  - Every `.mkv`, `.mka` and `.webm` in the local media folder (`CUTLASS_LOCAL_MEDIA`, by default `local-files` beside the checkout) is read to the end against libavformat and seeked through. Here that was one 1920x1080 WebM of 2334 packets.
  - The measure test, below.
- On every run: 12 seeds × 250 mutants through the demuxer, and 400 random packet sequences through the muxer, each read back.
- The desktop app's tests open a WebM with Opus written by `MatroskaMux` and compare the decoded samples from the start and after a seek. The probe crate checks a WebM's size and duration.

### Fuzz campaigns

The demux campaign mutates valid seed files (bit flips, interesting bytes, inserted and deleted runs, truncation, spliced copies, eight-octet and one-octet sizes set huge or unknown) and runs open, a read to the end and seeks. The mux campaign sends random packet sequences, a tenth of them hostile, through the muxer and reads each finished file back. Both ran 600 seconds, in parallel, in a release build on the M5 Pro:

| Code | Demux | Mux | Failures |
| --- | --- | --- | --- |
| `15f7b79a` | 23,154,000 mutants from 231,540 seeds; 13,029,218 opened; 811,490,457 packets read; 118,697,135 seeks landed; slowest input 147.8 ms | 7,765,272 sequences; 5,431,225 finished and read back | none |
| `8664e0a3` | 22,934,400 mutants from 229,344 seeds; 12,907,358 opened; 804,228,878 packets read; 117,583,400 seeks landed; slowest input 145.0 ms | 8,025,045 sequences; 5,612,525 finished and read back | none |

"None" means no panic and no failed comparison. The second run covers the muxer's Opus padding, the trim in samples and the cached seek starts. The slowest input took about 145 ms in both runs; its cause has not been looked into.

## Speed against FFmpeg

Measured on 12 November 2026 on an Apple M5 Pro running macOS 27.0.1, against FFmpeg 8.1.1, with rustc 1.99.1 and a release build. Each side is a whole process timed with `/usr/bin/time -l`, start-up included: one warm-up, then five runs each in turn, the median shown, one thread each. The Rust side is `examples/bench.rs`, the FFmpeg side `bench/ffmpeg_matroska.c`, which calls libavformat directly. Instructions retired is the measure; wall time on this machine is too noisy at these lengths. To run it again: `crates/containers/matroska/bench/against_ffmpeg.sh <work folder>`.

The files, made by FFmpeg: H.264 1920x1080 at 6 Mb/s with B-frames and a keyframe every 2 s plus AAC, 120 s (92 MB, 9226 packets); VP9 640x360 at 1 Mb/s plus Opus, 60 s (7.6 MB, 4801 packets); FLAC, 600 s (8.1 MB, 6250 packets). The script stops unless both sides read the same packet count and bytes.

| File | Job | Cutlass | libavformat | Past open: Cutlass / libavformat |
| --- | --- | ---: | ---: | ---: |
| H.264 + AAC `.mkv` | open | 15.7 M | 285.3 M | |
| | read every packet | 94.5 M | 415.3 M | 78.8 M / 129.9 M (0.61×) |
| | 200 seeks | 83.1 M | 321.2 M | 67.5 M / 35.9 M (1.88×) |
| | remux | 218.4 M | 620.6 M | 202.7 M / 335.3 M (0.60×) |
| VP9 + Opus `.webm` | open | 15.8 M | 285.3 M | |
| | read every packet | 34.1 M | 322.7 M | 18.3 M / 37.4 M (0.49×) |
| | 200 seeks | 62.6 M | 297.4 M | 46.9 M / 12.1 M (3.88×) |
| | remux | 58.8 M | 357.7 M | 43.1 M / 72.4 M (0.59×) |
| FLAC `.mka` | open | 15.7 M | 285.2 M | |
| | read every packet | 38.2 M | 334.7 M | 22.5 M / 49.6 M (0.45×) |
| | 200 seeks | 33.6 M | 291.6 M | 17.8 M / 6.4 M (2.77×) |
| | remux | 68.5 M | 363.6 M | 52.8 M / 78.4 M (0.67×) |

Instructions retired. Peak memory was 2.1 to 4.4 MB for Cutlass and 9.6 to 22.4 MB for libavformat.

How to read it:

- Most of libavformat's total is loading and starting the FFmpeg libraries, which are linked dynamically; the Rust side is one static binary. The last column takes each side's `open` run off its other runs, which leaves the container work, and is the fairer comparison.
- Reading every packet and remuxing take 0.45 to 0.67 times libavformat's instructions past open.
- Seeking takes 1.9 to 3.9 times as many. The two do different work: `av_seek_frame` goes to the cue point and stops, while this crate walks the block headers from the cue's Cluster to the target, to land on the last keyframe at or before it and to count the preroll packets the `Demux` contract reports. In the FLAC file this lands on the frame at or before each target, where libavformat lands on the Cluster start, up to a few seconds earlier. A profile of the seek run puts nearly all of its time in the kernel's `read` and `lseek`, not in parsing.
- For remux, the libavformat side also calls `avformat_find_stream_info`, which its muxer needs to rebuild decode times for H.264 B-frames, and that decodes a few frames. Its other jobs skip that call.

What a seek reads (`tests/external/measure.rs`: 20 s of 640x360 H.264 at 4 Mb/s with a keyframe every 2 s, plus AAC, 10.2 MB): opening reads 4,096 bytes. With Cues, six seeks read 16 KB to 209 KB each and inspect no frame payload. Without Cues, the first seek reads 176 KB to build the Cluster list, and the others cost about what they do with Cues.

## Limits and gaps

| Item | State |
| --- | --- |
| Export | `write_movie` does not write MKV or WebM, and the desktop export choices do not offer them |
| Precision at the usual 1 ms scale | Times are rounded to the millisecond, as the file stores them. An Opus pre-skip of 2.5 ms becomes a 3 ms edit, so the desktop app starts playing 24 samples late; `Trim` still has the exact pre-skip. A file with a 0.5 ms scale plays exactly (tested) |
| DiscardPadding on demux | Ignored. A remuxed Opus file gets its DiscardPadding back from the TOC byte and the BlockDuration, to within a millisecond |
| Lacing on mux | Not written |
| Compressed or encrypted tracks | Opened with an unknown codec tag; not decompressed or decrypted |
| Codec delay for audio other than Opus, on mux | Written only when `set_trim` gives the encoder delay before the track's first packet. Nothing in the export path calls `set_trim` today |
| Chained Segments | Not read past the first, and not tested |
| mkvtoolnix | Not installed; not used as an oracle |

## Readiness

Demux: ready for import. Files from FFmpeg and GStreamer read packet for packet as libavformat reads them, seeking uses Cues and works without them, the desktop app plays WebM with Opus, and the fuzz campaigns found nothing.

Mux: writes Matroska and WebM that libavformat and GStreamer read back, with Cues and a SeekHead, keeping codec delay and Opus end padding. It is not wired to export, so the app cannot make an MKV or WebM yet.
