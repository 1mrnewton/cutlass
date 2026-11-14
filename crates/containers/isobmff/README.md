# cutlass-isobmff

`cutlass-isobmff` reads and writes the ISO base media file format: MP4, QuickTime (`.mov`), M4A, M4V, 3GP, and fragmented MP4. It demuxes a file into timed packets and muxes packets back into a file. It does not decode the packets. The crate is safe Rust (`#![forbid(unsafe_code)]`). The desktop app opens these files with `IsoBmffFile`, and the export crate writes MP4, MOV, and M4A files with `Mp4Mux`.

## What it supports

A file opens when its `ftyp` box names a known major or compatible brand, or when a QuickTime file without `ftyp` starts with `moov`, `mdat`, `wide`, `free`, `skip`, or `pnot`. The known brands are the list in `src/brand.rs` (`isom`, `iso2` to `iso9`, `mp41`, `mp42`, `M4A `, `M4V `, `qt  `, the 3GPP brands, `dash`, `cmfc`, `heic`, `avif`, and others).

Reading:

| Part of the file | What the demuxer does |
| --- | --- |
| Box sizes | 32-bit size, size 0 (to the end of the parent), 64-bit `largesize`, and `uuid` boxes. |
| Index place | `moov` before or after `mdat`. |
| Sample tables | `stts`, `ctts`, `stsc`, `stsz`, `stz2`, `stco`, `co64`, `stss`. A missing `stss` means every sample is a sync sample. |
| Fragments | `mvex`, `moof`, `tfhd`, `tfdt`, `trun`, read in order. Seek uses `mfra` or `sidx`; with neither, it walks the `moof` boxes. |
| Edit list | `elst` is returned on the stream. It is not applied to packet times. |
| Display matrix | The `mvhd` and `tkhd` matrices are multiplied. A clockwise turn of 0, 90, 180, or 270 degrees is reported on the stream. |
| HEIC and AVIF | Each item of the `meta` box is one packet at timescale 1, the primary item first. |

One packet is one sample. `dts` and `pts` are ticks of the track's own timescale, and `pts` is `dts` plus the composition offset. Packets come back in file order (lowest byte offset first).

The sample entry names the codec:

| Sample entry | Codec id |
| --- | --- |
| `avc1`, `avc3` | H.264 |
| `hvc1`, `hev1`, `dvh1`, `dvhe` | HEVC |
| `av01` | AV1 |
| `vp08`, `vp09` | VP8, VP9 |
| `mp4v` | MPEG-4 Visual, or MPEG-2 or Motion JPEG when the `esds` object type says so |
| `mp2v`, `m2v1`, `hdv*`, `mx3n` to `mx5p`, `xd5*`, `xdv*`, `xdhd`, `xdh2`, `AVmp`, `m1v `, `m1v1`, `mpeg`, `mp1v` | MPEG-2 |
| `jpeg`, `mjpa`, `mjpb` | Motion JPEG |
| `apch`, `apcn`, `apcs`, `apco`, `ap4h`, `ap4x` | ProRes |
| `mp4a` | AAC, or MP3, AC-3, or E-AC-3 when the object type says so |
| `.mp3`, `Opus`, `fLaC`, `alac` | MP3, Opus, FLAC, ALAC |
| `ac-3`, `sac3`, `ec-3` | AC-3, E-AC-3 |
| `sowt`, `twos`, `raw `, `in24`, `in32`, `fl32`, `fl64`, `lpcm`, `NONE` | PCM |

The configuration boxes `avcC`, `hvcC`, `av1C`, `vpcC`, `esds`, `alac`, `dOps`, `dac3`, `dec3`, and `dfLa` are kept as the stream's extradata. Three entries are reshaped: an ALAC track gives the bare magic cookie (the `alac` box joined with its `chan` box), `IsoBmffFile` turns an Opus `dOps` into an `OpusHead`, and a ProRes entry, which has no configuration box, gives its fourcc. A QuickTime Motion JPEG entry gives its fourcc too, followed by the two `fiel` bytes when that box is present. An unknown sample entry is `CodecId::Unknown` and its packets are still returned.

Writing, with `IsoBmffMux`:

| `FileBrand` | Major brand | Compatible brands |
| --- | --- | --- |
| `Mp4` | `isom` | `isom`, `iso2`, `mp41` |
| `Mov` | `qt  ` | `qt  ` |
| `M4a` | `M4A ` | `M4A `, `mp42`, `isom` |
| `M4v` | `M4V ` | `M4V `, `mp42`, `isom` |
| `ThreeGp` | `3gp4` | `3gp4`, `isom` |

The default layout is `ftyp`, `mdat`, `moov`. `moov_at_end: false` puts `moov` first and holds the payloads in memory until `finish`. `fragmented: true` writes `moov` with `mvex`, then `moof`/`mdat` pairs, then `mfra`, under the brands `iso5`, `iso6`, `mp41`. Decode times on a track must start at 0 and continue by each packet's duration. `set_edits` writes an `elst`.

`Mp4Mux` is the export muxer. It accepts H.264, HEVC, MPEG-4 Visual, MPEG-2, ProRes, VP8, AV1, and Motion JPEG pictures, and AAC, ALAC, Opus, AC-3, E-AC-3, and 16-bit PCM sound. Motion JPEG is written only to a QuickTime movie. An AV1 track is an `av01` entry with its `av1C` record in a file that lists the `av01` brand; each sample is the temporal unit without its temporal delimiter, padding and redundant frame headers, every OBU with a size (AV1 Codec ISO Media File Format Binding 2.1 and 2.4).

What is refused:

- An unknown major brand with no known compatible brand: `Error::UnsupportedBrand`. A file that is not this family: `Error::NotThisFamily`.
- A box size that runs past its parent: `Error::HostileSize`. A short file: `Error::Truncated`.
- Media data in another file (an external data reference): `Error::ExternalReference`.
- A sample outside `mdat`: `Error::SampleExtendsPastMedia`.
- A `sidx` that points at another `sidx`, a HEIF item with more than one extent, and a HEIF construction method other than 0 or 1: `Error::Unsupported`.
- A seek on a track whose `stss` is empty: `Error::SeekLimited`. Reading such a track in order still works.
- In the muxer, Vorbis, JPEG, PNG, WebP, and GIF tracks: `Error::CodecRejected`.

## Using it

There are two sets of entry points.

- `IsoBmff` (demux) and `IsoBmffMux` (mux) use the types this crate exports itself: `Demux`, `Mux`, `Source`, `Packet`, `Stream`, `StreamParams`, `Error`. `FileSource` and `SliceSource` implement `Source`.
- `IsoBmffFile` and `Mp4Mux` wrap those two and implement `cutlass_container::Demux` and `cutlass_container::Mux`. The app and the export crate use these.

`probe` looks at the first bytes of a file and says whether the brand is known. It needs at most `PROBE_BYTES` (64). `open` is the final answer.

```rust
use cutlass_isobmff::{Demux, FileSource, IsoBmff};

let mut source = FileSource::open(path)?;
let mut movie = IsoBmff::open(&mut source)?;
let _movie_timescale = movie.movie_timescale();
while let Some(packet) = movie.read_packet(&mut source)? {
    let _presentation_ticks = packet.pts;
    let _track_timescale = packet.timescale;
}
```

```rust
use cutlass_isobmff::{IsoBmffMux, Mux, MuxOptions};

let mut mux = IsoBmffMux::new(writer, MuxOptions::default());
let track_id = mux.add_stream(&params)?;
mux.write_packet(&packet)?;
mux.finish()?;
```

`seek` moves to the sync sample at or before a decode time on the named track. A failed seek leaves the read position where it was. `finish` writes the index; a muxer dropped before `finish` leaves a file with no `moov`, and `open` refuses it.

`cargo run --release -p cutlass-isobmff --example demux` asks FFmpeg for a short MP4, opens it, and prints the stream count, the packet count, and the first timestamp. It looks for FFmpeg only in the Homebrew location on macOS and prints `skip: ffmpeg is not installed` when it is not there.

## How it is tested

```sh
cargo test -p cutlass-isobmff
```

- Unit tests in `src/behavior.rs` build files by hand and through the muxer. They cover both index places, the sample tables, negative composition offsets, edit lists, the display matrix, seek, fragments, a HEIF item, an unknown codec, and hostile input (short header, wrong family, size past the parent, 40 nested boxes, external reference).
- `src/oracle.rs` compares against FFmpeg. `ffmpeg_and_gstreamer_see_the_same_packets` reads FFmpeg's progressive MP4, fragmented MP4, and QuickTime files and compares packet count, payload hash, size, duration, keyframe flag, and composition offset with `ffmpeg -c copy -f framehash`. `ffmpeg_reads_files_this_muxer_wrote` does the reverse for files this muxer wrote. `ffmpeg_motion_jpeg_files_demux_and_remux` covers Motion JPEG. These tests print `not available` and pass when `ffmpeg` or `ffprobe` is not on the `PATH`. GStreamer's `gst-discoverer-1.0` is asked for the codec and width when it is installed.
- Two fuzz loops run only when asked: `ISOBMFF_FUZZ_SECS=<seconds>` for probe, open, read, and seek on mutated input, and `ISOBMFF_MUX_FUZZ_SECS=<seconds>` for a mux round trip. Without the variable they print `not run` and pass.
- Four tests read real media from a folder kept outside the repository, and one of them rewrites the files only when `ISOBMFF_LOCAL_WRITE=1`. They pass without checking anything when the folder or the named file is absent.
- `open_and_seek_stay_under_a_coarse_bound` opens a 4000-sample file and fails if open takes 2 s or a seek takes 500 ms.
- `cargo bench -p cutlass-isobmff --bench against_ffmpeg` checks packet count, sizes, and times against `ffprobe -show_packets` before it times anything.

## Speed against FFmpeg

Measured on 10 November 2026 at commit `ed0743d2`, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1, against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.

**Read every packet, mp4.** Input: one 1975 MB file. Nothing is decoded.

|  | Cutlass | FFmpeg (libavformat) |
| --- | ---: | ---: |
| Wall time | 0.16 s | 0.17 s |
| CPU time | 0.15 s | 0.16 s |
| Peak memory | 14 MB | 29 MB |
| Packets returned | 131,520 | 131,520 |

Cutlass is **1.1× faster** than FFmpeg here.

Writing files was not measured.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Limits, from `src/limits.rs`:

| Limit | Value |
| --- | --- |
| `moov`, `moof`, `meta`, or `sidx` held in memory | 64 MiB |
| Tracks | 64 |
| Samples per track | 4,000,000 |
| One packet | 64 MiB |
| Box nesting | 32 |
| Boxes visited at open | 2,000,000 |
| Edits per track | 4,096 |
| Sample descriptions per track | 16 |
| Payload held when `moov` is written first | 256 MiB |

Gaps:

- Edit lists are returned and never applied to packet times.
- A flip or any matrix that is not a quarter turn is reported as no turn. The muxer writes only the identity matrix.
- Only the first sample description of a track is used.
- A `trun` version 0 composition offset is read as unsigned. `ctts` is read as signed in both versions.
- HEIF: a grid item is returned as its own packet and the picture is not assembled. The muxer does not write HEIF items. AVIF items go through the same code as HEIC items but have no test file.
- `sidx` is parsed and tested on its own; no test seeks a whole file through it. The muxer does not write `sidx`.
- The muxer always writes `co64` with one sample per chunk. `mvhd`, `tkhd`, `mdhd`, and `elst` are version 0, so a duration above `u32::MAX` is stored as `u32::MAX`.
- If a write fails in the middle of a fragment after its `moof` went out, that `moof` is not taken back.
- The FFmpeg comparison in this crate uses H.264, AAC, and Motion JPEG files. Most other sample entries are tested with hand-built entries. The AV1, VP9, FLAC, and MP3 entries have no test in this crate.
- FFmpeg (`ffmpeg` and `ffprobe` share one library) is the only tool the packet comparison uses. GStreamer confirms codec and width only.

## Specifications

These are the documents the source comments name:

- ISO/IEC 14496-12, ISO base media file format.
- ISO/IEC 14496-1, for the `esds` object type indications.
- Encapsulation of Opus in ISO Base Media File Format, version 0.6.8, and RFC 7845 (`OpusHead`).
- Apple's `ALACMagicCookieDescription.txt`, for the ALAC cookie.
- SMPTE RDD 36, for the color bytes of a ProRes frame header.
