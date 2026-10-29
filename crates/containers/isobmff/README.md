# cutlass-isobmff

## What this crate is

`cutlass-isobmff` is a native safe-Rust demuxer and muxer for the ISO base media file format. It opens a file into timed packets and writes packets back into a file. The payload of a packet is the sample bytes. This crate does not decode those bytes into pictures or PCM.

Timestamps stay in the file timescale. A packet's `dts` and `pts` are ticks of that track's media timescale. The crate does not convert them to microseconds. `#![forbid(unsafe_code)]` is set, and the crate has no `unsafe` block.

## Specification

The boxes this crate reads and writes follow ISO/IEC 14496-12:2022, together with the sample-entry layouts those brands use for AVC, HEVC, and HEIF items. The walker covers `ftyp`, a headerless QuickTime start, `moov` and its sample tables (`stts`, `ctts`, `stsc`, `stsz`, `stz2`, `stco`, `co64`, `stss`), `elst`, `sidx`, `moof` / `tfhd` / `tfdt` / `trun`, `mfra`, and HEIF `meta` items. Edition 8 additions are not implemented.

## API

Open and read go through `IsoBmff`, which implements `Demux`. Write goes through `IsoBmffMux`, which implements `Mux`.

```rust
use cutlass_isobmff::{Demux, FileSource, IsoBmff, Mux};

let mut source = FileSource::open(path)?;
let mut movie = IsoBmff::open(&mut source)?;
let _movie_timescale = movie.movie_timescale();
while let Some(packet) = movie.read_packet(&mut source)? {
    let _presentation_ticks = packet.pts;
    let _track_timescale = packet.timescale;
}
```

`movie_timescale()` is the movie header timescale. Each `Packet` carries its own track timescale. `pts` is the sample-table presentation time (`dts` plus the composition offset). `read_packet` returns samples in file order. `seek` moves to the sync sample at or before a decode time on one track, and to the sample at or before the scaled time on the others.

`probe` looks at the first bytes of a header and reports whether the brand is one this crate knows. `open` is the authority: a brand list that continues past the 64-byte probe is decided there.

```rust
use cutlass_isobmff::{IsoBmffMux, Mux, MuxOptions};

let mut mux = IsoBmffMux::new(writer, MuxOptions::default());
let track_id = mux.add_stream(&params)?;
mux.write_packet(&packet)?;
mux.finish()?;
```

`MuxOptions` selects `FileBrand` (`Mp4`, `Mov`, `M4a`, `M4v`, `ThreeGp`), whether `moov` is last, and whether output is fragmented. Decode times on a track start at 0 and then equal the previous decode time plus that sample's duration. `set_edits` writes the edit list. `finish` writes the index. Dropping the muxer before `finish` leaves the index unwritten.

`FileSource` and `SliceSource` implement `Source`.

## Support

A file opens when `ftyp` has a known major brand or a known compatible brand, or when a headerless QuickTime file starts with `moov`, `mdat`, `wide`, `free`, `skip`, or `pnot`. Known brands:

`isom`, `iso2`, `iso3`, `iso4`, `iso5`, `iso6`, `iso7`, `iso8`, `iso9`, `mp41`, `mp42`, `mp71`, `avc1`, `dash`, `cmfc`, `msdh`, `msix`, `piff`, `M4A `, `M4B `, `M4P `, `M4V `, `qt  `, `3gp4`, `3gp5`, `3gp6`, `3gp7`, `3gp8`, `3gp9`, `3g2a`, `3g2b`, `3g2c`, `3ge6`, `3ge7`, `3gg6`, `kddi`, `mif1`, `msf1`, `miaf`, `heic`, `heix`, `hevc`, `hevx`, `avif`, `avis`.

The muxer writes these `ftyp` brands:

| `FileBrand` | Major | Compatible |
| --- | --- | --- |
| `Mp4` | `isom` | `isom`, `iso2`, `mp41` |
| `Mov` | `qt  ` | `qt  ` |
| `M4a` | `M4A ` | `M4A `, `mp42`, `isom` |
| `M4v` | `M4V ` | `M4V `, `mp42`, `isom` |
| `ThreeGp` | `3gp4` | `3gp4`, `isom` |

Fragmented output forces major `iso5` and compatible `iso5`, `iso6`, `mp41`. The demuxer reads a fragmented movie sequentially from `moof` and `trun`, then seeks through `mfra` or `sidx`. A `sidx` reference to another index is rejected. The muxer writes `mfra` and does not write `sidx`.

An edit list is stored on `Stream.edits`. An empty edit has `media_time: None`. Packet `dts` and `pts` stay on the sample table; the edit is not applied to them. `set_edits` writes `elst` version 0.

Tracks are video, audio, or data. The sample-description walker maps `avc1` / `avc3` to H.264, `hvc1` / `hev1` / `dvh1` / `dvhe` to HEVC, `av01` to AV1, `vp08` and `vp09` to VP8 and VP9, `mp4v` to MPEG-4 visual, `mp2v` to MPEG-2, `jpeg` / `mjpa` / `mjpb` to Motion JPEG, and the ProRes fourccs `apch`, `apcn`, `apcs`, `apco`, `ap4h`, `ap4x`. Audio maps `mp4a` to AAC (or MP3 / AC-3 from the object type), `.mp3` to MP3, `Opus`, `fLaC`, `alac`, `ac-3` / `sac3` / `ec-3`, and PCM fourccs `sowt`, `twos`, `raw `, `in24`, `in32`, `fl32`, `fl64`, `lpcm`, `NONE`. Configuration records (`avcC`, `hvcC`, `av1C`, `esds`, and the matching audio boxes) are kept as extradata. This crate does not parse the codec bitstream inside a sample. An unknown fourcc is `CodecId::Unknown` and the samples are still returned. The muxer rejects Vorbis, JPEG, PNG, WebP, and GIF.

HEIC and AVIF items are packets at timescale 1, one packet per item, including a grid item's own payload. The still image is not assembled. Construction methods 0 and 1 are read. Any other construction method is rejected. A `moov` in the same file wins, and the items are ignored. The muxer does not write HEIF items.

An external data reference fails at open. Public bounds include a 64-byte probe, a 64 MiB metadata box, 64 tracks, 4,000,000 samples, and a 64 MiB packet.

## Accuracy

The bench below writes one MP4 with ffmpeg 8.1.1: `testsrc2` at 64×64, 10 fps, for 0.4 s, plus a 1000 Hz sine at 48000 Hz, encoded as H.264 and AAC, with `-use_editlist 0`. Before any timing, `IsoBmff` reads every packet and compares packet count, payload size, and `pts` with `ffprobe -show_packets`. The file has no edit list, so ffprobe's packet pts are the sample-table pts this crate returns.

That comparison matched. The same run reported 2 streams and 24 packets: H.264 at timescale 10240 (4 packets) and AAC at timescale 48000 (20 packets).

The muxer then wrote the first 4 packets of each track (8 packets) with `IsoBmffMux` and `MuxOptions::default`. ffprobe read that file back with the same packet count, sizes, and pts.

A file that does contain an edit list still returns sample-table timestamps from this crate. ffprobe's default packet pts include the edit, so those two pts values differ by the edit. This bench's file was generated without one, and the match above is for that file.

## Performance

One sample. The cutlass figure is a fresh open plus a read of every packet, median of 3 after one warmup. The ffprobe figure is `ffprobe -show_packets` on the same file, median of 3 after one warmup. The ffmpeg time includes process startup.

ffmpeg version: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`

`sysctl -n machdep.cpu.brand_string`: `Apple M5 Pro`

Date: `Thu Oct 29 21:04:04 GMT 2026`

Command: `cargo bench -p cutlass-isobmff`

Bench stdout:

```text
ffmpeg_version: ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
cpu: Apple M5 Pro
date: Thu Oct 29 21:04:04 GMT 2026
command: /opt/homebrew/bin/ffmpeg -y -hide_banner -loglevel error -f lavfi -i testsrc2=size=64x64:rate=10:duration=0.4 -f lavfi -i sine=frequency=1000:sample_rate=48000:duration=0.4 -c:v libx264 -preset ultrafast -pix_fmt yuv420p -g 10 -bf 0 -c:a aac -b:a 64k -shortest -use_editlist 0 /var/folders/vq/1wh9t5d5487bq7cf3y400n4r0000gn/T/cutlass-isobmff-bench/sample.mp4
streams: 2
packets: 24
stream 0: H264 timescale 10240 packets 4
stream 1: Aac timescale 48000 packets 20
compare: packet count, sizes, and pts match ffprobe -show_packets
sample: 1 file, median of 3 after 1 warmup
cutlass_open_read_s: 0.000033 0.000033 0.000035 median 0.000033
ffprobe_show_packets_s: 0.017368 0.017467 0.017906 median 0.017467
ffmpeg time includes process startup
mux_packets: 8
mux_compare: packet count, sizes, and pts match ffprobe -show_packets
```

The medians are 0.000033 s for open and read, and 0.017467 s for ffprobe.

## Example

```sh
cargo run --release -p cutlass-isobmff --example demux
```

The example asks `/opt/homebrew/bin/ffmpeg` for the same short MP4, opens it with `IsoBmff`, and prints the stream count, the packet count, and the first packet's presentation timestamp. If that ffmpeg binary is absent it prints `skip: ffmpeg is not installed` and exits 0.

## Gaps

Codec bitstreams are not parsed. Vorbis, JPEG, PNG, WebP, and GIF are rejected by the muxer. Unknown sample entries are returned as packets labeled `CodecId::Unknown`.

Edit lists are exposed and are not applied to packet timestamps. Fragmented files are read and written as described above; `sidx` is parsed, a reference to another index is rejected, and the muxer does not emit `sidx`. HEIF items are packets, the image is not assembled, and construction methods other than 0 and 1 are rejected. The muxer does not write HEIF items.

Only the first sample description is exposed on a track. The muxer writes `co64` chunk offsets. `mvhd`, `tkhd`, `mdhd`, and `elst` are version 0, and a duration that does not fit in `u32` is stored as `u32::MAX`. An external media file is refused at open. Edition 8 boxes are not implemented.
