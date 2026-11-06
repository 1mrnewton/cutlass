# cutlass-ogg

## What this crate is

`cutlass-ogg` demuxes and muxes Ogg with the Opus mapping. The implementation is native safe Rust, and the crate forbids `unsafe`. It reads pages, checks their checksums, joins packets across pages, and gives each Opus packet its time from the page granule positions. Packet bytes are the Opus packets as stored; this crate does not decode them.

FFmpeg 8.1.1 and GStreamer 1.28.3 are the test oracles. The crate links `cutlass-codec` and `cutlass-container` only.

## Specification

- RFC 3533, The Ogg Encapsulation Format, for pages, lacing, the checksum, and grouping.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec, for `OpusHead`, `OpusTags`, granule positions, pre-skip, end trimming, and seeking with pre-roll.
- RFC 6716 section 3.1, for the TOC byte that gives each packet's duration.

The longer record of sources, limits, decisions, and test results is `STATUS.md`.

## API

`OggDemux` implements `cutlass_container::Demux`. `OggMux` implements `cutlass_container::Mux`. `probe` and `OggFamily` recognize the five bytes `OggS\0`.

`OggDemux::open` takes a random-access reader and the file length. `open_forward` reads once from the start: it cannot seek, and the end of the stream is unknown until it is read. Each Opus stream is a `StreamParams::try_audio(Known(Opus), 48000, channels, OpusHead)` with timescale 48000. Any other mapping is listed as `Unknown("ogg:vorbis")`, `Unknown("ogg:flac")`, and so on, and its pages are skipped. `opus(track)` gives the parsed `OpusHead`, the `OpusTags`, and the first and last sample positions.

Times are in the granule domain: a packet's presentation and decode time is the granule position of its first sample, before pre-skip. FFmpeg's `pts` is this value minus the pre-skip. Each stream carries `Trim::new(Some(pre_skip), trailing)` and one `Edit` whose `media_time` is `start + pre_skip`, so the engine plays the same samples as an RFC 7845 player.

```rust
use std::fs::File;

use cutlass_container::{Demux, SeekTo};
use cutlass_ogg::OggDemux;

let file = File::open(path)?;
let length = file.metadata()?.len();
let mut demux = OggDemux::open(file, length)?;
let stream = &demux.streams()[0];
let pre_skip = stream.trim().encoder_delay();
while let Some(packet) = demux.read_packet()? {
    let _granule_time = packet.timestamp().presentation();
    let _opus = packet.bytes();
}
demux.seek(SeekTo { track: 0, time: 48_000 * 60, end: None })?;
```

`OggMux::new` writes Ogg Opus as packets arrive. The `OpusHead` comes from the stream's extradata or from `set_config` before the first packet; an MP4 `dOps` body is accepted and rewritten. Packets must be contiguous in time per stream, and only the last packet may be shorter than its TOC duration: it becomes end trimming on the last page.

```rust
use std::io::Cursor;

use cutlass_codec::{CodecId, StreamCodec, StreamParams, Timescale};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_ogg::OggMux;

let mut mux = OggMux::new(Cursor::new(Vec::new()));
let track = mux.add_stream(NewStream {
    params: StreamParams::try_audio(StreamCodec::Known(CodecId::Opus), 48_000, 2, opus_head)?,
    timescale: Timescale::new(48_000),
    language: None,
})?;
for packet in &packets {
    mux.write_packet(packet)?;
}
mux.finish(&[TrackDuration { track, timescale: Timescale::new(48_000), ticks }])?;
let bytes = mux.into_inner().into_inner();
```

## Support

| Kind | Demux | Mux |
| --- | --- | --- |
| Opus, channel mapping family 0 and 1 | Opens, times, trims, seeks | Writes |
| Opus, channel mapping family 255 | Parsed, not checked against an independent file | Accepted |
| Several logical streams in one file | Each Opus stream is a track | Each stream is a track |
| Vorbis, FLAC, Theora, Speex, other mappings | Listed as `Unknown`, pages skipped | Refused |
| Chained files | First link only, `chained()` is true | One link |
| Damaged pages | Checksum failures resync, lost packets become `gaps()` | n/a |
| Opus in WebM or Matroska | Out of scope | Out of scope |

## Tests

`cargo test -p cutlass-ogg` runs the unit and behavior tests and a short mutation campaign. The tests in `tests/external` need ffmpeg, ffprobe, and the `local-files` folder (`CUTLASS_LOCAL_FILES` names it), and skip when one is missing. GStreamer checks skip when `gst-launch-1.0` is not installed.
