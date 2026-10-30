# cutlass-mp3-container

## What this crate is

Native safe Rust demuxer and muxer for an MPEG audio Layer III file. The library forbids `unsafe`. A packet is one Layer III frame, including its four-byte header. `pts`, `duration`, and `Mp3Demux::duration` are sample counts at that frame's sample rate. This crate does not decode those samples to PCM.

The codec crate is `cutlass-mp3` in `crates/codecs/mp3`. This package only frames the file.

## Specification

Frame layout and the Layer III byte count follow ISO/IEC 11172-3:1993 clause 2.4.1.3 and ISO/IEC 13818-3:1998 clause 2.4.2.3. MPEG-2.5 rates (8 kHz, 11.025 kHz, 12 kHz) use the MPEG-2 Layer III size rules with the base rate divided by four. A Xing or Info tag, including the LAME encoder-delay and padding fields, and a VBRI header at byte 36 of the first frame, are an index. ID3v2 sizes are the synchsafe integers in the id3.org 2.4.0 header. ID3v1 is the 128-byte `TAG`. An APEv2 footer ends the audio.

## API

Open a file with `Mp3Demux::open`. The reader must implement `Read + Seek`. The value implements `Demux`:

- `container_id` returns `ContainerId::Mp3`.
- `streams` is one stream. `timescale` and `CodecParams::sample_rate` are the frame sample rate. `channels` is 1 or 2.
- `duration` is the sample count when a Xing, Info, VBRI, or constant-bitrate schedule provides it. `duration_exact` says whether that count came from a frame total.
- `next_packet` returns the next audio frame. A Xing, Info, or VBRI frame is not returned. `Packet::pts` and `Packet::duration` use `Packet::timescale`.
- `seek` takes a sample time and returns `SeekLanding` (`pts`, `indexed`, `preroll_frames`). `index_kind` reports `IndexKind::XingToc`, `VbriToc`, `ConstantBitrate`, `AverageBytes`, or `None`.
- `cancel` makes the next read or seek return `ContainerError::Cancelled`.

`Mp3Demux` also implements `cutlass_container::Demux`. `read_packet` yields a `cutlass_codec::Packet` with the same frame bytes and sample timestamps. `consumed` is the number of bytes read from the input.

Build a file with `Mp3Mux::new`. The writer must implement `Write + Seek`. The value implements `Mux`:

- `add_stream` accepts one `CodecParams` whose codec is `CodecId::Mp3`.
- `write_packet` stores the frame bytes unchanged. `pts` must be the next sample on that stream, and `duration` must be the header's samples per frame.
- `finish` fills the leading Xing or Info frame with the frame count, byte count, and seek table.
- `into_inner` returns the writer after `finish`.

`probe` classifies a prefix as `Probe::Matched`, `Probe::NeedMore`, or `Probe::Reject`. Public bounds are `MAX_FRAME_BYTES`, `MAX_ID3_BYTES`, `MAX_RESYNC_BYTES`, and `SEEK_WINDOW_BYTES`.

`Mp3Mux` implements this crate's `Mux` trait. It does not implement `cutlass_container::Mux`.

## Support

`SUPPORT.md` is the capability record. Summary:

| File | Demux | Mux |
| --- | --- | --- |
| MPEG-1, MPEG-2, and MPEG-2.5 Layer III | One frame per packet | One `CodecId::Mp3` stream |
| Constant and variable bitrate | Sequential read; seek uses the index when one exists | Info when every packet bitrate matches, otherwise Xing |
| Xing, Info, VBRI | Index only, not a packet | Leading Xing or Info frame |
| Free format | Stride taken from the next sync | Packets written as given |
| Encoder delay and padding | `Trim` on the stream | LAME slot when both values are at most 4095 |
| ID3v2, ID3v1, APEv2 | Skipped | Not written |
| Layer I and Layer II | `Unsupported` | `Unsupported` |

A positive seek with no Xing, VBRI, or constant schedule returns `SeekUnavailable` and leaves the read cursor where it was.

## Accuracy

One file, checked before the clock started. ffmpeg 8.1.1 (`/opt/homebrew/bin/ffmpeg`) wrote a two-second stereo MP3 with libmp3lame at 128 kbit/s and 48 kHz. The file was 33069 bytes. `Mp3Demux` and ffprobe agreed on 85 packets. Every packet size matched. Every presentation timestamp matched after ffprobe ticks were converted with the stream time base and the first packet was aligned to the demuxer's first `pts`. ffprobe did not have an extra leading packet (`dropped_leading_tag=no`). The comparison is frame sizes and timestamps. The demuxer does not return PCM, so the bench does not compare a decoded waveform.

The same run wrote eight stereo frames with `Mp3Mux` and opened that file in ffprobe. ffprobe's packet count, sizes, and timestamps matched those eight frames (`dropped_leading_tag=no`).

## Performance

One sample: the 33069-byte file above. Each cutlass time is `Mp3Demux::open` plus a read of every frame. Each ffprobe time is a new process that lists every packet, so it includes process startup. The three nanosecond samples are printed in sorted order. The median is the middle value, after one untimed warmup.

Command, from this worktree:

```text
cargo bench -p cutlass-mp3-container
```

ffmpeg version, from `/opt/homebrew/bin/ffmpeg -version`:

```text
ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
```

`sysctl -n machdep.cpu.brand_string`:

```text
Apple M5 Pro
```

Date of the run: Thu Oct 29 21:01:13 GMT 2026.

Bench stdout:

```text
ffmpeg: /opt/homebrew/bin/ffmpeg
sample: ffmpeg libmp3lame, 2 s, stereo, 48000 Hz, 128 kbit/s
sample_bytes: 33069
channels: 2
compare: packets=85 sizes=match pts=match dropped_leading_tag=no
cutlass_samples_ns: 88083 90583 90792
cutlass_median_ns: 90583
cutlass_median_s: 0.000090583
ffprobe_samples_ns: 18865500 18889208 19165000
ffprobe_median_ns: 18889208
ffprobe_median_s: 0.018889208
timing: median of 3 after 1 warmup
ffprobe_time_includes_process_startup: yes
mux: packets=8 sizes=match pts=match dropped_leading_tag=no
```

## Example

`examples/demux.rs` writes eight stereo frames with `Mp3Mux` and reads them with `Mp3Demux`. Pass a path to open that file instead.

```text
cargo run --release -p cutlass-mp3-container --example demux
```

```text
packets: 8
duration_samples: 9216
timescale: 48000
first_pts: 0
last_pts: 8064
```

`duration_samples` is the demuxer's duration. `first_pts` and `last_pts` are the first and last frame timestamps, in samples.

## Gaps

Layer I and Layer II are refused. ID3 text and pictures are not exposed, and the muxer writes no ID3 tag. Encoder delay and trailing padding stay on `Trim`. They are not subtracted from `pts`, and no samples are dropped. A lone free-format frame, with no second header inside the resync window, is `Unsupported`.

`Mp3Mux` does not implement `cutlass_container::Mux`. The desktop package lists this crate as a dependency. `engine`, `export`, and the desktop source do not open an MP3 through it. Sample decode stays in `cutlass-mp3`.

The timed result above is one two-second file. It is not a seek measurement and not a long-file measurement. The earlier seek budget and the fuzz campaign are recorded in `SUPPORT.md`.
