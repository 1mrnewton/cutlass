# MP3 container

Demux and mux for an MPEG audio Layer III file. The packet is one frame, including its header. Times are samples at the frame's sample rate. This crate does not decode samples. The `mp3` codec crate is still empty.

Checkout `mp3-container` at `6dfa2b6`, parent `109d16f`. Both directions were requested.

## Scope

Normative layout:

- ISO/IEC 11172-3:1993 clause 2.4.1.3 and ISO/IEC 13818-3:1998 clause 2.4.2.3 for the header and the Layer III byte count.
- Fraunhofer MPEG-2.5 rates: 8 kHz, 11.025 kHz, and 12 kHz, using the MPEG-2 Layer III frame rules.
- Xing and Info tags, including the LAME 12-bit encoder delay and trailing padding.
- VBRI at byte 36 of the first frame, version 1.
- ID3v2 synchsafe size (id3.org 2.4.0 header) and the 128-byte ID3v1 `TAG`.
- APEv2 footer at the end of the file.

Read before implementation: FFmpeg 8.1.1 `libavformat/mp3dec.c`, `mp3enc.c`, `id3v2.c`, `id3v1.c`, and `libavcodec/mpegaudiodecheader.c` with `mpegaudiotabs.h`; GStreamer `gstmpegaudioparse.c`; LAME `VbrTag.c`; mpg123 `parse.c`; the Symphonia MP3 demuxer. Those trees were not copied into this repo.

`crates/engine` and `crates/export` are empty. Playback and export were not exercised. Tests call `Demux` and `Mux` directly. Timestamps are not converted to microseconds.

## Free format

Free format (bitrate index 0) is supported when a second compatible free-format header within 4096 bytes fixes the stride. A lone free-format frame at the start of a file that ends before that second header is `Unsupported { what: "free format" }`. The muxer accepts a free-format packet and writes a Xing frame in front of it. The stride is learned from the audio frames, not from the Xing frame.

## What is not a packet

A Xing, Info, or VBRI frame is an index. It is not returned. Encoder delay and trailing padding are `Trim` on the stream and the same fields on `CodecParams`. They are the 12-bit values in the LAME extension. This crate does not add the decoder filterbank delay and does not drop samples.

ID3v2 at the start (up to four tags) and ID3v1 or APEv2 at the end are skipped. Title, artist, and pictures are not exposed. The muxer does not write an ID3 tag.

Layer I and Layer II are `Unsupported { what: "layer" }` when two headers confirm the layer. They are not resynchronized into silence.

## Capability matrix

| Capability | Required | Demux | Mux | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| MPEG-1 Layer III | Yes | One frame per packet, timescale = sample rate | One `CodecId::Mp3` stream | Unit tests, ffprobe, mpg123 | Verified |
| MPEG-2 and MPEG-2.5 Layer III | Yes | 576 samples per frame; reservoir preroll 12 | Header must match the stream | `mpeg25_uses_the_file_sample_rate_and_a_twelve_frame_reservoir` | Verified |
| CBR, including padding | Yes | Stride or bitrate schedule | Info tag when every packet bitrate matches | 48 kHz round trip; 44.1 kHz formula tests | Verified |
| VBR with Xing or VBRI | Yes | TOC or VBRI table | Xing tag | Xing, VBRI, and LAME VBR oracle | Verified |
| VBR without an index | Yes | Sequential read works | n/a | `SeekUnavailable` on a positive seek | Verified |
| Free format | Yes, decision recorded above | Stride from the next sync | Opaque packets, file marked Xing | Free-format unit tests | Verified |
| Gapless delay and padding | Yes as metadata | `Trim` only | LAME slot, delay and padding ≤ 4095 | Round trip 576/100, frame kept | Verified |
| ID3v2, ID3v1, APEv2 | Skip, bounded | Not packets | Not written | ID3 and APE tests; hostile ID3 is `Limit` | Verified |
| Layer I and II | No | `Unsupported` when confirmed | `Unsupported` | Layer II fixture | Unsupported |
| ID3 text and pictures | No | Skipped with the tag | No field to write | By scope | Unsupported |
| Sample decode | No | Outside this crate | Outside this crate | `crates/codecs/mp3` is empty | Outside scope |
| Engine and export | Integration | Traits only | Traits only | Scaffolds are empty | Not run |

## Limits

| Limit | Value |
| --- | --- |
| ID3v2 tag skipped, never allocated | 16 MiB |
| APEv2 span | 16 MiB |
| Resync without a confirmed frame | 64 KiB |
| Frame, including free format | 4096 bytes |
| Seek search either side of the index | 4096 bytes |
| VBRI entries stored | 4096 |
| Leading ID3v2 tags | 4 |
| Read buffer | 32 KiB |

A declared ID3 size above the limit returns `Limit { what: "id3" }` and does not allocate the tag. A positive seek with no Xing, VBRI, or constant schedule returns `SeekUnavailable` and does not move the read cursor.

Reservoir preroll reported to the codec, and not returned as packets: 9 frames for MPEG-1, 12 for MPEG-2 and MPEG-2.5.

## Oracles

ffprobe 8.1.1 (libavformat) is the per-packet oracle. Packet times use time base 1/14112000 and are converted with `ticks * sample_rate / 14112000`, then shifted so the first compared audio frame is presentation time 0. One leading ffprobe packet is dropped when that makes the size sequences equal, which is the Xing or Info frame ffprobe sometimes still emits. Payload sizes and those times matched for:

- ffmpeg 8.1.1 `libmp3lame`, 128 kbit/s, 48 kHz, 0.12 s sine
- LAME 3.100 `-V 4` from a 44.1 kHz stereo wav
- a file this muxer wrote, opened again by ffprobe

mpg123 1.33.5 is a second implementation. It confirms the sample rate and the frame count (its count is this crate's count, or one more when it still counts the tag). It does not print every packet. ffmpeg's libmp3lame and the `lame` binary are the same encoder library.

GStreamer is installed, and `gstmpegaudioparse.c` was read. `gst-launch-1.0` on this machine does not finish: the plugin scanner fails to load `libgobject` and `libglib` and then stalls. GStreamer was not used as an oracle. That is `not available`, not a pass.

Local files under `../local-files` were left in place: `thunder_distant_10s.mp3`, `sfx_A_sin_20260721_231942.mp3`, `Seamless_gentle_rain_#1-1784676090807.mp3` (full demux, monotonic sample clock, ffprobe rate and channel match), and `rain_bedroom_night_30min.mp3` (indexed seek to half the duration). The test skips when that folder is missing.

## Performance

Apple M5 Pro, 18 cores, 48 GiB (`hw.memsize` 51539607552). Release test `indexed_seek_does_not_scan_and_release_time_stays_in_budget`.

File: 1,536,384 bytes, 4000 MPEG-1 frames at 128 kbit/s and 48 kHz, Info tag at the start. The read is a `Cursor` over that buffer.

| Measure | Result | Budget |
| --- | --- | --- |
| Open through the first packet | 278 µs | 30 ms |
| Sequential read of 4000 frames | 1.87 ms | 80 ms |
| Indexed seek to frame 2000 | 2.3 µs | 20 ms |
| Bytes the reader returned for that seek and the next frame | 98,304 | 128 KiB |

The 98,304 bytes are three fills of the 32 KiB buffer (the seek window, then the frame header, then the frame). They are not a scan from the start of the 1.5 MiB file. The same bound is asserted on a 1000-frame file in `seek_lands_on_the_frame_that_contains_the_sample`.

`/usr/bin/time -l` on that release test: maximum resident set size 5,963,776 bytes, peak memory footprint 4,915,584. That includes the test harness and two copies of the in-memory file. A per-allocation count was not available; a counting allocator would need `unsafe`, which this crate forbids. The demuxer retains a 32 KiB buffer, the current frame, and at most a 100-byte Xing TOC or 4096 VBRI entries. It does not allocate a table proportional to the file.

## Fuzz

In-process campaign, not libFuzzer. Target: `probe`, `Mp3Demux::open`, `next_packet`, `seek`, and `Mp3Mux` with at most six packets. Corpus: random buffers up to 8 KiB, a truncated valid file, a bit flip in a valid file, and an ID3v2 header whose declared size is over the limit. Seed step `0x9E3779B97F4A7C15` from `0x00C0FFEE`.

A 3 second smoke exercised 513,823 seeds with no panic.

The recorded campaign ran 1800 seconds on the same machine and exercised 306,072,002 seeds. Exit status 0. No panic, no finding, and no minimized regression.

```text
MP3_FUZZ_SECS=1800 cargo test --release -p cutlass-mp3-container --offline --test external -- --ignored --nocapture
```

## Builds

Commands from this checkout, exit status 0:

```text
cargo fmt --all -- --check
cargo lint
cargo test --workspace --offline
cargo deny check
```

`cargo deny check` reported unmatched license allowances already listed in `deny.toml` (`Apache-2.0 WITH LLVM-exception`, `BSD-3-Clause`, `ISC`, `Zlib`) and then `advisories ok, bans ok, licenses ok, sources ok`. No new crates.io dependency. Production code is safe Rust (`unsafe_code` is forbidden). The test harness allows `expect` and `panic` because `clippy.toml`'s test exception does not apply to an integration-test crate.

Release tests:

```text
cargo test --release -p cutlass-mp3-container --offline --test external -- indexed_seek
```

Exit status 0.

## Conclusion

Demux and mux of an unqualified Layer III file are complete for the checks in this file. Layer I, Layer II, ID3 metadata, and sample coding stay unsupported. Engine and export integration was not run because those crates are empty. GStreamer was not an oracle because its plugin scanner does not finish on this machine.
