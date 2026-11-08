# Motion JPEG support record

Checkout: branch `mjpeg`, worktree `cutlass-mjpeg`, started from `development` at `e025f6ce`. Native safe Rust in `cutlass-mjpeg`, on the JPEG core in `cutlass-jpeg`. No `unsafe`, no libjpeg, no FFmpeg, no VideoToolbox, and no subprocess in either library. FFmpeg, libjpeg-turbo, and GStreamer are test oracles only.

This file is the scope record. A row stays planned until a test or a recorded run named here has passed. Numbers are from runs on this machine, not estimates.

## Specifications and sources read

Normative:

- ITU-T T.81 (1992) | ISO/IEC 10918-1. Markers, interchange and abbreviated formats (B.4, B.5), Huffman coding, sequential and progressive scans, restart intervals, the DCT.
- ITU-T T.83 | ISO/IEC 10918-2. Decoder compliance: each reconstructed sample within 1 of the reference output. That is the tolerance used where the reference decoder uses a different IDCT.
- ITU-T T.871. JFIF: full-range BT.601 YCbCr, chroma sited between luma samples.
- Apple QuickTime File Format, Motion JPEG format A and B, and the `fiel` atom.

Read before the Cutlass code, for the algorithm, the conventions, and the edge cases. Nothing was copied, linked, or vendored.

- FFmpeg 8.1.1 `libavcodec/mjpegdec.c` (default Huffman tables, interlace detection from the first SOF height, the `AVID` comment, AVI1 and Apple `mjpg` APP1 headers, EOI field handling), `mjpegbdec.c`, `mjpegenc.c`, `jrevdct.c`, `jfdctint_template.c`, `libavformat/mov.c` (`fiel`).
- GStreamer 1.28.3 `gst-plugins-good/ext/jpeg/gstjpegdec.c` (standard Huffman tables when all four are absent, the field-pair height rule, field decode into alternate rows).
- libjpeg-turbo 3.2.0 behaviour through `djpeg` and `cjpeg`, used as an oracle. GStreamer's `jpegdec` calls libjpeg and outputs only I420, so it is not a separate oracle.

## Conventions this crate follows

These are Motion JPEG rules that are not in T.81. Each one is taken from the projects above, not guessed.

| Convention | Rule | Source |
| --- | --- | --- |
| Missing DHT | A table slot never defined by DHT uses the T.81 Annex K.3 table of that class: id 0 luminance, id 1 chrominance. A DHT replaces it | FFmpeg always preloads K.3; GStreamer inserts them when all four are absent; the AVI MJPEG note drops DHT from every frame |
| Abbreviated frames | A packet may omit DQT. Tables persist from earlier packets and from codec extradata (T.81 B.5). Tables a packet defines replace earlier ones only after that packet decodes. A scan whose table was never defined is `Malformed` | T.81 B.5 |
| Two fields in one packet | Two complete JPEG images in one packet, same width, where the first image's height `h` and the container height `H` satisfy `2h - 16 < H <= 2h` and `h > 8`, are two fields woven into one frame of height `H` | GStreamer `gstjpegdec.c`; FFmpeg uses the first SOF height against the container height |
| Field order | First stored field is the top field, unless the `fiel` atom says bottom stored first (detail 6 or 14) | QuickTime `fiel`; GStreamer top-first default. FFmpeg maps detail 14 to "bottom coded first, top displayed first" and only honours detail 6; this crate follows Apple's definition |
| Format A header | An APP1 `mjpg` header in each field is accepted and not needed; the second field is found by its SOI | QuickTime format A; FFmpeg reads and discards it |
| Color | Three components are full-range BT.601 YCbCr unless Adobe transform 0 or `R`,`G`,`B` ids say RGB. A COM segment starting `AVID` marks studio range | T.871; FFmpeg `AVID` handling |
| EXIF | Ignored for video frames. Orientation is not applied to Motion JPEG | Video players do not rotate per frame |

## Capability matrix

Decoder acceptance and encoder output are separate columns.

| Capability | Decode | Encode | Status |
| --- | --- | --- | --- |
| Baseline SOF0, 8-bit Huffman | Yes | Yes, every frame | Planned |
| Extended sequential SOF1, 8-bit Huffman | Yes | No | Planned |
| Progressive SOF2 | Yes, through the shared core | No | Planned |
| Grey (1 component) | Yes | Yes | Planned |
| YCbCr 4:4:4, 4:2:2, 4:2:0 | Yes | Yes, chosen by the caller; default 4:2:0 | Planned |
| YCbCr 4:4:0 and other legal sampling factors 1..=4 | Yes; 4:4:0 stays 4:4:0, the rest become 4:4:4 on the planar path | No | Planned |
| RGB (Adobe transform 0 or `RGB` ids) | Yes, RGBA output; planar path refuses | No | Planned |
| Quantization tables 8-bit and 16-bit, Huffman tables, custom tables | Yes | Annex K tables scaled by quality; standard Huffman tables | Planned |
| Restart intervals and RSTn | Yes, strict sequence | Optional interval in MCUs | Planned |
| Byte stuffing and fill bytes | Yes | Yes | Planned |
| Odd sizes, MCU padding, cropping | Cropped to T.81 A.1.1 component sizes | Edge replication into the padded MCU | Planned |
| Size, sampling, or table change between frames | Each packet stands alone apart from carried tables | Each frame stands alone | Planned |
| Missing DHT, abbreviated DQT | As in the conventions above | Always writes DQT and DHT | Planned |
| Two fields per packet (interlaced) | Woven into one frame; field order exposed on the crate API | Not emitted | Planned |
| Motion JPEG format A (`mjpa`) | Yes | No | Planned |
| Motion JPEG format B (`mjpb`) | Refused, `Unsupported` (`motion jpeg format b`): no independent encoder exists to validate it | No | Refused |
| Arithmetic, lossless, hierarchical, JPEG-LS, 12-bit, CMYK | Refused by name, from the JPEG core | No | Refused |
| Quality control | n/a | `extra` one byte 1..=100, default 75 | Planned |
| Bitrate control | n/a | Per-frame quality search to the frame's bit budget | Planned |

## Integration

| Path | Change | Status |
| --- | --- | --- |
| Engine playback | `register_decoders`: `Decoder` producing `Rgba8` with the JFIF conversion and libjpeg-style triangle chroma upsampling | Planned |
| Planar consumers | `register_video_decoders`: `VideoDecoder` producing `Gray8`, `Yuv420p`, `Yuv422p`, `Yuv440p`, or `Yuv444p`, full range, BT.601 | Planned |
| Export | `register_encoders`: `Encoder` accepting `Rgba8` or `Yuv420p` CPU frames; `register_video_encoders` | Planned |
| ISOBMFF demux | `jpeg`, `mjpa`, `mjpb` sample entries pass their type and any `fiel` payload as extradata | Planned |
| ISOBMFF mux | The export muxer accepts Motion JPEG and writes a `jpeg` sample entry | Planned |
| Desktop, iOS, Android | Registered after the platform decoders in the desktop, and in `ffi` | Planned |
| Desktop export menu | Motion JPEG offered for `.mov` only | Planned |

Field order has no slot in `PictureInfo`, `VideoFrame`, or the engine's frame. Fields are woven before those contracts, and the order is available from the crate API. Nothing downstream of the decoder deinterlaces.

## Resource limits

| Limit | Value |
| --- | --- |
| Packet | 64 MiB |
| Edge | 16 384 pixels |
| Pixels per frame | 16 777 216 |
| Images per packet | 2 |
| Markers per image | 4 096 |
| Pending input | None: a packet is decoded in `push` |
| Output queue | At most one frame per packet |
| Encoder delay | None |

## Gates

| Gate | Budget |
| --- | --- |
| Planar decode against FFmpeg (`-idct int`, raw `yuvj*` planes) | Every sample within 1 (T.83) on grey, 4:4:4, 4:2:2, 4:2:0, 4:4:0 |
| RGBA decode against `djpeg -dct int` | Bit-exact on grey, 4:4:4, 4:2:2, 4:2:0, 4:4:0 |
| Encoder output | `ffmpeg -xerror` and `djpeg` decode every frame with no warning |
| Encoder quality | Same quality and sampling as `cjpeg -dct int`: size within 2 % and PSNR within 0.2 dB |
| Rate control | Mean bitrate within 10 % of the target over a clip, when the target lies inside the quality range |
| Interlaced | Woven output equals the two independently decoded fields interleaved |
| Real media | Fixtures made from `../local-files` footage by FFmpeg and cjpeg, decode, seek, export, and reopen with ffprobe |
| Malformed input | Truncation at every byte of a frame, bit flips, and hostile headers return errors; the next good packet decodes |
| Fuzz | At least 30 minutes on the decoder entry, recorded |
| Decode speed | 1920×1080 4:2:0 to planar and to RGBA, one thread, release: under 16 ms per frame |
| Encode speed | The same frame at a fixed quality: under 33 ms |
| Memory | No growth across 1 000 frames; peak recorded |
| Checks | `cargo fmt`, `cargo clippy -D warnings`, and `cargo test` on the changed crates and their direct consumers. The user asked for no full-workspace run |
