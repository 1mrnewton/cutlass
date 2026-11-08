# cutlass-mjpeg

Native safe Rust Motion JPEG decoder and encoder in `cutlass-mjpeg`, on the JPEG core in `cutlass-jpeg`. No `unsafe`. No libjpeg, no FFmpeg, no VideoToolbox, no subprocess. FFmpeg 8.1.1 and libjpeg-turbo 3.2.0 `cjpeg` and `djpeg` are oracles in the tests and the bench only, started as separate processes.

`STATUS.md` is the scope record: the sources read, the Motion JPEG conventions with their origin, the capability matrix with the test behind each row, limits, gates, and the recorded results.

## What this crate is

One packet is one frame, and every frame is a key frame. A packet holds one JPEG image, or two field images of one interlaced frame.

- `MjpegDecoder` implements `VideoDecoder`. `decode(&[u8])` returns a `Picture`. `Picture::into_frame` gives planar samples at their coded sampling, and `Picture::to_rgba` gives straight RGBA. `last_field_order` reports the field order of the last two-field packet.
- `MjpegPictureDecoder` implements `Decoder` and hands the engine `Rgba8` pictures at the packet's time.
- `MjpegEncoder` implements `Encoder` (RGBA CPU frames, what export sends) and `VideoEncoder` (planar frames). `encode_rgba` and `encode_frame` are the same paths without the traits.

`register_decoders`, `register_video_decoders`, `register_encoders`, and `register_video_encoders` register `CodecId::Mjpeg`. The desktop app and `cutlass-ffi` register `register_decoders` and `register_encoders` ahead of the platform codecs. VideoToolbox opens a Motion JPEG stream and then fails a 4:2:2 frame, and a factory moves to the next codec only when one refuses at open, so the platform codec cannot go first.

## API

`open` takes a `CodecParams` whose `codec` is `CodecId::Mjpeg`. Any other id is `Unsupported`.

Decoder `extra`:

- Empty: a plain Motion JPEG stream (AVI, MP4 `mp4v` object type 0x6C, raw).
- A QuickTime sample-entry type, `jpeg` or `mjpa`, optionally followed by the two `fiel` bytes. `mjpb` is `Unsupported` (`motion jpeg format b`).
- A JPEG table stream (SOI, DQT/DHT, EOI) whose tables serve every packet.

Anything else is refused. Tables a packet defines carry to later packets that omit them (T.81 B.5). `reset` drops the carried tables. A failed packet leaves the decoder as it was.

Encoder `extra`: empty, or one byte `1..=100` for a fixed quality. Without a quality, a nonzero `bitrate` selects `RateMode::Bitrate`, which searches each frame's quality to its share of the bitrate and carries the difference to the next frame. Otherwise the quality is 75. `with_sampling` chooses 4:2:0 (default), 4:2:2, 4:4:4, or grey for RGBA input. `with_restart_interval` writes restart markers. Planar input keeps its own sampling. An encoder opened at 0×0 accepts any frame size, and one opened at a size refuses another. `extradata()` is `jpeg`, the sample-entry type a movie muxer writes.

## Support

Decoded:

- Baseline SOF0, extended sequential SOF1, and progressive SOF2 Huffman frames, precision 8. 8-bit and 16-bit quantization tables.
- Grey, YCbCr at any legal sampling (4:4:4, 4:2:2, 4:2:0, and 4:4:0 kept on the planar path, others upsampled to 4:4:4), and RGB (Adobe transform 0 or `R`,`G`,`B` ids, `Gbrp` on the planar path).
- Restart intervals, byte stuffing, fill bytes, odd sizes cropped from the padded MCUs.
- Frames without DHT (Annex K.3 tables), frames without DQT (carried tables), and tables in the extradata.
- Two field images in one packet, woven when the stream's first packet holds an image under three quarters of the container height followed by a second image, as FFmpeg decides; a progressive stream may later drop to half the container height. Field order from `fiel`, then from an `AVID` comment. Apple `mjpg` APP1 headers (format A).
- The `CS=ITU601` comment, read as a C string, marks studio range.
- A size or sampling change from one packet to the next.

Encoded: baseline SOF0 with JFIF APP0, Annex K quantizers scaled the libjpeg way, the standard Huffman tables, and optional restart markers, at a fixed quality or a bitrate.

Refused, by name: format B, arithmetic, lossless, differential, hierarchical, JPEG-LS, precision other than 8, CMYK, a packet over 64 MiB, a frame over 16 384 on an edge or 16 777 216 pixels, more than two images in a packet, field images that do not pair, and a lone field when `fiel` or an `AVI1` marker says the track is interlaced.

## Accuracy

With `MJPEG_REQUIRE_ORACLES=1` a missing oracle fails a test instead of skipping it.

- Decoded RGBA equals `djpeg -dct int` byte for byte on FFmpeg and cjpeg streams: 4:2:0, 4:2:2, 4:4:4, 4:4:0, grey, RGB, progressive, SOF1 with 16-bit tables, restart markers, DHT stripped, and field pairs.
- Planar samples are within 1 of FFmpeg's `-idct int` planes (T.83).
- The encoder's frames open in `ffmpeg -xerror` with no message. At quality 30, 75, and 95 and every sampling, their size is within 2 % of `cjpeg -dct int`, and `djpeg` decodes them to the same samples as cjpeg's.
- Frames from two clips in `../local-files`, coded by FFmpeg, decode equal to `djpeg`. The desktop plays, seeks, and exports a Motion JPEG movie made from `16929689_1920_1080_30fps.mp4`.

## Performance

`cargo bench -p cutlass-mjpeg --bench throughput`, Apple M5 Pro, one thread, 1920×1080 frames of `16929689_1920_1080_30fps.mp4`:

| Work | Time per frame |
| --- | --- |
| Encode RGBA 4:2:0, quality 75 (210 KB) | 14.59 ms |
| Encode RGBA 4:2:0 at 60 Mb/s (60.0 Mb/s delivered) | 28.23 ms |
| Decode 4:2:0 to planar | 3.33 ms |
| Decode 4:2:0 to RGBA | 6.76 ms |
| `ffmpeg -threads 1` decode of the same stream, process start included | 2.84 ms |

Resident memory grew 480 KiB between frame 100 and frame 1 000 for decode, and not at all for encode. `STATUS.md` has the full output.

## Gaps

- Format B is refused.
- The encoder writes whole progressive frames only; it does not write field pairs, and the movie muxer writes no `fiel`.
- Export codes 4:2:0 from RGBA. 4:2:2 and 4:4:4 encode is on the crate API only.
- With the Rust codec first, VideoToolbox's Motion JPEG decoder is not used.
- No coverage-guided fuzzing: `cargo-fuzz` is not installed. The seeded mutation campaign is recorded in `STATUS.md`.
- Linux, Windows, iOS, and Android devices were not part of these runs.
