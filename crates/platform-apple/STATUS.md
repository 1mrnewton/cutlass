# Apple platform codecs

Checkout: branch `platform-apple`. VideoToolbox, AudioToolbox, and ImageIO behind `Decoder` and `Encoder`. The `unsafe` is confined to `src/apple/imp` and only calls those frameworks plus CoreFoundation, CoreMedia, CoreVideo, CoreGraphics, and CoreAudio. There is no bindgen crate and no bundled codec library.

Linux and other non-Apple builds compile the same crate and every opener returns `Unsupported`, so a factory walk continues to a software codec. This record was written on macOS (arm64). An iOS device was not used.

## What this crate does

The desktop registers these openers first, then PNG and AAC. `Unsupported` continues that walk. Any other error stops it. Oversize pictures decline with `Unsupported` rather than `Limit`, so a later codec with a larger cap can still run. An empty picture or an empty packet is `InvalidApi`.

Stills go through ImageIO: JPEG, PNG, WebP, GIF, and the file helpers for HEIC (`public.heic`) and AVIF (`public.avif`). ImageIO expands to 8-bit straight RGBA. Registering it for PNG ahead of `cutlass-png` hides that crate's 16-bit and indexed path on Apple. 16-bit and indexed `push_picture` return `Unsupported`, which is too late for the factory to fall through. GIF buffers frames until `drain` emits one file.

Audio goes through AudioConverter. PCM is little-endian interleaved and round-trips exactly. AAC attaches the magic cookie and any priming info on the first packet. Vorbis has no format id and declines. A bitrate the converter will not set is ignored and the converter's default stands. ALAC, FLAC, MP3, Opus, and AC-3 open only when `AudioConverterNew` accepts them.

Video goes through VideoToolbox. A decoded frame is a retained BGRA pixel buffer named by a platform token. `to_cpu` copies that buffer to tightly packed RGBA, and `release_buffer` drops the retain. The session is asked for BGRA when the codec is H.264 or HEVC; ProRes and the other sized formats use the decoder's own pixel format and are kept only when that format is BGRA. An encoder accepts a token this crate still holds and passes that buffer to VideoToolbox. A token this crate did not publish is `Unsupported`. H.264 and HEVC need parameter sets, either in `CodecParams.extra` or in the first packet (`avcC`, the private `HVCC` envelope, or in-band Annex B). Other video codecs get a format description from the codec id and the frame size. VP8 declines immediately. The software renderer cannot sample the token, and the Metal renderer is not wired to it yet. `with_pixel_buffer` is the borrow a later renderer can use.

Edges must be even and at most 8192, and a frame must be at most 16,777,216 pixels. A packet must be non-empty and at most 16 MiB. Audio is at most 8 channels and 192 kHz. When `bitrate` is 0, video uses `max(width * height * 8, 64_000)` bits per second. H.264 asks for Baseline and HEVC for Main, with reordering off and a keyframe interval of 30. A property the session does not support is ignored.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| PNG, opaque 8-bit RGBA | ImageIO | ImageIO | `a_png_round_trip_keeps_opaque_pixels`, `a_png_file_round_trips_through_the_file_helpers` | Verified on this machine. Not the `cutlass-png` coded-domain path |
| JPEG | ImageIO | ImageIO | `jpeg_keeps_the_frame_size_and_the_red_channel` | Verified. Lossy |
| GIF, several frames | ImageIO | One file on `drain` | `gif_frames_leave_on_drain` | Verified |
| WebP | ImageIO when the encode succeeds | Same | `webp_keeps_the_frame_size_and_the_red_channel_when_imageio_can` | Encode declined on this machine |
| HEIC file helper | ImageIO | ImageIO | `heic_round_trips_when_imageio_can_write_it` | Verified on this machine for a 2×2 buffer |
| AVIF file helper | Not separately tested | Not separately tested | — | Not run |
| 16-bit still | — | Declined | `a_sixteen_bit_still_is_declined` | Unsupported |
| Local PNG or JPEG | ImageIO | — | `a_local_still_decodes_when_the_folder_is_present` | Verified here. The test returns without asserting when `../../local-files` is missing |
| PCM f32 | Exact | Exact | `pcm_round_trips_the_samples` | Verified |
| AAC-LC | AudioConverter | AudioConverter | `aac_round_trips_a_tone` | Verified. SNR above 20 dB after a lag search, not an ffmpeg oracle |
| Empty, non-finite, out-of-range, cancelled, wrong rate | Reject, buffer unchanged | Same | audio tests of those names | Verified |
| Vorbis, VP8 | Declined | Declined | `vorbis_is_declined`, `vp8_is_declined` | Unsupported |
| H.264, even size, Baseline | VideoToolbox, retained buffer | VideoToolbox | `h264_round_trips_a_solid_color` | Verified. A broken unit then a real packet still decodes. RGBA comes from `to_cpu` |
| HEVC | VideoToolbox | VideoToolbox | `hevc_round_trips_a_solid_color_when_the_hardware_can` | Verified on this machine |
| ProRes 422, MJPEG | VideoToolbox | VideoToolbox | the matching round-trip tests | Verified on this machine |
| AV1, VP9, MPEG-4, MPEG-2 | Open or decline | Same | `later_video_codecs_decline_or_open` | Capability query. A wrong picture fails the test. A decline is `Unsupported` |
| Odd size, oversize | Declined | Declined | `an_odd_h264_size_is_declined` | Unsupported, so a later opener can run |
| Platform frame token | Retained BGRA buffer | Encoded when this crate published it | `h264_round_trips_a_solid_color`, `a_retained_frame_encodes_without_a_cpu_copy` | Verified. A token this crate did not publish stays `Unsupported`. The renderer does not sample the buffer |
| ffmpeg 8.1.1 oracle | Not run | Not run | — | Not run |
| Short random H.264 packets | Fail closed | — | `random_packets_fail_closed` | Verified. 32 packets, not a 30-minute run |
| 30-minute fuzz | Not run | Not run | — | Not run |
| Release speed and memory | Not measured | Not measured | — | Not run |
| iOS device | Not run | Not run | Host was macOS | Not run |
| Engine preview and export | The engine crate is empty | Same | Desktop `register_codecs` only builds the factories | Registered, not played |

## Still required before calling this complete

- An ffmpeg decode of the AAC, H.264, and HEVC this crate emits, and a decode here of an ffmpeg elementary stream.
- A finished fuzz run of at least 30 minutes, with the seed and the findings written down.
- Release timings for one decode and one encode on this machine.
- A pass on an iOS device. The frameworks are the same entry points; the device was not here.
- AVIF covered the same way HEIC is.
- The Metal renderer sampling a retained token through `with_pixel_buffer`. The token and the encode path exist. Playback still copies if a caller asks for RGBA, and nothing in the renderer reads the buffer yet.
