# Windows platform codecs

Checkout: branch `platform-windows`. Media Foundation and the Windows Imaging Component behind `Decoder` and `Encoder`, beside the WASAPI preview queue. The `unsafe` is confined to `src/mf/imp` and `src/wasapi.rs` and only calls Media Foundation, WIC, COM, and the Direct3D 11 device a decoder is handed. There is no bindings crate: interfaces are called through vtable slots whose order, and every GUID, was checked against the Windows SDK 10.0.26100 headers.

Linux and macOS builds compile the same crate and every opener returns `Unsupported`, so a factory walk continues to a software codec. This record was written on Windows 11 Pro (x86_64) with an Intel UHD Graphics 630 and the HEVC Video Extensions package installed.

## What this crate does

The desktop registers these openers first on Windows, then the Rust codecs. `Unsupported` continues that walk. The Settings hardware switches leave the H.264 and HEVC openers out, the same as on macOS. AAC, JPEG, and PNG are registered either way.

Video is H.264 and HEVC. Hardware goes first. A decoder is the codec's decoder transform handed a Direct3D 11 device, so DXVA decodes on the GPU and the picture is copied back once; it is asked for low latency, so a picture leaves as soon as display order allows rather than once sixteen are held, which with a device would run it out of surfaces. An encoder is a hardware encoder transform found by `MFTEnumEx`, such as Intel Quick Sync, NVENC, or AMF, driven through its asynchronous event queue. When the hardware path will not open, or refuses its first unit, the software transform runs instead: `CMSH264DecoderMFT` and `CMSH264EncoderMFT` for H.264, and whatever HEVC transform is registered for HEVC.

A decoded picture is CPU RGBA, or CPU planar 4:2:0 when `CodecParams.pixel` is `Yuv420p`; NV12 to RGBA is BT.709 limited range. The encoder takes CPU RGBA or planar 4:2:0, and a platform frame token is `Unsupported`. The decoder takes length-prefixed units with an `avcC` or `hvcC` (in `extra` or on the packet as `SideData::Config`), or Annex B with in-band parameter sets. A frame the transform pads to 16 is cropped to its display aperture.

The encoder asks for H.264 Baseline or HEVC Main, no B pictures, and a keyframe every 30 pictures. Units leave as 4-byte length-prefixed NAL units without the in-band parameter sets, and the first keyframe carries the configuration as `SideData::Config`: an `avcC`, or for HEVC a real `hvcC` with the profile, tier, and level copied from the SPS, which is what the MP4 muxer writes verbatim. The VideoToolbox encoder writes a private `HVCC` envelope there instead. The transform is rebuilt at the frame rate the first picture's duration names, and `reset` builds a new one, so the next unit is a keyframe with the configuration again. When `bitrate` is 0 it uses `max(width * height * 8, 64_000)`.

Audio is AAC through `CMSAACDecMFT` and `AACMFTEncoder`. Packets are raw access units and the configuration is the AudioSpecificConfig, which is what the Rust AAC codec writes; an `esds` is accepted too. The decoder gives `f32` or `i16` and, like the Rust decoder, leaves trimming to the edit list. The encoder takes 44.1 or 48 kHz in mono, stereo, or 5.1, at the nearest of 96, 128, 160, or 192 kbit/s; any other rate or layout declines to the Rust encoder. Measured against the Rust decoder, the Microsoft encoder adds no priming, so its packets carry no `SideData::Gapless`.

Stills are JPEG and PNG through WIC, one file a packet. Samples are straight RGBA, upright: a JPEG's EXIF orientation is applied. A 16-bit PNG stays 16 bits, so registering WIC ahead of `cutlass-png` does not lose that depth; an indexed PNG is expanded to RGBA, and an indexed still to encode is `Unsupported`.

Edges must be even and at most 8192, a frame at most 16,777,216 pixels, a packet non-empty and at most 16 MiB, audio at most 8 channels and 192 kHz. `nal.rs` and `limits.rs` began as copies of the Apple crate's; they could move to a shared crate.

## The desktop export

The desktop's Export dialog now offers an MP4 beside the PNG frame. `Shell::export` writes the whole cut as H.264 or HEVC with 48 kHz stereo AAC through the desktop's own encoder factory, so the platform encoders go first on Windows and macOS. The bitrate follows the short side, as on the phone. The file is written beside its destination and renamed only when finished. `cutlass-isobmff`'s `Mp4Mux` now takes HEVC as well as H.264; its box writer already wrote `hvc1` and `hvcC`.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| H.264, every engine | DXVA and software | Quick Sync and software | `h264_round_trips_on_every_engine` | Verified at 64×48 and 1920×1080, all four pairings |
| HEVC | DXVA and software | Quick Sync | `hevc_round_trips_when_windows_can`, `an_hevc_config_is_a_real_hvcc` | Verified at 64×48 and 1920×1080. No software HEVC encoder here |
| H.264 behaviour | Planar 4:2:0, Annex B, crop, reset | Timestamps, avcC on the first keyframe | the H.264 tests | Verified |
| B pictures in display order | DXVA, low latency | — | `a_named_movie_decodes_in_display_order_on_the_platform` on a local B-picture movie | Verified on two local movies: 300 pictures at 1080×1080 and 240 at 3840×2160 in order. The Quick Sync encoder makes no B pictures through MF, so `b_pictures_come_back_in_display_order` skips here |
| AAC | Exact layout, `f32` or `i16` | 44.1/48 kHz, 1/2/6 channels | `aac_round_trips_a_tone_through_windows`, `rust_aac_decodes_on_windows`, `the_encoder_adds_no_priming` | Verified. 29.8 dB tone SNR; lag 0 under the Rust decoder |
| AAC limits | — | Other rates and layouts declined | `a_rate_or_layout_the_encoder_lacks_is_declined`, `a_bad_sample_adds_nothing` | Verified |
| PNG 8-bit and 16-bit | WIC | WIC | `a_png_round_trips_exactly_with_alpha`, `a_sixteen_bit_png_keeps_sixteen_bits`, `a_rust_png_decodes_on_windows` | Verified exact, both ways with `cutlass-png` |
| JPEG | WIC, EXIF orientation | WIC | `jpeg_keeps_the_size_and_the_red`, `exif_orientation_turns_the_picture_upright`, `the_pattern_fixture_matches_the_rust_jpeg_decoder` | Verified. The engine fixture is within 2 levels of `cutlass-jpeg` |
| Hostile input | Error or no picture | — | `random_packets_fail_closed`, `hostile_and_truncated_files_fail_closed` | Verified. Short runs, not a fuzz run |
| Desktop MP4 export | Opens again through the desktop readers | H.264 and HEVC with AAC | `export_writes_an_h264_movie_that_opens_again`, `export_writes_an_hevc_movie_where_the_platform_can`, `a_named_movie_exports_and_opens_again` | Verified on a colour project, and on a local 31-second 1080×1080 movie exported at 720p in a release build: 928 pictures back in order, 48 kHz AAC, 75 s to export |
| AV1, VP9, MPEG-2, MP3 | Not wired | Not wired | — | Not done. The extensions for VP9 and AV1 are installed here |
| WebP, GIF, HEIC stills | Not wired | Not wired | — | Not done |
| Platform frame token into the renderer | — | — | — | Not done. Pictures cross as CPU bytes |
| ffmpeg oracle | Not run | Not run | — | ffmpeg is not installed here |
| Release speed and memory | Not measured | Not measured | — | Not run |

## Still required before calling this complete

- A machine with NVIDIA or AMD graphics, and one without a GPU encoder, through the same tests.
- An ffmpeg decode of the H.264, HEVC, and AAC this crate emits.
- A fuzz run, and release timings for decode and encode.
- Handing the decoder's Direct3D surface to the renderer instead of copying it back, as `with_pixel_buffer` allows on Apple.
- An export progress bar and a way to stop. The desktop export runs on the interface thread, as the PNG export always has, so a movie export holds the window until it finishes; the 31-second movie took 75 s, slower than real time, and where that time goes has not been measured.
