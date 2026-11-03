# Windows platform codecs

Checkout: branch `platform-windows`. Media Foundation behind `Decoder` and `Encoder`, beside the WASAPI preview queue. The `unsafe` is confined to `src/mf/imp` and `src/wasapi.rs` and only calls Media Foundation and COM. There is no bindings crate: interfaces are called through vtable slots whose order was checked against the Windows SDK 10.0.26100 headers.

Linux and macOS builds compile the same crate and every opener returns `Unsupported`, so a factory walk continues to a software codec. This record was written on Windows 11 Pro (x86_64).

## What this crate does

The desktop registers these openers first on Windows, then the Rust codecs. `Unsupported` continues that walk. The Settings hardware switches leave the Media Foundation picture codecs out, the same as on macOS.

Video is H.264 only so far, through the in-box software transforms `CMSH264DecoderMFT` and `CMSH264EncoderMFT`, in synchronous mode and without a Direct3D device. Pictures cross as NV12 in system memory. A decoded picture is CPU RGBA, or CPU planar 4:2:0 when `CodecParams.pixel` is `Yuv420p`. NV12 to RGBA uses BT.709 limited range, the same fixed point `cutlass-encode` uses the other way. The encoder takes CPU RGBA or planar 4:2:0. A platform frame token is `Unsupported`.

The decoder takes length-prefixed units with an `avcC` (in `extra` or on the packet as `SideData::Config`), or Annex B with in-band parameter sets. The `avcC` sets go ahead of the first unit after open, a new config, or a reset. The transform reorders internally, so pictures leave in display order. A frame the transform pads to 16 is cropped to its display aperture.

The encoder asks for Baseline, so there are no B-frames. It emits 4-byte length-prefixed units without the in-band SPS and PPS, and the first keyframe carries the `avcC` as `SideData::Config`, like the VideoToolbox encoder. The transform is rebuilt at the frame rate the first picture's duration names. `reset` builds a new transform, so the next unit is a keyframe with the config again. When `bitrate` is 0 it uses `max(width * height * 8, 64_000)`.

Edges must be even and at most 8192, a frame at most 16,777,216 pixels, a packet non-empty and at most 16 MiB. `nal.rs` and `limits.rs` are copies of the Apple crate's; they could move to a shared crate.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| H.264, even size, Baseline | MF software transform, CPU RGBA | MF software transform | `h264_round_trips_a_solid_color` | Verified on this machine. Six units, timestamps kept, avcC on the first keyframe |
| H.264 to planar 4:2:0 | CPU Yuv420p | — | `h264_decodes_to_cpu_yuv_when_asked` | Verified |
| Annex B with in-band sets | Decoded without an avcC | — | `annex_b_units_decode_without_an_avcc` | Verified |
| Size not a multiple of 16 | Cropped to the aperture | Encoded | `a_frame_size_that_is_not_a_multiple_of_sixteen_is_cropped` | Verified at 200×120 |
| Reset | Flush | New transform, keyframe and config again | `reset_starts_again_on_a_keyframe_with_the_config` | Verified |
| Factory order and hardware switch | First opener | First opener | `the_factory_uses_media_foundation_first` | Verified |
| Odd size, other codecs | Declined | Declined | `an_odd_size_or_another_codec_is_declined` | Unsupported, so a later opener can run |
| Platform frame token | — | Declined | `a_platform_frame_is_declined` | Unsupported |
| Empty packet | InvalidApi | — | `an_empty_packet_is_an_invalid_call` | Verified |
| Short random packets | Error or no picture | — | `random_packets_fail_closed` | Verified. 32 packets, not a fuzz run |
| HEVC, AV1, VP9, MPEG-2 | Not wired | Not wired | — | Not done |
| AAC, MP3, other audio | Not wired | Not wired | — | Not done |
| JPEG, PNG stills (WIC) | Not wired | Not wired | — | Not done |
| Hardware transforms (async, D3D11) | Not wired | Not wired | — | Not done. The software transforms are what runs |
| ffmpeg oracle | Not run | Not run | — | Not run |
| Release speed and memory | Not measured | Not measured | — | Not run |
| Engine preview and export | Registered in the desktop factory | Same | Desktop `register_codecs_with` | Registered. Export not run end to end on Windows |

## Still required before calling this complete

- HEVC through `CMSH265DecoderMFT` and the HEVC encoder, using the HEVC records already in `nal.rs`.
- AAC decode and encode through `CMSAACDecMFT` and `AACMFTEncoder`, with the audio caps already in `limits.rs`.
- Hardware transforms from `MFTEnumEx` with a D3D11 device manager, and the asynchronous event model they need.
- WIC for the still formats, as ImageIO serves them on Apple.
- An ffmpeg decode of the H.264 this crate emits, a fuzz run, and release timings.
- An export from the desktop on Windows that is opened in another player.
