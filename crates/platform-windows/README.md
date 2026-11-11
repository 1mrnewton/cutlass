# cutlass-platform-windows

Windows' system codecs behind the workspace `Decoder` and `Encoder` traits: Media Foundation for H.264, HEVC and AAC, and the Windows Imaging Component (WIC) for JPEG and PNG. It decodes and encodes. The crate also holds the WASAPI output queue that preview playback writes to.

The crate is not `#![forbid(unsafe_code)]`. Its lints deny `unsafe` everywhere except two places: the module `src/mf/imp` (the `unsafe` blocks are in `com.rs` and `image.rs`) and `src/wasapi.rs`. On a target that is not Windows those are left out, every opener declines with `CodecError::Unsupported` so the factory moves on to a Rust codec, and `OutputQueue::open` reports `QueueError::NoDevice`.

The desktop app registers these codecs on Windows, and `cutlass-playback` opens the queue.

## What it supports

### System libraries

There is no bindings crate and no bundled codec library. COM interfaces are called through hand-written vtable slots and GUIDs in `src/mf/imp/com.rs`.

| Library | Used for |
| --- | --- |
| Media Foundation (`mfplat`) | Finding and running the H.264, HEVC and AAC transforms |
| Direct3D 11 (`d3d11`) | The device a hardware video decoder is handed |
| WIC, through COM (`ole32`, `shlwapi`) | Reading and writing JPEG and PNG |
| WASAPI, through COM (`ole32`, `kernel32`) | The shared-mode output stream |

### What it registers

| Kind | Codec ids | Notes |
| --- | --- | --- |
| Video, decode and encode | H.264, HEVC | Left out when the caller passes `video: false` |
| Sound, decode and encode | AAC | Always registered |
| Stills, decode and encode | JPEG, PNG | Always registered |

A factory asks openers in the order they were registered and takes the first that accepts. On Windows the desktop app registers some Rust codecs first (FLAC, ALAC, MP3, AC-3, E-AC-3, Opus, Vorbis, WebP, Motion JPEG), then this crate, then the remaining Rust codecs. None of the earlier ones share an id with this crate, so it is asked first for H.264 and HEVC when the hardware switches in Settings are on, and always for AAC, JPEG and PNG.

### Video

- Hardware is tried first. A decoder is the codec's decoder transform handed a Direct3D 11 device and asked for low latency. An encoder is a hardware encoder transform found with `MFTEnumEx`.
- When the hardware path does not open, or refuses its first unit, the software transform for the same codec runs instead. For HEVC that is whatever HEVC transform Windows has registered.
- A decoded picture is CPU RGBA, or CPU planar 4:2:0 when the caller asks for `PixelFormat::Yuv420p`. NV12 to RGBA uses BT.709 limited range. A picture the transform pads is cropped to its real size.
- The decoder takes length-prefixed units with an `avcC` or `hvcC` (in `CodecParams.extra` or as `SideData::Config` on a packet), or Annex B with in-band parameter sets. Pictures leave in display order.
- The encoder takes CPU RGBA or planar 4:2:0. It asks for H.264 Baseline or HEVC Main, no B pictures, a keyframe every 30 pictures, and a bitrate of `max(width * height * 8, 64_000)` when none is given.
- Encoded units leave as 4-byte length-prefixed NAL units without in-band parameter sets. The first keyframe carries the configuration as `SideData::Config`: an `avcC`, or for HEVC a standard `hvcC`.

### Sound

- AAC packets are raw access units. The configuration is an AudioSpecificConfig; an `esds` is accepted too.
- The decoder gives interleaved `f32` or `i16`.
- The encoder takes 44.1 or 48 kHz in mono, stereo or 5.1, at the nearest of 96, 128, 160 or 192 kbit/s. Any other rate or layout declines, so the Rust AAC encoder takes it. Its packets carry no `SideData::Gapless`.

### Stills

- One packet is one file. Samples come back as straight RGBA, turned upright by a JPEG's EXIF orientation.
- A 16-bit PNG stays 16 bits. An indexed PNG is expanded to RGBA.
- The encoder takes 8-bit or 16-bit gray, gray with alpha, RGB or RGBA. An indexed still is `Unsupported`.

## Using it

```rust
use cutlass_decode::DecodeFactory;
use cutlass_encode::EncodeFactory;

fn platform_codecs(hardware_video: bool) -> (DecodeFactory, EncodeFactory) {
    let mut decoders = DecodeFactory::new();
    let mut encoders = EncodeFactory::new();
    cutlass_platform_windows::register_decoders_with(&mut decoders, hardware_video);
    cutlass_platform_windows::register_encoders_with(&mut encoders, hardware_video);
    (decoders, encoders)
}
```

- `register_decoders`, `register_decoders_with`, `register_encoders`, `register_encoders_with` exist on every target.
- `OutputQueue::open(rate, channels)` opens the default output device in shared mode. It accepts only 48 kHz stereo. `set_sample` replaces the slice of interleaved `f32` the device thread plays; past its end the queue plays silence.
- `QueueError` is the queue's error type.

## How it is tested

```text
cargo test -p cutlass-platform-windows
```

All tests are unit tests inside `src/`. `nal.rs`, `nv12.rs`, `slice.rs`, `limits.rs` and `format.rs` run on any host. The tests under `src/mf/imp` are compiled only on Windows and call the real system codecs:

- Video (`tests.rs`): H.264 round trips on each pairing of hardware and software decoder and encoder, decode to 4:2:0, Annex B without an `avcC`, cropping of a size that is not a multiple of 16, reset, HEVC round trip when Windows can, the HEVC configuration checked as a standard `hvcC`, B pictures in display order, and 32 short random packets that must not produce a picture.
- Sound (`audio_tests.rs`): an AAC tone round trip, the Rust AAC encoder's output decoded here, no priming from the encoder, and refusal of rates and layouts the encoder lacks.
- Stills (`image_tests.rs`): exact PNG round trips at 8 and 16 bits, a PNG written by `cutlass-png` decoded here, JPEG size and orientation, a fixture compared with `cutlass-jpeg`, and hostile and truncated files.
- The test that opens a real WASAPI device is `#[ignore]`d.

What has not been done: none of these tests were run for this README (they need a Windows host), no output is compared with FFmpeg, and there is no fuzz campaign. Results depend on the graphics hardware and the codec packages installed on the machine that runs them.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Picture edge | 8192, and even for video |
| Pixels in one picture | 16,777,216 |
| One packet | 16 MiB |
| Sound channels | 8 |
| Sample rate | 192 kHz |
| One slice given to the output queue | 64 KiB |

- Only H.264, HEVC, AAC, JPEG and PNG are wired. AV1, VP9, MPEG-2, MP3 and WebP, GIF or HEIC stills do not go through Windows.
- A decoded picture is always copied back to CPU memory. The Direct3D surface is not handed to the renderer, and a platform frame token given to the encoder is `Unsupported`.
- The output queue only replaces its slice. It has no way to append a stream or report how many frames have played, which the Apple queue has, and it is fixed at 48 kHz stereo.
- For H.264 the crate names Microsoft's software decoder and encoder as the fallback. For HEVC it names none: it uses a hardware transform or whatever HEVC transform Windows lists, and declines when there is neither.
- `nal.rs` and `limits.rs` have near copies in `cutlass-platform-apple`; nothing is shared between the two crates.
