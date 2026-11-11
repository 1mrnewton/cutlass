# cutlass-platform-apple

Apple's system codecs behind the workspace `Decoder` and `Encoder` traits: VideoToolbox for video, AudioToolbox for sound, ImageIO for stills. It decodes and encodes. The crate also holds the Core Audio output queue that preview playback writes to, the import of a decoded picture into Metal without a copy, and three macOS helpers for the menu bar, Finder documents and thread priority.

The crate is not `#![forbid(unsafe_code)]`. Its lints deny `unsafe` everywhere except the module `src/apple/imp`, which is the only place that calls the frameworks. On a target that is not Apple the crate still builds, `src/apple/imp` is left out, and every opener declines with `CodecError::Unsupported` so the factory moves on to a Rust codec.

The desktop app registers these codecs on macOS and uses the Metal import and the helpers. `cutlass-playback` opens the queue.

## What it supports

### System frameworks

`src/apple/imp/sys.rs` declares the C functions by hand; there is no generated binding and no bundled codec library.

| Framework | Used for |
| --- | --- |
| VideoToolbox | Video decode and encode sessions |
| CoreMedia, CoreVideo | Format descriptions, sample buffers, pixel buffers, the Metal texture cache |
| AudioToolbox | `AudioConverter` for sound codecs, `AudioQueue` for the output queue |
| ImageIO, CoreGraphics | Reading and writing stills |
| CoreFoundation | The retained values the others take and return |
| Metal, through `objc2-metal` and `wgpu` | Wrapping a decoded picture's planes as textures |
| AppKit and Foundation classes, through `objc2` | The menu and open-document helpers (macOS only) |

### What it registers

| Kind | Codec ids | Notes |
| --- | --- | --- |
| Video, decode and encode | H.264, HEVC, AV1, VP8, VP9, MPEG-4, MPEG-2, Motion JPEG, ProRes | Left out when the caller passes `video: false`. VP8 always declines. The others open only when VideoToolbox opens a session on this machine |
| Sound, decode and encode | AAC, Opus, MP3, FLAC, PCM, Vorbis, ALAC, AC-3, E-AC-3 | Vorbis always declines. The others open only when `AudioConverterNew` accepts the format |
| Stills, decode | JPEG, PNG, WebP | GIF is left to `cutlass-gif` |
| Stills, encode | JPEG, PNG, WebP, GIF | A GIF collects its frames and leaves as one file on `drain` |

A factory asks openers in the order they were registered and takes the first that accepts. On macOS the desktop app registers the Rust FLAC, ALAC, MP3, AC-3, E-AC-3, Opus, Vorbis, WebP and Motion JPEG codecs first, then this crate, then the remaining Rust codecs. So this crate is asked first for H.264, HEVC, AV1, VP9, MPEG-4, MPEG-2 and ProRes when the hardware switches in Settings are on, and always for AAC, PCM, JPEG, PNG and GIF encode. For the other ids it is a fallback.

### Video

- Decode is asynchronous. A push submits one unit and returns the pictures that have finished. Pictures are held for the stream's reorder depth and leave in display order.
- A caller that asks for `PixelFormat::Yuv420p` gets 4:2:0. A decoder from `register_surface_decoders_with` keeps that picture in its IOSurface as a `cutlass_decode::Surface` when it can, and copies the planes out only when asked. Otherwise the planes are copied to CPU memory.
- A caller that asks for anything else gets a BGRA pixel buffer named by a platform token. `to_cpu` copies it to RGBA and `release_buffer` drops it.
- H.264 and HEVC need parameter sets: an `avcC` or `hvcC` in `CodecParams.extra` or on the packet, or in-band Annex B. Units may be length-prefixed or Annex B.
- A session the system invalidates is opened again on the next push or reset.
- The encoder takes RGBA CPU frames or a platform token this crate published. It asks for H.264 Baseline or HEVC Main, no frame reordering, a keyframe every 30 frames, and a bitrate of `max(width * height * 8, 64_000)` when none is given. ProRes takes its profile (`apco`, `apcs`, `apcn`, `apch`) from `CodecParams.extra`.

### Sound

- PCM is little-endian and interleaved. Compressed packets are one access unit each.
- The decoder takes the codec's magic cookie from `CodecParams.extra` or from `SideData::Config` on a packet. A bare AAC AudioSpecificConfig is wrapped as an ES descriptor first. HE-AAC is opened at its full output rate.
- The encoder puts its magic cookie on the first packet as `SideData::Config`.

### Stills

- ImageIO output is 8-bit straight RGBA, turned upright by the file's orientation.
- The encoder accepts 8-bit input only. A 16-bit still is declined with `Unsupported`.
- `decode_file` and `encode_file` reach the other ImageIO types by name, including HEIC (`public.heic`) and AVIF (`public.avif`).

## Using it

```rust
use cutlass_decode::DecodeFactory;
use cutlass_encode::EncodeFactory;

fn platform_codecs(hardware_video: bool) -> (DecodeFactory, EncodeFactory) {
    let mut decoders = DecodeFactory::new();
    let mut encoders = EncodeFactory::new();
    cutlass_platform_apple::register_surface_decoders_with(&mut decoders, hardware_video);
    cutlass_platform_apple::register_encoders_with(&mut encoders, hardware_video);
    (decoders, encoders)
}
```

Available on every target:

- `register_decoders`, `register_decoders_with`, `register_surface_decoders_with`, `register_encoders`, `register_encoders_with`.

Apple targets only:

- `import_surface(device, surface)` returns the luma and chroma planes of a decoded picture as two `wgpu` textures, or `None` when the surface is not from this crate or the device is not Metal.
- `OutputQueue::open(rate, channels)` opens the default output for interleaved `f32`. `set_sample` replaces what is held, `append` adds to it, `played_frames` is the clock a player follows, and `sink` gives a `StreamSink` another thread can append through.
- `to_cpu`, `with_pixel_buffer`, `release_buffer` work on platform tokens.
- `decode_file`, `encode_file` for stills by ImageIO type name.
- `run_at_interface_priority` moves the calling thread to the user-interactive class.

macOS only:

- `listen_for_opened_files`, `wake_on_opened_files`, `take_opened_files` receive the documents Finder asks the app to open.
- `menu_keys_shown_only` makes named menus show their key equivalents without taking the key presses.

## How it is tested

```text
cargo test -p cutlass-platform-apple
```

All tests are unit tests inside `src/`. `nal.rs`, `reorder.rs`, `status.rs` and `limits.rs` run on any host. The tests under `src/apple/imp` run only on an Apple host and call the real frameworks:

- Video: H.264 round trip of a solid color, decode to 4:2:0, display order after drain, reset, a session reopened after it is lost, a surface picture compared with the planar copy, a surface imported as two Metal planes, a retained frame encoded without a CPU copy, and a Rust MPEG-4 export decoded by VideoToolbox. HEVC, ProRes and Motion JPEG round trips pass when the machine can open them. AV1, VP9, MPEG-4 and MPEG-2 are only required to open or decline cleanly.
- Sound: PCM and AAC round trips, HE-AAC configurations, E-AC-3 frame counts, and refusal of empty, non-finite and out-of-range input.
- Stills: PNG, JPEG and GIF round trips; WebP and HEIC when ImageIO can write them; a 16-bit still declined; random bytes refused.
- The output queue's copy and append logic. The test that opens a real output device is `#[ignore]`d.

What has not been done: no output is compared with FFmpeg or any other decoder, there is no fuzz campaign (one test pushes 32 random H.264 packets), AVIF has no test, and nothing has been run on an iOS device. Results depend on the hardware of the machine that runs the tests.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Picture edge | 8192, and even for video |
| Pixels in one picture | 16,777,216 |
| One packet | 16 MiB |
| Sound channels | 8 |
| Sample rate | 192 kHz |
| Reorder depth | 16 pictures |

A picture over the size limits declines with `Unsupported`, not `Limit`, so a later opener with a larger cap can still run.

- PNG and JPEG are decoded by ImageIO ahead of `cutlass-png` and `cutlass-jpeg` on macOS, so a 16-bit PNG arrives as 8 bits.
- A platform token stays alive until `release_buffer` is called for it. Nothing outside this crate calls it today; the engine asks for 4:2:0 and does not receive tokens.
- The HEVC encoder hands out its parameter sets in a private `HVCC` envelope, not a standard `hvcC` record.
- The encoder sets an expected frame rate of 30 and real-time mode whatever the caller's frame rate is, and does not take 4:2:0 input.
- A bitrate the audio converter refuses is ignored and the converter's default stands.
- Only the 4:2:0 path keeps a picture on the GPU. A decoder for a codec other than H.264 or HEVC, asked for a non-4:2:0 picture, returns `Unsupported` unless VideoToolbox happens to produce BGRA.
- `nal.rs` and `limits.rs` have near copies in `cutlass-platform-windows`; nothing is shared between the two crates.
