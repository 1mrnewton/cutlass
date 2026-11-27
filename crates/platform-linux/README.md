# cutlass-platform-linux

Linux's hardware video codecs behind the workspace `Decoder` and `Encoder` traits: NVDEC and NVENC on an NVIDIA GPU, and VA-API on Intel, AMD and any other GPU with a VA-API driver. The crate also holds the ALSA output queue that preview playback writes to.

The crate is not `#![forbid(unsafe_code)]`. Its lints deny `unsafe` everywhere except `src/alsa.rs`, `src/dl.rs` and the module `src/video/imp`, and every `unsafe` block carries a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks` is denied). On a target that is not Linux those are left out, every opener declines with `CodecError::Unsupported` so the factory moves on to a Rust codec, and `OutputQueue::open` reports `QueueError::NoDevice`.

The desktop app registers these codecs on Linux, ahead of the Rust codecs, and `cutlass-playback` opens the queue.

## What it supports

### System libraries

There is no bindings crate and nothing is linked at build time. Each library is opened with `dlopen` the first time it is needed, so the build needs no headers and a machine without one of them runs the Rust codecs instead.

| Library | Comes with | Used for |
| --- | --- | --- |
| `libcuda.so.1` | The NVIDIA driver | The CUDA context NVDEC and NVENC run on, and copying pictures back |
| `libnvcuvid.so.1` | The NVIDIA driver | NVDEC: NVIDIA's bitstream parser and decoder |
| `libnvidia-encode.so.1` | The NVIDIA driver | NVENC |
| `libva.so.2`, `libva-drm.so.2` | libva (`libva2`, `libva-drm2`) and a driver such as `intel-media-va-driver` or Mesa's `va-driver` | VA-API on a DRM render node |
| `libasound.so.2` | alsa-lib | The output queue |

The C structs are kept as zeroed bytes with their fields written at fixed offsets. Those offsets were printed once by the C programs in `abi/` (`nvenc_abi.c` against FFmpeg's nv-codec-headers 12.0, `va_abi.c` against libva 2.22) into `src/video/imp/nvenc_abi.rs` and `src/video/imp/va/abi.rs`. NVENC API 12.0 needs NVIDIA driver 520 or newer.

### What it registers

| Backend | Decode | Encode |
| --- | --- | --- |
| NVIDIA | H.264, HEVC, AV1 | H.264, HEVC |
| VA-API | H.264, HEVC | H.264, HEVC |

`register_decoders_with(factory, video)` and `register_encoders_with(factory, video)` register NVIDIA first, then VA-API, and nothing when `video` is false. A factory asks openers in the order they were registered and takes the first that accepts, so the desktop's Rust codecs, registered after, are asked only when both backends decline. An opener declines (`CodecError::Unsupported`) when its library, a device, the profile, the bit depth or the size is missing; that is the only error an encoder opener returns, because `EncodeFactory` stops at any other.

`CUTLASS_VIDEO_BACKEND` (`BACKEND_VAR`) limits which backends may open: `nvidia`, `vaapi`, both comma-separated, or `none`. Unset, both may. `CUTLASS_VAAPI_DEVICE` names one render node, such as `/dev/dri/renderD129`; unset, every node from `renderD128` to `renderD191` is probed once and the first that offers what an opener needs is used. The first time a backend opens for a codec, or declines for a reason, it says so once in the log.

### Video decode

- **NVDEC.** NVIDIA's own parser reads the stream, keeps the references and calls back in display order. 8-bit streams come out as NV12 and deeper ones as P016. Before opening, the stream is checked against `cuvidGetDecoderCaps`.
- **VA-API.** VA-API takes parsed headers, so the stateless front ends in `cutlass_hevc::stateless` and `cutlass_h264::stateless` do the parsing, reference marking, reference lists and output order the Rust decoders do, and this crate fills VA's picture, scaling-matrix and slice buffers from them the way FFmpeg's `vaapi_hevc.c` and `vaapi_h264.c` do. HEVC Main and Main 10, and H.264 Constrained Baseline, Main and High (8-bit 4:2:0, progressive) are taken. A profile the GPU lacks is decoded on the nearest superset it offers. Pictures are read back with `vaGetImage`, or `vaDeriveImage` where the driver refuses that.
- **Fallback.** A hardware decoder can open and then refuse the stream once it reads the parameter sets: an interlaced H.264 stream, a 4:4:4 profile, a size over the GPU's limit. Until the first picture comes out, every packet pushed since open or reset is kept; if the hardware fails before then, the Rust decoder for the codec opens and the packets are pushed into it, so the caller sees pictures, not an error.
- A decoded picture is CPU planar 8-bit 4:2:0 (`PixelFormat::Yuv420p`), cropped to the display area. 10-bit pictures are shifted down to 8 bits, as the Rust HEVC decoder does. Colour comes from the stream's VUI, and BT.709 when it names none.
- The decoders take length-prefixed units with an `avcC` or `hvcC` (in `CodecParams.extra` or as `SideData::Config` on a packet), or Annex B with in-band parameter sets. AV1 takes the `av1C` configuration OBUs.

### Video encode

- Both encoders take CPU RGBA or planar 4:2:0. A platform frame token or a picture of another size is `Unsupported`.
- The settings match Windows and Apple: H.264 High and HEVC Main, no B pictures, an IDR every 30 pictures, and a bitrate of `max(width * height * 8, 64_000)` when none is given. With `constant_rate` the rate is constant; otherwise it is variable with twice the target as its peak.
- **NVENC.** Preset P4 with high-quality tuning, one host NV12 buffer that NVENC allocates, encoded synchronously. The session starts at the first frame, so the frame rate comes from that frame's duration.
- **VA-API.** H.264 High, else Main, else Constrained Baseline, and HEVC Main, on the full entrypoint or the low-power one when that is all there is. The sequence, picture and slice headers are written by this crate (`src/video/imp/va/header.rs`) and handed to the driver as packed headers, with the parameter buffers FFmpeg's `vaapi_encode_h264.c` and `vaapi_encode_h265.c` fill. Rate control is CBR or VBR as asked, each falling back to the other and then to constant QP 26. For HEVC the coding tree and transform sizes and the tools come from what the driver reports (`VAConfigAttribEncHEVCBlockSizes` and `EncHEVCFeatures`, which low-power Intel needs), and P pictures go out as B slices with both lists the same where the driver asks for that.
- Encoded units leave as 4-byte length-prefixed NAL units without in-band parameter sets. The first keyframe after open or reset carries the configuration as `SideData::Config`: an `avcC`, or an `hvcC` with the VPS, SPS and PPS.

### Sound

`OutputQueue::open(rate, channels)` plays 48 kHz stereo `f32` on ALSA's default device, which on a desktop is usually PipeWire or PulseAudio behind ALSA's plugin. One thread owns the PCM and writes periods from the stored slice, and silence past its end. `set_sample` replaces the slice.

## Using it

```rust
use cutlass_decode::DecodeFactory;
use cutlass_encode::EncodeFactory;

fn platform_codecs(hardware_video: bool) -> (DecodeFactory, EncodeFactory) {
    let mut decoders = DecodeFactory::new();
    let mut encoders = EncodeFactory::new();
    cutlass_platform_linux::register_decoders_with(&mut decoders, hardware_video);
    cutlass_platform_linux::register_encoders_with(&mut encoders, hardware_video);
    // Register the Rust codecs after these.
    (decoders, encoders)
}
```

- `register_decoders`, `register_decoders_with`, `register_encoders`, `register_encoders_with` and `probe_video` exist on every target.
- `probe_video()` returns lines describing the NVIDIA GPU and what NVDEC and NVENC offer there, and each VA-API render node with its driver, profiles and entrypoints. The `video_probe` example prints them:

```text
cargo run -p cutlass-platform-linux --example video_probe
```

## How it is tested

```text
cargo test -p cutlass-platform-linux
```

Tests that need no GPU run on any Linux machine:

- The struct layouts against the generated offsets, the bit writer (exp-Golomb codes, alignment, emulation prevention), levels from size, rate and bitrate, rate-control choice, and reading pictures out of padded surfaces.
- **VA-API decode buffers.** A recording accelerator checks every picture's VA buffers against the references the stateless front end holds, on x265 and x264 streams (several profiles, B pyramids, weighted prediction, several slices, scaling lists) and on the Rust H.264 encoder's stream. The x264 and x265 streams need an `ffmpeg` with libx264 and libx265 and are skipped without one.
- **VA-API encode headers.** The headers the encoders write are read back by the stateless front ends, for each H.264 profile and for HEVC with the default sizes and with low-power Intel's 64×64 tree blocks, every tool on and GPB slices, and compared with the parameter buffers sent with the same picture: the reference named, frame numbers and order counts, slice types, and where the slice data starts.
- Every opener declines cleanly with no GPU, and the factories then reach the Rust codecs.

Tests that need a GPU print `skipped:` and pass without one, unless `CUTLASS_REQUIRE_NVIDIA` or `CUTLASS_REQUIRE_VAAPI` is set:

- Encoding on NVENC and on VA-API and decoding with the Rust decoder at 320×240, 1920×1080 and (VA-API) 642×362, at 35 dB PSNR or better, with an IDR at pictures 0 and 30 and a configuration that parses as `avcC` or `hvcC`.
- Reset starts again on a keyframe carrying the configuration; RGBA input with an NTSC rate and a constant rate; refusal of AV1 encode, odd and oversized pictures, 10-bit input, platform frames and a wrong size.
- NVDEC and VA-API decode compared picture by picture with the Rust decoder, VA-API decoding what VA-API encoded, NVDEC taking Annex B without a configuration, decoding again after drain and reset, and a stream the hardware refuses (an interlaced H.264 stream on VA-API) falling back to Rust.

```text
CUTLASS_REQUIRE_VAAPI=1 CUTLASS_REQUIRE_NVIDIA=1 cargo test -p cutlass-platform-linux
```

What has not been done: the GPU tests were not run for this README (they were written on a machine without a GPU), so nothing here has yet met a real driver. The test that opens a real ALSA device is `#[ignore]`d.

## Limits and gaps

| Limit | Value |
| --- | --- |
| Picture edge | 8192, and even for video |
| Pixels in one picture | 16,777,216 |
| One packet | 16 MiB |

- A decoded picture is always copied back to CPU memory. The GPU surface is not handed to the renderer, and a platform frame token given to an encoder is `Unsupported`.
- VA-API does not decode AV1, VP9, interlaced H.264 or 4:2:2 and 4:4:4 profiles here; NVDEC does not decode VP9. Those go to the Rust decoders, or for VP9 nowhere.
- Encoding is 8-bit 4:2:0 only, one slice a picture, with one reference and no B pictures.
- The output queue is fixed at 48 kHz stereo and only replaces its slice.
