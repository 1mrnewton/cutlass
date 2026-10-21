# Codecs

One engine on desktop and mobile. FFmpeg stays out of the shipping binary.
Each codec below decodes in Rust when the platform has no decoder. Hardware
is used in front of that when the device has it.

Export stays narrow: H.264 and AAC everywhere, HEVC where a hardware encoder
exists, ProRes encode on Apple silicon only. A machine with no HEVC encoder
exports H.264.

Android and Linux hardware vary by device. Query the codec at startup and
skip it when the device does not list it.

## Video

| Codec | macOS / iOS | Android | Windows | Linux hardware | Software fallback |
|---|---|---|---|---|---|
| H.264 | Decode and encode | Decode and encode | Decode and encode | Decode and encode, when the GPU has it | Decode and encode |
| HEVC | Decode and encode | Decode; encode on many phones | Decode and encode, when the GPU has it | Decode and encode, when the GPU has it | Decode. Export uses hardware, or H.264 when it does not |
| VP9 | Decode on Apple silicon | Decode | Decode on Windows 10 and later | Decode on many GPUs | Decode |
| VP8 | No | Decode | No | Rare | Decode |
| AV1 | Decode on M3 and later | Decode on newer phones | Decode on newer GPUs | Decode on newer GPUs | Decode |
| MPEG-4 Part 2 | Decode | Usually decode | Unreliable | Rare | Decode |
| MPEG-2 | Decode | Some devices | Needs an extra extension | Some GPUs | Decode |
| MJPEG | Decode | Usually decode | Usually decode | Rare | Decode |
| ProRes | Decode and encode | No | No | No | Decode, so the file still opens. Encode stays on Apple silicon |

## Audio

Linux has no system audio codecs, so every format here is the Rust decoder.

| Codec | macOS / iOS | Android | Windows | Linux hardware | Software fallback |
|---|---|---|---|---|---|
| AAC | Decode and encode | Decode and encode | Decode and encode | None | Decode and encode |
| MP3 | Decode | Decode | Decode | None | Decode |
| PCM, WAV, AIFF | Decode and encode | Decode and encode | Decode and encode | None | Decode and encode |
| FLAC | Decode | Decode | Decode | None | Decode |
| Opus | Partial on recent OS versions | Decode | Decode on Windows 10 and later | None | Decode |
| Vorbis | No | Decode | No | None | Decode |
| ALAC | Decode and encode | No | No | None | Decode |
| AC-3 | Decode | Rare | Rare | None | Decode |

## Linux

VA-API for Intel and AMD. NVDEC and NVENC for NVIDIA. The Rust decoder or
encoder takes over when the GPU cannot do that codec, including a machine
with no GPU encoder. There is no ProRes hardware. Export is H.264 and AAC.

## Containers

MP4, MOV, MKV, WebM, AVI, and the plain audio files (WAV, AIFF, MP3, FLAC,
M4A, Ogg).

## Out of scope

WMA, AMR, Theora, VC-1, DV, DNx, camera RAW, and E-AC-3.
