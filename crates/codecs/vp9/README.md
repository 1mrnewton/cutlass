# cutlass-vp9

## What this crate is

`cutlass-vp9` is a native safe Rust VP9 decoder and encoder. The crate forbids `unsafe`. FFmpeg is not linked. Callers pass access units and pictures through `VideoDecoder` and `VideoEncoder`.

## Specification

The bitstream and the reconstruction follow the VP9 Bitstream & Decoding Process Specification, version 0.7, 22 February 2017. The bool decoder follows that document's one-bit process.

## API

The crate exports `Vp9Decoder`, `Vp9Encoder`, `Decoder`, `register_decoders`, and `register_encoders`.

`Vp9Decoder::open` takes a `CodecParams` whose codec is `CodecId::Vp9` and whose pixel format is `PixelFormat::Yuv420p`. The decoder implements `VideoDecoder`. `push` consumes one `Packet` and returns `VideoOutput` values. Each output carries a `VideoFrame` plus `pts`, `duration`, and `timescale`. `drain` is empty. `reset` drops reference pictures. `cancel_flag` is the flag `push` checks between frames.

`Vp9Encoder::open` takes the same kind of `CodecParams`. A `bitrate` of 0 keeps quantizer 32. A positive bitrate moves the quantizer. `Vp9Encoder::with_quantizer` fixes the quantizer, and 0 is lossless. The encoder implements `VideoEncoder`. `push` consumes one `VideoInput` (`VideoFrame`, `pts`, `duration`, `timescale`) and returns `Packet` values. `drain` is empty. `reset` drops the reference picture.

`Decoder::new` and `Decoder::decode` take the bytes of one frame. A shown frame returns the reconstructed width, height, and `y`, `u`, and `v` planes. A frame that is not shown returns no picture. `reset` drops references and probabilities.

`register_decoders` adds the profile-0 opener to a `VideoDecodeFactory`. `register_encoders` adds it to a `VideoEncodeFactory`.

## Support

Profile 0 only: 8-bit 4:2:0. Profiles 1, 2, and 3 are refused. An open call also refuses a pixel format other than `PixelFormat::Yuv420p`. A coded edge outside 1..=4096 is `CodecError::Limit`. Profile 0 with the RGB color space is malformed. A reference scale this decoder cannot use is `CodecError::Unsupported`.

| Frame | Decoder | Encoder |
| --- | --- | --- |
| Key frame | Accepted | Emitted when the picture index is a multiple of 30, including the first picture. |
| Inter frame | Accepted | Emitted between those key frames, using the previous picture as LAST. |
| Show-existing | Accepted. Displays a stored reference. | Not emitted. |
| Hidden frame | Accepted. The call succeeds and returns no picture. | Not emitted. |
| Superframe | Accepted by `Vp9Decoder::push` as one packet of several frames. | Not emitted. |

The encoder emits only when both edges are multiples of 8. Every emitted frame is shown and error resilient, and it is stored in every reference slot.

## Accuracy

One sample, compared before timing. The bytes are tightly packed 8-bit 4:2:0, `y` then `u` then `v`, against ffmpeg's `vp9` decoder. These lines are copied from the bench stdout below.

```text
pixels: 3456000 bytes, 0 mismatches against ffmpeg vp9
ffmpeg vp9 decoded 8 frames, 0 mismatches
```

The first line is 30 frames of the libvpx-vp9 stream. The second is 8 frames from `Vp9Encoder`, decoded by ffmpeg's `vp9` decoder and compared with `Vp9Decoder`.

The crate tests compare stored YUV the same way for key frames, error-resilient inter frames, inter frames that adapt probabilities, alt-ref, scaled references, segmentation, and a 52×52 loop-filtered pair.

## Performance

One sample: ffmpeg `testsrc2` at 320×240, profile 0, 8-bit 4:2:0. Decode is 30 frames (one key frame, then inter frames). Encode is 8 frames of the same pattern at quantizer 32. ffmpeg's wall time includes process startup. The cutlass times are the codec loop.

Date: 2026-10-29

Machine: `Apple M5 Pro` (`sysctl -n machdep.cpu.brand_string`)

FFmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`

Command: `cargo bench -p cutlass-vp9`

```text
320x240 profile 0 4:2:0, one thread
cutlass times the codec loop. ffmpeg times include process startup.
decode libvpx-vp9 ivf, 30 frames, median of 3 after 1 warmup
pixels: 3456000 bytes, 0 mismatches against ffmpeg vp9
cutlass     0.019 s     1580.6 fps
ffmpeg      0.024 s     1275.1 fps
encode testsrc2, 8 frames, quantizer 32, one run after 1 warmup
cutlass     0.100 s       79.8 fps
ffmpeg vp9 decoded 8 frames, 0 mismatches
```

## Example

`examples/roundtrip.rs` decodes `fixtures/vp9_red32_q_0.bin` with `Vp9Decoder` and prints the picture size and frame count.

```text
cargo run --release -p cutlass-vp9 --example roundtrip
```

```text
32x32 frames 1
```

## Gaps

- Profiles 1, 2, and 3 are refused. Those are 8-bit 4:4:4, and 10-bit and 12-bit 4:2:0 and 4:4:4.
- The encoder does not emit show-existing frames, hidden frames, more than one tile, a loop filter, segmentation, compound prediction, or sub-pixel motion. Intra blocks are DC. Inter motion is an integer LAST vector. Transforms are no larger than 8×8. Quantizer 0 uses a 4×4 Walsh-Hadamard transform.
- The encoder refuses a size that is not a multiple of 8. The decoder can reconstruct those sizes.
- An edge above 4096 is refused.
- Playback and a WebM container are outside this crate.
