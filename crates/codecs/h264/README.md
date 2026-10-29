# cutlass-h264

## What this crate is

Native safe Rust decoder and encoder for progressive 8-bit 4:2:0 H.264. The crate forbids `unsafe`. It does not link FFmpeg, a C library, or another H.264 implementation. FFmpeg is an external oracle for the tests, the example, and the bench. The example and the bench exit 0 after printing `skip: ffmpeg is not installed` when that program is absent.

The support record is [STATUS.md](STATUS.md). The scope is [SCOPE.md](SCOPE.md).

## Specification

Recommendation ITU-T H.264 (V16) (06/2026) | ISO/IEC 14496-10 is the in-force specification. The syntax, decoding process, and CABAC tables were read from Recommendation ITU-T H.264 (V14) (08/2021), clauses 6, 7, 8, and 9 and Annexes A, B, and C. Reconstruction follows clauses 7, 8, and 9 of that text. V16's new supplemental enhancement information messages are skipped.

ISO/IEC 14496-15, the AVC decoder configuration record, was not obtained. Annex B and length-prefixed samples are implemented as described in [SCOPE.md](SCOPE.md).

## API

`H264Decoder` and `H264Encoder` open from `CodecParams`. `nal_length_size` 0 is Annex B. `register_decoders` and `register_encoders` add those openers to a factory. One `push` is one access unit or one picture. `drain` flushes pictures still held for reorder.

```rust
use cutlass_decode::{CodecId, CodecParams, VideoDecoder, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_h264::{H264Decoder, H264Encoder};

fn encode_one() -> Result<(), Box<dyn std::error::Error>> {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    let mut encoder = H264Encoder::open(&params)?;
    let packets = encoder.push(&VideoInput {
        picture: Yuv420p::solid(16, 16, 40, 80, 160)?,
        pts: 0,
        duration: 1,
        timescale: 30,
        force_keyframe: true,
    })?;
    let mut decoder = H264Decoder::open(&params)?;
    for packet in &packets {
        decoder.push(packet)?;
    }
    decoder.drain()?;
    Ok(())
}
```

## Support

| Capability | Decode | Encode |
| --- | --- | --- |
| Progressive 8-bit 4:2:0 | Baseline, Main, and High | High CABAC |
| CAVLC and CABAC | Both | CABAC |
| I slices | Accepted | Intra 4×4 DC |
| P slices | Accepted, including skip and partitions through 4×4 | Skip when the picture matches the reconstruction, otherwise 16×16, 16×8, 8×16, or four 8×8 blocks. Motion is a diamond at integer, half-pel, and quarter-pel |
| B slices | Accepted, including skip and direct | One held picture, 16×16, list 0, list 1, or both |
| 4×4 and 8×8 transform, deblock, frame crop | Accepted | Inter luma uses an 8×8 transform when that residual is closer than sixteen 4×4 transforms |
| Weighted prediction | Explicit and implicit | Refused (`weighted_pred_flag` 0, `weighted_bipred_idc` 0) |
| Picture order count | Types 0, 1, and 2 | Type 0, sliding window |
| Framing | Annex B, length prefixes of 1, 2, and 4 bytes, and `avcC` | Annex B when `nal_length_size` is 0. Otherwise 4-byte lengths and `avcC`. Any other length size is refused |
| Rate |  | Frame QP when `bitrate` is set |
| Slices per picture | More than one, including the deblock across the boundary unless that slice turns it off | One |

Caps, from [SCOPE.md](SCOPE.md): 4096×4096, 8 388 608 pixels, 16 reference frames, 8 MiB for one NAL, 256 PPS ids, 32 SPS ids.

## Accuracy

Decode tests compare this decoder to FFmpeg's raw `yuv420p` and require every sample inside the cropped rectangle to match. The assert is zero difference. The tests are `decodes_one_independent_intra_picture`, `idr_returns_pictures_still_waiting`, `baseline_intra_matches_ffmpeg`, `baseline_ip_matches_ffmpeg`, `main_cabac_ipb_matches_ffmpeg`, `main_cabac_temporal_direct_matches_ffmpeg`, `two_slices_match_ffmpeg_across_the_boundary`, `cropped_picture_matches_ffmpeg`, `jvt_quant_matrices_match_ffmpeg`, `high_cabac_8x8_matches_ffmpeg`, `high_implicit_weights_match_ffmpeg`, and `main_explicit_weights_match_ffmpeg`.

Encoder tests that require this decoder and FFmpeg to reconstruct the same samples from this encoder's bitstream are `eight_by_eight_transform_matches_ffmpeg`, `ramp_ac_matches_ffmpeg`, `chroma_ramp_matches_ffmpeg`, `repeated_ramp_residual_matches_ffmpeg`, `shifted_texture_matches_ffmpeg`, `half_pixel_shift_matches_ffmpeg`, `split_partition_matches_ffmpeg`, `vertical_split_matches_ffmpeg`, `quadrant_blocks_match_ffmpeg`, and `b_frame_comes_out_in_display_order`.

`testsrc2_meets_the_luma_and_rate_gates` requires luma PSNR of at least 30 dB on the synthetic clip and a payload within 0.75×–1.25× of `bitrate * duration`. `natural_clip_meets_the_luma_gate` requires at least 28 dB on one natural clip and the same rate window. An earlier run of those gates is recorded in [STATUS.md](STATUS.md).

The bench below uses the same zero-sample rule on a 1280×720 High Annex B stream before it will print a speed. That comparison disagreed. See Performance.

## Performance

Command, from this worktree:

```text
cargo bench -p cutlass-h264
```

Date: Thu Oct 29 21:09:50 GMT 2026.

CPU: `Apple M5 Pro`.

FFmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`.

FFmpeg's timed commands include process startup. This was one sample. The bench prints a speed only after samples match. This run stopped first. No speed line was printed, and the encode timing did not run. The library harness in the same command reported `0 passed; 0 failed; 53 ignored`.

```text
     Running benches/against_ffmpeg.rs (target/release/deps/against_ffmpeg-4d995ec6a7ad3ead)
Error: "decode pixels disagree: 132262391 samples, first byte 240 (this decoder 127, ffmpeg 145), max delta 250"
error: bench failed, to rerun pass `-p cutlass-h264 --bench against_ffmpeg`

Caused by:
  process didn't exit successfully: `/Users/mrnewton/dev/cutlass/cutlass-h264-sw-enc-dec/target/release/deps/against_ffmpeg-4d995ec6a7ad3ead --bench` (exit status: 1)
```

The decode input is 120 frames of `testsrc2` at 1280×720, High profile, libx264 `veryfast`, CABAC, two B frames, Annex B. The packed buffer is Y, then U, then V. Byte 240 is inside the first picture's luma. Exit status 1.

## Example

`examples/roundtrip.rs` asks FFmpeg for a 64×64 High-profile Annex B file in a temp directory, decodes that file with `H264Decoder`, and prints the frame count and the picture size.

```text
cargo run -p cutlass-h264 --example roundtrip
```

When FFmpeg is absent the process prints `skip: ffmpeg is not installed` and exits 0.

## Gaps

The decoder refuses interlaced, field, and MBAFF pictures (`frame_mbs_only_flag` 0), FMO, data partitioning, SP/SI, scalable, multiview, and 3D NAL units, chroma other than 4:2:0, bit depth other than 8, separate colour planes, and QP-prime transform bypass. Supplemental enhancement information is skipped. A picture past the caps above is refused.

The encoder refuses weighted prediction, FMO, and more than one slice per picture. B slices stay 16×16: they do not search smaller partitions, and they do not emit skip or direct. Intra blocks stay DC 4×4. 4×4 sub-macroblocks are not emitted.

Official ITU conformance bitstreams were not obtained. The desktop shell does not register this crate. Linux and Windows were not run for this record.

Details and the earlier measurements are in [STATUS.md](STATUS.md) and [SCOPE.md](SCOPE.md).
