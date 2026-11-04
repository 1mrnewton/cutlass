# H.264 scope

Checkout: `8fce4a643865` on `h264-sw-enc-dec`. Crate `cutlass-h264`. Direction: decode and encode. Decode follows the recommendation's software process for the profiles and tools below. Encode emits progressive High, 8-bit, 4:2:0, CABAC. Container demux and mux stay in the container crates. The codec boundary is one access unit: Annex B or length-prefixed NAL units.

`docs/codecs/h264/decode.md` is the decode grind list. This file is the contract. A row there that is still open stays open here.

## Normative text

- Recommendation ITU-T H.264 (V16) (06/2026) is in force. Reconstruction follows clauses 7, 8, and 9 as read from V14 (08/2021), plus the annexes named below. V16 adds SEI. Picture timing and film grain change decoder output. The other V16 messages are consumed by size.
- Annex A bounds the coded picture. Annex B is the byte stream. Annex D is the SEI that changes output time or the displayed picture. Annex E is sample aspect, timing, and colour tags. Annexes G, H, and I are SVC, MVC, and 3D. Those three are partial.
- ISO/IEC 14496-15 (`avcC`) was not obtained. Length prefixes of 1, 2, and 4 bytes follow the record this crate already parses, checked against FFmpeg `libavformat/avc.c`.

Official ITU conformance bitstreams were not obtained. That gap stays open. A self-encoded file does not close it.

## Sources read before the Cutlass code

- The V14 text for NAL units, parameter sets, slice headers, macroblock and residual syntax, intra and inter prediction, weighted prediction, the inverse transform, deblock, POC, reference lists, DPB marking, CAVLC, and CABAC, then the annexes as each tool was added. Where a project and the recommendation disagree, the recommendation wins.
- FFmpeg 8.1.1: `libavcodec/h264dec.c` and the `h264_*.c` files next to it, plus `libavformat/avc.c`. Hardware decoders (`vaapi_h264.c`, `vdpau_h264.c`, `nvdec_h264.c`, `dxva2_h264.c`, `d3d12va_h264.c`, `vulkan_h264.c`) are ignored. FFmpeg has no software H.264 encoder. Encode checks use libx264. FFmpeg 8.1.1 refuses data partitions, FMO, separate colour planes, bit depths 11 and 13, and NAL types 20 and 21. Those paths use the recommendation as the oracle.
- GStreamer: `gsth264parse` for framing, and `gstx264enc` / `gstopenh264dec` as wrappers. GStreamer has no software reconstruction of its own.
- OpenH264 and x264 were read for the first progressive 8-bit encoder and decoder. The JM reference software was not retrieved.

## Decode

A normal sequence parameter set is accepted for profiles 66, 77, 88, 100, 110, 122, and 244 (Baseline, Main, Extended, High, High 10, High 4:2:2, High 4:4:4 Predictive). The constraint byte is read and ignored, so a constrained Baseline or constrained High stream is decoded as that `profile_idc`. A subset sequence parameter set is accepted for profiles 83, 86, 118, 128, 135, 138, and 139. Any other `profile_idc`, including 44 and 134, is refused.

| Capability | Decode | Gate |
| --- | --- | --- |
| Progressive and interlaced 4:2:0, 4:2:2, and 4:4:4, 8 through 14 bits | Accept. Luma and chroma depths match. | Independent pictures match inside the cropped rectangle. 11-bit and 13-bit follow the recommendation, because FFmpeg refuses them. |
| CAVLC and CABAC | Accept | Same |
| I, P, B, SP, and SI. Skip, direct, partitions through 4×4 | Accept. SP and SI are 8-bit 4:2:0. | Same. SP and SI outside that depth and chroma are `unsupported`. |
| Weighted prediction, 4×4 and 8×8 transforms, custom scaling lists, deblock, frame crop | Accept | Bit-exact against FFmpeg where FFmpeg decodes the tool |
| POC types 0, 1, and 2. DPB, MMCO, reference-list modification, `frame_num` gaps | Accept | Display order matches |
| Slice groups, map types 0 through 6 | Accept | The recommendation. FFmpeg's build leaves FMO out. |
| Data partitions, NAL types 2, 3, and 4 | Accept | The recommendation. FFmpeg 8.1.1 does not decode them. |
| Separate colour planes. QP′Y-zero transform bypass | Accept | Planes follow the recommendation. Bypass matches FFmpeg. |
| Annex B and length prefixes of 1, 2, and 4 bytes. `avcC` | Accept | Both framings |
| Picture timing and film grain | Picture timing can change the output time. Film grain changes the displayed picture and stays off the reference. | Timing matches the recommendation. Grain follows H.274, which need not match FFmpeg's synthesis. |
| VUI sample aspect, timing, colour, transfer, matrix | Tags on the picture. Reserved ids stay unspecified. | A defined idc is the tag. A transfer this workspace does not implement stays an unspecified curve. |
| Annex A picture size | A picture that fits any defined level is decoded, including one larger than the `level_idc` stored in the SPS. A picture outside every level is refused. | Flat pictures at the recorded sizes match FFmpeg |
| SVC, MVC, and 3D | Partial. The tools with tests in `annex_g.rs` reconstruct. The open rows below are refused. | The recommendation, except where an ordinary P residual is also checked against FFmpeg |

Reconstruction is exact inside the cropped rectangle. Lossy comparison is only for the encoder.

## Encode

The encoder emits one High profile (`profile_idc` 100), level 3.1, 8-bit 4:2:0, CABAC stream. POC type is 0. The reference window slides. Each picture is one slice. The first picture is intra 4×4 DC. A later picture is P, or one held picture becomes a B slice at 16×16. P motion is a radius-1 diamond at integer, half-pel, and quarter-pel, and a macroblock may be 16×16, 16×8, 8×16, or four 8×8 blocks. Inter luma uses an 8×8 transform when that residual is closer than sixteen 4×4 transforms. `weighted_pred_flag` and `weighted_bipred_idc` stay 0.

| Capability | Encode | Gate |
| --- | --- | --- |
| High, progressive, 8-bit, 4:2:0, CABAC | Emit | This decoder and FFmpeg reconstruct the same samples |
| I, P, and B | Emit. B stays 16×16 and does not emit skip or direct. | Motion on a moving clip is not all intra |
| Rate | Frame QP tracks `bitrate` | 0.75×–1.25× of `bitrate * duration` over the named clip |
| Quality | Useful High-profile coding | Luma PSNR at least 30 dB on the named synthetic clip, at least 28 dB on one natural clip |
| Framing | `avcC` plus 4-byte lengths, or Annex B when `nal_length_size` is 0 | Both framings. Any other length size is refused |

## Limits

Decoder caps are code limits on top of Annex A. A picture past every defined level is refused before a frame is allocated.

| Cap | Decoder | Encoder |
| --- | --- | --- |
| Coded picture | 139,264 macroblocks, edge 16,880 samples. Levels 6, 6.1, and 6.2 share that `MaxFS`. | Each edge at most 4,096, and at most 8,388,608 pixels. Both edges even. |
| Reference frames | 16 | The frames the High encoder keeps |
| One NAL | 8 MiB | The slice it just wrote |
| SPS ids | 32 | One |
| PPS ids | 256 | One |
| CPB count in the HRD | 32 | The encoder writes no HRD |
| Threads | One. This crate does not spawn. | One |

## Still refused

- `chroma_format_idc` 0.
- A bit depth outside 8 through 14, or a luma depth that differs from chroma.
- SP and SI outside 8-bit 4:2:0. An SP slice that selects the 8×8 transform is malformed.
- Inter-layer prediction with `quality_id` other than 0, `ref_layer_dq_id` other than 0, an inter-layer deblock idc other than 1, constrained intra resampling, or slice skip. A scan index other than 0 through 15. Resampling of a field, an MBAFF picture, separate colour planes, or chroma other than 4:2:0.
- An `Intra_16×16` coefficient stays in the sample residual. The per-slice adaptive transform-coefficient flag is refused. Header-prediction sources 1 and 3, and a texture inter slice, are refused. `TextureFirstFlag` 0 is refused for view synthesis. `dmvp_flag` 0 is refused when view synthesis, adaptive luminance compensation, and run-length skip are off. A view-synthesis partition other than `P_L0_16×16`, `P_L0_L0_16×8`, `P_L0_L0_8×16`, `P_8×8`, or `B_L0_16×16` is refused. A predicted texture slice is refused.
- `svc_vui_parameters_present_flag`, `mvc_vui_parameters_present_flag`, and `mvcd_vui_parameters_present_flag`.
- Error concealment. An incomplete picture is an error. The next good IDR still decodes.
- A thread pool. That waits on a shared Cutlass pool and does not live in this crate.
- Encoder tools this crate only decodes: weighted prediction, FMO, more than one slice, interlaced pictures, bit depths other than 8, chroma other than 4:2:0, B partitions below 16×16, B skip, B direct, and 4×4 sub-macroblocks.

## Numeric gates

Machine: Apple M5 Pro, one thread, release. Corpus: ffmpeg `testsrc2` 320×240 at 30 fps for 2 seconds at 500 kbps, one natural clip from `../local-files` scaled to 640×360 for 30 frames at 800 kbps, a libx264 1280×720 High stream, and `12471039-uhd_3840_2160_24fps.mp4` from `../local-files`. Those files are not copied into the repo.

- Decode: samples match an independent decoder on the supported set. Tolerance is zero inside the cropped rectangle. Tools FFmpeg refuses match the recommendation.
- Encode rate: total payload bits within 0.75×–1.25× of `bitrate * duration`.
- Encode quality: luma PSNR at least 30 dB on the synthetic clip and 28 dB on the natural clip. Chroma is reported.
- Speed: 1280×720 High decode at 30 fps, one thread, on this machine. 3840×2160 was measured and has no realtime gate.
- Memory: one 1280×720 decode stays under 64 MiB resident above the process baseline.
- Fuzz: at least 30 minutes on the decoder entry and a bounded encoder campaign. Random bytes return an error and leave the decoder reusable.

## Milestones

1. Parameter sets, Annex B, length prefixes, and `avcC`. Done.
2. Progressive 8-bit 4:2:0 reconstruction through High, including CABAC, B slices, weights, and deblock. Done.
3. High 10, the other depths through 14, 4:2:2, and 4:4:4. Done.
4. Separate colour planes, lossless bypass, interlaced pictures, slice groups, data partitions, SP, and SI. Done.
5. Picture timing, film grain, Annex E tags, and Annex A sizes. Done.
6. Annexes G, H, and I. Partial. The open rows are listed above.
7. High CABAC encoder with inter prediction, one B frame, and rate control. Done for the tools the encoder emits.
8. ITU conformance bitstreams. Not available.
