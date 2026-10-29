# H.264 scope

Checkout: branch `h264-sw-enc-dec`. Crate `cutlass-h264`. Direction: decode and encode. Domain: progressive 8-bit 4:2:0 video for import and export. Container demux and mux stay in the container crates. The codec boundary is one access unit: Annex B or length-prefixed NAL units.

## Normative text

- Recommendation ITU-T H.264 (V16) (06/2026) | ISO/IEC 14496-10, in force. The published summary of V16 adds supplemental enhancement information messages. It does not change the 8-bit 4:2:0 reconstruction process this crate implements.
- The syntax, decoding process, and CABAC tables were read from Recommendation ITU-T H.264 (V14) (08/2021), clauses 6, 7, 8, and 9 and Annexes A, B, and C. V15 (08/2024) removed a placeholder annex and renumbered later annexes. Clauses 7 through 9 are the reconstruction text used here. V16’s new SEI messages are skipped, not applied.
- ISO/IEC 14496-15 AVC decoder configuration record (`avcC`) and length-prefixed samples, as read from FFmpeg `libavformat/avc.c` and the ISOBMFF sample description this workspace already stores as opaque extradata. The ISO text itself was not obtained.

Official ITU conformance bitstreams were not obtained. That gap stays visible in `STATUS.md`.

## Sources read before the Cutlass code

- ITU-T H.264 (V14) text for NAL units, SPS, PPS, slice headers, macroblock and residual syntax, intra prediction, inter prediction, direct modes, weighted prediction, inverse transform, deblocking, POC, reference lists, DPB marking, CAVLC, and CABAC. Returned to those clauses while writing each stage.
- FFmpeg 8.1 (the tree current on 2026-10-28): `libavcodec/h264dec.c`, `h264_slice.c`, `h264_cabac.c`, `h264_cavlc.c`, `h264_mb.c`, `h264_parse.c`, `h264_ps.c`, `h264_refs.c`, `h264_loopfilter.c`, `h264pred.c`, `h264data.c`, `h264idct_template.c`, `h264qpel_template.c`, `h264chroma_template.c`, `cabac.c`, `h2645_parse.c`, and `libavformat/avc.c`. FFmpeg has no native software H.264 encoder. Its encoder path is libx264.
- GStreamer: `gsth264parse` (Annex B and AVCC framing only), `gstopenh264dec` / `gstopenh264enc` (OpenH264), and `gstx264enc` (x264). GStreamer does not contain its own reconstruction. OpenH264’s decoder gained Main-profile tools in later releases and is not treated as a High-profile oracle.
- OpenH264 decoder and encoder cores: `decode_slice.cpp`, `parse_mb_syn_cavlc.cpp`, `parse_mb_syn_cabac.cpp`, `rec_mb.cpp`, `manage_dec_ref.cpp`, `cabac_decoder.cpp`, `encoder.cpp`, `md.cpp`, `mv_pred.cpp`, `set_mb_syn_cavlc.cpp`, `set_mb_syn_cabac.cpp`, `deblocking.cpp`, `ratectl.cpp`.
- x264 encoder: `encoder/encoder.c`, `analyse.c`, `macroblock.c`, `me.c`, `cabac.c`, `cavlc.c`, `ratecontrol.c`.
- The JM reference software was not retrieved. The recommendation is the reference. Where a project and the recommendation disagree, the recommendation wins.

## What is required

| Capability | Decode | Encode | Gate |
| --- | --- | --- | --- |
| Baseline, Main, and High, progressive, 8-bit, 4:2:0 | Accept | Emit High | Independent pictures match the reconstruction |
| CAVLC and CABAC | Accept | Emit CABAC | Same |
| I, P, and B slices, skip, direct, partitions down to 4x4 | Accept | Emit I, P, and B, 16x16 through 8x8 | Motion on a moving clip is not all intra |
| SPS, PPS, DPB, POC types 0, 1, and 2, reorder | Accept | POC type 0, sliding window | Display order matches |
| Explicit and implicit weighted prediction | Accept | Not emitted (`weighted_pred_flag` 0, `weighted_bipred_idc` 0) | Streams that set the flags still decode |
| 4x4 and 8x8 transform, deblock, frame crop | Accept | Emit | Bit-exact against an independent decoder |
| Annex B and 1-, 2-, and 4-byte length prefixes, `avcC` | Accept | `avcC` plus 4-byte lengths, or Annex B when `nal_length_size` is 0 | Both framings |
| Rate | n/a | Frame QP tracks a bitrate | 0.75×–1.25× of the requested rate over the named clip |
| Quality | n/a | Useful High-profile coding | Luma PSNR at least 30 dB on the named synthetic clip, at least 28 dB on one natural clip |

Reconstruction is exact. Lossy comparison is only between the encoder input and an independent decode of its output.

## Justified exclusions

- Interlaced, field, and MBAFF pictures (`frame_mbs_only_flag` 0). The local library is progressive. The decoder returns `unsupported`.
- FMO (`num_slice_groups_minus1` > 0), data partitioning, SP/SI, scalable, multiview, and 3D extension NAL units.
- Chroma other than 4:2:0, bit depth other than 8, separate colour planes, and QP-prime transform bypass.
- SEI is skipped. It does not change samples. Film grain SEI is not applied.
- Encoder does not emit weighted prediction, FMO, or more than one slice per picture. It still decodes those when the exclusion above does not apply.
- Levels above the coded-size cap are refused. Caps: 4096×4096, 8 388 608 pixels, 16 reference frames, 8 MiB for one NAL, 256 PPS ids, 32 SPS ids.

## Numeric gates, named before tuning

Machine: this Mac, one thread, release build. Corpus: ffmpeg `testsrc2` 320×240 30 fps for 2 seconds at 500 kbps, and one natural clip from `../local-files` scaled to 640×360 for 30 frames at 800 kbps. Those files are not copied into the repo.

- Decode: samples match an independent decoder on the supported set. Tolerance is zero inside the cropped rectangle.
- Encode rate: total payload bits within 0.75×–1.25× of `bitrate * duration`.
- Encode quality: luma PSNR at least 30 dB (synthetic) and 28 dB (natural) against the source, after the independent decode. Chroma is reported, not gated, because the declared gate is luma.
- Speed budget: 1280×720 High decode at 30 fps, one thread, on this machine. A miss stays a miss.
- Memory: one 1280×720 decode stays under 64 MiB resident above the process baseline. A miss stays a miss.
- Fuzz: at least 30 minutes on the decoder entry and a bounded encoder campaign. Random bytes must return an error and leave the decoder reusable. The campaign is not a substitute for the short deterministic hostile tests.

## Milestones

1. Shared picture contracts, without changing the AAC audio traits.
2. Parameter sets, Annex B, and length prefixes.
3. Reconstruction: transform, intra, inter, deblock, DPB.
4. CAVLC and CABAC slice decode, bit-exact on independent streams.
5. High-profile encoder with inter prediction, B-frame reorder, and rate control.
6. Independent oracles, hostile input, fuzz, and release measurements.
