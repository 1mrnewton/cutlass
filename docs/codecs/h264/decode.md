# H.264 software decode

This file is the grind list for `cutlass-h264` decode. It is the software decoder only. VideoToolbox, VAAPI, NVDEC, and the other hardware decoders are out of this work. The encoder is out of this work.

The finish line is the recommendation’s software decoding process, head to head. FFmpeg is the independent oracle for pictures. When FFmpeg and the recommendation disagree, the recommendation wins. Read FFmpeg. Do not copy it, do not link it, and do not add `unsafe`.

`crates/codecs/h264/SCOPE.md` left 10-bit, other chroma, interlaced, and several annexes out. That list is not the finish line for this file.

## Where to read

| Source | What it decides |
| --- | --- |
| ITU-T H.264 (V14, 08/2021), clauses 7, 8, and 9 | Syntax, decoding process, CAVLC, CABAC. This is the reconstruction text. |
| The same recommendation, Annex A | Levels and the largest picture a level allows. |
| Annex B | Byte stream (start codes). |
| Annex D, and the SEI added in V16 | Picture timing and film grain. Other messages are parsed and do not change the samples. |
| Annexes G, H, and I | SVC, MVC, and 3D. A slice extension with no inter-layer prediction is decoded. Same-size inter-layer intra prediction copies the reference layer and adds a coded residual. Same-size list-0 motion is inherited. The reference layer's list-0 residual, including a nonzero vector, is added to an enhancement that codes its own motion. Base mode and that residual can both apply, and the enhancement adds a coded residual of its own. List 0 can name an older reference than the newest. An intra non-base MVC view is decoded. An intra Annex I depth view is decoded. Resampling, list 1, inter-view prediction, and a 3D-AVC header are still open. |
| V16 (06/2026) | The text in force. |
| ISO/IEC 14496-15 `avcC` | Not obtained. Length prefixes follow the record this crate already parses. |
| `../pub-packages/ffmpeg/libavcodec/h264dec.c` and the `h264_*.c` files next to it | Software decoder. Ignore `vaapi_h264.c`, `vdpau_h264.c`, `nvdec_h264.c`, `dxva2_h264.c`, `d3d12va_h264.c`, `vulkan_h264.c`. |
| `crates/codecs/h264/SCOPE.md` | What the crate claimed before this file. Use it for the tools already built. Do not use its exclusion list to stop a spec step. |

Official ITU conformance bitstreams were not obtained. That gap stays open until the streams are in hand. Do not invent a pass.

## Caps today

Picture size follows Annex A. The other rows are code limits. Do not raise one so a broken or endless stream can allocate past every defined level.

| Cap | Value now |
| --- | --- |
| Coded edge | 16,880 (1,055 macroblocks) |
| Coded macroblocks | 139,264 |
| Reference frames | 16 |
| One NAL | 8 MiB |
| SPS ids | 32 |
| PPS ids | 256 |
| CPB count in the HRD | 32 |

3840×2160 is 24,300 macroblocks. Level 5.1 and level 5.2 allow it. `12471039-uhd_3840_2160_24fps.mp4` matches FFmpeg inside the cropped rectangle: 351 pictures, 21348 ms, 16.4 fps, release, Apple M5 Pro. 4096×2160 is 34,560 macroblocks. Level 5.2 allows it. Levels 6, 6.1, and 6.2 share MaxFS 139,264 and an edge of 1,055 macroblocks. A picture past those is refused. The earlier measured run is 1280×720.

## Head to head

Rows are the software decoder. “FFmpeg” means `libavcodec`’s `h264` decoder with hardware decode left off. A blank speed cell means that tool was not timed.

| Tool | Spec | FFmpeg software | Cutlass | Grind |
| --- | --- | --- | --- | --- |
| Progressive 8-bit 4:2:0 | Clauses 7–8 | Yes | Yes | Done |
| Baseline, Main, High (`profile_idc` 66, 77, 100) | Annex A | Yes | Yes | Done |
| Extended (`88`) | Annex A | Yes | Profile 88, including a data-partitioned slice | Done |
| 10-bit 4:2:0 | High 10. `bit_depth_luma_minus8` is 2 | Yes | Pictures match | Done |
| Bit depths 9, 12, and 14 | `bit_depth_luma_minus8` 1, 4, and 6 | Yes, at 4:2:0, 4:2:2, and 4:4:4 | Pictures match | Done |
| Bit depths 11 and 13 | `bit_depth_luma_minus8` 3 and 5 | Refused (`Unimplemented luma bit depth`) | PCM and DC prediction follow the recommendation | Done. The recommendation is the oracle. |
| 4:2:2 and 4:4:4 | High 4:2:2, High 4:4:4 | Yes, at 8, 9, 10, 12, and 14 bits | Pictures match. 11 and 13 follow the recommendation. | Done |
| Separate colour planes | `separate_colour_plane_flag` | Refused | Three monochrome planes of one picture match the recommendation | Done. The recommendation is the oracle. |
| QP-prime transform bypass (lossless) | Clause 8.5 | Yes | Pictures match. Clause 8.5.15 runs for every profile. | Done |
| CAVLC and CABAC | Clause 9 | Yes | Yes | Done |
| I, P, and B, skip, direct, partitions through 4×4 | Clause 7.3.5 / 7.4.5 | Yes | Yes | Done |
| SP and SI | Clause 8.6 | I macroblocks. Clause 8.6 is not run. | Pictures match. SP P and the SI macroblock follow clause 8.6. | Done |
| 4×4 and 8×8 transform, custom scaling lists | Clause 8.5 | Yes | Yes | Done |
| Intra 4×4, 8×8, 16×16, I_PCM | Clause 8.3 | Yes | Yes | Done |
| Inter: quarter-sample luma, eighth-sample chroma, weighted prediction | Clause 8.4 | Yes | Yes | Done |
| Spatial and temporal direct, B skip, B direct 16×16 and 8×8 | Clause 8.4.1.2 | Yes | Yes | Done |
| Deblock, including across a slice boundary | Clause 8.7 | Yes | Yes. `disable_deblocking_filter_idc` 0, 1, and 2. | Done |
| Frame crop, visible size | Clause 7.4.2.1 | Yes | Yes | Done |
| POC types 0, 1, and 2 | Clause 8.2.1 | Yes | Yes | Done |
| DPB, sliding window, MMCO 1–6, IDR flush | Clause 8.2.5 | Yes | Yes | Done |
| Reference list modification | Clause 7.4.3.1 | Yes | Yes | Done |
| `frame_num` gaps | Clause 8.2.5.2 | Yes | Yes, when the SPS allows them | Done |
| Redundant pictures | Clause 7.4.3 | Yes | Discarded | Done |
| FMO / slice groups | Clause 8.2.2 | Compiled out (`FMO` is 0). The recommendation still has it. | Pictures match | Done |
| Data partitions (NAL types 2, 3, and 4) | Clause 7.3.2.9 | 8.1.1 refuses | Pictures follow the recommendation | Done |
| Interlaced, PAFF, MBAFF | Clauses 7–8, field pictures | Yes | Pictures match | Done |
| SVC, MVC, 3D (NAL types 14, 15, 20, 21) | Annexes G, H, I | Separate decoders | A slice extension with no inter-layer prediction is decoded. Same-size inter-layer intra prediction copies the base and adds a coded residual. Same-size list-0 motion is inherited. The reference layer's list-0 residual, including a nonzero vector, is added when the enhancement codes its own motion. Base mode and that residual can both apply, and the enhancement adds a coded residual of its own. List 0 can name an older reference than the newest. An intra non-base MVC view is decoded. An intra Annex I depth view is decoded. Resampling, list 1, inter-view prediction, and a 3D-AVC header are refused. A prefix or a subset SPS does not stop an Annex A picture. | Yet |
| Annex B and length prefixes of 1, 2, and 4 bytes | Annex B and 14496-15 | Yes | Yes | Done |
| `avcC` | 14496-15 | Yes | Yes | Done |
| SEI that changes the output picture or timing | Annex D, V16 | Picture timing, recovery point, buffering period. Film grain can change the displayed picture. | Picture timing changes the output time. Film grain changes the displayed picture and stays off the reference. Other SEI leaves the samples alone. | Done |
| VUI colour, range, matrix | Annex E | Exported | Every defined colour, transfer, and matrix idc is a tag on the picture. Reserved values stay unspecified. | Done |
| VUI sample aspect and timing | Annex E | Exported | Sample aspect and the sequence clock are on the picture. | Done |
| `level_idc` and MaxFS | Annex A | Recorded. A larger legal picture still decodes. | `level_idc` is stored. A picture inside any defined level is decoded, including one larger than the level the SPS names. A picture outside every level is refused. | Done |
| Error concealment | Not in the decoding process | Conceals missing slices | An incomplete picture is an error. The next good IDR still decodes. | Out. Do not conceal. |
| Frame threads and slice threads | Not in the decoding process | Both, from a shared pool in `libavcodec`. `-threads 1` forces one thread. Omitting `-threads` uses FFmpeg’s own count. | One thread. The decoder does not spawn. | Later. A shared Cutlass pool, not a pool inside this crate. |
| 1280×720 High, one thread | — | 140 ms, 849 fps, 120 frames | 420 ms, 286 fps, 120 frames | Speed |
| 3840×2160 High 8-bit 4:2:0, one thread | Legal at this size | Pictures match. Not timed. | 21348 ms, 16.4 fps, 351 frames | Done |

The 720p numbers are the recorded `against_ffmpeg` run: median of 3 after one warmup, Apple M5 Pro, release, FFmpeg 8.1.1. FFmpeg’s row includes process startup. Cutlass was about 3.0× slower. That run is throughput, not a second pixel check. Pixel checks on the supported tools are bit-exact inside the cropped rectangle against FFmpeg.

## Implemented

- One access unit in, pictures in output order out. Annex B, or a 1-, 2-, or 4-byte length prefix. `avcC` in `CodecParams.extra`.
- NAL types 1, 2, and 5 are slices. 3 and 4 complete a data-partitioned slice. 7 and 8 are parameter sets. 6 is SEI. 9, 10, 11, and 12 are skipped. Type 14 is a prefix and is skipped. Type 15 is a subset SPS. Type 20 with `no_inter_layer_pred_flag` 1 is a slice of the subset SPS. Same-size inter-layer intra prediction copies the base layer in that access unit, adds a coded residual, and the enhancement is the picture. List-0 motion is inherited from an inter reference-layer macroblock, and the samples come from the decoded picture buffer. The reference layer's list-0 residual, including a nonzero vector, is added to an enhancement that codes its own list-0 motion. Base mode and that residual can both apply, and the enhancement adds a coded residual of its own. List 0 can name an older reference than the newest. Type 20 with `svc_extension_flag` 0 is an intra non-base MVC view. Type 21 with `avc_3d_extension_flag` 0 is an intra Annex I depth view. Resampling, list 1, a reference base picture, inter-view prediction, and type 21 with `avc_3d_extension_flag` 1 are refused when they are the picture. MVC and that 3D-AVC header are skipped when an Annex A slice is the picture. Any other type fails while the access unit is walked.
- SPS and PPS, including High-profile fields, scaling lists, and frame crop. `level_idc` is stored. The picture must fit a level in Table A-1.
- Slice header for I, P, B, SP, and SI, including list modification, weights, decoded-reference marking, and `slice_qs_delta`.
- Macroblocks: I 4×4, I 8×8, I 16×16, I_PCM, P partitions and P skip, B partitions, B skip, B direct 16×16 and 8×8, spatial and temporal direct. An SP P macroblock and an SI macroblock are reconstructed by clause 8.6.
- Reconstruction: inverse 4×4 and 8×8, intra and inter prediction, weighted prediction, deblock, then crop to the visible size. 4:2:0, 4:2:2, and 4:4:4 samples are 8 through 14 bits. Field pictures weave into a frame. An MBAFF pair is two frame macroblocks or a top field and a bottom field. Slice groups use the map in clause 8.2.2. One group stays raster order.
- DPB output order, POC 0, 1, and 2, gaps, long-term marks, redundant pictures dropped.
- A bad packet returns an error and leaves a later good IDR decodable.
- `#![forbid(unsafe_code)]`. No SIMD intrinsics. No second thread.

## Yet to do

Do these in order. Each step is software decode. It is done when a new test fails if the behavior is removed, the pictures match an independent software decode inside the cropped rectangle, and `cargo fmt --all -- --check`, `cargo lint`, and `cargo test --workspace` pass.

The oracle command is `ffmpeg -hwaccel none -threads 1`. That is one software thread. This pass does not add a thread pool.

1. **Prove the 8-bit 4:2:0 path.** Done. `12471039-uhd_3840_2160_24fps.mp4` matches FFmpeg: 351 pictures, 21348 ms, 16.4 fps, release, Apple M5 Pro.
2. **High 10.** Done. A libx264 High 10 4:2:0 stream matches `ffmpeg -hwaccel none -threads 1` as `yuv420p10le`. The samples stay 10-bit. The 3840×2160 High file still matches: 351 pictures, 21348 ms, 16.4 fps, release, Apple M5 Pro.
3. **Bit depths 9, 12, and 14.** Done. libx264 on this machine writes 8-bit and 10-bit only, so the 4:2:0 test builds a 16×16 CAVLC stream. At 9, 12, and 14 bits that stream matches `ffmpeg -hwaccel none -threads 1` as `yuv420p9le`, `yuv420p12le`, and `yuv420p14le`: a flat I_16x16 DC picture, an I_16x16 picture with one luma DC coefficient, and an I_PCM picture. 11 and 13 follow the recommendation. PCM codes round-trip, and an unavailable intra DC prediction is the midpoint. FFmpeg refuses those two. The same depths at 4:2:2 and 4:4:4 are step 4. The High 10 stream still matches. The 3840×2160 High file is still the 8-bit check.
4. **4:2:2 and 4:4:4.** Done at 8, 9, 10, 11, 12, 13, and 14 bits. A 4:2:2 chroma block is 8×16 and its DC is eight coefficients. 4:4:4 has no separate chroma DC: each plane uses the luma residual. libx264 High 4:2:2 and High 4:4:4 streams at 8 and 10 bits, with B frames and deblock, match `ffmpeg -hwaccel none -threads 1` as `yuv422p`, `yuv444p`, `yuv422p10le`, and `yuv444p10le`. The other depths are a 16×16 CAVLC stream. At 8, 9, 10, 12, and 14 that stream matches FFmpeg: a flat I_16x16 picture, one luma DC coefficient, one chroma DC coefficient, and I_PCM. 11 and 13 follow the recommendation. PCM codes round-trip, and an unavailable intra DC prediction is the midpoint. FFmpeg refuses those two. `chroma_format_idc` 0 stays refused. The 4:2:0 pictures from steps 1–3 still match.
5. **Separate colour planes, then lossless bypass.** Done. A 16×16 High 4:4:4 Predictive picture with `separate_colour_plane_flag` is three monochrome slices. Plane 0 is a flat I_16×16. Planes 1 and 2 are I_PCM. A following P picture skips each plane and copies that plane of the reference. FFmpeg refuses the flag, so the recommendation is the oracle. A picture that omits a plane is an error. `colour_plane_id` 3 is an error. `chroma_format_idc` 0 stays refused. QP′Y-zero transform bypass matches `ffmpeg -hwaccel none -threads 1` on hand-built 4:2:0, 4:2:2, and 4:4:4 streams at QP′Y 0, including a profile 244 vertical 16×32 picture, and on a stream whose flag is set while QP′Y is 26. Clause 8.5.15 runs for every profile. A libx264 `-qp 0` 64×64 4:4:4 stream matches FFmpeg. The 4:2:0 pictures from steps 1–3 still match. No new 4K timing.
6. **Interlaced.** Done. Complementary fields weave even lines from the top field and odd lines from the bottom. A PAFF frame picture stays progressive. An MBAFF pair is two frame macroblocks or a top field and a bottom field. A 16×32 hand-built CAVLC stream matches `ffmpeg -hwaccel none -threads 1`: two frame I_16×16 macroblocks, and a field pair whose top is I_16×16 and whose bottom is I_PCM. libx264 MBAFF 64×64 streams match FFmpeg for CAVLC and CABAC I pictures, a predicted picture, all-field stripe pairs, and a picture that mixes frame pairs and field pairs. Field macroblocks use the field coefficient contexts. `direct_8x8_inference_flag` is 1 when the picture is interlaced. The 3840×2160 High file still matches: 351 pictures. No new 4K timing.
7. **FMO.** Done. Slice groups follow clause 8.2.2. The next macroblock is the next address in the same group. Hand-built 8-bit 4:2:0 CAVLC pictures match the recommendation: interleaved runs, a dispersed map, a foreground rectangle, box-out, reverse raster, a wipe, and an explicit map. An interlaced frame repeats each map unit on both rows. An adaptive pair shares one map unit. FFmpeg’s build leaves FMO out, so the recommendation is the oracle. A repeated macroblock is an error, and a later picture still decodes. A map type past 6, more than eight groups, and an explicit id past the group count are errors. The 3840×2160 High file still matches: 351 pictures. No new 4K timing.
8. **Data partitions, SP, and SI.** Done. NAL type 2 carries the slice header, `slice_id`, and the category-2 syntax. Type 3 carries intra residual and PCM. Type 4 carries inter residual. A 16×16 Extended CAVLC picture follows the recommendation: I_PCM, a flat I_16×16, a P skip of that picture, and a P macroblock whose only luma DC coefficient is +1. At QP 26 with deblock off, that 4×4 is 131 and the rest of the picture stays 128. FFmpeg 8.1.1 does not decode data partitions. The tree under `../pub-packages/ffmpeg` still has that decoder, and this machine has no built copy of it, so the recommendation is the oracle. An orphan partition, a partition out of order, a mismatched `slice_id`, an unused partition, CABAC with data partitioning, and a QS outside 0..=51 are errors. A later IDR still decodes. An SP P skip at QSY 0 stays 128. At QSY 26 the luma becomes 127 and the chroma stays 128. The same macroblock with a +1 luma DC coefficient is 131 when `sp_for_switch_flag` is 0 and 128 when it is 1. An SI macroblock with no residual is luma 127 and chroma 128. I_PCM inside SP and inside SI matches `ffmpeg -hwaccel none -threads 1`. The 3840×2160 High file still matches: 351 pictures. No new 4K timing.
9. **Annex D and the V16 SEI that changes output.** Done. NAL type 6 is parsed. Picture timing sets the output duration from Table D-1: `pic_struct` 0, 3, and 4 keep one frame, 1 and 2 are one field, 5 and 6 are one and a half frames, 7 is two frames, and 8 is three. `dpb_output_delay` shifts the presentation time only when `num_units_in_tick` and `time_scale` divide the packet timescale exactly. A delay that is not an integer number of packet ticks leaves the packet time. A flat 16×16 picture with picture timing, and one whose SEI is filler or an unknown payload type, match `ffmpeg -hwaccel none -threads 1` as `yuv420p`. Film grain follows H.274 equations 22–32 on a copy of the reconstructed picture. The reference stays clean, so a following P skip is still 128. Frequency filtering and auto-regression both change the luma and leave the chroma. The same message on the same picture is deterministic. FFmpeg's grain synthesis is a different approximation, and H.274 does not require the display method to match that approximation, so grain pictures follow the recommendation. Model 2, blending mode 2, a depth that does not match the picture, and a cancel flag leave the samples alone. Period 0 applies to that picture only. Period 1 persists until the next grain message or an IDR. A truncated payload, `pic_struct` past 8, a model value out of range, and a repetition period above 16384 are errors. A later IDR still decodes. Source picture timing and the other V16 messages are consumed by size and do not replace the decoded samples or the output time. The 3840×2160 High file still matches: 351 pictures. No new 4K timing.
10. **Annex E.** Done. Table E-1 sample aspect is on the picture. An absent flag, idc 0, an idc the table does not list, and an extended ratio with a zero edge are unspecified, width 0 and height 1. idc 1 is square, idc 2 is 12:11, idc 14 is 4:3, idc 16 is 2:1, and idc 255 carries the explicit width and height. `num_units_in_tick`, `time_scale`, and `fixed_frame_rate_flag` are on the picture. Absent timing information is zeros. A flat 16×16 picture with aspect 12:11, BT.709 tags, and a 60 000/1 001 clock matches `ffmpeg -hwaccel none -threads 1` as `yuv420p`, and `ffprobe` reports `12:11` and `bt709`. The defined colour, transfer, and matrix idc values are tags. Values 5 and 6 share BT.601. Transfer values 1 and 6 are BT.709. Transfer 16 is the perceptual quantizer and 18 is hybrid log-gamma. Transfer 14 and 15 stay BT.2020. Reserved idc values stay unspecified. The colour crate converts SMPTE 240M primaries as SMPTE 170M. A primary it has no chromaticities for follows the unspecified matrix. A transfer it does not implement stays an unspecified curve. The 3840×2160 High file still matches: 351 pictures. No new 4K timing.
11. **Annex A.** Done. `level_idc` is stored. Table A-1's greatest MaxFS is 139,264 macroblocks, shared by levels 6, 6.1, and 6.2. The edge is 1,055 macroblocks, 16,880 samples. A picture inside any defined level is decoded, including one larger than the level named in the SPS. A picture outside every level is refused before a frame is allocated. A later 16×16 IDR still decodes. Flat CAVLC pictures match `ffmpeg -hwaccel none -threads 1` as `yuv420p`: 480×16 at level 1, which is past that level's edge, 8,704×16 at level 5.2, which is past that level's edge, 16,880×16 at level 6.2, 4,096×2,160 at level 5.2, and 8,688×1,088 at level 6, which is past level 5.2's MaxFS. A width of 1,056 macroblocks, 1,024×137 macroblocks, a field whose frame height is 1,056 macroblocks, and a width of 1,000,000 macroblocks are refused. The stored picture edge is 16,880. The 3840×2160 High file still matches: 351 pictures. No new 4K timing.
12. **Annexes G, H, and I.** A coded slice extension (NAL 20) with `no_inter_layer_pred_flag` 1 reconstructs with the Annex A process. Profiles 83 and 86, `quality_id` above 0, and `slice_header_restriction_flag` 0 are included. The picture is a flat I_16×16. FFmpeg 8.1.1 does not decode NAL 20, so the recommendation is the oracle. `output_flag` 0 is reconstructed and is not displayed. A prefix NAL, a subset SPS, an MVC header, and a 3D NAL leave an Annex A slice as the picture, and that picture matches `ffmpeg -hwaccel none -threads 1`. Same-size inter-layer intra prediction is decoded. The base is I_PCM, luma 200 and chroma 40. The enhancement sets `base_mode_flag` and adds no residual, and its crop is 8×8, so the displayed picture is that crop of the base. A second enhancement adds one luma DC coefficient of +1 at QP 26. That 4×4 becomes 203 and the rest of the picture stays 200 and 40. A later access unit has a P base with that same coefficient on an I_PCM reference of luma 150 and chroma 90, so the base corner is 153. The enhancement inherits the zero motion and is the reference, luma 150 and chroma 90. A later pair uses an I_PCM whose left half is 100 and right half is 180. The base adds 3 on the top-left 4×4, so that block is 103. The enhancement codes its own motion of eight pixels to the right and adds that residual, so the top-left 4×4 is 183 and the rest of the luma is 180. When the base moves eight pixels and the enhancement stays still, the base corner is 183 and the enhancement corner is 103. The rest of the enhancement stays 100 on the left and 180 on the right. Another enhancement inherits that eight-pixel motion, adds the base residual, and adds its own coefficient on the next 4×4. The top-left 8×4 is 183 and the rest of the luma is 180. A later pair keeps two references. The older picture is split, 100 and 180. The newer picture is luma 40. The base uses reference index 1 and adds 3, so its corner is 103. One enhancement inherits that index and is the older picture, 100 and 180. Another predicts from the newer picture and adds the base residual, so its corner is 43 and the rest of the luma is 40. FFmpeg emits the base. The recommendation's picture is the enhancement. A reference base picture and NAL type 21 with `avc_3d_extension_flag` 1 are refused when they are the only picture. A broken enhancement in the same access unit is an error. Profiles 118 and 128 are a subset SPS with the MVC extension. A non-base view coded as an I slice is the picture. FFmpeg 8.1.1 does not decode that view, so the recommendation is the oracle. A view id that is not a non-base view is an error. Inter-view prediction is not applied. Profiles 135 and 138 are a subset SPS with the MVCD extension. NAL type 21 with `avc_3d_extension_flag` 0 is an intra depth view and the picture is that reconstruction. FFmpeg 8.1.1 does not decode it, so the recommendation is the oracle. A view that is not a depth view is an error. NAL type 21 with `avc_3d_extension_flag` 1 stays refused. A truncated extension header, a reserved header bit, `svc_vui_parameters_present_flag`, `mvc_vui_parameters_present_flag`, `mvcd_vui_parameters_present_flag`, and a truncated subset SPS are errors. A later IDR still decodes. The 3840×2160 High file still matches: 351 pictures. No new 4K timing. Resampling, list 1, inter-view prediction, and a 3D-AVC header are still open.
13. **Conformance streams.** When an ITU stream is obtained, run it and record the clause. Until then that stream stays not available. A self-encoded file does not close that line.

## Later

A shared Cutlass pool, used by decoders the way `libavcodec`’s frame and slice pools are. One pool decodes whole pictures. The other runs slices of one picture. The reconstruction waits on a row another worker is still writing. It does not live inside this crate, and it is not part of the steps above.

## Do not grind

- Hardware decode, and any change that makes the desktop skip this software decoder.
- The encoder.
- Error concealment. The decoding process does not invent missing pixels. A broken picture stays an error.
- Copying FFmpeg, OpenH264, or JM into this crate.
- Raising a cap so Cutlass accepts a picture no level in Annex A allows.
- A thread pool, in this crate or elsewhere. That pool is the later work.
