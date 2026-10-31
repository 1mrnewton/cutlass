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
| Annex D, and the SEI added in V16 | Messages. The ones that change the output picture or timing are still to do. |
| Annexes G, H, and I | SVC, MVC, and 3D. Still to do, after the single-layer decoder matches. |
| V16 (06/2026) | The text in force. |
| ISO/IEC 14496-15 `avcC` | Not obtained. Length prefixes follow the record this crate already parses. |
| `../pub-packages/ffmpeg/libavcodec/h264dec.c` and the `h264_*.c` files next to it | Software decoder. Ignore `vaapi_h264.c`, `vdpau_h264.c`, `nvdec_h264.c`, `dxva2_h264.c`, `d3d12va_h264.c`, `vulkan_h264.c`. |
| `crates/codecs/h264/SCOPE.md` | What the crate claimed before this file. Use it for the tools already built. Do not use its exclusion list to stop a spec step. |

Official ITU conformance bitstreams were not obtained. That gap stays open until the streams are in hand. Do not invent a pass.

## Caps today

These are the current code limits, not Annex A. A picture inside a level the recommendation defines, and outside this table, is a spec miss. Raise a cap when that step needs it. Do not raise one so a broken or endless stream can allocate like FFmpeg.

| Cap | Value now |
| --- | --- |
| Coded edge | 4096 |
| Coded pixels | 8,388,608 |
| Reference frames | 16 |
| One NAL | 8 MiB |
| SPS ids | 32 |
| PPS ids | 256 |
| CPB count in the HRD | 32 |

3840×2160 is 8,294,400 pixels, so that size already fits. `12471039-uhd_3840_2160_24fps.mp4` matches FFmpeg inside the cropped rectangle: 351 pictures, 21348 ms, 16.4 fps, release, Apple M5 Pro. 4096×2160 is 8,847,360 pixels. Annex A level 5.2 allows it. This decoder refuses it. The earlier measured run is 1280×720.

## Head to head

Rows are the software decoder. “FFmpeg” means `libavcodec`’s `h264` decoder with hardware decode left off. A blank speed cell means that tool was not timed.

| Tool | Spec | FFmpeg software | Cutlass | Grind |
| --- | --- | --- | --- | --- |
| Progressive 8-bit 4:2:0 | Clauses 7–8 | Yes | Yes | Done |
| Baseline, Main, High (`profile_idc` 66, 77, 100) | Annex A | Yes | Yes | Done |
| Extended (`88`) without data partitions | Annex A | Yes | The profile id is accepted. Partition NALs still fail. | Yet |
| 10-bit 4:2:0 | High 10. `bit_depth_luma_minus8` is 2 | Yes | Pictures match | Done |
| Bit depths 9, 12, and 14 | `bit_depth_luma_minus8` 1, 4, and 6 | Yes, at 4:2:0, 4:2:2, and 4:4:4 | Pictures match | Done |
| Bit depths 11 and 13 | `bit_depth_luma_minus8` 3 and 5 | Refused (`Unimplemented luma bit depth`) | PCM and DC prediction follow the recommendation | Done. The recommendation is the oracle. |
| 4:2:2 and 4:4:4 | High 4:2:2, High 4:4:4 | Yes, at 8, 9, 10, 12, and 14 bits | Pictures match. 11 and 13 follow the recommendation. | Done |
| Separate colour planes | `separate_colour_plane_flag` | Refused | Three monochrome planes of one picture match the recommendation | Done. The recommendation is the oracle. |
| QP-prime transform bypass (lossless) | Clause 8.5 | Yes | Pictures match. Clause 8.5.15 runs for every profile. | Done |
| CAVLC and CABAC | Clause 9 | Yes | Yes | Done |
| I, P, and B, skip, direct, partitions through 4×4 | Clause 7.3.5 / 7.4.5 | Yes | Yes | Done |
| SP and SI | Clause 7.3.5 | Parsed | Refused (`slice type`) | Yet |
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
| FMO / slice groups | Clause 8.2.2 | Compiled out (`FMO` is 0). The recommendation still has it. | Refused | Yet |
| Data partitions (NAL types 2, 3, 4) | Clause 7.3.2.9 | Yes | Refused | Yet |
| Interlaced, PAFF, MBAFF | Clauses 7–8, field pictures | Yes | Pictures match | Done |
| SVC, MVC, 3D (NAL types 14, 15, 20, 21) | Annexes G, H, I | Separate decoders | Refused | Yet, after the single layer matches |
| Annex B and length prefixes of 1, 2, and 4 bytes | Annex B and 14496-15 | Yes | Yes | Done |
| `avcC` | 14496-15 | Yes | Yes | Done |
| SEI that changes the output picture or timing | Annex D, V16 | Picture timing, recovery point, buffering period. Film grain can change the displayed picture. | The NAL is skipped | Yet |
| VUI colour, range, matrix | Annex E | Exported | Exported, for the idc values the parser maps | Done for those values. Map the remaining idc values. |
| VUI sample aspect and timing | Annex E | Exported | Read and dropped | Yet |
| `level_idc` and MaxFS | Annex A | Recorded. A larger legal picture still decodes. | Read and ignored. 4096×2160 is refused. | Yet |
| Error concealment | Not in the decoding process | Conceals missing slices | An incomplete picture is an error. The next good IDR still decodes. | Out. Do not conceal. |
| Frame threads and slice threads | Not in the decoding process | Both, from a shared pool in `libavcodec`. `-threads 1` forces one thread. Omitting `-threads` uses FFmpeg’s own count. | One thread. The decoder does not spawn. | Later. A shared Cutlass pool, not a pool inside this crate. |
| 1280×720 High, one thread | — | 140 ms, 849 fps, 120 frames | 420 ms, 286 fps, 120 frames | Speed |
| 3840×2160 High 8-bit 4:2:0, one thread | Legal at this size | Pictures match. Not timed. | 21348 ms, 16.4 fps, 351 frames | Done |

The 720p numbers are the recorded `against_ffmpeg` run: median of 3 after one warmup, Apple M5 Pro, release, FFmpeg 8.1.1. FFmpeg’s row includes process startup. Cutlass was about 3.0× slower. That run is throughput, not a second pixel check. Pixel checks on the supported tools are bit-exact inside the cropped rectangle against FFmpeg.

## Implemented

- One access unit in, pictures in output order out. Annex B, or a 1-, 2-, or 4-byte length prefix. `avcC` in `CodecParams.extra`.
- NAL types 1 and 5 are slices. 7 and 8 are parameter sets. 6, 9, 10, 11, and 12 are skipped. Types 2, 3, 4, 14, 15, 20, and 21 fail at the NAL header. Any other type fails while the access unit is walked.
- SPS and PPS, including High-profile fields, scaling lists, and frame crop. `level_idc` is not stored.
- Slice header for I, P, and B, including list modification, weights, and decoded-reference marking.
- Macroblocks: I 4×4, I 8×8, I 16×16, I_PCM, P partitions and P skip, B partitions, B skip, B direct 16×16 and 8×8, spatial and temporal direct.
- Reconstruction: inverse 4×4 and 8×8, intra and inter prediction, weighted prediction, deblock, then crop to the visible size. 4:2:0, 4:2:2, and 4:4:4 samples are 8 through 14 bits. Field pictures weave into a frame. An MBAFF pair is two frame macroblocks or a top field and a bottom field.
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
7. **FMO.** Slice groups, including maps FFmpeg’s build leaves out. The recommendation is the oracle where FFmpeg cannot decode the stream.
8. **Data partitions, SP, and SI.** NAL types 2, 3, and 4, and slice types SP and SI.
9. **Annex D and the V16 SEI that changes output.** Film grain and picture timing that change the displayed picture or the output time. Other SEI is parsed and leaves the samples alone.
10. **Annex E.** Sample aspect and timing on the picture. The remaining colour, transfer, and matrix idc values.
11. **Annex A.** A picture the declared level allows is decoded, including 4096×2160 at level 5.2 and the level 6 sizes. A picture past every defined level is refused.
12. **Annexes G, H, and I.** SVC, then MVC, then 3D. Only after the single-layer steps match.
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
