# H.264

Native safe Rust decoder and encoder in `cutlass-h264`. No `unsafe`. No bundled C, FFmpeg, or other H.264 library. FFmpeg is an independent test oracle only.

About **76%** of the scope in `SCOPE.md` is done. The progressive decoder tools are in place. The encoder emits I, P, and B, and the named rate and luma PSNR gates passed on this machine. The release gates are still open.

| Part | Done | Where it stands |
| --- | --- | --- |
| Decoder | ~90% | Progressive 8-bit 4:2:0 Baseline, Main, and High. CABAC and CAVLC. I, P, and B, skip, spatial and temporal direct, 4×4 and 8×8 transform, deblock, and explicit and implicit weighted prediction. The tests that build those streams match ffmpeg inside the cropped rectangle. An IDR returns pictures that were still waiting, unless the slice says to drop them, and the IDR itself stays displayed. Picture order count types 0, 1, and 2 are all applied. Annex B and length prefixes of 1, 2, and 4 bytes are accepted. Adaptive reference marking is applied; without it, marking stays a sliding window. A frame_num gap is filled with non-existing references when the sequence allows it, and rejected when it does not. A redundant picture is discarded. Deblocking crosses a slice boundary unless that slice turns the boundary filter off. A picture that is not a multiple of 16 is returned at its visible size. Inter chroma uses the inter quantisation matrices. |
| Encoder | ~90% | High CABAC. I slices are intra 4×4 DC. P slices skip when the next picture equals the reconstruction, and otherwise choose 16×16, 16×8, 8×16, or four 8×8 blocks. Motion is a diamond search at integer, half-pel, and quarter-pel. Inter luma uses an 8×8 transform when that residual is closer than sixteen 4×4 transforms. One held frame is coded as a B slice between two references and shown in display order. B motion is 16×16, list 0, list 1, or both. When `bitrate` is set, the frame QP moves so the payload stays near that rate. On this machine, ffmpeg `testsrc2` at 320×240, 60 frames, 500 kbps measured 39.94 dB luma and 1,016,856 bits (target 1,000,000). One natural clip from `../local-files`, scaled to 640×360 for 30 frames at 800 kbps, measured 35.09 dB luma and 812,424 bits (target 800,000). Both are inside the `SCOPE.md` windows. |
| Release evidence | ~20% | No readiness report. No 30-minute fuzz run. No 720p speed or memory measurement. The named bitrate and luma PSNR gates have been run. |

## Encoder, still open

- B slices do not search 16×8, 8×16, or 8×8, and they do not emit skip or direct.
- Intra blocks stay DC 4×4. The luma gates passed with that choice.
- Weighted prediction, 4×4 sub-macroblocks, and more than one slice per picture stay unemitted, as `SCOPE.md` already records.

## Before this crate is finished

- Hostile-input tests, a fuzz run of at least 30 minutes, and release measurements: 1280×720 High decode at 30 fps on one thread, and under 64 MiB extra resident memory.
- A readiness report. Official ITU conformance bitstreams were not obtained, so that gap stays visible.
- Registration as the software decoder and encoder when the platform has none for this codec.

Exclusions that stay exclusions are in `SCOPE.md`: interlaced and field pictures, FMO, data partitioning, SP/SI, scalable and multiview NAL units, chroma other than 4:2:0, and bit depth other than 8.
