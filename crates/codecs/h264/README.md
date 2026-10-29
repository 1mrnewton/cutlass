# H.264

Native safe Rust decoder and encoder in `cutlass-h264`. No `unsafe`. No bundled C, FFmpeg, or other H.264 library. FFmpeg is an independent test oracle only.

About **55%** of the scope in `SCOPE.md` is done. The decoder is most of that. The encoder has started inter prediction. The release gates are still open.

| Part | Done | Where it stands |
| --- | --- | --- |
| Decoder | ~80% | Progressive 8-bit 4:2:0 Baseline, Main, and High. CABAC and CAVLC. I, P, and B, skip, spatial direct, 4×4 and 8×8 transform, deblock, and explicit and implicit weighted prediction. The tests that build those streams match ffmpeg inside the cropped rectangle. An IDR returns pictures that were still waiting, unless the slice says to drop them, and the IDR itself stays displayed. |
| Encoder | ~35% | High CABAC intra 4×4. A P skip when the next picture equals the reconstruction. A P slice of 16×16 blocks with a zero motion vector and a residual. FFmpeg accepts that residual, including a repeated ramp. |
| Release evidence | ~20% | No readiness report. No 30-minute fuzz run. No 720p speed or memory measurement. The bitrate and PSNR gates have not been run. |

Current work is the decoder. The encoder stays where it is until these decoder gaps are closed.

## Decoder, still required

- Temporal direct. Spatial direct is what the tests use.
- POC types 1 and 2. Type 0 works. A type 1 picture currently uses the parsed delta as its order count.
- Memory-management control operations. The syntax is consumed. Marking is sliding-window only.
- A checked pass for 1-byte and 2-byte length prefixes. Annex B and 4-byte lengths work.

## Encoder, still required

- A real motion vector, and partitions from 16×16 down through 8×8.
- B slices and reorder.
- An 8×8 transform. The encoder emits 4×4, and every intra block is DC.
- Rate control. QP is fixed at 26, so the 0.75×–1.25× bitrate gate cannot pass yet.
- Luma PSNR at least 30 dB on `testsrc2` 320×240, 2 seconds, 500 kbps, and at least 28 dB on one natural clip scaled to 640×360, 30 frames, 800 kbps.

## Before this crate is finished

- Hostile-input tests, a fuzz run of at least 30 minutes, and release measurements: 1280×720 High decode at 30 fps on one thread, and under 64 MiB extra resident memory.
- A readiness report. Official ITU conformance bitstreams were not obtained, so that gap stays visible.
- Registration as the software decoder and encoder when the platform has none for this codec.

Exclusions that stay exclusions are in `SCOPE.md`: interlaced and field pictures, FMO, data partitioning, SP/SI, scalable and multiview NAL units, chroma other than 4:2:0, and bit depth other than 8.
