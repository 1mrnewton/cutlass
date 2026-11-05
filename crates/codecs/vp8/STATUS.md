# VP8 support record

Checkout: branch `vp8`, worktree `/Users/mrnewton/dev/cutlass/cutlass-vp8`, started from `development` at `3607813a`. Native safe Rust in `cutlass-vp8`. No `unsafe`. No libvpx, FFmpeg, or GStreamer code in the crate or its dependency graph. Those programs are test oracles only.

Request: a full VP8 decoder and encoder, built, tested, fixed, and benchmarked against ffmpeg.

## Rules that were loaded

All of `.cursor/rules` in this checkout: `codec-shipping`, `commit-often`, `container-shipping`, `development-worktree`, `local-media`, `merge-pull-request`, `no-capcut-names`, `perf`, `platforms`, `pull-request`, `rust-standards`, `source-of-truth`, `tests`, `think-like-a-pro`, `worktree-names`. Skills: `ship-codec` with every reference, `think-like-a-pro`, `cutlass-worktree`. There is no `AGENTS.md` in this tree.

## Specification and sources

- RFC 6386, *VP8 Data Format and Decoding Guide*, November 2011. Section 1 says the reference decoder source in Section 20 wins over the narrative. Section 20 is read as the normative definition.
- RFC 6386 Section 7 for the boolean encoder, and Section 20 (`dixie`) for every decode table and procedure.
- FFmpeg `libavcodec/vp8.c`, `vp8dsp.c`, `vp8data.h`, `vpx_rac.h`, `vp8_parser.c` from `../pub-packages/ffmpeg`, read for edge handling, MV clamping, reference updates, and threading layout.
- GStreamer `gst-plugins-bad/gst-libs/gst/codecparsers/gstvp8parser.c` (header parsing) and `gst-plugins-good/ext/vpx/gstvp8enc.c` / `gstvp8dec.c` (libvpx wrappers) from `../pub-packages/gstreamer`.
- GStreamer and FFmpeg ship no native VP8 encoder. Both wrap libvpx. The encoder here is designed from RFC 6386 and validated with two independent decoders.

### Where the sources disagree

| Case | RFC 6386 Section 20 | libvpx | FFmpeg | This crate |
| --- | --- | --- | --- | --- |
| Segment loop-filter level outside 0..63 before the ref/mode deltas | Clamped first | Clamped first | Not clamped | Clamped first (Section 20) |
| Segment quantizer index outside 0..127 before the per-plane deltas | Not clamped | Clamped first | Not clamped | Not clamped (Section 20) |
| `copy_buffer_to_golden = 2` in the same frame as `copy_buffer_to_alternate` | Golden takes the new altref | Golden takes the new altref | Golden takes the old altref | New altref (Section 20) |
| Loop-filter delta whose update flag is clear | Set to 0 | Kept | Kept | Kept. The Section 19.2 prose says the flag marks whether the value is updated, and every shipping encoder relies on that. Recorded as an interoperability choice. |
| `mb_no_coeff_skip = 0` | Reuses a stale skip flag | Skip is 0 | Skip is 0 | Skip is 0 |

The encoder never emits any of these cases.

## Required target

Decoder: the whole VP8 bitstream of RFC 6386.

- Versions 0 to 3: six-tap or bilinear interpolation, and full-pixel chroma for version 3. The loop-filter type comes from the header bit.
- Key frames and inter frames, a frame size change on any key frame, and odd sizes up to 16383×16383.
- Intra 16×16 DC, V, H, TM, and B_PRED with all ten sub-block modes. Key-frame contexts, and the 127/129 edge rules.
- Inter modes NEAREST, NEAR, ZERO, NEW, and SPLITMV with 16×8, 8×16, 8×8, and 4×4 partitions. LAST, GOLDEN, and ALTREF with sign bias, refresh flags, and copy flags.
- Probability updates for coefficients, MV, and modes; `refresh_entropy_probs` save and restore.
- Segmentation: map update and persistence, and absolute or delta quantizer and filter levels.
- Normal and simple loop filters with sharpness and ref/mode deltas.
- One, two, four, or eight token partitions; skip on or off; hidden frames (`show_frame = 0`).
- Colour-space and clamp bits, and the horizontal and vertical scale bits, reported on the decoder.

Encoder: version 0, 4:2:0 8-bit.

- Key frames and inter frames; intra 16×16 and B_PRED by rate-distortion choice.
- Inter NEAREST, NEAR, ZERO, NEW, and SPLITMV (all four partitionings), with quarter-pel motion search on LAST and GOLDEN.
- A periodic golden refresh, sign bias 0, adaptive coefficient/MV/skip probability updates, and a searched loop-filter level with ref/mode deltas.
- Optional four-segment activity quantization; one to eight token partitions.
- Constant quantizer and target-bitrate modes.

## Gates, declared before tuning

| Gate | Bound |
| --- | --- |
| Decode libvpx streams across speed, partitions, error-resilient, alt-ref with lag, sharpness, static threshold, versions 1–3, odd sizes, multiple key frames | YUV identical to ffmpeg `vp8` and to `libvpx` |
| Decode the independent 1920×1080 WebM in `../local-files` | YUV identical to ffmpeg `vp8` |
| Our encoder's packets | Decoded by ffmpeg `vp8` and by `libvpx`, both identical to our reconstruction |
| Lossy quality, 640×360 natural clip at 1 Mb/s | PSNR-Y within 2.0 dB of libvpx `-deadline good -cpu-used 4` at the same measured rate |
| Rate, 60 frames at a target of 1 Mb/s | Mean rate within 0.85×–1.15× of the target |
| Decode speed, release, one thread, Apple M5 Pro | 1920×1080 at ≥ 60 fps, with the ratio to ffmpeg recorded |
| Encode speed, release, one thread | 640×360 at ≥ 30 fps on the default preset |
| Decoder fuzz | ≥ 30 minutes, seed and findings recorded, findings 0 |

## Capability matrix

Status is `planned` until an executed check exists.

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Bool coder | Read | Write | unit tests | planned |
| Frame header, partitions | Accept | Emit | unit tests | planned |
| Intra 16×16, B_PRED, chroma | Accept | Emit | ffmpeg comparison | planned |
| Inter modes, split MV, three references | Accept | Emit (LAST, GOLDEN) | ffmpeg comparison | planned |
| Versions 1–3 | Accept | Error | ffmpeg comparison | planned |
| Segmentation | Accept | Emit | ffmpeg comparison | planned |
| Loop filter normal, simple, deltas, sharpness | Accept | Emit normal | ffmpeg comparison | planned |
| Token partitions 1/2/4/8 | Accept | Emit | ffmpeg comparison | planned |
| Hidden frames, alt-ref | Accept | Not emitted | ffmpeg comparison | planned |
| Lifecycle, malformed input, cancel | Specified | Specified | unit tests | planned |
| Fuzz | Required | n/a | fuzz test | planned |
| Benchmarks against ffmpeg | Required | Required | `cargo bench -p cutlass-vp8` | planned |

## Outside this crate

A WebM demuxer (`cutlass-matroska` is an empty scaffold), playback, and export. The ISOBMFF crate already maps `vp08`; the integration test drives that path.
