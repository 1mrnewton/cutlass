# AV1 support record

Safe Rust (`#![forbid(unsafe_code)]`), no C libraries, no dependencies
outside this workspace. FFmpeg 6.1 with libdav1d, libaom, SVT-AV1 and rav1e
is the oracle and the baseline; nothing of it is linked. Every number below
was measured on the development machine (4 cores, shared, so wall times
carry a few percent of noise) with the scripts in `tools/`.

Specification: AV1 Bitstream & Decoding Process Specification v1.0.0 with
errata 1. Tables (CDFs, scans, quantizer lookups, filters) are generated
from the specification text.

## Decoder

### Coverage

| Area | Status |
| --- | --- |
| Profiles 0/1/2; 8/10/12-bit; 4:2:0, 4:2:2, 4:4:4, monochrome | done |
| OBUs: sequence header, temporal delimiter, frame, frame header, redundant header, tile group, metadata, padding | done |
| Annex B length-delimited input (`Av1Core::decode_annexb`) | done |
| Large-scale tile (`OBU_TILE_LIST`) | not implemented (also absent from libdav1d/FFmpeg) |
| Operating point selection / spatial layers | operating point 0 only |
| Frame types: key, inter, intra-only, switch; hidden frames, show_existing_frame, error resilience, frame ids, order hints, frame size with refs | done |
| Superblocks 64/128, uniform and explicit tiles, multiple tile groups, context update tile | done |
| Intra: all modes, angle deltas, edge filter/upsampling, filter intra, CFL, palette, intra block copy | done |
| Inter: all single/compound modes, MV stack incl. temporal projection, OBMC, local and global warp, wedge, difference-weighted, distance-weighted, inter-intra (smooth and wedge), skip mode, dual/switchable filters | done |
| Scaled references and reference frame size changes | done |
| Transforms 4..64, all types, lossless WHT, quantizer matrices | done |
| Segmentation, delta q, delta lf | done |
| Deblocking, CDEF, loop restoration (Wiener, self-guided), super resolution | done |
| Film grain synthesis on output pictures | done |

### Correctness

- `tools/conf.py` + `tools/cases.py` generate 46 streams with FFmpeg's
  libaom, SVT-AV1 and rav1e encoders (intra and inter, every chroma format
  and bit depth, odd sizes, tiles, 128x128 superblocks, screen content with
  palette and intra block copy, lossless, super resolution, film grain,
  frame resizing with scaled references, long alt-ref GOPs, a stream with
  OBMC/warp/wedge/difference-weighted/distance-weighted/inter-intra enabled).
  `tools/checkall.sh`: **46/46 identical to libdav1d** (`-autoscale 0`, film
  grain applied), at 1, 2, 3 and 4 threads.
- Mutation fuzzing (`examples/av1fuzz.rs`): no panics. Corrupt packets
  return an error and leave the reference slots as they were; frame size is
  limited (`Av1Core::max_pixels`) to bound memory.

### Speed

`bench1080.ivf`: 1920x1080, 60 frames, 8-bit 4:2:0 (libaom). Wall time
including process start, best of 3; same machine and load for both.

| Threads | cutlass-av1 | libdav1d (FFmpeg) |
| --- | --- | --- |
| 1 | 0.77-0.79 s | 0.39-0.43 s |
| 2 | 0.73 s | 0.31 s |
| 3 | 0.48 s | 0.30 s |
| 4 | 0.48-0.49 s | 0.30 s |

Single threaded the decoder takes about 1.9x the time of libdav1d, which
uses hand-written assembly for its kernels; this crate relies on
autovectorization only. With 4 threads the gap is about 1.6x.
Threading: with 3+ threads parsing and reconstruction run on separate
threads, and deblocking, CDEF and loop restoration follow behind on a
third; motion field estimation runs alongside parsing, band by band. Tiles
are not decoded in parallel yet, and one frame is finished before the next
starts (about 20% of the 4-thread time is that end-of-frame tail).
The coefficient reader keeps the arithmetic decoder state in registers and
updates small CDFs as packed lanes; callgrind puts the whole decode at
6.3G instructions single threaded.

## Encoder

### Coverage

Produces 8/10/12-bit 4:2:0, 4:2:2, 4:4:4, monochrome, any size
(`tools/encmatrix.sh`: every format decodes in libdav1d to exactly the
encoder's reconstruction). Tools in use: hierarchical mini-GOPs with hidden
alt-refs and up to seven references, RD partition search to 64x64, all
intra modes with angle deltas, filter intra, CFL, palette, compound
prediction, temporal MV prediction, switchable filters, local warped
motion, OBMC, skip mode, RD transform type and size, RDOQ with exact
coefficient rates, per-frame deblocking and CDEF strength search, constant
q or bitrate rate control. Optional: global motion, alt-ref temporal
filtering. Not used: intra block copy, wedge/difference-weighted compound,
inter-intra, loop restoration, super resolution, segmentation.

### Quality and speed against FFmpeg's encoders

`tools/rdcmp.sh`, single thread each. Ours at speed 6 (q 60..220); libaom
`-cpu-used 6` (crf 25..49); SVT-AV1 `-preset 8` (crf 25..49); rav1e
`-speed 8` (qp 60..180). BD-rate: bits ours needs relative to the other
encoder for equal PSNR (negative = fewer bits).

| Clip | vs libaom cpu-used 6 | vs SVT-AV1 preset 8 | vs rav1e speed 8 |
| --- | --- | --- | --- |
| natural, 352x288, 30 frames (aom test clip) | +10.2% | +26.7% | -11.7% |
| synthetic `testsrc2`, 640x360, 10 frames | +69.5% | +29.5% | +1.1% |

(BD-rate over the overlapping PSNR range of each pair; the synthetic clip's
numbers are higher than in earlier versions of this file mostly because
the q grid now reaches lower rates, where the gap is wider.)

Encoding speed on the natural clip: ours 10-15 fps, libaom 24-32, SVT-AV1
29-43, rav1e 9-27.

Multi-threaded, 1920x1080, 20 frames, 4 threads each (wall time; the
encoder splits HD frames into tile columns and codes them in parallel, with
output independent of the thread count): ours 7.3 s, libaom
`-cpu-used 6 -row-mt 1` 2.7 s, SVT-AV1 `-preset 8` 1.5 s, rav1e `-speed 8`
13.0 s.

So the encoder beats rav1e at its default speed on natural video and is
behind libaom and SVT-AV1, which have years of tuning (temporal dependency
models, more search, assembly kernels). `tools/kbd.sh` (q 60..220 against
libaom crf 20..50, long GOP) puts the natural-video gap to libaom at +5% on
30 frames, +10% on 60 frames and +9% on a single key frame; synthetic
content (flat areas, text, fast cuts) is the weak spot.

## Checks

`cargo fmt`, `cargo clippy -p cutlass-av1 --all-targets -- -D warnings`
and `cargo test -p cutlass-av1 --release` (103 tests: transform, filter,
predictor and coder kernels against spec transliterations, header and
OBU round trips, Annex B, encoder/decoder API including reordered GOPs and
error recovery, thread-independent tile coding) pass.
