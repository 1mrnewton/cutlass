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
| 1 | 0.71 s | 0.41 s |
| 2 | 0.64 s | 0.37 s |
| 3 | 0.66 s | 0.34 s |
| 4 | 0.62 s | 0.37 s |

(Best of 8, interleaved, measured while other jobs kept the machine's
load average at about 8 on its 4 processors, so the threaded times mostly
show how much processor time each decoder needs; in the same runs the
previous design, which parsed every frame on the calling thread, took
0.85 s, 0.71 s and 0.85 s at 2, 3 and 4 threads. With the load average
at 4-6 this decoder took 0.36-0.39 s at 4 threads.) Single
threaded the decoder takes about 1.7x the time of libdav1d, which uses
hand-written assembly for its kernels; this crate relies on
autovectorization only (baseline x86-64: SSE2). With 2-4 threads the gap
is set by the processor time (see below).
Threading (2+ threads): the calling thread only reads the frame headers.
Each frame is a task on a thread of its own (the decoder keeps a pool of
them) that waits for what its parse depends on, and nothing else: the
CDFs of its primary reference frame (known when that frame's parse ends,
or at its start when the frame end CDF update is disabled) and the motion
fields and segment ids of the references it reads (when their parses
end). Frames no later frame depends on (half of bench1080) are parsed
beside the chain of referenced ones. With one tile column, a frame's
task hands each chunk of parsed blocks to a backend thread that
reconstructs it a superblock row behind and runs deblocking (in place,
one row behind) and CDEF and loop restoration (two rows behind). A
reconstruction that reads a reference still being decoded copies the
rows that reference publishes as final (only frames something waits for
publish them), waiting per chunk of blocks for the rows the chunk's
motion may reach (warped blocks and scaled references wait for the whole
picture), and reads the finished picture in place once it is done.
Multi-tile and super-resolved frames are decoded whole by their task
once the references are done. Up to as many frames as threads decode at
once; `Av1Core::decode_delayed`/`drain` return the shown frames with a
delay of as many frames (`Av1Decoder` uses them), and a tile data error
found later fails the frame's picture and the frames depending on it;
`Av1Core::decode` waits for the frames of each unit. Motion field
estimation runs alongside parsing, band by band. Tiles are not decoded in
parallel yet. The parse is no longer the critical path: on this file the
referenced frames' parses take about 0.2 s of processor time, the others
0.13 s, reconstruction and filters 0.55 s. Threaded decoding costs about
1.3x the processor time of single-threaded decoding (the parse records
blocks that reconstruction replays, page faults of the larger working
set: about 50k against 15k), so on a loaded machine the threaded times
suffer more than libdav1d's (0.57 s of processor time at 4 threads,
against 0.9-1.0 s). The `prof` feature prints the stage times, the
processor time of each kind of thread and the page faults from
`av1bench`.
The coefficient reader keeps the arithmetic decoder state in registers and
updates small CDFs as packed lanes; callgrind puts the whole decode at
5.07G instructions single threaded, and the parsing thread at 2.14G with
4 threads. CDEF filters fixed-stride padded blocks one 8-lane vector at a
time (0.46G of the total), and the motion field projection is shared by
runs of equal motion vectors. CDEF and loop restoration filter the
deblocked picture in place: CDEF keeps the pre-CDEF rows next to each
64-row boundary, loop restoration the deblocked rows next to each stripe
boundary (as libdav1d does), so no stage copies the picture. Deblocking
and the 8-bit Wiener filter run on 16-bit lanes.

## Encoder

### Coverage

Produces 8/10/12-bit 4:2:0, 4:2:2, 4:4:4, monochrome, any size
(`tools/encmatrix.sh`: every format decodes in libdav1d to exactly the
encoder's reconstruction). Tools in use: hierarchical mini-GOPs with hidden
alt-refs and up to seven references (up to 32 frames long on static
content, whose alt-refs are temporally filtered), RD partition search to
64x64, all intra modes with angle deltas, filter intra, CFL, luma and U/V
palettes (screen content), compound prediction with the NEAREST/NEAR/NEW
mode combinations, temporal MV prediction, switchable filters, local
warped motion, OBMC, skip mode, mode decisions charged with the exact
signalling cost from the CDFs, RD transform type and size, RDOQ with exact
coefficient rates, per-frame deblocking and CDEF strength search, constant
q or bitrate rate control. Optional: global motion, alt-ref temporal
filtering on all content. Not used: intra block copy,
wedge/difference-weighted compound, inter-intra, loop restoration, super
resolution, segmentation.

### Quality and speed against FFmpeg's encoders

`tools/rdcmp.sh`, single thread each. Ours at speed 6 (q 60..220); libaom
`-cpu-used 6` (crf 25..49); SVT-AV1 `-preset 8` (crf 25..49); rav1e
`-speed 8` (qp 60..180). BD-rate: bits ours needs relative to the other
encoder for equal PSNR (negative = fewer bits).

| Clip | vs libaom cpu-used 6 | vs SVT-AV1 preset 8 | vs rav1e speed 8 |
| --- | --- | --- | --- |
| natural, 352x288, 30 frames (aom test clip) | +0.4% | +15.7% | -17.6% |
| synthetic `testsrc2`, 640x360, 10 frames | +24.1% | -6.8% | -29.7% |

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
libaom crf 20..50, long GOP) now puts ours 3.5% ahead of libaom on the
30-frame natural clip, 1% behind on 60 frames (whose static second scene
alone is 16% behind) and 8% behind on a single key frame; synthetic
content (`testsrc2`, 10 frames) is 23% behind.

## Checks

`cargo fmt`, `cargo clippy -p cutlass-av1 --all-targets -- -D warnings`
and `cargo test -p cutlass-av1 --release` (103 tests: transform, filter,
predictor and coder kernels against spec transliterations, header and
OBU round trips, Annex B, encoder/decoder API including reordered GOPs and
error recovery, thread-independent tile coding) pass.
