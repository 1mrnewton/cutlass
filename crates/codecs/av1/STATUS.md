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
| Operating point selection (`Av1Decoder::select_operating_point`), layer dropping | done |
| Level limit: past level 6.3 is `Limit`; a tile that reads past its data is refused | done |
| Picture metadata: colour (named and CICP), chroma position, per-frame render size, timing info, decoding tick, `seq_level_idx` of the decoded point | done |
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
- Fixtures (`fixtures/`, 177 files, 86 streams from aomenc and hand-made
  OBU edits): every frame of 85 streams matches the libdav1d 1.5.3
  digests in `tests/libdav1d-digests.txt` (FFmpeg 8.1.1,
  `-c:v libdav1d -threads 1 -noautoscale -fps_mode passthrough`; grey
  streams on luma, with neutral chroma checked), checked by
  `tests/fixtures.rs` in CI without FFmpeg. `wide.ivf` (16448x64) is
  refused as past level 6.3, as libdav1d refuses it.
- Mutation fuzzing (`examples/av1fuzz.rs`): no panics. Corrupt packets
  return an error and leave the reference slots as they were; frame size is
  limited (`Av1Core::max_pixels`) to bound memory.

### Speed

`bench1080.ivf`: 1920x1080, 60 frames, 8-bit 4:2:0 (libaom). Wall time
including process start, median of 7 interleaved runs; same machine and
load for both (load average 7-9 on the 4 processors). CPU is user plus
system time.

| Threads | cutlass-av1 | libdav1d (FFmpeg) | cutlass-av1 CPU | libdav1d CPU |
| --- | --- | --- | --- | --- |
| 1 | 0.74 s | 0.45 s | 0.70 s | 0.43 s |
| 2 | 0.44 s | 0.31 s | 0.85 s | 0.49 s |
| 3 | 0.49 s | 0.32 s | 0.89 s | 0.52 s |
| 4 | 0.41 s | 0.32 s | 0.93 s | 0.54 s |

(Best runs: 0.69, 0.41, 0.45 and 0.34 s against libdav1d's 0.40, 0.31,
0.30 and 0.30 s. In the same interleaved runs the previous round's
decoder took 0.75, 0.45, 0.49 and 0.40 s (medians); an 11-run comparison
at 4 threads gave 0.53 s against its 0.56 s and libdav1d's 0.40 s.
Single threaded, callgrind counts 4.60G instructions (4.72G before
the MC pass rework below, 4.77G before that)
against libdav1d's 2.3G; its coefficient decoding is about as costly as
ours (1.2G), the rest is its assembly kernels. The previous design,
which parsed every frame on the calling thread, took 0.85 s, 0.71 s and
0.85 s at 2, 3 and 4 threads at a similar load.) Single
threaded the decoder takes about 1.7x the time of libdav1d, which uses
hand-written assembly for its kernels; this crate relies on
autovectorization only (baseline x86-64: SSE2). With 2-4 threads the gap
is set by the processor time (see below).
Multi-tile 4K (real files, 3840x2160 8-bit 4:2:0, Apple M5 Pro, 18
threads, wall time of the whole file, median of 3 interleaved runs at a
load average near 6; peak resident memory):

| File | Tiles | Before tile threads | Tile threads | libdav1d (FFmpeg) |
| --- | --- | --- | --- | --- |
| 2797 frames, 23.976 fps | 4x1 | 40.7 s, 683 MB | 23.3 s, 1099 MB | 5.7 s, 413 MB |
| 5326 frames, 25 fps | 8x4 | 66.8 s, 827 MB | 32.2 s, 1425 MB | 7.8 s, 423 MB |

Single threaded the same files take about 2.3x libdav1d's time. What is
left: every frame waits for its references to be done before its tiles
start (the chain of referenced frames decodes one frame at a time, about
3.7 ms each against libdav1d's 1.2 ms), the shells cost about 33 MB each
at 4K, and a frame decoding beside another may find no shell left
(57 of 258 frames on the 8x4 file).

Vector kernels: `cutlass-simd` holds NEON kernels the decoder calls on
aarch64. Each is a safe function on fixed-size arrays with a scalar twin
from the specification; other targets keep the lane code. So far: the
CDEF direction search, the 8-bit deblocking sample filter (sixteen lines
of an edge at a time), and the CDEF block filter for blocks with both
strengths. The first 240 frames of the two 4K files above, same output:

| File | Threads | Before the kernels | With them | libdav1d (FFmpeg) |
|---|---|---|---|---|
| 4x1 tiles | 1 | 5.88 s | 5.13 s | 2.54 s |
| 4x1 tiles | 18 | 1.24 s | 1.09 s | |
| 8x4 tiles | 1 | 4.75 s | 4.44 s | 1.88 s |
| 8x4 tiles | 18 | 1.30 s | 1.19 s | |

The lane code was already compiled to vectors, so a kernel wins only
where it packs more samples in a vector (8-bit deblocking) or where the
compiler had not found one (deblocking a vertical edge, the CDEF filter
with both strengths: 737 ns to 78 ns a block, a case neither file uses).
A vector is loaded from an array through two integer registers, not in
one instruction: a direct load needs a second kind of `unsafe`. Not on a
kernel yet: inter prediction, the inverse transforms, loop restoration,
and deblocking's edge decisions (about 12% of one thread's time).

Multi-tile frames in the pipeline, same two files, first 240 frames,
medians of 7 runs taken in turn on a busy machine, same output:

| File | Threads | Tiles parsed in turn | Tiles parsed at once |
|---|---|---|---|
| 4x1 tiles | 18 | 1.31 s | 1.12 s |
| 4x1 tiles | 8 | 1.33 s | 1.14 s |
| 4x1 tiles | 4 | 1.64 s | 1.60 s |
| 8x4 tiles | 18 | 1.28 s | 1.23 s |
| 8x4 tiles | 8 | 1.62 s | 1.55 s |

Reconstruction and the filters of a frame then got a thread each:
reconstruction keeps an unfiltered picture and hands every superblock
row it completes to the filters as a copy, and the filters work in the
final picture. Same files, medians of 7 runs taken in turn (the machine
was busy, so the ratios count more than the times):

| File | Threads | Before the pipeline took tiles | Tiles parsed at once | Filters on their own thread |
|---|---|---|---|---|
| 4x1 tiles | 18 | 1.19 s | 0.99 s | 0.86 s |
| 4x1 tiles | 8 | 1.18 s | 1.01 s | 0.87 s |
| 4x1 tiles | 4 | 1.55 s | 1.53 s | 1.18 s |
| 8x4 tiles | 18 | 1.25 s | 1.19 s | 1.22 s |
| 8x4 tiles | 8 | 1.59 s | 1.50 s | 1.25 s |
| 8x4 tiles | 4 | 1.75 s | 1.74 s | 1.38 s |

libdav1d takes 0.30 s and 0.31 s on these with all threads. Then, in
the same turn of work (best of 9 runs taken in turn, same output):

| File | Threads | Before | After | libdav1d (FFmpeg) |
|---|---|---|---|---|
| 4x1 tiles | 18 | 0.79 s | 0.63 s | 0.30 s |
| 4x1 tiles | 1 | 5.17 s | 4.49 s | 2.6 s |
| 8x4 tiles | 18 | 1.13 s | 0.89 s | 0.31 s |
| 8x4 tiles | 1 | 4.51 s | 3.87 s | 1.9 s |

What changed: published bands overlap, so a block's reference rows lie
in one band (a block across two went through a row-by-row copy, 9% of
the processor time); a frame no longer waits for a reference no block
near it reads; the context tile is parsed first on the thread already
running, and the parsers and the filters take and return logs in
batches (the parsers spent 15% of their time on the pool's lock); the
filters read 8 bytes of block info for each 4x4 unit, with the filter
levels worked out once for a block, where they read the 28-byte mode
info and worked a level out for every unit in every pass; the kernels
load a vector in one instruction; blocks taking both subpel filters go
through a dot product kernel (0.19 to 0.24 ns a sample against 0.34 to
0.40); and CDEF runs on every 64-row area but the last one deblocked.
Tried and dropped, each a little slower than what it replaced: a
deblocking kernel reading the plane's rows itself, and skipping the
narrower deblocking filters when no line uses their result.

The whole files take about 9 s (2.5 GB) and 22 s (3.4 GB) at 18 threads
against libdav1d's 5.7 s and 7.8 s. At most 8 frames decode at once.
On the first file the threaded decoder now keeps as many cores busy as
libdav1d does (about 11 of 18): what is left there is the work itself,
1.7x libdav1d's on one thread. A profile of one thread puts 25% in
CDEF, 21% in deblocking (a third of it the edge decisions), 17% in
inter prediction and 8% in the inverse transforms on the first file,
and 26% in inter prediction, 21% in deblocking and 21% in the inverse
transforms on the second. The second file is also held back by its
frames following their references: a frame trails by the superblock
rows the filters need plus its motion, and one frame in five has a
warped block and waits for whole references (0.7 s of waiting in 250
frames). Memory is mostly per frame in flight: two pictures, the mode
info, the block logs of the whole frame, the published bands (each
with the rows above it) and the frame states of its tile parsers.

Threading (2+ threads): the calling thread only reads the frame headers.
Each frame is a task on a thread of its own (the decoder keeps a pool of
them) that waits for what its parse depends on, and nothing else: the
CDFs of its primary reference frame (known when that frame's context
tile is parsed, or at its start when the frame end CDF update is
disabled), the saved motion of the references it projects (band by band
of 8 rows of 8x8 positions, as their parses save it) and the segment ids
of its primary reference (when that parse ends, and only when the frame
predicts from them). Frames no later frame depends on (half of bench1080)
are parsed beside the chain of referenced ones. A frame's tiles are
parsed on up to four threads, each into a frame state of its own; a
tile hands over its part of every superblock row (its block logs, loop
filter sizes, CDEF indices, restoration units and saved motion) and
whole rows go on in order to a backend thread that
reconstructs them a superblock row behind, in a picture that stays
unfiltered. Each finished row goes as a copy to a third thread, which
runs deblocking in the final picture as soon as the row arrives, and
CDEF and loop restoration one row behind. A reconstruction that reads a reference
still being decoded reads the bands of rows that reference publishes as
final (only frames something waits for publish them; the bands are
shared, not copied): a block whose rows lie in one band is predicted
from it in place, others go through the edge-emulation copy. It waits
per chunk of blocks for the rows the motion into each reference may
reach, taking the largest downward motion of the superblock row it is
in and the row above (warped blocks and scaled references wait for the
whole picture), and reads the finished picture in place once it is done.
Super-resolved frames, frames without in-loop filters, and every frame
when `Av1Core::frames_at_once` is off are decoded whole by their task
once the references are done. With 2+ threads the tiles of such a frame
(but not one using intra block copy) are decoded at the same time: the
frame's task decodes tiles into the frame, and helper threads into
shells of their own (a frame state and picture of the frame's size, from
a pool of at most `threads - 1` lent at once to all frames; a frame
finding none decodes its tiles alone), and what a helper's tile wrote is
then copied into the frame in bands of rows on several threads. Up to as many frames as threads decode at
once; `Av1Core::decode_delayed`/`drain` return the shown frames with a
delay of as many frames (`Av1Decoder` uses them), and a tile data error
found later fails the frame's picture and the frames depending on it;
`Av1Core::decode` waits for the frames of each unit. Motion field
estimation runs alongside parsing, band by band. The parse is no longer the critical path: on this file the
referenced frames' parses take about 0.2 s of processor time, the others
0.13 s, reconstruction and filters 0.55 s. Threaded decoding costs about
1.3x the processor time of single-threaded decoding (the parse records
blocks, about 70 bytes each, that reconstruction replays, and writes
the mode info a second time; callgrind counts 6% more instructions at 4
threads; page faults of the larger working set: about 38-40k against
14k, mostly pictures, published bands and block logs, with a resident
peak near 155 MB against 58 MB), so on a loaded machine the threaded
times suffer more than libdav1d's (libdav1d's processor time grows by
about a quarter from 1 to 4 threads, about as much as ours, so most of
the growth is the shared cores and caches). Frames others depend on get
no priority over the others yet: every frame task has a thread of its
own. The `prof` feature prints the stage times, the
processor time of each kind of thread and the page faults from
`av1bench`.
The coefficient reader keeps the arithmetic decoder state in registers and
updates small CDFs as packed lanes; callgrind puts the whole decode at
4.60G instructions single threaded. The largest kernels: CDEF 0.75G
(filter 0.46G, direction search 0.16G, which stays scalar: attempts to
vectorize its weighted squares with 16-bit products made the compiler
scalarize the partial sums instead), motion compensation 0.43G (the
vertical passes read a window of row slices checked once per row, and
the 2D path for widths 2-16 filters 1-4 rows of the intermediate as one
8/16-lane step with pre-splat taps: the compound hv kernel went from
0.25G to 0.17G; whole-sample copies of constant widths no longer call
memcpy per row), loop
filter 0.27G, inverse transforms about 0.3G. CDEF filters fixed-stride padded blocks one 8-lane vector at a
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
and `cargo test -p cutlass-av1 --release` (126 tests pass, 2 ignored: transform,
filter, predictor and coder kernels against spec transliterations, header
and OBU round trips, Annex B, encoder/decoder API including reordered GOPs
and error recovery, thread-independent tile coding; in `tests/`, the
libdav1d fixture check, picture metadata, operating points, the level
refusal, and OBU framing). The same suite passes in a debug build,
which checks arithmetic overflow.
