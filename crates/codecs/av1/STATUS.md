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

A later turn of work, measured against the `dav1d` 1.5.3 command line
tool (`--muxer null`) rather than through FFmpeg, on a machine busy with
other work (best of 5 runs at 18 threads, of 3 at one):

| File | Threads | Before | After | dav1d |
|---|---|---|---|---|
| 4x1 tiles | 18 | 0.63 s | 0.61 s | 0.30 s |
| 4x1 tiles | 1 | 4.49 s | 4.14 s | 2.55 s |
| 8x4 tiles | 18 | 0.89 s | 0.61 s | 0.31 s |
| 8x4 tiles | 1 | 3.87 s | 3.60 s | 1.91 s |

The whole files take 9.1 s (2.6 GB) and 15.8 s (3.6 GB) at 18 threads
against dav1d's 4.8 s and 7.8 s (0.4 GB each).

What changed: a warped block waits for the rows it reads of a reference
still being decoded and no longer for the whole reference (the warp
filter takes its 15 rows from the band that holds them); any block that
finds rows not published yet stops, the worker waits for those rows and
the block is predicted again, so the wait ahead of each chunk of blocks
is a shortcut and not what keeps reads in bounds (on these files no
block has to stop: the wait ahead already covers the rows). 8-bit CDEF
blocks whose taps all lie in the picture go through a byte kernel,
sixteen samples to a vector (9.7 ns against 13.9 for a 4x4 block with
one strength), and the taps derived from a block's strengths, direction
and damping are kept for the next block that shares them. Deblocking
gathers a row's edges into runs as it scans, and its kernel skips the
filters no line of a group takes. The row pass of an inverse transform
goes eight rows at a time whenever the block has eight, also with few
non-zero rows: the four-lane code is the slower one.
Tried and dropped: a vector kernel running the steps of the inverse
transforms (butterfly and Hadamard rotations on eight 16-bit columns)
from the same step tables. It took as long as the compiler's code for
the same loop: the time is in running a transform as a table of steps
through memory, not in the arithmetic of a step.

Where one thread's time goes against dav1d's, from samples at 1 ms of
both decoders on the same clips (approximate: dav1d's unnamed assembly
is assigned by address to the nearest named routine, and about 140 of
its samples have no owner; its deblocking is taken from its own
`--inloopfilters` switch, which gives 0.2 to 0.45 s):

| Stage, 8x4 tiles | cutlass-av1 | dav1d |
|---|---|---|
| Inter prediction | 0.94 s | 0.68 s |
| Inverse transforms | 0.72 s | 0.20 s |
| Deblocking | 0.68 s | about 0.25 s |
| Reading blocks and coefficients, mode info | 0.60 s | 0.34 s |
| Clearing and copying memory | 0.17 s | 0.02 s |
| Intra prediction | 0.14 s | 0.06 s |
| Motion vector prediction, motion field | 0.14 s | 0.17 s |

On the 4x1 file: CDEF 0.86 s against 0.67 s, deblocking 0.73 s against
0.25 to 0.45 s, inter prediction 0.62 s against 0.62 s, inverse
transforms 0.30 s against 0.10 s. So the inverse transforms are the
widest gap (3 to 3.5x), then deblocking, of which 0.28 s is deciding
the edges (dav1d builds bit masks as it reads a block, at next to no
cost). With all threads both decoders spend about 1.8x the processor
time they spend on one, so the same gaps hold there.

Then the inverse transforms of 8-bit blocks moved to a vector kernel
in `cutlass-simd` (same machine, same way of measuring):

| File | Threads | Before | After | dav1d |
|---|---|---|---|---|
| 4x1 tiles | 18 | 0.62 s | 0.59 s | 0.30 s |
| 4x1 tiles | 1 | 4.16 s | 3.97 s | 2.60 s |
| 8x4 tiles | 18 | 0.59 s | 0.54 s | 0.31 s |
| 8x4 tiles | 1 | 3.62 s | 3.21 s | 1.92 s |

The whole files took 7.8 s and 15.2 s at 18 threads in one run each,
against dav1d's 5.1 s and 8.5 s in the same sitting.

Both passes of a block run in 16-bit vectors, eight rows and then
eight columns to a vector, and the residual is added to sixteen samples
at a time. A one-dimensional DCT or ADST is straight-line code expanded
from its table of steps, once for each count of inputs that can be
non-zero (1, 4, 8, 16, 32): a step on entries known to be zero is no
code, and a rotation of one non-zero entry is two multiplies. The
identity transforms scale in 32 bits as outputs are rounded, and a
block with two of them touches only the samples under its
coefficients. A block with one row of coefficients, or only its first
coefficient, adds one row of residuals to every row. In isolation:

| Block, coefficients | Before | After |
|---|---|---|
| 64x64 DCT, 2x2 | 1.93 us | 0.66 us |
| 64x64 DCT, 32x32 | 3.43 us | 1.67 us |
| 32x32 DCT, 2x2 | 0.65 us | 0.18 us |
| 32x32 DCT, 32x32 | 1.20 us | 0.61 us |
| 16x16 DCT, 16x16 | 317 ns | 153 ns |
| 8x8 DCT, 8x8 | 96 ns | 50 ns |
| 8x8 ADST, 8x8 | 98 ns | 60 ns |
| 16x16 identity, 4x4 | 92 ns | 23 ns |
| 64x64, first coefficient only | 268 ns | 68 ns |

On one thread of the 8x4 file the inverse transforms now take about
0.29 s where they took 0.72 s, against dav1d's 0.20 s. 4x4 blocks with
a DCT or an ADST stay on the older code, which is the faster one for
them, as do 10-bit and 12-bit streams and builds without NEON. What is
left of the gap on these files is deblocking (its edge decisions
first), inter prediction, and reading blocks and coefficients.

A turn of work on one thread, counted in instructions retired and
processor cycles (`/usr/bin/time -l`, best of 3 runs for cycles, on a
machine busy with other work; dav1d 1.5.3 with `--threads 1 --muxer
null`), 240 frames of each file:

| File | Before | After | dav1d |
|---|---|---|---|
| 8x4 tiles, instructions | 55.9G | 47.4G | 27.1G |
| 8x4 tiles, cycles | 13.3G | 11.8G | 8.1G |
| 4x1 tiles, instructions | 74.8G | 71.8G | 45.8G |
| 4x1 tiles, cycles | 16.8G | 15.7G | 10.9G |

So one thread takes about 1.45x dav1d's cycles on both files. What
changed: 8-bit compound masks blend a row at a time, weights first
(10.4% of the 8x4 file's instructions); DC, vertical and horizontal
intra predictions fill their rows with fixed-size stores instead of a
call to memset or memcpy per row (1.7%); the inverse transform's values
between its passes are sized by the rows the column transforms read, so
a large block with few coefficients clears a few hundred bytes and not
4 KB (1.2%); vertical deblocking edges copy a fixed count of lines to
and from the kernel, so the copies unroll (1.8%); and the mode info
kept for every 4x4 is 16 bytes, one vector store, instead of 28 (the
loop filter deltas in an array of their own for the frames that have
them, the small fields packed): 0.9% of the instructions, and 3 to 5%
of the cycles, as less memory is written.
Tried and dropped: deblocking kernel entries that take a filter and
limits for each group of four lines (as the edges of four 4x4 units lie
in a picture), and edge decisions made sixteen units at a time in
vectors from per-unit level arrays. The first took 12% more
instructions in all (marshaling the groups cost more per call than the
separate runs it replaced); with the second the edge scan itself got
cheaper, but filling the level arrays cost as much, and the time stayed
the same. What is left of the gap: deblocking is about a fifth of the
time, most of it the filters themselves (the size-16 wide filter alone
is 6%); inter prediction's 8-tap filter is 15%; and reading blocks and
coefficients, and the mode info, follow.

A further turn, measured the same way (cycles best of 3; at 18 threads
best of 5, wall time and processor time):

| File | Threads | Before | After | dav1d |
|---|---|---|---|---|
| 8x4 tiles, instructions | 1 | 47.4G | 45.6G | 27.1G |
| 8x4 tiles, cycles | 1 | 11.8G | 10.2G | 8.1G |
| 4x1 tiles, instructions | 1 | 71.8G | 66.3G | 45.8G |
| 4x1 tiles, cycles | 1 | 15.7G | 13.5G | 11.0G |
| 8x4 tiles | 18 | 0.54 s | 0.46 s (4.3 s CPU) | 0.29 s (3.5 s) |
| 4x1 tiles | 18 | 0.59 s | 0.52 s (6.1 s CPU) | 0.28 s (4.1 s) |

One thread now takes 1.25x dav1d's cycles (1.45x before). What changed,
among smaller steps: 8-bit compound predictions blend with their masks
eight samples at a time in a `cutlass-simd` kernel on 16-bit lanes, and
the difference-weighted mask of 8-bit predictions is worked out on
16-bit lanes; a picture's rows are 64 samples longer when they would be
a multiple of 1024 samples (4K luma rows were 4096), so that the rows
of a block fall in different cache sets (about 4% fewer cycles); the
pool of pictures looks pictures up by that row length (it found no 4K
picture until then). Intra prediction at the top of a superblock row
reads a copy of the row above's last lines as reconstruction left them,
so a row's filters can run as soon as it is reconstructed. With threads,
a frame that no frame references, or that has no CDEF and no loop
restoration (its rows are then final 8 rows behind the last one
deblocked), is filtered in place behind its reconstruction on the same
thread, in the final picture: no second picture, no copy of every row,
no filter thread (22% less processor time on the 8x4 file, wall time
the same).
Tried and dropped: reconstructing the tile columns of a frame on threads
of their own, each into a picture of its own with the filters gathering
the rows (16% more processor time and slower: the extra pictures and
shells, and the frames waited longer for their references); filtering
every frame in place (18% less processor time, but 30% more wall time:
frames that others wait for are then final later, with recon and
filters of a row one after the other). Where the wall time goes at 18
threads (the `prof` timeline): 8 frames are always in flight, the most
allowed, and each new frame starts as one ends, about 1.8 ms apart on
the 4x1 file; a frame's parse takes 1.34 ms from receiving its primary
reference's CDFs to passing its own on (one tile's parse, as the
context tile is parsed first), against dav1d's whole frame interval of
1.2 ms. Allowing 12 frames in flight takes the 8x4 file to 0.41 s but
needs 0.3 GB more (2.0 GB at the peak against dav1d's 0.4 GB), so the
per-frame memory comes first: the frame states of the tile parsers
(full size, about 14 MB each at 4K), the published bands (with their
overlap, about twice the rows) and the second picture.

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
in and the row above (a scaled reference is waited for whole; a block
that still finds rows missing, as a warped block may, stops until they
are published), and reads the finished picture in place once it is done.
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

On two 4K 8-bit clips of 240 frames (one with CDEF, one deblocking only)
on a 6 performance + 12 efficiency core machine, single threaded the
decoder takes 12.7G and 9.6G cycles against libdav1d's 11.0G and 8.2G
(through FFmpeg). With 18 threads it takes 0.41 s and 0.36 s against
0.31 s and 0.31 s; both keep about 13 cores busy, so the gap is processor
time (5.3 s against 4.1 s), and up to 6 threads it is faster (0.49 s
against 0.61 s at 6). Of the threaded processor time about 9% is copies:
the bands published for frames still reading a reference (each holds
80 rows above the 64 it adds, so a block's rows lie in one band) and the
reconstructed rows handed to the filter thread. Tried and dropped there:
bands with less overlap (blocks across a band edge then gather their own
copy of the reference rows, which costs more than the overlap; warped
blocks read across bands by row made no difference), publishing fewer,
taller bands (less processor time, more wall time: later frames wait
longer), saving the motion field bands per superblock row on one thread
(the reads were not cache misses), a branch free motion field band
selection and CDEF direction choice (fewer instructions, no fewer
cycles), and taking the vertical deblocking edges off the filter thread
(skipping them altogether moved the 18-thread time by only as much as
their processor time).

Later, the per-4x4 maps (mode info, transform types and sizes, the loop
filter's transform sizes and block info) go round a ring of 64 rows on
one thread and in each tile parser, rather than through arrays the size
of the frame that the cache cannot hold, and the tile parsers read the
motion field where its thread puts it. Single threaded that takes 12.0G
and 9.1G cycles (libdav1d 11.0G and 8.2G); with 18 threads 0.37 s and
0.32 s on a quiet machine (user 5.0 s and 3.9 s), and with 8 threads 0.41 s
and 0.40 s against libdav1d's 0.52 s and 0.44 s. What is left single
threaded is spread over the blocks' bookkeeping (mode info, BlockDecoded
flags, the per-4x4 maps), about 0.5G against libdav1d's block decode;
the filters, motion compensation, the transforms and coefficient reading
are within a few percent of it. Threaded, about 10% of the processor time
is still copies (the published bands above, the rows handed to the filter
thread, and the loop filter transform sizes passed from the parsers).
Tried and dropped: walking the saved motion per block (fewer
instructions, but the short runs cost more than they saved), branch free
or vectorised motion saving (its time is waiting on memory), and copying
the samples of emulated edges in fixed chunks (the compiler turns the
chunk loops back into library calls).

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
