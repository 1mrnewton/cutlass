# cutlass-av1

An AV1 decoder and encoder. The crate is safe Rust (`#![forbid(unsafe_code)]`),
depends only on `cutlass-decode`, `cutlass-encode` and `cutlass-simd`, and links
no C library. Its sample loops run on the NEON kernels of `cutlass-simd` on
aarch64, and elsewhere on loops over small arrays that the compiler turns
into vector code.
The desktop app registers it for AV1 decode and encode, and its export panel
offers AV1 in an MP4. On macOS, when hardware decode or encode is on in
Settings, the VideoToolbox opener is asked first and this crate opens when that
one declines. The Windows platform crate registers no AV1 opener and Linux has
no platform codecs, so there it is the only one.

## What it supports

Decoder (`Av1Decoder`, over `dec::decoder::Av1Core`):

| Area | Accepted |
| --- | --- |
| Profiles | 0, 1 and 2: 8, 10 and 12 bits; 4:2:0, 4:2:2, 4:4:4, monochrome; odd sizes |
| Input | Low-overhead OBU temporal units (as in IVF and MP4), one per packet. An `av1C` box in `CodecParams::extra` supplies the configuration OBUs. `Av1Core::decode_annexb` takes Annex B length-delimited units |
| Frames | Key, inter, intra-only and switch frames, hidden frames, `show_existing_frame`, error resilient mode, frame ids, order hints, 64x64 and 128x128 superblocks, uniform and explicit tiles, several tile groups |
| Intra | All modes, angle deltas, edge filter, filter intra, chroma from luma, palette, intra block copy |
| Inter | Single and compound references, motion vector projection, OBMC, local and global warped motion, wedge, difference-weighted and distance-weighted compound, inter-intra, skip mode, switchable and dual filters, scaled references |
| Residual | Transforms of 4 to 64 points, lossless Walsh-Hadamard, quantizer matrices, segmentation, delta q, delta lf |
| Filters | Deblocking, CDEF, loop restoration (Wiener and self-guided), super resolution, film grain synthesis |
| Layers | Operating point 0 unless `select_operating_point` picks another; OBUs of other layers are dropped |
| Metadata | Content light level and mastering display OBUs, in the stream or in the `av1C` record's configuration OBUs, and ITU-T T.35 messages (HDR10+ is one). Scalability and timecode metadata are not read |

Pictures come out as planar bytes in a `PictureOutput`. Samples above 8 bits
are little-endian `u16`. Monochrome comes out as 4:2:0 with neutral chroma, an
identity matrix on 4:4:4 as GBR, and an odd 4:2:0 size is padded to even by
repeating the last column and row. `PictureInfo` carries the color primaries,
transfer and matrix (named, and as the sequence header's CICP codes), the
sample range, the chroma sample position, the render size, the sequence timing
info, `num_units_in_decoding_tick` and the `seq_level_idx` of the decoded
point. It also carries the content light level and the mastering display
colour volume: they hold from the OBU that gives them until a different
sequence header starts another sequence (6.7.1), through a repeated header and
a `reset`. A picture's ITU-T T.35 messages are on its `PictureOutput::t35`, in
stream order; a message before a hidden frame goes out when
`show_existing_frame` shows that frame.

The engine converts every layout the decoder publishes (4:2:0, 4:2:2, 4:4:0,
4:4:4 and GBR at 8, 10 and 12 bits) with the matrix and range the stream
names. 8-bit 4:2:0 with the BT.709, BT.601 or BT.2020 matrix, studio swing or
full range, is converted by the GPU layer shader; the other layouts are
converted on the CPU, 17 ms for a 1080p picture. An AVIF still opens through
the ISOBMFF reader and decodes here.

Refused by the decoder:

- A large-scale tile list (`OBU_TILE_LIST`), or an operating point past the
  sequence's last one: `CodecError::Unsupported`.
- A sequence wider or taller than 16384, or a frame with more luma samples
  than `Av1Decoder::set_max_pixels` allows (35,651,584 by default):
  `CodecError::Limit`.
- An empty packet: `CodecError::InvalidApi`.
- A light level, mastering display or T.35 OBU cut short:
  `CodecError::Truncated`.

A unit that fails leaves the reference frames as they were before the call.

Encoder (`Av1Encoder`, over `enc::encoder::Av1Enc`):

- Input: `Yuv420p`, `Yuv422p`, `Yuv444p` and `Gbrp` at 8, 10 and 12 bits, and
  one grey plane (`Gray8`, `Gray10`), coded as monochrome. Another pixel
  format is `Unsupported`; a side of 0 or above 16384 is `Limit`.
- `Av1VideoEncoder` is the encoder export opens: the shared adapter turns the
  RGBA frames export pushes into 8-bit 4:2:0 in BT.709 studio swing, and the
  stream is tagged so. `register_encoders` registers both.
- Colour: the primaries, transfer, matrix and range of `CodecParams` go into
  the sequence header. GBR is the identity matrix (sRGB and full range when
  unnamed); the identity matrix on other planes is `Unsupported`.
- The `av1C` record (marker, profile, level, tier, depth, chroma layout and the
  sequence header OBU) rides on the first packet as `SideData::Config` and is
  `Av1Encoder::config_record`. It is byte for byte the record FFmpeg writes for
  the same stream.
- The sequence header and the record name the lowest level of Annex A.3 whose
  picture size, edges, display and decode rates, tile counts and (with a
  bitrate target) bitrate hold, with the high tier where the main tier's rate
  is too low. A constant-quantizer stream is not checked against the level's
  bitrate.
- Constant quantizer index (`set_quality`), or a bitrate target when
  `CodecParams::bitrate` is not zero. With `CodecParams::constant_rate` each
  second gets its own budget, so a second that spends less leaves nothing for
  a later one.
- Key frames every 120 pictures (`set_keyint`), on request
  (`force_keyframe`), and at a scene cut (`set_scene_cut`, on by default): a
  picture that differs from the one before all at once, no closer to the key
  frame before it than a mini-GOP or a tenth of the key frame distance.
- Hierarchical mini-GOPs behind a hidden alt-ref (8 frames by default, 1 to 32
  through `set_gop`), shown later with `show_existing_frame`.
- Partition search up to 64x64; intra modes with angle deltas, filter intra,
  chroma from luma, and palette on screen content; motion search, compound
  prediction, temporal motion vector prediction, switchable filters, local
  warped motion, OBMC and skip mode; transform type and size chosen by rate
  and distortion; a per-frame search of the deblocking level and CDEF
  strengths.
- Tile and superblock-row threads (`set_threads`). Frames 1280 wide and up are
  split into tile columns when there are threads for them.
- Off by default, on `Av1Encoder` as settings: Wiener loop restoration
  (`set_loop_restoration`), global motion (`set_global_motion`), alt-ref
  temporal filtering of every mini-GOP (`set_temporal_filter`), explicit
  tiles (`set_tiles`).

In a file: `cutlass-isobmff` writes an AV1 track as an `av01` sample entry with
the `av1C` record and lists the `av01` brand, and `cutlass-matroska` writes
`V_AV1` with the record as CodecPrivate. Both store each temporal unit without
its temporal delimiter.

The encoder codes each frame through the decoder's own tile syntax code with a
writing arithmetic coder, and an internal `Av1Core` holds the references.

## Using it

`register_decoders` and `register_encoders` add the openers to a
`DecodeFactory` and an `EncodeFactory`. Directly:

```rust
use cutlass_av1::{AsView, Av1Decoder, Av1Encoder};
use cutlass_decode::{CodecId, CodecParams, Decoder};
use cutlass_encode::Encoder;

let params = CodecParams::video(CodecId::Av1, 352, 288);
let mut enc = Av1Encoder::open(&params)?;
enc.set_quality(120); // base q index; or set a bitrate in CodecParams
enc.set_speed(6);     // 0 slowest and best .. 10 fastest
enc.set_gop(8);       // 1 = no reordering, one packet per picture
let mut packets = Vec::new();
// for each picture: packets.extend(enc.push_frame(&frame)?);
packets.extend(enc.drain()?);

let mut dec = Av1Decoder::open(&params)?;
for p in &packets {
    for out in dec.push(p)? { /* out.as_video() */ }
}
for out in dec.drain()? { /* the pictures still being decoded */ }
```

One packet is one temporal unit. `Av1Decoder` decodes on as many threads as
the machine has unless `set_threads` says otherwise, and a picture may come out
of a later `push` or of `drain`. With two or more threads the parse,
reconstruction and filters of a frame run in a pipeline beside other frames',
and the tiles of a frame are parsed at the same time;
`Av1Core::frames_at_once` turns that off for less memory.

Lower level: `Av1Core::decode` takes one temporal unit and returns its shown
frames; `decode_delayed` and `drain` return them late so that frames decode
side by side. `Av1Enc::encode` takes plane slices and returns finished
temporal units, and `flush` returns the rest.

Examples: `av1dec` (IVF to raw planes), `av1enc` (Y4M to IVF), `av1bench`,
`av1fuzz` and `roundtrip`. Cargo features: `stats` counts the coding tools the
decoder sees, `prof` times the threaded decoder's stages, `tune` lets
environment variables override encoder constants.

## How it is tested

```
cargo test --release -p cutlass-av1
```

- `tests/fixtures.rs` decodes every stream in `fixtures/` (178 files, 87
  streams; each `.ivf` is one stream, and `name_N.obu` files are one stream
  with a packet per file) through `Av1Decoder`. Each shown frame of 86 streams
  must match the libdav1d digest stored in `tests/libdav1d-digests.txt`. The
  last stream, `wide.ivf` (16448x64), must be refused with `Limit`. This needs
  no FFmpeg, so it runs in CI.
- `tests/metadata.rs` checks the published color, timing, render size and
  operating point, and the light level, mastering display and T.35 messages:
  `hdr10.ivf` (SVT-AV1) must publish the side data ffprobe reports for it
  through libdav1d. `tests/obu.rs` checks OBU framing and the tile list
  refusal.
- `tests/fuzz.rs` runs on every `cargo test`: 510 damaged copies of the
  fixture streams (bits flipped, bytes overwritten, packets cut, headers
  overwritten, OBUs dropped, repeated and swapped, random metadata OBUs)
  decode on one thread and on three without a panic on any thread, and after
  each one a reset decoder decodes the stream as a fresh one does; 40 random
  encodes across sizes, formats and settings decode to the pictures that went
  in. `AV1_FUZZ_SECS=240 cargo test --release -p cutlass-av1 --test fuzz --
  --ignored` is the timed campaign; on 13 November 2026 it ran 93,138 damaged
  streams and 4,332 encodes with no panic.
- The desktop app's tests (`apps/desktop/src/readers.rs`, `shell.rs`) decode
  the files of `tests/mp4/` with the Rust decoder (HDR10 10-bit, 4:4:4, 4:2:2
  at 10 bits, 12-bit, GBR, full range, a BT.601 matrix, two AVIF stills) and
  check the pictures the engine makes of them; they export RGBA through the
  Rust encoder into MP4 and WebM, read both back, and, where FFmpeg is
  installed, compare its libdav1d decode byte for byte and its ffprobe tags.
- Every NEON and AVX2 kernel in `cutlass-simd` has a scalar twin written from
  the specification and a test comparing the two on random and edge inputs
  (`cargo test --release -p cutlass-simd`). `tests/fixtures.rs` checks the
  digests twice where the kernels are chosen at run time: with them, and
  with `cutlass_simd::av1::cpu::force_scalar` turning them off.
- Unit tests compare the transform, CDEF and arithmetic coder kernels with
  direct transcriptions of the specification, round-trip the sequence and
  frame headers through the writer and the parser, and check that the encoder
  writes the same bytes with one thread and with several and that its
  reconstruction is the decoder's. Others read the encoder's colour tags,
  level, `av1C` record, key frame schedule, scene cuts and constant rate back
  from the stream.
- In a separate measurement, the decoder's output for a 600-frame 1080p stream
  written by SVT-AV1 was byte-identical to that of FFmpeg's libdav1d.

Needs FFmpeg, and is not run by `cargo test`:

- The digests are rewritten from a local `ffmpeg` with libdav1d by
  `cargo test --release -p cutlass-av1 --test fixtures -- --ignored write_libdav1d_digests`.
- `tools/conf.py` with `tools/cases.py` writes 46 streams with FFmpeg's
  libaom, SVT-AV1 and rav1e encoders and compares the `av1dec` example with
  libdav1d; `tools/checkall.sh` repeats the comparison. Neither was run for
  this description: the FFmpeg on this machine has no libaom or rav1e
  encoder. `tools/encmatrix.sh` encodes 13 cases with `av1enc` and checks that
  libdav1d decodes each to the encoder's reconstruction: 13 of 13 matched on
  13 November 2026. `tools/rdcmp.sh`, `tools/kbd.sh` and `tools/bd.py` compare
  rate and distortion with FFmpeg's encoders and were not run.
- No official conformance suite (the Argon streams) has been run.
- `examples/av1fuzz.rs` corrupts IVF streams and checks that decoding does not
  panic. It is run by hand.
- `examples/roundtrip.rs` and `benches/decode.rs` print `skip` without `ffmpeg`.

## Speed against FFmpeg

**Decode, av1.** Measured on 13 November 2026 on `development` after the AV1
work of branch `av1` was merged in and `Av1Decoder` copied pictures out a row
at a time, on an Apple M5 Pro (18 cores, 48 GB) running macOS 27.0.1 with
other work running (load average about 4), against FFmpeg 8.1.1. Built with
`cargo build --release` (no profile) and rustc 1.99.1; median of 9 runs.
Input: 373 frames of 1920×1080 at 30 frames a second (8.1 MB), written by
FFmpeg with `-c:v libsvtav1 -preset 8 -crf 35`.

|  | Cutlass | FFmpeg (`libdav1d`) |
| --- | ---: | ---: |
| Wall time | 0.31 s | 0.51 s |
| Speed | 1,203 frames/s | 731 frames/s |
| CPU time | 1.87 s | 1.59 s |
| Peak memory | 377 MB | 110 MB |
| Threads allowed | 16 | 16 |

Cutlass is **1.6× faster** than FFmpeg here. Both decoders produced exactly the
same samples. Before the row copy, `Av1Decoder` packed each shown picture one
sample at a time (1.16 GB on this file): 46.8G instructions against 28.5G
after, and 0.67 s against FFmpeg's 0.52 s in the same harness.

**Decode, `Av1Core`, five clips.** Measured on 14 November 2026 at commit
`3c8dc585`, same machine, rustc 1.100.0, against FFmpeg 8.1.1 as Homebrew
builds it. `av1bench <clip> <threads>`, built as a release is (the fixtures'
profile from `scripts/pgo_profile.py`, through `scripts/pgo.py`), against
[`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c) decoding the
same IVF file with libdav1d through libavcodec. Each side is a whole process
under `/usr/bin/time -l`, the two run one after the other. Both decoders
produced the same bytes on all five clips, with Cutlass on 1 thread and on 18.
The clips: the two 4K 8-bit clips of 240 frames measured before, an 8-bit
1920×1080 clip of 373 frames, and 240 frames of 1920×1080 from an iPhone
HDR clip, written by FFmpeg with `-c:v libsvtav1 -preset 8 -crf 35` once as
`yuv420p` and once as `yuv420p10le`.

One thread, processor cycles (median of 7 runs):

| Clip | Cutlass | FFmpeg (`libdav1d`) | Cutlass ÷ FFmpeg |
| --- | ---: | ---: | ---: |
| 4K, CDEF on | 10.694G | 11.064G | 0.97 |
| 4K, deblocking only | 8.119G | 7.953G | 1.02 |
| 1080p, 373 frames | 5.075G | 4.754G | 1.07 |
| 1080p iPhone, 8-bit | 3.018G | 2.886G | 1.05 |
| 1080p iPhone, 10-bit | 4.821G | 3.558G | 1.35 |

18 threads, wall time and processor time (median of 9 runs):

| Clip | Cutlass wall | FFmpeg wall | Cutlass CPU | FFmpeg CPU |
| --- | ---: | ---: | ---: | ---: |
| 4K, CDEF on | 0.32 s | 0.31 s | 4.58 s | 4.54 s |
| 4K, deblocking only | 0.27 s | 0.31 s | 3.73 s | 4.46 s |
| 1080p, 373 frames | 0.30 s | 0.48 s | 1.68 s | 1.70 s |
| 1080p iPhone, 8-bit | 0.11 s | 0.18 s | 1.10 s | 1.05 s |
| 1080p iPhone, 10-bit | 0.21 s | 0.27 s | 1.84 s | 1.35 s |

On one thread the decoder takes 3% fewer cycles than libdav1d, which uses
hand-written assembly, on one 4K clip, and is 2% behind on the other, 5 to 7%
behind on 8-bit 1080p and 35% behind at 10 bits. On 18 threads it finishes
sooner on four of the five clips and is level on the fifth. The threaded run
was taken with other work on the machine (load average 7 to 8),
`/usr/bin/time` reports hundredths of a second, and the FFmpeg process takes
about 16 ms to start where `av1bench` takes about 2 ms.

On 13 November, at commit `67cc080d` of branch `av1` with rustc 1.99.1 and a
profile made that day, the decoder was ahead of the `ffmpeg` command on both
4K clips on one thread (1893.8 ms against 1918.3 ms on the deblocking-only
clip). It is not ahead on that clip today.

The profile helps 8 bits and costs 10. Built with `cargo build --release` and
no profile, the 8-bit iPhone clip took 3.154G cycles (3.098G with the
profile) and the 10-bit one 4.619G (4.836G with it), medians of 5 runs the
same day, where libdav1d took 2.893G and 3.584G. The fixtures the profile is
made from decode about 44 million luma samples of 8-bit 4:2:0 and 0.1 million of
10-bit 4:2:0, which is the likely reason; that has not been tested.

**Decode, x86-64.** Measured on 24 November 2026 at commit `0cd9f876`
against `development` at `bcab8681`, on 4 virtual cores of an Intel Xeon
(family 6 model 207, Emerald Rapids; AVX2 and AVX-512; 2.10 GHz nominal)
in a KVM guest with 15 GB, Ubuntu 24.04, rustc 1.100.0,
`cargo build --release` with no profile. The other side is the `dav1d` 1.4.1
command of Ubuntu (libdav1d's hand-written assembly), which picks AVX-512 on
this CPU; `--cpumask avx2` holds it to AVX2. Each run is a whole process:
`av1bench <clip> <threads>` against
`dav1d -i <clip> --threads <n> --muxer null`. No real footage could be
fetched on the machine, so the clips are synthetic: 60 frames of a rotating
Mandelbrot zoom, then 60 of a sub-sample pan over an upscaled still with
light temporal noise (half as many frames at 4K), 8-bit 4:2:0, written by
FFmpeg 6.1.1 with `-c:v libsvtav1 -preset 6|8|10 -crf 35 -g 60` (SVT-AV1
1.7.0) and once with `-c:v libaom-av1 -cpu-used 6 -crf 35`. Both decoders
produce the same bytes on all seven, with Cutlass on 1 thread, on 4, and with
its run-time kernels forced off.

Until this commit the x86-64 build had no kernel of its own: `Before` is the
decoder's loops as the compiler vectorizes them for the SSE2 baseline. `After`
picks AVX2 kernels at run time (motion compensation, warp, inverse transform,
CDEF, deblocking and its edge masks) and runs the remaining lane loops
(compound averages, loop restoration) compiled a second time for AVX2.

The guest exposes no hardware counters, so `perf stat` cannot count there.
Instructions are counted by `valgrind --tool=callgrind` 3.22 on one thread,
where both decoders see a CPU with AVX2 and no AVX-512:

| Clip | Before | After | libdav1d | Before ÷ libdav1d | After ÷ libdav1d |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1080p, SVT-AV1 preset 6 | 12.37G | 7.79G | 4.17G | 2.97 | 1.87 |
| 1080p, SVT-AV1 preset 8 | 10.35G | 6.16G | 3.55G | 2.91 | 1.73 |
| 1080p, SVT-AV1 preset 10 | 7.36G | 4.71G | 2.95G | 2.49 | 1.60 |
| 1080p, libaom cpu-used 6 | 11.79G | 7.13G | 3.82G | 3.08 | 1.86 |
| 4K, SVT-AV1 preset 6 | 22.32G | 12.53G | 6.00G | 3.72 | 2.09 |
| 4K, SVT-AV1 preset 8 | 18.63G | 9.83G | 5.20G | 3.58 | 1.89 |
| 4K, SVT-AV1 preset 10 | 12.73G | 7.20G | 4.28G | 2.97 | 1.68 |
| 58 IVF fixtures, one pass | 7.48G | 3.62G | 1.88G | 3.97 | 1.92 |

With the kernels forced off (`CUTLASS_AV1_SCALAR=1 av1bench`) the counts are
within 0.6% of `Before` on every clip.

One thread, wall time (median of 9 runs, the sides interleaved; the machine
is shared, and single runs vary by about 10%):

| Clip | Before | After | libdav1d, AVX2 | libdav1d, AVX-512 | After ÷ AVX2 | After ÷ AVX-512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p, preset 6 | 1.29 s | 1.10 s | 0.62 s | 0.62 s | 1.78 | 1.77 |
| 1080p, preset 8 | 1.00 s | 0.84 s | 0.48 s | 0.48 s | 1.75 | 1.74 |
| 1080p, preset 10 | 0.85 s | 0.73 s | 0.45 s | 0.39 s | 1.61 | 1.87 |
| 1080p, libaom | 1.20 s | 0.98 s | 0.57 s | 0.57 s | 1.71 | 1.72 |
| 4K, preset 6 | 2.44 s | 1.94 s | 1.26 s | 1.13 s | 1.54 | 1.71 |
| 4K, preset 8 | 2.00 s | 1.54 s | 1.02 s | 1.00 s | 1.52 | 1.54 |
| 4K, preset 10 | 1.46 s | 1.22 s | 0.78 s | 0.85 s | 1.55 | 1.44 |
| 58 IVF fixtures | 0.82 s | 0.60 s | — | 0.43 s | — | 1.39 |

Four threads, wall time and (processor time), median of 9:

| Clip | Before | After | libdav1d, AVX2 | libdav1d, AVX-512 | After ÷ AVX-512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1080p, preset 6 | 0.45 s (1.46) | 0.38 s (1.20) | 0.36 s (0.83) | 0.34 s (0.82) | 1.12 |
| 1080p, preset 8 | 0.39 s (1.21) | 0.32 s (0.98) | 0.29 s (0.69) | 0.28 s (0.66) | 1.14 |
| 1080p, preset 10 | 0.33 s (0.97) | 0.28 s (0.83) | 0.28 s (0.63) | 0.28 s (0.60) | 1.03 |
| 1080p, libaom | 0.44 s (1.38) | 0.36 s (1.09) | 0.29 s (0.74) | 0.28 s (0.72) | 1.30 |
| 4K, preset 6 | 0.77 s (2.63) | 0.60 s (2.01) | 0.62 s (1.45) | 0.58 s (1.38) | 1.03 |
| 4K, preset 8 | 0.68 s (2.23) | 0.53 s (1.71) | 0.58 s (1.26) | 0.55 s (1.19) | 0.96 |
| 4K, preset 10 | 0.52 s (1.65) | 0.45 s (1.42) | 0.52 s (1.09) | 0.49 s (1.01) | 0.92 |

The instructions fell by 36 to 47% and one thread's time by 14 to 23%:
Cutlass is still **1.4 to 1.9× slower** than libdav1d on one thread here,
where it was 1.7 to 2.2× slower. On four threads it finishes within 4% of
libdav1d or ahead of it on the 4K clips and 3 to 30% behind at 1080p, on
about 40 to 50% more processor time. The fixtures are 58 short processes a
pass, so their times include each one's start.

Where the instructions of two clips are, by area (the same builds with debug
info; `other decoding` is block and mode parsing, motion vector prediction
and the control around prediction):

| Area | 1080p preset 8, before | after | 4K preset 8, before | after |
| --- | ---: | ---: | ---: | ---: |
| Motion compensation | 3.03G | 1.46G | 3.16G | 1.60G |
| CDEF | 2.01G | 0.80G | 8.57G | 3.28G |
| Deblocking | 1.29G | 0.33G | 1.09G | 0.33G |
| Inverse transform | 0.57G | 0.25G | 1.40G | 0.42G |
| Loop restoration | 0.34G | 0.22G | 0.40G | 0.25G |
| Intra prediction | 0.06G | 0.06G | 0.17G | 0.17G |
| Coefficient and symbol decoding | 1.14G | 1.14G | 1.58G | 1.58G |
| Other decoding | 1.90G | 1.90G | 2.26G | 2.21G |
| All | 10.35G | 6.16G | 18.63G | 9.83G |

What is left of the gap: symbol decoding and the other decoding, which no
kernel touches, are 3.0G of the 6.2G at 1080p, against 3.6G for the whole of
libdav1d; CDEF is still a third of the 4K clip (the byte filter 1.7G, the
rows and padding around it 0.9G, the direction search 0.6G); motion
compensation of blocks four wide, the whole-sample copies and the edge
emulation stay on the decoder's loops; intra prediction, the self-guided
filter, film grain and everything at 10 and 12 bits have no AVX2 kernel.
libdav1d's own split is not known: the packaged library has no symbols.

Tried and not kept: an AVX2 Wiener kernel on `pmaddwd` took 248M instructions
on the 1080p preset 8 clip where the decoder's 16-bit lanes compiled for AVX2
take 220M (342M before). Earlier, under Rosetta 2, an SSE4.1 motion
compensation kernel was no faster than the loops it replaced.

The Arm side of the same change (the motion compensation kernel chosen by
the CPU's dot product flag where the build does not enable it, so that
Android, Linux and Windows on Arm run it) was compiled and linted for
`aarch64-unknown-linux-gnu` and `aarch64-apple-darwin` on this machine and
not run.

**Encode, av1.** Measured on 10 November 2026 at commit `ed0743d2`, same machine, FFmpeg 8.1.1, rustc 1.99.1. Input: 300 raw frames of 1920×1080 at 30 frames a second, asked for 4 Mbit/s.

|  | Cutlass | FFmpeg (`libsvtav1 preset 10`) | FFmpeg (`libsvtav1 preset 6`) |
| --- | ---: | ---: | ---: |
| Wall time | 11.05 s | 0.71 s | 3.02 s |
| Speed | 27 frames/s | 423 frames/s | 99 frames/s |
| CPU time | 94.41 s | 7.44 s | 38.37 s |
| Peak memory | 606 MB | 1252 MB | 1385 MB |
| Output size | 4.6 MB | 5.0 MB | 4.9 MB |
| Quality kept (PSNR) | 36.59 dB | 36.48 dB | 38.48 dB |
| Threads allowed | 16 | 16 | 16 |

Cutlass is **15.6× slower** than `libsvtav1 preset 10` here. Cutlass is **3.7× slower** than `libsvtav1 preset 6` here. Quality is the PSNR between the source pictures and what each encoder's output decodes to; higher is closer to the source. The encoders do not make the same choices, so read speed, size and quality together.

How it was measured: both sides get the same input from disk, do all of the work, and throw the result away. Each side is a whole process timed with `/usr/bin/time -l`, start-up included, run five times in turn; the tables show the median run. FFmpeg is given as many threads as the Rust code used. The Rust side is [`tools/media-bench`](../../../tools/media-bench), which registers only the Rust codecs. For decoding and packet reading the FFmpeg side is [`scripts/ffmpeg_decode.c`](../../../scripts/ffmpeg_decode.c), which calls libavformat and libavcodec directly; for encoding it is the `ffmpeg` command. The pictures come from a 4K street scene scaled to 1920×1080, and the sound from a music recording.

To run it again: `cargo build --release -p media-bench`, then `python3 scripts/bench_media.py --video <clip> --audio <song> --work <folder>`.

## Limits and gaps

Decoder:

- No large-scale tile decoding.
- A frame with super resolution is not pipelined: it is parsed and
  reconstructed on one thread.
- At most 16 frames decode at once.
- The decoder model info is published but no decoder model is run.
- Scalability and timecode metadata OBUs are not read.
- `Av1Decoder` copies every shown picture into packed bytes, a row at a time,
  on the thread that calls `push`.
- The vector kernels are NEON on aarch64 and AVX2 on x86-64, the AVX2 ones
  chosen at run time. An x86-64 CPU without AVX2, and any other target, runs
  the decoder's loops over small arrays, which the compiler vectorizes. On
  aarch64 the motion compensation kernels need the dot product extension
  (every Armv8.4 core, and most since the Cortex-A75 and A55); without it
  those blocks run the loops too.
- On x86-64 the 8-bit kernels are motion compensation of blocks eight wide
  and up, warp, the inverse transform, CDEF, deblocking and its edge masks.
  Intra prediction, loop restoration, the compound averages and everything
  at 10 and 12 bits run the decoder's loops, compiled for AVX2 where they
  are lanes. One thread is 1.4 to 1.9× slower than libdav1d there (see
  above).
- At 10 and 12 bits the motion compensation of blocks eight wide and up has
  a kernel; CDEF runs on the 16-bit kernel or on lanes; the inverse
  transform, loop restoration, film grain and super resolution run on the
  decoder's loops. One thread is 29% behind libdav1d at 10 bits without a
  profile and 35% behind with the release's (see above).
- Film grain and super resolution have no vector kernel at any depth; the
  Wiener filter has one at 8 bits, the self-guided filter none.

Encoder:

- Never chooses intra block copy, wedge or difference-weighted compound,
  inter-intra, super resolution, segmentation, 128x128 superblocks or film
  grain. The dual filter and loop restoration are off by default.
- Rate control is one pass: the quantizer follows what each key frame and
  mini-GOP spent, with no buffer model and no peak limit. A sudden hard
  passage overshoots before it settles: a second of changing texture after a
  still second spent 689 kbit at a constant 400 kbit/s.
- A bitrate target takes the frame rate from the first frame's duration and
  falls back to 30.
- Writes no timing info and no HDR metadata OBUs: the light level, mastering
  display and T.35 messages a decoder reads have no way in.

## Specifications

- AV1 Bitstream & Decoding Process Specification, version 1.0.0 with errata 1.
- ISO/IEC 23091-4 (ITU-T H.273) for the color codes.
- AV1 Codec ISO Media File Format Binding, for the `av1C` record.
