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
- Every NEON kernel in `cutlass-simd` has a scalar twin written from the
  specification and a test comparing the two on random and edge inputs
  (`cargo test --release -p cutlass-simd`).
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

**Decode, 4K, `Av1Core`.** Two 4K 8-bit clips of 240 frames, `av1bench`
against `ffmpeg -c:v libdav1d -threads N -f null`, whole-process wall time,
median of 11 to 21 runs taken in turn (13 November 2026, commit `67cc080d` of
branch `av1`, release build with a profile from the fixtures, not the clips):

| Clip | Threads | Cutlass | FFmpeg (`libdav1d`) |
| --- | ---: | ---: | ---: |
| CDEF on | 1 | 2453.9 ms | 2583.5 ms |
| Deblocking only | 1 | 1893.8 ms | 1918.3 ms |
| CDEF on | 18 | 312.3 ms | 321.9 ms |
| Deblocking only | 18 | 284.1 ms | 333.1 ms |

The decoder is faster than libdav1d, which uses hand-written assembly, on one
thread and on 18 on both clips, with the same output.

**Decode, one thread, 8 and 10 bits.** 240 frames of 1920×1080 from an iPhone
HDR clip, written by FFmpeg with `-c:v libsvtav1 -preset 8 -crf 35` once as
`yuv420p` and once as `yuv420p10le`; `av1bench <clip> 1` against
`ffmpeg -threads 1 -c:v libdav1d -f null`, whole-process cycle counts from
`/usr/bin/time -l`, the least of 5 to 9 runs taken in turn (13 November 2026,
`cargo build --release`, no profile). Both decoders produced the same bytes.

| Clip | Cutlass | FFmpeg (`libdav1d`) |
| --- | ---: | ---: |
| 8-bit | 3.251G cycles | 3.111G cycles |
| 10-bit | 4.668G cycles | 3.699G cycles |

On one thread without a profile the decoder is 4% behind libdav1d at 8 bits
and 26% behind at 10 bits on these clips. The same day's work took the 10-bit
clip from 5.063G cycles (a motion compensation kernel for 10 and 12 bits) and
the 8-bit clip from 3.356G (a Wiener kernel).

**Decode, x86-64.** Not measured on x86-64 hardware. The same 8-bit clip
decoded by the `x86_64-apple-darwin` build under Rosetta 2 took 2.1 to 2.4 s
on one thread, where the native build took 0.87 s; Rosetta's translation is
part of that. An SSE4.1 motion compensation kernel written and tested there
was no faster than the loops it replaced, and was not kept.

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
- The vector kernels are NEON only. Other targets run the decoder's loops
  over small arrays, which the compiler vectorizes; their speed on x86-64
  hardware is not measured.
- At 10 and 12 bits the motion compensation of blocks eight wide and up has
  a kernel; CDEF runs on the 16-bit kernel or on lanes; the inverse
  transform, loop restoration, film grain and super resolution run on the
  decoder's loops. One thread is 26% behind libdav1d at 10 bits (see above).
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
