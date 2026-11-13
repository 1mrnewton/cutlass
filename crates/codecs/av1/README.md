# cutlass-av1

An AV1 decoder and encoder. The crate is safe Rust (`#![forbid(unsafe_code)]`),
depends only on `cutlass-decode`, `cutlass-encode` and `cutlass-simd`, and links
no C library. Its sample loops run on the NEON kernels of `cutlass-simd` on
aarch64, and on their scalar twins elsewhere.
The desktop app registers it for AV1 decode and encode. On macOS, when hardware
decode or encode is on in Settings, the VideoToolbox opener is asked first and
this crate opens when that one declines. The Windows platform crate registers
no AV1 opener and Linux has no platform codecs, so there it is the only one.

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

Pictures come out as planar bytes in a `PictureOutput`. Samples above 8 bits
are little-endian `u16`. Monochrome comes out as 4:2:0 with neutral chroma, an
identity matrix on 4:4:4 as GBR, and an odd 4:2:0 size is padded to even by
repeating the last column and row. `PictureInfo` carries the color primaries,
transfer and matrix (named, and as the sequence header's CICP codes), the
chroma sample position, the render size, the sequence timing info,
`num_units_in_decoding_tick` and the `seq_level_idx` of the decoded point.

Refused by the decoder:

- A large-scale tile list (`OBU_TILE_LIST`), or an operating point past the
  sequence's last one: `CodecError::Unsupported`.
- A sequence wider or taller than 16384, or a frame with more luma samples
  than `Av1Core::max_pixels` (35,651,584 by default): `CodecError::Limit`.
- An empty packet: `CodecError::InvalidApi`.

A unit that fails leaves the reference frames as they were before the call.

Encoder (`Av1Encoder`, over `enc::encoder::Av1Enc`):

- Input: `Yuv420p`, `Yuv422p`, `Yuv444p` and `Gbrp` at 8, 10 and 12 bits. GBR
  is written with an identity matrix, sRGB transfer and full range. Another
  pixel format is `Unsupported`; a side of 0 or above 16384 is `Limit`.
  `Av1Enc` also codes monochrome (`EncConfig::mono`).
- Constant quantizer index (`set_quality`), or a bitrate target when
  `CodecParams::bitrate` is not zero.
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
- Options on `EncConfig` only: key frame distance (120 by default), Wiener
  loop restoration, global motion, alt-ref temporal filtering, explicit tiles.

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

- `tests/fixtures.rs` decodes every stream in `fixtures/` (177 files, 86
  streams; each `.ivf` is one stream, and `name_N.obu` files are one stream
  with a packet per file) through `Av1Decoder`. Each shown frame of 85 streams
  must match the libdav1d digest stored in `tests/libdav1d-digests.txt`. The
  last stream, `wide.ivf` (16448x64), must be refused with `Limit`. This needs
  no FFmpeg, so it runs in CI.
- `tests/metadata.rs` checks the published color, timing, render size and
  operating point. `tests/obu.rs` checks OBU framing and the tile list
  refusal.
- Every NEON kernel in `cutlass-simd` has a scalar twin written from the
  specification and a test comparing the two on random and edge inputs
  (`cargo test --release -p cutlass-simd`).
- Unit tests compare the transform, CDEF and arithmetic coder kernels with
  direct transcriptions of the specification, round-trip the sequence and
  frame headers through the writer and the parser, and check that the encoder
  writes the same bytes with one thread and with several and that its
  reconstruction is the decoder's.
- In a separate measurement, the decoder's output for a 600-frame 1080p stream
  written by SVT-AV1 was byte-identical to that of FFmpeg's libdav1d.

Needs FFmpeg, and is not run by `cargo test`:

- The digests are rewritten from a local `ffmpeg` with libdav1d by
  `cargo test --release -p cutlass-av1 --test fixtures -- --ignored write_libdav1d_digests`.
- `tools/conf.py` with `tools/cases.py` writes 46 streams with FFmpeg's
  libaom, SVT-AV1 and rav1e encoders and compares the `av1dec` example with
  libdav1d; `tools/checkall.sh` repeats the comparison. `tools/encmatrix.sh`
  encodes 13 cases with `av1enc` and checks that libdav1d decodes each
  to the encoder's reconstruction. `tools/rdcmp.sh`, `tools/kbd.sh` and
  `tools/bd.py` compare rate and distortion with FFmpeg's encoders. None of
  these scripts were run for this description.
- `examples/av1fuzz.rs` corrupts IVF streams and checks that decoding does not
  panic. It is run by hand.
- `examples/roundtrip.rs` and `benches/decode.rs` print `skip` without `ffmpeg`.

## Speed against FFmpeg

**Decode, av1.** Measured on 13 November 2026 with the AV1 work of branch
`av1` merged into `development` (`aac363e8`), on an Apple M5 Pro (18 cores,
48 GB) running macOS 27.0.1 with other work running (load average about 6),
against FFmpeg 8.1.1. Built with `cargo build --release` and rustc 1.99.1.
Input: 373 frames of 1920×1080 at 30 frames a second (8.1 MB), written by
FFmpeg with `-c:v libsvtav1 -preset 8 -crf 35`.

|  | Cutlass | FFmpeg (`libdav1d`) |
| --- | ---: | ---: |
| Wall time | 0.67 s | 0.52 s |
| Speed | 557 frames/s | 717 frames/s |
| CPU time | 2.40 s | 1.61 s |
| Peak memory | 332 MB | 110 MB |
| Threads allowed | 16 | 16 |

Cutlass is **1.3× slower** than FFmpeg here. Both decoders produced exactly the
same samples. The gap is outside the decoding itself: the `av1bench` example,
which decodes through `Av1Core` and keeps the pictures as they are, took 307 ms
on the same file with 16 threads against 537 ms for the `ffmpeg` command with
libdav1d (median of 9 runs taken in turn; that build used a profile from
`scripts/pgo_profile.py`, worth 3 to 5%). The likely cost is `Av1Decoder`
copying each shown picture into packed bytes one sample at a time (1.16 GB
here); that has not been profiled yet.

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
- `Av1Decoder` copies every shown picture into packed bytes one sample at a
  time.
- The vector kernels are NEON only; other targets run the scalar twins.

Encoder:

- Never chooses intra block copy, wedge or difference-weighted compound,
  inter-intra, super resolution, segmentation, 128x128 superblocks or film
  grain. The dual filter and loop restoration are off by default.
- Writes `seq_level_idx` 31 for every stream instead of the level the stream
  fits.
- `Av1Encoder` has no setting for the key frame distance, monochrome, loop
  restoration, global motion or the temporal filter; those exist on
  `EncConfig` only. A bitrate target takes the frame rate from the first
  frame's duration and falls back to 30.

## Specifications

- AV1 Bitstream & Decoding Process Specification, version 1.0.0 with errata 1.
- ISO/IEC 23091-4 (ITU-T H.273) for the color codes.
- AV1 Codec ISO Media File Format Binding, for the `av1C` record.
