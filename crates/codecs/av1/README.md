# cutlass-av1

Native safe Rust AV1 decoder and encoder in this crate. No `unsafe`. The build does not link libaom, dav1d, SVT-AV1, or FFmpeg. FFmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg` is an oracle for tests, the example, and the bench. `STATUS.md` is the support record. This file is the measured record of the same limits.

## What this crate is

`cutlass-av1` decodes and encodes AV1 packets through the shared codec traits. `Av1Decoder` reconstructs a picture from one temporal unit. `Av1Encoder` emits a keyframe. `register_decoders` and `register_encoders` add those types to the shared factories.

A frame the decoder does not implement returns `CodecError` before a picture is published. A failed packet leaves the reference slots as they were. `drain` is empty. `reset` drops the slots and keeps the sequence.

## Specification

The normative document is the AOMedia AV1 Bitstream & Decoding Process Specification, version 1.0.0 with Errata 1, 8 January 2019, tag `v1.0.0-errata1`. Syntax, the decoding process, the parsing process, and the default CDF, scan, quantizer, and block-size tables in that document are the rules this crate implements. Annex A names Main profile and levels. Annex B is the length-delimited temporal-unit format.

This crate reads low-overhead OBUs with `obu_has_size_field = 1`. That is the payload an IVF frame or an ISOBMFF sample carries after the container removes its own length. AV1 in ISOBMFF (`av1C`) and AV1 in Matroska stay in the container crates.

Film grain synthesis is part of the output process in that specification. When a frame sends its own grain parameters, the noise is added to the shown picture. The reference keeps the picture from before that step.

Main profile allows 10-bit 4:2:0 and monochrome. This crate rejects both, so it is not a Main-profile-compliant decoder.

## API

Open a decoder with `CodecParams::video`, which names 8-bit 4:2:0. The frame header carries the picture size. `Decoder::push` takes one temporal unit. `AsView` splits a shown picture back into planes.

```rust
use cutlass_av1::{AsView, Av1Decoder};
use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Packet};

fn shown_size(bytes: &[u8]) -> Result<(u32, u32), CodecError> {
    let params = CodecParams::video(CodecId::Av1, 64, 64);
    let mut decoder = Av1Decoder::open(&params)?;
    let outputs = decoder.push(&Packet::video(bytes.to_vec(), 0, 1, 30, true))?;
    let picture = outputs
        .first()
        .and_then(|output| output.as_video())
        .ok_or(CodecError::InvalidApi {
            what: "no picture",
        })?;
    Ok((picture.picture.y.width(), picture.picture.y.height()))
}
```

`Av1Encoder::open` takes the same `CodecParams` shape for a 4:2:0 picture whose edges are at most 4096. `set_quality` sets the quantizer index used when `bitrate` is zero. `Encoder::push_frame` emits a temporal delimiter, a sequence header, and one keyframe of DC-predicted 4×4 and 8×8 blocks with the 2D DCT. The loop filter and CDF update stay off. A picture that repeats the previous one is a frame that shows reference slot 0.

## Support

The decoder's scope is progressive 8-bit 4:2:0, `seq_profile` 0, one operating point, and dimensions up to 4096×2304. An edge above 4096 is `Limit`.

Implemented for that scope, and matched against dav1d on the keyframe fixtures named in `STATUS.md`:

- Sequence headers, frame headers, temporal delimiters, frame OBUs, padding, and metadata OBUs that are skipped.
- Eight reference slots, `show_existing_frame`, and a hidden frame (`show_frame = 0` returns no picture).
- Intra prediction, including the recursive filter on DC luma, chroma-from-luma, and directional modes.
- Inverse DCT through 64 points, identity and 1D transforms, and an extended intra transform set on a keyframe. A keyframe may select its transform size.
- Deblocking when the luma loop-filter levels are non-zero. A frame whose two luma levels are zero stays unfiltered.
- CDEF, uniform tiles, a quantizer matrix, and a superblock quantizer delta.
- Super resolution on a keyframe: reconstruction uses the coded width, and the shown picture is the upscaled width.
- Wiener, self-guided, and switchable loop restoration on a keyframe.
- Film grain on the shown picture: luma, chroma, overlap, and chroma grain scaled from luma. The reference is stored before grain.

Within-frame CDF adaptation is implemented. The encoder leaves CDEF, restoration, super resolution, and film grain off. The decoder applies those tools when a supported keyframe sends them.

The encoder emits keyframes only. Its quality and rate tests passed in `cargo test -p cutlass-av1` on this machine. It does not emit inter frames.

## Accuracy

`cargo test -p cutlass-av1` on this machine passed 61 tests. The embedded keyframe fixtures in that suite match dav1d byte for byte on the tools listed above, including CDEF, film grain, uniform tiles, a quantizer matrix, a superblock quantizer delta, super resolution, Wiener, self-guided, and switchable restoration, the recursive intra filter, and selectable transform sizes.

`a_production_av1_keyframe_matches_dav1d_when_present` passed. The bench below also printed `pixels: match` for that same 3840×2160 keyframe, compared with FFmpeg's libdav1d before either time was taken.

`single_reference_frames_match_dav1d` matches dav1d on error-resilient frames that keep the largest transform and leave compound, warped, overlapped, reference-motion, order hint, CDEF, restoration, and film grain off. `order_hint_frames_match_dav1d` matches with order hint on, each frame repeating the eight reference hints. `film_grain_stays_off_the_reference` predicts a second frame from the picture stored before grain. A later frame of the 2160p file is outside these checks.

Hostile packets in the suite return an error and leave a stored reference usable. The 30-minute fuzz campaign was not run.

## Performance

One sample, measured with `cargo bench -p cutlass-av1` on Thu Oct 29 21:08:47 GMT 2026.

`sysctl -n machdep.cpu.brand_string` printed `Apple M5 Pro`.

`/opt/homebrew/bin/ffmpeg -version` began `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`.

The library harness ignored 61 unit tests. `benches/decode.rs` then printed:

```text
workload: 3840x2160 keyframe, 37259 bytes, from New MacBook Pro with PROLED XDR Display： The first-ever Mac with Touch (2160p, av1).mp4
machine: Apple M5 Pro, 18 cores
ffmpeg: ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers
one sample. cutlass times one Av1Decoder::push in this process. ffmpeg times one process and includes startup. bench profile, no link-time optimization. cutlass decodes tiles in parallel.
pixels: match
cutlass-av1: 0.0305 s  one push
ffmpeg libdav1d -threads 1: 0.0338 s  one process, includes startup
```

`0.0305 s` is one `Av1Decoder::push` inside the bench process, after the pixel check. `0.0338 s` is the wall time of one `ffmpeg` process with `-threads 1` and `-c:v libdav1d`, and that figure includes process startup. The bench profile has no link-time optimization. This decoder decodes its tiles in parallel.

The 720×480 30 fps gate was not re-measured here. `STATUS.md` records that gate as a miss. Peak resident memory was not measured. The decoder keeps eight reference pictures plus the frame being decoded.

## Example

`examples/roundtrip.rs` asks ffmpeg to write one 64×64 AV1 keyframe, decodes that keyframe with `Av1Decoder`, and compares the picture to ffmpeg's libdav1d decode of the same file.

```text
cargo run --release -p cutlass-av1 --example roundtrip
```

On this machine that printed:

```text
64x64 match
```

If ffmpeg is missing, the example prints `skip: ffmpeg is not installed` and exits 0. The bench uses the same line.

## Gaps

Refused with a typed error before a picture is published:

- 10-bit, 12-bit, monochrome, 4:2:2, and 4:4:4.
- A profile other than 0, a second operating point, and the decoder model.
- 128×128 superblocks, large-scale tile lists, a tile group outside a frame, and non-uniform tile spacing.
- An OBU without a size field, and a reserved OBU type.
- A render size that differs from the coded size.
- Screen-content tools, palette, and intra block copy.
- Lossless (`base_q_idx` 0).
- Segmentation, a per-superblock loop-filter delta, and a loop-filter delta update.
- Film grain copied from another frame. A show-existing frame does not re-apply grain.
- Compound prediction, warped motion, overlapped motion, reference-frame motion vectors, short reference signaling, and global motion.
- CDF state carried from another frame, an inter frame that selects its transform size, a scaled reference, and interintra on an inter frame.

Super resolution on an inter frame is not proven.

The encoder does not emit inter frames, CDEF, restoration, super resolution, or film grain. Its blocks stay DC-predicted 4×4 and 8×8.

Still open, as `STATUS.md` records: the 30-minute packet fuzz, the 720×480 realtime gate, and a peak-memory measurement. Containers are outside this crate.
