# MPEG-2 video support record

Checkout: branch `mpeg2`, worktree `/Users/mrnewton/dev/cutlass/cutlass-mpeg2`, started from `development` at `e025f6ce`. The repository root is `/Users/mrnewton/dev/cutlass/cutlass-main` (the common git directory); the worktree is its sibling. Native safe Rust in `cutlass-mpeg2` with `#![forbid(unsafe_code)]`. No FFmpeg, GStreamer, or VideoToolbox code is in the crate or its dependency graph. Those programs are test oracles only.

Request: MPEG-2 video end to end in `crates/codecs/mpeg2`: a decoder, an encoder, and integration with playback and export.

## Rules and skills that were followed

- Every file in `.cursor/rules` of this checkout: `codec-shipping`, `commit-often`, `container-shipping`, `development-worktree`, `local-media`, `merge-pull-request`, the product-name ban, `perf`, `platforms`, `pull-request`, `rust-standards`, `source-of-truth`, `tests`, `think-like-a-pro`, `worktree-names`.
- Skill `ship-codec` with `references/implementation.md`, `verification.md`, `deliverables.md`, and `video-image.md`. This file is that skill's scope record, capability matrix, fixture manifest, and validation report.
- Skills `think-like-a-pro` and `cutlass-worktree`. There is no `AGENTS.md` in this tree.
- The workspace docs in `cutlass-docs`: `rust-workspace.md` names `mpeg2` as the MPEG-2 video crate that implements the `decode` and `encode` traits.

## Specification and sources

- ITU-T H.262 (07/1995), the text published as ISO/IEC 13818-2:1996. It is the edition ITU still serves without a login. Clause 6 for syntax and semantics, clause 7 for decoding, Annex A for the IDCT accuracy rule (IEEE 1180-1990), Annex B for every VLC table, Annex C for the VBV, clause 8 for profiles and levels.
- The 4:2:2 profile codes (`profile_and_level_indication` 0x85 and 0x82) and 11-bit `intra_dc_precision` come from the later amendments; they are taken from the amendment as cited by FFmpeg and GStreamer, since the 2012 edition needs a login.
- ISO/IEC 11172-2 behaviour (MPEG-1 video) is required of every MPEG-2 Simple, Main and High profile decoder by H.262 clause 8.1. Its differences (no extensions, odd-ified inverse quantisation, 8-bit escape, full-pel vectors, D-pictures) are taken from FFmpeg's `mpeg12dec.c` and the 11172-2 notes in H.262 Annex D.
- FFmpeg (`../pub-packages/ffmpeg`, commit `31ba8267`): `libavcodec/mpeg12dec.c`, `mpeg12.c`, `mpeg12data.c`, `mpeg12enc.c`, `mpegvideo_motion.c`, `mpegvideo_enc.c`, `ratecontrol.c`, `mpegvideo_parser.c`; `libavformat/isom.c` and `isom_tags.c` for the MP4 object types and QuickTime fourccs.
- GStreamer (`../pub-packages/gstreamer`): `gst-plugins-bad/gst-libs/gst/codecs/gstmpeg2decoder.c` and `gstmpeg2picture.c` for the reference and output queue and the open-GOP rule; `codecparsers/gstmpegvideoparser.c` for header parsing. GStreamer's own encoder (`ext/mpeg2enc`) wraps mjpegtools, which is not installed here.
- Conformance material: the ISO/IEC 13818-4 streams that FFmpeg's FATE suite carries (`sony-ct3.bs`, `tcela-6.bits`, `hhi_burst_422_short.bits`), plus FATE's field-coded, DVD and XDCAM streams. Fetched once into `/tmp/cutlass-mpeg2/fate`; not checked in.

## Existing code before this branch

- `crates/codecs/mpeg2` was an empty crate (one `#![forbid(unsafe_code)]` line).
- `cutlass-codec` already names `CodecId::Mpeg2`. `cutlass-isobmff` maps the `mp2v` sample entry to it in both directions, and maps `mp4v` to MPEG-4 Part 2 whatever the ESDS object type says.
- The Matroska, MPEG-TS and AVI container crates are empty. There is no program-stream crate.
- Playback opens `cutlass_decode::Decoder` through `DecodeFactory::open` and renders packed 4:2:0 only. Export opens `EncodeFactory::open`; a 4:2:0 `VideoEncoder` registered with `register_video` is wrapped by the RGBA to BT.709 adapter in `cutlass-encode`.
- On macOS the platform crate registers a VideoToolbox decoder for `CodecId::Mpeg2` ahead of the Rust codecs.

## Required target

### Decoder: the non-scalable syntax of H.262, and MPEG-1

- Simple, Main, 4:2:2 and High profile single-layer streams at every level. The profile and level byte is reported, not enforced.
- Chroma 4:2:0, 4:2:2 and 4:4:4. 8-bit samples.
- I, P and B pictures. Frame pictures and field pictures, both field orders, field pairs split across packets or in one packet.
- Frame, field, 16×8 and dual-prime prediction. Frame and field DCT. Concealment vectors parsed. Skipped macroblocks in P and B, frame and field.
- Both scans, both intra VLC tables, both quantiser scale types, `intra_dc_precision` 8 to 11 bits, every quantiser matrix including the 4:2:2 chroma matrices, saturation and mismatch control.
- Sequence display extension colour and display size, aspect ratio, frame rate with the extension, `repeat_first_field`, `top_field_first`, `progressive_frame`.
- MPEG-1 streams (no sequence extension): I, P, B and D pictures, full-pel vectors, 11172-2 inverse quantisation.
- Display order through a two-reference queue. After a reset or seek, B pictures of an open GOP with no forward reference are not shown (as GStreamer marks them decode-only and FFmpeg drops them).
- Sizes up to 16383×16383. The engine opens with an 8192×8192 pixel cap, refused with `Limit`.
- An elementary-stream splitter that turns a raw `.m2v` byte stream into access units, chunk-invariant.

### Encoder: Main profile, progressive 4:2:0, the stream export needs

- 4:2:0 8-bit progressive frame pictures (`progressive_sequence = 1`), which is what export composites.
- I, P and B pictures with a configurable GOP length and B-picture count, closed GOPs, a sequence header on every GOP, scene-cut I pictures and `force_keyframe`.
- Half-pel motion search for forward, backward and bidirectional prediction, with `f_code` chosen from the search range.
- Intra, inter, skipped and not-coded decisions by cost; adaptive quantisation from spatial activity.
- Rate control: a target bitrate with a VBV model (no underflow, the buffer size signalled), a constant-rate mode that stuffs to hold the rate, and a quality default when no bitrate is asked.
- Main profile at Main or High level, chosen from size and rate. Both quantiser scale types; intra VLC table B.15; the sequence display extension with BT.709 colour.

### Out of scope, with reasons

- SNR, spatial and temporal scalable enhancement layers and data partitioning: no profile a desktop editor imports uses them. A stream with a `sequence_scalable_extension` is refused with `Unsupported`.
- Interlaced and 4:2:2 or 4:4:4 encoding: export produces progressive 4:2:0 pictures (`VideoInput::picture` is `Yuv420p`).
- Encoding above 1920×1152: that is above High level, so no conformant MPEG-2 decoder must take it. Refused with `Unsupported`.
- Opening `.ts`, `.mpg`, `.vob` and `.m2v` files in the app: those containers (MPEG-TS, program stream, raw elementary stream) have no Cutlass container crate. That is container work for the `ship-container` workflow.
- Error concealment: the decode crate's contract is that a bad unit is an error and the decoder stays as it was.

## Gates, declared before tuning

| Gate | Bound |
| --- | --- |
| IDCT accuracy | IEEE 1180-1990 procedure: every range and sign passes peak, mean-square and mean error bounds; zero in gives zero out |
| Decode independent streams (FFmpeg encodes of synthetic and real content, every coding tool the encoder can be asked for) | Same picture count, order and timestamps as FFmpeg; every picture's Y, U and V PSNR against FFmpeg ≥ 45 dB, mean ≥ 50 dB |
| Decode the 13818-4 conformance streams and FATE field, 4:2:2, DVD and XDCAM streams | Decode without error; same picture count and order as FFmpeg; PSNR bounds as above |
| Second decoder | Decoder output against VideoToolbox (through GStreamer `vtdec`) on the 4:2:0 streams: PSNR bounds as above |
| Our streams in other decoders | FFmpeg decodes every encoder test stream with `-xerror` and no error log; VideoToolbox decodes the 4:2:0 ones; both within the PSNR bounds of our reconstruction |
| Quality | PSNR-Y at the same measured rate no more than 1.0 dB below FFmpeg `mpeg2video` (`-g 12 -bf 2`, otherwise defaults) on three natural clips |
| Rate | Mean rate within 0.90×–1.10× of the target over 150 frames; the VBV model never underflows |
| Decode speed, release, one thread, Apple M5 Pro | 1920×1080 4:2:0 at ≥ 120 fps, ratio to FFmpeg recorded |
| Encode speed, release, one thread | 1920×1080 at ≥ 30 fps on the default settings |
| Memory | Decoder peak resident set at 1080p ≤ 64 MB above the process baseline; bounded on a long stream |
| Decoder fuzz | ≥ 30 minutes, seed and findings recorded, findings 0 |
| Encoder fuzz | ≥ 30 minutes, seed and findings recorded, findings 0 |

## Capability matrix

Status is updated with evidence as each row lands. "Planned" means no code yet.

| Capability | Required | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Sequence, GOP, picture headers and extensions | Yes | Parse | Write | | Planned |
| Scalable extensions | No | `Unsupported` | Not emitted | | Planned |
| 4:2:0 | Yes | Yes | Yes | | Planned |
| 4:2:2, 4:4:4 | Yes (decode) | Yes | Out of scope | | Planned |
| I, P, B frame pictures | Yes | Yes | Yes | | Planned |
| Field pictures | Yes (decode) | Yes | Out of scope | | Planned |
| Frame/field/16×8/dual-prime prediction, field DCT | Yes (decode) | Yes | Frame only | | Planned |
| Alternate scan, B.15, q_scale_type, DC precision, matrices | Yes | Yes | Subset | | Planned |
| MPEG-1 streams | Yes (decode) | Yes | No | | Planned |
| Display order, open GOP after seek | Yes | Yes | Closed GOPs | | Planned |
| Rate control and VBV | Yes | n/a | Yes | | Planned |
| Elementary-stream splitter | Yes | Yes | n/a | | Planned |
| Playback registration | Yes | Yes | n/a | | Planned |
| Export registration | Yes | n/a | Yes | | Planned |
| ISOBMFF `mp4v` with MPEG-2 object types, QuickTime MPEG-2 fourccs | Yes | Demux map | `mp2v` | | Planned |
| Lifecycle: drain, reset, malformed input, cancel | Yes | Yes | Yes | | Planned |
| Fuzz | Yes | Yes | Yes | | Planned |
