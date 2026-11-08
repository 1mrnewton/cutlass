# MPEG-2 video support record

Checkout: branch `mpeg2`, worktree `/Users/mrnewton/dev/cutlass/cutlass-mpeg2`, started from `development` at `e025f6ce`. Head when this record was written: `2d925e2c`. Native safe Rust in `cutlass-mpeg2` with `#![forbid(unsafe_code)]`. No FFmpeg, GStreamer, or VideoToolbox code is linked. Those programs are test oracles only.

Request: MPEG-2 video end to end in `crates/codecs/mpeg2`: a decoder, an encoder, and integration with playback and export.

## Rules and skills

- Every file in `.cursor/rules` of this checkout. There is no `AGENTS.md` in this tree.
- Skill `ship-codec` with `references/implementation.md`, `verification.md`, `deliverables.md`, and `video-image.md`. This file is that skill's scope record, capability matrix, fixture manifest, and validation report.
- Skills `think-like-a-pro` and `cutlass-worktree`.
- `cutlass-docs/rust-workspace.md` names `mpeg2` as the crate that implements the `decode` and `encode` traits.

## Specification and sources

- ITU-T H.262 (07/1995), the text published as ISO/IEC 13818-2:1996. Clause 6 for syntax, clause 7 for decoding, Annex A for the IDCT (IEEE 1180-1990), Annex B for the VLC tables, Annex C for the VBV, clause 8 for profiles and levels.
- The 4:2:2 profile codes (`profile_and_level_indication` 0x82 and 0x85) come from the later amendments, as cited by FFmpeg, because the 2012 edition needs a login.
- ISO/IEC 11172-2 behaviour is required of every Simple, Main, and High profile decoder by H.262 clause 8.1.
- FFmpeg at `../pub-packages/ffmpeg`, commit `31ba826`. The installed oracle is FFmpeg 8.1.1. Files read: `libavcodec/mpeg12dec.c`, `mpeg12.c`, `mpeg12data.c`, `mpeg12enc.c`, `mpegvideo_dec.c`, `mpegvideo_enc.c`, `mpegvideo_motion.c`, `ratecontrol.c`, `mpegvideo_parser.c`; `libavformat/isom.c` and `isom_tags.c`.
- GStreamer at `../pub-packages/gstreamer`: `gstmpeg2decoder.c`, `gstmpeg2picture.c`, `gstmpegvideoparser.c`. GStreamer's encoder wraps mjpegtools, which is not installed here, so it was not an encode oracle.
- Conformance bytes from the FFmpeg FATE suite, fetched once into `/tmp/cutlass-mpeg2/fate` and not checked in. The small synthetic streams under `tests/fixtures` are checked in.

## What was already here

- `crates/codecs/mpeg2` was an empty crate.
- `CodecId::Mpeg2` already existed. ISOBMFF mapped only the `mp2v` sample entry to it, and mapped every `mp4v` entry to MPEG-4 Part 2.
- Matroska, MPEG-TS, and AVI are empty crates. There is no program-stream crate.
- Playback opens `DecodeFactory`. Export opens `EncodeFactory`, and a 4:2:0 `VideoEncoder` is fed by the RGBA to BT.709 adapter in `cutlass-encode`.

## Decoder

Simple, Main, 4:2:2, and High profile, one layer, 8-bit. The profile byte is reported, not enforced. Chroma 4:2:0, 4:2:2, and 4:4:4. I, P, B, and (MPEG-1) D pictures. Frame pictures and field pictures, both parities, a pair in one packet or split across packets. Frame, field, 16×8, and dual-prime prediction. Frame and field DCT. Both scans, both intra VLC tables, both quantiser scales, `intra_dc_precision` 8 to 11 bits, and the quantiser matrices including the chroma matrices. Sequence display colour, aspect, frame rate with the extension, `repeat_first_field`, `top_field_first`, `progressive_frame`.

Display order uses two references. After reset, B pictures of an open GOP with no forward reference are not shown. A bad unit is an error and the decoder stays as it was. There is no error concealment. Sizes are limited by the engine's pixel cap (8192×8192); a larger header returns `Limit`. A `sequence_scalable_extension` returns `Unsupported`.

The elementary-stream splitter returns the same access units however the bytes are chunked. Both fields of a field picture stay one unit, including when a `sequence_end_code` sits between them.

## Encoder

Main profile, progressive 4:2:0, 8-bit frame pictures. Closed GOPs. The GOP length is half the frame rate in pictures, at least one, with two B pictures between anchors (`M = 3`). `CodecParams` has no GOP fields, so the length is not a caller setting. A forced keyframe, or a cut, closes the GOP early and the next picture is intra.

A cut is a luma histogram distance above half the samples. Bins are four levels wide, counted on every other row. On the local recordings, ordinary motion stays under 0.25 of the samples; a hard splice of two of them scores about 1.7. The threshold is 0.5.

Motion search is half-pel, forward, backward, and bidirectional, and `f_code` follows the vector. The picture is intra, inter, skipped, or not coded by cost. Rate control is TM5 steps 1 and 2 with a VBV model. Activity modulation was tried and removed: it cost about 1.2 dB and did not earn its place. A zero bitrate asks for about 0.3 bits per sample. Constant-rate mode stuffs so the buffer does not overflow. The level is the lowest Main-profile level (Main, High-1440, High; Tables 8-10 to 8-13, with their bit-rate and VBV limits) that holds the size, the frame rate, and the bitrate. Above High level, or a bitrate above 80 Mb/s, `open` returns `Unsupported`. A frame rate that is not in Table 6-4 is refused at the first picture. Main profile High level holds at most 62 668 800 luma samples a second, so 1080p at 50 or 60 Hz has no Main profile level and is refused, and so is any picture taller or wider than 1920×1152 (portrait 1080×1920 and 720×1280 among them). `main_profile_level` answers the same question before an export starts.

The encoder writes `q_scale_type` 0 and intra VLC table B.15. The decoder reads both scales and both tables. The sequence display extension signals BT.709 limited range, which is what the export adapter produces.

Interlaced, 4:2:2, and 4:4:4 encoding are out of scope: export hands the encoder progressive 4:2:0. Scalable layers are refused, not written.

## Integration

- ISOBMFF demux (`crates/containers/isobmff/src/stsd.rs`): `mp4v` with an object type 0x60–0x65 or 0x6A is MPEG-2. The QuickTime MPEG video fourccs (`mp2v`, `m2v1`, `hdv1`–`hdva`, `mx3n`–`mx5p`, `xd51`–`xd5f`, `xdv1`–`xdvf`, `xdhd`, `xdh2`, `AVmp`, `m1v `, `m1v1`, `mpeg`, `mp1v`) are MPEG-2. Any other `mp4v` stays MPEG-4 Part 2.
- ISOBMFF mux (`mux.rs`, `file_mux.rs`): MPEG-2 is `mp4v` plus an `esds`. The object type follows the sequence extension profile (Main 0x61, Simple 0x60, SNR 0x62, Spatial 0x63, High 0x64, 4:2:2 0x65, no extension 0x6A). An empty configuration still muxes, with object type 0x61, and demuxes back as MPEG-2.
- Export (`crates/export/src/config.rs`) rounds an MPEG-2 coded size down to even edges.
- Desktop (`apps/desktop/src/lib.rs`, `view.rs`, `shell.rs`) registers the decoder and the encoder, lists MPEG-2 in the export picker, writes an MP4, refuses a MOV, and budgets twice the H.264 bitrate. With MPEG-2 picked, the dialog greys the short sides and frame rates Main profile cannot hold and names the reason; `Shell::export` refuses the same settings before the job starts (`mpeg2_settings_past_main_profile_are_refused_before_the_job`). The shell test decodes the exported file through the installed readers.
- Phones (`crates/ffi/src/codecs.rs`) register the Rust decoder after the platform decoders. Phone export stays H.264, so the encoder is not registered there. The phone test encodes eight frames and decodes them in display order.

## Capability matrix

| Capability | Decode | Encode | Status |
| --- | --- | --- | --- |
| Sequence, GOP, picture headers and extensions | Parse | Write | Verified. Header round trip in `headers.rs`. Fixtures below. |
| Scalable extensions | `Unsupported` | Not written | Verified. `scalable_layers_are_unsupported`. |
| 4:2:0 | Yes | Yes | Verified. Oracle, local files, export. |
| 4:2:2 | Yes | Out of scope | Verified. `chroma422.m2v`, `hhi_burst_422_short`, `xdcam8mp2`, `real_422`. |
| 4:4:4 | Yes | Out of scope | Verified. `chroma444.m2v` matches FFmpeg on every sample, including a predicted picture. |
| I, P, B frame pictures | Yes | Yes | Verified. |
| Field pictures | Yes | Out of scope | Verified. `tests/fields.rs`, `t.m2v`, `real_1080i`. |
| Frame, field, 16×8, dual-prime prediction, field DCT | Yes | Frame, half-pel | Verified on decode by `interlaced.m2v` and the FATE field streams. Dual-prime has a unit test for the vector formula. |
| Alternate scan, B.15, both quantiser scales, DC precision, matrices | Yes | `q_scale_type` 0, table B.15, zigzag, 8-bit DC | Decode verified by `interlaced.m2v` (alternate scan, B.15, non-linear scale, 10-bit DC). Encode writes the one combination above. |
| MPEG-1, including D pictures and full-pel vectors | Yes | No | Verified. `mpeg1.m1v`, `real_mpeg1`. |
| Display order, open GOP after a reset | Yes | Closed GOPs only | Verified. `lifecycle.rs`. |
| Rate control and VBV | n/a | TM5 steps 1–2, VBV, constant-rate stuffing | Verified. `encode.rs` rate and buffer tests. See the rate table. |
| Scene cut and forced keyframe | n/a | New GOP | Verified. `a_scene_cut_starts_a_gop_and_motion_does_not`, `a_forced_keyframe_closes_the_gop_there`. |
| Elementary-stream splitter | Yes | n/a | Verified. Chunk invariance in `lifecycle.rs` and the fuzzer. |
| Playback and export registration | Desktop and phones | Desktop MP4 | Verified by the desktop and ffi tests named below. |
| ISOBMFF `mp4v` and the QuickTime fourccs | Demux | Mux `mp4v` + `esds` | Verified. `stsd.rs`, `file_mux.rs`, `tests/isobmff.rs`. |
| Malformed input, drain, reset, cancel, size change | Fail closed, state kept | Same | Verified. `lifecycle.rs`, `encode.rs`, and the fuzzer's smoke run. |
| Fuzz, 30 minutes each | Seeded campaign | Seeded campaign | Verified. See Robustness. Zero findings. |

## Fixture manifest

Checked in under `tests/fixtures`, produced by FFmpeg 8.1.1 unless noted. The `.yuv` beside each stream is that FFmpeg's own decode. Comparison allows a sample error of 2, at most 3% of samples differing, and 55 dB on every plane. FFmpeg's two integer IDCTs differ by more than that from each other on these streams.

| File | What it is |
| --- | --- |
| `progressive.m2v` | 96×64 Main, open GOPs, I/P/B, `-g 6 -bf 2 -q:v 3` |
| `interlaced.m2v` | Frame pictures with field motion, field DCT, alternate scan, intra VLC B.15, non-linear quantiser, 10-bit DC |
| `chroma422.m2v` | 4:2:2 with a loaded non-intra matrix |
| `chroma444.m2v` | 16×16 High profile 4:4:4, written from H.262 because FFmpeg cannot encode 4:4:4. Twelve distinct block levels, then a P picture with a zero vector. SHA-256 `ba282d3ac4636265eed2742cb9a874464069706f1bc0bdde6fae4dd2f9258bc5` |
| `mpeg1.m1v` | 90×62 MPEG-1, a size that is not a multiple of 16 |

Not checked in, fetched from `https://fate-suite.ffmpeg.org/mpeg2/`: `sony-ct3.bs`, `tcela-6.bits`, `hhi_burst_422_short.bits`, `matrixbench_mpeg2.lq1.mpg`, `mpeg2_field_encoding.ts`, `xdcam8mp2-1s_small.ts`, `dvd_still_frame.vob`, `dvd_single_frame.vob`, `t.mpg`. Real elementary streams extracted from `../local-files` live in `/tmp/cutlass-mpeg2/es` and are not in the repo.

## Correctness

IEEE 1180-1990 (`dct.rs`): ranges 256/255, 5, and 300, both signs, 10 000 blocks each, plus zero in gives zero out. All four tests pass.

Independent streams, this machine, FFmpeg 8.1.1 `-fps_mode passthrough` against `m2v_yuv` built from `2d925e2c`. Frame counts match and the worst plane is at least 63 dB, except the one damaged stream noted below.

| Stream | Frames | Worst plane |
| --- | --- | --- |
| `progressive`, `interlaced`, `chroma422`, `mpeg1` fixtures | match | ≥ 64 dB (the committed test) |
| `chroma444` | 2 = 2, bit-exact | identical |
| `sony-ct3` | 7 = 7 | 66.0 dB |
| `tcela-6` | 4 = 4 | 65.8 dB |
| `hhi_burst_422_short` | 4 = 4 | 65.5 dB |
| `matrixbench_mpeg2.lq1` | 25 = 25 | 68.6 dB |
| `xdcam8mp2-1s_small` | 18 = 18 | 67.4 dB |
| `dvd_still_frame`, `dvd_single_frame` | 1 = 1 | 66.5 dB, 71.0 dB |
| `t` (field pictures, a sequence end between the last two fields) | 353 = 353 | 64.8 dB |
| `real_1080p` | 200 = 200 | 66.9 dB |
| `real_1080i` | 120 = 120 | 66.1 dB |
| `real_422` | 90 = 90 | 67.4 dB |
| `real_mpeg1` | 120 = 120 | 64.0 dB |
| `mpeg2_field_encoding` | 30 of FFmpeg's 31 | 67.7 dB on the 30 |

`mpeg2_field_encoding.ts` has a damaged coefficient. FFmpeg logs `ac-tex damaged at 0 22` and conceals one picture. Cutlass returns `Malformed` for that picture and keeps the 30 pictures already decoded. Concealment is out of scope.

VideoToolbox does not decode MPEG-2 on this Mac. `ffmpeg -hwaccel videotoolbox` reports `Failed setup for format videotoolbox_vld` and falls back to its software decoder. The second-decoder gate is unavailable here, not passed.

Our streams in FFmpeg: every encoder test, the ISOBMFF round trip, and the local-file encodes are decoded with `-xerror -err_detect explode`. The committed checks require at least 50 dB between our reconstruction and FFmpeg's. Measured agreement on the rate table is 66–72 dB.

## Encoding quality and rate

FFmpeg 8.1.1 `mpeg2video`, `-g` equal to our GOP, `-bf 2`, otherwise its defaults. Delta is our luma PSNR minus FFmpeg's at the same measured rate. Clips are 720×576 at 25, 1920×1080 at 25, 1920×1080 at 29.97, 1280×720 at 60, and 848×480 at 30, taken from `../local-files` and scaled where the name says so. Constant-rate rows are marked CBR.

| Clip | Target | Ratio | Ours Y | FFmpeg Y | Delta |
| --- | --- | --- | --- | --- | --- |
| sd25 | 0.8 Mb/s | 1.026 | 45.48 | 44.47 | +1.01 |
| sd25 | 1.5 Mb/s | 1.015 | 48.50 | 49.10 | −0.60 |
| sd25 | 4 Mb/s | 1.002 | 51.82 | above FFmpeg's range | |
| hd25 | 4 Mb/s | 1.026 | 50.27 | 50.59 | −0.32 |
| hd25 | 12 Mb/s | 1.005 | 53.57 | above FFmpeg's range | |
| hd2997 | 12 Mb/s | 1.000 | 41.69 | 42.07 | −0.39 |
| hd720p60 | 8 Mb/s | 1.000 | 38.22 | 36.10 | +2.12 |
| phone480 | 1 Mb/s | 1.001 | 39.22 | 36.99 | +2.23 |
| phone480 | 2.5 Mb/s | 1.000 | 46.19 | 45.97 | +0.21 |
| sd25 CBR | 2 Mb/s | 1.011 | 49.56 | 50.14 | −0.58 |
| hd2997 CBR | 15 Mb/s | 1.000 | 42.83 | 40.83 | +2.00 |

The 1.0 dB gate holds on every clip whose rate FFmpeg can reach. The largest shortfall is 0.60 dB. Decoders agree at 66–72 dB on these streams. The committed rate test requires 0.90×–1.10× over its run, and the VBV test requires the buffer never to go negative.

A cut costs a new intra picture. On a 720×576 splice at 3 Mb/s the worst picture rose from 41.8 dB to 44.4 dB and the rate stayed 3.00 Mb/s (it was 3.06 Mb/s with no detector). On 1 500 frames of an edited 640×360 recording at 1.5 Mb/s, mean luma rose from 47.56 dB to 48.26 dB at the same rate. The five clips in the table have no cut, and the detector leaves their bitstreams byte-identical, so the table still describes the final encoder.

## Lifecycle and integration checks

Scoped `cargo test` on this machine, all passing:

- `cutlass-mpeg2`: 44 lib, 14 encode, 5 fields, 2 fuzz smoke, 1 isobmff, 18 lifecycle, 1 local, 6 oracle.
- `cutlass-isobmff` lib tests, including the `mp4v` object-type and QuickTime fourcc tests.
- `cutlass-export` coded-size test.
- `cutlass` desktop tests filtered to export and codec registration.
- `cutlass-ffi` codec tests, including `the_phone_decodes_mpeg2`.

`cargo fmt -p cutlass-mpeg2 -- --check` and `cargo clippy -p cutlass-mpeg2 --all-targets -- -D warnings` pass on `2d925e2c`. The same clippy bar passed for `cutlass-isobmff`, `cutlass-export`, `cutlass`, and `cutlass-ffi` on the commits that touched them. Full-workspace `cargo test` and `cargo lint` were not run; another agent is using the original checkout, and the request says to keep this worktree's checks to the MPEG-2 crate and its direct callers.

## Robustness

The smoke fuzz under `cargo test` runs 400 decoder iterations and 60 encoder iterations. A refused decoder packet leaves the pictures around it unchanged. Splitting a clean stream in random chunks matches splitting it whole. A drained encoder run decodes to the pictures it accepted, in order, and a replay without the refused pushes writes the same bytes. One iteration may not exceed 20 seconds.

The timed campaigns used `MPEG2_FUZZ_SEED=20261108` and a budget of 1800 seconds. The same variables repeat them. The runner xors that seed with a per-target salt (`0xDEC` for the decoder, `0xE4C` for the encoder) and forces it odd; the seed printed below is the generator's starting state.

- Encoder, 8 November 2026, 15:06–15:36. Binary copied from the tree at `d4695bdf`. No encoder source has changed since. Printed seed `0x13526b9`. 118 459 iterations, 84 278 encoders opened, 1 265 473 pictures coded, 219 206 refused pushes, slowest iteration 1.49 s. `test result: ok`.
- Decoder, 15:20–15:50, binary of `2d925e2c` (the field-pair fix is in it; a 4:4:4 comment and this record came after, and neither changes decoding). Printed seed `0x1352519`. 6 273 828 iterations, the four modes within 1 568–1 569 thousand each, 2 720 941 refusals checked against the clean pictures, 61 571 821 pictures, slowest iteration 53 ms. `test result: ok`.

An earlier decoder campaign was on the binary from before the field-pair fix and was stopped. It is not evidence for this decoder. Both logs are in `/tmp/cutlass-mpeg2/fuzz/`. A failed assertion, a panic, or an iteration over 20 seconds fails the test, so a passing result is zero findings.

Two bugs the encoder fuzzer found, both fixed and covered by tests that fail without the fix:

- A cancelled `drain` set the drained flag before the waiting pictures were coded, so they were lost. `drain` now codes them unless it is cancelled, and a cancelled drain leaves them in place.
- `PictureRate::quant` overflowed an `i32` at one bit per second. The quantiser is clamped in floating point.

One bug the conformance compare found: a `sequence_end_code` between the two fields of a frame closed the pair. FFmpeg's FATE stream `t.mpg` ends that way. The splitter and the decoder now keep the pair open across the end code. `tests/fields.rs` fails if that is removed. After the fix, `t.m2v` matches FFmpeg's 353 frames.

## Performance

Release build, one thread, Apple M5 Pro, measured on the final binary while the two fuzz processes were also running. Resident set is `time -l` maximum resident. The rate column is file bits over the target.

| Clip | Frames | Encode | Encode RSS | Rate | Decode | Decode RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 720×576 25 | 100 | 290 fps | 12.5 MB | 1.002 | 1429 fps | 13.3 MB |
| 1920×1080 25 | 100 | 59 fps | 33.0 MB | 1.005 | 345 fps | 39.1 MB |
| 1920×1080 29.97 | 90 | 60 fps | 61.7 MB | 1.000 | 391 fps | 50.8 MB |
| 1280×720 60 | 120 | 139 fps | 25.8 MB | 1.000 | 1091 fps | 13.4 MB |
| 848×480 30 | 150 | 348 fps | 10.6 MB | 1.000 | 2500 fps | 9.0 MB |

The gates are 30 fps encode and 120 fps decode at 1080p, and a decoder resident set within 64 MB of the process baseline. Those hold. An earlier run of the same 1080p29.97 encode, before the cut detector and with a quieter machine, was 70 fps and 30 MB. The cut detector's histogram is a few percent of that encode. FFmpeg 8.1.1 `-threads 1` on that clip, measured then, encoded 90 frames in 0.54 s (about 167 fps) and decoded them in 0.11 s (about 820 fps).

## Conclusion

Decoder, encoder, ISOBMFF, desktop export, and phone playback are implemented. The gates that can run on this machine pass, including both 30-minute fuzz campaigns.

VideoToolbox has no MPEG-2 decoder on this Mac (`Failed setup for format videotoolbox_vld`), so the second-decoder comparison is unavailable. FFmpeg is the independent oracle for both directions. GStreamer's software MPEG-2 decoder is that same FFmpeg decoder, so it is not a second implementation.

Out of scope, and still out of scope: scalable layers, interlaced and 4:2:2 and 4:4:4 encoding, error concealment, and opening `.ts`, `.mpg`, `.vob`, or raw `.m2v` in the app. Those containers have no Cutlass crate yet.

Not the declared target, and named so they are not silent: the GOP length is derived from the frame rate rather than configured; quantiser scale type 1 is decoded but not written; spatial-activity quantiser modulation was measured and removed.
