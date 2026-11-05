# VP8 support record

Checkout: branch `vp8`, worktree `/Users/mrnewton/dev/cutlass/cutlass-vp8`, started from `development` at `3607813a`. Native safe Rust in `cutlass-vp8` with `#![forbid(unsafe_code)]`. No libvpx, FFmpeg, or GStreamer code is in the crate or its dependency graph. Those programs are test oracles only.

Request: a full VP8 decoder and encoder, end to end, built, tested, fixed, and benchmarked against ffmpeg, learning from the local FFmpeg and GStreamer checkouts and following the project skills and docs.

## Rules and skills that were followed

- Every file in `.cursor/rules` of this checkout: `codec-shipping`, `commit-often`, `container-shipping`, `development-worktree`, `local-media`, `merge-pull-request`, the product-name ban, `perf`, `platforms`, `pull-request`, `rust-standards`, `source-of-truth`, `tests`, `think-like-a-pro`, `worktree-names`.
- Skill `ship-codec` (`.cursor/skills/ship-codec/SKILL.md`) with `references/implementation.md`, `verification.md`, `deliverables.md`, and `video-image.md`. This file is that skill's scope record, capability matrix, fixture manifest, and validation report.
- Skills `think-like-a-pro` and `cutlass-worktree`.
- There is no `AGENTS.md` in this tree.

## Specification and sources

- RFC 6386, *VP8 Data Format and Decoding Guide*, November 2011. Section 1 says the reference decoder source in Section 20 wins over the narrative, so Section 20 is read as the normative definition.
  - Section 7 for the boolean encoder.
  - Section 9 for the frame header, versions, and reference updates.
  - Section 20 (`dixie`) for every decode table and procedure.
- FFmpeg `libavcodec/vp8.c`, `vp8dsp.c`, `vp8data.h`, `vpx_rac.h`, and `vp8_parser.c` from `../pub-packages/ffmpeg`, read for edge handling, MV clamping, reference updates, and threading layout.
- GStreamer, from `../pub-packages/gstreamer`:
  - `gst-plugins-bad/gst-libs/gst/codecparsers/gstvp8parser.c` for header parsing.
  - `gst-plugins-good/ext/vpx/gstvp8enc.c` and `gstvp8dec.c`, which wrap libvpx.
- Neither GStreamer nor FFmpeg ships a native VP8 encoder; both wrap libvpx. The encoder here is designed from RFC 6386 and checked with two independent decoders: FFmpeg's native `vp8`, and `libvpx`.

### Where the sources disagree

| Case | RFC 6386 Section 20 | libvpx | FFmpeg | This crate |
| --- | --- | --- | --- | --- |
| Segment loop-filter level outside 0..63 before the ref/mode deltas | Clamped first | Clamped first | Not clamped | Decoder clamps first (Section 20). The encoder keeps every segment level inside 0..63, so all three decoders agree on its streams. |
| Segment quantizer index outside 0..127 before the per-plane deltas | Not clamped | Clamped first | Not clamped | Decoder does not clamp (Section 20). The encoder never emits such an index. |
| `copy_buffer_to_golden = 2` in the same frame as `copy_buffer_to_alternate` | Golden takes the new altref | Golden takes the new altref | Golden takes the old altref | Decoder: new altref (Section 20). The encoder never sets both copies in one frame. |
| Loop-filter delta whose update flag is clear | Set to 0 | Kept | Kept | Kept. The Section 19.2 prose says the flag marks whether the value is updated, and every shipping encoder relies on that. This is an interoperability choice. |
| `mb_no_coeff_skip = 0` | Reuses a stale skip flag | Skip is 0 | Skip is 0 | Skip is 0. The encoder emits `mb_no_coeff_skip = 0` only when no macroblock skips, where all three readings agree. |

## Required target

### Decoder: the whole VP8 bitstream of RFC 6386

- Versions 0 to 3: six-tap or bilinear interpolation, and full-pixel chroma for version 3. The loop-filter type comes from the header bit.
- Key frames and inter frames, a frame size change on any key frame, and odd sizes up to 16383×16383.
- Intra 16×16 DC, V, H, TM, and B_PRED with all ten sub-block modes. Key-frame contexts and the 127/129 edge rules.
- Inter modes NEAREST, NEAR, ZERO, NEW, and SPLITMV with 16×8, 8×16, 8×8, and 4×4 partitions.
- LAST, GOLDEN, and ALTREF references with sign bias, refresh flags, and copy flags.
- Probability updates for coefficients, MV, and modes, with `refresh_entropy_probs` save and restore.
- Segmentation: map update and persistence, and absolute or delta quantizer and filter levels.
- Normal and simple loop filters with sharpness and ref/mode deltas.
- One, two, four, or eight token partitions. Skip on or off. Hidden frames (`show_frame = 0`).
- The colour-space, clamp, and horizontal and vertical scale bits, all reported by the decoder.

### Encoder: VP8 4:2:0 8-bit, every coding tool a decoder must accept

- Versions 0 to 3: six-tap or bilinear interpolation, the normal or simple loop filter, no loop filter for versions 2 and 3, and full-pixel chroma for version 3.
- Key and inter frames. Key frames come on an interval, on request (`force_keyframe`), and at scene cuts.
- Intra 16×16 and B_PRED, chosen by rate and distortion.
- Inter NEAREST, NEAR, ZERO, NEW, and SPLITMV with all four partitionings. Quarter-pel motion search runs on LAST, GOLDEN, and ALTREF.
- Golden refreshes. The old golden is copied to ALTREF (`copy_buffer_to_alternate = 2`).
- Optional hidden altrefs with lookahead:
  - A `show_frame = 0` frame codes the end of each golden group first.
  - The frames before it set the altref sign bias.
  - The frame it stands for copies it into golden (`copy_buffer_to_golden = 2`).
- Adaptive coefficient, MV, mode, skip, intra, and reference probability updates. `refresh_entropy_probs = 0` on hidden frames and in error-resilient mode.
- The loop-filter level is searched per frame, with libvpx's default ref/mode deltas and sharpness 0..7.
- Optional four-segment activity quantization, with segment filter offsets.
- One to eight token partitions.
- The display scale bits.
- `mb_no_coeff_skip = 0` when nothing skips.
- Constant quantizer and target bitrate modes.

### Out of scope, with reasons

- Encoder choices the decoder accepts but this encoder does not make:
  - Copy flag 1 (copy from LAST).
  - GOLDEN sign bias.
  - Absolute segment values.
  - Colour-space and clamp bits other than 0. libvpx writes 0 too.
- VP8 has no other bit depths or chroma formats.

## Gates, declared before tuning

| Gate | Bound |
| --- | --- |
| Decode libvpx streams across speed, partitions, error-resilient, alt-ref with lag, sharpness, static threshold, versions 1–3, odd sizes, and multiple key frames | YUV identical to ffmpeg `vp8` and to `libvpx` |
| Decode the independent 1920×1080 WebM in `../local-files` | YUV identical to ffmpeg `vp8` |
| Our encoder's packets, for every option | Decoded by ffmpeg `vp8` and by `libvpx`, both identical to our reconstruction |
| Lossy quality, 640×360 natural clips at 1 Mb/s, 150 frames | PSNR-Y within 2.0 dB of libvpx `-deadline good -cpu-used 4` at the same measured rate |
| Lossy chroma. Added later than the others, together with the scene-cut work. | PSNR-U and PSNR-V each within 2.0 dB of libvpx in the same runs |
| Rate, 150 frames at a target of 1 Mb/s | Mean rate within 0.85×–1.15× of the target |
| Decode speed, release, one thread, Apple M5 Pro | 1920×1080 at ≥ 60 fps, with the ratio to ffmpeg recorded |
| Encode speed, release, one thread | 640×360 at ≥ 30 fps on the default preset |
| Decoder fuzz | ≥ 30 minutes, seed and findings recorded, findings 0 |
| Encoder fuzz | ≥ 30 minutes, seed and findings recorded, findings 0 |

## Results against the gates

All numbers are from this machine (Apple M5 Pro, macOS 27): release builds, one thread, ffmpeg 8.1.1 from Homebrew with libvpx.

| Gate | Measured | Evidence | Result |
| --- | --- | --- | --- |
| Decode libvpx streams | 27 libvpx encodes, every pixel identical in both oracles | `libvpx_*_match_both_decoders` | Pass |
| Decode the 1080p WebM | 901 frames identical | `independent_webm_matches_ffmpeg` | Pass |
| Our packets in other decoders | Every encoder test stream is identical in ffmpeg `vp8`, `libvpx`, and our decoder. The streams cover modes, quantizers, sizes, partitions, bitrate, segments, versions 1–3, altref copies, hidden altrefs, error resilience, header options, and scene cuts. | `prove_encode::*`, `an_mp4_vp08_track_round_trips`, `the_export_path_writes_an_mp4_other_players_read` | Pass |
| PSNR-Y, default options and hidden altrefs | Ahead of libvpx on 5 of 6 runs; −0.18 dB on the 6th (see the quality table) | `gate_quality_and_rate_against_libvpx` | Pass |
| PSNR-U and PSNR-V | Above libvpx on all three clips, in both modes | same | Pass |
| Rate at 1 Mb/s | 1.003×–1.008× of 625,000 bytes on every clip and mode | same | Pass |
| Decode 1080p, one thread | testsrc2 270.2 fps; the 901-frame WebM 284.9 fps. ffmpeg `vp8` 519.5 and 827.4 fps, `libvpx` 454.5 and 877.3 fps, so 0.34×–0.52× of ffmpeg `vp8` | `cargo bench -p cutlass-vp8` | Pass |
| Encode 640×360, default preset | testsrc2 63.8 fps (59.6 fps with hidden altrefs); the three natural clips 44.8–48.9 fps in the gate test. libvpx good cpu-used 4: 877 fps | `cargo bench -p cutlass-vp8`, gate test | Pass |
| Decoder fuzz | 1800 s, 2,927,272 inputs to the core and engine decoders, findings 0 | `decoder_fuzz_does_not_panic` | Pass |
| Encoder fuzz | 1800 s on the final encoder: 41,235 rounds, 211,533 frames, findings 0 | `encoder_fuzz_does_not_panic` | Pass |
| `cargo fmt --all -- --check` | Clean for every file this branch touches. `crates/codecs/gif/src/quant.rs` already differs on `development` and is not touched here. | | Pass for this branch |
| `cargo clippy` with the workspace lints | Clean for `cutlass-vp8`, `cutlass-isobmff`, and the desktop app on macOS. `cutlass-vp8` and `cutlass-isobmff` are also clean with `--target x86_64-unknown-linux-gnu`. | | Pass, except the desktop Linux cross-lint, which cannot run here (see Not done) |
| `cargo test` | `cutlass-vp8`: 61 unit, oracle, and lifecycle tests and 2 integration tests pass in debug and in release; the gate is ignored by default and passes when run. `cutlass-isobmff`: 49 pass. Desktop `cutlass`: 250 pass, 2 ignored (a 4K HDR playback test and a test that needs `CUTLASS_MOVIE`, neither about VP8). | | Pass |

### Quality at 1 Mb/s, 640×360, 150 frames

Run with `cargo test --release -p cutlass-vp8 -- --ignored gate --nocapture`.

- The source is each clip's first 150 frames, scaled by ffmpeg to 640×360 yuv420p.
- libvpx runs `-deadline good -cpu-used 4 -threads 1` at the same bitrate.
- "Worst Y" is the lowest single-frame PSNR-Y.

| Clip | Mode | Ours Y / U / V, worst Y (dB) | Ours bytes, key frames | libvpx Y / U / V, worst Y (dB) | libvpx bytes |
| --- | --- | --- | --- | --- | --- |
| 16929689 | default | 41.47 / 47.68 / 46.58, 37.77 | 627,291, 1 | 40.54 / 46.70 / 45.55, 36.40 | 633,681 |
| 7688387224072670472 | default | 43.82 / 45.19 / 45.43, 38.06 | 628,267, 2 (one scene cut) | 44.00 / 44.79 / 44.98, 40.48 | 675,714 |
| IMG_8144 | default | 47.82 / 50.65 / 50.53, 44.09 | 629,641, 1 | 46.54 / 49.02 / 49.23, 39.56 | 660,568 |
| 16929689 | hidden altref | 41.29 / 47.53 / 46.44, 37.68 | 627,376, 1 | as above | 633,681 |
| 7688387224072670472 | hidden altref | 44.28 / 45.38 / 45.58, 40.21 | 627,879, 1 | as above | 675,714 |
| IMG_8144 | hidden altref | 47.82 / 50.55 / 50.38, 44.31 | 629,674, 1 | as above | 660,568 |
| 16929689 | segments on (reported only) | 41.14 / 47.28 / 46.20, 37.60 | 627,046, 1 | as above | 633,681 |

Findings from this table:

- **Worst frame.** In default mode on 7688387224072670472, the worst frame is 2.4 dB below libvpx's worst. It is 38.06 dB whether the scene cut is on or off, so the cut does not cause it: rate control gives that frame too few bits. Hidden-altref mode reaches 40.21 dB on the same clip. No worst-frame gate was declared, so this is a finding, not a failure.
- **Scene-cut cost.** The scene cut on 7688387224072670472 costs 0.08 dB of average PSNR-Y (43.90 dB with it off). In exchange the stream gets a random-access point at the cut. libvpx also places automatic key frames by default.
- **Rate accuracy.** libvpx spent 1–8% more bytes than its target. Ours stayed within 0.8% of it.

### Benchmark output

`cargo bench -p cutlass-vp8` on commit `f5e4ab9c`, with nothing else running:

| Case | cutlass | ffmpeg `vp8` | ffmpeg `libvpx` |
| --- | --- | --- | --- |
| Decode 1920×1080 testsrc2, 120 frames, 3002 kB | 270.2 fps | 519.5 fps | 454.5 fps |
| Decode the local 1080p WebM, 901 frames, 3645 kB | 284.9 fps | 827.4 fps | 877.3 fps |
| Encode 640×360 testsrc2, 150 frames at 1000 kb/s | 63.8 fps, 627,569 B, PSNR-Y 45.33 dB | none (FFmpeg has no native VP8 encoder) | 877.2 fps, 624,936 B, PSNR-Y 44.24 dB |
| Encode, hidden altrefs | 59.6 fps, 627,667 B, 160 packets (10 hidden) | | |

Every decode row has 0 mismatches against ffmpeg `vp8`, and libvpx decodes our encode with 0 mismatches.

The final code gives the same testsrc2 stream: 627,569 B at 45.33 dB, because scene cuts do not fire on it. Two runs on the final code shared the machine with other builds, an editor indexer, and an Android emulator (load average 6–10). They measured:

- Decode: 259.8 and 190.0 fps.
- WebM decode: 287.0 and 233.6 fps.
- Encode: 49.3 and 49.9 fps.
- Hidden-altref encode: 54.0 and 55.8 fps.

Those runs are recorded as loaded, not as the measurement. Every gate holds in both.

libvpx and ffmpeg's decoder use hand-written SIMD. This crate is safe Rust, with the loop filter and interpolation written so the compiler vectorizes them. Encoding is slower than libvpx because every macroblock is decided by rate and distortion over more candidates. At the same rate it reaches higher PSNR-Y on testsrc2 and on two of the three natural clips.

### Latency, delay, memory, and cancellation

Per-frame processing time, from the quiet run on `f5e4ab9c`:

| Path | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| Decode 1920×1080 | 3.55 ms | 5.13 ms | 5.63 ms | 5.83 ms |
| Encode 640×360 | 16.34 ms | 22.77 ms | 25.38 ms | 25.45 ms |

- **Decoder delay.** There is no reordering: each packet's picture is ready when `push` returns. A hidden frame returns no picture.
- **Encoder delay.**
  - With default options, each `push` returns that frame's packet.
  - With `hidden_altref`, the encoder looks `golden_interval` frames ahead (16 by default, at most 64). The first packet comes after frame 17, and `drain` returns the rest.
  - A hidden altref packet carries the pts of the next shown frame and a duration of 0, so a muxer keeps it in order without giving it display time.
- **Memory.** Measured with `/usr/bin/time -l`:
  - Streaming 1080p decode (`VP8_MEMORY=decode cargo bench -p cutlass-vp8`): max RSS 141 MB, peak footprint 27.6 MB.
  - Streaming 640×360 encode (`VP8_MEMORY=encode`): peak footprint 114 MB, of which about 104 MB is the benchmark's two copies of the 150 source frames.
  - The decoder keeps the current frame and the three references, shared by `Arc`. The encoder keeps those, plus at most `golden_interval` (≤ 64) queued inputs when `hidden_altref` is on.
- **Allocations** were not counted. A counting global allocator needs `unsafe`, which the workspace forbids.
- **Cancellation.** The encoder checks its cancel flag on every `push` and between queued frames. In the quiet benchmark run it stopped 9.4 ms after the flag was raised. Other runs measured 15.3, 4.6, and 8.5 ms, all at or under one frame's encode time.

## Capability matrix

| Capability | Required | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Bool coder | Yes | Read | Write | `random_bools_round_trip_at_every_probability`, `the_reference_two_byte_decoder_agrees`, `a_long_run_of_certain_ones_carries_into_written_bytes` | Verified |
| Frame header, partitions 1/2/4/8 | Yes | Accept | Emit | `a_written_key_header_parses_back`, `golden_keys_and_partitions_decode_identically`, libvpx `slices2/4/8` | Verified |
| Intra 16×16 DC, V, H, TM; B_PRED with all ten sub-modes; chroma | Yes | Accept | Emit | `every_macroblock_mode_is_chosen`, `diagonal_modes_match_the_reference_formulas`, libvpx streams | Verified |
| NEAREST, NEAR, ZERO, NEW, SPLITMV 16×8, 8×16, 8×8, 4×4 | Yes | Accept | Emit on LAST, GOLDEN, and ALTREF | `every_macroblock_mode_is_chosen` (all ten modes, identical in ffmpeg and libvpx) | Verified |
| LAST, GOLDEN, ALTREF; refresh flags | Yes | Accept | Emit | `altref_copies_decode_identically` (macroblocks predicted from ALTREF are counted), `golden_keys_and_partitions_decode_identically` | Verified |
| Copy flags | Yes | Accept 1 and 2 | Emit 2: golden to altref, and altref to golden | `altref_copies_decode_identically`, `hidden_altrefs_decode_identically`, libvpx `smooth_two_pass_altref` | Verified |
| Sign bias | Yes | Accept | ALTREF sign bias with hidden altrefs | `hidden_altrefs_decode_identically` | Verified |
| Hidden frames (`show_frame = 0`) | Yes | Accept, no picture | Emit as hidden altrefs, with lookahead | `hidden_altrefs_decode_identically` (fixed q, bitrate, version 1), `hidden_frames_produce_no_picture`, `playback_crops_odd_sizes_and_skips_hidden_frames` | Verified |
| `refresh_entropy_probs = 0` | Yes | Accept, save and restore | Hidden frames and error-resilient mode | `error_resilient_streams_decode_identically`, `an_inter_header_keeps_unsent_deltas_and_restores_probs` | Verified |
| Versions 1–3 (bilinear, simple filter, no filter, full-pixel chroma) | Yes | Accept | Emit | `versions_decode_identically`, `libvpx_versions_match_both_decoders` | Verified |
| Segmentation: map, persistence, relative quantizer and filter | Yes | Accept, also absolute | Emit (option `segments`, off by default) | `segments_decode_identically` (filter offsets present and in range), libvpx `resilient` | Verified |
| Loop filter: normal, simple, sharpness, ref/mode deltas | Yes | Accept | Level searched per frame; deltas; sharpness 0..7 | `libvpx_filter_options_match_both_decoders`, `header_options_decode_identically` | Verified |
| Probability updates for coefficients, modes, MV, skip, intra, and reference | Yes | Accept | Emit | `probability_updates_are_sent`, `an_inter_header_keeps_unsent_deltas_and_restores_probs` | Verified |
| Skip on and off | Yes | Accept | Off when no macroblock skips | `header_options_decode_identically` | Verified |
| Display scale bits | Yes | Accept, reported | Emit on key frames | `header_options_decode_identically` | Verified |
| Key frames by interval, on request, and at scene cuts | Yes | n/a | Emit | `scene_cuts_become_key_frames`, `export_and_playback_reach_vp8_through_the_engine_factories` (forced key) | Verified |
| Odd sizes, size change at a key frame | Yes | Accept | Emit | `libvpx_odd_sizes_match_both_decoders` (1×1, 17×9, 33×301, 193×97), `odd_sizes_decode_identically`, `size_changes_at_key_frames` | Verified |
| Constant quantizer and bitrate modes | Yes | n/a | Emit | `fixed_quantizers_decode_identically` (q 0, 10, 40, 80, 127), `bitrate_mode_decodes_identically`, gate test | Verified |
| Colour-space and clamp bits | Yes for decode | Accept, reported | Always 0 | libvpx streams | Decode verified. Encode of non-zero values is out of scope. |
| MP4 `vp08` with `vpcC` | Yes | Read through `cutlass-isobmff` | `vpcC` on the first packet; `Mp4Mux` writes `vp08` | `an_mp4_vp08_track_round_trips`, `a_vp8_track_is_vp08_with_its_vpcc`, `the_export_path_writes_an_mp4_other_players_read` (ffprobe reads BT.709) | Verified |
| Engine playback (`DecodeFactory`) | Yes | `EngineDecoder`: vpcC colour, odd sizes cropped to even, an empty packet is a dropped frame | n/a | `export_and_playback_reach_vp8_through_the_engine_factories`, `playback_crops_odd_sizes_and_skips_hidden_frames` | Verified |
| Engine export (`EncodeFactory`) | Yes | n/a | `EngineEncoder`: RGBA or yuv420p, bitrate, forced key frames, BT.709 vpcC | same, plus `the_export_path_writes_an_mp4_other_players_read` | Verified |
| Desktop app | Yes | VP8 in MP4 opens and plays through the registered decoder | The export dialog offers VP8 for MP4. A VP8 MOV is refused with "VP8 is written in an MP4". | `a_vp8_export_lands_an_mp4_that_plays_back`: export, reopen, decode at 1.5 s, BT.709 colours within 4 | Verified headless. Not clicked through in a running app. |
| Lifecycle: drain, reset, malformed input, cancel | Yes | Specified | Specified | `decoder_contract`, `encoder_contract`, `reset_drops_references`, `failed_frames_leave_the_decoder_as_it_was`, `malformed_headers_are_rejected` | Verified |
| Untrusted size | Yes | `with_max_pixels`. The engine opens with 8192² and refuses larger key frames with `Limit`. | Options and sizes validated | `malformed_headers_are_rejected`, `encoder_contract`, encoder fuzz | Verified |
| Fuzz | Yes | Core and engine decoder | Options, sizes, content, push, force-key, reset, drain, wrong sizes | `decoder_fuzz_does_not_panic`, `encoder_fuzz_does_not_panic` | Verified: 30-minute runs, findings 0 |
| Benchmarks against ffmpeg | Yes | Throughput, latency, memory | Throughput, latency, hidden-altref delay, cancel | `cargo bench -p cutlass-vp8` | Verified |
| Multithreading | Optional | One thread | One thread | | Not implemented |
| Platform codecs | Optional | Apple: none to use. Android: not investigated. | same | | CoreMedia in this SDK defines `kCMVideoCodecType_VP9` and no VP8 type, so VideoToolbox has no VP8 backend to add. Android MediaCodec VP8 was not looked at. |

## Fixtures

The libvpx streams are made at test time from `../local-files` sources. Each one is a command line in `src/prove.rs` that runs `ffmpeg -c:v libvpx` with the options its test name says. The synthetic sources are generated in code (`prove_encode::synthetic`) and need no files.

| File in `../local-files` | SHA-256 | Purpose |
| --- | --- | --- |
| `16929689_1920_1080_30fps.mp4` | `5c59ecb8561ad53e29db3d10ce81de0829fd0bcaee000cad3e6b67ad2a0efa11` | libvpx stream source, quality gate |
| `7688387224072670472.mp4` | `9563675c2cd9cc9b34fb63441d1310e61ad243fd6e15ca2c0864dc9243e5b4cd` | Quality gate; has a scene cut |
| `IMG_8144.MP4` | `7fbb20808c51d363e56e752c3128798be693931d85fd5a016e03f13ebd67b30f` | Quality gate |
| `file_example_WEBM_1920_3_7MB.webm` | `4c3d89f4130a6349bab2a6b9c57066879c95c87f966ca5ff4b8c689e4834184b` | Independent 1080p VP8 decode, benchmark |

None of these files are checked in. A test that needs them prints `skipped: no ../local-files` and passes without checking anything. A run without them is not evidence.

## Fuzz

Both fuzzers run in a debug build, so overflow checks are on.

### Decoder fuzz (`decoder_fuzz_does_not_panic`)

- **Settings:** `VP8_FUZZ_SECS`, and `VP8_FUZZ_SEED` (default `0x56503800f22d`).
- **Corpus:** three streams from our encoder:
  - 48×32 with default options.
  - 33×17 with four partitions and golden refreshes at a fine quantizer.
  - 16×16 with a key frame every second frame at a coarse quantizer.
- **Each round:**
  - It decodes a good prefix of one stream.
  - For each later frame it feeds one of three inputs: the frame itself, random bytes, or the frame with one to five mutations. The mutations are bit flip, byte replace, truncate, insert, header-byte replace, and append.
  - The same inputs go to the core decoder and to the engine's `EngineDecoder`, which is opened with random extradata. Both must agree.
- **Checks:**
  - Every picture must have the decoder's size and crop cleanly.
  - After the noise, the stream's key frame must decode to a picture.
- **Limit:** the decoder is capped at 65,536 pixels, so a mutated size stays cheap.

### Encoder fuzz (`encoder_fuzz_does_not_panic`)

- **Settings:** `VP8_ENC_FUZZ_SECS`, and `VP8_ENC_FUZZ_SEED` (default `0x565038454e43`).
- **Each round:**
  - It picks options: bitrate, quantizer, key and golden intervals, partitions, speed, segments, version, hidden altref, error resilience, sharpness, scale, and scene cut.
  - It picks a size from 1×1 to 80×64.
  - One round in eight breaks one setting or the size, and the encoder must refuse it.
  - It pushes up to 12 frames, each a moving picture, noise, or a flat picture, with random durations. Force-key, reset, and drain are mixed in, and sometimes a picture of the wrong size, which must be refused.
- **Checks:** every packet must decode in our decoder to exactly the encoder's reconstruction of each shown frame, and the count of shown frames must match.

### Recorded runs

| Fuzzer | Commit | Seed | Seconds | Volume | Findings |
| --- | --- | --- | --- | --- | --- |
| Decoder (core and engine) | `f5e4ab9c` | `0x56503800f22d` | 1800 | 2,927,272 inputs, 1,906,848 pictures, 1,016,368 errors | 0 |
| Encoder | `f5e4ab9c` | `0x565038454e43` | 1800 | 44,028 rounds, 5,477 refused, 224,474 frames, 168,754 packets | 0 |
| Encoder, with scene cuts | `1ea865d9` | `0x565038454e43` | 1800 | 41,235 rounds, 5,124 refused, 211,533 frames, 157,883 packets | 0 |
| Decoder (core only, before the engine adapter) | `d4414acf` | `0x56503800f22d` | 1800 | 5,922,430 inputs, 3,861,230 pictures, 2,053,099 errors | 0 |

The recorded line of the run on the final encoder:

```text
vp8 encoder fuzz seed 0x565038454e43 seconds 1800 rounds 41235 refused 5124 frames 211533 packets 157883 findings 0
test result: ok. 1 passed; 0 failed
```

Notes on these runs:

- **Final encoder.** Commits after `1ea865d9` change the quality report, the seed parser, the dev build profile, a GIF format fix, and this file, not the encoder.
- **Throughput.** The two runs on `f5e4ab9c` ran side by side, and the decoder run also drives the engine decoder, so it got through fewer inputs than the earlier core-only run.
- **Seeds.**
  - The third run was asked for seed `0x5650384b4559`. The seed parser then read only decimal, so it fell back to the default.
  - Its option draws still differ from the second run's, because scene cut is one more option.
  - Commit `133089fa` makes the seed variables accept `0x` hex and fail on a value that does not parse.
- **Packet counts.** These count only what `push` returned. Packets from `drain` are not counted, and frames queued for a hidden altref when a reset comes are dropped by design.
- **Build.** All four runs were debug builds at `opt-level = 0` with overflow checks. Commit `c3d73bc2` builds `cutlass-vp8` at `opt-level = 3` in dev, still with overflow checks. A 2-second run there covers 829 rounds.

## Integration

- **Playback.**
  - `cutlass_vp8::register_engine_decoders` puts `EngineDecoder` on the engine's `DecodeFactory`. The desktop app calls it in `register_codecs_with`.
  - `cutlass-isobmff` demuxes an MP4 `vp08` track. The decoder hands the engine a CPU yuv420p picture.
  - The `vpcC` colour bytes give the primaries and transfer. Without them the default is BT.601.
  - Odd sizes are cropped to even, because the engine's 4:2:0 pictures need even sizes.
  - Hidden frames return no picture, and an empty packet is a dropped frame.
- **Export.**
  - `register_engine_encoders` puts `EngineEncoder` on `EncodeFactory`.
  - The export pipeline's `Planar` adapter turns RGBA into BT.709 limited-range 4:2:0, and the encoder's `vpcC` says BT.709.
  - `Mp4Mux` writes the `vp08` sample entry.
  - The export dialog offers VP8 only when the container is MP4. A VP8 MOV is refused before any work starts, because QuickTime has no VP8 sample entry.
- **Seek.** The desktop test reopens the exported file and decodes the picture at 1.5 s. That picture is a single flat colour, so the test proves the seek path for VP8, not frame accuracy.

## Not done

- **No multithreading** in either direction. Decoding runs at a third to a half of ffmpeg's speed, and encoding at about a fourteenth of libvpx's.
- **No WebM.** There is no WebM muxer or demuxer: `cutlass-matroska` is an empty scaffold. VP8 export goes to MP4 only, and `.webm` files do not open in the app. The 1080p WebM test reads its packets with ffmpeg.
- **Encoder choices not made.** The decoder accepts all of these and is checked on libvpx streams that use them:
  - Copy flag 1 (copy from LAST).
  - GOLDEN sign bias.
  - Absolute segment values.
  - Colour-space and clamp bits other than 0.
- **Worst-frame quality.** In default mode on 7688387224072670472, the worst frame is 2.4 dB below libvpx's worst frame. No gate was declared for this.
- **Dev build profile.** `cutlass-vp8` builds at `opt-level = 3` in dev builds, like the sound codecs, with overflow checks still on. Without it, the desktop VP8 export test took 89 s on this machine, against a 300 s job limit in the test, and a debug app could not play 1080p VP8 in real time.
- **Not in CI.** These need `../local-files` and ffmpeg, and skip without them:
  - The libvpx oracle tests.
  - The 1080p WebM test.
  - The quality gate.
  - The export interop test.

  Without ffmpeg, the synthetic encoder round trips still check our decoder against the reconstruction.
- **Desktop Linux cross-lint not run.**
  - `cargo clippy -p cutlass --target x86_64-unknown-linux-gnu` fails on this Mac before it reaches our code: `glib-sys` and `gdk-sys` need a Linux pkg-config sysroot.
  - The shared crates this branch changes, `cutlass-vp8` and `cutlass-isobmff`, cross-lint clean.
  - The desktop change is three registration lines, a dialog entry, a MOV refusal, and a test.
- **Not tried in a running app.** The desktop path is verified by headless tests through the real shell, readers, and export job. Nobody has clicked through it.
- **VP9** is a separate crate and is still not registered in the app.
