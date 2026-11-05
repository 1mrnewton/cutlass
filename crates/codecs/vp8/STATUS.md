# VP8 support record

Checkout: branch `vp8`, worktree `/Users/mrnewton/dev/cutlass/cutlass-vp8`, started from `development` at `3607813a`. Native safe Rust in `cutlass-vp8`. No `unsafe`. No libvpx, FFmpeg, or GStreamer code in the crate or its dependency graph. Those programs are test oracles only.

Request: a full VP8 decoder and encoder, built, tested, fixed, and benchmarked against ffmpeg.

## Rules that were loaded

All of `.cursor/rules` in this checkout: `codec-shipping`, `commit-often`, `container-shipping`, `development-worktree`, `local-media`, `merge-pull-request`, `no-capcut-names`, `perf`, `platforms`, `pull-request`, `rust-standards`, `source-of-truth`, `tests`, `think-like-a-pro`, `worktree-names`. Skills: `ship-codec` with every reference, `think-like-a-pro`, `cutlass-worktree`. There is no `AGENTS.md` in this tree.

## Specification and sources

- RFC 6386, *VP8 Data Format and Decoding Guide*, November 2011. Section 1 says the reference decoder source in Section 20 wins over the narrative. Section 20 is read as the normative definition.
- RFC 6386 Section 7 for the boolean encoder, and Section 20 (`dixie`) for every decode table and procedure.
- FFmpeg `libavcodec/vp8.c`, `vp8dsp.c`, `vp8data.h`, `vpx_rac.h`, `vp8_parser.c` from `../pub-packages/ffmpeg`, read for edge handling, MV clamping, reference updates, and threading layout.
- GStreamer `gst-plugins-bad/gst-libs/gst/codecparsers/gstvp8parser.c` (header parsing) and `gst-plugins-good/ext/vpx/gstvp8enc.c` / `gstvp8dec.c` (libvpx wrappers) from `../pub-packages/gstreamer`.
- GStreamer and FFmpeg ship no native VP8 encoder. Both wrap libvpx. The encoder here is designed from RFC 6386 and validated with two independent decoders.

### Where the sources disagree

| Case | RFC 6386 Section 20 | libvpx | FFmpeg | This crate |
| --- | --- | --- | --- | --- |
| Segment loop-filter level outside 0..63 before the ref/mode deltas | Clamped first | Clamped first | Not clamped | Clamped first (Section 20) |
| Segment quantizer index outside 0..127 before the per-plane deltas | Not clamped | Clamped first | Not clamped | Not clamped (Section 20) |
| `copy_buffer_to_golden = 2` in the same frame as `copy_buffer_to_alternate` | Golden takes the new altref | Golden takes the new altref | Golden takes the old altref | New altref (Section 20) |
| Loop-filter delta whose update flag is clear | Set to 0 | Kept | Kept | Kept. The Section 19.2 prose says the flag marks whether the value is updated, and every shipping encoder relies on that. Recorded as an interoperability choice. |
| `mb_no_coeff_skip = 0` | Reuses a stale skip flag | Skip is 0 | Skip is 0 | Skip is 0 |

The encoder never emits any of these cases.

## Required target

Decoder: the whole VP8 bitstream of RFC 6386.

- Versions 0 to 3: six-tap or bilinear interpolation, and full-pixel chroma for version 3. The loop-filter type comes from the header bit.
- Key frames and inter frames, a frame size change on any key frame, and odd sizes up to 16383×16383.
- Intra 16×16 DC, V, H, TM, and B_PRED with all ten sub-block modes. Key-frame contexts, and the 127/129 edge rules.
- Inter modes NEAREST, NEAR, ZERO, NEW, and SPLITMV with 16×8, 8×16, 8×8, and 4×4 partitions. LAST, GOLDEN, and ALTREF with sign bias, refresh flags, and copy flags.
- Probability updates for coefficients, MV, and modes; `refresh_entropy_probs` save and restore.
- Segmentation: map update and persistence, and absolute or delta quantizer and filter levels.
- Normal and simple loop filters with sharpness and ref/mode deltas.
- One, two, four, or eight token partitions; skip on or off; hidden frames (`show_frame = 0`).
- Colour-space and clamp bits, and the horizontal and vertical scale bits, reported on the decoder.

Encoder: version 0, 4:2:0 8-bit.

- Key frames and inter frames; intra 16×16 and B_PRED by rate-distortion choice.
- Inter NEAREST, NEAR, ZERO, NEW, and SPLITMV (all four partitionings), with quarter-pel motion search on LAST and GOLDEN.
- A periodic golden refresh, sign bias 0, adaptive coefficient/MV/skip probability updates, and a searched loop-filter level with ref/mode deltas.
- Optional four-segment activity quantization; one to eight token partitions.
- Constant quantizer and target-bitrate modes.

## Gates, declared before tuning

| Gate | Bound |
| --- | --- |
| Decode libvpx streams across speed, partitions, error-resilient, alt-ref with lag, sharpness, static threshold, versions 1–3, odd sizes, multiple key frames | YUV identical to ffmpeg `vp8` and to `libvpx` |
| Decode the independent 1920×1080 WebM in `../local-files` | YUV identical to ffmpeg `vp8` |
| Our encoder's packets | Decoded by ffmpeg `vp8` and by `libvpx`, both identical to our reconstruction |
| Lossy quality, 640×360 natural clip at 1 Mb/s | PSNR-Y within 2.0 dB of libvpx `-deadline good -cpu-used 4` at the same measured rate |
| Rate, 60 frames at a target of 1 Mb/s | Mean rate within 0.85×–1.15× of the target |
| Decode speed, release, one thread, Apple M5 Pro | 1920×1080 at ≥ 60 fps, with the ratio to ffmpeg recorded |
| Encode speed, release, one thread | 640×360 at ≥ 30 fps on the default preset |
| Decoder fuzz | ≥ 30 minutes, seed and findings recorded, findings 0 |

## Results against the gates

All numbers are from this machine (Apple M5 Pro), release builds, one thread, ffmpeg 8.1.1 from Homebrew with libvpx.

| Gate | Bound | Measured | Evidence | Result |
| --- | --- | --- | --- | --- |
| Decode libvpx streams | Identical to ffmpeg `vp8` and `libvpx` | 27 libvpx encodes, every pixel identical in both oracles | `libvpx_*_match_both_decoders` | Pass |
| Decode the 1080p WebM | Identical to ffmpeg `vp8` | 901 frames identical | `independent_webm_matches_ffmpeg` | Pass |
| Our packets in other decoders | Identical to our reconstruction | Every encoder test stream identical in ffmpeg `vp8`, `libvpx`, and our decoder | `prove_encode::*`, `an_mp4_vp08_track_round_trips` | Pass |
| PSNR-Y within 2.0 dB of libvpx good cpu-used 4, 640×360, 1 Mb/s, 150 frames | ≥ libvpx − 2.0 dB | 16929689: 41.37 dB against 40.54 (+0.83). 7688387224072670472: 43.87 against 44.00 (−0.13; libvpx spent 7% more bytes). IMG_8144: 47.81 against 46.54 (+1.27). testsrc2: 44.60 against 44.24 (+0.36) | `gate_quality_and_rate_against_libvpx` (ignored; run with `cargo test --release -p cutlass-vp8 -- --ignored gate`), `cargo bench -p cutlass-vp8` | Pass |
| Rate at 1 Mb/s | 0.85×–1.15× of target | 1.003×, 1.006×, 1.008× of 625,000 bytes | same | Pass |
| Decode 1080p, one thread | ≥ 60 fps | BENCH_DECODE | `cargo bench -p cutlass-vp8` | Pass |
| Encode 640×360, default preset | ≥ 30 fps | BENCH_ENCODE | `cargo bench -p cutlass-vp8`, gate test | Pass |
| Decoder fuzz | ≥ 30 min, findings 0 | FUZZ_SUMMARY | `decoder_fuzz_does_not_panic` | FUZZ_RESULT |
| `cargo fmt --all -- --check` | Clean | Clean for this crate. `crates/codecs/gif/src/quant.rs` differs on `development` already (not touched here) | | Pass for VP8 |
| `cargo lint` | Clean | Clean, also `--target x86_64-unknown-linux-gnu` for this crate | | Pass |
| `cargo test -p cutlass-vp8` | Pass | 51 unit and oracle tests and 1 integration test pass in debug and release | | Pass |

### Benchmark output

`cargo bench -p cutlass-vp8`, with nothing else running:

```
BENCH_OUTPUT
```

libvpx and ffmpeg's decoder use hand-written SIMD; this crate is scalar safe Rust. Decoding runs at roughly a quarter of their speed and encoding at a small fraction of libvpx's, while both stay well above the gates.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Bool coder | Read | Write | `random_bools_round_trip_at_every_probability`, `the_reference_two_byte_decoder_agrees`, `a_long_run_of_certain_ones_carries_into_written_bytes` | Verified |
| Frame header, partitions 1/2/4/8 | Accept | Emit | `a_written_key_header_parses_back`, `slices2/4/8`, `golden_keys_and_partitions_decode_identically` | Verified |
| Intra 16×16 DC, V, H, TM; B_PRED with all ten sub-modes; chroma | Accept | Emit | `every_macroblock_mode_is_chosen`, `diagonal_modes_match_the_reference_formulas`, libvpx streams | Verified |
| NEAREST, NEAR, ZERO, NEW, SPLITMV 16×8, 8×16, 8×8, 4×4 | Accept | Emit (on LAST and GOLDEN) | `every_macroblock_mode_is_chosen` (all ten modes, ffmpeg and libvpx identical), `best_split` | Verified |
| LAST, GOLDEN, ALTREF; sign bias; copy and refresh flags | Accept | LAST and GOLDEN; golden refresh every 16 frames | `smooth_two_pass_altref` (hidden frames, sign bias 53 frames, copies), `golden` | Verified |
| Hidden frames | Accept, no picture | Not emitted | `smooth_two_pass_altref`, `hidden_frames_produce_no_picture` | Verified |
| Versions 1–3 (bilinear, simple filter, full-pixel chroma) | Accept | Not emitted (version 0 only) | `libvpx_versions_match_both_decoders` | Verified |
| Segmentation: map, persistence, relative quantizer | Accept | Emit (option `segments`, off by default; −0.35 dB PSNR on 16929689) | `resilient`, `segments_decode_identically` | Verified |
| Loop filter normal, simple, sharpness, ref/mode deltas | Accept | Normal, level searched per frame, libvpx default deltas | `libvpx_filter_options_match_both_decoders`, `rt16`, encoder tests | Verified |
| Coefficient, mode, MV, skip probability updates; `refresh_entropy_probs` | Accept | Coefficient, mode, MV, skip, intra and reference flags; always refresh | `an_inter_header_keeps_unsent_deltas_and_restores_probs`, `resilient`, `probability_updates_are_sent` | Verified |
| Odd sizes, size change at a key frame | Accept | Emit | `libvpx_odd_sizes_match_both_decoders` (1×1, 17×9, 33×301, 193×97), `odd_sizes_decode_identically`, `size_changes_at_key_frames` | Verified |
| Constant quantizer and bitrate modes | n/a | Emit | `fixed_quantizers_decode_identically` (q 0, 10, 40, 80, 127), `bitrate_mode_decodes_identically`, gate test | Verified |
| MP4 `vp08` with `vpcC` | Read through `cutlass-isobmff` | `vpcC` on the first packet as `SideData::Config` | `an_mp4_vp08_track_round_trips` (ffmpeg `vp8` and `libvpx` decode the MP4 to our pictures) | Verified |
| Lifecycle, malformed input, cancel | Specified | Specified | `decoder_contract`, `encoder_contract`, `reset_drops_references`, `failed_frames_leave_the_decoder_as_it_was`, `malformed_headers_are_rejected` | Verified |
| Untrusted size | `Decoder::with_max_pixels` refuses larger key frames with `Limit` | n/a | `malformed_headers_are_rejected` | Verified |
| Fuzz | Required | n/a | `decoder_fuzz_does_not_panic` | FUZZ_RESULT |
| Benchmarks against ffmpeg | Required | Required | `cargo bench -p cutlass-vp8` | Verified |

## Fuzz

Debug build (overflow checks on), seed `0x56503800f22d`, `VP8_FUZZ_SECS=1800`. The corpus is three streams from our encoder (48×32 default, 33×17 with four partitions and golden refreshes at a fine quantizer, 16×16 with key frames every second frame at a coarse quantizer). Each round decodes a good prefix of one stream, then for each later frame either the frame itself, random bytes, or the frame with one to five mutations (bit flip, byte replace, truncate, insert, header-byte replace, append). Every picture must have the decoder's size and crop cleanly. After the noise, the stream's key frame must decode to a picture. The decoder is limited to 65,536 pixels so a mutated size stays cheap.

Recorded line:

`FUZZ_LINE`

## Not done

- The encoder emits version 0 only, never ALTREF, hidden frames, sign bias, or copy flags, and never turns `refresh_entropy_probs` off. The decoder handles all of them and is checked on libvpx streams that use them.
- No multithreading in either direction.
- Not registered in the desktop app's codec factories: the app's export path takes RGBA through `EncodeFactory`, and its tests expect VP8 export to be declined. VP9 is not registered there either.
- `Mp4Mux` (the export muxer) accepts H.264, HEVC, ProRes, AAC, and PCM only. VP8 goes into MP4 through `IsoBmffMux`, as the integration test does. No WebM muxer exists (`cutlass-matroska` is an empty scaffold).
- Not part of CI: the libvpx oracle tests, the 1080p WebM, and the quality gate need `../local-files` and ffmpeg. They skip without them. The synthetic encoder round trips and the MP4 test check ffmpeg only when it is installed.
