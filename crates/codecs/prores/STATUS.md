# ProRes support record

Checkout: branch `prores`, cut from `development` at `57498d9b`. Native safe Rust in `cutlass-prores`: `#![forbid(unsafe_code)]`, workspace lints, no `unwrap`, `expect` or `panic` in library code, no printing. No C, no FFmpeg, no VideoToolbox in the crate. ffmpeg 8.1.1, ffprobe, VideoToolbox (through `ffmpeg -hwaccel videotoolbox`) and AVFoundation (through `avconvert`) are oracles and fixture generators only, started as processes from tests and the bench.

This file is the scope record. A row says verified only when a test named here ran and passed on this machine (Apple M5 Pro, 18 cores, macOS, rustc 1.99.1, 5 November 2026).

## Instructions that were loaded

`.cursor/skills/ship-codec/SKILL.md` with `references/video-image.md`, `references/verification.md` and `references/deliverables.md`, and the rules under `.cursor/rules`.

## Specifications and sources read

Normative:

- SMPTE RDD 36:2022, Apple ProRes Bitstream Syntax and Decoding Process. Frame and picture headers, slices, the entropy codes, scans, quantization, the inverse transform (Annex A accuracy test), alpha.
- Apple ProRes white paper, April 2022. Target data rates per profile and frame size.

Read for the algorithm and edge cases, not copied and not linked, from `../pub-packages`:

- FFmpeg `libavcodec/proresdec.c`, `proresdec.h`, `proresdsp.c`, `proresdata.c`, `prores_parser.c`, `proresenc_kostya.c`, `proresenc_kostya_common.c`, `proresenc_anatoliy.c`, and the simple IDCT. Where FFmpeg and RDD 36 disagree this crate follows RDD 36; the output clamp `4..=2^b - 5` matches FFmpeg because Apple's decoder agrees.
- GStreamer has no native ProRes codec. `gst-libav` maps it to FFmpeg, `gst/mxf/mxfprores.c` wraps it in MXF, and `sys/applemedia/vtdec.c` and `vtenc.c` call VideoToolbox.

Also measured, not read: the quantization matrices Apple's own encoder writes, taken from VideoToolbox streams' frame headers. The encoder uses those.

## Required target and scope

Decode all six RDD 36 profiles: Proxy, LT, 422, HQ, 4444 and 4444 XQ, progressive and interlaced, 4:2:2 and 4:4:4, 8- and 16-bit alpha, every slice width, custom and default matrices, color tags. Encode Proxy, LT, 422, HQ, 4444 with alpha and 4444 XQ at Apple's target rates. Write and read them in QuickTime movies through `cutlass-isobmff`. Register the codec for the app's import and export.

Resource limits, in `src/header.rs`: longest edge 16384, at most 67,108,864 pixels. A larger header is `Limit` before any plane is allocated.

## Gates

| Gate | Bound | Result |
| --- | --- | --- |
| Decode against ffmpeg | Every Y, Cb, Cr within 1; share off by 1 under 5% at 10 bits and 25% at 12; alpha identical | Pass, 39 streams |
| Decode against VideoToolbox | Off in no more samples than ffmpeg is, and by no more than 1 or ffmpeg's largest miss | Pass, every stream VideoToolbox decoded |
| Threads | One worker and many give identical bytes, decode and encode | Pass |
| Encode rate | 0.80× to 1.05× of Apple's target | Pass, all six at 1.00× |
| Encode quality | PSNR-Y at least prores_ks's minus 0.3 dB at no more than 1.05× prores_ks's rate | Pass, all six |
| Encode interop | VideoToolbox decodes every profile this encoder writes | Pass |
| End to end | ffmpeg movie → demux → decode → encode → mux; ffprobe stream description and packet times identical; AVFoundation reads 8 frames and the same color tags | Pass, BT.709 HQ and HLG 4444 with alpha |
| Real time | 2160p HQ decode at 29.97 fps or faster on all cores | Pass, 226.6 fps |
| Fuzz | 30 minutes on the decoder, seed and findings recorded | Pass, 336,504,629 cases, findings 0 |

The first decode gate also capped the share off by 1 at 1%. ffmpeg's own integer transform misses that against Apple's decoder (2.5% at 10 bits, 14% at 12 with some 2s), and this crate's float transform is closer to Apple's than ffmpeg's is, so the share gate became relative to VideoToolbox. The reason is written in `tests/external/decode_oracle.rs`.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Proxy, LT, 422, HQ (4:2:2, 10-bit) | Yes | Yes, at Apple's rate | `decode_oracle` (prores_ks, prores_aw, VideoToolbox streams), `encode_oracle`, `every_profile_round_trips_through_this_decoder` | Verified |
| 4444 and 4444 XQ (4:4:4, 12-bit) | Yes | Yes, at Apple's rate | Same | Verified |
| Alpha, 8- and 16-bit | Yes, exact | Yes, lossless, 4444 and XQ | `ks 4444 alpha 8/16`, `ks xq alpha 16`, `aw 4444 alpha`, `vt 4444 alpha`, `odd_sizes_and_alpha_survive`, `an_hlg_4444_movie_keeps_its_tags` | Verified |
| Interlaced, top and bottom field first | Yes | No, progressive only | `ks hq interlaced`, `ks hq interlaced tff`, `aw hq interlaced`, `ks 4444 interlaced alpha`, `ks hq interlaced 721x487` | Decode verified; encode not emitted |
| Slices of 1, 2, 4, 8 macroblocks | Yes | 8 (narrower at the right edge) | `ks hq 1 mb slices`, `2 mb`, `ks proxy 4 mb slices` | Verified |
| Default matrices (no matrix in header) | Yes | Matrices always written | `ks 422 default matrix` | Verified |
| Odd sizes | Yes | Yes, edge replicated | `ks hq 1917x1077`, `vt 4444 alpha 1917x1077`, `ks 4444 33x17 alpha`, `the_smallest_pictures_round_trip` (1×1 to 17×1) | Verified |
| Color tags (H.273 primaries, transfer, matrix) | Read from the frame header | Written from the frame or parameters | `ks lt hdr tags`, `an_hlg_4444_movie_keeps_its_tags` | Verified |
| 2160p | Yes | Yes | `a_4k_stream_matches_ffmpeg`, bench | Verified |
| QuickTime sample entries `apco`…`ap4x` | Demux sets extradata to the code | Mux writes the code's sample entry and a `colr` box from the first frame header | `every_prores_profile_survives_a_movie_round_trip`, `a_prores_frame_header_names_the_movie_color`, `e2e` | Verified |
| RGBA input | n/a | BT.709 limited range | `rgba_comes_back_as_its_bt709_colors` | Verified |
| App import | `ProresPictureDecoder`, 8-bit 4:2:0, alpha dropped | n/a | `prores_opens_in_rust_when_the_platform_is_off` | Verified, partial (8-bit) |
| App export | n/a | Proxy, LT, 422, HQ from the export panel | `a_prores_export_without_the_platform_reads_back` | Verified; 4444 and XQ not in the panel |
| Rate by `params.bitrate` | n/a | Not read; `with_budget_scale` | — | Unsupported |
| ProRes RAW | Not this crate | Not this crate | — | Outside scope |

## Accuracy

Command: `PRORES_REQUIRE_ORACLES=1 cargo test -p cutlass-prores --release --test external -- --nocapture --test-threads 1`. Exit 0, 7 passed, 35 s.

Fixtures are made at test time from `../local-files/16929689_1920_1080_30fps.mp4` and `12129324_3840_2160_30fps.mp4` by ffmpeg's `prores_ks` (22 cases, two of them 2160p) and `prores_aw` (8), and VideoToolbox through `-c:v prores_videotoolbox` (9). Each reference is ffmpeg's decode at its native format, and on a Mac VideoToolbox's decode through `-hwaccel videotoolbox`. Selected lines (max error, share of samples that differ):

| Stream | This crate vs ffmpeg, Y | This crate vs VideoToolbox | ffmpeg vs VideoToolbox |
| --- | --- | --- | --- |
| prores_ks HQ 2160p | 1, 3.43% | 1, 1.19% | 1, 2.52% |
| prores_ks 4444 2160p | 1, 15.8% | 1, 5.11% | 2, 13.5% |
| prores_aw Proxy | 1, 1.91% | 1, 0.47% | 4, 1.79% |
| prores_aw HQ | 1, 3.29% | 1, 1.31% | 1, 2.65% |
| prores_aw 4444 alpha | 1, 15.6%; alpha exact | 1, 5.83% | 2, 13.4% |
| prores_ks HQ interlaced | — | 1, 1.28% | 1, 2.74% |
| VideoToolbox HQ | — | 1, 1.33% | 1, 2.67% |
| VideoToolbox XQ | — | 1, 7.22% | 2, 15.6% |
| VideoToolbox 4444 alpha | alpha exact | 1, 5.84% | 2, 21.3% |

Two kinds of VideoToolbox reference are not a comparison. On odd sizes (`1917x1077`, `721x487`) VideoToolbox's output path shows the same edge artifacts against both decoders (max 220 and 613 for both); the gate still holds because both decoders miss by the same amount. On `33x17`, VideoToolbox fell back to ffmpeg's software decoder (ffmpeg is off in 0 samples), so that case is skipped. VideoToolbox refuses some generated alpha patterns with error -12912, so its alpha cases use a ramp.

The transform passes the RDD 36 Annex A test in `dct.rs`.

## Encoding quality

Command as above, `encode_oracle::every_profile_meets_its_rate_and_quality_gates`. Six frames of the 1080p29.97 clip at 10-bit 4:2:2 (12-bit 4:4:4 for 4444 and XQ). PSNR against that source after ffmpeg decodes each stream; prores_ks at its default for each profile; VideoToolbox through `prores_videotoolbox`.

| Profile | Encoder | Rate | PSNR Y / U / V | SSIM Y |
| --- | --- | --- | --- | --- |
| Proxy, target 45 | this crate (q 17) | 45.0 | 40.74 / 53.65 / 52.03 | 0.97738 |
| | prores_ks | 46.0 | 38.34 / 50.80 / 48.87 | 0.96944 |
| | VideoToolbox | 44.2 | 40.36 / 53.49 / 51.90 | 0.97623 |
| LT, target 102 | this crate (q 7) | 102.0 | 49.83 / 59.03 / 57.65 | 0.99572 |
| | prores_ks | 108.0 | 44.98 / 56.26 / 54.41 | 0.99204 |
| | VideoToolbox | 98.2 | 49.49 / 58.79 / 57.41 | 0.99503 |
| 422, target 147 | this crate (q 6) | 147.0 | 54.22 / 61.45 / 60.32 | 0.99813 |
| | prores_ks | 153.1 | 48.79 / 58.38 / 56.77 | 0.99639 |
| | VideoToolbox | 145.9 | 54.19 / 61.42 / 60.29 | 0.99773 |
| HQ, target 220 | this crate (q 4) | 220.0 | 59.50 / 64.64 / 63.73 | 0.99933 |
| | prores_ks | 226.1 | 54.35 / 60.94 / 59.73 | 0.99884 |
| | VideoToolbox | 218.7 | 59.51 / 64.62 / 63.73 | 0.99895 |
| 4444, target 330 | this crate (q 3) | 330.0 | 62.88 / 66.05 / 65.62 | 0.99976 |
| | this crate at prores_ks's rate | 313.8 | 62.21 / 65.71 / 65.26 | 0.99971 |
| | prores_ks | 313.8 | 58.85 / 64.39 / 63.68 | 0.99957 |
| | VideoToolbox | 333.9 | 63.22 / 66.25 / 65.84 | 0.99944 |
| XQ, target 495 | this crate (q 2) | 495.0 | 68.39 / 67.87 / 67.38 | 0.99997 |
| | this crate at prores_ks's rate | 391.3 | 66.12 / 65.98 / 65.54 | 0.99992 |
| | prores_ks | 391.2 | 63.40 / 66.95 / 66.40 | 0.99984 |
| | VideoToolbox | 530.5 | 69.02 / 68.94 / 68.36 | 0.99963 |

Rates are Mb/s at 29.97 fps. Worst case: XQ at prores_ks's rate is 2.7 dB better than prores_ks in Y but about 1 dB worse in U and V. VideoToolbox decodes every profile this encoder writes (`ffmpeg -hwaccel videotoolbox -i ours -f null -` exits 0). No human viewing check was done.

## End to end

`e2e::a_bt709_hq_movie_comes_back_through_this_crate` and `e2e::an_hlg_4444_movie_keeps_its_tags`. ffmpeg's `prores_ks` writes 8 frames of a local clip into a movie: 1080p HQ tagged BT.709, and 720p 4444 with 16-bit alpha tagged BT.2020 / HLG from the iPhone clip `IMG_9197.mov`. `IsoBmffFile` demuxes it (extradata is the sample-entry code), `ProresDecoder` decodes it, `ProresEncoder` re-encodes it, and `Mp4Mux` writes a new movie with the source's packet times.

- ffprobe's stream description (codec, tag, size, pixel format, primaries, transfer, matrix, frame rate, frames read) and every packet's pts and duration are identical between ffmpeg's movie and ours.
- ffmpeg's decode of ours against its decode of the original: HQ PSNR 67.47 dB (10-bit), 4444 with alpha 85.40 dB (12-bit).
- AVFoundation (`avconvert --preset PresetAppleProRes422LPCM`) reads 8 frames and keeps the color tags, BT.2020 / HLG included.

That last check found a gap: our muxer wrote no `colr` box, so AVFoundation re-tagged the HLG movie as BT.709 (ffmpeg's movie, which has the box, kept HLG). `Mp4Mux` now writes `colr` for a ProRes track from its first frame header.

## Lifecycle and robustness

`src/hardening.rs`, in the normal suite (`cargo test -p cutlass-prores`, 41 passed, 1 ignored):

- Every truncation of a valid frame is refused. Bad headers are refused by name (`frame identifier`, `frame size`, `frame header size`, `frame dimensions`, `chroma format`, `interlace mode`, `alpha channel type`, `picture header size`, `picture size`, `slice table`, `slice size`). Slice sizes past the picture, quantization indices 0, 225 and 255, component sizes past the slice, a coefficient run past the block, a codeword longer than 31 bits, empty alpha and alpha past its data are refused by name. An alpha run past the slice is clamped, as the spec says.
- `random_damage_never_panics`: 4,000 seeded mutations per run.
- An empty packet is `InvalidApi`; a raised cancel flag gives `Cancelled`; `reset` clears it; `drain` is empty. A refused frame leaves the decoder able to decode the next one. The `VideoDecoder` and `VideoEncoder` traits carry pts and duration through.
- The encoder refuses other codecs, unknown profile codes, zero sizes, short RGBA buffers, a size that differs from the parameters, and other pixel formats.

### Fuzz

cargo-fuzz is not installed, and installing it means downloading it and building libFuzzer's C++ runtime, which this task's rules do not allow, so no coverage-guided campaign was run. The in-process campaign instead:

Command: `PRORES_FUZZ_SECONDS=1800 cargo test --release -p cutlass-prores --lib fuzz_campaign -- --ignored --nocapture`. Seed `0x9e3779b97f4a7c15` (xorshift). Inputs: four valid frames (Proxy 40×24, HQ 72×40, 4444 33×17 with alpha, XQ 24×40 with alpha), mutated by bit flips, truncation, random byte runs, and damage inside the slice table and slice headers. Result, exit 0: 336,504,629 cases in 1800 s, 233,521,245 refused by name, no panic and no hang. Findings: none, so no regression was added.

Earlier runs: 60 s release, 9,137,968 cases, 6,339,380 refused, no panic; 90 s debug with overflow checks, 285,606 cases, no panic. The campaign is not coverage-guided; zero findings is evidence for these runs only.

## Performance

Command: `cargo bench -p cutlass-prores --bench throughput`. Apple M5 Pro, 18 cores, bench profile (release, no LTO), rustc 1.99.1. Real frames: ffmpeg decodes the first 6 frames of each clip to 10-bit 4:2:2 and 12-bit 4:4:4; each measurement codes 30 pictures cycling through them, in process. "All" is 18 workers; "one" is `with_threads(1)`. The one-worker encode is checked to give the same bytes as the all-worker encode. Other agents were building on this machine (load average 10 to 18), so the numbers are a lower bound and move by about 15% between runs.

| Size | Profile | Encode all | Encode one | Decode all | Decode one | Mb/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1080p | Proxy | 153.5 | 44.1 | 1212.4 | 184.8 | 45 |
| 1080p | LT | 135.5 | 33.8 | 1038.0 | 123.6 | 102 |
| 1080p | 422 | 150.6 | 28.3 | 852.3 | 88.3 | 147 |
| 1080p | HQ | 122.5 | 24.8 | 761.6 | 73.6 | 220 |
| 1080p | 4444 | 104.2 | 18.0 | 552.6 | 49.9 | 330 |
| 1080p | 4444 XQ | 101.3 | 16.0 | 434.3 | 39.3 | 495 |
| 2160p | Proxy | 55.5 | 12.3 | 465.9 | 51.2 | 181 |
| 2160p | LT | 48.9 | 9.1 | 366.9 | 32.6 | 410 |
| 2160p | 422 | 48.1 | 8.6 | 294.4 | 25.5 | 591 |
| 2160p | HQ | 42.3 | 6.8 | 226.6 | 18.3 | 884 |
| 2160p | 4444 | 28.3 | 4.8 | 163.7 | 12.7 | 1326 |
| 2160p | 4444 XQ | 26.1 | 4.0 | 124.6 | 10.2 | 1989 |

Budget: 2160p HQ decodes at 226.6 fps on all cores, 7.6 times real time at 29.97 fps. Met. One core decodes it at 18.3 fps, 0.61 times real time.

Beside it, one `ffmpeg -threads 0 -i hq.mov -f null -` process on the same 30 pictures: ffmpeg's ProRes decoder 121.9 fps, ffmpeg with VideoToolbox 83.1 fps. Both include process startup.

2160p encode is faster than real time for every profile except 4444 and XQ (28.3 and 26.1 fps). The first bench run, before the quantizer search started from the last frame's index, gave 13.3 fps for 2160p HQ; the search change made it three times faster with identical output on the encode oracle.

## Changes outside this crate

- `crates/codec/src/access/frame.rs`, `access/mod.rs`: `VideoFrame` carries an optional `Alpha { bit_depth, plane }`, set with `with_alpha`, which checks depth, size and stride. Every constructor sets `None`.
- `crates/containers/isobmff/src/file_mux.rs`: `ap4h` and `ap4x` are accepted ProRes sample entries; a ProRes track gets a `colr` box from its first frame header. `mux.rs`: `IsoBmffMux::set_color`.
- `crates/containers/isobmff/src/stsd.rs`: a ProRes sample entry without a configuration box demuxes with its code as extradata.
- `crates/ffi`: depends on `cutlass-prores`; `codecs.rs` registers the decoders and `export.rs` the encoders.
- `apps/desktop/src/lib.rs`: registers the decoders and encoders after the platform ones and H.264; `shell.rs` adds the ProRes export test.

## Build and lint

- `cargo fmt -p cutlass-prores -p cutlass-isobmff`.
- `cargo clippy -p cutlass-prores --all-targets -- -D warnings`: exit 0.
- `cargo clippy --workspace --exclude cutlass --all-targets --target x86_64-unknown-linux-gnu -- -D warnings`: exit 0.
- `cargo test -p cutlass-prores`: 41 passed, 1 ignored. `cargo test -p cutlass-isobmff`: 50 passed. `cargo test -p cutlass-ffi`: 77 passed. `cargo test -p cutlass --lib -- prores mov`: 31 passed, 2 ignored.

## Not run, partial, blocked

- Interlaced encode: not emitted. The encoder writes progressive frames.
- cargo-fuzz: not run, for the reason above. The in-process campaign ran instead.
- 4444 and 4444 XQ export from the app: partial. `ProresProfile` in `crates/export` has Proxy, LT, 422 and HQ; the codec writes all six through the API.
- App import: partial. The app's picture path takes 8-bit 4:2:0 from `ProresPictureDecoder` and drops alpha; the full-depth `VideoDecoder` output is registered but playback does not use it.
- `params.bitrate`: not read.
- Linux and Windows test runs: not run. Linux was linted, not tested; the external tests and the bench need ffmpeg and the local clips.
- Official ProRes conformance streams: not available. The decoder was checked against three encoders and two decoders instead.
- Peak memory: not sampled.
