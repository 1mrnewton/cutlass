# H.264 support record

Checkout: `491f9300f2ae781150d00720b0e323bc61bcca97` on `h264-sw-enc-dec`. Native safe Rust in `cutlass-h264`. No `unsafe`. No bundled C, FFmpeg, or other H.264 library.

This record is partial. Progressive 8-bit 4:2:0 decode and High CABAC encode are implemented. The release speed, memory, and fuzz runs below were executed on this machine. Playback and export are outside this record. The desktop shell does not register this crate.

## Specifications

- Recommendation ITU-T H.264 (V16) (06/2026) is in force. Reconstruction follows clauses 7, 8, and 9 as read from V14 (08/2021). V16’s new SEI messages are not applied.
- ISO/IEC 14496-15 was not obtained. Annex B and length-prefixed samples were checked against FFmpeg `libavformat/avc.c` and this workspace’s existing sample description.
- Official ITU conformance bitstreams were not obtained.

FFmpeg 8.1.1 is the independent oracle. It has no software H.264 encoder; encode checks use libx264. GStreamer in the local tree wraps x264 and OpenH264 and has no software reconstruction of its own. Those sources are not linked into this crate.

## Machine

Apple M5 Pro, 18 cores, arm64, macOS 27.0.1, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. Release profile, one thread. The decoder and encoder do not spawn threads.

## What this run measured

Hostile input is in the default library tests. A truncated access unit, a forbidden NAL bit, data partitioning, an unknown NAL type, a length past the buffer, a length above 8 MiB, a short `avcC`, an interlaced SPS, a 10-bit SPS, an oversized picture, and more than 16 references are rejected. After each rejection a known 16×16 IDR still reconstructs. `random_packets_fail_closed` does the same for 64 deterministic packets. `encoder_rejects_a_bad_configuration` rejects a zero size, a size above the cap, an odd width, a 1-byte length, the wrong codec id, a mismatched picture, and a zero timescale, then still emits a picture.

Decoder fuzz, release, seed `0xC0FFEE01`, 1800.00 seconds:

`cargo test -p cutlass-h264 --offline --release --lib thirty_minute_decoder_campaign -- --ignored --nocapture --test-threads=1`

458,022,150 packets. 310,437,207 returned an error. The rest returned pictures inside the coded-size cap, with plane lengths matching 4:2:0. Every 2,000 packets the decoder was reset and the known IDR still reconstructed. No panic. Exit status 0.

Encoder fuzz, release, seed `0xBEEF4201`, 120.00 seconds:

`cargo test -p cutlass-h264 --offline --release --lib bounded_encoder_campaign -- --ignored --nocapture --test-threads=1`

1,111,135 pictures of 16×16 were encoded and every packet was accepted by this decoder. 277,089 random configurations were rejected by `open`. No panic. Exit status 0.

720p High decode, release, one thread. ffmpeg 8.1.1 libx264, profile high, preset veryfast, CABAC, two B frames, `testsrc2` 1280×720, 30 frames:

`cargo test -p cutlass-h264 --offline --release --lib high_720p_decode_meets_the_release_gates -- --ignored --nocapture --test-threads=1`

30 pictures in 0.41 seconds, about 73 fps. The gate is 30 fps. Resident memory, from `ps -o rss=` sampled after the file was loaded and again after each picture, grew 29,008 KiB (28.3 MiB). The gate is 64 MiB above that baseline. Both gates passed. Exit status 0. The sample is resident size at those points, not a kernel peak between samples.

Encode quality, from the default tests on this machine: `testsrc2` 320×240, 60 frames, 500 kbps, luma 39.94 dB and 1,016,856 bits (target 1,000,000). One natural clip scaled to 640×360, 30 frames, 800 kbps, luma 35.09 dB and 812,424 bits (target 800,000). Gates are 30 dB, 28 dB, and 0.75×–1.25× of `bitrate * duration`.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Progressive 8-bit 4:2:0, Baseline, Main, High | Accept | Emit High CABAC | Library tests named in `README.md` | Verified on this machine for the cases those tests build |
| I, P, and B, partitions through 8×8 on P | Accept, including 4×4 | Emit I, P, and B. P searches through 8×8. B is 16×16 | Encoder tests and `b_frame_comes_out_in_display_order` | Verified. B skip, B direct, and B partitions below 16×16 are not emitted |
| 8×8 transform | Accept | Inter luma when it is closer than 4×4 | `eight_by_eight_transform_matches_ffmpeg` | Verified against ffmpeg |
| Rate and luma | n/a | Frame QP | `testsrc2_meets_the_luma_and_rate_gates`, `natural_clip_meets_the_luma_gate` | Verified on this machine. The natural test skips when `../local-files` or ffmpeg is absent |
| Hostile input | Error, then a later IDR still decodes | Error, then a later picture still encodes | `hostile_framing_is_rejected_and_the_decoder_stays_usable`, `hostile_parameter_sets_are_rejected`, `random_packets_fail_closed`, `encoder_rejects_a_bad_configuration` | Verified |
| 30-minute decoder fuzz | Run | n/a | `thirty_minute_decoder_campaign` | Verified, this machine, 1800.00 s, seed `0xC0FFEE01` |
| Bounded encoder fuzz | n/a | Run | `bounded_encoder_campaign` | Verified, this machine, 120.00 s, seed `0xBEEF4201` |
| 1280×720 High decode | At least 30 fps, under 64 MiB extra | n/a | `high_720p_decode_meets_the_release_gates` | Passed. About 73 fps, 29,008 KiB |
| ITU conformance bitstreams | Not run | Not run | Streams not obtained | Not available |
| Desktop registration | Not wired | Not wired | `apps/desktop` prints and returns | Not done |
| Linux and Windows | Not run | Not run | This file was written on macOS | Not run |

## Checks run with these edits

- `cargo fmt -p cutlass-h264`
- `cargo clippy -p cutlass-h264 --offline --all-targets -- -D warnings` passed
- `cargo test -p cutlass-h264 --offline --lib -- --test-threads=1` — 49 passed, 3 ignored, 0 failed

`cargo fmt --all -- --check`, `cargo lint`, `cargo test --workspace`, and `cargo deny check` were not run for this record.

## Still required before calling this complete

- Official ITU conformance bitstreams, or this gap stays as not available.
- Register the software decoder and encoder when the platform has none for this codec.
- Linux and Windows runs.
- The workspace fmt, lint, test, and deny commands above.
- Encoder gaps that are not release gates: B slices do not search below 16×16 and do not emit skip or direct. Intra blocks stay DC 4×4. Weighted prediction, 4×4 sub-macroblocks, and more than one slice per picture stay unemitted, as `SCOPE.md` records.
