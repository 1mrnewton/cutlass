# VP9 support record

Checkout: branch `cursor/vp9-encoder-decoder-d4af`. Native safe Rust in `cutlass-vp9`. No `unsafe`. No bundled C, FFmpeg, libvpx, or GStreamer library. Those tools are oracles only.

Playback, export, and a WebM container are outside this crate. `cutlass-matroska`, `cutlass-engine`, `cutlass-playback`, and `cutlass-export` are empty scaffolds, so the shipped path is the video codec contract: `VideoDecoder` and `VideoEncoder` in `cutlass-decode` and `cutlass-encode`.

## Instructions that were loaded

The codec-shipping skill (recovered from closed pull request 38, commit `d8b85f6a542c7ed75d56ebb18ca6a7618a9d44a5`) was followed: capability matrix before the codec, native safe Rust, independent decoder and encoder evidence, lifecycle, malformed input, a recorded fuzz campaign, and release measurements.

These paths are named by the ISOBMFF scope note and are not in this checkout, because `.cursor/` is gitignored and they are not in git history: `container-shipping`, `commit-often`, `perf`, `source-of-truth`, `no-capcut-names`, `local-media`, `rust-standards`, `platforms`, and `tests`. They were not invented.

Oracles on this machine: ffmpeg 6.1.1 (`/usr/bin/ffmpeg`). `ffprobe` lists native `vp9` and `libvpx-vp9`. GStreamer 1.24.2 `vp9dec` (gst-plugins-good, libvpx) is installed. Native ffmpeg `vp9` is an independent algorithm. `libvpx-vp9` and `vp9dec` are both libvpx. System libvpx is 1.14.0. It is not linked into this crate.

`cargo deny check` (cargo-deny 0.20.2) reported advisories ok, bans ok, licenses ok, sources ok. It warned that four allowed licenses were not encountered: `Apache-2.0 WITH LLVM-exception`, `BSD-3-Clause`, `ISC`, and `Zlib`.

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7, 22 February 2017.
- libvpx (BSD) was read where the specification text disagrees with bitstreams produced by libvpx. Those disagreements are recorded below. The libvpx source is not compiled into this crate.

Known specification mismatches this crate follows libvpx on, because that is what `libvpx-vp9` emits:

- Intra frames use `kf_partition_probs`. Inter frames use `partition_probs`. Section 9.3.2 swaps that test. Section 10.4 says the keyframe table is for intra frames, which matches libvpx.
- Partition context bytes are the libvpx `partition_context_lookup` values. The context bit is `mi_width_log2` of the block, and the left context is indexed by `mi_row & 7`. The formula `15 >> b_width_log2` does not match those bytes.
- Loop-filter levels ignore a stale `loop_filter_delta_update` when deltas are disabled. If deltas are disabled, every level is the segment level. If they are enabled, the saved deltas are applied.

## Required target

Profile 0, 8-bit, 4:2:0, key frames and inter frames. The decoder accepts partitions, transform sizes through 32×32, intra modes, inter modes including compound prediction, sub-pixel motion, the loop filter, segmentation, tiles, eight reference slots, `show_existing_frame`, error-resilient frames, probability adaptation, and reference scaling.

Profiles 1, 2, and 3 (4:2:2, 4:4:4, 10-bit, 12-bit) return `CodecError::Unsupported`. A coded edge above 4096 returns `CodecError::Limit`. That edge is a resource limit, not a profile cut.

The encoder emits profile 0, error-resilient, one tile, no loop filter, no segmentation, DC intra, a single LAST reference, integer motion, and transforms no larger than 8×8. It quantizes and can follow a target bitrate. The decoder still accepts libvpx streams that use the rest of profile 0.

## Gates

| Gate | Bound | Result |
| --- | --- | --- |
| Lossless q = 0 | Decoded YUV matches the source bytes | Pass |
| Flat 64×64, quantizer 32 | PSNR-Y at least 45 dB | Pass |
| Ramp 64×64, quantizer 32 | PSNR-Y at least 38 dB | Pass |
| Checker 64×64, quantizer 32 | PSNR-Y at least 28 dB | Pass |
| 16-frame motion, quantizer 32 | Mean PSNR-Y at least 32 dB | Pass |
| About 30 frames at 500 kbps | Mean rate inside 0.70×–1.25× | Pass |
| Decode 320×240 at 30 fps | At least 1× realtime on this machine | Pass, 498.0 fps |
| Fuzz | At least 30 minutes on the decoder, seed and findings recorded | Pass, findings 0 |

Encoder throughput is measured and recorded. It is not required to be realtime. On this 4-core Intel Xeon, 30 frames of 320×240 at quantizer 32 encoded at 41.8 fps. The decode measurement above is the same run. High-water mark was 4856 KiB.

Recorded line:

`vp9 320x240 q32 frames 30 decode_fps 498.0 encode_fps 41.8 hwm_kib 4856`

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Profile 0, 8-bit, 4:2:0, key and inter | Accept | Emit, within the encoder limits above | `lossless_constant_matches_the_source`, `lossless_motion_matches_the_source`, `lossy_keyframes_meet_the_psnr_gates`, `lossy_motion_meets_the_psnr_gate`, `factories_open_the_profile_0_codec` | Verified on this machine |
| Profiles 1–3 | Error | Error | `profile_one_is_unsupported`, `open_rejects_the_sizes_and_formats_outside_profile_0` | Verified |
| Edge above 4096 | Error `Limit` | Error `Limit` | `open_rejects_the_sizes_and_formats_outside_profile_0` | Verified |
| Lossless round trip | Exact YUV | Emit | `lossless_constant_matches_the_source`, `lossless_ramp_and_odd_blocks_match_the_source`, `lossless_motion_matches_the_source`, `lossless_red_keyframe_matches_ffmpeg` | Verified |
| Lossy PSNR and rate gates | n/a | Named bounds above | `lossy_keyframes_meet_the_psnr_gates`, `lossy_motion_meets_the_psnr_gate`, `thirty_frames_stay_near_500_kbps` | Verified |
| libvpx-vp9, native vp9, and GStreamer vp9dec decode our bitstreams | n/a | All three | `independent_decoders_match_our_yuv` | Verified on this machine |
| We decode a libvpx or ffmpeg bitstream | YUV matches the oracle | n/a | `lossy_keyframes_match_ffvp9`, `error_resilient_inter_matches_ffvp9`, `normal_inter_matches_ffvp9` (probability adaptation and previous-frame motion), `altref_inter_matches_ffvp9`, `scaled_references_match_libvpx` (1/2 and 4/5; an illegal ratio is `Unsupported`), `segmentation_matches_libvpx`, `odd_size_loop_filter_matches_libvpx` (52×52 with the loop filter) | Verified |
| `show_existing_frame` and a two-frame superframe | Accept | The superframe test builds both | `show_existing_and_a_superframe_round_trip` (native ffmpeg `vp9` returns both pictures; ffmpeg `libvpx-vp9` returns the last picture of one superframe packet) | Verified |
| Lifecycle, malformed input, cancel | Specified | Specified | `lifecycle_keeps_references_until_reset_or_cancel`, `failed_inter_does_not_drop_references`, `reset_drops_references`, `empty_packet_is_truncated`, `bad_marker_and_sync_are_malformed` | Verified |
| 30-minute fuzz | Required | n/a | `decoder_fuzz_does_not_panic` | Verified, findings 0 |
| Release speed and memory | Required | Measured | `release_320x240_meets_realtime_decode` | Verified |

## Fuzz

Debug build, seed `0x56503900d4af`, `VP9_FUZZ_SECS=1800`. The corpus is four encoded 32×16 frames plus short illegal headers. Mutations flip, truncate, insert, append, or replace the bytes. Every 32nd input decodes the corpus key, then the mutated bytes, then the key again, and the key must still produce a picture.

Recorded line:

`vp9 fuzz seed 0x56503900d4af seconds 1800 inputs 175512941 pictures 0 findings 0`

The run finished in 1800.03 seconds with exit 0. Mutated buffers produced no picture. The key-after-noise check did not fail.

## Not exercised

The segmentation temporal-update bit is implemented. The aq-mode 1 fixture (`segmentation_matches_libvpx`) uses map copy with `update_map` clear, including a predicted segment id of 4 on the inter frames. That fixture does not read the temporal bit.

## Still required before calling this complete

Nothing in the gates above is still open. Playback, export, and a WebM container remain outside this crate.
