# VP9 support record

Checkout: branch `cursor/vp9-encoder-decoder-d4af`. Native safe Rust in `cutlass-vp9`. No `unsafe`. No bundled C, FFmpeg, libvpx, or GStreamer library. Those tools are oracles only.

Playback, export, and a WebM container are outside this crate. `cutlass-matroska`, `cutlass-engine`, `cutlass-playback`, and `cutlass-export` are empty scaffolds, so the shipped path is the video codec contract: `VideoDecoder` and `VideoEncoder` in `cutlass-decode` and `cutlass-encode`.

## Instructions that were loaded

The codec-shipping skill (recovered from closed pull request 38, commit `d8b85f6a542c7ed75d56ebb18ca6a7618a9d44a5`) was followed: capability matrix before the codec, native safe Rust, independent decoder and encoder evidence, lifecycle, malformed input, a recorded fuzz campaign, and release measurements.

These paths are named by the ISOBMFF scope note and are not in this checkout, because `.cursor/` is gitignored and they are not in git history: `container-shipping`, `commit-often`, `perf`, `source-of-truth`, `no-capcut-names`, `local-media`, `rust-standards`, `platforms`, and `tests`. They were not invented.

GStreamer is not installed on this machine. `ffprobe` lists two VP9 decoders, native `vp9` and `libvpx-vp9`, and one encoder, `libvpx-vp9`. The two decoders are the independent decoder oracles. The libvpx encoder is the independent producer for decoder tests.

## Specifications

- VP9 Bitstream & Decoding Process Specification, version 0.7, 22 February 2017.
- libvpx (BSD) was read where the specification text disagrees with bitstreams produced by libvpx. Those disagreements are recorded below. The libvpx source is not compiled into this crate.

Known specification mismatches this crate follows libvpx on, because that is what `libvpx-vp9` emits:

- Intra frames use `kf_partition_probs`. Inter frames use `partition_probs`. Section 9.3.2 swaps that test. Section 10.4 says the keyframe table is for intra frames, which matches libvpx.
- Partition context bytes are the libvpx `partition_context_lookup` values. The context bit is `mi_width_log2` of the block, and the left context is indexed by `mi_row & 7`. The formula `15 >> b_width_log2` does not match those bytes.
- Loop-filter levels ignore a stale `loop_filter_delta_update` when deltas are disabled. If deltas are disabled, every level is the segment level. If they are enabled, the saved deltas are applied.

## Required target

Profile 0, 8-bit, 4:2:0, key frames and inter frames. That includes partitions, transform sizes through 32×32, intra modes, inter modes including compound prediction, sub-pixel motion, the loop filter, segmentation, tiles, eight reference slots, `show_existing_frame`, error-resilient frames, probability adaptation, and reference scaling.

Profiles 1, 2, and 3 (4:2:2, 4:4:4, 10-bit, 12-bit) return `CodecError::Unsupported`. A coded edge above 4096 returns `CodecError::Limit`. That edge is a resource limit, not a profile cut.

The encoder emits profile 0 with intra and inter pictures, motion search, quantization, and a target bitrate. It may leave tiles at 1×1 and may keep the transform size at 16×16 or smaller. The decoder still accepts libvpx streams that use the rest of profile 0.

## Gates

| Gate | Bound |
| --- | --- |
| Lossless q = 0 | Decoded YUV matches the source bytes |
| Flat 64×64, quantizer 32 | PSNR-Y at least 45 dB |
| Ramp 64×64, quantizer 32 | PSNR-Y at least 38 dB |
| Checker 64×64, quantizer 32 | PSNR-Y at least 28 dB |
| 16-frame motion, quantizer 32 | Mean PSNR-Y at least 32 dB |
| About 30 frames at 500 kbps | Mean rate inside 0.70×–1.25× |
| Decode 320×240 at 30 fps | At least 1× realtime on this machine, 4 threads |
| Fuzz | At least 30 minutes on the decoder, seed and findings recorded |

Encoder throughput is measured and recorded. It is not required to be realtime.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Profile 0, 8-bit, 4:2:0, key and inter | Accept | Emit | Not yet | In progress |
| Profiles 1–3 | Error | Error | Not yet | In progress |
| Edge above 4096 | Error `Limit` | Error `Limit` | Not yet | In progress |
| Lossless round trip | Exact YUV | Emit | Not yet | In progress |
| Lossy PSNR and rate gates | n/a | Named bounds above | Not yet | In progress |
| libvpx-vp9 and native vp9 decode our bitstreams | n/a | Both decoders | Not yet | In progress |
| We decode a libvpx-vp9 bitstream | Both oracles match our YUV | n/a | `lossless_red_keyframe_matches_ffmpeg`, `lossy_keyframes_match_ffvp9` (keyframes only) | In progress |
| Lifecycle, malformed input, cancel | Specified | Specified | Not yet | In progress |
| 30-minute fuzz | Required | Bounded | Not yet | Not run |
| Release speed and memory | Required | Measured | Not yet | Not run |

## Still required before calling this complete

The rows above must move to verified, with the test names written in the evidence column, or the record must say which gate failed.
