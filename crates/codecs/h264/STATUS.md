# H.264 support record

Checkout: `8fce4a643865` on `h264-sw-enc-dec`. Native safe Rust in `cutlass-h264`. No `unsafe`. No bundled C, FFmpeg, or other H.264 library.

This record is partial. Annex A decode for the profiles in `SCOPE.md` is implemented, and Annexes G, H, and I are partial. The encoder still emits progressive High, 8-bit, 4:2:0, CABAC. The desktop registers this decoder after the platform decoder, and export falls back to this encoder when no platform encoder takes the picture.

Writing this file did not re-run the suite, the bench, the fuzz campaigns, or the 3840×2160 file. Numbers below are the runs already recorded, named with the checkout that produced them.

## Specifications

- Recommendation ITU-T H.264 (V16) (06/2026) is in force. Reconstruction follows clauses 7, 8, and 9 as read from V14 (08/2021). Picture timing and film grain are applied. Other V16 SEI messages are consumed by size.
- ISO/IEC 14496-15 was not obtained. Annex B and length-prefixed samples were checked against FFmpeg `libavformat/avc.c`.
- Official ITU conformance bitstreams were not obtained.

FFmpeg 8.1.1 is the independent oracle where it decodes the tool. The command is `ffmpeg -hwaccel none -threads 1`. Encode checks use libx264. GStreamer in the local tree wraps x264 and OpenH264 and has no software reconstruction of its own. Those sources are not linked into this crate.

## Machine

Apple M5 Pro, arm64, macOS. Release profile, one thread. The decoder and encoder do not spawn threads.

## What was measured

Hostile input is in the default library tests. A truncated access unit, a forbidden NAL bit, a bad data partition, an unknown NAL type, a length past the buffer, a length above 8 MiB, a short `avcC`, a bit depth past 14, an oversized picture, and more than 16 references are rejected. After each rejection a known 16×16 IDR still reconstructs. `random_packets_fail_closed` does the same for 64 deterministic packets.

The runs at checkout `491f9300f2ae781150d00720b0e323bc61bcca97` were not repeated on `8fce4a643865`:

Decoder fuzz, release, seed `0xC0FFEE01`, 1800.00 seconds. `thirty_minute_decoder_campaign`. 458,022,150 packets. 310,437,207 returned an error. The rest returned pictures inside the coded-size cap of that checkout, with plane lengths matching 4:2:0. Every 2,000 packets the decoder was reset and the known IDR still reconstructed. No panic. Exit status 0.

Encoder fuzz, release, seed `0xBEEF4201`, 120.00 seconds. `bounded_encoder_campaign`. 1,111,135 pictures of 16×16 were encoded and every packet was accepted by this decoder. 277,089 random configurations were rejected by `open`. No panic. Exit status 0.

720p High decode gate, 30 frames of `testsrc2`, release, one thread. `high_720p_decode_meets_the_release_gates`. 30 pictures in 0.41 seconds, about 73 fps. The gate is 30 fps. Resident memory grew 29,008 KiB (28.3 MiB). The gate is 64 MiB above that baseline. Both gates passed. The sample is resident size at those points, not a kernel peak between samples.

Encode quality, from the default tests on that checkout: `testsrc2` 320×240, 60 frames, 500 kbps, luma 39.94 dB and 1,016,856 bits (target 1,000,000). One natural clip scaled to 640×360, 30 frames, 800 kbps, luma 35.09 dB and 812,424 bits (target 800,000). Gates are 30 dB, 28 dB, and 0.75×–1.25× of `bitrate * duration`.

`cargo bench -p cutlass-h264 --bench against_ffmpeg` on that checkout, release, one thread. Decode is 120 frames of that High stream, median of 3 after a warmup. Encode is 30 frames of `testsrc2` at 2,000,000 bits/s, one run after a warmup. libx264 is ultrafast. Cutlass times are the codec loop. FFmpeg times include process startup.

| | Cutlass | FFmpeg | FFmpeg / Cutlass |
| --- | --- | --- | --- |
| Decode | 0.42 s, 286.1 fps | 0.14 s, 849.1 fps | 3.0× |
| Encode | 7.33 s, 4.1 fps | 0.07 s, 432.3 fps | 105.6× |

`docs/codecs/h264/decode.md` records a later pixel check of `12471039-uhd_3840_2160_24fps.mp4`. That file matches FFmpeg inside the cropped rectangle: 351 pictures, 21348 ms, 16.4 fps, release, Apple M5 Pro. The same file was the 8-bit check after each later decode step in that document. This file's edit did not re-time it. There is no realtime gate at 3840×2160.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Baseline, Main, High, progressive 8-bit 4:2:0 | Accept | Emit High CABAC | `baseline_intra_matches_ffmpeg`, `baseline_ip_matches_ffmpeg`, `main_cabac_ipb_matches_ffmpeg`, `high_cabac_8x8_matches_ffmpeg` | Recorded against FFmpeg. Not re-run for this file. |
| I, P, and B, skip, direct, partitions through 4×4 | Accept | Emit I and P through 8×8. B is 16×16. | `main_cabac_temporal_direct_matches_ffmpeg`, `b_frame_comes_out_in_display_order` | Recorded. B skip, B direct, and B partitions below 16×16 are not emitted. |
| Weighted prediction, two slices, crop, custom quant matrices | Accept | Not emitted | `main_explicit_weights_match_ffmpeg`, `high_implicit_weights_match_ffmpeg`, `two_slices_match_ffmpeg_across_the_boundary`, `cropped_picture_matches_ffmpeg`, `jvt_quant_matrices_match_ffmpeg` | Recorded against FFmpeg |
| High 10, 4:2:0 | Accept | Not emitted | `high10_420_matches_ffmpeg` | Recorded against FFmpeg as `yuv420p10le` |
| Bit depths 9, 12, and 14, at 4:2:0, 4:2:2, and 4:4:4 | Accept | Not emitted | `bit_depths_9_through_14_reconstruct_420`, `hand_built_422_and_444_match_the_oracle` | Recorded. libx264 on this machine writes 8-bit and 10-bit only, so the odd depths are a 16×16 CAVLC stream. |
| Bit depths 11 and 13 | Accept | Not emitted | The same depth tests | Recorded against the recommendation. FFmpeg refuses these depths. |
| 4:2:2 and 4:4:4 at 8 and 10 bits, with B frames | Accept | Not emitted | `x264_422_and_444_match_ffmpeg`, `x264_noise_422_and_444_match_ffmpeg` | Recorded against FFmpeg at full chroma resolution, on one thread and on four: clean pictures, and noisy ones under CABAC and CAVLC, where 8×8 and 4×4 transform macroblocks with coefficients sit side by side. |
| Monochrome | Refused | Refused | `monochrome_chroma_format_is_refused` | Unsupported |
| Separate colour planes | Accept | Not emitted | `separate_colour_planes_follow_the_recommendation` | Recorded against the recommendation. FFmpeg refuses the flag. |
| QP′Y-zero transform bypass | Accept | Not emitted | `qpprime_bypass_matches_ffmpeg`, `x264_lossless_matches_ffmpeg`, `x264_noise_422_and_444_match_ffmpeg`, `many_encoder_settings_match_ffmpeg` | Recorded against FFmpeg, with noisy pictures that libx264 writes partly as I_PCM inside CABAC slices at 4:2:0 and 4:2:2. |
| Interlaced as macroblock-adaptive frames (MBAFF): I, P, and B, CABAC and CAVLC, 4:2:0 and 4:2:2 | Accept | Not emitted | `x264_interlaced_streams_match_ffmpeg`, `x264_interlaced_422_matches_ffmpeg`, `x264_mbaff_matches_ffmpeg`, `mbaff_pairs_match_ffmpeg` | Recorded against FFmpeg at 8-bit 4:2:0 and at 8-bit and 10-bit 4:2:2. 4:4:4 is not checked. Under constrained intra prediction, the samples left of a macroblock beside a pair of the other kind are judged by one macroblock of that pair, not by each macroblock they lie in. |
| Interlaced as field pictures (PAFF) | Intra field pairs. A field picture's reference lists are built from frames, not fields, so prediction between fields is not to the recommendation. | Not emitted | `complementary_fields_match_ffmpeg` | Recorded against FFmpeg for an intra pair. libx264 does not write field pictures. |
| Slice groups, map types 0 through 6 | Accept | Not emitted | `map_formulas_follow_the_recommendation` and the map tests in `fmo.rs` | Recorded against the recommendation. FFmpeg's build leaves FMO out. |
| Data partitions, SP, and SI | Accept. SP and SI are 8-bit 4:2:0. | Not emitted | `partition_i_pcm_and_i16_follow_the_recommendation`, `sp_residual_follows_the_switch_flag`, `sp_and_si_intra_pcm_match_ffmpeg` | Recorded. FFmpeg 8.1.1 does not decode data partitions. |
| Picture timing, film grain, other SEI | Timing and grain change output. Other SEI leaves the samples. | Not emitted | `picture_timing_changes_the_output_time`, `frequency_and_autoregressive_grain_change_the_display`, `grain_is_not_stored_in_the_reference` | Recorded. Grain follows H.274. |
| VUI aspect, timing, and colour tags | On the picture | Bitstream-restriction VUI only | `sample_aspect_and_timing_are_on_the_picture`, `the_remaining_colour_idc_values_are_tags` | Recorded. `ffprobe` reported `12:11` and `bt709` for the aspect test. |
| Annex A sizes | A picture inside any defined level is decoded. A picture outside every level is refused. | Edges and pixel count capped below that | `pictures_inside_annex_a_match_ffmpeg`, `a_picture_past_every_level_is_refused` | Recorded. The stored edge is 16,880. |
| 3840×2160 High 8-bit 4:2:0 | Accept | Above the encoder cap | `uhd_3840x2160_high_matches_ffmpeg` | Recorded in `decode.md`: 351 pictures, 16.4 fps. Not re-run for this file. |
| SVC, MVC, and 3D | Partial | Not emitted | The tests in `annex_g.rs` | Recorded for the cases those tests build. The open rows are in `SCOPE.md`. FFmpeg 8.1.1 emits the base picture and does not decode the extension. |
| Annex B, length prefixes of 1, 2, and 4 bytes, `avcC` | Accept | Annex B or 4-byte lengths and `avcC` | Decoder framing tests and `encoder_rejects_a_bad_configuration` | Recorded for the original 8-bit path |
| Hostile input | Error, then a later IDR still decodes | Error, then a later picture still encodes | `hostile_framing_is_rejected_and_the_decoder_stays_usable`, `hostile_parameter_sets_are_rejected`, `random_packets_fail_closed`, `encoder_rejects_a_bad_configuration` | In the default tests. The 30-minute campaigns were not repeated on this checkout. |
| 1280×720 High decode | The recorded run passed 30 fps and the memory gate | n/a | `high_720p_decode_meets_the_release_gates`, `against_ffmpeg` | Passed at `491f9300`. Not repeated here. |
| ITU conformance bitstreams | Not run | Not run | Streams not obtained | Not available |
| Desktop registration | Wired after the platform opener | Registered; export falls back to it when no platform encoder takes the picture | `a_real_movie_returns_a_software_picture_and_audio` | Done |
| Linux and Windows | Not run | Not run | The recorded runs are on this Mac | Not run |

## Checks for this edit

This edit rewrote `SCOPE.md` and `STATUS.md`. `cargo fmt`, `cargo lint`, `cargo test`, and `cargo deny check` were not run.

## Still required before calling this complete

- The open Annex G, H, and I rows in `SCOPE.md`.
- Official ITU conformance bitstreams, or this gap stays not available.
- Repeat the fuzz campaigns, the 720p speed and memory gate, and the encoder quality gate on this checkout.
- Linux and Windows runs.
- The workspace fmt, lint, test, and deny commands.
- Encoder gaps that are outside the encoder target: B slices do not search below 16×16 and do not emit skip or direct. Intra blocks stay DC 4×4. Weighted prediction, 4×4 sub-macroblocks, and more than one slice per picture stay unemitted.
