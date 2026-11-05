# FLAC support record

Branch `flac`. Native safe Rust in `cutlass-flac` (frames) and `cutlass-flac-container` (files). No `unsafe`. No C, no FFmpeg, no FLAC crate. `flac` 1.5.0, `metaflac` 1.5.0, and ffmpeg 8.1.1 are test oracles only.

## Specification

RFC 9639 (December 2024). The decoder accepts the whole frame format. The encoder writes fixed-blocking streams that use every subframe and residual coding, chosen the way libFLAC's presets choose them.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Constant, verbatim, fixed 0 to 4, LPC 1 to 32 | Exact | Chosen per subframe | `a_rich_frame_decodes_to_the_values_built_into_it`, `libflac_streams_decode_bit_exactly`, `our_streams_pass_flac_and_ffmpeg_bit_exactly` | Verified |
| Rice 4-bit and 5-bit parameters, escapes, partition orders 0 to 15 | Exact | Searched per partition order | same, plus `escaped_partitions_of_width_zero_are_silent` | Verified |
| Wasted bits | Exact | Detected per subframe | `extremes_noise_silence_and_wasted_bits_round_trip`, the `s24-wasted-bits` oracle case | Verified |
| Left/side, side/right, mid/side | Exact | Independent, estimated, or exhaustive choice | `every_channel_assignment_is_exercised`, `every_stereo_mode_round_trips_correlated_channels` | Verified |
| Depths 4 to 32 bits, 1 to 8 channels | Exact | Exact | `depths_channel_counts_and_signals_round_trip`, oracle cases `s4-mono` to `s32-extremes-side`, `s16-7ch`, `s16-8ch` | Verified |
| Every sample-rate code, including rates only STREAMINFO holds | Exact | Written | oracle cases `rate-7000-khz-code` to `rate-700000-streaminfo` | Verified |
| Variable blocking | Exact | Not written | `a_variable_block_size_stream_decodes_bit_exactly`, container `a_variable_block_stream_muxes_and_demuxes` | Verified on decode. The encoder always writes fixed blocking |
| Block sizes 16 to 65535 | Exact | Any size | `s16-block-16`, `s16-block-65535`, `s16-odd-block-1000` | Verified |
| Levels 0 to 8 | n/a | libFLAC presets | `every_level_round_trips_and_higher_levels_are_not_larger`; all nine levels pass `flac -t`, `flac -d`, and ffmpeg | Verified. Levels 6 to 8 approximate `subdivide_tukey` (see Gaps) |
| CRC-8, CRC-16, MD5 | Checked | Written | `every_flipped_bit_is_refused_or_decodes_without_panicking`, `the_signature_is_checked_only_over_a_complete_in_order_stream` | Verified |
| Malformed frames | Refused without panic, decoder unchanged | n/a | the decoder tests listed under Hardening | Verified |
| Factories, `SideData::Config` | Registered | Registered, opening and final STREAMINFO | `open_refuses_what_it_cannot_produce`, `open_refuses_mismatched_parameters`, e2e test | Verified |
| `.flac` files: demux, mux, seek, SEEKTABLE, STREAMINFO written back | `cutlass-flac-container` | `cutlass-flac-container` | container `tests.rs` and `tests/external` | Verified |
| ID3v2 before `fLaC` | Refused | Not written | container `malformed_metadata_is_refused` | Unsupported |
| Ogg FLAC, MP4 `dfLa` | Not done | Not done | n/a | Unsupported |
| Linux and Windows | Linux clippy only | Linux clippy only | `cargo clippy --target x86_64-unknown-linux-gnu` | Tests not run off macOS |

## Oracles

`FLAC_REQUIRE_ORACLES=1 cargo test --release -p cutlass-flac -p cutlass-flac-container` turns every missing tool into a failure. Without it, a test skips when its tool is missing. Tests that use `../local-files` skip when the folder is missing.

- `libflac_streams_decode_bit_exactly`: 29 streams written by `flac` across depths 4 to 32, 1 to 8 channels, every rate code, block sizes 16 to 65535, LPC order 32, and noise. Our samples equal `flac -d` and ffmpeg's.
- `our_streams_pass_flac_and_ffmpeg_bit_exactly`: the same signals encoded here at levels 0, 5, and 8, and one at all nine levels. Each file passes `flac -t`, and `flac -d` and ffmpeg return the source samples.
- Container `independently_written_files_demux_frame_for_frame`: files from `flac` (with pictures and tags, without a SEEKTABLE, with a dense one, tiny and big blocks, 24-bit 6-channel, 8-bit mono) and from ffmpeg. Every frame offset and size equals `flac --analyze` and ffprobe's packets. The MD5 matches.
- Container `seeks_land_on_the_frame_holding_the_target`: 25 seeks per file, each landing exactly.
- Container `files_muxed_here_pass_flac_metaflac_and_ffprobe`: remuxed files pass `flac -t`. `metaflac` reads the same STREAMINFO values as in the source and finds seek points. ffprobe lists the same packets as for the source, and the file demuxes back to the same frames.
- Container `a_real_recording_survives_demux_decode_encode_and_mux`: the first 90 s of `local-files/videoplayback.m4a` (stereo) and a speech WAV (mono), written by `flac -5`, then demuxed, decoded to `I16` and to `F32`, encoded, and muxed here. `flac -t` passes, ffmpeg's samples equal the source's, and `metaflac --show-md5sum` equals the source's. Stereo: 969 frames in 0.41 s (`I16`) and 0.45 s (`F32`), 3,217,317 bytes against libFLAC's 3,235,452.

## Compression against libFLAC

`compression_against_libflac_is_recorded` encodes each source here and with `flac` at the same level and compares the frame bytes. It fails if level 5 is ever more than 2% larger.

| Source | -0 | -5 | -8 |
| --- | --- | --- | --- |
| synthetic s16 stereo | −0.07% | −0.03% | −0.06% |
| synthetic s24 6-channel 96 kHz | −1.33% | −0.33% | −0.39% |
| synthetic s24 stereo | −1.30% | −0.05% | −0.33% |
| speech, 16 kHz mono, 60 s | −0.01% | −0.01% | −0.03% |
| `videoplayback.m4a`, 60 s | −0.00% | −0.31% | −0.15% |
| rain (MP3 source), 60 s | −0.00% | +0.01% | +0.03% |
| expo (M4A source), 60 s | +0.03% | −0.70% | −0.65% |

Worst level 5: +0.01%.

## Hardening

Decoder tests build frames by hand and damage them:

- Every truncation of a rich frame is refused. Every single flipped bit is refused or decodes without a panic.
- Refused: a bad CRC-16, reserved subframe types, a set padding bit, wasted bits that leave no sample bits, reserved residual methods, partitions that do not divide the block or are shorter than the predictor, a predictor longer than the block, LPC precision 15, a negative LPC shift, a residual outside 32 bits, a prediction or decorrelated sample outside the depth, nonzero frame padding, trailing bytes, and a header that disagrees with STREAMINFO.
- A failed push leaves the decoder as it was. MD5 is reported only for a complete, in-order stream (`Match`, `Mismatch`, `Incomplete`, `Unknown`).

Encoder tests: every level, depths 4 to 32, 1 to 8 channels, extremes, noise, silence, wasted bits, streams shorter than a block, block sizes 16 and 65535 with order 32, every stereo mode, out-of-range and uneven input (nothing is added), layout reordering, reset, push after drain, and cancel.

## Campaigns

In-process campaigns in `tests/external/fuzz.rs` of both crates. They are not libFuzzer. cargo-fuzz is not installed here and was not run.

- Decoder: frames encoded here with random settings, then bit flips, truncation, insertion, or random bodies behind a sync code. Half the time the CRC-16 is refreshed so the damage reaches the subframe readers. Only a panic fails.
- Encoder: random level or random settings (block size, LPC order, partition orders, stereo mode, window), depth, channel count, rate, signal, and push sizes. The round trip must be exact and the MD5 must match.
- Container: small valid files damaged by flips, truncation, insertion, metadata damage, duplicated ranges, or random bodies after `fLaC`, then probed, read, and seeked. The muxer is fed damaged and misnumbered frames.

The short runs are part of the suite: decoder 4,000 seeds, encoder 600, container 2,000.

Long runs at commit `ca71d388`, release, one thread each, all three at once. Each exited 0 with no panic:

| Campaign | Command | Time | Seeds | Findings |
| --- | --- | --- | --- | --- |
| Decoder | `FLAC_FUZZ_SECS=1800 cargo test --release -p cutlass-flac --test external fuzz::decoder_campaign -- --ignored --nocapture --exact` | 1800.0 s | 2,303,418 | 0 |
| Encoder | `FLAC_FUZZ_SECS=600 cargo test --release -p cutlass-flac --test external fuzz::encoder_campaign -- --ignored --nocapture --exact` | 600.0 s | 113,698 | 0 |
| Container | `FLAC_FUZZ_SECS=1800 cargo test --release -p cutlass-flac-container --test external fuzz::demux_and_mux_campaign -- --ignored --nocapture --exact` | 1800.0 s | 1,804,653 | 0 |

## Speed

`cargo bench -p cutlass-flac --bench against_flac`: the first 60 s of `local-files/videoplayback.m4a` as 44.1 kHz 16-bit stereo, or a synthetic signal when the folder is missing. Median of 3 after 1 warmup. Ours works in memory. The `flac` rows include process launch and file I/O, and their byte counts include metadata and padding.

Run Thu Nov 5 13:20 GMT 2026, after the campaigns finished:

```text
audio: local-files/videoplayback.m4a, 44100 Hz, 2 channels, 16 bits, 60.0 s
each time: median of 3 after 1 warmup; rtf is audio seconds per second
encode level 0: 0.0737 s, rtf 814x, 4814454 frame bytes
encode level 5: 0.1337 s, rtf 449x, 2143624 frame bytes
encode level 8: 0.7765 s, rtf 77x, 2072664 frame bytes
decode level 5 frames: 0.0614 s, rtf 978x
flac -0: 0.0451 s, rtf 1329x, 4822883 file bytes
flac -5: 0.0606 s, rtf 990x, 2158639 file bytes
flac -8: 0.1332 s, rtf 450x, 2084103 file bytes
flac -d of its -5 file: 0.0388 s, rtf 1546x
```

Ours takes 1.6 times libFLAC's time at level 0, 2.2 times at level 5, 5.8 times at level 8, and 1.6 times to decode. Every level is far above real time. The end-to-end test above runs 90 s of stereo through demux, decode, encode at level 5, and mux in 0.41 s in release.

## App wiring

- `apps/desktop`: the FLAC container reader is registered, and the Rust FLAC codecs are registered before the platform ones. On macOS AudioToolbox also claims FLAC and would otherwise win, and its encoder's cookie is not a STREAMINFO. The export panel offers FLAC when the picture is off. `.flac` sound-only export writes 16-bit FLAC. A movie with FLAC sound is refused.
- `crates/export`: `(Flac, Flac)` writes through `FlacMux`. The mix is handed to FLAC as 16-bit samples, as it is to PCM.
- `crates/probe`: a FLAC file is audio, with its duration from STREAMINFO, or from the frames when STREAMINFO has no sample count.
- `crates/ffi`: the phone reads FLAC files with the Rust decoder ahead of the Apple one. The phone exports movies only, so it has no FLAC export.

## Gaps

- The engine opens mono and stereo audio only, so a FLAC file with more channels does not import, although this crate decodes it.
- Export writes 16-bit FLAC only.
- The encoder writes fixed blocking only.
- Levels 6 to 8 use partial and punch-out Tukey windows in place of libFLAC's `subdivide_tukey`.
- No SIMD and no threads.
- ID3v2 before `fLaC` is refused. Ogg FLAC and MP4 `dfLa` are not done.
- For a variable-block stream, the frame ordinal reported after a seek is approximate. The timestamps are exact.
- cargo-fuzz was not run. Tests were not run on Linux or Windows.

## Machine

Apple M5 Pro, 18 cores, arm64, macOS 27.0.1, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. `flac` 1.5.0, `metaflac` 1.5.0, ffmpeg 8.1.1.
