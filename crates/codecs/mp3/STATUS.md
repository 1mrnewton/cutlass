# MP3 support record

Checkout: branch `mp3`, cut from `development` at `9ee1a701`. Native safe Rust in `cutlass-mp3`. No `unsafe`. No C, no FFmpeg, LAME, minimp3, Symphonia or other MP3 library in the build. FFmpeg, mpg123, LAME and afconvert are test oracles and fixture generators only.

Decode and encode of MPEG-1, MPEG-2 LSF and MPEG-2.5 Layer III are implemented and tested. The desktop app opens and exports MP3; the phone opens it. Perceptual encoder quality is pending human listening.

## Specifications

- ISO/IEC 11172-3:1993, Layer III: header, side information, scalefactors, Huffman tables (Annex B), requantization, stereo processing, alias reduction, IMDCT and windows, the polyphase synthesis and analysis filterbanks (Annex C), and the CRC-16.
- ISO/IEC 13818-3:1998, the lower sampling frequencies: one granule, the 9-bit `scalefac_compress` partitions and the intensity right-channel partitions, intensity positions with `intensity_scale`.
- MPEG-2.5 (8, 11.025, 12 kHz) is the Fraunhofer extension. It has no ISO text. The MPEG-2 rules apply, with the 8 kHz band tables FFmpeg and mpg123 use.
- ISO/IEC 11172-4 for the full-accuracy decoder bounds: RMS difference below `2^-15 / sqrt(12)` and no sample further than `2^-14`, at full scale 1.0.
- LAME's Xing/Info and LAME tag layout (`VbrTag.c`) for the encoder delay and padding, and Apple's `iTunSMPB` comment for the same trim.

FFmpeg 8.1.1 `libavcodec/mpegaudiodec_template.c` (the checkout in `../pub-packages/ffmpeg`) was read for its stereo processing and short-block IMDCT where the references disagree. GStreamer's MP3 elements (`gst-plugins-good/ext/mpg123`, `ext/lame`) wrap mpg123 and LAME, so they add no independent implementation. mpg123 1.33.5, LAME 3.100 and Apple's AudioToolbox were used as binaries. None of their source is in this crate. The Huffman, band and window tables are transcribed from the ISO annexes and checked by the oracle tests.

ISO/IEC 11172-4 conformance bitstreams were not obtained. Their role is taken by the LAME fixtures, the local media and the random-syntax streams below, all compared with two independent decoders.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| MPEG-1 32/44.1/48 kHz | Accept | Emit | `every_fixture_matches_*`, `every_sample_rate_and_layout_round_trips` | Verified |
| MPEG-2 LSF 16/22.05/24 kHz | Accept | Emit | Same | Verified |
| MPEG-2.5 8/11.025/12 kHz | Accept | Emit | Same | Verified |
| Mono, stereo, dual channel | Accept | Mono, joint stereo | Fixtures, `random_streams_match_ffmpeg_and_mpg123` | Verified. Dual channel decodes as two channels |
| Mid/side | Accept | Emit, per frame by energy | `stereo_32000_ms96.mp3`, encoder oracle tests | Verified |
| Intensity stereo, MPEG-1 and LSF, long, short and mixed, illegal positions | Accept | Not emitted | `src/synthetic.rs` streams vs ffmpeg and mpg123 | Verified, with the reference splits below |
| Long, start, short, stop blocks | Accept | Emit (attack detection) | Fixtures, synthetic streams, encoder oracle tests | Verified |
| Mixed blocks | Accept | Not emitted | Synthetic streams | Verified. MPEG-2.5 mixed blocks agree with ffmpeg only |
| `scfsi`, `preflag`, `scalefac_scale`, subblock gain, all Huffman tables, count1 table B | Accept | `scalefac_scale`, subblock gain, tables, count1 A/B emitted; `scfsi`, `preflag` not | Synthetic streams, encoder oracle tests | Verified |
| CRC | Checked; mismatch is `Malformed` | Not written | `a_crc_mismatch_is_malformed_and_main_data_is_not_covered` | Verified |
| Bit reservoir | Up to 511 bytes back; a frame whose main data is gone after a seek is skipped | Pending-frame FIFO, cap `min(511, 960 − frame)` (255 LSF) | `after_a_seek_the_preroll_frames_restore_identical_output`, `a_frame_after_reset_whose_main_data_is_gone_is_skipped_not_invented` | Verified |
| Free format | Accept (stride from the container) | Not emitted | `sine_44100_free400.mp3` vs mpg123 | Verified |
| Constant bitrate | n/a | Every legal rate, need-based split across channels | `constant_and_variable_rates_meet_their_quality_and_rate_gates` | Verified, rate within 0.2% |
| Variable bitrate V0–V9 | n/a | Noise target, smallest frame that holds it | Same | Verified |
| Gapless trim | `SideData::Gapless` honored; held tail, leading skip | `SideData::Gapless` on every packet, padding on the last | `gapless_timestamps_count_from_the_first_valid_sample`, `short_inputs_keep_their_length` | Verified |
| Xing/Info + LAME tag, `iTunSMPB` | Read by `cutlass-mp3-container` | Written by `Mp3Mux` from the encoder trim | `a_local_mp3_reencodes_and_reopens_in_every_reader` | Verified; see the reader table |
| Layer I, Layer II | `Unsupported` | n/a | `reserved_and_unsupported_header_fields_are_refused` | Unsupported by scope |
| ID3 metadata | Skipped by the container | `iTunSMPB` only | Container tests | By scope |

## Decoder correctness

Every comparison asserts the ISO/IEC 11172-4 bounds against the reference output, sample for sample, with the same length.

| Input | ffmpeg 8.1.1 (`mp3float`) | mpg123 1.33.5 | Test |
| --- | --- | --- | --- |
| Six LAME 3.100 fixtures (`tests/fixtures/README.md`) | Within bounds (free format: ffmpeg refuses above 320 kbps) | Within bounds | `every_fixture_matches_ffmpeg_*`, `every_fixture_matches_mpg123_*` |
| Local media under 2 MB (rain, three effects, thunder) | Within bounds | Not run | `local_media_matches_ffmpeg_within_the_iso_11172_4_bounds` |
| `rain_bedroom_night_30min.mp3`, 1798.6 s | Within bounds | Not run | `the_thirty_minute_local_file_matches_ffmpeg` (ignored; passed in release, 4.9 s) |
| Random valid streams, 9 rates × joint/stereo/mono/dual × 40 frames | Within bounds | Within bounds | `random_streams_match_ffmpeg_and_mpg123` |

### Where ffmpeg and mpg123 disagree

The random-syntax generator (`src/synthetic.rs`) writes standalone frames with `main_data_begin` 0, random block shapes, scalefactors, tables, regions, subblock gains, intensity bounds per window, and mode extensions. Four corners split the two references; each is checked against the reference that agrees, and the other one is confirmed to differ.

| Corner | This crate | ffmpeg | mpg123 | Test |
| --- | --- | --- | --- | --- |
| LSF intensity position from 16 to `2^slen − 2` (5-bit right-channel fields) | Legal; only `2^slen − 1` is illegal (ISO/IEC 13818-3) | Illegal from 16 (`sf_max = 16`) | Agrees with this crate (7 is its illegal value; never written) | `lsf_intensity_positions_from_16_match_mpg123` |
| MPEG-1 intensity position 8–15 | Illegal, like 7 | Agrees | Takes `tan(p·π/12)`; position 9 divides by zero (`inf`) | `mpeg1_positions_above_7_intensity_scfsi_and_mpeg25_mixed_blocks_match_ffmpeg` |
| `scfsi` on the intensity right channel | Positions from the reused scalefactors | Agrees | Differs | Same |
| MPEG-2.5 mixed blocks | 6 long bands, short from band 3 | Agrees | Differs | Same |

Two more rules keep a stream inside what ffmpeg decodes as specified: a short block only after a start or short block, and no pure short block after a mixed one. ffmpeg's short IMDCT writes the overlap of window 0 over lines 12–17 of the previous overlap before reading them (`mpegaudiodec_template.c`, `buf[4*(i + 6*2)]`), which drops a long block's tail under an illegal switch. This crate and mpg123 compute the overlap-add for any sequence.

## Encoder

Analysis: the 32-band polyphase analysis filter (the synthesis window divided by 32), the MDCT with normal, start, short and stop windows, and the alias butterflies; `analysis.rs` tests reconstruction through the decoder's own path below `1e-4`. Psychoacoustics: a model-2 style FFT model with tonality bounded by local spectral flatness, spreading, an adaptive absolute threshold, and attack detection on the short FFT windows; both channels share the block type. Quantization: per-band scalefactor search for the allowed noise, subblock gains, scalefactor lifting where noise is still above the threshold, Huffman table and region choice, count1 table choice. Stereo: mid/side by an energy test. Rate control: constant rate splits the frame budget by each channel's need and borrows from the reservoir; variable rate codes to the noise target and takes the smallest legal bitrate whose slot plus usable reservoir holds it. Lowpass by bitrate. `ENCODER_DELAY` is 528; the analysis and synthesis delay is 1057 = 528 + `DECODER_DELAY` 529.

`Mp3Encoder::open` maps `CodecParams`: `constant_rate` picks the nearest legal constant rate; otherwise the bitrate per channel pair maps to V0–V9, and bitrate 0 is V2. Export asks for constant rate.

### Independent decoders read what it writes

| Reader | Result | Test |
| --- | --- | --- |
| mpg123 1.33.5, gapless | Same length as the input; within ISO bounds of this crate's gapless decode | `the_reference_decoders_read_what_the_encoder_writes` |
| ffmpeg 8.1.1 | Full length (it ignores a LAME tag not named `LAME`/`Lavf`/`Lavc`); within ISO bounds of this crate's untrimmed decode | Same |
| lame 3.100 `--decode` | Exactly the input length; within 1.5 LSB of this crate's 16-bit decode | `lame_decodes_the_encoder_output_to_the_input_length` |
| Apple afinfo / afconvert | Reads the `iTunSMPB` comment: `N valid frames + 528 priming + P remainder`, sample-aligned output | `a_local_mp3_reencodes_and_reopens_in_every_reader` |

The LAME tag is signed `Cutlass`, not `LAME`, so ffmpeg (and Apple, without `iTunSMPB`) play the priming and padding. That gap stays: the tag does not claim another encoder.

### Gates locked by tests

On a synthetic noise- and click-heavy 44.1 kHz stereo signal (`music` in `tests/mp3/encoder.rs`), after the gapless trim: constant 64/128/192/320 kbps within 0.2% of the rate and SNR at least 20/26/30/32 dB (measured 23.8/29.0/32.2/33.6); V0/V2/V4/V6/V9 SNR at least 30/28/25/22/15 dB with falling rates (measured 319/245/192/148/71 kbps at 36.2/34.7/32.7/30.0/24.0 dB). Silence codes to silence in the smallest frames. Full-scale square waves and clicks stay finite and in range. Chunking does not change the stream.

### Against LAME 3.100

Ten 20 s 44.1 kHz stereo items: synthetic music, speech, a sweep, clicks, noise, and a tone left/noise right item, plus excerpts of the local thunder and rain and two talk recordings (`expo`, `fireship`). LAME through ffmpeg's `libmp3lame` (`-b:a 128k`, `-q:a 2`). Both decoded by this crate with the gapless trim. NMR is a crude Bark-band noise-to-mask ratio (lower is better), not a perceptual model; SNR rewards waveform matching, which LAME does not aim for. These are measurements of a scratch evaluator, not listening results.

Constant 128 kbps (both at 128.4–128.8 kbps; clicks/noise/stereo/sweep 134.5 vs 134.0):

| Item | Ours SNR | Ours NMR | LAME SNR | LAME NMR |
| --- | --- | --- | --- | --- |
| music | 38.9 | −4.34 | 25.7 | −3.84 |
| speech | 38.0 | −11.83 | 25.9 | −11.81 |
| thunder | 45.8 | −15.90 | 26.0 | −15.69 |
| rain | 13.4 | 4.23 | 17.2 | 4.94 |
| expo | 44.1 | −6.27 | 26.0 | −6.80 |
| fireship | 38.0 | −3.84 | 25.4 | −4.38 |
| clicks | 4.6 | 6.06 | 3.3 | 6.41 |
| noise | 4.0 | 4.98 | 3.1 | 5.79 |
| stereo | 22.4 | 8.56 | 19.4 | 1.55 |
| sweep | 47.0 | −9.68 | 26.0 | −11.61 |

V2:

| Item | Ours kbps | Ours NMR | LAME kbps | LAME NMR |
| --- | --- | --- | --- | --- |
| music | 133.8 | −5.24 | 120.4 | −4.94 |
| speech | 89.8 | −7.24 | 96.5 | −7.41 |
| thunder | 100.4 | −12.82 | 97.1 | −15.03 |
| rain | 208.7 | −2.81 | 181.7 | 1.34 |
| expo | 151.3 | −6.73 | 105.2 | −4.87 |
| fireship | 134.1 | −6.10 | 129.8 | −5.97 |
| clicks | 183.6 | −3.59 | 134.5 | 1.73 |
| noise | 224.3 | −2.35 | 191.7 | 2.12 |
| stereo | 147.4 | 0.62 | 140.7 | −0.61 |
| sweep | 52.3 | −3.29 | 126.5 | −10.99 |

V2 total rate: 1425.7 kbps against LAME's 1324.1 (+7.7%). Worst cases: the tone-left/noise-right item at 128 kbps (NMR 8.56 against 1.55: noise from the right channel's need spreads into the left's low bands), the sweep at V2 (under-coded at 52 kbps), and thunder at V2 (−12.8 against −15.0). The evaluator's `lag` readout on the stereo item reports a nine-period alias of the 440 Hz tone; every metric is computed at lag 0, and the gapless length is checked separately by the tests above.

Perceptual quality: pending human listening. No listening test was run.

## Lifecycle

Decoder: an empty packet is `InvalidApi` and changes nothing; a failed push leaves the decoder as it was; drain twice returns nothing the second time; without gapless side data every sample is returned; reset decodes the same samples again; a sample-rate or channel change mid-stream is `Malformed`; preroll frames after a seek restore identical output. Encoder: push after drain is refused until reset; a refused push (rate, layout, non-finite, out of range, uneven channels) adds nothing; 16-bit input codes as its float value; the cancel flag stops between frames; an empty stream drains to nothing.

## Robustness

- `every_truncation_of_every_frame_fails_without_harm`, `side_information_out_of_range_is_malformed`, `reserved_and_unsupported_header_fields_are_refused`, `random_and_mutated_frames_never_panic`.
- `tests/mp3/fuzz.rs`, in-process, not libFuzzer (cargo-fuzz is not installed). Decoder seeds mutate runs of up to 12 consecutive frames from the fixtures and from encoder output at every rate (bit flips, truncation, header and side-information rewrites, splices, random bytes behind a sync word, hostile gapless values, random timestamps, reset and drain mid-stream) and require finite output and no more audible samples than frames pushed. One seed in 16 also runs the encoder with random rate, layout, constant or variable mode, signal, chunking, refused pushes and reset, and requires every stream to decode to its input length. Seed step `0x9E3779B97F4A7C15` from `0x00C0FFEE`. `a_short_campaign_never_panics` runs 2 s on every test run. The long run, `MP3_FUZZ_SECS=600 cargo test --release -p cutlass-mp3 --test mp3 -- --ignored --nocapture campaign`, is recorded under Performance.

## Performance

`cargo bench -p cutlass-mp3 --bench speed`, Apple M5 Pro, release, one thread, best of 3, packets and PCM in memory. ffmpeg (`-threads 1`) and lame times include process launch and file reading. 5 Nov 2026; the 600 s fuzz campaign was running on another core.

| File | Decode, this crate | Decode, ffmpeg | Encode CBR 128 | LAME CBR 128 | Encode V2 | LAME V2 |
| --- | --- | --- | --- | --- | --- | --- |
| `rain_bedroom_night_30min.mp3`, 1798.6 s, 44.1 kHz stereo | 3.01 s (597×) | 1.24 s (1453×) | 30.8 s (58×) | 12.9 s (139×) | 22.3 s (81×) | 11.3 s (159×) |
| Seamless rain, 30 s, 48 kHz | 565× | 530× | 55× | 138× | 77× | 156× |
| thunder, 10 s | 546× | 308× | 88× | 160× | 96× | 193× |
| three effects, 5 s each | 598–622× | 170–174× | 50–69× | 113–147× | 69–84× | 147–176× |

The decoder runs at about 600× real time, 2.4× slower than ffmpeg on the long file. The encoder runs at 50–96× real time, about 2.2× slower than LAME. Memory was not measured.

## Not done

- ISO/IEC 11172-4 conformance bitstreams: not obtained.
- The encoder never writes intensity stereo, mixed blocks, `scfsi` or `preflag`; those decoder paths are checked with the random-syntax streams only.
- Encoder quality is below LAME on tonal-plus-noise stereo and on pure sweeps at V2, and V2 spends 7.7% more bits over the ten items. Human listening has not been done.
- ffmpeg plays our files with the priming and padding, because it trusts only a LAME-named tag.
- Phone export is movie-only (H.264/AAC), so the encoder is registered on the desktop, not the phone.
- Linux was cross-linted with clippy, not run. Windows was not built.
