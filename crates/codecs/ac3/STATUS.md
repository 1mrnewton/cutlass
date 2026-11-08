# AC-3 and E-AC-3 support record

Branch `ac3-eac3`, worktree `/Users/mrnewton/dev/cutlass/cutlass-ac3-eac3`. Native safe Rust in `cutlass-ac3`. No `unsafe`. No bundled C, FFmpeg, AudioToolbox call, or AC-3 crate. Decode and encode are both in scope, and the two syntaxes stay distinct.

The specification is ATSC A/52:2018. AC-3 is the main document. E-AC-3 is Annex E. Annex D is the alternate bit stream information. Mix levels are the powers of two in Annex D, not `10^(dB/20)`.

Read, and not copied: `../pub-packages/ffmpeg/libavcodec/ac3dec.c`, `ac3enc.c`, `ac3.c`, `eac3dec.c`, `eac3_data.c`, and `../pub-packages/ffmpeg/libavformat/mov.c`, `riff.c`. GStreamer's `ac3parse` and `ac3dec` under `../pub-packages/gstreamer/subprojects` were the second implementation. FFmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg` and `/usr/bin/afconvert` are the oracles. They are not a fallback.

`../local-files` has no AC-3 or E-AC-3 file, and no 5.1 recording. Program material from that folder was resampled or decoded to PCM, encoded by this crate, and checked against FFmpeg and Apple. The files stay where they are.

## Limits

| Limit | Value | Error |
| --- | --- | --- |
| Syncframe | at most 4096 bytes | `Truncated` or `Malformed` |
| Syncframes in one access unit | 432 (six blocks of eight programs, eight dependents each) | `Limit` `syncframes per access unit` |
| Sample rate | 32 kHz, 44.1 kHz, 48 kHz. Zero at open means the first frame decides | `Unsupported` otherwise, including E-AC-3 reduced rates |
| Channels | at most 8 | `Unsupported` `more than 8 channels` |
| Program | `substreamid` 0 through 7 | `InvalidApi` above |
| `bsid` | 0 through 8 is AC-3, 11 through 16 is E-AC-3 | `Unsupported` for 9, 10, and above 16 |
| Enhanced coupling | refused | `Unsupported` |
| Empty packet | refused, decoder unchanged | `InvalidApi` |
| Encoder input | finite, within a few ulps of full scale, even channel count, same rate and layout | `NonFinite`, `OutOfRange`, `UnevenChannels`, `InvalidApi` |
| Encoder layout | mono through 5.1 with LFE. A side pair and a back pair together are refused | `Unsupported` |
| AC-3 bit rate | a Table 5.18 rate. A non-zero request snaps to the nearest. Zero uses the default | `Unsupported` when even the coarse frame does not fit |
| E-AC-3 bit rate | any positive rate. Zero uses the default | `Limit` when a frame still cannot be packed |
| E-AC-3 blocks | 1, 2, 3, or 6. AC-3 is 6 | `Unsupported` otherwise |

Default bit rates, kbps, for 1, 2, 3, 4, 5, and 5.1 channels: AC-3 96, 192, 256, 320, 384, 448. E-AC-3 64, 128, 192, 224, 256, 384.

The encoder delay is 256 samples, one half transform. At 48 kHz that is 5.333 ms. The first packet carries `dac3` or `dec3`, and every packet carries the gapless delay and trailing padding.

## AC-3

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Mono, stereo, 3/0, 2/1, 3/1, 2/2, 3/2, with and without LFE | Speaker order. Surrounds are side channels. The single surround is back center. LFE is its own channel | Same layouts. Coded order is L C R Ls Rs, then LFE | `each_channel_carries_its_own_tone_in_speaker_order`, `every_layout_decodes_in_order_with_its_lfe` | Verified |
| 32, 44.1, and 48 kHz | Yes | Yes | `every_fixture_decodes_with_its_rate_layout_and_length`, `ffmpeg_decodes_our_streams` | Verified |
| Table 5.18 frame sizes, including the lowest legal rate | Yes | Coarse exponents when the frame cannot hold a full-band adaptive set | `ac3_syncframes_follow_table_5_18`, `the_lowest_rates_still_code_every_layout` | Verified |
| Coupling, rematrixing, block switching, exponent strategies | Yes | Coupling and rematrixing. A transient takes short blocks. A steady tone does not | decoder fixtures, `a_transient_stays_quiet_before_the_attack`, FATE `monsters_inc` and `millers_crossing` | Verified |
| Phase flags and delta bit allocation | Parsed and applied. A block that omits `deltbaie` keeps the previous block's segments | Stereo coupling writes a phase flag when a band's channels are out of phase. The rate control does not choose delta segments. A block that carries segments is written, and later blocks in the frame keep that mask | `an_inverted_stereo_band_keeps_its_phase`, `a_delta_segment_matches_ffmpeg`. No FATE sample in this set isolates either tool | Both verified against FFmpeg. The delta check is one segment on a mono AC-3 frame, above 60 dB |
| Dialogue level, heavy compression, downmix levels, surround EX, production info | Exposed on `Metadata`. Line and heavy modes apply `dynrng` and `compr` when asked. The default is full range | `dialnorm`, `bsmod`, `dsurmod`, and Annex D mix codes. A block peaking above half scale sends `dynrng`, and the frame sends `compr`, both bringing the peak back to half scale | `metadata_reports_what_the_encoder_wrote`, `metadata_round_trips`, `dynamic_range_off_ignores_the_words_and_line_mode_applies_them`, `a_loud_block_is_cut_to_half_scale`, `a_loud_frame_carries_heavy_compression`, FATE line and heavy | Verified |
| Downmix | Lo/Ro uses `cmixlev` and `surmixlev`, else −4.5 dB and −6 dB. Lt/Rt uses 0.707 unless extended levels exist. Mono is half the Lo/Ro sum. LFE joins a stereo or mono downmix at `lfemixlevcod` minus 4.5 dB when that code exists | Writes the levels it was given | `downmix_follows_the_a52_equations_with_the_stream_levels`, `lo_ro_downmix_matches_ffmpeg`, `mono_downmix_is_half_the_lo_ro_sum` | Verified |
| Independent decode | FFmpeg on the fixtures, to the dither floor. Apple per channel, to the dither floor | Middle of the file above 60 dB SNR against FFmpeg, at every layout and rate. Apple above 40 dB on the speakers it renders | `every_fixture_matches_ffmpeg_to_the_dither_floor`, `every_fixture_matches_apple_to_the_dither_floor`, `ffmpeg_decodes_our_streams`, `apple_decodes_our_streams` | Verified, with the Apple exceptions below |
| Timing | Packet time is kept. A seek with one preroll frame restores the overlap. Gapless side data trims the delay and the padding | Decoded length equals the input. The signal lines up after the 256-sample delay | `timestamps_pass_through_and_every_sample_is_returned`, `after_a_seek_one_preroll_frame_restores_the_overlap`, `gapless_side_data_trims_the_delay_and_padding_exactly`, `decoded_length_matches_the_input_exactly` | Verified |
| Lifecycle | Empty packet, a bad frame, and a rate or layout change leave the decoder as it was. `reset` repeats. A second `drain` is empty | Illegal input adds nothing. `reset` starts a new stream. Cancel stops before the next frame | decoder lifecycle tests, `illegal_input_fails_and_adds_nothing`, `reset_starts_a_fresh_stream_and_drain_ends_it`, `cancel_stops_before_the_next_frame_and_reset_encodes_again` | Verified |
| Concealment | Not implemented. `concealed` stays false | n/a | none | Not implemented |

Apple's decoder is not a discrete-channel oracle for every mode. It renders 2/1 as left, a silent center, and right, and drops the back-center surround of 2/1 and 3/1. With LFE and a center or a single surround, it folds the center into the fronts and drops LFE. The Apple test skips those speakers. FFmpeg of the same streams keeps the channels, so the bitstream is the one we wrote.

FFmpeg disagrees with A/52 on block-switched blocks. On those FATE samples Apple is the reference, and it matches.

## E-AC-3

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Mono, stereo, 5.0, 5.1 | Same speaker order as AC-3 | Same layouts as AC-3, one independent substream | fixtures, `every_layout_decodes_in_order_with_its_lfe` | Verified |
| 7.1 dependent substream | The dependent's channels join the program. A location outside the eleven speakers is `Unsupported`. `Extension::Core` plays the independent substream and counts every dependent frame. A dependent frame before any independent frame is `Malformed` | One dependent frame carries the back pair (`chanmap` bit 6). The independent frame stays 5.1 with the side pair. The `dec3` record lists one dependent and `chan_loc` bit 1. AC-3 refuses the layout. The default rate is 384 kbit/s plus 128 kbit/s; a chosen rate is split five sevenths to the independent frame | FATE `the_great_wall_7.1`, `a_core_decode_drops_the_dependent_substream`, `each_program_is_its_own_substream_and_a_leading_dependent_is_refused`, `a_seven_point_one_program_keeps_each_speaker` | Decode verified. Encode keeps each speaker in this decoder, including after an ISOBMFF seek (`a_seven_point_one_movie_keeps_each_speaker_through_seek`). FFmpeg's acmod 7 layout is the back pair, so it plays the dependent backs in place of the independent surrounds |
| Programs in one access unit | `substreamid` selects one independent program. The other programs in the unit are ignored | One program, id 0 | `each_program_is_its_own_substream_and_a_leading_dependent_is_refused` | Decode verified. The encoder writes program 0 |
| 32, 44.1, and 48 kHz | Yes | Yes | fixtures, `ffmpeg_decodes_our_streams` | Verified |
| Reduced sample rates | Refused | Not written | `the_eac3_reduced_rates_are_unsupported` | Unsupported |
| 1, 2, 3, and 6 blocks | Yes | Yes. One block uses D45 exponents. Two blocks use D45 then D25. A frame that cannot reach eight SNR steps above the nominal offset gives bandwidth back until it can. The count may change from one syncframe to the next | `eac3_holds_any_bit_rate_at_any_block_count`, `every_eac3_block_count_decodes_with_ffmpeg`, `a_stream_may_change_its_block_count`. FATE `serenity` is one block per frame | Our encoder verified against FFmpeg for mono, stereo, and 5.1, including one access unit that switches among 6, 1, 3, and 2. An independent 2-block or 3-block file is not in the FATE set |
| Spectral extension and AHT | Rebuilt from the side data | Spectral extension below 48 kbit/s per full-band channel, copying sub-band 2 through a begin at sub-band 5 and out to sub-band 13, with the blend on the copied signal. A steady uncoupled six-block frame sends one hybrid transform per channel. A gain (`gaqmod` 1, 2, or 3) is used when it is shorter than a fixed mantissa and no coefficient is more than one step further off. Mode 3 packs gains 1, 2, and 4 into 5-bit groups when that mix is shorter than either one-bit mode. Coupling stays in charge at 48 kbit/s per channel and above, and whenever exponents are refreshed | FATE `csi_miami` (`spx_blocks` and `aht_channels`), `spectral_extension_matches_ffmpeg`, `a_steady_tone_uses_the_hybrid_transform`, `gain_adaptive_quantization_matches_ffmpeg` | Decode verified. Encodes verified against FFmpeg, including a mix whose frames carry a gain |
| Coupling and rematrixing | Yes. The coupling band structure is kept from the default and replaced when the frame sends one. A stereo phase flag negates the right channel in that band | The band structure is always sent (`cplbndstrce` 1), including when it matches the default. A frame that uses spectral extension stays on the long transform. Any other short block turns coupling off for the whole E-AC-3 frame and leaves the hybrid transform off, because those tools need one long transform in every block. An out-of-phase band sets the phase flag and sums that band with the right channel negated | `coupling_and_rematrixing_carry_the_low_rate_programs`, `ffmpeg_decodes_an_eac3_frame_that_starts_with_a_transient`, `a_transient_stays_quiet_before_the_attack`, `an_inverted_stereo_band_keeps_its_phase`, FATE `matrix2` | Verified |
| Enhanced coupling | Refused | Not written | the frame parser | Unsupported |
| Transient pre-noise | The pre-noise is cross-faded with a copy of earlier audio in that channel (Annex E 3.7). The copy may come from the previous 1536 samples. A correction that lies entirely in samples already returned is left as it was | Not chosen. A test can force channel 0 | `the_middle_of_the_window_is_a_copy_of_earlier_audio`, `a_source_before_this_frame_is_read_from_the_history`, `a_forced_transient_window_is_copied_into_the_frame` | Applied when the frame carries it |
| Program scale `pgmscl` | Applied to the channels of the substream that carries it, before the overlap is stored. Code 0 mutes. In 1+1 mode `pgmscl2` scales the second channel. `extpgmscl` stays on `Metadata` because this decoder does not mix a second program | Written when `program_scale_db` is set, −50 to +12 dB, or −51 for mute. AC-3 refuses it. A dependent frame does not carry the word | `a_program_scale_cuts_the_substream`, `compression_words_read_as_the_spec_tables` | Verified for the substream that carries the word |
| Dialogue level, mix levels, LFE mix | Exposed. The downmix uses them | `dialnorm` and `lfemixlevcod` round-trip | `metadata_round_trips` | Verified |
| Dynamic range | Line mode matches FFmpeg's `drc_scale`. Heavy mode applies each frame's `compr` word. A complete mix uses the last dependent frame's words (Annex E 3.8.5); a core decode keeps the independent frame's. FFmpeg's E-AC-3 path never sets `compression_exists`, so it is not the heavy-mode oracle | A block whose peak is above half scale sends `dynrng` to bring that peak to half scale. Later blocks reuse the word until the peak changes. The frame also sends `compr` for the same cut. On a 7.1 program the dependent frame carries the peak of the whole mix, and the independent frame keeps its own channels. The default playback policy ignores both words | `line_and_heavy_compression_match_ffmpeg`, `eac3_heavy_compression_applies_each_frames_compr_word`, `a_loud_block_is_cut_to_half_scale`, `a_loud_frame_carries_heavy_compression`, `a_complete_mix_uses_the_dependent_range_word` | Line mode and this decoder's heavy mode are verified. FFmpeg is the heavy-mode oracle for AC-3 only |
| Dependent dynamic range | With dependents, dynamics not off, and no downmix, the last dependent's words apply to the whole program. A downmix uses independent substream 0 | n/a | the decoder | Implemented. Not in the FATE DRC check |
| Independent decode | Same gates as AC-3 | Same gates as AC-3, including 32 kHz | the same tests | Verified, with the same Apple exceptions |
| Timing and lifecycle | Same as AC-3 | Same as AC-3 | the same tests | Verified |

## Program material

Four seconds from `../local-files`, encoded at the default rate, compared on the middle of the file. SNR is this decoder against the source, against FFmpeg (`-drc_scale 0`), and against Apple. The test requires FFmpeg and Apple above 30 dB and the source above 15 dB. One second of stereo silence at 48 kHz stays under an RMS of 1e-3 for both formats. The music clips were muxed to ISOBMFF, read back byte for byte, sought to one second, and decoded.

| Clip | Format | Rate | Source | FFmpeg | Apple |
| --- | --- | --- | --- | --- | --- |
| Speech, 48 kHz mono, from a 16 kHz WAV | AC-3 | 96 kbps | 34.7 dB | 48.8 dB | 50.0 dB |
| Speech | E-AC-3 | 64 kbps | 23.3 dB | 35.2 dB | 36.5 dB |
| Music, 48 kHz stereo | AC-3 | 192 kbps | 28.8 dB | 41.7 dB | 43.7 dB |
| Music | E-AC-3 | 128 kbps | 18.6 dB | 32.2 dB | 32.9 dB |
| Talk, 44.1 kHz stereo | AC-3 | 192 kbps | 52.2 dB | 67.3 dB | not printed |
| Talk | E-AC-3 | 128 kbps | 37.8 dB | 52.1 dB | 54.0 dB |
| Thunder, 48 kHz stereo | AC-3 | 192 kbps | 48.8 dB | 51.9 dB | 53.4 dB |
| Thunder | E-AC-3 | 128 kbps | 46.2 dB | 49.3 dB | 50.6 dB |

The 44.1 kHz AC-3 frame table makes that clip 192003 bit/s. Every other clip matched its frame's implied rate within 1 percent. Talk AC-3 was above the 60 dB FFmpeg gate, so that run did not print Apple; the locked test still requires Apple above 30 dB, and it passed. A single 1536-sample window can sit far under the file figure: speech E-AC-3 reached 5.0 dB and talk E-AC-3 reached −0.3 dB. Those windows are quiet. The speech window is RMS 0.001 on our decode, on FFmpeg, and on the source. The talk windows are digital silence, RMS 0. The ratio there is dither against dither. Every packet of those files still decodes with `-f eac3`. The music E-AC-3 file used to be rejected (`expacc` out of range) when coupling turned on after a short first block.

## Containers, playback, and export

| Path | AC-3 | E-AC-3 | Evidence | Status |
| --- | --- | --- | --- | --- |
| ISOBMFF | `ac-3` and `sac3`, object type 0xA5, `dac3` | `ec-3`, object type 0xA6, `dec3` | `ac3_and_eac3_keep_separate_sample_entries` | Verified |
| WAV | Microsoft tag 0x2000. The Dolby GUID is also read | No Microsoft tag. Written and read as the Dolby GUID FFmpeg writes, in WAVEFORMATEXTENSIBLE | `an_eac3_wave_round_trips_through_its_guid`, `the_dolby_ac3_guid_is_ac3` | Verified. A 0x2000 file that actually holds E-AC-3 stays labeled AC-3. The decoder still reads the syncframe |
| M4A export | Sound-only, `ftyp M4A`, config and a syncframe that decodes. The encoder delay is an edit | Same | `sound_alone_writes_ac3_and_eac3_and_reopens`, `a_real_recording_exports_as_ac3_and_eac3_and_plays_back` | Empty timeline, and two seconds of a 48 kHz stereo recording |
| WAV export | Not offered | Not offered | `write_sound` | Unsupported |
| Desktop import | Rust decoder and encoder, registered ahead of the platform | Same | `ac3_and_eac3_open_for_import_and_export`, `an_ac3_movie_opens_seeks_and_reopens`, `a_surround_movie_keeps_each_speaker_through_seek`, `a_real_recording_exports_as_ac3_and_eac3_and_plays_back` | Installed readers open, seek, and reopen a 1 kHz tone, a 5.1 tone (side surrounds and LFE), and an exported recording. The running window was not exercised |
| Phone | The same Rust codecs, ahead of the platform | Same | `the_phone_opens_ac3_and_eac3` | Verified at the factory |
| Platform fallback | AudioToolbox fourcc `ac-3`, 1536 samples a frame | fourcc `ec-3`. A shorter frame is the Rust codec's job | the platform tables | Registered behind the Rust codec. Not compared |
| Matroska, MPEG-TS, CAF | Not implemented. Those container crates are empty, or have no mapping | Same | none | Not present |

`Mp4Mux` writes an AC-3 or E-AC-3 gapless trim as one edit. `media_time` is the 256-sample delay, and the segment duration drops the trailing padding. Packet timestamps still include both, because that is the frame. Opus is unchanged: it already shortens its last packet and writes the pre-skip from the head. A file opened from disk skips the delay through that edit. The real-recording test imports two seconds of 48 kHz stereo music, exports both codecs, and checks playback at the start and at one second against the source. The best lag inside ±8 samples is 0, and the source SNR is above 15 dB. It skips when `../local-files` is missing, and fails instead when `AC3_REQUIRE_ORACLES=1`. The media-header duration still counts the priming and the padding.

## Robustness

`bounded_mutations_do_not_panic` runs on every `cargo test`: 2048 mutated frames from the fixtures and from both encoders, and 128 encoder rounds of legal and hostile input. A panic, a non-finite sample, or a legal frame that does not decode fails it.

cargo-fuzz is not used. `libfuzzer-sys` builds C through `cc`, which `deny.toml` bans.

A two-minute release campaign on 8 November 2026 found nothing. Legal rounds include 7.1, a 40 kbit/s mono frame, an 80 kbit/s stereo frame, a 96 kbit/s mono frame, peaks above half scale, and E-AC-3 frames of one, two, and three blocks. Short frames are stereo or fewer than seven channels, and they use the default rate when the drawn rate is below 96 kbit/s, so a 40 kbit/s frame stays at six blocks. 7.1 stays at six blocks. The one-block, two-block, and three-block stereo streams are also decoder seeds. Release turns overflow checks off. The 300-second run below is the test profile, where an overflow panics, and it is an earlier encoder. This crate's dev profile still compiles the library at opt-level 3.

```text
AC3_FUZZ_SECS=120 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 120 inputs 7617224 errors 7603785 accepted 13439 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 120 rounds 426452 refused 212748 encoded 213704 findings 0
test result: ok. 1 passed; 0 failed; finished in 120.00s
```

Earlier encoders, kept so the longer runs stay visible. The first of these is the six-block-only encoder:

```text
AC3_FUZZ_SECS=120 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 120 inputs 9141348 errors 9127049 accepted 14299 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 120 rounds 567989 refused 284327 encoded 283662 findings 0
test result: ok. 1 passed; 0 failed; finished in 120.00s

AC3_FUZZ_SECS=120 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 120 inputs 9430442 errors 9416165 accepted 14277 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 120 rounds 668080 refused 333925 encoded 334155 findings 0
test result: ok. 1 passed; 0 failed; finished in 120.00s

AC3_FUZZ_SECS=1800 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 1800 inputs 139183455 errors 138970023 accepted 213432 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 1800 rounds 9772334 refused 4882999 encoded 4889335 findings 0
test result: ok. 1 passed; 0 failed; finished in 1800.00s

AC3_FUZZ_SECS=300 cargo test -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 300 inputs 22217058 errors 22183167 accepted 33891 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 300 rounds 1556225 refused 777633 encoded 778592 findings 0
test result: ok. 1 passed; 0 failed; finished in 300.00s
```

## Performance

`cargo build --release -p cutlass-ac3 --bench throughput`, then the bench binary, on 8 November 2026, on the encoder that writes spectral extension, the hybrid transform, dynamic range words, and 7.1. Apple M5 Pro, 18 cores, 48 GiB, macOS 27.0.1, rustc 1.99.1, ffmpeg 8.1.1. Ten seconds at 48 kHz. Each figure is the median of three runs after one warmup. Cutlass times the codec only. ffmpeg times a new process, including startup. `/usr/bin/time -l` on the whole bench: 2.89 s real, maximum resident set 117,669,888 bytes, peak footprint 116,752,864 bytes.

```text
encoder delay 256 samples (5.333 ms at 48000 Hz)
rss at start 2176 KiB
AC-3 stereo 192 kbps
  encode 27.602 ms, 362x realtime
  decode 8.887 ms, 1125x realtime
  decode frame latency p50 27.6 us, p95 28.3 us, p99 28.8 us over 313 frames
  rss before a further 60 s of decode 18896 KiB, after 18912 KiB
E-AC-3 stereo 192 kbps
  encode 19.867 ms, 503x realtime
  decode 8.627 ms, 1159x realtime
  decode frame latency p50 27.0 us, p95 27.9 us, p99 28.5 us over 313 frames
  rss before a further 60 s of decode 18960 KiB, after 18960 KiB
AC-3 5.1 448 kbps
  encode 56.812 ms, 176x realtime
  decode 21.337 ms, 469x realtime
  decode frame latency p50 66.2 us, p95 68.6 us, p99 69.9 us over 313 frames
  rss before a further 60 s of decode 61632 KiB, after 61632 KiB
E-AC-3 5.1 384 kbps
  encode 44.851 ms, 223x realtime
  decode 19.726 ms, 507x realtime
  decode frame latency p50 61.0 us, p95 83.5 us, p99 205.2 us over 313 frames
  rss before a further 60 s of decode 63040 KiB, after 63056 KiB
E-AC-3 mono 40 kbps
  encode 7.762 ms, 1288x realtime
  decode 3.500 ms, 2857x realtime
  decode frame latency p50 10.8 us, p95 11.1 us, p99 11.2 us over 313 frames
  rss before a further 60 s of decode 63184 KiB, after 63184 KiB
E-AC-3 7.1 default
  encode 62.268 ms, 161x realtime
  decode 26.289 ms, 380x realtime
  decode frame latency p50 81.9 us, p95 84.0 us, p99 85.1 us over 313 frames
  rss before a further 60 s of decode 114848 KiB, after 114848 KiB
cancel of a 5 s stereo push, flag already set: 0.293 ms
ffmpeg ac3 stereo, process time including startup: encode 32.157 ms (311x), decode 25.271 ms (396x)
ffmpeg eac3 stereo, process time including startup: encode 31.651 ms (316x), decode 24.720 ms (405x)
```

Resident set grew as the bench moved from stereo to 5.1 and then to 7.1, which is the allocator keeping the larger working set, and did not grow across a further minute of decoding the same packets. The E-AC-3 5.1 p99 is one slow frame in the sample, not the median. Cancel with the flag already set returns after the sample check and before a frame is coded.

## Commands run

```text
AC3_REQUIRE_ORACLES=1 AC3_FATE_SAMPLES=/tmp/cutlass-ac3-eac3/fixtures/fate \
  cargo test --release -p cutlass-ac3 -- --skip decoder_and_encoder_campaigns
AC3_FATE_SAMPLES=/tmp/cutlass-ac3-eac3/fixtures/fate \
  cargo test --release -p cutlass-ac3 --test ac3 a_core_decode_drops_the_dependent_substream
  # lib 59 passed; integration 48 passed, 1 ignored (the long campaign)
cargo clippy -p cutlass-ac3 --all-targets -- -D warnings
AC3_FUZZ_SECS=120 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
the release throughput bench, after the E-AC-3 block-length change
AC3_FUZZ_SECS=1800 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
AC3_FUZZ_SECS=300 cargo test -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
the release throughput bench
cargo test -p cutlass-isobmff --lib ac3_and_eac3_keep_separate_sample_entries
cargo test -p cutlass-riff --lib an_eac3_wave_round_trips_through_its_guid
cargo test -p cutlass-riff --lib the_dolby_ac3_guid_is_ac3
cargo test -p cutlass-export --lib sound_alone_writes_ac3_and_eac3_and_reopens
cargo test -p cutlass --lib an_ac3_movie_opens_seeks_and_reopens
cargo test --release -p cutlass --lib a_surround_movie_keeps_each_speaker_through_seek
cargo test -p cutlass --lib ac3_and_eac3_open_for_import_and_export
cargo test -p cutlass-ffi --lib the_phone_opens_ac3_and_eac3
cargo test --release -p cutlass-isobmff --lib ac3_and_eac3_trim_is_an_edit_around_the_played_samples
AC3_REQUIRE_ORACLES=1 cargo test --release -p cutlass --lib a_real_recording_exports_as_ac3_and_eac3_and_plays_back
cargo test --release -p cutlass-export -p cutlass -p cutlass-ffi --lib ac3
  # this encoder: desktop 2, export 1, ffi 1, all passed
cargo clippy -p cutlass-ac3 -p cutlass-isobmff -p cutlass-riff \
  -p cutlass-platform-apple -p cutlass-export -p cutlass-ffi -p cutlass \
  --all-targets -- -D warnings
```

Linux and Windows were not run.
