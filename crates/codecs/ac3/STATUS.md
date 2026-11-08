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
| Enhanced coupling in a short block | refused | `Unsupported` `enhanced coupling in a short block` |
| Empty packet | refused, decoder unchanged | `InvalidApi` |
| Encoder input | finite, within a few ulps of full scale, even channel count, same rate and layout | `NonFinite`, `OutOfRange`, `UnevenChannels`, `InvalidApi` |
| Encoder layout | mono through 5.1 with LFE. E-AC-3 also takes 7.1, a side pair plus a back pair. AC-3 refuses that layout, and any other mix of the two pairs is refused | `Unsupported` |
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
| Enhanced coupling | A long block is rebuilt from the coupling channel, the amplitudes, and the angles (Annex E 3.5). The first block that omits the band structure uses Table E2.13. The previous frame's last coupling spectrum is kept for the next frame, and `reset` drops it. A short block on a coupled channel is refused. Angle interpolation follows the relative-index reading of 3.5.5.3: two even bands pass through each band center, and a step across ±1 takes the short arc. Chaos code 7 scales a steady, non-first channel by 0.62 and leaves a transient and the first channel alone. The random sequence is this decoder's linear congruential generator; A/52 does not name one. A steady channel keeps its draw, and a transient draws again each block. When spectral extension is also on, the coupling end is omitted and lands on the first extension bin for every `spxbegf`. Line and heavy dynamics scale the coupling spectrum by the block's `dynrng` or `compr` gain before the coupled bins are rebuilt. FFmpeg does not decode the tool | Not written. `ecplinu` stays 0, so existing streams stay decodable by FFmpeg and Apple | `enhanced_coupling_puts_the_same_signal_in_both_channels`, `enhanced_coupling_in_a_short_block_is_refused`, `enhanced_coupling_takes_the_dynamic_range_gain`, `a_steady_coefficient_keeps_its_value_at_zero_angle`, `interpolation_holds_a_constant_angle_and_crosses_each_band_center`, `interpolation_takes_the_short_arc_across_the_branch_cut`, `chaos_scales_only_a_steady_channel_that_is_not_first`, `a_transient_draws_a_new_angle_each_block_and_a_steady_channel_does_not`, `spectral_extension_ends_enhanced_coupling_on_its_first_bin` | Self-consistency only: a synthetic two-block frame, and the angle and chaos formulas checked against Annex E 3.5.5. FFmpeg does not decode the tool, Apple was not tried on it, and there is no external sample, so nothing independent confirms this decode. The last block of each syncframe departs from the spec (see the limit under Readiness) |
| Transient pre-noise | The pre-noise is cross-faded with a copy of earlier audio in that channel (Annex E 3.7). The copy may come from the previous 1536 samples. A correction that lies entirely in samples already returned is left as it was | The first detected transient in each full-band channel is named, at 64-sample resolution, with a time-scaling length of 0, when that replacement already matches the source. The words are left out when the syncframe cannot hold them. A test can replace channel 0 | `the_middle_of_the_window_is_a_copy_of_earlier_audio`, `a_source_before_this_frame_is_read_from_the_history`, `a_forced_transient_window_is_copied_into_the_frame`, `a_transient_stays_quiet_before_the_attack` | Applied. FFmpeg skips the tool, so a stream that carries it is not sample-identical there |
| Program scale `pgmscl` | Parsed and kept on `Metadata` as `program_scale_db` (and `pgmscl2`, `extpgmscl`), not applied. It is mixing metadata: FFmpeg skips the word and Apple ignores it, so a stream with the mute code (raw 0, stored as −51) plays at full level here as it does there. `program_gain` turns the value into a gain for a mixer that wants it | Written when `program_scale_db` is set, −50 to +12 dB, or −51 for mute. AC-3 refuses it. A dependent frame does not carry the word | `a_program_scale_is_kept_and_not_applied`, `compression_words_read_as_the_spec_tables` | Verified: −6 dB and the mute code decode at full level, and match FFmpeg above 60 dB |
| Dialogue level, mix levels, LFE mix | Exposed. The downmix uses them | `dialnorm` and `lfemixlevcod` round-trip | `metadata_round_trips` | Verified |
| Dynamic range | Line mode matches FFmpeg's `drc_scale`. Heavy mode applies each frame's `compr` word. A complete mix uses the last dependent frame's words (Annex E 3.8.5); a core decode keeps the independent frame's. FFmpeg's E-AC-3 path never sets `compression_exists`, so it is not the heavy-mode oracle | By default no `dynrng` or `compr` is written, as FFmpeg's encoder writes none, so a player applying line-mode dynamic range does not turn mastered audio down. With `EncoderMetadata::compression` on, a block whose peak is above half scale sends `dynrng` to bring that peak to half scale. Later blocks reuse the word until the peak changes. The frame also sends `compr` for the same cut. On a 7.1 program the dependent frame carries the peak of the whole mix, and the independent frame keeps its own channels. The default playback policy ignores both words | `line_and_heavy_compression_match_ffmpeg`, `eac3_heavy_compression_applies_each_frames_compr_word`, `a_loud_program_carries_no_range_words_by_default`, `a_loud_block_is_cut_to_half_scale`, `a_loud_frame_carries_heavy_compression`, `a_complete_mix_uses_the_dependent_range_word` | Line mode and this decoder's heavy mode are verified. FFmpeg is the heavy-mode oracle for AC-3 only |
| Dependent dynamic range | With dependents, dynamics not off, and no downmix, the last dependent's words apply to the whole program. A downmix uses independent substream 0 | n/a | the decoder | Implemented. Not in the FATE DRC check |
| Independent decode | Same gates as AC-3 | Same gates as AC-3, including 32 kHz | the same tests | Verified, with the same Apple exceptions |
| Timing and lifecycle | Same as AC-3 | Same as AC-3 | the same tests | Verified |

## Program material

Four seconds from `../local-files`, encoded at the default rate, compared on the middle of the file. SNR is this decoder against the source, against FFmpeg (`-drc_scale 0`), and against Apple. The test requires FFmpeg and Apple above 30 dB and the source above 15 dB. One second of stereo silence at 48 kHz stays under an RMS of 1e-3 for both formats. The music clips were muxed to ISOBMFF, read back byte for byte, sought to one second, and decoded.

| Clip | Format | Rate | Source | FFmpeg | Apple |
| --- | --- | --- | --- | --- | --- |
| Speech, 48 kHz mono, from a 16 kHz WAV | AC-3 | 96 kbps | 34.7 dB | 48.8 dB | 50.0 dB |
| Speech | E-AC-3 | 64 kbps | 21.4 dB | 34.1 dB | 40.1 dB |
| Music, 48 kHz stereo | AC-3 | 192 kbps | 28.8 dB | 41.7 dB | 43.7 dB |
| Music | E-AC-3 | 128 kbps | 19.8 dB | 34.5 dB | 32.1 dB |
| Talk, 44.1 kHz stereo | AC-3 | 192 kbps | 52.2 dB | 67.3 dB | 68.6 dB |
| Talk | E-AC-3 | 128 kbps | 37.7 dB | 52.0 dB | 54.1 dB |
| Thunder, 48 kHz stereo | AC-3 | 192 kbps | 48.8 dB | 52.1 dB | 53.6 dB |
| Thunder | E-AC-3 | 128 kbps | 45.3 dB | 49.0 dB | 50.3 dB |

Stereo white noise and one impulse, one second at 48 kHz and the default rate, were measured the same day. Noise against the source is 3.6 dB for AC-3 and 1.4 dB for E-AC-3. FFmpeg's encoder on the same samples is 3.6 dB and 1.6 dB. An impulse decoded from this encoder matches FFmpeg's decode at 71.0 dB (AC-3) and 67.7 dB (E-AC-3). `noise_and_an_impulse_match_ffmpeg` keeps this encoder within 1 dB of FFmpeg's encoder against the source, and the impulse above 50 dB against FFmpeg's decode. FFmpeg's decode of the noise itself is not sample-close: unallocated bins are dithered by different generators, and an E-AC-3 transient is skipped by FFmpeg.

The same gate on one second of 5.1, layout `5.1(side)`, default rates 448 and 384 kbit/s, was measured the same day. Noise against the source is 2.3 dB for AC-3 and 1.7 dB for E-AC-3. FFmpeg's encoder is 2.6 dB and 2.1 dB. The impulse against the source is 2.3 dB and 3.7 dB, against FFmpeg's encoder 3.3 dB and 3.1 dB, and against FFmpeg's decode of this stream 71.7 dB and 65.6 dB. `surround_noise_and_an_impulse_match_ffmpeg` uses that gate.

E-AC-3 stereo at 128 kbit/s starts coupling at sub-band 3. The per-channel table waited until about 10 kHz, and the music clip was 18.6 dB against the source. Sub-band 3 raises that to 19.5 dB and FFmpeg's decode of the same stream from 31.7 dB to 34.1 dB. Apple stays at 32.0 dB. Thunder at the same rate loses 0.5 dB against the source. The reference encoder starts at sub-band 1, which scored 20.7 dB against the source and 27.9 dB against Apple, under the 30 dB floor.

The clip table and the stereo noise and impulse figures were measured again on 8 November 2026, after the encoder stopped writing `dynrng` and `compr` by default. The 5.1 figures did not change. Before that, these file figures were measured after block switching, gain-adaptive quantization, and transient pre-noise. The 128 kbit/s coupling start was measured the same day, after that. The 44.1 kHz AC-3 frame table makes that clip 192003 bit/s. Every other clip matched its frame's implied rate within 1 percent. A single 1536-sample window can sit far under the file figure: on an earlier encoder, speech E-AC-3 reached 5.0 dB and talk E-AC-3 reached −0.3 dB. Those windows are quiet. The speech window is RMS 0.001 on our decode, on FFmpeg, and on the source. The talk windows are digital silence, RMS 0. The ratio there is dither against dither. Every packet of those files still decodes with `-f eac3`. The music E-AC-3 file used to be rejected (`expacc` out of range) when coupling turned on after a short first block.

## Containers, playback, and export

| Path | AC-3 | E-AC-3 | Evidence | Status |
| --- | --- | --- | --- | --- |
| ISOBMFF | `ac-3` and `sac3`, object type 0xA5, `dac3` | `ec-3`, object type 0xA6, `dec3` | `ac3_and_eac3_keep_separate_sample_entries` | Verified |
| WAV | Microsoft tag 0x2000. The Dolby GUID is also read | No Microsoft tag. Written and read as the Dolby GUID FFmpeg writes, in WAVEFORMATEXTENSIBLE | `an_eac3_wave_round_trips_through_its_guid`, `the_dolby_ac3_guid_is_ac3` | Verified. A 0x2000 file that actually holds E-AC-3 stays labeled AC-3. The decoder still reads the syncframe |
| M4A export | Sound-only, `ftyp M4A`, config and a syncframe that decodes. The encoder delay is an edit | Same | `sound_alone_writes_ac3_and_eac3_and_reopens`, `a_real_recording_exports_as_ac3_and_eac3_and_plays_back` | Empty timeline, and two seconds of a 48 kHz stereo recording |
| WAV export | Not offered | Not offered | `write_sound` | Unsupported |
| Desktop import | Rust decoder and encoder, registered ahead of the platform | Same | `ac3_and_eac3_are_coded_by_the_rust_codec_ahead_of_the_platform`, `an_ac3_movie_opens_seeks_and_reopens`, `a_surround_movie_keeps_each_speaker_through_seek`, `a_bad_ac3_frame_is_skipped_and_the_rest_plays`, `a_real_recording_exports_as_ac3_and_eac3_and_plays_back`, `the_export_dialog_names_its_file_by_what_it_writes_and_reads_a_typed_bitrate` | Installed readers open, seek, and reopen a 1 kHz tone, a 5.1 tone (side surrounds and LFE), and an exported recording. A CRC-failed syncframe in the middle of a sweep is skipped. Its time is silence, and the sweep after it matches the intact file sample for sample, in one window and in 1024-sample windows. The export dialog offers both codecs: sound alone is an M4A, and with a picture they are the movie's audio. The running window was not exercised |
| Phone | The same Rust codecs, ahead of the platform | Same | `the_phone_codes_ac3_and_eac3_with_the_rust_codec` | The factory's packets and samples match the Rust codec. AudioToolbox's decoder, opened alone, does not |
| Platform fallback | AudioToolbox fourcc `ac-3`, 1536 samples a frame | fourcc `ec-3`. The description stays at six blocks, and a syncframe asks for its own `numblkscod` length | the platform tables | Registered behind the Rust codec. Not compared |
| Matroska, MPEG-TS, CAF | Not implemented. Those container crates are empty, or have no mapping | Same | none | Not present |

`Mp4Mux` writes an AC-3 or E-AC-3 gapless trim as one edit. `media_time` is the 256-sample delay, and the segment duration drops the trailing padding. Packet timestamps still include both, because that is the frame. Opus is unchanged: it already shortens its last packet and writes the pre-skip from the head. A file opened from disk skips the delay through that edit. The real-recording test imports two seconds of 48 kHz stereo music, exports both codecs, and checks playback at the start and at one second against the source. The best lag inside ±8 samples is 0, and the source SNR is above 15 dB. It skips when `../local-files` is missing, and fails instead when `AC3_REQUIRE_ORACLES=1`. The media-header duration still counts the priming and the padding. A CRC-failed syncframe is still an error from the decoder. The engine skips that packet, up to sixteen in one pull, and keeps the rest of the file. The skipped time plays as silence, so the sound after it stays on its own timestamps (`a_refused_audio_packet_is_silence_and_later_sound_keeps_its_time` in the engine).

## Robustness

`bounded_mutations_do_not_panic` runs on every `cargo test`: 2048 mutated frames from the fixtures and from both encoders, and 128 encoder rounds of legal and hostile input. A panic, a non-finite sample, or a legal frame that does not decode fails it.

cargo-fuzz is not used. `libfuzzer-sys` builds C through `cc`, which `deny.toml` bans.

A two-minute release campaign on 8 November 2026 found nothing. This binary writes gain-adaptive quantization and transient pre-noise when the drawn audio calls for them. Legal rounds include 7.1, a 40 kbit/s mono frame, an 80 kbit/s stereo frame, a 96 kbit/s mono frame, peaks above half scale, and E-AC-3 frames of one, two, and three blocks. Short frames are stereo or fewer than seven channels, and they use the default rate when the drawn rate is below 96 kbit/s, so a 40 kbit/s frame stays at six blocks. 7.1 stays at six blocks. The one-block, two-block, and three-block stereo streams are also decoder seeds. Release turns overflow checks off. The 300-second run below is the test profile, where an overflow panics, and it is an earlier encoder. This crate's dev profile still compiles the library at opt-level 3.

```text
AC3_FUZZ_SECS=120 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 120 inputs 8270929 errors 8256299 accepted 14630 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 120 rounds 449260 refused 224272 encoded 224988 findings 0
test result: ok. 1 passed; 0 failed; finished in 120.01s
```

Earlier encoders, kept so the longer runs stay visible. The first of these is the short-frame encoder from before gain-adaptive quantization and transient pre-noise. The next is the six-block-only encoder:

```text
AC3_FUZZ_SECS=120 cargo test --release -p cutlass-ac3 --test ac3 -- --ignored campaign --nocapture
ac3 decoder campaign: seed 0xac3dec0d seconds 120 inputs 7617224 errors 7603785 accepted 13439 findings 0
ac3 encoder campaign: seed 0xac3e0c0d seconds 120 rounds 426452 refused 212748 encoded 213704 findings 0
test result: ok. 1 passed; 0 failed; finished in 120.00s

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

`cargo build --release -p cutlass-ac3 --bench throughput`, then the bench binary, on 8 November 2026, on the encoder that writes spectral extension, the hybrid transform, gain-adaptive quantization, dynamic range words, block switching, transient pre-noise, and 7.1. Apple M5 Pro, 18 cores, 48 GiB, macOS 27.0.1, rustc 1.99.1, ffmpeg 8.1.1. Ten seconds at 48 kHz. Each figure is the median of three runs after one warmup. Cutlass times the codec only. ffmpeg times a new process, including startup. `/usr/bin/time -l` on the whole bench: 3.66 s real, maximum resident set 117,751,808 bytes, peak footprint 116,785,632 bytes. The run had 18,006 involuntary context switches, so the tails include scheduler noise. An earlier encoder's E-AC-3 5.1 p99 of 205 µs did not repeat.

```text
encoder delay 256 samples (5.333 ms at 48000 Hz)
rss at start 2208 KiB
AC-3 stereo 192 kbps
  encode 27.858 ms, 359x realtime
  decode 9.469 ms, 1056x realtime
  decode frame latency p50 27.6 us, p95 34.6 us, p99 35.5 us over 313 frames
  rss before a further 60 s of decode 19136 KiB, after 19152 KiB
E-AC-3 stereo 192 kbps
  encode 20.181 ms, 496x realtime
  decode 9.093 ms, 1100x realtime
  decode frame latency p50 27.0 us, p95 34.1 us, p99 35.1 us over 313 frames
  rss before a further 60 s of decode 19216 KiB, after 19216 KiB
AC-3 5.1 448 kbps
  encode 57.350 ms, 174x realtime
  decode 22.334 ms, 448x realtime
  decode frame latency p50 67.9 us, p95 76.9 us, p99 82.7 us over 313 frames
  rss before a further 60 s of decode 61968 KiB, after 61968 KiB
E-AC-3 5.1 384 kbps
  encode 45.443 ms, 220x realtime
  decode 20.314 ms, 492x realtime
  decode frame latency p50 61.9 us, p95 71.0 us, p99 79.8 us over 313 frames
  rss before a further 60 s of decode 62992 KiB, after 62992 KiB
E-AC-3 mono 40 kbps
  encode 7.864 ms, 1272x realtime
  decode 3.783 ms, 2644x realtime
  decode frame latency p50 11.3 us, p95 14.6 us, p99 15.1 us over 313 frames
  rss before a further 60 s of decode 63136 KiB, after 63136 KiB
E-AC-3 7.1 default
  encode 63.593 ms, 157x realtime
  decode 27.531 ms, 363x realtime
  decode frame latency p50 83.8 us, p95 92.7 us, p99 103.3 us over 313 frames
  rss before a further 60 s of decode 114960 KiB, after 114960 KiB
cancel of a 5 s stereo push, flag already set: 0.282 ms
ffmpeg ac3 stereo, process time including startup: encode 33.363 ms (300x), decode 24.251 ms (412x)
ffmpeg eac3 stereo, process time including startup: encode 33.608 ms (298x), decode 25.953 ms (385x)
```

Resident set grew as the bench moved from stereo to 5.1 and then to 7.1, which is the allocator keeping the larger working set, and did not grow across a further minute of decoding the same packets. Cancel with the flag already set returns after the sample check and before a frame is coded.

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
cargo test --release -p cutlass --lib ac3_and_eac3_are_coded_by_the_rust_codec_ahead_of_the_platform
cargo test --release -p cutlass-ffi --lib the_phone_codes_ac3_and_eac3_with_the_rust_codec
cargo test --release -p cutlass --lib a_bad_ac3_frame_is_skipped_and_the_rest_plays
cargo test --release -p cutlass-ac3 --test ac3 every_fixture_matches_the_committed_ffmpeg_pcm
cargo test --release -p cutlass-ac3 --test ac3 surround_noise_and_an_impulse_match_ffmpeg
cargo clippy -p cutlass-ac3 -p cutlass-isobmff -p cutlass-riff -p cutlass-export -p cutlass-ffi --all-targets --target x86_64-unknown-linux-gnu -- -D warnings
cargo test --release -p cutlass-isobmff --lib ac3_and_eac3_trim_is_an_edit_around_the_played_samples
AC3_REQUIRE_ORACLES=1 cargo test --release -p cutlass --lib a_real_recording_exports_as_ac3_and_eac3_and_plays_back
cargo test --release -p cutlass-export -p cutlass -p cutlass-ffi --lib ac3
  # this encoder: desktop 2, export 1, ffi 1, all passed
cargo clippy -p cutlass-ac3 -p cutlass-isobmff -p cutlass-riff \
  -p cutlass-platform-apple -p cutlass-export -p cutlass-ffi -p cutlass \
  --all-targets -- -D warnings
```

Linux and Windows were not run.

## Readiness

The declared target is in place on `ac3-eac3` at `/Users/mrnewton/dev/cutlass/cutlass-ac3-eac3`. Decode and encode of both syntaxes are native safe Rust in `cutlass-ac3`. ISOBMFF and WAV open them, M4A export writes them, and the desktop export dialog offers both.

`cargo test --release -p cutlass-ac3` on 8 November 2026: 81 library tests passed, 63 integration tests passed, and the thirty-minute campaign stayed ignored. That campaign was already run, twice for two minutes and once for thirty minutes, with no findings. The fixture check against the committed FFmpeg PCM passed in that same run, as did the live FFmpeg and Apple oracles and the program-material files.

Enhanced coupling at a syncframe end departs from A/52:2018 Annex E 3.5.5.1. The spec rebuilds the non-aliased coupling channel from the previous, current, and next blocks, and the next block of a syncframe's last block is the first block of the next syncframe; it is zero only when that block does not use enhanced coupling. This decoder renders a syncframe's audio before it has the next one, so it rebuilds the last block with the next block set to zero. That is every sixth block of a six-block frame, every third of a three-block frame, every second of a two-block frame, and every block of a one-block frame. At angle zero the coupled coefficients come out the same; at any other angle the imaginary part, and so the coupled bins of that block, differ from the spec. Following the spec needs output held back by one block, 256 samples, across syncframes, in step for every substream of the program and with transient pre-noise processing moved with it. That was not done. No stream this crate writes uses the tool, and FFmpeg does not decode it, so this decode rests on self-consistency checks only.

What this target does not do: E-AC-3 reduced sample rates, enhanced-coupling encode (`ecplinu` stays 0; FFmpeg does not decode the tool), a raw `.ac3` reader, Matroska, MPEG-TS, and CAF. WAV export stays PCM. The movie-header duration still counts priming and padding. The codec still returns an error for a bad syncframe; the engine skips up to sixteen of those and keeps going. Linux and Windows were not run as product shells.
