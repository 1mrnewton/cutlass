# PCM support record

Branch `pcm`. Native safe Rust in `cutlass-pcm`. No `unsafe`. No bundled C, FFmpeg, or GStreamer. Packet bytes are the samples a container already framed. This crate does not parse WAV, AIFF, or CAF.

Decode and encode are both in scope. s16, f32, and f64 encode are lossless for the samples the mix stores. s24 and s32 encode are refused: the mix has no buffer of that width, and quantizing would not return the original samples.

## Specifications and sources

PCM inside a WAV file is the layout this crate follows. The container is another crate. The rules used here:

- Microsoft `WAVEFORMATEX` format `0x0001`: 8-bit is unsigned offset-binary; 16, 24, and 32-bit are signed two's complement, little-endian, interleaved. Block align is channels times bytes per sample. Playback consumes a multiple of that block.
- Format `0x0003` is IEEE float, full scale 1.0, 32 or 64 bits.
- Format `0xFFFE` carries a subformat. Valid bits are informational. Microsoft documents them as left-justified. This crate does not guess a separate 24-in-32 coding. Packed s24 is three bytes. Left-justified 24-in-32 is ordinary s32.
- Formats `0x0006` and `0x0007` (A-law and μ-law) are refused by name.

Read before the implementation, and not copied:

- `../pub-packages/ffmpeg/libavcodec/pcm.c`. s16, s32, f32, and f64 are endian conversions. Packed s24 is three bytes, shifted left 8 into an `int32` on decode. A packet longer than one frame drops a trailing partial frame. NaN and infinity pass through.
- `../pub-packages/gstreamer/subprojects/gst-plugins-base/gst-libs/gst/audio/audio-format.h` and the matching `audio-format.c`. `S24` unpacks with a left shift of 8. `S24_32` is the low three bytes of a 32-bit word, which is not Microsoft's left-justified rule, so this crate does not implement that coding.
- `../pub-packages/gstreamer/subprojects/gst-plugins-base/gst/rawparse/gstrawaudioparse.c`. A short tail is held for the next buffer. At end of stream that tail is dropped.
- `../pub-packages/gstreamer/subprojects/gst-plugins-base/gst-libs/gst/riff/riff-media.c`. PCM depth follows the container width. Valid-bits are not applied as a narrower sample.

Deviations, locked by tests: a trailing partial frame is `Truncated`, not a shorter frame of zeros and not a dropped tail. A non-finite float is `CodecError::NonFinite` for the whole frame. Integer full scale is the positive maximum (`8388607` and `2147483647`), the same rule the mix uses for `i16` (`32767`). `i32::MIN` rounds to `-1.0` in `f32`. The s24 minimum stays below `-1.0`. Finite floats outside ±1 are kept.

## Limits

| Limit | Value | Error |
| --- | --- | --- |
| Packet | 16 MiB | `Limit` `packet bytes` |
| Decoded or encoded buffer | 32 MiB | `Limit` `allocation` |
| Sample rate | 1 through 384_000 Hz | `InvalidApi` at 0, `Limit` above |
| Channels | 1 through 8, from `ChannelLayout` | `InvalidApi` when missing or empty |
| Layout extra | empty, or exactly 3 bytes | `Malformed` otherwise |

An allocation whose size overflows is rejected before `Vec` reserves it. A failed frame leaves the decoder as it was. PCM has no delay: `drain` is empty, and `reset` keeps the open configuration.

## Layout bytes

`CodecParams.extra` is three bytes. Empty extra follows the mix format: `I16` is interleaved little-endian s16, `F32` is interleaved little-endian f32. Missing both is `Unsupported` `pcm layout`.

| Byte 0 | Coding | Decode | Encode |
| --- | --- | --- | --- |
| 1 | s16 | `SampleFormat::I16`, exact | same bits, requested endian and plane order |
| 2 | s24, packed | `F32`, divide by `8388607` | `Unsupported` `s24 encode` |
| 3 | s32 | `F32`, divide by `2147483647` | `Unsupported` `s32 encode` |
| 4 | f32 | `F32`, bit-exact | bit-exact |
| 5 | f64 | `F32` via `as f32` | `f64::from` of the `f32` sample |
| 6 | 8-bit | refused | refused |
| 7 | A-law | refused | refused |
| 8 | μ-law | refused | refused |
| 9 | s64 | refused | refused |
| 10 | s24 stored in 32 bits | refused | refused |

Byte 1 is endian: `0` little, `1` big. Byte 2 is arrangement: `0` interleaved, `1` planar. Planar audio is one plane per channel, concatenated in channel order, and decodes to interleaved mix frames. Any other tag is `Malformed`.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| s16, either endian, interleaved or planar | Exact `I16` | Exact `I16` | `s16le_interleaved_is_the_mix_sample`, `an_independent_s16_stream_is_not_produced_by_this_encoder`, `encode_round_trips_the_original_samples` | Verified |
| s24 packed, either endian | `F32`, positive max is 1.0, minimum stays below −1 | Refused | `endian_is_a_parameter_for_integer_pcm`, `ffmpeg_reads_an_independent_bitstream` | Verified on decode. Encode unsupported |
| s32, either endian | `F32`, positive max is 1.0, minimum rounds to −1 | Refused | `endian_is_a_parameter_for_integer_pcm` | Verified on decode. Encode unsupported |
| f32, either endian | Bit-exact, including values outside ±1 | Bit-exact | `a_non_finite_float_is_an_error_and_a_loud_finite_sample_is_kept`, `encode_round_trips_the_original_samples` | Verified |
| f64, either endian | Narrowed to `F32`. Non-finite input or a non-finite result is an error | Widens `f32` and narrows back to the same bits | same tests, including −0 | Verified |
| Planar to interleaved | Accept | Emit when the coding encodes | `planar_pcm_decodes_to_interleaved_frames` | Verified |
| Non-finite float | `NonFinite`, then a later finite frame decodes | `NonFinite`, nothing queued, clock unchanged | `a_failed_frame_does_not_poison_the_next_frame`, `a_bad_encode_adds_nothing_and_the_next_push_still_works` | Verified. Replacement with zero would fail these tests |
| Trailing partial frame | `Truncated` `pcm frame` | n/a | `a_partial_frame_is_an_error_and_not_silence` | Verified. Not silence |
| Short buffer and overflowing length | Error | Error | `a_short_buffer_and_an_overflowing_length_are_rejected` | Verified |
| Factory selection by `CodecId::Pcm` | Registered | Registered | `the_decoder_is_registered_and_refuses_a_bad_open`, `export_selects_the_encoder_by_codec_id`, desktop `pcm_falls_through_when_the_platform_declines_the_layout` | Verified |
| Independent s16 and f32 bitstream | n/a | ffmpeg 8.1.1 reads the packet back | `ffmpeg_reads_an_independent_bitstream` | Verified on this machine. Skips when ffmpeg is absent unless `PCM_REQUIRE_ORACLES` is set |
| ffmpeg s24le compared after widening to s32 | Matches the scale above | n/a | same test | Verified on this machine |
| Local WAV and M4A, first 0.2 s as s16le | Matches the ffmpeg dump | n/a | `local_files_match_an_independent_pcm_dump` | Verified on this machine when `../local-files` and ffmpeg are present. Skips when the folder is missing |
| 30-minute decoder fuzz | Run | n/a | `thirty_minute_decoder_campaign` | Verified, this machine, 1800.00 s, seed `0x5043_4D5F_4655_5A5A` |
| Bounded encoder fuzz | n/a | Run | `bounded_encoder_campaign` | Verified, this machine, 120.00 s, seed `0x5043_4D5F_454E_4300` |
| Release throughput | At least 10× realtime | At least 10× realtime | `release_decode_and_encode_throughput` | Passed. Decode about 25,392×, encode about 7,811×. Resident set grew 48 KiB |
| 8-bit, unsigned, A-law, μ-law, ADPCM, s64, 24-in-32 | Error | Error | `named_tools_are_refused` | Unsupported |
| Resample, remix, dither, container framing | Outside this crate | Outside this crate | n/a | Unsupported |
| Linux and Windows | Not run | Not run | This file was written on macOS | Not run |

## Desktop

`apps/desktop` registers the platform openers first, then PNG, AAC, AV1, and this crate. On macOS, AudioToolbox accepts little-endian interleaved f32 and i16 and is selected for those parameters. A 3-byte f64 layout is not a magic cookie, so that opener returns `Unsupported` and this decoder runs. That is `pcm_falls_through_when_the_platform_declines_the_layout`.

A platform rate above 192 kHz returns `Limit`, and the factory walk stops. The software opener is not asked. Opening `PcmDecoder` directly still accepts a rate through 384 kHz.

## Campaigns

These runs used commit `758dc542` on `pcm`, release, one thread. Exit status 0. No panic and no hang.

Decoder fuzz, seed `0x5043_4D5F_4655_5A5A`, 1800.00 seconds:

`cargo test -p cutlass-pcm --offline --release --lib thirty_minute_decoder_campaign -- --ignored --nocapture --test-threads=1`

18,360,653,459 packets. 14,409,105,067 returned an error. The rest returned PCM whose stored length matched the frame count. Every 10,000 packets a known s16 frame still decoded. Findings 0.

Encoder fuzz, seed `0x5043_4D5F_454E_4300`, 120.00 seconds:

`cargo test -p cutlass-pcm --offline --release --lib bounded_encoder_campaign -- --ignored --nocapture --test-threads=1`

317,009,906 inputs. 22,647,259 returned an error. 294,362,647 round-tripped through the decoder at the expected frame count. Every 1,000 inputs a known s16 frame still encoded. Random 3-byte layouts were opened and did not panic. Findings 0.

Release measurement, 48 kHz stereo, 480,000 frames (10 seconds), s16le, 8 timed iterations after 2 warmup iterations. The timed loop checks the frame count and the encoded byte length, so the compiler cannot delete it. `ps -o rss=` was sampled before the timed loop and after it.

`cargo test -p cutlass-pcm --offline --release --lib release_decode_and_encode_throughput -- --ignored --nocapture --test-threads=1`

Decode 0.003151 seconds, about 25,392× realtime. Encode 0.010241 seconds, about 7,811× realtime. Resident set 17,728 KiB before and 17,776 KiB after, growth 48 KiB. The gates are 10× realtime and growth under 64 MiB. Both passed. The test panics when `debug_assertions` is set, so this is not a debug figure. Resident size is those two samples, not a peak between them.

## Machine

Apple M5 Pro, 18 cores, arm64, macOS 27.0.1, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. ffmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg` for the oracle tests. The decoder and encoder do not spawn threads.

## Checks

Checked on the PCM sources in `758dc542`:

- `cargo fmt --all -- --check` passed
- `cargo lint` passed
- `cargo test --workspace` passed. The PCM library reported 21 passed, 3 ignored, 0 failed. The ignored tests are the campaigns above, which were run separately in release.

`cargo deny check` was not run. Linux and Windows were not run. There is no PCM conformance bitstream suite. The export crate is still empty, so this codec is selected by `EncodeFactory` and by the desktop registration, not by an export job.
