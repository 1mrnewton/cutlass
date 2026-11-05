# ALAC support record

Branch `alac`, cut from `development` at `a3b60b45`. Native safe Rust in `cutlass-alac`. No `unsafe`. No bundled C, FFmpeg, AudioToolbox call, or ALAC crate. Decode and encode are both in scope.

This record is complete for ALAC in MP4 and M4A and partial overall: CAF is not done (see Gaps).

## Specifications and sources

- Apple's open-source ALAC reference (macosforge `alac`, Apache 2.0), read online and not copied: `ALACDecoder.cpp`, `ALACEncoder.cpp`, `ag_dec.c`, `ag_enc.c`, `dp_dec.c`, `dp_enc.c`, `matrix_dec.c`, `matrix_enc.c`, `ALACBitUtilities.c`, `ALACAudioTypes.h`, and `ALACMagicCookieDescription.txt`. There is no other ALAC specification.
- `../pub-packages/ffmpeg/libavcodec/alac.c`, `alacenc.c`, `alac_data.c`, and `alacdsp.c`, to check field order, the channel map (`ff_alac_channel_layout_offsets`), and the cookie forms. `../pub-packages/ffmpeg/libavformat/mov.c` for how the `alac` and `chan` boxes are read.
- GStreamer has no native ALAC codec: `gst-libav` maps it to FFmpeg, and `qtdemux` and `qtmux` carry the cookie (`gst-plugins-good/gst/isomp4`). Read for the `wave`/`frma` cookie form, which `Cookie::parse` accepts.

Behavior that follows the reference and is locked by tests:

- Golomb arithmetic wraps as unsigned 32-bit. A zero run past the frame, a run parameter outside 1 through 16, and a Golomb limit of zero are errors where the reference would read out of bounds.
- The encoder's stereo search (mix weight 0 through 4 on an eighth of the frame, orders 4 and 8 after eight converging passes on a thirty-second) and the mono search are the reference's, including which buffer each pass reads. That is why the frames equal AudioToolbox's.
- The three-channel layout tag is `MPEG_3_0_B` (114), C L R, as AudioToolbox writes it.
- `i32::MIN` as a residual has no Golomb code, as in the reference. The encoder never produces one: no channel is wider than 17 bits after the byte shift.

## Limits

| Limit | Value | Error |
| --- | --- | --- |
| Packet | 16 MiB | `Limit` `packet bytes` |
| Frame length | 1 through 262144 samples | `Malformed` at 0, `Limit` above |
| Bit depth | 16, 20, 24, 32 | `Unsupported` otherwise |
| Channels | 1 through 8 | `Malformed` at 0, `Unsupported` above |
| Cookie version | 0 | `Unsupported` otherwise |
| Golomb limit `kb` | at most 31 | `Malformed` above |
| `chan` atom | tag count equal to the channel count | `Malformed` otherwise |

## Capability matrix

All "Verified" rows ran on this machine on 2026-11-05: Apple M5 Pro, macOS 27.0.1, rustc 1.99.1, ffmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg`, `/usr/bin/afconvert`, `/usr/bin/afinfo`. Real audio is from `../local-files` (an Expo talk as music and speech, and a speech WAV), cut and converted by ffmpeg into the test's temporary folder. Oracle tests skip with a message when a tool or `../local-files` is missing; `ALAC_REQUIRE_ORACLES=1` turns the skip into a failure, and the runs below set it.

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Stereo, 16 and 24-bit, real audio | Same integers as ffmpeg and `afconvert`, on streams from both | Lossless in this decoder, ffmpeg, and `afconvert`, default and fast, 4096 and 1001-sample frames | `real_stereo_from_both_encoders_decodes_like_both_decoders`, `real_stereo_in_both_modes_decodes_back_exactly` | Verified |
| Mono, real speech | Same as both, both encoders | Lossless in all three, 4096 and 4095-sample frames | `real_mono_speech_decodes_like_both_decoders`, `real_mono_speech_decodes_back_exactly` | Verified |
| 5.1, 16 and 24-bit, real audio | Same as both, every channel in place | Lossless in all three, every channel in place | `real_five_one_keeps_every_channel_in_place`, `real_five_one_decodes_back_with_every_channel_in_place` | Verified |
| 1 through 8 channels, 16 and 24-bit | Same as ffmpeg (in its channel order) and `afconvert` (frame order), streams from both encoders, 0.7 s so the last frame is partial | Lossless in all three, odd frame lengths 1003 through 1017 | `every_channel_count_decodes_in_reference_order`, `every_channel_count_decodes_back_in_place` | Verified |
| 20 and 32-bit | Same as ffmpeg and `afconvert` on `afconvert -d alac/2` and `alac/4` streams | Lossless in all three, default and fast | `twenty_and_thirty_two_bit_streams_decode_like_both_decoders`, `twenty_and_thirty_two_bit_decode_back_exactly` | Verified |
| Same frames as AudioToolbox | n/a | Default equals `afconvert -d alac`, fast equals `afconvert -d alac -q 0`, every frame byte for byte, 10 s stereo at 16 and 24-bit | `compression_matches_the_reference_encoders` | Verified on stereo. Mono and other counts: lossless only, bytes not compared |
| Compression | n/a | See Compression below | same test | Measured |
| Escape frames, partial frames, shifted bytes, every mix weight and order the search picks | Decoded in the streams above | Written in the streams above | the oracle tests; hand-built escape frames in `hand_built_frames_decode_and_their_faults_are_named` | Verified |
| DSE and FIL elements | Skipped | Not written | `hand_built_frames_decode_and_their_faults_are_named` | Verified |
| CCE and PCE elements | `Unsupported` | Not written | same test | Unsupported, as in the reference |
| Cookie forms: bare, with `chan`, MP4 `alac` box payload, full atom, `wave`/`frma` | Accept | Bare config plus `chan` for 3 to 8 channels | `config` unit tests, `opening_checks_the_parameters`, `malformed_cookies_are_refused` | Verified |
| MP4 and M4A: cookie and channel layout | `cutlass-isobmff` joins the `alac` and sibling `chan` boxes into the cookie and takes channels, depth, and rate from it | Splits the cookie into `alac` and `chan` boxes | isobmff `an_apple_entry_joins_into_the_caf_cookie_and_splits_back`, `an_alac_cookie_set_late_comes_back_whole_with_its_layout`; every oracle test reads this crate's files back | Verified |
| End to end: Apple `.m4a` → demux → decode → encode → mux → reopen | Through the factories | Through the factories and `Mp4Mux` | `an_apple_m4a_survives_decode_encode_and_mux_bit_for_bit`: stereo 16 and 24-bit and 5.1 16-bit; every frame but Apple's padded last one equal; cookie and layout tag preserved; ffmpeg, `afconvert`, and `afinfo` (format and layout lines) read it as they read the source | Verified |
| CAF | Not read | Not written | none | Not done |
| Trait PCM | 16-bit `I16` exact; 20 and 24-bit `F32` exact; 32-bit `F32`, 24 bits kept | `I16` and `F32` input; floats round to the depth's grid | `the_registered_decoder_gives_exact_sixteen_bit_pcm_at_each_packet_time`, `twenty_four_bit_pcm_is_exact_in_float`, `float_input_is_checked_and_rounds_to_the_grid` | Verified. 32-bit through the traits is partial; `decode_frame` is exact |
| Lifecycle | Empty packet `InvalidApi`; a failed frame changes nothing; config side data must describe the same stream; a padded frame is trimmed to its packet's duration; `drain` empty | Cookie on the first packet and again on the drained one; `reset` encodes as a new stream; cancel between frames; uneven, out-of-range, non-finite, and changed-format input refused | `lifecycle` tests | Verified |
| Malformed input | Every named fault is an error; truncations, bit flips, and random tails of real frames at six shapes never panic or overrun a frame; 100,000 random packets | n/a | `hostile` tests | Verified |
| Seeded campaigns (cargo-fuzz cannot build here: `libfuzzer-sys` needs `cc`, which `deny.toml` bans) | Decoder mutation and cookie campaigns | Random shapes and signals must round-trip | `decoder_campaign`, `cookie_campaign`, `round_trip_campaign` | Verified, 1800 s each in release, no findings |
| Factory registration | Registered | Registered | `the_registered_*` tests; desktop `alac_opens_for_import_and_export` | Verified |
| Export | n/a | Sound-only M4A, 24-bit from the float mix, clipped to full scale | export `sound_alone_writes_a_lossless_m4a` | Verified on an empty timeline only |
| Desktop import | Rust decoder behind AudioToolbox on macOS, alone elsewhere | Export dialog offers ALAC, no bitrate | desktop `alac_opens_for_import_and_export`, `the_export_dialog_names_its_file_by_what_it_writes_and_reads_a_typed_bitrate` | Verified at the factory and dialog level. Playing an ALAC file in the running app: not run |
| Linux | Lint | Lint | `cargo clippy --workspace --exclude cutlass --all-targets --target x86_64-unknown-linux-gnu -- -D warnings` | Lint passed. Tests not run on Linux |
| Windows | Not run | Not run | none | Not run |

## Compression

`compression_matches_the_reference_encoders`, the first 10 s of the Expo file, stereo, 44.1 kHz. Ratio is PCM bytes over ALAC payload bytes.

| Encoder | 16-bit bytes | Ratio | 24-bit bytes | Ratio |
| --- | --- | --- | --- | --- |
| cutlass default | 375,546 | 4.697 | 1,257,550 | 2.104 |
| `afconvert -d alac` | 375,546 | 4.697 | 1,257,550 | 2.104 |
| cutlass fast | 754,682 | 2.337 | 1,637,089 | 1.616 |
| `afconvert -d alac -q 0` | 754,682 | 2.337 | 1,637,089 | 1.616 |
| `ffmpeg -c:a alac` | 351,927 | 5.012 | 1,233,931 | 2.144 |

ffmpeg's encoder searches harder than Apple's and compresses 6.3% (16-bit) and 1.9% (24-bit) better. This crate follows Apple's encoder so its files match AudioToolbox's.

## Performance

`cargo bench -p cutlass-alac`, release build, Thu Nov 5 2026, Apple M5 Pro, ffmpeg 8.1.1. Source: the first 60 s of the Expo file. Every encode was decoded and compared before timing.

```text
60 s stereo at 44100 Hz; median of 3 after 1 warmup; x real time
16-bit, real speech and music
  cutlass Default: 2237590 bytes, encode 590x, decode 1000x
  cutlass Fast: 4465045 bytes, encode 611x, decode 567x
  ffmpeg: encode 594x, decode 1936x
24-bit, real speech and music
  cutlass Default: 7529590 bytes, encode 558x, decode 877x
  cutlass Fast: 9757045 bytes, encode 578x, decode 537x
  ffmpeg: encode 557x, decode 1895x
```

Cutlass rows time samples and packets already in memory, one thread. ffmpeg rows are a new process each, including startup and file input and output. ffmpeg decodes about twice as fast as this decoder; the decoder has not been tuned. Memory was not measured.

## Campaigns

Run together, three threads, at commit `b489853e` (the decoder, cookie, and encoder code they exercise is unchanged since):

```text
ALAC_FUZZ_SECS=1800 cargo test -p cutlass-alac --release -- --ignored campaign --nocapture
alac decoder campaign: 27757057 inputs, 22219109 errors, no panics
alac cookie campaign: 5854561390 inputs, 3383511118 accepted, no panics
alac round-trip campaign: 1806331 streams, all lossless
test result: ok. 3 passed; 0 failed; finished in 1800.05s
```

The same three ran 300 s in the dev profile, where integer overflow panics: 5,722,759 decoder inputs, 1,145,273,253 cookies, 346,191 streams, no panics, all lossless. Seeds are fixed in `src/tests/campaign.rs`. No findings, so no minimized regressions.

## Commands run

```text
ALAC_REQUIRE_ORACLES=1 cargo test -p cutlass-alac          # 49 passed, 3 ignored (the campaigns)
cargo test -p cutlass-isobmff                              # 53 passed
cargo test -p cutlass-export                               # 31 passed
cargo test -p cutlass -- alac_opens the_export_dialog_names # 2 passed
cargo clippy -p cutlass-alac -p cutlass-isobmff -p cutlass-export -p cutlass --all-targets -- -D warnings
cargo clippy -p cutlass-alac -p cutlass-isobmff -p cutlass-export --all-targets --target x86_64-unknown-linux-gnu -- -D warnings
cargo clippy --workspace --exclude cutlass --all-targets --target x86_64-unknown-linux-gnu -- -D warnings   # exit 0
cargo bench -p cutlass-alac
cargo run --release -p cutlass-alac --example roundtrip
```

## Gaps

- CAF (`.caf`) is not read or written: `crates/containers/caf` is still an empty crate. ALAC in CAF needs `desc`, `kuki`, `chan`, `pakt`, and `data` chunks behind the container traits, with its own oracle tests. Not done.
- 32-bit through the `Decoder` trait is `f32`, which keeps 24 bits of each sample, because the mix has no wider buffer. `AlacDecoder::decode_frame` is exact.
- AudioToolbox pads the last frame to full length and states the real length in the MP4 sample table. `Decoder::push` trims a frame to its packet's duration when the packet is on the stream's clock (`a_padded_last_frame_is_trimmed_to_its_packet_duration`); `decode_frame` returns the whole frame. Not checked in playback.
- The app's media reader and export accept mono and stereo only, so a 5.1 ALAC file is refused at import and cannot be exported. The codec handles one to eight channels.
- Mono and multichannel frames are checked lossless, not byte-equal with AudioToolbox.
- Not run: playback in the running app, Linux and Windows tests, memory measurement.
