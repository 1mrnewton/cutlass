# Opus support record

Branch `opus`. Native safe Rust in `cutlass-opus`. `#![forbid(unsafe_code)]`, workspace lints. No libopus, FFmpeg, or other C: ffmpeg (with libopus and its native Opus), the RFC `opus_compare` tool, and the RFC 8251 vectors are test oracles only.

The decoder is complete for RFC 6716 as updated by RFC 8251. The encoder is CELT-only: it does not code SILK or hybrid frames. Both are wired into the desktop app, the phone, probe, and export; the gaps are listed at the end.

## Specifications and sources

- RFC 6716, Definition of the Opus Audio Codec: the range coder, packet framing and padding, SILK (section 4.2), CELT (4.3), mode switching and redundancy (4.5), packet loss concealment, and self-delimited framing (appendix B).
- RFC 8251, Updates to the Opus Audio Codec: the fixes the decoder applies (hybrid folding, CELT energy and anti-collapse changes, padding length checks) and the new test vectors.
- RFC 7845, Ogg Encapsulation for the Opus Audio Codec: `OpusHead`, channel mapping families 0, 1, and 255, the Vorbis channel order, output gain, and pre-skip.
- Encapsulation of Opus in ISO Base Media File Format, version 0.6.8: the `dOps` box and the edit list that removes the pre-skip.
- The reference implementation attached to RFC 6716 (`opus-rfc6716` tarball, sha1 `86a927223e73d2476646a1b933fcd3fffb6ecc8c`) was read to check behavior the prose leaves open (the float CELT decoder, the decoder's mode transitions, the encoder analysis). Tables are transcribed from the RFC. No code is linked.

## What this crate does

`OpusDecoder` and `OpusEncoder` speak raw Opus packets through `cutlass_decode::Decoder` and `cutlass_encode::Encoder`; `register_decoders` and `register_encoders` install them for `CodecId::Opus`. The canonical configuration is the `OpusHead` packet. The decoder also takes a `dOps` box body.

Decoder:

- SILK-only (NB, MB, WB), hybrid (SWB, FB), and CELT-only (NB to FB) frames of 2.5 to 60 ms, mono and stereo streams, packets of frame-count codes 0 to 3 with padding, up to 120 ms per packet.
- Mode transitions as the reference decoder does them: CELT redundancy frames on SILK/CELT switches, the transition crossfade, the CELT reset rules, and the hybrid-to-SILK silence frame. The final range of every frame is kept (`final_range`), XORed over the streams of a multistream packet.
- Multistream (RFC 7845 families 1 and 255): self-delimited stream packets, coupled and uncoupled streams, the channel mapping including silent channels (255), and output gain. At most 8 output channels (`MAX_CHANNELS`, the `ChannelLayout` limit). Families 2 and 3 (ambisonics) are refused as unsupported.
- Output at 8, 12, 16, 24, or 48 kHz (`params.sample_rate`; any other rate decodes at 48 kHz), interleaved `f32` at full scale 1.0, not soft-clipped (as libopus's float API).
- Concealment (`conceal`, `conceal_output` with `concealed: true`): SILK and CELT packet loss concealment. Forward error correction from SILK LBRR data (`decode_fec`). Gap lengths are 2.5 ms steps up to 120 ms.
- The pre-skip is left to the container's edit, which the engine applies. A packet whose `duration` is shorter than its decoded length is trimmed at the end (`skip_trailing`).
- A failed push changes nothing (the stream state is restored); an empty packet is `InvalidApi`; `drain` returns nothing; `reset` returns to a fresh decoder.

Encoder:

- CELT-only, mono or stereo, 48 kHz input only, 20 ms frames, one frame per packet. `open_with` (crate-internal) also does 2.5, 5, and 10 ms, which the tests cover.
- `bitrate` 0 picks 64 kb/s mono or 96 kb/s stereo; any other value is clamped to 6 to 510 kb/s. `constant_rate` gives CBR (every packet `bitrate * 20 ms / 8` bytes); otherwise VBR. The bitrate picks the coded bandwidth (narrowband at the lowest rates, fullband from about 18 kb/s per channel).
- The analysis follows the reference float encoder: pre-emphasis, pitch pre-filter, transient detection, intra/inter energy trial, TF analysis, spreading, dynamic allocation and trim, intensity and dual stereo decisions, PVQ, anti-collapse.
- Packets count at 48 kHz from 0. The first carries the `OpusHead` (`SideData::Config`); the pre-skip is 120 samples. The last packet's duration is shortened so durations add up to the pre-skip plus the input, and it carries `SideData::Gapless`.
- NaN, infinities, and samples over full scale are refused (`NonFinite`, `OutOfRange`) without changing the encoder, as the AAC encoder refuses them.

## Containers and the app

- Ogg: `cutlass-ogg` (see its `STATUS.md`). `.opus` files open in the desktop app, on the phone, and in `cutlass-probe`, and sound alone exports to `.opus`.
- MP4: `cutlass-isobmff` reads a `dOps` box as an `OpusHead`, reports the pre-skip as the stream's encoder delay, and keeps the file's edit. It writes `dOps` from an `OpusHead` or a `dOps` body, and an Opus track without edits gets the edit that starts play after the pre-skip and ends at the samples. `Mp4Mux` accepts Opus tracks. The `roll` sample group (`sgpd`/`sbgp`) ffmpeg writes is not written.
- Registration: the desktop (`apps/desktop/src/lib.rs`) and the phone (`crates/ffi/src/codecs.rs`, `export.rs`) register the Rust Opus decoder and encoder ahead of the platform codecs, so Opus decodes and encodes the same everywhere. `the_rust_opus_codec_goes_ahead_of_the_platform_one` in both crates checks the factories give this crate's exact packets and samples, and on Apple platforms that the platform codec, put first, gives different ones.
- Export: `write_sound` writes `(Opus, Ogg)`. The desktop export offers Opus for sound alone, at 48 kHz; a movie with Opus sound is refused ("Opus sound is written alone, in an Opus file").

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| RFC 8251 vectors 01 to 12 (CELT, SILK NB/MB/WB, hybrid SWB/FB, all switches) | Every final range matches; `opus_compare` passes in stereo and mono | n/a | `the_rfc_8251_vectors_match_every_final_range_and_pass_opus_compare` | Verified on this machine with the vectors; skipped without them |
| SILK-only, hybrid, CELT-only from libopus | Same sample count as libopus, at least 60 dB SNR against it | n/a | `files_ffmpeg_writes_decode_like_libopus` | Verified when ffmpeg with libopus is present |
| Frame sizes 2.5 to 60 ms, code 3 packets | Accept | 20 ms (2.5 to 20 ms internal) | vectors; `lowdelay-2.5ms`, `silk-60ms` interop cases; encoder frame tests | Verified |
| Multistream family 1 (5.1) | Accept | Not built | `a_five_one_stream_maps_each_decoded_channel_to_its_output`; libopus 5.1 `.opus` at 74 dB SNR against libopus | Decode verified. Encode not built |
| Family 255 | Accept (up to 8 channels) | Not built | header and mapping unit tests | Partial: no independent file |
| Output gain | Applied | Written as 0 | `output_gain_scales_the_samples` | Verified |
| API rates 8 to 24 kHz | Accept | 48 kHz only | `lower_api_rates_decode_the_same_packets` | Decode verified. Encode at other rates not built |
| Concealment and FEC | `conceal`, `decode_fec` | No LBRR written | `concealment_is_finite_marked_and_bounded`; SILK FEC compared with the RFC `opus_demo` during development | Decode verified; the engine does not call them (files do not lose packets) |
| CELT-only encode, VBR and CBR, 6 to 510 kb/s | n/a | Emit | `encoder_meets_the_interop_rate_and_quality_gates` | Verified against the gates below |
| SILK and hybrid encode | n/a | Not built | n/a | Not built: speech below about 24 kb/s codes as CELT, which libopus would code as SILK or hybrid |
| DTX, in-band FEC encode, multistream encode | n/a | Not built | n/a | Not built |
| Ogg Opus | Through `cutlass-ogg` | Through `cutlass-ogg` | interop tests, `sound_alone_writes_an_opus_file`, `an_opus_file_opens_seeks_and_decodes_after_the_pre_skip` | Verified |
| Opus in MP4 (`dOps`, pre-skip edit, end trim) | Through `cutlass-isobmff` | Through `cutlass-isobmff` | `an_opus_head_is_stored_as_dops_with_a_pre_skip_edit`, MP4 interop cases | Verified; the `roll` sample group is not written |
| Opus in WebM or Matroska | n/a | n/a | n/a | Needs the Matroska container. The `.webm` in `local-files` is Vorbis |
| More than two channels in the app | Decoder yes | n/a | n/a | The engine plays mono and stereo only; a 5.1 Opus file imports with a layout error |
| Linux and Windows | Not run | Not run | macOS; Linux cross-clippy only | Not run |

## Commands and results

All on this machine: Apple M5 Pro (`sysctl -n machdep.cpu.brand_string`), macOS, ffmpeg 8.1.1 at `/opt/homebrew/bin` with libopus, 5 November 2026.

`cargo fmt -p cutlass-opus`, `cargo clippy -p cutlass-opus --all-targets -- -D warnings`, and `cargo test -p cutlass-opus` pass: 91 unit tests, the encoder oracle, the interop tests, and the campaigns. Without ffmpeg the oracle and interop tests skip; without the vectors the vector test skips.

### Test vectors

The RFC 8251 vectors, `https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz` (sha256 `6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55`). `opus_compare` compiled from the RFC 6716 reference source above. The test decodes each vector as `opus_demo -d 48000 <channels>` does (float, soft clip, 16 bits) and compares every final range with the `.bit` records.

`CUTLASS_OPUS_VECTORS=<opus_newvectors> CUTLASS_OPUS_COMPARE=<opus_compare> cargo test --release -p cutlass-opus --lib decode:: -- --nocapture`

| Vector | Content | Packets | Final ranges | opus_compare stereo | opus_compare mono |
| --- | --- | --- | --- | --- | --- |
| 01 | CELT FB stereo | 2147 | all | PASS 99.9% | PASS 98.2% |
| 02 | SILK NB | 1185 | all | PASS 100.0% | PASS 96.2% |
| 03 | SILK MB | 998 | all | PASS 100.0% | PASS 96.7% |
| 04 | SILK WB | 1265 | all | PASS 100.0% | PASS 96.5% |
| 05 | hybrid SWB | 2037 | all | PASS 99.8% | PASS 96.3% |
| 06 | hybrid FB | 1876 | all | PASS 99.8% | PASS 96.3% |
| 07 | CELT, all sizes and bandwidths | 4186 | all | PASS 99.8% | PASS 98.1% |
| 08 | CELT and SILK | 1247 | all | PASS 99.8% | PASS 96.5% |
| 09 | CELT and SILK | 1337 | all | PASS 99.8% | PASS 98.1% |
| 10 | CELT and hybrid | 1912 | all | PASS 99.8% | PASS 97.7% |
| 11 | CELT | 553 | all | PASS 99.8% | PASS 98.5% |
| 12 | SILK and hybrid | 1332 | all | PASS 99.8% | PASS 99.8% |

Final ranges match in both the stereo and the mono decode of every vector.

### Files from ffmpeg

`cargo test --release -p cutlass-opus --test interop -- --nocapture`. Ten seconds of speech from `local-files` (synthetic without it), encoded by ffmpeg into `.opus` and `.mp4`, read by `OggDemux` or `IsoBmffFile`, decoded here, cut by the first edit, and compared with ffmpeg's libopus decode of the `.opus`:

| Case | Packets (SILK / hybrid / CELT) | Samples (= libopus) | SNR against libopus |
| --- | --- | --- | --- |
| libopus 96k stereo | 0 / 0 / 501 | 480000 | 75.8 dB |
| libopus 8k mono, 8 kHz input (SILK NB) | 501 / 0 / 0 | 480000 | exact |
| libopus 16k mono, 16 kHz input (SILK WB) | 501 / 0 / 0 | 480000 | exact |
| libopus voip 32k stereo | 274 / 204 / 23 | 480000 | 80.6 dB |
| libopus 128k, 2.5 ms | 0 / 0 / 4001 | 480000 | 75.8 dB |
| libopus 12k mono, 60 ms | 167 / 0 / 0 | 480000 | exact |
| libopus voip 24k, FEC on, 20% loss assumed | 302 / 199 / 0 | 480000 | 82.5 dB |
| ffmpeg native 96k | 0 / 0 / 501 | 480000 | 76.2 dB |
| libopus 5.1, 256k | 0 / 0 / 501 | 480000 | 74.2 dB |

Every MP4 case gives exactly the samples of its Ogg case. ffmpeg's own MP4 reader is not a reference for the count: it plays the last packet whole (past the end trim), and for a track it wrote at 8 or 16 kHz it skips the edit's media time as if counted at 48 kHz (52 samples instead of 312). The SNR against ffmpeg's MP4 decode is checked only for 48 kHz tracks.

### Files written here

The same test encodes nine and a bit seconds (479,667 samples, so the last packet is trimmed) at 96 kb/s, muxed by `OggMux` and `Mp4Mux`. ffprobe reports `opus`, 48000 Hz, 2 channels, `initial_padding=120` for both. ffmpeg's native decoder and libopus decode the `.opus` to exactly 479,667 samples; for the `.mp4`, ffprobe's `duration_ts` is 479,667 and both decoders give 479,880 (the last packet whole, as above). libopus and this decoder agree at 75.7 dB on both files.

### Encoder rate and quality

`cargo test --release -p cutlass-opus --test encoder_oracle -- --nocapture` (about 40 s). Gates, fixed before tuning: every file decodes in ffmpeg (native and libopus) to the input's exact sample count; VBR mean within 10% of the request on non-sparse items; every CBR packet exactly `bitrate * 20 ms / 8` bytes; mean per-band log-spectral distance (21 CELT bands) at most 1.5 dB above libopus's at 64, 96, and 128 kb/s stereo, both decoded by libopus.

Rates (kb/s, VBR) on the eleven recordings (speech, rain, thunder, effects, film): 32.1 to 33.5 at 32k mono, 64.2 to 66.6 at 64k mono, 64.0 to 65.9 at 64k stereo, 94.8 to 96.8 at 96k stereo, 125.8 to 129.0 at 128k stereo. Sparse items (silence 1.2, clicks 5 to 15) spend less, as libopus does.

Mean LSD (dB), ours against libopus, stereo VBR:

| Item | 64k | 96k | 128k |
| --- | --- | --- | --- |
| speech (expo) | 5.89 / 5.96 | 5.80 / 5.86 | 5.76 / 5.78 |
| speech (gemini) | 11.84 / 11.72 | 11.68 / 11.68 | 11.66 / 11.65 |
| film (fireship) | 6.18 / 6.25 | 5.87 / 6.01 | 5.76 / 5.85 |
| film (dbx) | 6.59 / 6.93 | 6.43 / 6.64 | 6.30 / 6.42 |
| rain (bedroom) | 1.09 / 1.20 | 0.74 / 0.84 | 0.56 / 0.66 |
| thunder | 1.36 / 1.42 | 1.24 / 1.22 | 1.21 / 1.19 |
| pink noise | 0.72 / 1.12 | 0.47 / 0.94 | 0.36 / 0.78 |
| sine 1 kHz | 1.49 / 0.87 | 0.81 / 0.59 | 0.52 / 0.45 |

The worst margin over libopus is the pure sine at 64k (+0.62 dB). LSD is an objective proxy: perceptual quality has not been judged in a listening test.

### Speed

`cargo bench -p cutlass-opus --bench against_ffmpeg` (median of three after a warmup, one thread; ffmpeg's times include its process start-up and file reading, so they are lower bounds on its codec speed; another test run shared the machine):

| Work | This crate | ffmpeg libopus | ffmpeg native |
| --- | --- | --- | --- |
| Decode CELT 96k stereo | 410× real time | 202× | 195× |
| Decode voip 24k stereo (SILK + hybrid) | 449× | 213× | 205× |
| Decode SILK WB 16k mono | 1760× | 283× | 266× |
| Encode 96k stereo VBR | 158× | 119× | n/a |
| Encode 128k stereo CBR | 142× | 113× | n/a |

The encoder oracle measures 151× to 174× real time over 170 s of stereo at 64 to 128 kb/s. Memory is not measured.

### Hardening

cargo-fuzz is not installed, and libfuzzer-sys builds C++, which this crate's rules do not allow. `tests/campaign.rs` runs seeded in-process campaigns instead, 2 s each on every test run (`CUTLASS_OPUS_CAMPAIGN_SECONDS` sets the length):

- Decoder: random bytes and bit flips, truncations, splices, TOC and code 3 rewrites of real packets (this crate's encoder, and with `CUTLASS_OPUS_VECTORS` every vector packet), through mono, stereo, and 5.1 decoders at every API rate, with 5.1 packets built from self-delimited mutated streams, mixed with concealment and FEC calls. No panic, every sample finite, a refused packet leaves the final range as it was, and after a reset the decoder matches a fresh one.
- Encoder: random bitrates (including 0 and over the limit), rate modes, channel counts, push sizes, and NaN, infinities, over-full-scale, square-wave, and denormal input. No panic, hostile samples refused without changing the encoder, every packet parses and decodes to finite samples, and durations add up to the pre-skip plus the accepted input.
- In the unit tests, `hostile_packets_fail_closed` runs 6000 more random packets and truncations.

One longer run, in the test profile (integer overflow checks on), with the vectors in the corpus, on commit `66248da2`:

`CUTLASS_OPUS_CAMPAIGN_SECONDS=300 CUTLASS_OPUS_VECTORS=<opus_newvectors> cargo test -p cutlass-opus --test campaign -- --nocapture`

| Campaign | Result |
| --- | --- |
| Decoder, 300 s | 6,751,168 calls, 3,132,221 packets refused, no failure |
| Encoder, 300 s | 99,410 runs, no failure |

## Partial or not run

- SILK and hybrid encoding are not built; the encoder is CELT-only at 48 kHz, 20 ms in its public API, with no DTX, in-band FEC, or multistream encoding.
- Family 255 decoding is checked by unit tests only, and output stops at 8 channels.
- The engine plays mono and stereo only, so a 5.1 Opus file does not import yet.
- The engine's probe duration is the last packet's end, so it counts the pre-skip (2.5 ms for this encoder, 6.5 ms for libopus), as it counts AAC priming in an MP4.
- The engine does not call concealment or FEC; files do not lose packets.
- `Mp4Mux` does not write the `roll` sample group; movie export with Opus sound is refused, so Opus in MP4 is written only by direct `Mp4Mux` use (the tests).
- Opus in WebM needs the Matroska container.
- Perceptual quality has not been judged in a listening test.
- cargo-fuzz not run (reason above). Memory not measured.
- Linux and Windows not run; shared crates were cross-linted for `x86_64-unknown-linux-gnu`.
