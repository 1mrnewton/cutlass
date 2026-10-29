# AAC support record

Checkout: branch `aac`. Native safe Rust in `cutlass-aac`. No `unsafe`. No bundled C, FFmpeg, or other AAC library.

This record is partial. AAC-LC encode and decode are implemented and covered by the tests named below. HE-AAC is not implemented. Playback and export are outside this record.

## Specifications

- ISO/IEC 14496-3:2009, subpart 4, for AAC-LC syntax, the filterbank, and the tools this decoder reads.
- ISO/IEC 13818-7 for MPEG-2 AAC and the ADTS header, including the CRC bit selection.
- ISO/IEC 11172-3 for the CRC-16 polynomial `x^16 + x^15 + x^2 + 1`, initial value `0xFFFF`.
- ISO/IEC 14496-3 subclause 1.7 for one-program LOAS/LATM.
- ISO/IEC 14496-1 for the `esds` descriptors, and ISO/IEC 14496-12 / 14496-14 for the audio movie boxes.
- 3GPP TS 26.404 and TS 26.405 describe SBR and parametric stereo. Those tools are not implemented. The QMF window coefficients sit in a figure in TS 26.404 and in the TS 26.410 reference tables. They are not transcribed here.

FFmpeg 8.1.1 (`libavcodec` AAC, `adtsenc`, `latmenc`, the mov muxer) was read to check field layout and float scale. Its source is not in this crate. Official ISO/IEC 14496-26 conformance vectors were not obtained.

## What this crate does

`AacEncoder` and `AacDecoder` speak raw AAC access units. `AudioSpecificConfig` is `CodecParams.extra`. Frame length is 1024 or 960. Sample-rate indexes 0 through 12 are 96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, and 7350. An explicit 24-bit rate is parsed and then rejected. MPEG-2 rejects 960-sample frames, the 7350 Hz index, and an explicit rate.

Channel layouts follow AAC element order, not wave order. Mono, stereo, 5.1 (configuration 6), and 7.1 (configuration 12) round-trip. Configuration 7 is the wide 7.1 layout. Configurations 11 and 12 cannot be written in the 3-bit ADTS channel field. The decoder bound is 8 channels. Configuration 13 is rejected.

The decoder reads SCE, CPE, LFE, DSE, FIL, and END, plus ICS, Huffman (including escapes), inverse quantization, sine and KBD windows, MDCT, overlap-add, pulse, TNS, PNS, mid/side, and intensity. Coupling, an in-band program config, prediction, gain control, a core coder, and `extensionFlag3` are errors. DRC is skipped and not applied. A failed frame does not change overlap. One encoder or decoder is `Send` and is not internally locked.

The encoder uses one frame of lookahead, KBD windows, a shared block sequence for non-LFE channels, mid/side when the channels correlate, and an average or variable rate. It does not emit TNS, PNS, intensity, or pulse. Average mode is the mean payload rate. Variable quality is 1 through 10. The bit reservoir is at most 6144 bits per channel. Leading delay is one frame and is carried on `SideData::Gapless`. Packet N reconstructs input frame N−1.

Float PCM full scale is 1.0. Analysis coefficients are multiplied by 65536 before quantization, and the decoder divides by 65536 before synthesis, so a float decoder that uses the 32768 / `(N·32768)` pair matches level. Integer PCM uses the 16-bit full scale.

## Transport, outside this crate

`cutlass-adts` frames one AAC stream as ADTS or as single-program LOAS/LATM. ADTS can store the header CRC. One protected block leaves the element CRC for this decoder (`protection_crc`). Two to four raw blocks are accepted only when the header carries byte positions. A multi-block frame without positions is rejected. ADTS cannot carry HE signaling or channel configurations above 7.

`cutlass-isobmff` writes and reads one audio track: `ftyp` (`M4A `), `moov`, `mp4a`/`esds`, and `mdat`. Samples are raw access units. The edit list stores encoder delay and trailing padding. A video track, a fragment, and an ADTS header inside a sample are rejected. This is not a general MP4 reader.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| AAC-LC, 1024, MPEG-4 | Accept | Emit | `a_sine_clears_the_mono_snr_gate`, `a_silent_frame_decodes_to_one_frame_of_zeros` | Verified on this machine |
| AAC-LC, 960 | Accept | Emit | `a_nine_sixty_frame_clears_the_mono_gate` | Verified on this machine |
| MPEG-2 LC limits | Reject 960, 7350 Hz, explicit rate | Same | `config` tests, `mpeg2_rejects_the_7350_hz_index` | Verified |
| Stereo, 5.1, 7.1 | Accept | Emit | `a_stereo_sine_clears_the_per_channel_gate`, `five_one_keeps_each_channel_and_limits_the_lfe`, `seven_one_round_trips_a_distinct_tone_per_channel` | Verified on this machine |
| Average rate over 2 s of noise | n/a | Within 0.70×–1.10× of 64 kbps | `noise_over_two_seconds_stays_inside_the_rate_window` | Verified |
| ffmpeg 8.1.1 float interop | Decode its stream | It decodes ours | `ffmpeg_decodes_our_access_units`, `we_decode_an_ffmpeg_stream` | Verified here when ffmpeg is present. Skipped, not passed, when it is absent |
| ADTS, including CRC and positioned multi-block | This crate checks the element CRC | The ADTS crate writes it | `a_wrong_crc_does_not_advance_the_decoder`, `a_protected_block_keeps_its_crc_and_payload`, `two_raw_blocks_round_trip_when_positions_are_present`, `ffmpeg_adts_survives_our_demux_and_mux` | Verified. ffmpeg does not check the CRC, so the element CRC is checked by this decoder |
| LOAS/LATM, one program | Opaque payload | The ADTS crate frames it | `one_program_round_trips_and_reuses_the_config`, `ffmpeg_decodes_a_latm_remux_of_its_own_aac` | Verified here when ffmpeg is present |
| M4A, one AAC track, priming edit | The ISOBMFF crate | The ISOBMFF crate | `priming_edit_round_trips_raw_samples`, `ffmpeg_m4a_round_trips_through_our_boxes` | Verified here when ffmpeg is present |
| HE-AAC v1 (SBR) and v2 (SBR+PS) | Error `sbr` | Error `sbr` | `config` rejects the extension | Unsupported. Lowband-only output is not treated as HE |
| Main, SSR, LTP, ER, LD, ELD, USAC/xHE | Error | Error | Named audio-object-type errors | Unsupported |
| ADIF | Error | Error | No reader | Unsupported |
| ISO 14496-26 vectors | Not run | Not run | Vectors not obtained | Not available |
| 30-minute fuzz | Ignored test exists | n/a | `thirty_minute_decoder_campaign`, `thirty_minute_adts_campaign` | Not run to completion for this record. Short random inputs are `random_access_units_fail_closed` and `random_bytes_fail_closed` |
| Release speed and memory | Not measured | Not measured | No benchmark recorded | Not run |

## Gates these tests lock

Mono 48 kHz, 1 kHz sine at half scale, 96 kbps, 1024 or 960: SNR at least 40 dB after delay trim. Stereo 128 kbps: at least 35 dB per channel. Near Nyquist (0.45 × sample rate), stereo 128 kbps: at least 20 dB. Silence at 128 kbps: absolute sample below `1e-4`. DC 0.25: mean error below 0.02. 5.1 at 384 kbps: full-band channels at least 30 dB, LFE at least 25 dB. 7.1 at 512 kbps: at least 25 dB per channel. 16-bit sine: at least 40 dB. A unit impulse is finite; its peak bound is not asserted.

## Still required before calling this complete

- A real SBR decoder and encoder, then parametric stereo. AOT 5 and 29 must not return the core alone.
- The ISO 14496-26 vectors, or an explicit record that they cannot be obtained.
- A finished fuzz campaign of at least 30 minutes for the decoder and for ADTS, with seed, duration, and findings written down.
- Release measurements on one thread against the decode and encode throughput budgets, and a memory bound, on the machine that runs them.
- Linux and Windows runs. This file was written on macOS.
