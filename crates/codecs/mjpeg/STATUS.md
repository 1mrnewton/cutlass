# Motion JPEG support record

Checkout: branch `mjpeg`, worktree `cutlass-mjpeg`, started from `development` at `e025f6ce`. Native safe Rust in `cutlass-mjpeg`, on the JPEG core in `cutlass-jpeg`. No `unsafe`, no libjpeg, no FFmpeg, no VideoToolbox, and no subprocess in either library. FFmpeg, libjpeg-turbo, and GStreamer are test oracles only.

This file is the scope record. A row is marked done only when a test or a recorded run named here has passed. Numbers are from runs on this machine, not estimates.

## Specifications and sources read

Normative:

- ITU-T T.81 (1992) | ISO/IEC 10918-1. Markers, interchange and abbreviated formats (B.4, B.5), Huffman coding, sequential and progressive scans, restart intervals, the DCT.
- ITU-T T.83 | ISO/IEC 10918-2. Decoder compliance: each reconstructed sample within 1 of the reference output. That is the tolerance used where the reference decoder uses a different IDCT.
- ITU-T T.871. JFIF: full-range BT.601 YCbCr, chroma sited between luma samples.
- Apple QuickTime File Format, Motion JPEG format A and B, and the `fiel` atom.
- ISO/IEC 14496-1 object type 0x6C (JPEG) for an `mp4v` entry in an MP4 file.

Read before the Cutlass code, for the algorithm, the conventions, and the edge cases. Nothing was copied, linked, or vendored.

- FFmpeg 8.1.1 `libavcodec/mjpegdec.c` (default Huffman tables, interlace detection from the first SOF height, the `AVID` comment, the `CS=ITU601` comment, AVI1 and Apple `mjpg` APP1 headers, EOI field handling), `mjpegbdec.c`, `mjpegenc.c`, `jrevdct.c`, `jfdctint_template.c`, `libavformat/mov.c` (`fiel`), `movenc.c` (`jpeg` entry, `fiel` written from field order), `isom.c` (`mp4v` object type 0x6C).
- GStreamer 1.28.3 `gst-plugins-good/ext/jpeg/gstjpegdec.c` (standard Huffman tables when all four are absent, the field-pair height rule, field decode into alternate rows).
- libjpeg-turbo 3.2.0 behaviour through `djpeg` and `cjpeg`, used as an oracle. GStreamer's `jpegdec` calls libjpeg and outputs only I420, so it is not a separate oracle.

## Conventions this crate follows

These are Motion JPEG rules that are not in T.81. Each one is taken from the projects above, not guessed.

| Convention | Rule | Source |
| --- | --- | --- |
| Missing DHT | A table slot never defined by DHT uses the T.81 Annex K.3 table of that class: id 0 luminance, id 1 chrominance. A DHT replaces it | FFmpeg always preloads K.3; GStreamer inserts them when all four are absent; the AVI MJPEG note drops DHT from every frame |
| Abbreviated frames | A packet may omit DQT. Tables persist from earlier packets and from codec extradata (T.81 B.5). Tables a packet defines replace earlier ones only after that packet decodes. A scan whose table was never defined is `Malformed` | T.81 B.5 |
| Two fields in one packet | Two complete JPEG images in one packet, same width, where the first image's height `h` and the container height `H` satisfy `2h - 16 < H <= 2h` and `h > 8`, are two fields woven into one frame of height `H` | GStreamer `gstjpegdec.c`; FFmpeg uses the first SOF height against the container height |
| Field order | First stored field is the top field, unless the `fiel` atom says bottom stored first (detail 6 or 14). Without `fiel`, a COM segment starting `AVID` whose byte 12 is 1 (NTSC) stores the bottom field first, and 2 (PAL) the top field | QuickTime `fiel`; FFmpeg `parse_avid`; GStreamer top-first default. FFmpeg maps detail 14 to "bottom coded first, top displayed first" and only honours detail 6; this crate follows Apple's definition |
| Format A header | An APP1 `mjpg` header in each field is accepted and not needed; the second field is found by its SOI | QuickTime format A; FFmpeg reads and discards it |
| Color | Three components are full-range BT.601 YCbCr unless Adobe transform 0 or `R`,`G`,`B` ids say RGB. A COM segment marks studio range when, read as a C string (one final `\n` dropped, text ending at the first NUL), it is exactly `CS=ITU601`. FFmpeg's encoder writes that comment with a NUL | T.871; FFmpeg `cs_itu601` and `mjpegenc_common.c` |
| EXIF | Ignored for video frames. Orientation is not applied to Motion JPEG | Video players do not rotate per frame |

## Capability matrix

Decoder acceptance and encoder output are separate columns. Evidence names the test that locks the row.

| Capability | Decode | Encode | Status | Evidence |
| --- | --- | --- | --- | --- |
| Baseline SOF0, 8-bit Huffman | Yes | Yes, every frame | Done | `ffmpeg_motion_jpeg_matches_djpeg_exactly_and_ffmpeg_within_one`, `this_encoder_opens_cleanly_and_matches_cjpeg_size_and_quality` |
| Extended sequential SOF1, 8-bit Huffman | Yes | No | Done | `cjpeg_440_rgb_and_restart_frames_match_djpeg` (`cjpeg -quality 5` writes SOF1) |
| Progressive SOF2 | Yes, through the shared core | No | Done | `cjpeg_440_rgb_and_restart_frames_match_djpeg` (`-progressive`) |
| Grey (1 component) | Yes, `Gray8` | Yes | Done | `cjpeg_440_rgb_and_restart_frames_match_djpeg` (`-grayscale`), encoder oracle with `Sampling::Gray` |
| YCbCr 4:4:4, 4:2:2, 4:2:0 | Yes | Yes, chosen by the caller; default 4:2:0 | Done | FFmpeg `yuvj444p`/`yuvj422p`/`yuvj420p` streams; encoder oracle against `cjpeg -sample 1x1`/`2x1`/`2x2` |
| YCbCr 4:4:0 and other legal sampling factors 1..=4 | Yes; 4:4:0 stays `Yuv440p`, the rest become `Yuv444p` on the planar path | No | Done | `cjpeg_440_rgb_and_restart_frames_match_djpeg` (`-sample 1x2`, planes within 1 of FFmpeg `yuvj440p`) |
| RGB (Adobe transform 0 or `RGB` ids) | Yes: RGBA on the engine path, `Gbrp` on the planar path | No | Done | `cjpeg_440_rgb_and_restart_frames_match_djpeg` (`-rgb`) |
| Quantization tables 8-bit and 16-bit, Huffman tables, custom tables | Yes | Annex K tables scaled by quality; standard Huffman tables | Done | `-quality 5` writes 16-bit DQT; asserted in the fixture |
| Restart intervals and RSTn | Yes, strict sequence | Optional interval in MCUs | Done | `cjpeg ... -restart 1`; fuzz corpus restart 1 and 2; encoder restart in fuzz encoder cases |
| Byte stuffing and fill bytes | Yes | Yes | Done | Every real-footage frame (stuffing is common); fuzz insert mutations |
| Odd sizes, MCU padding, cropping | Cropped to T.81 A.1.1 component sizes | Edge replication into the padded MCU | Done | 99×67, 53×38, 23×17, 641×361 fixtures; `planar_frames_keep_their_coded_sampling_and_odd_sizes` |
| Size, sampling, or table change between frames | Each packet stands alone apart from carried tables | Each frame stands alone; an encoder opened at 0×0 takes any size, one opened at a size refuses another | Done | `size_and_sampling_may_change_between_packets`, `tables_carry_to_an_abbreviated_packet_and_reset_drops_them` |
| Missing DHT, abbreviated DQT | As in the conventions above | Always writes DQT and DHT | Done | `frames_without_dht_decode_like_djpeg_with_its_default_tables` (FFmpeg `-huffman default` with DHT stripped), `tables_in_the_extradata_serve_every_packet`, `a_failed_packet_keeps_the_tables_it_did_not_finish` |
| Two fields per packet (interlaced) | Woven into one frame; field order on the crate API | Not emitted | Done | `cjpeg_field_pairs_weave_like_the_djpeg_fields_interleaved` (72×48, 72×45, 40×34; top and bottom first), `two_fields_weave_into_the_container_frame`, `field_packets_that_do_not_pair_are_refused` |
| Field order from `fiel` and `AVID` | Yes | n/a | Done | `the_bottom_field_first_comes_from_fiel_then_the_avid_comment`; ISOBMFF oracle reads FFmpeg's `setfield=bff` movie as `fiel 02 06` |
| Motion JPEG format A (`mjpa`) | Yes | No | Done | `format_a_headers_and_padding_between_fields_are_skipped` |
| Motion JPEG format B (`mjpb`) | Refused, `Unsupported` (`motion jpeg format b`): no independent encoder exists to validate it | No | Refused | `format_b_and_foreign_extradata_are_refused` |
| Arithmetic, lossless, hierarchical, JPEG-LS, 12-bit, CMYK | Refused by name, from the JPEG core | No | Refused | `cutlass-jpeg` `named_unsupported_frames_stay_named`, `arithmetic_coding_is_refused_by_name`, `precision_cmyk_and_component_count_are_named` |
| Studio range (`CS=ITU601`) | Tagged `Limited` on the planar path; RGBA converted as studio BT.601 | Refused on the planar input | Done | `a_studio_range_comment_tags_the_frame_and_the_rgba_conversion`; FFmpeg `yuv420p` stream compared with FFmpeg's own RGB |
| Quality control | n/a | `extra` one byte 1..=100, default 75 | Done | Size within 2 % and PSNR within 0.2 dB of `cjpeg -dct int` at quality 30, 75, 95 for every sampling; decoded samples equal cjpeg's |
| Bitrate control | n/a | Per-frame quality search to the frame's bit budget, with carried credit | Done | `bitrate_mode_holds_the_mean_rate_within_ten_percent`; the 1080p bench delivered 60.0 Mb/s for a 60 Mb/s target |

## Integration

| Path | Change | Status | Evidence |
| --- | --- | --- | --- |
| Engine playback | `register_decoders`: `Decoder` producing `Rgba8` with the JFIF conversion and libjpeg-style triangle chroma upsampling | Done | `the_picture_decoder_hands_the_engine_rgba_with_the_packet_time` |
| Planar consumers | `register_video_decoders`: `VideoDecoder` producing `Gray8`, `Yuv420p`, `Yuv422p`, `Yuv440p`, `Yuv444p`, or `Gbrp`; BT.601, full range unless `CS=ITU601` | Done | Planes within 1 of FFmpeg `-idct int` |
| Export | `register_encoders`: `Encoder` accepting `Rgba8` CPU frames, which is what export sends. `register_video_encoders` and `MjpegEncoder::encode_frame` take full-range `Gray8`, `Yuv420p`, `Yuv422p`, or `Yuv444p` | Done | `the_encoder_writes_frames_this_decoder_and_the_core_read_back`, `planar_input_keeps_its_sampling_through_encode_and_decode` |
| ISOBMFF demux | `jpeg`, `mjpa`, `mjpb` sample entries pass their type and any `fiel` payload as extradata. `mp4v` with object type 0x6C is Motion JPEG with empty extradata: a decoder-specific descriptor there has no meaning for JPEG and is dropped, as FFmpeg's decoder ignores it. Object types 0x60..=0x65 and 0x6A are MPEG-2 | Done | `a_motion_jpeg_entry_carries_its_code_and_field_order`, `an_mp4v_entry_is_motion_jpeg_only_for_object_type_0x6c`, `ffmpeg_motion_jpeg_files_demux_and_remux` (packets equal to FFmpeg's) |
| ISOBMFF mux | The movie muxer writes Motion JPEG as a `jpeg` entry, in a `.mov` only; an MP4 is refused | Done | `a_motion_jpeg_movie_is_a_jpeg_entry_with_every_frame`, `motion_jpeg_is_refused_outside_a_movie_or_with_foreign_extradata`; FFmpeg reads the remux as `mjpeg,jpeg` |
| Desktop | Registered ahead of the platform decoders and encoders. VideoToolbox opens a Motion JPEG stream and then fails a 4:2:2 frame (`Malformed`), and the factory falls through only on a refusal at open, so it cannot be the first choice | Done | `motion_jpeg_decodes_in_rust_ahead_of_the_platform` (fails with the platform first), `motion_jpeg_opens_in_rust_when_the_platform_is_off` |
| iOS, Android (`ffi`) | Registered ahead of the platform in `codecs::decoders` and `export::encoders` | Done | `a_phone_codes_motion_jpeg_in_rust_ahead_of_the_platform` (fails when the order is reversed) |
| Desktop export | Motion JPEG offered for `.mov` only, at five times the H.264 rate for the size, capped at 400 Mb/s | Done | `a_motion_jpeg_export_lands_a_mov_that_plays_back`, `motion_jpeg_is_given_five_times_the_h264_rate` |
| Playback, seek, export of real footage | A Motion JPEG movie made by FFmpeg from `16929689_1920_1080_30fps.mp4` (854×480 `yuvj422p`) | Done | `a_real_motion_jpeg_movie_plays_seeks_and_exports`: frames 45, 9, 30 within a mean of 1.5 of FFmpeg; export opens in `ffmpeg -xerror` with no message and PSNR > 30 dB against the source |

Field order has no slot in `PictureInfo`, `VideoFrame`, or the engine's frame. Fields are woven before those contracts, and the order is available from the crate API (`MjpegDecoder::last_field_order`). Nothing downstream of the decoder deinterlaces.

## Resource limits

| Limit | Value |
| --- | --- |
| Packet | 64 MiB, refused before it is read (`an_oversized_packet_is_refused_before_it_is_read`) |
| Edge | 16 384 pixels |
| Pixels per frame | 16 777 216 |
| Images per packet | 2 |
| Markers per image | 4 096 |
| Pending input | None: a packet is decoded in `push` |
| Output queue | At most one frame per packet |
| Encoder delay | None |

## Gates and results

Tools: FFmpeg 8.1.1 (`/opt/homebrew/bin/ffmpeg`), libjpeg-turbo 3.2.0 `cjpeg` and `djpeg`. Machine: Apple M5 Pro, 18 cores. `MJPEG_REQUIRE_ORACLES=1` makes a missing tool fail the oracle tests instead of skipping them; every `cutlass-mjpeg` run recorded here set it, and the desktop and ISOBMFF oracle tests found FFmpeg and ran.

| Gate | Budget | Result |
| --- | --- | --- |
| Planar decode against FFmpeg (`-idct int`, raw `yuvj*` planes) | Every sample within 1 (T.83) on grey, 4:4:4, 4:2:2, 4:2:0, 4:4:0 | Met |
| RGBA decode against `djpeg -dct int` | Bit-exact on grey, 4:4:4, 4:2:2, 4:2:0, 4:4:0, RGB, progressive, SOF1 | Met, including 5 frames each of two real clips, scaled to 720 rows as `yuvj420p` and to 641×361 as `yuvj422p` |
| Encoder output | `ffmpeg -xerror` and `djpeg` decode every frame with no warning | Met |
| Encoder quality | Same quality and sampling as `cjpeg -dct int`: size within 2 % and PSNR within 0.2 dB | Met; decoded samples are equal to cjpeg's |
| Rate control | Mean bitrate within 10 % of the target over a clip | Met |
| Interlaced | Woven output equals the two independently decoded fields interleaved | Met |
| Real media | Fixtures made from `../local-files` footage by FFmpeg and cjpeg, decode, seek, export, and reopen with FFmpeg | Met. `../local-files` holds no Motion JPEG file, so every Motion JPEG input is made from its H.264 footage |
| Malformed input | Truncation at every byte of a frame, bit flips, and hostile headers return errors; the next good packet decodes | Met (`every_truncation_and_bit_flip_fails_closed_and_the_next_packet_decodes`, 3 000-case mutation test) |
| Fuzz | At least 30 minutes on the decoder entry, recorded | Met: 2 000 s, 117 147 093 cases, no panic |
| Decode speed | 1920×1080 4:2:0 to planar and to RGBA, one thread, release: under 16 ms per frame | Met: 3.33 ms planar, 6.76 ms RGBA |
| Encode speed | The same frame at a fixed quality: under 33 ms | Met: 14.59 ms at quality 75 |
| Memory | No growth across 1 000 frames; peak recorded | Met: +480 KiB decode, none encode; bench peak 527 MB resident |
| Checks | `cargo fmt`, `cargo clippy -D warnings`, and `cargo test` on the changed crates and their direct consumers. The user asked for no full-workspace run | Met: see Checks |

## Performance

Command: `cargo bench -p cutlass-mjpeg --bench throughput`, its binary run under `/usr/bin/time -l` at 15:12 UTC on Sun Nov 8 2026 with the machine otherwise idle. Six 1920×1080 frames from `16929689_1920_1080_30fps.mp4`, 60 pictures per pass, best of 5 passes, one thread.

```text
encode RGBA 4:2:0 q75: 14.59 ms/frame, 210 KB/frame, 50.3 Mb/s at 30 fps
encode RGBA 4:2:2 q75: 16.87 ms/frame, 220 KB/frame
encode RGBA 4:2:0 at 60 Mb/s: 28.23 ms/frame, 60.0 Mb/s delivered
decode this encoder 4:2:0 q75: planar 3.33 ms/frame, RGBA 6.76 ms/frame; ffmpeg -threads 1 2.84 ms/frame
decode this encoder 4:2:2 q75: planar 3.76 ms/frame, RGBA 6.69 ms/frame; ffmpeg -threads 1 3.17 ms/frame
decode ffmpeg 4:2:2 -q:v 3: planar 3.80 ms/frame, RGBA 6.85 ms/frame; ffmpeg -threads 1 3.36 ms/frame
memory: decode resident 182064 KiB at frame 100, 182544 KiB at 1000 (+480 KiB); encode at 60 Mb/s 187648 KiB, 152912 KiB (-34736 KiB)
       64.20 real        63.86 user         0.40 sys
           526630912  maximum resident set size
           195002896  peak memory footprint
```

An earlier run, with the fuzz campaign on another core, gave 13.93, 16.95, and 28.38 ms to encode and 3.35 and 6.76 ms to decode.

The FFmpeg time is `ffmpeg -threads 1 -f mjpeg` decoding the same stream to raw planes, total wall time over the frame count, so it includes process start. Rate mode costs about twice a fixed quality because it counts the frame's scan bits at two or more qualities, starting from the last frame's, before writing. The peak is the whole bench process: six RGBA source frames, every encoded stream, and FFmpeg's decoded output, as well as the codec. Resident size between frame 100 and frame 1 000 is the growth gate; the codec's own buffers are reused from frame to frame.

## Fuzz

`MJPEG_FUZZ_SECONDS=2000 MJPEG_FUZZ_SEED=20261108 cargo test --release -p cutlass-mjpeg fuzz_campaign -- --ignored --nocapture`, from 14:38:10 to 15:11:32 UTC on commit `f13e5bcd`. Later commits changed only tests, the bench, and documents.

```text
fuzz campaign: seed 20261108, 117147093 cases in 2000 s, 93400448 refused, 16426569 accepted, 7320076 encoder cases, no panics
```

The corpus is eight packets: 4:2:0 with restart 2, 4:2:2 without DHT, 4:4:4 17×13 with restart 1, grey, an abbreviated packet whose tables are in the extradata, a 40×47 field pair, a format A pair, and an `AVID` bottom-first pair. Each case applies one of eight mutations (bit flips, truncation, a random span, a header byte, a segment length, a splice, inserted markers or bytes, a duplicated range). One case in sixteen encodes a random 1..70 picture at a random sampling, quality, and restart interval and decodes it back. Every accepted picture is checked for size limits and consistent buffers, and after every case the unmutated packet must still decode to its reference picture, so a refused packet cannot disturb the decoder's tables or state. `mutated_packets_fail_closed_and_leave_the_decoder_as_it_was` runs 3 000 of these cases in every `cargo test`.

The JPEG core's own campaign also ran, because this branch rewrote its entropy decoder, IDCT, and planar path: `JPEG_FUZZ_SECS=1200 cargo test --release -p cutlass-jpeg decoder_fuzz_campaign -- --ignored`, 14:47:57 to 15:07:59 UTC on `aa357b8f`, printed `fuzz_cases=805305614 seed=0x4a504500ffd8ffd9 secs=1200` with no panic. That older campaign flips one byte of an 8×8 grey image or feeds random bytes, so it is shallow next to the Motion JPEG one.

## Checks

On `aa357b8f`, with `CARGO_TARGET_DIR` and `TMPDIR` under `/tmp/cutlass-mjpeg-work`:

| Command | Result |
| --- | --- |
| `cargo fmt -p cutlass-jpeg -p cutlass-mjpeg -p cutlass-isobmff -p cutlass -p cutlass-ffi -- --check` | Clean |
| `cargo clippy -p cutlass-jpeg -p cutlass-mjpeg -p cutlass-isobmff -p cutlass -p cutlass-ffi --all-targets -- -D warnings` | Clean |
| `MJPEG_REQUIRE_ORACLES=1 JPEG_REQUIRE_ORACLES=1 cargo test --no-fail-fast -p cutlass-jpeg -p cutlass-mjpeg -p cutlass-isobmff -p cutlass-ffi` | jpeg 28 passed, 2 ignored (timing budget, fuzz campaign); mjpeg 30 passed, 1 ignored (fuzz campaign); isobmff 65 passed; ffi 87 passed |
| `cargo test --no-fail-fast -p cutlass` (desktop) | 436 passed, 6 ignored; sandbox 2 passed; doc tests 8 ignored |

`a_real_motion_jpeg_movie_plays_seeks_and_exports` was run alone with `--nocapture` and printed no skip line. The desktop test build prints one linker notice, that the debug test binary's unwind section is over 16 MB; it is not a compiler or clippy warning. The repository's `cargo lint` and `cargo test --workspace` cover the whole workspace and were not run, as the task asked.

## Gaps

- Motion JPEG format B (`mjpb`) is refused. No independent encoder was available to validate a decoder for it.
- The encoder writes whole progressive frames only. Two-field packets are decoded, not encoded, and the movie muxer writes no `fiel` box.
- Arithmetic, lossless, hierarchical, JPEG-LS, 12-bit, and CMYK frames are refused.
- The fuzz campaign is a seeded mutation fuzzer run as an ignored test, the repository's convention. `cargo-fuzz` is not installed, so there was no coverage-guided run.
- With Motion JPEG registered ahead of the platform, VideoToolbox's Motion JPEG decoder is not used. A 4:2:0 frame it can decode is decoded in Rust too.
- Export always codes 4:2:0 from RGBA. Planar 4:2:2 and 4:4:4 input is on the crate API, not in the export menu.
- `../local-files` holds no Motion JPEG media. Real-file tests use Motion JPEG made from that footage by FFmpeg and cjpeg.
- Linux, Windows, iOS, and Android devices were not part of these runs. `ffi` tests ran on macOS.
