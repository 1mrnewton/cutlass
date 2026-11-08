# WebP support record

Checkout: branch `webp`, cut from `development` at `57498d9b`. Native safe Rust in `cutlass-webp`: `#![forbid(unsafe_code)]`, workspace lints, no `#[allow]`. No libwebp, no `image` crate, no `webp` crate, no ImageIO, and no subprocess in the library. The still container is another crate.

This file is the scope record. A row is verified only when the test named here ran and passed. Every number below is from a run on this machine, not an estimate.

## Specifications and sources read

Normative:

- RFC 9649, WebP Image Format (November 2024): RIFF layout, `VP8X`, `ICCP`, `ANIM`, `ANMF`, `ALPH`, `EXIF`, `XMP `, and the VP8L lossless bitstream (section 3).
- RFC 6386, VP8 Data Format and Decoding Guide: the key frame a lossy WebP carries, the boolean coder, tokens, the transforms and the loop filter.

Read before the Cutlass code, for the algorithm and the edge cases. No source was copied into this repo, and none is linked.

- FFmpeg 8.1.1 `libavcodec/webp.c` (VP8L, `ALPH`, the RIFF walk), `vp8.c` (key frames), `libwebpenc*.c` (a libwebp wrapper, no encoder of its own), and `libavformat/webp_anim_dec.c`, `webpenc.c`.
- GStreamer `subprojects/gst-plugins-bad/ext/webp` (`gstwebpdec.c`, `gstwebpenc.c`): both call libwebp's `WebPDecode` and `WebPEncode` and implement neither bitstream.
- libwebp 1.6.0 is the reference implementation and the oracle (`cwebp`, `dwebp`, `webpinfo`, `img2webp`, `gif2webp`, `webpmux`, and Pillow 11.3.0's animation decoder). Its source is not on this machine and was not downloaded; its encoder choices (quantizer matrices, `QualityToCompression`, segment analysis, lambdas, chroma DC error diffusion, flatness penalties, token-probability refresh, transparent-area cleanup) are from its documented behaviour and checked against its output, not copied.
- The VP8 tables (`lossy/tables.rs`) are the RFC 6386 tables, as on the `vp8` branch's `crates/codecs/vp8`.

Where FFmpeg and libwebp differ, this crate follows the RFC and libwebp. FFmpeg clamps a VP8L backward reference that starts before the picture to the first pixel; libwebp and this crate refuse it.

## What this crate is

`WebpDecoder` and `WebpEncoder` speak one WebP file per still packet, and the `still` container's split for animations: the file start through the first `ANMF`, then one `ANMF` per packet. Each animation packet decodes to the whole composed canvas, kept across packets until `reset`. A whole animated file as one packet returns every canvas, timed in milliseconds.

The encoder turns a still into one file and a run of frames into an animation on `drain`. Frames after the first are coded as the smallest even-cornered rectangle that changed, with no blending and no disposal, so every frame is exact for lossless and independent of the previous frame's rounding for lossy. A frame equal to the last is a 1×1 frame.

`register_decoders` and `register_encoders` register `CodecId::Webp`. `apps/desktop` and `crates/ffi` register this crate ahead of the platform codecs: ImageIO can open a still WebP but not a bare `ANMF` packet, so an animated file must reach this decoder.

## Resource limits

Declared before the decoder, in `limits.rs`:

| Limit | Value |
| --- | --- |
| Edge of one bitstream | 16 383 (the format's 14 bits) |
| Pixels in one picture or canvas | 2^26 (8192 × 8192) |
| Canvas pixels returned by one call | 2^26 |
| Frames in one packet | 4096 |
| Chunks in one packet | 16 384 |
| ICC profile | 4 MiB |
| VP8L decoding-table entries for one image | 2^25 |

A hostile size returns `Limit` before anything that size is allocated: a 5-byte VP8L header for 16 383 × 16 383, a 2^24 × 2^24 canvas, and 4097 frames in one packet are tests.

## Gates

Set before the encoders were tuned. They have not been changed since.

| Gate | Budget |
| --- | --- |
| Independent decode | `cwebp`, `img2webp`, `gif2webp`, `webpmux`, ffmpeg's `libvpx` key frames: this decoder's RGBA equals `dwebp`'s exactly, its YUV planes equal `dwebp -yuv`, and opaque lossy pictures equal ffmpeg's native decoder; animations equal Pillow's libwebp canvas by canvas |
| Lossless encode | Exact pixels back from `dwebp` and from this decoder, `webpinfo` clean, ImageIO opens it. Size reported against `cwebp -lossless -z N -exact` |
| Lossy encode, per picture and quality | PSNR at most 1.0 dB and SSIM at most 0.02 below `cwebp` at our size; size 0.6× to 1.6× `cwebp` at the same `-q` and `-m` |
| Lossy encode, mean of a run | PSNR at most 0.5 dB and SSIM at most 0.01 below; geometric mean size 0.8× to 1.25× |
| Lossy decode everywhere | Every encoded file decodes identically in this crate, `dwebp` and ffmpeg; `webpinfo` clean; ImageIO within 30 dB PSNR (it upsamples chroma its own way) |
| Alpha | Exact through `ALPH` in every file |
| Hostile input | An error, never a panic; the next packet on the same decoder still decodes |
| Fuzz | In-tree mutation campaign of at least 1 000 000 inputs with no panic, plus random encoder settings |
| Workspace | `cargo fmt`, clippy with `-D warnings` on macOS and for `x86_64-unknown-linux-gnu`, tests of every touched crate |

PSNR is over RGB where alpha is not zero. SSIM is over BT.601 luma in 8×8 windows, step 4. `cwebp`'s quality at our size is interpolated in log size between the two `-q` values whose files bracket ours.

## Capability matrix

| Capability | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- |
| Simple lossless (`VP8L`) | Every transform, colour cache, meta prefix codes | Efforts 0 to 9, transforms chosen per image | `cwebp_lossless_files_decode_to_the_exact_pixels_of_dwebp_and_imageio`, `encoded_files_decode_exactly_in_dwebp_webpinfo_and_imageio`, `every_effort_round_trips` | Verified |
| Simple lossy (`VP8 `) | Key frames of every version, 1 to 8 partitions, segments, filter deltas, simple and normal filters | Methods 0 to 6, quality 0 to 100 | `cwebp_lossy_files_decode_to_the_exact_pixels_of_dwebp_and_ffmpeg` (nine `cwebp` settings), `libvpx_key_frames_with_partitions_and_filter_deltas_match_dwebp`, `lossy_files_decode_everywhere_and_meet_the_gates`, `every_method_and_odd_sizes_decode_like_dwebp` | Verified |
| Extended (`VP8X`) with `ALPH` | Raw and lossless alpha, all four filters, pre-processing flag | Lossless alpha, best of four filters from method 3 | `cwebp_alpha_in_every_coding_matches_dwebp`, `every_alpha_filter_matches_dwebp` | Verified |
| `ICCP` | Passed through on `icc_profile`, not applied | Written from `icc_profile` | `an_icc_profile_is_written_and_read_back`, `icc_profiles_pass_through_and_alpha_sets_the_colour_type` | Verified |
| `EXIF`, `XMP ` | Skipped | Not written | `an_extended_still_carries_its_canvas_and_icc` | Verified, not applied |
| Animation (`ANIM`, `ANMF`) | Offsets, blending, disposal, loop count, durations, lossy and lossless frames mixed | Changed-rectangle frames, loop count through the muxer | `animations_compose_like_libwebp_packet_by_packet`, `lossless_animations_come_back_frame_for_frame`, `lossy_animations_and_stills_decode_like_libwebp` | Verified |
| Still container end to end | One packet per still, one per frame | Muxed by `StillMux` | `a_cwebp_picture_goes_through_the_container_and_back_out_to_libwebp_and_imageio`, `an_animation_from_a_local_video_goes_through_the_container_and_back_out` | Verified |
| App import and export | Desktop and phone factories | Desktop: Save still as `.webp`; export dialog GIF row has a WebP format | `the_desktop_codes_webp_in_rust_ahead_of_the_platform`, `a_still_lands_the_canvas_at_the_playhead_as_png_or_jpeg`, `an_export_writes_a_mov_the_sound_alone_a_gif_and_a_caption_file`, `the_phone_decodes_webp_in_rust`, export's `a_webp_export_writes_an_animation_the_still_container_reopens` | Verified |
| VP8 inter frames, hidden frames | Refused (`vp8 inter frame in webp`, `vp8 hidden frame`) | Not emitted | `a_partition_past_the_end_and_a_bool_decoder_overrun_are_errors` | Verified, refused |
| Hostile and truncated input | Error, no panic | n/a | `tests/malformed.rs`, `tests/fuzz.rs` | Verified |
| Trellis quantization, `-sns`/`-f`/`-sharpness` knobs, target size or PSNR | n/a | Not implemented; libwebp's defaults are fixed | Recorded so a caller does not expect them | Unsupported |
| Incremental decoding | A packet is whole | n/a | | Unsupported |

## Interoperability consequences

- The decoder matches `dwebp` byte for byte, including the loop filter, fancy chroma upsampling and alpha filtering. ImageIO upsamples chroma differently and premultiplies alpha, so it is compared by PSNR.
- `ANIM`'s background colour is ignored and disposal clears to transparent, as libwebp's animation decoder does.
- The lossy encoder replaces the colour under fully transparent 8×8 blocks with the first block of their run, and weights chroma by alpha where a 2×2 block is partly transparent, as `cwebp` does without `-exact`. `exact` keeps that colour.
- A small picture (64 macroblocks or fewer) also tries the other qualities in steps of 10 and keeps a frame that is no larger and no worse than the one asked for. A flat 64×64 colour at `-q 30` comes out exact at 80 bytes, as `cwebp -q 100` does.

## Lifecycle

`tests/lifecycle.rs`. One still packet is one picture. A failed packet leaves the decoder usable. A later animation packet with no first packet before it is refused, and a failed frame leaves the canvas as it was. `reset` forgets the canvas. The encoder buffers frames until `drain`; a second `drain` is empty; frames after a drain start a new file; `reset` drops waiting frames; a still in the middle of an animation is its own file. A raised cancel flag makes `push_frame`, `push_picture` and `drain` return `Cancelled`. The decoder's `push` hands over each picture as straight RGBA, opaque where the file has no alpha; audio `push` on the encoder is `InvalidApi`.

## Lossless size against `cwebp`

22 PNG pictures (photographs, frames of the local videos, screenshots, synthetic gradients, alpha and odd sizes), `cwebp 1.6.0 -lossless -exact -z N` against this encoder at effort N, release, Apple M5 Pro. Size is ours over theirs, summed over the set; time is the total for the set.

| Effort | Size | Ours | cwebp |
| --- | --- | --- | --- |
| 0 | 0.933 | 0.69 s | 0.41 s |
| 1 | 1.025 | 0.80 s | 1.46 s |
| 2 | 1.042 | 0.84 s | 2.10 s |
| 3 | 1.029 | 8.15 s | 2.51 s |
| 4 | 1.028 | 8.78 s | 2.66 s |
| 5 | 1.021 | 9.97 s | 2.82 s |
| 6 | 1.022 | 13.79 s | 3.45 s |
| 7 | 1.018 | 16.97 s | 4.21 s |
| 8 | 1.016 | 81.07 s | 7.27 s |
| 9 | 1.015 | 210 s | 47 s |

The effort 8 time is from before its optimal parse dropped to one round.

## Lossy quality against `cwebp`

`WEBP_SOURCE_DIR=<set> cargo test --release -p cutlass-webp measure_lossy_against_cwebp -- --ignored --nocapture`, method 4, qualities 30, 50, 75, 90, the same 22 pictures (88 encodes). Every per-picture and mean gate passed.

| Set | Mean PSNR deficit | Worst | Mean SSIM deficit | Worst | Size at same `-q` (geometric mean, range) |
| --- | --- | --- | --- | --- | --- |
| All 88 | −2.30 dB | 0.31 dB (checker q90) | −0.0038 | 0.0014 (webm540 q30) | 0.981 (0.76 to 1.04) |
| 60 larger than 64 macroblocks | −0.64 dB | 0.31 dB | −0.0053 | 0.0014 | 0.991 (0.76 to 1.04) |

A negative deficit is better than `cwebp`. Photographs land within 0.1 dB either side: `uhd720` q75 33.81 vs 33.80 dB, `street1080` q75 38.67 vs 38.15 dB, `portrait720` q50 36.73 vs 36.72 dB. The test in the default suite runs the synthetic picture, a frame of the first local video, and the local PNG at qualities 30, 75 and 95.

## Hostile input

`tests/malformed.rs` crafts each case with a bit writer, so the named check is the one that fires:

- A backward reference before the first pixel (the legal copy beside it decodes), and a copy past the last pixel.
- An over-subscribed code-length code.
- A transform used twice.
- A colour cache size outside 1 to 11.
- Every truncation of a VP8L, VP8 and ALPH bitstream and of a whole extended file.
- A partition that runs past the end, and a boolean decoder that reads past its data.
- An inter frame.
- The size limits above.
- An `ANMF` frame outside its canvas.

`tests/fuzz.rs` mutates six seeds this crate writes (lossless and lossy, with and without alpha, with ICC, and two animations). The mutations are bit flips, chosen bytes, cuts, splices, random bytes, chunk-size patches and deletions. Each mutant is decoded whole and packet by packet, inside `catch_unwind`. The default run is 4000 rounds. A second campaign encodes random sizes, pictures and settings, and decodes what it wrote: lossless pixels must be exact, and alpha must be exact for every setting.

Campaign of record: four seeds of 250 000 rounds each (1 000 000 inputs), debug build. The command for each seed was `WEBP_FUZZ_ROUNDS=250000 WEBP_FUZZ_SEED=$seed cargo test -p cutlass-webp mutated_files_decode`.

- No panic. About 44 500 inputs per seed still decoded to a picture.
- The first campaign found one panic: an `i32` multiply overflowing in the VP8 inverse transform. It now wraps as libwebp's C does, and the edge coefficients are a regression test in `lossy/dsp.rs`.
- The slowest input took 5.1 s in debug and 69 ms in release. It was 398 bytes declaring a 4 194 325 × 13 canvas, inside the 2^26-pixel limit, so the cost is the canvas it asks for.
- Encoder campaign: 3000 rounds, no failure.

## Throughput

`cargo bench -p cutlass-webp`, release, one thread, Apple M5 Pro, macOS 27.0.1. The picture is one 1920 × 1080 frame of `12129324_3840_2160_30fps.mp4` from `local-files`. Each of our times is the median of five runs. The libwebp 1.6.0 times are its own `-v` timings, with no process start-up. The decode of both files was checked equal to `dwebp` first.

| Operation | Cutlass | libwebp |
| --- | --- | --- |
| Decode lossy q75 m4 (283 554 bytes) | 25.4 ms, 81.5 MP/s | `dwebp` 13.0 ms |
| Decode lossless z6 (1 985 010 bytes) | 37.5 ms, 55.3 MP/s | `dwebp` 24.0 ms |
| Encode lossy q75 m0 | 102.7 ms, 20.2 MP/s, 315 182 bytes | 29.0 ms, 71.5 MP/s, 339 448 bytes |
| Encode lossy q75 m4 | 363.0 ms, 5.7 MP/s, 280 544 bytes | 99.0 ms, 20.9 MP/s, 283 554 bytes |
| Encode lossy q75 m6 | 568.7 ms, 3.6 MP/s, 280 214 bytes | 219.0 ms, 9.5 MP/s, 268 478 bytes |
| Encode lossless z0 | 128.3 ms, 16.2 MP/s, 2 111 580 bytes | 53.0 ms, 39.1 MP/s, 2 093 936 bytes |
| Encode lossless z6 | 1417.7 ms, 1.5 MP/s, 1 983 092 bytes | 607.0 ms, 3.4 MP/s, 1 985 010 bytes |

libwebp uses SIMD, and this crate uses none. At method 6, `cwebp`'s trellis quantization makes a file 4 percent smaller than ours on this frame.

## Readiness

Decode and encode of WebP stills and animations are in `cutlass-webp`, registered in the desktop and phone factories, and reachable from export. This is not a release.

### Not run

- `cargo-fuzz`: not installed, and not installed for this work. The in-tree campaign above stands in; it is not coverage-guided.
- Official WebP conformance files beyond what libwebp's tools and libvpx write.
- Linux and Windows execution. Linux was cross-checked with clippy only.
- A peak-RSS profile.
- `cargo deny`.
- Lossy corpus runs at methods other than 4. Methods 0 to 6 are covered for decodability and odd sizes by `every_method_and_odd_sizes_decode_like_dwebp`, but not against the quality gates.
