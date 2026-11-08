# MPEG-4 Visual

ISO/IEC 14496-2, rectangular 4:2:0. The decoder and the encoder are separate: a tool the decoder accepts is not a tool the encoder writes.

## Decoder

Playback registers this decoder after the platform decoders. On Apple, VideoToolbox is asked first and this decoder runs when that opener is absent or hardware decode is off.

| Tool | Decoder |
| --- | --- |
| I, P, and B pictures | Accepted. With `low_delay` clear, pictures leave in display order. |
| S-VOP / GMC | Up to three warping points. A warp outside that range is refused. No brightness change. |
| Quarter-sample motion | Accepted, including the older quarter-sample chroma and filter fixes named in user data. |
| H.263 and MPEG quantisers | Both. An MPEG stream may carry its own matrices. |
| Interlace | Accepted, including field motion. |
| Data partitioning and RVLC | Accepted, except data partitioning combined with interlace. |
| Resync markers | Read inside a picture. |
| Short header | H.263 baseline. Four motion vectors and the optional H.263 modes are refused. |
| Shape, static sprites, N-bit, NEWPRED, reduced resolution, studio | Refused. |
| Scalability | A flag with no enhancement layer is ignored. A real enhancement layer is refused. |

A picture is at most 8192×8192 pixels, and each VOL edge is at most 8191. An odd edge is shown even, dropping the last column or row. Colour primaries, transfer, and matrix are passed through. The picture description has no place for the full-range flag, so that bit is not visible to playback.

## Encoder

`encode_intra` and the I-then-P stream are Simple: half-sample motion, one or four vectors, the H.263 quantiser, a fixed quantiser, and a 25 Hz clock. The size has to be a whole number of macroblocks.

`encode_advanced` adds B pictures, quarter-sample motion, and the MPEG quantiser with the default matrices. A stream with a B picture or quarter-sample motion is Advanced Simple, and `low_delay` is clear when a B picture is present. The quantiser still does not change from picture to picture. The clock is 25 Hz.

Export uses `EngineEncoder`. It is Advanced Simple and progressive:

- The first picture is an I-VOP. Each later picture is held, the next one is a P-VOP, and the held one is a B-VOP between them. A forced keyframe finishes the held picture as a P-VOP, then writes an I-VOP.
- Motion is quarter-sample, searched within ±8 pixels, so every vector fits `fcode` 1. One vector or four.
- The quantiser is H.263, from 1 to 31. A named bitrate moves it. Zero bitrate keeps the starting quantiser. `constant_rate` starts the budget again each second.
- The layer's time resolution is the picture timescale, from 1 to 65535. Export's frame-rate numerator fits. A later picture that uses another timescale is refused.
- A width or height that is not a multiple of 16 is padded by repeating the edge. The layer keeps the visible size. Both edges must be even and at most 8191.
- The configuration is the video object and the layer, and it contains no picture. Each packet is one VOP.

GMC, static sprites, interlace, data partitioning, resync markers, RVLC, short headers, custom matrices, shape, N-bit video, NEWPRED, reduced resolution, and scalability are not written. The export choice is that progressive Advanced Simple stream, not every tool in the specification.

The desktop and the phone ask a platform encoder first. This encoder runs when that opener is absent or declines the parameters. The phone's own export still writes H.264. On this Mac, `VTCompressionSessionCreate` for MPEG-4 returned `-12908` (`kVTCouldNotFindVideoEncoderErr`) and no session. Opening maps that status to unsupported, so export uses this encoder.

## Checks

`cargo test -p cutlass-mpeg4` — 53 lib tests passed, 1 ignored, and the decode and MP4 integration tests passed, including the elementary-stream corpus beside the checkout. FFmpeg 8.1.1, one thread, the simple inverse transform, and raw 4:2:0 compared bit-exactly against this decoder for intra, predicted, four-vector, bidirectional, quarter-sample, and MPEG-quantiser streams, for a 64×40 padded stream, and for 32 pictures that cross a one-second boundary. The checked-in `asp.mp4` matches that same FFmpeg decode. Without FFmpeg those checks skip with a message; `MPEG4_REQUIRE_ORACLES=1` turns each skip into a failure. The bidirectional and quarter-sample encoder streams also run without FFmpeg, as in CI: each test holds a digest of the stream and of FFmpeg's decode of it, and checks that the encoder still writes that stream and this decoder still gives that decode. The decoder fixtures under `tests/fixtures` carry FFmpeg's decode beside them and run everywhere.

A 64×48 stream at 30 pictures a second stayed within ±25% of 200, 400, and 800 kbit/s over one second.

Release gates, Apple M5 Pro, 18 cores, 48 GiB, rustc 1.99.1, `cargo test -p cutlass-mpeg4 --release --lib release_decode_and_encode_meet_their_gates -- --ignored --nocapture`. A 1280×720 simple stream, 30 pictures, decoded in 0.051 s and grew the resident set by 8688 KiB. The budget was 1 s and 64 MiB. Advanced Simple encode of 320×240, 30 pictures, at 800 kbit/s took 1.827 s and grew 768 KiB. The budget was 5 s and 64 MiB.

Decoder fuzz, debug test profile so overflow checks stay on, pixel cap 1920×1088, seed `0x4d5045344432`, `MPEG4_FUZZ_SECS=1800`. 42,014,484 inputs, 95,057,282 pictures, 11,258,426 rejected inputs, findings 0. The core decoder and the engine decoder agreed, and a good intra picture still decoded after each input. Encoder fuzz, same profile, seed `0x4d504534454e`, `MPEG4_ENC_FUZZ_SECS=1800`. 6,541,489 inputs, each successful stream decoded back to the same picture count, findings 0. Both runs were 2026-11-08 17:13:20Z through 18:13:20Z.

`../local-files` was probed with ffprobe: 63 media files, none MPEG-4 Visual. The test skips when that folder is missing.

A layer with resync markers disabled is still scanned for video packets, as FFmpeg does, until its pictures show they lack end stuffing (FFmpeg's `padding_bug_score` and `FF_BUG_NO_PADDING`). `cargo fmt -p cutlass-mpeg4 -- --check` and `cargo clippy -p cutlass-mpeg4 --all-targets -- -D warnings` are clean.
