# cutlass-webp

Native safe Rust WebP decoder and encoder in `cutlass-webp`. No `unsafe`. No libwebp, no `image` crate, no `webp` crate, no ImageIO, and no subprocess in the library. Tests and the benchmark start `cwebp`, `dwebp`, `webpinfo`, `img2webp`, `gif2webp`, `webpmux`, `sips`, ffmpeg and Python's Pillow as separate processes when they are installed, to compare against.

The support record, with every gate and measurement, is [STATUS.md](STATUS.md).

## What this crate is

`WebpDecoder` takes one WebP packet and returns its pictures. A still file is one packet and one picture. An animated file is split by the `still` container into one packet per frame: the file start through the first `ANMF`, then each `ANMF`. Each packet decodes to the whole composed canvas. A whole animated file handed over as one packet returns every canvas, timed in milliseconds.

`WebpEncoder` takes a still (`push_picture`) and returns one complete `.webp` file as one packet. It takes frames (`push_frame`) and, on `drain`, returns an animated file as one packet per frame, in the same split the `still` container reads and writes. Frames after the first are stored as the smallest even-cornered rectangle that changed.

`register_decoders` and `register_encoders` register `CodecId::Webp`. The desktop and the phone register them ahead of the platform codecs.

## Specification

RFC 9649 (WebP Image Format, 2024): the RIFF container, `VP8X`, `ICCP`, `ANIM`, `ANMF`, `ALPH`, `EXIF` and `XMP ` (skipped), and the VP8L lossless bitstream. RFC 6386 (VP8): the key frame a lossy WebP carries.

## API

`WebpDecoder::open` and `WebpEncoder::open` take a `CodecParams` whose `codec` is `CodecId::Webp`. Any other id is `Unsupported` (`webp`).

`WebpOptions` is the encoder's `extra`: four bytes, `lossless` (0 or 1), `quality` (0 to 100), `effort` (0 to 6 lossy, 0 to 9 lossless), `exact` (0 or 1). Empty `extra` is the default: lossy, quality 75, effort 4, as `cwebp`'s defaults. `WebpOptions::lossless()` is lossless effort 6. Any other `extra` is `Unsupported` (`webp encoder options`).

- Lossy `quality` and `effort` mean what `cwebp -q` and `cwebp -m` mean. Alpha is coded losslessly in `ALPH`.
- Lossless `effort` means what `cwebp -lossless -z` means. `quality` is not used.
- `exact` keeps the colour under fully transparent pixels. Without it that colour is replaced, as `cwebp` does, because it is never seen.

Decoded pictures are `StillImage` values with 8-bit samples stored as `u16`: `ColorType::Rgb`, or `ColorType::Rgba` when the file carries alpha. Animations are always `Rgba`. An `ICCP` profile is passed through on `icc_profile` and not applied. Frames are composed with libwebp's blending and disposal.

The decoder's `push` hands over each picture of `decode_picture` as straight RGBA, opaque where the file has no alpha. The encoder's audio `push` is `InvalidApi`. An empty packet is `InvalidApi`. `drain` of the decoder is empty. `reset` forgets the animation canvas, or the frames waiting in the encoder. The encoder's cancel flag makes the next call return `Cancelled`.

## Support

Decoded: every VP8L feature, VP8 key frames (every version, partitions, segments, filter deltas, simple and normal loop filters), `ALPH` raw and lossless with every filter and the level-reduction pre-processing flag, `VP8X` with `ICCP`, animations with offsets, blending and disposal, and files with `EXIF` and `XMP`.

Encoded: lossless with per-image transforms, LZ77 with a cost-based parse, a colour cache, and grouped prefix codes, at efforts 0 to 9; lossy with rate-distortion mode choice, four segments, fitted token probabilities and a searched loop filter, at methods 0 to 6; alpha; ICC profiles; animations.

Refused by name: VP8 inter frames (`vp8 inter frame in webp`), hidden frames, a bitstream whose size differs from its canvas or frame, and every case in the limits below.

Limits in `limits.rs`: longest edge 16 383 pixels, at most 8192 × 8192 pixels in one picture or canvas, at most 2^26 canvas pixels returned by one call, 4096 frames and 16 384 chunks in one packet, 4 MiB of ICC profile, and 2^25 decoding-table entries for one VP8L image. A size past those limits returns `Limit` before anything that size is allocated.

## Example

```text
cargo bench -p cutlass-webp
```

prints decode and encode throughput beside `cwebp` and `dwebp` on one 1920×1080 frame; see STATUS.md for the numbers of the last run.

## Gaps

- No trellis quantization, so lossy methods 5 and 6 are a second pass with fitted costs and not libwebp's trellis. The quality gates pass without it.
- Lossless efforts 3 and above are three to five times slower than `cwebp -z` at the same effort (eleven at effort 8), and 1.5 to 3 percent larger.
- Lossy encoding is several times slower than `cwebp` (see STATUS.md). One thread; no SIMD.
- `EXIF` and `XMP ` are skipped on decode and not written. ICC is passed through, not applied.
- No incremental decoding: a packet is a whole still or a whole frame.
- The VP8 key-frame code here is this crate's own. A shared `cutlass-vp8` crate is on another branch; once it lands, the decoder side can move there.
