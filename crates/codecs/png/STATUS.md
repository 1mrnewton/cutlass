# PNG support record

Checkout: branch `png`. Scope recorded at `eb60bbf`. The decoder and encoder landed in `4a82563`. Native safe Rust in `cutlass-png`. No `unsafe`. No libpng, no `png` crate, no ImageIO, and no subprocess in the library.

This file is the scope record. The readiness section at the bottom is filled after the gates run. A row stays planned until a test named here has been executed.

## Specifications and sources read

Normative:

- W3C Portable Network Graphics (PNG) Specification (Third Edition), Recommendation, 24 June 2025, `https://www.w3.org/TR/png-3/`. This is the syntax this crate implements. It is not yet an ISO standard.
- ISO/IEC 15948:2004, the current International Standard, equivalent to the W3C Second Edition. Static-image rules used here agree with that edition. Third-edition additions (APNG, `cICP`, `mDCV`, `cLLI`, `eXIf`) are not required for a still.
- Errata for the Third Edition, `https://www.w3.org/2025/06/REC-PNG-20250624-errata` (10 September 2025). The only entry is a non-conformance correction to an `mDCV` example (`3C 05` to `3D 13` for white-point x). This crate does not emit that example.
- RFC 1950 (zlib wrapper) and RFC 1951 (deflate). The IDAT payload is one zlib stream. Compression method 0, window at most 32768 bytes, no preset dictionary. Adler-32 is checked. Chunk CRC-32 is ISO-3309 / ITU-T V.42, the same polynomial as zlib's CRC-32.

Also read before the Cutlass code, for the algorithm and the edge cases:

- FFmpeg 8.1.1 `libavcodec/pngdec.c`, `pngenc.c`, and `png.c` (the installed ffmpeg is 8.1.1). Filter reconstruction, Adam7 placement, and the per-scanline filter heuristic.
- libpng 1.6.58 `pngrutil.c` (`png_read_filter_row_*`, `png_do_read_interlace`) and the installed headers.
- GStreamer 1.28.3 `gst-plugins-good` `ext/libpng/gstpngdec.c`. The plugin does not implement inflate or filters. It calls libpng and then asks libpng to expand the picture.

Where those projects disagree with the specification, this crate follows the specification. The consequences are in the matrix.

## What this crate is

`PngDecoder` and `PngEncoder` speak one PNG datastream per packet. A still is one packet and one picture. The `still` container is still a stub, so a file test hands the file bytes to the codec as that packet. Opening the file is not a second PNG parser.

Samples stay in the declared color type, bit depth, and alpha domain. 1-bit, 2-bit, and 4-bit values stay in `0 .. 2^depth - 1`. They are not scaled to 8-bit. 16-bit samples stay 16-bit, big-endian on the wire and host integers in `StillImage`. Indexed pixels stay indices. `tRNS` stays a color key or a palette alpha table. Gamma, chromaticity, sRGB, and ICC are recorded and not applied. The color crate owns conversion. Alpha is straight, not premultiplied.

The shared `Decoder` and `Encoder` traits stay audio-shaped for AAC. A still uses `decode_picture` and `push_picture`. The factories register `CodecId::Png` the same way AAC registers itself. Tests open that registration. The desktop binary does not build a factory yet.

## Resource limits

Declared before implementation:

| Limit | Value |
| --- | --- |
| Edge | 16 384 pixels |
| Pixels | 16 777 216 |
| One chunk | 16 MiB |
| Concatenated IDAT | 128 MiB |
| Inflated image | The exact filtered size implied by IHDR, or an error |
| ICC and each text chunk | Inside the chunk cap; text payload after inflate at most 1 MiB |
| zlib window | The CINFO window, at most 32 768 |
| Palette | 256 entries |

A dimension above `2^31 - 1` is illegal PNG even when it fits in `u32`.

## Gates

Lossless means byte-identical samples in the declared domain. Metadata is compared on its own. No encoder byte-identity requirement.

| Gate | Budget |
| --- | --- |
| Independent decode | ffmpeg 8.1.1 and ImageMagick (libpng) files, exact samples |
| Independent encode | Those tools decode this encoder, exact samples |
| Analytic fixtures | Hand-built stored zlib and hand-computed filters, exact samples |
| Self round trip | Extra coverage, not the proof |
| Decode throughput | 1920×1080 RGBA8, one thread, release, under 250 ms on this machine |
| Encode throughput | The same frame, under 500 ms |
| Memory | One image, one previous scanline, and a 32 KiB distance window. No growth across images |
| Fuzz | At least 30 minutes on the decoder entry, random bytes, recorded seed and findings. A shorter run is not that gate |
| Workspace | `cargo fmt --all -- --check`, `cargo lint`, `cargo test --workspace --all-targets`, `cargo deny check` |

Hardware for the timing gates: the machine that runs them, named in the readiness section when the numbers exist.

## Capability matrix

| Capability | Required | Decode | Encode | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Signature, IHDR, IDAT, IEND, CRC | Yes | Accept a legal datastream. Bad CRC, bad signature, truncation, and trailing bytes are errors | Emit this order | `tests.rs` hand-built streams and malformed cases | Verified |
| Greyscale 1, 2, 4, 8, 16 | Yes | Exact samples in range | Emit | 1-bit analytic and ffmpeg `monob`. 2-bit and 4-bit ImageMagick. 8-bit ImageMagick both ways. 16-bit ffmpeg both ways. Adam7 placement for 1, 2, and 4 | Verified |
| Truecolor 8 and 16 | Yes | Exact R, G, B | Emit | ffmpeg 8.1.1 both ways for 8-bit, including `-pred mixed`. ffmpeg `rgb48be` both the foreign file and this encoder | Verified |
| Greyscale and truecolor with alpha, 8 and 16 | Yes | Exact straight alpha | Emit | ffmpeg `ya8`, `rgba`, `ya16be`, and `rgba64be` foreign files, plus this encoder's `ya8` and `rgba` | Verified |
| Indexed 1, 2, 4, 8, with PLTE | Yes | Exact indices. Palette kept | Emit | Analytic 8-bit indices. ffmpeg `pal8` both ways for 8-bit. 1/2/4 self round trip | Verified for 8-bit. Independent 1/2/4 not run |
| `tRNS` key and palette alpha | Yes | Kept, not expanded | Emit | Hand-built key and short palette table. Self round trip. Not applied to samples | Verified in this crate. No independent metadata oracle |
| Filter method 0, types 0–4, including Paeth | Yes | Reconstruct per the spec | Choose per scanline by minimum sum of absolute byte values | Hand-computed None, Sub, Up, Average, and Paeth. ffmpeg `-pred mixed` | Verified |
| Interlace 0 and Adam7 | Yes | Full image | Both. `CodecParams.extra` empty or `[0]` is method 0. `[1]` is Adam7 | Hand-built 2×2 Adam7. ImageMagick `-interlace PNG` both ways. Self round trip of every required depth | Verified |
| Multiple IDAT, including empty IDAT | Yes | Concatenate. Non-consecutive IDAT is an error | Split the zlib stream at 8192 bytes | One-byte IDAT chunks plus an empty IDAT. A `tEXt` between IDAT chunks is an error | Verified |
| gAMA, cHRM, sRGB, iCCP, sBIT, pHYs, bKGD, tEXt, zTXt, iTXt | Yes to preserve, no to apply | Record. Do not convert | Emit when set. Do not invent an sRGB gAMA | Samples unchanged beside the metadata. Round trip keeps gamma, sRGB, cHRM, pHYs, text, ICC, sBIT, bKGD, and a compressed iTXt. A `tEXt` after IDAT is kept. A keyword with a leading space is refused | Verified in this crate |
| Unknown ancillary chunk | Yes | Ignore, including APNG `acTL`, `fcTL`, and `fdAT` | Do not emit | `orNG`, `acTL`, and `fcTL` leave the samples intact. `fdAT` before IDAT is ignored. `fdAT` between IDAT chunks is an error | Verified |
| Unknown critical chunk | Yes | Error | Do not emit | `ORNG` is rejected | Verified |
| Filter method other than 0 | Yes | Error. FFmpeg accepts MNG method 64 (LOCO). Those files do not decode here | Method 0 only | Method 64 returns `Unsupported` | Verified |
| APNG animation | No | The static IDAT image only | Not emitted | Outside the still | Unsupported |
| `cICP`, `mDCV`, `cLLI`, `eXIf`, `hIST`, `sPLT`, `tIME` | No | Ignored ancillary | Not emitted | Not preserved | Unsupported |
| Color conversion, premultiplied alpha, sample scaling | No | Not done | Not done | Color crate | Unsupported |
| ImageIO and other platform PNG codecs | No | Not called | Not called | This crate is the software path | Outside this implementation |

## Interoperability consequences

- GStreamer 1.28.3 `pngdec` expands grey below 8 bits, palette images, and grey-with-alpha before the application sees the buffer. Its buffers are not the coded sample domain. Comparison with GStreamer pixels is not a pass for this crate.
- libpng `png_do_read_interlace` can replicate pixels for a display transform. Reconstruction here uses the Adam7 pass geometry in the specification, the same placement FFmpeg's `png_put_interlaced_row` performs.
- Paeth ties prefer the left predictor, then the above predictor. That is the specification order. libpng's row function uses a strict `<` test and lands on the same predictor.
- FFmpeg filter method 64 is rejected.
- An sRGB writer in this crate does not add gAMA 45455 or the sRGB cHRM values unless the caller set them. The specification recommends those chunks and requires those numbers only when the chunks are present.
- Unused low bits at the end of a packed scanline are written as zero. The specification leaves them unspecified. A decoder ignores them.
- `tRNS` samples narrower than 16 bits are masked to the bit depth before they are stored.

## Lifecycle

One push is one complete datastream. A short buffer is `Truncated`, not `NeedMoreData`. The container owns partial reads. A failed push leaves the decoder unchanged. `drain` is empty, including a second drain. `reset` clears the cancel flag on the encoder. `push` of an empty packet is `InvalidApi`. Audio `push` on this codec is `InvalidApi`. There is no inter-frame state.

## Readiness

Partial. Decode and encode of a static PNG datastream are in `cutlass-png` and the required sample domains round-trip. The rows marked partial above are the gaps. This is not a release.

### Implementation

`PngDecoder::decode_picture` and `PngEncoder::push_picture` take one PNG datastream per packet. zlib and the filters live in this crate. `register_decoders` and `register_encoders` register `CodecId::Png`. The `still` container is still an empty stub, so file tests pass the file bytes as the packet. `apps/desktop` does not build a factory, so it does not call those register functions.

### Rules

Followed `.cursor/skills/ship-codec/SKILL.md` and `references/video-image.md`, `implementation.md`, `verification.md`, and `deliverables.md`, plus the rules under `.cursor/rules`. Product boundaries follow `cutlass-docs/rust-workspace.md` and `cutlass-components.md`. No missing skill file.

### Correctness

Lossless comparison is exact samples in the declared domain. Metadata is compared on its own and is not applied.

- Analytic stored-zlib streams, not produced by this encoder: grey, Sub, Up, Average, Paeth, 1-bit, 16-bit, indexed plus `tRNS`, and a 2×2 Adam7 image.
- ffmpeg 8.1.1 (`/opt/homebrew/bin/ffmpeg`) writes files this decoder matches for `rgb24` (`-pred mixed` included), `rgba`, `ya8`, `ya16be`, `rgb48be`, `rgba64be`, `pal8` indices, and `monob`. It also matches this encoder for `rgb24`, `rgba`, `ya8`, `gray16be`, `rgb48be`, and `pal8`.
- ImageMagick 7.1.2-21 writes 2-bit and 4-bit grey and an Adam7 grey file this decoder matches, and it matches this encoder's Adam7 grey file. Installed libpng headers are 1.6.58. The magick binary's linked libpng soname was not printed by `otool`.
- python3 3.9.6 zlib: inflate of level-9 streams, and decompress of this encoder's streams, for lengths 0 through 95 plus the earlier samples.
- Local file `/Users/mrnewton/dev/cutlass/local-files/04-motion-graphics.png` (2946×1603, 8-bit RGBA, no interlace, 836790 bytes) matches ffmpeg `rgba`. Re-encoding it and decoding with ffmpeg matches those same bytes. The file was not modified.
- ISO/IEC 15948 official conformance suite: not available. Not run.
- GStreamer 1.28.3 is a libpng wrapper that expands the picture. Not used as a coded-sample oracle.

### Encoding quality

Lossless. The release timing test encodes a frame and decodes it back to the same samples. No lossy ladder.

### Lifecycle

Covered in `tests.rs`: empty packet is `InvalidApi`, audio `push` is `InvalidApi`, `drain` twice is empty, a failed picture does not block the next one, cancel then `reset` allows another encode, and the first factory registration wins. Playback and export are not wired. Not run.

### Robustness

Fast property loop: 24 random stills, seed `0x1234_5678_9abc_def0`, both interlace modes, exact samples. 48 random buffers up to 80 bytes return errors. Malformed fixtures cover CRC, truncation, signature, filter type 5, filter method 64, a zero edge, trailing bytes, non-consecutive IDAT, a critical unknown chunk, missing PLTE, PLTE on grey, `tRNS` on RGBA, `gAMA` after PLTE, and an edge of 20000.

Fuzz campaign, release, one thread, `decoder_fuzz_campaign`: seed `0x504e47f00d0a1a0a`, 1800 seconds, 2 760 000 674 inputs, exit 0, no panic. Two thirds of the inputs are random buffers of 0 to 2047 bytes. One third flips one byte in a stored-zlib 2×2 truecolor PNG. This is not a coverage-guided fuzzer and it did not minimize a crash, because there was no crash. A structured hostile corpus beyond that mutation is the malformed tests above.

### Build

On this machine, after `4a82563`:

- `cargo fmt --all -- --check` exit 0
- `cargo lint` exit 0
- `cargo test --workspace --all-targets --offline` exit 0. PNG: 25 passed, 2 ignored
- `cargo deny check` exit 0. Advisories, bans, licenses, and sources ok. The unmatched license allowances in `deny.toml` were already there. No new dependency.

`#![forbid(unsafe_code)]` stays. Library code has no `unwrap` or `expect`.

### Performance

Apple M5 Pro, Darwin arm64, 48 GiB, rustc 1.99.1, `--release`, one thread. `full_hd_rgba_stays_inside_the_time_budget` below is the earlier fixed-Huffman run. The bench table is the current compressor.

| Frame | Encode | Decode | Output |
| --- | --- | --- | --- |
| 1920×1080 RGBA8 gradient | 53 ms | 34 ms | 54958 bytes |
| 1920×1080 RGBA8 high-bit noise | 159 ms | 61 ms | 8760375 bytes |

Both of those runs were inside 500 ms encode and 250 ms decode. Linux and Windows were not run.

`cargo bench -p cutlass-png` (`benches/against_ffmpeg.rs`) repeats each pass 5 times after one warmup and reports the median wall time. Same machine, release, one thread. ffmpeg 8.1.1 at `/opt/homebrew/bin/ffmpeg`. Pixels matched before the timed passes. An ffmpeg row includes process start. `ffmpeg -version` itself is 18.6 ms (18.4–19.0). Cutlass stays in-process. Encode `out` is the PNG size. Decode `out` is the raw sample count.

Matches are copied in runs into one reused scanline. A contiguous row is widened straight into the sample buffer. Literals still land one byte at a time. Sub, Average, and Paeth reconstruct each byte of a pixel together, because those bytes do not depend on their neighbors. The encoder picks a dynamic Huffman block when it is smaller than a fixed block, probes the previous scanline, and defers a short match when the next byte matches further. Once a copy of 32 bytes is in hand, the search checks only a few more places. A length-3 match farther than 4096 bytes is stored as literals.

| Picture | Side | Op | Median | Out | Settings |
| --- | --- | --- | --- | --- | --- |
| 1920×1080 RGBA8 gradient | cutlass | encode | 36.1 ms | 9652 | filter search, dynamic or fixed Huffman |
| 1920×1080 RGBA8 gradient | ffmpeg | encode | 46.2 ms | 59164 | pred Paeth, zlib default |
| 1920×1080 RGBA8 gradient | ffmpeg | encode | 53.6 ms | 10548 | pred mixed, zlib default |
| 1920×1080 RGBA8 gradient | cutlass | decode | 19.2 ms | 8294400 | the cutlass PNG |
| 1920×1080 RGBA8 gradient | ffmpeg | decode | 27.4 ms | 8294400 | the same PNG |
| 1920×1080 RGBA8 noise | cutlass | encode | 237.3 ms | 8311144 | filter search, dynamic or fixed Huffman |
| 1920×1080 RGBA8 noise | ffmpeg | encode | 130.4 ms | 8322399 | pred Paeth, zlib default |
| 1920×1080 RGBA8 noise | ffmpeg | encode | 144.4 ms | 8322399 | pred mixed, zlib default |
| 1920×1080 RGBA8 noise | cutlass | decode | 57.7 ms | 8294400 | the cutlass PNG |
| 1920×1080 RGBA8 noise | ffmpeg | decode | 57.3 ms | 8294400 | the same PNG |
| motion-graphics.png | cutlass | decode | 33.8 ms | 18889752 | the file on disk |
| motion-graphics.png | ffmpeg | decode | 63.6 ms | 18889752 | the same file |
| motion-graphics.png | cutlass | encode | 139.3 ms | 720638 | filter search, dynamic or fixed Huffman |
| motion-graphics.png | ffmpeg | encode | 109.1 ms | 707538 | pred Paeth, zlib default |
| motion-graphics.png | ffmpeg | encode | 139.1 ms | 702900 | pred mixed, zlib default |

On the real file, 1110 of 1603 rows are Paeth and 433 are Up. Inflating that file is 19 ms and the whole decode is 34 ms. ffmpeg's wall time on it is 64 ms. Decode wall time is ahead of ffmpeg on these three pictures. On the noise frame their inflater itself is still faster, because that picture is almost all literals and ffmpeg's row still includes about 19 ms of process start. The gradient encodes to 9652 bytes, smaller than ffmpeg's mixed filter at 10548. The real file encodes to 720638 bytes in 139 ms, against ffmpeg mixed at 702900 bytes in 139 ms. Noise encodes to 8311144 bytes against ffmpeg 8322399, in 237 ms, inside the 500 ms budget.

Memory: the inflater keeps a distance window of the size named in the zlib header, at most 32 KiB. Reconstruction keeps one scanline and the previous scanline, and writes the sample buffer directly. The inflated image is not retained. No heap profile. The sample buffer is the one image the gate names. Linux and Windows were not run.

### Conclusion

Partial. Static PNG decode and encode work for the domains covered above, with the independent evidence named there. The 30 minute decoder campaign finished with no panic. Still open: independent indexed 1/2/4-bit files, the still-container open path, desktop registration, the ISO conformance suite, and Linux/Windows. APNG animation and the ignored metadata chunks stay unsupported. Do not publish this.
