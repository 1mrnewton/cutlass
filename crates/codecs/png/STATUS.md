# PNG support record

Checkout: branch `png` at `eb60bbf` when this scope was written. Native safe Rust in `cutlass-png`. No `unsafe`. No libpng, no `png` crate, no ImageIO, and no subprocess in the library.

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
| Signature, IHDR, IDAT, IEND, CRC | Yes | Accept a legal datastream. Bad CRC, bad signature, truncation, and trailing bytes are errors | Emit this order | Planned tests in this crate | Planned |
| Greyscale 1, 2, 4, 8, 16 | Yes | Exact samples in range | Emit | Analytic plus ffmpeg | Planned |
| Truecolor 8 and 16 | Yes | Exact R, G, B | Emit | Analytic plus ffmpeg | Planned |
| Greyscale and truecolor with alpha, 8 and 16 | Yes | Exact straight alpha | Emit | Analytic plus ffmpeg | Planned |
| Indexed 1, 2, 4, 8, with PLTE | Yes | Exact indices. Palette kept | Emit | Analytic plus an independent decoder | Planned |
| `tRNS` key and palette alpha | Yes | Kept, not expanded | Emit | Metadata compared apart from samples | Planned |
| Filter method 0, types 0–4, including Paeth | Yes | Reconstruct per the spec | Choose per scanline by minimum sum of absolute byte values | Hand-computed rows plus independent files | Planned |
| Interlace 0 and Adam7 | Yes | Full image | Both. `CodecParams.extra` empty or `[0]` is method 0. `[1]` is Adam7 | Independent Adam7 file | Planned |
| Multiple IDAT, including empty IDAT | Yes | Concatenate. Non-consecutive IDAT is an error | Split the zlib stream | Planned | Planned |
| gAMA, cHRM, sRGB, iCCP, sBIT, pHYs, bKGD, tEXt, zTXt, iTXt | Yes to preserve, no to apply | Record. Do not convert | Emit when set. Do not invent an sRGB gAMA | Metadata tests | Planned |
| Unknown ancillary chunk | Yes | Ignore, including APNG `acTL`, `fcTL`, and `fdAT` | Do not emit | Default image still decodes | Planned |
| Unknown critical chunk | Yes | Error | Do not emit | Planned | Planned |
| Filter method other than 0 | Yes | Error. FFmpeg accepts MNG method 64 (LOCO). Those files do not decode here | Method 0 only | Planned | Planned |
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

Not run. The matrix rows above are planned. No implementation is claimed in this revision of the file.
