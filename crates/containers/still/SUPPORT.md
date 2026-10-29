# Still image container

Demux and mux for a JPEG, PNG, WebP, or GIF file that is not inside a movie. The crate identifies the file, splits it into packets, and writes those packets back. It does not decode pixels. Huffman, inflate, and LZW are not implemented here.

`crates/engine` and `crates/export` are empty. Playback and export were not exercised. Tests call `Demux` and `Mux` directly. Timestamps stay in the file timescale and are not converted to microseconds. `crates/codecs/jpeg`, `crates/codecs/webp`, and `crates/codecs/gif` are empty scaffolds. `crates/codecs/png` accepts one PNG datastream in the packet bytes. This crate does not depend on that decoder. A still packet's bytes are that datastream, and extradata is empty.

## Scope

Normative layout:

- ITU-T T.81 for the JPEG markers. SOI starts the file, a segment length includes its own two bytes, and EOI ends the scan. The entropy-coded scan is not Huffman-decoded. `FF 00` is a stuffed byte.
- W3C PNG. A chunk is a 32-bit length, a type, that many data bytes, and a CRC-32 over the type and the data. The CRC polynomial is the ISO HDLC polynomial `0xEDB88320`. An `acTL` sequence is not split.
- Google's WebP RIFF container. A still file is one packet. An animated file has the VP8X animation flag, an `ANIM` chunk, and one `ANMF` chunk per frame. Duration is milliseconds. A stored 0 becomes 100 ms. Values 1 through 10 are kept.
- GIF87a and GIF89a. A graphic-control delay is a count of hundredths of a second. A missing graphic-control extension, and a stored delay of 0, both use 10 hundredths (100 ms). GIF89a says a delay of 0 does not name a wait. FFmpeg 8.1.1 substitutes 10. A stored delay above 0 is kept, and the packet bytes are not rewritten.

Read before implementation: FFmpeg 8.1.1 `libavformat/gifdec.c` and `libavcodec/gif_parser.c`; the WebP container specification for `VP8X`, `ANIM`, and `ANMF`; the PNG chunk layout; ITU-T T.81 marker structure. FFmpeg 8.1.1 has no animated-WebP demuxer (`image data not found` from ffprobe). Those trees were not copied into this repo.

Packet choice, recorded because the JPEG, WebP, and GIF codec crates are scaffolds:

- One JPEG, one PNG, or one still WebP (including a VP8X file whose animation flag is clear, and a lossless `VP8L` file) is one packet. The packet is the file bytes. There is no timescale and no duration. Trailing bytes after EOI or IEND stay in that packet.
- GIF is one packet per image, in file order. The first packet starts at byte 0, so it carries the header and the global color table. Each later packet starts where the previous image ended. The trailer stays on the last packet. Concatenating the packets is the file. Disposal is the graphic-control packed field inside the packet. Extradata is empty.
- Animated WebP is one packet per `ANMF`, in file order. The first packet starts at byte 0, so it carries `VP8X` and `ANIM`. Later packets start where the previous packet ended. The last packet runs through the end of the file. Disposal and blend stay in the `ANMF` header inside the packet. Extradata is empty.
- The Netscape loop count and the `ANIM` loop count are metadata. `None` means the file did not store one. `Some(0)` means repeat forever. Packets are one pass either way.
- GIF plain text is skipped. It is not a packet.
- Only the first GIF or animated-WebP frame is a keyframe. A later seek prerolls to that frame. A still picture is a keyframe at tick 0, and the packet timestamp stays missing.

## What is not supported

| Case | Result |
| --- | --- |
| APNG frame sequence | One still packet. Frames are not split. |
| HEIC, AVIF | Refused. Another family, or not a still signature. |
| Pixel decode, Huffman, inflate, LZW | Outside this crate. |
| Fragmented mux output | Not written. Finish is one progressive file. |
| Invented frame rate | Not reported. ffprobe's 1/25 clock on a still picture is not copied. |
| Infinite loop as packets | The loop count is metadata. The packet list is one pass. |
| GIF plain text | Skipped. |

## Capability matrix

| Capability | Required | Demux | Mux | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| JPEG | Yes | One packet, the file, no clock | Those bytes unchanged | Unit test; ffprobe size equals the file | Verified |
| PNG | Yes | One packet, the file, no clock | Those bytes unchanged | Unit test; ffprobe; local `04-motion-graphics.png` | Verified |
| Still WebP, lossy and lossless | Yes | One packet, the file, no clock | Those bytes unchanged | Unit test; ffprobe; webpinfo reports no animation | Verified |
| Animated WebP | Yes | One packet per `ANMF`, file order, millisecond delays | Concatenation, optional `ANIM` loop patch | Unit test; PIL file against webpinfo 1.6.0, durations 100 then 400 | Verified |
| GIF87a and GIF89a | Yes | One packet per image, timescale 100 | Concatenation, optional Netscape loop | Unit test; ImageMagick GIF against ffprobe, delays 5 and 20 | Verified |
| Missing or zero GIF delay | Yes | Duration 10, bytes unchanged | Unchanged when the loop is not set | Unit test locks the default | Verified |
| Loop count | Metadata | `Some(0)` does not add packets | `set_loop_count` writes Netscape or `ANIM` | Unit test | Verified |
| Probe | Yes | SOI, PNG signature, `GIF87a`/`GIF89a`, RIFF `WEBP` | Re-parsed on finish | Truncated prefixes need more bytes; movies are refused | Verified |
| Hostile size | Yes | Declared size checked before that many bytes are kept | Failed finish has no bytes and no durations | JPEG segment, PNG chunk, RIFF size, GIF sub-block, nested `ANMF`, 4097 GIF frames | Verified |
| Seek | Yes | Index built at open, `payload_scans` 0 | Outside demux | Still picture, GIF preroll, forward-only, cancel | Verified |
| Movie container | Refuse | At most 12 bytes read | Not wrapped | ftyp, EBML, AVI, WAVE; local mp4 and webm | Verified |
| Sample or pixel decode | No | Outside this crate | Outside this crate | Codec scaffolds, except PNG's separate decoder | Outside scope |
| Engine and export | Integration | Traits only | Traits only | Those crates are empty | Not run |

## Limits

| Limit | Value |
| --- | --- |
| File, and one packet | 32 MiB |
| GIF or animated WebP frames | 4096 |
| WebP chunk nesting | 2 (RIFF children, then `ANMF` children) |
| Probe window | 12 bytes |
| GIF timescale | 100 ticks per second |
| GIF default delay | 10 ticks |
| WebP animation timescale | 1000 ticks per second |
| WebP default delay | 100 ticks |

A declared file length above the cap is refused after the signature and before the body is read. A JPEG segment, PNG chunk, RIFF chunk, or GIF sub-block whose declared size exceeds the remaining bytes is `OpenError::HostileSize`. Nothing of that declared size is allocated. A failed open returns the error and no demuxer, so there is no partial packet list.

## Oracles

ffprobe 8.1.1 is the packet oracle for JPEG, PNG, still WebP, and GIF. ImageMagick  produced the files. Packet `pos` and `size` named the same slices this crate returns. GIF delays matched at time base 1/100, including an unequal pair of 5 and 20. ffprobe reports a still picture with a 1/25 duration of 1. This crate reports no timestamp and timescale 0. That invented clock is not a mismatch.

ffprobe 8.1.1 does not demux animated WebP (`image data not found`) and emits one packet for the whole file. webpinfo 1.6.0 is the animation oracle: frame order, `ANMF` offset, stored duration, and loop count. A PIL 11.3.0 file with durations 100 ms and 400 ms and loop 0 matched this crate's packets, and concatenating the packets was the file.

GStreamer 1.28.3 is installed. `pngdec` accepted the local PNG. `gst-discoverer-1.0` reports a GIF and a WebP as one image with duration 0 and frame rate 0/1, and it does not list frame packets. It is not the packet oracle. That is a limit of what it prints, not a pass.

Two libavformat tools would be one implementation. The GIF byte comparison is ffprobe. The animated WebP comparison is libwebp's `webpinfo`.

## Local files

`../local-files` is present on this machine. `04-motion-graphics.png` opened as one PNG packet whose bytes are the file, with empty extradata and no duration. `12129324_3840_2160_30fps.mp4` and `file_example_WEBM_1920_3_7MB.webm` were refused from the signature. Each read stopped by 12 bytes. The folder has no JPEG, GIF, or WebP sample. Those kinds are covered by the generated files above. The test skips when `../local-files` is missing.

## Performance

Apple M5 Pro, 18 cores, 48 GiB (`hw.memsize` 51539607552), arm64. Release test `open_scans_a_large_jpeg_and_seek_does_not_read_it_again`.

File: 8,388,635 bytes. SOI, a 1×1 SOF, SOS, 8 MiB of `0x00`, EOI. The read is a `Cursor` over that buffer.

| Measure | Result | Budget |
| --- | --- | --- |
| Open, including the marker scan | 2.98 ms, then 3.03 ms | 100 ms |
| Seek to tick 0 | 15.0 µs, then 2.6 µs | 5 ms |
| Bytes read for that seek | 0 | 0 |

`payload_scans` is 0. `consumed` stays at the file length. `/usr/bin/time -l` around `cargo test --release -p cutlass-still --offline --test external open_scans` reported maximum resident set size 53,411,840 bytes and peak memory footprint 28,951,296. That includes cargo, the test harness, the file, and the packet copy returned by the read. A counting allocator was not used. It would need `unsafe`, which this crate forbids. The demuxer keeps one copy of the file, at most 32 MiB, and an index of at most 4096 entries.

## Fuzz

In-process campaign, not libFuzzer. Target: `probe`, `StillDemux::open`, `read_packet`, `seek`, and `StillMux` with at most four packets. Corpus: random buffers up to 8 KiB, a truncated JPEG, a RIFF size of `u32::MAX`, and a bit flip in a JPEG or a GIF. Seed step `0x9E3779B97F4A7C15` from `0x00C0FFEE`.

A 3 second smoke exercised 2,232,259 seeds with no panic.

The recorded campaign:

```text
STILL_FUZZ_SECS=1800 cargo test --release -p cutlass-still --offline --test external -- --ignored --nocapture
```

## Builds

Commands from this checkout. Results are filled in after they run.
