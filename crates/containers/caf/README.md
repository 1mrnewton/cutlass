# cutlass-caf

Reader and writer for Apple's Core Audio Format (`.caf`), version 1. Safe Rust only (`#![forbid(unsafe_code)]`), no dependencies beyond `cutlass-codec` and `cutlass-container`, and no foreign library, process or OS parser at run time. afconvert, afinfo and ffmpeg are used only by tests and the bench.

## Readiness

Ready for the editor to open CAF sound and for export to write it. Every CAF file a Mac ships with opens and reads (2891 files), files from afconvert and ffmpeg give the same packets as libavformat and the same counts as afinfo, and files this crate writes are read by afinfo and ffprobe and decoded by afconvert and ffmpeg to the same samples as the originals. Open, read and seek take fewer retired instructions than libavformat on every file measured. What is missing is listed under [Gaps](#gaps).

## Support matrix

`FEATURES` in `src/features.rs` holds the same rows for code.

| Row | Demux | Mux | Evidence |
| --- | --- | --- | --- |
| `caff` header, version 1 | Yes | Yes | all tests |
| Other versions | `UnsupportedBrand` | — | `tests/hostile.rs` |
| Linear PCM, integer 8/16/24/32/64 bit, float 32/64, either endian | Yes, codec `Pcm` codings 1–6 and 9 | Yes | afconvert and ffmpeg files match libavformat; PCM written here decodes to the source in afconvert and ffmpeg |
| a-law, µ-law | Yes, codings 7 and 8 | Yes | as above |
| AAC (`kuki` = `ES_Descriptor`) | Yes, AudioSpecificConfig as extradata | Yes | as above; export AAC reads in afinfo, decodes in afconvert |
| ALAC (`kuki` = ALACSpecificConfig, bare or `frma`/`alac` atoms) | Yes | Yes | remux of afconvert and ffmpeg ALAC decodes to the source samples; 20 minutes decode bit-exact through `media-bench` |
| FLAC (`kuki` = `dfLa`, or none as ffmpeg writes it) | Yes, STREAMINFO as extradata | Yes, needs the STREAMINFO | remux of afconvert FLAC decodes to the source |
| Opus | Yes, an `OpusHead` is made (1 or 2 channels) with the pre-skip at 48 kHz | Yes, no `kuki`, as AudioToolbox accepts | desktop test plays from the start what the decoder gives after the pre-skip |
| MP3 (`.mp3`), AC-3 (`ac-3`) | Yes | Yes | ffmpeg files match libavformat; export files read in afinfo, decode in afconvert |
| E-AC-3 (`ec-3`) | Yes | Yes | files written here are read by afinfo and decoded by afconvert; libavformat has no `ec-3` CAF tag and no third-party file was available |
| Other format ids (`ima4`, `QDM2`, ...) | Packets framed, `StreamCodec::Unknown` | Refused (`UnknownCodec`) | afconvert IMA4 matches afinfo |
| `pakt`: sizes, frames, or both | Yes, bounded | Written after `data` | all oracle tests |
| Priming and remainder | `Trim` plus one edit; timestamps are not shifted | From `set_trim` and a short last packet | afinfo counts equal on every remux |
| `data` of size -1 | Runs to the end of the file; a forward input reads whole frames until the source ends | Written while recording, patched at finish | 501 system files |
| `chan` | Parsed and exposed | Mono or stereo by default, any layout through `set_channel_layout` | tests |
| `info` | Parsed and exposed | Through `set_info` | tests |
| Other chunks (`mark`, `regn`, `strg`, `uuid`, `ovvw`, `midi`, `free`) | Listed in `chunks()`, otherwise skipped | Not written | system files |
| Seek | Arithmetic for constant packets, checkpoints every 64 packets in the table for variable ones; never reads payload | — | `consumed()` unchanged after every seek in tests, real files and the fuzz campaign |

## Behaviour

- Times are in frames at the file's sample rate, starting at zero, priming included. The priming and remainder are reported as `Trim`, and an edit starts at the priming and lasts the valid frames, as the Ogg crate does for Opus pre-skip. libavformat instead starts at minus the priming and shortens the last packet; the oracle tests account for that.
- PCM with one frame a packet is returned in runs of up to 4096 bytes, as libavformat groups it, so packet sizes and times match ffprobe's exactly.
- A broken chunk after `data` (seen in 51 macOS system sounds: a header one byte off claiming an absurd size, or a few stray bytes) ends the walk and is reported by `stopped_at()`; the file still opens. Essential chunks (`desc`, `kuki`, `pakt`, `chan`) that run past the file are refused.
- ffmpeg ends a FLAC packet table with an entry of no bytes and no frames. It is counted, as AudioToolbox counts it, and skipped when reading, as libavformat stops at it.
- `packet_count()` is the number of CAF packets (frames, for PCM), which is what afinfo prints.
- The muxer writes the header at the first packet with `data` of size -1, streams the payload, and at finish writes the `pakt` after `data`, fixes the data size, sets `mFramesPerPacket` (all packets but a shorter last one equal: that count; otherwise 0 with size and frame pairs), and rewrites a same-length `kuki` (the AAC buffer size and bit rates, the final FLAC STREAMINFO). A shorter last packet becomes remainder; the remainder is the larger of that and the trailing padding from `set_trim`. PCM with no extradata is 16-bit little-endian, as for WAV.
- AAC cookies are written with four-byte descriptor lengths, as AudioToolbox writes them; libavformat's reader skips four bytes as if a full box header were there and only reads such cookies correctly.

## Integration

- Desktop readers (`apps/desktop/src/readers.rs`), platform readers (`crates/ffi/src/codecs.rs`) and `tools/media-bench` open CAF next to AIFF.
- `crates/probe` reports a CAF file as audio lasting all its frames, priming and remainder included, as it counts AAC priming in an MP4.
- `crates/export` writes sound alone to CAF for PCM, AAC, ALAC, FLAC, Opus, MP3, AC-3 and E-AC-3 when the request names `ContainerId::Caf`. The desktop export panel does not offer CAF: it picks the container from the codec, and adding a choice is a product decision left open.

## Limits

`Limits::standard()` applies: header reads 8 MiB, one packet 16 MiB, `kuki` 1 MiB, packet table entries 16 million. Also 64 channels, 4096 chunks, 2^20 frames a packet, 64 bits a channel, `info` 64 KiB, varints of at most 5 bytes that fit 32 bits. A table's declared count is checked against the limit and against the bytes in the chunk before anything is reserved, and the table is decoded in 64 KiB blocks. It is held as 4 bytes a packet for each varying field plus 16 bytes a 64 packets: measured with `/usr/bin/time -l`, an open of a table of 16 million size and frame pairs peaks at 135 MB (2.6 MB for a one-packet file), and the table's sum of sizes must fit in the data.

## Tests

All counts are from 12 November 2026 on an Apple M5 Pro running macOS 27.0.1, rustc 1.99.1, FFmpeg 8.1.1.

- `cargo test -p cutlass-caf`: 71 unit and behaviour tests on files built byte by byte (`src/tests`): PCM runs, size -1 data on random and forward input, every table shape, seek landings with no payload read, mux round trips for every PCM coding, AAC, ALAC, FLAC and Opus, and hostile input — lying counts and sizes, long and large varints, broken descriptions and cookies, duplicate chunks, every prefix and many single-byte changes of good files, failing readers and writers.
- `tests/external/oracle.rs` (skips with a message when a tool is missing, as on Linux CI):
  - afconvert writes 13 formats (`aac`, `alac`, `opus`, `flac`, `ulaw`, `alaw`, `LEI16`, `BEI24`, `LEF32`, `BEF64`, `I8`, `BEI32`, `ima4`); this crate's packets, times and payload equal libavformat's, and its counts equal afinfo's packets, bytes, valid, priming and remainder. libavformat refuses the IMA4 file (its `pakt` says 0 packets and comes before `data`, which `cafdec.c` cannot handle), so that one is checked against afinfo only.
  - ffmpeg writes 12 (`ac3`, `libmp3lame`, `alac`, `flac`, `pcm_s16be`, `pcm_s24le`, `pcm_s32be`, `pcm_f32le`, `pcm_f64be`, `pcm_alaw`, `pcm_mulaw`, `pcm_s8`); the same comparisons hold, afinfo included.
  - 12 files remuxed through this crate (8 from afconvert, 4 from ffmpeg) give afinfo and ffprobe the same counts and packets as the originals, and afconvert and ffmpeg decode them to the same samples; lossless ones decode to the source. ffmpeg's FLAC without a `kuki` is refused by the muxer, as intended. AudioToolbox's AAC decode of one file differs from run to run on this noisy source, so for AAC only the length is compared there.
  - PCM written here in five layouts decodes to the source in afconvert and ffmpeg. E-AC-3 written here is read by afinfo and decoded by afconvert.
- `tests/external/local.rs`: every CAF under `/System/Library` and `/Library/Audio` on this Mac — 2891 files (1749 AAC, 517 Opus at 24 and 48 kHz, 621 packed LPCM of 16, 24 and float 32 bits, 4 IMA4), 1,105,611 packets, 633 MB of payload, 51 with a broken chunk after data, 501 with data of size -1 — opens, reads with times that add up, and seeks to its first, middle and last packet without reading payload; every 25th (116) also matches afinfo and libavformat. Each also reads forward, as a file still being copied in is read, with the same packets. Two MPEG-4 files named `.caf` are refused as not CAF. The default run covers 2327 of them; `CUTLASS_CAF_ROOTS=/System/Library:/Library/Audio` covers all.
- `tests/external/fuzz.rs`: an in-process campaign (not libFuzzer) builds files from chunks with honest and lying sizes and mutates muxed files and three system sounds; on whatever opens, packets stay within the limit, times never go back, and a seek reads nothing and lands at or before its target. Random muxes that finish must read back byte for byte with valid, priming and remainder adding up. A 2,000-seed run is part of `cargo test`. Two release campaigns ran on 12 November 2026 with no panic and no failed check: 1800 s from origin `0xcaffee` (built at `9deefabe`) ran 64,329,069 cases — 9,088,771 files opened, 213,858,321 packets read, 36,084,664 seeks landed, 16,773,896 muxes finished and read back — in at most 23 MB resident; 900 s from origin `0xbeef` (built at `8547df8f`, with the 32 KiB buffer) ran 33,402,925 cases (4,722,586 opened, 111,120,652 packets, 18,749,316 seeks, 8,716,827 muxes) in at most 18 MB. About 14% of cases open; the rest are refused at the header, which is also checked for not panicking.
- `tests/external/measure.rs`: in ten hours of 48 kHz AAC (1,687,500 packets) open takes about 5 ms and a thousand seeks with a packet read about 1.4 ms in a release build, visiting at most 63 table entries a seek.
- Outside this crate: two desktop reader tests (a PCM CAF decodes, an Opus CAF plays after the priming), a probe test, and an export test that writes all eight codecs to CAF.

## Speed against libavformat

`cargo bench -p cutlass-caf`, same machine, commit `fc8f889c`. 20 minutes of 48 kHz stereo (a tone and noise from ffmpeg), written by afconvert. Each side does the job in one process, `repeat` times; the figure is the median of five runs less the median of five runs that only start up, divided by `repeat`, so neither side pays for process start or library loading. Instructions are those `/usr/bin/time -l` reports as retired; wall time on this machine is too noisy to compare.

| File | Job | Repeat | Cutlass instructions a job | libavformat instructions a job | Ratio | Cutlass peak RSS | libavformat peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAC (18.9 MB, 56,253 packets) | open | 2000 | 5,166,544 | 12,456,919 | 0.41 | 3.4 MB | 13.7 MB |
| | read | 20 | 62,483,970 | 301,544,345 | 0.21 | 3.4 MB | 13.9 MB |
| | seek | 20000 | 15,120 | 20,794 | 0.73 | 2.6 MB | 10.7 MB |
| ALAC (110 MB) | open | 2000 | 1,444,248 | 3,377,061 | 0.43 | 2.8 MB | 10.6 MB |
| | read | 20 | 85,997,586 | 111,944,475 | 0.77 | 2.8 MB | 10.6 MB |
| | seek | 20000 | 17,271 | 18,999 | 0.91 | 2.4 MB | 9.5 MB |
| Opus (11 MB) | open | 2000 | 5,496,097 | 13,258,080 | 0.41 | 3.5 MB | 13.9 MB |
| | read | 20 | 56,125,761 | 197,959,443 | 0.28 | 3.2 MB | 12.5 MB |
| | seek | 20000 | 15,089 | 17,298 | 0.87 | 2.6 MB | 10.7 MB |
| PCM s16 (230 MB) | open | 2000 | 201,153 | 243,988 | 0.82 | 2.4 MB | 8.9 MB |
| | read | 20 | 191,620,570 | 309,833,172 | 0.62 | 2.4 MB | 8.9 MB |
| | seek | 20000 | 15,568 | 17,953 | 0.87 | 2.3 MB | 8.9 MB |

The jobs: open reads the header and the packet table and no packet (`avformat_open_input` with the CAF demuxer named, so no probe runs, and no `avformat_find_stream_info`); read is an open and every packet, dropped; seek is one open and then `repeat` seeks to the same pseudo-random times, each followed by one packet read. Before timing, both sides must print the same packet count, bytes and a checksum of first bytes or landing times. Differences that remain: libavformat runs its AAC and Opus parsers over every packet it reads (`AVSTREAM_PARSE_HEADERS` in `cafdec.c`), work this crate does not do, which accounts for much of the AAC and Opus read gap; ALAC and PCM have no parser, so their rows compare demuxing alone. libavformat keeps a 24-byte index entry a packet; this crate keeps 4 bytes and a checkpoint a 64. A second full run gave every figure within 0.5% of these. Writing was not timed.

## Gaps

- The Rust PCM decoder does not decode 8-bit (signed in CAF, as afinfo and libavformat report), a-law, µ-law or 64-bit integer PCM (codings 6–9), so such CAF files open but do not play. That is a decoder gap, not this crate's.
- LPCM whose samples are not packed (24 bits in 4 bytes, 20-bit and the like) or with more than one frame a packet opens as `Unknown("lpcm")`: framed, not handed to the PCM decoder. None of the 2891 system files is like that.
- The muxer refuses formats it does not know (IMA4 and the like), FLAC without a STREAMINFO (AudioToolbox cannot decode ffmpeg's cookie-less FLAC CAF), and Opus with more than two channels. Opus with more than two channels also opens as `Unknown("opus")`.
- `mark`, `regn`, `strg`, `uuid`, `ovvw`, `midi` and other chunks are not exposed beyond their place in `chunks()`, and are not written.
- E-AC-3 has been checked only on files this crate wrote.
- AudioToolbox drops 288 frames of decoder delay from an E-AC-3 CAF that names no priming. Files from export name the encoder's priming, and afconvert decodes each of the eight export codecs in CAF to exactly half a second (22,050 frames, 24,000 for Opus).

## Sources

- Apple, *Core Audio Format Specification* (developer.apple.com documentation archive, read 12 November 2026), and `CAFFile.h` from the macOS 27 SDK (AudioToolbox) for the packed big-endian structures. `kuki` follows the codec: AAC is an MPEG-4 `ES_Descriptor` (ISO/IEC 14496-1), ALAC is Apple's `ALACMagicCookieDescription.txt`, FLAC is a `dfLa` box (FLAC in ISOBMFF).
- Read for behaviour, not copied: FFmpeg `libavformat/cafdec.c`, `cafenc.c`, `caf.c`, `mov_esds.c`, `isom.c`. GStreamer has no CAF demuxer or muxer: `gst-plugins-base/gst/typefind` only recognises `caff`.
- afinfo and afconvert (AudioToolbox) are the second, independent implementation, used as oracles only.

## Running

- `cargo test -p cutlass-caf` — everything above but the long runs.
- `CAF_FUZZ_SECS=1800 cargo test --release -p cutlass-caf --test external fuzz_campaign -- --ignored --nocapture`
- `CAF_MEASURE_PACKETS=16000000 /usr/bin/time -l cargo test --release -p cutlass-caf --test external long_table_open -- --ignored --nocapture` (run the test binary itself under `time` for its own memory).
- `cargo bench -p cutlass-caf` (`CAF_BENCH_WORK=<folder>` keeps the generated files).
- `cargo run --release -p cutlass-caf --example demux -- <file.caf> [--packets]` and `--example remux -- <in.caf> <out.caf>`.
