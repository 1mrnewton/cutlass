# WAV and RF64

Demux and mux for classic RIFF WAVE and RF64. One PCM, A-law, or μ-law packet is one sample frame. One MPEG packet is one frame. Other compressed audio is one `nBlockAlign` block, or the whole data chunk when the block align is 1 and the chunk fits in the packet limit. Timestamps are ticks of the sample rate. This crate does not decode samples.

`crates/engine` and `crates/export` do not register this family. Playback and export were not exercised. Tests call `Demux` and `Mux` directly.

## Sources

Normative layout:

- Multimedia Programming Interface and Data Specifications 1.0 (IBM/Microsoft, 1991), RIFF and WAVE.
- Microsoft WAVEFORMATEX and Multiple Channel Audio Data and WAVE Files (WAVEFORMATEXTENSIBLE).
- EBU Tech 3306, RF64. The first chunk of an RF64 file is `ds64`.

Read before implementation, from the local trees, not copied into this repo:

- FFmpeg `31ba826` at `../pub-packages/ffmpeg`: `libavformat/wavdec.c`, `wavenc.c`, `riff.h`, `riffdec.c`, `riffenc.c`, `riff.c`.
- GStreamer `4f441b9` at `../pub-packages/gstreamer`: `subprojects/gst-plugins-good/gst/wavparse/gstwavparse.c` and `gst/wavenc/gstwavenc.c`.
- libsndfile `wav.c`. It clips an oversized data chunk to the bytes that are present. This crate returns `OpenError::HostileSize` instead. FFmpeg warns and continues on some oversized sizes. The declared size is an error here when it passes the parent or the file.

A GUID is a media subtype only when its tail is `00 00 10 00 80 00 00 aa 00 38 9b 71` and the first field fits in a `u16`. Other bases, including ones FFmpeg accepts, stay `StreamCodec::Unknown` with a `wav-guid:` tag.

## Capability matrix

| Capability | Demux | Mux | Status |
| --- | --- | --- | --- |
| RIFF WAVE PCM integer | One frame per packet, timescale = sample rate | PCMWAVEFORMAT, no `fact` for tag `0x0001` | Verified |
| IEEE float | Tag `0x0003`, duration from the data chunk | `fact` written | Verified |
| WAVEFORMATEXTENSIBLE | Mask at extradata offset 8 | Written for a mask, more than two channels, a valid-bit width, or a source that was already extensible | Verified |
| RF64 and `ds64` | `0xFFFFFFFF` sizes come from `ds64` | `WavMux::rf64` writes `ds64` even for a small file | Verified |
| A-law and μ-law | Sample frames, codec `wav:0006` or `wav:0007` | Refused. The tag is an unknown codec | Verified |
| MPEG in WAV (`0x0055`, `0x0050`) | One frame, `0x0055` is `CodecId::Mp3` | One frame per packet | Verified |
| ADPCM and other block codecs | One block, duration from `fact`, `ds64`, or the block formula | Packet duration must match a known block | Verified for IMA ADPCM |
| Unknown codec tag | Stream kept, tag `wav:XXXX` | Refused | Verified |
| Unknown chunk ≤ 64 KiB | Kept | Written back before `data` | Verified |
| Unknown chunk above 64 KiB | Named on `DroppedChunk`, not allocated | Not rewritten | Verified |
| Odd chunk pad | Pad is not payload | Pad byte written outside the size field | Verified |
| `fact` | PCM uses the data size when `fact` disagrees. Compressed `fact` and `ds64` sample counts must match | Written when the stored tag is not `0x0001` | Verified |
| Hostile size, truncated header, nested LIST | Error, no stream list | — | Verified |
| RIFX, BW64 | Family is recognized, open returns `UnsupportedBrand` | Not written | Unsupported |
| AIFF, wavl, SMV, sample decode | Not this crate | Not this crate | Unsupported |
| Language | Not read | Accepted on the stream and not written | Unsupported |
| Engine and export | Traits only | Traits only | Not run |

A 96 kHz 16-bit stereo stream stays PCMWAVEFORMAT. Sample rate alone does not force WAVEFORMATEXTENSIBLE.

PCM duration is `data_len / block_align`. A `ds64` sample count that differs is `OpenError::Source` with the text `ds64 sample count disagrees with the data chunk`. A 32-bit data size that is not `0xFFFFFFFF` and differs from `ds64` is `ds64 data size disagrees with the data chunk`.

## Limits

| Limit | Value |
| --- | --- |
| Probe | 12 bytes |
| LIST and RIFF depth | 8 |
| Chunk headers in one file | 4096 |
| Unknown chunk kept | 64 KiB |
| Larger unknown chunk | Named and skipped |
| `ds64` table rows | 64 |
| MPEG resync | 64 KiB |
| Header bytes (`Limits::standard`) | 8 MiB |
| One packet | 16 MiB |
| Extradata | 1 MiB |
| Tracks | 1 |

A forward reader stops at `data`. Chunks after `data` are not seen, and seek returns `SeekError::ForwardOnly`. A non-data chunk larger than 64 KiB on a forward reader is `HostileSize` and is not drained.

PCM and other constant-block seek is `data_start + index * block_align`. `index_used` is true and `payload_scans` is 0. The data payload is not read during open or seek. MPEG seek walks frame headers.

## Oracles

ffprobe (libavformat) checks sample rate, channel count, codec name, and `duration_ts` on an ffmpeg PCM file, a float file, a file this muxer wrote, and an RF64 file this muxer wrote. Packet boundaries differ: ffprobe groups PCM samples, and this crate returns one frame per packet. The comparison is the sample count and the concatenated frames.

Python `wave` is the second reader. It is not libavformat. On the ffmpeg 44.1 kHz stereo PCM file and on the muxed file, channel count, rate, sample width, frame count, and frame bytes matched.

`afinfo` (AudioToolbox) accepted the same PCM file and reported 44100 Hz.

`gst-launch-1.0` with `wavparse` accepted that PCM file. `gst-inspect-1.0 wavparse --exists` returned 0.

The local file `GOOGLE IS BACK! (Gemini 3.8 Flash).wav` (32,325,228 bytes, 16,162,575 samples, 16 kHz mono) was opened from `../local-files`. Seek to the first, middle, and last frame did not read the payload. ffprobe's `duration_ts` matched. Python `wave` read the first frame and those bytes matched the first packet. The test skips when `../local-files` is missing.

## Fuzz

In-process campaign, not libFuzzer. `fuzz_seeds_do_not_panic` runs 64 fixed seeds on every `cargo test`.

`fuzz_campaign` ran in release for 1800 seconds (`WAV_FUZZ_SECS=1800`, `CARGO_TARGET_DIR=/tmp/cutlass-riff-fuzz`). It exercised 1,061,805,649 seeds. The generator starts at `0x00c0ffee` and adds `0x9E3779B97F4A7C15` before each seed. The run printed `origin 0x875500da6be97893`, which is that accumulator after the last seed. The corpus is random bytes up to 4 KiB, a truncated valid PCM file, one flipped byte, a chunk size of `u32::MAX`, a nested LIST, and a small PCM mux. No panic. No finding was minimized because there was none.

## Measurement

Release build. Apple M5 Pro, `hw.memsize` 51,539,607,552. One run each. Seek did not increase `consumed` and `payload_scans` was 0.

| File | Bytes | Samples | Open | Seek | Header bytes read |
| --- | --- | --- | --- | --- | --- |
| Generated 44.1 kHz stereo | 32,044 | 8,000 | 12.666 µs | 8.25 µs | 44 |
| Local 16 kHz mono | 32,325,228 | 16,162,575 | 3.709 µs | 833 ns | 78 |

Wall-clock asserts (open under 50 ms, seek under 20 ms) run only when debug assertions are off.

## Checks

`cargo fmt --all -- --check`, `cargo lint`, and `cargo test --workspace` passed on this branch. `cargo deny check` reported advisories, bans, licenses, and sources ok, with existing unmatched-license and duplicate-crate warnings that this change did not add. Engine and export registration was not run.
