# Vorbis support record

Branch `vorbis`, worktree `cutlass-vorbis`, started from `e025f6ce`. Native safe Rust in `cutlass-vorbis`: `#![forbid(unsafe_code)]`, workspace lints, no libvorbis, FFmpeg, or other C. libvorbis 1.3.7 (through GStreamer 1.28.3 `vorbisenc` and `vorbisdec`) and FFmpeg 8.1.1's own Vorbis decoder and encoder are test oracles only.

This file is the scope record. The evidence sections are filled in as each gate runs; a gate that has not run says so.

## Specifications and sources

- Vorbis I specification (Xiph.Org, revision of 2020-07-04, with the 2015-02-26 single-entry codebook errata): bit packing, codebooks, floors 0 and 1, residues 0, 1, and 2, mappings, modes, windows, the audio packet, and Appendix A (Ogg encapsulation: negative start, end trim).
- RFC 5334 (Ogg media types) and the Ogg Vorbis mapping in Appendix A for the container side.
- Read for behavior the prose leaves open (not copied, not linked): FFmpeg `libavcodec/vorbisdec.c`, `vorbis.c`, `vorbis_parser.c`, `vorbisdsp.c`, `vorbisenc.c`; `libavformat/oggparsevorbis.c`; GStreamer `ext/ogg/gstoggstream.c` and `vorbis_parse.c`.

## Scope

Decode and encode, both. "Both" means each direction is checked against an independent implementation, with the self round trip only as extra coverage.

Channels: the decoder takes 1 to 8 channels, the `ChannelLayout` limit, in the Vorbis channel order. More than 8 is refused as `Unsupported`. The engine plays mono and stereo and the export path writes mono and stereo, so the encoder is required for 1 and 2 channels. It also takes 3 to 8 so a multichannel round trip can be tested. The only real Vorbis file in `local-files` is stereo (the `.webm`), and the recordings with more channels use other codecs.

### Capability matrix

| Area | Decoder accepts | Encoder emits |
| --- | --- | --- |
| Identification header | version 0, 1 to 8 channels, any nonzero rate, blocksizes 64 to 8192 with short <= long | as decode; 8 to 192 kHz input; blocksizes 256/2048 (rate <= 50 kHz) or 512/4096 (above) |
| Comment header | parsed, vendor and fields bounded, ignored for decode | vendor string and no fields |
| Codebooks | ordered and sparse lengths, lookup types 0, 1, 2, single-entry books (errata), full and partial trees refused as the spec says | ordered or sparse lengths, lookup type 1 |
| Floor 0 | LSP, any order, bark map, amplitude/offset | not emitted |
| Floor 1 | all partitions, classes, subclasses, multipliers 1 to 4, up to 65 posts | multiplier 2, a fixed post layout per blocksize |
| Residue 0, 1, 2 | all, with the 8-pass cascade and do-not-decode flags | 1 for mono and uncoupled, 2 for coupled channels |
| Mapping | submaps 1 to 16, coupling steps, mux | 1 submap, square-polar coupling of each channel pair |
| Modes | 1 to 64 modes, short and long | 2 modes (short, long) |
| Windows | short/long transitions, hybrid windows from the packet's window flags | short/long switching from a transient detector |
| In-band headers | a header packet resets and takes the new setup (chained link) | n/a |

### Timing contract

- Packet `n` decodes to `previous_blocksize / 4 + current_blocksize / 4` samples; the first packet after open or `reset` decodes to none.
- The container gives each packet that duration (the first packet's is 0). Pts is the stream position of the packet's first returned sample. A packet stamped shorter than it decodes is an end trim (`skip_trailing`). A negative start (granule of the first page below its samples) is the stream's first edit, applied by the engine, as Opus pre-skip is.
- The encoder's first packet has pts 0 and duration 0. Packet `k` returns the samples between the centers of blocks `k - 1` and `k`, so output starts at input sample 0 with no delay to trim. The last packet's duration is shortened to the input length and it carries `SideData::Gapless` with the trailing padding.

### Limits

Every size comes from the stream and is checked before allocation: codebook entries at most 2^24 (spec), dimensions 1 to 65535, total VQ values bounded by the header's bit length, at most 256 codebooks, 64 floors, residues, mappings, and modes. Residue partitions are bounded by `blocksize / 2`. Configuration (Xiph-laced) at most 16 MiB. A packet never makes the decoder allocate more than its configuration's fixed buffers.

### Not in scope

- More than 8 channels (the `ChannelLayout` limit).
- Encoder floor 0, residue 0, lookup type 2, more than two modes. The decoder takes them; oracle streams for them come from this crate's test-only setup writer, cross-decoded by libvorbis and FFmpeg.
- Vorbis in Matroska/WebM (the Matroska crate is a scaffold). The `.webm` stream is copied into Ogg by ffmpeg for testing.

## Acceptance gates

Fixed before implementation; the results go below.

Decoder:

1. libvorbis files (GStreamer `vorbisenc`) at qualities -0.1, 0.1, 0.4, 0.6, 1.0 and managed bitrates, 8 to 96 kHz, mono, stereo, 5.1, 7.1: the same samples as libvorbis (`vorbisdec`), in count exactly and to at least 90 dB SNR per channel, after the container's trim.
2. FFmpeg native encoder files: the same samples as FFmpeg's native decoder, exact count, at least 90 dB SNR.
3. The `.webm` stream from `local-files`, copied to Ogg: agrees with both oracles as above.
4. Floor 0, residue 0 and 1, lookup type 2, and many-mode streams this crate writes: libvorbis and FFmpeg (lookup type 2: libvorbis only, FFmpeg refuses it) decode them to this decoder's samples at 90 dB.
5. Malformed and truncated input fails with typed errors and leaves the decoder usable. Bounded fuzz campaigns run with no panic, recorded below.

Encoder:

1. Every encode decodes in libvorbis and FFmpeg's native decoder to exactly the input's sample count, through `OggMux`, and both agree with this crate's decoder at 90 dB.
2. Quality mode: the rate rises with quality, and silence costs at most 2 kb/s.
3. With a bitrate: the mean over a non-sparse item is within 10% of the request.
4. At libvorbis's rate for quality 0.4 and 0.6 (stereo 44.1 kHz), this encoder's mean per-band log-spectral distance (critical bands, decoded by libvorbis) is at most 1.5 dB above libvorbis's at the same rate (within 10%).

Integration: an Ogg Vorbis file opens, seeks, and plays in the desktop reader, probes, exports sound alone to `.ogg`, and the export reopens with the same samples.

Performance (release, this machine): decode at least 100x real time for 44.1 kHz stereo; encode at least 20x; peak memory recorded.

## Evidence

Run on 8 November 2026. `CARGO_TARGET_DIR=/tmp/cutlass-vorbis-target cargo test -p cutlass-vorbis`. GStreamer 1.28.3 `vorbisenc`/`vorbisdec` and FFmpeg 8.1.1 were present, so the oracles ran. FFmpeg's native decoder is invoked with `-cpuflags 0`: its aarch64 NEON inverse coupling treats a +0 magnitude as positive, and the spec's `magnitude > 0` does not.

Two reference limits, not decoder bugs. GStreamer `vorbisdec` emits one sample past the end granule. FFmpeg's native decoder skips the end trim when the first audio page is also the last, so a short file comes back a block long (`sweep_8000_mono_q0.ogg` 4096 against a granule of 4000; `ffmpeg_stereo_44100.ogg` 22528 against 22080). SNR is over the common prefix. Our count is the granule.

### Decoder

libvorbis files (10 s, `local-files` audio), our count against the granule, SNR against FFmpeg and against libvorbis:

| File | Samples | FFmpeg | libvorbis | SNR, dB |
| --- | --- | --- | --- | --- |
| q −0.1 stereo 44.1 kHz | 441000 | 441000 | 441001 | 137.3 / 135.5 |
| q 0.4 stereo 44.1 kHz | 441000 | 441000 | 441001 | 137.8 / 135.7 |
| q 1.0 stereo 48 kHz | 480000 | 480000 | 480000 | 138.4 / 136.0 |
| q 0.6 mono 22.05 kHz | 220500 | 220500 | 220501 | 138.1 / 135.8 |
| q 0.1 mono 8 kHz | 80000 | 80000 | 80000 | 138.9 / 136.3 |
| q 0.5 stereo 96 kHz | 960000 | 960000 | 960000 | 138.2 / 136.1 |
| managed 64 kb/s stereo 44.1 kHz | 441000 | 441000 | 441001 | 137.7 / 135.7 |
| q 0.4 5.1 48 kHz | 480000 | 480000 | 480000 | 137.8 / 135.7 |
| q 0.4 7.1 48 kHz | 480000 | 480000 | 480000 | 137.8 / 135.7 |

FFmpeg's native encoder, stereo only, same check: q1 and q4 at 44.1 kHz (264640 samples, 137.4/135.8 and 137.3/135.4 dB), q8 at 48 kHz (288000, libvorbis 288001, 137.3/135.3), q3 at 22.05 kHz (132352, 137.2/135.3), q6 at 32 kHz (192000, 137.8/135.5).

Checked-in fixtures agree the same way, 135.2 to 139.2 dB, aside from the two short files above. The local `.webm` track, copied to Ogg, is 1,465,920 samples of digital silence at 48 kHz and matches both decoders exactly.

Gates 1 to 3 pass. The malformed-packet and lifecycle tests in `tests/decoder.rs` pass. A recorded fuzz campaign has not run.

Gate 4, streams this crate writes (`tests/synthetic.rs`), mono 44.1 kHz, short block 256 and long block 2048. The same sample count as this decoder, SNR against FFmpeg and libvorbis:

| Stream | Samples | SNR, dB |
| --- | --- | --- |
| Residue 0, floor 1, lookup type 1 | 32768 | 142.4 / 137.6 |
| Residue 1, floor 1, lookup type 1 | 32768 | 143.6 / 136.5 |
| Floor 0, residue 1 | 32768 | 144.3 / 134.1 |
| Lookup type 2 | 32768 | libvorbis 136.0; FFmpeg refuses the book |
| Four modes, short and long blocks | 126336 | 139.0 / 135.5 |

### Encoder

Every item below is muxed with `OggMux` and decoded by this crate, FFmpeg's native decoder, and libvorbis. Our count and FFmpeg's are the input length. libvorbis is that length or one sample over. Agreement is 135.1 to 138.8 dB, and infinite on silence.

| Check | Result |
| --- | --- |
| Silence, stereo 44.1 kHz, 2 s, quality 0.5 | 0.35 kb/s of audio packets |
| Quality, stereo 44.1 kHz, 1 s | q 0.2 at 165.6 kb/s, q 0.8 at 261.3 kb/s |
| 64 kb/s, dense stereo 44.1 kHz, 4 s | 69.5 kb/s, 1.086 of the request |
| 128 kb/s, same item | 116.6 kb/s, 0.911 of the request |
| Matched to libvorbis q 0.4 (105.3 kb/s, LSD 0.85 dB) | 110.6 kb/s, LSD 2.25 dB |
| Matched to libvorbis q 0.6 (159.9 kb/s, LSD 0.39 dB) | 145.8 kb/s, LSD 1.85 dB |

The log-spectral distance is the RMS over 1-Bark bands of a 2048-point Hann, hop 1024, frames above −70 dBFS, decoded by libvorbis. The limit is 1.5 dB above libvorbis, so 2.35 and 1.89. Gates 1 to 4 pass.

Interop items: mono 44.1 kHz q 0.4, stereo 44.1 kHz q 0.3, stereo 48 kHz q 0.7, 5.1 48 kHz q 0.5, mono 8 kHz q 0.2, stereo 96 kHz q 0.4.

### Not run

- Vorbis in the desktop reader: open, seek, play, probe, export to `.ogg`, reopen.
- A Vorbis fuzz campaign, and release measurements of decode and encode throughput, latency, and memory.
