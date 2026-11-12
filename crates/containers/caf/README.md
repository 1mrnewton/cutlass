# cutlass-caf

Reader and writer for Apple's Core Audio Format (`.caf`). Work in progress: this file is the scope record the implementation is checked against, and it is rewritten as a readiness report when the crate is done.

## Scope record

- Family: Core Audio Format, package `cutlass-caf`. Direction: demux and mux. File type: `.caf`, `mFileVersion` 1.
- Use: open CAF sound in the editor (Apple devices write ALAC and AAC in CAF; macOS system sounds are LPCM, IMA4, AAC and Opus in CAF) and write sound-only CAF from export.
- Normative sources: Apple, *Core Audio Format Specification* (developer.apple.com documentation archive, read 12 Nov 2026), and `CAFFile.h` from the macOS 27 SDK (AudioToolbox), which gives the packed big-endian structs. The `kuki` contents follow the codec: AAC is an MPEG-4 `ES_Descriptor` (ISO/IEC 14496-1), ALAC is Apple's `ALACMagicCookieDescription.txt`, FLAC is a `dfLa` box (FLAC in ISOBMFF).
- Implementations read: FFmpeg `libavformat/cafdec.c`, `cafenc.c`, `caf.c`, `mov_esds.c` and `isom.c` (descriptor lengths) in `../pub-packages/ffmpeg`. GStreamer has no CAF demuxer or muxer: `gst-plugins-base/gst/typefind` only recognises `caff`, and `pbutils/descriptions.c` names it. Apple's `afinfo` and `afconvert` (AudioToolbox, not libavformat) are the second, independent implementation, used only as test oracles and fixture generators.

## Planned matrix

| Capability | Demux | Mux |
| --- | --- | --- |
| `caff` header, version 1 | Accept; other versions `UnsupportedBrand` | Write |
| `desc` | Required first chunk, 32 bytes | Write |
| Linear PCM, int 8/16/24/32/64 and float 32/64, both endians | Constant packets, grouped up to 4096 bytes like libavformat | Write |
| a-law, µ-law | Same, codec `Pcm` coding 7 and 8 | Write |
| ALAC (`kuki` = ALACSpecificConfig, bare or `frma` form) | Packets from `pakt` | Write |
| AAC (`kuki` = `ES_Descriptor`) | AudioSpecificConfig as extradata | Write |
| Opus | Synthesised `OpusHead` (1 or 2 channels) | Write without `kuki`, as AudioToolbox accepts |
| FLAC (`kuki` = `dfLa`) | STREAMINFO as extradata | Write |
| MP3, AC-3, E-AC-3 | Packets from `pakt` | Write |
| Other format ids (`ima4`, `QDM2`, ...) | `StreamCodec::Unknown`, packets still framed | Refused |
| `pakt` sizes, frames, or both | Varint table, bounded | Written after `data` |
| Priming and remainder | `Trim` and an edit, not applied to timestamps | From `set_trim` and a short last packet |
| `data` size -1 | Data runs to the end of the file | Written while recording, patched at finish |
| `chan`, `info` | Parsed and exposed | Written for mono and stereo, `info` on request |
| Seek | Arithmetic for constant packets, `pakt` checkpoints for variable ones; no payload scan | — |

Limits: header 8 MiB, one packet 16 MiB, `kuki` 1 MiB, packet table bounded by `Limits::max_index_entries` and by the bytes of the `pakt` chunk, 64 channels, 4096 chunks.

Acceptance gates: packet bytes, sizes and times equal to libavformat's (after its `-priming` shift) on files from both afconvert and ffmpeg; afinfo and ffprobe read the files this crate writes with the same packet count, valid, priming and remainder frames; afconvert decodes muxed ALAC and PCM back to the source samples; no panic in the fuzz campaign; seek reads no payload.
