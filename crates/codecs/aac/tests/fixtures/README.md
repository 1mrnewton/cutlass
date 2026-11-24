# AAC decoder fixtures

AAC-LC streams written by FFmpeg's native `aac` encoder, with FFmpeg's own
decode of each one. `src/oracle.rs` decodes the `.aac` files and compares
the result with the `.f32` files, so the tests need neither FFmpeg nor any
outside media.

Made on 24 November 2026 with FFmpeg 6.1.1 (`6.1.1-3ubuntu5`, Ubuntu 24.04,
x86-64) by `python3 make_fixtures.py` run in this folder. The source is
synthetic and free of third-party content: half a second of stereo, with
plucked notes at different pan positions, a kick, snare and hi-hat pattern,
and a quiet independent hiss in each channel (numpy `default_rng`, seeds 1,
2 and 3).

Each `.aac` is ADTS without a CRC. Each `.f32` is
`ffmpeg -c:a aac -i <file>.aac -f f32le`: interleaved stereo, little-endian
32-bit float, one 1024-sample frame per access unit with nothing trimmed.

| File | Encoder options | Access units | TNS filters (long windows) | Noise bands | Intensity bands | Mid/side bands | Short-window channels | SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tools_44100_128k_nopns.aac` | `-b:a 128k -aac_pns 0`, 44.1 kHz | 23 | 8 (4) | 0 | 307 | 540 | 11 | `ee2203ddd662393bac1d0f01e72f98a3d4f896eb1458c0dcffa1827e1e263dd2` |
| `tools_48000_64k_nopns.aac` | `-b:a 64k -aac_pns 0`, 48 kHz | 25 | 23 (6) | 0 | 287 | 417 | 20 | `0bffab596af97e9e7bb58ad660e2e26e3d3cecdf0c0182337f43e27d08973092` |
| `tools_44100_96k_pns.aac` | `-b:a 96k`, 44.1 kHz | 23 | 5 (4) | 458 | 94 | 329 | 12 | `b5e6028a3c09a1488abdc4a645e7a2165059c60ecd30b6a60fdda3c1705baf85` |

| Reference | SHA-256 |
| --- | --- |
| `tools_44100_128k_nopns.f32` | `2efc69840d821e41af430ae0471a8acd1a8ded926e7b9160a679e08bd71ef1b1` |
| `tools_48000_64k_nopns.f32` | `9a9188b0329b2f07840dc97cd5ec1839452f0034c0f0c8941cdb17b752033365` |
| `tools_44100_96k_pns.f32` | `5b773945cd04a63a1b7520595d67d300045806cb0e03a4dcf89f5e110f736d53` |

The counts are what the decoder's own parser finds in each stream; the
oracle tests assert that the tools are present, so a rebuilt fixture that
lost one would fail. Intensity stereo, mid/side and TNS are on by the
encoder's defaults.

FFmpeg has no correlated noise substitution (a noise band with the mid/side
flag set in both channels), and these streams do not use it. That case is
tested on hand-built access units in `src/decode.rs`.
