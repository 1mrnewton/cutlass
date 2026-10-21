# Fixtures

Small media files for codec tests. Each one is under 1 MB. Tests read them
through `cutlass-test-fixtures` and never call an external program.

## Video

`scripts/make-video-fixtures.py` regenerates `fixtures/video/`. It needs
ffmpeg built with libx264 and libx265, and the output is deterministic. The
frames are FFmpeg's `testsrc2` pattern, so there is nothing here we cannot
redistribute.

Each directory holds three files:

- `config.bin`, the codec configuration record (`avcC` or `hvcC`).
- `packets.bin`, the compressed frames in decode order, concatenated.
- `stream.txt`, the manifest: codec, size, timestamps, and one CRC-32 per
  decoded frame.

The checksum is the CRC-32 (zlib polynomial) of the decoded picture in NV12,
or P010 for 10-bit, with row padding left out. FFmpeg's decoder produces it.
H.264 and HEVC decoding is bit-exact, so another correct decoder matches.

Generated with FFmpeg 8.1.1, x264 r3222, and x265 4.2.

| Fixture | What it covers |
| --- | --- |
| `h264-high-bframes-128x72` | High profile, B-frames, limited range |
| `h264-high-full-range-128x72` | Full range stays full range |
| `h264-high-cropped-100x58` | Size that is not a multiple of 16 |
| `hevc-main-bframes-128x72` | Main profile, B-frames |
| `hevc-main10-bframes-128x72` | Main 10, decoded as P010 |
