# Vorbis test fixtures

Made on macOS from synthetic 16-bit WAV files, with libvorbis 1.3.7 through
GStreamer 1.28.3 `vorbisenc ! oggmux`, and with FFmpeg 8.1.1's own Vorbis
encoder (not libvorbis). The WAV generator writes, per sample `i` at rate
`r`, with `noise` 48000 values of Python `random.seed(7)` `uniform(-1, 1)`:

- `stereo`: 440 Hz at 0.4 on the left; 660 Hz at 0.4 plus `0.1 · noise` on the right.
- `clicks`: every 0.25 s, `0.9 · exp(-60 p) · noise` for the first 0.1 s of the period (`p` seconds into it), silence after; the right channel reads `noise` 977 samples later.
- `sweep`: `0.5 · sin(2π (200 + 1500 t) t)`.
- `surround`: channel `c` is `0.3 · sin(2π · 220 (c + 1) t) + 0.05 · noise[i (c + 1)]`.

All are free of third-party content.

| File | Source | Command | Exercises | SHA-256 |
| --- | --- | --- | --- | --- |
| `stereo_44100_q4.ogg` | stereo, 44.1 kHz, 0.5 s | `vorbisenc quality=0.4` | coupled stereo, residue 2, floor 1 | `f140a95d28da583bf24dbd02ba567402526c2d121e725d80c96319157ae91216` |
| `clicks_48000_q9.ogg` | clicks, 48 kHz stereo, 1 s | `vorbisenc quality=0.9` | short and long blocks and every transition, silent blocks | `ef55833c74d714679d584e4b7c56b160f23f6191395784cb82a02b55d24c0b90` |
| `sweep_8000_mono_q0.ogg` | sweep, 8 kHz mono, 0.5 s | `vorbisenc quality=0.0` | mono, low rate | `bc92c0ff7b253c378decf8cb7cee9ed1883c38f4824b37a22dca05a7f9f8efe7` |
| `surround51_48000_q4.ogg` | surround, 48 kHz 5.1, 0.25 s | `vorbisenc quality=0.4` | six channels, multiple coupling steps, the LFE submap | `db2a3ebfce449f87bd8be23a248b4df313205f2d94cf005675cc9e91e5a609ac` |
| `ffmpeg_stereo_44100.ogg` | stereo, 44.1 kHz, 0.5 s | `ffmpeg -c:a vorbis -strict -2 -q:a 4` | another encoder's setup and codebooks | `72cb20fcf1311cdc34b2766b3b15019444bc623cdacb65b5dff81cb94fbfaf00` |
| `chained_mono_stereo.ogg` | the sweep file then the stereo file | `cat` | a chained stream: new headers, rate, and channels mid-stream | `3fe2f7c226f1abdf69b2063983ad5b7fe9e3bde7716b1cd66a19d79f95ef54a7` |

The expected sample counts in `tests/decoder.rs` are the source lengths (the
end-of-stream granules), not decoder output. The FFmpeg encoder does not
trim its end, so its file ends at granule 22080. Reference PCM is made at
test time by FFmpeg's decoder and libvorbis when they are installed.
