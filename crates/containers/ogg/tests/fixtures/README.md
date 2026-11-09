# Ogg Vorbis fixtures

Copies of the `cutlass-vorbis` fixtures, so this crate's tests do not read
another crate's folder. `crates/codecs/vorbis/tests/fixtures/README.md`
gives the synthetic signals and the exact commands. All are free of
third-party content.

| File | Producer | Layout |
| --- | --- | --- |
| `stereo_44100_q4.ogg` | libvorbis 1.3.7, GStreamer 1.28.3 `oggmux` | stereo, two modes, blocks 256 and 2048 |
| `sweep_8000_mono_q0.ogg` | libvorbis 1.3.7, GStreamer 1.28.3 `oggmux` | mono, 8 kHz |
| `surround51_48000_q4.ogg` | libvorbis 1.3.7, GStreamer 1.28.3 `oggmux` | 5.1 |
| `ffmpeg_stereo_44100.ogg` | FFmpeg 8.1.1 native encoder and Ogg muxer | stereo, one mode |
| `chained_mono_stereo.ogg` | the sweep file then the stereo file | a chain of two links |

| File | SHA-256 |
| --- | --- |
| `stereo_44100_q4.ogg` | `f140a95d28da583bf24dbd02ba567402526c2d121e725d80c96319157ae91216` |
| `sweep_8000_mono_q0.ogg` | `bc92c0ff7b253c378decf8cb7cee9ed1883c38f4824b37a22dca05a7f9f8efe7` |
| `surround51_48000_q4.ogg` | `db2a3ebfce449f87bd8be23a248b4df313205f2d94cf005675cc9e91e5a609ac` |
| `ffmpeg_stereo_44100.ogg` | `72cb20fcf1311cdc34b2766b3b15019444bc623cdacb65b5dff81cb94fbfaf00` |
| `chained_mono_stereo.ogg` | `3fe2f7c226f1abdf69b2063983ad5b7fe9e3bde7716b1cd66a19d79f95ef54a7` |
