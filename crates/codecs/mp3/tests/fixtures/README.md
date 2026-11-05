# MP3 test fixtures

Made on macOS with LAME 3.100 (`LAME 64bits version 3.100`) from synthetic
16-bit WAV files. The WAV generator writes, per sample `i` at rate `r`:

- `sine`: `int(0.5 * sin(2π · 1000 · i / r) * 32767)` on every channel.
- `clicks`: decaying noise bursts every 0.25 s, `exp(-60 p) · U(-0.9, 0.9)` for the first 0.1 s of each burst, Python `random.seed(7)`.
- `sweep`: a linear chirp from 20 Hz to 0.45 · r at half scale (right channel at 1.01 × the phase).
- `stereo`: 440 Hz at 0.4 on the left; 660 Hz at 0.4 plus 0.1 uniform noise on the right.

All are free of third-party content.

| File | Source | Command | Exercises | SHA-256 |
| --- | --- | --- | --- | --- |
| `sine_44100_j128_crc.mp3` | sine, 44.1 kHz stereo, 0.5 s | `lame -b 128 -m j -p` | MPEG-1, joint stereo, CRC, Info tag after a CRC | `9f138576db58f901ea62e4f65115725f5c9707be9d58ee278633007c5ca50142` |
| `clicks_48000_v2.mp3` | clicks, 48 kHz stereo, 0.5 s | `lame -V 2` | VBR, Xing tag, short and mixed-transition blocks, reservoir | `a162254a388a7b22708afce008f9837f89cad39fb1e29ca53d0f4d7176582cd3` |
| `sweep_22050_mono64.mp3` | sweep, 22.05 kHz mono, 0.5 s | `lame -b 64 -m m` | MPEG-2 LSF, mono, LSF scalefactors | `190e1158644502afe629f77e747fb04cd0b2189ca2274e24811f6210317a8895` |
| `sine_8000_mono16.mp3` | sine, 8 kHz mono, 0.5 s | `lame -b 16 -m m` | MPEG-2.5, no tag (the frame is too small for one) | `db7eb093cd4f391184d9d3fb26eb6859520832e7e8a4c0822b402e56764be026` |
| `stereo_32000_ms96.mp3` | stereo, 32 kHz, 0.5 s | `lame -b 96 -m f` | MPEG-1, forced mid/side | `60198f4c936f9610b303226abcb2746bdde0847211d8ff014d291e914b276285` |
| `sine_44100_free400.mp3` | sine, 44.1 kHz stereo, 0.25 s | `lame --freeformat -b 400` | free format at 400 kbps | `c4699e48d0b0c810db1492951b81e1142d022091d096aabea12f3156ab6d7677` |

The expected sample counts in `tests/mp3/decoder.rs` are the source lengths,
not decoder output. Reference PCM is produced at test time by FFmpeg 8.1.1
(`mp3float`) and mpg123 1.33.5 when they are installed. FFmpeg refuses free
format above 320 kbps, so `sine_44100_free400.mp3` is checked against mpg123
only.
