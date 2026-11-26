# CAF fixtures

Small files for `tests/external/damaged.rs`. With FFmpeg installed the test also compares the decoded samples against `ffmpeg -i <file> -f s16le -`.

| File | What it is | What plays |
| --- | --- | --- |
| `data-past-end.caf` | 16-bit big-endian mono PCM at 8 kHz; `data` states 100 frames and the file ends inside frame 70 | 69 frames |
| `pakt-past-end.caf` | ALAC, 8 kHz mono, four packets in the packet table (3003, 3003, 3003 and 1519 bytes); the file is cut inside the third | The first two packets, 8,192 frames |
| `rate-22254.caf` | 16-bit PCM whose `mSampleRate` is 244800/11 = 22254.5454… Hz | 100 frames at 22255 Hz |

The PCM files hold the first 100 samples of `((state >> 8) & 0xFFFF) - 32768` of `state = (state * 1103515245 + 12345) & 0x7FFFFFFF`, starting from 12345, written byte by byte.

`pakt-past-end.caf` is FFmpeg 6.1.1's ALAC encode (`-bitexact`) of the triangle wave `(abs(i mod 64 - 32) - 16) * 256`, with its chunks put in the order Apple's tools write them, `desc`, `kuki`, `pakt`, `data`, so the table is still there when the sound is cut.
