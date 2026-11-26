# WAV fixtures

Small files that are unfinished or cut, for `tests/external/damaged.rs`. Each holds 16-bit little-endian sound at 8 kHz taken from one sequence: the samples `((state >> 8) & 0xFFFF) - 32768` of `state = (state * 1103515245 + 12345) & 0x7FFFFFFF`, starting from 12345. The test builds the same sequence, so what each file must play is known without another reader. With FFmpeg installed the test also compares against `ffmpeg -i <file> -f s16le -`.

| File | What is wrong with it | What plays |
| --- | --- | --- |
| `streamed-ffmpeg-pipe.wav` | Written by `ffmpeg -f wav pipe:1` (FFmpeg 6.1.1, `-bitexact`): the RIFF size and the `data` size are both `0xFFFFFFFF`, and there is no `ds64` | 200 mono frames |
| `data-size-zero.wav` | RIFF size 0 and `data` size 0, as a writer that never went back to the header leaves them | 200 mono frames |
| `data-past-end.wav` | `data` states 400 bytes; the file ends after 301 of them, inside frame 151 | 150 mono frames |
| `data-partial-frame.wav` | Stereo; `data` is 403 bytes, three more than 100 frames | 100 stereo frames |
| `rf64-sample-count-zero.wav` | RF64 whose `ds64` sample count is 0 | 200 mono frames |
| `bw64.wav` | The RF64 layout under the `BW64` brand | 200 mono frames |
| `trailing-chunk-cut.wav` | The sound is whole; the `LIST` chunk after it is cut short | 200 mono frames |

Only the first file needs FFmpeg to make. The rest are a header and the sequence, written byte by byte.
