# AIFF fixtures

Small files for `tests/external/damaged.rs`: unfinished or cut files, and corners of the format. Each holds mono sound at 8 kHz (one file at 244800/11 Hz) made from the first 100 samples of one 16-bit sequence: `((state >> 8) & 0xFFFF) - 32768` of `state = (state * 1103515245 + 12345) & 0x7FFFFFFF`, starting from 12345. The test builds the same sequence, so the 16-bit sound each file must give is known without another reader. With FFmpeg installed the test also compares against `ffmpeg -i <file> -f s16le -`.

| File | What it is | What plays |
| --- | --- | --- |
| `raw-unsigned.aifc` | AIFF-C compression `raw `: 8-bit offset binary, 128 for silence | 100 frames, the top byte of each sample |
| `no-final-pad.aiff` | 99 signed 8-bit frames; `SSND` is an odd 107 bytes and the pad byte after it is missing | 99 frames |
| `ssnd-past-end.aiff` | `COMM` and `SSND` state 100 16-bit frames; the file ends inside frame 70 | 69 frames |
| `form-past-end.aiff` | The FORM size is 1,000 bytes more than the file | 100 frames |
| `zero-frames.aiff` | `numSampleFrames` is 0; `SSND` holds 100 frames | 100 frames |
| `rate-22254.aiff` | Sample rate 244800/11 = 22254.5454… Hz | 100 frames at 22255 Hz |
| `depth-12.aiff` | 12-bit samples, two bytes each, low four bits zero | 100 frames, read as 16-bit |
| `depth-20.aiff` | 20-bit samples, three bytes each | 100 frames, read as 24-bit |
| `fl32-upper.aifc` | Compression `FL32`, the upper-case float tag | 100 frames |
| `fl64-upper.aifc` | Compression `FL64` | 100 frames |

All were written byte by byte from the sequence; none needs another tool to make.
