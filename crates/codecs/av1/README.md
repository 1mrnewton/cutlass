# cutlass-av1

Native AV1 decoder and encoder. The support record is `STATUS.md`. These percentages are a scope estimate of that record, not a line count and not a coverage report.

About **45%** of the full decoder-and-encoder job is done. The constrained 8-bit 4:2:0 path is further along than a file a normal encoder writes.

| Slice | Done | What that covers |
| --- | --- | --- |
| Decoder | ~60% | Headers, references, intra, and the in-loop chain through restoration. A default encode still will not open. |
| Encoder | ~25% | Keyframes only. The rate and quality gates for that path already pass. |
| Proof | ~40% | Exact dav1d matches on the supported subset. Speed, the long fuzz run, and a real production file are still open. |

## Decoder

| Piece | Done |
| --- | --- |
| OBUs, sequence and frame headers, eight reference slots, show-existing, hidden frames, failed-packet rollback | ~90% |
| Intra prediction, partitions, chroma-from-luma, 2D inverse DCT through 64 | ~80% |
| Identity and 1D transforms | ~40%. The inverse exists. The coefficient order is not proven. |
| Single-reference inter | ~20%. Compound, warped, overlapped, order hint, and scaled references are refused. |
| Deblocking | ~70% |
| CDEF | ~90% |
| Super resolution | ~75%. Proven on a keyframe. Not proven on an inter frame. |
| Loop restoration | ~90%. Wiener, self-guided, and switchable match dav1d on keyframes. |
| Film grain | 0% |
| 10-bit, 12-bit, monochrome, 4:2:2, 4:4:4 | 0% |

Current decoder work stays on this table. The encoder is left as it is until these decode gaps move.

## Encoder

| Piece | Done |
| --- | --- |
| Temporal delimiter, sequence header, DC keyframe, show-existing | ~90% |
| Rate control and the 28 dB / 500 kbit/s gates | ~80% |
| Inter frames | 0% |
| CDEF, restoration, super resolution, film grain | 0% |

## Proof

| Gate | Done |
| --- | --- |
| dav1d pixel match on the supported tools | ~70% |
| Hostile input returns an error | ~80% |
| 30-minute fuzz | 0%. 200 random packets is the stand-in. |
| 720×480 real time on one thread | 0%. About five round trips a second. |
| A production 2160p file plays | 0%. It fails closed. |

Film grain, order hint, and compound prediction are the tools still in front of that 2160p file.
