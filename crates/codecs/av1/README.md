# cutlass-av1

Native AV1 decoder and encoder. The support record is `STATUS.md`. These percentages are a scope estimate of that record, not a line count and not a coverage report.

About **58%** of the full decoder-and-encoder job is done. The constrained 8-bit 4:2:0 path is further along than a file a normal encoder writes.

| Slice | Done | What that covers |
| --- | --- | --- |
| Decoder | ~80% | Headers, references, intra (including the recursive filter and the extended transform set), film grain, order hint, uniform tiles, a quantizer matrix, and a superblock quantizer delta. A normal file can be opened. Its picture is not yet a dav1d match. |
| Encoder | ~25% | Keyframes only. The rate and quality gates for that path already pass. |
| Proof | ~40% | Exact dav1d matches on the supported subset. Speed, the long fuzz run, and a real production file are still open. |

## Decoder

| Piece | Done |
| --- | --- |
| OBUs, sequence and frame headers, eight reference slots, show-existing, hidden frames, failed-packet rollback | ~90% |
| Intra prediction, partitions, chroma-from-luma, 2D inverse DCT through 64 | ~80% |
| Identity and 1D transforms | ~70%. Horizontal-only and vertical-only coefficient contexts follow the specification. An extended-set keyframe matches dav1d. |
| Single-reference inter | ~40%. An error-resilient frame may carry an order hint. Compound, warped, overlapped, reference motion vectors, and scaled references are refused. |
| Deblocking | ~70% |
| CDEF | ~90% |
| Super resolution | ~75%. Proven on a keyframe. Not proven on an inter frame. |
| Loop restoration | ~90%. Wiener, self-guided, and switchable match dav1d on keyframes. |
| Film grain | ~80%. Luma, chroma, autoregression, overlap, and chroma-from-luma match dav1d. The noise is not stored in a reference. Copying grain from another frame, and re-applying it on show-existing, are still refused. |
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
| A production 2160p file plays | 0%. The picture is produced and does not match dav1d. |

The 2160p keyframe now parses, including its four uniform tiles, quantizer matrix, superblock quantizer delta, and selectable transform sizes. After the loop filter, luma differs in a few thousand samples and chroma differs in about a quarter of its samples. Compound prediction and carrying CDF state from another frame are still refused. An inter frame that selects its transform size is still refused.
