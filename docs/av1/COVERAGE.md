# Feature coverage

Status key: `done` has a test or an oracle compare in this tree; `partial`
parses and implements the path but is not yet oracle-compared; `gap` is not
implemented and is rejected or untested. A gap is not a pass.

| Family | Status | Evidence |
| --- | --- | --- |
| Profiles 0–2, depths, layouts, mono, identity | partial | Sequence/color parser enforces the spec combinations. Pixel formats exist. Oracle clips not yet compared for 4:4:4 / 12-bit. |
| Still / reduced header, both superblocks | partial | Syntax implemented. Needs fixtures. |
| Low-overhead, Annex B, no-size OBU, reserved types | partial | Unit tests for truncation, LEB128, inconsistent lengths. |
| Frame OBU, tile groups across packets, EOF | partial | One frame OBU with one tile is assembled. Multiple tiles, split tile groups, and EOF mid-frame are not completed reconstructions. |
| Show / hidden / show-existing | gap | A finished shown keyframe is presented. Hidden frames and show-existing return `Unsupported`. |
| References, short signaling, order hint | gap | Refresh flags and order-hint bits are parsed. Eight reference slots are not stored. |
| Entropy CDFs and adaptation | partial | Symbol decoder unit test and a libaom-matching range encoder used to build fixtures. Mode and coefficient defaults generated from libaom v3.13.1. Adaptation is parsed only when disabled. |
| Partitions, intra modes, CfL, palette, intrabc | partial | Implied split plus NONE on a 16×16 keyframe. DC, vertical, and horizontal predictors have unit tests. Only DC is reached from a chroma-bearing block. CfL, palette, and intrabc are rejected. |
| Motion, compound, warp, OBMC | gap | |
| Segmentation, delta-Q, delta-LF, lossless | partial | Header syntax. A reduced-still 16×16 lossless 8-bit 4:2:0 keyframe reconstructs. Segmentation and delta-q are rejected before pixels. |
| Coefficients, 19 transforms | partial | 4×4 `all_zero` and a positive DC level through dequant and the lossless Walsh–Hadamard kernel. The level-1 case is checked against the specification arithmetic (sample 129 at the origin). The other 18 transform sizes and non-lossless types are not reconstructed. One libaom v3.13.1 lossless 16×16 constant-128 frame matches `aomdec --rawvideo --skip-film-grain` byte for byte when palette, intrabc, filter-intra, CDEF, restoration, matrices, and CDF updates are disabled. A non-constant frame from the same encoder stops at an unsupported partition. |
| Deblock, CDEF, superres, restoration, grain | gap | Required before claiming sample equality. |
| Operating points, metadata | partial | Parsed and scoped to the sequence. |
| Tile list | unsupported | Explicit error. Not counted as coverage. |
| API reset/drain/backpressure | partial | Unit tests around the state machine. |
| SIMD / multi-worker | not claimed | `workers` must be 1. Scalar only. |

Oracle build and corpus hashes are recorded in `REFERENCES.md`. Unrun 12-bit
and SIMD checks stay unrun.
