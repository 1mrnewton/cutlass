# Specification to code

Section numbers are the v1.0.0 Errata 1 markdown (`06` syntax, `07` semantics,
`08` decoding, `09` parsing). The PDF numbers those as sections 5–8.

| Spec | Code |
| --- | --- |
| 4 / descriptors `f`, `uvlc`, `le`, `leb128`, `su`, `ns` | `bits.rs` |
| 5.2 low-overhead, 5.3 OBU header | `obu.rs` |
| 5.5 sequence header, color config | `sequence.rs` |
| 5.6–5.8 timing, operating points | `sequence.rs` |
| 5.9 uncompressed header through film grain | `header.rs` (grain params rejected when present) |
| 5.9 frame OBU, tile group | `obu.rs`, `lib.rs`, `tiledec.rs` |
| 5.10 metadata | bounded and ignored in `lib.rs`; payloads are not retained |
| 5.11 tile list | recognized in `obu.rs`, rejected as unsupported |
| 8.2 symbol decoder, CDF update | `symbol.rs` (adaptation exists; frames require it disabled) |
| 8.3 CDF selection | `cdf.rs`, `coeff_cdf.rs`, `tiledec.rs`, `coeff.rs` |
| Annex A levels | parsed, not used as an allocation cap |
| Annex B | `annexb.rs` |
| Annex D / 7.5 large-scale tiles | unsupported mode |
| Annex E decoder model | timing syntax in the sequence header; the decoder does not sleep |
| 7.8 motion field, 7.9 MV prediction | not implemented |
| 7.10 prediction | `intra.rs` (DC, vertical, horizontal only) |
| 7.11 palette, 7.11.4 CfL | not implemented; rejected before pixels |
| 7.12 dequant, 7.13 inverse transform | `quant.rs`, `txfm.rs` (4×4 lossless WHT only) |
| 7.14 deblock, 7.15 CDEF | header syntax in `header.rs`; nonzero levels rejected |
| 7.16 upscale, 7.17 restoration | header syntax; non-1:1 superres and restoration rejected |
| 7.18 output and film grain | presentation in `lib.rs`; grain application rejected |
| 7.19–7.21 reference and show-existing | refresh flags parsed; slots and show-existing rejected |

Profile and level legality is checked where the specification states a
requirement (profile/bit-depth/subsampling combinations, reserved bits,
operating-point idc). Level limits do not replace `DecoderLimits`.
