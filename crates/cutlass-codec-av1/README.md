# cutlass-codec-av1

Native AV1 decoder for Cutlass. Decode only. The production path does not
link libaom, dav1d, libgav1, FFmpeg, or an operating-system decoder.

Specification, state lifetimes, and the coverage matrix are in `docs/av1/`.

## What this build reconstructs

Reduced-still and ordinary intra frames that stay inside a narrow lossless
8-bit 4:2:0 path: one uniform tile, tools that this build rejects left off,
DC / vertical / horizontal luma prediction, and 4×4 lossless Walsh–Hadamard
coefficients. `capabilities()` does not advertise a profile until that
profile's legal configurations reconstruct.

Other tools return `DecodeError::Unsupported` before a frame is published.
Large-scale tile lists are an explicit unsupported mode. Film grain is parsed
only far enough to reject application; reconstruction output is grain-free.

## Input

`Av1Decoder::submit` takes one complete low-overhead OBU packet or one Annex B
temporal unit, chosen by `DecoderConfig::framing`. The two framings are not
auto-detected. An OBU with `obu_has_size_field == 0` needs the enclosing
length from that framing. IVF is not a decoder input; `av1dec` strips IVF
length prefixes before calling `submit`.

```bash
cargo run -p cutlass-codec-av1 --example av1dec -- --grain-off --dump out.raw clip.ivf
```

`--dump` writes tight planes in Y, U, V order. 8-bit samples are bytes.
10/12-bit samples are little-endian `u16` (in-memory storage is native-endian).

## Limits

Defaults live in `DecoderLimits`: 4096 pixels on a side, 4096×2176 total
pixels, 16 MiB packets, 8 MiB OBUs, one worker. `workers` above 1 is rejected.
Sequence header maxima are not an allocation grant.
