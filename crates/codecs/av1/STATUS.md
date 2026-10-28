# AV1 support record

Checkout: branch `av1`. Native safe Rust in `cutlass-av1`. No `unsafe`. No libaom, dav1d, SVT-AV1, rav1e, or FFmpeg linked into the build. Those programs are test oracles only.

This record is the scope written before the implementation. The readiness section at the bottom is filled after the checks that actually ran.

## Specifications

- AOMedia AV1 Bitstream & Decoding Process Specification, version 1.0.0 with Errata 1 (8 January 2019). Tag `v1.0.0-errata1`. This supersedes 1.0.0. The syntax, decoding process, parsing process, and the default CDF, scan, quantizer, and block-size tables in that document are normative.
- Annex A of that specification for Main profile and levels. Annex B is the length-delimited temporal-unit format. This crate reads low-overhead OBUs (`obu_has_size_field = 1`), which is what an IVF frame and an ISOBMFF sample carry after the container strips its own length.
- There is no later AV1 bitstream amendment that changes the 1.0.0-errata1 decoding process. AV1 in ISOBMFF (`av1C`) and AV1 in Matroska are container bindings. They are out of this crate. Film grain synthesis is section 7.17 of the same specification. It is applied to output pictures and is not stored in reference frames.

## Sources read before the Cutlass code

- The specification markdown for syntax (section 5), semantics, the decoding process (section 7), the symbol parser (section 8), Annex A, and the default CDF and scan tables.
- FFmpeg 8.1 `libavcodec/av1dec.c` parses headers, reference frames, and film-grain metadata, then reconstructs only through a hardware accelerator. It is not a software tile decoder. `libdav1d.c` is the software decoder and it calls dav1d. `libaomenc.c` and `libsvtav1.c` call libaom and SVT-AV1. `cbs_av1` parses the same OBU syntax. Where a wrapper and the specification disagree, the specification wins.
- GStreamer `gstav1parse` splits low-overhead and Annex B streams into OBUs and temporal units. It does not reconstruct samples. The gst-plugins-rs dav1d and rav1e elements were not in the tree that was checked out; the plugin that was read is a parser in front of an external decoder.
- dav1d 1.5.x (`obu.c`, `msac.c`, `decode.c`, `recon_tmpl.c`, `ipred_tmpl.c`, `itx_1d.c`, `lf_apply_tmpl.c`) is the production decoder that was read. The arithmetic coder keeps the complement of the raw bits in its window. The specification's `read_symbol` is the rule this crate implements.
- rav1e (`encoder.rs`, `ec.rs`, `predict.rs`) is the production encoder that was read. It chooses partitions and modes, then range-codes symbols. libaom is the reference encoder behind `aomenc`. SVT-AV1 is a second production encoder behind FFmpeg's wrapper. This crate does not copy those sources.

## What this crate is required to do

Progressive 8-bit 4:2:0, `seq_profile` 0, one operating point, dimensions up to 4096×2304. Sequence headers, frame headers, temporal delimiters, frame OBUs, tile groups, padding, and metadata OBUs that are skipped. Eight reference slots. Decode order versus presentation order, including `show_frame = 0` and `show_existing_frame`. Key, intra-only, and inter frames. Single-reference inter prediction. Square and rectangular partitions that the specification uses for a 64×64 superblock. Intra DC, vertical, horizontal, smooth, Paeth, and directional modes. Inverse DCT through 64 points, and ADST where a 8- or 16-point intra set needs it. Loop filtering. A real quantizer and a rate update on the encoder, not a fixed-quantizer demo.

Main profile allows 10-bit 4:2:0 and monochrome. This crate rejects both. It is not a Main-profile-compliant decoder. The rejection is a typed error before the frame is published.

Film grain, super-resolution, and loop restoration are legal in profile 0 and are not required to be switched on. A stream that turns one of them on is rejected. The picture is not published with that stage skipped. Consequence: a production encode that leaves restoration or film grain on will not open.

Also rejected, with the same rule: 4:2:2, 4:4:4, 12-bit, scalability beyond operating point 0, large-scale tile lists, warped motion, OBMC, palette, intra block copy, chroma-from-luma, filter intra, masked and distance-weighted compound, 128×128 superblocks, quantization matrices, and screen-content tools. The encoder does not emit them.

## Gates declared before tuning

- Decoder versus dav1d 1.5.3, and versus `aomdec` when it is present: exact 8-bit 4:2:0 samples on streams whose tools are in the supported set. dav1d with `--inloopfilters none` is only a reconstruction diagnostic. The acceptance compare uses the default in-loop filters.
- Encoder: `dav1d` accepts the bitstream. Pixels match this decoder. Over 30 frames of 320×180 at 500 kbit/s, the average rate lands in 0.60×–1.40× of the target. A 64×64 step pattern at quality index 40 has luma PSNR at least 28 dB against the source after an independent decode. These are lossy gates. Byte-identical encoder output is not required.
- One packet that fails does not change reference frames. A second `drain` is empty. `reset` drops references and keeps the configuration.
- Hostile input returns `Truncated`, `Malformed`, `Unsupported`, or `Limit`. It does not panic.
- Fuzz budget: 30 minutes on the decoder packet entry. A short random test is the CI stand-in. The 30-minute run is recorded only if it finished.
- Speed budget: one thread, release, on this machine. 720×480 4:2:0 at 30 fps is the playback target (real time). A miss stays a miss.
- Memory: one sequence stays within eight reference frames plus the frame being decoded, each plane bounded by the 4096×2304 limit.

## Fixture rules

`aomenc` and `dav1d` are oracles. A test skips, and says so, when the binary or `../local-files` is absent. Local media stays in that folder. An AV1 sample in that folder is opened by extracting its video payload with the oracle, not by a container parser in this crate.
