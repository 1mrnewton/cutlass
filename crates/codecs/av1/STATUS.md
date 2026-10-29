# AV1 support record

Checkout: branch `av1`. Native safe Rust in `cutlass-av1`. No `unsafe`. No libaom, dav1d, SVT-AV1, rav1e, or FFmpeg linked into the build. Those programs are test oracles only.

This record is the scope written before the implementation. The readiness section at the bottom is filled after the checks that actually ran.

## Specifications

- AOMedia AV1 Bitstream & Decoding Process Specification, version 1.0.0 with Errata 1 (8 January 2019). Tag `v1.0.0-errata1`. This supersedes 1.0.0. The syntax, decoding process, parsing process, and the default CDF, scan, quantizer, and block-size tables in that document are normative.
- Annex A of that specification for Main profile and levels. Annex B is the length-delimited temporal-unit format. This crate reads low-overhead OBUs (`obu_has_size_field = 1`), which is what an IVF frame and an ISOBMFF sample carry after the container strips its own length.
- There is no later AV1 bitstream amendment that changes the 1.0.0-errata1 decoding process. AV1 in ISOBMFF (`av1C`) and AV1 in Matroska are container bindings. They are out of this crate. Film grain synthesis is part of the output process in that specification. It is applied to output pictures and is not stored in reference frames.

## Sources read before the Cutlass code

- The specification markdown for syntax (section 5), semantics, the decoding process (section 7), the symbol parser (section 8), Annex A, and the default CDF and scan tables.
- FFmpeg 8.1 `libavcodec/av1dec.c` parses headers, reference frames, and film-grain metadata, then reconstructs only through a hardware accelerator. It is not a software tile decoder. `libdav1d.c` is the software decoder and it calls dav1d. `libaomenc.c` and `libsvtav1.c` call libaom and SVT-AV1. `cbs_av1` parses the same OBU syntax. Where a wrapper and the specification disagree, the specification wins.
- GStreamer `gstav1parse` splits low-overhead and Annex B streams into OBUs and temporal units. It does not reconstruct samples. The gst-plugins-rs dav1d and rav1e elements were not in the tree that was checked out; the plugin that was read is a parser in front of an external decoder.
- dav1d 1.5.x (`obu.c`, `msac.c`, `decode.c`, `recon_tmpl.c`, `ipred_tmpl.c`, `itx_1d.c`, `lf_apply_tmpl.c`) is the production decoder that was read. The arithmetic coder keeps the complement of the raw bits in its window. The specification's `read_symbol` is the rule this crate implements.
- rav1e (`encoder.rs`, `ec.rs`, `predict.rs`) is the production encoder that was read. It chooses partitions and modes, then range-codes symbols. libaom is the reference encoder behind `aomenc`. SVT-AV1 is a second production encoder behind FFmpeg's wrapper. This crate does not copy those sources.

## What this crate is required to do

Progressive 8-bit 4:2:0, `seq_profile` 0, one operating point, dimensions up to 4096×2304. Sequence headers, frame headers, temporal delimiters, frame OBUs, tile groups, padding, and metadata OBUs that are skipped. Eight reference slots. Decode order versus presentation order, including `show_frame = 0` and `show_existing_frame`. Key, intra-only, and inter frames. Single-reference inter prediction. Square and rectangular partitions that the specification uses for a 64×64 superblock. Intra DC, vertical, horizontal, smooth, Paeth, and directional modes. Inverse DCT through 64 points, and ADST where a 8- or 16-point intra set needs it. Loop filtering. A real quantizer and a rate update on the encoder, not a fixed-quantizer demo.

Main profile allows 10-bit 4:2:0 and monochrome. This crate rejects both. It is not a Main-profile-compliant decoder. The rejection is a typed error before the frame is published.

Film grain is legal in profile 0 and is not required to be switched on. When a frame sends its own parameters, the noise is added to the shown picture and the reference keeps the picture from before that step. A frame that copies grain parameters from another frame is rejected, and a show-existing frame does not re-apply the reference's grain. Wiener, self-guided, and switchable restoration are applied for 8-bit 4:2:0 after CDEF. Super resolution is applied for 8-bit 4:2:0: the frame is reconstructed at the coded width and the shown picture is the upscaled width. The encoder leaves both off. A frame whose two luma loop-filter levels are zero is left unfiltered, including when a reference delta is present.

Also rejected, with the same rule: 4:2:2, 4:4:4, 12-bit, scalability beyond operating point 0, large-scale tile lists, warped motion, OBMC, palette, intra block copy, masked and distance-weighted compound, 128×128 superblocks, screen-content tools, and lossless (`base_q_idx` 0, which forces 4×4 Walsh-Hadamard). A DC luma block no larger than 32 on a side may use the recursive intra filter. A frame may name a quantizer matrix level per plane. The matrix scales the quantizer before the inverse transform. Level 15 and an identity transform leave the quantizer unchanged. Segmentation, which can pick a different level per segment, is still rejected. The encoder does not emit them. Chroma-from-luma is implemented for profile 0 and matched on an `aomenc` stream that uses it. CDEF is applied after the loop filter for 8-bit 4:2:0. The encoder still leaves it off.

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

## Readiness

This is what the code does after `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and `cargo deny check`. Those four passed on this machine. Nothing was published.

### Decoder

Partial. Progressive 8-bit 4:2:0 keyframes, intra-only frames, and single-reference inter frames are reconstructed. Sequence headers, frame headers, temporal delimiters, frame OBUs, eight reference slots, `show_existing_frame`, and a hidden frame (`show_frame = 0`, empty output) are implemented. A failed packet does not change the slots. `drain` is empty. `reset` drops the slots and keeps the sequence. An inter frame reads each reference's frame-id delta when the sequence numbers frames.

Matched exactly against dav1d 1.5.3, with default in-loop filters:

- The embedded 64×64 constrained keyframe in the crate.
- An embedded 64×64 three-frame `aomenc` stream: one keyframe and two single-reference inter frames. That stream is error resilient, keeps the largest transform, uses identity global motion, turns CDF updates off, and leaves compound, warped, overlapped, reference-motion, order-hint, CDEF, restoration, and film grain off. The second frame's luma is the first frame shifted by 8 samples.
- An embedded 64×64 three-frame `aomenc` stream with order hint on. Each error-resilient frame repeats the eight reference hints. The pictures match dav1d. Changing one signaled hint makes the frame fail, and the stored reference is still usable.
- Two embedded `aomenc` keyframes with CDEF on and the loop filter off: a 64×64 step, and a 64×128 ramp that crosses a 64-sample stripe. Both match dav1d with the filter applied.
- One embedded `aomenc` keyframe that codes a 64×64 horizontal ramp at width 43 and upscales it. The shown picture matches dav1d. Inter frames that use super resolution are not in that stream.
- One embedded `aomenc` keyframe whose luma uses the self-guided restoration filter. Chroma selects no restoration. The shown picture matches dav1d.
- One embedded `aomenc` keyframe whose luma uses the Wiener restoration filter. Chroma selects no restoration. The shown picture matches dav1d.
- One embedded `aomenc` keyframe whose luma chooses the restoration filter per unit. Chroma selects no restoration. The shown picture matches dav1d.
- Film grain on embedded `aomenc` 64×64 frames: luma-only grain, lag-1 chroma grain with overlap, chroma grain scaled from luma at lag 2, and a second frame that predicts the first. The shown pictures match dav1d. The second frame matches only if the reference was stored before grain.
- One embedded `aomenc` 128×128 keyframe in a uniform 2×2 tile grid. Each tile is its own entropy stream. Prediction does not cross a tile edge. The shown picture matches dav1d. Non-uniform tile spacing is still refused.
- One embedded `aomenc` 64×64 keyframe whose chroma is not flat. Chroma above-right availability is taken in the chroma 4×4 grid. The shown picture matches dav1d.
- One embedded `aomenc` 64×64 keyframe that names a quantizer matrix. Luma and chroma both use level 3. The shown picture matches dav1d.
- One embedded `aomenc` 128×128 keyframe whose header signals a superblock quantizer delta. The running index moves at the first coded block of a superblock. The shown picture matches dav1d. A per-superblock loop-filter delta is still refused.
- One embedded `aomenc` 64×64 keyframe that uses the recursive intra filter on DC luma (horizontal and Paeth). The shown picture matches dav1d.
- One embedded `aomenc` 64×64 keyframe that selects the transform size and may use the extended intra set. The shown picture matches dav1d. Horizontal-only and vertical-only coefficient contexts use the specification's class numbers.
- Chroma decoded-neighbor flags use the chroma superblock grid. An extended intra edge stops at the tile boundary. A 2160p keyframe from `../local-files` matches dav1d when that file and dav1d are present.
- When the files are present: `aomenc` streams for a step, a ramp, chroma-from-luma, AB partitions, angle delta, rectangular transforms, and 64-point transforms. Deblocking runs on the constrained keyframe (loop-filter levels are non-zero there) and those samples match.

Within-frame CDF adaptation is implemented. The inter streams above turned it off.

Unsupported, refused with a typed error before a picture is published: compound prediction, warped motion, overlapped motion, reference-frame motion vectors, short reference signaling, CDF state carried from another frame, an inter frame that selects its transform size, scaled references, lossless, a per-superblock loop-filter delta, film grain copied from another frame, 10-bit, 12-bit, monochrome, and the other tools named above. A keyframe may select its transform size. A 2160p Main-profile sample from `../local-files` was pushed after `ffmpeg` copied one frame out of the container. That keyframe matches dav1d. A later frame from the same file is not part of that check.

### Encoder

Partial. It emits a temporal delimiter, a sequence header, and a keyframe of DC-predicted 4×4 and 8×8 blocks with the 2D DCT, loop filter off, and CDF update off. A repeated picture is a frame that shows reference slot 0. dav1d 1.5.3 accepted one 64×64 step bitstream and its pixels matched this decoder. A constant-128 picture is exact. The same step at quality index 40 has luma PSNR at least 28 dB against the source. Thirty 320×180 frames at 500 kbit/s landed inside 0.60×–1.40× of the target. One scaled 64×64 frame from `../local-files` round-tripped through this encoder and this decoder.

The encoder does not emit inter frames, film grain, super resolution, restoration, or CDEF. The decoder applies film grain; the encoder still leaves it off.

### Gates still open

- Fuzz: not run. The 30-minute packet campaign did not run. The stand-in is 200 random packets, which leave the stored reference usable and return an error. A LEB128 value that does not fit in `u32` returns `Limit` instead of panicking.
- Speed: miss. One release thread encoded and decoded one 720×480 4:2:0 frame in 0.19s. That is about five round trips a second, short of 30 fps playback. `cargo bench -p cutlass-av1 --bench decode` later decoded one 3840×2160 keyframe from `../local-files` on an Apple M5 Pro. Two runs of five samples had medians 0.329 s and 0.331 s; one sample reached 0.351 s. FFmpeg's libdav1d decoded that same frame in 0.0063–0.0064 s on one thread and 0.0018–0.0019 s with its default threads. The pixels matched.
- Memory: the decoder keeps eight reference pictures plus the frame being decoded. Planes are cropped to the visible size, and a frame edge above 4096 is `Limit`. Peak resident memory was not measured.
- Containers: not this crate. AV1 inside MP4, WebM, and AVIF stays with the container crates.
