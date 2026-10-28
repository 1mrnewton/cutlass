# Video and image codec implementation details

## Configuration and image domain

Resolve profiles/levels, dimensions, bit depth, chroma subsampling, planar/packed layout, component order, strides, crop/visible region, alpha, color primaries/transfer/matrix/range, and required metadata. Validate combinations and codec-specific maxima. Do not conflate coded size, display size, and buffer allocation size.

Keep color conversion, scaling, tone mapping, and compositor behavior explicit. A decoder that outputs correct coded-domain samples must not be blamed for a separately applied display conversion; equally, a pipeline with the wrong color metadata is not integrated correctly.

Bound pixel count, stride/plane sizes, reference buffers, partitions, tiles/slices, recursion, expanded image data, and entropy work before allocating or processing. Check odd sizes, minimum dimensions, alignment/padding, and high-precision arithmetic.

## Video decoder checklist

Implement all required entropy coding/adaptation/reset, prediction, transform/inverse quantization, reconstruction, reference handling, loop/restoration filters, partitioning, tiles/slices, and display-order behavior for the advertised target.

For interframe codecs, cover required inter prediction and reference updates, motion vectors/interpolation, compound/weighted prediction, skipped blocks, temporal dependencies, decode versus presentation order, random-access points, and required preroll. An intra/keyframe-only or lossless-only subset is scaffolding unless explicitly requested as the final target.

Cover format-specific features such as H.264 parameter sets/DPB/reordering, HEVC parameter sets/tiles/reference management, VP8/VP9 reference and entropy state, and AV1 sequence/frame headers, OBUs, reference slots, super-resolution, restoration, film grain, and show-existing/hidden frames where required. Derive the precise checklist from the applicable standard; this list is not an exhaustive codec specification.

Treat required film grain or other reconstruction-affecting tools deliberately. Expose optional output stages if the format permits, compare pre/post stages correctly, and do not silently claim rendered output with a required stage skipped.

Test resolution/profile changes, damaged references, loss/recovery, seeking, flush/reset, end-of-stream delayed frames, and cancellation. Never release a buffer still owned by a reference frame or expose stale output from the previous source.

## Video encoder checklist

Implement valid headers/config records, intra/inter decisions as scoped, reference/reorder management, prediction/motion search, transforms/quantization, entropy coding, in-loop reconstruction/filtering, and normative profile/level/buffer constraints.

Support justified rate-distortion decisions, rate/quality modes, bounded search, keyframe/GOP behavior, scene changes, latency/lookahead, and drain. Validate actual bitrate, buffer model where applicable, output timing, random access, and closed/open sequence behavior as advertised.

Evaluate quality at matched actual bitrate and stated preset/latency constraints. Use appropriate luma/chroma/perceptual metrics and visual inspection of difficult sequences. Do not infer useful compression or subjective quality from valid output alone.

Do not implement a decoder when only encoding was requested; independent decoders are the primary interoperability evidence. A reconstruction loop needed by the encoder is an internal algorithm, not permission to expand product scope.

## Still-image and animation checklist

Treat still images separately from interframe video. Resolve single image versus animation, precision, grayscale/color, alpha semantics, progressive/interlaced modes, metadata, orientation, ICC/color handling, palette/transparency, and lossless/lossy variants.

Implement the requested format modes, integrity checks, termination, and legal ordering of chunks/markers. Bound metadata and decompressed output. Handle partial/progressive input explicitly if advertised.

For animation, test disposal/blend modes, frame rectangles, timing, loop counts, partial updates, and canvas history. Single-frame success does not imply animation support.

For lossless promises, compare exact samples/indices/alpha in the declared domain; distinguish equivalent rendering from identical encoded bytes or palette representation. For lossy images, compare meaningful perceptual/rate behavior and document unavoidable source-domain conversions.

## Verification corpus and engine paths

Use official vectors and independently encoded natural, synthetic, and difficult content. Include flat fields, impulses/checkerboards, ramps, high-frequency detail, screen/text, noise, motion/occlusion, scene cuts, fades, high dynamic range where supported, extreme aspect ratios, alpha gradients, and per-plane patterns.

Cover supported profiles/depths/chroma, odd/min/max dimensions, inter tools, reference/reorder cases, tiles/slices, lossless modes, progressive/animated modes, corruption, truncation, and hostile expansion sizes. Add tool-specific fixtures rather than relying on a producer's defaults.

Validate raw/framed payloads, codec configuration records, and relevant container bindings (for example start-code versus length-prefixed forms). Avoid putting transport framing where a muxer expects raw samples.

Exercise import -> demux -> decode -> engine picture/render boundary, random-access/seek/scrub/loop, timeline cuts, and encode -> mux -> independent reopen for requested directions. Verify frame ordering/duration, color/crop/alpha metadata, and audio/video sync. Identify unimplemented application boundaries accurately.
