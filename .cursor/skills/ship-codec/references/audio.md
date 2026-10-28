# Audio codec implementation details

## Source and output contracts

Specify PCM type/precision, endian/layout where relevant, channel identities/order, sample rate, sample count, valid range, ownership, and conversion policy. A compressed codec's profile is not a PCM “bit depth.” Follow shared engine PCM contracts instead of introducing ad hoc buffers.

Validate NaN/infinity, out-of-range floats, uneven channel lengths, integer extrema, empty buffers, and partial frames. Do not silently clip normalized float reconstruction, resample, remix, or discard channels. Make conversions explicit at their owner boundary.

Model core versus rendered sample rate/count for extension codecs. Track frame samples, lookahead, analysis/synthesis history, priming, delay, padding, and valid trim separately. Use rational/sample-based timestamps, including long 44.1 kHz paths.

Define drain separately from seek/reset. Preserve the last valid input samples when an encoder pads its final frame; never expose padding as original content when the container supplies valid trim metadata. Do not invent gapless guarantees for a transport that cannot carry that metadata.

## Algorithm checklist

Identify the codec's actual tool set from the standard and support matrix. Implement applicable:

- Framing/configuration, channel elements/maps, entropy codebooks or range coding, state initialization/reset, scale factors, and legal bit budgets.
- Analysis/synthesis filterbanks, MDCT or other transforms, windows, normalization, overlap-add, block switching, and history.
- Prediction, residuals, quantization/inverse quantization, stereo/coupling, noise substitution/shaping, in-loop/post filters, and required extension tools.
- Psychoacoustic/masking/bit-allocation or format-appropriate encoder decisions, bounded search, rate/reservoir control, useful quality ladders, and legal tool signaling.
- Lossless predictors/residual coding, checksums, exact integer precision, and bounded reconstruction where applicable.

Verify transforms/filterbanks against an independent direct mathematical implementation and analytic invariants. Test impulses at all relevant boundary positions, long/short transitions, first/last frames, extrema, and finite behavior.

## AAC specialization

For AAC tasks, explicitly resolve LC versus HE v1/v2 and other AOT/profile requirements. For unqualified Cutlass AAC import/export requests, propose LC plus HE v1/v2 for the requested direction(s), unless active project or user requirements specify a different target. State this assumption before implementing. Do not silently add every AAC-family technology; do not count LC core fallback as HE support. Preserve any explicit existing scope when continuing work.

Cover applicable ASC/GASpecificConfig, indexed/explicit frequencies, 1024/960 frames and short windows, PCE/channelConfiguration, SCE/CPE/CCE/LFE and other raw elements, Huffman/sign/escape coding, scale-factor bands, pulse, TNS, PNS, M/S/intensity/coupling, sine/KBD windows, overlap, and order of operations.

For HE, implement real SBR QMF/frequency tables/patch/envelope/noise/harmonic/gain handling and PS parameter/decorrelation/mixing reconstruction/analysis. Test explicit/implicit/hierarchical/mixed signaling, valid rate modes, channel reconstruction, delay, and parameter reuse. Refuse or explicitly identify unsupported extensions such as USAC/xHE-AAC, LD/ELD, or ER variants.

Keep ADTS/ADIF/LOAS/LATM and MP4 adapters at the transport/container boundary, each with an explicit support decision. Validate CRC/multi-raw-block and mux-config reuse modes as applicable. Do not insert ADTS headers into MP4 samples. Produce correct raw payload/config data and preserve trim/edit metadata through mux/demux.

Use ISO/IEC 13818-7 and 14496-3 provisions applicable to the advertised stream, and ISO/IEC 14496-26 conformance methodology. Record editions/amendments and verify critical constants against accessible normative sources. Do not claim access to unavailable standards.

## Audio verification corpus

Use silence, DC, positive/negative impulses, boundary impulses, tones/sweeps/near-Nyquist content, near-silence, full-scale extrema, correlated/uncorrelated noise, percussion/transients, speech, music, tonal stress samples, stereo isolation/anti-phase, and per-channel IDs/LFE.

Cover every supported rate/layout/frame mode, representative bitrate/quality ladders, legal extreme requests, short/fractional-frame source lengths, all relevant tail lengths, long streams, and changing configurations. Validate channel association, polarity, leakage, DC, clipping, bandwidth, transient pre-echo, and stereo image.

For decoders, compare independent reference output under the appropriate conformance tolerance. For encoders, decode with independent implementations and compare actual bitrate and perceptual quality at matched conditions. Assess SBR/PS/noise tools using suitable methods; waveform SNR alone is inadequate.

For lossless audio, require bit-exact original PCM in the promised precision/domain. For gapless tests, assert exact valid sample counts when metadata permits; otherwise state the transport's limitation.

Exercise preview start/pause/resume, seek with history/preroll, looping, clips/joins, export boundaries, cancellation, and audio/video sync through implemented engine boundaries. Keep decoding/encoding off the real-time device callback.
