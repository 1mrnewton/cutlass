# cutlass-aac

## What this crate is

`cutlass-aac` is a native safe Rust AAC-LC encoder and decoder. The crate forbids `unsafe` and does not link FFmpeg or any other AAC library. FFmpeg is an external oracle used by the tests and by `benches/against_ffmpeg.rs`.

`AacEncoder` and `AacDecoder` read and write raw AAC access units. ADTS and LOAS/LATM framing live in `cutlass-adts`. The longer support record is `STATUS.md`.

## Specification

The comments in this crate and `STATUS.md` name these documents:

- ISO/IEC 14496-3:2009, subpart 4, for AAC-LC syntax, the filterbank, and the tools the decoder reads.
- ISO/IEC 13818-7 for MPEG-2 AAC and for which bits the ADTS CRC covers. A packet that carries `SideData::Protection` is checked with that CRC.
- ISO/IEC 11172-3 for the CRC-16 polynomial `x^16 + x^15 + x^2 + 1`, starting at `0xFFFF`.
- 3GPP TS 26.404 and TS 26.405 describe spectral band replication and parametric stereo. Those tools are rejected.

`STATUS.md` also names ISO/IEC 14496-3 subclause 1.7 for one-program LOAS/LATM, and ISO/IEC 14496-1, 14496-12, and 14496-14 for the audio movie boxes. Those formats are outside this crate.

Indexed rates are the 13 MPEG-4 rates from 96000 Hz down to 7350 Hz. MPEG-2 AAC-LC stops at 8000 Hz, uses a 1024-sample frame, and has no spectral band replication. An explicit 24-bit rate is parsed and then rejected.

## API

Float PCM full scale is 1.0. Integer PCM uses the 16-bit full scale. `AacEncoder::open` reads `CodecParams` and emits 1024-sample AAC-LC frames. A `bitrate` of 0 selects 64 kbps per channel. `AacEncoder::from_config` can select a 960-sample frame, MPEG-2 limits, an average bitrate, or variable quality from 1 to 10. `AacDecoder::open` reads `CodecParams.extra` as an `AudioSpecificConfig`. An empty `extra` builds an AAC-LC configuration from the sample rate and layout. `push_packet` returns one interleaved `f32` frame. `drain_pcm` flushes the overlap tail. `config` returns the `AudioConfig` (sample rate, layout, frame length).

`samples` is interleaved `f32` at full scale 1.0. `open`, `push`, `drain`, and `push_packet` return `CodecError`. `Pcm::from_f32` returns `PcmError`, so a caller maps that error before using `?` on the codec calls.

```rust
use cutlass_aac::{AacDecoder, AacEncoder, EncoderConfig, Profile, RateControl};
use cutlass_decode::{ChannelLayout, CodecId, CodecParams};
use cutlass_encode::{AudioInput, Encoder};

let mut params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
params.bitrate = 96_000;

let mut encoder = AacEncoder::open(&params)?;
let pcm = cutlass_decode::Pcm::from_f32(48_000, ChannelLayout::mono(), samples)?;
let mut packets = encoder.push(&AudioInput { pcm })?;
packets.extend(encoder.drain()?);

let mut decoder = AacDecoder::open(&params)?;
for packet in &packets {
    let _pcm = decoder.push_packet(packet)?;
}
let _tail = decoder.drain_pcm()?;

let _variable = AacEncoder::from_config(EncoderConfig {
    profile: Profile::AacLc,
    mpeg2: false,
    sample_rate: 48_000,
    layout: ChannelLayout::mono(),
    frame_length: 1024,
    rate: RateControl::Variable {
        quality: 6,
        cap: None,
    },
})?;
```

`register_decoders` and `register_encoders` register this crate on a `DecodeFactory` and an `EncodeFactory`.

## Support

Raw AAC-LC access units are implemented for encode and decode. Frame length is 1024 or 960. Sample-rate indexes 0 through 12 are accepted. Channel configurations 1, 2, 3, 4, 5, 6, 7, 11, and 12 are accepted, in AAC element order: mono, stereo, the 3-, 4-, and 5-channel layouts, 5.1 (configuration 6), wide 7.1 (configuration 7), configuration 11, and 7.1 (configuration 12). Configuration 13 and more than 8 channels are rejected.

The decoder reads SCE, CPE, LFE, DSE, FIL, and END. It applies Huffman (including escapes), inverse quantization, sine and KBD windows, the MDCT, overlap-add, pulse, TNS, PNS, mid/side, and intensity. DRC in a fill element is skipped and not applied. Coupling, an in-band program config, prediction, gain control, a core coder, `extensionFlag3`, and spectral band replication are errors. A failed frame leaves the overlap and the sample clock unchanged. An empty packet, and an access unit larger than 8192 bytes, are errors.

The encoder uses KBD windows, one frame of lookahead, and one block-switch decision for the non-LFE channels. It uses mid/side when the two channels of a pair are strongly correlated. Average mode aims at a mean payload rate and does not pad access units to one size. Variable mode takes a quality from 1 to 10 and an optional bitrate cap. The bit reservoir is at most 6144 bits per channel. Leading delay is one frame and is carried on `SideData::Gapless`. The encoder does not emit prediction, coupling, PNS, intensity, TNS, or pulse.

MPEG-2 encode and decode reject a 960-sample frame, the 7350 Hz index, and an explicit rate. HE-AAC v1, HE-AAC v2, Main, SSR, LTP, ER AAC, LD, ELD, USAC/xHE, and ADIF are unsupported. An SBR or parametric-stereo configuration returns the error `sbr` rather than a lowband-only decode. `push_frame` rejects a picture.

ADTS, LOAS/LATM, and M4A are not parsed here.

## Accuracy

On the one bench sample, decode of ffmpeg's AAC-LC stream matches ffmpeg's decoder with a maximum absolute sample error of `1.48953870e-3` on the overlap.

```text
decode compare: agree, ours 146432 samples, ffmpeg 145408 samples, lag 0 samples, snr 76.015162 dB, max abs error 1.48953870e-3, compared 145408
```

The bench treats that overlap as agreement when SNR is at least 40 dB and the two lengths differ by at most 4096 samples. This run's SNR is 76.015162 dB at lag 0. The lengths differ by 1024 samples, and the shared prefix of 145408 samples was compared. ffmpeg also decoded a bitstream from this encoder (145408 samples).

The tests in this crate keep round-trip floors. They fail below the floor and do not record a tighter score: mono 48 kHz, 1 kHz, half scale, 96 kbps, 1024 or 960 samples, SNR at least 40 dB after delay trim; stereo 128 kbps at least 35 dB per channel; near Nyquist, stereo 128 kbps, at least 20 dB; silence at 128 kbps below `1e-4`; DC 0.25 with mean error below 0.02; 5.1 at 384 kbps, full-band at least 30 dB and LFE at least 25 dB; 7.1 at 512 kbps, at least 25 dB per channel; 16-bit sine at least 40 dB. `ffmpeg_decodes_our_access_units` and `we_decode_an_ffmpeg_stream` require SNR of at least 40 dB and a lag of at most 1024 samples when ffmpeg is present.

## Performance

One sample: 48000 Hz, 1 channel, 144000 samples, 1000 Hz sine, AAC-LC at 96000 bps. Median of 3 after 1 warmup for decode and for encode.

Command: `cargo bench -p cutlass-aac`

ffmpeg: `ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers`

`sysctl -n machdep.cpu.brand_string`: `Apple M5 Pro`

Date: `Thu Oct 29 21:07:12 GMT 2026`

```text
sample: 48000 Hz, 1 channel, 144000 samples, 1000 Hz sine, AAC-LC 96000 bps
decode gate: snr at least 40 dB, length difference at most 4096 samples
decode compare: agree, ours 146432 samples, ffmpeg 145408 samples, lag 0 samples, snr 76.015162 dB, max abs error 1.48953870e-3, compared 145408
decode: median of 3 after 1 warmup
cutlass 0.004806 s
ffmpeg 0.020315 s
encode: median of 3 after 1 warmup
cutlass 0.005373 s
ffmpeg 0.047526 s
ffmpeg decoded the cutlass bitstream: 145408 samples
cutlass decodes packets already in memory. ffmpeg's time starts at process launch.
```

ffmpeg's decode and encode times include process startup.

## Example

`examples/roundtrip.rs` asks ffmpeg for 0.1 seconds of a 1 kHz sine at 48 kHz, AAC-LC at 96 kbps, in an ADTS file, then decodes the raw access units.

```text
cargo run --release -p cutlass-aac --example roundtrip
```

Output from that command on this machine:

```text
sample rate: 48000
channels: 1
samples: 7168
```

`samples` is the decoded PCM frame count, including delay and the flushed tail. If ffmpeg is absent, the example prints `skip: ffmpeg is not installed` and exits 0.

## Gaps

- Spectral band replication and parametric stereo are unimplemented. Object types 5 and 29 return an error.
- Main, SSR, LTP, ER AAC, LD, ELD, and USAC/xHE are unimplemented.
- ADIF is unimplemented.
- The encoder does not emit TNS, PNS, intensity, or pulse.
- ISO/IEC 14496-26 conformance vectors were not obtained.
- `thirty_minute_decoder_campaign` is `#[ignore]`. The short random-input test is `random_access_units_fail_closed`.
- The numbers above are one 3-second sine. A memory bound was not measured. Linux and Windows were not run.
- `STATUS.md` is the support record. This crate does not frame ADTS, LOAS/LATM, or M4A.
