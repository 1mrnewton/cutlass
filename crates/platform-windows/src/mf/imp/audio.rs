//! Media Foundation AAC decode and encode.
//!
//! The in-box transforms are `CMSAACDecMFT` and `AACMFTEncoder`. Packets are
//! raw AAC access units, one per packet, and the configuration is the
//! AudioSpecificConfig, which is what the Rust AAC codec writes and reads.
//! An `esds` from an MP4 is accepted too; the AudioSpecificConfig inside it
//! is what the decoder is handed.
//!
//! The decoder gives `f32` or `i16` interleaved PCM, whichever the caller
//! asked for. HE-AAC decodes at its full output rate. Like the Rust AAC
//! decoder it keeps every sample and leaves trimming to the edit list, so
//! `skip_leading` stays zero.
//!
//! The encoder takes 16-bit PCM at 44.1 or 48 kHz in mono, stereo, or 5.1.
//! Any other rate or layout is [`CodecError::Unsupported`], so the Rust AAC
//! encoder takes it. Its bitrate is one of 96, 128, 160, or 192 kbit/s, the
//! nearest to the one asked for. Each packet is 1024 samples, clocked from
//! the samples accepted, and the first carries the AudioSpecificConfig. The
//! Microsoft encoder adds no priming of its own: its first decoded sample is
//! the first sample it was given, so there is no [`SideData::Gapless`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    AudioOutput, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
    SampleFormat, SideData,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet as Encoded, PictureInput};

use super::com::{self, Com, Hresult};
use super::transform::Transform;
use super::video::{from_hns, to_hns};
use crate::format::Guid;
use crate::limits::{self, audio_ok};

const CLSID_AAC_DECODER: Guid = Guid {
    data1: 0x32D1_86A7,
    data2: 0x218F,
    data3: 0x4C75,
    data4: [0x88, 0x76, 0xDD, 0x77, 0x27, 0x3A, 0x89, 0x99],
};
const CLSID_AAC_ENCODER: Guid = Guid {
    data1: 0x93AF_0C51,
    data2: 0x2275,
    data3: 0x45D2,
    data4: [0xA3, 0x5B, 0xF2, 0xBA, 0x21, 0xCA, 0xED, 0x00],
};

/// `WAVE_FORMAT_PCM` on the Media Foundation base GUID.
const PCM: Guid = Guid {
    data1: 0x0000_0001,
    ..com::MF_AUDIO_FORMAT_FLOAT
};

/// Samples in one AAC-LC access unit.
const FRAME: u32 = 1024;

/// The bytes a second `AACMFTEncoder` accepts: 96, 128, 160, 192 kbit/s.
const BYTE_RATES: [u32; 4] = [12_000, 16_000, 20_000, 24_000];

pub struct AudioDecoder {
    transform: Transform,
    layout: ChannelLayout,
    format: SampleFormat,
    /// What the transform writes, which may differ from `format`.
    float_out: bool,
    rate: u32,
    timescale: u32,
}

pub struct AudioEncoder {
    transform: Transform,
    rate: u32,
    layout: ChannelLayout,
    format: SampleFormat,
    byte_rate: u32,
    config: Vec<u8>,
    /// Samples accepted, for the input clock.
    accepted: u64,
    /// Packets emitted, for the output clock.
    emitted: u64,
    drained: bool,
    cancel: Arc<AtomicBool>,
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    aac_only(params.codec)?;
    let layout = layout_of(params)?;
    let format = params.sample_format.unwrap_or(SampleFormat::F32);
    com::enter()?;
    let config = match specific_config(&params.extra) {
        Some(config) => config.to_vec(),
        None => low_complexity_config(params.sample_rate, layout.len())?,
    };
    let mft = com::transform(&CLSID_AAC_DECODER).map_err(|_| decline())?;
    let mut transform = Transform::new(mft, false, None).map_err(|_| decline())?;
    let input = aac_type(params.sample_rate, layout.len(), &config, None).map_err(|_| decline())?;
    com::set_input_type(transform.mft(), &input).map_err(|_| decline())?;
    let float_out =
        choose_pcm(transform.mft(), format == SampleFormat::F32).map_err(|_| decline())?;
    let rate = com::output_current_type(transform.mft())
        .ok()
        .and_then(|current| com::get_u32(&current, &com::MF_MT_AUDIO_SAMPLES_PER_SECOND))
        .unwrap_or(params.sample_rate);
    transform.start().map_err(|_| decline())?;
    Ok(Box::new(AudioDecoder {
        transform,
        layout,
        format,
        float_out,
        rate,
        timescale: params.sample_rate,
    }))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    aac_only(params.codec)?;
    let layout = layout_of(params)?;
    if !matches!(params.sample_rate, 44_100 | 48_000) || !matches!(layout.len(), 1 | 2 | 6) {
        return Err(CodecError::Unsupported {
            what: "media foundation aac format",
        });
    }
    com::enter()?;
    let byte_rate = byte_rate(params.bitrate);
    let (transform, config) = build_encoder(params.sample_rate, layout.len(), byte_rate)?;
    Ok(Box::new(AudioEncoder {
        transform,
        rate: params.sample_rate,
        format: params.sample_format.unwrap_or(SampleFormat::F32),
        layout,
        byte_rate,
        config,
        accepted: 0,
        emitted: 0,
        drained: false,
        cancel: Arc::new(AtomicBool::new(false)),
    }))
}

impl Decoder for AudioDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Aac
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        limits::packet_ok(packet.data.len())?;
        com::enter()?;
        if packet.timescale != 0 {
            self.timescale = packet.timescale;
        }
        let sample = com::sample_from(
            &packet.data,
            to_hns(packet.pts, self.timescale),
            to_hns(packet.duration, self.timescale),
        )
        .map_err(|_| malformed())?;
        let samples = self
            .transform
            .feed(&sample, &mut no_change)
            .map_err(|_| malformed())?;
        self.outputs(&samples)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        com::enter()?;
        let samples = self
            .transform
            .drain(&mut no_change)
            .map_err(|_| malformed())?;
        self.outputs(&samples)
    }

    fn reset(&mut self) {
        if com::enter().is_ok() {
            self.transform.flush();
        }
    }
}

impl AudioDecoder {
    fn outputs(&self, samples: &[Com]) -> Result<Vec<Output>, CodecError> {
        let mut out = Vec::with_capacity(samples.len());
        for sample in samples {
            let bytes = com::sample_bytes(sample).map_err(|_| malformed())?;
            if bytes.is_empty() {
                continue;
            }
            let pcm = self.pcm(&bytes)?;
            let pts = com::sample_time(sample).map_or(0, |time| from_hns(time, self.timescale));
            out.push(Output::Audio(AudioOutput {
                pcm,
                skip_leading: 0,
                pts,
                timescale: self.timescale,
                skip_trailing: 0,
                concealed: false,
            }));
        }
        Ok(out)
    }

    fn pcm(&self, bytes: &[u8]) -> Result<Pcm, CodecError> {
        let floats: Vec<f32> = if self.float_out {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        } else {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f32::from(i16::from_le_bytes(*chunk)) / 32_768.0)
                .collect()
        };
        let pcm = match self.format {
            SampleFormat::F32 => {
                if floats.iter().any(|sample| !sample.is_finite()) {
                    return Err(malformed());
                }
                Pcm::from_f32(self.rate, self.layout.clone(), floats)
            }
            SampleFormat::I16 => Pcm::from_i16(
                self.rate,
                self.layout.clone(),
                floats.into_iter().map(to_i16).collect(),
            ),
        };
        pcm.map_err(|_| malformed())
    }
}

impl Encoder for AudioEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Aac
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Encoded>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let pcm = &input.pcm;
        if pcm.rate != self.rate || pcm.layout != self.layout {
            return Err(CodecError::InvalidApi {
                what: "audio format",
            });
        }
        let samples = pcm_i16(pcm, self.format)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        com::enter()?;
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let frames = u64::from(pcm.samples);
        let time = to_hns(i64::try_from(self.accepted).unwrap_or(i64::MAX), self.rate);
        let duration = to_hns(i64::try_from(frames).unwrap_or(i64::MAX), self.rate);
        let sample = com::sample_from(&bytes, time, duration).map_err(|_| refused())?;
        let out = self
            .transform
            .feed(&sample, &mut no_change)
            .map_err(|_| refused())?;
        self.accepted += frames;
        self.drained = false;
        self.packets(&out)
    }

    fn push_picture(&mut self, _input: &PictureInput) -> Result<Vec<Encoded>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Encoded>, CodecError> {
        Err(CodecError::Unsupported { what: "picture" })
    }

    fn drain(&mut self) -> Result<Vec<Encoded>, CodecError> {
        if self.drained || self.accepted == 0 {
            self.drained = true;
            return Ok(Vec::new());
        }
        com::enter()?;
        let out = self
            .transform
            .drain(&mut no_change)
            .map_err(|_| refused())?;
        self.drained = true;
        self.packets(&out)
    }

    /// A new transform, so the priming and the configuration come again.
    fn reset(&mut self) {
        self.accepted = 0;
        self.emitted = 0;
        self.drained = false;
        if com::enter().is_err() {
            return;
        }
        if let Ok((transform, config)) = build_encoder(self.rate, self.layout.len(), self.byte_rate)
        {
            self.transform = transform;
            self.config = config;
        }
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl AudioEncoder {
    fn packets(&mut self, samples: &[Com]) -> Result<Vec<Encoded>, CodecError> {
        let mut packets = Vec::with_capacity(samples.len());
        for sample in samples {
            let data = com::sample_bytes(sample).map_err(|_| refused())?;
            if data.is_empty() {
                continue;
            }
            let pts = self.emitted.saturating_mul(u64::from(FRAME));
            let mut packet = Encoded {
                stream: 0,
                data,
                pts: i64::try_from(pts).unwrap_or(i64::MAX),
                duration: i64::from(FRAME),
                timescale: self.rate,
                keyframe: true,
                side: Vec::new(),
            };
            if self.emitted == 0 {
                packet.side.push(SideData::Config(self.config.clone()));
            }
            self.emitted += 1;
            packets.push(packet);
        }
        Ok(packets)
    }
}

fn build_encoder(
    rate: u32,
    channels: usize,
    byte_rate: u32,
) -> Result<(Transform, Vec<u8>), CodecError> {
    let mft = com::transform(&CLSID_AAC_ENCODER).map_err(|_| decline())?;
    let mut transform = Transform::new(mft, false, None).map_err(|_| decline())?;
    let channels = u32::try_from(channels).map_err(|_| decline())?;
    let input = pcm_type(rate, channels).map_err(|_| decline())?;
    com::set_input_type(transform.mft(), &input).map_err(|_| decline())?;
    let output = com::media_type().map_err(|_| decline())?;
    com::set_guid(&output, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_AUDIO)
        .and_then(|()| com::set_guid(&output, &com::MF_MT_SUBTYPE, &com::MF_AUDIO_FORMAT_AAC))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_AUDIO_BITS_PER_SAMPLE, 16))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_AUDIO_SAMPLES_PER_SECOND, rate))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_AUDIO_NUM_CHANNELS, channels))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_AUDIO_AVG_BYTES_PER_SECOND, byte_rate))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_AAC_PAYLOAD_TYPE, 0))
        .and_then(|()| {
            com::set_u32(
                &output,
                &com::MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION,
                0x29,
            )
        })
        .and_then(|()| com::set_output_type(transform.mft(), &output))
        .map_err(|_| decline())?;
    let config = com::output_current_type(transform.mft())
        .ok()
        .and_then(|current| com::get_blob_vec(&current, &com::MF_MT_USER_DATA))
        .and_then(|info| info.get(WAVE_INFO..).map(<[u8]>::to_vec))
        .filter(|config| config.len() >= 2)
        .ok_or_else(decline)?;
    transform.start().map_err(|_| decline())?;
    Ok((transform, config))
}

/// Bytes of `HEAACWAVEINFO` after its `WAVEFORMATEX`, ahead of the
/// AudioSpecificConfig in `MF_MT_USER_DATA`.
const WAVE_INFO: usize = 12;

fn aac_type(
    rate: u32,
    channels: usize,
    config: &[u8],
    bitrate: Option<u32>,
) -> Result<Com, Hresult> {
    let channels = u32::try_from(channels).map_err(|_| com::MF_E_NOTACCEPTING)?;
    let kind = com::media_type()?;
    com::set_guid(&kind, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_AUDIO)?;
    com::set_guid(&kind, &com::MF_MT_SUBTYPE, &com::MF_AUDIO_FORMAT_AAC)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_SAMPLES_PER_SECOND, rate)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_NUM_CHANNELS, channels)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_BITS_PER_SAMPLE, 16)?;
    com::set_u32(&kind, &com::MF_MT_AAC_PAYLOAD_TYPE, 0)?;
    com::set_u32(&kind, &com::MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, 0xFE)?;
    if let Some(bitrate) = bitrate {
        com::set_u32(&kind, &com::MF_MT_AUDIO_AVG_BYTES_PER_SECOND, bitrate / 8)?;
    }
    // wPayloadType, wAudioProfileLevelIndication, wStructType, and two
    // reserved fields, then the AudioSpecificConfig.
    let mut info = vec![0_u8; WAVE_INFO];
    info[2..4].copy_from_slice(&0xFE_u16.to_le_bytes());
    info.extend_from_slice(config);
    com::set_blob(&kind, &com::MF_MT_USER_DATA, &info)?;
    Ok(kind)
}

fn pcm_type(rate: u32, channels: u32) -> Result<Com, Hresult> {
    let block = 2 * channels;
    let kind = com::media_type()?;
    com::set_guid(&kind, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_AUDIO)?;
    com::set_guid(&kind, &com::MF_MT_SUBTYPE, &PCM)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_BITS_PER_SAMPLE, 16)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_SAMPLES_PER_SECOND, rate)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_NUM_CHANNELS, channels)?;
    com::set_u32(&kind, &com::MF_MT_AUDIO_BLOCK_ALIGNMENT, block)?;
    com::set_u32(
        &kind,
        &com::MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
        rate.saturating_mul(block),
    )?;
    Ok(kind)
}

/// Take `f32` output when the decoder offers it and the caller wants it,
/// else 16-bit PCM. Returns whether the output is `f32`.
fn choose_pcm(mft: &Com, want_float: bool) -> Result<bool, Hresult> {
    let mut index = 0;
    let mut pcm = None;
    while let Some(kind) = com::output_available_type(mft, index)? {
        match com::get_guid(&kind, &com::MF_MT_SUBTYPE) {
            Some(subtype) if subtype == com::MF_AUDIO_FORMAT_FLOAT && want_float => {
                com::set_output_type(mft, &kind)?;
                return Ok(true);
            }
            Some(subtype)
                if subtype == PCM
                    && com::get_u32(&kind, &com::MF_MT_AUDIO_BITS_PER_SAMPLE) == Some(16) =>
            {
                pcm.get_or_insert(kind);
            }
            _ => {}
        }
        index += 1;
    }
    let kind = pcm.ok_or(com::MF_E_NO_MORE_TYPES)?;
    com::set_output_type(mft, &kind)?;
    Ok(false)
}

/// The AudioSpecificConfig, from itself or from an `esds` ES descriptor.
fn specific_config(extra: &[u8]) -> Option<&[u8]> {
    if extra.first() != Some(&0x03) {
        return (!extra.is_empty()).then_some(extra);
    }
    let mut at = 0;
    while at < extra.len() {
        let tag = extra[at];
        at += 1;
        let mut length = 0_usize;
        for _ in 0..4 {
            let byte = *extra.get(at)?;
            at += 1;
            length = (length << 7) | usize::from(byte & 0x7f);
            if byte & 0x80 == 0 {
                break;
            }
        }
        match tag {
            // ES descriptor: id and flags, then optional fields.
            0x03 => {
                let flags = *extra.get(at + 2)?;
                at += 3;
                if flags & 0x80 != 0 {
                    at += 2;
                }
                if flags & 0x40 != 0 {
                    at += 1 + usize::from(*extra.get(at)?);
                }
                if flags & 0x20 != 0 {
                    at += 2;
                }
            }
            // Decoder config: thirteen fixed bytes before what it holds.
            0x04 => at += 13,
            0x05 => return extra.get(at..at.checked_add(length)?),
            _ => at = at.checked_add(length)?,
        }
    }
    None
}

/// A two-byte AAC-LC AudioSpecificConfig for a stream that named none.
fn low_complexity_config(rate: u32, channels: usize) -> Result<Vec<u8>, CodecError> {
    const RATES: [u32; 12] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000,
    ];
    let index = RATES
        .iter()
        .position(|candidate| *candidate == rate)
        .ok_or(CodecError::Unsupported {
            what: "aac sample rate",
        })?;
    let channels = match channels {
        1..=6 => channels,
        8 => 7,
        _ => {
            return Err(CodecError::Unsupported {
                what: "aac channels",
            });
        }
    };
    let bits = (2_u16 << 11) | ((index as u16) << 7) | ((channels as u16) << 3);
    Ok(bits.to_be_bytes().to_vec())
}

/// Interleaved 16-bit samples, after the same checks the Rust encoder makes.
fn pcm_i16(pcm: &Pcm, format: SampleFormat) -> Result<Vec<i16>, CodecError> {
    if pcm.format != format {
        return Err(CodecError::InvalidApi {
            what: "sample format",
        });
    }
    match format {
        SampleFormat::F32 => {
            let samples = pcm.as_f32().ok_or(CodecError::InvalidApi {
                what: "sample format",
            })?;
            if samples.iter().any(|sample| !sample.is_finite()) {
                return Err(CodecError::NonFinite);
            }
            if samples.iter().any(|sample| sample.abs() > 1.0) {
                return Err(CodecError::OutOfRange);
            }
            Ok(samples.iter().copied().map(to_i16).collect())
        }
        SampleFormat::I16 => pcm
            .as_i16()
            .map(<[i16]>::to_vec)
            .ok_or(CodecError::InvalidApi {
                what: "sample format",
            }),
    }
}

fn to_i16(sample: f32) -> i16 {
    (sample * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16
}

/// The nearest byte rate the encoder takes. 0 asks for 128 kbit/s.
fn byte_rate(bitrate: u32) -> u32 {
    let want = if bitrate == 0 { 16_000 } else { bitrate / 8 };
    BYTE_RATES
        .into_iter()
        .min_by_key(|rate| rate.abs_diff(want))
        .unwrap_or(16_000)
}

fn layout_of(params: &CodecParams) -> Result<ChannelLayout, CodecError> {
    let Some(layout) = params.layout.clone() else {
        return Err(CodecError::Unsupported {
            what: "audio layout",
        });
    };
    audio_ok(params.sample_rate, layout.len())?;
    Ok(layout)
}

fn aac_only(codec: CodecId) -> Result<(), CodecError> {
    if codec == CodecId::Aac {
        Ok(())
    } else {
        Err(CodecError::Unsupported {
            what: "media foundation audio codec",
        })
    }
}

/// An audio transform does not change its output type mid-stream here.
fn no_change(_mft: &Com) -> Result<(), Hresult> {
    Ok(())
}

fn decline() -> CodecError {
    CodecError::Unsupported {
        what: "media foundation aac",
    }
}

fn malformed() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "media foundation aac",
    }
}

fn refused() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "media foundation aac encode",
    }
}
