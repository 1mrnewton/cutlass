//! AudioToolbox audio conversion.
//!
//! PCM is little-endian interleaved. Compressed packets are one access unit.
//! The encoder's magic cookie is `SideData::Config` on the first packet, and
//! the decoder wants that cookie in `CodecParams.extra` or on the packet.

use std::mem::size_of_val;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    AudioOutput, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
    SampleFormat, SideData,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet as Encoded, PictureInput};

use super::sys::{
    self, AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioStreamPacketDescription,
};
use crate::limits::{self, audio_ok};
use crate::status::open_status;

const COOKIE: u32 = u32::from_be_bytes(*b"cmgc");
const COOKIE_IN: u32 = u32::from_be_bytes(*b"dmgc");
const BITRATE: u32 = u32::from_be_bytes(*b"brat");
const PRIME: u32 = u32::from_be_bytes(*b"prim");

struct Converter {
    raw: *mut std::ffi::c_void,
}

impl Converter {
    fn new(
        source: &AudioStreamBasicDescription,
        destination: &AudioStreamBasicDescription,
    ) -> Result<Self, CodecError> {
        let mut raw = ptr::null_mut();
        let status = unsafe { sys::AudioConverterNew(source, destination, &mut raw) };
        open_status(status, "audiotoolbox")?;
        if raw.is_null() {
            return Err(CodecError::Unsupported {
                what: "audiotoolbox",
            });
        }
        Ok(Self { raw })
    }
}

impl Drop for Converter {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = sys::AudioConverterDispose(self.raw);
            }
        }
    }
}

unsafe impl Send for Converter {}

pub struct AudioDecoder {
    codec: CodecId,
    converter: Converter,
    rate: u32,
    layout: ChannelLayout,
    format: SampleFormat,
    primed: bool,
}

pub struct AudioEncoder {
    codec: CodecId,
    converter: Converter,
    rate: u32,
    layout: ChannelLayout,
    format: SampleFormat,
    frame: u32,
    pending: Vec<u8>,
    cookie: Vec<u8>,
    leading: u32,
    produced: bool,
    drained: bool,
    cancel: Arc<AtomicBool>,
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    let layout = layout_of(params)?;
    let format = params.sample_format.unwrap_or(SampleFormat::F32);
    let source = compressed_desc(params.codec, params.sample_rate, layout.len())?;
    let destination = pcm_desc(params.sample_rate, layout.len(), format)?;
    let converter = Converter::new(&source, &destination)?;
    if !params.extra.is_empty() {
        set_bytes(converter.raw, COOKIE_IN, &params.extra)?;
    }
    Ok(Box::new(AudioDecoder {
        codec: params.codec,
        converter,
        rate: params.sample_rate,
        layout,
        format,
        primed: false,
    }))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    let layout = layout_of(params)?;
    let format = params.sample_format.unwrap_or(SampleFormat::F32);
    let source = pcm_desc(params.sample_rate, layout.len(), format)?;
    let destination = compressed_desc(params.codec, params.sample_rate, layout.len())?;
    let converter = Converter::new(&source, &destination)?;
    if params.bitrate != 0 && params.codec != CodecId::Pcm {
        let rate = params.bitrate;
        unsafe {
            let _ =
                sys::AudioConverterSetProperty(converter.raw, BITRATE, 4, (&raw const rate).cast());
        }
    }
    let cookie = get_bytes(converter.raw, COOKIE).unwrap_or_default();
    let leading = prime_leading(converter.raw);
    let frame = if params.codec == CodecId::Pcm {
        1
    } else {
        destination.frames_per_packet.max(1)
    };
    Ok(Box::new(AudioEncoder {
        codec: params.codec,
        converter,
        rate: params.sample_rate,
        layout,
        format,
        frame,
        pending: Vec::new(),
        cookie,
        leading,
        produced: false,
        drained: false,
        cancel: Arc::new(AtomicBool::new(false)),
    }))
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

fn compressed_desc(
    codec: CodecId,
    rate: u32,
    channels: usize,
) -> Result<AudioStreamBasicDescription, CodecError> {
    let (format_id, frames) = match codec {
        CodecId::Aac => (u32::from_be_bytes(*b"aac "), 1024),
        CodecId::Alac => (u32::from_be_bytes(*b"alac"), 4096),
        CodecId::Flac => (u32::from_be_bytes(*b"flac"), 4096),
        CodecId::Mp3 => (u32::from_be_bytes(*b".mp3"), 1152),
        CodecId::Opus => (u32::from_be_bytes(*b"opus"), 960),
        CodecId::Ac3 => (u32::from_be_bytes(*b"ac-3"), 1536),
        CodecId::Pcm => return pcm_desc(rate, channels, SampleFormat::F32),
        CodecId::Vorbis => return Err(CodecError::Unsupported { what: "vorbis" }),
        _ => {
            return Err(CodecError::Unsupported {
                what: "audiotoolbox",
            });
        }
    };
    let channels = u32::try_from(channels).map_err(|_| CodecError::Limit { what: "channels" })?;
    Ok(AudioStreamBasicDescription {
        sample_rate: f64::from(rate),
        format_id,
        format_flags: 0,
        bytes_per_packet: 0,
        frames_per_packet: frames,
        bytes_per_frame: 0,
        channels,
        bits: 0,
        reserved: 0,
    })
}

fn pcm_desc(
    rate: u32,
    channels: usize,
    format: SampleFormat,
) -> Result<AudioStreamBasicDescription, CodecError> {
    let channels = u32::try_from(channels).map_err(|_| CodecError::Limit { what: "channels" })?;
    let (flags, bits) = match format {
        SampleFormat::F32 => (1u32 | 8, 32u32),
        SampleFormat::I16 => (4 | 8, 16),
    };
    let bytes = bits / 8 * channels;
    Ok(AudioStreamBasicDescription {
        sample_rate: f64::from(rate),
        format_id: u32::from_be_bytes(*b"lpcm"),
        format_flags: flags,
        bytes_per_packet: bytes,
        frames_per_packet: 1,
        bytes_per_frame: bytes,
        channels,
        bits,
        reserved: 0,
    })
}

fn set_bytes(
    converter: *mut std::ffi::c_void,
    property: u32,
    bytes: &[u8],
) -> Result<(), CodecError> {
    let size = u32::try_from(bytes.len()).map_err(|_| CodecError::Limit { what: "cookie" })?;
    let status =
        unsafe { sys::AudioConverterSetProperty(converter, property, size, bytes.as_ptr().cast()) };
    if status != 0 {
        return Err(CodecError::Unsupported {
            what: "audiotoolbox cookie",
        });
    }
    Ok(())
}

fn get_bytes(converter: *mut std::ffi::c_void, property: u32) -> Option<Vec<u8>> {
    let mut bytes = vec![0u8; 4096];
    let mut size = u32::try_from(bytes.len()).ok()?;
    let status = unsafe {
        sys::AudioConverterGetProperty(converter, property, &mut size, bytes.as_mut_ptr().cast())
    };
    if status != 0 || size == 0 {
        return None;
    }
    let size = usize::try_from(size).ok()?;
    if size > bytes.len() {
        return None;
    }
    bytes.truncate(size);
    Some(bytes)
}

fn prime_leading(converter: *mut std::ffi::c_void) -> u32 {
    let mut info = sys::AudioConverterPrimeInfo {
        leading: 0,
        trailing: 0,
    };
    let mut size = u32::try_from(size_of_val(&info)).unwrap_or(0);
    let status = unsafe {
        sys::AudioConverterGetProperty(converter, PRIME, &mut size, (&raw mut info).cast())
    };
    if status == 0 { info.leading } else { 0 }
}

struct Feed {
    data: *const u8,
    size: u32,
    channels: u32,
    frames: u32,
    bytes_per_frame: u32,
    packet: bool,
    desc: AudioStreamPacketDescription,
    used: bool,
}

unsafe extern "C" fn feed_input(
    _converter: *mut std::ffi::c_void,
    packets: *mut u32,
    data: *mut AudioBufferList,
    descriptions: *mut *mut AudioStreamPacketDescription,
    user: *mut std::ffi::c_void,
) -> sys::OSStatus {
    unsafe {
        if packets.is_null() || data.is_null() || user.is_null() {
            return -50;
        }
        let feed = &mut *user.cast::<Feed>();
        if feed.used || feed.size == 0 {
            *packets = 0;
            return 0;
        }
        if feed.packet {
            (*data).buffers = 1;
            (*data).buffer = AudioBuffer {
                channels: feed.channels,
                size: feed.size,
                data: feed.data.cast_mut().cast(),
            };
            feed.desc.size = feed.size;
            if !descriptions.is_null() {
                *descriptions = &raw mut feed.desc;
            }
            *packets = 1;
            feed.used = true;
            return 0;
        }
        let asked = *packets;
        if asked == 0 {
            return 0;
        }
        let give = asked.min(feed.frames);
        let bytes = give.saturating_mul(feed.bytes_per_frame);
        (*data).buffers = 1;
        (*data).buffer = AudioBuffer {
            channels: feed.channels,
            size: bytes,
            data: feed.data.cast_mut().cast(),
        };
        if !descriptions.is_null() {
            *descriptions = ptr::null_mut();
        }
        *packets = give;
        let advance = usize::try_from(bytes).unwrap_or(0);
        feed.data = feed.data.wrapping_add(advance);
        feed.frames = feed.frames.saturating_sub(give);
        feed.size = feed.size.saturating_sub(bytes);
        if feed.frames == 0 {
            feed.used = true;
        }
        0
    }
}

fn convert(
    converter: *mut std::ffi::c_void,
    feed: &mut Feed,
    out_packets: u32,
    out_bytes: usize,
    variable: bool,
) -> Result<Vec<u8>, CodecError> {
    let mut output = vec![0u8; out_bytes];
    let channels = feed.channels;
    let mut list = AudioBufferList {
        buffers: 1,
        buffer: AudioBuffer {
            channels,
            size: u32::try_from(output.len()).map_err(|_| CodecError::Limit { what: "pcm" })?,
            data: output.as_mut_ptr().cast(),
        },
    };
    let mut count = out_packets;
    let mut desc = AudioStreamPacketDescription {
        start: 0,
        frames: 0,
        size: 0,
    };
    let desc_ptr = if variable { &mut desc } else { ptr::null_mut() };
    let feed_ptr: *mut Feed = feed;
    let status = unsafe {
        sys::AudioConverterFillComplexBuffer(
            converter,
            feed_input,
            feed_ptr.cast(),
            &mut count,
            &mut list,
            desc_ptr,
        )
    };
    if status != 0 && status != 1 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "audiotoolbox",
        });
    }
    if variable && desc.size != 0 {
        let start = usize::try_from(desc.start).unwrap_or(0);
        let size = usize::try_from(desc.size).unwrap_or(0);
        if start.saturating_add(size) <= output.len() && size != 0 {
            return Ok(output[start..start + size].to_vec());
        }
    }
    let written = usize::try_from(list.buffer.size).unwrap_or(0);
    output.truncate(written.min(output.len()));
    Ok(output)
}

impl Decoder for AudioDecoder {
    fn codec(&self) -> CodecId {
        self.codec
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        limits::packet_ok(packet.data.len())?;
        if self.codec == CodecId::Pcm {
            let pcm = bytes_to_pcm(&packet.data, self.rate, self.layout.clone(), self.format)?;
            return Ok(vec![Output::Audio(AudioOutput {
                pcm,
                pts: packet.pts,
                timescale: if packet.timescale == 0 {
                    self.rate
                } else {
                    packet.timescale
                },
                skip_leading: 0,
                skip_trailing: 0,
                concealed: false,
            })]);
        }
        if !self.primed {
            if let Some(SideData::Config(cookie)) = packet
                .side
                .iter()
                .find(|side| matches!(side, SideData::Config(_)))
            {
                let _ = set_bytes(self.converter.raw, COOKIE_IN, cookie);
            }
            self.primed = true;
        }
        let channels = u32::try_from(self.layout.len()).unwrap_or(1);
        let bytes_per_frame = match self.format {
            SampleFormat::F32 => 4 * channels,
            SampleFormat::I16 => 2 * channels,
        };
        let mut feed = Feed {
            data: packet.data.as_ptr(),
            size: u32::try_from(packet.data.len()).map_err(|_| CodecError::Limit {
                what: "packet bytes",
            })?,
            channels,
            frames: 0,
            bytes_per_frame,
            packet: self.codec != CodecId::Pcm,
            desc: AudioStreamPacketDescription {
                start: 0,
                frames: 0,
                size: 0,
            },
            used: false,
        };
        let frames = if self.codec == CodecId::Pcm {
            packet.data.len() / usize::try_from(bytes_per_frame).unwrap_or(1)
        } else {
            1024
        };
        let out_packets = u32::try_from(frames).unwrap_or(1).max(1);
        let bytes = convert(
            self.converter.raw,
            &mut feed,
            out_packets,
            frames.saturating_mul(usize::try_from(bytes_per_frame).unwrap_or(1)),
            false,
        )?;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let pcm = bytes_to_pcm(&bytes, self.rate, self.layout.clone(), self.format)?;
        let skip = packet
            .side
            .iter()
            .find_map(|side| match side {
                SideData::Gapless {
                    encoder_delay,
                    sample_rate,
                    ..
                } if *sample_rate == self.rate => Some(*encoder_delay),
                _ => None,
            })
            .unwrap_or(0);
        Ok(vec![Output::Audio(AudioOutput {
            pcm,
            pts: packet.pts,
            timescale: if packet.timescale == 0 {
                self.rate
            } else {
                packet.timescale
            },
            skip_leading: skip,
            skip_trailing: 0,
            concealed: false,
        })])
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        unsafe {
            let _ = sys::AudioConverterReset(self.converter.raw);
        }
        self.primed = false;
    }
}

impl Encoder for AudioEncoder {
    fn codec(&self) -> CodecId {
        self.codec
    }

    fn push(&mut self, input: &AudioInput) -> Result<Vec<Encoded>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let bytes = pcm_bytes(&input.pcm, self.format)?;
        if input.pcm.rate != self.rate || input.pcm.layout != self.layout {
            return Err(CodecError::InvalidApi {
                what: "audio format",
            });
        }
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        self.pending.extend_from_slice(&bytes);
        self.pull(false)
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
        if self.drained {
            return Ok(Vec::new());
        }
        let channels = self.layout.len().max(1);
        let sample = match self.format {
            SampleFormat::F32 => 4 * channels,
            SampleFormat::I16 => 2 * channels,
        };
        let frame_bytes = sample.saturating_mul(usize::try_from(self.frame).unwrap_or(1));
        if frame_bytes != 0 && !self.pending.is_empty() {
            let extra = frame_bytes - (self.pending.len() % frame_bytes);
            if extra != frame_bytes {
                self.pending.extend(std::iter::repeat_n(0, extra));
            }
        }
        let packets = self.pull(true)?;
        self.drained = true;
        self.pending.clear();
        Ok(packets)
    }

    fn reset(&mut self) {
        unsafe {
            let _ = sys::AudioConverterReset(self.converter.raw);
        }
        self.pending.clear();
        self.produced = false;
        self.drained = false;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl AudioEncoder {
    fn pull(&mut self, flush: bool) -> Result<Vec<Encoded>, CodecError> {
        let channels = u32::try_from(self.layout.len()).unwrap_or(1);
        let bytes_per_frame = match self.format {
            SampleFormat::F32 => 4 * channels,
            SampleFormat::I16 => 2 * channels,
        };
        let frame_bytes = bytes_per_frame.saturating_mul(self.frame.max(1));
        let mut packets = Vec::new();
        while self.pending.len() >= usize::try_from(frame_bytes).unwrap_or(usize::MAX)
            || (flush && !self.pending.is_empty())
        {
            let take = usize::try_from(frame_bytes)
                .unwrap_or(self.pending.len())
                .min(self.pending.len());
            if take == 0 {
                break;
            }
            let chunk: Vec<u8> = self.pending.drain(..take).collect();
            let frames = u32::try_from(chunk.len() / usize::try_from(bytes_per_frame).unwrap_or(1))
                .unwrap_or(0);
            let mut feed = Feed {
                data: chunk.as_ptr(),
                size: u32::try_from(chunk.len()).unwrap_or(0),
                channels,
                frames,
                bytes_per_frame,
                packet: false,
                desc: AudioStreamPacketDescription {
                    start: 0,
                    frames: 0,
                    size: 0,
                },
                used: false,
            };
            let encoded = if self.codec == CodecId::Pcm {
                chunk
            } else {
                convert(self.converter.raw, &mut feed, 1, 16 * 1024, true)?
            };
            if encoded.is_empty() {
                break;
            }
            let mut packet = Encoded {
                stream: 0,
                data: encoded,
                pts: 0,
                duration: i64::from(frames),
                timescale: self.rate,
                keyframe: true,
                side: Vec::new(),
            };
            if !self.produced {
                if !self.cookie.is_empty() {
                    packet.side.push(SideData::Config(self.cookie.clone()));
                }
                if self.leading != 0 {
                    packet.side.push(SideData::Gapless {
                        encoder_delay: self.leading,
                        trailing_padding: 0,
                        sample_rate: self.rate,
                    });
                }
                self.produced = true;
            }
            packets.push(packet);
            if flush {
                break;
            }
        }
        Ok(packets)
    }
}

fn pcm_bytes(pcm: &Pcm, format: SampleFormat) -> Result<Vec<u8>, CodecError> {
    if pcm.format != format {
        return Err(CodecError::InvalidApi {
            what: "sample format",
        });
    }
    match format {
        SampleFormat::F32 => {
            let Some(samples) = pcm.as_f32() else {
                return Err(CodecError::InvalidApi {
                    what: "sample format",
                });
            };
            for sample in samples {
                if !sample.is_finite() {
                    return Err(CodecError::NonFinite);
                }
                if sample.abs() > 1.0 {
                    return Err(CodecError::OutOfRange);
                }
            }
            let mut bytes = Vec::with_capacity(samples.len() * 4);
            for sample in samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            Ok(bytes)
        }
        SampleFormat::I16 => {
            let Some(samples) = pcm.as_i16() else {
                return Err(CodecError::InvalidApi {
                    what: "sample format",
                });
            };
            let mut bytes = Vec::with_capacity(samples.len() * 2);
            for sample in samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            Ok(bytes)
        }
    }
}

fn bytes_to_pcm(
    bytes: &[u8],
    rate: u32,
    layout: ChannelLayout,
    format: SampleFormat,
) -> Result<Pcm, CodecError> {
    match format {
        SampleFormat::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "pcm",
                });
            }
            let mut samples = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.as_chunks::<4>().0 {
                let sample = f32::from_le_bytes(*chunk);
                if !sample.is_finite() {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "pcm",
                    });
                }
                samples.push(sample);
            }
            Pcm::from_f32(rate, layout, samples).map_err(|_| CodecError::InvalidApi { what: "pcm" })
        }
        SampleFormat::I16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "pcm",
                });
            }
            let mut samples = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.as_chunks::<2>().0 {
                samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
            Pcm::from_i16(rate, layout, samples).map_err(|_| CodecError::InvalidApi { what: "pcm" })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::ChannelLayout;

    fn audio_params(codec: CodecId, rate: u32) -> CodecParams {
        let mut params = CodecParams::audio(codec, rate, ChannelLayout::mono());
        params.bitrate = 64_000;
        params
    }

    #[test]
    fn pcm_round_trips_the_samples() {
        let samples = vec![0.0, 0.25, -0.5, 1.0];
        let mut encoder = open_encoder(&audio_params(CodecId::Pcm, 48_000)).expect("encode open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), samples.clone()).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("push");
        assert!(!packets.is_empty());
        let mut decoder = open_decoder(&audio_params(CodecId::Pcm, 48_000)).expect("decode open");
        let mut got = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).expect("decode") {
                let Output::Audio(audio) = output else {
                    panic!("audio")
                };
                got.extend_from_slice(audio.pcm.as_f32().expect("f32"));
            }
        }
        assert_eq!(got, samples);
    }

    #[test]
    fn a_non_finite_sample_adds_nothing() {
        let mut encoder = open_encoder(&audio_params(CodecId::Pcm, 48_000)).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![f32::NAN]).expect("pcm");
        assert_eq!(
            encoder.push(&AudioInput { pcm }),
            Err(CodecError::NonFinite)
        );
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.5]).expect("pcm");
        assert!(encoder.push(&AudioInput { pcm }).is_ok());
    }

    #[test]
    fn aac_round_trips_a_tone() {
        let rate = 48_000u32;
        let frames = 1024 * 4;
        let samples: Vec<f32> = (0..frames)
            .map(|index| (index as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
            .collect();
        let mut encoder = open_encoder(&audio_params(CodecId::Aac, rate)).expect("encode open");
        let pcm = Pcm::from_f32(rate, ChannelLayout::mono(), samples.clone()).expect("pcm");
        let mut packets = encoder.push(&AudioInput { pcm }).expect("encode");
        packets.extend(encoder.drain().expect("drain"));
        assert!(!packets.is_empty(), "aac produced no packets");
        let mut decoder = open_decoder(&audio_params(CodecId::Aac, rate)).expect("decode open");
        let mut got = Vec::new();
        for packet in &packets {
            for output in decoder.push(packet).expect("decode") {
                let Output::Audio(audio) = output else {
                    panic!("audio");
                };
                got.extend_from_slice(audio.pcm.as_f32().expect("f32"));
            }
        }
        let window = 1024;
        assert!(
            got.len() >= window,
            "decoded {} samples from {} packets",
            got.len(),
            packets.len()
        );
        let mut best = f64::NEG_INFINITY;
        let mut best_lag = 0usize;
        let last = got.len() - window;
        for lag in 0..=last {
            let mut error = 0.0f64;
            let mut energy = 0.0f64;
            for index in 0..window {
                let delta = f64::from(got[lag + index] - samples[index]);
                error += delta * delta;
                energy += f64::from(samples[index]) * f64::from(samples[index]);
            }
            let snr = 10.0 * (energy / error.max(1e-12)).log10();
            if snr > best {
                best = snr;
                best_lag = lag;
            }
        }
        assert!(
            best > 20.0,
            "best snr {best} at lag {best_lag}, decoded {}",
            got.len()
        );
    }

    #[test]
    fn an_empty_push_adds_no_packets() {
        let mut encoder = open_encoder(&audio_params(CodecId::Pcm, 48_000)).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), Vec::new()).expect("pcm");
        assert!(encoder.push(&AudioInput { pcm }).expect("push").is_empty());
    }

    #[test]
    fn a_sample_past_full_scale_adds_nothing() {
        let mut encoder = open_encoder(&audio_params(CodecId::Pcm, 48_000)).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![1.1]).expect("pcm");
        assert_eq!(
            encoder.push(&AudioInput { pcm }),
            Err(CodecError::OutOfRange)
        );
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.25]).expect("pcm");
        assert!(encoder.push(&AudioInput { pcm }).is_ok());
    }

    #[test]
    fn a_cancelled_encoder_accepts_nothing() {
        let mut encoder = open_encoder(&audio_params(CodecId::Pcm, 48_000)).expect("open");
        encoder
            .cancel_flag()
            .store(true, Ordering::Relaxed);
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.25]).expect("pcm");
        assert_eq!(
            encoder.push(&AudioInput { pcm }),
            Err(CodecError::Cancelled)
        );
    }

    #[test]
    fn the_wrong_sample_rate_leaves_the_buffer_empty() {
        let mut encoder = open_encoder(&audio_params(CodecId::Pcm, 48_000)).expect("open");
        let pcm = Pcm::from_f32(44_100, ChannelLayout::mono(), vec![0.25]).expect("pcm");
        assert_eq!(
            encoder.push(&AudioInput { pcm }),
            Err(CodecError::InvalidApi {
                what: "audio format"
            })
        );
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.25]).expect("pcm");
        let packets = encoder.push(&AudioInput { pcm }).expect("push");
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].data.len(), 4);
    }

    #[test]
    fn reset_drops_a_partial_aac_frame() {
        let mut encoder = open_encoder(&audio_params(CodecId::Aac, 48_000)).expect("open");
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.2; 100]).expect("pcm");
        assert!(encoder.push(&AudioInput { pcm }).expect("partial").is_empty());
        encoder.reset();
        let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.2; 100]).expect("pcm");
        assert!(encoder.push(&AudioInput { pcm }).expect("again").is_empty());
        let drained = encoder.drain().expect("drain");
        assert!(!drained.is_empty());
        assert!(encoder.drain().expect("second").is_empty());
    }

    #[test]
    fn vorbis_is_declined() {
        let err = match open_encoder(&audio_params(CodecId::Vorbis, 48_000)) {
            Err(err) => err,
            Ok(_) => panic!("vorbis"),
        };
        assert_eq!(err, CodecError::Unsupported { what: "vorbis" });
    }
}
