//! Media Foundation H.264 decode and encode.
//!
//! Both sides use the in-box Microsoft transforms in their synchronous,
//! software form: `CMSH264DecoderMFT` and `CMSH264EncoderMFT`. Neither is
//! given a Direct3D device, so pictures cross as NV12 in system memory.
//! A decoded picture is CPU RGBA, or planar 4:2:0 when the caller asks for
//! [`PixelFormat::Yuv420p`]. The encoder takes either of those.
//!
//! The decoder wants Annex B. A length-prefixed unit is rewritten with
//! start codes, and the `avcC` parameter sets go in front of the first unit
//! after open, a new configuration, or a reset. The transform reorders
//! internally, so pictures leave in display order. `drain` asks it for
//! every picture it still holds.
//!
//! The encoder writes Annex B with the parameter sets in band. Each unit
//! leaves as 4-byte length-prefixed NAL units without those sets, and the
//! first keyframe carries the `avcC` as [`SideData::Config`], which is what
//! the VideoToolbox encoder emits too. The encoder asks for Baseline, so
//! there are no B-frames and units leave in input order.
//!
//! A transform this Windows cannot create, such as on an N edition without
//! the media feature pack, is [`CodecError::Unsupported`], so the factory
//! moves on to the Rust codec.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, SideData, Transfer,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, FrameTime, PictureInput};

use super::com::{self, Com, Hresult};
use crate::limits::{self, even_picture};
use crate::nal::{self, AvcConfig, Nal};
use crate::nv12::{self, Layout};

/// Media Foundation time is in 100-nanosecond units.
const HNS: i128 = 10_000_000;

/// Output samples one call may produce before the loop gives up on a
/// transform that never asks for more input.
const MAX_OUTPUTS: usize = 64;

pub struct VideoDecoder {
    width: u32,
    height: u32,
    planar: bool,
    /// 0 when units arrive as Annex B.
    length_size: u8,
    /// SPS and PPS from the `avcC`, with no start codes.
    sets: Vec<Vec<u8>>,
    send_sets: bool,
    transform: Com,
    output: OutputSlot,
    layout: Option<Layout>,
    timescale: u32,
}

pub struct VideoEncoder {
    width: u32,
    height: u32,
    bitrate: u32,
    rate: (u32, u32),
    transform: Com,
    output: OutputSlot,
    started: bool,
    attached: bool,
    /// Times of the pictures handed in and not yet out, by their
    /// Media Foundation time, so a unit leaves with the caller's own clock.
    times: VecDeque<(i64, FrameTime)>,
    cancel: Arc<AtomicBool>,
}

/// Whether the transform allocates its output samples, and how large the
/// caller's must be when it does not.
#[derive(Clone, Copy)]
struct OutputSlot {
    provides: bool,
    size: usize,
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    h264_only(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    com::enter()?;
    let transform = com::transform(&com::CLSID_H264_DECODER).map_err(|_| decline())?;
    let input = com::media_type().map_err(|_| decline())?;
    com::set_guid(&input, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_VIDEO)
        .and_then(|()| com::set_guid(&input, &com::MF_MT_SUBTYPE, &com::MF_VIDEO_FORMAT_H264))
        .and_then(|()| com::set_pair(&input, &com::MF_MT_FRAME_SIZE, width, height))
        .and_then(|()| com::set_input_type(&transform, &input))
        .map_err(|_| decline())?;
    let mut decoder = VideoDecoder {
        width,
        height,
        planar: params.pixel == PixelFormat::Yuv420p,
        length_size: params.nal_length_size,
        sets: Vec::new(),
        send_sets: false,
        transform,
        output: OutputSlot {
            provides: false,
            size: 0,
        },
        layout: None,
        timescale: 30,
    };
    decoder.choose_output().map_err(|_| decline())?;
    if !params.extra.is_empty() {
        decoder.configure(&params.extra)?;
    }
    start(&decoder.transform).map_err(|_| decline())?;
    Ok(Box::new(decoder))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    h264_only(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    if !matches!(
        params.pixel,
        PixelFormat::None | PixelFormat::Rgba8 | PixelFormat::Yuv420p
    ) {
        return Err(CodecError::Unsupported {
            what: "pixel format",
        });
    }
    com::enter()?;
    let bitrate = target_bitrate(params);
    let rate = (30, 1);
    let (transform, output) = build_encoder(width, height, bitrate, rate)?;
    Ok(Box::new(VideoEncoder {
        width,
        height,
        bitrate,
        rate,
        transform,
        output,
        started: false,
        attached: false,
        times: VecDeque::new(),
        cancel: Arc::new(AtomicBool::new(false)),
    }))
}

impl Decoder for VideoDecoder {
    fn codec(&self) -> CodecId {
        CodecId::H264
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        limits::packet_ok(packet.data.len())?;
        com::enter()?;
        if let Some(config) = config_of(packet) {
            self.configure(config)?;
        }
        let mut unit = self.annex_b(&packet.data)?;
        if self.send_sets {
            let mut with_sets =
                nal::to_annex_b(&self.sets.iter().cloned().map(nal_of).collect::<Vec<_>>());
            with_sets.append(&mut unit);
            unit = with_sets;
        }
        let timescale = if packet.timescale == 0 {
            30
        } else {
            packet.timescale
        };
        let sample = com::sample_from(
            &unit,
            to_hns(packet.pts, timescale),
            to_hns(packet.duration, timescale),
        )
        .map_err(|_| malformed())?;
        self.timescale = timescale;
        let mut outputs = Vec::new();
        let mut code = com::process_input(&self.transform, &sample);
        if code == com::MF_E_NOTACCEPTING {
            outputs = self.collect()?;
            code = com::process_input(&self.transform, &sample);
        }
        if code < 0 {
            return Err(malformed());
        }
        self.send_sets = false;
        outputs.extend(self.collect()?);
        Ok(outputs)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        com::enter()?;
        com::message(&self.transform, com::MFT_MESSAGE_NOTIFY_END_OF_STREAM)
            .and_then(|()| com::message(&self.transform, com::MFT_MESSAGE_COMMAND_DRAIN))
            .map_err(|_| malformed())?;
        let outputs = self.collect();
        let _ = com::message(&self.transform, com::MFT_MESSAGE_NOTIFY_START_OF_STREAM);
        outputs
    }

    fn reset(&mut self) {
        if com::enter().is_err() {
            return;
        }
        let _ = com::message(&self.transform, com::MFT_MESSAGE_COMMAND_FLUSH);
        self.send_sets = !self.sets.is_empty();
    }
}

impl VideoDecoder {
    /// Take the parameter sets and NAL length size from an `avcC`.
    fn configure(&mut self, extra: &[u8]) -> Result<(), CodecError> {
        let config = nal::avc_config(extra)?;
        self.length_size = config.length_size;
        self.sets = config.sps.into_iter().chain(config.pps).collect();
        self.send_sets = true;
        Ok(())
    }

    /// The unit with start codes. Length framing is tried first, as on
    /// Apple: a 256 to 511 byte NAL has a prefix that reads as a start code.
    fn annex_b(&self, data: &[u8]) -> Result<Vec<u8>, CodecError> {
        let size = if self.length_size == 0 {
            4
        } else {
            self.length_size
        };
        if let Ok(nals) = nal::split_lengths(data, size) {
            return Ok(nal::to_annex_b(&nals));
        }
        nal::annex_b(data)?;
        Ok(data.to_vec())
    }

    /// Pick NV12 out of the transform's output types and read where the
    /// picture sits in its buffer.
    fn choose_output(&mut self) -> Result<(), Hresult> {
        let mut index = 0;
        let chosen = loop {
            let Some(kind) = com::output_available_type(&self.transform, index)? else {
                return Err(com::MF_E_NO_MORE_TYPES);
            };
            if com::get_guid(&kind, &com::MF_MT_SUBTYPE) == Some(com::MF_VIDEO_FORMAT_NV12) {
                break kind;
            }
            index += 1;
        };
        com::set_output_type(&self.transform, &chosen)?;
        let current = com::output_current_type(&self.transform)?;
        let (columns, rows) =
            com::get_pair(&current, &com::MF_MT_FRAME_SIZE).unwrap_or((self.width, self.height));
        let stride = com::get_u32(&current, &com::MF_MT_DEFAULT_STRIDE)
            .and_then(|stride| i32::try_from(stride).ok())
            .filter(|stride| *stride > 0)
            .map_or(columns as usize, |stride| stride as usize);
        let (width, height) = com::get_blob::<16>(&current, &com::MF_MT_MINIMUM_DISPLAY_APERTURE)
            .and_then(aperture)
            .unwrap_or((self.width.min(columns), self.height.min(rows)));
        let layout = Layout {
            width: width & !1,
            height: height & !1,
            stride,
            rows: rows as usize,
        };
        self.layout = layout.len().is_ok().then_some(layout);
        let info = com::output_stream_info(&self.transform)?;
        self.output = OutputSlot {
            provides: info.flags & com::MFT_OUTPUT_STREAM_PROVIDES_SAMPLES != 0,
            size: info.size as usize,
        };
        Ok(())
    }

    /// Every picture the transform has ready.
    fn collect(&mut self) -> Result<Vec<Output>, CodecError> {
        let mut outputs = Vec::new();
        for _ in 0..MAX_OUTPUTS {
            let provided = self.output.sample().map_err(|_| malformed())?;
            match com::process_output(&self.transform, provided) {
                Ok(sample) => outputs.push(Output::Picture(self.picture(&sample)?)),
                Err(com::MF_E_TRANSFORM_NEED_MORE_INPUT) => return Ok(outputs),
                Err(com::MF_E_TRANSFORM_STREAM_CHANGE) => {
                    self.choose_output().map_err(|_| malformed())?;
                }
                Err(_) => return Err(malformed()),
            }
        }
        Ok(outputs)
    }

    fn picture(&self, sample: &Com) -> Result<PictureOutput, CodecError> {
        let Some(layout) = self.layout else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "picture size",
            });
        };
        let planar = self.planar;
        let bytes = com::with_sample_bytes(sample, |bytes| {
            if planar {
                nv12::to_planar(layout, bytes)
            } else {
                nv12::to_rgba(layout, bytes)
            }
        })
        .map_err(|_| malformed())??;
        let pts = com::sample_time(sample).map_or(0, |time| from_hns(time, self.timescale));
        let info = PictureInfo {
            width: layout.width,
            height: layout.height,
            pixel: if planar {
                PixelFormat::Yuv420p
            } else {
                PixelFormat::Rgba8
            },
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Bt709,
        };
        PictureOutput::cpu(info, bytes, pts, self.timescale)
    }
}

impl Encoder for VideoEncoder {
    fn codec(&self) -> CodecId {
        CodecId::H264
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "audio input",
        })
    }

    fn push_picture(&mut self, _input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture input",
        })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        if frame.width() != self.width || frame.height() != self.height {
            return Err(CodecError::Unsupported { what: "frame" });
        }
        let picture = match frame.pixel() {
            PixelFormat::Rgba8 => nv12::from_rgba(self.width, self.height, frame.bytes())?,
            PixelFormat::Yuv420p => nv12::from_planar(self.width, self.height, frame.bytes())?,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            }
        };
        com::enter()?;
        let time = frame.time();
        if !self.started {
            self.match_rate(time)?;
        }
        let hns = to_hns(time.pts(), time.timescale());
        let sample = com::sample_from(&picture, hns, to_hns(time.duration(), time.timescale()))
            .map_err(|_| refused())?;
        let mut packets = Vec::new();
        let mut code = com::process_input(&self.transform, &sample);
        if code == com::MF_E_NOTACCEPTING {
            packets = self.collect()?;
            code = com::process_input(&self.transform, &sample);
        }
        if code < 0 {
            return Err(refused());
        }
        self.started = true;
        self.times.push_back((hns, time));
        packets.extend(self.collect()?);
        Ok(packets)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if !self.started {
            return Ok(Vec::new());
        }
        com::enter()?;
        com::message(&self.transform, com::MFT_MESSAGE_NOTIFY_END_OF_STREAM)
            .and_then(|()| com::message(&self.transform, com::MFT_MESSAGE_COMMAND_DRAIN))
            .map_err(|_| refused())?;
        let packets = self.collect();
        let _ = com::message(&self.transform, com::MFT_MESSAGE_NOTIFY_START_OF_STREAM);
        self.times.clear();
        packets
    }

    /// A new transform, so the next picture is a keyframe that carries the
    /// configuration again.
    fn reset(&mut self) {
        self.times.clear();
        self.attached = false;
        self.started = false;
        if com::enter().is_err() {
            return;
        }
        if let Ok((transform, output)) =
            build_encoder(self.width, self.height, self.bitrate, self.rate)
        {
            self.transform = transform;
            self.output = output;
        }
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl VideoEncoder {
    /// Rate control budgets bits per picture, so the transform is rebuilt
    /// at the rate the first picture's duration names.
    fn match_rate(&mut self, time: FrameTime) -> Result<(), CodecError> {
        let rate = frame_rate(time);
        if rate == self.rate {
            return Ok(());
        }
        let (transform, output) = build_encoder(self.width, self.height, self.bitrate, rate)?;
        self.transform = transform;
        self.output = output;
        self.rate = rate;
        Ok(())
    }

    fn collect(&mut self) -> Result<Vec<Packet>, CodecError> {
        let mut packets = Vec::new();
        for _ in 0..MAX_OUTPUTS {
            let provided = self.output.sample().map_err(|_| refused())?;
            match com::process_output(&self.transform, provided) {
                Ok(sample) => {
                    if let Some(packet) = self.packet(&sample)? {
                        packets.push(packet);
                    }
                }
                Err(com::MF_E_TRANSFORM_NEED_MORE_INPUT) => return Ok(packets),
                Err(com::MF_E_TRANSFORM_STREAM_CHANGE) => {
                    let kind = com::output_available_type(&self.transform, 0)
                        .ok()
                        .flatten()
                        .ok_or_else(refused)?;
                    com::set_output_type(&self.transform, &kind).map_err(|_| refused())?;
                    let info = com::output_stream_info(&self.transform).map_err(|_| refused())?;
                    self.output = OutputSlot::from_info(&info);
                }
                Err(_) => return Err(refused()),
            }
        }
        Ok(packets)
    }

    /// One length-prefixed unit, with the `avcC` on the first keyframe.
    fn packet(&mut self, sample: &Com) -> Result<Option<Packet>, CodecError> {
        let bytes = com::sample_bytes(sample).map_err(|_| refused())?;
        let nals = nal::annex_b(&bytes)?;
        let mut sps = Vec::new();
        let mut pps = Vec::new();
        let mut units = Vec::new();
        let mut keyframe = false;
        for unit in nals {
            match nal::h264_type(&unit.bytes) {
                Some(7) => sps.push(unit.bytes),
                Some(8) => pps.push(unit.bytes),
                Some(9) => {}
                kind => {
                    keyframe |= kind == Some(5);
                    units.push(unit);
                }
            }
        }
        if units.is_empty() {
            return Ok(None);
        }
        let hns = com::sample_time(sample).unwrap_or(0);
        let time = self.take_time(hns, sample)?;
        let mut packet = Packet {
            stream: 0,
            data: nal::length_prefixed(&units, 4)?,
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
            keyframe,
            side: Vec::new(),
        };
        if keyframe && !self.attached && !sps.is_empty() && !pps.is_empty() {
            let config = nal::write_avc_config(&AvcConfig {
                length_size: 4,
                profile: sps[0].get(1).copied().unwrap_or(66),
                compatibility: sps[0].get(2).copied().unwrap_or(0),
                level: sps[0].get(3).copied().unwrap_or(0x1f),
                sps,
                pps,
            })?;
            packet.side.push(SideData::Config(config));
            self.attached = true;
        }
        Ok(Some(packet))
    }

    /// The caller's time for the picture at `hns`. Baseline does not
    /// reorder, so this is the oldest one; a time the transform moved is
    /// matched by value first.
    fn take_time(&mut self, hns: i64, sample: &Com) -> Result<FrameTime, CodecError> {
        let at = self
            .times
            .iter()
            .position(|(time, _)| *time == hns)
            .unwrap_or(0);
        if let Some((_, time)) = self.times.remove(at) {
            return Ok(time);
        }
        let timescale = 30;
        let duration = com::sample_duration(sample).map_or(1, |hns| from_hns(hns, timescale));
        FrameTime::new(from_hns(hns, timescale), duration.max(1), timescale)
    }
}

impl OutputSlot {
    fn from_info(info: &com::OutputStreamInfo) -> Self {
        Self {
            provides: info.flags & com::MFT_OUTPUT_STREAM_PROVIDES_SAMPLES != 0,
            size: info.size as usize,
        }
    }

    /// A sample for the transform to fill, or `None` when it brings its own.
    fn sample(self) -> Result<Option<Com>, Hresult> {
        if self.provides {
            Ok(None)
        } else {
            com::empty_sample(self.size).map(Some)
        }
    }
}

fn build_encoder(
    width: u32,
    height: u32,
    bitrate: u32,
    (numerator, denominator): (u32, u32),
) -> Result<(Com, OutputSlot), CodecError> {
    let transform = com::transform(&com::CLSID_H264_ENCODER).map_err(|_| decline())?;
    let output = com::media_type().map_err(|_| decline())?;
    com::set_guid(&output, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_VIDEO)
        .and_then(|()| com::set_guid(&output, &com::MF_MT_SUBTYPE, &com::MF_VIDEO_FORMAT_H264))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_AVG_BITRATE, bitrate))
        .and_then(|()| com::set_pair(&output, &com::MF_MT_FRAME_RATE, numerator, denominator))
        .and_then(|()| com::set_pair(&output, &com::MF_MT_FRAME_SIZE, width, height))
        .and_then(|()| com::set_pair(&output, &com::MF_MT_PIXEL_ASPECT_RATIO, 1, 1))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_INTERLACE_MODE, com::PROGRESSIVE))
        .and_then(|()| com::set_u32(&output, &com::MF_MT_MPEG2_PROFILE, com::H264_BASELINE))
        .and_then(|()| com::set_output_type(&transform, &output))
        .map_err(|_| decline())?;
    let input = com::media_type().map_err(|_| decline())?;
    com::set_guid(&input, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_VIDEO)
        .and_then(|()| com::set_guid(&input, &com::MF_MT_SUBTYPE, &com::MF_VIDEO_FORMAT_NV12))
        .and_then(|()| com::set_pair(&input, &com::MF_MT_FRAME_RATE, numerator, denominator))
        .and_then(|()| com::set_pair(&input, &com::MF_MT_FRAME_SIZE, width, height))
        .and_then(|()| com::set_pair(&input, &com::MF_MT_PIXEL_ASPECT_RATIO, 1, 1))
        .and_then(|()| com::set_u32(&input, &com::MF_MT_INTERLACE_MODE, com::PROGRESSIVE))
        .and_then(|()| com::set_input_type(&transform, &input))
        .map_err(|_| decline())?;
    let info = com::output_stream_info(&transform).map_err(|_| decline())?;
    start(&transform).map_err(|_| decline())?;
    Ok((transform, OutputSlot::from_info(&info)))
}

fn start(transform: &Com) -> Result<(), Hresult> {
    com::message(transform, com::MFT_MESSAGE_NOTIFY_BEGIN_STREAMING)?;
    com::message(transform, com::MFT_MESSAGE_NOTIFY_START_OF_STREAM)
}

/// `MFVideoArea`: two `MFOffset` then a `SIZE`. Only the size is used; a
/// cropped origin is not something these transforms write.
fn aperture(area: [u8; 16]) -> Option<(u32, u32)> {
    let width = i32::from_le_bytes([area[8], area[9], area[10], area[11]]);
    let height = i32::from_le_bytes([area[12], area[13], area[14], area[15]]);
    match (u32::try_from(width), u32::try_from(height)) {
        (Ok(width), Ok(height)) if width > 0 && height > 0 => Some((width, height)),
        _ => None,
    }
}

fn h264_only(codec: CodecId) -> Result<(), CodecError> {
    if codec == CodecId::H264 {
        Ok(())
    } else {
        Err(CodecError::Unsupported {
            what: "media foundation codec",
        })
    }
}

fn video_size(width: u32, height: u32) -> Result<(u32, u32), CodecError> {
    match even_picture(width, height) {
        Ok(()) => Ok((width, height)),
        Err(CodecError::Limit { .. }) => Err(CodecError::Unsupported {
            what: "picture size",
        }),
        Err(err) => Err(err),
    }
}

/// The bitrate asked for, or the VideoToolbox default of 8 bits a pixel.
fn target_bitrate(params: &CodecParams) -> u32 {
    if params.bitrate != 0 {
        return params.bitrate;
    }
    let pixels = u64::from(params.width).saturating_mul(u64::from(params.height));
    u32::try_from(pixels.saturating_mul(8).max(64_000)).unwrap_or(u32::MAX)
}

/// Pictures per second as a reduced fraction, from one picture's duration.
fn frame_rate(time: FrameTime) -> (u32, u32) {
    let numerator = time.timescale();
    let Ok(denominator) = u32::try_from(time.duration()) else {
        return (30, 1);
    };
    if denominator == 0 {
        return (30, 1);
    }
    let divisor = gcd(numerator, denominator);
    (numerator / divisor, denominator / divisor)
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn to_hns(ticks: i64, timescale: u32) -> i64 {
    let scaled = i128::from(ticks) * HNS;
    let timescale = i128::from(timescale.max(1));
    i64::try_from(round_div(scaled, timescale)).unwrap_or(i64::MAX)
}

fn from_hns(hns: i64, timescale: u32) -> i64 {
    let scaled = i128::from(hns) * i128::from(timescale.max(1));
    i64::try_from(round_div(scaled, HNS)).unwrap_or(i64::MAX)
}

/// Division that rounds half away from zero.
fn round_div(value: i128, divisor: i128) -> i128 {
    let half = divisor / 2;
    if value >= 0 {
        (value + half) / divisor
    } else {
        (value - half) / divisor
    }
}

fn nal_of(bytes: Vec<u8>) -> Nal {
    Nal { bytes }
}

fn config_of(packet: &Packet) -> Option<&[u8]> {
    packet.side.iter().find_map(|side| match side {
        SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.as_slice()),
        _ => None,
    })
}

fn decline() -> CodecError {
    CodecError::Unsupported {
        what: "media foundation h264",
    }
}

fn malformed() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "media foundation h264",
    }
}

fn refused() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "media foundation h264 encode",
    }
}
