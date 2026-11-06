//! Media Foundation H.264 and HEVC decode and encode.
//!
//! Hardware goes first. A decoder is the codec's decoder transform handed a
//! Direct3D 11 device, so DXVA decodes on the GPU and the picture is copied
//! back once. An encoder is a hardware encoder transform, such as Intel
//! Quick Sync, NVENC, or AMF, found by `MFTEnumEx`. When the hardware path
//! will not open, or refuses its first unit, the same codec runs as the
//! software transform: `CMSH264DecoderMFT` and `CMSH264EncoderMFT` for
//! H.264, and whatever HEVC transform is registered, which on most machines
//! is the HEVC Video Extensions package.
//!
//! Pictures cross as NV12. A decoded picture is CPU RGBA, or planar 4:2:0
//! when the caller asks for [`PixelFormat::Yuv420p`]. The encoder takes
//! either of those. A platform token is [`CodecError::Unsupported`].
//!
//! The decoder wants Annex B. A length-prefixed unit is rewritten with start
//! codes, and the configuration's parameter sets go in front of the first
//! unit after open, a new configuration, or a reset. The transform reorders
//! internally, so pictures leave in display order.
//!
//! The encoder writes Annex B with the parameter sets in band. Each unit
//! leaves as 4-byte length-prefixed NAL units without those sets, and the
//! first keyframe carries the `avcC` or `hvcC` as [`SideData::Config`]. B
//! pictures are turned off and the keyframe interval is 30, as on Apple, so
//! units leave in input order.
//!
//! A codec this Windows has no transform for is [`CodecError::Unsupported`],
//! so the factory moves on to the Rust codec.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, SideData, Transfer,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, FrameTime, PictureInput};

use super::com::{self, Com, DeviceManager, Hresult};
use super::transform::Transform;
use crate::format::Guid;
use crate::limits::{self, even_picture};
use crate::nal::{self, AvcConfig, Nal};
use crate::nv12::{self, Layout};

/// Media Foundation time is in 100-nanosecond units.
const HNS: i128 = 10_000_000;

/// Pictures between keyframes, as VideoToolbox is asked for.
const KEYFRAME_INTERVAL: u32 = 30;

/// The two codecs this module runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    H264,
    Hevc,
}

impl Kind {
    fn of(codec: CodecId) -> Result<Self, CodecError> {
        match codec {
            CodecId::H264 => Ok(Self::H264),
            CodecId::Hevc => Ok(Self::Hevc),
            _ => Err(CodecError::Unsupported {
                what: "media foundation codec",
            }),
        }
    }

    fn codec(self) -> CodecId {
        match self {
            Self::H264 => CodecId::H264,
            Self::Hevc => CodecId::Hevc,
        }
    }

    fn subtype(self) -> Guid {
        match self {
            Self::H264 => com::MF_VIDEO_FORMAT_H264,
            Self::Hevc => com::MF_VIDEO_FORMAT_HEVC,
        }
    }

    fn profile(self) -> u32 {
        match self {
            Self::H264 => com::H264_BASELINE,
            Self::Hevc => com::HEVC_MAIN,
        }
    }

    /// The NAL type, from the header's low five bits or bits 1 to 6.
    fn nal_type(self, nal: &[u8]) -> Option<u8> {
        match self {
            Self::H264 => nal::h264_type(nal),
            Self::Hevc => nal::hevc_type(nal),
        }
    }

    fn is_parameter_set(self, kind: u8) -> bool {
        match self {
            Self::H264 => matches!(kind, 7 | 8),
            Self::Hevc => matches!(kind, 32..=34),
        }
    }

    fn is_delimiter(self, kind: u8) -> bool {
        match self {
            Self::H264 => kind == 9,
            Self::Hevc => kind == 35,
        }
    }

    /// IDR for H.264; an intra random access point for HEVC.
    fn is_keyframe(self, kind: u8) -> bool {
        match self {
            Self::H264 => kind == 5,
            Self::Hevc => matches!(kind, 16..=21),
        }
    }

    /// Parameter sets and NAL length size from an `avcC` or `hvcC`.
    fn read_config(self, extra: &[u8]) -> Result<(u8, Vec<Vec<u8>>), CodecError> {
        match self {
            Self::H264 => {
                let config = nal::avc_config(extra)?;
                Ok((
                    config.length_size,
                    config.sps.into_iter().chain(config.pps).collect(),
                ))
            }
            Self::Hevc => {
                let config = nal::read_hevc_sets(extra)?;
                Ok((config.length_size, config.nals))
            }
        }
    }

    fn write_config(self, sets: &[Vec<u8>]) -> Result<Vec<u8>, CodecError> {
        match self {
            Self::H264 => {
                let (sps, pps): (Vec<_>, Vec<_>) = sets
                    .iter()
                    .cloned()
                    .partition(|set| nal::h264_type(set) == Some(7));
                let first = sps.first().cloned().unwrap_or_default();
                nal::write_avc_config(&AvcConfig {
                    length_size: 4,
                    profile: first.get(1).copied().unwrap_or(66),
                    compatibility: first.get(2).copied().unwrap_or(0),
                    level: first.get(3).copied().unwrap_or(0x1f),
                    sps,
                    pps,
                })
            }
            Self::Hevc => nal::write_hevc_config(sets),
        }
    }

    /// The decoder transforms to try, best first. `hardware` asks for DXVA.
    fn decoders(self, hardware: bool) -> Vec<Candidate> {
        let mut found = Vec::new();
        let software = || -> Vec<Com> {
            let input = (com::MF_MEDIA_TYPE_VIDEO, self.subtype());
            let output = (com::MF_MEDIA_TYPE_VIDEO, com::MF_VIDEO_FORMAT_NV12);
            com::enumerate(
                &com::MFT_CATEGORY_VIDEO_DECODER,
                com::ENUM_SOFTWARE,
                &input,
                &output,
            )
        };
        let make = |dxva: bool| -> Vec<Candidate> {
            let mut list = Vec::new();
            if self == Self::H264 {
                list.push(Candidate::Class(com::CLSID_H264_DECODER, dxva));
            }
            list.extend(
                software()
                    .into_iter()
                    .map(|found| Candidate::Activate(found, dxva)),
            );
            list
        };
        if hardware {
            found.extend(make(true));
        }
        found.extend(make(false));
        found
    }

    /// The encoder transforms to try, best first.
    fn encoders(self, hardware: bool) -> Vec<Candidate> {
        let input = (com::MF_MEDIA_TYPE_VIDEO, com::MF_VIDEO_FORMAT_NV12);
        let output = (com::MF_MEDIA_TYPE_VIDEO, self.subtype());
        let mut found = Vec::new();
        if hardware {
            found.extend(
                com::enumerate(
                    &com::MFT_CATEGORY_VIDEO_ENCODER,
                    com::ENUM_HARDWARE,
                    &input,
                    &output,
                )
                .into_iter()
                .map(Candidate::Hardware),
            );
        }
        if self == Self::H264 {
            found.push(Candidate::Class(com::CLSID_H264_ENCODER, false));
        }
        found.extend(
            com::enumerate(
                &com::MFT_CATEGORY_VIDEO_ENCODER,
                com::ENUM_SOFTWARE,
                &input,
                &output,
            )
            .into_iter()
            .map(|found| Candidate::Activate(found, false)),
        );
        found
    }
}

/// One way to make a transform. The flag on a software transform asks for a
/// Direct3D device.
enum Candidate {
    Class(Guid, bool),
    Activate(Com, bool),
    Hardware(Com),
}

impl Candidate {
    fn make(&self) -> Result<(Transform, bool), Hresult> {
        match self {
            Self::Class(clsid, dxva) => {
                let mft = com::transform(clsid)?;
                Self::with_device(mft, *dxva)
            }
            Self::Activate(activate, dxva) => {
                let mft = com::activate(activate)?;
                Self::with_device(mft, *dxva)
            }
            Self::Hardware(activate) => {
                let mft = com::activate(activate)?;
                Ok((Transform::new(mft, true, None)?, true))
            }
        }
    }

    fn with_device(mft: Com, dxva: bool) -> Result<(Transform, bool), Hresult> {
        if !dxva {
            return Ok((Transform::new(mft, false, None)?, false));
        }
        if !com::d3d11_aware(&mft) {
            return Err(com::MF_E_NOTACCEPTING);
        }
        let device = DeviceManager::create()?;
        device.attach(&mft)?;
        Ok((Transform::new(mft, true, Some(device))?, true))
    }
}

pub struct VideoDecoder {
    kind: Kind,
    width: u32,
    height: u32,
    planar: bool,
    /// 0 when units arrive as Annex B.
    length_size: u8,
    /// Parameter sets from the configuration, with no start codes.
    sets: Vec<Vec<u8>>,
    send_sets: bool,
    transform: Transform,
    picture: PictureSlot,
    timescale: u32,
    /// Whether a picture has come out yet. Until one has, a hardware
    /// decoder that refuses a unit is swapped for the software one.
    produced: bool,
}

/// Where the picture sits in the transform's output buffers.
#[derive(Clone, Copy, Default)]
struct PictureSlot {
    width: u32,
    height: u32,
    columns: u32,
    rows: u32,
    stride: usize,
}

pub struct VideoEncoder {
    kind: Kind,
    width: u32,
    height: u32,
    bitrate: u32,
    rate: (u32, u32),
    transform: Transform,
    started: bool,
    attached: bool,
    /// Parameter sets from the transform's output type, for an encoder that
    /// does not repeat them in band.
    header: Vec<Vec<u8>>,
    /// Times of the pictures handed in and not yet out, by their
    /// Media Foundation time, so a unit leaves with the caller's own clock.
    times: VecDeque<(i64, FrameTime)>,
    cancel: Arc<AtomicBool>,
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(open_decoder_on(params, true)?))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(open_encoder_on(params, true)?))
}

/// The decoder, on the GPU first when `hardware` is set.
pub fn open_decoder_on(params: &CodecParams, hardware: bool) -> Result<VideoDecoder, CodecError> {
    let kind = Kind::of(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    com::enter()?;
    let (mut length_size, mut sets) = (params.nal_length_size, Vec::new());
    if !params.extra.is_empty() {
        (length_size, sets) = kind.read_config(&params.extra)?;
    }
    let mut picture = PictureSlot {
        width,
        height,
        ..PictureSlot::default()
    };
    let transform = kind
        .decoders(hardware)
        .iter()
        .find_map(|candidate| build_decoder(kind, candidate, &mut picture).ok())
        .ok_or_else(decline)?;
    Ok(VideoDecoder {
        kind,
        width,
        height,
        planar: params.pixel == PixelFormat::Yuv420p,
        length_size,
        send_sets: !sets.is_empty(),
        sets,
        transform,
        picture,
        timescale: 30,
        produced: false,
    })
}

/// The encoder, on a hardware transform first when `hardware` is set.
pub fn open_encoder_on(params: &CodecParams, hardware: bool) -> Result<VideoEncoder, CodecError> {
    let kind = Kind::of(params.codec)?;
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
    let transform = first_encoder(kind, width, height, bitrate, rate, hardware)?;
    Ok(VideoEncoder {
        kind,
        width,
        height,
        bitrate,
        rate,
        transform,
        started: false,
        attached: false,
        header: Vec::new(),
        times: VecDeque::new(),
        cancel: Arc::new(AtomicBool::new(false)),
    })
}

#[cfg(test)]
impl VideoDecoder {
    /// Whether DXVA is decoding.
    pub fn hardware(&self) -> bool {
        self.transform.hardware()
    }
}

#[cfg(test)]
impl VideoEncoder {
    /// Whether a hardware encoder transform is encoding.
    pub fn hardware(&self) -> bool {
        self.transform.hardware()
    }
}

impl Decoder for VideoDecoder {
    fn codec(&self) -> CodecId {
        self.kind.codec()
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        limits::packet_ok(packet.data.len())?;
        com::enter()?;
        if let Some(config) = config_of(packet) {
            (self.length_size, self.sets) = self.kind.read_config(config)?;
            self.send_sets = true;
        }
        let mut unit = self.annex_b(&packet.data)?;
        if self.send_sets {
            let sets: Vec<Nal> = self.sets.iter().cloned().map(nal_of).collect();
            let mut with_sets = nal::to_annex_b(&sets);
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
        let samples = match self.feed(&sample) {
            Ok(samples) => samples,
            Err(_) if self.transform.hardware() && !self.produced => {
                self.fall_back()?;
                self.feed(&sample).map_err(|_| malformed())?
            }
            Err(_) => return Err(malformed()),
        };
        self.send_sets = false;
        self.pictures(&samples)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        com::enter()?;
        let mut picture = self.picture;
        let kind = self.kind;
        let samples = self
            .transform
            .drain(&mut |mft| choose_output(kind, mft, &mut picture))
            .map_err(|_| malformed())?;
        self.picture = picture;
        self.pictures(&samples)
    }

    fn reset(&mut self) {
        if com::enter().is_err() {
            return;
        }
        self.transform.flush();
        self.send_sets = !self.sets.is_empty();
    }
}

impl VideoDecoder {
    fn feed(&mut self, sample: &Com) -> Result<Vec<Com>, Hresult> {
        let mut picture = self.picture;
        let kind = self.kind;
        let samples = self
            .transform
            .feed(sample, &mut |mft| choose_output(kind, mft, &mut picture));
        self.picture = picture;
        samples
    }

    /// Swap a hardware decoder that refused its first unit for software.
    fn fall_back(&mut self) -> Result<(), CodecError> {
        let mut picture = PictureSlot {
            width: self.width,
            height: self.height,
            ..PictureSlot::default()
        };
        self.transform = self
            .kind
            .decoders(false)
            .iter()
            .find_map(|candidate| build_decoder(self.kind, candidate, &mut picture).ok())
            .ok_or_else(malformed)?;
        self.picture = picture;
        self.send_sets = !self.sets.is_empty();
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

    fn pictures(&mut self, samples: &[Com]) -> Result<Vec<Output>, CodecError> {
        let mut out = Vec::with_capacity(samples.len());
        for sample in samples {
            out.push(Output::Picture(self.picture(sample)?));
        }
        self.produced |= !out.is_empty();
        Ok(out)
    }

    fn picture(&self, sample: &Com) -> Result<PictureOutput, CodecError> {
        let slot = self.picture;
        let planar = self.planar;
        let bytes = com::with_sample_bytes(sample, |bytes| {
            let layout = slot.layout(bytes.len())?;
            if planar {
                nv12::to_planar(layout, bytes)
            } else {
                nv12::to_rgba(layout, bytes)
            }
        })
        .map_err(|_| malformed())??;
        let pts = com::sample_time(sample).map_or(0, |time| from_hns(time, self.timescale));
        let info = PictureInfo::new(
            slot.width & !1,
            slot.height & !1,
            if planar {
                PixelFormat::Yuv420p
            } else {
                PixelFormat::Rgba8
            },
            ColorPrimaries::Bt709,
            Transfer::Bt709,
        );
        PictureOutput::cpu(info, bytes, pts, self.timescale)
    }
}

impl PictureSlot {
    /// The layout of a buffer `len` bytes long. The software decoder pads
    /// its rows to the default stride. A Direct3D surface copies out with
    /// rows as wide as the frame. Whichever stride fills the buffer exactly
    /// is the one used.
    fn layout(self, len: usize) -> Result<Layout, CodecError> {
        let columns = self.columns.max(self.width) as usize;
        let rows = self.rows.max(self.height) as usize;
        let mut strides = vec![self.stride, columns];
        if rows > 0 && (len * 2).is_multiple_of(rows * 3) {
            strides.push(len * 2 / (rows * 3));
        }
        let fits = |stride: usize| {
            let layout = Layout {
                width: self.width & !1,
                height: self.height & !1,
                stride,
                rows,
            };
            layout.len().ok().map(|need| (need, layout))
        };
        strides
            .iter()
            .filter_map(|stride| fits(*stride))
            .find(|(need, _)| *need == len)
            .or_else(|| {
                strides
                    .iter()
                    .filter_map(|stride| fits(*stride))
                    .find(|(need, _)| *need <= len)
            })
            .map(|(_, layout)| layout)
            .ok_or(CodecError::Truncated {
                at_bit: (len as u64).saturating_mul(8),
                what: "nv12 buffer",
            })
    }
}

fn build_decoder(
    kind: Kind,
    candidate: &Candidate,
    picture: &mut PictureSlot,
) -> Result<Transform, Hresult> {
    let (mut transform, _) = candidate.make()?;
    // A transform without the attribute decodes the same, only later.
    let _ = com::set_low_latency(transform.mft());
    let input = com::media_type()?;
    com::set_guid(&input, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_VIDEO)?;
    com::set_guid(&input, &com::MF_MT_SUBTYPE, &kind.subtype())?;
    com::set_pair(
        &input,
        &com::MF_MT_FRAME_SIZE,
        picture.width,
        picture.height,
    )?;
    com::set_input_type(transform.mft(), &input)?;
    choose_output(kind, transform.mft(), picture)?;
    transform.start()?;
    Ok(transform)
}

/// Pick NV12 out of the transform's output types and read where the
/// picture sits in its buffer.
fn choose_output(_kind: Kind, mft: &Com, picture: &mut PictureSlot) -> Result<(), Hresult> {
    let mut index = 0;
    let chosen = loop {
        let Some(kind) = com::output_available_type(mft, index)? else {
            return Err(com::MF_E_NO_MORE_TYPES);
        };
        if com::get_guid(&kind, &com::MF_MT_SUBTYPE) == Some(com::MF_VIDEO_FORMAT_NV12) {
            break kind;
        }
        index += 1;
    };
    com::set_output_type(mft, &chosen)?;
    let current = com::output_current_type(mft)?;
    let (columns, rows) =
        com::get_pair(&current, &com::MF_MT_FRAME_SIZE).unwrap_or((picture.width, picture.height));
    let stride = com::get_u32(&current, &com::MF_MT_DEFAULT_STRIDE)
        .and_then(|stride| i32::try_from(stride).ok())
        .filter(|stride| *stride > 0)
        .map_or(columns as usize, |stride| stride as usize);
    let (width, height) = com::get_blob::<16>(&current, &com::MF_MT_MINIMUM_DISPLAY_APERTURE)
        .and_then(aperture)
        .unwrap_or((picture.width.min(columns), picture.height.min(rows)));
    *picture = PictureSlot {
        width,
        height,
        columns,
        rows,
        stride,
    };
    Ok(())
}

impl Encoder for VideoEncoder {
    fn codec(&self) -> CodecId {
        self.kind.codec()
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
        let samples = match self.feed(&sample) {
            Ok(samples) => samples,
            Err(_) if self.transform.hardware() && !self.started => {
                self.transform = first_encoder(
                    self.kind,
                    self.width,
                    self.height,
                    self.bitrate,
                    self.rate,
                    false,
                )?;
                self.feed(&sample).map_err(|_| refused())?
            }
            Err(_) => return Err(refused()),
        };
        self.started = true;
        self.times.push_back((hns, time));
        self.packets(&samples)
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if !self.started {
            return Ok(Vec::new());
        }
        com::enter()?;
        let mut header = std::mem::take(&mut self.header);
        let kind = self.kind;
        let samples = self
            .transform
            .drain(&mut |mft| encoder_output_changed(kind, mft, &mut header))
            .map_err(|_| refused());
        self.header = header;
        let packets = self.packets(&samples?);
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
        let hardware = self.transform.hardware();
        if let Ok(transform) = first_encoder(
            self.kind,
            self.width,
            self.height,
            self.bitrate,
            self.rate,
            hardware,
        ) {
            self.transform = transform;
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
        let hardware = self.transform.hardware();
        self.transform = first_encoder(
            self.kind,
            self.width,
            self.height,
            self.bitrate,
            rate,
            hardware,
        )?;
        self.rate = rate;
        Ok(())
    }

    fn feed(&mut self, sample: &Com) -> Result<Vec<Com>, Hresult> {
        let mut header = std::mem::take(&mut self.header);
        let kind = self.kind;
        let samples = self.transform.feed(sample, &mut |mft| {
            encoder_output_changed(kind, mft, &mut header)
        });
        self.header = header;
        samples
    }

    fn packets(&mut self, samples: &[Com]) -> Result<Vec<Packet>, CodecError> {
        let mut packets = Vec::with_capacity(samples.len());
        for sample in samples {
            if let Some(packet) = self.packet(sample)? {
                packets.push(packet);
            }
        }
        Ok(packets)
    }

    /// One length-prefixed unit, with the configuration on the first
    /// keyframe.
    fn packet(&mut self, sample: &Com) -> Result<Option<Packet>, CodecError> {
        let bytes = com::sample_bytes(sample).map_err(|_| refused())?;
        let nals = nal::annex_b(&bytes)?;
        let mut sets = Vec::new();
        let mut units = Vec::new();
        let mut keyframe = false;
        for unit in nals {
            match self.kind.nal_type(&unit.bytes) {
                Some(kind) if self.kind.is_parameter_set(kind) => sets.push(unit.bytes),
                Some(kind) if self.kind.is_delimiter(kind) => {}
                kind => {
                    keyframe |= kind.is_some_and(|kind| self.kind.is_keyframe(kind));
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
        if keyframe && !self.attached {
            if sets.is_empty() {
                sets = self.header.clone();
            }
            if let Ok(config) = self.kind.write_config(&sets) {
                packet.side.push(SideData::Config(config));
                self.attached = true;
            }
        }
        Ok(Some(packet))
    }

    /// The caller's time for the picture at `hns`, matched by value, else
    /// the oldest one still waiting.
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

/// An encoder that changes its output type mid-stream: take its first
/// offered type, and keep any parameter sets it now carries.
fn encoder_output_changed(kind: Kind, mft: &Com, header: &mut Vec<Vec<u8>>) -> Result<(), Hresult> {
    let offered = com::output_available_type(mft, 0)?.ok_or(com::MF_E_NO_MORE_TYPES)?;
    com::set_output_type(mft, &offered)?;
    if let Ok(current) = com::output_current_type(mft) {
        *header = sequence_header(kind, &current);
    }
    Ok(())
}

/// The parameter sets in `MF_MT_MPEG_SEQUENCE_HEADER`, which a hardware
/// encoder fills when its output type is set.
fn sequence_header(kind: Kind, output: &Com) -> Vec<Vec<u8>> {
    com::get_blob_vec(output, &com::MF_MT_MPEG_SEQUENCE_HEADER)
        .and_then(|bytes| nal::annex_b(&bytes).ok())
        .map(|nals| {
            nals.into_iter()
                .map(|nal| nal.bytes)
                .filter(|bytes| {
                    kind.nal_type(bytes)
                        .is_some_and(|t| kind.is_parameter_set(t))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The first encoder candidate that takes the types, best first.
fn first_encoder(
    kind: Kind,
    width: u32,
    height: u32,
    bitrate: u32,
    rate: (u32, u32),
    hardware: bool,
) -> Result<Transform, CodecError> {
    kind.encoders(hardware)
        .iter()
        .find_map(|candidate| build_encoder(kind, candidate, width, height, bitrate, rate).ok())
        .ok_or_else(decline)
}

fn build_encoder(
    kind: Kind,
    candidate: &Candidate,
    width: u32,
    height: u32,
    bitrate: u32,
    (numerator, denominator): (u32, u32),
) -> Result<Transform, Hresult> {
    let (mut transform, _) = candidate.make()?;
    let mft = transform.mft();
    let _ = com::set_codec_u32(mft, &com::CODECAPI_B_PICTURE_COUNT, b_pictures());
    let _ = com::set_codec_u32(mft, &com::CODECAPI_GOP_SIZE, KEYFRAME_INTERVAL);
    let _ = com::set_codec_u32(mft, &com::CODECAPI_MEAN_BIT_RATE, bitrate);
    let output = com::media_type()?;
    com::set_guid(&output, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_VIDEO)?;
    com::set_guid(&output, &com::MF_MT_SUBTYPE, &kind.subtype())?;
    com::set_u32(&output, &com::MF_MT_AVG_BITRATE, bitrate)?;
    com::set_pair(&output, &com::MF_MT_FRAME_RATE, numerator, denominator)?;
    com::set_pair(&output, &com::MF_MT_FRAME_SIZE, width, height)?;
    com::set_pair(&output, &com::MF_MT_PIXEL_ASPECT_RATIO, 1, 1)?;
    com::set_u32(&output, &com::MF_MT_INTERLACE_MODE, com::PROGRESSIVE)?;
    com::set_u32(&output, &com::MF_MT_MPEG2_PROFILE, kind.profile())?;
    com::set_output_type(mft, &output)?;
    let input = com::media_type()?;
    com::set_guid(&input, &com::MF_MT_MAJOR_TYPE, &com::MF_MEDIA_TYPE_VIDEO)?;
    com::set_guid(&input, &com::MF_MT_SUBTYPE, &com::MF_VIDEO_FORMAT_NV12)?;
    com::set_pair(&input, &com::MF_MT_FRAME_RATE, numerator, denominator)?;
    com::set_pair(&input, &com::MF_MT_FRAME_SIZE, width, height)?;
    com::set_pair(&input, &com::MF_MT_PIXEL_ASPECT_RATIO, 1, 1)?;
    com::set_u32(&input, &com::MF_MT_INTERLACE_MODE, com::PROGRESSIVE)?;
    com::set_input_type(mft, &input)?;
    // Some hardware encoders take these only once their types are set.
    let _ = com::set_codec_u32(mft, &com::CODECAPI_B_PICTURE_COUNT, b_pictures());
    let _ = com::set_codec_u32(mft, &com::CODECAPI_GOP_SIZE, KEYFRAME_INTERVAL);
    transform.start()?;
    Ok(transform)
}

/// B pictures between anchors. Always none, so units leave in input
/// order; a test asks for some to check the decoder puts them back.
fn b_pictures() -> u32 {
    #[cfg(test)]
    {
        TEST_B_PICTURES.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        0
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_B_PICTURES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
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

pub(super) fn to_hns(ticks: i64, timescale: u32) -> i64 {
    let scaled = i128::from(ticks) * HNS;
    let timescale = i128::from(timescale.max(1));
    i64::try_from(round_div(scaled, timescale)).unwrap_or(i64::MAX)
}

pub(super) fn from_hns(hns: i64, timescale: u32) -> i64 {
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

pub(super) fn config_of(packet: &Packet) -> Option<&[u8]> {
    packet.side.iter().find_map(|side| match side {
        SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.as_slice()),
        _ => None,
    })
}

fn decline() -> CodecError {
    CodecError::Unsupported {
        what: "media foundation video",
    }
}

fn malformed() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "media foundation video",
    }
}

fn refused() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "media foundation video encode",
    }
}
