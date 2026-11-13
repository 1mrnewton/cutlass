//! MediaCodec H.264, HEVC, VP9 and AV1 decode.
//!
//! The platform's own pick for the type goes first, which is the hardware
//! decoder on a phone that has one, then Android's software decoders by
//! name. A decoder that will not configure is skipped at open. One that
//! opens and then gives pictures in a layout this crate cannot read, or
//! refuses a unit before its first picture, is swapped for the next one and
//! handed the units again, so a vendor format falls back to software.
//!
//! H.264 and HEVC units cross as Annex B: a length-prefixed unit is
//! rewritten with start codes, and the parameter sets go in `csd-0` and
//! `csd-1` and again in front of the first unit after open, a reset or a new
//! configuration. VP9 frames and AV1 temporal units cross as they are, with
//! an `av1C` in `csd-0`.
//!
//! MediaCodec decodes on its own threads, so a picture can leave a few
//! pushes after its unit went in. `drain` sends end of stream, waits for
//! every picture, then flushes so the next push starts again. Pictures leave
//! as planar 4:2:0, eight or ten bits, or as RGBA when the caller asked for
//! it, with the exact time of their packet.

use std::time::{Duration, Instant};

use cutlass_color::nv12::{self, Layout};
use cutlass_decode::nal::{self, Nal};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, Decoder, Output, Packet,
    PictureInfo, PictureOutput, PixelFormat, Transfer,
};

use super::ndk::{
    BufferInfo, Codec, Dequeued, FLAG_CODEC_CONFIG, FLAG_END_OF_STREAM, Format, Status,
};
use crate::media::clock::{self, Clock};
use crate::media::kind::{Kind, config_of, is_software};
use crate::planes::{MAX_EDGE, Planes};

/// Longest a push waits for the codec to free an input buffer.
const INPUT_WAIT: Duration = Duration::from_secs(5);

/// Longest a drain waits for the last picture.
const DRAIN_WAIT: Duration = Duration::from_secs(10);

/// How long one wait for output lasts while the codec is busy.
const POLL_US: i64 = 2_000;

/// Largest compressed unit this crate hands to MediaCodec.
const MAX_PACKET: usize = 16 * 1024 * 1024;

/// Units kept for a swap to the next decoder before the first picture.
const MAX_REPLAY: usize = 240;

/// Where a decoder comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    /// The platform's pick for the type.
    Platform,
    Named(&'static str),
}

/// One unit as it was queued, kept until the first picture.
struct Unit {
    bytes: Vec<u8>,
    micros: i64,
}

pub struct VideoDecoder {
    kind: Kind,
    width: u32,
    height: u32,
    rgba: bool,
    /// 0 when units arrive as Annex B.
    length_size: u8,
    /// Parameter sets, with no start codes.
    sets: Vec<Vec<u8>>,
    extra: Vec<u8>,
    send_sets: bool,
    codec: Codec,
    name: String,
    /// The candidates after the one running.
    rest: Vec<Candidate>,
    planes: Option<Planes>,
    color: Color,
    clock: Clock<i64>,
    timescale: u32,
    produced: bool,
    replay: Option<Vec<Unit>>,
    /// A unit went in since the last drain.
    fed: bool,
}

/// What the output format says about the picture's color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Color {
    primaries: ColorPrimaries,
    transfer: Transfer,
    matrix: ColorMatrix,
}

impl Default for Color {
    fn default() -> Self {
        Self {
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Bt709,
            matrix: ColorMatrix::Bt709,
        }
    }
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(open_decoder_on(params, true)?))
}

/// The decoder, the platform's pick first when `hardware` is set.
pub fn open_decoder_on(params: &CodecParams, hardware: bool) -> Result<VideoDecoder, CodecError> {
    let kind = Kind::of(params.codec)?;
    let (width, height) = video_size(params.width, params.height)?;
    let (mut length_size, mut sets) = (params.nal_length_size, Vec::new());
    if !params.extra.is_empty() && kind.uses_nal() {
        (length_size, sets) = kind.read_config(&params.extra)?;
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    if hardware {
        candidates.push(Candidate::Platform);
    }
    candidates.extend(
        kind.software_decoders()
            .iter()
            .map(|name| Candidate::Named(name)),
    );
    let mut rest = candidates;
    let extra = if kind.uses_nal() {
        Vec::new()
    } else {
        params.extra.clone()
    };
    let (codec, name) = next_codec(kind, &mut rest, width, height, &sets, &extra)?;
    Ok(VideoDecoder {
        kind,
        width,
        height,
        rgba: params.pixel == PixelFormat::Rgba8,
        length_size,
        send_sets: !sets.is_empty(),
        sets,
        extra,
        codec,
        name,
        rest,
        planes: None,
        color: Color::default(),
        clock: Clock::new(),
        timescale: 30,
        produced: false,
        replay: Some(Vec::new()),
        fed: false,
    })
}

impl VideoDecoder {
    /// The MediaCodec name decoding, such as `c2.android.avc.decoder`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether a vendor decoder is running rather than Android's software one.
    pub fn hardware(&self) -> bool {
        !is_software(&self.name)
    }
}

impl Decoder for VideoDecoder {
    fn codec(&self) -> CodecId {
        self.kind.codec()
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        if packet.data.len() > MAX_PACKET {
            return Err(CodecError::Limit {
                what: "packet bytes",
            });
        }
        if let Some(config) = config_of(packet)
            && self.kind.uses_nal()
        {
            (self.length_size, self.sets) = self.kind.read_config(config)?;
            self.send_sets = !self.sets.is_empty();
        }
        let unit = self.unit(&packet.data)?;
        self.timescale = if packet.timescale == 0 {
            30
        } else {
            packet.timescale
        };
        let micros = clock::to_micros(packet.pts, self.timescale);
        self.clock.insert(micros, packet.pts);
        // Kept before it is queued, so a swap to the next decoder sees it.
        if let Some(replay) = self.replay.as_mut() {
            if self.produced || replay.len() >= MAX_REPLAY {
                self.replay = None;
            } else {
                replay.push(Unit {
                    bytes: unit.clone(),
                    micros,
                });
            }
        }
        let mut out = Vec::new();
        match self.feed(&unit, micros, &mut out) {
            Ok(()) => {}
            Err(error) if self.can_swap() => {
                self.swap(error)?;
                out.clear();
                self.replay_all(&mut out)?;
            }
            Err(error) => return Err(error),
        }
        self.send_sets = false;
        self.fed = true;
        if self.produced {
            self.replay = None;
        }
        Ok(out)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        if !self.fed {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let result = self.finish(&mut out);
        let result = match result {
            Err(error) if self.can_swap() => {
                self.swap(error)?;
                out.clear();
                self.replay_all(&mut out)?;
                self.finish(&mut out)
            }
            other => other,
        };
        self.restart();
        result.map(|()| out)
    }

    fn reset(&mut self) {
        self.restart();
        self.replay = if self.produced {
            None
        } else {
            Some(Vec::new())
        };
    }
}

impl VideoDecoder {
    /// The unit as MediaCodec takes it.
    fn unit(&self, data: &[u8]) -> Result<Vec<u8>, CodecError> {
        if !self.kind.uses_nal() {
            return Ok(data.to_vec());
        }
        let mut unit = annex_b(data, self.length_size)?;
        if self.send_sets {
            let sets: Vec<Nal> = self
                .sets
                .iter()
                .map(|bytes| Nal {
                    bytes: bytes.clone(),
                })
                .collect();
            let mut with_sets = nal::to_annex_b(&sets);
            with_sets.append(&mut unit);
            unit = with_sets;
        }
        Ok(unit)
    }

    /// Queue one unit, then take every picture that is ready.
    fn feed(&mut self, unit: &[u8], micros: i64, out: &mut Vec<Output>) -> Result<(), CodecError> {
        let index = self.input_slot(out)?;
        let written = self
            .codec
            .with_input(index, |buffer| {
                let target = buffer.get_mut(..unit.len())?;
                target.copy_from_slice(unit);
                Some(unit.len())
            })
            .flatten()
            .ok_or(CodecError::Limit {
                what: "mediacodec input buffer",
            })?;
        self.codec
            .queue_input(index, written, micros, 0)
            .map_err(refused)?;
        while self.take(0, out)? {}
        Ok(())
    }

    /// An empty input buffer, taking pictures while the codec is full.
    fn input_slot(&mut self, out: &mut Vec<Output>) -> Result<usize, CodecError> {
        let start = Instant::now();
        loop {
            if let Some(index) = self.codec.input_slot(0).map_err(refused)? {
                return Ok(index);
            }
            if !self.take(POLL_US, out)? && start.elapsed() > INPUT_WAIT {
                return Err(stalled());
            }
        }
    }

    /// End of stream, then every picture up to it.
    fn finish(&mut self, out: &mut Vec<Output>) -> Result<(), CodecError> {
        let index = self.input_slot(out)?;
        self.codec
            .queue_input(index, 0, 0, FLAG_END_OF_STREAM)
            .map_err(refused)?;
        let start = Instant::now();
        loop {
            match self.next_output(POLL_US * 5, out)? {
                Taken::Ended => return Ok(()),
                Taken::Picture | Taken::Changed => {}
                Taken::Nothing if start.elapsed() > DRAIN_WAIT => return Err(stalled()),
                Taken::Nothing => {}
            }
        }
    }

    /// Take one output if one comes within `timeout_us`. True when the
    /// codec gave something, so the caller should ask again.
    fn take(&mut self, timeout_us: i64, out: &mut Vec<Output>) -> Result<bool, CodecError> {
        Ok(!matches!(
            self.next_output(timeout_us, out)?,
            Taken::Nothing
        ))
    }

    fn next_output(&mut self, timeout_us: i64, out: &mut Vec<Output>) -> Result<Taken, CodecError> {
        match self.codec.dequeue_output(timeout_us) {
            Dequeued::TryAgain => Ok(Taken::Nothing),
            Dequeued::FormatChanged => {
                self.read_format()?;
                Ok(Taken::Changed)
            }
            Dequeued::Failed(code) => Err(refused(Status(i32::try_from(code).unwrap_or(-1)))),
            Dequeued::Buffer { index, info } => {
                let ended = info.flags & FLAG_END_OF_STREAM != 0;
                let picture = if info.size > 0 && info.flags & FLAG_CODEC_CONFIG == 0 {
                    Some(self.picture(index, info))
                } else {
                    None
                };
                self.codec.release_output(index);
                if let Some(picture) = picture {
                    out.push(Output::Picture(picture?));
                    self.produced = true;
                }
                Ok(if ended { Taken::Ended } else { Taken::Picture })
            }
        }
    }

    fn read_format(&mut self) -> Result<(), CodecError> {
        let format = self.codec.output_format().ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "mediacodec output format",
        })?;
        self.planes = Some(Planes::of(&format.described())?);
        self.color = color_of(&format);
        Ok(())
    }

    fn picture(&mut self, index: usize, info: BufferInfo) -> Result<PictureOutput, CodecError> {
        if self.planes.is_none() {
            self.read_format()?;
        }
        let Some(planes) = self.planes else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "mediacodec output format",
            });
        };
        let bytes = self
            .codec
            .with_output(index, info, |buffer| planes.read(buffer))
            .ok_or(CodecError::Malformed {
                at_bit: 0,
                what: "mediacodec output buffer",
            })??;
        let micros = info.presentation_us;
        let (pixel, bytes) = if self.rgba {
            (PixelFormat::Rgba8, to_rgba(&planes, &bytes)?)
        } else {
            (planes.pixel(), bytes)
        };
        let pts = self
            .clock
            .take(micros)
            .unwrap_or_else(|| clock::from_micros(micros, self.timescale));
        let mut info = PictureInfo::new(
            planes.width,
            planes.height,
            pixel,
            self.color.primaries,
            self.color.transfer,
        );
        info.matrix = self.color.matrix;
        PictureOutput::cpu(info, bytes, pts, self.timescale)
    }

    fn can_swap(&self) -> bool {
        !self.produced && self.replay.is_some() && !self.rest.is_empty()
    }

    /// Move to the next candidate after `error` from the one running.
    fn swap(&mut self, error: CodecError) -> Result<(), CodecError> {
        let (codec, name) = next_codec(
            self.kind,
            &mut self.rest,
            self.width,
            self.height,
            &self.sets,
            &self.extra,
        )
        .map_err(|_| error)?;
        self.codec = codec;
        self.name = name;
        self.planes = None;
        Ok(())
    }

    /// Queue every unit since open again, on a fresh decoder.
    fn replay_all(&mut self, out: &mut Vec<Output>) -> Result<(), CodecError> {
        let units = self.replay.take().unwrap_or_default();
        for unit in &units {
            self.feed(&unit.bytes, unit.micros, out)?;
        }
        self.replay = Some(units);
        Ok(())
    }

    /// Flush so the next push starts a new stream with the parameter sets.
    fn restart(&mut self) {
        let _ = self.codec.flush();
        self.clock.clear();
        self.send_sets = !self.sets.is_empty();
        self.fed = false;
    }
}

enum Taken {
    Nothing,
    Changed,
    Picture,
    Ended,
}

/// The first candidate that configures and starts.
fn next_codec(
    kind: Kind,
    rest: &mut Vec<Candidate>,
    width: u32,
    height: u32,
    sets: &[Vec<u8>],
    extra: &[u8],
) -> Result<(Codec, String), CodecError> {
    while !rest.is_empty() {
        let candidate = rest.remove(0);
        let codec = match candidate {
            Candidate::Platform => Codec::by_type(kind.mime(), false),
            Candidate::Named(name) => Codec::by_name(name),
        };
        let Some(mut codec) = codec else {
            continue;
        };
        let name = codec.name().unwrap_or_default();
        // The platform's pick may be a software decoder named later.
        rest.retain(|next| !matches!(next, Candidate::Named(known) if *known == name));
        let Some(format) = Format::new() else {
            continue;
        };
        format.set_str(c"mime", kind.mime());
        format.set_i32(c"width", i32::try_from(width).unwrap_or(0));
        format.set_i32(c"height", i32::try_from(height).unwrap_or(0));
        let input = (width as usize * height as usize * 3 / 2).max(1 << 20);
        format.set_i32(c"max-input-size", i32::try_from(input).unwrap_or(i32::MAX));
        for (key, data) in [c"csd-0", c"csd-1"].iter().zip(kind.csd(sets, extra)) {
            format.set_buffer(key, &data);
        }
        if codec.configure(&format, false).is_err() || codec.start().is_err() {
            continue;
        }
        return Ok((codec, name));
    }
    Err(CodecError::Unsupported {
        what: "mediacodec decoder",
    })
}

/// Start codes for a length-prefixed unit. Length framing is tried first, as
/// on Apple and Windows: a 256 to 511 byte NAL has a prefix that reads as a
/// start code.
fn annex_b(data: &[u8], length_size: u8) -> Result<Vec<u8>, CodecError> {
    let size = if length_size == 0 { 4 } else { length_size };
    if let Ok(nals) = nal::split_lengths(data, size) {
        return Ok(nal::to_annex_b(&nals));
    }
    nal::annex_b(data)?;
    Ok(data.to_vec())
}

fn video_size(width: u32, height: u32) -> Result<(u32, u32), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    if width > MAX_EDGE || height > MAX_EDGE {
        return Err(CodecError::Unsupported {
            what: "picture size",
        });
    }
    Ok((width, height))
}

/// Eight-bit planar 4:2:0 to RGBA, through NV12. Ten bits drop two.
fn to_rgba(planes: &Planes, yuv: &[u8]) -> Result<Vec<u8>, CodecError> {
    let eight;
    let yuv = if planes.depth > 8 {
        eight = yuv
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (u16::from_le_bytes(*pair) >> 2).min(255) as u8)
            .collect::<Vec<u8>>();
        eight.as_slice()
    } else {
        yuv
    };
    let packed = nv12::from_planar(planes.width, planes.height, yuv)?;
    nv12::to_rgba(Layout::packed(planes.width, planes.height), &packed)
}

/// `color-standard`, `color-transfer`, from the output format.
fn color_of(format: &Format) -> Color {
    let mut color = Color::default();
    match format.i32(c"color-standard") {
        Some(2 | 4) => {
            color.primaries = ColorPrimaries::Bt601;
            color.matrix = ColorMatrix::Bt601;
        }
        Some(6) => {
            color.primaries = ColorPrimaries::Bt2020;
            color.matrix = ColorMatrix::Bt2020;
        }
        _ => {}
    }
    match format.i32(c"color-transfer") {
        Some(6) => color.transfer = Transfer::Pq,
        Some(7) => color.transfer = Transfer::Hlg,
        _ => {}
    }
    color
}

fn refused(_status: Status) -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "mediacodec refused the unit",
    }
}

fn stalled() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "mediacodec stopped taking units",
    }
}
