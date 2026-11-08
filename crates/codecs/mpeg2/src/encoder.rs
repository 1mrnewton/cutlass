//! The MPEG-2 Main profile encoder: progressive 4:2:0 frame pictures in
//! closed groups of pictures with two B pictures between anchors, TM5 rate
//! control, and a VBV buffer model. A cut in the pictures, or a forced
//! keyframe, starts a new GOP.
//!
//! Packets leave in decode order. Every I picture carries the sequence
//! header, its extensions, and a GOP header, so any keyframe starts a
//! stream. The last packet before [`VideoEncoder::drain`] ends with
//! `sequence_end_code`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, Yuv420p};
use cutlass_encode::{EncodeFactory, VideoEncoder, VideoInput};

use crate::bits::BitWriter;
use crate::frame::{Frame, Geometry};
use crate::headers::{
    CHROMA_420, GopHeader, PICTURE_B, PICTURE_I, PICTURE_P, SEQUENCE_END, SequenceDisplayExtension,
    SequenceExtension, SequenceHeader,
};
use crate::picture::{Coded, Job, Quantisers, analyse, decode_params, write_picture};
use crate::rate::{Rate, RateConfig};
use crate::slice::{Picture, SliceData, decode_picture};
use crate::tables::{DEFAULT_INTRA, DEFAULT_NON_INTRA, FRAME_RATES};

/// B pictures between two anchors.
const B_FRAMES: usize = 2;

/// Quantiser offsets tried, in turn, when a picture would underflow the
/// decoder buffer. The last forces the coarsest quantiser.
const VBV_RETRIES: [i32; 4] = [3, 8, 16, 31];

/// Main profile levels (Table 8-11 to 8-13): `profile_and_level_indication`,
/// largest frame, largest frame rate code, luma samples per second, bit
/// rate, and VBV buffer size.
struct Level {
    indication: u8,
    width: u32,
    height: u32,
    max_rate_code: u8,
    luma_rate: u64,
    bitrate: u32,
    vbv_bits: u32,
}

const LEVELS: [Level; 3] = [
    Level {
        indication: 0x48,
        width: 720,
        height: 576,
        max_rate_code: 5,
        luma_rate: 10_368_000,
        bitrate: 15_000_000,
        vbv_bits: 1_835_008,
    },
    Level {
        indication: 0x46,
        width: 1440,
        height: 1152,
        max_rate_code: 8,
        luma_rate: 47_001_600,
        bitrate: 60_000_000,
        vbv_bits: 7_340_032,
    },
    Level {
        indication: 0x44,
        width: 1920,
        height: 1152,
        max_rate_code: 8,
        luma_rate: 62_668_800,
        bitrate: 80_000_000,
        vbv_bits: 9_781_248,
    },
];

/// The highest level this encoder writes.
const TOP: &Level = &LEVELS[LEVELS.len() - 1];

const TOO_LARGE: CodecError = CodecError::Unsupported {
    what: "picture larger than 1920×1152, the most Main profile High level holds",
};
const TOO_FAST: CodecError = CodecError::Unsupported {
    what: "bit rate above 80 Mb/s, the most Main profile High level holds",
};

/// The `profile_and_level_indication` of the lowest Main profile level
/// that holds `width` × `height` at `frames_per_second` and `bitrate` bits a
/// second (zero leaves the rate to the encoder), as [`Mpeg2Encoder`] picks
/// it at the first picture. An export can ask this before it starts.
///
/// # Errors
///
/// [`CodecError::Unsupported`] naming the limit: a picture past 1920×1152,
/// a frame rate that Table 6-4 does not list, more luma samples a second
/// than High level allows (1080p at 50 or 60 Hz), or a bit rate past
/// 80 Mb/s.
pub fn main_profile_level(
    width: u32,
    height: u32,
    frames_per_second: f64,
    bitrate: u32,
) -> Result<u8, CodecError> {
    pick_level(width, height, frames_per_second, bitrate).map(|(_, level)| level.indication)
}

/// The Table 6-4 `frame_rate_code` within 0.1% of `rate`, and the lowest
/// level that holds the stream.
fn pick_level(
    width: u32,
    height: u32,
    rate: f64,
    bitrate: u32,
) -> Result<(usize, &'static Level), CodecError> {
    if width > TOP.width || height > TOP.height {
        return Err(TOO_LARGE);
    }
    if bitrate > TOP.bitrate {
        return Err(TOO_FAST);
    }
    let Some(code) = (1..FRAME_RATES.len()).find(|&c| {
        let (n, d) = FRAME_RATES[c];
        (f64::from(n) / f64::from(d) - rate).abs() <= rate * 0.001
    }) else {
        return Err(CodecError::Unsupported {
            what: "frame rate outside Table 6-4: 23.976, 24, 25, 29.97, 30, 50, 59.94 or 60",
        });
    };
    let (n, d) = FRAME_RATES[code];
    let luma = u64::from(width) * u64::from(height);
    let luma_rate = (luma as f64 * f64::from(n) / f64::from(d)).ceil() as u64;
    LEVELS
        .iter()
        .find(|level| {
            width <= level.width
                && height <= level.height
                && code <= usize::from(level.max_rate_code)
                && luma_rate <= level.luma_rate
                && bitrate <= level.bitrate
        })
        .map(|level| (code, level))
        .ok_or(CodecError::Unsupported {
            what: "more luma samples a second than Main profile High level holds",
        })
}

/// Registers the encoder for export.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register_video(CodecId::Mpeg2, open_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(Mpeg2Encoder::open(params)?))
}

/// Everything fixed once the first picture gives the frame rate.
struct Stream {
    rate: Rate,
    /// Sequence header and its extensions, repeated before every I picture.
    headers: Vec<u8>,
    /// Sequence header and sequence extension.
    extradata: Vec<u8>,
    gop_size: u32,
    /// Frames per second the time code counts.
    time_code_rate: u64,
}

/// A source picture waiting for its turn.
struct Source {
    frame: Frame,
    pts: i64,
    duration: i64,
    timescale: u32,
    temporal_reference: u16,
}

/// Encodes 4:2:0 pictures as an MPEG-2 Main profile elementary stream.
pub struct Mpeg2Encoder {
    width: u32,
    height: u32,
    geo: Geometry,
    bitrate: u32,
    constant: bool,
    cancel: Arc<AtomicBool>,
    quant: Box<Quantisers>,
    stream: Option<Stream>,
    /// Pictures that come after the next anchor in coding order.
    pending: Vec<Source>,
    /// Display position of the next picture in its GOP.
    gop_pos: u32,
    /// Pictures accepted since the stream started, for the time code.
    frames: u64,
    /// Reconstructed older and newer anchors.
    refs: [Option<Frame>; 2],
    /// The newest packet, kept so drain can end the sequence in it.
    held: Option<Packet>,
    drained: bool,
    /// Luma histogram of the last accepted picture, to find a cut.
    last_luma: Option<Box<Histogram>>,
}

/// Luma levels in 64 bins of four.
type Histogram = [u32; 64];

/// Histogram distance, as a share of the picture's samples (0 to 2), above
/// which a picture starts a new GOP. Motion and lighting on the local
/// recordings stay under 0.25; a cut between two of them is about 1.7.
const CUT: f64 = 0.5;

/// Histogram of every other luma row. Four tables take neighbouring
/// samples, so runs of one level do not wait on a single counter.
fn histogram(luma: &[u8], width: usize) -> Box<Histogram> {
    let mut lanes = [[0u32; 64]; 4];
    for row in luma.chunks_exact(width.max(1)).step_by(2) {
        let (quads, rest) = row.as_chunks::<4>();
        for quad in quads {
            for (lane, &sample) in lanes.iter_mut().zip(quad) {
                lane[usize::from(sample >> 2)] += 1;
            }
        }
        for &sample in rest {
            lanes[0][usize::from(sample >> 2)] += 1;
        }
    }
    let mut bins = Box::new([0u32; 64]);
    for lane in &lanes {
        for (bin, count) in bins.iter_mut().zip(lane) {
            *bin += count;
        }
    }
    bins
}

/// Whether `next` shows a different scene from `last`: the pictures share
/// too few luma levels for prediction to help.
fn scene_cut(last: &Histogram, next: &Histogram) -> bool {
    let samples: u64 = next.iter().map(|&count| u64::from(count)).sum();
    let distance: u64 = last
        .iter()
        .zip(next)
        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
        .sum();
    distance as f64 > CUT * samples as f64
}

impl Mpeg2Encoder {
    /// Opens an encoder for `params.width` × `params.height`. A zero
    /// `bitrate` picks about 0.3 bits per sample.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mpeg2 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let (width, height) = (params.width, params.height);
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(CodecError::Unsupported {
                what: "picture size",
            });
        }
        if width > TOP.width || height > TOP.height {
            return Err(TOO_LARGE);
        }
        if params.bitrate > TOP.bitrate {
            return Err(TOO_FAST);
        }
        let geo = Geometry {
            mb_width: width.div_ceil(16) as usize,
            mb_height: height.div_ceil(16) as usize,
            chroma_format: CHROMA_420,
        };
        Ok(Self {
            width,
            height,
            geo,
            bitrate: params.bitrate,
            constant: params.constant_rate,
            cancel: Arc::new(AtomicBool::new(false)),
            quant: Quantisers::new(&DEFAULT_INTRA, &DEFAULT_NON_INTRA),
            stream: None,
            pending: Vec::new(),
            gop_pos: 0,
            frames: 0,
            refs: [None, None],
            held: None,
            drained: false,
            last_luma: None,
        })
    }

    /// Chooses the frame rate code, level, and rate from the first picture.
    fn open_stream(&self, duration: i64, timescale: u32) -> Result<Stream, CodecError> {
        let (code, level) = pick_level(
            self.width,
            self.height,
            timescale as f64 / duration as f64,
            self.bitrate,
        )?;
        let (n, d) = FRAME_RATES[code];
        let frame_rate = f64::from(n) / f64::from(d);
        let luma = u64::from(self.width) * u64::from(self.height);
        let default_rate = (0.3 * luma as f64 * frame_rate).max(1_000_000.0);
        let requested = if self.bitrate == 0 {
            None
        } else {
            Some(f64::from(self.bitrate))
        };
        let bitrate = requested
            .unwrap_or(default_rate)
            .min(f64::from(level.bitrate));
        let (peak, vbv_bits) = if self.constant {
            // A constant rate signals a real vbv_delay, at most 0.728 s.
            let fits = (bitrate * 0.7 / 16_384.0).floor() * 16_384.0;
            (bitrate, fits.clamp(16_384.0, f64::from(level.vbv_bits)))
        } else {
            (
                (2.0 * bitrate).min(f64::from(level.bitrate)),
                f64::from(level.vbv_bits),
            )
        };
        let bit_rate_400 = (peak / 400.0).ceil() as u32;
        let vbv_units = (vbv_bits / 16_384.0) as u32;
        let header = SequenceHeader {
            horizontal_size_value: (self.width & 0xfff) as u16,
            vertical_size_value: (self.height & 0xfff) as u16,
            // Square samples.
            aspect_ratio_information: 1,
            frame_rate_code: code as u8,
            bit_rate_value: bit_rate_400 & 0x3_ffff,
            vbv_buffer_size_value: (vbv_units & 0x3ff) as u16,
            constrained_parameters: false,
            intra_matrix: None,
            non_intra_matrix: None,
        };
        let extension = SequenceExtension {
            profile_and_level: level.indication,
            progressive_sequence: true,
            chroma_format: CHROMA_420,
            horizontal_size_extension: (self.width >> 12) as u8,
            vertical_size_extension: (self.height >> 12) as u8,
            bit_rate_extension: (bit_rate_400 >> 18) as u16,
            vbv_buffer_size_extension: (vbv_units >> 10) as u8,
            low_delay: false,
            frame_rate_extension_n: 0,
            frame_rate_extension_d: 0,
        };
        // The planar path converts to BT.709 limited range.
        let display = SequenceDisplayExtension {
            video_format: 5,
            colour: Some((1, 1, 1)),
            display_horizontal_size: self.width as u16,
            display_vertical_size: self.height as u16,
        };
        let mut w = BitWriter::new();
        header.write(&mut w);
        extension.write(&mut w);
        let extradata = w.clone().finish();
        display.write(&mut w);
        let headers = w.finish();
        let time_code_rate = u64::from(n.div_ceil(d));
        Ok(Stream {
            rate: Rate::new(RateConfig {
                bitrate,
                peak,
                vbv_bits,
                frame_rate,
                constant: self.constant,
            }),
            headers,
            extradata,
            gop_size: (time_code_rate as u32 / 2).max(1),
            time_code_rate,
        })
    }

    /// Copies the visible picture into coded-size planes, repeating the
    /// right column and bottom row into the padding.
    fn pad(&self, picture: &Yuv420p) -> Frame {
        let mut frame = Frame::new(&self.geo, 0);
        let sources = [picture.y(), picture.u(), picture.v()];
        for (c, src) in sources.into_iter().enumerate() {
            let (w, h) = if c == 0 {
                (self.width as usize, self.height as usize)
            } else {
                (self.width as usize / 2, self.height as usize / 2)
            };
            let (stride, rows) = self.geo.plane(c);
            let plane = &mut frame.planes[c];
            for y in 0..rows {
                let line = &src[y.min(h - 1) * w..][..w];
                let dst = &mut plane[y * stride..][..stride];
                dst[..w].copy_from_slice(line);
                let last = line[w - 1];
                dst[w..].fill(last);
            }
        }
        frame
    }

    /// Packets in a GOP of `gop_size`: P and B pictures after the I.
    fn gop_counts(gop_size: u32) -> (u32, u32) {
        let m = B_FRAMES as u32 + 1;
        let p = (1..gop_size)
            .filter(|&k| k % m == 0 || k + 1 == gop_size)
            .count() as u32;
        (p, gop_size.saturating_sub(1 + p))
    }

    /// Ends the open GOP: the newest waiting picture becomes a P picture
    /// and the rest B pictures between it and the previous anchor.
    fn close_gop(&mut self, out: &mut Vec<Packet>) -> Result<(), CodecError> {
        if let Some(last) = self.pending.pop() {
            out.push(self.encode(PICTURE_P, last)?);
            self.flush_b(out)?;
        }
        Ok(())
    }

    fn flush_b(&mut self, out: &mut Vec<Packet>) -> Result<(), CodecError> {
        for source in std::mem::take(&mut self.pending) {
            out.push(self.encode(PICTURE_B, source)?);
        }
        Ok(())
    }

    /// Codes one picture, rebuilds it when later pictures predict from it,
    /// and returns its packet.
    fn encode(&mut self, kind: u8, source: Source) -> Result<Packet, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let Some(stream) = self.stream.as_mut() else {
            return Err(CodecError::InvalidApi {
                what: "picture before the stream opened",
            });
        };
        let geo = self.geo;
        let refs: [Option<&Frame>; 2] = match kind {
            PICTURE_I => [None, None],
            PICTURE_P => [self.refs[1].as_ref(), None],
            _ => [self.refs[0].as_ref(), self.refs[1].as_ref()],
        };
        if (kind == PICTURE_P && refs[0].is_none()) || (kind == PICTURE_B && refs[1].is_none()) {
            return Err(CodecError::InvalidApi {
                what: "picture without its reference",
            });
        }
        let mut prefix = Vec::new();
        if kind == PICTURE_I {
            let (p, b) = Self::gop_counts(stream.gop_size);
            stream.rate.start_gop(p, b);
            prefix.extend_from_slice(&stream.headers);
            let mut w = BitWriter::new();
            GopHeader {
                time_code: time_code(self.frames, stream.time_code_rate),
                closed_gop: true,
                broken_link: false,
            }
            .write(&mut w);
            prefix.extend(w.finish());
        }
        let macroblocks = geo.mb_width * geo.mb_height;
        let mut picture_rate = stream.rate.begin(kind, macroblocks);
        let analysis = analyse(&geo, kind, &source.frame, refs, picture_rate.base_quant());
        let mut retries = VBV_RETRIES.iter();
        let coded = loop {
            let job = Job {
                geo,
                kind,
                temporal_reference: source.temporal_reference,
                vbv_delay: stream.rate.vbv_delay(),
                src: &source.frame,
                refs,
                quant: &self.quant,
            };
            picture_rate.restart();
            let coded = write_picture(&job, &analysis, &mut picture_rate)?;
            let bits = ((prefix.len() + coded.data.len()) * 8) as f64;
            if bits > stream.rate.max_bits()
                && let Some(&offset) = retries.next()
            {
                picture_rate.offset = offset;
                continue;
            }
            break coded;
        };
        let mut data = prefix;
        data.extend_from_slice(&coded.data);
        let short = stream.rate.min_bits() - (data.len() * 8) as f64;
        if short > 0.0 {
            // Zero bytes before the next start code are stuffing (6.2.1).
            data.resize(data.len() + (short / 8.0).ceil() as usize, 0);
        }
        stream.rate.end(&picture_rate, (data.len() * 8) as f64);
        if kind != PICTURE_B {
            let recon = reconstruct(&geo, kind, &coded, analysis.f_code, refs)?;
            self.refs[0] = self.refs[1].take();
            self.refs[1] = Some(recon);
        }
        Ok(Packet::video(
            data,
            source.pts,
            source.duration,
            source.timescale,
            kind == PICTURE_I,
        ))
    }

    /// Keeps the newest packet back and releases the one held before.
    fn release(&mut self, mut packets: Vec<Packet>) -> Vec<Packet> {
        let Some(newest) = packets.pop() else {
            return packets;
        };
        let mut out = Vec::with_capacity(packets.len() + 1);
        out.extend(self.held.replace(newest));
        out.extend(packets);
        out
    }
}

/// `time_code` of 6.3.8 for picture `frames`, without drop-frame counting.
fn time_code(frames: u64, rate: u64) -> u32 {
    let rate = rate.max(1);
    let seconds = frames / rate;
    let pictures = (frames % rate) as u32;
    let hours = ((seconds / 3600) % 24) as u32;
    let minutes = ((seconds / 60) % 60) as u32;
    let seconds = (seconds % 60) as u32;
    (hours << 19) | (minutes << 13) | (1 << 12) | (seconds << 6) | pictures
}

/// Decodes the written picture exactly as a decoder will, so later
/// pictures predict from the same samples.
fn reconstruct(
    geo: &Geometry,
    kind: u8,
    coded: &Coded,
    f_code: [[u8; 2]; 2],
    refs: [Option<&Frame>; 2],
) -> Result<Frame, CodecError> {
    let matrices = [
        DEFAULT_INTRA,
        DEFAULT_NON_INTRA,
        DEFAULT_INTRA,
        DEFAULT_NON_INTRA,
    ];
    let picture = Picture {
        geo: *geo,
        params: decode_params(kind, f_code, geo.mb_height * 16),
        matrices: &matrices,
        fwd: refs[0],
        bwd: refs[1],
    };
    let slices: Vec<SliceData<'_>> = coded
        .slices
        .iter()
        .map(|&(vertical_position, offset)| SliceData {
            vertical_position,
            data: &coded.data[offset..],
            at_bit: 0,
        })
        .collect();
    let mut frame = Frame::new(geo, 0);
    decode_picture(&picture, &mut frame, &slices, None)?;
    Ok(frame)
}

impl VideoEncoder for Mpeg2Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg2
    }

    fn push(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if input.picture.width() != self.width || input.picture.height() != self.height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        if input.timescale == 0 {
            return Err(CodecError::InvalidApi { what: "timescale" });
        }
        if input.duration < 0 {
            return Err(CodecError::InvalidApi { what: "duration" });
        }
        let duration = input.duration.max(1);
        if self.stream.is_none() {
            self.stream = Some(self.open_stream(duration, input.timescale)?);
        }
        let gop_size = self.stream.as_ref().map_or(1, |stream| stream.gop_size);
        let luma = histogram(input.picture.y(), input.picture.width() as usize);
        let cut = self
            .last_luma
            .as_deref()
            .is_some_and(|last| scene_cut(last, &luma));
        self.drained = false;
        let mut source = Source {
            frame: self.pad(&input.picture),
            pts: input.pts,
            duration,
            timescale: input.timescale,
            temporal_reference: 0,
        };
        let mut out = Vec::new();
        if input.force_keyframe || cut || self.refs[1].is_none() || self.gop_pos >= gop_size {
            self.close_gop(&mut out)?;
            out.push(self.encode(PICTURE_I, source)?);
            self.gop_pos = 1;
        } else {
            source.temporal_reference = self.gop_pos as u16;
            self.gop_pos += 1;
            if self.pending.len() >= B_FRAMES || self.gop_pos >= gop_size {
                out.push(self.encode(PICTURE_P, source)?);
                self.flush_b(&mut out)?;
            } else {
                self.pending.push(source);
            }
        }
        self.frames += 1;
        self.last_luma = Some(luma);
        Ok(self.release(out))
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if self.drained {
            return Ok(Vec::new());
        }
        let mut tail = Vec::new();
        self.close_gop(&mut tail)?;
        self.drained = true;
        let mut out: Vec<Packet> = self.held.take().into_iter().chain(tail).collect();
        if let Some(last) = out.last_mut() {
            last.data.extend_from_slice(&[0, 0, 1, SEQUENCE_END]);
        }
        // The sequence ended: the next picture starts a new one.
        self.refs = [None, None];
        self.gop_pos = 0;
        Ok(out)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.refs = [None, None];
        self.held = None;
        self.gop_pos = 0;
        self.frames = 0;
        self.drained = false;
        self.last_luma = None;
        if let Some(stream) = self.stream.as_mut() {
            stream.rate.reset();
        }
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    fn extradata(&self) -> &[u8] {
        self.stream
            .as_ref()
            .map_or(&[][..], |stream| &stream.extradata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_codes_count_hours_minutes_seconds_and_pictures() {
        let code = time_code(25 * 3661 + 7, 25);
        assert_eq!(code >> 19, 1);
        assert_eq!((code >> 13) & 63, 1);
        assert_eq!((code >> 12) & 1, 1);
        assert_eq!((code >> 6) & 63, 1);
        assert_eq!(code & 63, 7);
    }

    #[test]
    fn gop_counts_end_on_an_anchor() {
        // I B B P B B P B B P B P for twelve pictures.
        assert_eq!(Mpeg2Encoder::gop_counts(12), (4, 7));
        assert_eq!(Mpeg2Encoder::gop_counts(1), (0, 0));
        assert_eq!(Mpeg2Encoder::gop_counts(2), (1, 0));
    }
}
