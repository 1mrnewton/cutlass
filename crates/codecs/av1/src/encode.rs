//! Public AV1 encoder over the shared [`Encoder`] trait.
//!
//! The coding itself lives in [`crate::enc`]. This type maps the input pixel
//! format to the AV1 bit depth and subsampling and the colour names of
//! [`CodecParams`] to the sequence header's codes, starts the encoder on the
//! first frame (when the frame rate for a bitrate target is known), and wraps
//! each temporal unit in a [`Packet`]. The first packet carries the `av1C`
//! record as [`SideData::Config`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorRange, MatrixCoefficients, Packet, PixelFormat, SideData,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput};

use crate::enc::encoder::{Av1Enc, EncConfig, EncodedUnit, InputFrame, RateControl};
use crate::yuv::VideoInput;

/// Rust AV1 encoder. `bitrate` of zero keeps the quality index.
pub struct Av1Encoder {
    width: u32,
    height: u32,
    pixel: PixelFormat,
    bitrate: u32,
    qindex: u8,
    speed: u8,
    gop: u32,
    keyint: u32,
    loop_restoration: bool,
    global_motion: bool,
    temporal_filter: bool,
    threads: usize,
    color: ColorTags,
    enc: Option<Av1Enc>,
    /// The `av1C` record of `enc` has gone out on a packet.
    config_sent: bool,
    /// The next picture pushed is a key frame.
    force_key: bool,
    cancel: Arc<AtomicBool>,
    /// Timestamps of queued pictures, by display index.
    times: std::collections::VecDeque<(i64, i64, u32)>,
}

/// Bit depth, horizontal and vertical subsampling, and whether the format
/// is one grey plane, which is coded as monochrome.
fn format_of(pixel: PixelFormat) -> Option<(u32, bool, bool, bool)> {
    Some(match pixel {
        PixelFormat::Yuv420p => (8, true, true, false),
        PixelFormat::Yuv422p => (8, true, false, false),
        PixelFormat::Yuv444p | PixelFormat::Gbrp => (8, false, false, false),
        PixelFormat::Yuv420p10 => (10, true, true, false),
        PixelFormat::Yuv422p10 => (10, true, false, false),
        PixelFormat::Yuv444p10 | PixelFormat::Gbrp10 => (10, false, false, false),
        PixelFormat::Yuv420p12 => (12, true, true, false),
        PixelFormat::Yuv422p12 => (12, true, false, false),
        PixelFormat::Yuv444p12 | PixelFormat::Gbrp12 => (12, false, false, false),
        PixelFormat::Gray8 => (8, true, true, true),
        PixelFormat::Gray10 => (10, true, true, true),
        _ => return None,
    })
}

/// The colour codes (ISO/IEC 23091-2) and range flag of a sequence header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColorTags {
    primaries: u8,
    transfer: u8,
    matrix: u8,
    full_range: bool,
}

fn is_gbr(pixel: PixelFormat) -> bool {
    matches!(
        pixel,
        PixelFormat::Gbrp | PixelFormat::Gbrp10 | PixelFormat::Gbrp12
    )
}

/// What the sequence header says about the colour of `params`.
///
/// GBR planes are the identity matrix. Without names they are sRGB, full
/// range, as the format's 4:4:4 sRGB case is (AV1 5.5.2). Other planes keep
/// the names given, and an unnamed range is limited, the flag's zero. The
/// identity matrix on planes that are not GBR is refused.
fn color_tags(params: &CodecParams) -> Result<ColorTags, CodecError> {
    let limited = matches!(params.range, ColorRange::Limited | ColorRange::Studio);
    if is_gbr(params.pixel) {
        let primaries = match params.primaries.cicp() {
            2 => 1,
            named => named,
        };
        let transfer = match params.transfer.cicp() {
            2 => 13,
            named => named,
        };
        // BT.709 primaries with the sRGB curve and the identity matrix are
        // full range by the syntax: the flag is not coded.
        let srgb = primaries == 1 && transfer == 13;
        return Ok(ColorTags {
            primaries,
            transfer,
            matrix: 0,
            full_range: srgb || !limited,
        });
    }
    if params.matrix == MatrixCoefficients::Rgb {
        return Err(CodecError::Unsupported {
            what: "identity matrix on planes that are not GBR",
        });
    }
    Ok(ColorTags {
        primaries: params.primaries.cicp(),
        transfer: params.transfer.cicp(),
        matrix: params.matrix.cicp(),
        full_range: params.range == ColorRange::Full,
    })
}

impl Av1Encoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Av1 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if format_of(params.pixel).is_none() {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let color = color_tags(params)?;
        if params.width == 0 || params.height == 0 || params.width > 16384 || params.height > 16384
        {
            return Err(CodecError::Limit { what: "frame size" });
        }
        Ok(Self {
            width: params.width,
            height: params.height,
            pixel: params.pixel,
            bitrate: params.bitrate,
            qindex: 100,
            speed: 6,
            gop: 8,
            keyint: 120,
            loop_restoration: false,
            global_motion: false,
            temporal_filter: false,
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            color,
            enc: None,
            config_sent: false,
            force_key: false,
            cancel: Arc::new(AtomicBool::new(false)),
            times: std::collections::VecDeque::new(),
        })
    }

    /// Quantizer index used when `bitrate` is zero. Zero is raised to one.
    pub fn set_quality(&mut self, qindex: u8) {
        self.qindex = qindex.clamp(1, 255);
        self.enc = None;
    }

    /// Mini-GOP length. 1 gives one packet per picture with no delay; longer
    /// GOPs reorder frames behind a hidden alt-ref (packets come out later,
    /// and `drain` returns the rest).
    pub fn set_gop(&mut self, gop: u32) {
        self.gop = gop.clamp(1, 32);
        self.enc = None;
    }

    /// Worker threads (tile-parallel coding; frames 1280 wide and up are
    /// split into tile columns automatically). Output does not depend on it
    /// for a given tiling.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, 64);
        self.enc = None;
    }

    /// Speed preset: 0 is slowest and best, 10 fastest.
    pub fn set_speed(&mut self, speed: u8) {
        self.speed = speed.min(10);
        self.enc = None;
    }

    /// Pictures from one key frame to the next (120 by default; 1 makes
    /// every picture a key frame). Zero is raised to one.
    pub fn set_keyint(&mut self, pictures: u32) {
        self.keyint = pictures.max(1);
        self.enc = None;
    }

    /// Codes Wiener loop restoration where it lowers the error (off by
    /// default; speeds 8 and up leave it off).
    pub fn set_loop_restoration(&mut self, on: bool) {
        self.loop_restoration = on;
        self.enc = None;
    }

    /// Estimates a global motion model for each reference (off by default).
    pub fn set_global_motion(&mut self, on: bool) {
        self.global_motion = on;
        self.enc = None;
    }

    /// Filters every hidden alt-ref with its neighbours (off by default,
    /// when only the alt-refs of still content are filtered).
    pub fn set_temporal_filter(&mut self, on: bool) {
        self.temporal_filter = on;
        self.enc = None;
    }

    /// Makes the next picture pushed a key frame. The key frame distance is
    /// counted from it.
    pub fn force_keyframe(&mut self) {
        self.force_key = true;
    }

    fn encoder(&mut self, duration: i64, timescale: u32) -> Result<&mut Av1Enc, CodecError> {
        if self.enc.is_none() {
            let Some((bd, sx, sy, mono)) = format_of(self.pixel) else {
                return Err(CodecError::Unsupported {
                    what: "pixel format",
                });
            };
            let mut cfg = EncConfig::new(self.width as usize, self.height as usize);
            cfg.bit_depth = bd;
            cfg.sub_x = sx;
            cfg.sub_y = sy;
            cfg.mono = mono;
            cfg.speed = self.speed;
            cfg.gop = self.gop;
            cfg.keyint = self.keyint;
            cfg.loop_restoration = self.loop_restoration;
            cfg.global_motion = self.global_motion;
            cfg.temporal_filter = self.temporal_filter;
            cfg.threads = self.threads;
            cfg.color_primaries = self.color.primaries;
            cfg.transfer = self.color.transfer;
            cfg.matrix = self.color.matrix;
            cfg.full_range = self.color.full_range;
            let fps = if duration > 0 && timescale > 0 {
                f64::from(timescale) / duration as f64
            } else {
                30.0
            };
            cfg.fps = fps;
            cfg.rc = if self.bitrate > 0 {
                RateControl::Bitrate {
                    bps: u64::from(self.bitrate),
                    fps,
                }
            } else {
                RateControl::Qindex(self.qindex)
            };
            self.enc = Some(Av1Enc::new(cfg)?);
            self.config_sent = false;
        }
        self.enc
            .as_mut()
            .ok_or(CodecError::InvalidApi { what: "encoder" })
    }

    /// Encodes tightly packed planes in `self.pixel` layout.
    fn encode_packed(
        &mut self,
        bytes: &[u8],
        pts: i64,
        duration: i64,
        timescale: u32,
    ) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let (w, h) = (self.width as usize, self.height as usize);
        // One grey plane has no chroma planes.
        let (cw, ch) = self
            .pixel
            .chroma_wh(self.width, self.height)
            .map_or((0, 0), |(cw, ch)| (cw as usize, ch as usize));
        let wide = format_of(self.pixel).is_some_and(|f| f.0 > 8);
        let need = (w * h + 2 * cw * ch) * if wide { 2 } else { 1 };
        if bytes.len() != need {
            return Err(CodecError::InvalidApi {
                what: "picture byte length",
            });
        }
        let force = std::mem::take(&mut self.force_key);
        let enc = self.encoder(duration, timescale)?;
        if force {
            enc.force_keyframe();
        }
        let data = if wide {
            let s: Vec<u16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            let (y, rest) = s.split_at(w * h);
            let (u, v) = rest.split_at(cw * ch);
            enc.encode(&InputFrame {
                planes: [y, u, v],
                strides: [w, cw, cw],
            })?
        } else {
            let (y, rest) = bytes.split_at(w * h);
            let (u, v) = rest.split_at(cw * ch);
            enc.encode(&InputFrame {
                planes: [y, u, v],
                strides: [w, cw, cw],
            })?
        };
        self.times.push_back((pts, duration, timescale));
        Ok(self.packets(data))
    }

    /// Wraps temporal units in packets with their pictures' timestamps. The
    /// first packet of an encoder carries its `av1C` record.
    fn packets(&mut self, units: Vec<EncodedUnit>) -> Vec<Packet> {
        let mut packets = Vec::with_capacity(units.len());
        for u in units {
            let (pts, duration, timescale) = self.times.pop_front().unwrap_or((0, 0, 1));
            let key = u.key || is_key(&u.data);
            let mut packet = Packet::video(u.data, pts, duration, timescale, key);
            if !self.config_sent
                && let Some(enc) = &self.enc
            {
                packet.side.push(SideData::Config(enc.config_record()));
                self.config_sent = true;
            }
            packets.push(packet);
        }
        packets
    }

    /// The `av1C` record of the stream being written (the bytes an ISOBMFF
    /// sample entry or a Matroska `CodecPrivate` holds), once the first
    /// picture has been pushed. The first packet carries the same bytes.
    #[must_use]
    pub fn config_record(&self) -> Option<Vec<u8>> {
        self.enc.as_ref().map(Av1Enc::config_record)
    }

    /// Encode one 4:2:0 picture given as separate planes.
    pub fn push_picture(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.pixel != PixelFormat::Yuv420p
            || input.picture.y.width() != self.width
            || input.picture.y.height() != self.height
        {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        let mut bytes = Vec::with_capacity(input.picture.y.samples().len() * 3 / 2);
        bytes.extend_from_slice(input.picture.y.samples());
        bytes.extend_from_slice(input.picture.u.samples());
        bytes.extend_from_slice(input.picture.v.samples());
        self.encode_packed(&bytes, input.pts, input.duration, input.timescale)
    }
}

/// True when the temporal unit carries a sequence header (the encoder emits
/// one with every key frame).
fn is_key(data: &[u8]) -> bool {
    // The second OBU follows the 2-byte temporal delimiter.
    data.get(2).is_some_and(|h| (h >> 3) & 15 == 1)
}

impl Encoder for Av1Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "audio" })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        if frame.pixel() != self.pixel
            || frame.width() != self.width
            || frame.height() != self.height
        {
            return Err(CodecError::InvalidApi {
                what: "frame format",
            });
        }
        let time = frame.time();
        self.encode_packed(frame.bytes(), time.pts(), time.duration(), time.timescale())
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        let wide = format_of(self.pixel).is_some_and(|f| f.0 > 8);
        let units = match self.enc.as_mut() {
            None => return Ok(Vec::new()),
            Some(e) if wide => e.flush::<u16>()?,
            Some(e) => e.flush::<u8>()?,
        };
        Ok(self.packets(units))
    }

    fn reset(&mut self) {
        self.enc = None;
        self.config_sent = false;
        self.force_key = false;
        self.times.clear();
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_decode::{
        CodecError, CodecId, CodecParams, ColorPrimaries, ColorRange, Decoder, MatrixCoefficients,
        Output, Packet, PixelFormat, SideData, Transfer,
    };
    use cutlass_encode::{CpuFrame, Encoder, FrameInput, FrameTime};

    use super::Av1Encoder;
    use crate::dec::obu::{OBU_SEQUENCE_HEADER, parse_sequence_header, split_obus};
    use crate::decode::Av1Decoder;

    /// A moving ramp in `pixel` layout, `index` pictures in.
    fn frame(pixel: PixelFormat, width: u32, height: u32, index: u32) -> FrameInput {
        // One grey plane has no chroma.
        let (cw, ch) = pixel.chroma_wh(width, height).unwrap_or((0, 0));
        let wide = !matches!(
            pixel,
            PixelFormat::Yuv420p
                | PixelFormat::Yuv422p
                | PixelFormat::Yuv444p
                | PixelFormat::Gbrp
                | PixelFormat::Gray8
        );
        let mut bytes = Vec::new();
        for (w, h) in [(width, height), (cw, ch), (cw, ch)] {
            for y in 0..h {
                for x in 0..w {
                    let v = (x * 3 + y * 2 + index * 5) % 200 + 20;
                    if wide {
                        bytes.extend_from_slice(&((v as u16) << 2).to_le_bytes());
                    } else {
                        bytes.push(v as u8);
                    }
                }
            }
        }
        let time = FrameTime::new(i64::from(index), 1, 30).expect("time");
        FrameInput::Cpu(CpuFrame::new(width, height, pixel, bytes, time).expect("frame"))
    }

    fn encode(encoder: &mut Av1Encoder, pixel: PixelFormat, frames: u32) -> Vec<Packet> {
        let mut packets = Vec::new();
        for i in 0..frames {
            packets.extend(encoder.push_frame(&frame(pixel, 64, 48, i)).expect("push"));
        }
        packets.extend(encoder.drain().expect("drain"));
        packets
    }

    fn config(packet: &Packet) -> Option<&[u8]> {
        packet.side.iter().find_map(|side| match side {
            SideData::Config(config) => Some(config.as_slice()),
            _ => None,
        })
    }

    fn pictures(decoder: &mut Av1Decoder, packets: &[Packet]) -> Vec<cutlass_decode::PictureInfo> {
        let mut out = Vec::new();
        for p in packets {
            out.extend(decoder.push(p).expect("decode"));
        }
        out.extend(decoder.drain().expect("drain"));
        out.iter()
            .map(|o| match o {
                Output::Picture(p) => p.info(),
                _ => panic!("a picture"),
            })
            .collect()
    }

    #[test]
    fn the_first_packet_carries_the_av1c_record() {
        let params = CodecParams::video(CodecId::Av1, 64, 48);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(4);
        assert!(encoder.config_record().is_none());
        let packets = encode(&mut encoder, PixelFormat::Yuv420p, 6);
        assert_eq!(packets.len(), 6);
        let record = config(&packets[0]).expect("config on the first packet");
        assert!(packets[1..].iter().all(|p| config(p).is_none()));
        assert_eq!(encoder.config_record().as_deref(), Some(record));

        // Marker and version, then profile 0, 8-bit 4:2:0, no delay.
        assert_eq!(record[0], 0x81);
        assert_eq!(record[1] >> 5, 0);
        assert_eq!(record[2] & 0x7c, 0b0000_1100);
        assert_eq!(record[3], 0);
        // The configuration OBU is the key frame's sequence header.
        let obus = split_obus(&record[4..]).expect("config OBUs");
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, OBU_SEQUENCE_HEADER);
        let in_stream = split_obus(&packets[0].data).expect("first unit");
        let stream_header = in_stream
            .iter()
            .find(|o| o.obu_type == OBU_SEQUENCE_HEADER)
            .expect("sequence header in the key frame");
        assert_eq!(obus[0].payload, stream_header.payload);
        let seq = parse_sequence_header(obus[0].payload).expect("sequence header");
        assert_eq!(record[1] & 31, seq.operating_points[0].seq_level_idx);
        // 64x48 at 30 pictures a second is level 2.0, index 0.
        assert_eq!(record[1] & 31, 0);

        // A decoder opened with the record decodes the stream.
        let mut with_record = params.clone();
        with_record.extra = record.to_vec();
        let mut decoder = Av1Decoder::open(&with_record).expect("decoder");
        assert_eq!(pictures(&mut decoder, &packets).len(), 6);
    }

    #[test]
    fn the_av1c_record_names_the_bit_depth_and_chroma_layout() {
        for (pixel, profile, bits) in [
            (PixelFormat::Yuv420p10, 0, 0b0100_1100),
            (PixelFormat::Yuv444p, 1, 0b0000_0000),
            (PixelFormat::Yuv422p, 2, 0b0000_1000),
            (PixelFormat::Yuv422p12, 2, 0b0110_1000),
        ] {
            let mut params = CodecParams::video(CodecId::Av1, 64, 48);
            params.pixel = pixel;
            let mut encoder = Av1Encoder::open(&params).expect("encoder");
            encoder.set_gop(1);
            let packets = encode(&mut encoder, pixel, 1);
            let record = config(&packets[0]).expect("config");
            assert_eq!(record[1] >> 5, profile, "{pixel:?}");
            assert_eq!(record[2] & 0x7c, bits, "{pixel:?}");
        }
    }

    #[test]
    fn a_reset_sends_the_record_again() {
        let params = CodecParams::video(CodecId::Av1, 64, 48);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(1);
        let first = encode(&mut encoder, PixelFormat::Yuv420p, 2);
        encoder.reset();
        let second = encode(&mut encoder, PixelFormat::Yuv420p, 2);
        assert!(config(&first[0]).is_some());
        assert_eq!(config(&first[0]), config(&second[0]));
        assert!(config(&second[1]).is_none());
    }

    #[test]
    fn the_color_names_reach_the_decoded_picture() {
        let mut params = CodecParams::video(CodecId::Av1, 64, 48);
        params.pixel = PixelFormat::Yuv420p10;
        params.primaries = ColorPrimaries::Bt2020;
        params.transfer = Transfer::Pq;
        params.matrix = MatrixCoefficients::Bt2020;
        params.range = ColorRange::Limited;
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(1);
        let packets = encode(&mut encoder, PixelFormat::Yuv420p10, 1);
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let info = pictures(&mut decoder, &packets)[0];
        assert_eq!(
            (info.cicp_primaries, info.cicp_transfer, info.cicp_matrix),
            (9, 16, 9)
        );
        assert_eq!(info.primaries, ColorPrimaries::Bt2020);
        assert_eq!(info.transfer, Transfer::Pq);
        let seq = sequence_of(&packets[0]);
        assert!(!seq.color_range);
    }

    fn sequence_of(packet: &Packet) -> crate::dec::obu::SequenceHeader {
        let obus = split_obus(&packet.data).expect("unit");
        let header = obus
            .iter()
            .find(|o| o.obu_type == OBU_SEQUENCE_HEADER)
            .expect("sequence header");
        parse_sequence_header(header.payload).expect("parse")
    }

    #[test]
    fn a_full_range_and_an_unnamed_stream_say_so() {
        let mut params = CodecParams::video(CodecId::Av1, 64, 48);
        params.range = ColorRange::Full;
        params.matrix = MatrixCoefficients::Bt601;
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(1);
        let seq = sequence_of(&encode(&mut encoder, PixelFormat::Yuv420p, 1)[0]);
        assert!(seq.color_range);
        assert_eq!(
            (
                seq.color_primaries,
                seq.transfer_characteristics,
                seq.matrix_coefficients
            ),
            (2, 2, 6)
        );

        let unnamed = CodecParams::video(CodecId::Av1, 64, 48);
        let mut encoder = Av1Encoder::open(&unnamed).expect("encoder");
        encoder.set_gop(1);
        let seq = sequence_of(&encode(&mut encoder, PixelFormat::Yuv420p, 1)[0]);
        assert!(!seq.color_range);
        assert_eq!(
            (
                seq.color_primaries,
                seq.transfer_characteristics,
                seq.matrix_coefficients
            ),
            (2, 2, 2)
        );
    }

    #[test]
    fn gbr_planes_are_the_identity_matrix() {
        // Unnamed GBR is sRGB, full range.
        let mut params = CodecParams::video(CodecId::Av1, 64, 48);
        params.pixel = PixelFormat::Gbrp;
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(1);
        let seq = sequence_of(&encode(&mut encoder, PixelFormat::Gbrp, 1)[0]);
        assert_eq!(
            (
                seq.color_primaries,
                seq.transfer_characteristics,
                seq.matrix_coefficients
            ),
            (1, 13, 0)
        );
        assert!(seq.color_range);

        // Named GBR keeps its primaries and curve.
        params.pixel = PixelFormat::Gbrp10;
        params.primaries = ColorPrimaries::Bt2020;
        params.transfer = Transfer::Hlg;
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(1);
        let packets = encode(&mut encoder, PixelFormat::Gbrp10, 1);
        let seq = sequence_of(&packets[0]);
        assert_eq!(
            (
                seq.color_primaries,
                seq.transfer_characteristics,
                seq.matrix_coefficients
            ),
            (9, 18, 0)
        );
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        assert_eq!(
            pictures(&mut decoder, &packets)[0].pixel,
            PixelFormat::Gbrp10
        );
    }

    #[test]
    fn the_identity_matrix_on_yuv_planes_is_refused() {
        let mut params = CodecParams::video(CodecId::Av1, 64, 48);
        params.matrix = MatrixCoefficients::Rgb;
        assert!(matches!(
            Av1Encoder::open(&params),
            Err(CodecError::Unsupported { .. })
        ));
    }

    #[test]
    fn the_key_frame_distance_is_a_setting() {
        let params = CodecParams::video(CodecId::Av1, 64, 48);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(2);
        encoder.set_keyint(3);
        let packets = encode(&mut encoder, PixelFormat::Yuv420p, 8);
        let keys: Vec<usize> = (0..8).filter(|&i| packets[i].keyframe).collect();
        assert_eq!(keys, [0, 3, 6]);
        // Zero is one: every picture is a key frame.
        encoder.set_keyint(0);
        let packets = encode(&mut encoder, PixelFormat::Yuv420p, 3);
        assert!(packets.iter().all(|p| p.keyframe));
    }

    #[test]
    fn loop_restoration_is_a_setting() {
        let params = CodecParams::video(CodecId::Av1, 64, 48);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(1);
        let off = encode(&mut encoder, PixelFormat::Yuv420p, 2);
        assert!(!sequence_of(&off[0]).enable_restoration);
        encoder.set_loop_restoration(true);
        let on = encode(&mut encoder, PixelFormat::Yuv420p, 2);
        assert!(sequence_of(&on[0]).enable_restoration);
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        assert_eq!(pictures(&mut decoder, &on).len(), 2);
    }

    /// Pictures of noise that changes from one to the next, so that no
    /// mini-GOP is found still and filtered without the setting.
    fn encode_noise(encoder: &mut Av1Encoder, frames: u32) -> Vec<Packet> {
        let mut state = 0x2545_f491u32;
        let mut packets = Vec::new();
        for i in 0..frames {
            let bytes: Vec<u8> = (0..64 * 48 * 3 / 2)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 24) as u8
                })
                .collect();
            let time = FrameTime::new(i64::from(i), 1, 30).expect("time");
            let frame = CpuFrame::new(64, 48, PixelFormat::Yuv420p, bytes, time).expect("frame");
            packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("push"));
        }
        packets.extend(encoder.drain().expect("drain"));
        packets
    }

    #[test]
    fn global_motion_and_the_temporal_filter_are_settings() {
        let params = CodecParams::video(CodecId::Av1, 64, 48);
        let mut plain = Av1Encoder::open(&params).expect("encoder");
        plain.set_gop(4);
        let base = encode_noise(&mut plain, 9);
        for setting in 0..2 {
            let mut encoder = Av1Encoder::open(&params).expect("encoder");
            encoder.set_gop(4);
            if setting == 0 {
                encoder.set_global_motion(true);
            } else {
                encoder.set_temporal_filter(true);
            }
            let packets = encode_noise(&mut encoder, 9);
            assert_eq!(packets.len(), 9);
            let mut decoder = Av1Decoder::open(&params).expect("decoder");
            assert_eq!(pictures(&mut decoder, &packets).len(), 9, "{setting}");
            if setting == 1 {
                // A filtered alt-ref is not the picture shown: the stream
                // differs from the one without the filter.
                let bytes =
                    |ps: &[Packet]| -> Vec<u8> { ps.iter().flat_map(|p| p.data.clone()).collect() };
                assert_ne!(bytes(&packets), bytes(&base));
            }
        }
    }

    #[test]
    fn a_grey_plane_is_coded_as_monochrome() {
        for (pixel, out) in [
            (PixelFormat::Gray8, PixelFormat::Yuv420p),
            (PixelFormat::Gray10, PixelFormat::Yuv420p10),
        ] {
            let mut params = CodecParams::video(CodecId::Av1, 64, 48);
            params.pixel = pixel;
            let mut encoder = Av1Encoder::open(&params).expect("encoder");
            encoder.set_gop(2);
            encoder.set_quality(40);
            let packets = encode(&mut encoder, pixel, 3);
            assert_eq!(packets.len(), 3);
            let seq = sequence_of(&packets[0]);
            assert!(seq.mono_chrome);
            // The record says monochrome, with both chroma bits set.
            let record = config(&packets[0]).expect("config");
            assert_eq!(record[2] & 0b0001_1100, 0b0001_1100);
            // The decoder shows 4:2:0 with neutral chroma and the luma back.
            let mut decoder = Av1Decoder::open(&params).expect("decoder");
            let mut shown = Vec::new();
            for p in &packets {
                shown.extend(decoder.push(p).expect("decode"));
            }
            shown.extend(decoder.drain().expect("drain"));
            assert_eq!(shown.len(), 3);
            let Output::Picture(first) = &shown[0] else {
                panic!("a picture");
            };
            assert_eq!(first.info().pixel, out);
            let cutlass_decode::PictureSamples::Cpu(bytes) = first.samples() else {
                panic!("cpu samples");
            };
            let FrameInput::Cpu(source) = frame(pixel, 64, 48, 0) else {
                panic!("cpu frame");
            };
            let luma = source.bytes();
            let step = if pixel == PixelFormat::Gray8 { 1 } else { 2 };
            let sample = |b: &[u8], i: usize| -> i32 {
                if step == 1 {
                    i32::from(b[i])
                } else {
                    i32::from(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
                }
            };
            let tolerance = if step == 1 { 6 } else { 24 };
            for i in 0..64 * 48 {
                assert!((sample(bytes, i) - sample(luma, i)).abs() <= tolerance);
            }
            let neutral = if step == 1 { 128 } else { 512 };
            for i in 64 * 48..64 * 48 + 2 * 32 * 24 {
                assert_eq!(sample(bytes, i), neutral);
            }
        }
    }

    #[test]
    fn a_forced_key_frame_starts_a_stream_a_decoder_can_enter() {
        let params = CodecParams::video(CodecId::Av1, 64, 48);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_gop(4);
        let mut packets = Vec::new();
        for i in 0..10 {
            if i == 5 {
                encoder.force_keyframe();
            }
            packets.extend(
                encoder
                    .push_frame(&frame(PixelFormat::Yuv420p, 64, 48, i))
                    .expect("push"),
            );
        }
        packets.extend(encoder.drain().expect("drain"));
        assert_eq!(packets.len(), 10);
        let keys: Vec<usize> = (0..10).filter(|&i| packets[i].keyframe).collect();
        assert_eq!(keys, [0, 5]);
        for (i, p) in packets.iter().enumerate() {
            assert_eq!(p.pts, i as i64);
        }
        // Decoding can start at the forced key frame.
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        assert_eq!(pictures(&mut decoder, &packets[5..]).len(), 5);
        let mut whole = Av1Decoder::open(&params).expect("decoder");
        assert_eq!(pictures(&mut whole, &packets).len(), 10);
    }
}
