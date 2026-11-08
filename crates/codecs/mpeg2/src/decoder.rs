//! Sequence state, picture management and display order.
//!
//! One `push` takes one access unit: usually one frame, or both fields of
//! one frame, with whatever sequence and GOP headers precede it. A field
//! whose pair arrives in the next packet waits for it. I and P frames are
//! shown when the next I or P frame arrives (7.12), B frames at once.

use std::sync::Arc;

use cutlass_decode::{
    CodecError, ColorMatrix, ColorPrimaries, ColorRange, Packet, SampleAspect, SequenceTiming,
    Transfer, VideoOutput, Yuv420p, Yuv422p, Yuv444p,
};

use crate::bits::start_codes;
use crate::frame::{Frame, Geometry};
use crate::headers::{
    CHROMA_420, CHROMA_422, EXT_PICTURE_CODING, EXT_PICTURE_SPATIAL, EXT_PICTURE_TEMPORAL,
    EXT_QUANT_MATRIX, EXT_SEQUENCE, EXT_SEQUENCE_DISPLAY, EXT_SEQUENCE_SCALABLE, EXTENSION, GROUP,
    GopHeader, Matrix, PICTURE_B, PICTURE_D, PICTURE_I, PICTURE_P, PICTURE_START,
    PictureCodingExtension, PictureHeader, QuantMatrixExtension, SEQUENCE_END, SEQUENCE_ERROR,
    SEQUENCE_HEADER, SLICE_FIRST, SLICE_LAST, STRUCTURE_FRAME, STRUCTURE_TOP,
    SequenceDisplayExtension, SequenceExtension, SequenceHeader, USER_DATA, extension_id,
};
use crate::slice::{Picture, PictureParams, SliceData, decode_picture};
use crate::tables::{DEFAULT_INTRA, DEFAULT_NON_INTRA, FRAME_RATES};

/// Largest coded picture the engine opens by default, in luma samples.
pub const DEFAULT_MAX_PIXELS: u64 = 8192 * 8192;

/// What a decoded frame reports, fixed when its first picture was decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameInfo {
    geo: Geometry,
    width: u32,
    height: u32,
    colour: Option<(u8, u8, u8)>,
    sample_aspect: SampleAspect,
    frame_rate: Option<(u32, u32)>,
    pts: i64,
    duration: i64,
    timescale: u32,
    keyframe: bool,
}

#[derive(Clone, Debug)]
struct Decoded {
    frame: Arc<Frame>,
    info: FrameInfo,
}

/// A first field waiting for its pair. `frame` is `None` when that field
/// was dropped, so its pair is dropped too.
#[derive(Clone, Debug)]
struct OpenField {
    frame: Option<Arc<Frame>>,
    info: FrameInfo,
    parity: usize,
    coding_type: u8,
}

#[derive(Clone, Debug)]
struct Sequence {
    header: SequenceHeader,
    extension: Option<SequenceExtension>,
    display: Option<SequenceDisplayExtension>,
    geo: Geometry,
    width: u32,
    height: u32,
}

impl Sequence {
    fn mpeg1(&self) -> bool {
        self.extension.is_none()
    }

    fn frame_rate(&self) -> Option<(u32, u32)> {
        let (n, d) = *FRAME_RATES.get(usize::from(self.header.frame_rate_code))?;
        if n == 0 {
            return None;
        }
        let (en, ed) = self.extension.map_or((0, 0), |ext| {
            (
                u32::from(ext.frame_rate_extension_n),
                u32::from(ext.frame_rate_extension_d),
            )
        });
        Some((n * (en + 1), d * (ed + 1)))
    }

    fn sample_aspect(&self) -> SampleAspect {
        let code = self.header.aspect_ratio_information;
        if self.mpeg1() {
            // 11172-2 Table 2-D.4: height over width of one sample, × 10 000.
            const PEL: [u32; 15] = [
                0, 10000, 6735, 7031, 7615, 8055, 8437, 8935, 9157, 9815, 10255, 10695, 10950,
                11575, 12015,
            ];
            return match PEL.get(usize::from(code)) {
                Some(&h) if h != 0 => reduce(10000, h),
                _ => SampleAspect::UNSPECIFIED,
            };
        }
        let (dar_w, dar_h) = match code {
            1 => return SampleAspect::ratio(1, 1),
            2 => (4, 3),
            3 => (16, 9),
            4 => (221, 100),
            _ => return SampleAspect::UNSPECIFIED,
        };
        let (w, h) = match self.display {
            Some(display)
                if display.display_horizontal_size > 0 && display.display_vertical_size > 0 =>
            {
                (
                    u32::from(display.display_horizontal_size),
                    u32::from(display.display_vertical_size),
                )
            }
            _ => (self.width, self.height),
        };
        reduce(dar_w * h, dar_h * w)
    }
}

fn reduce(w: u32, h: u32) -> SampleAspect {
    let mut a = w;
    let mut b = h;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    if a == 0 {
        return SampleAspect::UNSPECIFIED;
    }
    SampleAspect::ratio(w / a, h / a)
}

#[derive(Clone, Debug)]
struct State {
    sequence: Option<Sequence>,
    /// Intra luma, non-intra luma, intra chroma, non-intra chroma.
    matrices: [Matrix; 4],
    gop_closed: bool,
    gop_broken: bool,
    /// Reference frames completed since the last GOP header.
    refs_since_gop: u32,
    older: Option<Arc<Frame>>,
    newer: Option<Arc<Frame>>,
    /// The newest I or P frame, shown when the next one arrives.
    held: Option<Decoded>,
    open_field: Option<OpenField>,
    /// After a reset, P and B pictures wait for an I picture.
    need_intra: bool,
}

impl State {
    fn new() -> Self {
        Self {
            sequence: None,
            matrices: default_matrices(),
            gop_closed: false,
            gop_broken: false,
            refs_since_gop: 0,
            older: None,
            newer: None,
            held: None,
            open_field: None,
            need_intra: true,
        }
    }
}

fn default_matrices() -> [Matrix; 4] {
    [
        DEFAULT_INTRA,
        DEFAULT_NON_INTRA,
        DEFAULT_INTRA,
        DEFAULT_NON_INTRA,
    ]
}

/// Native MPEG-2 and MPEG-1 video decoder.
#[derive(Clone, Debug)]
pub struct Mpeg2Decoder {
    state: State,
    max_pixels: u64,
}

/// One picture header, its extensions, and its slices.
struct PictureUnit<'a> {
    header: PictureHeader,
    coding: Option<PictureCodingExtension>,
    quant: Option<QuantMatrixExtension>,
    slices: Vec<SliceData<'a>>,
}

/// Timestamps of the packet being decoded.
#[derive(Clone, Copy)]
struct Clock {
    pts: i64,
    duration: i64,
    timescale: u32,
    /// Frames already started in this packet.
    started: i64,
}

impl Mpeg2Decoder {
    /// A decoder that refuses coded pictures above `max_pixels` luma samples.
    #[must_use]
    pub fn new(max_pixels: u64) -> Self {
        Self {
            state: State::new(),
            max_pixels,
        }
    }

    /// Reads the sequence header and extensions that a container stores as
    /// codec configuration. Empty input is no configuration.
    pub fn configure(&mut self, extra: &[u8]) -> Result<(), CodecError> {
        if extra.is_empty() {
            return Ok(());
        }
        let mut state = self.state.clone();
        let mut out = Vec::new();
        let clock = Clock {
            pts: 0,
            duration: 0,
            timescale: 0,
            started: 0,
        };
        self.units(&mut state, extra, clock, &mut out)?;
        if !out.is_empty() || state.sequence.is_none() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "configuration without a sequence header",
            });
        }
        self.state = state;
        Ok(())
    }

    /// Decodes one access unit and returns the frames it completed, in display order.
    pub fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let mut state = self.state.clone();
        let mut done = Vec::new();
        let clock = Clock {
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
            started: 0,
        };
        self.units(&mut state, &packet.data, clock, &mut done)?;
        let out = done.iter().map(output).collect::<Result<Vec<_>, _>>()?;
        self.state = state;
        Ok(out)
    }

    /// Releases the frame held for display order and any unpaired field.
    pub fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        let mut state = self.state.clone();
        let mut done = Vec::new();
        if let Some(open) = state.open_field.take() {
            finish_unpaired(&mut state, open, &mut done);
        }
        if let Some(held) = state.held.take() {
            done.push(held);
        }
        let out = done.iter().map(output).collect::<Result<Vec<_>, _>>()?;
        self.state = state;
        Ok(out)
    }

    /// Drops references and the display queue. The sequence header stays.
    pub fn reset(&mut self) {
        let sequence = self.state.sequence.take();
        let matrices = self.state.matrices;
        self.state = State::new();
        self.state.sequence = sequence;
        self.state.matrices = matrices;
    }

    fn units(
        &self,
        st: &mut State,
        data: &[u8],
        mut clock: Clock,
        done: &mut Vec<Decoded>,
    ) -> Result<(), CodecError> {
        let starts = start_codes(data);
        let unit = |k: usize| -> Option<(u8, &[u8], u64)> {
            let at = *starts.get(k)?;
            let code = *data.get(at + 3)?;
            let end = starts.get(k + 1).copied().unwrap_or(data.len());
            Some((code, &data[at + 4..end.max(at + 4)], (at as u64 + 4) * 8))
        };
        let mut k = 0;
        while let Some((code, payload, at_bit)) = unit(k) {
            match code {
                SEQUENCE_HEADER => {
                    let header = SequenceHeader::parse(payload)?;
                    let mut extension = None;
                    let mut display = None;
                    k += 1;
                    while let Some((next, payload, _)) = unit(k) {
                        if next == EXTENSION {
                            match extension_id(payload) {
                                Some(EXT_SEQUENCE) => {
                                    extension = Some(SequenceExtension::parse(payload)?);
                                }
                                Some(EXT_SEQUENCE_DISPLAY) => {
                                    display = Some(SequenceDisplayExtension::parse(payload)?);
                                }
                                Some(EXT_SEQUENCE_SCALABLE) => {
                                    return Err(CodecError::Unsupported {
                                        what: "MPEG-2 scalable layers",
                                    });
                                }
                                _ => {}
                            }
                        } else if next != USER_DATA {
                            break;
                        }
                        k += 1;
                    }
                    self.sequence(st, header, extension, display, done)?;
                }
                GROUP => {
                    let gop = GopHeader::parse(payload)?;
                    st.gop_closed = gop.closed_gop;
                    st.gop_broken = gop.broken_link;
                    st.refs_since_gop = 0;
                    k += 1;
                }
                PICTURE_START => {
                    let header = PictureHeader::parse(payload)?;
                    let mut coding = None;
                    let mut quant = None;
                    k += 1;
                    while let Some((next, payload, _)) = unit(k) {
                        if next == EXTENSION {
                            match extension_id(payload) {
                                Some(EXT_PICTURE_CODING) => {
                                    coding = Some(PictureCodingExtension::parse(payload)?);
                                }
                                Some(EXT_QUANT_MATRIX) => {
                                    quant = Some(QuantMatrixExtension::parse(payload)?);
                                }
                                Some(EXT_PICTURE_SPATIAL | EXT_PICTURE_TEMPORAL) => {
                                    return Err(CodecError::Unsupported {
                                        what: "MPEG-2 scalable layers",
                                    });
                                }
                                _ => {}
                            }
                        } else if next != USER_DATA {
                            break;
                        }
                        k += 1;
                    }
                    let mut slices = Vec::new();
                    while let Some((next, payload, at_bit)) = unit(k) {
                        if !(SLICE_FIRST..=SLICE_LAST).contains(&next) {
                            break;
                        }
                        slices.push(SliceData {
                            vertical_position: next,
                            data: payload,
                            at_bit,
                        });
                        k += 1;
                    }
                    let unit = PictureUnit {
                        header,
                        coding,
                        quant,
                        slices,
                    };
                    self.picture(st, &unit, &mut clock, done)?;
                }
                SEQUENCE_END => {
                    if let Some(open) = st.open_field.take() {
                        finish_unpaired(st, open, done);
                    }
                    if let Some(held) = st.held.take() {
                        done.push(held);
                    }
                    k += 1;
                }
                SEQUENCE_ERROR => {
                    return Err(CodecError::Malformed {
                        at_bit,
                        what: "sequence_error_code",
                    });
                }
                SLICE_FIRST..=SLICE_LAST => {
                    return Err(CodecError::Malformed {
                        at_bit,
                        what: "slice without a picture header",
                    });
                }
                _ => k += 1,
            }
        }
        Ok(())
    }

    fn sequence(
        &self,
        st: &mut State,
        header: SequenceHeader,
        extension: Option<SequenceExtension>,
        display: Option<SequenceDisplayExtension>,
        done: &mut Vec<Decoded>,
    ) -> Result<(), CodecError> {
        let (h_ext, v_ext) = extension.map_or((0, 0), |ext| {
            (
                u32::from(ext.horizontal_size_extension),
                u32::from(ext.vertical_size_extension),
            )
        });
        let width = u32::from(header.horizontal_size_value) | h_ext << 12;
        let height = u32::from(header.vertical_size_value) | v_ext << 12;
        if width == 0 || height == 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "zero picture size",
            });
        }
        let progressive = extension.is_none_or(|ext| ext.progressive_sequence);
        let mb_width = width.div_ceil(16) as usize;
        let mb_height = if progressive {
            height.div_ceil(16) as usize
        } else {
            2 * height.div_ceil(32) as usize
        };
        let pixels = (mb_width as u64 * 16) * (mb_height as u64 * 16);
        if pixels > self.max_pixels {
            return Err(CodecError::Limit {
                what: "MPEG-2 picture size",
            });
        }
        let geo = Geometry {
            mb_width,
            mb_height,
            chroma_format: extension.map_or(CHROMA_420, |ext| ext.chroma_format),
        };
        let changed = st.sequence.as_ref().is_none_or(|old| {
            old.geo != geo
                || old.width != width
                || old.height != height
                || old.mpeg1() != extension.is_none()
        });
        if changed {
            if let Some(open) = st.open_field.take() {
                finish_unpaired(st, open, done);
            }
            if let Some(held) = st.held.take() {
                done.push(held);
            }
            st.older = None;
            st.newer = None;
            st.need_intra = true;
        }
        let mut matrices = default_matrices();
        if let Some(matrix) = header.intra_matrix {
            matrices[0] = matrix;
            matrices[2] = matrix;
        }
        if let Some(matrix) = header.non_intra_matrix {
            matrices[1] = matrix;
            matrices[3] = matrix;
        }
        st.matrices = matrices;
        st.sequence = Some(Sequence {
            header,
            extension,
            display,
            geo,
            width,
            height,
        });
        Ok(())
    }

    fn picture(
        &self,
        st: &mut State,
        unit: &PictureUnit<'_>,
        clock: &mut Clock,
        done: &mut Vec<Decoded>,
    ) -> Result<(), CodecError> {
        let PictureUnit {
            header,
            coding,
            quant,
            ref slices,
        } = *unit;
        let Some(seq) = st.sequence.clone() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "picture before a sequence header",
            });
        };
        let mpeg1 = seq.mpeg1();
        let coding_type = header.picture_coding_type;
        let mut params = match (mpeg1, coding) {
            (true, _) => PictureParams {
                mpeg1: true,
                coding_type,
                field: None,
                second_field: false,
                f_code: [[header.forward_f_code; 2], [header.backward_f_code; 2]],
                full_pel: [header.full_pel_forward, header.full_pel_backward],
                intra_dc_precision: 0,
                frame_pred_frame_dct: true,
                concealment: false,
                q_scale_type: false,
                intra_vlc: false,
                alternate_scan: false,
                top_field_first: false,
                vertical_size: seq.height as usize,
            },
            (false, Some(ext)) => {
                if coding_type == PICTURE_D {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "D picture in an MPEG-2 sequence",
                    });
                }
                PictureParams {
                    mpeg1: false,
                    coding_type,
                    field: (ext.picture_structure != STRUCTURE_FRAME)
                        .then_some(usize::from(ext.picture_structure != STRUCTURE_TOP)),
                    second_field: false,
                    f_code: ext.f_code,
                    full_pel: [false; 2],
                    intra_dc_precision: ext.intra_dc_precision,
                    frame_pred_frame_dct: ext.frame_pred_frame_dct,
                    concealment: ext.concealment_motion_vectors,
                    q_scale_type: ext.q_scale_type,
                    intra_vlc: ext.intra_vlc_format,
                    alternate_scan: ext.alternate_scan,
                    top_field_first: ext.top_field_first,
                    vertical_size: seq.height as usize,
                }
            }
            (false, None) => {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "picture without a picture coding extension",
                });
            }
        };
        if let Some(quant) = quant {
            if let Some(matrix) = quant.intra {
                st.matrices[0] = matrix;
                st.matrices[2] = matrix;
            }
            if let Some(matrix) = quant.non_intra {
                st.matrices[1] = matrix;
                st.matrices[3] = matrix;
            }
            if let Some(matrix) = quant.chroma_intra {
                st.matrices[2] = matrix;
            }
            if let Some(matrix) = quant.chroma_non_intra {
                st.matrices[3] = matrix;
            }
        }
        if slices.is_empty() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "picture without slices",
            });
        }
        let is_b = coding_type == PICTURE_B;
        // A second field pairs with an open first field of the other parity.
        let open = st.open_field.take();
        let pair = match (open, params.field) {
            (Some(open), Some(parity))
                if open.parity != parity && (open.coding_type == PICTURE_B) == is_b =>
            {
                Some(open)
            }
            (Some(open), _) => {
                finish_unpaired(st, open, done);
                None
            }
            (None, _) => None,
        };
        params.second_field = pair.is_some();
        // Can this picture be decoded from the references that are here?
        let first_type = pair.as_ref().map_or(coding_type, |open| open.coding_type);
        let decodable = match pair.as_ref() {
            Some(open) => open.frame.is_some(),
            None => match coding_type {
                PICTURE_I | PICTURE_D => true,
                PICTURE_P => !st.need_intra && st.newer.is_some(),
                _ => {
                    let leading = st.refs_since_gop <= 1;
                    !st.need_intra
                        && st.newer.is_some()
                        && !(leading && st.gop_broken)
                        && (st.older.is_some() || (leading && st.gop_closed))
                }
            },
        };
        if coding_type == PICTURE_I && pair.is_none() {
            st.need_intra = false;
        }
        let info = FrameInfo {
            geo: seq.geo,
            width: seq.width,
            height: seq.height,
            colour: seq.display.and_then(|display| display.colour),
            sample_aspect: seq.sample_aspect(),
            frame_rate: seq.frame_rate(),
            pts: clock.pts + clock.started * frame_ticks(clock, seq.frame_rate()),
            duration: clock.duration,
            timescale: clock.timescale,
            keyframe: first_type == PICTURE_I,
        };
        let (fwd, bwd) = match coding_type {
            PICTURE_P => (st.newer.clone(), None),
            PICTURE_B => (
                st.older.clone().or_else(|| st.newer.clone()),
                st.newer.clone(),
            ),
            _ => (None, None),
        };
        let fallback = st.newer.clone();
        let (mut frame, info) = match pair {
            Some(open) => (open.frame, open.info),
            None => {
                clock.started += 1;
                let frame = decodable.then(|| Arc::new(Frame::new(&seq.geo, 0)));
                (frame, info)
            }
        };
        if let Some(target) = frame.as_mut() {
            let picture = Picture {
                geo: seq.geo,
                params,
                matrices: &st.matrices,
                fwd: fwd.as_deref(),
                bwd: bwd.as_deref(),
            };
            decode_picture(&picture, Arc::make_mut(target), slices, fallback.as_deref())?;
        }
        match params.field {
            Some(parity) if !params.second_field => {
                st.open_field = Some(OpenField {
                    frame,
                    info,
                    parity,
                    coding_type,
                });
            }
            _ => {
                if let Some(frame) = frame {
                    complete(st, Decoded { frame, info }, first_type, &seq, done);
                }
            }
        }
        Ok(())
    }
}

/// Ticks of one frame in the packet timescale, for a second frame that a
/// single packet carries.
fn frame_ticks(clock: &Clock, rate: Option<(u32, u32)>) -> i64 {
    match rate {
        Some((n, d)) if n > 0 && clock.timescale > 0 => {
            (i64::from(clock.timescale) * i64::from(d) + i64::from(n) / 2) / i64::from(n)
        }
        _ => clock.duration,
    }
}

/// A frame whose fields, or whose frame picture, are all decoded.
fn complete(
    st: &mut State,
    decoded: Decoded,
    coding_type: u8,
    seq: &Sequence,
    done: &mut Vec<Decoded>,
) {
    if coding_type == PICTURE_B || coding_type == PICTURE_D {
        done.push(decoded);
        return;
    }
    st.older = st.newer.take();
    st.newer = Some(Arc::clone(&decoded.frame));
    st.refs_since_gop = st.refs_since_gop.saturating_add(1);
    if seq.extension.is_some_and(|ext| ext.low_delay) {
        if let Some(held) = st.held.take() {
            done.push(held);
        }
        done.push(decoded);
    } else if let Some(held) = st.held.replace(decoded) {
        done.push(held);
    }
}

/// A first field whose pair never came: its lines fill the missing field.
fn finish_unpaired(st: &mut State, open: OpenField, done: &mut Vec<Decoded>) {
    let Some(mut frame) = open.frame else {
        return;
    };
    let geo = open.info.geo;
    let target = Arc::make_mut(&mut frame);
    for c in 0..3 {
        let (stride, height) = geo.plane(c);
        let plane = &mut target.planes[c];
        for line in (0..height).step_by(2) {
            let (from, to) = if open.parity == 0 {
                (line, line + 1)
            } else {
                (line + 1, line)
            };
            plane.copy_within(from * stride..from * stride + stride, to * stride);
        }
    }
    let Some(seq) = st.sequence.clone() else {
        return;
    };
    complete(
        st,
        Decoded {
            frame,
            info: open.info,
        },
        open.coding_type,
        &seq,
        done,
    );
}

/// The cropped picture and its tags.
fn output(decoded: &Decoded) -> Result<VideoOutput, CodecError> {
    let info = &decoded.info;
    let geo = info.geo;
    let frame = &decoded.frame;
    // 4:2:0 and 4:2:2 pictures need an even width; the coded area has it.
    let width = (info.width as usize + 1) & !1;
    let height = (info.height as usize + 1) & !1;
    let crop = |c: usize, w: usize, h: usize| -> Vec<u8> {
        let (stride, _) = geo.plane(c);
        let mut out = Vec::with_capacity(w * h);
        for row in frame.planes[c].chunks_exact(stride).take(h) {
            out.extend_from_slice(&row[..w]);
        }
        out
    };
    let invalid = |_| CodecError::Limit {
        what: "MPEG-2 output picture",
    };
    let y = crop(0, width, height);
    let (picture, yuv422, yuv444) = match geo.chroma_format {
        CHROMA_420 => {
            let u = crop(1, width / 2, height / 2);
            let v = crop(2, width / 2, height / 2);
            let picture =
                Yuv420p::try_new(width as u32, height as u32, y, u, v).map_err(invalid)?;
            (picture, None, None)
        }
        CHROMA_422 => {
            let u = crop(1, width / 2, height);
            let v = crop(2, width / 2, height);
            let half = |plane: &[u8]| subsample(plane, width / 2, height, 1, 2);
            let picture =
                Yuv420p::try_new(width as u32, height as u32, y.clone(), half(&u), half(&v))
                    .map_err(invalid)?;
            let full = Yuv422p::try_new(width as u32, height as u32, y, u, v).map_err(invalid)?;
            (picture, Some(full), None)
        }
        _ => {
            let u = crop(1, width, height);
            let v = crop(2, width, height);
            let quarter = |plane: &[u8]| subsample(plane, width, height, 2, 2);
            let picture = Yuv420p::try_new(
                width as u32,
                height as u32,
                y.clone(),
                quarter(&u),
                quarter(&v),
            )
            .map_err(invalid)?;
            let full = Yuv444p::try_new(width as u32, height as u32, y, u, v).map_err(invalid)?;
            (picture, None, Some(full))
        }
    };
    let (primaries, transfer, matrix) = colour(info.colour);
    Ok(VideoOutput {
        picture,
        yuv420p10: None,
        yuv420_high: None,
        yuv422,
        yuv422_high: None,
        yuv444,
        yuv444_high: None,
        pts: info.pts,
        duration: info.duration,
        timescale: info.timescale,
        keyframe: info.keyframe,
        range: ColorRange::Limited,
        primaries,
        transfer,
        matrix,
        sample_aspect: info.sample_aspect,
        timing: info
            .frame_rate
            .map_or(SequenceTiming::ABSENT, |(n, d)| SequenceTiming {
                num_units: d,
                time_scale: 2 * n,
                fixed: true,
            }),
    })
}

/// Averages `fx`×`fy` neighbourhoods of a `w`×`h` plane, rounding half up.
fn subsample(plane: &[u8], w: usize, h: usize, fx: usize, fy: usize) -> Vec<u8> {
    let (ow, oh) = (w / fx, h / fy);
    let count = (fx * fy) as u32;
    let mut out = Vec::with_capacity(ow * oh);
    for y in 0..oh {
        for x in 0..ow {
            let mut sum = 0u32;
            for dy in 0..fy {
                for dx in 0..fx {
                    sum += u32::from(plane[(y * fy + dy) * w + x * fx + dx]);
                }
            }
            out.push(((sum + count / 2) / count) as u8);
        }
    }
    out
}

/// Tables 6-7 to 6-9. Without a colour description the tags are unspecified.
pub(crate) fn colour(colour: Option<(u8, u8, u8)>) -> (ColorPrimaries, Transfer, ColorMatrix) {
    let Some((p, t, m)) = colour else {
        return (
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
            ColorMatrix::Unspecified,
        );
    };
    let primaries = match p {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        _ => ColorPrimaries::Unspecified,
    };
    let transfer = match t {
        1 => Transfer::Bt709,
        4..=6 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        11 => Transfer::Iec61966_2_4,
        12 => Transfer::Bt1361,
        13 => Transfer::Srgb,
        14 | 15 => Transfer::Bt2020,
        16 => Transfer::Pq,
        17 => Transfer::Smpte428,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
    };
    let matrix = match m {
        0 => ColorMatrix::Identity,
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        8 => ColorMatrix::Ycgco,
        9 | 10 => ColorMatrix::Bt2020,
        _ => ColorMatrix::Unspecified,
    };
    (primaries, transfer, matrix)
}

impl Default for Mpeg2Decoder {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_PIXELS)
    }
}
