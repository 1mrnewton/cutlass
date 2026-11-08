//! The stream layer: start codes, headers, references, VOP timing, and
//! display order (ISO/IEC 14496-2 Sections 6.2, 6.3 and 7.6.9).

use cutlass_decode::CodecError;

use crate::block::Quant;
use crate::frame::Frame;
use crate::header::{
    Colour, Encoder, GOV_START, ShortHeader, SpriteUsage, USER_DATA, VISUAL_OBJECT, VOP_START,
    VOS_END, VOS_START, Vol, VopType, is_short_header, is_video_object, is_vol, parse_gov,
    parse_short_header, parse_user_data, parse_visual_object, parse_vol, parse_vop, units,
};
use crate::mc::Warp;
use crate::vop::{self, Fixes, Grid, Layer, Picture, Times};

/// One displayed picture: packed 4:2:0 over its even-sized top-left.
pub(crate) struct Image {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes: Vec<u8>,
}

/// What one packet gave.
#[derive(Default)]
pub(crate) struct Decoded {
    /// Pictures that became due for display, in display order.
    pub(crate) pictures: Vec<Image>,
    /// The packet's time belongs to a picture: one this packet decoded,
    /// or the B-VOP a packed bitstream put in the previous packet.
    pub(crate) timed: bool,
}

/// VOP times in VOL ticks (Section 6.3.5's modulo time base).
#[derive(Clone, Copy, Debug, Default)]
struct Clock {
    /// Seconds of the latest reference VOP.
    time_base: i64,
    /// Seconds of the reference before it, which B-VOPs count from.
    last_time_base: i64,
    /// Time of the latest reference VOP.
    last_non_b: i64,
    /// Distance between the two latest reference VOPs.
    pp: i64,
    /// The first B-VOP's distance, which field times are rounded to.
    t_frame: i64,
}

/// Rounds `a / b` half away from zero.
fn rounded_div(a: i64, b: i64) -> i64 {
    if a >= 0 {
        (a + (b >> 1)) / b
    } else {
        (a - (b >> 1)) / b
    }
}

fn narrow(v: i64) -> i32 {
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// The coded size of the current layer and what it was signalled by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Vol(u32, u32),
    Short(u32, u32),
}

/// Decodes MPEG-4 Visual elementary stream packets.
pub(crate) struct Decoder {
    max_pixels: u64,
    vol: Option<Vol>,
    source: Option<Source>,
    colour: Colour,
    encoder: Encoder,
    fixes: Fixes,
    grid: Option<Grid>,
    /// The reference before `newer`: a B-VOP's forward reference.
    older: Option<Frame>,
    /// The latest reference: a P-VOP's reference, a B-VOP's backward one.
    newer: Option<Frame>,
    /// `newer` has not been displayed yet.
    held: bool,
    /// Each picture is displayed when it is decoded.
    low_delay: bool,
    clock: Clock,
    /// Pictures decoded in packets that held more than one, whose
    /// not-coded placeholder packets are still to come.
    packed_extra: usize,
    spare: Vec<Frame>,
}

impl Decoder {
    pub(crate) fn with_max_pixels(max_pixels: u64) -> Self {
        Self {
            max_pixels,
            vol: None,
            source: None,
            colour: Colour::default(),
            encoder: Encoder::default(),
            fixes: Fixes::default(),
            grid: None,
            older: None,
            newer: None,
            held: false,
            low_delay: false,
            clock: Clock::default(),
            packed_extra: 0,
            spare: Vec::new(),
        }
    }

    /// The colour the visual object header signalled.
    pub(crate) fn colour(&self) -> Colour {
        self.colour
    }

    /// Reads the decoder configuration: VOS, visual object, VOL, and user
    /// data headers. A VOP in it is an error.
    pub(crate) fn configure(&mut self, extra: &[u8]) -> Result<(), CodecError> {
        let mut flushed = Vec::new();
        for unit in units(extra) {
            if unit.code == VOP_START {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "VOP in the decoder configuration",
                });
            }
            self.header(unit.code, unit.payload, &mut flushed)?;
        }
        Ok(())
    }

    /// Decodes one packet: any headers, then each VOP in it, or one
    /// short-header picture.
    pub(crate) fn decode(&mut self, data: &[u8]) -> Result<Decoded, CodecError> {
        let mut out = Decoded::default();
        if is_short_header(data) {
            out.timed = self.short(data, &mut out.pictures)?;
            return Ok(out);
        }
        if self.packed_extra > 0 && self.is_placeholder(data) {
            // The B-VOP this stands for was decoded with the packet before;
            // the placeholder's own header is not part of the timeline.
            self.packed_extra -= 1;
            out.timed = true;
            return Ok(out);
        }
        let mut coded = 0;
        for unit in units(data) {
            if unit.code == VOP_START {
                if self.vop(unit.payload, &mut out.pictures)? {
                    coded += 1;
                }
            } else {
                self.header(unit.code, unit.payload, &mut out.pictures)?;
            }
        }
        if coded > 0 {
            out.timed = true;
            self.packed_extra += coded - 1;
        }
        Ok(out)
    }

    /// The pictures still waiting for display.
    pub(crate) fn drain(&mut self) -> Vec<Image> {
        let mut out = Vec::new();
        self.flush(&mut out);
        out
    }

    /// Forgets the references, as after a seek. Headers and timing stay.
    pub(crate) fn reset(&mut self) {
        self.drop_references();
        self.packed_extra = 0;
    }

    fn header(
        &mut self,
        code: u8,
        payload: &[u8],
        pictures: &mut Vec<Image>,
    ) -> Result<(), CodecError> {
        match code {
            c if is_vol(c) => self.set_vol(parse_vol(payload)?, pictures)?,
            VISUAL_OBJECT => self.colour = parse_visual_object(payload),
            USER_DATA => {
                parse_user_data(payload, &mut self.encoder);
                self.fixes = Fixes {
                    xvid_ilace: self.fixes.xvid_ilace,
                    ..Fixes::from_encoder(&self.encoder)
                };
            }
            GOV_START => {
                if let Some(seconds) = parse_gov(payload) {
                    self.clock.time_base = i64::from(seconds);
                }
            }
            // A sequence header names a profile. The layer header is what
            // decides which tools the pictures use.
            VOS_START | VOS_END => {}
            c if is_video_object(c) => {}
            _ => {}
        }
        Ok(())
    }

    /// Whether every VOP in `data` is not coded: the packet a packed
    /// bitstream leaves after the one that carried two VOPs.
    fn is_placeholder(&self, data: &[u8]) -> bool {
        let Some(vol) = &self.vol else {
            return false;
        };
        let mut vops = 0;
        for unit in units(data) {
            if unit.code != VOP_START {
                continue;
            }
            match parse_vop(unit.payload, vol, self.fixes.divx_413) {
                Ok(vop) if !vop.coded => vops += 1,
                _ => return false,
            }
        }
        vops > 0
    }

    fn check_size(&self, width: u32, height: u32) -> Result<(), CodecError> {
        if u64::from(width) * u64::from(height) > self.max_pixels {
            return Err(CodecError::Limit {
                what: "picture size",
            });
        }
        Ok(())
    }

    /// Switches to the layer `source`, displaying and dropping the
    /// references of a layer of another size.
    fn set_source(&mut self, source: Source, pictures: &mut Vec<Image>) {
        if self.source != Some(source) {
            self.flush(pictures);
            self.drop_references();
            self.grid = None;
            self.spare.clear();
            self.source = Some(source);
        }
    }

    fn set_vol(&mut self, vol: Vol, pictures: &mut Vec<Image>) -> Result<(), CodecError> {
        self.check_size(vol.width, vol.height)?;
        let changed = self.source != Some(Source::Vol(vol.width, vol.height));
        self.set_source(Source::Vol(vol.width, vol.height), pictures);
        // Without vol_control_parameters a layer may still hold B-VOPs, so
        // pictures wait for the next reference unless the layer says they
        // need not.
        if vol.vol_control || changed {
            self.low_delay = vol.vol_control && vol.low_delay;
        }
        // Field distances round to the first B-VOP's distance in this layer.
        self.clock.t_frame = 0;
        self.vol = Some(vol);
        Ok(())
    }

    /// Decodes one VOP. Returns whether it was a coded picture this
    /// decoder decoded.
    fn vop(&mut self, payload: &[u8], pictures: &mut Vec<Image>) -> Result<bool, CodecError> {
        let Some(mut vol) = self.vol.clone() else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "VOP before its video object layer",
            });
        };
        if self.source != Some(Source::Vol(vol.width, vol.height)) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "VOP after a short-header picture",
            });
        }
        let hdr = parse_vop(payload, &vol, self.fixes.divx_413)?;
        if hdr.time_bits != vol.time_bits {
            // The width the stream uses holds until the next VOL.
            vol.time_bits = hdr.time_bits;
            self.vol = Some(vol.clone());
        }
        let Some(times) =
            self.advance_clock(&vol, hdr.kind, hdr.modulo_time_base, hdr.time_increment)
        else {
            return Ok(false);
        };
        if !hdr.coded {
            return Ok(false);
        }
        if hdr.kind == VopType::B {
            // A B-VOP needs both references; after a seek it has only one.
            if self.older.is_none() || self.newer.is_none() {
                return Ok(false);
            }
            self.low_delay = false;
        } else if hdr.kind != VopType::I && self.newer.is_none() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "predicted VOP without a reference",
            });
        }
        let warp = if hdr.kind == VopType::S {
            let Some(warp) = Warp::new(
                &hdr.trajectory,
                vol.warping_points,
                vol.warping_accuracy,
                vol.width,
                vol.height,
                self.fixes.divx_413,
            ) else {
                return Err(CodecError::Unsupported {
                    what: "global motion warp out of range",
                });
            };
            Some(warp)
        } else {
            None
        };
        let mbw = vol.width.div_ceil(16) as usize;
        let mbh = vol.height.div_ceil(16) as usize;
        let quant = if vol.mpeg_quant {
            Quant::Mpeg {
                intra: &vol.intra_matrix,
                inter: &vol.inter_matrix,
            }
        } else {
            Quant::H263
        };
        let edge = if self.fixes.edge {
            (vol.width, vol.height)
        } else {
            (mbw as u32 * 16, mbh as u32 * 16)
        };
        let layer = Layer {
            mbw,
            mbh,
            edge,
            size: (vol.width, vol.height),
            interlaced: vol.interlaced,
            quarter: vol.quarter_sample,
            quant,
            partitioned: vol.data_partitioning,
            rvlc: vol.rvlc,
            gmc: vol.sprite == SpriteUsage::Gmc,
            warping_points: vol.warping_points,
            time_bits: vol.time_bits,
            short: false,
            resync: vol.resync,
            fixes: self.fixes,
        };
        let pic = Picture {
            kind: hdr.kind,
            quant: hdr.quant,
            rounding: hdr.rounding,
            fcode_forward: hdr.fcode_forward,
            fcode_backward: hdr.fcode_backward,
            dc_threshold: hdr.dc_threshold,
            alternate_scan: hdr.alternate_scan,
            top_field_first: hdr.top_field_first,
            warp,
            times,
            data_start: hdr.data_start,
            gob_rows: 0,
        };
        self.picture(payload, &layer, &pic, (vol.width, vol.height), pictures)?;
        Ok(true)
    }

    /// Moves the clock to a VOP and returns its B-VOP distances, or
    /// `None` for a B-VOP whose times do not fall between its references.
    fn advance_clock(
        &mut self,
        vol: &Vol,
        kind: VopType,
        modulo: u32,
        increment: u32,
    ) -> Option<Times> {
        let c = &mut self.clock;
        let resolution = i64::from(vol.time_resolution);
        let modulo = i64::from(modulo);
        let increment = i64::from(increment);
        if kind != VopType::B {
            c.last_time_base = c.time_base;
            c.time_base += modulo;
            let time = c.time_base * resolution + increment;
            c.pp = time - c.last_non_b;
            c.last_non_b = time;
            return Some(Times::default());
        }
        let time = (c.last_time_base + modulo) * resolution + increment;
        let pb = c.pp - (c.last_non_b - time);
        if c.pp <= 0 || pb <= 0 || pb >= c.pp {
            return None;
        }
        if c.t_frame == 0 {
            c.t_frame = pb;
        }
        let t = c.t_frame;
        let base = rounded_div(c.last_non_b - c.pp, t);
        let mut pp_field = (rounded_div(c.last_non_b, t) - base) * 2;
        let mut pb_field = (rounded_div(time, t) - base) * 2;
        if pp_field <= pb_field || pb_field <= 1 {
            if vol.interlaced {
                return None;
            }
            pp_field = 4;
            pb_field = 2;
        }
        Some(Times {
            pp: narrow(c.pp),
            pb: narrow(pb),
            pp_field: narrow(pp_field),
            pb_field: narrow(pb_field),
        })
    }

    /// Decodes one short-header picture. Returns whether it decoded.
    fn short(&mut self, data: &[u8], pictures: &mut Vec<Image>) -> Result<bool, CodecError> {
        let hdr: ShortHeader = parse_short_header(data)?;
        self.check_size(hdr.width, hdr.height)?;
        self.set_source(Source::Short(hdr.width, hdr.height), pictures);
        self.vol = None;
        // Short-header pictures have no B-VOPs.
        self.low_delay = true;
        if hdr.inter && self.newer.is_none() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "predicted picture without a reference",
            });
        }
        let (mbw, mbh) = ((hdr.width / 16) as usize, (hdr.height / 16) as usize);
        let layer = Layer {
            mbw,
            mbh,
            edge: (hdr.width, hdr.height),
            size: (hdr.width, hdr.height),
            interlaced: false,
            quarter: false,
            quant: Quant::H263,
            partitioned: false,
            rvlc: false,
            gmc: false,
            warping_points: 0,
            time_bits: 0,
            short: true,
            resync: false,
            fixes: Fixes::default(),
        };
        let pic = Picture {
            kind: if hdr.inter { VopType::P } else { VopType::I },
            quant: hdr.quant,
            rounding: false,
            fcode_forward: 1,
            fcode_backward: 1,
            dc_threshold: 0,
            alternate_scan: false,
            top_field_first: false,
            warp: None,
            times: Times::default(),
            data_start: hdr.data_start,
            gob_rows: hdr.gob_rows as usize,
        };
        self.picture(data, &layer, &pic, (hdr.width, hdr.height), pictures)?;
        Ok(true)
    }

    /// Decodes the macroblocks of one picture and updates the references
    /// and the display queue. A failed picture leaves both unchanged.
    fn picture(
        &mut self,
        data: &[u8],
        layer: &Layer<'_>,
        pic: &Picture,
        size: (u32, u32),
        pictures: &mut Vec<Image>,
    ) -> Result<(), CodecError> {
        let (mbw, mbh) = (layer.mbw, layer.mbh);
        let grid = match &mut self.grid {
            Some(grid) if grid.fits(mbw, mbh) => grid,
            slot => slot.insert(Grid::new(mbw, mbh)),
        };
        let mut frame = match self.spare.pop() {
            Some(frame) if frame.mbw == mbw && frame.mbh == mbh => frame,
            _ => Frame::new(mbw, mbh),
        };
        let (past, future) = match pic.kind {
            VopType::I => (None, None),
            VopType::P | VopType::S => (self.newer.as_ref(), None),
            VopType::B => (self.older.as_ref(), self.newer.as_ref()),
        };
        if let Err(error) = vop::decode(data, layer, pic, grid, past, future, &mut frame) {
            // Old Xvid builds write dct_type in every interlaced P
            // macroblock and say so nowhere but their FourCC. A P-VOP that
            // fails as written and decodes whole that way is one of them.
            let xvid = Layer {
                fixes: Fixes {
                    xvid_ilace: true,
                    ..layer.fixes
                },
                ..*layer
            };
            let retry = layer.interlaced
                && !layer.fixes.xvid_ilace
                && matches!(pic.kind, VopType::P | VopType::S)
                && self.encoder.divx_version.is_none()
                && self.encoder.lavc_build.is_none();
            if !(retry && vop::decode(data, &xvid, pic, grid, past, future, &mut frame).is_ok()) {
                self.spare.push(frame);
                return Err(error);
            }
            self.fixes.xvid_ilace = true;
        }
        if pic.kind == VopType::B {
            pictures.push(image(&frame, size));
            self.spare.push(frame);
            return Ok(());
        }
        if self.held
            && let Some(newer) = &self.newer
        {
            pictures.push(image(newer, size));
        }
        if let Some(old) = self.older.take() {
            self.spare.push(old);
        }
        self.older = self.newer.take();
        if self.low_delay {
            pictures.push(image(&frame, size));
            self.held = false;
        } else {
            self.held = true;
        }
        self.newer = Some(frame);
        Ok(())
    }

    /// Displays the held reference.
    fn flush(&mut self, pictures: &mut Vec<Image>) {
        if self.held
            && let Some(newer) = &self.newer
            && let Some(size) = self.size()
        {
            pictures.push(image(newer, size));
        }
        self.held = false;
    }

    fn size(&self) -> Option<(u32, u32)> {
        match self.source? {
            Source::Vol(w, h) | Source::Short(w, h) => Some((w, h)),
        }
    }

    fn drop_references(&mut self) {
        for frame in [self.older.take(), self.newer.take()].into_iter().flatten() {
            self.spare.push(frame);
        }
        self.held = false;
    }
}

/// The even-sized top-left of `frame` for display.
fn image(frame: &Frame, (width, height): (u32, u32)) -> Image {
    let width = (width & !1).max(2);
    let height = (height & !1).max(2);
    Image {
        width,
        height,
        bytes: frame.packed(width as usize, height as usize),
    }
}
