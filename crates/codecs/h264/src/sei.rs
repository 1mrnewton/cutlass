//! Supplemental enhancement information, Annex D.
//!
//! Picture timing changes the output time. Film grain selects the model
//! applied when the picture is displayed. Every other payload, including a
//! V16 message whose payload type is not assigned here, is consumed by its
//! size and does not change samples or time. Source picture timing describes
//! the source interval. It does not replace the decoded output time.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::grain::{Component, FilmGrain, Interval};

/// One SEI payload. `kind` is the payload type.
#[derive(Clone, Debug)]
pub(crate) struct Payload {
    pub(crate) kind: u32,
    pub(crate) data: Vec<u8>,
}

/// Clock lengths from the SPS. Zero delay lengths mean the HRD is absent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TimingClock {
    pub(crate) pic_struct: bool,
    pub(crate) cpb_bits: u8,
    pub(crate) dpb_bits: u8,
    pub(crate) offset_bits: u8,
    pub(crate) num_units: u32,
    pub(crate) time_scale: u32,
}

impl TimingClock {
    pub(crate) fn absent() -> Self {
        Self {
            pic_struct: false,
            cpb_bits: 0,
            dpb_bits: 0,
            offset_bits: 0,
            num_units: 0,
            time_scale: 0,
        }
    }
}

/// Film grain that persists across pictures in one coded video sequence.
#[derive(Clone, Debug, Default)]
pub(crate) struct GrainHold {
    grain: Option<FilmGrain>,
    origin: i32,
    period: u32,
}

impl GrainHold {
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn store(&mut self, grain: FilmGrain, poc: i32, period: u32) {
        self.grain = Some(grain);
        self.origin = poc;
        self.period = period;
    }

    /// The grain for a picture at `poc`, when a message on that picture did
    /// not replace it. Period 1 lasts until the next message or a new
    /// sequence. A larger period also ends once the picture order count
    /// passes the origin by that many counts.
    pub(crate) fn for_picture(&self, poc: i32) -> Option<FilmGrain> {
        let grain = self.grain.clone()?;
        if poc < self.origin {
            return None;
        }
        if self.period > 1 {
            let span = i32::try_from(self.period).unwrap_or(i32::MAX);
            if poc > self.origin.saturating_add(span) {
                return None;
            }
        }
        Some(grain)
    }
}

/// What a film-grain payload does to the current picture.
pub(crate) enum GrainChoice {
    Cancel,
    Ignore,
    Apply(FilmGrain, u32),
}

const CLOCKS: [u32; 9] = [1, 1, 1, 2, 2, 3, 3, 2, 3];

pub(crate) fn payloads(rbsp: &[u8]) -> Result<Vec<Payload>, CodecError> {
    let mut bits = Bits::new(rbsp);
    let mut out = Vec::new();
    while bits.more_rbsp() {
        let kind = ff_sum(&mut bits)?;
        let size = ff_sum(&mut bits)?;
        let size = usize::try_from(size).map_err(|_| mal())?;
        if size > bits.rest().len() {
            return Err(mal());
        }
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(take_byte(&mut bits)?);
        }
        out.push(Payload { kind, data });
    }
    bits.rbsp_trailing().map_err(|_| mal())?;
    Ok(out)
}

/// Output `(pts, duration)` after picture timing. The duration is the packet
/// duration times the Table D-1 field count, divided by two. The presentation
/// time moves by `dpb_output_delay` only when that delay is an integer number
/// of packet ticks.
pub(crate) fn picture_timing(
    payload: &[u8],
    clock: &TimingClock,
    pts: i64,
    duration: i64,
    timescale: u32,
) -> Result<(i64, i64), CodecError> {
    let mut bits = Bits::new(payload);
    let mut pts = pts;
    let mut duration = duration;
    if clock.cpb_bits > 0 {
        let _removal = take_u(&mut bits, u32::from(clock.cpb_bits))?;
        let delay = take_u(&mut bits, u32::from(clock.dpb_bits))?;
        pts = shift_pts(pts, delay, clock, timescale);
    }
    if clock.pic_struct {
        let pic_struct = take_u(&mut bits, 4)?;
        let fields = display_fields(pic_struct)?;
        duration = duration.saturating_mul(i64::from(fields)) / 2;
        let clocks = CLOCKS.get(pic_struct as usize).copied().unwrap_or(0);
        for _ in 0..clocks {
            if take_flag(&mut bits)? {
                skip_timestamp(&mut bits, clock.offset_bits)?;
            }
        }
    }
    Ok((pts, duration))
}

pub(crate) fn film_grain(payload: &[u8], bit_depth: u32) -> Result<GrainChoice, CodecError> {
    let mut bits = Bits::new(payload);
    if take_flag(&mut bits)? {
        return Ok(GrainChoice::Cancel);
    }
    let model = take_u(&mut bits, 2)?;
    if model > 1 {
        return Ok(GrainChoice::Ignore);
    }
    let (luma_depth, chroma_depth, mismatch) = colour(&mut bits, bit_depth)?;
    let blend = take_u(&mut bits, 2)?;
    if blend > 1 {
        return Ok(GrainChoice::Ignore);
    }
    let scale = take_u(&mut bits, 4)?;
    let mut present = [false; 3];
    for flag in &mut present {
        *flag = take_flag(&mut bits)?;
    }
    let mut comp = [None, None, None];
    for (index, flag) in present.iter().enumerate() {
        if *flag {
            let depth = if index == 0 { luma_depth } else { chroma_depth };
            comp[index] = Some(component(&mut bits, model, depth)?);
        }
    }
    let period = take_ue(&mut bits)?;
    if period > 16_384 {
        return Err(mal());
    }
    if mismatch {
        return Ok(GrainChoice::Ignore);
    }
    let scale = u8::try_from(scale).map_err(|_| mal())?;
    let model = u8::try_from(model).map_err(|_| mal())?;
    let blend = u8::try_from(blend).map_err(|_| mal())?;
    Ok(GrainChoice::Apply(
        FilmGrain {
            model,
            blend,
            scale,
            comp,
        },
        period,
    ))
}

fn colour(bits: &mut Bits<'_>, bit_depth: u32) -> Result<(u32, u32, bool), CodecError> {
    if !take_flag(bits)? {
        return Ok((bit_depth, bit_depth, false));
    }
    let luma = take_u(bits, 3)?.saturating_add(8);
    let chroma = take_u(bits, 3)?.saturating_add(8);
    let _full = take_flag(bits)?;
    let _primaries = take_u(bits, 8)?;
    let _transfer = take_u(bits, 8)?;
    let _matrix = take_u(bits, 8)?;
    Ok((luma, chroma, luma != bit_depth || chroma != bit_depth))
}

fn component(bits: &mut Bits<'_>, model: u32, depth: u32) -> Result<Component, CodecError> {
    let intervals = take_u(bits, 8)?.saturating_add(1);
    let values = take_u(bits, 3)?.saturating_add(1);
    if values > 6 {
        return Err(mal());
    }
    let mut list = Vec::with_capacity(usize::try_from(intervals).unwrap_or(0));
    for _ in 0..intervals {
        let low = take_byte(bits)?;
        let high = take_byte(bits)?;
        let mut value = [0; 6];
        for slot in value.iter_mut().take(usize::try_from(values).unwrap_or(0)) {
            *slot = take_se(bits)?;
        }
        infer(&mut value, model, values);
        if !in_range(&value, model, depth) {
            return Err(mal());
        }
        list.push(Interval { low, high, value });
    }
    Ok(Component { intervals: list })
}

fn infer(value: &mut [i32; 6], model: u32, present: u32) {
    if present == 0 {
        value[0] = 0;
    }
    if present <= 1 {
        value[1] = if model == 0 { 8 } else { 0 };
    }
    if present <= 2 {
        value[2] = if model == 0 { value[1] } else { 0 };
    }
    if present <= 3 {
        value[3] = 0;
    }
    if present <= 4 {
        value[4] = if model == 0 { 0 } else { 1 };
    }
    if present <= 5 {
        value[5] = 0;
    }
}

fn in_range(value: &[i32; 6], model: u32, depth: u32) -> bool {
    let depth = depth.clamp(1, 30);
    let full = (1i32 << depth) - 1;
    let half = 1i32 << depth.saturating_sub(1);
    let (lo, hi) = if model == 0 {
        (0, full)
    } else {
        (-half, half - 1)
    };
    if value.iter().any(|item| *item < lo || *item > hi) {
        return false;
    }
    model != 0
        || (value[1] >= 0
            && value[1] < 16
            && value[2] >= 0
            && value[2] < 16
            && value[3] >= 0
            && value[3] <= value[1]
            && value[4] >= 0
            && value[4] <= value[2])
}

fn display_fields(pic_struct: u32) -> Result<u32, CodecError> {
    match pic_struct {
        0 | 3 | 4 => Ok(2),
        1 | 2 => Ok(1),
        5 | 6 => Ok(3),
        7 => Ok(4),
        8 => Ok(6),
        _ => Err(CodecError::Malformed {
            at_bit: 0,
            what: "pic struct",
        }),
    }
}

fn shift_pts(pts: i64, delay: u32, clock: &TimingClock, timescale: u32) -> i64 {
    if clock.time_scale == 0 {
        return pts;
    }
    let ticks = u64::from(delay)
        .saturating_mul(u64::from(clock.num_units))
        .saturating_mul(u64::from(timescale));
    let scale = u64::from(clock.time_scale);
    if scale == 0 || ticks % scale != 0 {
        return pts;
    }
    let Ok(add) = i64::try_from(ticks / scale) else {
        return pts;
    };
    pts.saturating_add(add)
}

fn skip_timestamp(bits: &mut Bits<'_>, offset_bits: u8) -> Result<(), CodecError> {
    let _ct = take_u(bits, 2)?;
    let _nuit = take_flag(bits)?;
    let _counting = take_u(bits, 5)?;
    let full = take_flag(bits)?;
    let _discontinuity = take_flag(bits)?;
    let _dropped = take_flag(bits)?;
    let _frames = take_u(bits, 8)?;
    if full {
        let _seconds = take_u(bits, 6)?;
        let _minutes = take_u(bits, 6)?;
        let _hours = take_u(bits, 5)?;
    } else if take_flag(bits)? {
        let _seconds = take_u(bits, 6)?;
        if take_flag(bits)? {
            let _minutes = take_u(bits, 6)?;
            if take_flag(bits)? {
                let _hours = take_u(bits, 5)?;
            }
        }
    }
    if offset_bits > 0 {
        let _offset = take_u(bits, u32::from(offset_bits))?;
    }
    Ok(())
}

fn ff_sum(bits: &mut Bits<'_>) -> Result<u32, CodecError> {
    let mut total = 0u32;
    loop {
        let byte = u32::from(take_byte(bits)?);
        if byte == 255 {
            total = total.checked_add(255).ok_or_else(mal)?;
        } else {
            return total.checked_add(byte).ok_or_else(mal);
        }
    }
}

fn take_byte(bits: &mut Bits<'_>) -> Result<u8, CodecError> {
    let value = take_u(bits, 8)?;
    u8::try_from(value).map_err(|_| mal())
}

fn take_u(bits: &mut Bits<'_>, n: u32) -> Result<u32, CodecError> {
    bits.u(n).map_err(|_| mal())
}

fn take_flag(bits: &mut Bits<'_>) -> Result<bool, CodecError> {
    bits.bit().map_err(|_| mal())
}

fn take_ue(bits: &mut Bits<'_>) -> Result<u32, CodecError> {
    bits.ue().map_err(|_| mal())
}

fn take_se(bits: &mut Bits<'_>) -> Result<i32, CodecError> {
    bits.se().map_err(|_| mal())
}

fn mal() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "sei",
    }
}
