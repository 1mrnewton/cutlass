//! Picture order counts, clause 8.2.1, and what is carried from one
//! reference picture to the next to work them out. The software decoder
//! and [`crate::stateless`] both count with these.

use cutlass_decode::CodecError;

use crate::set::Sps;

/// The slice-header elements a picture's order count is worked out from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrderSyntax {
    pub frame_num: u32,
    /// `nal_ref_idc` is not 0.
    pub nal_ref: bool,
    /// `field_pic_flag`.
    pub field: bool,
    /// `pic_order_cnt_lsb`.
    pub poc_lsb: i32,
    /// For order count type 0 the second entry is
    /// `delta_pic_order_cnt_bottom`; for type 1 the two are
    /// `delta_pic_order_cnt`.
    pub delta_poc: [i32; 2],
}

/// What the order counts of the next picture depend on: the previous
/// reference picture's `frame_num`, its offset and its order count, and the
/// largest long-term frame index allowed.
#[derive(Clone, Debug)]
pub(crate) struct Poc {
    pub prev_frame_num: u32,
    pub prev_frame_num_offset: u32,
    pub prev_poc_msb: i32,
    pub prev_poc_lsb: i32,
    pub have_ref: bool,
    pub max_long: Option<u8>,
}

impl Poc {
    pub(crate) fn new() -> Self {
        Self {
            prev_frame_num: 0,
            prev_frame_num_offset: 0,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            have_ref: false,
            max_long: None,
        }
    }

    /// The first slice of a reference picture with these counts has been
    /// read: the pictures after it count from it.
    pub(crate) fn follow(&mut self, frame_num: u32, counts: &Counts) {
        self.prev_frame_num = frame_num;
        self.prev_frame_num_offset = counts.frame_num_offset;
        self.have_ref = true;
    }
}

/// A picture's order counts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Counts {
    /// The picture's own count: a frame's is the smaller of its fields'.
    pub poc: i32,
    /// `TopFieldOrderCnt` and `BottomFieldOrderCnt`. A field picture has its
    /// one count in both.
    pub fields: [i32; 2],
    /// `FrameNumOffset`.
    pub frame_num_offset: u32,
}

/// Clause 8.2.1 for one slice: the picture's counts. A reference picture
/// of type 0 updates the counts the next picture's are worked out from.
pub(crate) fn counts(
    syntax: &OrderSyntax,
    sps: &Sps,
    state: &mut Poc,
    idr: bool,
) -> Result<Counts, CodecError> {
    let (poc, frame_num_offset) = picture_order(syntax, sps, state, idr)?;
    let fields = field_orders(sps, syntax, poc);
    let poc = if syntax.field {
        poc
    } else {
        fields[0].min(fields[1])
    };
    Ok(Counts {
        poc,
        fields,
        frame_num_offset,
    })
}

pub(crate) fn picture_order(
    syntax: &OrderSyntax,
    sps: &Sps,
    state: &mut Poc,
    idr: bool,
) -> Result<(i32, u32), CodecError> {
    let max_fn = 1u32 << sps.log2_max_frame_num();
    let offset = if idr || !state.have_ref {
        0
    } else if syntax.frame_num < state.prev_frame_num {
        state.prev_frame_num_offset.wrapping_add(max_fn)
    } else {
        state.prev_frame_num_offset
    };
    if idr && syntax.nal_ref {
        state.prev_poc_msb = 0;
        state.prev_poc_lsb = if sps.pic_order_cnt_type == 0 {
            syntax.poc_lsb
        } else {
            0
        };
    }
    let poc = if sps.pic_order_cnt_type == 1 {
        poc_type1(
            sps,
            offset,
            syntax.frame_num,
            syntax.nal_ref,
            syntax.delta_poc,
        )?
    } else if sps.pic_order_cnt_type == 2 {
        let abs = i32::try_from(offset.saturating_add(syntax.frame_num)).unwrap_or(0);
        if syntax.nal_ref { 2 * abs } else { 2 * abs - 1 }
    } else if idr {
        syntax.poc_lsb
    } else {
        let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb();
        let prev_msb = if state.have_ref {
            state.prev_poc_msb
        } else {
            0
        };
        let prev_lsb = if state.have_ref {
            state.prev_poc_lsb
        } else {
            0
        };
        let msb = if syntax.poc_lsb < prev_lsb && prev_lsb - syntax.poc_lsb >= max_lsb / 2 {
            prev_msb + max_lsb
        } else if syntax.poc_lsb > prev_lsb && syntax.poc_lsb - prev_lsb > max_lsb / 2 {
            prev_msb - max_lsb
        } else {
            prev_msb
        };
        if syntax.nal_ref {
            state.prev_poc_msb = msb;
            state.prev_poc_lsb = syntax.poc_lsb;
        }
        msb + syntax.poc_lsb
    };
    Ok((poc, offset))
}

/// Clause 8.2.1: the order counts of a frame's top and bottom fields, from
/// the count [`picture_order`] gave. Type 0 gives the top field's and sends
/// the bottom one as a difference; type 1 gives the smaller of the two,
/// which lie a known distance apart; type 2 gives both the same. A field
/// picture has the one count.
pub(crate) fn field_orders(sps: &Sps, syntax: &OrderSyntax, poc: i32) -> [i32; 2] {
    if syntax.field {
        return [poc; 2];
    }
    match sps.pic_order_cnt_type {
        0 => [poc, poc.saturating_add(syntax.delta_poc[1])],
        1 => {
            let apart = sps
                .offset_for_top_to_bottom_field
                .saturating_add(syntax.delta_poc[1]);
            let top = poc.saturating_sub(apart.min(0));
            [top, top.saturating_add(apart)]
        }
        _ => [poc; 2],
    }
}

/// Clause 8.2.1.2. A frame's order count is the smaller of the two fields.
pub(crate) fn poc_type1(
    sps: &Sps,
    frame_num_offset: u32,
    frame_num: u32,
    nal_ref: bool,
    delta: [i32; 2],
) -> Result<i32, CodecError> {
    let mut abs_frame_num = if sps.offset_for_ref_frame.is_empty() {
        0
    } else {
        i64::from(frame_num_offset) + i64::from(frame_num)
    };
    if !nal_ref && abs_frame_num > 0 {
        abs_frame_num -= 1;
    }
    let mut expected = 0i64;
    if abs_frame_num > 0 {
        let cycle = sps.offset_for_ref_frame.len() as i64;
        let per_cycle: i64 = sps
            .offset_for_ref_frame
            .iter()
            .map(|offset| i64::from(*offset))
            .sum();
        let cycle_count = (abs_frame_num - 1) / cycle;
        let in_cycle = usize::try_from((abs_frame_num - 1) % cycle).unwrap_or(0);
        expected = cycle_count * per_cycle;
        for offset in sps.offset_for_ref_frame.iter().take(in_cycle + 1) {
            expected += i64::from(*offset);
        }
    }
    if !nal_ref {
        expected += i64::from(sps.offset_for_non_ref_pic);
    }
    let top = expected + i64::from(delta[0]);
    let bottom = top + i64::from(sps.offset_for_top_to_bottom_field) + i64::from(delta[1]);
    i32::try_from(top.min(bottom)).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "picture order count",
    })
}

/// The order count and `FrameNumOffset` of a frame clause 8.2.5.2 infers
/// for a missing `frame_num`. Type 0 has none to give; such a frame is never
/// output, so 0 serves.
pub(crate) fn ghost_order(
    sps: &Sps,
    state: &Poc,
    frame_num: u32,
) -> Result<(i32, u32), CodecError> {
    let max_fn = 1u32 << sps.log2_max_frame_num();
    let offset = if frame_num < state.prev_frame_num {
        state.prev_frame_num_offset.wrapping_add(max_fn)
    } else {
        state.prev_frame_num_offset
    };
    let poc = if sps.pic_order_cnt_type == 1 {
        poc_type1(sps, offset, frame_num, true, [0, 0])?
    } else if sps.pic_order_cnt_type == 2 {
        let abs = i32::try_from(offset.saturating_add(frame_num)).unwrap_or(0);
        2 * abs
    } else {
        0
    };
    Ok((poc, offset))
}
