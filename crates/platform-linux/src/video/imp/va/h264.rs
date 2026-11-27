//! H.264 decoding on VA-API: progressive 8-bit 4:2:0 Constrained Baseline,
//! Main and High.
//!
//! `cutlass-h264`'s stateless front end reads the bitstream: parameter
//! sets, slice headers, order counts, `frame_num` gaps, reference marking
//! and lists, and the decoded picture buffer. This file turns each picture
//! it hands over into VA-API's picture, matrix and slice buffers, with the
//! meanings FFmpeg's `vaapi_h264.c` gives them. Interlaced and other
//! streams the front end refuses go to the Rust decoder through the
//! fallback.

use std::sync::Arc;

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Output, Packet};
use cutlass_h264::stateless::{
    self, Accelerator, Picture, PredWeightTable, Reference, Slice, Sps, StatelessDecoder,
};

use super::abi::{self, h264_iq, h264_pic, h264_slice, picture_h264};
use super::decode::{self, Session, Shape, byte, flag, short};
use super::display::{self, Node};
use super::surface::{Buffer, Readback, Surface, Window};
use crate::video::backend::{note_decline, note_open};
use crate::video::colour::Signal;
use crate::video::fallback::Fallback;
use crate::video::imp::blob::{Base, Blob, Field};

pub const BACKEND: &str = "VA-API";

/// Surfaces beyond the references and the pictures held for display
/// order: the picture being decoded and a margin.
const EXTRA_SURFACES: usize = 3;

/// `ReferenceFrames` holds 16.
const MAX_REFS: usize = 16;

/// `RefPicList0` and `RefPicList1` hold 32.
const MAX_LIST: usize = 32;

/// The profiles that decode a stream of each `profile_idc`, the exact one
/// first: a stream decodes on a profile whose tools include its own.
fn profiles(profile_idc: u8) -> &'static [i32] {
    match profile_idc {
        66 => &[
            abi::VAProfileH264ConstrainedBaseline,
            abi::VAProfileH264Main,
            abi::VAProfileH264High,
        ],
        77 => &[abi::VAProfileH264Main, abi::VAProfileH264High],
        100 => &[abi::VAProfileH264High],
        _ => &[],
    }
}

/// The VA-API H.264 opener. Declines when libva is missing or no render
/// node decodes H.264.
pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(open(params)?))
}

/// The VA-API H.264 decoder, inside the fallback to the Rust decoder.
pub fn open(params: &CodecParams) -> Result<Fallback, CodecError> {
    let decline = CodecError::Unsupported { what: "vaapi" };
    if params.codec != CodecId::H264 {
        return Err(decline);
    }
    if params.width != 0 || params.height != 0 {
        crate::video::limits::video_size(params.width, params.height)?;
    }
    if super::api::va().is_none() {
        note_decline(BACKEND, params.codec, false, "libva is not installed");
        return Err(decline);
    }
    let Some(node) = display::find(abi::VAProfileH264High, abi::VAEntrypointVLD)
        .or_else(|| display::find(abi::VAProfileH264Main, abi::VAEntrypointVLD))
    else {
        note_decline(BACKEND, params.codec, false, "no render node decodes H.264");
        return Err(decline);
    };
    let inner = StatelessDecoder::open(params, Accel::new(node))?;
    let decoder = VaDecoder {
        inner,
        readback: Readback::new(node),
    };
    note_open(BACKEND, params.codec, false);
    Ok(Fallback::new(params, Box::new(decoder), BACKEND))
}

/// Whether a render node decodes H.264 here.
#[cfg(test)]
pub fn available() -> bool {
    super::api::va().is_some()
        && display::find(abi::VAProfileH264High, abi::VAEntrypointVLD).is_some()
}

/// The stateless decoder with its pictures read back.
struct VaDecoder {
    inner: StatelessDecoder<Accel>,
    readback: Readback,
}

impl VaDecoder {
    fn pictures(
        &mut self,
        decoded: Vec<stateless::Output<Arc<Surface>>>,
    ) -> Result<Vec<Output>, CodecError> {
        decoded
            .into_iter()
            .map(|picture| {
                let sps = &picture.sps;
                let (left, top, width, height) = sps.crop_rect()?;
                let window = Window {
                    left,
                    top,
                    width: width & !1,
                    height: height & !1,
                };
                decode::picture(
                    &mut self.readback,
                    &picture.surface,
                    window,
                    signal(sps),
                    picture.pts,
                    picture.timescale,
                )
            })
            .collect()
    }
}

impl Decoder for VaDecoder {
    fn codec(&self) -> CodecId {
        CodecId::H264
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let decoded = self.inner.push(packet)?;
        self.pictures(decoded)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let decoded = self.inner.drain()?;
        self.pictures(decoded)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// The colour description of the SPS's VUI, with the values H.264 infers
/// when it is absent.
fn signal(sps: &Sps) -> Signal {
    let vui = &sps.vui;
    Signal {
        primaries: vui.colour_primaries,
        transfer: vui.transfer_characteristics,
        matrix: vui.matrix_coefficients,
        full_range: Some(vui.video_full_range_flag),
    }
}

/// Decodes pictures on one render node.
struct Accel {
    node: &'static Node,
    session: Option<Session>,
}

impl Accel {
    fn new(node: &'static Node) -> Self {
        Self {
            node,
            session: None,
        }
    }
}

impl Accelerator for Accel {
    type Surface = Arc<Surface>;

    fn sequence(&mut self, sps: &Sps) -> Result<(), CodecError> {
        // The front end takes progressive 8-bit 4:2:0 only.
        let Some(profile) = profiles(sps.profile_idc)
            .iter()
            .copied()
            .find(|profile| self.node.offers(*profile, abi::VAEntrypointVLD))
        else {
            return Err(CodecError::Unsupported {
                what: "this GPU does not decode the H.264 profile",
            });
        };
        let (width, height) = sps.coded_size();
        let shape = Shape {
            profile,
            width,
            height,
            depth: 8,
            surfaces: usize::from(sps.max_num_ref_frames) + sps.output_delay() + EXTRA_SURFACES,
        };
        decode::renew(self.node, &mut self.session, shape)
    }

    fn missing(&mut self, _sps: &Sps, _frame_num: u32) -> Result<Arc<Surface>, CodecError> {
        // Never output or in a list; FFmpeg leaves its samples as they are
        // for hardware too.
        decode::current(&self.session)?.take()
    }

    fn decode(&mut self, picture: &Picture<'_, Arc<Surface>>) -> Result<Arc<Surface>, CodecError> {
        let node = self.node;
        let session = decode::current(&self.session)?;
        let target = session.take()?;
        let context = session.context.id;
        let params = [
            Buffer::blob(
                node,
                context,
                abi::VAPictureParameterBufferType,
                &picture_params(picture, target.id, |surface| surface.id),
            )?,
            Buffer::blob(
                node,
                context,
                abi::VAIQMatrixBufferType,
                &iq_matrix(picture.pps),
            )?,
        ];
        let mut slices = Vec::with_capacity(picture.slices.len() * 2);
        for slice in picture.slices {
            let blob = slice_params(
                slice,
                picture.refs,
                picture.pps.weighted_bipred_idc,
                |surface| surface.id,
            );
            slices.push(Buffer::blob(
                node,
                context,
                abi::VASliceParameterBufferType,
                &blob,
            )?);
            slices.push(Buffer::new(
                node,
                context,
                abi::VASliceDataBufferType,
                slice.nal,
            )?);
        }
        session
            .context
            .render(target.id, &params, &slices)
            .inspect_err(|_| {
                cutlass_logger::debug!("VA-API: H.264 frame_num {} failed", picture.frame_num)
            })?;
        Ok(target)
    }
}

/// One `VAPictureH264`.
fn set_picture(blob: &mut Blob, at: Base, id: u32, frame_idx: u32, flags: u32, order: (i32, i32)) {
    blob.set(picture_h264::PICTURE_ID.of(at), id);
    blob.set(picture_h264::FRAME_IDX.of(at), frame_idx);
    blob.set(picture_h264::FLAGS.of(at), flags);
    blob.set(picture_h264::TOP_FIELD_ORDER_CNT.of(at), order.0);
    blob.set(picture_h264::BOTTOM_FIELD_ORDER_CNT.of(at), order.1);
}

fn set_invalid(blob: &mut Blob, at: Base) {
    set_picture(
        blob,
        at,
        abi::VA_INVALID_SURFACE,
        0,
        abi::VA_PICTURE_H264_INVALID,
        (0, 0),
    );
}

/// A reference frame as a `VAPictureH264`.
fn set_reference<S>(blob: &mut Blob, at: Base, reference: &Reference<S>, id: u32) {
    let flags = if reference.long_term {
        abi::VA_PICTURE_H264_LONG_TERM_REFERENCE
    } else {
        abi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
    };
    set_picture(
        blob,
        at,
        id,
        reference.frame_num,
        flags,
        (
            reference.top_field_order_cnt,
            reference.bottom_field_order_cnt,
        ),
    );
}

/// `VAPictureParameterBufferH264`.
fn picture_params<S>(picture: &Picture<'_, S>, target: u32, id: impl Fn(&S) -> u32) -> Blob {
    use h264_pic as p;
    let (sps, pps) = (picture.sps, picture.pps);
    let mut blob = Blob::new(p::SIZE);
    let reference = picture.nal_ref_idc != 0;
    set_picture(
        &mut blob,
        p::CURR_PIC,
        target,
        picture.frame_num,
        if reference {
            abi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
        } else {
            0
        },
        (picture.top_field_order_cnt, picture.bottom_field_order_cnt),
    );
    for slot in 0..MAX_REFS {
        if let Some(at) = p::REFERENCE_FRAMES.at(slot) {
            set_invalid(&mut blob, at);
        }
    }
    for (slot, reference) in picture.refs.iter().take(MAX_REFS).enumerate() {
        if let Some(at) = p::REFERENCE_FRAMES.at(slot) {
            set_reference(&mut blob, at, reference, id(&reference.surface));
        }
    }
    let (width, height) = sps.coded_size();
    blob.set(
        p::PICTURE_WIDTH_IN_MBS_MINUS1,
        u16::try_from(width / 16 - 1).unwrap_or(u16::MAX),
    );
    blob.set(
        p::PICTURE_HEIGHT_IN_MBS_MINUS1,
        u16::try_from(height / 16 - 1).unwrap_or(u16::MAX),
    );
    blob.set(p::BIT_DEPTH_LUMA_MINUS8, sps.bit_depth_luma_minus8);
    blob.set(p::BIT_DEPTH_CHROMA_MINUS8, sps.bit_depth_chroma_minus8);
    blob.set(p::NUM_REF_FRAMES, sps.max_num_ref_frames);
    let seq = [
        (p::CHROMA_FORMAT_IDC, u32::from(sps.chroma_format_idc)),
        (p::RESIDUAL_COLOUR_TRANSFORM_FLAG, 0),
        (
            p::GAPS_IN_FRAME_NUM_VALUE_ALLOWED_FLAG,
            flag(sps.gaps_in_frame_num_value_allowed_flag),
        ),
        (p::FRAME_MBS_ONLY_FLAG, flag(sps.frame_mbs_only_flag)),
        (
            p::MB_ADAPTIVE_FRAME_FIELD_FLAG,
            flag(sps.mb_adaptive_frame_field_flag),
        ),
        (
            p::DIRECT_8X8_INFERENCE_FLAG,
            flag(sps.direct_8x8_inference_flag),
        ),
        // Clause A.3.3.2: levels 3.1 and up.
        (p::MIN_LUMA_BI_PRED_SIZE8X8, flag(sps.level_idc >= 31)),
        (
            p::LOG2_MAX_FRAME_NUM_MINUS4,
            u32::from(sps.log2_max_frame_num_minus4),
        ),
        (p::PIC_ORDER_CNT_TYPE, u32::from(sps.pic_order_cnt_type)),
        (
            p::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
            u32::from(sps.log2_max_pic_order_cnt_lsb_minus4),
        ),
        (
            p::DELTA_PIC_ORDER_ALWAYS_ZERO_FLAG,
            flag(sps.delta_pic_order_always_zero_flag),
        ),
    ];
    for (field, value) in seq {
        blob.set_bits(field, value);
    }
    blob.set(p::PIC_INIT_QP_MINUS26, byte(pps.pic_init_qp_minus26));
    blob.set(p::PIC_INIT_QS_MINUS26, byte(pps.pic_init_qs_minus26));
    blob.set(p::CHROMA_QP_INDEX_OFFSET, byte(pps.chroma_qp_index_offset));
    blob.set(
        p::SECOND_CHROMA_QP_INDEX_OFFSET,
        byte(pps.second_chroma_qp_index_offset),
    );
    let pic = [
        (
            p::ENTROPY_CODING_MODE_FLAG,
            flag(pps.entropy_coding_mode_flag),
        ),
        (p::WEIGHTED_PRED_FLAG, flag(pps.weighted_pred_flag)),
        (p::WEIGHTED_BIPRED_IDC, u32::from(pps.weighted_bipred_idc)),
        (
            p::TRANSFORM_8X8_MODE_FLAG,
            flag(pps.transform_8x8_mode_flag),
        ),
        (p::FIELD_PIC_FLAG, 0),
        (
            p::CONSTRAINED_INTRA_PRED_FLAG,
            flag(pps.constrained_intra_pred_flag),
        ),
        (
            p::PIC_ORDER_PRESENT_FLAG,
            flag(pps.bottom_field_pic_order_in_frame_present_flag),
        ),
        (
            p::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG,
            flag(pps.deblocking_filter_control_present_flag),
        ),
        (
            p::REDUNDANT_PIC_CNT_PRESENT_FLAG,
            flag(pps.redundant_pic_cnt_present_flag),
        ),
        (p::REFERENCE_PIC_FLAG, flag(reference)),
    ];
    for (field, value) in pic {
        blob.set_bits(field, value);
    }
    // frame_num is below 2^16: log2_max_frame_num_minus4 is at most 12.
    blob.set(
        p::FRAME_NUM,
        u16::try_from(picture.frame_num).unwrap_or(u16::MAX),
    );
    blob
}

/// `VAIQMatrixBufferH264`: the picture's lists in coded zig-zag order,
/// the 8×8 ones of Intra Y and Inter Y.
fn iq_matrix(pps: &stateless::Pps) -> Blob {
    use h264_iq as q;
    let mut blob = Blob::new(q::SIZE);
    for (list, values) in pps.scaling_list_4x4().iter().enumerate() {
        for (index, value) in values.iter().enumerate() {
            blob.set_at(q::SCALING_LIST_4X4, list * 16 + index, *value);
        }
    }
    for (list, values) in pps.scaling_list_8x8().iter().take(2).enumerate() {
        for (index, value) in values.iter().enumerate() {
            blob.set_at(q::SCALING_LIST_8X8, list * 64 + index, *value);
        }
    }
    blob
}

/// `VASliceParameterBufferH264` for one slice, whose data buffer is its
/// whole NAL unit. `bipred_idc` is the PPS's `weighted_bipred_idc`.
fn slice_params<S>(
    slice: &Slice<'_>,
    refs: &[Reference<S>],
    bipred_idc: u8,
    id: impl Fn(&S) -> u32,
) -> Blob {
    use h264_slice as s;
    let header = slice.header;
    let kind = header.slice_kind();
    let mut blob = Blob::new(s::SIZE);
    blob.set(
        s::SLICE_DATA_SIZE,
        u32::try_from(slice.nal.len()).unwrap_or(u32::MAX),
    );
    blob.set(s::SLICE_DATA_OFFSET, 0);
    blob.set(s::SLICE_DATA_FLAG, abi::VA_SLICE_DATA_FLAG_ALL);
    // RBSP bits from the NAL unit header to the end of the slice header, as
    // FFmpeg counts them; the drivers skip the escapes themselves.
    blob.set(
        s::SLICE_DATA_BIT_OFFSET,
        u16::try_from(slice.header_bits).unwrap_or(u16::MAX),
    );
    blob.set(
        s::FIRST_MB_IN_SLICE,
        u16::try_from(header.first_mb_in_slice).unwrap_or(u16::MAX),
    );
    blob.set(s::SLICE_TYPE, kind);
    let b = kind == 1;
    let lists = match kind {
        0 | 3 => 1,
        1 => 2,
        _ => 0,
    };
    blob.set(
        s::DIRECT_SPATIAL_MV_PRED_FLAG,
        u8::from(b && header.direct_spatial_mv_pred_flag),
    );
    blob.set(
        s::NUM_REF_IDX_L0_ACTIVE_MINUS1,
        if lists > 0 {
            header.num_ref_idx_l0_active_minus1
        } else {
            0
        },
    );
    blob.set(
        s::NUM_REF_IDX_L1_ACTIVE_MINUS1,
        if lists > 1 {
            header.num_ref_idx_l1_active_minus1
        } else {
            0
        },
    );
    blob.set(s::CABAC_INIT_IDC, header.cabac_init_idc);
    blob.set(s::SLICE_QP_DELTA, byte(header.slice_qp_delta));
    blob.set(
        s::DISABLE_DEBLOCKING_FILTER_IDC,
        header.disable_deblocking_filter_idc,
    );
    blob.set(
        s::SLICE_ALPHA_C0_OFFSET_DIV2,
        byte(header.slice_alpha_c0_offset_div2),
    );
    blob.set(
        s::SLICE_BETA_OFFSET_DIV2,
        byte(header.slice_beta_offset_div2),
    );

    for (list, array) in [s::REF_PIC_LIST0, s::REF_PIC_LIST1].into_iter().enumerate() {
        for index in 0..MAX_LIST {
            if let Some(at) = array.at(index) {
                set_invalid(&mut blob, at);
            }
        }
        if list >= lists {
            continue;
        }
        for (index, reference) in slice.ref_lists[list].iter().take(MAX_LIST).enumerate() {
            if let (Some(at), Some(frame)) = (array.at(index), refs.get(*reference)) {
                set_reference(&mut blob, at, frame, id(&frame.surface));
            }
        }
    }

    let counts = [
        usize::from(header.num_ref_idx_l0_active_minus1) + 1,
        usize::from(header.num_ref_idx_l1_active_minus1) + 1,
    ];
    weights(
        &mut blob,
        header.pred_weight_table.as_ref(),
        b && bipred_idc == 2,
        counts,
        lists,
    );
    blob
}

/// The weights of each list, with the defaults where the stream codes
/// none, as FFmpeg fills them: VA-API reads every entry up to the active
/// count. Implicit weighting has FFmpeg's denominators of 5.
fn weights(
    blob: &mut Blob,
    table: Option<&PredWeightTable>,
    implicit: bool,
    counts: [usize; 2],
    lists: usize,
) {
    use h264_slice as s;
    let (luma_denom, chroma_denom) = match (table, implicit) {
        (Some(table), _) => (table.luma_log2_weight_denom, table.chroma_log2_weight_denom),
        (None, true) => (5, 5),
        (None, false) => (0, 0),
    };
    blob.set(s::LUMA_LOG2_WEIGHT_DENOM, luma_denom);
    blob.set(s::CHROMA_LOG2_WEIGHT_DENOM, chroma_denom);
    let fields = [
        (
            s::LUMA_WEIGHT_L0_FLAG,
            s::LUMA_WEIGHT_L0,
            s::LUMA_OFFSET_L0,
            s::CHROMA_WEIGHT_L0_FLAG,
            s::CHROMA_WEIGHT_L0,
            s::CHROMA_OFFSET_L0,
        ),
        (
            s::LUMA_WEIGHT_L1_FLAG,
            s::LUMA_WEIGHT_L1,
            s::LUMA_OFFSET_L1,
            s::CHROMA_WEIGHT_L1_FLAG,
            s::CHROMA_WEIGHT_L1,
            s::CHROMA_OFFSET_L1,
        ),
    ];
    let luma_default = 1_i32 << luma_denom;
    let chroma_default = 1_i32 << chroma_denom;
    for (list, (luma_flag, luma_weight, luma_offset, chroma_flag, chroma_weight, chroma_offset)) in
        fields.into_iter().enumerate().take(lists)
    {
        let entries = table.map_or(
            &[][..],
            |table| {
                if list == 0 { &table.l0 } else { &table.l1 }
            },
        );
        blob.set(
            luma_flag,
            u8::from(entries.iter().any(|entry| entry.luma_weight_flag)),
        );
        blob.set(
            chroma_flag,
            u8::from(entries.iter().any(|entry| entry.chroma_weight_flag)),
        );
        for index in 0..counts[list].min(MAX_LIST) {
            let entry = entries.get(index);
            let luma = entry.map_or((luma_default, 0), |e| (e.luma_weight, e.luma_offset));
            blob.set_at(luma_weight, index, short(luma.0));
            blob.set_at(luma_offset, index, short(luma.1));
            for plane in 0..2 {
                let chroma = entry.map_or((chroma_default, 0), |e| {
                    (e.chroma_weight[plane], e.chroma_offset[plane])
                });
                blob.set_at(chroma_weight, index * 2 + plane, short(chroma.0));
                blob.set_at(chroma_offset, index * 2 + plane, short(chroma.1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds every buffer of each picture the front end hands over, as
    /// the VA-API accelerator does, and checks them against the picture.
    struct Recorder {
        next: u32,
        pictures: usize,
        slices: usize,
    }

    impl Accelerator for Recorder {
        type Surface = u32;

        fn sequence(&mut self, _sps: &Sps) -> Result<(), CodecError> {
            Ok(())
        }

        fn missing(&mut self, _sps: &Sps, _frame_num: u32) -> Result<u32, CodecError> {
            self.next += 1;
            Ok(self.next)
        }

        fn decode(&mut self, picture: &Picture<'_, u32>) -> Result<u32, CodecError> {
            self.next += 1;
            let target = self.next;
            let pic = picture_params(picture, target, |surface| *surface);
            let curr = h264_pic::CURR_PIC;
            assert_eq!(pic.get(picture_h264::PICTURE_ID.of(curr)), target);
            assert_eq!(
                pic.get(picture_h264::TOP_FIELD_ORDER_CNT.of(curr)),
                picture.top_field_order_cnt
            );
            assert_eq!(u32::from(pic.get(h264_pic::FRAME_NUM)), picture.frame_num);
            for (slot, reference) in picture.refs.iter().enumerate() {
                let at = h264_pic::REFERENCE_FRAMES.at(slot).expect("16 at most");
                assert_eq!(pic.get(picture_h264::PICTURE_ID.of(at)), reference.surface);
                assert_eq!(pic.get(picture_h264::FRAME_IDX.of(at)), reference.frame_num);
            }
            if let Some(at) = h264_pic::REFERENCE_FRAMES.at(picture.refs.len()) {
                assert_eq!(
                    pic.get(picture_h264::FLAGS.of(at)),
                    abi::VA_PICTURE_H264_INVALID
                );
            }
            iq_matrix(picture.pps);
            for slice in picture.slices {
                let blob = slice_params(
                    slice,
                    picture.refs,
                    picture.pps.weighted_bipred_idc,
                    |surface| *surface,
                );
                assert_eq!(
                    blob.get(h264_slice::SLICE_DATA_SIZE) as usize,
                    slice.nal.len()
                );
                assert_eq!(
                    u32::from(blob.get(h264_slice::SLICE_DATA_BIT_OFFSET)),
                    slice.header_bits
                );
                assert!((slice.header_bits as usize) < slice.nal.len() * 8);
                assert_eq!(blob.get(h264_slice::SLICE_TYPE), slice.header.slice_kind());
                let lists = [h264_slice::REF_PIC_LIST0, h264_slice::REF_PIC_LIST1];
                for (list, array) in lists.into_iter().enumerate() {
                    for (index, reference) in slice.ref_lists[list].iter().enumerate() {
                        let at = array.at(index).expect("32 at most");
                        let frame = &picture.refs[*reference];
                        assert_eq!(blob.get(picture_h264::PICTURE_ID.of(at)), frame.surface);
                        assert_eq!(
                            blob.get(picture_h264::TOP_FIELD_ORDER_CNT.of(at)),
                            frame.top_field_order_cnt
                        );
                    }
                    let past = array.at(slice.ref_lists[list].len()).expect("in range");
                    assert_eq!(
                        blob.get(picture_h264::PICTURE_ID.of(past)),
                        abi::VA_INVALID_SURFACE
                    );
                }
                self.slices += 1;
            }
            self.pictures += 1;
            Ok(target)
        }
    }

    #[test]
    fn the_buffers_of_each_picture_name_it_and_its_references() {
        for (name, packets, params) in crate::video::imp::tests::h264_streams() {
            let recorder = Recorder {
                next: 0,
                pictures: 0,
                slices: 0,
            };
            let mut decoder = StatelessDecoder::open(&params, recorder).expect("opens");
            let mut out = 0;
            for packet in &packets {
                out += decoder.push(packet).expect("push").len();
            }
            out += decoder.drain().expect("drain").len();
            let recorder = decoder.accelerator();
            assert_eq!(recorder.pictures, packets.len(), "{name}");
            assert!(recorder.slices >= recorder.pictures, "{name}");
            assert_eq!(out, recorder.pictures, "{name}");
            eprintln!(
                "{name}: {} pictures, {} slices",
                recorder.pictures, recorder.slices
            );
        }
    }

    #[test]
    fn a_profile_decodes_on_itself_then_on_its_supersets() {
        assert_eq!(profiles(66)[0], abi::VAProfileH264ConstrainedBaseline);
        assert_eq!(
            profiles(77),
            [abi::VAProfileH264Main, abi::VAProfileH264High]
        );
        assert_eq!(profiles(100), [abi::VAProfileH264High]);
        assert!(profiles(110).is_empty());
    }

    #[test]
    fn missing_weights_are_the_defaults_and_implicit_ones_use_five() {
        let mut blob = Blob::new(h264_slice::SIZE);
        weights(&mut blob, None, true, [2, 1], 2);
        assert_eq!(blob.get(h264_slice::LUMA_LOG2_WEIGHT_DENOM), 5);
        let second = h264_slice::LUMA_WEIGHT_L0.at(1).expect("in range");
        assert_eq!(blob.get(second), 32);
        assert_eq!(blob.get(h264_slice::LUMA_WEIGHT_L0_FLAG), 0);
        let unused = h264_slice::LUMA_WEIGHT_L1.at(1).expect("in range");
        assert_eq!(blob.get(unused), 0);
    }
}
