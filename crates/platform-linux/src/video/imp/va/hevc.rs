//! HEVC Main and Main 10 decoding on VA-API.
//!
//! `cutlass-hevc`'s stateless front end reads the bitstream: parameter
//! sets, slice headers, order counts, reference sets and lists, and the
//! decoded picture buffer. This file turns each picture it hands over into
//! VA-API's picture, matrix and slice buffers, with the meanings FFmpeg's
//! `vaapi_hevc.c` gives them, which is what the Intel and AMD drivers are
//! tested against.

use std::sync::Arc;

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Output, Packet};
use cutlass_hevc::stateless::{
    self, Accelerator, Picture, PredWeightTable, Reference, RpsSet, ScalingList, Slice, SliceType,
    Sps, StatelessDecoder,
};

use super::abi::{self, hevc_iq, hevc_pic, hevc_slice, picture_hevc};
use super::decode::{self, Session, Shape, byte, flag};
use super::display::{self, Node};
use super::surface::{Buffer, Readback, Surface, Window};
use crate::video::backend::{note_decline, note_open};
use crate::video::colour::Signal;
use crate::video::fallback::Fallback;
use crate::video::imp::blob::{Base, Blob, Field};

pub const BACKEND: &str = "VA-API";

/// Surfaces beyond the decoded picture buffer: the picture being decoded
/// and a margin for references the stream lacks.
const EXTRA_SURFACES: usize = 3;

/// `ReferenceFrames` and each `RefPicList` hold 15.
const MAX_REFS: usize = 15;

/// "No picture" in a reference list.
const NO_REF: u8 = 0xff;

/// The VA-API HEVC opener. Declines when libva is missing or no render
/// node decodes HEVC.
pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(open(params)?))
}

/// The VA-API HEVC decoder, inside the fallback to the Rust decoder.
pub fn open(params: &CodecParams) -> Result<Fallback, CodecError> {
    let decline = CodecError::Unsupported { what: "vaapi" };
    if params.codec != CodecId::Hevc {
        return Err(decline);
    }
    if params.width != 0 || params.height != 0 {
        crate::video::limits::video_size(params.width, params.height)?;
    }
    if super::api::va().is_none() {
        note_decline(BACKEND, params.codec, false, "libva is not installed");
        return Err(decline);
    }
    let main = |node: &&Node| node.offers(abi::VAProfileHEVCMain, abi::VAEntrypointVLD);
    let main10 = |node: &&Node| node.offers(abi::VAProfileHEVCMain10, abi::VAEntrypointVLD);
    let nodes = display::nodes();
    let Some(node) = nodes
        .iter()
        .find(|node| main(node) && main10(node))
        .or_else(|| display::find(abi::VAProfileHEVCMain, abi::VAEntrypointVLD))
    else {
        note_decline(BACKEND, params.codec, false, "no render node decodes HEVC");
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

/// Whether a render node decodes HEVC here.
#[cfg(test)]
pub fn available() -> bool {
    super::api::va().is_some()
        && display::find(abi::VAProfileHEVCMain, abi::VAEntrypointVLD).is_some()
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
                let window = Window::cropped(
                    sps.pic_width_in_luma_samples,
                    sps.pic_height_in_luma_samples,
                    sps.crop(),
                )?;
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
        CodecId::Hevc
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

/// The colour description of the SPS's VUI, with the values H.265 infers
/// when it is absent.
fn signal(sps: &Sps) -> Signal {
    let vui = &sps.vui;
    Signal {
        primaries: vui.colour_primaries,
        transfer: vui.transfer_characteristics,
        matrix: vui.matrix_coeffs,
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
        if sps.chroma_format_idc != 1 || sps.separate_colour_plane_flag {
            return Err(CodecError::Unsupported {
                what: "VA-API decodes 4:2:0 HEVC only",
            });
        }
        let depth = sps.bit_depth();
        let profile = match (depth, sps.bit_depth_chroma()) {
            (8, 8) => abi::VAProfileHEVCMain,
            (10, 10) => abi::VAProfileHEVCMain10,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "VA-API decodes 8 and 10-bit HEVC only",
                });
            }
        };
        let shape = Shape {
            profile,
            width: sps.pic_width_in_luma_samples,
            height: sps.pic_height_in_luma_samples,
            depth,
            surfaces: sps.max_dec_pic_buffering() + EXTRA_SURFACES,
        };
        decode::renew(self.node, &mut self.session, shape)
    }

    fn missing(&mut self, _sps: &Sps, _poc: i32) -> Result<Arc<Surface>, CodecError> {
        // Left as it is, as FFmpeg leaves it for hardware: it stands in for
        // a picture the stream lost and is never output.
        decode::current(&self.session)?.take()
    }

    fn decode(&mut self, picture: &Picture<'_, Arc<Surface>>) -> Result<Arc<Surface>, CodecError> {
        let node = self.node;
        let session = decode::current(&self.session)?;
        let target = session.take()?;
        let context = session.context.id;
        let slots = slots(picture.refs);
        let mut params = vec![Buffer::blob(
            node,
            context,
            abi::VAPictureParameterBufferType,
            &picture_params(picture, target.id, &slots, |surface| surface.id),
        )?];
        if let Some(list) = ScalingList::active(picture.sps, picture.pps) {
            params.push(Buffer::blob(
                node,
                context,
                abi::VAIQMatrixBufferType,
                &iq_matrix(list),
            )?);
        }
        let mut slices = Vec::with_capacity(picture.slices.len() * 2);
        for (index, slice) in picture.slices.iter().enumerate() {
            let last = index + 1 == picture.slices.len();
            let blob = slice_params(slice, &slots, last, picture.sps.chroma_format_idc != 0);
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
                cutlass_logger::debug!("VA-API: HEVC picture {} failed", picture.poc)
            })?;
        Ok(target)
    }
}

/// The `ReferenceFrames` slot of each of `refs`: those the current picture
/// uses first, so a stream that keeps more than 15 drops only pictures
/// kept for later ones.
fn slots<S>(refs: &[Reference<'_, S>]) -> Vec<Option<u8>> {
    let mut order: Vec<usize> = (0..refs.len()).collect();
    order.sort_by_key(|&index| matches!(refs[index].set, RpsSet::Foll));
    let mut slots = vec![None; refs.len()];
    for (slot, &index) in order.iter().take(MAX_REFS).enumerate() {
        slots[index] = u8::try_from(slot).ok();
    }
    slots
}

fn small(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

fn set_picture(blob: &mut Blob, at: Base, id: u32, poc: i32, flags: u32) {
    blob.set(picture_hevc::PICTURE_ID.of(at), id);
    blob.set(picture_hevc::PIC_ORDER_CNT.of(at), poc);
    blob.set(picture_hevc::FLAGS.of(at), flags);
}

/// `VAPictureParameterBufferHEVC`.
fn picture_params<S>(
    picture: &Picture<'_, S>,
    target: u32,
    slots: &[Option<u8>],
    id: impl Fn(&S) -> u32,
) -> Blob {
    use hevc_pic as p;
    let (sps, pps) = (picture.sps, picture.pps);
    let mut blob = Blob::new(p::SIZE);
    set_picture(&mut blob, p::CURR_PIC, target, picture.poc, 0);
    for slot in 0..MAX_REFS {
        if let Some(at) = p::REFERENCE_FRAMES.at(slot) {
            set_picture(
                &mut blob,
                at,
                abi::VA_INVALID_SURFACE,
                0,
                abi::VA_PICTURE_HEVC_INVALID,
            );
        }
    }
    for (reference, slot) in picture.refs.iter().zip(slots) {
        let Some(at) = slot.and_then(|slot| p::REFERENCE_FRAMES.at(usize::from(slot))) else {
            continue;
        };
        let mut flags = match reference.set {
            RpsSet::StCurrBefore => abi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
            RpsSet::StCurrAfter => abi::VA_PICTURE_HEVC_RPS_ST_CURR_AFTER,
            RpsSet::LtCurr => abi::VA_PICTURE_HEVC_RPS_LT_CURR,
            RpsSet::Foll => 0,
        };
        if reference.long_term {
            flags |= abi::VA_PICTURE_HEVC_LONG_TERM_REFERENCE;
        }
        set_picture(&mut blob, at, id(reference.surface), reference.poc, flags);
    }

    blob.set(
        p::PIC_WIDTH_IN_LUMA_SAMPLES,
        u16::try_from(sps.pic_width_in_luma_samples).unwrap_or(u16::MAX),
    );
    blob.set(
        p::PIC_HEIGHT_IN_LUMA_SAMPLES,
        u16::try_from(sps.pic_height_in_luma_samples).unwrap_or(u16::MAX),
    );
    let pcm = sps.pcm.as_ref();
    let bits = [
        (p::CHROMA_FORMAT_IDC, u32::from(sps.chroma_format_idc)),
        (
            p::SEPARATE_COLOUR_PLANE_FLAG,
            flag(sps.separate_colour_plane_flag),
        ),
        (p::PCM_ENABLED_FLAG, flag(sps.pcm_enabled_flag)),
        (
            p::SCALING_LIST_ENABLED_FLAG,
            flag(sps.scaling_list_enabled_flag),
        ),
        (
            p::TRANSFORM_SKIP_ENABLED_FLAG,
            flag(pps.transform_skip_enabled_flag),
        ),
        (p::AMP_ENABLED_FLAG, flag(sps.amp_enabled_flag)),
        (
            p::STRONG_INTRA_SMOOTHING_ENABLED_FLAG,
            flag(sps.strong_intra_smoothing_enabled_flag),
        ),
        (
            p::SIGN_DATA_HIDING_ENABLED_FLAG,
            flag(pps.sign_data_hiding_enabled_flag),
        ),
        (
            p::CONSTRAINED_INTRA_PRED_FLAG,
            flag(pps.constrained_intra_pred_flag),
        ),
        (
            p::CU_QP_DELTA_ENABLED_FLAG,
            flag(pps.cu_qp_delta_enabled_flag),
        ),
        (p::WEIGHTED_PRED_FLAG, flag(pps.weighted_pred_flag)),
        (p::WEIGHTED_BIPRED_FLAG, flag(pps.weighted_bipred_flag)),
        (
            p::TRANSQUANT_BYPASS_ENABLED_FLAG,
            flag(pps.transquant_bypass_enabled_flag),
        ),
        (p::TILES_ENABLED_FLAG, flag(pps.tiles_enabled_flag)),
        (
            p::ENTROPY_CODING_SYNC_ENABLED_FLAG,
            flag(pps.entropy_coding_sync_enabled_flag),
        ),
        (
            p::PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
            flag(pps.pps_loop_filter_across_slices_enabled_flag),
        ),
        (
            p::LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG,
            flag(pps.loop_filter_across_tiles_enabled_flag),
        ),
        (
            p::PCM_LOOP_FILTER_DISABLED_FLAG,
            flag(pcm.is_some_and(|pcm| pcm.pcm_loop_filter_disabled_flag)),
        ),
    ];
    for (field, value) in bits {
        blob.set_bits(field, value);
    }

    blob.set(
        p::SPS_MAX_DEC_PIC_BUFFERING_MINUS1,
        u8::try_from(sps.max_dec_pic_buffering().saturating_sub(1)).unwrap_or(u8::MAX),
    );
    blob.set(p::BIT_DEPTH_LUMA_MINUS8, sps.bit_depth_luma_minus8);
    blob.set(p::BIT_DEPTH_CHROMA_MINUS8, sps.bit_depth_chroma_minus8);
    if let Some(pcm) = pcm {
        blob.set(
            p::PCM_SAMPLE_BIT_DEPTH_LUMA_MINUS1,
            pcm.pcm_sample_bit_depth_luma_minus1,
        );
        blob.set(
            p::PCM_SAMPLE_BIT_DEPTH_CHROMA_MINUS1,
            pcm.pcm_sample_bit_depth_chroma_minus1,
        );
        blob.set(
            p::LOG2_MIN_PCM_LUMA_CODING_BLOCK_SIZE_MINUS3,
            pcm.log2_min_pcm_luma_coding_block_size_minus3,
        );
        blob.set(
            p::LOG2_DIFF_MAX_MIN_PCM_LUMA_CODING_BLOCK_SIZE,
            pcm.log2_diff_max_min_pcm_luma_coding_block_size,
        );
    }
    blob.set(
        p::LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3,
        sps.log2_min_luma_coding_block_size_minus3,
    );
    blob.set(
        p::LOG2_DIFF_MAX_MIN_LUMA_CODING_BLOCK_SIZE,
        sps.log2_diff_max_min_luma_coding_block_size,
    );
    blob.set(
        p::LOG2_MIN_TRANSFORM_BLOCK_SIZE_MINUS2,
        sps.log2_min_luma_transform_block_size_minus2,
    );
    blob.set(
        p::LOG2_DIFF_MAX_MIN_TRANSFORM_BLOCK_SIZE,
        sps.log2_diff_max_min_luma_transform_block_size,
    );
    blob.set(
        p::MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA,
        sps.max_transform_hierarchy_depth_intra,
    );
    blob.set(
        p::MAX_TRANSFORM_HIERARCHY_DEPTH_INTER,
        sps.max_transform_hierarchy_depth_inter,
    );
    blob.set(p::INIT_QP_MINUS26, byte(pps.init_qp_minus26));
    blob.set(p::DIFF_CU_QP_DELTA_DEPTH, pps.diff_cu_qp_delta_depth);
    blob.set(p::PPS_CB_QP_OFFSET, byte(pps.pps_cb_qp_offset));
    blob.set(p::PPS_CR_QP_OFFSET, byte(pps.pps_cr_qp_offset));
    blob.set(
        p::LOG2_PARALLEL_MERGE_LEVEL_MINUS2,
        pps.log2_parallel_merge_level_minus2,
    );
    if pps.tiles_enabled_flag {
        blob.set(
            p::NUM_TILE_COLUMNS_MINUS1,
            small(pps.num_tile_columns_minus1),
        );
        blob.set(p::NUM_TILE_ROWS_MINUS1, small(pps.num_tile_rows_minus1));
        for (index, width) in picture.column_widths.iter().enumerate() {
            let value = u16::try_from(width.saturating_sub(1)).unwrap_or(u16::MAX);
            blob.set_at(p::COLUMN_WIDTH_MINUS1, index, value);
        }
        for (index, height) in picture.row_heights.iter().enumerate() {
            let value = u16::try_from(height.saturating_sub(1)).unwrap_or(u16::MAX);
            blob.set_at(p::ROW_HEIGHT_MINUS1, index, value);
        }
    }

    let irap = (16..=23).contains(&picture.nal_unit_type);
    let idr = matches!(picture.nal_unit_type, 19 | 20);
    let slice_bits = [
        (
            p::LISTS_MODIFICATION_PRESENT_FLAG,
            flag(pps.lists_modification_present_flag),
        ),
        (
            p::LONG_TERM_REF_PICS_PRESENT_FLAG,
            flag(sps.long_term_ref_pics_present_flag),
        ),
        (
            p::SPS_TEMPORAL_MVP_ENABLED_FLAG,
            flag(sps.sps_temporal_mvp_enabled_flag),
        ),
        (
            p::CABAC_INIT_PRESENT_FLAG,
            flag(pps.cabac_init_present_flag),
        ),
        (
            p::OUTPUT_FLAG_PRESENT_FLAG,
            flag(pps.output_flag_present_flag),
        ),
        (
            p::DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG,
            flag(pps.dependent_slice_segments_enabled_flag),
        ),
        (
            p::PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG,
            flag(pps.pps_slice_chroma_qp_offsets_present_flag),
        ),
        (
            p::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG,
            flag(sps.sample_adaptive_offset_enabled_flag),
        ),
        (
            p::DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG,
            flag(pps.deblocking_filter_override_enabled_flag),
        ),
        (
            p::PPS_DISABLE_DEBLOCKING_FILTER_FLAG,
            flag(pps.pps_deblocking_filter_disabled_flag),
        ),
        (
            p::SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG,
            flag(pps.slice_segment_header_extension_present_flag),
        ),
        (p::RAP_PIC_FLAG, flag(irap)),
        (p::IDR_PIC_FLAG, flag(idr)),
        // FFmpeg's value: drivers read it as "intra random access".
        (p::INTRA_PIC_FLAG, flag(irap)),
    ];
    for (field, value) in slice_bits {
        blob.set_bits(field, value);
    }
    blob.set(
        p::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4,
        sps.log2_max_pic_order_cnt_lsb_minus4,
    );
    blob.set(
        p::NUM_SHORT_TERM_REF_PIC_SETS,
        sps.num_short_term_ref_pic_sets,
    );
    blob.set(p::NUM_LONG_TERM_REF_PIC_SPS, sps.num_long_term_ref_pics_sps);
    blob.set(
        p::NUM_REF_IDX_L0_DEFAULT_ACTIVE_MINUS1,
        pps.num_ref_idx_l0_default_active_minus1,
    );
    blob.set(
        p::NUM_REF_IDX_L1_DEFAULT_ACTIVE_MINUS1,
        pps.num_ref_idx_l1_default_active_minus1,
    );
    blob.set(p::PPS_BETA_OFFSET_DIV2, byte(pps.pps_beta_offset_div2));
    blob.set(p::PPS_TC_OFFSET_DIV2, byte(pps.pps_tc_offset_div2));
    blob.set(
        p::NUM_EXTRA_SLICE_HEADER_BITS,
        pps.num_extra_slice_header_bits,
    );
    let st_rps_bits = picture.slices.first().map_or(0, |slice| slice.st_rps_bits);
    blob.set(p::ST_RPS_BITS, st_rps_bits);
    blob
}

/// `VAIQMatrixBufferHEVC`: the lists in coded order, the 32×32 ones of
/// matrixId 0 and 3.
fn iq_matrix(list: &ScalingList) -> Blob {
    use hevc_iq as q;
    let mut blob = Blob::new(q::SIZE);
    for (matrix, values) in list.scaling_list_4x4().iter().enumerate() {
        for (index, value) in values.iter().enumerate() {
            blob.set_at(q::SCALING_LIST_4X4, matrix * 16 + index, *value);
        }
    }
    let wide = [
        (q::SCALING_LIST_8X8, list.scaling_list_8x8().to_vec()),
        (q::SCALING_LIST_16X16, list.scaling_list_16x16().to_vec()),
        (q::SCALING_LIST_32X32, list.scaling_list_32x32().to_vec()),
    ];
    for (array, lists) in wide {
        for (matrix, values) in lists.iter().enumerate() {
            for (index, value) in values.iter().enumerate() {
                blob.set_at(array, matrix * 64 + index, *value);
            }
        }
    }
    for (index, value) in list.scaling_list_dc_16x16().iter().enumerate() {
        blob.set_at(q::SCALING_LIST_DC_16X16, index, *value);
    }
    for (index, value) in list.scaling_list_dc_32x32().iter().enumerate() {
        blob.set_at(q::SCALING_LIST_DC_32X32, index, *value);
    }
    blob
}

/// `VASliceParameterBufferHEVC` for one slice segment, whose data buffer
/// is its whole NAL unit.
fn slice_params(slice: &Slice<'_>, slots: &[Option<u8>], last: bool, chroma: bool) -> Blob {
    use hevc_slice as s;
    let header = slice.header;
    let mut blob = Blob::new(s::SIZE);
    blob.set(
        s::SLICE_DATA_SIZE,
        u32::try_from(slice.nal.len()).unwrap_or(u32::MAX),
    );
    blob.set(s::SLICE_DATA_OFFSET, 0);
    blob.set(s::SLICE_DATA_FLAG, abi::VA_SLICE_DATA_FLAG_ALL);
    blob.set(
        s::SLICE_DATA_BYTE_OFFSET,
        u32::try_from(slice.data_offset).unwrap_or(u32::MAX),
    );
    blob.set(s::SLICE_SEGMENT_ADDRESS, header.slice_segment_address);
    for index in 0..s::REF_PIC_LIST.len {
        blob.set_at(s::REF_PIC_LIST, index, NO_REF);
    }
    for (list, entries) in slice.ref_lists.iter().enumerate() {
        for (index, reference) in entries.iter().take(MAX_REFS).enumerate() {
            let slot = slots.get(*reference).copied().flatten().unwrap_or(NO_REF);
            blob.set_at(s::REF_PIC_LIST, list * MAX_REFS + index, slot);
        }
    }
    let intra = header.slice_type == SliceType::I;
    let bits = [
        (s::LAST_SLICE_OF_PIC, flag(last)),
        (
            s::DEPENDENT_SLICE_SEGMENT_FLAG,
            flag(header.dependent_slice_segment_flag),
        ),
        (s::SLICE_TYPE, u32::from(header.slice_type as u8)),
        (s::COLOR_PLANE_ID, 0),
        (s::SLICE_SAO_LUMA_FLAG, flag(header.slice_sao_luma_flag)),
        (s::SLICE_SAO_CHROMA_FLAG, flag(header.slice_sao_chroma_flag)),
        (s::MVD_L1_ZERO_FLAG, flag(header.mvd_l1_zero_flag)),
        (s::CABAC_INIT_FLAG, flag(header.cabac_init_flag)),
        (
            s::SLICE_TEMPORAL_MVP_ENABLED_FLAG,
            flag(header.slice_temporal_mvp_enabled_flag),
        ),
        (
            s::SLICE_DEBLOCKING_FILTER_DISABLED_FLAG,
            flag(header.slice_deblocking_filter_disabled_flag),
        ),
        (
            s::COLLOCATED_FROM_L0_FLAG,
            flag(header.collocated_from_l0_flag),
        ),
        (
            s::SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG,
            flag(header.slice_loop_filter_across_slices_enabled_flag),
        ),
    ];
    for (field, value) in bits {
        blob.set_bits(field, value);
    }
    blob.set(
        s::COLLOCATED_REF_IDX,
        if header.slice_temporal_mvp_enabled_flag {
            header.collocated_ref_idx
        } else {
            NO_REF
        },
    );
    let active = |list: usize| {
        let len = slice.ref_lists[list].len().min(MAX_REFS);
        u8::try_from(len.saturating_sub(1)).unwrap_or(0)
    };
    blob.set(s::NUM_REF_IDX_L0_ACTIVE_MINUS1, active(0));
    blob.set(s::NUM_REF_IDX_L1_ACTIVE_MINUS1, active(1));
    blob.set(s::SLICE_QP_DELTA, byte(header.slice_qp_delta));
    blob.set(s::SLICE_CB_QP_OFFSET, byte(header.slice_cb_qp_offset));
    blob.set(s::SLICE_CR_QP_OFFSET, byte(header.slice_cr_qp_offset));
    blob.set(
        s::SLICE_BETA_OFFSET_DIV2,
        byte(header.slice_beta_offset_div2),
    );
    blob.set(s::SLICE_TC_OFFSET_DIV2, byte(header.slice_tc_offset_div2));
    if let Some(table) = &header.pred_weight_table {
        weights(&mut blob, table, chroma);
    }
    blob.set(
        s::FIVE_MINUS_MAX_NUM_MERGE_CAND,
        if intra {
            0
        } else {
            header.five_minus_max_num_merge_cand
        },
    );
    let escaped = slice
        .data_offset
        .saturating_sub(slice.header_bits as usize / 8);
    blob.set(
        s::SLICE_DATA_NUM_EMU_PREVN_BYTES,
        u16::try_from(escaped).unwrap_or(u16::MAX),
    );
    blob
}

/// The explicit weights of `pred_weight_table()`, as deltas and the
/// derived chroma offsets.
fn weights(blob: &mut Blob, table: &PredWeightTable, chroma: bool) {
    use hevc_slice as s;
    blob.set(s::LUMA_LOG2_WEIGHT_DENOM, table.luma_log2_weight_denom);
    if chroma {
        blob.set(
            s::DELTA_CHROMA_LOG2_WEIGHT_DENOM,
            byte(table.delta_chroma_log2_weight_denom),
        );
    }
    let lists = [
        (
            s::DELTA_LUMA_WEIGHT_L0,
            s::LUMA_OFFSET_L0,
            s::DELTA_CHROMA_WEIGHT_L0,
            s::CHROMA_OFFSET_L0,
        ),
        (
            s::DELTA_LUMA_WEIGHT_L1,
            s::LUMA_OFFSET_L1,
            s::DELTA_CHROMA_WEIGHT_L1,
            s::CHROMA_OFFSET_L1,
        ),
    ];
    for (entries, (luma_weight, luma_offset, chroma_weight, chroma_offset)) in
        table.entries.iter().zip(lists)
    {
        for (index, entry) in entries.iter().take(MAX_REFS).enumerate() {
            blob.set_at(luma_weight, index, byte(entry.delta_luma_weight));
            blob.set_at(luma_offset, index, byte(entry.luma_offset));
            for plane in 0..2 {
                let at = index * 2 + plane;
                blob.set_at(chroma_weight, at, byte(entry.delta_chroma_weight[plane]));
                blob.set_at(chroma_offset, at, byte(entry.chroma_offset[plane]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_the_picture_uses_get_slots_first() {
        let surface = 0_u8;
        let reference = |set| Reference {
            surface: &surface,
            poc: 0,
            long_term: false,
            set,
        };
        let mut refs: Vec<Reference<'_, u8>> = (0..15).map(|_| reference(RpsSet::Foll)).collect();
        refs.push(reference(RpsSet::StCurrBefore));
        let slots = slots(&refs);
        assert_eq!(slots[15], Some(0));
        assert_eq!(slots[0], Some(1));
        assert_eq!(slots[13], Some(14));
        assert_eq!(slots[14], None);
    }

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

        fn missing(&mut self, _sps: &Sps, _poc: i32) -> Result<u32, CodecError> {
            self.next += 1;
            Ok(self.next)
        }

        fn decode(&mut self, picture: &Picture<'_, u32>) -> Result<u32, CodecError> {
            self.next += 1;
            let target = self.next;
            let slots = slots(picture.refs);
            let pic = picture_params(picture, target, &slots, |surface| *surface);
            let curr = hevc_pic::CURR_PIC;
            assert_eq!(pic.get(picture_hevc::PICTURE_ID.of(curr)), target);
            assert_eq!(pic.get(picture_hevc::PIC_ORDER_CNT.of(curr)), picture.poc);
            assert_eq!(
                u32::from(pic.get(hevc_pic::PIC_WIDTH_IN_LUMA_SAMPLES)),
                picture.sps.pic_width_in_luma_samples
            );
            if let Some(list) = ScalingList::active(picture.sps, picture.pps) {
                iq_matrix(list);
            }
            for (index, slice) in picture.slices.iter().enumerate() {
                let last = index + 1 == picture.slices.len();
                let blob = slice_params(slice, &slots, last, true);
                assert_eq!(
                    blob.get(hevc_slice::SLICE_DATA_SIZE) as usize,
                    slice.nal.len()
                );
                assert_eq!(
                    blob.get(hevc_slice::SLICE_DATA_BYTE_OFFSET) as usize,
                    slice.data_offset
                );
                assert!(slice.data_offset < slice.nal.len());
                assert_eq!(blob.bits(hevc_slice::LAST_SLICE_OF_PIC), u32::from(last));
                assert_eq!(
                    blob.bits(hevc_slice::SLICE_TYPE),
                    u32::from(slice.header.slice_type as u8)
                );
                for (list, entries) in slice.ref_lists.iter().enumerate() {
                    for (at, reference) in entries.iter().enumerate() {
                        let entry = hevc_slice::REF_PIC_LIST.at(list * MAX_REFS + at);
                        let slot = blob.get(entry.expect("in range"));
                        let frame = hevc_pic::REFERENCE_FRAMES
                            .at(usize::from(slot))
                            .expect("a slot");
                        assert_eq!(
                            pic.get(picture_hevc::PIC_ORDER_CNT.of(frame)),
                            picture.refs[*reference].poc
                        );
                        assert_eq!(
                            pic.get(picture_hevc::PICTURE_ID.of(frame)),
                            *picture.refs[*reference].surface
                        );
                    }
                }
                self.slices += 1;
            }
            self.pictures += 1;
            Ok(target)
        }
    }

    #[test]
    fn the_buffers_of_each_picture_name_it_and_its_references() {
        for (name, bytes, _) in crate::video::imp::tests::hevc_streams() {
            let packets = crate::video::imp::tests::hevc_access_units(&bytes);
            let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
            params.nal_length_size = 0;
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
            assert!(recorder.pictures >= 1, "{name}");
            assert!(recorder.slices >= recorder.pictures, "{name}");
            assert_eq!(out, recorder.pictures, "{name}");
            eprintln!(
                "{name}: {} pictures, {} slices",
                recorder.pictures, recorder.slices
            );
        }
    }

    #[test]
    fn the_matrix_takes_every_list_in_coded_order() {
        let mut list = ScalingList::default_lists();
        list.dc[1][3] = 42;
        let blob = iq_matrix(&list);
        let coded = list.scaling_list_32x32();
        let second = hevc_iq::SCALING_LIST_32X32.at(64 + 63).expect("in range");
        assert_eq!(blob.get(second), coded[1][63]);
        let dc = hevc_iq::SCALING_LIST_DC_32X32.at(1).expect("in range");
        assert_eq!(blob.get(dc), 42);
        let flat = hevc_iq::SCALING_LIST_4X4.at(5 * 16 + 15).expect("in range");
        assert_eq!(blob.get(flat), 16);
    }
}
