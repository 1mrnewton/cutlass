//! HEVC Main encoding on VA-API, with the parameter buffers FFmpeg's
//! `vaapi_encode_h265.c` fills for a stream without B pictures.
//!
//! The block sizes and coding tools follow what the driver reports in
//! `VAConfigAttribEncHEVCBlockSizes` and `VAConfigAttribEncHEVCFeatures`,
//! which low-power encoders need; a driver that reports neither gets
//! FFmpeg's defaults. A driver that takes no P slices
//! (`VA_PREDICTION_DIRECTION_BI_NOT_EMPTY`) gets B slices with the same
//! reference in both lists.

use cutlass_decode::{CodecError, CodecId, CodecParams};
use cutlass_encode::Encoder;

use super::abi::{
    self, hevc_block_sizes, hevc_enc_pic, hevc_enc_seq, hevc_enc_slice, hevc_features, picture_hevc,
};
use super::bits::Packed;
use super::encode::{self, Engine, Pictures, RateControl, Settings, Syntax, VaEncoder, field};
use super::header::{Blocks, HEVC_B, HEVC_I, Hevc, Step, Tools, rate16};
use crate::video::imp::blob::{Array, Base, Blob, Field};
use crate::video::kind::Kind;
use crate::video::rate;

/// The QP of a stream without rate control, and where rate control
/// starts.
const INIT_QP: u8 = 26;

/// What surfaces are aligned to when the driver names no block sizes, as
/// FFmpeg does.
const DEFAULT_ALIGN: u32 = 16;

/// Open a VA-API HEVC encoder, or decline so the next opener is asked.
pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    let settings = Settings::of(params, CodecId::Hevc)?;
    let choice = encode::choose(&settings, &[abi::VAProfileHEVCMain])?;
    let (blocks, align) = blocks(choice.block_sizes);
    let tools = tools(choice.features, choice.rc);
    let syntax = HevcSyntax::new(&settings, blocks, align, tools, choice.gpb);
    let size = (syntax.headers.width, syntax.headers.height);
    let engine = Engine::open(choice, syntax, size, settings.width, settings.height)?;
    Ok(VaEncoder::boxed(engine, Kind::Hevc))
}

/// Whether a render node encodes HEVC here.
#[cfg(test)]
pub fn available() -> bool {
    encode::offered(&[abi::VAProfileHEVCMain])
}

fn byte(value: u32) -> u8 {
    u8::try_from(value).unwrap_or(u8::MAX)
}

/// The block sizes the driver reports, and the alignment of the surfaces
/// that follows: the minimum coding block. FFmpeg's defaults when the
/// driver reports none or sizes HEVC does not allow.
pub fn blocks(value: u32) -> (Blocks, u32) {
    if value == abi::VA_ATTRIB_NOT_SUPPORTED {
        return (Blocks::DEFAULT, DEFAULT_ALIGN);
    }
    use hevc_block_sizes as b;
    let get = |bits| field(value, bits);
    let ctb = get(b::LOG2_MAX_CODING_TREE_BLOCK_SIZE_MINUS3) + 3;
    let min_cb = get(b::LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3) + 3;
    let max_tb = get(b::LOG2_MAX_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2) + 2;
    let min_tb = get(b::LOG2_MIN_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2) + 2;
    let allowed = (4..=6).contains(&ctb)
        && min_cb <= ctb
        && min_tb < min_cb
        && min_tb <= max_tb
        && max_tb <= ctb.min(5);
    if !allowed {
        return (Blocks::DEFAULT, DEFAULT_ALIGN);
    }
    let deepest = ctb - min_tb;
    let blocks = Blocks {
        min_cb,
        ctb,
        min_tb,
        max_tb,
        depth_inter: get(b::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER).min(deepest),
        depth_intra: get(b::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA).min(deepest),
    };
    (blocks, 1 << min_cb)
}

/// The coding tools the driver offers or needs, each on when it is either,
/// as FFmpeg sets them. CU QP deltas are on when rate control needs them
/// and the driver allows them. FFmpeg's defaults when the driver reports
/// none: asymmetric motion partitions only.
pub fn tools(value: u32, rc: RateControl) -> Tools {
    let rate_controlled = rc.mode != abi::VA_RC_CQP;
    if value == abi::VA_ATTRIB_NOT_SUPPORTED {
        return Tools {
            amp: true,
            cu_qp_delta: rate_controlled,
            ..Tools::default()
        };
    }
    use hevc_features as f;
    let on = |bits| field(value, bits) != 0;
    Tools {
        amp: on(f::AMP),
        sao: on(f::SAO),
        temporal_mvp: on(f::TEMPORAL_MVP),
        strong_intra_smoothing: on(f::STRONG_INTRA_SMOOTHING),
        sign_data_hiding: on(f::SIGN_DATA_HIDING),
        transform_skip: on(f::TRANSFORM_SKIP),
        cu_qp_delta: match field(value, f::CU_QP_DELTA) {
            0 => false,
            1 => rate_controlled,
            _ => true,
        },
    }
}

/// The headers and parameter buffers of an HEVC stream.
#[derive(Clone, Debug)]
pub struct HevcSyntax {
    pub headers: Hevc,
    bitrate: u32,
}

impl HevcSyntax {
    pub fn new(settings: &Settings, blocks: Blocks, align: u32, tools: Tools, gpb: bool) -> Self {
        let width = settings.width.next_multiple_of(align);
        let height = settings.height.next_multiple_of(align);
        let rate = rate16(rate::DEFAULT_RATE);
        Self {
            headers: Hevc {
                level_idc: Hevc::level(width, height, rate, settings.bitrate),
                width,
                height,
                crop: (width - settings.width, height - settings.height),
                blocks,
                tools,
                gpb,
                init_qp: INIT_QP,
                colour: settings.colour,
                rate,
            },
            bitrate: settings.bitrate,
        }
    }

    /// Whether the slice of `step` predicts motion from its reference.
    fn temporal_mvp(&self, step: Step) -> bool {
        self.headers.tools.temporal_mvp && !step.idr
    }
}

fn invalid(blob: &mut Blob, at: Base) {
    blob.set(picture_hevc::PICTURE_ID.of(at), abi::VA_INVALID_ID);
    blob.set(picture_hevc::FLAGS.of(at), abi::VA_PICTURE_HEVC_INVALID);
}

fn picture(blob: &mut Blob, at: Base, surface: u32, poc: i32, flags: u32) {
    blob.set(picture_hevc::PICTURE_ID.of(at), surface);
    blob.set(picture_hevc::PIC_ORDER_CNT.of(at), poc);
    blob.set(picture_hevc::FLAGS.of(at), flags);
}

/// Every entry of `list` invalid, then the reference first.
fn list(blob: &mut Blob, list: Array<Base>, reference: Option<(u32, i32)>) {
    for index in 0..list.len {
        if let Some(at) = list.at(index) {
            invalid(blob, at);
        }
    }
    if let (Some((surface, poc)), Some(at)) = (reference, list.at(0)) {
        picture(
            blob,
            at,
            surface,
            poc,
            abi::VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE,
        );
    }
}

/// The reference of `step` with its order count: the picture before.
fn reference(step: Step, pictures: Pictures) -> Option<(u32, i32)> {
    pictures
        .reference
        .filter(|_| !step.idr)
        .map(|surface| (surface, Hevc::poc(step) - 1))
}

impl Syntax for HevcSyntax {
    fn sets(&self) -> Vec<Vec<u8>> {
        vec![self.headers.vps(), self.headers.sps(), self.headers.pps()]
    }

    fn set_rate(&mut self, rate: (u32, u32)) {
        let h = &mut self.headers;
        h.rate = rate;
        h.level_idc = Hevc::level(h.width, h.height, rate, self.bitrate);
    }

    fn sequence(&self, rc: RateControl, period: u32) -> Blob {
        use hevc_enc_seq as q;
        let h = &self.headers;
        let (blocks, tools) = (h.blocks, h.tools);
        let mut blob = Blob::new(q::SIZE);
        blob.set(q::GENERAL_PROFILE_IDC, 1);
        blob.set(q::GENERAL_LEVEL_IDC, h.level_idc);
        blob.set(q::GENERAL_TIER_FLAG, 0);
        blob.set(q::INTRA_PERIOD, period);
        blob.set(q::INTRA_IDR_PERIOD, period);
        blob.set(q::IP_PERIOD, 1);
        blob.set(q::BITS_PER_SECOND, rc.sequence_bits());
        blob.set(
            q::PIC_WIDTH_IN_LUMA_SAMPLES,
            u16::try_from(h.width).unwrap_or(u16::MAX),
        );
        blob.set(
            q::PIC_HEIGHT_IN_LUMA_SAMPLES,
            u16::try_from(h.height).unwrap_or(u16::MAX),
        );
        blob.set_bits(q::CHROMA_FORMAT_IDC, 1);
        blob.set_bits(q::BIT_DEPTH_LUMA_MINUS8, 0);
        blob.set_bits(q::BIT_DEPTH_CHROMA_MINUS8, 0);
        blob.set_bits(q::SCALING_LIST_ENABLED_FLAG, 0);
        blob.set_bits(
            q::STRONG_INTRA_SMOOTHING_ENABLED_FLAG,
            u32::from(tools.strong_intra_smoothing),
        );
        blob.set_bits(q::AMP_ENABLED_FLAG, u32::from(tools.amp));
        blob.set_bits(q::SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG, u32::from(tools.sao));
        blob.set_bits(q::PCM_ENABLED_FLAG, 0);
        blob.set_bits(
            q::SPS_TEMPORAL_MVP_ENABLED_FLAG,
            u32::from(tools.temporal_mvp),
        );
        // No B pictures: every picture is low delay.
        blob.set_bits(q::LOW_DELAY_SEQ, 1);
        blob.set(
            q::LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3,
            byte(blocks.min_cb - 3),
        );
        blob.set(
            q::LOG2_DIFF_MAX_MIN_LUMA_CODING_BLOCK_SIZE,
            byte(blocks.ctb - blocks.min_cb),
        );
        blob.set(
            q::LOG2_MIN_TRANSFORM_BLOCK_SIZE_MINUS2,
            byte(blocks.min_tb - 2),
        );
        blob.set(
            q::LOG2_DIFF_MAX_MIN_TRANSFORM_BLOCK_SIZE,
            byte(blocks.max_tb - blocks.min_tb),
        );
        blob.set(
            q::MAX_TRANSFORM_HIERARCHY_DEPTH_INTER,
            byte(blocks.depth_inter),
        );
        blob.set(
            q::MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA,
            byte(blocks.depth_intra),
        );
        // The VUI goes in the packed sequence header, as FFmpeg sends it.
        blob.set(q::VUI_PARAMETERS_PRESENT_FLAG, 0);
        blob
    }

    fn picture(&self, step: Step, pictures: Pictures) -> Blob {
        use hevc_enc_pic as p;
        let h = &self.headers;
        let tools = h.tools;
        let mut blob = Blob::new(p::SIZE);
        picture(
            &mut blob,
            p::DECODED_CURR_PIC,
            pictures.current,
            Hevc::poc(step),
            0,
        );
        list(&mut blob, p::REFERENCE_FRAMES, reference(step, pictures));
        blob.set(p::CODED_BUF, pictures.coded);
        let collocated = if self.temporal_mvp(step) { 0 } else { 0xff };
        blob.set(p::COLLOCATED_REF_PIC_INDEX, collocated);
        blob.set(p::LAST_PICTURE, 0);
        blob.set(p::PIC_INIT_QP, h.init_qp);
        blob.set(p::DIFF_CU_QP_DELTA_DEPTH, 0);
        blob.set(p::LOG2_PARALLEL_MERGE_LEVEL_MINUS2, 0);
        blob.set(p::NUM_REF_IDX_L0_DEFAULT_ACTIVE_MINUS1, 0);
        blob.set(p::NUM_REF_IDX_L1_DEFAULT_ACTIVE_MINUS1, 0);
        blob.set(p::SLICE_PIC_PARAMETER_SET_ID, 0);
        blob.set(p::NAL_UNIT_TYPE, byte(Hevc::nal_unit_type(step)));
        blob.set_bits(p::IDR_PIC_FLAG, u32::from(step.idr));
        // 1 is an I picture, 2 a P picture.
        blob.set_bits(p::CODING_TYPE, if step.idr { 1 } else { 2 });
        blob.set_bits(p::REFERENCE_PIC_FLAG, 1);
        blob.set_bits(
            p::SIGN_DATA_HIDING_ENABLED_FLAG,
            u32::from(tools.sign_data_hiding),
        );
        blob.set_bits(
            p::TRANSFORM_SKIP_ENABLED_FLAG,
            u32::from(tools.transform_skip),
        );
        blob.set_bits(p::CU_QP_DELTA_ENABLED_FLAG, u32::from(tools.cu_qp_delta));
        blob
    }

    fn slice(&self, step: Step, pictures: Pictures) -> Blob {
        use hevc_enc_slice as l;
        let h = &self.headers;
        let kind = h.slice_type(step);
        let mut blob = Blob::new(l::SIZE);
        blob.set(l::SLICE_SEGMENT_ADDRESS, 0);
        blob.set(l::NUM_CTU_IN_SLICE, h.ctus());
        blob.set(l::SLICE_TYPE, kind);
        blob.set(l::SLICE_PIC_PARAMETER_SET_ID, 0);
        blob.set(l::NUM_REF_IDX_L0_ACTIVE_MINUS1, 0);
        blob.set(l::NUM_REF_IDX_L1_ACTIVE_MINUS1, 0);
        let reference = reference(step, pictures).filter(|_| kind != HEVC_I);
        list(&mut blob, l::REF_PIC_LIST0, reference);
        list(
            &mut blob,
            l::REF_PIC_LIST1,
            reference.filter(|_| kind == HEVC_B),
        );
        blob.set(l::MAX_NUM_MERGE_CAND, 5);
        blob.set(l::SLICE_QP_DELTA, 0);
        let mvp = u32::from(self.temporal_mvp(step));
        blob.set_bits(l::LAST_SLICE_OF_PIC_FLAG, 1);
        blob.set_bits(l::SLICE_TEMPORAL_MVP_ENABLED_FLAG, mvp);
        blob.set_bits(l::SLICE_SAO_LUMA_FLAG, u32::from(h.tools.sao));
        blob.set_bits(l::SLICE_SAO_CHROMA_FLAG, u32::from(h.tools.sao));
        blob.set_bits(l::COLLOCATED_FROM_L0_FLAG, mvp);
        blob
    }

    fn slice_header(&self, step: Step) -> Packed {
        Packed::partial(self.headers.slice_header(step))
    }
}
