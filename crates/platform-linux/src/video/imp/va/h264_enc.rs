//! H.264 encoding on VA-API: High, else Main, else Constrained Baseline,
//! whichever the GPU offers first, with the parameter buffers FFmpeg's
//! `vaapi_encode_h264.c` fills for a stream without B pictures.

use cutlass_decode::{CodecError, CodecId, CodecParams};
use cutlass_encode::Encoder;

use super::abi::{self, h264_enc_pic, h264_enc_seq, h264_enc_slice, picture_h264};
use super::bits::Packed;
use super::encode::{self, Engine, Pictures, RateControl, Settings, Syntax, VaEncoder};
use super::header::{H264, H264_LOG2_MAX_FRAME_NUM, H264Profile, Step, rate16};
use crate::video::imp::blob::{Array, Base, Blob, Field};
use crate::video::kind::Kind;
use crate::video::rate;

/// The QP of a stream without rate control, and where rate control
/// starts.
const INIT_QP: u8 = 26;

/// The VA profiles in the order they are tried.
const PROFILES: [(i32, H264Profile); 3] = [
    (abi::VAProfileH264High, H264Profile::High),
    (abi::VAProfileH264Main, H264Profile::Main),
    (
        abi::VAProfileH264ConstrainedBaseline,
        H264Profile::ConstrainedBaseline,
    ),
];

/// Open a VA-API H.264 encoder, or decline so the next opener is asked.
pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    let settings = Settings::of(params, CodecId::H264)?;
    let profiles = PROFILES.map(|(profile, _)| profile);
    let choice = encode::choose(&settings, &profiles)?;
    let profile = PROFILES
        .iter()
        .find(|(va, _)| *va == choice.profile)
        .map_or(H264Profile::High, |(_, profile)| *profile);
    let syntax = H264Syntax::new(&settings, profile);
    let size = (
        syntax.headers.width_mbs * 16,
        syntax.headers.height_mbs * 16,
    );
    let engine = Engine::open(choice, syntax, size, settings.width, settings.height)?;
    Ok(VaEncoder::boxed(engine, Kind::H264))
}

/// Whether a render node encodes H.264 here.
#[cfg(test)]
pub fn available() -> bool {
    encode::offered(&PROFILES.map(|(profile, _)| profile))
}

/// The headers and parameter buffers of an H.264 stream.
#[derive(Clone, Debug)]
pub struct H264Syntax {
    pub headers: H264,
    bitrate: u32,
}

impl H264Syntax {
    pub fn new(settings: &Settings, profile: H264Profile) -> Self {
        let width_mbs = settings.width.div_ceil(16);
        let height_mbs = settings.height.div_ceil(16);
        let rate = rate16(rate::DEFAULT_RATE);
        Self {
            headers: H264 {
                profile,
                level_idc: H264::level(width_mbs, height_mbs, rate, settings.bitrate, profile),
                width_mbs,
                height_mbs,
                crop: (
                    width_mbs * 16 - settings.width,
                    height_mbs * 16 - settings.height,
                ),
                init_qp: INIT_QP,
                colour: settings.colour,
                rate,
            },
            bitrate: settings.bitrate,
        }
    }
}

/// The picture before `step`, which a P picture references.
fn before(step: Step) -> Step {
    Step {
        idr: step.index == 1,
        index: step.index.saturating_sub(1),
        idr_id: step.idr_id,
    }
}

fn invalid(blob: &mut Blob, at: Base) {
    blob.set(picture_h264::PICTURE_ID.of(at), abi::VA_INVALID_ID);
    blob.set(picture_h264::FLAGS.of(at), abi::VA_PICTURE_H264_INVALID);
}

fn picture(blob: &mut Blob, at: Base, surface: u32, step: Step, flags: u32) {
    blob.set(picture_h264::PICTURE_ID.of(at), surface);
    blob.set(picture_h264::FRAME_IDX.of(at), H264::frame_num(step));
    blob.set(picture_h264::FLAGS.of(at), flags);
    blob.set(picture_h264::TOP_FIELD_ORDER_CNT.of(at), H264::poc(step));
    blob.set(picture_h264::BOTTOM_FIELD_ORDER_CNT.of(at), H264::poc(step));
}

/// Every entry of `list` invalid, then the reference first.
fn list(blob: &mut Blob, list: Array<Base>, reference: Option<(u32, Step)>) {
    for index in 0..list.len {
        if let Some(at) = list.at(index) {
            invalid(blob, at);
        }
    }
    if let (Some((surface, step)), Some(at)) = (reference, list.at(0)) {
        picture(
            blob,
            at,
            surface,
            step,
            abi::VA_PICTURE_H264_SHORT_TERM_REFERENCE,
        );
    }
}

impl Syntax for H264Syntax {
    fn sets(&self) -> Vec<Vec<u8>> {
        vec![self.headers.sps(), self.headers.pps()]
    }

    fn set_rate(&mut self, rate: (u32, u32)) {
        let h = &mut self.headers;
        h.rate = rate;
        h.level_idc = H264::level(h.width_mbs, h.height_mbs, rate, self.bitrate, h.profile);
    }

    fn sequence(&self, rc: RateControl, period: u32) -> Blob {
        use h264_enc_seq as q;
        let h = &self.headers;
        let mut blob = Blob::new(q::SIZE);
        blob.set(q::SEQ_PARAMETER_SET_ID, 0);
        blob.set(q::LEVEL_IDC, h.level_idc);
        blob.set(q::INTRA_PERIOD, period);
        blob.set(q::INTRA_IDR_PERIOD, period);
        blob.set(q::IP_PERIOD, 1);
        blob.set(q::BITS_PER_SECOND, rc.sequence_bits());
        blob.set(q::MAX_NUM_REF_FRAMES, 1);
        blob.set(
            q::PICTURE_WIDTH_IN_MBS,
            u16::try_from(h.width_mbs).unwrap_or(u16::MAX),
        );
        blob.set(
            q::PICTURE_HEIGHT_IN_MBS,
            u16::try_from(h.height_mbs).unwrap_or(u16::MAX),
        );
        blob.set_bits(q::CHROMA_FORMAT_IDC, 1);
        blob.set_bits(q::FRAME_MBS_ONLY_FLAG, 1);
        blob.set_bits(q::DIRECT_8X8_INFERENCE_FLAG, 1);
        blob.set_bits(q::LOG2_MAX_FRAME_NUM_MINUS4, H264_LOG2_MAX_FRAME_NUM - 4);
        blob.set_bits(q::PIC_ORDER_CNT_TYPE, 2);
        blob.set(q::BIT_DEPTH_LUMA_MINUS8, 0);
        blob.set(q::BIT_DEPTH_CHROMA_MINUS8, 0);
        if h.crop != (0, 0) {
            blob.set(q::FRAME_CROPPING_FLAG, 1);
            blob.set(q::FRAME_CROP_RIGHT_OFFSET, h.crop.0 / 2);
            blob.set(q::FRAME_CROP_BOTTOM_OFFSET, h.crop.1 / 2);
        }
        blob.set(q::VUI_PARAMETERS_PRESENT_FLAG, 1);
        blob.set_bits(q::TIMING_INFO_PRESENT_FLAG, 1);
        blob.set_bits(q::BITSTREAM_RESTRICTION_FLAG, 1);
        blob.set_bits(q::LOG2_MAX_MV_LENGTH_HORIZONTAL, 15);
        blob.set_bits(q::LOG2_MAX_MV_LENGTH_VERTICAL, 15);
        blob.set_bits(q::MOTION_VECTORS_OVER_PIC_BOUNDARIES_FLAG, 1);
        blob.set(q::NUM_UNITS_IN_TICK, h.rate.1);
        blob.set(q::TIME_SCALE, h.rate.0 * 2);
        blob
    }

    fn picture(&self, step: Step, pictures: Pictures) -> Blob {
        use h264_enc_pic as p;
        let h = &self.headers;
        let mut blob = Blob::new(p::SIZE);
        picture(&mut blob, p::CURR_PIC, pictures.current, step, 0);
        list(
            &mut blob,
            p::REFERENCE_FRAMES,
            pictures.reference.map(|surface| (surface, before(step))),
        );
        blob.set(p::CODED_BUF, pictures.coded);
        blob.set(p::PIC_PARAMETER_SET_ID, 0);
        blob.set(p::SEQ_PARAMETER_SET_ID, 0);
        blob.set(p::LAST_PICTURE, 0);
        blob.set(
            p::FRAME_NUM,
            u16::try_from(H264::frame_num(step)).unwrap_or(0),
        );
        blob.set(p::PIC_INIT_QP, h.init_qp);
        blob.set(p::NUM_REF_IDX_L0_ACTIVE_MINUS1, 0);
        blob.set(p::NUM_REF_IDX_L1_ACTIVE_MINUS1, 0);
        blob.set_bits(p::IDR_PIC_FLAG, u32::from(step.idr));
        blob.set_bits(p::REFERENCE_PIC_FLAG, 1);
        blob.set_bits(p::ENTROPY_CODING_MODE_FLAG, u32::from(h.profile.cabac()));
        blob.set_bits(
            p::TRANSFORM_8X8_MODE_FLAG,
            u32::from(h.profile.transform_8x8()),
        );
        blob.set_bits(p::DEBLOCKING_FILTER_CONTROL_PRESENT_FLAG, 1);
        blob
    }

    fn slice(&self, step: Step, pictures: Pictures) -> Blob {
        use h264_enc_slice as l;
        let h = &self.headers;
        let mut blob = Blob::new(l::SIZE);
        blob.set(l::MACROBLOCK_ADDRESS, 0);
        blob.set(l::NUM_MACROBLOCKS, h.width_mbs * h.height_mbs);
        blob.set(l::MACROBLOCK_INFO, abi::VA_INVALID_ID);
        blob.set(l::SLICE_TYPE, if step.idr { 2 } else { 0 });
        blob.set(l::PIC_PARAMETER_SET_ID, 0);
        blob.set(l::IDR_PIC_ID, (step.idr_id & 0xffff) as u16);
        blob.set(l::PIC_ORDER_CNT_LSB, 0);
        blob.set(l::NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG, 0);
        blob.set(l::NUM_REF_IDX_L0_ACTIVE_MINUS1, 0);
        blob.set(l::NUM_REF_IDX_L1_ACTIVE_MINUS1, 0);
        let reference = pictures
            .reference
            .filter(|_| !step.idr)
            .map(|surface| (surface, before(step)));
        list(&mut blob, l::REF_PIC_LIST0, reference);
        list(&mut blob, l::REF_PIC_LIST1, None);
        blob.set(l::CABAC_INIT_IDC, 0);
        blob.set(l::SLICE_QP_DELTA, 0);
        blob.set(l::DISABLE_DEBLOCKING_FILTER_IDC, 0);
        blob.set(l::SLICE_ALPHA_C0_OFFSET_DIV2, 0);
        blob.set(l::SLICE_BETA_OFFSET_DIV2, 0);
        blob
    }

    fn slice_header(&self, step: Step) -> Packed {
        Packed::partial(self.headers.slice_header(step))
    }
}
