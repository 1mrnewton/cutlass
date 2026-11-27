//! The VA-API encoders' headers, read back by the stateless front ends of
//! `cutlass-h264` and `cutlass-hevc`, and checked against the parameter
//! buffers sent with the same pictures. No GPU is needed: a byte stands in
//! for each slice's data, which the front ends hand on without reading.

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet};
use cutlass_h264::stateless as h264;
use cutlass_hevc::stateless as hevc;

use super::abi::{self, h264_enc_pic, h264_enc_slice, hevc_enc_pic, hevc_enc_slice};
use super::abi::{picture_h264, picture_hevc};
use super::bits::{Packed, Writer};
use super::encode::{Pictures, Settings, Syntax};
use super::h264_enc::H264Syntax;
use super::header::{Blocks, H264, H264Profile, Hevc, Step, Tools, rate16};
use super::hevc_enc::HevcSyntax;
use crate::video::imp::blob::Field;
use crate::video::rate::{KEYFRAME_INTERVAL, Vui};

const PICTURES: u32 = 35;
const CODED: u32 = 99;
const COLOUR: Vui = Vui {
    primaries: 9,
    transfer: 16,
    matrix: 9,
    full_range: false,
    described: true,
};

fn settings(codec: CodecId, width: u32, height: u32) -> Settings {
    Settings {
        codec,
        width,
        height,
        bitrate: 2_000_000,
        constant: false,
        colour: Some(COLOUR),
    }
}

/// The pictures the engine encodes: an IDR every 30.
fn steps() -> Vec<Step> {
    (0..PICTURES)
        .map(|n| Step {
            idr: n % KEYFRAME_INTERVAL == 0,
            index: n % KEYFRAME_INTERVAL,
            idr_id: n / KEYFRAME_INTERVAL,
        })
        .collect()
}

/// Annex B access units: the parameter sets before each IDR, then the
/// slice header, aligned, and a byte for the slice data.
fn stream(syntax: &dyn Syntax, header: impl Fn(Step) -> Writer) -> Vec<Packet> {
    steps()
        .into_iter()
        .zip(0..)
        .map(|(step, pts)| {
            let mut data = Vec::new();
            if step.idr {
                data.extend(Packed::units(&syntax.sets()).data);
            }
            let mut slice = header(step);
            if !slice.bit_len().is_multiple_of(8) {
                slice.align();
            }
            slice.u(8, 0xa5);
            data.extend(Packed::units(&[slice.into_bytes()]).data);
            Packet {
                stream: 0,
                data,
                pts,
                duration: 1,
                timescale: 30,
                keyframe: step.idr,
                side: Vec::new(),
            }
        })
        .collect()
}

fn annex_b(codec: CodecId, width: u32, height: u32) -> CodecParams {
    let mut params = CodecParams::video(codec, width, height);
    params.nal_length_size = 0;
    params
}

/// Surface ids the recorders hand out: the picture's number plus one.
fn surface(picture: usize) -> u32 {
    u32::try_from(picture).expect("picture") + 1
}

struct H264Recorder {
    syntax: H264Syntax,
    decoded: usize,
    sequences: usize,
}

impl h264::Accelerator for H264Recorder {
    type Surface = u32;

    fn sequence(&mut self, sps: &h264::Sps) -> Result<(), CodecError> {
        let h = &self.syntax.headers;
        assert_eq!(sps.profile_idc, h.profile.idc());
        assert_eq!(sps.level_idc, h.level_idc);
        assert_eq!(
            sps.constraint_set_flags[1],
            h.profile != H264Profile::High,
            "constraint_set1_flag"
        );
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_order_cnt_type, 2);
        assert_eq!(sps.max_num_ref_frames, 1);
        assert_eq!(sps.pic_width_in_mbs_minus1 + 1, h.width_mbs);
        assert_eq!(sps.pic_height_in_map_units_minus1 + 1, h.height_mbs);
        assert!(sps.frame_mbs_only_flag);
        assert_eq!(sps.frame_cropping_flag, h.crop != (0, 0));
        assert_eq!(sps.frame_crop_right_offset * 2, h.crop.0);
        assert_eq!(sps.frame_crop_bottom_offset * 2, h.crop.1);
        let vui = &sps.vui;
        assert!(vui.video_signal_type_present_flag);
        assert!(vui.colour_description_present_flag);
        assert_eq!(
            (
                vui.colour_primaries,
                vui.transfer_characteristics,
                vui.matrix_coefficients
            ),
            (9, 16, 9)
        );
        assert!(vui.timing_info_present_flag);
        assert_eq!(
            (vui.time_scale, vui.num_units_in_tick),
            (h.rate.0 * 2, h.rate.1)
        );
        assert!(vui.bitstream_restriction_flag);
        assert_eq!(
            (vui.max_num_reorder_frames, vui.max_dec_frame_buffering),
            (0, 1)
        );
        self.sequences += 1;
        Ok(())
    }

    fn missing(&mut self, _sps: &h264::Sps, _frame_num: u32) -> Result<u32, CodecError> {
        panic!("the stream has no frame_num gaps");
    }

    fn decode(&mut self, picture: &h264::Picture<'_, u32>) -> Result<u32, CodecError> {
        let step = steps()[self.decoded];
        let h = &self.syntax.headers;
        assert_eq!(picture.idr, step.idr);
        assert_ne!(picture.nal_ref_idc, 0);
        assert_eq!(picture.frame_num, H264::frame_num(step));
        assert_eq!(picture.top_field_order_cnt, H264::poc(step));
        assert_eq!(picture.pps.entropy_coding_mode_flag, h.profile.cabac());
        assert_eq!(
            picture.pps.transform_8x8_mode_flag,
            h.profile.transform_8x8()
        );
        assert_eq!(picture.pps.pic_init_qp_minus26, 0);
        let [slice] = picture.slices else {
            panic!("one slice a picture");
        };
        assert_eq!(slice.header.slice_type % 5, if step.idr { 2 } else { 0 });
        assert_eq!(
            slice.header_bits as usize,
            h.slice_header(step).bit_len(),
            "the slice data starts where the packed header ends"
        );
        let current = surface(self.decoded);
        let reference = (!step.idr).then(|| surface(self.decoded - 1));
        let pictures = Pictures {
            current,
            reference,
            coded: CODED,
        };
        let pic = self.syntax.picture(step, pictures);
        let params = self.syntax.slice(step, pictures);
        assert_eq!(
            pic.get(picture_h264::PICTURE_ID.of(h264_enc_pic::CURR_PIC)),
            current
        );
        assert_eq!(
            pic.get(picture_h264::TOP_FIELD_ORDER_CNT.of(h264_enc_pic::CURR_PIC)),
            picture.top_field_order_cnt
        );
        assert_eq!(
            u32::from(pic.get(h264_enc_pic::FRAME_NUM)),
            picture.frame_num
        );
        assert_eq!(pic.get(h264_enc_pic::CODED_BUF), CODED);
        if step.idr {
            assert_eq!(
                u32::from(params.get(h264_enc_slice::IDR_PIC_ID)),
                slice.header.idr_pic_id
            );
        }
        let slot = h264_enc_pic::REFERENCE_FRAMES.at(0).expect("slot");
        let first = h264_enc_slice::REF_PIC_LIST0.at(0).expect("slot");
        if step.idr {
            assert!(picture.refs.is_empty());
            assert!(slice.ref_lists[0].is_empty());
            assert_eq!(
                pic.get(picture_h264::FLAGS.of(slot)),
                abi::VA_PICTURE_H264_INVALID
            );
        } else {
            let [held] = picture.refs else {
                panic!("one reference");
            };
            assert_eq!(slice.ref_lists[0], [0]);
            assert_eq!(Some(held.surface), reference);
            for (blob, at) in [(&pic, slot), (&params, first)] {
                assert_eq!(blob.get(picture_h264::PICTURE_ID.of(at)), held.surface);
                assert_eq!(blob.get(picture_h264::FRAME_IDX.of(at)), held.frame_num);
                assert_eq!(
                    blob.get(picture_h264::TOP_FIELD_ORDER_CNT.of(at)),
                    held.top_field_order_cnt
                );
                assert_eq!(
                    blob.get(picture_h264::FLAGS.of(at)),
                    abi::VA_PICTURE_H264_SHORT_TERM_REFERENCE
                );
            }
        }
        self.decoded += 1;
        Ok(current)
    }
}

#[test]
fn h264_headers_read_back_as_the_parameter_buffers_say() {
    for profile in [
        H264Profile::High,
        H264Profile::Main,
        H264Profile::ConstrainedBaseline,
    ] {
        let mut syntax = H264Syntax::new(&settings(CodecId::H264, 100, 60), profile);
        syntax.set_rate(rate16((30_000, 1001)));
        assert_eq!(syntax.headers.crop, (12, 4));
        let packets = stream(&syntax, |step| syntax.headers.slice_header(step));
        let recorder = H264Recorder {
            syntax: syntax.clone(),
            decoded: 0,
            sequences: 0,
        };
        let mut decoder = h264::StatelessDecoder::open(&annex_b(CodecId::H264, 100, 60), recorder)
            .expect("opens");
        let mut shown = Vec::new();
        for packet in &packets {
            shown.extend(decoder.push(packet).expect("parses"));
        }
        shown.extend(decoder.drain().expect("drains"));
        let order: Vec<i64> = shown.iter().map(|output| output.pts).collect();
        assert_eq!(order, (0..i64::from(PICTURES)).collect::<Vec<_>>());
        assert_eq!(decoder.accelerator().decoded, PICTURES as usize);
        assert_eq!(decoder.accelerator().sequences, 1, "{profile:?}");
    }
}

struct HevcRecorder {
    syntax: HevcSyntax,
    decoded: usize,
}

impl hevc::Accelerator for HevcRecorder {
    type Surface = u32;

    fn sequence(&mut self, sps: &hevc::Sps) -> Result<(), CodecError> {
        let h = &self.syntax.headers;
        let ptl = &sps.profile_tier_level;
        assert_eq!(ptl.general_profile_idc, 1);
        assert_eq!(ptl.general_level_idc, h.level_idc);
        assert!(ptl.general_progressive_source_flag && ptl.general_frame_only_constraint_flag);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(
            (
                sps.pic_width_in_luma_samples,
                sps.pic_height_in_luma_samples
            ),
            (h.width, h.height)
        );
        assert_eq!(sps.conformance_window_flag, h.crop != (0, 0));
        assert_eq!(
            (
                sps.conf_win_right_offset * 2,
                sps.conf_win_bottom_offset * 2
            ),
            h.crop
        );
        let blocks = h.blocks;
        assert_eq!(
            u32::from(sps.log2_min_luma_coding_block_size_minus3) + 3,
            blocks.min_cb
        );
        assert_eq!(
            u32::from(sps.log2_diff_max_min_luma_coding_block_size),
            blocks.ctb - blocks.min_cb
        );
        assert_eq!(
            u32::from(sps.log2_min_luma_transform_block_size_minus2) + 2,
            blocks.min_tb
        );
        assert_eq!(
            u32::from(sps.log2_diff_max_min_luma_transform_block_size),
            blocks.max_tb - blocks.min_tb
        );
        assert_eq!(
            u32::from(sps.max_transform_hierarchy_depth_inter),
            blocks.depth_inter
        );
        let tools = h.tools;
        assert_eq!(sps.amp_enabled_flag, tools.amp);
        assert_eq!(sps.sample_adaptive_offset_enabled_flag, tools.sao);
        assert_eq!(sps.sps_temporal_mvp_enabled_flag, tools.temporal_mvp);
        assert_eq!(
            sps.strong_intra_smoothing_enabled_flag,
            tools.strong_intra_smoothing
        );
        assert_eq!(sps.sps_max_dec_pic_buffering_minus1[0], 1);
        assert_eq!(sps.sps_max_num_reorder_pics[0], 0);
        let vui = &sps.vui;
        assert!(vui.video_signal_type_present_flag && vui.colour_description_present_flag);
        assert_eq!(
            (
                vui.colour_primaries,
                vui.transfer_characteristics,
                vui.matrix_coeffs
            ),
            (9, 16, 9)
        );
        assert!(vui.vui_timing_info_present_flag);
        assert_eq!((vui.vui_time_scale, vui.vui_num_units_in_tick), h.rate);
        Ok(())
    }

    fn missing(&mut self, _sps: &hevc::Sps, _poc: i32) -> Result<u32, CodecError> {
        panic!("the stream lacks no reference");
    }

    fn decode(&mut self, picture: &hevc::Picture<'_, u32>) -> Result<u32, CodecError> {
        let step = steps()[self.decoded];
        let h = &self.syntax.headers;
        let tools = h.tools;
        assert_eq!(picture.poc, Hevc::poc(step));
        assert_eq!(u32::from(picture.nal_unit_type), Hevc::nal_unit_type(step));
        let pps = picture.pps;
        assert_eq!(pps.sign_data_hiding_enabled_flag, tools.sign_data_hiding);
        assert_eq!(pps.transform_skip_enabled_flag, tools.transform_skip);
        assert_eq!(pps.cu_qp_delta_enabled_flag, tools.cu_qp_delta);
        assert_eq!(pps.init_qp_minus26, 0);
        let [slice] = picture.slices else {
            panic!("one slice a picture");
        };
        let header = slice.header;
        assert_eq!(header.slice_type as u8, h.slice_type(step));
        assert_eq!(
            slice.header_bits as usize,
            h.slice_header(step).bit_len(),
            "the slice data starts where the packed header ends"
        );
        let mvp = tools.temporal_mvp && !step.idr;
        assert_eq!(header.slice_temporal_mvp_enabled_flag, mvp);
        assert_eq!(header.slice_sao_luma_flag, tools.sao);
        assert_eq!(header.slice_sao_chroma_flag, tools.sao);
        assert_eq!(header.five_minus_max_num_merge_cand, 0);
        let current = surface(self.decoded);
        let reference = (!step.idr).then(|| surface(self.decoded - 1));
        let pictures = Pictures {
            current,
            reference,
            coded: CODED,
        };
        let pic = self.syntax.picture(step, pictures);
        let params = self.syntax.slice(step, pictures);
        let curr = hevc_enc_pic::DECODED_CURR_PIC;
        assert_eq!(pic.get(picture_hevc::PICTURE_ID.of(curr)), current);
        assert_eq!(pic.get(picture_hevc::PIC_ORDER_CNT.of(curr)), picture.poc);
        assert_eq!(
            u32::from(pic.get(hevc_enc_pic::NAL_UNIT_TYPE)),
            Hevc::nal_unit_type(step)
        );
        assert_eq!(params.get(hevc_enc_slice::SLICE_TYPE), h.slice_type(step));
        assert_eq!(params.get(hevc_enc_slice::NUM_CTU_IN_SLICE), h.ctus());
        let lists = [hevc_enc_slice::REF_PIC_LIST0, hevc_enc_slice::REF_PIC_LIST1];
        let slot = hevc_enc_pic::REFERENCE_FRAMES.at(0).expect("slot");
        if step.idr {
            assert!(picture.refs.is_empty());
            assert_eq!(
                pic.get(picture_hevc::FLAGS.of(slot)),
                abi::VA_PICTURE_HEVC_INVALID
            );
        } else {
            let [held] = picture.refs else {
                panic!("one reference");
            };
            assert_eq!(Some(*held.surface), reference);
            assert_eq!(held.poc, picture.poc - 1);
            assert_eq!(pic.get(picture_hevc::PICTURE_ID.of(slot)), *held.surface);
            assert_eq!(pic.get(picture_hevc::PIC_ORDER_CNT.of(slot)), held.poc);
            for (list, entries) in lists.iter().zip(&slice.ref_lists) {
                let at = list.at(0).expect("slot");
                if entries.is_empty() {
                    assert_eq!(
                        params.get(picture_hevc::FLAGS.of(at)),
                        abi::VA_PICTURE_HEVC_INVALID
                    );
                } else {
                    assert_eq!(entries, &[0]);
                    assert_eq!(params.get(picture_hevc::PICTURE_ID.of(at)), *held.surface);
                    assert_eq!(params.get(picture_hevc::PIC_ORDER_CNT.of(at)), held.poc);
                }
            }
            assert_eq!(slice.ref_lists[1].is_empty(), !h.gpb);
        }
        self.decoded += 1;
        Ok(current)
    }
}

#[test]
fn hevc_headers_read_back_as_the_parameter_buffers_say() {
    let every = Tools {
        amp: true,
        sao: true,
        temporal_mvp: true,
        strong_intra_smoothing: true,
        sign_data_hiding: true,
        transform_skip: true,
        cu_qp_delta: true,
    };
    let low_power = Blocks {
        min_cb: 3,
        ctb: 6,
        min_tb: 2,
        max_tb: 5,
        depth_inter: 2,
        depth_intra: 2,
    };
    let cases = [
        (Blocks::DEFAULT, 16, Tools::default(), false, (100, 60)),
        (low_power, 8, every, true, (100, 60)),
        (low_power, 8, every, false, (1920, 1080)),
    ];
    for (blocks, align, tools, gpb, (width, height)) in cases {
        let mut syntax = HevcSyntax::new(
            &settings(CodecId::Hevc, width, height),
            blocks,
            align,
            tools,
            gpb,
        );
        syntax.set_rate(rate16((25, 1)));
        let h = &syntax.headers;
        assert_eq!(h.width % align, 0);
        assert_eq!((h.width - h.crop.0, h.height - h.crop.1), (width, height));
        let packets = stream(&syntax, |step| syntax.headers.slice_header(step));
        let recorder = HevcRecorder {
            syntax: syntax.clone(),
            decoded: 0,
        };
        let mut decoder =
            hevc::StatelessDecoder::open(&annex_b(CodecId::Hevc, width, height), recorder)
                .expect("opens");
        let mut shown = Vec::new();
        for packet in &packets {
            shown.extend(decoder.push(packet).expect("parses"));
        }
        shown.extend(decoder.drain().expect("drains"));
        let order: Vec<i64> = shown.iter().map(|output| output.pts).collect();
        assert_eq!(order, (0..i64::from(PICTURES)).collect::<Vec<_>>());
        assert_eq!(decoder.accelerator().decoded, PICTURES as usize);
    }
}

#[test]
fn reported_block_sizes_and_features_set_the_hevc_tools() {
    use super::hevc_enc::{blocks, tools};
    use abi::{hevc_block_sizes as b, hevc_features as f};
    let put = |pairs: &[(crate::video::imp::blob::Bits, u32)]| {
        pairs
            .iter()
            .fold(0, |value, (bits, field)| value | (field << bits.shift))
    };
    // Intel's low-power HEVC: 64x64 tree blocks, 8x8 coding blocks,
    // transforms 4x4 to 32x32.
    let reported = put(&[
        (b::LOG2_MAX_CODING_TREE_BLOCK_SIZE_MINUS3, 3),
        (b::LOG2_MIN_LUMA_CODING_BLOCK_SIZE_MINUS3, 0),
        (b::LOG2_MAX_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2, 3),
        (b::LOG2_MIN_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2, 0),
        (b::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTER, 3),
        (b::MAX_MAX_TRANSFORM_HIERARCHY_DEPTH_INTRA, 2),
    ]);
    let (found, align) = blocks(reported);
    assert_eq!(
        found,
        Blocks {
            min_cb: 3,
            ctb: 6,
            min_tb: 2,
            max_tb: 5,
            depth_inter: 3,
            depth_intra: 2,
        }
    );
    assert_eq!(align, 8);
    assert_eq!(blocks(abi::VA_ATTRIB_NOT_SUPPORTED), (Blocks::DEFAULT, 16));
    // A minimum transform as large as the minimum coding block is not
    // HEVC: the defaults stand.
    let wrong = put(&[
        (b::LOG2_MAX_CODING_TREE_BLOCK_SIZE_MINUS3, 2),
        (b::LOG2_MIN_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2, 1),
        (b::LOG2_MAX_LUMA_TRANSFORM_BLOCK_SIZE_MINUS2, 3),
    ]);
    assert_eq!(blocks(wrong), (Blocks::DEFAULT, 16));

    let vbr = super::encode::RateControl {
        mode: abi::VA_RC_VBR,
        target: 1,
    };
    let cqp = super::encode::RateControl {
        mode: abi::VA_RC_CQP,
        target: 1,
    };
    let features = put(&[
        (f::AMP, 1),
        (f::SAO, 2),
        (f::TEMPORAL_MVP, 1),
        (f::TRANSFORM_SKIP, 0),
        (f::CU_QP_DELTA, 1),
    ]);
    let on = tools(features, vbr);
    assert!(on.amp && on.sao && on.temporal_mvp && on.cu_qp_delta);
    assert!(!on.transform_skip && !on.sign_data_hiding);
    assert!(
        !tools(features, cqp).cu_qp_delta,
        "optional without rate control"
    );
    let required = put(&[(f::CU_QP_DELTA, 2)]);
    assert!(tools(required, cqp).cu_qp_delta);
    let unknown = tools(abi::VA_ATTRIB_NOT_SUPPORTED, vbr);
    assert_eq!(
        unknown,
        Tools {
            amp: true,
            cu_qp_delta: true,
            ..Tools::default()
        }
    );
}

#[test]
fn the_h264_sequence_buffer_matches_the_sps() {
    use abi::h264_enc_seq as q;
    let syntax = H264Syntax::new(&settings(CodecId::H264, 1920, 1080), H264Profile::High);
    let rc = super::encode::RateControl {
        mode: abi::VA_RC_VBR,
        target: 2_000_000,
    };
    let blob = syntax.sequence(rc, KEYFRAME_INTERVAL);
    assert_eq!(blob.get(q::INTRA_IDR_PERIOD), KEYFRAME_INTERVAL);
    assert_eq!(blob.get(q::IP_PERIOD), 1);
    assert_eq!(blob.get(q::BITS_PER_SECOND), 2_000_000);
    assert_eq!(blob.get(q::PICTURE_WIDTH_IN_MBS), 120);
    assert_eq!(blob.get(q::PICTURE_HEIGHT_IN_MBS), 68);
    assert_eq!(blob.get(q::FRAME_CROPPING_FLAG), 1);
    assert_eq!(blob.get(q::FRAME_CROP_BOTTOM_OFFSET), 4);
    assert_eq!(blob.bits(q::PIC_ORDER_CNT_TYPE), 2);
    assert_eq!(blob.bits(q::LOG2_MAX_FRAME_NUM_MINUS4), 4);
    assert_eq!(blob.get(q::LEVEL_IDC), syntax.headers.level_idc);
    assert_eq!(blob.get(q::TIME_SCALE), 60);
}
