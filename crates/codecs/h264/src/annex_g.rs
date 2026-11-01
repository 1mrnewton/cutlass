//! Annex G slice extension.
//!
//! A slice with no inter-layer prediction is the Annex A reconstruction.
//! Same-size intra base mode copies the reference layer and the enhancement
//! crop is the picture. FFmpeg 8.1.1 does not decode NAL type 20, so that
//! picture follows the recommendation. FFmpeg still emits the Annex A base.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, VideoOutput};

use crate::H264Decoder;
use crate::nal::{annex_b_extension, annex_b_nal};
use crate::set::parse_subset_sps;

#[test]
fn a_slice_extension_without_inter_layer_prediction_is_a_picture() {
    for profile in [83u8, 86] {
        for restriction in [true, false] {
            let rbsp = subset(profile, restriction, 0, false);
            let parsed = parse_subset_sps(&rbsp).expect("subset");
            assert_eq!(parsed.slice_header_restriction, restriction);
            let annex = svc_annex(&rbsp, 0, true, restriction);
            let picture = decode(&annex);
            assert!(
                flat(&picture),
                "profile {profile} restriction {restriction}"
            );
            if let Some(ffmpeg) = ffmpeg_bin() {
                assert!(
                    ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
                    "ffmpeg decoded a slice extension"
                );
            }
        }
    }
    let annex = svc_annex(&subset(83, true, 0, false), 1, true, true);
    assert!(flat(&decode(&annex)));
}

#[test]
fn output_flag_keeps_the_extension_off_the_display() {
    let annex = svc_annex(&subset(83, true, 0, false), 0, false, true);
    let mut decoder = open();
    let pictures = decoder
        .push(&Packet::video(annex, 0, 1, 30, true))
        .expect("decode");
    assert!(pictures.is_empty());
    assert!(decoder.drain().expect("drain").is_empty());
    let (y, u, v) = decoder.picture_at(0).expect("held");
    assert!(
        y.iter()
            .chain(u.iter())
            .chain(v.iter())
            .all(|sample| *sample == 128)
    );
}

#[test]
fn inter_layer_intra_prediction_copies_the_base() {
    let parsed = parse_subset_sps(&layer_subset()).expect("subset");
    assert!(parsed.inter_layer_deblock);
    assert_eq!(parsed.svc_spatial, 0);
    assert!(!parsed.seq_tcoeff);
    assert_eq!(parsed.visible().expect("visible"), (8, 8));
    let base = pcm_annex();
    let base_picture = decode(&base);
    assert_eq!(base_picture.picture.width(), 16);
    assert_eq!(base_picture.picture.height(), 16);
    assert!(painted(&base_picture, 200, 40));
    let annex = inter_layer_annex();
    let picture = decode(&annex);
    assert_eq!(picture.picture.width(), 8);
    assert_eq!(picture.picture.height(), 8);
    assert!(painted(&picture, 200, 40));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &base);
        assert_eq!(raw, paint(16, 200, 40));
        let layered = ffmpeg_raw(&ffmpeg, &annex);
        assert_eq!(layered, paint(16, 200, 40));
    }
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn a_non_base_mvc_view_is_an_intra_picture() {
    for profile in [118u8, 128] {
        let parsed = parse_subset_sps(&mvc_subset(profile, false)).expect("mvc");
        assert_eq!(parsed.views, vec![0, 1], "profile {profile}");
    }
    let annex = mvc_annex(&mvc_header(1));
    let picture = decode(&annex);
    assert!(flat(&picture));
    assert_eq!(picture.picture.width(), 16);
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a non-base view"
        );
    }
    let err = open()
        .push(&Packet::video(mvc_annex(&mvc_header(2)), 0, 1, 30, false))
        .expect_err("view");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "view id",
                ..
            }
        ),
        "{err:?}"
    );
    let vui = parse_subset_sps(&mvc_subset(118, true)).expect_err("vui");
    assert!(matches!(vui, CodecError::Unsupported { what: "mvc vui" }));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn an_annex_i_depth_view_is_an_intra_picture() {
    for profile in [135u8, 138] {
        let parsed = parse_subset_sps(&mvcd_subset(profile, false)).expect("mvcd");
        assert_eq!(parsed.views, vec![0], "profile {profile}");
        assert_eq!(parsed.depth_views, vec![true], "profile {profile}");
    }
    let annex = depth_annex(&mvc_header(0));
    let picture = decode(&annex);
    assert!(flat(&picture));
    assert_eq!(picture.picture.width(), 16);
    if let Some(ffmpeg) = ffmpeg_bin() {
        assert!(
            ffmpeg_bytes(&ffmpeg, &annex).is_empty(),
            "ffmpeg decoded a depth view"
        );
    }
    let err = open()
        .push(&Packet::video(depth_annex(&mvc_header(1)), 0, 1, 30, false))
        .expect_err("view");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "view id",
                ..
            }
        ),
        "{err:?}"
    );
    let vui = parse_subset_sps(&mvcd_subset(138, true)).expect_err("vui");
    assert!(matches!(vui, CodecError::Unsupported { what: "mvcd vui" }));
    let mut decoder = open();
    let _ = decode_with(&mut decoder, &annex);
    assert!(flat(&decode_with(&mut decoder, &base_idr())));
}

#[test]
fn an_annex_a_picture_survives_svc_mvc_and_3d_nal_units() {
    let annex = base_with_extensions();
    let picture = decode(&annex);
    assert!(flat(&picture));
    if let Some(ffmpeg) = ffmpeg_bin() {
        let raw = ffmpeg_raw(&ffmpeg, &annex);
        assert_eq!(raw, vec![128u8; 16 * 16 + 8 * 8 * 2]);
    }
}

#[test]
fn the_svc_extension_syntax_is_consumed() {
    let parsed = parse_subset_sps(&subset(83, true, 1, false)).expect("ess");
    assert!(parsed.slice_header_restriction);
    let err = parse_subset_sps(&subset(83, true, 3, false)).expect_err("ess");
    assert!(
        matches!(
            err,
            CodecError::Malformed {
                what: "extended spatial scalability",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_refused_extension_leaves_the_next_idr() {
    let mut decoder = open();
    let refused = [
        annex_b_nal(3, 20, &[]),
        annex_b_extension(3, 20, &[0xc0, 0x80, 0x04], &[0x80]),
        svc_annex_header(svc_bytes(false, 0, true)),
        annex_b_extension(3, 20, &[0xc0, 0x80, 0x17], &[0x80]),
        annex_b_extension(3, 20, &[0x00, 0x00, 0x03], &[0x80]),
        annex_b_extension(3, 21, &[0x80, 0x00], &[0x80]),
        annex_b_nal(3, 15, &subset(83, true, 0, true)),
        annex_b_nal(3, 15, &truncated_scalable()),
        stored_base_picture(),
        base_with_broken_layer(),
        residual_layer(),
    ];
    let mut errors = Vec::new();
    for annex in refused {
        errors.push(
            decoder
                .push(&Packet::video(annex, 0, 1, 30, false))
                .expect_err("refused"),
        );
    }
    assert!(matches!(
        errors[0],
        CodecError::Truncated {
            what: "extension header",
            ..
        }
    ));
    assert!(matches!(
        errors[1],
        CodecError::Malformed {
            what: "extension header",
            ..
        }
    ));
    assert!(matches!(
        errors[2],
        CodecError::Unsupported {
            what: "inter-layer prediction"
        }
    ));
    assert!(matches!(
        errors[3],
        CodecError::Unsupported {
            what: "reference base picture"
        }
    ));
    assert!(matches!(errors[4], CodecError::Truncated { .. }));
    assert!(matches!(errors[5], CodecError::Unsupported { what: "3d" }));
    assert!(matches!(
        errors[6],
        CodecError::Unsupported { what: "svc vui" }
    ));
    assert!(matches!(errors[7], CodecError::Truncated { .. }));
    assert!(matches!(
        errors[8],
        CodecError::Unsupported {
            what: "reference base picture"
        }
    ));
    assert!(matches!(errors[9], CodecError::Truncated { .. }));
    assert!(matches!(
        errors[10],
        CodecError::Unsupported {
            what: "inter-layer prediction"
        }
    ));
    let picture = decode_with(&mut decoder, &base_idr());
    assert!(flat(&picture));
}

fn stored_base_picture() -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &subset(83, false, 0, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        20,
        &svc_bytes(true, 0, true),
        &idr_slice(true, false, true),
    ));
    out
}

fn svc_annex(rbsp: &[u8], quality: u8, output: bool, restriction: bool) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, rbsp);
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        20,
        &svc_bytes(true, quality, output),
        &idr_slice(quality == 0, restriction, false),
    ));
    out
}

fn svc_annex_header(header: [u8; 3]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &subset(83, true, 0, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 20, &header, &[0x80]));
    out
}

fn base_with_extensions() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &subset(83, true, 0, false)));
    out.extend(annex_b_nal(3, 15, &baseline_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(3, 14, &svc_bytes(true, 0, true), &[0x80]));
    out.extend(annex_b_nal(3, 5, &idr_slice(true, true, false)));
    out.extend(annex_b_extension(3, 20, &[0x00, 0x00, 0x03], &[0x80]));
    out.extend(annex_b_extension(3, 21, &[0x80, 0x00], &[0x80]));
    out
}

fn base_idr() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &idr_slice(true, true, false)));
    out
}

fn svc_bytes(no_inter_layer_pred: bool, quality: u8, output: bool) -> [u8; 3] {
    let b1 = (u8::from(no_inter_layer_pred) << 7) | (quality & 0x0f);
    let b2 = 0x03 | (u8::from(output) << 2);
    [0xc0, b1, b2]
}

/// IDR view, `view_id` in the low bits, anchor, reserved 1.
fn mvc_header(view: u16) -> [u8; 3] {
    let view = view & 0x03ff;
    let b1 = (view >> 2) as u8;
    let b2 = ((view as u8) << 6) | 0x05;
    [0x00, b1, b2]
}

fn mvc_annex(header: &[u8; 3]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &mvc_subset(118, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        20,
        header,
        &idr_slice(true, true, false),
    ));
    out
}

fn depth_annex(header: &[u8; 3]) -> Vec<u8> {
    let mut out = annex_b_nal(3, 15, &mvcd_subset(138, false));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_extension(
        3,
        21,
        header,
        &idr_slice(true, true, false),
    ));
    out
}

fn mvcd_subset(profile: u8, vui: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(u32::from(profile), 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(1);
    w.bit(vui);
    if !vui {
        w.bit(false);
        w.bit(false);
        w.trailing();
    }
    w.bytes
}

fn mvc_subset(profile: u8, vui: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(u32::from(profile), 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.ue(1);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bits(30, 8);
    w.ue(0);
    w.bits(0, 3);
    w.ue(0);
    w.ue(1);
    w.ue(1);
    w.bit(vui);
    if !vui {
        w.bit(false);
        w.trailing();
    }
    w.bytes
}

/// IDR, dependency 1, quality 0, inter-layer prediction on, output, reserved 3.
fn ilp_header() -> [u8; 3] {
    [0xc0, 0x10, 0x07]
}

fn pcm_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out
}

fn inter_layer_annex() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &layer_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &base_mode_slice(false),
    ));
    out
}

fn residual_layer() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &layer_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &pcm_slice(200, 40)));
    out.extend(annex_b_extension(
        3,
        20,
        &ilp_header(),
        &base_mode_slice(true),
    ));
    out
}

fn base_with_broken_layer() -> Vec<u8> {
    let mut out = annex_b_nal(3, 7, &baseline_sps());
    out.extend(annex_b_nal(3, 15, &layer_subset()));
    out.extend(annex_b_nal(3, 8, &pps()));
    out.extend(annex_b_nal(3, 5, &idr_slice(true, true, false)));
    out.extend(annex_b_extension(3, 20, &ilp_header(), &[0x80]));
    out
}

fn layer_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(83, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(true);
    w.ue(0);
    w.ue(4);
    w.ue(0);
    w.ue(4);
    w.bit(false);
    w.bit(true);
    w.bits(0, 2);
    w.bit(false);
    w.bits(1, 2);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn pcm_slice(luma: u8, chroma: u8) -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(25);
    w.align_zeros();
    w.bytes.extend(std::iter::repeat_n(luma, 256));
    w.bytes.extend(std::iter::repeat_n(chroma, 128));
    w.bit = w.bytes.len() * 8;
    w.trailing();
    w.bytes
}

fn base_mode_slice(residual: bool) -> Vec<u8> {
    let mut w = Writer::new();
    i_header(&mut w);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bit(true);
    w.bit(false);
    w.bit(residual);
    w.ue(3);
    w.trailing();
    w.bytes
}

fn i_header(w: &mut Writer) {
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.ue(0);
    w.bits(0, 4);
    w.bit(false);
    w.bit(false);
    w.se(0);
    w.ue(1);
}

fn open() -> H264Decoder {
    let mut params = CodecParams::video(CodecId::H264, 16, 16);
    params.nal_length_size = 0;
    H264Decoder::open(&params).expect("open")
}

fn decode(annex: &[u8]) -> VideoOutput {
    decode_with(&mut open(), annex)
}

fn decode_with(decoder: &mut H264Decoder, annex: &[u8]) -> VideoOutput {
    let mut pictures = decoder
        .push(&Packet::video(annex.to_vec(), 0, 1, 30, true))
        .expect("decode");
    pictures.extend(decoder.drain().expect("drain"));
    assert_eq!(pictures.len(), 1);
    pictures.remove(0)
}

fn flat(picture: &VideoOutput) -> bool {
    painted(picture, 128, 128)
}

fn painted(picture: &VideoOutput, luma: u8, chroma: u8) -> bool {
    picture.picture.y().iter().all(|sample| *sample == luma)
        && picture
            .picture
            .u()
            .iter()
            .chain(picture.picture.v())
            .all(|sample| *sample == chroma)
}

fn paint(edge: usize, luma: u8, chroma: u8) -> Vec<u8> {
    let mut raw = vec![luma; edge * edge];
    raw.extend(std::iter::repeat_n(chroma, (edge / 2) * (edge / 2) * 2));
    raw
}

fn baseline_sps() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn baseline_subset() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(66, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn truncated_scalable() -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(83, 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.trailing();
    w.bytes
}

fn subset(profile: u8, restriction: bool, spatial: u32, vui: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bits(u32::from(profile), 8);
    w.bits(0, 8);
    w.bits(30, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(1);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.bit(true);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.bit(false);
    w.bits(spatial, 2);
    if spatial <= 2 {
        w.bit(false);
        w.bits(1, 2);
        if spatial == 1 {
            w.bit(false);
            w.bits(1, 2);
            w.se(0);
            w.se(0);
            w.se(0);
            w.se(0);
        }
        w.bit(false);
        w.bit(restriction);
        w.bit(vui);
        if !vui {
            w.bit(false);
            w.trailing();
        }
    }
    w.bytes
}

fn pps() -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bit(false);
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.bit(false);
    w.bits(0, 2);
    w.se(0);
    w.se(0);
    w.se(0);
    w.bit(true);
    w.bit(false);
    w.bit(false);
    w.trailing();
    w.bytes
}

fn idr_slice(base_quality: bool, restriction: bool, store_ref_base: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.ue(0);
    w.ue(2);
    w.ue(0);
    w.bits(0, 4);
    w.ue(0);
    w.bits(0, 4);
    if base_quality {
        w.bit(false);
        w.bit(false);
        if !restriction {
            w.bit(store_ref_base);
        }
    }
    w.se(0);
    w.ue(1);
    if !restriction {
        w.bits(0, 4);
        w.bits(15, 4);
    }
    w.ue(3);
    w.ue(0);
    w.se(0);
    w.bit(true);
    w.trailing();
    w.bytes
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn ffmpeg_bytes(ffmpeg: &std::path::Path, annex: &[u8]) -> Vec<u8> {
    spawn(ffmpeg, annex).stdout
}

fn ffmpeg_raw(ffmpeg: &std::path::Path, annex: &[u8]) -> Vec<u8> {
    let output = spawn(ffmpeg, annex);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn spawn(ffmpeg: &std::path::Path, annex: &[u8]) -> std::process::Output {
    let mut child = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    std::io::Write::write_all(child.stdin.as_mut().expect("stdin"), annex).expect("write");
    child.wait_with_output().expect("wait")
}

struct Writer {
    bytes: Vec<u8>,
    bit: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit: 0,
        }
    }

    fn bit(&mut self, value: bool) {
        let index = self.bit / 8;
        if index == self.bytes.len() {
            self.bytes.push(0);
        }
        if value {
            self.bytes[index] |= 1 << (7 - (self.bit % 8));
        }
        self.bit += 1;
    }

    fn bits(&mut self, value: u32, n: u32) {
        for shift in (0..n).rev() {
            self.bit(((value >> shift) & 1) == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let zeros = 31 - code.leading_zeros();
        for _ in 0..zeros {
            self.bit(false);
        }
        self.bits(code, zeros + 1);
    }

    fn align_zeros(&mut self) {
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }

    fn se(&mut self, value: i32) {
        let code = if value > 0 {
            value.unsigned_abs() * 2 - 1
        } else {
            value.unsigned_abs() * 2
        };
        self.ue(code);
    }

    fn trailing(&mut self) {
        self.bit(true);
        while !self.bit.is_multiple_of(8) {
            self.bit(false);
        }
    }
}
